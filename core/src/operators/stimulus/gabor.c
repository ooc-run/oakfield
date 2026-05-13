#include "oakfield/operators/stimulus/gabor.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"

#include <limits.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_GABOR_EPS 1.0e-9
#define STIM_GABOR_VDSP_MIN_LEN 64U

#if defined(__APPLE__)
static inline void stim_gabor_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void stim_gabor_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void stim_gabor_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

typedef struct SimStimulusGaborState {
    SimStimulusGaborConfig config;
    SimClockMode           clock_mode;
    double                 locked_time;
    size_t                 last_step_index;
    bool                   clock_initialized;
    double*                envelope;
    size_t                 envelope_capacity;
    char                   symbolic[160];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_env;
    double* vdsp_aux;
    double* vdsp_theta;
    double* vdsp_value;
    size_t  vdsp_capacity;
#endif
} SimStimulusGaborState;

static void gabor_normalize(SimStimulusGaborConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->amplitude))
        config->amplitude = 0.0;
    if (!isfinite(config->wavenumber))
        config->wavenumber = 0.0;
    if (!isfinite(config->kx))
        config->kx = 0.0;
    if (!isfinite(config->ky))
        config->ky = 0.0;
    if (!isfinite(config->omega))
        config->omega = 0.0;
    if (!isfinite(config->phase))
        config->phase = 0.0;
    if (!isfinite(config->sigma_x) || config->sigma_x <= STIM_GABOR_EPS)
        config->sigma_x = 0.0;
    if (!isfinite(config->sigma_y) || config->sigma_y <= STIM_GABOR_EPS)
        config->sigma_y = 0.0;
    if (!isfinite(config->time_offset))
        config->time_offset = 0.0;

    sim_stimulus_coord_normalize(&config->coord);

    if (!isfinite(config->rotation))
        config->rotation = 0.0;
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0)
        config->nominal_dt = 0.0;

    if (!config->use_wavevector &&
        (fabs(config->kx) > STIM_GABOR_EPS || fabs(config->ky) > STIM_GABOR_EPS)) {
        config->use_wavevector = true;
    }
}

static SimClockMode gabor_resolve_clock_mode(const SimContext*             context,
                                             const char*                   op_name,
                                             const SimStimulusGaborConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_GABOR_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_gabor");
    }
    return resolved;
}

static void gabor_refresh_symbolic(SimStimulusGaborState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimStimulusGaborConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "gabor A=%.3g σ=(%.3g,%.3g) k=%.3g ω=%.3g",
                    cfg->amplitude,
                    cfg->sigma_x,
                    cfg->sigma_y,
                    cfg->wavenumber,
                    cfg->omega);
#else
    (void) state;
#endif
}

static const char* gabor_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusGaborState* state = (const SimStimulusGaborState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

typedef struct StimulusGaborIRParams {
    bool needs_dt;
    bool needs_step_index;
    bool needs_time;
} StimulusGaborIRParams;

static SimIRNodeId
gabor_ir_binary(SimIRBuilder* builder, SimIRNodeType type, SimIRNodeId lhs, SimIRNodeId rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId gabor_ir_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    if (operand == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_call(builder, kind, operand);
}

static SimIRNodeId gabor_build_ir(SimIRBuilder*                 builder,
                                  const SimStimulusGaborConfig* config,
                                  bool                          complex_output,
                                  StimulusGaborIRParams*        params) {
    if (builder == NULL || config == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt         = false;
    params->needs_step_index = false;
    params->needs_time       = false;

    SimIRNodeId index    = sim_ir_builder_index(builder);
    SimIRNodeId spacing  = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin   = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId x_offset = gabor_ir_binary(builder, SIM_IR_NODE_MUL, spacing, index);
    SimIRNodeId x        = gabor_ir_binary(builder, SIM_IR_NODE_ADD, origin, x_offset);
    if (x == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    bool needs_dt =
        config->scale_by_dt || (config->fixed_clock && config->nominal_dt <= STIM_GABOR_EPS);
    SimIRNodeId dt_scaled = SIM_IR_INVALID_NODE;
    if (needs_dt) {
        params->needs_dt    = true;
        SimIRNodeId dt_node = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
        dt_scaled           = dt_node;
    }

    SimIRNodeId t = SIM_IR_INVALID_NODE;
    if (config->fixed_clock) {
        params->needs_step_index = true;
        SimIRNodeId step_index   = sim_ir_builder_param(builder, SIM_IR_PARAM_STEP_INDEX);
        SimIRNodeId increment    = SIM_IR_INVALID_NODE;
        if (config->nominal_dt > STIM_GABOR_EPS) {
            increment = sim_ir_builder_constant(builder, config->nominal_dt);
        } else {
            increment = dt_scaled;
        }
        SimIRNodeId scaled_step = gabor_ir_binary(builder, SIM_IR_NODE_MUL, step_index, increment);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = gabor_ir_binary(builder, SIM_IR_NODE_ADD, scaled_step, time_offset);
    } else {
        params->needs_time      = true;
        SimIRNodeId time_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_TIME);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t                       = gabor_ir_binary(builder, SIM_IR_NODE_ADD, time_node, time_offset);
    }

    if (t == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId velocity         = sim_ir_builder_constant(builder, config->coord.velocity_x);
    SimIRNodeId drift            = gabor_ir_binary(builder, SIM_IR_NODE_MUL, velocity, t);
    SimIRNodeId sample_x         = gabor_ir_binary(builder, SIM_IR_NODE_SUB, x, drift);
    SimIRNodeId center_base      = sim_ir_builder_constant(builder, config->coord.center_x);
    SimIRNodeId diff             = gabor_ir_binary(builder, SIM_IR_NODE_SUB, sample_x, center_base);
    double      sigma_x          = (config->sigma_x > STIM_GABOR_EPS) ? config->sigma_x : 1.0;
    double      inv_two_sigma_sq = -0.5 / (sigma_x * sigma_x);
    SimIRNodeId inv_sigma        = sim_ir_builder_constant(builder, inv_two_sigma_sq);
    SimIRNodeId diff_sq          = gabor_ir_binary(builder, SIM_IR_NODE_MUL, diff, diff);
    SimIRNodeId exponent         = gabor_ir_binary(builder, SIM_IR_NODE_MUL, diff_sq, inv_sigma);
    SimIRNodeId envelope         = gabor_ir_call(builder, SIM_IR_CALL_EXP, exponent);
    SimIRNodeId amplitude        = sim_ir_builder_constant(builder, config->amplitude);
    SimIRNodeId envelope_scaled  = gabor_ir_binary(builder, SIM_IR_NODE_MUL, amplitude, envelope);

    SimIRNodeId scale = SIM_IR_INVALID_NODE;
    if (config->scale_by_dt) {
        scale = dt_scaled;
    } else {
        scale = sim_ir_builder_constant(builder, 1.0);
    }

    SimIRNodeId wavenumber = sim_ir_builder_constant(builder, config->wavenumber);
    SimIRNodeId omega      = sim_ir_builder_constant(builder, config->omega);
    SimIRNodeId phase      = sim_ir_builder_constant(builder, config->phase);
    SimIRNodeId kx         = gabor_ir_binary(builder, SIM_IR_NODE_MUL, wavenumber, sample_x);
    SimIRNodeId omega_t    = gabor_ir_binary(builder, SIM_IR_NODE_MUL, omega, t);
    SimIRNodeId phase_t    = gabor_ir_binary(builder, SIM_IR_NODE_SUB, kx, omega_t);
    SimIRNodeId phase_full = gabor_ir_binary(builder, SIM_IR_NODE_ADD, phase_t, phase);

    if (complex_output) {
        SimIRNodeId base_scaled = gabor_ir_binary(builder, SIM_IR_NODE_MUL, envelope_scaled, scale);
        if (base_scaled == SIM_IR_INVALID_NODE) {
            return SIM_IR_INVALID_NODE;
        }
        SimIRNodeId zero   = sim_ir_builder_constant(builder, 0.0);
        SimIRNodeId packed = sim_ir_builder_complex_pack(builder, base_scaled, zero);
        if (packed == SIM_IR_INVALID_NODE) {
            return SIM_IR_INVALID_NODE;
        }
        SimIRNodeId angle = phase_full;
        if (config->rotation != 0.0) {
            SimIRNodeId rotation = sim_ir_builder_constant(builder, config->rotation);
            angle                = gabor_ir_binary(builder, SIM_IR_NODE_ADD, phase_full, rotation);
        }
        return sim_ir_builder_complex_rotate(builder, packed, angle);
    }

    SimIRNodeId cos_phase = gabor_ir_call(builder, SIM_IR_CALL_COS, phase_full);
    SimIRNodeId value     = gabor_ir_binary(builder, SIM_IR_NODE_MUL, envelope_scaled, cos_phase);
    if (value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return gabor_ir_binary(builder, SIM_IR_NODE_MUL, value, scale);
}

static void gabor_destroy(void* state_ptr) {
    SimStimulusGaborState* state = (SimStimulusGaborState*) state_ptr;
    if (!state)
        return;
    free(state->envelope);
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
#endif
    free(state);
}

static SimResult gabor_ensure_envelope(SimStimulusGaborState* state, size_t count) {
    if (!state)
        return SIM_RESULT_INVALID_ARGUMENT;
    if (count == 0U)
        return SIM_RESULT_OK;
    if (state->envelope_capacity >= count)
        return SIM_RESULT_OK;

    double* resized = (double*) realloc(state->envelope, count * sizeof(double));
    if (!resized)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->envelope          = resized;
    state->envelope_capacity = count;
    return SIM_RESULT_OK;
}

#if defined(SIM_HAVE_VDSP)
static bool gabor_vdsp_ensure_buffers(SimStimulusGaborState* state, size_t width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* resized = (double*) realloc(state->vdsp_block, width * 4U * sizeof(double));
    if (resized == NULL) {
        return false;
    }

    state->vdsp_block    = resized;
    state->vdsp_capacity = width;
    state->vdsp_env      = resized;
    state->vdsp_aux      = resized + width;
    state->vdsp_theta    = resized + width * 2U;
    state->vdsp_value    = resized + width * 3U;
    return true;
}

static bool gabor_linear_map(const SimStimulusGaborConfig* cfg, double* out_u_x, double* out_u_y) {
    if (cfg == NULL || out_u_x == NULL || out_u_y == NULL) {
        return false;
    }

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                *out_u_x = 0.0;
                *out_u_y = 1.0;
            } else {
                *out_u_x = 1.0;
                *out_u_y = 0.0;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE:
            *out_u_x = cos(cfg->coord.angle);
            *out_u_y = sin(cfg->coord.angle);
            return true;
        default:
            break;
    }

    return false;
}

static bool gabor_vdsp_eval_nonseparable_envelope_row(SimStimulusGaborState*        state,
                                                      const SimStimulusGaborConfig* cfg,
                                                      size_t                        width,
                                                      double                        raw_x0,
                                                      double                        raw_y,
                                                      double                        dx,
                                                      double                        t) {
    if (state == NULL || cfg == NULL || width == 0U) {
        return false;
    }

    double sigma_x            = (cfg->sigma_x > STIM_GABOR_EPS) ? cfg->sigma_x : 1.0;
    double sigma_y            = (cfg->sigma_y > STIM_GABOR_EPS) ? cfg->sigma_y : sigma_x;
    double center_x           = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double center_y           = cfg->coord.center_y + cfg->coord.velocity_y * t;
    double inv_two_sigma_x_sq = 0.5 / (sigma_x * sigma_x);
    double inv_two_sigma_y_sq = 0.5 / (sigma_y * sigma_y);
    double angle              = cfg->coord.angle;
    double dx0                = raw_x0 - center_x;
    double dy0                = raw_y - center_y;

    if (!isfinite(dx0) || !isfinite(dy0) || !isfinite(dx) || !isfinite(inv_two_sigma_x_sq) ||
        !isfinite(inv_two_sigma_y_sq)) {
        return false;
    }

    const vDSP_Length len = (vDSP_Length) width;
    if (fabs(angle) <= STIM_GABOR_EPS) {
        double start      = dx0;
        double step       = dx;
        double exponent_y = -(dy0 * dy0) * inv_two_sigma_y_sq;
        vDSP_vrampD(&start, &step, state->vdsp_env, 1, len);
        vDSP_vsqD(state->vdsp_env, 1, state->vdsp_env, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_env, width, -inv_two_sigma_x_sq);
        vDSP_vsaddD(state->vdsp_env, 1, &exponent_y, state->vdsp_env, 1, len);
    } else {
        double sin_a    = sin(angle);
        double cos_a    = cos(angle);
        double xr_start = cos_a * dx0 + sin_a * dy0;
        double xr_step  = cos_a * dx;
        double yr_start = -sin_a * dx0 + cos_a * dy0;
        double yr_step  = -sin_a * dx;

        if (!isfinite(xr_start) || !isfinite(xr_step) || !isfinite(yr_start) ||
            !isfinite(yr_step)) {
            return false;
        }

        vDSP_vrampD(&xr_start, &xr_step, state->vdsp_env, 1, len);
        vDSP_vrampD(&yr_start, &yr_step, state->vdsp_aux, 1, len);
        vDSP_vsqD(state->vdsp_env, 1, state->vdsp_env, 1, len);
        vDSP_vsqD(state->vdsp_aux, 1, state->vdsp_aux, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_env, width, -inv_two_sigma_x_sq);
        sim_accel_scale_inplace_real(state->vdsp_aux, width, -inv_two_sigma_y_sq);
        vDSP_vaddD(state->vdsp_env, 1, state->vdsp_aux, 1, state->vdsp_env, 1, len);
    }

    const int vforce_len = (int) width;
    vvexp(state->vdsp_env, state->vdsp_env, &vforce_len);
    sim_accel_scale_inplace_real(state->vdsp_env, width, cfg->amplitude);
    return true;
}

static bool gabor_vdsp_eval_separable_envx_row(SimStimulusGaborState*        state,
                                               const SimStimulusGaborConfig* cfg,
                                               size_t                        width,
                                               double                        sample_x0,
                                               double                        dx) {
    if (state == NULL || cfg == NULL || width == 0U) {
        return false;
    }

    double sigma_x            = (cfg->sigma_x > STIM_GABOR_EPS) ? cfg->sigma_x : 1.0;
    double inv_two_sigma_x_sq = 0.5 / (sigma_x * sigma_x);
    double start              = sample_x0 - cfg->coord.center_x;
    double step               = dx;
    if (!isfinite(start) || !isfinite(step) || !isfinite(inv_two_sigma_x_sq)) {
        return false;
    }

    const vDSP_Length len = (vDSP_Length) width;
    vDSP_vrampD(&start, &step, state->vdsp_env, 1, len);
    vDSP_vsqD(state->vdsp_env, 1, state->vdsp_env, 1, len);
    sim_accel_scale_inplace_real(state->vdsp_env, width, -inv_two_sigma_x_sq);
    const int vforce_len = (int) width;
    vvexp(state->vdsp_env, state->vdsp_env, &vforce_len);
    return true;
}

static bool gabor_try_vdsp_linear_rows(SimStimulusGaborState* state,
                                       const SimField*        field,
                                       bool                   is_complex,
                                       double*                dst_real,
                                       SimComplexDouble*      dst_complex,
                                       size_t                 count,
                                       double                 scale,
                                       double                 t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t rank = field->layout.rank;
    if (rank == 0U || rank > 2U) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_GABOR_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!gabor_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusGaborConfig* cfg            = &state->config;
    bool                          use_wavevector = cfg->use_wavevector;
    bool   separable = (!use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double u_x       = 0.0;
    double u_y       = 0.0;

    if (!use_wavevector && !separable && !gabor_linear_map(cfg, &u_x, &u_y)) {
        return false;
    }

    double raw_x0     = cfg->coord.origin_x;
    double raw_y0     = cfg->coord.origin_y;
    double sample_x0  = raw_x0 - cfg->coord.velocity_x * t;
    double sample_y0  = raw_y0 - cfg->coord.velocity_y * t;
    double dx         = cfg->coord.spacing_x;
    double dy         = cfg->coord.spacing_y;
    double theta_bias = cfg->phase - cfg->omega * t;
    double re_scale   = scale;
    double im_scale   = 0.0;

    if (is_complex) {
        re_scale = scale * cos(cfg->rotation);
        im_scale = scale * sin(cfg->rotation);
    }

    if (!isfinite(raw_x0) || !isfinite(raw_y0) || !isfinite(sample_x0) || !isfinite(sample_y0) ||
        !isfinite(dx) || !isfinite(dy) || !isfinite(theta_bias) || !isfinite(re_scale) ||
        !isfinite(im_scale)) {
        return false;
    }
    if (re_scale == 0.0 && (!is_complex || im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;

    for (size_t row = 0U; row < height; ++row) {
        size_t offset      = row * width;
        double raw_y       = raw_y0 + (double) row * dy;
        double sample_y    = sample_y0 + (double) row * dy;
        double theta_start = 0.0;
        double theta_step  = 0.0;

        if (separable) {
            double sigma_y = (cfg->sigma_y > STIM_GABOR_EPS)
                                 ? cfg->sigma_y
                                 : ((cfg->sigma_x > STIM_GABOR_EPS) ? cfg->sigma_x : 1.0);
            double inv_two_sigma_y_sq = 0.5 / (sigma_y * sigma_y);
            double diff_y             = sample_y - cfg->coord.center_y;
            double env_y              = exp(-(diff_y * diff_y) * inv_two_sigma_y_sq);
            double theta_y            = cfg->wavenumber * sample_y + theta_bias;

            if (!isfinite(env_y) || !isfinite(theta_y) ||
                !gabor_vdsp_eval_separable_envx_row(state, cfg, width, sample_x0, dx)) {
                return false;
            }

            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_MULTIPLY) {
                double row_scale = cfg->amplitude * env_y;
                theta_start      = cfg->wavenumber * sample_x0 + theta_y;
                theta_step       = cfg->wavenumber * dx;
                if (!isfinite(row_scale) || !isfinite(theta_start) || !isfinite(theta_step)) {
                    return false;
                }
                vDSP_vrampD(&theta_start, &theta_step, state->vdsp_theta, 1, len);
                if (!is_complex) {
                    vvcos(state->vdsp_value, state->vdsp_theta, &vforce_len);
                    vDSP_vmulD(state->vdsp_env, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, row_scale);
                    vDSP_vsmaD(state->vdsp_value,
                               1,
                               &re_scale,
                               dst_real + offset,
                               1,
                               dst_real + offset,
                               1,
                               len);
                } else {
                    vvsincos(state->vdsp_aux, state->vdsp_value, state->vdsp_theta, &vforce_len);
                    vDSP_vmulD(state->vdsp_env, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    vDSP_vmulD(state->vdsp_env, 1, state->vdsp_aux, 1, state->vdsp_aux, 1, len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, row_scale);
                    sim_accel_scale_inplace_real(state->vdsp_aux, width, row_scale);
                }
            } else {
                double theta_x_start = cfg->wavenumber * sample_x0 + theta_bias;
                double theta_x_step  = cfg->wavenumber * dx;
                double fy_re         = env_y * cos(theta_y);
                double fy_im         = env_y * sin(theta_y);
                if (!isfinite(theta_x_start) || !isfinite(theta_x_step) || !isfinite(fy_re) ||
                    !isfinite(fy_im)) {
                    return false;
                }
                vDSP_vrampD(&theta_x_start, &theta_x_step, state->vdsp_theta, 1, len);
                if (!is_complex) {
                    vvcos(state->vdsp_value, state->vdsp_theta, &vforce_len);
                    vDSP_vmulD(state->vdsp_env, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, cfg->amplitude);
                    double row_bias = cfg->amplitude * fy_re;
                    vDSP_vsaddD(state->vdsp_value, 1, &row_bias, state->vdsp_value, 1, len);
                    vDSP_vsmaD(state->vdsp_value,
                               1,
                               &re_scale,
                               dst_real + offset,
                               1,
                               dst_real + offset,
                               1,
                               len);
                } else {
                    vvsincos(state->vdsp_aux, state->vdsp_value, state->vdsp_theta, &vforce_len);
                    vDSP_vmulD(state->vdsp_env, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    vDSP_vmulD(state->vdsp_env, 1, state->vdsp_aux, 1, state->vdsp_aux, 1, len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, cfg->amplitude);
                    sim_accel_scale_inplace_real(state->vdsp_aux, width, cfg->amplitude);
                    double re_bias = cfg->amplitude * fy_re;
                    double im_bias = cfg->amplitude * fy_im;
                    if (re_bias != 0.0) {
                        vDSP_vsaddD(state->vdsp_value, 1, &re_bias, state->vdsp_value, 1, len);
                    }
                    if (im_bias != 0.0) {
                        vDSP_vsaddD(state->vdsp_aux, 1, &im_bias, state->vdsp_aux, 1, len);
                    }
                }
            }
        } else {
            if (!gabor_vdsp_eval_nonseparable_envelope_row(
                    state, cfg, width, raw_x0, raw_y, dx, t)) {
                return false;
            }

            if (use_wavevector) {
                theta_start = cfg->kx * sample_x0 + cfg->ky * sample_y + theta_bias;
                theta_step  = cfg->kx * dx;
            } else {
                theta_start = cfg->wavenumber * (u_x * sample_x0 + u_y * sample_y) + theta_bias;
                theta_step  = cfg->wavenumber * u_x * dx;
            }
            if (!isfinite(theta_start) || !isfinite(theta_step)) {
                return false;
            }

            vDSP_vrampD(&theta_start, &theta_step, state->vdsp_theta, 1, len);
            if (!is_complex) {
                vvcos(state->vdsp_value, state->vdsp_theta, &vforce_len);
                vDSP_vmulD(state->vdsp_env, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                vDSP_vsmaD(state->vdsp_value,
                           1,
                           &re_scale,
                           dst_real + offset,
                           1,
                           dst_real + offset,
                           1,
                           len);
            } else {
                vvsincos(state->vdsp_aux, state->vdsp_value, state->vdsp_theta, &vforce_len);
                vDSP_vmulD(state->vdsp_env, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                vDSP_vmulD(state->vdsp_env, 1, state->vdsp_aux, 1, state->vdsp_aux, 1, len);
            }
        }

        if (is_complex) {
            SimComplexDouble* row_ptr = dst_complex + offset;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vDSP_vsmaD(state->vdsp_value, 1, &re_scale, row_re, 2, row_re, 2, len);
            if (im_scale != 0.0) {
                double neg_im = -im_scale;
                vDSP_vsmaD(state->vdsp_aux, 1, &neg_im, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_value, 1, &im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_aux, 1, &re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static double
gabor_drive_time(SimStimulusGaborState* state, double base_time, double dt, size_t step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_GABOR_EPS) {
                return ((double) step_index) * state->config.nominal_dt + state->config.time_offset;
            }
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_ACCUMULATED_STATEFUL:
        default:
            break;
    }

    if (!state->config.fixed_clock) {
        state->clock_initialized = false;
        return current_time;
    }

    double increment = (state->config.nominal_dt > STIM_GABOR_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

static void gabor_eval_base_1d(const SimStimulusGaborConfig* cfg,
                               double                        u,
                               double                        t,
                               double                        center,
                               double                        sigma,
                               double                        k,
                               double                        omega,
                               double                        phase,
                               double*                       out_re,
                               double*                       out_im) {
    double sigma_use        = (sigma > STIM_GABOR_EPS) ? sigma : 1.0;
    double inv_two_sigma_sq = 0.5 / (sigma_use * sigma_use);
    double diff             = u - center;
    double envelope         = exp(-diff * diff * inv_two_sigma_sq);
    double theta            = k * u - omega * t + phase;
    double s                = 0.0;
    double c                = 0.0;
    stim_gabor_sincos(theta, &s, &c);

    if (out_re != NULL)
        *out_re = envelope * c;
    if (out_im != NULL)
        *out_im = envelope * s;
}

static void gabor_eval_envelope(const SimStimulusGaborConfig* cfg,
                                const SimField*               field,
                                double*                       out,
                                size_t                        count,
                                double                        t) {
    if (!cfg || !out || count == 0U)
        return;

    if (cfg->amplitude == 0.0) {
        memset(out, 0, count * sizeof(double));
        return;
    }

    double sigma_x            = (cfg->sigma_x > STIM_GABOR_EPS) ? cfg->sigma_x : 1.0;
    double sigma_y            = (cfg->sigma_y > STIM_GABOR_EPS) ? cfg->sigma_y : sigma_x;
    double inv_two_sigma_x_sq = 0.5 / (sigma_x * sigma_x);
    double inv_two_sigma_y_sq = 0.5 / (sigma_y * sigma_y);
    double center_x           = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double center_y           = cfg->coord.center_y + cfg->coord.velocity_y * t;
    double angle              = cfg->coord.angle;
    double sin_a              = 0.0;
    double cos_a              = 1.0;

    if (angle != 0.0) {
        stim_gabor_sincos(angle, &sin_a, &cos_a);
    }

    for (size_t i = 0U; i < count; ++i) {
        double x = 0.0;
        double y = 0.0;
        if (field != NULL &&
            sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            out[i] = 0.0;
            continue;
        }
        double dx = x - center_x;
        double dy = y - center_y;
        double xr = dx;
        double yr = dy;
        if (angle != 0.0) {
            xr = cos_a * dx + sin_a * dy;
            yr = -sin_a * dx + cos_a * dy;
        }
        double exponent = -(xr * xr * inv_two_sigma_x_sq + yr * yr * inv_two_sigma_y_sq);
        double envelope = exp(exponent);
        out[i]          = cfg->amplitude * envelope;
    }
}

static SimResult gabor_step(void*               state_ptr,
                            struct SimContext*  context,
                            struct SimOperator* self,
                            size_t              substep_index,
                            double              dt_sub,
                            void*               scratch,
                            size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    SimStimulusGaborState* state = (SimStimulusGaborState*) state_ptr;
    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    void* raw_data = sim_field_data(field);
    if (!raw_data)
        return SIM_RESULT_INVALID_ARGUMENT;

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (!is_complex) {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        if (field->element_size != sizeof(SimComplexDouble))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double t          = gabor_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U || state->config.amplitude == 0.0) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;

#if defined(SIM_HAVE_VDSP)
    if (gabor_try_vdsp_linear_rows(state,
                                   field,
                                   is_complex,
                                   is_complex ? NULL : (double*) raw_data,
                                   is_complex ? sim_field_complex_data(field) : NULL,
                                   count,
                                   scale,
                                   t)) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }
#endif

    double k              = state->config.wavenumber;
    double omega          = state->config.omega;
    double phase0         = state->config.phase;
    bool   separable      = (state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool   use_wavevector = state->config.use_wavevector;
    double center_x       = 0.0;
    double center_y       = 0.0;
    double sigma_x        = 1.0;
    double sigma_y        = 1.0;

    if (separable) {
        center_x = state->config.coord.center_x;
        center_y = state->config.coord.center_y;
        sigma_x  = (state->config.sigma_x > STIM_GABOR_EPS) ? state->config.sigma_x : 1.0;
        sigma_y  = (state->config.sigma_y > STIM_GABOR_EPS) ? state->config.sigma_y : 1.0;
    }

    if (!separable || use_wavevector) {
        SimResult prep = gabor_ensure_envelope(state, count);
        if (prep != SIM_RESULT_OK)
            return prep;

        gabor_eval_envelope(&state->config, field, state->envelope, count, t);
    }

    if (!is_complex) {
        double* dst = (double*) raw_data;
        for (size_t i = 0U; i < count; ++i) {
            double x        = 0.0;
            double y        = 0.0;
            double sample_x = 0.0;
            double sample_y = 0.0;
            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                continue;
            }
            sim_stimulus_coord_sample_xy(&state->config.coord, x, y, t, &sample_x, &sample_y);

            double value = 0.0;
            if (use_wavevector) {
                double spatial = state->config.kx * sample_x + state->config.ky * sample_y;
                double theta   = spatial - omega * t + phase0;
                value          = state->envelope[i] * cos(theta);
            } else if (separable) {
                double fx_re = 0.0;
                double fy_re = 0.0;
                gabor_eval_base_1d(
                    &state->config, sample_x, t, center_x, sigma_x, k, omega, phase0, &fx_re, NULL);
                gabor_eval_base_1d(
                    &state->config, sample_y, t, center_y, sigma_y, k, omega, phase0, &fy_re, NULL);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    value = state->config.amplitude * (fx_re + fy_re);
                } else {
                    value = state->config.amplitude * (fx_re * fy_re);
                }
            } else {
                double u       = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                double spatial = k * u;
                double theta   = spatial - omega * t + phase0;
                value          = state->envelope[i] * cos(theta);
            }
            if (isfinite(value)) {
                dst[i] += scale * value;
            }
        }
    } else {
        SimComplexDouble* dst      = sim_field_complex_data(field);
        double            rotation = state->config.rotation;
        double            sin_r    = 0.0;
        double            cos_r    = 1.0;
        if (rotation != 0.0)
            stim_gabor_sincos(rotation, &sin_r, &cos_r);

        for (size_t i = 0U; i < count; ++i) {
            double x        = 0.0;
            double y        = 0.0;
            double sample_x = 0.0;
            double sample_y = 0.0;
            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                continue;
            }
            sim_stimulus_coord_sample_xy(&state->config.coord, x, y, t, &sample_x, &sample_y);

            double re = 0.0;
            double im = 0.0;
            if (use_wavevector) {
                double spatial = state->config.kx * sample_x + state->config.ky * sample_y;
                double theta   = spatial - omega * t + phase0;
                double s       = 0.0;
                double c       = 0.0;
                stim_gabor_sincos(theta, &s, &c);
                double env = state->envelope[i];
                re         = env * c;
                im         = env * s;
            } else if (separable) {
                double fx_re = 0.0;
                double fx_im = 0.0;
                double fy_re = 0.0;
                double fy_im = 0.0;
                gabor_eval_base_1d(&state->config,
                                   sample_x,
                                   t,
                                   center_x,
                                   sigma_x,
                                   k,
                                   omega,
                                   phase0,
                                   &fx_re,
                                   &fx_im);
                gabor_eval_base_1d(&state->config,
                                   sample_y,
                                   t,
                                   center_y,
                                   sigma_y,
                                   k,
                                   omega,
                                   phase0,
                                   &fy_re,
                                   &fy_im);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = state->config.amplitude * (fx_re + fy_re);
                    im = state->config.amplitude * (fx_im + fy_im);
                } else {
                    re = state->config.amplitude * (fx_re * fy_re - fx_im * fy_im);
                    im = state->config.amplitude * (fx_re * fy_im + fx_im * fy_re);
                }
            } else {
                double u       = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                double spatial = k * u;
                double theta   = spatial - omega * t + phase0;
                double s, c;
                stim_gabor_sincos(theta, &s, &c);
                double env = state->envelope[i];
                re         = env * c;
                im         = env * s;
            }

            double out_re = re * cos_r - im * sin_r;
            double out_im = re * sin_r + im * cos_r;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += scale * out_re;
                dst[i].im += scale * out_im;
            }
        }
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_gabor_operator(struct SimContext*            context,
                                          const SimStimulusGaborConfig* config,
                                          size_t*                       out_index) {
    if (!context)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusGaborConfig local = { 0 };
    if (config)
        local = *config;

    gabor_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "stimulus_gabor", (config != NULL), (config != NULL) ? config->scale_by_dt : true);

    SimStimulusGaborState* state =
        (SimStimulusGaborState*) calloc(1U, sizeof(SimStimulusGaborState));
    if (!state)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config            = local;
    state->clock_mode        = gabor_resolve_clock_mode(context, "stimulus_gabor", &state->config);
    state->locked_time       = 0.0;
    state->last_step_index   = 0U;
    state->clock_initialized = false;
    state->envelope          = NULL;
    state->envelope_capacity = 0U;
    gabor_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_gabor");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stimulus_gabor";
    sim_operator_info_set_schema_identity(&info, "stimulus_gabor");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    bool needs_complex =
        sim_field_is_complex(sim_context_field(context, state->config.field_index));

    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_gabor")) {
        SimField* field = sim_context_field(context, local.field_index);
        if (field != NULL) {
            bool is_complex = sim_field_is_complex(field);
            if ((!is_complex && field->element_size != sizeof(double)) ||
                (is_complex && field->element_size != sizeof(SimComplexDouble))) {
                field = NULL;
            }
        }
        if (field != NULL) {
            size_t rank    = sim_field_rank(field);
            bool   axis_ok = (local.coord.mode == SIM_STIMULUS_COORD_AXIS &&
                              local.coord.axis == SIM_STIMULUS_AXIS_X);
            bool   y_ok    = (fabs(local.coord.origin_y) <= STIM_GABOR_EPS &&
                              fabs(local.coord.center_y) <= STIM_GABOR_EPS &&
                              fabs(local.coord.velocity_y) <= STIM_GABOR_EPS &&
                              fabs(local.coord.angle) <= STIM_GABOR_EPS);
            if (rank != 1U || !axis_ok || local.use_wavevector || !y_ok) {
                field = NULL;
            }
        }
        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };
                StimulusGaborIRParams              ir_params   = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId delta      = gabor_build_ir(builder, &local, is_complex, &ir_params);
                SimIRNodeId sum = gabor_ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

                if (field_node != SIM_IR_INVALID_NODE && delta != SIM_IR_INVALID_NODE &&
                    sum != SIM_IR_INVALID_NODE) {
                    int max_param = -1;
                    if (ir_params.needs_dt) {
                        max_param = (int) SIM_IR_PARAM_DT;
                    }
                    if (ir_params.needs_step_index && (int) SIM_IR_PARAM_STEP_INDEX > max_param) {
                        max_param = (int) SIM_IR_PARAM_STEP_INDEX;
                    }
                    if (ir_params.needs_time && (int) SIM_IR_PARAM_TIME > max_param) {
                        max_param = (int) SIM_IR_PARAM_TIME;
                    }
                    size_t param_count = (max_param >= 0) ? (size_t) max_param + 1U : 0U;

                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = sum;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = param_count;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = gabor_destroy;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    if (local.field_index < 64U) {
                        kdesc.read_mask |= (1ULL << local.field_index);
                        kdesc.write_mask |= (1ULL << local.field_index);
                    }

                    result = sim_context_register_operator(context, &kdesc, out_index);
                    if (result == SIM_RESULT_OK) {
                        registered_kernel = true;
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = gabor_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = gabor_symbolic,
                                .destroy       = gabor_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        gabor_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_gabor_config(struct SimContext*      context,
                                    size_t                  operator_index,
                                    SimStimulusGaborConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusGaborState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusGaborState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusGaborState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_gabor_update(struct SimContext*            context,
                                    size_t                        operator_index,
                                    const SimStimulusGaborConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusGaborState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusGaborState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusGaborState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    SimStimulusGaborConfig local = state->config;
    if (config)
        local = *config;

    gabor_normalize(&local);
    state->config     = local;
    state->clock_mode = gabor_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_gabor"), &state->config);
    gabor_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
