#include "oakfield/operators/stimulus/heat_kernel.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

#include "oakfield/field.h"
#include "sim_accel.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STIM_HEAT_KERNEL_STATE_MAGIC 0x48544b4cu
#define STIM_HEAT_KERNEL_EPS 1.0e-9
#define STIM_HEAT_KERNEL_VDSP_MIN_LEN 64U

#if defined(__APPLE__)
static inline void stim_heat_kernel_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void stim_heat_kernel_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void stim_heat_kernel_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

typedef struct SimStimulusHeatKernelState {
    uint32_t                    magic;
    SimStimulusHeatKernelConfig config;
    SimClockMode                clock_mode;
    double                      locked_time;
    size_t                      last_step_index;
    bool                        clock_initialized;
    double*                     buffer;
    size_t                      buffer_capacity;
    char                        symbolic[192];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusHeatKernelState;

typedef struct StimulusHeatKernelIRParams {
    bool needs_dt;
    bool needs_step_index;
    bool needs_time;
} StimulusHeatKernelIRParams;

static double heat_kernel_base_sigma(double sigma, double fallback) {
    if (!isfinite(sigma) || sigma <= STIM_HEAT_KERNEL_EPS) {
        return (fallback > STIM_HEAT_KERNEL_EPS) ? fallback : 1.0;
    }
    return sigma;
}

static double heat_kernel_effective_time(double t) {
    return (t > 0.0) ? t : 0.0;
}

static double heat_kernel_sigma_sq(double sigma0, double diffusivity, double t) {
    double sigma_sq = sigma0 * sigma0 + 2.0 * diffusivity * heat_kernel_effective_time(t);
    if (!isfinite(sigma_sq) || sigma_sq <= STIM_HEAT_KERNEL_EPS) {
        return sigma0 * sigma0;
    }
    return sigma_sq;
}

static void heat_kernel_normalize(SimStimulusHeatKernelConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->diffusivity)) {
        config->diffusivity = 0.0;
    }
    config->diffusivity = fabs(config->diffusivity);
    if (!isfinite(config->sigma_x) || config->sigma_x <= STIM_HEAT_KERNEL_EPS) {
        config->sigma_x = 0.0;
    }
    if (!isfinite(config->sigma_y) || config->sigma_y <= STIM_HEAT_KERNEL_EPS) {
        config->sigma_y = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0) {
        config->nominal_dt = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static SimClockMode heat_kernel_resolve_clock_mode(const SimContext*                  context,
                                                   const char*                        op_name,
                                                   const SimStimulusHeatKernelConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_HEAT_KERNEL_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_heat_kernel");
    }
    return resolved;
}

static void heat_kernel_refresh_symbolic(SimStimulusHeatKernelState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusHeatKernelConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "heat_kernel A=%.3g D=%.3g sigma=(%.3g,%.3g)%s",
                    cfg->amplitude,
                    cfg->diffusivity,
                    heat_kernel_base_sigma(cfg->sigma_x, 1.0),
                    heat_kernel_base_sigma(cfg->sigma_y, cfg->sigma_x),
                    cfg->preserve_mass ? " preserve_mass" : "");
#else
    (void) state;
#endif
}

static const char* heat_kernel_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusHeatKernelState* state = (const SimStimulusHeatKernelState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimIRNodeId
heat_kernel_ir_binary(SimIRBuilder* builder, SimIRNodeType type, SimIRNodeId lhs, SimIRNodeId rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId
heat_kernel_ir_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    if (operand == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_call(builder, kind, operand);
}

static SimIRNodeId heat_kernel_ir_pow(SimIRBuilder* builder, SimIRNodeId base, double exponent) {
    SimIRNodeId exponent_node = sim_ir_builder_constant(builder, exponent);
    return heat_kernel_ir_binary(builder, SIM_IR_NODE_POW, base, exponent_node);
}

static SimIRNodeId heat_kernel_build_ir(SimIRBuilder*                      builder,
                                        const SimStimulusHeatKernelConfig* config,
                                        bool                               complex_output,
                                        StimulusHeatKernelIRParams*        params) {
    if (builder == NULL || config == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt         = false;
    params->needs_step_index = false;
    params->needs_time       = false;

    SimIRNodeId index    = sim_ir_builder_index(builder);
    SimIRNodeId spacing  = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin   = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId x_offset = heat_kernel_ir_binary(builder, SIM_IR_NODE_MUL, spacing, index);
    SimIRNodeId x        = heat_kernel_ir_binary(builder, SIM_IR_NODE_ADD, origin, x_offset);
    if (x == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    bool needs_dt =
        config->scale_by_dt || (config->fixed_clock && config->nominal_dt <= STIM_HEAT_KERNEL_EPS);
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
        if (config->nominal_dt > STIM_HEAT_KERNEL_EPS) {
            increment = sim_ir_builder_constant(builder, config->nominal_dt);
        } else {
            increment = dt_scaled;
        }
        SimIRNodeId scaled_step =
            heat_kernel_ir_binary(builder, SIM_IR_NODE_MUL, step_index, increment);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = heat_kernel_ir_binary(builder, SIM_IR_NODE_ADD, scaled_step, time_offset);
    } else {
        params->needs_time      = true;
        SimIRNodeId time_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_TIME);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = heat_kernel_ir_binary(builder, SIM_IR_NODE_ADD, time_node, time_offset);
    }
    if (t == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId t_abs = heat_kernel_ir_call(builder, SIM_IR_CALL_ABS, t);
    SimIRNodeId t_pos =
        heat_kernel_ir_binary(builder,
                              SIM_IR_NODE_MUL,
                              sim_ir_builder_constant(builder, 0.5),
                              heat_kernel_ir_binary(builder, SIM_IR_NODE_ADD, t, t_abs));

    SimIRNodeId center_base = sim_ir_builder_constant(builder, config->coord.center_x);
    SimIRNodeId velocity    = sim_ir_builder_constant(builder, config->coord.velocity_x);
    SimIRNodeId center =
        heat_kernel_ir_binary(builder,
                              SIM_IR_NODE_ADD,
                              center_base,
                              heat_kernel_ir_binary(builder, SIM_IR_NODE_MUL, velocity, t));
    SimIRNodeId diff = heat_kernel_ir_binary(builder, SIM_IR_NODE_SUB, x, center);

    double      sigma0_value = heat_kernel_base_sigma(config->sigma_x, 1.0);
    SimIRNodeId sigma0_sq    = sim_ir_builder_constant(builder, sigma0_value * sigma0_value);
    SimIRNodeId sigma_sq     = heat_kernel_ir_binary(
        builder,
        SIM_IR_NODE_ADD,
        sigma0_sq,
        heat_kernel_ir_binary(builder,
                              SIM_IR_NODE_MUL,
                              sim_ir_builder_constant(builder, 2.0 * config->diffusivity),
                              t_pos));
    SimIRNodeId diff_sq = heat_kernel_ir_binary(builder, SIM_IR_NODE_MUL, diff, diff);
    SimIRNodeId exponent =
        heat_kernel_ir_binary(builder,
                              SIM_IR_NODE_MUL,
                              sim_ir_builder_constant(builder, -0.5),
                              heat_kernel_ir_binary(builder, SIM_IR_NODE_DIV, diff_sq, sigma_sq));
    SimIRNodeId envelope = heat_kernel_ir_call(builder, SIM_IR_CALL_EXP, exponent);

    SimIRNodeId norm = sim_ir_builder_constant(builder, 1.0);
    if (config->preserve_mass) {
        SimIRNodeId sigma_eff = heat_kernel_ir_pow(builder, sigma_sq, 0.5);
        norm                  = heat_kernel_ir_binary(
            builder, SIM_IR_NODE_DIV, sim_ir_builder_constant(builder, sigma0_value), sigma_eff);
    }

    SimIRNodeId amplitude = sim_ir_builder_constant(builder, config->amplitude);
    SimIRNodeId value =
        heat_kernel_ir_binary(builder,
                              SIM_IR_NODE_MUL,
                              amplitude,
                              heat_kernel_ir_binary(builder, SIM_IR_NODE_MUL, norm, envelope));
    if (value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId scale = config->scale_by_dt ? dt_scaled : sim_ir_builder_constant(builder, 1.0);
    value             = heat_kernel_ir_binary(builder, SIM_IR_NODE_MUL, value, scale);

    if (complex_output && value != SIM_IR_INVALID_NODE) {
        SimIRNodeId packed =
            sim_ir_builder_complex_pack(builder, value, sim_ir_builder_constant(builder, 0.0));
        if (packed != SIM_IR_INVALID_NODE && config->rotation != 0.0) {
            packed = sim_ir_builder_complex_rotate(
                builder, packed, sim_ir_builder_constant(builder, config->rotation));
        }
        value = packed;
    }

    return value;
}

static void heat_kernel_destroy(void* state_ptr) {
    SimStimulusHeatKernelState* state = (SimStimulusHeatKernelState*) state_ptr;
    if (state == NULL) {
        return;
    }
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
#endif
    free(state->buffer);
    free(state);
}

static SimResult heat_kernel_ensure_buffer(SimStimulusHeatKernelState* state, size_t count) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (count == 0U || state->buffer_capacity >= count) {
        return SIM_RESULT_OK;
    }

    double* resized = (double*) realloc(state->buffer, count * sizeof(double));
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->buffer          = resized;
    state->buffer_capacity = count;
    return SIM_RESULT_OK;
}

static double heat_kernel_drive_time(SimStimulusHeatKernelState* state,
                                     double                      base_time,
                                     double                      dt,
                                     size_t                      step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_HEAT_KERNEL_EPS) {
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

    double increment =
        (state->config.nominal_dt > STIM_HEAT_KERNEL_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

#if defined(SIM_HAVE_VDSP)
static bool heat_kernel_vdsp_ensure_buffers(SimStimulusHeatKernelState* state, size_t width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }
    if (width > SIZE_MAX / (2U * sizeof(double))) {
        return false;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 2U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = width;
    state->vdsp_value    = block;
    state->vdsp_work     = block + width;
    return true;
}

static bool
heat_kernel_linear_map(const SimStimulusHeatKernelConfig* cfg, double* out_u_x, double* out_u_y) {
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
        case SIM_STIMULUS_COORD_ANGLE: {
            double s = sin(cfg->coord.angle);
            double c = cos(cfg->coord.angle);
            *out_u_x = c;
            *out_u_y = s;
            return true;
        }
        default:
            break;
    }

    return false;
}

static bool heat_kernel_try_vdsp_linear_rows(SimStimulusHeatKernelState* state,
                                             const SimField*             field,
                                             bool                        is_complex,
                                             double*                     dst_real,
                                             SimComplexDouble*           dst_complex,
                                             size_t                      count,
                                             double                      scale,
                                             double                      t) {
    if (state == NULL || field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_HEAT_KERNEL_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!heat_kernel_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusHeatKernelConfig* cfg             = &state->config;
    double                             output_re_scale = scale;
    double                             output_im_scale = 0.0;
    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            stim_heat_kernel_sincos(cfg->rotation, &sin_r, &cos_r);
        }
        output_re_scale = scale * cos_r;
        output_im_scale = scale * sin_r;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    double sigma0_x      = heat_kernel_base_sigma(cfg->sigma_x, 1.0);
    double sigma0_y      = heat_kernel_base_sigma(cfg->sigma_y, sigma0_x);
    double sigma_sq_x    = heat_kernel_sigma_sq(sigma0_x, cfg->diffusivity, t);
    double sigma_sq_y    = heat_kernel_sigma_sq(sigma0_y, cfg->diffusivity, t);
    double sigma_eff_x   = sqrt(sigma_sq_x);
    double sigma_eff_y   = sqrt(sigma_sq_y);
    double norm_x        = cfg->preserve_mass ? (sigma0_x / sigma_eff_x) : 1.0;
    double norm_y        = cfg->preserve_mass ? (sigma0_y / sigma_eff_y) : 1.0;
    double inv_two_sig_x = -0.5 / sigma_sq_x;
    double inv_two_sig_y = -0.5 / sigma_sq_y;
    double x0_raw        = cfg->coord.origin_x;
    double y0_raw        = cfg->coord.origin_y;
    double x0_sample     = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0_sample     = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx            = cfg->coord.spacing_x;
    double dy            = cfg->coord.spacing_y;
    double center_x      = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double center_y      = cfg->coord.center_y + cfg->coord.velocity_y * t;

    if (!isfinite(norm_x) || !isfinite(norm_y) || !isfinite(inv_two_sig_x) ||
        !isfinite(inv_two_sig_y) || !isfinite(x0_raw) || !isfinite(y0_raw) ||
        !isfinite(x0_sample) || !isfinite(y0_sample) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(center_x) || !isfinite(center_y)) {
        return false;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double x_start = x0_raw - center_x;
        for (size_t row = 0U; row < height; ++row) {
            double y      = y0_raw + (double) row * dy;
            double diff_y = y - center_y;
            double env_y  = norm_y * exp(diff_y * diff_y * inv_two_sig_y);
            if (!isfinite(y) || !isfinite(diff_y) || !isfinite(env_y)) {
                return false;
            }

            vDSP_vrampD(&x_start, &dx, state->vdsp_value, 1, len);
            vDSP_vsqD(state->vdsp_value, 1, state->vdsp_work, 1, len);
            sim_accel_scale_inplace_real(state->vdsp_work, width, inv_two_sig_x);
            vvexp(state->vdsp_value, state->vdsp_work, &vforce_len);
            sim_accel_scale_inplace_real(state->vdsp_value, width, norm_x);

            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                sim_accel_add_scalar_real(state->vdsp_value, width, env_y);
            } else {
                sim_accel_scale_inplace_real(state->vdsp_value, width, env_y);
            }
            sim_accel_scale_inplace_real(state->vdsp_value, width, cfg->amplitude);

            if (!is_complex) {
                sim_accel_copy_scale_real(
                    state->vdsp_value, dst_real + row * width, width, scale, true);
            } else {
                sim_accel_accumulate_real_to_complex(state->vdsp_value,
                                                     dst_complex + row * width,
                                                     width,
                                                     output_re_scale,
                                                     output_im_scale);
            }
        }
        return true;
    }

    {
        double u_x = 0.0;
        double u_y = 0.0;
        if (!heat_kernel_linear_map(cfg, &u_x, &u_y)) {
            return false;
        }

        double start_x = u_x * x0_sample - center_x;
        double step_x  = u_x * dx;
        if (!isfinite(u_x) || !isfinite(u_y) || !isfinite(start_x) || !isfinite(step_x)) {
            return false;
        }

        for (size_t row = 0U; row < height; ++row) {
            double sample_y = y0_sample + (double) row * dy;
            double start    = start_x + u_y * sample_y;
            if (!isfinite(sample_y) || !isfinite(start)) {
                return false;
            }

            vDSP_vrampD(&start, &step_x, state->vdsp_value, 1, len);
            vDSP_vsqD(state->vdsp_value, 1, state->vdsp_work, 1, len);
            sim_accel_scale_inplace_real(state->vdsp_work, width, inv_two_sig_x);
            vvexp(state->vdsp_value, state->vdsp_work, &vforce_len);
            sim_accel_scale_inplace_real(state->vdsp_value, width, cfg->amplitude * norm_x);

            if (!is_complex) {
                sim_accel_copy_scale_real(
                    state->vdsp_value, dst_real + row * width, width, scale, true);
            } else {
                sim_accel_accumulate_real_to_complex(state->vdsp_value,
                                                     dst_complex + row * width,
                                                     width,
                                                     output_re_scale,
                                                     output_im_scale);
            }
        }
    }

    return true;
}
#endif

static void heat_kernel_eval(const SimStimulusHeatKernelConfig* cfg,
                             const SimField*                    field,
                             double*                            out,
                             size_t                             count,
                             double                             t) {
    if (cfg == NULL || out == NULL) {
        return;
    }

    bool   separable     = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double sigma0_x      = heat_kernel_base_sigma(cfg->sigma_x, 1.0);
    double sigma0_y      = heat_kernel_base_sigma(cfg->sigma_y, sigma0_x);
    double sigma_sq_x    = heat_kernel_sigma_sq(sigma0_x, cfg->diffusivity, t);
    double sigma_sq_y    = heat_kernel_sigma_sq(sigma0_y, cfg->diffusivity, t);
    double sigma_eff_x   = sqrt(sigma_sq_x);
    double sigma_eff_y   = sqrt(sigma_sq_y);
    double norm_x        = cfg->preserve_mass ? (sigma0_x / sigma_eff_x) : 1.0;
    double norm_y        = cfg->preserve_mass ? (sigma0_y / sigma_eff_y) : 1.0;
    double center_x      = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double center_y      = cfg->coord.center_y + cfg->coord.velocity_y * t;
    double inv_two_sig_x = 0.5 / sigma_sq_x;
    double inv_two_sig_y = 0.5 / sigma_sq_y;

    for (size_t i = 0U; i < count; ++i) {
        double x = 0.0;
        double y = 0.0;
        if (field != NULL &&
            sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            out[i] = 0.0;
            continue;
        }

        if (separable) {
            double dx    = x - center_x;
            double dy    = y - center_y;
            double env_x = norm_x * exp(-(dx * dx) * inv_two_sig_x);
            double env_y = norm_y * exp(-(dy * dy) * inv_two_sig_y);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                out[i] = cfg->amplitude * (env_x + env_y);
            } else {
                out[i] = cfg->amplitude * (env_x * env_y);
            }
        } else {
            double u    = sim_stimulus_coord_u(&cfg->coord, x, y, t);
            double diff = u;
            if (cfg->coord.mode != SIM_STIMULUS_COORD_RADIAL &&
                cfg->coord.mode != SIM_STIMULUS_COORD_POLAR &&
                cfg->coord.mode != SIM_STIMULUS_COORD_AZIMUTH &&
                cfg->coord.mode != SIM_STIMULUS_COORD_ELLIPTIC &&
                cfg->coord.mode != SIM_STIMULUS_COORD_SPIRAL) {
                diff = u - center_x;
            }
            out[i] = cfg->amplitude * norm_x * exp(-(diff * diff) * inv_two_sig_x);
        }
    }
}

static SimResult heat_kernel_step(void*               state_ptr,
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

    SimStimulusHeatKernelState* state = (SimStimulusHeatKernelState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    void* raw_data = sim_field_data(field);
    if (raw_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count      = 0U;
    bool   is_complex = false;
    if (field->element_size == sizeof(double)) {
        count = sim_field_bytes(field) / sizeof(double);
    } else if (field->element_size == sizeof(SimComplexDouble)) {
        count      = sim_field_bytes(field) / sizeof(SimComplexDouble);
        is_complex = true;
    } else {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double drive_time = heat_kernel_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U || state->config.amplitude == 0.0) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;
#if defined(SIM_HAVE_VDSP)
    if (heat_kernel_try_vdsp_linear_rows(state,
                                         field,
                                         is_complex,
                                         sim_field_real_data(field),
                                         sim_field_complex_data(field),
                                         count,
                                         scale,
                                         drive_time)) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }
#endif

    SimResult prep = heat_kernel_ensure_buffer(state, count);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    heat_kernel_eval(&state->config, field, state->buffer, count, drive_time);

    if (!is_complex) {
        double* dst = (double*) raw_data;
        for (size_t i = 0U; i < count; ++i) {
            dst[i] += scale * state->buffer[i];
        }
    } else {
        SimComplexDouble* dst   = sim_field_complex_data(field);
        double            sin_r = 0.0;
        double            cos_r = 1.0;
        if (state->config.rotation != 0.0) {
            stim_heat_kernel_sincos(state->config.rotation, &sin_r, &cos_r);
        }

        for (size_t i = 0U; i < count; ++i) {
            double value = scale * state->buffer[i];
            dst[i].re += value * cos_r;
            dst[i].im += value * sin_r;
        }
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_heat_kernel_operator(struct SimContext*                 context,
                                                const SimStimulusHeatKernelConfig* config,
                                                size_t*                            out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusHeatKernelConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }
    heat_kernel_normalize(&local);

    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_heat_kernel",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusHeatKernelState* state =
        (SimStimulusHeatKernelState*) calloc(1U, sizeof(SimStimulusHeatKernelState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->magic  = STIM_HEAT_KERNEL_STATE_MAGIC;
    state->config = local;
    state->clock_mode =
        heat_kernel_resolve_clock_mode(context, "stimulus_heat_kernel", &state->config);
    state->locked_time       = 0.0;
    state->last_step_index   = 0U;
    state->clock_initialized = false;
    state->buffer            = NULL;
    state->buffer_capacity   = 0U;
    heat_kernel_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_heat_kernel");

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
    info.abstract_id       = "stimulus_heat_kernel";
    sim_operator_info_set_schema_identity(&info, "stimulus_heat_kernel");
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
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_heat_kernel")) {
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
            bool   y_ok    = (fabs(local.coord.origin_y) <= STIM_HEAT_KERNEL_EPS &&
                              fabs(local.coord.center_y) <= STIM_HEAT_KERNEL_EPS &&
                              fabs(local.coord.velocity_y) <= STIM_HEAT_KERNEL_EPS);
            if (rank != 1U || !axis_ok || !y_ok) {
                field = NULL;
            }
        }
        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };
                StimulusHeatKernelIRParams         ir_params   = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId delta = heat_kernel_build_ir(builder, &local, is_complex, &ir_params);
                SimIRNodeId sum =
                    heat_kernel_ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

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
                    kdesc.destroy               = heat_kernel_destroy;
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
                                .fn                = heat_kernel_step,
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
                                .symbolic      = heat_kernel_symbolic,
                                .destroy       = heat_kernel_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        heat_kernel_destroy(state);
    }

    return result;
}

static SimStimulusHeatKernelState* heat_kernel_state_from_operator(SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    if (op->kernel != NULL) {
        return (SimStimulusHeatKernelState*) sim_operator_payload(op);
    }
    return (SimStimulusHeatKernelState*) sim_split_state(op);
}

SimResult sim_stimulus_heat_kernel_config(struct SimContext*           context,
                                          size_t                       operator_index,
                                          SimStimulusHeatKernelConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusHeatKernelState* state = heat_kernel_state_from_operator(op);
    if (state == NULL || state->magic != STIM_HEAT_KERNEL_STATE_MAGIC) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_heat_kernel_update(struct SimContext*                 context,
                                          size_t                             operator_index,
                                          const SimStimulusHeatKernelConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusHeatKernelState* state = heat_kernel_state_from_operator(op);
    if (state == NULL || state->magic != STIM_HEAT_KERNEL_STATE_MAGIC) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusHeatKernelConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    heat_kernel_normalize(&local);
    state->config     = local;
    state->clock_mode = heat_kernel_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_heat_kernel"), &state->config);
    heat_kernel_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
