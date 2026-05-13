#include "oakfield/operators/stimulus/sinusoidal.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"

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

#define STIM_SINUSOIDAL_RENORM_INTERVAL 256U
#define STIM_SINUSOIDAL_EPS 1.0e-12
#define STIM_SINUSOIDAL_VDSP_MIN_LEN 64U

#if defined(__APPLE__)
static inline void sincos_pair(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void sincos_pair(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void sincos_pair(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

static void sinusoidal_normalize(SimStimulusSinusoidalConfig* config,
                                 SimStimulusSinusoidalMode    mode) {
    if (!config) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->wavenumber)) {
        config->wavenumber = 0.0;
    }
    if (!isfinite(config->kx)) {
        config->kx = 0.0;
    }
    if (!isfinite(config->ky)) {
        config->ky = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);

    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0) {
        config->nominal_dt = 0.0;
    }
    if (!isfinite(config->kdot)) {
        config->kdot = 0.0;
    }
    if (!isfinite(config->wdot)) {
        config->wdot = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }
    if (mode != SIM_STIMULUS_SINUSOIDAL_CHIRP) {
        config->kdot = 0.0;
        config->wdot = 0.0;
    }
    if (!config->use_wavevector &&
        (fabs(config->kx) > STIM_SINUSOIDAL_EPS || fabs(config->ky) > STIM_SINUSOIDAL_EPS)) {
        config->use_wavevector = true;
    }
}

static SimClockMode sinusoidal_resolve_clock_mode(const SimContext*                  context,
                                                  const char*                        op_name,
                                                  const SimStimulusSinusoidalConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_SINUSOIDAL_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_sinusoidal");
    }
    return resolved;
}

static const char* sinusoidal_mode_name(SimStimulusSinusoidalMode mode) {
    switch (mode) {
        case SIM_STIMULUS_SINUSOIDAL_STANDING:
            return "standing";
        case SIM_STIMULUS_SINUSOIDAL_CHIRP:
            return "stimulus_chirp";
        case SIM_STIMULUS_SINUSOIDAL_SINE:
        default:
            return "sine";
    }
}

static void sinusoidal_refresh_symbolic(SimStimulusSinusoidalState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimStimulusSinusoidalConfig* cfg = &state->config;

    switch (state->mode) {
        case SIM_STIMULUS_SINUSOIDAL_STANDING:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "standing A=%.3g k=%.3g ω=%.3g",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->omega);
            break;
        case SIM_STIMULUS_SINUSOIDAL_CHIRP:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "chirp A=%.3g k=%.3g+%.3gt ω=%.3g+%.3gt",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->kdot,
                            cfg->omega,
                            cfg->wdot);
            break;
        case SIM_STIMULUS_SINUSOIDAL_SINE:
        default:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "sine A=%.3g k=%.3g ω=%.3g φ=%.3g",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->omega,
                            cfg->phase);
            break;
    }
#else
    (void) state;
#endif
}

static const char* sinusoidal_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusSinusoidalState* state = (const SimStimulusSinusoidalState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

typedef struct StimulusIRParams {
    bool needs_dt;
    bool needs_step_index;
    bool needs_time;
} StimulusIRParams;

static SimIRNodeId
ir_binary(SimIRBuilder* builder, SimIRNodeType type, SimIRNodeId lhs, SimIRNodeId rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId ir_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    if (operand == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_call(builder, kind, operand);
}

static SimIRNodeId sinusoidal_build_ir(SimIRBuilder*                      builder,
                                       const SimStimulusSinusoidalConfig* config,
                                       SimStimulusSinusoidalMode          mode,
                                       StimulusIRParams*                  params) {
    if (builder == NULL || config == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt         = false;
    params->needs_step_index = false;
    params->needs_time       = false;

    SimIRNodeId index    = sim_ir_builder_index(builder);
    SimIRNodeId spacing  = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin   = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId x_offset = ir_binary(builder, SIM_IR_NODE_MUL, spacing, index);
    SimIRNodeId x        = ir_binary(builder, SIM_IR_NODE_ADD, origin, x_offset);
    if (x == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    bool needs_dt =
        config->scale_by_dt || (config->fixed_clock && config->nominal_dt <= STIM_SINUSOIDAL_EPS);
    SimIRNodeId dt_node   = SIM_IR_INVALID_NODE;
    SimIRNodeId dt_scaled = SIM_IR_INVALID_NODE;
    if (needs_dt) {
        params->needs_dt = true;
        dt_node          = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
        dt_scaled        = dt_node;
    }

    SimIRNodeId t = SIM_IR_INVALID_NODE;
    if (config->fixed_clock) {
        params->needs_step_index = true;
        SimIRNodeId step_index   = sim_ir_builder_param(builder, SIM_IR_PARAM_STEP_INDEX);
        SimIRNodeId increment    = SIM_IR_INVALID_NODE;
        if (config->nominal_dt > STIM_SINUSOIDAL_EPS) {
            increment = sim_ir_builder_constant(builder, config->nominal_dt);
        } else {
            increment = dt_scaled;
        }
        SimIRNodeId scaled_step = ir_binary(builder, SIM_IR_NODE_MUL, step_index, increment);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t                       = ir_binary(builder, SIM_IR_NODE_ADD, scaled_step, time_offset);
    } else {
        params->needs_time      = true;
        SimIRNodeId time_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_TIME);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t                       = ir_binary(builder, SIM_IR_NODE_ADD, time_node, time_offset);
    }

    if (t == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId amplitude  = sim_ir_builder_constant(builder, config->amplitude);
    SimIRNodeId wavenumber = sim_ir_builder_constant(builder, config->wavenumber);
    SimIRNodeId omega      = sim_ir_builder_constant(builder, config->omega);
    SimIRNodeId phase      = sim_ir_builder_constant(builder, config->phase);
    SimIRNodeId value      = SIM_IR_INVALID_NODE;

    switch (mode) {
        case SIM_STIMULUS_SINUSOIDAL_STANDING: {
            SimIRNodeId omega_t     = ir_binary(builder, SIM_IR_NODE_MUL, omega, t);
            SimIRNodeId cos_time    = ir_call(builder, SIM_IR_CALL_COS, omega_t);
            SimIRNodeId amplitude_t = ir_binary(builder, SIM_IR_NODE_MUL, amplitude, cos_time);
            SimIRNodeId kx          = ir_binary(builder, SIM_IR_NODE_MUL, wavenumber, x);
            SimIRNodeId phase_x     = ir_binary(builder, SIM_IR_NODE_ADD, kx, phase);
            SimIRNodeId cos_phase   = ir_call(builder, SIM_IR_CALL_COS, phase_x);
            value                   = ir_binary(builder, SIM_IR_NODE_MUL, amplitude_t, cos_phase);
            break;
        }
        case SIM_STIMULUS_SINUSOIDAL_CHIRP: {
            SimIRNodeId kdot = sim_ir_builder_constant(builder, config->kdot);
            SimIRNodeId wdot = sim_ir_builder_constant(builder, config->wdot);
            SimIRNodeId k_t  = ir_binary(
                builder, SIM_IR_NODE_ADD, wavenumber, ir_binary(builder, SIM_IR_NODE_MUL, kdot, t));
            SimIRNodeId w_t = ir_binary(
                builder, SIM_IR_NODE_ADD, omega, ir_binary(builder, SIM_IR_NODE_MUL, wdot, t));
            SimIRNodeId kx         = ir_binary(builder, SIM_IR_NODE_MUL, k_t, x);
            SimIRNodeId wtt        = ir_binary(builder, SIM_IR_NODE_MUL, w_t, t);
            SimIRNodeId phase_t    = ir_binary(builder, SIM_IR_NODE_SUB, kx, wtt);
            SimIRNodeId phase_full = ir_binary(builder, SIM_IR_NODE_ADD, phase_t, phase);
            SimIRNodeId sin_phase  = ir_call(builder, SIM_IR_CALL_SIN, phase_full);
            value                  = ir_binary(builder, SIM_IR_NODE_MUL, amplitude, sin_phase);
            break;
        }
        case SIM_STIMULUS_SINUSOIDAL_SINE:
        default: {
            SimIRNodeId kx         = ir_binary(builder, SIM_IR_NODE_MUL, wavenumber, x);
            SimIRNodeId omega_t    = ir_binary(builder, SIM_IR_NODE_MUL, omega, t);
            SimIRNodeId phase_t    = ir_binary(builder, SIM_IR_NODE_SUB, kx, omega_t);
            SimIRNodeId phase_full = ir_binary(builder, SIM_IR_NODE_ADD, phase_t, phase);
            SimIRNodeId sin_phase  = ir_call(builder, SIM_IR_CALL_SIN, phase_full);
            value                  = ir_binary(builder, SIM_IR_NODE_MUL, amplitude, sin_phase);
            break;
        }
    }

    if (value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId scale = SIM_IR_INVALID_NODE;
    if (config->scale_by_dt) {
        scale = dt_scaled;
    } else {
        scale = sim_ir_builder_constant(builder, 1.0);
    }
    return ir_binary(builder, SIM_IR_NODE_MUL, value, scale);
}

static void sinusoidal_destroy(void* state_ptr) {
    SimStimulusSinusoidalState* state = (SimStimulusSinusoidalState*) state_ptr;
    if (!state) {
        return;
    }
    free(state->buffer);
    free(state->vdsp_block);
    free(state);
}

static SimResult
sinusoidal_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusSinusoidalState* state = (SimStimulusSinusoidalState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->snapshot_locked_time       = state->locked_time;
    state->snapshot_last_step_index   = state->last_step_index;
    state->snapshot_clock_initialized = state->clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult
sinusoidal_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusSinusoidalState* state = (SimStimulusSinusoidalState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->locked_time       = state->snapshot_locked_time;
    state->last_step_index   = state->snapshot_last_step_index;
    state->clock_initialized = state->snapshot_clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult sinusoidal_ensure_buffer(SimStimulusSinusoidalState* state, size_t count) {
    double* resized;

    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (state->buffer_capacity >= count) {
        return SIM_RESULT_OK;
    }

    resized = (double*) realloc(state->buffer, count * sizeof(double));
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->buffer          = resized;
    state->buffer_capacity = count;
    return SIM_RESULT_OK;
}

#if defined(SIM_HAVE_VDSP)
static bool sinusoidal_vdsp_ensure_buffers(SimStimulusSinusoidalState* state, size_t width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* resized = (double*) realloc(state->vdsp_block, width * 2U * sizeof(double));
    if (resized == NULL) {
        return false;
    }

    state->vdsp_block    = resized;
    state->vdsp_capacity = width;
    state->vdsp_theta    = resized;
    state->vdsp_value    = resized + width;
    return true;
}

static bool sinusoidal_try_vdsp_linear_rows(SimStimulusSinusoidalState* state,
                                            const SimField*             field,
                                            bool                        is_complex,
                                            double*                     dst_real,
                                            SimComplexDouble*           dst_complex,
                                            size_t                      count,
                                            double                      scale,
                                            double                      t) {
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

    const SimStimulusSinusoidalConfig* cfg            = &state->config;
    bool                               use_wavevector = cfg->use_wavevector;
    bool   separable = (!use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double basis_x_x = 0.0;
    double basis_x_y = 0.0;
    double basis_y_x = 0.0;
    double basis_y_y = 0.0;

    if (use_wavevector) {
        basis_x_x = cfg->kx;
        basis_x_y = cfg->ky;
    } else {
        switch (cfg->coord.mode) {
            case SIM_STIMULUS_COORD_AXIS:
                if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                    basis_x_x = 0.0;
                    basis_x_y = 1.0;
                } else {
                    basis_x_x = 1.0;
                    basis_x_y = 0.0;
                }
                break;
            case SIM_STIMULUS_COORD_ANGLE: {
                double s  = sin(cfg->coord.angle);
                double c  = cos(cfg->coord.angle);
                basis_x_x = c;
                basis_x_y = s;
                break;
            }
            case SIM_STIMULUS_COORD_SEPARABLE:
                basis_x_x = 1.0;
                basis_x_y = 0.0;
                basis_y_x = 0.0;
                basis_y_y = 1.0;
                break;
            default:
                return false;
        }
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_SINUSOIDAL_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!sinusoidal_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double x0       = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0       = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx       = cfg->coord.spacing_x;
    double dy       = cfg->coord.spacing_y;
    double k_t      = cfg->wavenumber;
    double w_t      = cfg->omega;
    double time_cos = 1.0;
    bool   use_cos  = (state->mode == SIM_STIMULUS_SINUSOIDAL_STANDING);

    if (state->mode == SIM_STIMULUS_SINUSOIDAL_CHIRP) {
        k_t = cfg->wavenumber + cfg->kdot * t;
        w_t = cfg->omega + cfg->wdot * t;
    } else if (state->mode == SIM_STIMULUS_SINUSOIDAL_STANDING) {
        time_cos = cos(cfg->omega * t);
    }

    double phase_bias = cfg->phase;
    if (!use_cos) {
        phase_bias -= w_t * t;
    }

    double spatial_scale = 1.0;
    if (use_wavevector && state->mode == SIM_STIMULUS_SINUSOIDAL_CHIRP) {
        double k0 = hypot(cfg->kx, cfg->ky);
        if (k0 > STIM_SINUSOIDAL_EPS) {
            spatial_scale = k_t / k0;
        }
    }

    double step_x = 0.0;
    if (use_wavevector) {
        step_x = spatial_scale * basis_x_x * dx;
    } else {
        step_x = k_t * basis_x_x * dx;
    }

    double output_re_scale = scale;
    double output_im_scale = 0.0;
    if (is_complex) {
        output_re_scale = scale * cos(cfg->rotation);
        output_im_scale = scale * sin(cfg->rotation);
    }

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(k_t) ||
        !isfinite(w_t) || !isfinite(phase_bias) || !isfinite(step_x) || !isfinite(time_cos) ||
        !isfinite(output_re_scale) || !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;

    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double start    = phase_bias;

        if (use_wavevector) {
            start += spatial_scale * (basis_x_x * x0 + basis_x_y * sample_y);
        } else {
            start += k_t * (basis_x_x * x0 + basis_x_y * sample_y);
        }
        if (!isfinite(start)) {
            return false;
        }

        vDSP_vrampD(&start, &step_x, state->vdsp_theta, 1, len);
        if (use_cos) {
            vvcos(state->vdsp_value, state->vdsp_theta, &vforce_len);
        } else {
            vvsin(state->vdsp_value, state->vdsp_theta, &vforce_len);
        }

        if (separable) {
            double y_theta = k_t * (basis_y_x * x0 + basis_y_y * sample_y) + phase_bias;
            double y_value = use_cos ? cos(y_theta) : sin(y_theta);
            if (!isfinite(y_theta) || !isfinite(y_value)) {
                return false;
            }

            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                double row_scale = cfg->amplitude;
                double row_bias  = cfg->amplitude * y_value;
                if (use_cos) {
                    row_scale *= time_cos;
                    row_bias *= time_cos;
                }
                sim_accel_scale_inplace_real(state->vdsp_value, width, row_scale);
                sim_accel_add_scalar_real(state->vdsp_value, width, row_bias);
            } else {
                double row_scale = cfg->amplitude * y_value;
                if (use_cos) {
                    row_scale *= time_cos * time_cos;
                }
                sim_accel_scale_inplace_real(state->vdsp_value, width, row_scale);
            }
        } else {
            double row_scale = cfg->amplitude;
            if (use_cos) {
                row_scale *= time_cos;
            }
            sim_accel_scale_inplace_real(state->vdsp_value, width, row_scale);
        }

        size_t offset = row * width;
        if (!is_complex) {
            double* row_ptr = dst_real + offset;
            vDSP_vsmaD(state->vdsp_value, 1, &output_re_scale, row_ptr, 1, row_ptr, 1, len);
        } else {
            SimComplexDouble* row_ptr = dst_complex + offset;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;
            vDSP_vsmaD(state->vdsp_value, 1, &output_re_scale, row_re, 2, row_re, 2, len);
            if (output_im_scale != 0.0) {
                vDSP_vsmaD(state->vdsp_value, 1, &output_im_scale, row_im, 2, row_im, 2, len);
            }
        }
    }

    return true;
}
#endif

static double sinusoidal_drive_time(SimStimulusSinusoidalState* state,
                                    double                      base_time,
                                    double                      dt,
                                    size_t                      step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE: {
            if (state->config.nominal_dt > STIM_SINUSOIDAL_EPS) {
                return ((double) step_index) * state->config.nominal_dt + state->config.time_offset;
            }
            state->clock_initialized = false;
            return current_time;
        }
        case SIM_CLOCK_ACCUMULATED_STATEFUL:
        default:
            break;
    }

    if (!state->config.fixed_clock) {
        state->clock_initialized = false;
        return current_time;
    }

    double increment =
        (state->config.nominal_dt > STIM_SINUSOIDAL_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

static void eval_sine(const SimStimulusSinusoidalConfig* cfg,
                      const SimField*                    field,
                      double*                            out,
                      size_t                             count,
                      double                             t) {
    size_t i;
    double origin  = cfg->coord.origin_x;
    double spacing = cfg->coord.spacing_x;

    if (out == NULL || count == 0U) {
        return;
    }

    if (cfg->amplitude == 0.0) {
        for (i = 0U; i < count; ++i) {
            out[i] = 0.0;
        }
        return;
    }

    size_t rank = field ? sim_field_rank(field) : 1U;
    if (rank > 1U) {
        for (i = 0U; i < count; ++i) {
            size_t ix = 0U;
            size_t iy = 0U;
            if (sim_field_index_to_xy(field, i, &ix, &iy) != SIM_RESULT_OK) {
                out[i] = 0.0;
                continue;
            }
            (void) iy;
            double x     = origin + (double) ix * spacing;
            double phase = cfg->wavenumber * x - cfg->omega * t + cfg->phase;
            out[i]       = cfg->amplitude * sin(phase);
        }
        return;
    }

    double delta  = cfg->wavenumber * spacing;
    double phase0 = cfg->wavenumber * origin - cfg->omega * t + cfg->phase;

    if (fabs(delta) < STIM_SINUSOIDAL_EPS) {
        double value = cfg->amplitude * sin(phase0);
        for (i = 0U; i < count; ++i) {
            out[i] = value;
        }
        return;
    }

    double sin_delta, cos_delta;
    sincos_pair(delta, &sin_delta, &cos_delta);

    double sin_p, cos_p;
    sincos_pair(phase0, &sin_p, &cos_p);
    unsigned int mask = STIM_SINUSOIDAL_RENORM_INTERVAL - 1U;

    for (i = 0U; i < count; ++i) {
        out[i] = cfg->amplitude * sin_p;

        double next_sin = sin_p * cos_delta + cos_p * sin_delta;
        double next_cos = cos_p * cos_delta - sin_p * sin_delta;
        sin_p           = next_sin;
        cos_p           = next_cos;

        if (((unsigned int) (i + 1U) & mask) == 0U) {
            double phase_i = phase0 + delta * (double) (i + 1U);
            sincos_pair(phase_i, &sin_p, &cos_p);
        }
    }
}

static void eval_standing(const SimStimulusSinusoidalConfig* cfg,
                          const SimField*                    field,
                          double*                            out,
                          size_t                             count,
                          double                             t) {
    size_t i;
    double origin  = cfg->coord.origin_x;
    double spacing = cfg->coord.spacing_x;

    if (out == NULL || count == 0U) {
        return;
    }

    if (cfg->amplitude == 0.0) {
        for (i = 0U; i < count; ++i) {
            out[i] = 0.0;
        }
        return;
    }

    size_t rank = field ? sim_field_rank(field) : 1U;
    if (rank > 1U) {
        double amplitude = cfg->amplitude * cos(cfg->omega * t);
        for (i = 0U; i < count; ++i) {
            size_t ix = 0U;
            size_t iy = 0U;
            if (sim_field_index_to_xy(field, i, &ix, &iy) != SIM_RESULT_OK) {
                out[i] = 0.0;
                continue;
            }
            (void) iy;
            double x     = origin + (double) ix * spacing;
            double phase = cfg->wavenumber * x + cfg->phase;
            out[i]       = amplitude * cos(phase);
        }
        return;
    }

    double time_phase = cfg->omega * t;
    double amplitude  = cfg->amplitude * cos(time_phase);
    double delta      = cfg->wavenumber * spacing;
    double phase0     = cfg->wavenumber * origin + cfg->phase;

    if (fabs(delta) < STIM_SINUSOIDAL_EPS) {
        double value = amplitude * cos(phase0);
        for (i = 0U; i < count; ++i) {
            out[i] = value;
        }
        return;
    }

    double sin_delta, cos_delta;
    sincos_pair(delta, &sin_delta, &cos_delta);

    double sin_kx, cos_kx;
    sincos_pair(phase0, &sin_kx, &cos_kx);
    unsigned int mask = STIM_SINUSOIDAL_RENORM_INTERVAL - 1U;

    for (i = 0U; i < count; ++i) {
        out[i] = amplitude * cos_kx;

        double next_cos = cos_kx * cos_delta - sin_kx * sin_delta;
        double next_sin = sin_kx * cos_delta + cos_kx * sin_delta;
        cos_kx          = next_cos;
        sin_kx          = next_sin;

        if (((unsigned int) (i + 1U) & mask) == 0U) {
            double phase_i = phase0 + delta * (double) (i + 1U);
            sincos_pair(phase_i, &sin_kx, &cos_kx);
        }
    }
}

static void eval_chirp(const SimStimulusSinusoidalConfig* cfg,
                       const SimField*                    field,
                       double*                            out,
                       size_t                             count,
                       double                             t) {
    size_t i;
    double origin  = cfg->coord.origin_x;
    double spacing = cfg->coord.spacing_x;

    if (out == NULL || count == 0U) {
        return;
    }

    if (cfg->amplitude == 0.0) {
        for (i = 0U; i < count; ++i) {
            out[i] = 0.0;
        }
        return;
    }

    size_t rank = field ? sim_field_rank(field) : 1U;
    if (rank > 1U) {
        double k_t = cfg->wavenumber + cfg->kdot * t;
        double w_t = cfg->omega + cfg->wdot * t;
        for (i = 0U; i < count; ++i) {
            size_t ix = 0U;
            size_t iy = 0U;
            if (sim_field_index_to_xy(field, i, &ix, &iy) != SIM_RESULT_OK) {
                out[i] = 0.0;
                continue;
            }
            (void) iy;
            double x     = origin + (double) ix * spacing;
            double phase = k_t * x - w_t * t + cfg->phase;
            out[i]       = cfg->amplitude * sin(phase);
        }
        return;
    }

    double k_t    = cfg->wavenumber + cfg->kdot * t;
    double w_t    = cfg->omega + cfg->wdot * t;
    double delta  = k_t * spacing;
    double phase0 = k_t * origin - w_t * t + cfg->phase;

    if (fabs(delta) < STIM_SINUSOIDAL_EPS) {
        double value = cfg->amplitude * sin(phase0);
        for (i = 0U; i < count; ++i) {
            out[i] = value;
        }
        return;
    }

    double sin_delta, cos_delta;
    sincos_pair(delta, &sin_delta, &cos_delta);

    double sin_p, cos_p;
    sincos_pair(phase0, &sin_p, &cos_p);
    unsigned int mask = STIM_SINUSOIDAL_RENORM_INTERVAL - 1U;

    for (i = 0U; i < count; ++i) {
        out[i] = cfg->amplitude * sin_p;

        double next_sin = sin_p * cos_delta + cos_p * sin_delta;
        double next_cos = cos_p * cos_delta - sin_p * sin_delta;
        sin_p           = next_sin;
        cos_p           = next_cos;

        if (((unsigned int) (i + 1U) & mask) == 0U) {
            double phase_i = phase0 + delta * (double) (i + 1U);
            sincos_pair(phase_i, &sin_p, &cos_p);
        }
    }
}

static double sinusoidal_base_1d(const SimStimulusSinusoidalConfig* cfg,
                                 SimStimulusSinusoidalMode          mode,
                                 double                             u,
                                 double                             t,
                                 double                             k_t,
                                 double                             w_t,
                                 double                             time_cos) {
    switch (mode) {
        case SIM_STIMULUS_SINUSOIDAL_STANDING:
            return time_cos * cos(k_t * u + cfg->phase);
        case SIM_STIMULUS_SINUSOIDAL_CHIRP:
            return sin(k_t * u - w_t * t + cfg->phase);
        case SIM_STIMULUS_SINUSOIDAL_SINE:
        default:
            return sin(k_t * u - w_t * t + cfg->phase);
    }
}

static void sinusoidal_eval_generic(const SimStimulusSinusoidalConfig* cfg,
                                    SimStimulusSinusoidalMode          mode,
                                    const SimField*                    field,
                                    double*                            out,
                                    size_t                             count,
                                    double                             t) {
    if (cfg == NULL || field == NULL || out == NULL || count == 0U) {
        return;
    }

    if (cfg->amplitude == 0.0) {
        for (size_t i = 0U; i < count; ++i) {
            out[i] = 0.0;
        }
        return;
    }

    bool   separable      = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool   use_wavevector = cfg->use_wavevector;
    double amplitude      = cfg->amplitude;
    double k_t            = cfg->wavenumber;
    double w_t            = cfg->omega;
    if (mode == SIM_STIMULUS_SINUSOIDAL_CHIRP) {
        k_t = cfg->wavenumber + cfg->kdot * t;
        w_t = cfg->omega + cfg->wdot * t;
    }
    double time_cos = (mode == SIM_STIMULUS_SINUSOIDAL_STANDING) ? cos(cfg->omega * t) : 1.0;

    double k0      = 0.0;
    double k_scale = 0.0;
    if (use_wavevector) {
        k0 = hypot(cfg->kx, cfg->ky);
        if (mode == SIM_STIMULUS_SINUSOIDAL_CHIRP && k0 > STIM_SINUSOIDAL_EPS) {
            k_scale = k_t / k0;
        }
    }

    for (size_t i = 0U; i < count; ++i) {
        double x        = 0.0;
        double y        = 0.0;
        double sample_x = 0.0;
        double sample_y = 0.0;
        if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            out[i] = 0.0;
            continue;
        }
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

        double base = 0.0;
        if (use_wavevector) {
            double dot     = cfg->kx * sample_x + cfg->ky * sample_y;
            double spatial = dot;
            if (mode == SIM_STIMULUS_SINUSOIDAL_CHIRP && k0 > STIM_SINUSOIDAL_EPS) {
                spatial = k_scale * dot;
            }
            if (mode == SIM_STIMULUS_SINUSOIDAL_STANDING) {
                base = time_cos * cos(spatial + cfg->phase);
            } else {
                base = sin(spatial - w_t * t + cfg->phase);
            }
        } else if (separable) {
            double fx = sinusoidal_base_1d(cfg, mode, sample_x, t, k_t, w_t, time_cos);
            double fy = sinusoidal_base_1d(cfg, mode, sample_y, t, k_t, w_t, time_cos);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base = fx + fy;
            } else {
                base = fx * fy;
            }
        } else {
            double u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
            base     = sinusoidal_base_1d(cfg, mode, u, t, k_t, w_t, time_cos);
        }

        out[i] = amplitude * base;
    }
}

static SimResult sinusoidal_step(void*               state_ptr,
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

    SimStimulusSinusoidalState* state = (SimStimulusSinusoidalState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    void*  raw_data   = sim_field_data(field);
    size_t count      = 0U;
    bool   is_complex = false;

    if (raw_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    is_complex = sim_field_is_complex(field);
    if (!is_complex) {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double drive_time = sinusoidal_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    if (state->config.amplitude == 0.0) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    double            scale       = state->config.scale_by_dt ? dt_sub : 1.0;
    double            rotation    = state->config.rotation;
    double            sin_r       = 0.0;
    double            cos_r       = 1.0;
    double*           dst_real    = NULL;
    SimComplexDouble* dst_complex = NULL;

    if (!is_complex) {
        dst_real = (double*) raw_data;
    } else {
        dst_complex = sim_field_complex_data(field);
    }

    if (rotation != 0.0) {
        sincos_pair(rotation, &sin_r, &cos_r);
    }

#if defined(SIM_HAVE_VDSP)
    if (sinusoidal_try_vdsp_linear_rows(
            state, field, is_complex, dst_real, dst_complex, count, scale, drive_time)) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }
#endif

    SimResult prep = sinusoidal_ensure_buffer(state, count);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    size_t rank = sim_field_rank(field);
    bool   use_fast =
        (rank == 1U && state->config.coord.mode == SIM_STIMULUS_COORD_AXIS &&
         state->config.coord.axis == SIM_STIMULUS_AXIS_X && !state->config.use_wavevector &&
         fabs(state->config.coord.velocity_x) <= STIM_SINUSOIDAL_EPS &&
         fabs(state->config.coord.velocity_y) <= STIM_SINUSOIDAL_EPS);

    if (use_fast) {
        switch (state->mode) {
            case SIM_STIMULUS_SINUSOIDAL_STANDING:
                eval_standing(&state->config, field, state->buffer, count, drive_time);
                break;
            case SIM_STIMULUS_SINUSOIDAL_CHIRP:
                eval_chirp(&state->config, field, state->buffer, count, drive_time);
                break;
            case SIM_STIMULUS_SINUSOIDAL_SINE:
            default:
                eval_sine(&state->config, field, state->buffer, count, drive_time);
                break;
        }
    } else {
        sinusoidal_eval_generic(
            &state->config, state->mode, field, state->buffer, count, drive_time);
    }

    if (!is_complex) {
        sim_accel_copy_scale_real(state->buffer, dst_real, count, scale, true);
    } else {
        sim_accel_accumulate_real_to_complex(
            state->buffer, dst_complex, count, scale * cos_r, scale * sin_r);
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

static SimResult sinusoidal_register(struct SimContext*                 context,
                                     const SimStimulusSinusoidalConfig* config,
                                     SimStimulusSinusoidalMode          mode,
                                     const char*                        prefix,
                                     size_t*                            out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusSinusoidalConfig local = { 0 };
    if (config) {
        local = *config;
    }

    sinusoidal_normalize(&local, mode);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         prefix ? prefix : "stimulus_sine",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusSinusoidalState* state =
        (SimStimulusSinusoidalState*) calloc(1U, sizeof(SimStimulusSinusoidalState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    state->mode   = mode;
    state->clock_mode =
        sinusoidal_resolve_clock_mode(context, prefix ? prefix : "stimulus_sine", &state->config);
    state->locked_time       = 0.0;
    state->last_step_index   = 0U;
    state->clock_initialized = false;
    state->buffer            = NULL;
    state->buffer_capacity   = 0U;
    sinusoidal_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), prefix ? prefix : "stimulus_sine");
    const char* schema_key = prefix ? prefix : "stimulus_sine";

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
    info.abstract_id       = schema_key;
    sim_operator_info_set_schema_identity(&info, schema_key);
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
            context, &op_config, 0ULL, SIM_DET_NONE, schema_key)) {
        SimField* field = sim_context_field(context, local.field_index);
        if (field != NULL) {
            bool is_complex = sim_field_is_complex(field);
            if ((!is_complex && field->element_size != sizeof(double)) ||
                (is_complex && field->element_size != sizeof(SimComplexDouble))) {
                field = NULL;
            }
        }
        if (field != NULL) {
            size_t rank = sim_field_rank(field);
            if (rank != 1U || local.coord.mode != SIM_STIMULUS_COORD_AXIS ||
                local.coord.axis != SIM_STIMULUS_AXIS_X || local.use_wavevector) {
                field = NULL;
            }
        }
        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };
                StimulusIRParams                   ir_params   = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId delta      = sinusoidal_build_ir(builder, &local, mode, &ir_params);
                if (is_complex && delta != SIM_IR_INVALID_NODE) {
                    SimIRNodeId zero   = sim_ir_builder_constant(builder, 0.0);
                    SimIRNodeId packed = sim_ir_builder_complex_pack(builder, delta, zero);
                    if (packed != SIM_IR_INVALID_NODE && local.rotation != 0.0) {
                        SimIRNodeId angle = sim_ir_builder_constant(builder, local.rotation);
                        packed            = sim_ir_builder_complex_rotate(builder, packed, angle);
                    }
                    delta = packed;
                }
                SimIRNodeId sum = ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

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
                    kdesc.destroy               = sinusoidal_destroy;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    kdesc.save_state            = sinusoidal_save;
                    kdesc.restore_state         = sinusoidal_restore;
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
                                .fn                = sinusoidal_step,
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
                                .symbolic      = sinusoidal_symbolic,
                                .save_state    = sinusoidal_save,
                                .restore_state = sinusoidal_restore,
                                .destroy       = sinusoidal_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        sinusoidal_destroy(state);
    }

    return result;
}

SimResult sim_add_stimulus_sine_operator(struct SimContext*                 context,
                                         const SimStimulusSinusoidalConfig* config,
                                         size_t*                            out_index) {
    return sinusoidal_register(
        context, config, SIM_STIMULUS_SINUSOIDAL_SINE, "stimulus_sine", out_index);
}

SimResult sim_add_stimulus_standing_operator(struct SimContext*                 context,
                                             const SimStimulusSinusoidalConfig* config,
                                             size_t*                            out_index) {
    return sinusoidal_register(
        context, config, SIM_STIMULUS_SINUSOIDAL_STANDING, "stimulus_standing", out_index);
}

SimResult sim_add_stimulus_chirp_operator(struct SimContext*                 context,
                                          const SimStimulusSinusoidalConfig* config,
                                          size_t*                            out_index) {
    return sinusoidal_register(
        context, config, SIM_STIMULUS_SINUSOIDAL_CHIRP, "stimulus_chirp", out_index);
}

SimResult sim_stimulus_sinusoidal_config(struct SimContext*           context,
                                         size_t                       operator_index,
                                         SimStimulusSinusoidalConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusSinusoidalState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusSinusoidalState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusSinusoidalState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_sinusoidal_update(struct SimContext*                 context,
                                         size_t                             operator_index,
                                         const SimStimulusSinusoidalConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusSinusoidalState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusSinusoidalState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusSinusoidalState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusSinusoidalConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    sinusoidal_normalize(&local, state->mode);
    state->config     = local;
    state->clock_mode = sinusoidal_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_sinusoidal"), &state->config);
    sinusoidal_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
