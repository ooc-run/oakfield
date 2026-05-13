#include "oakfield/operators/stimulus/gaussian.h"
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
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STIM_GAUSSIAN_STATE_MAGIC 0x47415553u
#define STIM_GAUSSIAN_EPS 1.0e-9
#define STIM_GAUSSIAN_RENORM_INTERVAL 256U
#define STIM_GAUSSIAN_VDSP_MIN_LEN 64U

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

typedef struct SimStimulusGaussianState {
    uint32_t                  magic;
    SimStimulusGaussianConfig config;
    SimClockMode              clock_mode;
    double                    locked_time;
    size_t                    last_step_index;
    bool                      clock_initialized;
    double                    snapshot_locked_time;
    size_t                    snapshot_last_step_index;
    bool                      snapshot_clock_initialized;
    double*                   buffer;
    size_t                    buffer_capacity;
    char                      symbolic[160];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusGaussianState;

static void gaussian_normalize(SimStimulusGaussianConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->sigma_x) || config->sigma_x <= STIM_GAUSSIAN_EPS) {
        config->sigma_x = 0.0;
    }
    if (!isfinite(config->sigma_y) || config->sigma_y <= STIM_GAUSSIAN_EPS) {
        config->sigma_y = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);

    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0) {
        config->nominal_dt = 0.0;
    }
}

static SimClockMode gaussian_resolve_clock_mode(const SimContext*                context,
                                                const char*                      op_name,
                                                const SimStimulusGaussianConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_GAUSSIAN_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_gaussian_pulse");
    }
    return resolved;
}

static void gaussian_refresh_symbolic(SimStimulusGaussianState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimStimulusGaussianConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "gaussian amp=%.3g center=(%.3g,%.3g) sigma=(%.3g,%.3g) vel=(%.3g,%.3g)",
                    cfg->amplitude,
                    cfg->coord.center_x,
                    cfg->coord.center_y,
                    cfg->sigma_x,
                    cfg->sigma_y,
                    cfg->coord.velocity_x,
                    cfg->coord.velocity_y);
#else
    (void) state;
#endif
}

static const char* gaussian_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusGaussianState* state = (const SimStimulusGaussianState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

typedef struct StimulusGaussianIRParams {
    bool needs_dt;
    bool needs_step_index;
    bool needs_time;
} StimulusGaussianIRParams;

static SimIRNodeId
gaussian_ir_binary(SimIRBuilder* builder, SimIRNodeType type, SimIRNodeId lhs, SimIRNodeId rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId
gaussian_ir_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    if (operand == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_call(builder, kind, operand);
}

static SimIRNodeId gaussian_build_ir(SimIRBuilder*                    builder,
                                     const SimStimulusGaussianConfig* config,
                                     StimulusGaussianIRParams*        params) {
    if (builder == NULL || config == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt         = false;
    params->needs_step_index = false;
    params->needs_time       = false;

    SimIRNodeId index    = sim_ir_builder_index(builder);
    SimIRNodeId spacing  = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin   = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId x_offset = gaussian_ir_binary(builder, SIM_IR_NODE_MUL, spacing, index);
    SimIRNodeId x        = gaussian_ir_binary(builder, SIM_IR_NODE_ADD, origin, x_offset);
    if (x == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    bool needs_dt =
        config->scale_by_dt || (config->fixed_clock && config->nominal_dt <= STIM_GAUSSIAN_EPS);
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
        if (config->nominal_dt > STIM_GAUSSIAN_EPS) {
            increment = sim_ir_builder_constant(builder, config->nominal_dt);
        } else {
            increment = dt_scaled;
        }
        SimIRNodeId scaled_step =
            gaussian_ir_binary(builder, SIM_IR_NODE_MUL, step_index, increment);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = gaussian_ir_binary(builder, SIM_IR_NODE_ADD, scaled_step, time_offset);
    } else {
        params->needs_time      = true;
        SimIRNodeId time_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_TIME);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = gaussian_ir_binary(builder, SIM_IR_NODE_ADD, time_node, time_offset);
    }

    if (t == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId center_base = sim_ir_builder_constant(builder, config->coord.center_x);
    SimIRNodeId velocity    = sim_ir_builder_constant(builder, config->coord.velocity_x);
    SimIRNodeId center =
        gaussian_ir_binary(builder,
                           SIM_IR_NODE_ADD,
                           center_base,
                           gaussian_ir_binary(builder, SIM_IR_NODE_MUL, velocity, t));
    SimIRNodeId diff             = gaussian_ir_binary(builder, SIM_IR_NODE_SUB, x, center);
    double      sigma_x          = (config->sigma_x > STIM_GAUSSIAN_EPS) ? config->sigma_x : 1.0;
    double      inv_two_sigma_sq = -0.5 / (sigma_x * sigma_x);
    SimIRNodeId inv_sigma        = sim_ir_builder_constant(builder, inv_two_sigma_sq);
    SimIRNodeId diff_sq          = gaussian_ir_binary(builder, SIM_IR_NODE_MUL, diff, diff);
    SimIRNodeId exponent         = gaussian_ir_binary(builder, SIM_IR_NODE_MUL, diff_sq, inv_sigma);
    SimIRNodeId envelope         = gaussian_ir_call(builder, SIM_IR_CALL_EXP, exponent);
    SimIRNodeId amplitude        = sim_ir_builder_constant(builder, config->amplitude);
    SimIRNodeId value = gaussian_ir_binary(builder, SIM_IR_NODE_MUL, amplitude, envelope);
    if (value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId scale = SIM_IR_INVALID_NODE;
    if (config->scale_by_dt) {
        scale = dt_scaled;
    } else {
        scale = sim_ir_builder_constant(builder, 1.0);
    }
    return gaussian_ir_binary(builder, SIM_IR_NODE_MUL, value, scale);
}

static void gaussian_destroy(void* state_ptr) {
    SimStimulusGaussianState* state = (SimStimulusGaussianState*) state_ptr;
    if (!state) {
        return;
    }
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
#endif
    free(state->buffer);
    free(state);
}

static SimResult
gaussian_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusGaussianState* state = (SimStimulusGaussianState*) userdata;
    if (state == NULL || state->magic != STIM_GAUSSIAN_STATE_MAGIC) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->snapshot_locked_time       = state->locked_time;
    state->snapshot_last_step_index   = state->last_step_index;
    state->snapshot_clock_initialized = state->clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult
gaussian_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusGaussianState* state = (SimStimulusGaussianState*) userdata;
    if (state == NULL || state->magic != STIM_GAUSSIAN_STATE_MAGIC) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->locked_time       = state->snapshot_locked_time;
    state->last_step_index   = state->snapshot_last_step_index;
    state->clock_initialized = state->snapshot_clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult gaussian_ensure_buffer(SimStimulusGaussianState* state, size_t count) {
    if (!state) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (state->buffer_capacity >= count) {
        return SIM_RESULT_OK;
    }

    double* resized = (double*) realloc(state->buffer, count * sizeof(double));
    if (!resized) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->buffer          = resized;
    state->buffer_capacity = count;
    return SIM_RESULT_OK;
}

static double gaussian_drive_time(SimStimulusGaussianState* state,
                                  double                    base_time,
                                  double                    dt,
                                  size_t                    step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_GAUSSIAN_EPS) {
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
        (state->config.nominal_dt > STIM_GAUSSIAN_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

#if defined(SIM_HAVE_VDSP)
static bool gaussian_vdsp_ensure_buffers(SimStimulusGaussianState* state, size_t width) {
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
gaussian_linear_map(const SimStimulusGaussianConfig* cfg, double* out_u_x, double* out_u_y) {
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

static bool gaussian_try_vdsp_linear_rows(SimStimulusGaussianState* state,
                                          const SimField*           field,
                                          bool                      is_complex,
                                          double*                   dst_real,
                                          SimComplexDouble*         dst_complex,
                                          size_t                    count,
                                          double                    scale,
                                          double                    t) {
    if (state == NULL || field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_GAUSSIAN_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!gaussian_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusGaussianConfig* cfg             = &state->config;
    double                           output_re_scale = scale;
    double                           output_im_scale = 0.0;
    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            sincos_pair(cfg->rotation, &sin_r, &cos_r);
        }
        output_re_scale = scale * cos_r;
        output_im_scale = scale * sin_r;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    double sigma_x            = (cfg->sigma_x > STIM_GAUSSIAN_EPS) ? cfg->sigma_x : 1.0;
    double sigma_y            = (cfg->sigma_y > STIM_GAUSSIAN_EPS) ? cfg->sigma_y : sigma_x;
    double inv_two_sigma_x_sq = -0.5 / (sigma_x * sigma_x);
    double inv_two_sigma_y_sq = -0.5 / (sigma_y * sigma_y);
    double x0_raw             = cfg->coord.origin_x;
    double y0_raw             = cfg->coord.origin_y;
    double x0_sample          = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0_sample          = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx                 = cfg->coord.spacing_x;
    double dy                 = cfg->coord.spacing_y;
    double center_x           = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double center_y           = cfg->coord.center_y + cfg->coord.velocity_y * t;

    if (!isfinite(inv_two_sigma_x_sq) || !isfinite(inv_two_sigma_y_sq) || !isfinite(x0_raw) ||
        !isfinite(y0_raw) || !isfinite(x0_sample) || !isfinite(y0_sample) || !isfinite(dx) ||
        !isfinite(dy) || !isfinite(center_x) || !isfinite(center_y)) {
        return false;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double x_start = x0_raw - center_x;
        for (size_t row = 0U; row < height; ++row) {
            double y      = y0_raw + (double) row * dy;
            double diff_y = y - center_y;
            double env_y  = exp(diff_y * diff_y * inv_two_sigma_y_sq);
            if (!isfinite(y) || !isfinite(diff_y) || !isfinite(env_y)) {
                return false;
            }

            vDSP_vrampD(&x_start, &dx, state->vdsp_value, 1, len);
            vDSP_vsqD(state->vdsp_value, 1, state->vdsp_work, 1, len);
            sim_accel_scale_inplace_real(state->vdsp_work, width, inv_two_sigma_x_sq);
            vvexp(state->vdsp_value, state->vdsp_work, &vforce_len);

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
        if (!gaussian_linear_map(cfg, &u_x, &u_y)) {
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
            sim_accel_scale_inplace_real(state->vdsp_work, width, inv_two_sigma_x_sq);
            vvexp(state->vdsp_value, state->vdsp_work, &vforce_len);
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
    }

    return true;
}
#endif

static void gaussian_eval(const SimStimulusGaussianConfig* cfg,
                          const SimField*                  field,
                          double*                          out,
                          size_t                           count,
                          double                           t) {
    if (!cfg || !out) {
        return;
    }

    bool   separable          = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double sigma_x            = (cfg->sigma_x > STIM_GAUSSIAN_EPS) ? cfg->sigma_x : 1.0;
    double sigma_y            = (cfg->sigma_y > STIM_GAUSSIAN_EPS) ? cfg->sigma_y : sigma_x;
    double inv_two_sigma_x_sq = 0.5 / (sigma_x * sigma_x);
    double inv_two_sigma_y_sq = 0.5 / (sigma_y * sigma_y);
    double center_x           = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double center_y           = cfg->coord.center_y + cfg->coord.velocity_y * t;

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
            double env_x = exp(-(dx * dx) * inv_two_sigma_x_sq);
            double env_y = exp(-(dy * dy) * inv_two_sigma_y_sq);
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
            double envelope = exp(-(diff * diff) * inv_two_sigma_x_sq);
            out[i]          = cfg->amplitude * envelope;
        }
    }
}

static SimResult gaussian_step(void*               state_ptr,
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

    SimStimulusGaussianState* state = (SimStimulusGaussianState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    void* raw_data = sim_field_data(field);
    if (!raw_data) {
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
    double drive_time = gaussian_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U || state->config.amplitude == 0.0) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;

#if defined(SIM_HAVE_VDSP)
    if (gaussian_try_vdsp_linear_rows(state,
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

    SimResult prep = gaussian_ensure_buffer(state, count);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    gaussian_eval(&state->config, field, state->buffer, count, drive_time);

    if (!is_complex) {
        double* dst = (double*) raw_data;
        for (size_t i = 0U; i < count; ++i) {
            dst[i] += scale * state->buffer[i];
        }
    } else {
        SimComplexDouble* dst      = sim_field_complex_data(field);
        double            rotation = state->config.rotation;
        double            sin_r    = 0.0;
        double            cos_r    = 1.0;

        if (rotation != 0.0) {
            sincos_pair(rotation, &sin_r, &cos_r);
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

SimResult sim_add_stimulus_gaussian_operator(struct SimContext*               context,
                                             const SimStimulusGaussianConfig* config,
                                             size_t*                          out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusGaussianConfig local = { 0 };
    if (config) {
        local = *config;
    }
    gaussian_normalize(&local);

    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_gaussian_pulse",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusGaussianState* state =
        (SimStimulusGaussianState*) calloc(1U, sizeof(SimStimulusGaussianState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->magic  = STIM_GAUSSIAN_STATE_MAGIC;
    state->config = local;
    state->clock_mode =
        gaussian_resolve_clock_mode(context, "stimulus_gaussian_pulse", &state->config);
    state->locked_time       = 0.0;
    state->last_step_index   = 0U;
    state->clock_initialized = false;
    state->buffer            = NULL;
    state->buffer_capacity   = 0U;
    gaussian_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_gaussian_pulse");

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
    info.abstract_id       = "stimulus_gaussian_pulse";
    sim_operator_info_set_schema_identity(&info, "stimulus_gaussian_pulse");
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
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_gaussian")) {
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
            bool   y_ok    = (fabs(local.coord.origin_y) <= STIM_GAUSSIAN_EPS &&
                              fabs(local.coord.center_y) <= STIM_GAUSSIAN_EPS &&
                              fabs(local.coord.velocity_y) <= STIM_GAUSSIAN_EPS);
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
                StimulusGaussianIRParams           ir_params   = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId delta      = gaussian_build_ir(builder, &local, &ir_params);
                if (is_complex && delta != SIM_IR_INVALID_NODE) {
                    SimIRNodeId zero   = sim_ir_builder_constant(builder, 0.0);
                    SimIRNodeId packed = sim_ir_builder_complex_pack(builder, delta, zero);
                    if (packed != SIM_IR_INVALID_NODE && local.rotation != 0.0) {
                        SimIRNodeId angle = sim_ir_builder_constant(builder, local.rotation);
                        packed            = sim_ir_builder_complex_rotate(builder, packed, angle);
                    }
                    delta = packed;
                }
                SimIRNodeId sum = gaussian_ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

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
                    kdesc.destroy               = gaussian_destroy;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    kdesc.save_state            = gaussian_save;
                    kdesc.restore_state         = gaussian_restore;
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
                                .fn                = gaussian_step,
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
                                .symbolic      = gaussian_symbolic,
                                .save_state    = gaussian_save,
                                .restore_state = gaussian_restore,
                                .destroy       = gaussian_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        gaussian_destroy(state);
    }

    return result;
}

SimResult sim_stimulus_gaussian_config(struct SimContext*         context,
                                       size_t                     operator_index,
                                       SimStimulusGaussianConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusGaussianState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusGaussianState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusGaussianState*) sim_split_state(op);
    }
    if (state == NULL || state->magic != STIM_GAUSSIAN_STATE_MAGIC) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_gaussian_update(struct SimContext*               context,
                                       size_t                           operator_index,
                                       const SimStimulusGaussianConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusGaussianState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusGaussianState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusGaussianState*) sim_split_state(op);
    }
    if (state == NULL || state->magic != STIM_GAUSSIAN_STATE_MAGIC) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusGaussianConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    gaussian_normalize(&local);
    state->config     = local;
    state->clock_mode = gaussian_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_gaussian_pulse"), &state->config);
    gaussian_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
