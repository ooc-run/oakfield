#include "oakfield/operators/stimulus/airy_beam.h"

#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "sim_accel.h"
#include "oakfield/math/airy.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_AIRY_MIN_SCALE 1.0e-6
#define STIM_AIRY_DEFAULT_SCALE 0.45
#define STIM_AIRY_DEFAULT_APODIZATION 0.12
#define STIM_AIRY_EXP_CLAMP 60.0
#define STIM_AIRY_VDSP_MIN_LEN 64U

typedef struct SimStimulusAiryBeamState {
    SimStimulusAiryBeamConfig config;
    double                    inv_scale_u;
    double                    inv_scale_v;
    double                    coord_angle_sin;
    double                    coord_angle_cos;
    double                    ellipse_inv_u;
    double                    ellipse_inv_v;
    double                    rotation_sin;
    double                    rotation_cos;
    char                      symbolic[256];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusAiryBeamState;

typedef struct SimStimulusAiryBeamStepParams {
    double coord_offset_x;
    double coord_offset_y;
    double coord_spiral_time;
    double center_u;
    double center_v;
    double orientation_sin;
    double orientation_cos;
    double phase_bias;
    double write_scale;
} SimStimulusAiryBeamStepParams;

static void airy_beam_normalize(SimStimulusAiryBeamConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->scale_u) || fabs(config->scale_u) < STIM_AIRY_MIN_SCALE) {
        config->scale_u = STIM_AIRY_DEFAULT_SCALE;
    }
    if (!isfinite(config->scale_v) || fabs(config->scale_v) < STIM_AIRY_MIN_SCALE) {
        config->scale_v = config->scale_u;
    }
    config->scale_u = fabs(config->scale_u);
    config->scale_v = fabs(config->scale_v);

    if (!isfinite(config->apodization_u)) {
        config->apodization_u = STIM_AIRY_DEFAULT_APODIZATION;
    }
    if (!isfinite(config->apodization_v)) {
        config->apodization_v = config->apodization_u;
    }
    if (!isfinite(config->center_u)) {
        config->center_u = 0.0;
    }
    if (!isfinite(config->center_v)) {
        config->center_v = 0.0;
    }
    if (!isfinite(config->velocity_u)) {
        config->velocity_u = 0.0;
    }
    if (!isfinite(config->velocity_v)) {
        config->velocity_v = 0.0;
    }
    if (!isfinite(config->orientation)) {
        config->orientation = 0.0;
    }
    if (!isfinite(config->orientation_rate)) {
        config->orientation_rate = 0.0;
    }
    if (!isfinite(config->carrier_u)) {
        config->carrier_u = 0.0;
    }
    if (!isfinite(config->carrier_v)) {
        config->carrier_v = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void airy_beam_refresh_symbolic(SimStimulusAiryBeamState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusAiryBeamConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "airy_beam A=%.3g scale=(%.3g,%.3g) a=(%.3g,%.3g)",
                    cfg->amplitude,
                    cfg->scale_u,
                    cfg->scale_v,
                    cfg->apodization_u,
                    cfg->apodization_v);
#else
    (void) state;
#endif
}

static void airy_beam_refresh_derived(SimStimulusAiryBeamState* state) {
    if (state == NULL) {
        return;
    }

    state->inv_scale_u     = 1.0 / state->config.scale_u;
    state->inv_scale_v     = 1.0 / state->config.scale_v;
    state->coord_angle_sin = sin(state->config.coord.angle);
    state->coord_angle_cos = cos(state->config.coord.angle);
    state->ellipse_inv_u   = 1.0 / state->config.coord.ellipse_u;
    state->ellipse_inv_v   = 1.0 / state->config.coord.ellipse_v;
    state->rotation_sin    = sin(state->config.rotation);
    state->rotation_cos    = cos(state->config.rotation);
}

static const char* airy_beam_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusAiryBeamState* state = (const SimStimulusAiryBeamState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void airy_beam_prepare_step(const SimStimulusAiryBeamState* state,
                                   double                          t,
                                   double                          write_scale,
                                   SimStimulusAiryBeamStepParams*  out_params) {
    double theta = 0.0;

    if (state == NULL || out_params == NULL) {
        return;
    }

    theta = state->config.orientation + state->config.orientation_rate * t;

    out_params->coord_offset_x    = state->config.coord.velocity_x * t;
    out_params->coord_offset_y    = state->config.coord.velocity_y * t;
    out_params->coord_spiral_time = state->config.coord.spiral_angular_velocity * t;
    out_params->center_u          = state->config.center_u + state->config.velocity_u * t;
    out_params->center_v          = state->config.center_v + state->config.velocity_v * t;
    out_params->orientation_sin   = sin(theta);
    out_params->orientation_cos   = cos(theta);
    out_params->phase_bias        = state->config.phase - state->config.omega * t;
    out_params->write_scale       = write_scale;
}

static void airy_beam_map_coord(const SimStimulusAiryBeamState*      state,
                                const SimStimulusAiryBeamStepParams* params,
                                double                               x,
                                double                               y,
                                double*                              out_u,
                                double*                              out_v) {
    const SimStimulusCoordConfig* coord    = &state->config.coord;
    double                        sample_x = x - params->coord_offset_x;
    double                        sample_y = y - params->coord_offset_y;

    switch (coord->mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (coord->axis == SIM_STIMULUS_AXIS_Y) {
                *out_u = sample_y;
                *out_v = sample_x;
            } else {
                *out_u = sample_x;
                *out_v = sample_y;
            }
            return;
        case SIM_STIMULUS_COORD_ANGLE: {
            *out_u = sample_x * state->coord_angle_cos + sample_y * state->coord_angle_sin;
            *out_v = -sample_x * state->coord_angle_sin + sample_y * state->coord_angle_cos;
            return;
        }
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR:
        case SIM_STIMULUS_COORD_AZIMUTH:
        case SIM_STIMULUS_COORD_ELLIPTIC:
        case SIM_STIMULUS_COORD_SPIRAL: {
            double dx = sample_x - coord->center_x;
            double dy = sample_y - coord->center_y;

            if (coord->mode == SIM_STIMULUS_COORD_SPIRAL) {
                double r  = hypot(dx, dy);
                double th = atan2(dy, dx);
                *out_u = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                         params->coord_spiral_time;
                *out_v = th;
            } else if (coord->mode == SIM_STIMULUS_COORD_POLAR) {
                *out_u = hypot(dx, dy);
                *out_v = atan2(dy, dx);
            } else if (coord->mode == SIM_STIMULUS_COORD_AZIMUTH) {
                *out_u = atan2(dy, dx);
                *out_v = hypot(dx, dy);
            } else if (coord->mode == SIM_STIMULUS_COORD_ELLIPTIC) {
                double ur = dx * state->coord_angle_cos + dy * state->coord_angle_sin;
                double vr = -dx * state->coord_angle_sin + dy * state->coord_angle_cos;
                *out_u    = ur * state->ellipse_inv_u;
                *out_v    = vr * state->ellipse_inv_v;
            } else {
                *out_u = dx;
                *out_v = dy;
            }
            return;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            *out_u = sample_x;
            *out_v = sample_y;
            return;
    }
}

static double airy_beam_eval_real_uv(const SimStimulusAiryBeamState*      state,
                                     const SimStimulusAiryBeamStepParams* params,
                                     double                               u,
                                     double                               v) {
    const SimStimulusAiryBeamConfig* cfg = &state->config;
    double                           du  = u - params->center_u;
    double                           dv  = v - params->center_v;
    double ur       = du * params->orientation_cos + dv * params->orientation_sin;
    double vr       = -du * params->orientation_sin + dv * params->orientation_cos;
    double su       = ur * state->inv_scale_u;
    double sv       = vr * state->inv_scale_v;
    double airy_u   = sim_airy_ai_f64(su);
    double airy_v   = sim_airy_ai_f64(sv);
    double expo_arg = cfg->apodization_u * su + cfg->apodization_v * sv;
    if (expo_arg > STIM_AIRY_EXP_CLAMP) {
        expo_arg = STIM_AIRY_EXP_CLAMP;
    } else if (expo_arg < -STIM_AIRY_EXP_CLAMP) {
        expo_arg = -STIM_AIRY_EXP_CLAMP;
    }

    return airy_u * airy_v * exp(expo_arg) *
           cos(cfg->carrier_u * ur + cfg->carrier_v * vr + params->phase_bias);
}

static void airy_beam_eval_complex_uv(const SimStimulusAiryBeamState*      state,
                                      const SimStimulusAiryBeamStepParams* params,
                                      double                               u,
                                      double                               v,
                                      double*                              out_re,
                                      double*                              out_im) {
    const SimStimulusAiryBeamConfig* cfg = &state->config;
    double                           du  = u - params->center_u;
    double                           dv  = v - params->center_v;
    double ur       = du * params->orientation_cos + dv * params->orientation_sin;
    double vr       = -du * params->orientation_sin + dv * params->orientation_cos;
    double su       = ur * state->inv_scale_u;
    double sv       = vr * state->inv_scale_v;
    double airy_u   = sim_airy_ai_f64(su);
    double airy_v   = sim_airy_ai_f64(sv);
    double expo_arg = cfg->apodization_u * su + cfg->apodization_v * sv;
    if (expo_arg > STIM_AIRY_EXP_CLAMP) {
        expo_arg = STIM_AIRY_EXP_CLAMP;
    } else if (expo_arg < -STIM_AIRY_EXP_CLAMP) {
        expo_arg = -STIM_AIRY_EXP_CLAMP;
    }

    double envelope = airy_u * airy_v * exp(expo_arg);
    double phase    = cfg->carrier_u * ur + cfg->carrier_v * vr + params->phase_bias;
    *out_re         = envelope * cos(phase);
    *out_im         = envelope * sin(phase);
}

#if defined(SIM_HAVE_VDSP)
static bool airy_beam_vdsp_ensure_buffers(SimStimulusAiryBeamState* state, size_t width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    if (width > SIZE_MAX / (5U * sizeof(double))) {
        return false;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 5U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = width;
    state->vdsp_u        = block;
    state->vdsp_v        = block + width;
    state->vdsp_phase    = block + width * 2U;
    state->vdsp_value    = block + width * 3U;
    state->vdsp_work     = block + width * 4U;
    return true;
}

static bool airy_beam_linear_map(const SimStimulusAiryBeamState* state,
                                 double*                         out_u_x,
                                 double*                         out_u_y,
                                 double*                         out_v_x,
                                 double*                         out_v_y) {
    if (state == NULL || out_u_x == NULL || out_u_y == NULL || out_v_x == NULL || out_v_y == NULL) {
        return false;
    }

    switch (state->config.coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (state->config.coord.axis == SIM_STIMULUS_AXIS_Y) {
                *out_u_x = 0.0;
                *out_u_y = 1.0;
                *out_v_x = 1.0;
                *out_v_y = 0.0;
            } else {
                *out_u_x = 1.0;
                *out_u_y = 0.0;
                *out_v_x = 0.0;
                *out_v_y = 1.0;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE:
            *out_u_x = state->coord_angle_cos;
            *out_u_y = state->coord_angle_sin;
            *out_v_x = -state->coord_angle_sin;
            *out_v_y = state->coord_angle_cos;
            return true;
        case SIM_STIMULUS_COORD_SEPARABLE:
            *out_u_x = 1.0;
            *out_u_y = 0.0;
            *out_v_x = 0.0;
            *out_v_y = 1.0;
            return true;
        default:
            break;
    }

    return false;
}

static bool airy_beam_try_vdsp_linear_rows(SimStimulusAiryBeamState*            state,
                                           const SimField*                      field,
                                           bool                                 is_complex,
                                           double*                              dst_real,
                                           SimComplexDouble*                    dst_complex,
                                           size_t                               count,
                                           const SimStimulusAiryBeamStepParams* params,
                                           double                               t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL || params == NULL ||
        !airy_beam_linear_map(state, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_AIRY_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!airy_beam_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusAiryBeamConfig* cfg = &state->config;
    double                           x0  = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double                           y0  = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double                           dx  = cfg->coord.spacing_x;
    double                           dy  = cfg->coord.spacing_y;
    double                           ur_bias =
        -params->center_u * params->orientation_cos - params->center_v * params->orientation_sin;
    double vr_bias =
        params->center_u * params->orientation_sin - params->center_v * params->orientation_cos;
    double ur_x         = params->orientation_cos * u_x + params->orientation_sin * v_x;
    double ur_y         = params->orientation_cos * u_y + params->orientation_sin * v_y;
    double vr_x         = -params->orientation_sin * u_x + params->orientation_cos * v_x;
    double vr_y         = -params->orientation_sin * u_y + params->orientation_cos * v_y;
    double ur_step      = ur_x * dx;
    double vr_step      = vr_x * dx;
    double phase_x      = cfg->carrier_u * ur_x + cfg->carrier_v * vr_x;
    double phase_y      = cfg->carrier_u * ur_y + cfg->carrier_v * vr_y;
    double phase_step   = phase_x * dx;
    double phase_bias   = cfg->carrier_u * ur_bias + cfg->carrier_v * vr_bias + params->phase_bias;
    double output_scale = params->write_scale;
    double output_re_scale = output_scale;
    double output_im_scale = 0.0;

    if (is_complex) {
        output_re_scale = output_scale * state->rotation_cos;
        output_im_scale = output_scale * state->rotation_sin;
    }

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(params->center_u) || !isfinite(params->center_v) ||
        !isfinite(params->orientation_cos) || !isfinite(params->orientation_sin) ||
        !isfinite(ur_bias) || !isfinite(vr_bias) || !isfinite(ur_x) || !isfinite(ur_y) ||
        !isfinite(vr_x) || !isfinite(vr_y) || !isfinite(ur_step) || !isfinite(vr_step) ||
        !isfinite(phase_x) || !isfinite(phase_y) || !isfinite(phase_step) ||
        !isfinite(phase_bias) || !isfinite(output_re_scale) || !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y    = y0 + (double) row * dy;
        double ur_start    = ur_x * x0 + ur_y * sample_y + ur_bias;
        double vr_start    = vr_x * x0 + vr_y * sample_y + vr_bias;
        double phase_start = phase_x * x0 + phase_y * sample_y + phase_bias;

        if (!isfinite(sample_y) || !isfinite(ur_start) || !isfinite(vr_start) ||
            !isfinite(phase_start)) {
            return false;
        }

        vDSP_vrampD(&ur_start, &ur_step, state->vdsp_u, 1, len);
        vDSP_vrampD(&vr_start, &vr_step, state->vdsp_v, 1, len);
        vDSP_vrampD(&phase_start, &phase_step, state->vdsp_phase, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_u, width, state->inv_scale_u);
        sim_accel_scale_inplace_real(state->vdsp_v, width, state->inv_scale_v);

        for (size_t i = 0U; i < width; ++i) {
            double expo_arg =
                cfg->apodization_u * state->vdsp_u[i] + cfg->apodization_v * state->vdsp_v[i];
            if (expo_arg > STIM_AIRY_EXP_CLAMP) {
                expo_arg = STIM_AIRY_EXP_CLAMP;
            } else if (expo_arg < -STIM_AIRY_EXP_CLAMP) {
                expo_arg = -STIM_AIRY_EXP_CLAMP;
            }
            state->vdsp_value[i] =
                sim_airy_ai_f64(state->vdsp_u[i]) * sim_airy_ai_f64(state->vdsp_v[i]);
            state->vdsp_work[i] = expo_arg;
        }

        vvexp(state->vdsp_u, state->vdsp_work, &vforce_len);
        vDSP_vmulD(state->vdsp_value, 1, state->vdsp_u, 1, state->vdsp_value, 1, len);

        if (!is_complex) {
            vvcos(state->vdsp_work, state->vdsp_phase, &vforce_len);
            vDSP_vmulD(state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
            sim_accel_copy_scale_real(
                state->vdsp_value, dst_real + row * width, width, output_scale, true);
        } else {
            SimComplexDouble* row_ptr = dst_complex + row * width;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vvsincos(state->vdsp_work, state->vdsp_u, state->vdsp_phase, &vforce_len);
            vDSP_vmulD(state->vdsp_u, 1, state->vdsp_value, 1, state->vdsp_u, 1, len);
            vDSP_vmulD(state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_work, 1, len);
            vDSP_vsmaD(state->vdsp_u, 1, &output_re_scale, row_re, 2, row_re, 2, len);
            if (output_im_scale != 0.0) {
                double neg_output_im_scale = -output_im_scale;
                vDSP_vsmaD(state->vdsp_work, 1, &neg_output_im_scale, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_u, 1, &output_im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_work, 1, &output_re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static void airy_beam_destroy(void* state_ptr) {
    SimStimulusAiryBeamState* state = (SimStimulusAiryBeamState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block    = NULL;
        state->vdsp_capacity = 0U;
    }
#endif
    free(state);
}

static SimResult airy_beam_step(void*               state_ptr,
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

    SimStimulusAiryBeamState* state = (SimStimulusAiryBeamState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;
    if (!is_complex) {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    if (count == 0U || state->config.amplitude == 0.0) {
        return SIM_RESULT_OK;
    }

    double                        scale  = state->config.scale_by_dt ? dt_sub : 1.0;
    double                        t      = sim_context_time(context) + state->config.time_offset;
    SimStimulusAiryBeamStepParams params = { 0 };

    airy_beam_prepare_step(state, t, state->config.amplitude * scale, &params);

#if defined(SIM_HAVE_VDSP)
    if (airy_beam_try_vdsp_linear_rows(state,
                                       field,
                                       is_complex,
                                       sim_field_real_data(field),
                                       sim_field_complex_data(field),
                                       count,
                                       &params,
                                       t)) {
        return SIM_RESULT_OK;
    }
#endif

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double x     = 0.0;
            double y     = 0.0;
            double u     = 0.0;
            double v     = 0.0;
            double value = 0.0;

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            airy_beam_map_coord(state, &params, x, y, &u, &v);
            value = params.write_scale * airy_beam_eval_real_uv(state, &params, u, v);
            if (isfinite(value)) {
                dst[i] += value;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double x      = 0.0;
            double y      = 0.0;
            double u      = 0.0;
            double v      = 0.0;
            double re     = 0.0;
            double im     = 0.0;
            double out_re = 0.0;
            double out_im = 0.0;

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            airy_beam_map_coord(state, &params, x, y, &u, &v);
            airy_beam_eval_complex_uv(state, &params, u, v, &re, &im);

            out_re = params.write_scale * (re * state->rotation_cos - im * state->rotation_sin);
            out_im = params.write_scale * (re * state->rotation_sin + im * state->rotation_cos);
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += out_re;
                dst[i].im += out_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_airy_beam_operator(struct SimContext*               context,
                                              const SimStimulusAiryBeamConfig* config,
                                              size_t*                          out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusAiryBeamConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    airy_beam_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_airy_beam",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimField* field = sim_context_field(context, local.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    bool needs_complex = sim_field_is_complex(field);

    SimStimulusAiryBeamState* state = (SimStimulusAiryBeamState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    airy_beam_refresh_derived(state);
    airy_beam_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_airy_beam");

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
    info.abstract_id       = "stimulus_airy_beam";
    sim_operator_info_set_schema_identity(&info, "stimulus_airy_beam");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimOperatorConfig op_config = sim_operator_config_defaults();

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = airy_beam_step,
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
                                .symbolic      = airy_beam_symbolic,
                                .destroy       = airy_beam_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        airy_beam_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_airy_beam_config(struct SimContext*         context,
                                        size_t                     operator_index,
                                        SimStimulusAiryBeamConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusAiryBeamState* state = (SimStimulusAiryBeamState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_airy_beam_update(struct SimContext*               context,
                                        size_t                           operator_index,
                                        const SimStimulusAiryBeamConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusAiryBeamState* state = (SimStimulusAiryBeamState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusAiryBeamConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    airy_beam_normalize(&local);
    state->config = local;
    airy_beam_refresh_derived(state);
    airy_beam_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
