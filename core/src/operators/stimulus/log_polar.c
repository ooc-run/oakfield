#include "oakfield/operators/stimulus/log_polar.h"

#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

#include "oakfield/field.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_LOG_POLAR_EPS 1.0e-9
#define STIM_LOG_POLAR_VDSP_MIN_LEN 64U

typedef struct SimStimulusLogPolarState {
    SimStimulusLogPolarConfig config;
    char                      symbolic[192];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_tmp;
    double* vdsp_sin;
    double* vdsp_cos;
    size_t  vdsp_capacity;
#endif
} SimStimulusLogPolarState;

static void log_polar_normalize(SimStimulusLogPolarConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->radial_frequency)) {
        config->radial_frequency = 0.0;
    }
    if (!isfinite(config->angular_frequency)) {
        config->angular_frequency = 0.0;
    }
    if (!isfinite(config->orientation)) {
        config->orientation = 0.0;
    }
    if (!isfinite(config->orientation_rate)) {
        config->orientation_rate = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }
    if (!isfinite(config->radius_floor) || config->radius_floor <= STIM_LOG_POLAR_EPS) {
        config->radius_floor = 0.25;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
    if (config->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        config->coord.mode = SIM_STIMULUS_COORD_AXIS;
    }
    if (config->coord.mode == SIM_STIMULUS_COORD_POLAR) {
        config->coord.mode = SIM_STIMULUS_COORD_RADIAL;
    }
}

static void log_polar_refresh_symbolic(SimStimulusLogPolarState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusLogPolarConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "log_polar A=%.3g kr=%.3g m=%.3g eps=%.3g",
                    cfg->amplitude,
                    cfg->radial_frequency,
                    cfg->angular_frequency,
                    cfg->radius_floor);
#else
    (void) state;
#endif
}

static const char* log_polar_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusLogPolarState* state = (const SimStimulusLogPolarState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void log_polar_map_frame(const SimStimulusLogPolarConfig* cfg,
                                double                           x,
                                double                           y,
                                double                           t,
                                double*                          out_u,
                                double*                          out_v,
                                double*                          out_spiral_phase) {
    double u            = x;
    double v            = y;
    double spiral_phase = 0.0;

    if (cfg == NULL || out_u == NULL || out_v == NULL || out_spiral_phase == NULL) {
        return;
    }

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &u, &v);

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                double tmp = u;
                u          = v;
                v          = tmp;
            }
            break;
        case SIM_STIMULUS_COORD_ANGLE: {
            double s        = sin(cfg->coord.angle);
            double c        = cos(cfg->coord.angle);
            double sample_x = u;
            double sample_y = v;
            u               = sample_x * c + sample_y * s;
            v               = -sample_x * s + sample_y * c;
            break;
        }
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR:
        case SIM_STIMULUS_COORD_AZIMUTH:
        case SIM_STIMULUS_COORD_ELLIPTIC:
        case SIM_STIMULUS_COORD_SPIRAL: {
            sim_stimulus_coord_centered_xy(&cfg->coord, x, y, t, &u, &v);

            if (cfg->coord.mode == SIM_STIMULUS_COORD_AZIMUTH) {
                spiral_phase = atan2(v, u);
            } else if (cfg->coord.mode == SIM_STIMULUS_COORD_ELLIPTIC) {
                sim_stimulus_coord_elliptic_local(&cfg->coord, u, v, &u, &v);
            } else if (cfg->coord.mode == SIM_STIMULUS_COORD_SPIRAL) {
                double radius = hypot(u, v);
                double angle  = atan2(v, u);
                spiral_phase  = cfg->coord.spiral_pitch * radius + cfg->coord.spiral_arms * angle +
                                cfg->coord.spiral_phase + cfg->coord.spiral_angular_velocity * t;
            }
            break;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            break;
    }

    *out_u            = u;
    *out_v            = v;
    *out_spiral_phase = spiral_phase;
}

static void log_polar_eval(const SimStimulusLogPolarConfig* cfg,
                           double                           u,
                           double                           v,
                           double                           t,
                           double                           spiral_phase,
                           double*                          out_re,
                           double*                          out_im) {
    double theta = 0.0;
    double s     = 0.0;
    double c     = 1.0;
    double ur    = u;
    double vr    = v;
    double rho   = 0.0;
    double arg   = 0.0;

    if (out_re == NULL || out_im == NULL || cfg == NULL) {
        return;
    }

    theta = cfg->orientation + cfg->orientation_rate * t;
    if (theta != 0.0) {
        s  = sin(theta);
        c  = cos(theta);
        ur = u * c + v * s;
        vr = -u * s + v * c;
    }

    rho = log(hypot(ur, vr) + cfg->radius_floor);
    arg = cfg->radial_frequency * rho + cfg->angular_frequency * atan2(vr, ur) + spiral_phase -
          cfg->omega * t + cfg->phase;

    *out_re = cos(arg);
    *out_im = sin(arg);
}

static void log_polar_destroy(void* state_ptr) {
    SimStimulusLogPolarState* state = (SimStimulusLogPolarState*) state_ptr;
    if (state != NULL) {
#if defined(SIM_HAVE_VDSP)
        free(state->vdsp_block);
#endif
    }
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool log_polar_vdsp_ensure_buffers(SimStimulusLogPolarState* state, size_t width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* resized = (double*) realloc(state->vdsp_block, width * 6U * sizeof(double));
    if (resized == NULL) {
        return false;
    }

    state->vdsp_block    = resized;
    state->vdsp_capacity = width;
    state->vdsp_u        = resized;
    state->vdsp_v        = resized + width;
    state->vdsp_phase    = resized + width * 2U;
    state->vdsp_tmp      = resized + width * 3U;
    state->vdsp_sin      = resized + width * 4U;
    state->vdsp_cos      = resized + width * 5U;
    return true;
}

static void log_polar_fill_constant(double* data, size_t count, double value) {
    if (data == NULL || count == 0U) {
        return;
    }

    const vDSP_Length len = (vDSP_Length) count;
    vDSP_vfillD(&value, data, 1, len);
}

static bool log_polar_build_frame_row(SimStimulusLogPolarState*        state,
                                      const SimStimulusLogPolarConfig* cfg,
                                      size_t                           width,
                                      double                           sample_x0,
                                      double                           sample_y,
                                      double                           dx,
                                      double                           t,
                                      bool*                            out_has_phase_array) {
    if (state == NULL || cfg == NULL || out_has_phase_array == NULL) {
        return false;
    }
    if (!log_polar_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    *out_has_phase_array  = false;
    const vDSP_Length len = (vDSP_Length) width;

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                log_polar_fill_constant(state->vdsp_u, width, sample_y);
                vDSP_vrampD(&sample_x0, &dx, state->vdsp_v, 1, len);
            } else {
                vDSP_vrampD(&sample_x0, &dx, state->vdsp_u, 1, len);
                log_polar_fill_constant(state->vdsp_v, width, sample_y);
            }
            break;
        case SIM_STIMULUS_COORD_ANGLE: {
            double s       = sin(cfg->coord.angle);
            double c       = cos(cfg->coord.angle);
            double u_start = sample_x0 * c + sample_y * s;
            double u_step  = dx * c;
            double v_start = -sample_x0 * s + sample_y * c;
            double v_step  = -dx * s;
            if (!isfinite(u_start) || !isfinite(u_step) || !isfinite(v_start) ||
                !isfinite(v_step)) {
                return false;
            }
            vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
            vDSP_vrampD(&v_start, &v_step, state->vdsp_v, 1, len);
            break;
        }
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR:
        case SIM_STIMULUS_COORD_AZIMUTH:
        case SIM_STIMULUS_COORD_SPIRAL: {
            double u_start = sample_x0 - cfg->coord.center_x;
            double u_step  = dx;
            double v_const = sample_y - cfg->coord.center_y;
            if (!isfinite(u_start) || !isfinite(u_step) || !isfinite(v_const)) {
                return false;
            }
            vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
            log_polar_fill_constant(state->vdsp_v, width, v_const);

            if (cfg->coord.mode == SIM_STIMULUS_COORD_AZIMUTH) {
                const int vforce_len = (int) width;
                vvatan2(state->vdsp_phase, state->vdsp_v, state->vdsp_u, &vforce_len);
                *out_has_phase_array = true;
            } else if (cfg->coord.mode == SIM_STIMULUS_COORD_SPIRAL) {
                const int vforce_len = (int) width;
                vDSP_vsqD(state->vdsp_u, 1, state->vdsp_tmp, 1, len);
                vDSP_vsqD(state->vdsp_v, 1, state->vdsp_sin, 1, len);
                vDSP_vaddD(state->vdsp_tmp, 1, state->vdsp_sin, 1, state->vdsp_tmp, 1, len);
                vvsqrt(state->vdsp_tmp, state->vdsp_tmp, &vforce_len);
                vvatan2(state->vdsp_phase, state->vdsp_v, state->vdsp_u, &vforce_len);
                sim_accel_scale_inplace_real(state->vdsp_tmp, width, cfg->coord.spiral_pitch);
                sim_accel_scale_inplace_real(state->vdsp_phase, width, cfg->coord.spiral_arms);
                vDSP_vaddD(state->vdsp_tmp, 1, state->vdsp_phase, 1, state->vdsp_phase, 1, len);
                sim_accel_add_scalar_real(state->vdsp_phase,
                                          width,
                                          cfg->coord.spiral_phase +
                                              cfg->coord.spiral_angular_velocity * t);
                *out_has_phase_array = true;
            }
            break;
        }
        case SIM_STIMULUS_COORD_ELLIPTIC: {
            double s = sin(cfg->coord.angle);
            double c = cos(cfg->coord.angle);
            double ellipse_u =
                (fabs(cfg->coord.ellipse_u) > STIM_LOG_POLAR_EPS) ? cfg->coord.ellipse_u : 1.0;
            double ellipse_v = (fabs(cfg->coord.ellipse_v) > STIM_LOG_POLAR_EPS)
                                   ? cfg->coord.ellipse_v
                                   : ellipse_u;
            double dx0       = sample_x0 - cfg->coord.center_x;
            double dy0       = sample_y - cfg->coord.center_y;
            double u_start   = (dx0 * c + dy0 * s) / ellipse_u;
            double u_step    = (dx * c) / ellipse_u;
            double v_start   = (-dx0 * s + dy0 * c) / ellipse_v;
            double v_step    = (-dx * s) / ellipse_v;
            if (!isfinite(u_start) || !isfinite(u_step) || !isfinite(v_start) ||
                !isfinite(v_step)) {
                return false;
            }
            vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
            vDSP_vrampD(&v_start, &v_step, state->vdsp_v, 1, len);
            break;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            return false;
    }

    return true;
}

static bool log_polar_try_vdsp_rows(SimStimulusLogPolarState* state,
                                    const SimField*           field,
                                    bool                      is_complex,
                                    double*                   dst_real,
                                    SimComplexDouble*         dst_complex,
                                    size_t                    count,
                                    double                    scale,
                                    double                    t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if ((!is_complex && dst_real == NULL) || (is_complex && dst_complex == NULL)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_LOG_POLAR_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!log_polar_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusLogPolarConfig* cfg       = &state->config;
    double                           sample_x0 = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double                           sample_y0 = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double                           dx        = cfg->coord.spacing_x;
    double                           dy        = cfg->coord.spacing_y;
    double                           theta     = cfg->orientation + cfg->orientation_rate * t;
    double                           theta_s   = 0.0;
    double                           theta_c   = 1.0;
    double                           re_scale  = cfg->amplitude * scale;
    double                           im_scale  = 0.0;

    if (theta != 0.0) {
        theta_s = sin(theta);
        theta_c = cos(theta);
    }
    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            sin_r = sin(cfg->rotation);
            cos_r = cos(cfg->rotation);
        }
        re_scale = cfg->amplitude * scale * cos_r;
        im_scale = cfg->amplitude * scale * sin_r;
    }

    if (!isfinite(sample_x0) || !isfinite(sample_y0) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(theta_s) || !isfinite(theta_c) || !isfinite(re_scale) || !isfinite(im_scale)) {
        return false;
    }
    if (re_scale == 0.0 && (!is_complex || im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    const double      arg_bias   = cfg->phase - cfg->omega * t;

    for (size_t row = 0U; row < height; ++row) {
        double sample_y        = sample_y0 + (double) row * dy;
        bool   has_phase_array = false;
        if (!isfinite(sample_y) ||
            !log_polar_build_frame_row(
                state, cfg, width, sample_x0, sample_y, dx, t, &has_phase_array)) {
            return false;
        }

        if (theta != 0.0) {
            sim_accel_copy_scale_real(state->vdsp_u, state->vdsp_tmp, width, theta_c, false);
            vDSP_vsmaD(state->vdsp_v, 1, &theta_s, state->vdsp_tmp, 1, state->vdsp_tmp, 1, len);
            sim_accel_copy_scale_real(state->vdsp_u, state->vdsp_cos, width, -theta_s, false);
            vDSP_vsmaD(state->vdsp_v, 1, &theta_c, state->vdsp_cos, 1, state->vdsp_cos, 1, len);
        } else {
            sim_accel_copy_scale_real(state->vdsp_u, state->vdsp_tmp, width, 1.0, false);
            sim_accel_copy_scale_real(state->vdsp_v, state->vdsp_cos, width, 1.0, false);
        }

        vDSP_vsqD(state->vdsp_tmp, 1, state->vdsp_u, 1, len);
        vDSP_vsqD(state->vdsp_cos, 1, state->vdsp_sin, 1, len);
        vDSP_vaddD(state->vdsp_u, 1, state->vdsp_sin, 1, state->vdsp_u, 1, len);
        vvsqrt(state->vdsp_u, state->vdsp_u, &vforce_len);
        sim_accel_add_scalar_real(state->vdsp_u, width, cfg->radius_floor);
        vvlog(state->vdsp_u, state->vdsp_u, &vforce_len);
        sim_accel_scale_inplace_real(state->vdsp_u, width, cfg->radial_frequency);

        vvatan2(state->vdsp_v, state->vdsp_cos, state->vdsp_tmp, &vforce_len);
        sim_accel_scale_inplace_real(state->vdsp_v, width, cfg->angular_frequency);
        vDSP_vaddD(state->vdsp_u, 1, state->vdsp_v, 1, state->vdsp_u, 1, len);
        if (has_phase_array) {
            vDSP_vaddD(state->vdsp_u, 1, state->vdsp_phase, 1, state->vdsp_u, 1, len);
        }
        sim_accel_add_scalar_real(state->vdsp_u, width, arg_bias);

        vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_u, &vforce_len);

        size_t offset = row * width;
        if (!is_complex) {
            double* row_ptr = dst_real + offset;
            vDSP_vsmaD(state->vdsp_cos, 1, &re_scale, row_ptr, 1, row_ptr, 1, len);
        } else {
            SimComplexDouble* row_ptr = dst_complex + offset;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vDSP_vsmaD(state->vdsp_cos, 1, &re_scale, row_re, 2, row_re, 2, len);
            if (im_scale != 0.0) {
                double neg_im = -im_scale;
                vDSP_vsmaD(state->vdsp_sin, 1, &neg_im, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_cos, 1, &im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_sin, 1, &re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static SimResult log_polar_step(void*               state_ptr,
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

    SimStimulusLogPolarState* state = (SimStimulusLogPolarState*) state_ptr;
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

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;
    double t     = sim_context_time(context) + state->config.time_offset;

#if defined(SIM_HAVE_VDSP)
    if (log_polar_try_vdsp_rows(state,
                                field,
                                is_complex,
                                is_complex ? NULL : (double*) sim_field_data(field),
                                is_complex ? sim_field_complex_data(field) : NULL,
                                count,
                                scale,
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
            double x            = 0.0;
            double y            = 0.0;
            double u            = 0.0;
            double v            = 0.0;
            double spiral_phase = 0.0;
            double re           = 0.0;
            double im           = 0.0;
            double value        = 0.0;

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            log_polar_map_frame(&state->config, x, y, t, &u, &v, &spiral_phase);
            log_polar_eval(&state->config, u, v, t, spiral_phase, &re, &im);

            (void) im;
            value = state->config.amplitude * re;
            if (isfinite(value)) {
                dst[i] += scale * value;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        double sin_r = 0.0;
        double cos_r = 1.0;
        if (state->config.rotation != 0.0) {
            sin_r = sin(state->config.rotation);
            cos_r = cos(state->config.rotation);
        }

        for (size_t i = 0U; i < count; ++i) {
            double x            = 0.0;
            double y            = 0.0;
            double u            = 0.0;
            double v            = 0.0;
            double spiral_phase = 0.0;
            double re           = 0.0;
            double im           = 0.0;
            double out_re       = 0.0;
            double out_im       = 0.0;

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            log_polar_map_frame(&state->config, x, y, t, &u, &v, &spiral_phase);
            log_polar_eval(&state->config, u, v, t, spiral_phase, &re, &im);

            re *= state->config.amplitude;
            im *= state->config.amplitude;
            out_re = re * cos_r - im * sin_r;
            out_im = re * sin_r + im * cos_r;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += scale * out_re;
                dst[i].im += scale * out_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_log_polar_operator(struct SimContext*               context,
                                              const SimStimulusLogPolarConfig* config,
                                              size_t*                          out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusLogPolarConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    log_polar_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_log_polar",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusLogPolarState* state = (SimStimulusLogPolarState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    log_polar_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_log_polar");

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
    info.abstract_id       = "stimulus_log_polar";
    sim_operator_info_set_schema_identity(&info, "stimulus_log_polar");
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

    SimOperatorConfig op_config = sim_operator_config_defaults();

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0U, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = log_polar_step,
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
                                .symbolic      = log_polar_symbolic,
                                .destroy       = log_polar_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        log_polar_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_log_polar_config(struct SimContext*         context,
                                        size_t                     operator_index,
                                        SimStimulusLogPolarConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLogPolarState* state = (SimStimulusLogPolarState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_log_polar_update(struct SimContext*               context,
                                        size_t                           operator_index,
                                        const SimStimulusLogPolarConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLogPolarState* state = (SimStimulusLogPolarState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusLogPolarConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    log_polar_normalize(&local);
    state->config = local;
    log_polar_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
