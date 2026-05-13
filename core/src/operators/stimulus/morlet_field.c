#include "oakfield/operators/stimulus/morlet_field.h"

#include "operators/common/operator_utils.h"

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

#define STIM_MORLET_EPS 1.0e-12
#define STIM_MORLET_MAX_SCALES 64U
#define STIM_MORLET_VDSP_MIN_LEN 64U

typedef struct SimStimulusMorletFieldState {
    SimStimulusMorletFieldConfig config;
    char                         symbolic[256];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_env;
    double* vdsp_sin;
    double* vdsp_cos;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
} SimStimulusMorletFieldState;

static void morlet_field_normalize(SimStimulusMorletFieldConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (config->scale_count == 0U) {
        config->scale_count = 4U;
    }
    if (config->scale_count > STIM_MORLET_MAX_SCALES) {
        config->scale_count = STIM_MORLET_MAX_SCALES;
    }
    if (!isfinite(config->base_wavenumber) || fabs(config->base_wavenumber) <= STIM_MORLET_EPS) {
        config->base_wavenumber = 1.0;
    }
    if (!isfinite(config->scale_growth) || fabs(config->scale_growth) <= STIM_MORLET_EPS) {
        config->scale_growth = 2.0;
    }
    config->scale_growth = fabs(config->scale_growth);

    if (!isfinite(config->sigma_base) || config->sigma_base <= STIM_MORLET_EPS) {
        config->sigma_base = 0.75;
    }
    if (!isfinite(config->sigma_growth) || fabs(config->sigma_growth) <= STIM_MORLET_EPS) {
        config->sigma_growth = 1.5;
    }
    config->sigma_growth = fabs(config->sigma_growth);

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
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }

    if (!config->use_wavevector &&
        (fabs(config->kx) > STIM_MORLET_EPS || fabs(config->ky) > STIM_MORLET_EPS)) {
        config->use_wavevector = true;
    }

    if (config->use_wavevector && fabs(config->kx) <= STIM_MORLET_EPS &&
        fabs(config->ky) <= STIM_MORLET_EPS) {
        config->kx = 1.0;
        config->ky = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void morlet_field_refresh_symbolic(SimStimulusMorletFieldState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusMorletFieldConfig* cfg = &state->config;
    if (cfg->use_wavevector) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "morlet_field A=%.3g S=%u k0=%.3g g=%.3g sigma0=%.3g k=(%.3g,%.3g)",
                        cfg->amplitude,
                        cfg->scale_count,
                        cfg->base_wavenumber,
                        cfg->scale_growth,
                        cfg->sigma_base,
                        cfg->kx,
                        cfg->ky);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "morlet_field A=%.3g S=%u k0=%.3g g=%.3g sigma0=%.3g",
                        cfg->amplitude,
                        cfg->scale_count,
                        cfg->base_wavenumber,
                        cfg->scale_growth,
                        cfg->sigma_base);
    }
#else
    (void) state;
#endif
}

static const char* morlet_field_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusMorletFieldState* state = (const SimStimulusMorletFieldState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void morlet_field_map_coord(const SimStimulusMorletFieldConfig* cfg,
                                   double                              x,
                                   double                              y,
                                   double                              t,
                                   double*                             out_u,
                                   double*                             out_v) {
    if (cfg->use_wavevector) {
        double norm     = hypot(cfg->kx, cfg->ky);
        double ux       = 1.0;
        double uy       = 0.0;
        double sample_x = x;
        double sample_y = y;
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);
        if (norm > STIM_MORLET_EPS) {
            ux = cfg->kx / norm;
            uy = cfg->ky / norm;
        }
        *out_u = ux * sample_x + uy * sample_y;
        *out_v = -uy * sample_x + ux * sample_y;
        return;
    }

    const SimStimulusCoordConfig* coord    = &cfg->coord;
    double                        sample_x = x;
    double                        sample_y = y;
    sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);
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
            double s = sin(coord->angle);
            double c = cos(coord->angle);
            *out_u   = sample_x * c + sample_y * s;
            *out_v   = -sample_x * s + sample_y * c;
            return;
        }
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            *out_u = hypot(dx, dy);
            *out_v = atan2(dy, dx);
            return;
        }
        case SIM_STIMULUS_COORD_AZIMUTH: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            *out_u = atan2(dy, dx);
            *out_v = hypot(dx, dy);
            return;
        }
        case SIM_STIMULUS_COORD_ELLIPTIC: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            sim_stimulus_coord_elliptic_polar(coord, dx, dy, out_u, out_v);
            return;
        }
        case SIM_STIMULUS_COORD_SPIRAL: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            double r  = hypot(dx, dy);
            double th = atan2(dy, dx);
            *out_u    = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                        coord->spiral_angular_velocity * t;
            *out_v    = th;
            return;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            *out_u = sample_x;
            *out_v = sample_y;
            return;
    }
}

static void morlet_field_eval_uv(const SimStimulusMorletFieldConfig* cfg,
                                 double                              u,
                                 double                              v,
                                 double                              t,
                                 double*                             out_re,
                                 double*                             out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s     = sin(theta);
    double c     = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du       = u - center_u;
    double dv       = v - center_v;

    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;

    double re_sum = 0.0;
    double im_sum = 0.0;

    double k_scale     = 1.0;
    double sigma_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->scale_count; ++i) {
        double k     = cfg->base_wavenumber * k_scale;
        double sigma = cfg->sigma_base * sigma_scale;
        if (sigma <= STIM_MORLET_EPS) {
            sigma = 1.0;
        }

        double r2       = ur * ur + vr * vr;
        double envelope = exp(-0.5 * r2 / (sigma * sigma));

        double phase      = k * ur - cfg->omega * t + cfg->phase;
        double carrier_re = cos(phase);
        double carrier_im = sin(phase);

        if (cfg->zero_mean) {
            double correction = exp(-0.5 * (sigma * k) * (sigma * k));
            carrier_re -= correction;
        }

        re_sum += envelope * carrier_re;
        im_sum += envelope * carrier_im;

        k_scale *= cfg->scale_growth;
        sigma_scale *= cfg->sigma_growth;
    }

    double norm = 1.0;
    if (cfg->scale_count > 0U) {
        norm = 1.0 / sqrt((double) cfg->scale_count);
    }

    *out_re = re_sum * norm;
    *out_im = im_sum * norm;
}

static void morlet_field_destroy(void* state_ptr) {
    SimStimulusMorletFieldState* state = (SimStimulusMorletFieldState*) state_ptr;
    if (state != NULL) {
#if defined(SIM_HAVE_VDSP)
        free(state->vdsp_block);
#endif
    }
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool morlet_field_vdsp_ensure_buffers(SimStimulusMorletFieldState* state, size_t width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* resized = (double*) realloc(state->vdsp_block, width * 8U * sizeof(double));
    if (resized == NULL) {
        return false;
    }

    state->vdsp_block    = resized;
    state->vdsp_capacity = width;
    state->vdsp_u        = resized;
    state->vdsp_v        = resized + width;
    state->vdsp_phase    = resized + width * 2U;
    state->vdsp_env      = resized + width * 3U;
    state->vdsp_sin      = resized + width * 4U;
    state->vdsp_cos      = resized + width * 5U;
    state->vdsp_accum_re = resized + width * 6U;
    state->vdsp_accum_im = resized + width * 7U;
    return true;
}

static bool morlet_field_vdsp_eval_row(SimStimulusMorletFieldState*        state,
                                       const SimStimulusMorletFieldConfig* cfg,
                                       size_t                              width,
                                       double                              u_start,
                                       double                              u_step,
                                       double                              v_start,
                                       double                              v_step,
                                       double                              t) {
    if (state == NULL || cfg == NULL || width == 0U) {
        return false;
    }
    if (!morlet_field_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double theta    = cfg->orientation + cfg->orientation_rate * t;
    double theta_s  = 0.0;
    double theta_c  = 1.0;
    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du_start = u_start - center_u;
    double dv_start = v_start - center_v;
    double ur_start = du_start;
    double ur_step  = u_step;
    double vr_start = dv_start;
    double vr_step  = v_step;

    if (theta != 0.0) {
        theta_s  = sin(theta);
        theta_c  = cos(theta);
        ur_start = du_start * theta_c + dv_start * theta_s;
        ur_step  = u_step * theta_c + v_step * theta_s;
        vr_start = -du_start * theta_s + dv_start * theta_c;
        vr_step  = -u_step * theta_s + v_step * theta_c;
    }

    if (!isfinite(ur_start) || !isfinite(ur_step) || !isfinite(vr_start) || !isfinite(vr_step)) {
        return false;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    const double      zero       = 0.0;
    vDSP_vfillD(&zero, state->vdsp_accum_re, 1, len);
    vDSP_vfillD(&zero, state->vdsp_accum_im, 1, len);

    double k_scale     = 1.0;
    double sigma_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->scale_count; ++i) {
        double k     = cfg->base_wavenumber * k_scale;
        double sigma = cfg->sigma_base * sigma_scale;
        if (sigma <= STIM_MORLET_EPS) {
            sigma = 1.0;
        }

        double envelope_scale = -0.5 / (sigma * sigma);
        double phase_start    = k * ur_start - cfg->omega * t + cfg->phase;
        double phase_step     = k * ur_step;

        if (!isfinite(k) || !isfinite(sigma) || !isfinite(envelope_scale) ||
            !isfinite(phase_start) || !isfinite(phase_step)) {
            return false;
        }

        vDSP_vrampD(&ur_start, &ur_step, state->vdsp_u, 1, len);
        vDSP_vrampD(&vr_start, &vr_step, state->vdsp_v, 1, len);
        vDSP_vsqD(state->vdsp_u, 1, state->vdsp_env, 1, len);
        vDSP_vsqD(state->vdsp_v, 1, state->vdsp_phase, 1, len);
        vDSP_vaddD(state->vdsp_env, 1, state->vdsp_phase, 1, state->vdsp_env, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_env, width, envelope_scale);
        vvexp(state->vdsp_env, state->vdsp_env, &vforce_len);

        vDSP_vrampD(&phase_start, &phase_step, state->vdsp_phase, 1, len);
        vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_phase, &vforce_len);

        if (cfg->zero_mean) {
            double correction = exp(-0.5 * (sigma * k) * (sigma * k));
            if (!isfinite(correction)) {
                return false;
            }
            sim_accel_add_scalar_real(state->vdsp_cos, width, -correction);
        }

        vDSP_vmulD(state->vdsp_env, 1, state->vdsp_cos, 1, state->vdsp_cos, 1, len);
        vDSP_vmulD(state->vdsp_env, 1, state->vdsp_sin, 1, state->vdsp_sin, 1, len);
        vDSP_vaddD(state->vdsp_accum_re, 1, state->vdsp_cos, 1, state->vdsp_accum_re, 1, len);
        vDSP_vaddD(state->vdsp_accum_im, 1, state->vdsp_sin, 1, state->vdsp_accum_im, 1, len);

        k_scale *= cfg->scale_growth;
        sigma_scale *= cfg->sigma_growth;
    }

    if (cfg->scale_count > 0U) {
        double norm = 1.0 / sqrt((double) cfg->scale_count);
        sim_accel_scale_inplace_real(state->vdsp_accum_re, width, norm);
        sim_accel_scale_inplace_real(state->vdsp_accum_im, width, norm);
    }

    return true;
}

static bool morlet_field_try_vdsp_rows(SimStimulusMorletFieldState* state,
                                       const SimField*              field,
                                       bool                         is_complex,
                                       double*                      dst_real,
                                       SimComplexDouble*            dst_complex,
                                       size_t                       count,
                                       double                       scale,
                                       double                       t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if ((!is_complex && dst_real == NULL) || (is_complex && dst_complex == NULL)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    const SimStimulusMorletFieldConfig* cfg = &state->config;
    bool separable = (!cfg->use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool linear_mode =
        (cfg->coord.mode == SIM_STIMULUS_COORD_AXIS || cfg->coord.mode == SIM_STIMULUS_COORD_ANGLE);
    if (!cfg->use_wavevector && !separable && !linear_mode) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_MORLET_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!morlet_field_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double re_scale = cfg->amplitude * scale;
    double im_scale = 0.0;
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

    if (!isfinite(re_scale) || !isfinite(im_scale)) {
        return false;
    }
    if (re_scale == 0.0 && (!is_complex || im_scale == 0.0)) {
        return true;
    }

    double sample_x0 = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double sample_y0 = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double raw_x0    = cfg->coord.origin_x;
    double raw_y0    = cfg->coord.origin_y;
    double dx        = cfg->coord.spacing_x;
    double dy        = cfg->coord.spacing_y;

    double basis_u_x = 1.0;
    double basis_u_y = 0.0;
    double basis_v_x = 0.0;
    double basis_v_y = 1.0;
    if (cfg->use_wavevector) {
        double norm = hypot(cfg->kx, cfg->ky);
        if (norm <= STIM_MORLET_EPS) {
            return false;
        }
        basis_u_x = cfg->kx / norm;
        basis_u_y = cfg->ky / norm;
        basis_v_x = -basis_u_y;
        basis_v_y = basis_u_x;
    } else if (!separable && cfg->coord.mode == SIM_STIMULUS_COORD_AXIS &&
               cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
        basis_u_x = 0.0;
        basis_u_y = 1.0;
        basis_v_x = 1.0;
        basis_v_y = 0.0;
    } else if (!separable && cfg->coord.mode == SIM_STIMULUS_COORD_ANGLE) {
        double s  = sin(cfg->coord.angle);
        double c  = cos(cfg->coord.angle);
        basis_u_x = c;
        basis_u_y = s;
        basis_v_x = -s;
        basis_v_y = c;
    }

    if (!isfinite(sample_x0) || !isfinite(sample_y0) || !isfinite(raw_x0) || !isfinite(raw_y0) ||
        !isfinite(dx) || !isfinite(dy) || !isfinite(basis_u_x) || !isfinite(basis_u_y) ||
        !isfinite(basis_v_x) || !isfinite(basis_v_y)) {
        return false;
    }

    const vDSP_Length len = (vDSP_Length) width;
    for (size_t row = 0U; row < height; ++row) {
        double row_sample_y = sample_y0 + (double) row * dy;
        double row_raw_y    = raw_y0 + (double) row * dy;
        bool   ok           = false;

        if (separable) {
            double y_re = 0.0;
            double y_im = 0.0;
            ok          = morlet_field_vdsp_eval_row(state, cfg, width, raw_x0, dx, 0.0, 0.0, t);
            if (!ok) {
                return false;
            }
            morlet_field_eval_uv(cfg, row_raw_y, 0.0, t, &y_re, &y_im);
            if (!isfinite(y_re) || !isfinite(y_im)) {
                return false;
            }

            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                sim_accel_add_scalar_real(state->vdsp_accum_re, width, y_re);
                sim_accel_add_scalar_real(state->vdsp_accum_im, width, y_im);
            } else {
                sim_accel_copy_scale_real(state->vdsp_accum_re, state->vdsp_u, width, y_re, false);
                vDSP_vsmaD(state->vdsp_accum_im,
                           1,
                           &(double){ -y_im },
                           state->vdsp_u,
                           1,
                           state->vdsp_u,
                           1,
                           len);
                sim_accel_copy_scale_real(state->vdsp_accum_re, state->vdsp_v, width, y_im, false);
                vDSP_vsmaD(state->vdsp_accum_im, 1, &y_re, state->vdsp_v, 1, state->vdsp_v, 1, len);
                sim_accel_copy_scale_real(state->vdsp_u, state->vdsp_accum_re, width, 1.0, false);
                sim_accel_copy_scale_real(state->vdsp_v, state->vdsp_accum_im, width, 1.0, false);
            }
        } else {
            double u_start = basis_u_x * sample_x0 + basis_u_y * row_sample_y;
            double u_step  = basis_u_x * dx;
            double v_start = basis_v_x * sample_x0 + basis_v_y * row_sample_y;
            double v_step  = basis_v_x * dx;
            ok = morlet_field_vdsp_eval_row(state, cfg, width, u_start, u_step, v_start, v_step, t);
            if (!ok) {
                return false;
            }
        }

        size_t offset = row * width;
        if (!is_complex) {
            double* row_ptr = dst_real + offset;
            vDSP_vsmaD(state->vdsp_accum_re, 1, &re_scale, row_ptr, 1, row_ptr, 1, len);
        } else {
            SimComplexDouble* row_ptr = dst_complex + offset;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vDSP_vsmaD(state->vdsp_accum_re, 1, &re_scale, row_re, 2, row_re, 2, len);
            if (im_scale != 0.0) {
                double neg_im = -im_scale;
                vDSP_vsmaD(state->vdsp_accum_im, 1, &neg_im, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_accum_re, 1, &im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_accum_im, 1, &re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static SimResult morlet_field_step(void*               state_ptr,
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

    SimStimulusMorletFieldState* state = (SimStimulusMorletFieldState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
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

    if (count == 0U || state->config.amplitude == 0.0 || state->config.scale_count == 0U) {
        return SIM_RESULT_OK;
    }

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;
    double t     = sim_context_time(context) + state->config.time_offset;
    bool   separable =
        (!state->config.use_wavevector && state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

#if defined(SIM_HAVE_VDSP)
    if (morlet_field_try_vdsp_rows(state,
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
            double x = 0.0;
            double y = 0.0;
            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            double re = 0.0;
            double im = 0.0;

            if (separable) {
                double rx = 0.0;
                double ix = 0.0;
                double ry = 0.0;
                double iy = 0.0;
                morlet_field_eval_uv(&state->config, x, 0.0, t, &rx, &ix);
                morlet_field_eval_uv(&state->config, y, 0.0, t, &ry, &iy);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = rx + ry;
                    im = ix + iy;
                } else {
                    re = rx * ry - ix * iy;
                    im = rx * iy + ix * ry;
                }
            } else {
                double u = 0.0;
                double v = 0.0;
                morlet_field_map_coord(&state->config, x, y, t, &u, &v);
                morlet_field_eval_uv(&state->config, u, v, t, &re, &im);
            }

            (void) im;
            double value = state->config.amplitude * re;
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
            double x = 0.0;
            double y = 0.0;
            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            double re = 0.0;
            double im = 0.0;

            if (separable) {
                double rx = 0.0;
                double ix = 0.0;
                double ry = 0.0;
                double iy = 0.0;
                morlet_field_eval_uv(&state->config, x, 0.0, t, &rx, &ix);
                morlet_field_eval_uv(&state->config, y, 0.0, t, &ry, &iy);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = rx + ry;
                    im = ix + iy;
                } else {
                    re = rx * ry - ix * iy;
                    im = rx * iy + ix * ry;
                }
            } else {
                double u = 0.0;
                double v = 0.0;
                morlet_field_map_coord(&state->config, x, y, t, &u, &v);
                morlet_field_eval_uv(&state->config, u, v, t, &re, &im);
            }

            re *= state->config.amplitude;
            im *= state->config.amplitude;

            double out_re = re * cos_r - im * sin_r;
            double out_im = re * sin_r + im * cos_r;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += scale * out_re;
                dst[i].im += scale * out_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_morlet_field_operator(struct SimContext*                  context,
                                                 const SimStimulusMorletFieldConfig* config,
                                                 size_t*                             out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusMorletFieldConfig local = { 0 };
    local.zero_mean                    = true;
    if (config != NULL) {
        local = *config;
    }

    morlet_field_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_morlet_field",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusMorletFieldState* state = (SimStimulusMorletFieldState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    morlet_field_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_morlet_field");

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
    info.abstract_id       = "stimulus_morlet_field";
    sim_operator_info_set_schema_identity(&info, "stimulus_morlet_field");
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

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = morlet_field_step,
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
                                .symbolic      = morlet_field_symbolic,
                                .destroy       = morlet_field_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        morlet_field_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_morlet_field_config(struct SimContext*            context,
                                           size_t                        operator_index,
                                           SimStimulusMorletFieldConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusMorletFieldState* state = (SimStimulusMorletFieldState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_morlet_field_update(struct SimContext*                  context,
                                           size_t                              operator_index,
                                           const SimStimulusMorletFieldConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusMorletFieldState* state = (SimStimulusMorletFieldState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusMorletFieldConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    morlet_field_normalize(&local);
    state->config = local;
    morlet_field_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
