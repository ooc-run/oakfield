#include "oakfield/operators/stimulus/steerable_wavelet.h"

#include "operators/common/operator_utils.h"
#include "static_cache.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_STEERABLE_EPS 1.0e-12
#define STIM_STEERABLE_MAX_SCALES 64U
#define STIM_STEERABLE_VDSP_MIN_LEN 64U

#if defined(__APPLE__)
static inline void steerable_wavelet_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void steerable_wavelet_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void steerable_wavelet_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

typedef struct SimStimulusSteerableWaveletState {
    SimStimulusSteerableWaveletConfig config;
    double*                           band_k_values;
    double*                           band_log_radius_values;
    size_t                            allocated_scales;
    double                            scale_norm;
    double                            inv_radial_bandwidth;
    double                            simoncelli_exponent;
    double                            wavevector_ux;
    double                            wavevector_uy;
    double                            coord_angle_sin;
    double                            coord_angle_cos;
    double                            ellipse_inv_u;
    double                            ellipse_inv_v;
    double                            rotation_sin;
    double                            rotation_cos;
    SimStimulusStaticCache            cache;
    double*                           axis_real_x;
    double*                           axis_imag_x;
    double*                           axis_real_y;
    double*                           axis_imag_y;
    size_t                            axis_x_capacity;
    size_t                            axis_y_capacity;
    size_t                            axis_extent_x;
    size_t                            axis_extent_y;
    double                            axis_cache_time;
    bool                              axis_cache_valid;
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_r;
    double* vdsp_log_r;
    double* vdsp_phase;
    double* vdsp_weight;
    double* vdsp_sin;
    double* vdsp_cos;
    double* vdsp_orient_re;
    double* vdsp_orient_im;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
    char symbolic[256];
} SimStimulusSteerableWaveletState;

static bool steerable_wavelet_family_valid(SimStimulusSteerableWaveletFamily family) {
    return family == SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI ||
           family == SIM_STIMULUS_STEERABLE_WAVELET_RIESZ;
}

static const char* steerable_wavelet_family_name(SimStimulusSteerableWaveletFamily family) {
    switch (family) {
        case SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI:
            return "simoncelli";
        case SIM_STIMULUS_STEERABLE_WAVELET_RIESZ:
            return "riesz";
        default:
            return "unknown";
    }
}

static void steerable_wavelet_normalize(SimStimulusSteerableWaveletConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!steerable_wavelet_family_valid(config->family)) {
        config->family = SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI;
    }
    if (config->order == 0U) {
        config->order = 1U;
    }
    if (config->order > 16U) {
        config->order = 16U;
    }
    if (config->scale_count == 0U) {
        config->scale_count = 4U;
    }
    if (config->scale_count > STIM_STEERABLE_MAX_SCALES) {
        config->scale_count = STIM_STEERABLE_MAX_SCALES;
    }
    if (!isfinite(config->base_wavenumber) || fabs(config->base_wavenumber) <= STIM_STEERABLE_EPS) {
        config->base_wavenumber = 1.0;
    }
    if (!isfinite(config->scale_growth) || fabs(config->scale_growth) <= STIM_STEERABLE_EPS) {
        config->scale_growth = 2.0;
    }
    config->scale_growth = fabs(config->scale_growth);
    if (!isfinite(config->radial_bandwidth) || config->radial_bandwidth <= STIM_STEERABLE_EPS) {
        config->radial_bandwidth = 0.65;
    }
    if (!isfinite(config->angular_sharpness) || config->angular_sharpness <= STIM_STEERABLE_EPS) {
        config->angular_sharpness = 1.0;
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
        (fabs(config->kx) > STIM_STEERABLE_EPS || fabs(config->ky) > STIM_STEERABLE_EPS)) {
        config->use_wavevector = true;
    }

    if (config->use_wavevector && fabs(config->kx) <= STIM_STEERABLE_EPS &&
        fabs(config->ky) <= STIM_STEERABLE_EPS) {
        config->kx = 1.0;
        config->ky = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void steerable_wavelet_refresh_symbolic(SimStimulusSteerableWaveletState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusSteerableWaveletConfig* cfg = &state->config;
    if (cfg->use_wavevector) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "steerable_wavelet %s A=%.3g S=%u N=%u k0=%.3g g=%.3g k=(%.3g,%.3g)",
                        steerable_wavelet_family_name(cfg->family),
                        cfg->amplitude,
                        cfg->scale_count,
                        cfg->order,
                        cfg->base_wavenumber,
                        cfg->scale_growth,
                        cfg->kx,
                        cfg->ky);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "steerable_wavelet %s A=%.3g S=%u N=%u k0=%.3g g=%.3g",
                        steerable_wavelet_family_name(cfg->family),
                        cfg->amplitude,
                        cfg->scale_count,
                        cfg->order,
                        cfg->base_wavenumber,
                        cfg->scale_growth);
    }
#else
    (void) state;
#endif
}

static const char* steerable_wavelet_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusSteerableWaveletState* state =
        (const SimStimulusSteerableWaveletState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static double steerable_wavelet_wrap_pi(double angle) {
    if (!isfinite(angle)) {
        return 0.0;
    }
    const double two_pi  = 2.0 * M_PI;
    double       wrapped = fmod(angle + M_PI, two_pi);
    if (wrapped < 0.0) {
        wrapped += two_pi;
    }
    return wrapped - M_PI;
}

static void steerable_wavelet_invalidate_caches(SimStimulusSteerableWaveletState* state) {
    if (state == NULL) {
        return;
    }
    sim_stimulus_static_cache_invalidate(&state->cache);
    state->axis_cache_valid = false;
    state->axis_extent_x    = 0U;
    state->axis_extent_y    = 0U;
    state->axis_cache_time  = 0.0;
}

static SimResult steerable_wavelet_ensure_scale_capacity(SimStimulusSteerableWaveletState* state,
                                                         unsigned int desired) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (desired == 0U || state->allocated_scales >= desired) {
        return SIM_RESULT_OK;
    }

    double* band_k = (double*) realloc(state->band_k_values, (size_t) desired * sizeof(double));
    if (band_k == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    double* band_log =
        (double*) realloc(state->band_log_radius_values, (size_t) desired * sizeof(double));
    if (band_log == NULL) {
        state->band_k_values = band_k;
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->band_k_values          = band_k;
    state->band_log_radius_values = band_log;
    state->allocated_scales       = desired;
    return SIM_RESULT_OK;
}

static void steerable_wavelet_refresh_precomputed(SimStimulusSteerableWaveletState* state) {
    const SimStimulusSteerableWaveletConfig* cfg;
    double                                   band_scale;

    if (state == NULL) {
        return;
    }

    cfg = &state->config;

    state->scale_norm = (cfg->scale_count > 0U) ? (1.0 / sqrt((double) cfg->scale_count)) : 1.0;
    state->inv_radial_bandwidth = 1.0 / cfg->radial_bandwidth;

    state->simoncelli_exponent = (double) cfg->order * cfg->angular_sharpness;
    if (state->simoncelli_exponent < 1.0e-6) {
        state->simoncelli_exponent = 1.0;
    }

    band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->scale_count; ++i) {
        double k             = cfg->base_wavenumber * band_scale;
        double target_radius = fabs(k);
        if (target_radius <= STIM_STEERABLE_EPS) {
            target_radius = 1.0;
        }

        state->band_k_values[i]          = k;
        state->band_log_radius_values[i] = log(target_radius + STIM_STEERABLE_EPS);
        band_scale *= cfg->scale_growth;
    }

    {
        double wave_norm = hypot(cfg->kx, cfg->ky);
        if (wave_norm > STIM_STEERABLE_EPS) {
            state->wavevector_ux = cfg->kx / wave_norm;
            state->wavevector_uy = cfg->ky / wave_norm;
        } else {
            state->wavevector_ux = 1.0;
            state->wavevector_uy = 0.0;
        }
    }

    steerable_wavelet_sincos(cfg->coord.angle, &state->coord_angle_sin, &state->coord_angle_cos);
    state->ellipse_inv_u = 1.0 / cfg->coord.ellipse_u;
    state->ellipse_inv_v = 1.0 / cfg->coord.ellipse_v;
    steerable_wavelet_sincos(cfg->rotation, &state->rotation_sin, &state->rotation_cos);
}

static bool steerable_wavelet_is_time_invariant(const SimStimulusSteerableWaveletState* state) {
    const SimStimulusSteerableWaveletConfig* cfg;

    if (state == NULL) {
        return false;
    }

    cfg = &state->config;
    if (fabs(cfg->orientation_rate) > STIM_STEERABLE_EPS || fabs(cfg->omega) > STIM_STEERABLE_EPS) {
        return false;
    }

    if (!cfg->use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        return true;
    }

    return sim_stimulus_coord_is_time_invariant(&cfg->coord);
}

static SimResult steerable_wavelet_ensure_axis_capacity(SimStimulusSteerableWaveletState* state,
                                                        size_t                            extent_x,
                                                        size_t extent_y) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->axis_x_capacity < extent_x) {
        double* new_real = (double*) realloc(state->axis_real_x, extent_x * sizeof(double));
        if (new_real == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->axis_real_x = new_real;

        double* new_imag = (double*) realloc(state->axis_imag_x, extent_x * sizeof(double));
        if (new_imag == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->axis_imag_x     = new_imag;
        state->axis_x_capacity = extent_x;
    }

    if (state->axis_y_capacity < extent_y) {
        double* new_real = (double*) realloc(state->axis_real_y, extent_y * sizeof(double));
        if (new_real == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->axis_real_y = new_real;

        double* new_imag = (double*) realloc(state->axis_imag_y, extent_y * sizeof(double));
        if (new_imag == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->axis_imag_y     = new_imag;
        state->axis_y_capacity = extent_y;
    }

    return SIM_RESULT_OK;
}

static void steerable_wavelet_map_coord(const SimStimulusSteerableWaveletState* state,
                                        double                                  x,
                                        double                                  y,
                                        double                                  t,
                                        double*                                 out_u,
                                        double*                                 out_v) {
    const SimStimulusSteerableWaveletConfig* cfg;
    const SimStimulusCoordConfig*            coord;

    if (state == NULL || out_u == NULL || out_v == NULL) {
        return;
    }

    cfg   = &state->config;
    coord = &cfg->coord;

    if (cfg->use_wavevector) {
        double sample_x = x;
        double sample_y = y;
        sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);
        *out_u = state->wavevector_ux * sample_x + state->wavevector_uy * sample_y;
        *out_v = -state->wavevector_uy * sample_x + state->wavevector_ux * sample_y;
        return;
    }

    {
        double sample_x = x;
        double sample_y = y;
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
            case SIM_STIMULUS_COORD_ANGLE:
                *out_u = sample_x * state->coord_angle_cos + sample_y * state->coord_angle_sin;
                *out_v = -sample_x * state->coord_angle_sin + sample_y * state->coord_angle_cos;
                return;
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
                double ur = 0.0;
                double vr = 0.0;
                sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
                ur     = (dx * state->coord_angle_cos + dy * state->coord_angle_sin) *
                         state->ellipse_inv_u;
                vr     = (-dx * state->coord_angle_sin + dy * state->coord_angle_cos) *
                         state->ellipse_inv_v;
                *out_u = hypot(ur, vr);
                *out_v = atan2(vr, ur);
                return;
            }
            case SIM_STIMULUS_COORD_SPIRAL: {
                double dx = 0.0;
                double dy = 0.0;
                double r  = 0.0;
                double th = 0.0;
                sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
                r      = hypot(dx, dy);
                th     = atan2(dy, dx);
                *out_u = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                         coord->spiral_angular_velocity * t;
                *out_v = th;
                return;
            }
            case SIM_STIMULUS_COORD_SEPARABLE:
            default:
                *out_u = sample_x;
                *out_v = sample_y;
                return;
        }
    }
}

static void steerable_wavelet_combine_separable(const SimStimulusSteerableWaveletConfig* cfg,
                                                double                                   rx,
                                                double                                   ix,
                                                double                                   ry,
                                                double                                   iy,
                                                double*                                  out_re,
                                                double*                                  out_im) {
    if (out_re == NULL || out_im == NULL) {
        return;
    }

    if (cfg != NULL && cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
        *out_re = rx + ry;
        *out_im = ix + iy;
        return;
    }

    *out_re = rx * ry - ix * iy;
    *out_im = rx * iy + ix * ry;
}

static void steerable_wavelet_eval_uv(const SimStimulusSteerableWaveletState* state,
                                      double                                  u,
                                      double                                  v,
                                      double                                  steer,
                                      double                                  phase_t,
                                      double*                                 out_re,
                                      double*                                 out_im) {
    const SimStimulusSteerableWaveletConfig* cfg;
    double                                   r;
    double                                   phi;
    double                                   delta;
    double                                   delta_sin;
    double                                   cos_delta;
    double                                   angular_env;
    double                                   orient_re = 1.0;
    double                                   orient_im = 0.0;
    double                                   log_r;
    double                                   re_sum = 0.0;
    double                                   im_sum = 0.0;

    if (state == NULL || out_re == NULL || out_im == NULL) {
        return;
    }

    cfg   = &state->config;
    r     = hypot(u, v);
    phi   = atan2(v, u);
    delta = phi - steer;

    steerable_wavelet_sincos(delta, &delta_sin, &cos_delta);
    (void) delta_sin;
    if (cfg->family == SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI) {
        if (cos_delta <= 0.0) {
            *out_re = 0.0;
            *out_im = 0.0;
            return;
        }
        angular_env = pow(cos_delta, state->simoncelli_exponent);
    } else {
        double orient_delta = (double) cfg->order * delta;
        angular_env = pow(fmax(fabs(cos_delta), STIM_STEERABLE_EPS), cfg->angular_sharpness);
        steerable_wavelet_sincos(orient_delta, &orient_im, &orient_re);
    }

    if (angular_env <= 0.0) {
        *out_re = 0.0;
        *out_im = 0.0;
        return;
    }

    log_r = log(r + STIM_STEERABLE_EPS);
    for (unsigned int i = 0U; i < cfg->scale_count; ++i) {
        double radial_u = (log_r - state->band_log_radius_values[i]) * state->inv_radial_bandwidth;
        double radial_env = exp(-0.5 * radial_u * radial_u);
        double carrier_re = 0.0;
        double carrier_im = 0.0;
        double weight     = radial_env * angular_env;
        steerable_wavelet_sincos(state->band_k_values[i] * r + phase_t, &carrier_im, &carrier_re);

        if (cfg->family == SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI) {
            re_sum += weight * carrier_re;
            im_sum += weight * carrier_im;
        } else {
            double mixed_re = carrier_re * orient_re - carrier_im * orient_im;
            double mixed_im = carrier_re * orient_im + carrier_im * orient_re;
            re_sum += weight * mixed_re;
            im_sum += weight * mixed_im;
        }
    }

    *out_re = re_sum * state->scale_norm;
    *out_im = im_sum * state->scale_norm;
}

static void steerable_wavelet_eval_xy_raw(const SimStimulusSteerableWaveletState* state,
                                          double                                  x,
                                          double                                  y,
                                          double                                  t,
                                          double                                  steer,
                                          double                                  phase_t,
                                          double*                                 out_re,
                                          double*                                 out_im) {
    const SimStimulusSteerableWaveletConfig* cfg;

    if (state == NULL || out_re == NULL || out_im == NULL) {
        return;
    }

    cfg = &state->config;
    if (!cfg->use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double rx = 0.0;
        double ix = 0.0;
        double ry = 0.0;
        double iy = 0.0;
        steerable_wavelet_eval_uv(state, x, 0.0, steer, phase_t, &rx, &ix);
        steerable_wavelet_eval_uv(state, y, 0.0, steer, phase_t, &ry, &iy);
        steerable_wavelet_combine_separable(cfg, rx, ix, ry, iy, out_re, out_im);
        return;
    }

    {
        double u = 0.0;
        double v = 0.0;
        steerable_wavelet_map_coord(state, x, y, t, &u, &v);
        steerable_wavelet_eval_uv(state, u, v, steer, phase_t, out_re, out_im);
    }
}

static void steerable_wavelet_apply_output(const SimStimulusSteerableWaveletState* state,
                                           double                                  base_re,
                                           double                                  base_im,
                                           bool                                    complex_output,
                                           double*                                 out_re,
                                           double*                                 out_im) {
    double re;
    double im;

    if (state == NULL || out_re == NULL || out_im == NULL) {
        return;
    }

    re = state->config.amplitude * base_re;
    im = state->config.amplitude * base_im;
    if (complex_output) {
        *out_re = re * state->rotation_cos - im * state->rotation_sin;
        *out_im = re * state->rotation_sin + im * state->rotation_cos;
        return;
    }

    *out_re = re;
    *out_im = im;
}

#if defined(SIM_HAVE_VDSP)
static bool steerable_wavelet_vdsp_ensure_buffers(SimStimulusSteerableWaveletState* state,
                                                  size_t                            width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 12U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block     = block;
    state->vdsp_capacity  = width;
    state->vdsp_u         = block;
    state->vdsp_v         = block + width;
    state->vdsp_r         = block + width * 2U;
    state->vdsp_log_r     = block + width * 3U;
    state->vdsp_phase     = block + width * 4U;
    state->vdsp_weight    = block + width * 5U;
    state->vdsp_sin       = block + width * 6U;
    state->vdsp_cos       = block + width * 7U;
    state->vdsp_orient_re = block + width * 8U;
    state->vdsp_orient_im = block + width * 9U;
    state->vdsp_accum_re  = block + width * 10U;
    state->vdsp_accum_im  = block + width * 11U;
    return true;
}

static bool steerable_wavelet_linear_basis(const SimStimulusSteerableWaveletState* state,
                                           double*                                 out_u_x,
                                           double*                                 out_u_y,
                                           double*                                 out_v_x,
                                           double*                                 out_v_y) {
    const SimStimulusSteerableWaveletConfig* cfg;

    if (state == NULL || out_u_x == NULL || out_u_y == NULL || out_v_x == NULL || out_v_y == NULL) {
        return false;
    }

    cfg = &state->config;
    if (cfg->use_wavevector) {
        *out_u_x = state->wavevector_ux;
        *out_u_y = state->wavevector_uy;
        *out_v_x = -state->wavevector_uy;
        *out_v_y = state->wavevector_ux;
        return true;
    }

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
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
        default:
            return false;
    }
}

static bool steerable_wavelet_vdsp_eval_row(SimStimulusSteerableWaveletState* state,
                                            size_t                            width,
                                            double                            u_start,
                                            double                            u_step,
                                            double                            v_start,
                                            double                            v_step,
                                            double                            steer,
                                            double                            phase_t) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (!steerable_wavelet_vdsp_ensure_buffers(state, width)) {
        return false;
    }
    if (!isfinite(u_start) || !isfinite(u_step) || !isfinite(v_start) || !isfinite(v_step) ||
        !isfinite(steer) || !isfinite(phase_t)) {
        return false;
    }

    const SimStimulusSteerableWaveletConfig* cfg        = &state->config;
    const vDSP_Length                        len        = (vDSP_Length) width;
    const int                                vforce_len = (int) width;
    const double                             zero       = 0.0;

    vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
    vDSP_vrampD(&v_start, &v_step, state->vdsp_v, 1, len);

    vDSP_vsqD(state->vdsp_u, 1, state->vdsp_r, 1, len);
    vDSP_vsqD(state->vdsp_v, 1, state->vdsp_phase, 1, len);
    vDSP_vaddD(state->vdsp_r, 1, state->vdsp_phase, 1, state->vdsp_r, 1, len);
    vvsqrt(state->vdsp_r, state->vdsp_r, &vforce_len);

    sim_accel_copy_scale_real(state->vdsp_r, state->vdsp_log_r, width, 1.0, false);
    sim_accel_add_scalar_real(state->vdsp_log_r, width, STIM_STEERABLE_EPS);
    vvlog(state->vdsp_log_r, state->vdsp_log_r, &vforce_len);

    vvatan2(state->vdsp_phase, state->vdsp_v, state->vdsp_u, &vforce_len);
    sim_accel_add_scalar_real(state->vdsp_phase, width, -steer);
    vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_phase, &vforce_len);

    if (cfg->family == SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI) {
        const double exponent = state->simoncelli_exponent;
        for (size_t i = 0U; i < width; ++i) {
            double c = state->vdsp_cos[i];
            if (!isfinite(c)) {
                return false;
            }
            state->vdsp_phase[i] = (c > 0.0) ? c : 0.0;
        }
        vvpows(state->vdsp_weight, &exponent, state->vdsp_phase, &vforce_len);
    } else {
        const double sharpness = cfg->angular_sharpness;
        sim_accel_scale_inplace_real(state->vdsp_phase, width, (double) cfg->order);
        vvsincos(state->vdsp_orient_im, state->vdsp_orient_re, state->vdsp_phase, &vforce_len);
        for (size_t i = 0U; i < width; ++i) {
            double c = state->vdsp_cos[i];
            if (!isfinite(c)) {
                return false;
            }
            state->vdsp_phase[i] = fmax(fabs(c), STIM_STEERABLE_EPS);
        }
        vvpows(state->vdsp_weight, &sharpness, state->vdsp_phase, &vforce_len);
    }

    vDSP_vfillD(&zero, state->vdsp_accum_re, 1, len);
    vDSP_vfillD(&zero, state->vdsp_accum_im, 1, len);

    for (unsigned int i = 0U; i < cfg->scale_count; ++i) {
        double band_log = state->band_log_radius_values[i];
        double k        = state->band_k_values[i];

        if (!isfinite(band_log) || !isfinite(k)) {
            return false;
        }

        sim_accel_copy_scale_real(state->vdsp_log_r, state->vdsp_phase, width, 1.0, false);
        sim_accel_add_scalar_real(state->vdsp_phase, width, -band_log);
        sim_accel_scale_inplace_real(state->vdsp_phase, width, state->inv_radial_bandwidth);
        vDSP_vsqD(state->vdsp_phase, 1, state->vdsp_phase, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_phase, width, -0.5);
        vvexp(state->vdsp_phase, state->vdsp_phase, &vforce_len);
        vDSP_vmulD(state->vdsp_phase, 1, state->vdsp_weight, 1, state->vdsp_phase, 1, len);

        sim_accel_copy_scale_real(state->vdsp_r, state->vdsp_u, width, k, false);
        sim_accel_add_scalar_real(state->vdsp_u, width, phase_t);
        vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_u, &vforce_len);

        if (cfg->family == SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI) {
            vDSP_vmulD(state->vdsp_phase, 1, state->vdsp_cos, 1, state->vdsp_cos, 1, len);
            vDSP_vmulD(state->vdsp_phase, 1, state->vdsp_sin, 1, state->vdsp_sin, 1, len);
            vDSP_vaddD(state->vdsp_accum_re, 1, state->vdsp_cos, 1, state->vdsp_accum_re, 1, len);
            vDSP_vaddD(state->vdsp_accum_im, 1, state->vdsp_sin, 1, state->vdsp_accum_im, 1, len);
        } else {
            vDSP_vmulD(state->vdsp_cos, 1, state->vdsp_orient_re, 1, state->vdsp_u, 1, len);
            vDSP_vmulD(state->vdsp_sin, 1, state->vdsp_orient_im, 1, state->vdsp_v, 1, len);
            vDSP_vsubD(state->vdsp_v, 1, state->vdsp_u, 1, state->vdsp_u, 1, len);

            vDSP_vmulD(state->vdsp_cos, 1, state->vdsp_orient_im, 1, state->vdsp_cos, 1, len);
            vDSP_vmulD(state->vdsp_sin, 1, state->vdsp_orient_re, 1, state->vdsp_sin, 1, len);
            vDSP_vaddD(state->vdsp_cos, 1, state->vdsp_sin, 1, state->vdsp_v, 1, len);

            vDSP_vmulD(state->vdsp_phase, 1, state->vdsp_u, 1, state->vdsp_u, 1, len);
            vDSP_vmulD(state->vdsp_phase, 1, state->vdsp_v, 1, state->vdsp_v, 1, len);
            vDSP_vaddD(state->vdsp_accum_re, 1, state->vdsp_u, 1, state->vdsp_accum_re, 1, len);
            vDSP_vaddD(state->vdsp_accum_im, 1, state->vdsp_v, 1, state->vdsp_accum_im, 1, len);
        }
    }

    sim_accel_scale_inplace_real(state->vdsp_accum_re, width, state->scale_norm);
    sim_accel_scale_inplace_real(state->vdsp_accum_im, width, state->scale_norm);
    return true;
}

static bool steerable_wavelet_try_vdsp_rows(SimStimulusSteerableWaveletState* state,
                                            const SimField*                   field,
                                            bool                              is_complex,
                                            double*                           dst_real,
                                            SimComplexDouble*                 dst_complex,
                                            size_t                            count,
                                            double                            scale,
                                            double                            t,
                                            double                            steer,
                                            double                            phase_t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL ||
        !steerable_wavelet_linear_basis(state, &u_x, &u_y, &v_x, &v_y)) {
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
    if (width == 0U || height == 0U || width < STIM_STEERABLE_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!steerable_wavelet_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusSteerableWaveletConfig* cfg = &state->config;
    double sample_x0                             = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double sample_y0                             = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx                                    = cfg->coord.spacing_x;
    double dy                                    = cfg->coord.spacing_y;
    double output_re_scale                       = cfg->amplitude * scale;
    double output_im_scale                       = 0.0;

    if (is_complex) {
        output_re_scale = cfg->amplitude * scale * state->rotation_cos;
        output_im_scale = cfg->amplitude * scale * state->rotation_sin;
    }

    if (!isfinite(sample_x0) || !isfinite(sample_y0) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(u_x) || !isfinite(u_y) || !isfinite(v_x) || !isfinite(v_y) ||
        !isfinite(output_re_scale) || !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len = (vDSP_Length) width;
    for (size_t row = 0U; row < height; ++row) {
        double row_sample_y = sample_y0 + (double) row * dy;
        double u_start      = u_x * sample_x0 + u_y * row_sample_y;
        double u_step       = u_x * dx;
        double v_start      = v_x * sample_x0 + v_y * row_sample_y;
        double v_step       = v_x * dx;

        if (!steerable_wavelet_vdsp_eval_row(
                state, width, u_start, u_step, v_start, v_step, steer, phase_t)) {
            return false;
        }

        size_t offset = row * width;
        if (!is_complex) {
            double* row_ptr = dst_real + offset;
            vDSP_vsmaD(state->vdsp_accum_re, 1, &output_re_scale, row_ptr, 1, row_ptr, 1, len);
        } else {
            SimComplexDouble* row_ptr = dst_complex + offset;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vDSP_vsmaD(state->vdsp_accum_re, 1, &output_re_scale, row_re, 2, row_re, 2, len);
            if (output_im_scale != 0.0) {
                double neg_im = -output_im_scale;
                vDSP_vsmaD(state->vdsp_accum_im, 1, &neg_im, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_accum_re, 1, &output_im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_accum_im, 1, &output_re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static SimResult steerable_wavelet_fill_static_cache(void*                               userdata,
                                                     const SimStimulusStaticCacheLayout* layout,
                                                     bool                                need_imag,
                                                     double*                             out_real,
                                                     double*                             out_imag) {
    SimStimulusSteerableWaveletState* state = (SimStimulusSteerableWaveletState*) userdata;
    if (state == NULL || layout == NULL || out_real == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    {
        double steer   = steerable_wavelet_wrap_pi(state->config.orientation);
        double phase_t = state->config.phase;
        for (size_t idx = 0U; idx < layout->count; ++idx) {
            size_t ix       = 0U;
            size_t iy       = 0U;
            double x        = 0.0;
            double y        = 0.0;
            double base_re  = 0.0;
            double base_im  = 0.0;
            double value_re = 0.0;
            double value_im = 0.0;

            sim_stimulus_static_cache_index_to_xy(layout, idx, &ix, &iy);
            x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
            y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;

            steerable_wavelet_eval_xy_raw(state, x, y, 0.0, steer, phase_t, &base_re, &base_im);
            steerable_wavelet_apply_output(
                state, base_re, base_im, need_imag, &value_re, &value_im);

            out_real[idx] = isfinite(value_re) ? value_re : 0.0;
            if (need_imag && out_imag != NULL) {
                out_imag[idx] = isfinite(value_im) ? value_im : 0.0;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult steerable_wavelet_ensure_static_cache(SimStimulusSteerableWaveletState*   state,
                                                       const SimStimulusStaticCacheLayout* layout,
                                                       bool need_imag) {
    if (state == NULL || layout == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    return sim_stimulus_static_cache_ensure(
        &state->cache, layout, need_imag, steerable_wavelet_fill_static_cache, state);
}

static SimResult steerable_wavelet_ensure_axis_cache(SimStimulusSteerableWaveletState*   state,
                                                     const SimStimulusStaticCacheLayout* layout,
                                                     double                              t,
                                                     double                              steer,
                                                     double                              phase_t) {
    if (state == NULL || layout == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->axis_cache_valid && state->axis_extent_x == layout->extent_x &&
        state->axis_extent_y == layout->extent_y && state->axis_cache_time == t) {
        return SIM_RESULT_OK;
    }

    SimResult prep =
        steerable_wavelet_ensure_axis_capacity(state, layout->extent_x, layout->extent_y);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    for (size_t ix = 0U; ix < layout->extent_x; ++ix) {
        double x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
        steerable_wavelet_eval_uv(
            state, x, 0.0, steer, phase_t, &state->axis_real_x[ix], &state->axis_imag_x[ix]);
    }
    for (size_t iy = 0U; iy < layout->extent_y; ++iy) {
        double y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;
        steerable_wavelet_eval_uv(
            state, y, 0.0, steer, phase_t, &state->axis_real_y[iy], &state->axis_imag_y[iy]);
    }

    state->axis_extent_x    = layout->extent_x;
    state->axis_extent_y    = layout->extent_y;
    state->axis_cache_time  = t;
    state->axis_cache_valid = true;
    return SIM_RESULT_OK;
}

static double kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    {
        double value = kernel->params[param];
        return isfinite(value) ? value : 0.0;
    }
}

static SimResult steerable_wavelet_kernel_value(SimStimulusSteerableWaveletState* state,
                                                const KernelIR*                   kernel,
                                                size_t                            element_index,
                                                bool                              imag_component,
                                                bool                              complex_output,
                                                double*                           out_value) {
    const SimStimulusSteerableWaveletConfig* cfg;
    const SimKernelIRBinding*                binding;
    const size_t*                            shape;
    const size_t*                            strides;
    size_t                                   rank;
    SimStimulusStaticCacheLayout             layout      = { 0 };
    bool                                     have_layout = false;
    bool                                     separable;
    double                                   dt;
    double                                   t;
    double                                   scale;

    if (state == NULL || kernel == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    cfg = &state->config;
    if (cfg->amplitude == 0.0 || cfg->scale_count == 0U) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }
    if (kernel->bindings == NULL || kernel->binding_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    dt    = kernel_param_value(kernel, SIM_IR_PARAM_DT);
    t     = kernel_param_value(kernel, SIM_IR_PARAM_TIME) + cfg->time_offset;
    scale = cfg->scale_by_dt ? dt : 1.0;

    binding = &kernel->bindings[0];
    shape   = binding->shape;
    strides = binding->strides;
    rank    = binding->rank;
    if ((shape == NULL || strides == NULL || rank == 0U) && binding->field != NULL) {
        shape   = binding->field->layout.shape;
        strides = binding->field->layout.strides;
        rank    = binding->field->layout.rank;
    }
    if (shape == NULL || strides == NULL || rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_stimulus_static_cache_layout_from_arrays(shape, strides, rank, &layout) ==
        SIM_RESULT_OK) {
        have_layout = true;
    }

    if (have_layout && steerable_wavelet_is_time_invariant(state)) {
        SimResult prep = steerable_wavelet_ensure_static_cache(state, &layout, complex_output);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }
        {
            double value = imag_component ? state->cache.imag[element_index]
                                          : state->cache.real[element_index];
            value *= scale;
            *out_value = isfinite(value) ? value : 0.0;
            return SIM_RESULT_OK;
        }
    }

    separable = (!cfg->use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    if (have_layout && separable) {
        double    steer   = steerable_wavelet_wrap_pi(cfg->orientation + cfg->orientation_rate * t);
        double    phase_t = cfg->phase - cfg->omega * t;
        SimResult prep    = steerable_wavelet_ensure_axis_cache(state, &layout, t, steer, phase_t);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }
    }

    {
        size_t ix       = 0U;
        size_t iy       = 0U;
        double x        = 0.0;
        double y        = 0.0;
        double base_re  = 0.0;
        double base_im  = 0.0;
        double value_re = 0.0;
        double value_im = 0.0;
        double steer    = steerable_wavelet_wrap_pi(cfg->orientation + cfg->orientation_rate * t);
        double phase_t  = cfg->phase - cfg->omega * t;

        if (binding->field != NULL) {
            if (sim_field_index_to_xy(binding->field, element_index, &ix, &iy) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
        } else if (sim_kernel_binding_index_to_xy(binding, element_index, &ix, &iy) !=
                   SIM_RESULT_OK) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        x = cfg->coord.origin_x + (double) ix * cfg->coord.spacing_x;
        y = cfg->coord.origin_y + (double) iy * cfg->coord.spacing_y;

        if (separable && have_layout && state->axis_cache_valid && ix < state->axis_extent_x &&
            iy < state->axis_extent_y) {
            steerable_wavelet_combine_separable(cfg,
                                                state->axis_real_x[ix],
                                                state->axis_imag_x[ix],
                                                state->axis_real_y[iy],
                                                state->axis_imag_y[iy],
                                                &base_re,
                                                &base_im);
        } else {
            steerable_wavelet_eval_xy_raw(state, x, y, t, steer, phase_t, &base_re, &base_im);
        }

        steerable_wavelet_apply_output(
            state, base_re, base_im, complex_output, &value_re, &value_im);
        {
            double value = imag_component ? value_im : value_re;
            value *= scale;
            *out_value = isfinite(value) ? value : 0.0;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult steerable_wavelet_ir_eval_real(void*           userdata,
                                                const KernelIR* kernel,
                                                size_t          element_index,
                                                size_t          component,
                                                double*         out_value) {
    (void) component;

    return steerable_wavelet_kernel_value((SimStimulusSteerableWaveletState*) userdata,
                                          kernel,
                                          element_index,
                                          false,
                                          false,
                                          out_value);
}

static SimResult steerable_wavelet_ir_eval_complex_real(void*           userdata,
                                                        const KernelIR* kernel,
                                                        size_t          element_index,
                                                        size_t          component,
                                                        double*         out_value) {
    (void) component;

    return steerable_wavelet_kernel_value((SimStimulusSteerableWaveletState*) userdata,
                                          kernel,
                                          element_index,
                                          false,
                                          true,
                                          out_value);
}

static SimResult steerable_wavelet_ir_eval_complex_imag(void*           userdata,
                                                        const KernelIR* kernel,
                                                        size_t          element_index,
                                                        size_t          component,
                                                        double*         out_value) {
    (void) component;

    return steerable_wavelet_kernel_value(
        (SimStimulusSteerableWaveletState*) userdata, kernel, element_index, true, true, out_value);
}

static void steerable_wavelet_destroy(void* state_ptr) {
    SimStimulusSteerableWaveletState* state = (SimStimulusSteerableWaveletState*) state_ptr;
    if (state == NULL) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
#endif
    free(state->band_k_values);
    free(state->band_log_radius_values);
    free(state->axis_real_x);
    free(state->axis_imag_x);
    free(state->axis_real_y);
    free(state->axis_imag_y);
    sim_stimulus_static_cache_destroy(&state->cache);
    free(state);
}

static SimResult steerable_wavelet_step(void*               state_ptr,
                                        struct SimContext*  context,
                                        struct SimOperator* self,
                                        size_t              substep_index,
                                        double              dt_sub,
                                        void*               scratch,
                                        size_t              scratch_size) {
    SimStimulusSteerableWaveletState* state = (SimStimulusSteerableWaveletState*) state_ptr;
    SimField*                         field;
    SimStimulusStaticCacheLayout      layout      = { 0 };
    bool                              have_layout = false;
    bool                              is_complex;
    size_t                            count = 0U;
    double                            scale;
    double                            t;
    double                            steer;
    double                            phase_t;
    bool                              separable;

    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    is_complex = sim_field_is_complex(field);
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

    if (sim_stimulus_static_cache_layout_from_arrays(
            field->layout.shape, field->layout.strides, field->layout.rank, &layout) ==
            SIM_RESULT_OK &&
        layout.count == count) {
        have_layout = true;
    }

    scale = state->config.scale_by_dt ? dt_sub : 1.0;
    t     = sim_context_time(context) + state->config.time_offset;
    steer =
        steerable_wavelet_wrap_pi(state->config.orientation + state->config.orientation_rate * t);
    phase_t = state->config.phase - state->config.omega * t;
    separable =
        (!state->config.use_wavevector && state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

    if (have_layout && steerable_wavelet_is_time_invariant(state)) {
        SimResult prep = steerable_wavelet_ensure_static_cache(state, &layout, is_complex);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }

        if (!is_complex) {
            double* dst = (double*) sim_field_data(field);
            if (dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            for (size_t idx = 0U; idx < count; ++idx) {
                dst[idx] += scale * state->cache.real[idx];
            }
        } else {
            SimComplexDouble* dst = sim_field_complex_data(field);
            if (dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            for (size_t idx = 0U; idx < count; ++idx) {
                dst[idx].re += scale * state->cache.real[idx];
                dst[idx].im += scale * state->cache.imag[idx];
            }
        }
        return SIM_RESULT_OK;
    }

#if defined(SIM_HAVE_VDSP)
    if (steerable_wavelet_try_vdsp_rows(state,
                                        field,
                                        is_complex,
                                        is_complex ? NULL : (double*) sim_field_data(field),
                                        is_complex ? sim_field_complex_data(field) : NULL,
                                        count,
                                        scale,
                                        t,
                                        steer,
                                        phase_t)) {
        return SIM_RESULT_OK;
    }
#endif

    if (have_layout && separable) {
        SimResult prep = steerable_wavelet_ensure_axis_cache(state, &layout, t, steer, phase_t);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }
    }

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t idx = 0U; idx < count; ++idx) {
            size_t ix       = 0U;
            size_t iy       = 0U;
            double x        = 0.0;
            double y        = 0.0;
            double base_re  = 0.0;
            double base_im  = 0.0;
            double value_re = 0.0;
            double value_im = 0.0;

            if (have_layout) {
                sim_stimulus_static_cache_index_to_xy(&layout, idx, &ix, &iy);
            } else if (sim_field_index_to_xy(field, idx, &ix, &iy) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
            y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;

            if (separable && have_layout && state->axis_cache_valid && ix < state->axis_extent_x &&
                iy < state->axis_extent_y) {
                steerable_wavelet_combine_separable(&state->config,
                                                    state->axis_real_x[ix],
                                                    state->axis_imag_x[ix],
                                                    state->axis_real_y[iy],
                                                    state->axis_imag_y[iy],
                                                    &base_re,
                                                    &base_im);
            } else {
                steerable_wavelet_eval_xy_raw(state, x, y, t, steer, phase_t, &base_re, &base_im);
            }

            steerable_wavelet_apply_output(state, base_re, base_im, false, &value_re, &value_im);
            (void) value_im;
            if (isfinite(value_re)) {
                dst[idx] += scale * value_re;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t idx = 0U; idx < count; ++idx) {
            size_t ix       = 0U;
            size_t iy       = 0U;
            double x        = 0.0;
            double y        = 0.0;
            double base_re  = 0.0;
            double base_im  = 0.0;
            double value_re = 0.0;
            double value_im = 0.0;

            if (have_layout) {
                sim_stimulus_static_cache_index_to_xy(&layout, idx, &ix, &iy);
            } else if (sim_field_index_to_xy(field, idx, &ix, &iy) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
            y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;

            if (separable && have_layout && state->axis_cache_valid && ix < state->axis_extent_x &&
                iy < state->axis_extent_y) {
                steerable_wavelet_combine_separable(&state->config,
                                                    state->axis_real_x[ix],
                                                    state->axis_imag_x[ix],
                                                    state->axis_real_y[iy],
                                                    state->axis_imag_y[iy],
                                                    &base_re,
                                                    &base_im);
            } else {
                steerable_wavelet_eval_xy_raw(state, x, y, t, steer, phase_t, &base_re, &base_im);
            }

            steerable_wavelet_apply_output(state, base_re, base_im, true, &value_re, &value_im);
            if (isfinite(value_re) && isfinite(value_im)) {
                dst[idx].re += scale * value_re;
                dst[idx].im += scale * value_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimStimulusSteerableWaveletState* steerable_wavelet_operator_state(SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    if (op->kernel != NULL) {
        return (SimStimulusSteerableWaveletState*) sim_operator_payload(op);
    }
    return (SimStimulusSteerableWaveletState*) sim_split_state(op);
}

SimResult
sim_add_stimulus_steerable_wavelet_operator(struct SimContext*                       context,
                                            const SimStimulusSteerableWaveletConfig* config,
                                            size_t*                                  out_index) {
    SimStimulusSteerableWaveletConfig local = { 0 };
    SimStimulusSteerableWaveletState* state;
    char                              name[SIM_OPERATOR_NAME_MAX + 1U];
    SimOperatorInfo                   info;
    SimOperatorConfig                 op_config;
    SimResult                         result            = SIM_RESULT_OK;
    bool                              registered_kernel = false;
    bool                              needs_complex;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (config != NULL) {
        local = *config;
    }

    steerable_wavelet_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_steerable_wavelet",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    state = (SimStimulusSteerableWaveletState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    result = steerable_wavelet_ensure_scale_capacity(state, local.scale_count);
    if (result != SIM_RESULT_OK) {
        steerable_wavelet_destroy(state);
        return result;
    }

    state->config = local;
    steerable_wavelet_refresh_precomputed(state);
    steerable_wavelet_refresh_symbolic(state);

    sim_operator_make_unique_name(name, sizeof(name), "stimulus_steerable_wavelet");

    info                   = sim_operator_info_defaults();
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
    info.abstract_id       = "stimulus_steerable_wavelet";
    sim_operator_info_set_schema_identity(&info, "stimulus_steerable_wavelet");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    needs_complex = sim_field_is_complex(sim_context_field(context, state->config.field_index));

    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    op_config = sim_operator_config_defaults();

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_steerable_wavelet")) {
        SimField* field = sim_context_field(context, local.field_index);
        if (field != NULL) {
            bool field_is_complex = sim_field_is_complex(field);
            if ((!field_is_complex && field->element_size != sizeof(double)) ||
                (field_is_complex && field->element_size != sizeof(SimComplexDouble))) {
                field = NULL;
            }
        }

        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc      = { 0 };
                SimOperatorDescriptor              kdesc            = { 0 };
                bool                               field_is_complex = sim_field_is_complex(field);
                SimIRType                          field_type =
                    field_is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId sample_node =
                    field_is_complex
                        ? sim_ir_builder_stateful(builder,
                                                  steerable_wavelet_ir_eval_complex_real,
                                                  state,
                                                  "stimulus_steerable_wavelet")
                        : sim_ir_builder_stateful(builder,
                                                  steerable_wavelet_ir_eval_real,
                                                  state,
                                                  "stimulus_steerable_wavelet");

                if (field_is_complex && sample_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId sample_im =
                        sim_ir_builder_stateful(builder,
                                                steerable_wavelet_ir_eval_complex_imag,
                                                state,
                                                "stimulus_steerable_wavelet_im");
                    if (sample_im != SIM_IR_INVALID_NODE) {
                        sample_node = sim_ir_builder_complex_pack(builder, sample_node, sample_im);
                    }
                }

                if (field_node != SIM_IR_INVALID_NODE && sample_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId sum =
                        sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, field_node, sample_node);
                    if (sum != SIM_IR_INVALID_NODE) {
                        bindings[0].ir_field_index      = 0U;
                        bindings[0].context_field_index = local.field_index;

                        outputs[0].ir_field_index = 0U;
                        outputs[0].expression     = sum;

                        kernel_desc.builder           = builder;
                        kernel_desc.bindings          = bindings;
                        kernel_desc.binding_count     = 1U;
                        kernel_desc.outputs           = outputs;
                        kernel_desc.output_count      = 1U;
                        kernel_desc.param_count       = (size_t) SIM_IR_PARAM_TIME + 1U;
                        kernel_desc.required_features = 0ULL;

                        kdesc.name     = name;
                        kdesc.evaluate = NULL;
                        kdesc.destroy  = steerable_wavelet_destroy;
                        kdesc.userdata = state;
                        kdesc.kernel   = &kernel_desc;
                        kdesc.info     = info;
                        kdesc.config   = op_config;
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
    }

    if (registered_kernel) {
        return result;
    }

    {
        SimSplitPort port = { .context_field_index = state->config.field_index,
                              .require_complex     = needs_complex };

        SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

        SimSplitSubstep substep = { .name              = NULL,
                                    .fn                = steerable_wavelet_step,
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
                                    .symbolic      = steerable_wavelet_symbolic,
                                    .destroy       = steerable_wavelet_destroy,
                                    .info          = info,
                                    .config        = op_config,
                                    .scratch       = { 0U, 0U } };

        result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
        if (result != SIM_RESULT_OK) {
            steerable_wavelet_destroy(state);
        }
        return result;
    }
}

SimResult sim_stimulus_steerable_wavelet_config(struct SimContext*                 context,
                                                size_t                             operator_index,
                                                SimStimulusSteerableWaveletConfig* out_config) {
    SimOperator*                      op;
    SimStimulusSteerableWaveletState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = steerable_wavelet_operator_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_steerable_wavelet_update(struct SimContext* context,
                                                size_t             operator_index,
                                                const SimStimulusSteerableWaveletConfig* config) {
    SimOperator*                      op;
    SimStimulusSteerableWaveletState* state;
    SimStimulusSteerableWaveletConfig local;
    SimResult                         prep;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = steerable_wavelet_operator_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local = state->config;
    if (config != NULL) {
        local = *config;
    }

    steerable_wavelet_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context,
        sim_operator_schema_key_or(op, "stimulus_steerable_wavelet"),
        true,
        local.scale_by_dt);

    prep = steerable_wavelet_ensure_scale_capacity(state, local.scale_count);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    state->config = local;
    steerable_wavelet_refresh_precomputed(state);
    steerable_wavelet_invalidate_caches(state);
    steerable_wavelet_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
