#include "oakfield/operators/stimulus/log_spectral_grid.h"

#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_seed.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_LOG_GRID_EPS 1.0e-12
#define STIM_LOG_GRID_MAX_RADIAL_BINS 128U
#define STIM_LOG_GRID_MAX_ANGULAR_BINS 256U
#define STIM_LOG_GRID_MAX_MODES 16384U
#define STIM_LOG_GRID_VDSP_MIN_LEN 64U

typedef struct stim_log_grid_pcg32 {
    uint64_t state;
    uint64_t inc;
} stim_log_grid_pcg32_t;

static uint32_t stim_log_grid_pcg32_random(stim_log_grid_pcg32_t* rng) {
    uint64_t old        = rng->state;
    rng->state          = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t) (old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void
stim_log_grid_pcg32_seed(stim_log_grid_pcg32_t* rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc   = (initseq << 1u) | 1u;
    (void) stim_log_grid_pcg32_random(rng);
    rng->state += initstate;
    (void) stim_log_grid_pcg32_random(rng);
}

static double stim_log_grid_uniform(stim_log_grid_pcg32_t* rng) {
    return ldexp(stim_log_grid_pcg32_random(rng), -32);
}

typedef struct SimStimulusLogSpectralGridState {
    SimStimulusLogSpectralGridConfig config;
    double*                          mode_kx_values;
    double*                          mode_ky_values;
    double*                          mode_phase_values;
    double*                          mode_weight_values;
    unsigned int                     allocated_modes;
    unsigned int                     active_modes;
    double                           mode_norm;
    char                             symbolic[224];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta;
    double* vdsp_cos;
    double* vdsp_sin;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
} SimStimulusLogSpectralGridState;

static unsigned int log_spectral_grid_mode_count(const SimStimulusLogSpectralGridConfig* config) {
    if (config == NULL || config->radial_bins == 0U || config->angular_bins == 0U) {
        return 0U;
    }

    uint64_t total = (uint64_t) config->radial_bins * (uint64_t) config->angular_bins;
    if (total > STIM_LOG_GRID_MAX_MODES) {
        total = STIM_LOG_GRID_MAX_MODES;
    }
    return (unsigned int) total;
}

static void log_spectral_grid_normalize(SimStimulusLogSpectralGridConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }

    if (!isfinite(config->k_min)) {
        config->k_min = 0.2;
    }
    if (!isfinite(config->k_max)) {
        config->k_max = 2.0;
    }
    config->k_min = fabs(config->k_min);
    config->k_max = fabs(config->k_max);
    if (config->k_min <= STIM_LOG_GRID_EPS) {
        config->k_min = 0.2;
    }
    if (config->k_max <= STIM_LOG_GRID_EPS) {
        config->k_max = 2.0;
    }
    if (config->k_min > config->k_max) {
        double tmp    = config->k_min;
        config->k_min = config->k_max;
        config->k_max = tmp;
    }

    if (config->radial_bins == 0U) {
        config->radial_bins = 6U;
    }
    if (config->radial_bins > STIM_LOG_GRID_MAX_RADIAL_BINS) {
        config->radial_bins = STIM_LOG_GRID_MAX_RADIAL_BINS;
    }
    if (config->angular_bins == 0U) {
        config->angular_bins = 12U;
    }
    if (config->angular_bins > STIM_LOG_GRID_MAX_ANGULAR_BINS) {
        config->angular_bins = STIM_LOG_GRID_MAX_ANGULAR_BINS;
    }

    uint64_t total = (uint64_t) config->radial_bins * (uint64_t) config->angular_bins;
    if (total > STIM_LOG_GRID_MAX_MODES) {
        uint64_t per_radial = STIM_LOG_GRID_MAX_MODES / (uint64_t) config->radial_bins;
        if (per_radial == 0U) {
            per_radial = 1U;
        }
        config->angular_bins = (unsigned int) per_radial;
    }

    if (!isfinite(config->spectral_slope)) {
        config->spectral_slope = 0.0;
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
        (fabs(config->kx) > STIM_LOG_GRID_EPS || fabs(config->ky) > STIM_LOG_GRID_EPS)) {
        config->use_wavevector = true;
    }

    if (config->use_wavevector && fabs(config->kx) <= STIM_LOG_GRID_EPS &&
        fabs(config->ky) <= STIM_LOG_GRID_EPS) {
        config->kx = 1.0;
        config->ky = 0.0;
    }

    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void log_spectral_grid_refresh_symbolic(SimStimulusLogSpectralGridState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusLogSpectralGridConfig* cfg = &state->config;
    if (cfg->use_wavevector) {
        (void) snprintf(
            state->symbolic,
            sizeof(state->symbolic),
            "log_spectral_grid A=%.3g k=[%.3g,%.3g] R=%u A=%u slope=%.3g kbase=(%.3g,%.3g)",
            cfg->amplitude,
            cfg->k_min,
            cfg->k_max,
            cfg->radial_bins,
            cfg->angular_bins,
            cfg->spectral_slope,
            cfg->kx,
            cfg->ky);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "log_spectral_grid A=%.3g k=[%.3g,%.3g] R=%u A=%u slope=%.3g",
                        cfg->amplitude,
                        cfg->k_min,
                        cfg->k_max,
                        cfg->radial_bins,
                        cfg->angular_bins,
                        cfg->spectral_slope);
    }
#else
    (void) state;
#endif
}

static const char* log_spectral_grid_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusLogSpectralGridState* state =
        (const SimStimulusLogSpectralGridState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void log_spectral_grid_map_sample_coord(const SimStimulusLogSpectralGridConfig* cfg,
                                               double                                  sample_x,
                                               double                                  sample_y,
                                               double                                  t,
                                               double*                                 out_u,
                                               double*                                 out_v) {
    if (cfg->use_wavevector) {
        double norm = hypot(cfg->kx, cfg->ky);
        double ux   = 1.0;
        double uy   = 0.0;
        if (norm > STIM_LOG_GRID_EPS) {
            ux = cfg->kx / norm;
            uy = cfg->ky / norm;
        }
        *out_u = ux * sample_x + uy * sample_y;
        *out_v = -uy * sample_x + ux * sample_y;
        return;
    }

    const SimStimulusCoordConfig* coord = &cfg->coord;
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
            double dx = sample_x - coord->center_x;
            double dy = sample_y - coord->center_y;
            *out_u    = hypot(dx, dy);
            *out_v    = atan2(dy, dx);
            return;
        }
        case SIM_STIMULUS_COORD_AZIMUTH: {
            double dx = sample_x - coord->center_x;
            double dy = sample_y - coord->center_y;
            *out_u    = atan2(dy, dx);
            *out_v    = hypot(dx, dy);
            return;
        }
        case SIM_STIMULUS_COORD_ELLIPTIC: {
            double dx = sample_x - coord->center_x;
            double dy = sample_y - coord->center_y;
            sim_stimulus_coord_elliptic_polar(coord, dx, dy, out_u, out_v);
            return;
        }
        case SIM_STIMULUS_COORD_SPIRAL: {
            double dx = sample_x - coord->center_x;
            double dy = sample_y - coord->center_y;
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

static void log_spectral_grid_map_coord(const SimStimulusLogSpectralGridConfig* cfg,
                                        double                                  x,
                                        double                                  y,
                                        double                                  t,
                                        double*                                 out_u,
                                        double*                                 out_v) {
    double sample_x = x;
    double sample_y = y;

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);
    log_spectral_grid_map_sample_coord(cfg, sample_x, sample_y, t, out_u, out_v);
}

static SimResult log_spectral_grid_rebuild_modes(SimStimulusLogSpectralGridState* state) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    unsigned int desired = log_spectral_grid_mode_count(&state->config);
    if (desired == 0U) {
        state->active_modes    = 0U;
        state->mode_norm       = 1.0;
        state->allocated_modes = 0U;
        free(state->mode_kx_values);
        free(state->mode_ky_values);
        free(state->mode_phase_values);
        free(state->mode_weight_values);
        state->mode_kx_values     = NULL;
        state->mode_ky_values     = NULL;
        state->mode_phase_values  = NULL;
        state->mode_weight_values = NULL;
        return SIM_RESULT_OK;
    }

    double* mode_kx_values     = (double*) malloc((size_t) desired * sizeof(double));
    double* mode_ky_values     = (double*) malloc((size_t) desired * sizeof(double));
    double* mode_phase_values  = (double*) malloc((size_t) desired * sizeof(double));
    double* mode_weight_values = (double*) malloc((size_t) desired * sizeof(double));
    if (mode_kx_values == NULL || mode_ky_values == NULL || mode_phase_values == NULL ||
        mode_weight_values == NULL) {
        free(mode_kx_values);
        free(mode_ky_values);
        free(mode_phase_values);
        free(mode_weight_values);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    stim_log_grid_pcg32_t rng;
    stim_log_grid_pcg32_seed(&rng, state->config.seed, state->config.seed ^ 0x9E3779B97F4A7C15ULL);

    double log_k_min = log(fmax(state->config.k_min, STIM_LOG_GRID_EPS));
    double log_k_max = log(fmax(state->config.k_max, STIM_LOG_GRID_EPS));
    double slope     = state->config.spectral_slope;

    unsigned int index  = 0U;
    double       sum_sq = 0.0;
    for (unsigned int r = 0U; r < state->config.radial_bins && index < desired; ++r) {
        double k = 0.0;
        if (state->config.radial_bins <= 1U || fabs(log_k_max - log_k_min) <= STIM_LOG_GRID_EPS) {
            k = exp(0.5 * (log_k_min + log_k_max));
        } else {
            double frac = (double) r / (double) (state->config.radial_bins - 1U);
            k           = exp(log_k_min + frac * (log_k_max - log_k_min));
        }

        for (unsigned int a = 0U; a < state->config.angular_bins && index < desired; ++a) {
            double angle          = 2.0 * M_PI * ((double) a / (double) state->config.angular_bins);
            mode_kx_values[index] = k * cos(angle);
            mode_ky_values[index] = k * sin(angle);

            double mode_phase = 0.0;
            if (state->config.random_phase) {
                mode_phase = 2.0 * M_PI * stim_log_grid_uniform(&rng);
            }
            mode_phase_values[index] = mode_phase;

            double weight = 1.0;
            if (slope != 0.0 && k > STIM_LOG_GRID_EPS) {
                weight = pow(k, -0.5 * slope);
            }
            mode_weight_values[index] = weight;
            sum_sq += weight * weight;

            ++index;
        }
    }

    if (index == 0U) {
        free(mode_kx_values);
        free(mode_ky_values);
        free(mode_phase_values);
        free(mode_weight_values);
        return SIM_RESULT_INVALID_STATE;
    }

    free(state->mode_kx_values);
    free(state->mode_ky_values);
    free(state->mode_phase_values);
    free(state->mode_weight_values);

    state->mode_kx_values     = mode_kx_values;
    state->mode_ky_values     = mode_ky_values;
    state->mode_phase_values  = mode_phase_values;
    state->mode_weight_values = mode_weight_values;
    state->allocated_modes    = desired;
    state->active_modes       = index;
    state->mode_norm          = (sum_sq > STIM_LOG_GRID_EPS) ? sqrt(sum_sq / (double) index) : 1.0;
    return SIM_RESULT_OK;
}

static void log_spectral_grid_eval_uv(const SimStimulusLogSpectralGridState* state,
                                      double                                 u,
                                      double                                 v,
                                      double                                 t,
                                      double*                                out_re,
                                      double*                                out_im) {
    if (state == NULL || out_re == NULL || out_im == NULL || state->active_modes == 0U ||
        state->mode_kx_values == NULL || state->mode_ky_values == NULL ||
        state->mode_phase_values == NULL || state->mode_weight_values == NULL) {
        if (out_re != NULL) {
            *out_re = 0.0;
        }
        if (out_im != NULL) {
            *out_im = 0.0;
        }
        return;
    }

    double theta = state->config.orientation + state->config.orientation_rate * t;
    double s     = sin(theta);
    double c     = cos(theta);

    double ur = u * c + v * s;
    double vr = -u * s + v * c;

    double re_sum = 0.0;
    double im_sum = 0.0;
    double norm   = 1.0;
    if (state->active_modes > 0U) {
        norm = 1.0 / sqrt((double) state->active_modes);
    }
    double mode_norm = (state->mode_norm > STIM_LOG_GRID_EPS) ? state->mode_norm : 1.0;

    for (unsigned int i = 0U; i < state->active_modes; ++i) {
        double arg  = state->mode_kx_values[i] * ur + state->mode_ky_values[i] * vr -
                      state->config.omega * t + state->config.phase + state->mode_phase_values[i];
        double cphi = cos(arg);
        double sphi = sin(arg);
        double w    = norm * (state->mode_weight_values[i] / mode_norm);
        re_sum += w * cphi;
        im_sum += w * sphi;
    }

    *out_re = re_sum;
    *out_im = im_sum;
}

static void log_spectral_grid_destroy(void* state_ptr) {
    SimStimulusLogSpectralGridState* state = (SimStimulusLogSpectralGridState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state->mode_kx_values);
    free(state->mode_ky_values);
    free(state->mode_phase_values);
    free(state->mode_weight_values);
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
    state->vdsp_block = NULL;
#endif
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool log_spectral_grid_vdsp_ensure_buffers(SimStimulusLogSpectralGridState* state,
                                                  size_t                           width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_block != NULL && state->vdsp_capacity >= width) {
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
    state->vdsp_theta    = block;
    state->vdsp_cos      = block + width;
    state->vdsp_sin      = block + width * 2U;
    state->vdsp_accum_re = block + width * 3U;
    state->vdsp_accum_im = block + width * 4U;
    return true;
}

static bool log_spectral_grid_linear_map(const SimStimulusLogSpectralGridConfig* cfg,
                                         double*                                 out_u_x,
                                         double*                                 out_u_y,
                                         double*                                 out_v_x,
                                         double*                                 out_v_y) {
    if (cfg == NULL || out_u_x == NULL || out_u_y == NULL || out_v_x == NULL || out_v_y == NULL) {
        return false;
    }

    if (cfg->use_wavevector) {
        double norm = hypot(cfg->kx, cfg->ky);
        double ux   = 1.0;
        double uy   = 0.0;
        if (norm > STIM_LOG_GRID_EPS) {
            ux = cfg->kx / norm;
            uy = cfg->ky / norm;
        }
        *out_u_x = ux;
        *out_u_y = uy;
        *out_v_x = -uy;
        *out_v_y = ux;
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
        case SIM_STIMULUS_COORD_ANGLE: {
            double s = sin(cfg->coord.angle);
            double c = cos(cfg->coord.angle);
            *out_u_x = c;
            *out_u_y = s;
            *out_v_x = -s;
            *out_v_y = c;
            return true;
        }
        default:
            break;
    }

    return false;
}

static bool log_spectral_grid_try_vdsp_linear_rows(SimStimulusLogSpectralGridState* state,
                                                   const SimField*                  field,
                                                   bool                             is_complex,
                                                   double*                          dst_real,
                                                   SimComplexDouble*                dst_complex,
                                                   size_t                           count,
                                                   double                           scale,
                                                   double                           t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL || state->mode_kx_values == NULL ||
        state->mode_ky_values == NULL || state->mode_phase_values == NULL ||
        state->mode_weight_values == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR ||
        !log_spectral_grid_linear_map(&state->config, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_LOG_GRID_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!log_spectral_grid_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double mode_norm = (state->mode_norm > STIM_LOG_GRID_EPS) ? state->mode_norm : 1.0;
    double inv_sqrtM = 1.0 / sqrt((double) state->active_modes);
    double gain      = scale * state->config.amplitude * inv_sqrtM / mode_norm;
    double orient    = state->config.orientation + state->config.orientation_rate * t;
    double orient_s  = sin(orient);
    double orient_c  = cos(orient);
    double phase_t   = -state->config.omega * t + state->config.phase;
    double x0        = state->config.coord.origin_x - state->config.coord.velocity_x * t;
    double y0        = state->config.coord.origin_y - state->config.coord.velocity_y * t;
    double dx        = state->config.coord.spacing_x;
    double dy        = state->config.coord.spacing_y;
    double du        = u_x * dx;
    double dv        = v_x * dx;
    double dur       = du * orient_c + dv * orient_s;
    double dvr       = -du * orient_s + dv * orient_c;

    if (!isfinite(gain) || !isfinite(orient_s) || !isfinite(orient_c) || !isfinite(phase_t) ||
        !isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(dur) ||
        !isfinite(dvr)) {
        return false;
    }
    if (gain == 0.0) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    double            sin_r      = 0.0;
    double            cos_r      = 1.0;
    if (is_complex && state->config.rotation != 0.0) {
        sin_r = sin(state->config.rotation);
        cos_r = cos(state->config.rotation);
    }

    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double u0       = u_x * x0 + u_y * sample_y;
        double v0       = v_x * x0 + v_y * sample_y;
        double ur0      = u0 * orient_c + v0 * orient_s;
        double vr0      = -u0 * orient_s + v0 * orient_c;

        if (!isfinite(ur0) || !isfinite(vr0)) {
            return false;
        }

        vDSP_vclrD(state->vdsp_accum_re, 1, len);
        vDSP_vclrD(state->vdsp_accum_im, 1, len);

        for (unsigned int i = 0U; i < state->active_modes; ++i) {
            double weight = gain * state->mode_weight_values[i];
            double start  = state->mode_kx_values[i] * ur0 + state->mode_ky_values[i] * vr0 +
                            phase_t + state->mode_phase_values[i];
            double step   = state->mode_kx_values[i] * dur + state->mode_ky_values[i] * dvr;
            if (!isfinite(weight) || !isfinite(start) || !isfinite(step)) {
                return false;
            }

            vDSP_vrampD(&start, &step, state->vdsp_theta, 1, len);
            if (is_complex) {
                vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
            } else {
                vvcos(state->vdsp_cos, state->vdsp_theta, &vforce_len);
            }
            vDSP_vsmaD(
                state->vdsp_cos, 1, &weight, state->vdsp_accum_re, 1, state->vdsp_accum_re, 1, len);
            if (is_complex) {
                vDSP_vsmaD(state->vdsp_sin,
                           1,
                           &weight,
                           state->vdsp_accum_im,
                           1,
                           state->vdsp_accum_im,
                           1,
                           len);
            }
        }

        size_t offset = row * width;
        if (!is_complex) {
            for (size_t i = 0U; i < width; ++i) {
                double value = state->vdsp_accum_re[i];
                if (isfinite(value)) {
                    dst_real[offset + i] += value;
                }
            }
            continue;
        }

        for (size_t i = 0U; i < width; ++i) {
            double re     = state->vdsp_accum_re[i];
            double im     = state->vdsp_accum_im[i];
            double out_re = re * cos_r - im * sin_r;
            double out_im = re * sin_r + im * cos_r;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst_complex[offset + i].re += out_re;
                dst_complex[offset + i].im += out_im;
            }
        }
    }

    return true;
}
#endif

static SimResult log_spectral_grid_step(void*               state_ptr,
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

    SimStimulusLogSpectralGridState* state = (SimStimulusLogSpectralGridState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (field->layout.rank == 0U || field->layout.rank > 2U) {
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

    if (count == 0U || state->config.amplitude == 0.0 || state->active_modes == 0U) {
        return SIM_RESULT_OK;
    }

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;
    double t     = sim_context_time(context) + state->config.time_offset;
    bool   separable =
        (!state->config.use_wavevector && state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    SimFieldPatch patch = { 0 };

#if defined(SIM_HAVE_VDSP)
    if (!separable && log_spectral_grid_try_vdsp_linear_rows(state,
                                                             field,
                                                             is_complex,
                                                             (double*) sim_field_data(field),
                                                             sim_field_complex_data(field),
                                                             count,
                                                             scale,
                                                             t)) {
        return SIM_RESULT_OK;
    }
#endif

    if (sim_field_patch_full(sim_field_width(field), sim_field_height(field), &patch) !=
        SIM_RESULT_OK) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t row_offset = 0U; row_offset < patch.height; ++row_offset) {
            SimStimulusCoordRow row       = { 0 };
            size_t              dst_index = (patch.y0 + row_offset) * patch.field_width + patch.x0;
            double              x         = 0.0;
            double              y         = 0.0;
            double              sample_x  = 0.0;
            double              sample_y  = 0.0;

            if (sim_stimulus_coord_patch_row(&state->config.coord, &patch, row_offset, t, &row) !=
                SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            x        = row.x;
            y        = row.y;
            sample_x = row.sample_x;
            sample_y = row.sample_y;

            for (size_t col = 0U; col < row.width; ++col) {
                double re = 0.0;
                double im = 0.0;

                if (separable) {
                    double rx = 0.0;
                    double ix = 0.0;
                    double ry = 0.0;
                    double iy = 0.0;
                    log_spectral_grid_eval_uv(state, x, 0.0, t, &rx, &ix);
                    log_spectral_grid_eval_uv(state, y, 0.0, t, &ry, &iy);
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
                    log_spectral_grid_map_sample_coord(
                        &state->config, sample_x, sample_y, t, &u, &v);
                    log_spectral_grid_eval_uv(state, u, v, t, &re, &im);
                }

                (void) im;
                double value = state->config.amplitude * re;
                if (isfinite(value)) {
                    dst[dst_index] += scale * value;
                }

                dst_index += 1U;
                x += row.x_step;
                y += row.y_step;
                sample_x += row.sample_x_step;
                sample_y += row.sample_y_step;
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

        for (size_t row_offset = 0U; row_offset < patch.height; ++row_offset) {
            SimStimulusCoordRow row       = { 0 };
            size_t              dst_index = (patch.y0 + row_offset) * patch.field_width + patch.x0;
            double              x         = 0.0;
            double              y         = 0.0;
            double              sample_x  = 0.0;
            double              sample_y  = 0.0;

            if (sim_stimulus_coord_patch_row(&state->config.coord, &patch, row_offset, t, &row) !=
                SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            x        = row.x;
            y        = row.y;
            sample_x = row.sample_x;
            sample_y = row.sample_y;

            for (size_t col = 0U; col < row.width; ++col) {
                double re = 0.0;
                double im = 0.0;

                if (separable) {
                    double rx = 0.0;
                    double ix = 0.0;
                    double ry = 0.0;
                    double iy = 0.0;
                    log_spectral_grid_eval_uv(state, x, 0.0, t, &rx, &ix);
                    log_spectral_grid_eval_uv(state, y, 0.0, t, &ry, &iy);
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
                    log_spectral_grid_map_sample_coord(
                        &state->config, sample_x, sample_y, t, &u, &v);
                    log_spectral_grid_eval_uv(state, u, v, t, &re, &im);
                }

                re *= state->config.amplitude;
                im *= state->config.amplitude;

                double out_re = re * cos_r - im * sin_r;
                double out_im = re * sin_r + im * cos_r;
                if (isfinite(out_re) && isfinite(out_im)) {
                    dst[dst_index].re += scale * out_re;
                    dst[dst_index].im += scale * out_im;
                }

                dst_index += 1U;
                x += row.x_step;
                y += row.y_step;
                sample_x += row.sample_x_step;
                sample_y += row.sample_y_step;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult
sim_add_stimulus_log_spectral_grid_operator(struct SimContext*                      context,
                                            const SimStimulusLogSpectralGridConfig* config,
                                            size_t*                                 out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusLogSpectralGridConfig local = { 0 };
    local.random_phase                     = true;
    if (config != NULL) {
        local = *config;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_log_spectral_grid"),
                                     sim_context_operator_count(context));
    }

    log_spectral_grid_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_log_spectral_grid",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusLogSpectralGridState* state =
        (SimStimulusLogSpectralGridState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config             = local;
    state->mode_kx_values     = NULL;
    state->mode_ky_values     = NULL;
    state->mode_phase_values  = NULL;
    state->mode_weight_values = NULL;
    state->allocated_modes    = 0U;
    state->active_modes       = 0U;
    state->mode_norm          = 1.0;

    SimResult prep = log_spectral_grid_rebuild_modes(state);
    if (prep != SIM_RESULT_OK) {
        log_spectral_grid_destroy(state);
        return prep;
    }
    log_spectral_grid_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_log_spectral_grid");

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
    info.abstract_id       = "stimulus_log_spectral_grid";
    sim_operator_info_set_schema_identity(&info, "stimulus_log_spectral_grid");
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
                                .fn                = log_spectral_grid_step,
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
                                .symbolic      = log_spectral_grid_symbolic,
                                .destroy       = log_spectral_grid_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        log_spectral_grid_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_log_spectral_grid_config(struct SimContext*                context,
                                                size_t                            operator_index,
                                                SimStimulusLogSpectralGridConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLogSpectralGridState* state = (SimStimulusLogSpectralGridState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_log_spectral_grid_update(struct SimContext* context,
                                                size_t             operator_index,
                                                const SimStimulusLogSpectralGridConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLogSpectralGridState* state = (SimStimulusLogSpectralGridState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusLogSpectralGridConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }
    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_log_spectral_grid"), operator_index);
    }

    log_spectral_grid_normalize(&local);
    state->config = local;

    SimResult prep = log_spectral_grid_rebuild_modes(state);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    log_spectral_grid_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
