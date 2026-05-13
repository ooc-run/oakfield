#include "oakfield/operators/stimulus/worley_noise.h"
#include "operators/common/operator_utils.h"
#include "static_cache.h"

#include "oakfield/backend.h"
#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_seed.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_WORLEY_FREQ_EPS 1.0e-9
#define STIM_WORLEY_MINKOWSKI_MIN 1.0
#define STIM_WORLEY_MINKOWSKI_MAX 16.0

typedef struct SimStimulusWorleyNoiseState {
    SimStimulusWorleyNoiseConfig config;
    SimStimulusStaticCache       cache;
    uint64_t                     hash_basis[2][2];
    char                         symbolic[160];
} SimStimulusWorleyNoiseState;

typedef struct SimStimulusWorleyLinearMap {
    double u0;
    double v0;
    double du_x;
    double dv_x;
    double du_y;
    double dv_y;
} SimStimulusWorleyLinearMap;

static uint64_t stim_worley_mix_u64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

static double stim_worley_uniform01(uint64_t value) {
    return (double) (stim_worley_mix_u64(value) >> 11U) * (1.0 / 9007199254740992.0);
}

static uint64_t stim_worley_hash_basis(uint64_t seed, unsigned channel, unsigned axis) {
    uint64_t hash = stim_worley_mix_u64(seed ^ 0xD1B54A32D192ED03ULL);
    hash          = stim_worley_mix_u64(hash ^ (uint64_t) (uint32_t) channel);
    hash          = stim_worley_mix_u64(hash ^ (uint64_t) axis * 0x94D049BB133111EBULL);
    return hash;
}

static uint64_t
stim_worley_hash_feature_from_basis(uint64_t basis, int64_t cell_x, int64_t cell_y) {
    uint64_t hash = stim_worley_mix_u64(basis ^ (uint64_t) cell_x);
    hash          = stim_worley_mix_u64(hash ^ ((uint64_t) cell_y << 1U));
    return hash;
}

static inline void stim_worley_update_best(double distance, double* f1, double* f2) {
    if (distance < *f1) {
        *f2 = *f1;
        *f1 = distance;
    } else if (distance < *f2) {
        *f2 = distance;
    }
}

static void worley_noise_normalize(SimStimulusWorleyNoiseConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->feature_frequency) ||
        fabs(config->feature_frequency) < STIM_WORLEY_FREQ_EPS) {
        config->feature_frequency = 4.0;
    }
    config->feature_frequency = fabs(config->feature_frequency);

    if (!isfinite(config->jitter)) {
        config->jitter = 1.0;
    }
    if (config->jitter < 0.0) {
        config->jitter = 0.0;
    }
    if (config->jitter > 1.0) {
        config->jitter = 1.0;
    }

    if (!isfinite(config->distance_exponent)) {
        config->distance_exponent = 2.0;
    }
    if (config->distance_exponent < STIM_WORLEY_MINKOWSKI_MIN) {
        config->distance_exponent = STIM_WORLEY_MINKOWSKI_MIN;
    }
    if (config->distance_exponent > STIM_WORLEY_MINKOWSKI_MAX) {
        config->distance_exponent = STIM_WORLEY_MINKOWSKI_MAX;
    }

    if (config->distance_metric < SIM_STIMULUS_WORLEY_EUCLIDEAN ||
        config->distance_metric > SIM_STIMULUS_WORLEY_MINKOWSKI) {
        config->distance_metric = SIM_STIMULUS_WORLEY_EUCLIDEAN;
    }
    if (config->output_mode < SIM_STIMULUS_WORLEY_F1 ||
        config->output_mode > SIM_STIMULUS_WORLEY_F2_MINUS_F1) {
        config->output_mode = SIM_STIMULUS_WORLEY_F2_MINUS_F1;
    }

    sim_stimulus_coord_normalize(&config->coord);

    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }
}

static const char* worley_metric_name(SimStimulusWorleyDistanceMetric metric) {
    switch (metric) {
        case SIM_STIMULUS_WORLEY_MANHATTAN:
            return "L1";
        case SIM_STIMULUS_WORLEY_CHEBYSHEV:
            return "Linf";
        case SIM_STIMULUS_WORLEY_MINKOWSKI:
            return "Lp";
        case SIM_STIMULUS_WORLEY_EUCLIDEAN:
        default:
            return "L2";
    }
}

static const char* worley_output_name(SimStimulusWorleyOutputMode mode) {
    switch (mode) {
        case SIM_STIMULUS_WORLEY_F1:
            return "F1";
        case SIM_STIMULUS_WORLEY_F2:
            return "F2";
        case SIM_STIMULUS_WORLEY_F2_MINUS_F1:
        default:
            return "F2-F1";
    }
}

static void worley_noise_refresh_symbolic(SimStimulusWorleyNoiseState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusWorleyNoiseConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "worley A=%.3g f=%.3g %s %s",
                    cfg->amplitude,
                    cfg->feature_frequency,
                    worley_metric_name(cfg->distance_metric),
                    worley_output_name(cfg->output_mode));
#else
    (void) state;
#endif
}

static void worley_noise_refresh_hash_basis(SimStimulusWorleyNoiseState* state) {
    if (state == NULL) {
        return;
    }

    for (unsigned channel = 0U; channel < 2U; ++channel) {
        for (unsigned axis = 0U; axis < 2U; ++axis) {
            state->hash_basis[channel][axis] =
                stim_worley_hash_basis(state->config.seed, channel, axis);
        }
    }
}

static const char* worley_noise_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusWorleyNoiseState* state = (const SimStimulusWorleyNoiseState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void worley_noise_destroy(void* state_ptr) {
    SimStimulusWorleyNoiseState* state = (SimStimulusWorleyNoiseState*) state_ptr;
    if (state == NULL) {
        return;
    }
    sim_stimulus_static_cache_destroy(&state->cache);
    free(state);
}

static void worley_noise_map_coord(const SimStimulusCoordConfig* coord,
                                   double                        x,
                                   double                        y,
                                   double                        t,
                                   double*                       out_u,
                                   double*                       out_v) {
    double u = x;
    double v = y;

    if (coord == NULL) {
        if (out_u != NULL) {
            *out_u = u;
        }
        if (out_v != NULL) {
            *out_v = v;
        }
        return;
    }

    switch (coord->mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (coord->axis == SIM_STIMULUS_AXIS_Y) {
                u = y;
                v = x;
            } else {
                u = x;
                v = y;
            }
            break;
        case SIM_STIMULUS_COORD_ANGLE:
            sim_stimulus_coord_rotate_xy(x, y, coord->angle, &u, &v);
            break;
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            u = hypot(dx, dy);
            v = atan2(dy, dx);
            break;
        }
        case SIM_STIMULUS_COORD_AZIMUTH: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            u = atan2(dy, dx);
            v = hypot(dx, dy);
            break;
        }
        case SIM_STIMULUS_COORD_ELLIPTIC: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            sim_stimulus_coord_elliptic_polar(coord, dx, dy, &u, &v);
            break;
        }
        case SIM_STIMULUS_COORD_SPIRAL: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
            {
                double r  = hypot(dx, dy);
                double th = atan2(dy, dx);
                u = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                    coord->spiral_angular_velocity * t;
                v = r;
            }
            break;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            u = x;
            v = y;
            break;
    }

    if (out_u != NULL) {
        *out_u = u;
    }
    if (out_v != NULL) {
        *out_v = v;
    }
}

static bool worley_noise_prepare_linear_map(const SimStimulusWorleyNoiseConfig* cfg,
                                            double                              t,
                                            SimStimulusWorleyLinearMap*         out_map) {
    if (cfg == NULL || out_map == NULL) {
        return false;
    }

    const SimStimulusCoordConfig* coord = &cfg->coord;
    double                        x0    = coord->origin_x - coord->velocity_x * t;
    double                        y0    = coord->origin_y - coord->velocity_y * t;
    double                        dx    = coord->spacing_x;
    double                        dy    = coord->spacing_y;
    double                        freq  = cfg->feature_frequency;

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(freq)) {
        return false;
    }

    switch (coord->mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (coord->axis == SIM_STIMULUS_AXIS_Y) {
                out_map->u0   = y0 * freq;
                out_map->v0   = x0 * freq;
                out_map->du_x = 0.0;
                out_map->dv_x = dx * freq;
                out_map->du_y = dy * freq;
                out_map->dv_y = 0.0;
            } else {
                out_map->u0   = x0 * freq;
                out_map->v0   = y0 * freq;
                out_map->du_x = dx * freq;
                out_map->dv_x = 0.0;
                out_map->du_y = 0.0;
                out_map->dv_y = dy * freq;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE: {
            double c      = cos(coord->angle);
            double s      = sin(coord->angle);
            out_map->u0   = (x0 * c + y0 * s) * freq;
            out_map->v0   = (-x0 * s + y0 * c) * freq;
            out_map->du_x = dx * c * freq;
            out_map->dv_x = -dx * s * freq;
            out_map->du_y = dy * s * freq;
            out_map->dv_y = dy * c * freq;
            return isfinite(out_map->u0) && isfinite(out_map->v0) && isfinite(out_map->du_x) &&
                   isfinite(out_map->dv_x) && isfinite(out_map->du_y) && isfinite(out_map->dv_y);
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
            out_map->u0   = x0 * freq;
            out_map->v0   = y0 * freq;
            out_map->du_x = dx * freq;
            out_map->dv_x = 0.0;
            out_map->du_y = 0.0;
            out_map->dv_y = dy * freq;
            return true;
        default:
            return false;
    }
}

static inline void worley_noise_linear_map_sample(const SimStimulusWorleyLinearMap* map,
                                                  size_t                            ix,
                                                  size_t                            iy,
                                                  double*                           out_u,
                                                  double*                           out_v) {
    double u = map->u0 + (double) ix * map->du_x + (double) iy * map->du_y;
    double v = map->v0 + (double) ix * map->dv_x + (double) iy * map->dv_y;

    if (out_u != NULL) {
        *out_u = u;
    }
    if (out_v != NULL) {
        *out_v = v;
    }
}

static bool worley_noise_dense_layout(const SimStimulusStaticCacheLayout* layout) {
    if (layout == NULL || layout->extent_x == 0U || layout->extent_y == 0U) {
        return false;
    }
    if (layout->stride_x != 1U) {
        return false;
    }
    if (layout->rank <= 1U) {
        return layout->stride_y == 1U;
    }
    return layout->stride_y == layout->extent_x;
}

static double worley_output_value(double f1, double f2, SimStimulusWorleyOutputMode mode) {
    if (!isfinite(f1)) {
        return 0.0;
    }
    if (!isfinite(f2)) {
        f2 = f1;
    }

    switch (mode) {
        case SIM_STIMULUS_WORLEY_F1:
            return f1;
        case SIM_STIMULUS_WORLEY_F2:
            return f2;
        case SIM_STIMULUS_WORLEY_F2_MINUS_F1:
        default:
            return fmax(0.0, f2 - f1);
    }
}

static double worley_component_value_scaled(const SimStimulusWorleyNoiseState* state,
                                            double                             u,
                                            double                             v,
                                            unsigned                           channel) {
    const SimStimulusWorleyNoiseConfig* cfg = &state->config;
    double                              f1  = DBL_MAX;
    double                              f2  = DBL_MAX;
    int64_t                             base_x;
    int64_t                             base_y;
    uint64_t                            basis_x;
    uint64_t                            basis_y;

    if (state == NULL) {
        return 0.0;
    }

    base_x  = (int64_t) floor(u);
    base_y  = (int64_t) floor(v);
    basis_x = state->hash_basis[channel & 1U][0];
    basis_y = state->hash_basis[channel & 1U][1];

    switch (cfg->distance_metric) {
        case SIM_STIMULUS_WORLEY_MANHATTAN:
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x   = base_x + (int64_t) dx;
                    int64_t cell_y   = base_y + (int64_t) dy;
                    double  jitter_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_x, cell_x, cell_y));
                    double jitter_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_y, cell_x, cell_y));
                    double feature_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_x - 0.5);
                    double feature_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_y - 0.5);
                    double distance  = fabs(u - feature_x) + fabs(v - feature_y);
                    stim_worley_update_best(distance, &f1, &f2);
                }
            }
            return worley_output_value(f1, f2, cfg->output_mode);

        case SIM_STIMULUS_WORLEY_CHEBYSHEV:
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x   = base_x + (int64_t) dx;
                    int64_t cell_y   = base_y + (int64_t) dy;
                    double  jitter_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_x, cell_x, cell_y));
                    double jitter_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_y, cell_x, cell_y));
                    double feature_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_x - 0.5);
                    double feature_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_y - 0.5);
                    double distance  = fmax(fabs(u - feature_x), fabs(v - feature_y));
                    stim_worley_update_best(distance, &f1, &f2);
                }
            }
            return worley_output_value(f1, f2, cfg->output_mode);

        case SIM_STIMULUS_WORLEY_MINKOWSKI: {
            double exponent     = cfg->distance_exponent;
            double inv_exponent = 1.0 / exponent;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x   = base_x + (int64_t) dx;
                    int64_t cell_y   = base_y + (int64_t) dy;
                    double  jitter_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_x, cell_x, cell_y));
                    double jitter_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_y, cell_x, cell_y));
                    double feature_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_x - 0.5);
                    double feature_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_y - 0.5);
                    double distance =
                        pow(pow(fabs(u - feature_x), exponent) + pow(fabs(v - feature_y), exponent),
                            inv_exponent);
                    stim_worley_update_best(distance, &f1, &f2);
                }
            }
            return worley_output_value(f1, f2, cfg->output_mode);
        }

        case SIM_STIMULUS_WORLEY_EUCLIDEAN:
        default: {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x   = base_x + (int64_t) dx;
                    int64_t cell_y   = base_y + (int64_t) dy;
                    double  jitter_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_x, cell_x, cell_y));
                    double jitter_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_y, cell_x, cell_y));
                    double feature_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_x - 0.5);
                    double feature_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_y - 0.5);
                    double dx_u      = u - feature_x;
                    double dy_v      = v - feature_y;
                    double distance2 = dx_u * dx_u + dy_v * dy_v;
                    stim_worley_update_best(distance2, &f1, &f2);
                }
            }
            return worley_output_value(sqrt(f1), sqrt(f2), cfg->output_mode);
        }
    }
}

static void worley_complex_value_scaled(const SimStimulusWorleyNoiseState* state,
                                        double                             u,
                                        double                             v,
                                        double*                            out_re,
                                        double*                            out_im) {
    const SimStimulusWorleyNoiseConfig* cfg   = &state->config;
    double                              f1_re = DBL_MAX;
    double                              f2_re = DBL_MAX;
    double                              f1_im = DBL_MAX;
    double                              f2_im = DBL_MAX;
    int64_t                             base_x;
    int64_t                             base_y;
    uint64_t                            basis_re_x;
    uint64_t                            basis_re_y;
    uint64_t                            basis_im_x;
    uint64_t                            basis_im_y;

    if (out_re != NULL) {
        *out_re = 0.0;
    }
    if (out_im != NULL) {
        *out_im = 0.0;
    }
    if (state == NULL) {
        return;
    }

    base_x     = (int64_t) floor(u);
    base_y     = (int64_t) floor(v);
    basis_re_x = state->hash_basis[0][0];
    basis_re_y = state->hash_basis[0][1];
    basis_im_x = state->hash_basis[1][0];
    basis_im_y = state->hash_basis[1][1];

    switch (cfg->distance_metric) {
        case SIM_STIMULUS_WORLEY_MANHATTAN:
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x = base_x + (int64_t) dx;
                    int64_t cell_y = base_y + (int64_t) dy;

                    double jitter_re_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_x, cell_x, cell_y));
                    double jitter_re_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_y, cell_x, cell_y));
                    double feature_re_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_re_x - 0.5);
                    double feature_re_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_re_y - 0.5);
                    double distance_re  = fabs(u - feature_re_x) + fabs(v - feature_re_y);
                    stim_worley_update_best(distance_re, &f1_re, &f2_re);

                    double jitter_im_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_x, cell_x, cell_y));
                    double jitter_im_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_y, cell_x, cell_y));
                    double feature_im_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_im_x - 0.5);
                    double feature_im_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_im_y - 0.5);
                    double distance_im  = fabs(u - feature_im_x) + fabs(v - feature_im_y);
                    stim_worley_update_best(distance_im, &f1_im, &f2_im);
                }
            }
            break;

        case SIM_STIMULUS_WORLEY_CHEBYSHEV:
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x = base_x + (int64_t) dx;
                    int64_t cell_y = base_y + (int64_t) dy;

                    double jitter_re_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_x, cell_x, cell_y));
                    double jitter_re_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_y, cell_x, cell_y));
                    double feature_re_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_re_x - 0.5);
                    double feature_re_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_re_y - 0.5);
                    double distance_re  = fmax(fabs(u - feature_re_x), fabs(v - feature_re_y));
                    stim_worley_update_best(distance_re, &f1_re, &f2_re);

                    double jitter_im_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_x, cell_x, cell_y));
                    double jitter_im_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_y, cell_x, cell_y));
                    double feature_im_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_im_x - 0.5);
                    double feature_im_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_im_y - 0.5);
                    double distance_im  = fmax(fabs(u - feature_im_x), fabs(v - feature_im_y));
                    stim_worley_update_best(distance_im, &f1_im, &f2_im);
                }
            }
            break;

        case SIM_STIMULUS_WORLEY_MINKOWSKI: {
            double exponent     = cfg->distance_exponent;
            double inv_exponent = 1.0 / exponent;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x = base_x + (int64_t) dx;
                    int64_t cell_y = base_y + (int64_t) dy;

                    double jitter_re_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_x, cell_x, cell_y));
                    double jitter_re_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_y, cell_x, cell_y));
                    double feature_re_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_re_x - 0.5);
                    double feature_re_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_re_y - 0.5);
                    double distance_re  = pow(pow(fabs(u - feature_re_x), exponent) +
                                                  pow(fabs(v - feature_re_y), exponent),
                                              inv_exponent);
                    stim_worley_update_best(distance_re, &f1_re, &f2_re);

                    double jitter_im_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_x, cell_x, cell_y));
                    double jitter_im_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_y, cell_x, cell_y));
                    double feature_im_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_im_x - 0.5);
                    double feature_im_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_im_y - 0.5);
                    double distance_im  = pow(pow(fabs(u - feature_im_x), exponent) +
                                                  pow(fabs(v - feature_im_y), exponent),
                                              inv_exponent);
                    stim_worley_update_best(distance_im, &f1_im, &f2_im);
                }
            }
            break;
        }

        case SIM_STIMULUS_WORLEY_EUCLIDEAN:
        default:
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int64_t cell_x = base_x + (int64_t) dx;
                    int64_t cell_y = base_y + (int64_t) dy;

                    double jitter_re_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_x, cell_x, cell_y));
                    double jitter_re_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_re_y, cell_x, cell_y));
                    double feature_re_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_re_x - 0.5);
                    double feature_re_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_re_y - 0.5);
                    double dx_re        = u - feature_re_x;
                    double dy_re        = v - feature_re_y;
                    double distance_re2 = dx_re * dx_re + dy_re * dy_re;
                    stim_worley_update_best(distance_re2, &f1_re, &f2_re);

                    double jitter_im_x = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_x, cell_x, cell_y));
                    double jitter_im_y = stim_worley_uniform01(
                        stim_worley_hash_feature_from_basis(basis_im_y, cell_x, cell_y));
                    double feature_im_x = (double) cell_x + 0.5 + cfg->jitter * (jitter_im_x - 0.5);
                    double feature_im_y = (double) cell_y + 0.5 + cfg->jitter * (jitter_im_y - 0.5);
                    double dx_im        = u - feature_im_x;
                    double dy_im        = v - feature_im_y;
                    double distance_im2 = dx_im * dx_im + dy_im * dy_im;
                    stim_worley_update_best(distance_im2, &f1_im, &f2_im);
                }
            }
            f1_re = sqrt(f1_re);
            f2_re = sqrt(f2_re);
            f1_im = sqrt(f1_im);
            f2_im = sqrt(f2_im);
            break;
    }

    if (out_re != NULL) {
        *out_re = worley_output_value(f1_re, f2_re, cfg->output_mode);
    }
    if (out_im != NULL) {
        *out_im = worley_output_value(f1_im, f2_im, cfg->output_mode);
    }
}

static double worley_component_value(const SimStimulusWorleyNoiseState* state,
                                     double                             x,
                                     double                             y,
                                     unsigned                           channel) {
    const SimStimulusWorleyNoiseConfig* cfg = &state->config;
    double                              u   = 0.0;
    double                              v   = 0.0;

    worley_noise_map_coord(&cfg->coord, x, y, 0.0, &u, &v);
    u *= cfg->feature_frequency;
    v *= cfg->feature_frequency;
    return worley_component_value_scaled(state, u, v, channel);
}

static void worley_complex_value(const SimStimulusWorleyNoiseState* state,
                                 double                             x,
                                 double                             y,
                                 double*                            out_re,
                                 double*                            out_im) {
    const SimStimulusWorleyNoiseConfig* cfg = &state->config;
    double                              u   = 0.0;
    double                              v   = 0.0;

    worley_noise_map_coord(&cfg->coord, x, y, 0.0, &u, &v);
    u *= cfg->feature_frequency;
    v *= cfg->feature_frequency;
    worley_complex_value_scaled(state, u, v, out_re, out_im);
}

static bool worley_noise_can_use_static_cache(const SimStimulusWorleyNoiseState* state) {
    return state != NULL && sim_stimulus_coord_is_time_invariant(&state->config.coord);
}

static void
worley_noise_accumulate_cache_real(double* dst, const double* src, size_t count, double scale) {
    if (dst == NULL || src == NULL || count == 0U) {
        return;
    }

    if (sim_accel_scan_real_finite_maxabs(src, count, NULL)) {
        sim_accel_copy_scale_real(src, dst, count, scale, true);
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double value = src[i];
        if (isfinite(value)) {
            dst[i] += scale * value;
        }
    }
}

static void worley_noise_accumulate_cache_complex(SimComplexDouble* dst,
                                                  const double*     src_re,
                                                  const double*     src_im,
                                                  size_t            count,
                                                  double            scale) {
    if (dst == NULL || src_re == NULL || src_im == NULL || count == 0U) {
        return;
    }

    if (sim_accel_scan_real_finite_maxabs(src_re, count, NULL) &&
        sim_accel_scan_real_finite_maxabs(src_im, count, NULL)) {
        sim_accel_accumulate_real_to_complex(src_re, dst, count, scale, 0.0);
        sim_accel_accumulate_real_to_complex(src_im, dst, count, 0.0, scale);
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double value_re = src_re[i];
        double value_im = src_im[i];
        if (isfinite(value_re)) {
            dst[i].re += scale * value_re;
        }
        if (isfinite(value_im)) {
            dst[i].im += scale * value_im;
        }
    }
}

static bool worley_noise_fill_linear_cache(const SimStimulusWorleyNoiseState*  state,
                                           const SimStimulusStaticCacheLayout* layout,
                                           bool                                need_imag,
                                           double*                             out_real,
                                           double*                             out_imag) {
    SimStimulusWorleyLinearMap map;

    if (state == NULL || layout == NULL || out_real == NULL) {
        return false;
    }
    if (!worley_noise_dense_layout(layout) ||
        !worley_noise_prepare_linear_map(&state->config, 0.0, &map)) {
        return false;
    }

    double scale  = state->config.amplitude;
    size_t width  = layout->extent_x;
    size_t height = layout->extent_y;

    for (size_t row = 0U; row < height; ++row) {
        size_t offset = row * width;
        double u      = map.u0 + (double) row * map.du_y;
        double v      = map.v0 + (double) row * map.dv_y;

        if (!need_imag || out_imag == NULL) {
            for (size_t col = 0U; col < width; ++col) {
                double sample          = worley_component_value_scaled(state, u, v, 0U);
                double value           = scale * sample;
                out_real[offset + col] = isfinite(value) ? value : 0.0;
                u += map.du_x;
                v += map.dv_x;
            }
            continue;
        }

        for (size_t col = 0U; col < width; ++col) {
            double sample_re = 0.0;
            double sample_im = 0.0;
            worley_complex_value_scaled(state, u, v, &sample_re, &sample_im);
            sample_re *= scale;
            sample_im *= scale;
            out_real[offset + col] = isfinite(sample_re) ? sample_re : 0.0;
            out_imag[offset + col] = isfinite(sample_im) ? sample_im : 0.0;
            u += map.du_x;
            v += map.dv_x;
        }
    }

    return true;
}

static SimResult worley_noise_fill_static_cache(void*                               userdata,
                                                const SimStimulusStaticCacheLayout* layout,
                                                bool                                need_imag,
                                                double*                             out_real,
                                                double*                             out_imag) {
    SimStimulusWorleyNoiseState* state = (SimStimulusWorleyNoiseState*) userdata;
    if (state == NULL || layout == NULL || out_real == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (worley_noise_fill_linear_cache(state, layout, need_imag, out_real, out_imag)) {
        return SIM_RESULT_OK;
    }

    const SimStimulusWorleyNoiseConfig* cfg = &state->config;
    for (size_t idx = 0U; idx < layout->count; ++idx) {
        size_t ix = 0U;
        size_t iy = 0U;
        sim_stimulus_static_cache_index_to_xy(layout, idx, &ix, &iy);

        double x         = cfg->coord.origin_x + (double) ix * cfg->coord.spacing_x;
        double y         = cfg->coord.origin_y + (double) iy * cfg->coord.spacing_y;
        double sample_re = worley_component_value(state, x, y, 0U);
        double value_re  = cfg->amplitude * sample_re;
        out_real[idx]    = isfinite(value_re) ? value_re : 0.0;

        if (need_imag && out_imag != NULL) {
            double sample_im = worley_component_value(state, x, y, 1U);
            double value_im  = cfg->amplitude * sample_im;
            out_imag[idx]    = isfinite(value_im) ? value_im : 0.0;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult worley_noise_ensure_static_cache(SimStimulusWorleyNoiseState*        state,
                                                  const SimStimulusStaticCacheLayout* layout,
                                                  bool                                need_imag) {
    if (state == NULL || layout == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    return sim_stimulus_static_cache_ensure(
        &state->cache, layout, need_imag, worley_noise_fill_static_cache, state);
}

static bool worley_noise_try_linear_rows(const SimStimulusWorleyNoiseState* state,
                                         const SimField*                    field,
                                         double                             scale,
                                         double                             t,
                                         double*                            dst_real,
                                         SimComplexDouble*                  dst_complex,
                                         size_t                             count) {
    SimStimulusWorleyLinearMap map;
    double                     row_scale;
    size_t                     width;
    size_t                     height;

    if (state == NULL || field == NULL || count == 0U) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }
    if (!worley_noise_prepare_linear_map(&state->config, t, &map)) {
        return false;
    }

    row_scale = scale * state->config.amplitude;
    if (!isfinite(row_scale)) {
        return false;
    }

    width  = sim_field_width(field);
    height = sim_field_height(field);
    if (width == 0U || height == 0U || width > count || width * height != count) {
        return false;
    }

    for (size_t row = 0U; row < height; ++row) {
        size_t offset = row * width;
        double u      = map.u0 + (double) row * map.du_y;
        double v      = map.v0 + (double) row * map.dv_y;

        if (dst_real != NULL) {
            for (size_t col = 0U; col < width; ++col) {
                double sample = worley_component_value_scaled(state, u, v, 0U);
                if (isfinite(sample)) {
                    dst_real[offset + col] += row_scale * sample;
                }
                u += map.du_x;
                v += map.dv_x;
            }
            continue;
        }

        for (size_t col = 0U; col < width; ++col) {
            double sample_re = 0.0;
            double sample_im = 0.0;
            worley_complex_value_scaled(state, u, v, &sample_re, &sample_im);
            if (isfinite(sample_re) && isfinite(sample_im)) {
                dst_complex[offset + col].re += row_scale * sample_re;
                dst_complex[offset + col].im += row_scale * sample_im;
            }
            u += map.du_x;
            v += map.dv_x;
        }
    }

    return true;
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

static SimResult worley_noise_kernel_value(SimStimulusWorleyNoiseState* state,
                                           const KernelIR*              kernel,
                                           size_t                       element_index,
                                           unsigned                     channel,
                                           double*                      out_value) {
    if (state == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->config.amplitude == 0.0) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    if (kernel == NULL || kernel->bindings == NULL || kernel->binding_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimKernelIRBinding* binding = &kernel->bindings[0];
    const size_t*             shape   = binding->shape;
    const size_t*             strides = binding->strides;
    size_t                    rank    = binding->rank;
    if ((shape == NULL || strides == NULL || rank == 0U) && binding->field != NULL) {
        shape   = binding->field->layout.shape;
        strides = binding->field->layout.strides;
        rank    = binding->field->layout.rank;
    }
    if (shape == NULL || strides == NULL || rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    assert(rank == 1U || rank == 2U);

    SimStimulusStaticCacheLayout layout = { 0 };
    SimResult                    layout_rc =
        sim_stimulus_static_cache_layout_from_arrays(shape, strides, rank, &layout);
    if (layout_rc != SIM_RESULT_OK) {
        return layout_rc;
    }
    assert(element_index < layout.count);

    {
        size_t                     ix    = 0U;
        size_t                     iy    = 0U;
        double                     dt    = kernel_param_value(kernel, SIM_IR_PARAM_DT);
        double                     t     = kernel_param_value(kernel, SIM_IR_PARAM_TIME);
        double                     scale = state->config.scale_by_dt ? dt : 1.0;
        SimStimulusWorleyLinearMap linear_map;
        bool has_linear_map = worley_noise_prepare_linear_map(&state->config, t, &linear_map);

        if (worley_noise_can_use_static_cache(state)) {
            SimResult prep = worley_noise_ensure_static_cache(state, &layout, channel != 0U);
            if (prep != SIM_RESULT_OK) {
                return prep;
            }
            double value = (channel == 0U) ? state->cache.real[element_index]
                                           : state->cache.imag[element_index];
            value *= scale;
            *out_value = isfinite(value) ? value : 0.0;
            return SIM_RESULT_OK;
        }

        SimResult coord_rc = sim_kernel_binding_index_to_xy(binding, element_index, &ix, &iy);
        double    u        = 0.0;
        double    v        = 0.0;
        double    sample;

        assert(coord_rc == SIM_RESULT_OK);
        if (has_linear_map) {
            worley_noise_linear_map_sample(&linear_map, ix, iy, &u, &v);
            sample = worley_component_value_scaled(state, u, v, channel);
        } else {
            double x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
            double y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;
            double sample_x = x - state->config.coord.velocity_x * t;
            double sample_y = y - state->config.coord.velocity_y * t;
            sample          = worley_component_value(state, sample_x, sample_y, channel);
        }
        *out_value = isfinite(sample) ? scale * state->config.amplitude * sample : 0.0;
    }
    return SIM_RESULT_OK;
}

static SimResult worley_noise_ir_eval(void*           userdata,
                                      const KernelIR* kernel,
                                      size_t          element_index,
                                      size_t          component,
                                      double*         out_value) {
    (void) component;

    if (userdata == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    return worley_noise_kernel_value(
        (SimStimulusWorleyNoiseState*) userdata, kernel, element_index, 0U, out_value);
}

static SimResult worley_noise_ir_eval_imag(void*           userdata,
                                           const KernelIR* kernel,
                                           size_t          element_index,
                                           size_t          component,
                                           double*         out_value) {
    (void) component;

    if (userdata == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    return worley_noise_kernel_value(
        (SimStimulusWorleyNoiseState*) userdata, kernel, element_index, 1U, out_value);
}

static SimResult worley_noise_step(void*               state_ptr,
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

    SimStimulusWorleyNoiseState* state = (SimStimulusWorleyNoiseState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->config.amplitude == 0.0) {
        return SIM_RESULT_OK;
    }

    {
        SimField* field = sim_context_field(context, state->config.field_index);
        bool      is_complex;
        size_t    count;
        double    scale;

        if (field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        is_complex = sim_field_is_complex(field);
        if (is_complex) {
            count = sim_field_bytes(field) / sizeof(SimComplexDouble);
        } else {
            if (field->element_size != sizeof(double)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            count = sim_field_bytes(field) / sizeof(double);
        }
        if (count == 0U) {
            return SIM_RESULT_OK;
        }

        SimStimulusStaticCacheLayout layout    = { 0 };
        SimResult                    layout_rc = sim_stimulus_static_cache_layout_from_arrays(
            field->layout.shape, field->layout.strides, field->layout.rank, &layout);
        if (layout_rc != SIM_RESULT_OK) {
            return layout_rc;
        }

        size_t width  = layout.extent_x;
        size_t height = layout.extent_y;
        scale         = state->config.scale_by_dt ? dt_sub : 1.0;
        double t      = sim_context_time(context);

        assert(field->layout.rank == 1U || field->layout.rank == 2U);
        assert(width > 0U);
        assert(height > 0U);
        assert(layout.count == count);

        if (!is_complex) {
            double*                    dst_real = (double*) sim_field_data(field);
            SimStimulusWorleyLinearMap linear_map;
            bool has_linear_map = worley_noise_prepare_linear_map(&state->config, t, &linear_map);
            if (dst_real == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            if (worley_noise_can_use_static_cache(state)) {
                SimResult prep = worley_noise_ensure_static_cache(state, &layout, false);
                if (prep != SIM_RESULT_OK) {
                    return prep;
                }
                worley_noise_accumulate_cache_real(dst_real, state->cache.real, count, scale);
                return SIM_RESULT_OK;
            }

            if (worley_noise_try_linear_rows(state, field, scale, t, dst_real, NULL, count)) {
                return SIM_RESULT_OK;
            }

            for (size_t idx = 0U; idx < count; ++idx) {
                size_t    ix = 0U;
                size_t    iy = 0U;
                double    x;
                double    y;
                double    u;
                double    v;
                double    sample;
                SimResult coord_rc = sim_field_index_to_xy(field, idx, &ix, &iy);
                assert(coord_rc == SIM_RESULT_OK);
                assert(ix < width);
                assert(iy < height);

                if (has_linear_map) {
                    worley_noise_linear_map_sample(&linear_map, ix, iy, &u, &v);
                    sample = worley_component_value_scaled(state, u, v, 0U);
                } else {
                    x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
                    y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;
                    sample = worley_component_value(state,
                                                    x - state->config.coord.velocity_x * t,
                                                    y - state->config.coord.velocity_y * t,
                                                    0U);
                }
                if (isfinite(sample)) {
                    dst_real[idx] += scale * state->config.amplitude * sample;
                }
            }
        } else {
            SimComplexDouble*          dst_complex = sim_field_complex_data(field);
            SimStimulusWorleyLinearMap linear_map;
            bool has_linear_map = worley_noise_prepare_linear_map(&state->config, t, &linear_map);
            if (dst_complex == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            if (worley_noise_can_use_static_cache(state)) {
                SimResult prep = worley_noise_ensure_static_cache(state, &layout, true);
                if (prep != SIM_RESULT_OK) {
                    return prep;
                }
                worley_noise_accumulate_cache_complex(
                    dst_complex, state->cache.real, state->cache.imag, count, scale);
                return SIM_RESULT_OK;
            }

            if (worley_noise_try_linear_rows(state, field, scale, t, NULL, dst_complex, count)) {
                return SIM_RESULT_OK;
            }

            for (size_t idx = 0U; idx < count; ++idx) {
                size_t    ix = 0U;
                size_t    iy = 0U;
                double    x;
                double    y;
                double    u;
                double    v;
                double    sample_re;
                double    sample_im;
                SimResult coord_rc = sim_field_index_to_xy(field, idx, &ix, &iy);
                assert(coord_rc == SIM_RESULT_OK);
                assert(ix < width);
                assert(iy < height);

                if (has_linear_map) {
                    worley_noise_linear_map_sample(&linear_map, ix, iy, &u, &v);
                    worley_complex_value_scaled(state, u, v, &sample_re, &sample_im);
                } else {
                    x = state->config.coord.origin_x + (double) ix * state->config.coord.spacing_x;
                    y = state->config.coord.origin_y + (double) iy * state->config.coord.spacing_y;
                    worley_complex_value(state,
                                         x - state->config.coord.velocity_x * t,
                                         y - state->config.coord.velocity_y * t,
                                         &sample_re,
                                         &sample_im);
                }
                if (isfinite(sample_re) && isfinite(sample_im)) {
                    dst_complex[idx].re += scale * state->config.amplitude * sample_re;
                    dst_complex[idx].im += scale * state->config.amplitude * sample_im;
                }
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_worley_noise_operator(struct SimContext*                  context,
                                                 const SimStimulusWorleyNoiseConfig* config,
                                                 size_t*                             out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    {
        SimStimulusWorleyNoiseConfig local = { 0 };
        SimStimulusWorleyNoiseState* state;
        SimOperatorInfo              info;
        SimSplitPort                 port;
        SimSplitAccess               access;
        SimSplitSubstep              substep;
        SimOperatorConfig            op_config;
        SimResult                    result            = SIM_RESULT_OK;
        bool                         registered_kernel = false;
        char                         name[SIM_OPERATOR_NAME_MAX + 1U];

        if (config != NULL) {
            local = *config;
        }
        if (local.seed == 0ULL) {
            local.seed = sim_seed_derive(sim_context_seed(context),
                                         sim_seed_tag("stimulus_worley_noise"),
                                         sim_context_operator_count(context));
        }

        worley_noise_normalize(&local);
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context,
                                             "stimulus_worley_noise",
                                             (config != NULL),
                                             (config != NULL) ? config->scale_by_dt : true);

        state = (SimStimulusWorleyNoiseState*) calloc(1U, sizeof(*state));
        if (state == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->config = local;
        worley_noise_refresh_hash_basis(state);
        worley_noise_refresh_symbolic(state);

        sim_operator_make_unique_name(name, sizeof(name), "stimulus_worley_noise");

        info                   = sim_operator_info_defaults();
        info.category          = SIM_OPERATOR_CATEGORY_NOISE;
        info.warp_level        = SIM_WARP_LEVEL_NONE;
        info.is_noise          = true;
        info.is_spectral       = false;
        info.is_local          = true;
        info.is_nonlocal       = false;
        info.is_linear         = true;
        info.is_warp           = false;
        info.is_differentiable = false;
        info.preserves_real    = true;
        info.preferred_dt      = 0.0;
        info.abstract_id       = "stimulus_worley_noise";
        sim_operator_info_set_schema_identity(&info, "stimulus_worley_noise");
        info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR | SIM_OPERATOR_ALG_COMMUTES_WITH_NOISE;
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

        port.context_field_index = state->config.field_index;
        port.require_complex     = needs_complex;

        access.port = 0;
        access.mode = SIM_ACCESS_RW;

        substep.name              = NULL;
        substep.fn                = worley_noise_step;
        substep.accesses          = &access;
        substep.access_count      = 1U;
        substep.dt_scale          = 1.0;
        substep.barrier_after     = false;
        substep.error_measure     = NULL;
        substep.required_features = 0U;

        op_config = sim_operator_config_defaults();

        if (sim_operator_should_register_kernel_for_schema(
                context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_worley_noise")) {
            SimField* field = sim_context_field(context, local.field_index);
            if (field != NULL) {
                bool is_complex = sim_field_is_complex(field);
                if ((!is_complex && field->element_size != sizeof(double)) ||
                    (is_complex && field->element_size != sizeof(SimComplexDouble))) {
                    field = NULL;
                }
            }

            if (field != NULL) {
                SimIRBuilder* builder = sim_context_ir_builder(context);
                if (builder != NULL) {
                    SimOperatorKernelBindingDescriptor bindings[1];
                    SimOperatorKernelOutputDescriptor  outputs[1];
                    SimOperatorKernelDescriptor        kernel_desc = { 0 };
                    SimOperatorDescriptor              kdesc       = { 0 };
                    bool                               is_complex  = sim_field_is_complex(field);
                    SimIRType                          field_type =
                        is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                    SimIRNodeId field_node =
                        sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                    SimIRNodeId sample_node = sim_ir_builder_stateful(
                        builder, worley_noise_ir_eval, state, "stimulus_worley_noise");

                    if (is_complex && sample_node != SIM_IR_INVALID_NODE) {
                        SimIRNodeId sample_im = sim_ir_builder_stateful(
                            builder, worley_noise_ir_eval_imag, state, "stimulus_worley_noise_im");
                        if (sample_im != SIM_IR_INVALID_NODE) {
                            sample_node =
                                sim_ir_builder_complex_pack(builder, sample_node, sample_im);
                        }
                    }

                    if (field_node != SIM_IR_INVALID_NODE && sample_node != SIM_IR_INVALID_NODE) {
                        SimIRNodeId sum = sim_ir_builder_binary(
                            builder, SIM_IR_NODE_ADD, field_node, sample_node);
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
                            kdesc.destroy  = worley_noise_destroy;
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
            SimSplitDescriptor desc = { .name          = name,
                                        .ports         = &port,
                                        .port_count    = 1U,
                                        .substeps      = &substep,
                                        .substep_count = 1U,
                                        .state         = state,
                                        .symbolic      = worley_noise_symbolic,
                                        .destroy       = worley_noise_destroy,
                                        .info          = info,
                                        .config        = op_config,
                                        .scratch       = { 0U, 0U } };

            result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
            if (result != SIM_RESULT_OK) {
                worley_noise_destroy(state);
            }
            return result;
        }
    }
}

SimResult sim_stimulus_worley_noise_config(struct SimContext*            context,
                                           size_t                        operator_index,
                                           SimStimulusWorleyNoiseConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    {
        SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
        SimStimulusWorleyNoiseState* state = NULL;

        if (op == NULL) {
            return SIM_RESULT_NOT_FOUND;
        }
        if (op->kernel != NULL) {
            state = (SimStimulusWorleyNoiseState*) sim_operator_payload(op);
        } else {
            state = (SimStimulusWorleyNoiseState*) sim_split_state(op);
        }
        if (state == NULL) {
            return SIM_RESULT_INVALID_STATE;
        }

        *out_config = state->config;
    }
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_worley_noise_update(struct SimContext*                  context,
                                           size_t                              operator_index,
                                           const SimStimulusWorleyNoiseConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    {
        SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
        SimStimulusWorleyNoiseState* state = NULL;
        SimStimulusWorleyNoiseConfig local;

        if (op == NULL) {
            return SIM_RESULT_NOT_FOUND;
        }
        if (op->kernel != NULL) {
            state = (SimStimulusWorleyNoiseState*) sim_operator_payload(op);
        } else {
            state = (SimStimulusWorleyNoiseState*) sim_split_state(op);
        }
        if (state == NULL) {
            return SIM_RESULT_INVALID_STATE;
        }

        local = state->config;
        if (config != NULL) {
            local = *config;
        }

        if (local.seed == 0ULL) {
            local.seed = sim_seed_derive(
                sim_context_seed(context), sim_seed_tag("stimulus_worley_noise"), operator_index);
        }

        worley_noise_normalize(&local);
        local.scale_by_dt = sim_operator_resolve_scale_by_dt(
            context,
            sim_operator_schema_key_or(op, "stimulus_worley_noise"),
            true,
            local.scale_by_dt);
        state->config = local;
        worley_noise_refresh_hash_basis(state);
        sim_stimulus_static_cache_invalidate(&state->cache);
        worley_noise_refresh_symbolic(state);
    }

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
