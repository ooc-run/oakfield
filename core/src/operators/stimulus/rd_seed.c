#include "oakfield/operators/stimulus/rd_seed.h"
#include "operators/common/operator_utils.h"

#include "sim_accel.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_seed.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_RD_SEED_EPS 1.0e-12
#define STIM_RD_SEED_MAX_COUNT 512U
#define STIM_RD_SEED_VDSP_MIN_LEN 64U

typedef struct {
    uint64_t state;
    uint64_t inc;
} stim_rd_seed_pcg32_t;

static uint32_t stim_rd_seed_pcg32_random(stim_rd_seed_pcg32_t* rng) {
    uint64_t old        = rng->state;
    rng->state          = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t) (old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void
stim_rd_seed_pcg32_seed(stim_rd_seed_pcg32_t* rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc   = (initseq << 1u) | 1u;
    (void) stim_rd_seed_pcg32_random(rng);
    rng->state += initstate;
    (void) stim_rd_seed_pcg32_random(rng);
}

static double stim_rd_seed_uniform(stim_rd_seed_pcg32_t* rng) {
    return ldexp(stim_rd_seed_pcg32_random(rng), -32); /* [0,1) */
}

typedef struct SimStimulusRDSeedState {
    SimStimulusRDSeedConfig config;
    double*                 seed_u;
    double*                 seed_v;
    double*                 seed_phase;
    unsigned int            seed_capacity;
    char                    symbolic[224];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_accum;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusRDSeedState;

typedef struct SimStimulusRDSeedBounds {
    double u_min;
    double u_max;
    double v_min;
    double v_max;
    double u_span;
    double v_span;
    double avg_span;
} SimStimulusRDSeedBounds;

static bool rd_seed_mode_valid(SimStimulusRDSeedMode mode) {
    return mode == SIM_STIMULUS_RD_SEED_SPOTS || mode == SIM_STIMULUS_RD_SEED_STRIPES ||
           mode == SIM_STIMULUS_RD_SEED_LABYRINTH || mode == SIM_STIMULUS_RD_SEED_RINGS;
}

static const char* rd_seed_mode_name(SimStimulusRDSeedMode mode) {
    switch (mode) {
        case SIM_STIMULUS_RD_SEED_SPOTS:
            return "spots";
        case SIM_STIMULUS_RD_SEED_STRIPES:
            return "stripes";
        case SIM_STIMULUS_RD_SEED_LABYRINTH:
            return "labyrinth";
        case SIM_STIMULUS_RD_SEED_RINGS:
            return "rings";
        default:
            return "spots";
    }
}

static void rd_seed_normalize(SimStimulusRDSeedConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }
    if (!isfinite(config->scale)) {
        config->scale = 6.0;
    }
    config->scale = fabs(config->scale);
    if (config->scale <= STIM_RD_SEED_EPS) {
        config->scale = 6.0;
    }
    if (!isfinite(config->threshold)) {
        config->threshold = 0.5;
    }
    if (config->threshold < 0.0) {
        config->threshold = 0.0;
    } else if (config->threshold > 1.0) {
        config->threshold = 1.0;
    }
    if (!isfinite(config->sharpness)) {
        config->sharpness = 12.0;
    }
    config->sharpness = fabs(config->sharpness);
    if (config->sharpness <= STIM_RD_SEED_EPS) {
        config->sharpness = 1.0;
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

    if (config->seed_count == 0U) {
        config->seed_count = 24U;
    }
    if (config->seed_count > STIM_RD_SEED_MAX_COUNT) {
        config->seed_count = STIM_RD_SEED_MAX_COUNT;
    }

    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }

    if (!rd_seed_mode_valid(config->mode)) {
        config->mode = SIM_STIMULUS_RD_SEED_SPOTS;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void rd_seed_refresh_symbolic(SimStimulusRDSeedState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusRDSeedConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "rd_seed mode=%s A=%.3g count=%u scale=%.3g",
                    rd_seed_mode_name(cfg->mode),
                    cfg->amplitude,
                    cfg->seed_count,
                    cfg->scale);
#else
    (void) state;
#endif
}

static const char* rd_seed_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusRDSeedState* state = (const SimStimulusRDSeedState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult rd_seed_build_tables(const SimStimulusRDSeedConfig* cfg,
                                      double**                       out_u,
                                      double**                       out_v,
                                      double**                       out_phase) {
    if (cfg == NULL || out_u == NULL || out_v == NULL || out_phase == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_u     = NULL;
    *out_v     = NULL;
    *out_phase = NULL;

    if (cfg->seed_count == 0U) {
        return SIM_RESULT_OK;
    }

    double* seed_u     = (double*) malloc((size_t) cfg->seed_count * sizeof(double));
    double* seed_v     = (double*) malloc((size_t) cfg->seed_count * sizeof(double));
    double* seed_phase = (double*) malloc((size_t) cfg->seed_count * sizeof(double));
    if (seed_u == NULL || seed_v == NULL || seed_phase == NULL) {
        free(seed_u);
        free(seed_v);
        free(seed_phase);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    stim_rd_seed_pcg32_t rng;
    stim_rd_seed_pcg32_seed(&rng, cfg->seed, cfg->seed ^ 0x9E3779B97F4A7C15ULL);
    for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
        seed_u[i]     = stim_rd_seed_uniform(&rng);
        seed_v[i]     = stim_rd_seed_uniform(&rng);
        seed_phase[i] = 2.0 * M_PI * stim_rd_seed_uniform(&rng);
    }

    *out_u     = seed_u;
    *out_v     = seed_v;
    *out_phase = seed_phase;
    return SIM_RESULT_OK;
}

static SimResult rd_seed_rebuild_tables(SimStimulusRDSeedState* state) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double*   seed_u     = NULL;
    double*   seed_v     = NULL;
    double*   seed_phase = NULL;
    SimResult result     = rd_seed_build_tables(&state->config, &seed_u, &seed_v, &seed_phase);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    free(state->seed_u);
    free(state->seed_v);
    free(state->seed_phase);
    state->seed_u        = seed_u;
    state->seed_v        = seed_v;
    state->seed_phase    = seed_phase;
    state->seed_capacity = state->config.seed_count;
    return SIM_RESULT_OK;
}

static void rd_seed_destroy(void* state_ptr) {
    SimStimulusRDSeedState* state = (SimStimulusRDSeedState*) state_ptr;
    if (state == NULL) {
        return;
    }

    free(state->seed_u);
    free(state->seed_v);
    free(state->seed_phase);
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
#endif
    free(state);
}

static void rd_seed_map_coord(const SimStimulusCoordConfig* coord,
                              double                        x,
                              double                        y,
                              double                        t,
                              double*                       out_u,
                              double*                       out_v) {
    double u = x;
    double v = y;
    sim_stimulus_coord_sample_xy(coord, x, y, t, &u, &v);
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
                double tmp = u;
                u          = v;
                v          = tmp;
            }
            break;
        case SIM_STIMULUS_COORD_ANGLE: {
            double c        = cos(coord->angle);
            double s        = sin(coord->angle);
            double sample_x = u;
            double sample_y = v;
            u               = sample_x * c + sample_y * s;
            v               = -sample_x * s + sample_y * c;
            break;
        }
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
            double r  = hypot(dx, dy);
            double th = atan2(dy, dx);
            u         = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                        coord->spiral_angular_velocity * t;
            v         = r;
            break;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            break;
    }

    if (out_u != NULL) {
        *out_u = u;
    }
    if (out_v != NULL) {
        *out_v = v;
    }
}

static double rd_seed_clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double rd_seed_sigmoid(double value) {
    if (value >= 40.0) {
        return 1.0;
    }
    if (value <= -40.0) {
        return 0.0;
    }
    return 1.0 / (1.0 + exp(-value));
}

#if defined(SIM_HAVE_VDSP)
static bool rd_seed_vdsp_ensure_buffers(SimStimulusRDSeedState* state, size_t width) {
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
    state->vdsp_accum    = block + width * 2U;
    state->vdsp_value    = block + width * 3U;
    state->vdsp_work     = block + width * 4U;
    return true;
}

static bool rd_seed_linear_map(const SimStimulusCoordConfig* coord,
                               double*                       out_u_x,
                               double*                       out_u_y,
                               double*                       out_v_x,
                               double*                       out_v_y) {
    if (coord == NULL || out_u_x == NULL || out_u_y == NULL || out_v_x == NULL || out_v_y == NULL) {
        return false;
    }

    switch (coord->mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (coord->axis == SIM_STIMULUS_AXIS_Y) {
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
            double c = cos(coord->angle);
            double s = sin(coord->angle);
            *out_u_x = c;
            *out_u_y = s;
            *out_v_x = -s;
            *out_v_y = c;
            return true;
        }
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

static bool rd_seed_measure_linear_bounds(const SimStimulusRDSeedState* state,
                                          const SimField*               field,
                                          double                        t,
                                          SimStimulusRDSeedBounds*      out_bounds) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL || out_bounds == NULL ||
        !rd_seed_linear_map(&state->config.coord, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U) {
        out_bounds->u_min = out_bounds->u_max = 0.0;
        out_bounds->v_min = out_bounds->v_max = 0.0;
        out_bounds->u_span = out_bounds->v_span = out_bounds->avg_span = 1.0;
        return true;
    }

    double x0 = state->config.coord.origin_x - state->config.coord.velocity_x * t;
    double y0 = state->config.coord.origin_y - state->config.coord.velocity_y * t;
    double x1 = x0 + (double) (width - 1U) * state->config.coord.spacing_x;
    double y1 = y0 + (double) (height - 1U) * state->config.coord.spacing_y;
    double u_min;
    double u_max;
    double v_min;
    double v_max;

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(x1) || !isfinite(y1)) {
        return false;
    }

    {
        double xs[4]        = { x0, x1, x0, x1 };
        double ys[4]        = { y0, y0, y1, y1 };
        size_t corner_count = (field->layout.rank > 1U) ? 4U : 2U;
        double u            = u_x * xs[0] + u_y * ys[0];
        double v            = v_x * xs[0] + v_y * ys[0];

        u_min = u_max = u;
        v_min = v_max = v;
        for (size_t i = 1U; i < corner_count; ++i) {
            u = u_x * xs[i] + u_y * ys[i];
            v = v_x * xs[i] + v_y * ys[i];
            if (!isfinite(u) || !isfinite(v)) {
                return false;
            }
            if (u < u_min) {
                u_min = u;
            }
            if (u > u_max) {
                u_max = u;
            }
            if (v < v_min) {
                v_min = v;
            }
            if (v > v_max) {
                v_max = v;
            }
        }
    }

    out_bounds->u_min  = u_min;
    out_bounds->u_max  = u_max;
    out_bounds->v_min  = v_min;
    out_bounds->v_max  = v_max;
    out_bounds->u_span = u_max - u_min;
    out_bounds->v_span = v_max - v_min;
    if (out_bounds->u_span <= STIM_RD_SEED_EPS) {
        out_bounds->u_span = 1.0;
    }
    if (out_bounds->v_span <= STIM_RD_SEED_EPS) {
        out_bounds->v_span = 1.0;
    }
    out_bounds->avg_span = 0.5 * (out_bounds->u_span + out_bounds->v_span);
    if (out_bounds->avg_span <= STIM_RD_SEED_EPS) {
        out_bounds->avg_span = 1.0;
    }
    return true;
}

static void rd_seed_finalize_row(SimStimulusRDSeedState* state,
                                 size_t                  width,
                                 bool                    is_complex,
                                 double*                 dst_real,
                                 SimComplexDouble*       dst_complex,
                                 double                  rotation,
                                 double                  scale) {
    if (state == NULL || width == 0U) {
        return;
    }

    for (size_t i = 0U; i < width; ++i) {
        double raw = rd_seed_clamp01(state->vdsp_accum[i]);
        if (state->config.sharpness > STIM_RD_SEED_EPS) {
            raw = rd_seed_sigmoid(state->config.sharpness * (raw - state->config.threshold));
        }
        state->vdsp_accum[i] = state->config.bias + state->config.amplitude * rd_seed_clamp01(raw);
    }

    if (!is_complex) {
        sim_accel_copy_scale_real(state->vdsp_accum, dst_real, width, scale, true);
    } else {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (rotation != 0.0) {
            sin_r = sin(rotation);
            cos_r = cos(rotation);
        }
        sim_accel_accumulate_real_to_complex(
            state->vdsp_accum, dst_complex, width, scale * cos_r, scale * sin_r);
    }
}

static bool rd_seed_try_vdsp_rows(SimStimulusRDSeedState* state,
                                  const SimField*         field,
                                  bool                    is_complex,
                                  double*                 dst_real,
                                  SimComplexDouble*       dst_complex,
                                  size_t                  count,
                                  double                  t,
                                  double                  scale) {
    double                  u_x = 0.0;
    double                  u_y = 0.0;
    double                  v_x = 0.0;
    double                  v_y = 0.0;
    SimStimulusRDSeedBounds bounds;

    if (state == NULL || field == NULL ||
        !rd_seed_linear_map(&state->config.coord, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_RD_SEED_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX ||
        state->config.seed_count == 0U || state->seed_u == NULL || state->seed_v == NULL ||
        state->seed_phase == NULL) {
        return false;
    }
    if (!rd_seed_vdsp_ensure_buffers(state, width) ||
        !rd_seed_measure_linear_bounds(state, field, t, &bounds)) {
        return false;
    }

    double x0     = state->config.coord.origin_x - state->config.coord.velocity_x * t;
    double y0     = state->config.coord.origin_y - state->config.coord.velocity_y * t;
    double dx     = state->config.coord.spacing_x;
    double dy     = state->config.coord.spacing_y;
    double u_step = u_x * dx;
    double v_step = v_x * dx;

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(u_step) ||
        !isfinite(v_step)) {
        return false;
    }

    const vDSP_Length len         = (vDSP_Length) width;
    const int         vforce_len  = (int) width;
    double            phase_drive = state->config.phase - state->config.omega * t;
    double            avg_span    = bounds.avg_span;
    if (!isfinite(phase_drive) || avg_span <= STIM_RD_SEED_EPS || !isfinite(avg_span)) {
        return false;
    }

    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double u_start  = u_x * x0 + u_y * sample_y;
        double v_start  = v_x * x0 + v_y * sample_y;

        if (!isfinite(sample_y) || !isfinite(u_start) || !isfinite(v_start)) {
            return false;
        }

        vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
        vDSP_vrampD(&v_start, &v_step, state->vdsp_v, 1, len);
        vDSP_vclrD(state->vdsp_accum, 1, len);

        switch (state->config.mode) {
            case SIM_STIMULUS_RD_SEED_SPOTS: {
                double sigma = avg_span / (6.0 * state->config.scale);
                if (sigma <= STIM_RD_SEED_EPS || !isfinite(sigma)) {
                    return false;
                }
                double inv_two_sigma2 = -0.5 / (sigma * sigma);
                for (unsigned int seed = 0U; seed < state->config.seed_count; ++seed) {
                    double center_u     = bounds.u_min + state->seed_u[seed] * bounds.u_span;
                    double center_v     = bounds.v_min + state->seed_v[seed] * bounds.v_span;
                    double neg_center_u = -center_u;
                    double neg_center_v = -center_v;
                    if (!isfinite(center_u) || !isfinite(center_v)) {
                        return false;
                    }

                    vDSP_vsaddD(state->vdsp_u, 1, &neg_center_u, state->vdsp_value, 1, len);
                    vDSP_vsaddD(state->vdsp_v, 1, &neg_center_v, state->vdsp_work, 1, len);
                    vDSP_vsqD(state->vdsp_value, 1, state->vdsp_value, 1, len);
                    vDSP_vsqD(state->vdsp_work, 1, state->vdsp_work, 1, len);
                    vDSP_vaddD(
                        state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, inv_two_sigma2);
                    vvexp(state->vdsp_value, state->vdsp_value, &vforce_len);
                    vDSP_vaddD(
                        state->vdsp_accum, 1, state->vdsp_value, 1, state->vdsp_accum, 1, len);
                }
                sim_accel_scale_inplace_real(
                    state->vdsp_accum, width, 1.0 / (double) state->config.seed_count);
                break;
            }
            case SIM_STIMULUS_RD_SEED_STRIPES: {
                double k = state->config.scale * (2.0 * M_PI / avg_span);
                for (unsigned int seed = 0U; seed < state->config.seed_count; ++seed) {
                    double angle  = state->seed_phase[seed];
                    double c      = cos(angle);
                    double s      = sin(angle);
                    double offset = phase_drive + 2.0 * M_PI * state->seed_v[seed];
                    vDSP_vsmulD(state->vdsp_u, 1, &c, state->vdsp_value, 1, len);
                    if (s != 0.0) {
                        vDSP_vsmaD(
                            state->vdsp_v, 1, &s, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    }
                    sim_accel_scale_inplace_real(state->vdsp_value, width, k);
                    sim_accel_add_scalar_real(state->vdsp_value, width, offset);
                    vvcos(state->vdsp_value, state->vdsp_value, &vforce_len);
                    vDSP_vaddD(
                        state->vdsp_accum, 1, state->vdsp_value, 1, state->vdsp_accum, 1, len);
                }
                sim_accel_scale_inplace_real(
                    state->vdsp_accum, width, 0.5 / (double) state->config.seed_count);
                sim_accel_add_scalar_real(state->vdsp_accum, width, 0.5);
                break;
            }
            case SIM_STIMULUS_RD_SEED_LABYRINTH: {
                double k          = state->config.scale * (2.0 * M_PI / avg_span);
                double norm_scale = 0.5 / sqrt((double) state->config.seed_count);
                for (unsigned int seed = 0U; seed < state->config.seed_count; ++seed) {
                    double angle   = state->seed_phase[seed];
                    double c       = cos(angle);
                    double s       = sin(angle);
                    double offset  = phase_drive + 2.0 * M_PI * state->seed_v[seed];
                    double offset2 = state->seed_u[seed] * M_PI;
                    vDSP_vsmulD(state->vdsp_u, 1, &c, state->vdsp_value, 1, len);
                    if (s != 0.0) {
                        vDSP_vsmaD(
                            state->vdsp_v, 1, &s, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    }
                    sim_accel_scale_inplace_real(state->vdsp_value, width, k);
                    sim_accel_add_scalar_real(state->vdsp_value, width, offset);
                    vvsin(state->vdsp_work, state->vdsp_value, &vforce_len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, 1.7);
                    sim_accel_add_scalar_real(state->vdsp_value, width, offset2);
                    vvcos(state->vdsp_value, state->vdsp_value, &vforce_len);
                    sim_accel_scale_inplace_real(state->vdsp_value, width, 0.45);
                    vDSP_vaddD(
                        state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
                    vDSP_vaddD(
                        state->vdsp_accum, 1, state->vdsp_value, 1, state->vdsp_accum, 1, len);
                }
                sim_accel_scale_inplace_real(state->vdsp_accum, width, norm_scale);
                for (size_t i = 0U; i < width; ++i) {
                    state->vdsp_accum[i] = 0.5 + 0.5 * tanh(1.25 * state->vdsp_accum[i]);
                }
                break;
            }
            case SIM_STIMULUS_RD_SEED_RINGS:
            default: {
                double k              = state->config.scale * (2.0 * M_PI / avg_span);
                double envelope_scale = -0.75 / avg_span;
                for (unsigned int seed = 0U; seed < state->config.seed_count; ++seed) {
                    double center_u     = bounds.u_min + state->seed_u[seed] * bounds.u_span;
                    double center_v     = bounds.v_min + state->seed_v[seed] * bounds.v_span;
                    double neg_center_u = -center_u;
                    double neg_center_v = -center_v;
                    double theta_bias   = phase_drive + state->seed_phase[seed];
                    if (!isfinite(center_u) || !isfinite(center_v) || !isfinite(theta_bias)) {
                        return false;
                    }

                    vDSP_vsaddD(state->vdsp_u, 1, &neg_center_u, state->vdsp_value, 1, len);
                    vDSP_vsaddD(state->vdsp_v, 1, &neg_center_v, state->vdsp_work, 1, len);
                    vDSP_vsqD(state->vdsp_value, 1, state->vdsp_value, 1, len);
                    vDSP_vsqD(state->vdsp_work, 1, state->vdsp_work, 1, len);
                    vDSP_vaddD(
                        state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
                    vvsqrt(state->vdsp_value, state->vdsp_value, &vforce_len);

                    sim_accel_copy_scale_real(
                        state->vdsp_value, state->vdsp_work, width, 1.0, false);
                    sim_accel_scale_inplace_real(state->vdsp_work, width, k);
                    sim_accel_add_scalar_real(state->vdsp_work, width, theta_bias);
                    vvcos(state->vdsp_work, state->vdsp_work, &vforce_len);
                    sim_accel_scale_inplace_real(state->vdsp_work, width, 0.5);
                    sim_accel_add_scalar_real(state->vdsp_work, width, 0.5);

                    sim_accel_scale_inplace_real(state->vdsp_value, width, envelope_scale);
                    vvexp(state->vdsp_value, state->vdsp_value, &vforce_len);
                    vDSP_vmulD(state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_work, 1, len);
                    vDSP_vaddD(
                        state->vdsp_accum, 1, state->vdsp_work, 1, state->vdsp_accum, 1, len);
                }
                sim_accel_scale_inplace_real(
                    state->vdsp_accum, width, 1.0 / (double) state->config.seed_count);
                break;
            }
        }

        rd_seed_finalize_row(state,
                             width,
                             is_complex,
                             dst_real != NULL ? dst_real + row * width : NULL,
                             dst_complex != NULL ? dst_complex + row * width : NULL,
                             state->config.rotation,
                             scale);
    }

    return true;
}
#endif

static double rd_seed_eval_pattern(const SimStimulusRDSeedState*  state,
                                   const SimStimulusRDSeedBounds* bounds,
                                   double                         u,
                                   double                         v,
                                   double                         t) {
    const SimStimulusRDSeedConfig* cfg = &state->config;
    if (cfg->seed_count == 0U || state->seed_u == NULL || state->seed_v == NULL ||
        state->seed_phase == NULL || bounds == NULL) {
        return 0.0;
    }

    double phase_drive = cfg->phase - cfg->omega * t;
    double u_span      = bounds->u_span;
    double v_span      = bounds->v_span;
    double avg_span    = bounds->avg_span;
    if (avg_span <= STIM_RD_SEED_EPS) {
        avg_span = 1.0;
    }

    double raw = 0.0;
    switch (cfg->mode) {
        case SIM_STIMULUS_RD_SEED_SPOTS: {
            double sigma = avg_span / (6.0 * cfg->scale);
            if (sigma <= STIM_RD_SEED_EPS) {
                sigma = avg_span * 0.02 + STIM_RD_SEED_EPS;
            }
            double inv_two_sigma2 = 0.5 / (sigma * sigma);
            double sum            = 0.0;
            for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
                double center_u = bounds->u_min + state->seed_u[i] * u_span;
                double center_v = bounds->v_min + state->seed_v[i] * v_span;
                double du       = u - center_u;
                double dv       = v - center_v;
                double d2       = du * du + dv * dv;
                sum += exp(-d2 * inv_two_sigma2);
            }
            raw = sum / (double) cfg->seed_count;
            break;
        }
        case SIM_STIMULUS_RD_SEED_STRIPES: {
            double k   = cfg->scale * (2.0 * M_PI / avg_span);
            double sum = 0.0;
            for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
                double angle = state->seed_phase[i];
                double c     = cos(angle);
                double s     = sin(angle);
                double proj  = c * u + s * v;
                double theta = k * proj + phase_drive + 2.0 * M_PI * state->seed_v[i];
                sum += cos(theta);
            }
            raw = 0.5 + 0.5 * (sum / (double) cfg->seed_count);
            break;
        }
        case SIM_STIMULUS_RD_SEED_LABYRINTH: {
            double k   = cfg->scale * (2.0 * M_PI / avg_span);
            double sum = 0.0;
            for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
                double angle = state->seed_phase[i];
                double c     = cos(angle);
                double s     = sin(angle);
                double proj  = c * u + s * v;
                double theta = k * proj + phase_drive + 2.0 * M_PI * state->seed_v[i];
                sum += sin(theta) + 0.45 * cos(1.7 * theta + state->seed_u[i] * M_PI);
            }
            double norm = sum / sqrt((double) cfg->seed_count);
            raw         = 0.5 + 0.5 * tanh(1.25 * norm);
            break;
        }
        case SIM_STIMULUS_RD_SEED_RINGS:
        default: {
            double k   = cfg->scale * (2.0 * M_PI / avg_span);
            double sum = 0.0;
            for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
                double center_u = bounds->u_min + state->seed_u[i] * u_span;
                double center_v = bounds->v_min + state->seed_v[i] * v_span;
                double du       = u - center_u;
                double dv       = v - center_v;
                double r        = hypot(du, dv);
                double theta    = k * r + phase_drive + state->seed_phase[i];
                double envelope = exp(-0.75 * r / avg_span);
                sum += (0.5 + 0.5 * cos(theta)) * envelope;
            }
            raw = sum / (double) cfg->seed_count;
            break;
        }
    }

    raw = rd_seed_clamp01(raw);
    if (cfg->sharpness > STIM_RD_SEED_EPS) {
        raw = rd_seed_sigmoid(cfg->sharpness * (raw - cfg->threshold));
    }
    return rd_seed_clamp01(raw);
}

static SimResult rd_seed_measure_bounds(const SimStimulusRDSeedState* state,
                                        SimField*                     field,
                                        double                        t,
                                        SimStimulusRDSeedBounds*      out_bounds) {
    if (state == NULL || field == NULL || out_bounds == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const size_t count = sim_field_bytes(field) / field->element_size;
    if (count == 0U) {
        out_bounds->u_min    = 0.0;
        out_bounds->u_max    = 0.0;
        out_bounds->v_min    = 0.0;
        out_bounds->v_max    = 0.0;
        out_bounds->u_span   = 1.0;
        out_bounds->v_span   = 1.0;
        out_bounds->avg_span = 1.0;
        return SIM_RESULT_OK;
    }

    bool first = true;
    for (size_t i = 0U; i < count; ++i) {
        double x = 0.0;
        double y = 0.0;
        if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        double u = 0.0;
        double v = 0.0;
        rd_seed_map_coord(&state->config.coord, x, y, t, &u, &v);
        if (!isfinite(u) || !isfinite(v)) {
            continue;
        }

        if (first) {
            out_bounds->u_min = out_bounds->u_max = u;
            out_bounds->v_min = out_bounds->v_max = v;
            first                                 = false;
        } else {
            if (u < out_bounds->u_min) {
                out_bounds->u_min = u;
            }
            if (u > out_bounds->u_max) {
                out_bounds->u_max = u;
            }
            if (v < out_bounds->v_min) {
                out_bounds->v_min = v;
            }
            if (v > out_bounds->v_max) {
                out_bounds->v_max = v;
            }
        }
    }

    if (first) {
        out_bounds->u_min = out_bounds->u_max = 0.0;
        out_bounds->v_min = out_bounds->v_max = 0.0;
    }

    out_bounds->u_span = out_bounds->u_max - out_bounds->u_min;
    out_bounds->v_span = out_bounds->v_max - out_bounds->v_min;
    if (out_bounds->u_span <= STIM_RD_SEED_EPS) {
        out_bounds->u_span = 1.0;
    }
    if (out_bounds->v_span <= STIM_RD_SEED_EPS) {
        out_bounds->v_span = 1.0;
    }
    out_bounds->avg_span = 0.5 * (out_bounds->u_span + out_bounds->v_span);
    if (out_bounds->avg_span <= STIM_RD_SEED_EPS) {
        out_bounds->avg_span = 1.0;
    }
    return SIM_RESULT_OK;
}

static SimResult rd_seed_step(void*               state_ptr,
                              struct SimContext*  context,
                              struct SimOperator* self,
                              size_t              substep_index,
                              double              dt_sub,
                              void*               scratch,
                              size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) dt_sub;
    (void) scratch;
    (void) scratch_size;

    SimStimulusRDSeedState* state = (SimStimulusRDSeedState*) state_ptr;
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

    if (count == 0U) {
        return SIM_RESULT_OK;
    }
    if (state->config.amplitude == 0.0 && state->config.bias == 0.0) {
        return SIM_RESULT_OK;
    }

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;
    double t     = sim_context_time(context) + state->config.time_offset;

#if defined(SIM_HAVE_VDSP)
    if (rd_seed_try_vdsp_rows(state,
                              field,
                              is_complex,
                              sim_field_real_data(field),
                              sim_field_complex_data(field),
                              count,
                              t,
                              scale)) {
        return SIM_RESULT_OK;
    }
#endif

    SimStimulusRDSeedBounds bounds;
    SimResult               rc = rd_seed_measure_bounds(state, field, t, &bounds);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

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

            double u = 0.0;
            double v = 0.0;
            rd_seed_map_coord(&state->config.coord, x, y, t, &u, &v);
            double seed_value = rd_seed_eval_pattern(state, &bounds, u, v, t);
            double value      = scale * (state->config.bias + state->config.amplitude * seed_value);
            if (isfinite(value)) {
                dst[i] += value;
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

            double u = 0.0;
            double v = 0.0;
            rd_seed_map_coord(&state->config.coord, x, y, t, &u, &v);
            double seed_value = rd_seed_eval_pattern(state, &bounds, u, v, t);
            double re         = scale * (state->config.bias + state->config.amplitude * seed_value);
            double im         = 0.0;
            double out_re     = re * cos_r - im * sin_r;
            double out_im     = re * sin_r + im * cos_r;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += out_re;
                dst[i].im += out_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_rd_seed_operator(struct SimContext*             context,
                                            const SimStimulusRDSeedConfig* config,
                                            size_t*                        out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRDSeedConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_rd_seed"),
                                     sim_context_operator_count(context));
    }

    rd_seed_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_rd_seed",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusRDSeedState* state = (SimStimulusRDSeedState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config        = local;
    state->seed_u        = NULL;
    state->seed_v        = NULL;
    state->seed_phase    = NULL;
    state->seed_capacity = 0U;

    SimResult tables_rc = rd_seed_rebuild_tables(state);
    if (tables_rc != SIM_RESULT_OK) {
        rd_seed_destroy(state);
        return tables_rc;
    }
    rd_seed_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_rd_seed");

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
    info.abstract_id       = "stimulus_rd_seed";
    sim_operator_info_set_schema_identity(&info, "stimulus_rd_seed");
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
                                .fn                = rd_seed_step,
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
                                .symbolic      = rd_seed_symbolic,
                                .destroy       = rd_seed_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        rd_seed_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_rd_seed_config(struct SimContext*       context,
                                      size_t                   operator_index,
                                      SimStimulusRDSeedConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusRDSeedState* state = (SimStimulusRDSeedState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_rd_seed_update(struct SimContext*             context,
                                      size_t                         operator_index,
                                      const SimStimulusRDSeedConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusRDSeedState* state = (SimStimulusRDSeedState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusRDSeedConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }
    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_rd_seed"), operator_index);
    }

    rd_seed_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "stimulus_rd_seed"), true, local.scale_by_dt);

    SimStimulusRDSeedState probe = *state;
    probe.config                 = local;
    probe.seed_u                 = NULL;
    probe.seed_v                 = NULL;
    probe.seed_phase             = NULL;
    probe.seed_capacity          = 0U;

    SimResult rc = rd_seed_rebuild_tables(&probe);
    if (rc != SIM_RESULT_OK) {
        free(probe.seed_u);
        free(probe.seed_v);
        free(probe.seed_phase);
        return rc;
    }

    free(state->seed_u);
    free(state->seed_v);
    free(state->seed_phase);
    state->seed_u        = probe.seed_u;
    state->seed_v        = probe.seed_v;
    state->seed_phase    = probe.seed_phase;
    state->seed_capacity = probe.seed_capacity;
    state->config        = local;
    rd_seed_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
