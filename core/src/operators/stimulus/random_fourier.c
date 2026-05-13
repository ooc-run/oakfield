#include "oakfield/operators/stimulus/random_fourier.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_seed.h"
#include "oakfield/backend.h"
#include "oakfield/operator_identity.h"

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_RFF_EPS 1.0e-9
#define STIM_RFF_MAX_FEATURES 256U
#define STIM_RFF_VDSP_MIN_LEN 64U

/* Local PCG32 RNG for reproducible feature generation. */
typedef struct {
    uint64_t state;
    uint64_t inc;
} stim_rff_pcg32_t;

static uint32_t stim_rff_pcg32_random(stim_rff_pcg32_t* rng) {
    uint64_t old        = rng->state;
    rng->state          = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t) (old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void stim_rff_pcg32_seed(stim_rff_pcg32_t* rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc   = (initseq << 1u) | 1u;
    (void) stim_rff_pcg32_random(rng);
    rng->state += initstate;
    (void) stim_rff_pcg32_random(rng);
}

static double stim_rff_uniform(stim_rff_pcg32_t* rng) {
    return ldexp(stim_rff_pcg32_random(rng), -32); /* [0,1) */
}

#if defined(__APPLE__)
static inline void stim_rff_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void stim_rff_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void stim_rff_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

typedef struct SimStimulusRandomFourierState {
    SimStimulusRandomFourierConfig config;
    double*                        k_values;
    double*                        phi_values;
    double*                        weight_values; /* per-feature spectral weight (pre-pow) */
    unsigned int                   allocated_features;
    double                         feature_norm;
    SimClockMode                   clock_mode;
    double                         locked_time;
    size_t                         last_step_index;
    bool                           clock_initialized;
    double                         kernel_cached_time;
    size_t                         kernel_cached_step_index;
    bool                           kernel_cached_valid;
    double                         snapshot_locked_time;
    size_t                         snapshot_last_step_index;
    bool                           snapshot_clock_initialized;
    double                         snapshot_kernel_cached_time;
    size_t                         snapshot_kernel_cached_step_index;
    bool                           snapshot_kernel_cached_valid;
    char                           symbolic[160];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta;
    double* vdsp_cos;
    double* vdsp_sin;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
} SimStimulusRandomFourierState;

static void random_fourier_normalize(SimStimulusRandomFourierConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude))
        config->amplitude = 0.0;
    if (!isfinite(config->k_min))
        config->k_min = 0.0;
    if (!isfinite(config->k_max))
        config->k_max = 0.0;
    if (!isfinite(config->kx))
        config->kx = 0.0;
    if (!isfinite(config->ky))
        config->ky = 0.0;
    if (!isfinite(config->omega))
        config->omega = 0.0;

    sim_stimulus_coord_normalize(&config->coord);

    if (!isfinite(config->time_offset))
        config->time_offset = 0.0;
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0)
        config->nominal_dt = 0.0;
    if (!isfinite(config->spectral_slope))
        config->spectral_slope = 0.0;
    if (config->feature_count == 0U)
        config->feature_count = 8U;
    if (config->feature_count > STIM_RFF_MAX_FEATURES)
        config->feature_count = STIM_RFF_MAX_FEATURES;
    if (config->seed == 0ULL)
        config->seed = 1ULL;
    if (!config->use_wavevector &&
        (fabs(config->kx) > STIM_RFF_EPS || fabs(config->ky) > STIM_RFF_EPS)) {
        config->use_wavevector = true;
    }

    /* Ensure k_min <= k_max for sampling. */
    if (config->k_min > config->k_max) {
        double tmp    = config->k_min;
        config->k_min = config->k_max;
        config->k_max = tmp;
    }
}

static SimClockMode
random_fourier_resolve_clock_mode(const SimContext*                     context,
                                  const char*                           op_name,
                                  const SimStimulusRandomFourierConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_RFF_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_random_fourier");
    }
    return resolved;
}

static void random_fourier_refresh_symbolic(SimStimulusRandomFourierState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusRandomFourierConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "rff A=%.3g k∈[%.3g,%.3g] M=%u",
                    cfg->amplitude,
                    cfg->k_min,
                    cfg->k_max,
                    cfg->feature_count);
#else
    (void) state;
#endif
}

static const char* random_fourier_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusRandomFourierState* state = (const SimStimulusRandomFourierState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void random_fourier_eval_base(const SimStimulusRandomFourierState* state,
                                     double                               u,
                                     double                               t,
                                     double                               weight_base,
                                     double                               omega,
                                     double                               slope,
                                     double                               feature_norm,
                                     double*                              out_re,
                                     double*                              out_im) {
    double re_sum = 0.0;
    double im_sum = 0.0;

    if (state == NULL || state->config.feature_count == 0U) {
        if (out_re != NULL)
            *out_re = 0.0;
        if (out_im != NULL)
            *out_im = 0.0;
        return;
    }

    unsigned int M = state->config.feature_count;
    for (unsigned int m = 0U; m < M; ++m) {
        double k     = state->k_values[m];
        double theta = k * u - omega * t + state->phi_values[m];
        double s     = 0.0;
        double c     = 0.0;
        stim_rff_sincos(theta, &s, &c);
        double local_weight = weight_base;
        if (slope != 0.0 && state->weight_values != NULL) {
            local_weight *= state->weight_values[m] / feature_norm;
        }
        re_sum += local_weight * c;
        im_sum += local_weight * s;
    }

    if (out_re != NULL)
        *out_re = re_sum;
    if (out_im != NULL)
        *out_im = im_sum;
}

static void random_fourier_eval_base_wavevector(const SimStimulusRandomFourierState* state,
                                                double                               x,
                                                double                               y,
                                                double                               t,
                                                double                               weight_base,
                                                double                               omega,
                                                double                               slope,
                                                double                               feature_norm,
                                                double*                              out_re,
                                                double*                              out_im) {
    double re_sum = 0.0;
    double im_sum = 0.0;

    if (state == NULL || state->config.feature_count == 0U) {
        if (out_re != NULL)
            *out_re = 0.0;
        if (out_im != NULL)
            *out_im = 0.0;
        return;
    }

    const SimStimulusRandomFourierConfig* cfg      = &state->config;
    double                                dot      = cfg->kx * x + cfg->ky * y;
    double                                base_k   = hypot(cfg->kx, cfg->ky);
    bool                                  has_base = (base_k > STIM_RFF_EPS);

    unsigned int M = state->config.feature_count;
    for (unsigned int m = 0U; m < M; ++m) {
        double k       = state->k_values[m];
        double spatial = has_base ? ((k / base_k) * dot) : (k * x);
        double theta   = spatial - omega * t + state->phi_values[m];
        double s       = 0.0;
        double c       = 0.0;
        stim_rff_sincos(theta, &s, &c);
        double local_weight = weight_base;
        if (slope != 0.0 && state->weight_values != NULL) {
            local_weight *= state->weight_values[m] / feature_norm;
        }
        re_sum += local_weight * c;
        im_sum += local_weight * s;
    }

    if (out_re != NULL)
        *out_re = re_sum;
    if (out_im != NULL)
        *out_im = im_sum;
}

static void random_fourier_destroy(void* state_ptr) {
    SimStimulusRandomFourierState* state = (SimStimulusRandomFourierState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state->k_values);
    free(state->phi_values);
    free(state->weight_values);
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
    state->vdsp_block = NULL;
#endif
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool random_fourier_vdsp_ensure_buffers(SimStimulusRandomFourierState* state, size_t width) {
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

static bool random_fourier_linear_projection(const SimStimulusRandomFourierConfig* cfg,
                                             double*                               out_proj_x,
                                             double*                               out_proj_y) {
    if (cfg == NULL || out_proj_x == NULL || out_proj_y == NULL) {
        return false;
    }

    if (cfg->use_wavevector) {
        double base_k = hypot(cfg->kx, cfg->ky);
        if (base_k > STIM_RFF_EPS) {
            *out_proj_x = cfg->kx / base_k;
            *out_proj_y = cfg->ky / base_k;
        } else {
            *out_proj_x = 1.0;
            *out_proj_y = 0.0;
        }
        return true;
    }

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                *out_proj_x = 0.0;
                *out_proj_y = 1.0;
            } else {
                *out_proj_x = 1.0;
                *out_proj_y = 0.0;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE: {
            double s = 0.0;
            double c = 1.0;
            stim_rff_sincos(cfg->coord.angle, &s, &c);
            *out_proj_x = c;
            *out_proj_y = s;
            return true;
        }
        default:
            break;
    }

    return false;
}

static bool random_fourier_try_vdsp_linear_rows(SimStimulusRandomFourierState* state,
                                                const SimField*                field,
                                                bool                           is_complex,
                                                double*                        dst_real,
                                                SimComplexDouble*              dst_complex,
                                                size_t                         count,
                                                double                         scale,
                                                double                         t) {
    double proj_x = 0.0;
    double proj_y = 0.0;

    if (state == NULL || field == NULL || state->k_values == NULL || state->phi_values == NULL ||
        state->weight_values == NULL ||
        !random_fourier_linear_projection(&state->config, &proj_x, &proj_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_RFF_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }

    if (!random_fourier_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    unsigned int M            = state->config.feature_count;
    double       omega        = state->config.omega;
    double       feature_norm = (state->feature_norm > STIM_RFF_EPS) ? state->feature_norm : 1.0;
    double weight_base = (M > 0U) ? (scale * state->config.amplitude / sqrt((double) M)) : 0.0;
    double x0          = state->config.coord.origin_x - state->config.coord.velocity_x * t;
    double y0          = state->config.coord.origin_y - state->config.coord.velocity_y * t;
    double dx          = state->config.coord.spacing_x;
    double dy          = state->config.coord.spacing_y;
    double du          = proj_x * dx;
    double omega_t     = -omega * t;

    if (!isfinite(weight_base) || !isfinite(x0) || !isfinite(y0) || !isfinite(dx) ||
        !isfinite(dy) || !isfinite(du) || !isfinite(omega_t)) {
        return false;
    }
    if (weight_base == 0.0) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;

    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double u0       = proj_x * x0 + proj_y * sample_y;
        if (!isfinite(u0)) {
            return false;
        }

        vDSP_vclrD(state->vdsp_accum_re, 1, len);
        if (is_complex) {
            vDSP_vclrD(state->vdsp_accum_im, 1, len);
        }

        for (unsigned int m = 0U; m < M; ++m) {
            double local_weight = weight_base;
            if (state->config.spectral_slope != 0.0) {
                local_weight *= state->weight_values[m] / feature_norm;
            }
            double start = state->k_values[m] * u0 + omega_t + state->phi_values[m];
            double step  = state->k_values[m] * du;
            if (!isfinite(local_weight) || !isfinite(start) || !isfinite(step)) {
                return false;
            }

            vDSP_vrampD(&start, &step, state->vdsp_theta, 1, len);
            if (is_complex) {
                vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
            } else {
                vvcos(state->vdsp_cos, state->vdsp_theta, &vforce_len);
            }
            vDSP_vsmaD(state->vdsp_cos,
                       1,
                       &local_weight,
                       state->vdsp_accum_re,
                       1,
                       state->vdsp_accum_re,
                       1,
                       len);
            if (is_complex) {
                vDSP_vsmaD(state->vdsp_sin,
                           1,
                           &local_weight,
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
            double value_re = state->vdsp_accum_re[i];
            double value_im = state->vdsp_accum_im[i];
            if (isfinite(value_re) && isfinite(value_im)) {
                dst_complex[offset + i].re += value_re;
                dst_complex[offset + i].im += value_im;
            }
        }
    }

    return true;
}
#endif

static SimResult
random_fourier_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusRandomFourierState* state = (SimStimulusRandomFourierState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->snapshot_locked_time              = state->locked_time;
    state->snapshot_last_step_index          = state->last_step_index;
    state->snapshot_clock_initialized        = state->clock_initialized;
    state->snapshot_kernel_cached_time       = state->kernel_cached_time;
    state->snapshot_kernel_cached_step_index = state->kernel_cached_step_index;
    state->snapshot_kernel_cached_valid      = state->kernel_cached_valid;
    return SIM_RESULT_OK;
}

static SimResult
random_fourier_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusRandomFourierState* state = (SimStimulusRandomFourierState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->locked_time              = state->snapshot_locked_time;
    state->last_step_index          = state->snapshot_last_step_index;
    state->clock_initialized        = state->snapshot_clock_initialized;
    state->kernel_cached_time       = state->snapshot_kernel_cached_time;
    state->kernel_cached_step_index = state->snapshot_kernel_cached_step_index;
    state->kernel_cached_valid      = state->snapshot_kernel_cached_valid;
    return SIM_RESULT_OK;
}

static SimResult random_fourier_ensure_features(SimStimulusRandomFourierState* state) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    unsigned int desired = state->config.feature_count;
    if (desired == 0U) {
        return SIM_RESULT_OK;
    }
    if (state->allocated_features == desired && state->k_values != NULL &&
        state->phi_values != NULL) {
        return SIM_RESULT_OK;
    }

    double* k_vals   = (double*) realloc(state->k_values, (size_t) desired * sizeof(double));
    double* phi_vals = (double*) realloc(state->phi_values, (size_t) desired * sizeof(double));
    double* w_vals   = (double*) realloc(state->weight_values, (size_t) desired * sizeof(double));
    if (k_vals == NULL || phi_vals == NULL || w_vals == NULL) {
        free(k_vals);
        free(phi_vals);
        free(w_vals);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->k_values           = k_vals;
    state->phi_values         = phi_vals;
    state->weight_values      = w_vals;
    state->allocated_features = desired;

    stim_rff_pcg32_t rng;
    stim_rff_pcg32_seed(&rng, state->config.seed, state->config.seed ^ 0x9E3779B97F4A7C15ULL);

    double k_min = state->config.k_min;
    double k_max = state->config.k_max;
    double dk    = k_max - k_min;

    double slope  = state->config.spectral_slope;
    double sum_sq = 0.0;

    for (unsigned int i = 0U; i < desired; ++i) {
        double u = stim_rff_uniform(&rng);
        double v = stim_rff_uniform(&rng);

        double k   = (dk > 0.0) ? (k_min + dk * u) : k_min;
        double phi = 2.0 * M_PI * v;

        state->k_values[i]   = k;
        state->phi_values[i] = phi;

        if (slope != 0.0) {
            double freq = fabs(k);
            if (freq > STIM_RFF_EPS) {
                double amp_scale        = pow(freq, -0.5 * slope);
                state->weight_values[i] = amp_scale;
                sum_sq += amp_scale * amp_scale;
            } else {
                state->weight_values[i] = 1.0;
                sum_sq += 1.0;
            }
        } else {
            state->weight_values[i] = 1.0;
            sum_sq += 1.0;
        }
    }

    if (sum_sq > STIM_RFF_EPS) {
        state->feature_norm = sqrt(sum_sq / (double) desired);
    } else {
        state->feature_norm = 1.0;
    }

    return SIM_RESULT_OK;
}

static double random_fourier_drive_time(SimStimulusRandomFourierState* state,
                                        double                         base_time,
                                        double                         dt,
                                        size_t                         step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_RFF_EPS) {
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

    double increment = (state->config.nominal_dt > STIM_RFF_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

static double kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static double random_fourier_kernel_time(SimStimulusRandomFourierState* state,
                                         const KernelIR*                kernel,
                                         double                         dt_sub,
                                         size_t                         step_index) {
    if (state->kernel_cached_valid && state->kernel_cached_step_index == step_index) {
        return state->kernel_cached_time;
    }

    double base_time          = kernel_param_value(kernel, SIM_IR_PARAM_TIME);
    double t                  = random_fourier_drive_time(state, base_time, dt_sub, step_index);
    state->kernel_cached_time = t;
    state->kernel_cached_step_index = step_index;
    state->kernel_cached_valid      = true;
    state->last_step_index          = step_index;
    return t;
}

static SimResult random_fourier_kernel_value(SimStimulusRandomFourierState* state,
                                             const KernelIR*                kernel,
                                             size_t                         element_index,
                                             bool                           use_sin,
                                             double*                        out_value) {
    if (out_value == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRandomFourierConfig* cfg = &state->config;
    double                          dt  = kernel_param_value(kernel, SIM_IR_PARAM_DT);
    size_t step_index = (size_t) kernel_param_value(kernel, SIM_IR_PARAM_STEP_INDEX);
    double t          = random_fourier_kernel_time(state, kernel, dt, step_index);

    if (cfg->feature_count == 0U) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    if (cfg->amplitude == 0.0) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    SimResult prep = random_fourier_ensure_features(state);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    double       scale          = cfg->scale_by_dt ? dt : 1.0;
    double       omega          = cfg->omega;
    unsigned int M              = cfg->feature_count;
    double       slope          = cfg->spectral_slope;
    double       feature_norm   = (state->feature_norm > STIM_RFF_EPS) ? state->feature_norm : 1.0;
    double       weight_base    = (M > 0U) ? (1.0 / sqrt((double) M)) : 0.0;
    bool         separable      = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool         use_wavevector = cfg->use_wavevector;
    if (kernel->bindings == NULL || kernel->binding_count == 0U) {
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

    size_t count = 1U;
    for (size_t i = 0U; i < rank; ++i) {
        if (shape[i] == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count *= shape[i];
    }
    assert(element_index < count);

    size_t    ix       = 0U;
    size_t    iy       = 0U;
    SimResult coord_rc = sim_kernel_binding_index_to_xy(binding, element_index, &ix, &iy);
    assert(coord_rc == SIM_RESULT_OK);
    size_t extent_x = shape[rank - 1U];
    size_t extent_y = (rank == 1U) ? 1U : shape[rank - 2U];
    assert(ix < extent_x);
    assert(iy < extent_y);
    double x        = cfg->coord.origin_x + (double) ix * cfg->coord.spacing_x;
    double y        = cfg->coord.origin_y + (double) iy * cfg->coord.spacing_y;
    double sample_x = 0.0;
    double sample_y = 0.0;
    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

    double base_re = 0.0;
    double base_im = 0.0;
    if (use_wavevector) {
        random_fourier_eval_base_wavevector(state,
                                            sample_x,
                                            sample_y,
                                            t,
                                            weight_base,
                                            omega,
                                            slope,
                                            feature_norm,
                                            &base_re,
                                            &base_im);
    } else if (separable) {
        double fx_re = 0.0;
        double fx_im = 0.0;
        double fy_re = 0.0;
        double fy_im = 0.0;
        random_fourier_eval_base(
            state, sample_x, t, weight_base, omega, slope, feature_norm, &fx_re, &fx_im);
        random_fourier_eval_base(
            state, sample_y, t, weight_base, omega, slope, feature_norm, &fy_re, &fy_im);
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            base_re = fx_re + fy_re;
            base_im = fx_im + fy_im;
        } else {
            base_re = fx_re * fy_re - fx_im * fy_im;
            base_im = fx_re * fy_im + fx_im * fy_re;
        }
    } else {
        double u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
        random_fourier_eval_base(
            state, u, t, weight_base, omega, slope, feature_norm, &base_re, &base_im);
    }

    double value = cfg->amplitude * (use_sin ? base_im : base_re);
    value *= scale;
    *out_value = isfinite(value) ? value : 0.0;
    return SIM_RESULT_OK;
}

static SimResult random_fourier_ir_eval(void*           userdata,
                                        const KernelIR* kernel,
                                        size_t          element_index,
                                        size_t          component,
                                        double*         out_value) {
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRandomFourierState* state = (SimStimulusRandomFourierState*) userdata;
    return random_fourier_kernel_value(state, kernel, element_index, false, out_value);
}

static SimResult random_fourier_ir_eval_imag(void*           userdata,
                                             const KernelIR* kernel,
                                             size_t          element_index,
                                             size_t          component,
                                             double*         out_value) {
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRandomFourierState* state = (SimStimulusRandomFourierState*) userdata;
    return random_fourier_kernel_value(state, kernel, element_index, true, out_value);
}

static SimResult random_fourier_step(void*               state_ptr,
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

    SimStimulusRandomFourierState* state = (SimStimulusRandomFourierState*) state_ptr;
    if (state == NULL || context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusRandomFourierConfig* cfg   = &state->config;
    SimField*                       field = sim_context_field(context, cfg->field_index);
    if (field == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    void* raw_data = sim_field_data(field);
    if (raw_data == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (!is_complex) {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        if (field->element_size != sizeof(SimComplexDouble))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double t          = random_fourier_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U || cfg->amplitude == 0.0 || cfg->feature_count == 0U) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    SimResult prep = random_fourier_ensure_features(state);
    if (prep != SIM_RESULT_OK)
        return prep;

    double       scale          = cfg->scale_by_dt ? dt_sub : 1.0;
    double       omega          = cfg->omega;
    unsigned int M              = cfg->feature_count;
    double       slope          = cfg->spectral_slope;
    double       feature_norm   = (state->feature_norm > STIM_RFF_EPS) ? state->feature_norm : 1.0;
    double       weight_base    = (M > 0U) ? (1.0 / sqrt((double) M)) : 0.0;
    bool         separable      = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool         use_wavevector = cfg->use_wavevector;

    double*           dst_real    = NULL;
    SimComplexDouble* dst_complex = NULL;
    if (!is_complex) {
        dst_real = (double*) raw_data;
    } else {
        dst_complex = sim_field_complex_data(field);
    }

#if defined(SIM_HAVE_VDSP)
    if (!separable && random_fourier_try_vdsp_linear_rows(
                          state, field, is_complex, dst_real, dst_complex, count, scale, t)) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }
#endif

    for (size_t i = 0U; i < count; ++i) {
        double x        = 0.0;
        double y        = 0.0;
        double sample_x = 0.0;
        double sample_y = 0.0;
        if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

        double base_re = 0.0;
        double base_im = 0.0;
        if (use_wavevector) {
            random_fourier_eval_base_wavevector(state,
                                                sample_x,
                                                sample_y,
                                                t,
                                                weight_base,
                                                omega,
                                                slope,
                                                feature_norm,
                                                &base_re,
                                                &base_im);
        } else if (separable) {
            double fx_re = 0.0;
            double fx_im = 0.0;
            double fy_re = 0.0;
            double fy_im = 0.0;
            random_fourier_eval_base(
                state, sample_x, t, weight_base, omega, slope, feature_norm, &fx_re, &fx_im);
            random_fourier_eval_base(
                state, sample_y, t, weight_base, omega, slope, feature_norm, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
            random_fourier_eval_base(
                state, u, t, weight_base, omega, slope, feature_norm, &base_re, &base_im);
        }

        double value_re = cfg->amplitude * base_re;
        double value_im = cfg->amplitude * base_im;
        if (!is_complex) {
            if (isfinite(value_re)) {
                dst_real[i] += scale * value_re;
            }
        } else {
            if (isfinite(value_re) && isfinite(value_im)) {
                dst_complex[i].re += scale * value_re;
                dst_complex[i].im += scale * value_im;
            }
        }
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_random_fourier_operator(struct SimContext*                    context,
                                                   const SimStimulusRandomFourierConfig* config,
                                                   size_t* out_index) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusRandomFourierConfig local = { 0 };
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_random_fourier"),
                                     sim_context_operator_count(context));
    }

    random_fourier_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_random_fourier",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusRandomFourierState* state =
        (SimStimulusRandomFourierState*) calloc(1U, sizeof(SimStimulusRandomFourierState));
    if (state == NULL)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config = local;
    state->clock_mode =
        random_fourier_resolve_clock_mode(context, "stimulus_random_fourier", &state->config);
    state->k_values                 = NULL;
    state->phi_values               = NULL;
    state->weight_values            = NULL;
    state->allocated_features       = 0U;
    state->feature_norm             = 1.0;
    state->locked_time              = 0.0;
    state->last_step_index          = 0U;
    state->clock_initialized        = false;
    state->kernel_cached_time       = 0.0;
    state->kernel_cached_step_index = SIZE_MAX;
    state->kernel_cached_valid      = false;
    random_fourier_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_random_fourier");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_NOISE;
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
    info.abstract_id       = "stimulus_random_fourier";
    sim_operator_info_set_schema_identity(&info, "stimulus_random_fourier");
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

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = random_fourier_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_random_fourier")) {
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

                bool        is_complex  = sim_field_is_complex(field);
                SimIRType   field_type  = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node  = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId sample_node = sim_ir_builder_stateful(
                    builder, random_fourier_ir_eval, state, "stimulus_random_fourier");
                if (is_complex && sample_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId sample_im = sim_ir_builder_stateful(
                        builder, random_fourier_ir_eval_imag, state, "stimulus_random_fourier_im");
                    if (sample_im != SIM_IR_INVALID_NODE) {
                        sample_node = sim_ir_builder_complex_pack(builder, sample_node, sample_im);
                    }
                }
                SimIRNodeId sum =
                    sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, field_node, sample_node);

                if (field_node != SIM_IR_INVALID_NODE && sample_node != SIM_IR_INVALID_NODE &&
                    sum != SIM_IR_INVALID_NODE) {
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

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.save_state            = random_fourier_save;
                    kdesc.restore_state         = random_fourier_restore;
                    kdesc.destroy               = random_fourier_destroy;
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

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = random_fourier_symbolic,
                                .save_state    = random_fourier_save,
                                .restore_state = random_fourier_restore,
                                .destroy       = random_fourier_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        random_fourier_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_random_fourier_config(struct SimContext*              context,
                                             size_t                          operator_index,
                                             SimStimulusRandomFourierConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusRandomFourierState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusRandomFourierState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusRandomFourierState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_random_fourier_update(struct SimContext*                    context,
                                             size_t                                operator_index,
                                             const SimStimulusRandomFourierConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusRandomFourierState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusRandomFourierState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusRandomFourierState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    SimStimulusRandomFourierConfig local = state->config;
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_random_fourier"), operator_index);
    }

    random_fourier_normalize(&local);
    state->config     = local;
    state->clock_mode = random_fourier_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_random_fourier"), &state->config);

    /* Regenerate features to reflect updated config. */
    state->allocated_features  = 0U;
    state->feature_norm        = 1.0;
    state->kernel_cached_valid = false;
    random_fourier_ensure_features(state);
    random_fourier_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
