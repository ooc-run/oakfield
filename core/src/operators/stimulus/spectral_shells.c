#include "oakfield/operators/stimulus/spectral_shells.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

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

#define STIM_SHELLS_EPS 1.0e-9
#define STIM_SHELLS_MAX_SHELLS 128U
#define STIM_SHELLS_MAX_MODES_PER_SHELL 256U
#define STIM_SHELLS_MAX_MODES 8192U
#define STIM_SHELLS_VDSP_MIN_LEN 64U

typedef struct {
    uint64_t state;
    uint64_t inc;
} stim_shells_pcg32_t;

static uint32_t stim_shells_pcg32_random(stim_shells_pcg32_t* rng) {
    uint64_t old        = rng->state;
    rng->state          = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t) (old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void stim_shells_pcg32_seed(stim_shells_pcg32_t* rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc   = (initseq << 1u) | 1u;
    (void) stim_shells_pcg32_random(rng);
    rng->state += initstate;
    (void) stim_shells_pcg32_random(rng);
}

static double stim_shells_uniform(stim_shells_pcg32_t* rng) {
    return ldexp(stim_shells_pcg32_random(rng), -32); /* [0,1) */
}

#if defined(__APPLE__)
static inline void stim_shells_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void stim_shells_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void stim_shells_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

typedef struct SimStimulusSpectralShellsState {
    SimStimulusSpectralShellsConfig config;
    double*                         kx_values;
    double*                         ky_values;
    double*                         phi_values;
    double*                         weight_values;
    unsigned int                    allocated_modes;
    unsigned int                    active_modes;
    double                          mode_norm;
    bool                            mode_layout_1d;
    SimClockMode                    clock_mode;
    double                          locked_time;
    size_t                          last_step_index;
    bool                            clock_initialized;
    double                          snapshot_locked_time;
    size_t                          snapshot_last_step_index;
    bool                            snapshot_clock_initialized;
    char                            symbolic[192];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta;
    double* vdsp_cos;
    double* vdsp_sin;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
} SimStimulusSpectralShellsState;

static unsigned int spectral_shells_mode_count(const SimStimulusSpectralShellsConfig* config) {
    if (config == NULL || config->shell_count == 0U || config->modes_per_shell == 0U) {
        return 0U;
    }

    uint64_t total = (uint64_t) config->shell_count * (uint64_t) config->modes_per_shell;
    if (total > STIM_SHELLS_MAX_MODES) {
        total = STIM_SHELLS_MAX_MODES;
    }
    return (unsigned int) total;
}

static void spectral_shells_normalize(SimStimulusSpectralShellsConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->k_min)) {
        config->k_min = 0.0;
    }
    if (!isfinite(config->k_max)) {
        config->k_max = 0.0;
    }
    if (config->k_min < 0.0) {
        config->k_min = fabs(config->k_min);
    }
    if (config->k_max < 0.0) {
        config->k_max = fabs(config->k_max);
    }
    if (config->k_min > config->k_max) {
        double tmp    = config->k_min;
        config->k_min = config->k_max;
        config->k_max = tmp;
    }

    if (!isfinite(config->shell_width) || config->shell_width < 0.0) {
        config->shell_width = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0) {
        config->nominal_dt = 0.0;
    }
    if (!isfinite(config->spectral_slope)) {
        config->spectral_slope = 0.0;
    }

    if (config->shell_count == 0U) {
        config->shell_count = 4U;
    }
    if (config->shell_count > STIM_SHELLS_MAX_SHELLS) {
        config->shell_count = STIM_SHELLS_MAX_SHELLS;
    }
    if (config->modes_per_shell == 0U) {
        config->modes_per_shell = 8U;
    }
    if (config->modes_per_shell > STIM_SHELLS_MAX_MODES_PER_SHELL) {
        config->modes_per_shell = STIM_SHELLS_MAX_MODES_PER_SHELL;
    }

    uint64_t total = (uint64_t) config->shell_count * (uint64_t) config->modes_per_shell;
    if (total > STIM_SHELLS_MAX_MODES) {
        uint64_t per_shell_cap = STIM_SHELLS_MAX_MODES / (uint64_t) config->shell_count;
        if (per_shell_cap == 0U) {
            per_shell_cap = 1U;
        }
        config->modes_per_shell = (unsigned int) per_shell_cap;
    }

    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static SimClockMode
spectral_shells_resolve_clock_mode(const SimContext*                      context,
                                   const char*                            op_name,
                                   const SimStimulusSpectralShellsConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_SHELLS_EPS, &forced);

    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_spectral_shells");
    }
    return resolved;
}

static void spectral_shells_refresh_symbolic(SimStimulusSpectralShellsState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusSpectralShellsConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "spectral_shells A=%.3g k=[%.3g,%.3g] shells=%u modes=%u",
                    cfg->amplitude,
                    cfg->k_min,
                    cfg->k_max,
                    cfg->shell_count,
                    cfg->modes_per_shell);
#else
    (void) state;
#endif
}

static const char* spectral_shells_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusSpectralShellsState* state = (const SimStimulusSpectralShellsState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void spectral_shells_destroy(void* state_ptr) {
    SimStimulusSpectralShellsState* state = (SimStimulusSpectralShellsState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state->kx_values);
    free(state->ky_values);
    free(state->phi_values);
    free(state->weight_values);
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
    state->vdsp_block = NULL;
#endif
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool spectral_shells_vdsp_ensure_buffers(SimStimulusSpectralShellsState* state,
                                                size_t                          width) {
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

static bool spectral_shells_try_vdsp_linear_rows(SimStimulusSpectralShellsState* state,
                                                 const SimField*                 field,
                                                 bool                            is_complex,
                                                 double*                         dst_real,
                                                 SimComplexDouble*               dst_complex,
                                                 size_t                          count,
                                                 double                          scale,
                                                 double                          t,
                                                 bool                            layout_1d) {
    if (state == NULL || field == NULL || state->kx_values == NULL || state->ky_values == NULL ||
        state->phi_values == NULL || state->weight_values == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_SHELLS_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || count != plane || width > (size_t) INT_MAX) {
        return false;
    }

    if (!spectral_shells_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double mode_norm = (state->mode_norm > STIM_SHELLS_EPS) ? state->mode_norm : 1.0;
    double inv_sqrtM = 1.0 / sqrt((double) state->active_modes);
    double gain      = scale * state->config.amplitude * inv_sqrtM / mode_norm;
    if (!isfinite(gain) || !isfinite(state->config.omega) || !isfinite(t) ||
        !isfinite(state->config.coord.origin_x) || !isfinite(state->config.coord.spacing_x) ||
        (!layout_1d &&
         (!isfinite(state->config.coord.origin_y) || !isfinite(state->config.coord.spacing_y)))) {
        return false;
    }
    if (gain == 0.0) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    const double      theta_time = -state->config.omega * t;
    const double      origin_x   = state->config.coord.origin_x;
    const double      spacing_x  = state->config.coord.spacing_x;
    const double      origin_y   = layout_1d ? 0.0 : state->config.coord.origin_y;
    const double      spacing_y  = layout_1d ? 0.0 : state->config.coord.spacing_y;

    for (size_t row = 0U; row < height; ++row) {
        double y = origin_y + (double) row * spacing_y;

        vDSP_vclrD(state->vdsp_accum_re, 1, len);
        if (is_complex) {
            vDSP_vclrD(state->vdsp_accum_im, 1, len);
        }

        for (unsigned int m = 0U; m < state->active_modes; ++m) {
            double weight = gain * state->weight_values[m];
            double start  = state->kx_values[m] * origin_x + state->ky_values[m] * y + theta_time +
                            state->phi_values[m];
            double step   = state->kx_values[m] * spacing_x;
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
            double* row_ptr    = dst_real + offset;
            bool    all_finite = true;
            for (size_t i = 0U; i < width; ++i) {
                if (!isfinite(state->vdsp_accum_re[i])) {
                    all_finite = false;
                    break;
                }
            }
            if (all_finite) {
                vDSP_vaddD(row_ptr, 1, state->vdsp_accum_re, 1, row_ptr, 1, len);
            } else {
                for (size_t i = 0U; i < width; ++i) {
                    if (isfinite(state->vdsp_accum_re[i])) {
                        row_ptr[i] += state->vdsp_accum_re[i];
                    }
                }
            }
            continue;
        }

        SimComplexDouble* row_ptr    = dst_complex + offset;
        bool              all_finite = true;
        for (size_t i = 0U; i < width; ++i) {
            if (!isfinite(state->vdsp_accum_re[i]) || !isfinite(state->vdsp_accum_im[i])) {
                all_finite = false;
                break;
            }
        }
        if (all_finite) {
            double* row_re = &row_ptr[0].re;
            double* row_im = &row_ptr[0].im;
            vDSP_vaddD(row_re, 2, state->vdsp_accum_re, 1, row_re, 2, len);
            vDSP_vaddD(row_im, 2, state->vdsp_accum_im, 1, row_im, 2, len);
        } else {
            for (size_t i = 0U; i < width; ++i) {
                if (isfinite(state->vdsp_accum_re[i]) && isfinite(state->vdsp_accum_im[i])) {
                    row_ptr[i].re += state->vdsp_accum_re[i];
                    row_ptr[i].im += state->vdsp_accum_im[i];
                }
            }
        }
    }

    return true;
}
#endif

static SimResult
spectral_shells_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusSpectralShellsState* state = (SimStimulusSpectralShellsState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->snapshot_locked_time       = state->locked_time;
    state->snapshot_last_step_index   = state->last_step_index;
    state->snapshot_clock_initialized = state->clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult
spectral_shells_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusSpectralShellsState* state = (SimStimulusSpectralShellsState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->locked_time       = state->snapshot_locked_time;
    state->last_step_index   = state->snapshot_last_step_index;
    state->clock_initialized = state->snapshot_clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult spectral_shells_ensure_modes(SimStimulusSpectralShellsState* state,
                                              bool                            layout_1d) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    unsigned int desired = spectral_shells_mode_count(&state->config);
    if (desired == 0U) {
        state->active_modes    = 0U;
        state->mode_norm       = 1.0;
        state->mode_layout_1d  = layout_1d;
        state->allocated_modes = 0U;
        free(state->kx_values);
        free(state->ky_values);
        free(state->phi_values);
        free(state->weight_values);
        state->kx_values     = NULL;
        state->ky_values     = NULL;
        state->phi_values    = NULL;
        state->weight_values = NULL;
        return SIM_RESULT_OK;
    }

    if (state->active_modes == desired && state->allocated_modes == desired &&
        state->mode_layout_1d == layout_1d && state->kx_values != NULL &&
        state->ky_values != NULL && state->phi_values != NULL && state->weight_values != NULL) {
        return SIM_RESULT_OK;
    }

    double* kx_values     = (double*) malloc((size_t) desired * sizeof(double));
    double* ky_values     = (double*) malloc((size_t) desired * sizeof(double));
    double* phi_values    = (double*) malloc((size_t) desired * sizeof(double));
    double* weight_values = (double*) malloc((size_t) desired * sizeof(double));
    if (kx_values == NULL || ky_values == NULL || phi_values == NULL || weight_values == NULL) {
        free(kx_values);
        free(ky_values);
        free(phi_values);
        free(weight_values);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    stim_shells_pcg32_t rng;
    stim_shells_pcg32_seed(&rng, state->config.seed, state->config.seed ^ 0xC2B2AE3D27D4EB4FULL);

    double k_min       = state->config.k_min;
    double k_max       = state->config.k_max;
    double range       = k_max - k_min;
    double shell_step  = (state->config.shell_count > 1U && range > STIM_SHELLS_EPS)
                             ? (range / (double) (state->config.shell_count - 1U))
                             : 0.0;
    double shell_width = state->config.shell_width;
    if (shell_width <= STIM_SHELLS_EPS && range > STIM_SHELLS_EPS) {
        shell_width = range / (double) state->config.shell_count;
    }

    double       slope  = state->config.spectral_slope;
    double       sum_sq = 0.0;
    unsigned int index  = 0U;

    for (unsigned int shell = 0U; shell < state->config.shell_count && index < desired; ++shell) {
        double center   = (state->config.shell_count > 1U && range > STIM_SHELLS_EPS)
                              ? (k_min + shell_step * (double) shell)
                              : ((range > STIM_SHELLS_EPS) ? (0.5 * (k_min + k_max)) : k_min);
        double shell_lo = center;
        double shell_hi = center;

        if (range > STIM_SHELLS_EPS && shell_width > STIM_SHELLS_EPS) {
            shell_lo = center - 0.5 * shell_width;
            shell_hi = center + 0.5 * shell_width;
            if (shell_lo < k_min) {
                shell_lo = k_min;
            }
            if (shell_hi > k_max) {
                shell_hi = k_max;
            }
            if (shell_lo > shell_hi) {
                shell_lo = center;
                shell_hi = center;
            }
        }

        for (unsigned int mode = 0U; mode < state->config.modes_per_shell && index < desired;
             ++mode) {
            double u = stim_shells_uniform(&rng);
            double v = stim_shells_uniform(&rng);
            double w = stim_shells_uniform(&rng);

            double radius = center;
            if (shell_hi > shell_lo + STIM_SHELLS_EPS) {
                radius = shell_lo + (shell_hi - shell_lo) * u;
            }
            if (!isfinite(radius)) {
                radius = 0.0;
            }
            if (radius < 0.0) {
                radius = -radius;
            }

            if (layout_1d) {
                double sign      = (v < 0.5) ? -1.0 : 1.0;
                kx_values[index] = sign * radius;
                ky_values[index] = 0.0;
            } else {
                double angle     = 2.0 * M_PI * v;
                kx_values[index] = radius * cos(angle);
                ky_values[index] = radius * sin(angle);
            }

            phi_values[index] = 2.0 * M_PI * w;

            double local_weight = 1.0;
            if (slope != 0.0 && radius > STIM_SHELLS_EPS) {
                local_weight = pow(radius, -0.5 * slope);
            }
            weight_values[index] = local_weight;
            sum_sq += local_weight * local_weight;

            ++index;
        }
    }

    if (index == 0U) {
        free(kx_values);
        free(ky_values);
        free(phi_values);
        free(weight_values);
        return SIM_RESULT_INVALID_STATE;
    }

    free(state->kx_values);
    free(state->ky_values);
    free(state->phi_values);
    free(state->weight_values);
    state->kx_values       = kx_values;
    state->ky_values       = ky_values;
    state->phi_values      = phi_values;
    state->weight_values   = weight_values;
    state->allocated_modes = desired;
    state->active_modes    = index;
    state->mode_layout_1d  = layout_1d;
    if (sum_sq > STIM_SHELLS_EPS) {
        state->mode_norm = sqrt(sum_sq / (double) index);
    } else {
        state->mode_norm = 1.0;
    }

    return SIM_RESULT_OK;
}

static double spectral_shells_drive_time(SimStimulusSpectralShellsState* state,
                                         double                          base_time,
                                         double                          dt,
                                         size_t                          step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_SHELLS_EPS) {
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

    double increment = (state->config.nominal_dt > STIM_SHELLS_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

static void spectral_shells_eval(const SimStimulusSpectralShellsState* state,
                                 double                                x,
                                 double                                y,
                                 double                                t,
                                 double*                               out_re,
                                 double*                               out_im) {
    double re_sum = 0.0;
    double im_sum = 0.0;

    if (state == NULL || state->active_modes == 0U || state->kx_values == NULL ||
        state->ky_values == NULL || state->phi_values == NULL || state->weight_values == NULL) {
        if (out_re != NULL) {
            *out_re = 0.0;
        }
        if (out_im != NULL) {
            *out_im = 0.0;
        }
        return;
    }

    double mode_norm = (state->mode_norm > STIM_SHELLS_EPS) ? state->mode_norm : 1.0;
    double inv_sqrtM = 1.0 / sqrt((double) state->active_modes);
    double omega     = state->config.omega;

    for (unsigned int m = 0U; m < state->active_modes; ++m) {
        double theta =
            state->kx_values[m] * x + state->ky_values[m] * y - omega * t + state->phi_values[m];
        double s = 0.0;
        double c = 0.0;
        stim_shells_sincos(theta, &s, &c);

        double weight = inv_sqrtM * (state->weight_values[m] / mode_norm);
        re_sum += weight * c;
        im_sum += weight * s;
    }

    if (out_re != NULL) {
        *out_re = re_sum;
    }
    if (out_im != NULL) {
        *out_im = im_sum;
    }
}

static SimResult spectral_shells_step(void*               state_ptr,
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

    SimStimulusSpectralShellsState* state = (SimStimulusSpectralShellsState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusSpectralShellsConfig* cfg   = &state->config;
    SimField*                        field = sim_context_field(context, cfg->field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field->layout.rank == 0U || field->layout.rank > 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    void* raw_data = sim_field_data(field);
    if (raw_data == NULL) {
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

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double t          = spectral_shells_drive_time(state, base_time, dt_sub, step_index);

    bool      layout_1d = (field->layout.rank == 1U);
    SimResult prep      = spectral_shells_ensure_modes(state, layout_1d);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    if (count == 0U || cfg->amplitude == 0.0 || state->active_modes == 0U) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    double scale = cfg->scale_by_dt ? dt_sub : 1.0;

#if defined(SIM_HAVE_VDSP)
    if (spectral_shells_try_vdsp_linear_rows(state,
                                             field,
                                             is_complex,
                                             (double*) raw_data,
                                             sim_field_complex_data(field),
                                             count,
                                             scale,
                                             t,
                                             layout_1d)) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }
#endif

    if (!is_complex) {
        double* dst = (double*) raw_data;
        for (size_t i = 0U; i < count; ++i) {
            double x = 0.0;
            double y = 0.0;
            if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (layout_1d) {
                y = 0.0;
            }

            double re = 0.0;
            double im = 0.0;
            spectral_shells_eval(state, x, y, t, &re, &im);
            (void) im;

            double value = cfg->amplitude * re;
            if (isfinite(value)) {
                dst[i] += scale * value;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            double x = 0.0;
            double y = 0.0;
            if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (layout_1d) {
                y = 0.0;
            }

            double re = 0.0;
            double im = 0.0;
            spectral_shells_eval(state, x, y, t, &re, &im);

            double out_re = cfg->amplitude * re;
            double out_im = cfg->amplitude * im;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += scale * out_re;
                dst[i].im += scale * out_im;
            }
        }
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_spectral_shells_operator(struct SimContext*                     context,
                                                    const SimStimulusSpectralShellsConfig* config,
                                                    size_t* out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusSpectralShellsConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_spectral_shells"),
                                     sim_context_operator_count(context));
    }

    spectral_shells_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_spectral_shells",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusSpectralShellsState* state =
        (SimStimulusSpectralShellsState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config            = local;
    state->kx_values         = NULL;
    state->ky_values         = NULL;
    state->phi_values        = NULL;
    state->weight_values     = NULL;
    state->allocated_modes   = 0U;
    state->active_modes      = 0U;
    state->mode_norm         = 1.0;
    state->mode_layout_1d    = true;
    state->locked_time       = 0.0;
    state->last_step_index   = 0U;
    state->clock_initialized = false;
    state->clock_mode =
        spectral_shells_resolve_clock_mode(context, "stimulus_spectral_shells", &state->config);
    spectral_shells_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_spectral_shells");

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
    info.abstract_id       = "stimulus_spectral_shells";
    sim_operator_info_set_schema_identity(&info, "stimulus_spectral_shells");
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
                                .fn                = spectral_shells_step,
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
                                .symbolic      = spectral_shells_symbolic,
                                .save_state    = spectral_shells_save,
                                .restore_state = spectral_shells_restore,
                                .destroy       = spectral_shells_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        spectral_shells_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_spectral_shells_config(struct SimContext*               context,
                                              size_t                           operator_index,
                                              SimStimulusSpectralShellsConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusSpectralShellsState* state = (SimStimulusSpectralShellsState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_spectral_shells_update(struct SimContext*                     context,
                                              size_t                                 operator_index,
                                              const SimStimulusSpectralShellsConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusSpectralShellsState* state = (SimStimulusSpectralShellsState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusSpectralShellsConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_spectral_shells"), operator_index);
    }

    spectral_shells_normalize(&local);
    state->config     = local;
    state->clock_mode = spectral_shells_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_spectral_shells"), &state->config);

    state->active_modes    = 0U;
    state->allocated_modes = 0U;
    state->mode_norm       = 1.0;
    free(state->kx_values);
    free(state->ky_values);
    free(state->phi_values);
    free(state->weight_values);
    state->kx_values     = NULL;
    state->ky_values     = NULL;
    state->phi_values    = NULL;
    state->weight_values = NULL;

    spectral_shells_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
