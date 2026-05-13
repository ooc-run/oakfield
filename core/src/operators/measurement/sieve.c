#include "oakfield/operators/measurement/sieve.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIEVE_SYMBOLIC_CAPACITY 240
#define SIEVE_SIGMA_MIN 1.0e-3
#define SIEVE_SAMPLE_SPACING_MIN 1.0e-9
#define SIEVE_TAPS_MIN 3U

typedef struct SimSieveOperatorState {
    SimSieveOperatorConfig config;
    double*                kernel;
    double*                kernel_temp;
    unsigned int           kernel_size;
    double*                scratch;
    SimComplexDouble*      scratch_complex;
    size_t                 scratch_capacity;
    char                   symbolic[SIEVE_SYMBOLIC_CAPACITY];
} SimSieveOperatorState;

static const char* sieve_mode_name(SimSieveMode mode) {
    switch (mode) {
        case SIM_SIEVE_MODE_LOW_PASS:
            return "low_pass";
        case SIM_SIEVE_MODE_HIGH_PASS:
            return "high_pass";
        case SIM_SIEVE_MODE_BAND_PASS_DOG:
            return "band_pass_dog";
        case SIM_SIEVE_MODE_BAND_STOP_DOG:
            return "band_stop_dog";
        case SIM_SIEVE_MODE_SAVGOL_SMOOTH:
            return "savgol_smooth";
        case SIM_SIEVE_MODE_SAVGOL_DERIVATIVE:
            return "savgol_derivative";
        case SIM_SIEVE_MODE_HANN_LOW_PASS:
            return "hann_low_pass";
        case SIM_SIEVE_MODE_HANN_HIGH_PASS:
            return "hann_high_pass";
        case SIM_SIEVE_MODE_BLACKMAN_LOW_PASS:
            return "blackman_low_pass";
        case SIM_SIEVE_MODE_BLACKMAN_HIGH_PASS:
            return "blackman_high_pass";
        case SIM_SIEVE_MODE_TUKEY_LOW_PASS:
            return "tukey_low_pass";
        case SIM_SIEVE_MODE_TUKEY_HIGH_PASS:
            return "tukey_high_pass";
        default:
            return "low_pass";
    }
}

static bool sieve_mode_uses_sigma(SimSieveMode mode) {
    return mode != SIM_SIEVE_MODE_SAVGOL_SMOOTH && mode != SIM_SIEVE_MODE_SAVGOL_DERIVATIVE;
}

static bool sieve_mode_uses_sigma2(SimSieveMode mode) {
    return mode == SIM_SIEVE_MODE_BAND_PASS_DOG || mode == SIM_SIEVE_MODE_BAND_STOP_DOG;
}

static bool sieve_mode_uses_poly_order(SimSieveMode mode) {
    return mode == SIM_SIEVE_MODE_SAVGOL_SMOOTH || mode == SIM_SIEVE_MODE_SAVGOL_DERIVATIVE;
}

static bool sieve_mode_is_derivative(SimSieveMode mode) {
    return mode == SIM_SIEVE_MODE_SAVGOL_DERIVATIVE;
}

static bool sieve_mode_is_tukey(SimSieveMode mode) {
    return mode == SIM_SIEVE_MODE_TUKEY_LOW_PASS || mode == SIM_SIEVE_MODE_TUKEY_HIGH_PASS;
}

static bool sieve_mode_is_high_pass_like(SimSieveMode mode) {
    return mode == SIM_SIEVE_MODE_HIGH_PASS || mode == SIM_SIEVE_MODE_BAND_STOP_DOG ||
           mode == SIM_SIEVE_MODE_HANN_HIGH_PASS || mode == SIM_SIEVE_MODE_BLACKMAN_HIGH_PASS ||
           mode == SIM_SIEVE_MODE_TUKEY_HIGH_PASS;
}

static unsigned int
sieve_normalize_taps(unsigned int taps, unsigned int previous_taps, bool has_previous_taps) {
    if (taps < SIEVE_TAPS_MIN) {
        taps = SIEVE_TAPS_MIN;
    }
    if ((taps % 2U) == 0U) {
        if (has_previous_taps && previous_taps >= SIEVE_TAPS_MIN && taps < previous_taps) {
            taps -= 1U;
        } else {
            taps += 1U;
        }
    }
    return taps;
}

static double sieve_clamp(double v, double lo, double hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static double sieve_effective_window_alpha(const SimSieveOperatorConfig* config) {
    double alpha;

    if (config == NULL) {
        return 0.0;
    }

    alpha = (isfinite(config->window_alpha) && config->window_alpha > 0.0) ? config->window_alpha
                                                                           : config->sigma;
    return sieve_clamp(alpha, 0.0, 1.0);
}

static void sieve_normalize_config(SimSieveOperatorConfig* config,
                                   unsigned int            previous_taps,
                                   bool                    has_previous_taps) {
    if (!config) {
        return;
    }

    config->taps = sieve_normalize_taps(config->taps, previous_taps, has_previous_taps);

    if (!isfinite(config->sigma) || config->sigma < SIEVE_SIGMA_MIN) {
        config->sigma = SIEVE_SIGMA_MIN;
    }
    if (!isfinite(config->sigma2) || config->sigma2 < SIEVE_SIGMA_MIN) {
        config->sigma2 = 2.0 * config->sigma;
    }

    if (!isfinite(config->gain)) {
        config->gain = 1.0;
    }
    if (!isfinite(config->sample_spacing) || config->sample_spacing <= 0.0) {
        config->sample_spacing = 1.0;
    } else if (config->sample_spacing < SIEVE_SAMPLE_SPACING_MIN) {
        config->sample_spacing = SIEVE_SAMPLE_SPACING_MIN;
    }
    if (!isfinite(config->window_alpha) || config->window_alpha < 0.0) {
        config->window_alpha = 0.0;
    } else if (config->window_alpha > 1.0) {
        config->window_alpha = 1.0;
    }

    switch (config->mode) {
        case SIM_SIEVE_MODE_LOW_PASS:
        case SIM_SIEVE_MODE_HIGH_PASS:
        case SIM_SIEVE_MODE_BAND_PASS_DOG:
        case SIM_SIEVE_MODE_BAND_STOP_DOG:
        case SIM_SIEVE_MODE_SAVGOL_SMOOTH:
        case SIM_SIEVE_MODE_SAVGOL_DERIVATIVE:
        case SIM_SIEVE_MODE_HANN_LOW_PASS:
        case SIM_SIEVE_MODE_HANN_HIGH_PASS:
        case SIM_SIEVE_MODE_BLACKMAN_LOW_PASS:
        case SIM_SIEVE_MODE_BLACKMAN_HIGH_PASS:
        case SIM_SIEVE_MODE_TUKEY_LOW_PASS:
        case SIM_SIEVE_MODE_TUKEY_HIGH_PASS:
            break;
        default:
            config->mode = SIM_SIEVE_MODE_LOW_PASS;
            break;
    }

    switch (config->boundary) {
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_PERIODIC:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            break;
        default:
            config->boundary = SIM_IR_BOUNDARY_NEUMANN;
            break;
    }

    if (config->poly_order >= config->taps) {
        config->poly_order = (config->taps > 1U) ? (config->taps - 1U) : 0U;
    }
    if (config->derivative_order >= config->taps) {
        config->derivative_order = (config->taps > 1U) ? (config->taps - 1U) : 0U;
    }
    if (sieve_mode_is_derivative(config->mode)) {
        if (config->derivative_order < 1U) {
            config->derivative_order = 1U;
        }
        if (config->poly_order < config->derivative_order) {
            config->poly_order = config->derivative_order;
        }
    }

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static void sieve_refresh_symbolic(SimSieveOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimSieveOperatorConfig* cfg         = &state->config;
    char                          primary[48] = { 0 };
    char                          extra[96]   = { 0 };

    if (sieve_mode_uses_sigma(cfg->mode)) {
        (void) snprintf(primary, sizeof(primary), " sigma=%.3g", cfg->sigma);
    }

    if (sieve_mode_uses_sigma2(cfg->mode)) {
        (void) snprintf(extra, sizeof(extra), " s2=%.3g", cfg->sigma2);
    } else if (sieve_mode_uses_poly_order(cfg->mode)) {
        (void) snprintf(extra, sizeof(extra), " poly=%u", cfg->poly_order);
        if (sieve_mode_is_derivative(cfg->mode)) {
            size_t offset = strlen(extra);
            (void) snprintf(extra + offset,
                            sizeof(extra) - offset,
                            " deriv=%u dx=%.3g",
                            cfg->derivative_order,
                            cfg->sample_spacing);
        }
    } else if (sieve_mode_is_tukey(cfg->mode)) {
        (void) snprintf(extra, sizeof(extra), " alpha=%.3g", sieve_effective_window_alpha(cfg));
    }

    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "sieve mode=%s taps=%u%s%s gain=%.3g boundary=%s scale_by_dt=%s",
                    sieve_mode_name(cfg->mode),
                    cfg->taps,
                    primary,
                    extra,
                    cfg->gain,
                    sim_boundary_policy_name(cfg->boundary),
                    cfg->scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static bool sieve_alloc_kernel(SimSieveOperatorState* state, unsigned int taps) {
    double* kernel_new      = (double*) malloc(taps * sizeof(double));
    double* kernel_temp_new = (double*) malloc(taps * sizeof(double));
    if (!kernel_new || !kernel_temp_new) {
        free(kernel_new);
        free(kernel_temp_new);
        return false;
    }

    free(state->kernel);
    free(state->kernel_temp);
    state->kernel      = kernel_new;
    state->kernel_temp = kernel_temp_new;

    state->kernel_size = taps;
    return true;
}

static void sieve_make_gaussian(double* kernel, unsigned int taps, double sigma) {
    unsigned int radius = taps / 2U;

    double sum = 0.0;
    for (unsigned int k = 0U; k < taps; ++k) {
        int    offset = (int) k - (int) radius;
        double x      = (double) offset;
        double value  = exp(-(x * x) / (2.0 * sigma * sigma));
        kernel[k]     = value;
        sum += value;
    }

    if (sum <= 0.0) {
        for (unsigned int k = 0U; k < taps; ++k) {
            kernel[k] = (1.0 / (double) taps);
        }
    } else {
        double inv_sum = 1.0 / sum;
        for (unsigned int k = 0U; k < taps; ++k) {
            kernel[k] *= inv_sum;
        }
    }
}

static void
sieve_make_window(double* kernel, unsigned int taps, SimSieveMode mode, double tukey_alpha) {
    unsigned int n     = taps;
    double       sum   = 0.0;
    double       alpha = sieve_clamp(tukey_alpha, 0.0, 1.0);
    for (unsigned int k = 0U; k < n; ++k) {
        double w = 1.0;
        double t = (double) k / (double) (n - 1U);
        switch (mode) {
            case SIM_SIEVE_MODE_HANN_LOW_PASS:
            case SIM_SIEVE_MODE_HANN_HIGH_PASS:
                w = 0.5 - 0.5 * cos(2.0 * M_PI * t);
                break;
            case SIM_SIEVE_MODE_BLACKMAN_LOW_PASS:
            case SIM_SIEVE_MODE_BLACKMAN_HIGH_PASS:
                w = 0.42 - 0.5 * cos(2.0 * M_PI * t) + 0.08 * cos(4.0 * M_PI * t);
                break;
            case SIM_SIEVE_MODE_TUKEY_LOW_PASS:
            case SIM_SIEVE_MODE_TUKEY_HIGH_PASS:
                if (t < alpha / 2.0) {
                    w = 0.5 * (1.0 + cos(M_PI * (2.0 * t / alpha - 1.0)));
                } else if (t <= 1.0 - alpha / 2.0) {
                    w = 1.0;
                } else {
                    w = 0.5 * (1.0 + cos(M_PI * (2.0 * t / alpha - 2.0 / alpha + 1.0)));
                }
                break;
            default:
                w = 1.0;
                break;
        }
        kernel[k] = w;
        sum += w;
    }

    if (sum > 0.0) {
        double inv_sum = 1.0 / sum;
        for (unsigned int k = 0U; k < n; ++k) {
            kernel[k] *= inv_sum;
        }
    }
}

static double sieve_factorial(unsigned int n) {
    double f = 1.0;
    for (unsigned int i = 2U; i <= n; ++i) {
        f *= (double) i;
    }
    return f;
}

static bool sieve_invert_matrix(double* a, double* inv, unsigned int n) {
    for (unsigned int r = 0U; r < n; ++r) {
        for (unsigned int c = 0U; c < n; ++c) {
            inv[r * n + c] = (r == c) ? 1.0 : 0.0;
        }
    }

    for (unsigned int i = 0U; i < n; ++i) {
        double       pivot     = a[i * n + i];
        unsigned int pivot_row = i;
        for (unsigned int r = i + 1U; r < n; ++r) {
            double candidate = fabs(a[r * n + i]);
            if (candidate > fabs(pivot)) {
                pivot     = a[r * n + i];
                pivot_row = r;
            }
        }

        if (fabs(pivot) < 1.0e-12) {
            return false;
        }

        if (pivot_row != i) {
            for (unsigned int c = 0U; c < n; ++c) {
                double tmp_a         = a[i * n + c];
                a[i * n + c]         = a[pivot_row * n + c];
                a[pivot_row * n + c] = tmp_a;

                double tmp_inv         = inv[i * n + c];
                inv[i * n + c]         = inv[pivot_row * n + c];
                inv[pivot_row * n + c] = tmp_inv;
            }
        }

        double inv_pivot = 1.0 / a[i * n + i];
        for (unsigned int c = 0U; c < n; ++c) {
            a[i * n + c] *= inv_pivot;
            inv[i * n + c] *= inv_pivot;
        }

        for (unsigned int r = 0U; r < n; ++r) {
            if (r == i) {
                continue;
            }
            double factor = a[r * n + i];
            for (unsigned int c = 0U; c < n; ++c) {
                a[r * n + c] -= factor * a[i * n + c];
                inv[r * n + c] -= factor * inv[i * n + c];
            }
        }
    }

    return true;
}

static bool sieve_build_savgol(double*      kernel,
                               unsigned int taps,
                               unsigned int order,
                               unsigned int deriv_order,
                               double       sample_spacing) {
    if (order < deriv_order) {
        return false;
    }

    unsigned int n   = taps;
    unsigned int m   = order + 1U;
    double*      ata = (double*) calloc(m * m, sizeof(double));
    double*      inv = (double*) calloc(m * m, sizeof(double));
    if (!ata || !inv) {
        free(ata);
        free(inv);
        return false;
    }

    int radius = (int) taps / 2;
    for (unsigned int i = 0U; i < m; ++i) {
        for (unsigned int j = 0U; j < m; ++j) {
            double sum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                sum += pow((double) k, (double) (i + j));
            }
            ata[i * m + j] = sum;
        }
    }

    if (!sieve_invert_matrix(ata, inv, m)) {
        free(ata);
        free(inv);
        return false;
    }

    double factor = sieve_factorial(deriv_order);
    for (unsigned int k = 0U; k < n; ++k) {
        int    x     = (int) k - radius;
        double coeff = 0.0;
        for (unsigned int j = 0U; j < m; ++j) {
            coeff += inv[deriv_order * m + j] * pow((double) x, (double) j);
        }
        kernel[k] = coeff * factor;
    }

    free(ata);
    free(inv);

    if (deriv_order == 0U) {
        double sum = 0.0;
        for (unsigned int k = 0U; k < n; ++k) {
            sum += kernel[k];
        }
        if (fabs(sum) > 1.0e-12) {
            double inv_sum = 1.0 / sum;
            for (unsigned int k = 0U; k < n; ++k) {
                kernel[k] *= inv_sum;
            }
        }
    } else {
        double spacing_scale = pow(sample_spacing, (double) deriv_order);
        if (!isfinite(spacing_scale) || fabs(spacing_scale) < 1.0e-18) {
            return false;
        }
        spacing_scale = 1.0 / spacing_scale;
        for (unsigned int k = 0U; k < n; ++k) {
            kernel[k] *= spacing_scale;
        }
    }

    return true;
}

static size_t
sieve_resolve_index(ptrdiff_t index, size_t count, SimIRBoundaryPolicy boundary, bool* out_valid) {
    if (out_valid != NULL) {
        *out_valid = true;
    }

    if (count == 0U) {
        if (out_valid != NULL) {
            *out_valid = false;
        }
        return 0U;
    }

    switch (boundary) {
        case SIM_IR_BOUNDARY_PERIODIC: {
            ptrdiff_t mod = index % (ptrdiff_t) count;
            if (mod < 0) {
                mod += (ptrdiff_t) count;
            }
            return (size_t) mod;
        }
        case SIM_IR_BOUNDARY_REFLECTIVE:
            if (index < 0) {
                index = -index;
            }
            if ((size_t) index >= count) {
                index = (ptrdiff_t) (count - 1U);
            }
            return (size_t) index;
        case SIM_IR_BOUNDARY_DIRICHLET:
            if (index < 0 || (size_t) index >= count) {
                if (out_valid != NULL) {
                    *out_valid = false;
                }
                return 0U;
            }
            return (size_t) index;
        case SIM_IR_BOUNDARY_NEUMANN:
        default:
            break;
    }

    if (index < 0) {
        return 0U;
    }
    if ((size_t) index >= count) {
        return count - 1U;
    }
    return (size_t) index;
}

static SimResult sieve_rebuild_kernel(SimSieveOperatorState* state) {
    if (!state) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    unsigned int taps   = state->config.taps;
    double       sigma  = state->config.sigma;
    double       sigma2 = state->config.sigma2;

    if (!sieve_alloc_kernel(state, taps)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    switch (state->config.mode) {
        case SIM_SIEVE_MODE_LOW_PASS:
        case SIM_SIEVE_MODE_HIGH_PASS:
            sieve_make_gaussian(state->kernel, taps, sigma);
            break;
        case SIM_SIEVE_MODE_BAND_PASS_DOG:
        case SIM_SIEVE_MODE_BAND_STOP_DOG:
            if (sigma2 < sigma) {
                double tmp = sigma;
                sigma      = sigma2;
                sigma2     = tmp;
            }
            sieve_make_gaussian(state->kernel, taps, sigma);
            sieve_make_gaussian(state->kernel_temp, taps, sigma2);
            for (unsigned int k = 0U; k < taps; ++k) {
                state->kernel[k] = state->kernel[k] - state->kernel_temp[k];
            }
            break;
        case SIM_SIEVE_MODE_SAVGOL_SMOOTH:
            if (!sieve_build_savgol(state->kernel, taps, state->config.poly_order, 0U, 1.0)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            break;
        case SIM_SIEVE_MODE_SAVGOL_DERIVATIVE:
            if (!sieve_build_savgol(state->kernel,
                                    taps,
                                    state->config.poly_order,
                                    state->config.derivative_order,
                                    state->config.sample_spacing)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            break;
        case SIM_SIEVE_MODE_HANN_LOW_PASS:
        case SIM_SIEVE_MODE_HANN_HIGH_PASS:
        case SIM_SIEVE_MODE_BLACKMAN_LOW_PASS:
        case SIM_SIEVE_MODE_BLACKMAN_HIGH_PASS:
        case SIM_SIEVE_MODE_TUKEY_LOW_PASS:
        case SIM_SIEVE_MODE_TUKEY_HIGH_PASS:
            sieve_make_window(state->kernel,
                              taps,
                              state->config.mode,
                              sieve_effective_window_alpha(&state->config));
            break;
        default:
            sieve_make_gaussian(state->kernel, taps, sigma);
            break;
    }

    return SIM_RESULT_OK;
}

static SimResult sieve_ensure_scratch(SimSieveOperatorState* state, size_t count, bool is_complex) {
    if (!state) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->scratch_capacity >= count) {
        return SIM_RESULT_OK;
    }

    if (is_complex) {
        SimComplexDouble* scratch_complex =
            (SimComplexDouble*) realloc(state->scratch_complex, count * sizeof(SimComplexDouble));
        if (!scratch_complex) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->scratch_complex = scratch_complex;
    } else {
        double* scratch = (double*) realloc(state->scratch, count * sizeof(double));
        if (!scratch) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->scratch = scratch;
    }

    state->scratch_capacity = count;
    return SIM_RESULT_OK;
}

static void sieve_destroy(void* state_ptr) {
    SimSieveOperatorState* state = (SimSieveOperatorState*) state_ptr;
    if (!state) {
        return;
    }

    free(state->kernel);
    free(state->kernel_temp);
    free(state->scratch);
    free(state->scratch_complex);
    free(state);
}

static const char* sieve_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimSieveOperatorState* state = (const SimSieveOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult
sieve_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimSieveOperatorState* state = (SimSieveOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* input_field  = sim_context_field(context, state->config.input_field);
    SimField* output_field = sim_context_field(context, state->config.output_field);
    if (!input_field || !output_field) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_operator_field_domain_is_f64_or_c64(input_field) ||
        !sim_operator_field_domain_is_f64_or_c64(output_field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    bool input_complex  = sim_field_is_complex(input_field);
    bool output_complex = sim_field_is_complex(output_field);
    if (input_complex != output_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    const bool is_complex = input_complex;

    size_t count;
    if (is_complex) {
        count               = sim_field_element_count(&input_field->layout);
        size_t output_count = sim_field_element_count(&output_field->layout);
        if (count != output_count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else {
        if (input_field->element_size != sizeof(double) ||
            output_field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count               = sim_field_bytes(input_field) / sizeof(double);
        size_t output_count = sim_field_bytes(output_field) / sizeof(double);
        if (count != output_count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    SimResult ensure_result = sieve_ensure_scratch(state, count, is_complex);
    if (ensure_result != SIM_RESULT_OK) {
        return ensure_result;
    }

    const double*                 kernel = state->kernel;
    unsigned int                  taps   = state->kernel_size;
    unsigned int                  radius = taps / 2U;
    const SimSieveOperatorConfig* cfg    = &state->config;
    const double                  scale  = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    bool subtract_from_input = sieve_mode_is_high_pass_like(cfg->mode);

    if (is_complex) {
        /* Complex field path */
        SimComplexDouble* input_data  = sim_field_complex_data(input_field);
        SimComplexDouble* output_data = sim_field_complex_data(output_field);
        if (!input_data || !output_data) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        /* Apply convolution to complex data */
        for (size_t i = 0U; i < count; ++i) {
            double accum_re = 0.0;
            double accum_im = 0.0;
            for (unsigned int k = 0U; k < taps; ++k) {
                ptrdiff_t offset = (ptrdiff_t) i + (ptrdiff_t) k - (ptrdiff_t) radius;
                bool      valid  = false;
                size_t    index  = sieve_resolve_index(offset, count, cfg->boundary, &valid);
                if (!valid) {
                    continue;
                }
                accum_re += kernel[k] * input_data[index].re;
                accum_im += kernel[k] * input_data[index].im;
            }
            state->scratch_complex[i].re = accum_re;
            state->scratch_complex[i].im = accum_im;
        }

        /* Apply mode combination and gain */
        for (size_t i = 0U; i < count; ++i) {
            double original_re = input_data[i].re;
            double original_im = input_data[i].im;
            double filtered_re = state->scratch_complex[i].re;
            double filtered_im = state->scratch_complex[i].im;

            if (subtract_from_input) {
                filtered_re = original_re - filtered_re;
                filtered_im = original_im - filtered_im;
            }

            double value_re = filtered_re * cfg->gain;
            double value_im = filtered_im * cfg->gain;

            if (!isfinite(value_re) || !isfinite(value_im)) {
                continue;
            }

            if (cfg->accumulate) {
                output_data[i].re += scale * value_re;
                output_data[i].im += scale * value_im;
            } else {
                output_data[i].re = value_re;
                output_data[i].im = value_im;
            }
        }
    } else {
        /* Real field path (original code) */
        double* input_data  = (double*) sim_field_data(input_field);
        double* output_data = (double*) sim_field_data(output_field);
        if (!input_data || !output_data) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double accum = 0.0;
            for (unsigned int k = 0U; k < taps; ++k) {
                ptrdiff_t offset = (ptrdiff_t) i + (ptrdiff_t) k - (ptrdiff_t) radius;
                bool      valid  = false;
                size_t    index  = sieve_resolve_index(offset, count, cfg->boundary, &valid);
                if (!valid) {
                    continue;
                }
                accum += kernel[k] * input_data[index];
            }
            state->scratch[i] = accum;
        }

        for (size_t i = 0U; i < count; ++i) {
            double original = input_data[i];
            double filtered = state->scratch[i];
            if (subtract_from_input) {
                filtered = original - filtered;
            }
            double value = filtered * cfg->gain;

            if (!isfinite(value)) {
                continue;
            }

            if (cfg->accumulate) {
                output_data[i] += scale * value;
            } else {
                output_data[i] = value;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult sieve_step(void*               state_ptr,
                            struct SimContext*  context,
                            struct SimOperator* self,
                            size_t              substep_index,
                            double              dt_sub,
                            void*               scratch,
                            size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return sieve_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_sieve_operator(struct SimContext*            context,
                                 const SimSieveOperatorConfig* config,
                                 size_t*                       out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimSieveOperatorState* state =
        (SimSieveOperatorState*) calloc(1U, sizeof(SimSieveOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimSieveOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.input_field      = 0U;
        local.output_field     = 0U;
        local.taps             = 5U;
        local.sigma            = 1.0;
        local.sigma2           = 2.0;
        local.poly_order       = 3U;
        local.derivative_order = 1U;
        local.sample_spacing   = 1.0;
        local.window_alpha     = 0.0;
        local.gain             = 1.0;
        local.mode             = SIM_SIEVE_MODE_LOW_PASS;
        local.boundary         = SIM_IR_BOUNDARY_NEUMANN;
        local.accumulate       = false;
        local.scale_by_dt      = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "sieve", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    sieve_normalize_config(&local, 0U, false);
    state->config = local;

    SimResult kernel_result = sieve_rebuild_kernel(state);
    if (kernel_result != SIM_RESULT_OK) {
        sieve_destroy(state);
        return kernel_result;
    }

    sieve_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "sieve");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_MEASUREMENT;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = false;
    info.is_nonlocal       = true;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "sieve";
    sim_operator_info_set_schema_identity(&info, "sieve");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;

    SimField* input_field  = sim_context_field(context, state->config.input_field);
    SimField* output_field = sim_context_field(context, state->config.output_field);
    if (input_field == NULL || output_field == NULL) {
        sieve_destroy(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_operator_field_domain_is_f64_or_c64(input_field) ||
        !sim_operator_field_domain_is_f64_or_c64(output_field)) {
        sieve_destroy(state);
        return SIM_RESULT_TYPE_MISMATCH;
    }
    bool needs_complex =
        (input_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(input_field))) ||
        (output_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(output_field)));
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;
    info.representation.boundary                        = local.boundary;

    SimOperatorConfig op_config = sim_operator_config_defaults();
    op_config.boundary          = local.boundary;

    SimSplitPort ports[2] = {
        { .context_field_index = state->config.input_field, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = sieve_step,
                                .accesses          = accesses,
                                .access_count      = 2U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 2U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = sieve_symbolic,
                                .destroy       = sieve_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        sieve_destroy(state);
    }

    return result;
}

SimResult sim_sieve_config(struct SimContext*      context,
                           size_t                  operator_index,
                           SimSieveOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSieveOperatorState* state = (SimSieveOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_sieve_update(struct SimContext*            context,
                           size_t                        operator_index,
                           const SimSieveOperatorConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSieveOperatorState* state = (SimSieveOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimSieveOperatorConfig local = state->config;
    if (config) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "sieve", true, config->scale_by_dt);
    }

    sieve_normalize_config(&local, state->config.taps, true);
    state->config = local;

    SimResult kernel_result = sieve_rebuild_kernel(state);
    if (kernel_result != SIM_RESULT_OK) {
        return kernel_result;
    }

    sieve_refresh_symbolic(state);
    {
        SimOperatorConfig op_config = op->config;
        op_config.boundary          = local.boundary;
        sim_operator_config_set(op, &op_config);
    }
    op->info.representation.boundary = local.boundary;
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
