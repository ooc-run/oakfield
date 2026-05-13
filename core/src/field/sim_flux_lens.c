#include "oakfield/sim_flux_lens.h"

#include "oakfield/sim_context.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void flux_lens_clamp_band(double* lo, double* hi, double min_width, double max_k);

static bool flux_lens_add_size(size_t a, size_t b, size_t* out) {
    if (!out) {
        return false;
    }
    if (a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool flux_lens_mul_size(size_t a, size_t b, size_t* out) {
    if (!out) {
        return false;
    }
    if (a != 0U && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool flux_lens_state_bytes(size_t length, size_t* out_bytes) {
    size_t bytes = 0U;
    size_t block = 0U;

    if (!flux_lens_mul_size(length, sizeof(double complex), &block)) {
        return false;
    }
    if (!flux_lens_add_size(bytes, block, &bytes) || !flux_lens_add_size(bytes, block, &bytes)) {
        return false;
    }

    if (!flux_lens_mul_size(length, sizeof(double), &block)) {
        return false;
    }
    for (size_t i = 0U; i < 6U; ++i) {
        if (!flux_lens_add_size(bytes, block, &bytes)) {
            return false;
        }
    }

    *out_bytes = bytes;
    return true;
}

static bool flux_lens_workspace_bytes(size_t length, size_t* out_bytes) {
    size_t bytes = 0U;
    size_t block = 0U;

    if (!flux_lens_mul_size(length, sizeof(double complex), &block)) {
        return false;
    }
    if (!flux_lens_add_size(bytes, block, &bytes) || !flux_lens_add_size(bytes, block, &bytes)) {
        return false;
    }

    if (!flux_lens_mul_size(length, sizeof(double), &block)) {
        return false;
    }
    if (!flux_lens_add_size(bytes, block, &bytes)) {
        return false;
    }

    *out_bytes = bytes;
    return true;
}

static double flux_lens_frequency_bin(size_t index, size_t length) {
    ptrdiff_t signed_k =
        (index <= length / 2U) ? (ptrdiff_t) index : (ptrdiff_t) index - (ptrdiff_t) length;
    return (double) signed_k;
}

void flux_lens_init(FluxLensState* lens) {
    if (!lens) {
        return;
    }

    lens->enabled            = false;
    lens->locked             = false;
    lens->force_update       = false;
    lens->band_ready         = false;
    lens->field_index        = 0U;
    lens->smoothing          = 0.01;
    lens->min_bandwidth      = 60.0;
    lens->update_period      = 1U;
    lens->last_update_step   = 0U;
    lens->band_lo            = 0.0;
    lens->band_hi            = 0.0;
    lens->target_band_lo     = 0.0;
    lens->target_band_hi     = 0.0;
    lens->band_max_component = 0.0;
    lens->band_max_magnitude = 0.0;
    lens->max_k              = 0.0;
    lens->band_spec          = NULL;
    lens->band_phys          = NULL;
    lens->band_capacity      = 0U;
    lens->S                  = NULL;
    lens->Pi                 = NULL;
    lens->bucket_k           = NULL;
    lens->bucket_pi          = NULL;
    lens->bucket_S           = NULL;
    lens->bucket_absS        = NULL;
    lens->bucket_capacity    = 0U;
    lens->bucket_count       = 0U;
    lens->pi_min             = 0.0;
    lens->pi_max             = 0.0;
    lens->absS_total         = 0.0;
    memset(&lens->marks, 0, sizeof(lens->marks));
    lens->marks.valid   = false;
    lens->scratch_bytes = 0U;
}

void flux_lens_force_refresh(FluxLensState* lens) {
    if (!lens) {
        return;
    }
    lens->force_update = true;
}

void flux_lens_set_field_index(FluxLensState* lens, size_t field_index) {
    if (!lens) {
        return;
    }
    if (lens->field_index == field_index) {
        return;
    }
    lens->field_index  = field_index;
    lens->force_update = true;
    lens->band_ready   = false;
}

void flux_lens_scale_width(FluxLensState* lens, double scale) {
    if (!lens || !(scale > 0.0) || !isfinite(scale)) {
        return;
    }

    double lo = lens->band_lo;
    double hi = lens->band_hi;
    if (!(hi > lo)) {
        lo = lens->target_band_lo;
        hi = lens->target_band_hi;
    }
    if (!(hi > lo)) {
        lo = 0.0;
        hi = fmax(lens->min_bandwidth, 1.0);
    }

    double center        = 0.5 * (lo + hi);
    double width         = (hi - lo) * scale;
    lens->target_band_lo = center - 0.5 * width;
    lens->target_band_hi = center + 0.5 * width;
    flux_lens_clamp_band(
        &lens->target_band_lo, &lens->target_band_hi, lens->min_bandwidth, lens->max_k);
    if (lens->locked) {
        lens->band_lo = lens->target_band_lo;
        lens->band_hi = lens->target_band_hi;
    }
    lens->force_update = true;
}

void flux_lens_shift_center(FluxLensState* lens, double shift) {
    if (!lens || !isfinite(shift)) {
        return;
    }

    double lo = lens->band_lo;
    double hi = lens->band_hi;
    if (!(hi > lo)) {
        lo = lens->target_band_lo;
        hi = lens->target_band_hi;
    }
    if (!(hi > lo)) {
        lo = 0.0;
        hi = fmax(lens->min_bandwidth, 1.0);
    }

    double width         = hi - lo;
    double center        = 0.5 * (lo + hi) + shift;
    lens->target_band_lo = center - 0.5 * width;
    lens->target_band_hi = center + 0.5 * width;
    flux_lens_clamp_band(
        &lens->target_band_lo, &lens->target_band_hi, lens->min_bandwidth, lens->max_k);
    if (lens->locked) {
        lens->band_lo = lens->target_band_lo;
        lens->band_hi = lens->target_band_hi;
    }
    lens->force_update = true;
}

void flux_lens_set_band(FluxLensState* lens, double band_lo, double band_hi) {
    if (!lens) {
        return;
    }

    if (!isfinite(band_lo)) {
        band_lo = 0.0;
    }
    if (!isfinite(band_hi)) {
        band_hi = band_lo + lens->min_bandwidth;
    }
    if (band_hi < band_lo) {
        double tmp = band_hi;
        band_hi    = band_lo;
        band_lo    = tmp;
    }

    double max_k = lens->max_k;
    if (!(max_k > 0.0)) {
        max_k = fmax(band_hi, band_lo + lens->min_bandwidth);
    }

    flux_lens_clamp_band(&band_lo, &band_hi, lens->min_bandwidth, max_k);

    lens->band_lo        = band_lo;
    lens->band_hi        = band_hi;
    lens->target_band_lo = band_lo;
    lens->target_band_hi = band_hi;
    lens->band_ready     = false;
    lens->force_update   = true;
}

void flux_lens_workspace_init(FluxLensWorkspace* workspace) {
    if (!workspace) {
        return;
    }

    workspace->tmp_spec1   = NULL;
    workspace->tmp_phys1   = NULL;
    workspace->tmp_real1   = NULL;
    workspace->use_dealias = false;
    fft_plan_reset(&workspace->plan);
    workspace->capacity      = 0U;
    workspace->plan_ready    = false;
    workspace->scratch_bytes = 0U;
}

void flux_release_scratch(FluxLensWorkspace* workspace, SimContext* context) {
    if (!workspace) {
        return;
    }

    if (context && workspace->scratch_bytes > 0U) {
        sim_context_release_scratch(context, workspace->scratch_bytes);
    }

    fft_plan_destroy(&workspace->plan);
    free(workspace->tmp_spec1);
    free(workspace->tmp_phys1);
    free(workspace->tmp_real1);

    workspace->tmp_spec1     = NULL;
    workspace->tmp_phys1     = NULL;
    workspace->tmp_real1     = NULL;
    workspace->capacity      = 0U;
    workspace->plan_ready    = false;
    workspace->scratch_bytes = 0U;
}

static void flux_lens_release_state_buffers(FluxLensState* lens, SimContext* context) {
    if (!lens) {
        return;
    }

    if (context && lens->scratch_bytes > 0U) {
        sim_context_release_scratch(context, lens->scratch_bytes);
    }

    free(lens->band_spec);
    free(lens->band_phys);
    free(lens->S);
    free(lens->Pi);
    free(lens->bucket_k);
    free(lens->bucket_pi);
    free(lens->bucket_S);
    free(lens->bucket_absS);

    lens->band_spec       = NULL;
    lens->band_phys       = NULL;
    lens->S               = NULL;
    lens->Pi              = NULL;
    lens->bucket_k        = NULL;
    lens->bucket_pi       = NULL;
    lens->bucket_S        = NULL;
    lens->bucket_absS     = NULL;
    lens->band_capacity   = 0U;
    lens->bucket_capacity = 0U;
    lens->bucket_count    = 0U;
    lens->band_ready      = false;
    lens->force_update    = true;
    lens->max_k           = 0.0;
    lens->pi_min          = 0.0;
    lens->pi_max          = 0.0;
    lens->absS_total      = 0.0;
    lens->marks.valid     = false;
    lens->scratch_bytes   = 0U;
}

void flux_lens_release(FluxLensState* lens, FluxLensWorkspace* workspace, SimContext* context) {
    flux_lens_release_state_buffers(lens, context);
    flux_release_scratch(workspace, context);
}

static SimResult flux_lens_resize_state(FluxLensState* lens, SimContext* context, size_t length) {
    if (!lens) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (length == 0U) {
        flux_lens_release_state_buffers(lens, context);
        return SIM_RESULT_OK;
    }

    bool ready = (lens->band_capacity == length) && (lens->bucket_capacity == length) &&
                 lens->band_spec && lens->band_phys && lens->S && lens->Pi && lens->bucket_k &&
                 lens->bucket_pi && lens->bucket_S && lens->bucket_absS;
    lens->max_k = (double) (length / 2U);
    if (ready) {
        return SIM_RESULT_OK;
    }

    double complex* new_band_spec   = (double complex*) calloc(length, sizeof(double complex));
    double complex* new_band_phys   = (double complex*) calloc(length, sizeof(double complex));
    double*         new_S           = (double*) calloc(length, sizeof(double));
    double*         new_Pi          = (double*) calloc(length, sizeof(double));
    double*         new_bucket_k    = (double*) calloc(length, sizeof(double));
    double*         new_bucket_pi   = (double*) calloc(length, sizeof(double));
    double*         new_bucket_S    = (double*) calloc(length, sizeof(double));
    double*         new_bucket_absS = (double*) calloc(length, sizeof(double));

    if (!new_band_spec || !new_band_phys || !new_S || !new_Pi || !new_bucket_k || !new_bucket_pi ||
        !new_bucket_S || !new_bucket_absS) {
        free(new_band_spec);
        free(new_band_phys);
        free(new_S);
        free(new_Pi);
        free(new_bucket_k);
        free(new_bucket_pi);
        free(new_bucket_S);
        free(new_bucket_absS);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    size_t new_bytes = 0U;
    if (!flux_lens_state_bytes(length, &new_bytes)) {
        free(new_band_spec);
        free(new_band_phys);
        free(new_S);
        free(new_Pi);
        free(new_bucket_k);
        free(new_bucket_pi);
        free(new_bucket_S);
        free(new_bucket_absS);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (context && new_bytes > lens->scratch_bytes) {
        SimResult rc = sim_context_reserve_scratch(context, new_bytes - lens->scratch_bytes);
        if (rc != SIM_RESULT_OK) {
            free(new_band_spec);
            free(new_band_phys);
            free(new_S);
            free(new_Pi);
            free(new_bucket_k);
            free(new_bucket_pi);
            free(new_bucket_S);
            free(new_bucket_absS);
            return rc;
        }
    }

    free(lens->band_spec);
    free(lens->band_phys);
    free(lens->S);
    free(lens->Pi);
    free(lens->bucket_k);
    free(lens->bucket_pi);
    free(lens->bucket_S);
    free(lens->bucket_absS);

    if (context && new_bytes < lens->scratch_bytes) {
        sim_context_release_scratch(context, lens->scratch_bytes - new_bytes);
    }

    lens->band_spec       = new_band_spec;
    lens->band_phys       = new_band_phys;
    lens->S               = new_S;
    lens->Pi              = new_Pi;
    lens->bucket_k        = new_bucket_k;
    lens->bucket_pi       = new_bucket_pi;
    lens->bucket_S        = new_bucket_S;
    lens->bucket_absS     = new_bucket_absS;
    lens->band_capacity   = length;
    lens->bucket_capacity = length;
    lens->bucket_count    = 0U;
    lens->band_ready      = false;
    lens->force_update    = true;
    lens->marks.valid     = false;
    lens->scratch_bytes   = new_bytes;
    return SIM_RESULT_OK;
}

static SimResult
flux_lens_resize_workspace(FluxLensWorkspace* workspace, SimContext* context, size_t length) {
    if (!workspace) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (length == 0U) {
        flux_release_scratch(workspace, context);
        return SIM_RESULT_OK;
    }

    bool ready = (workspace->capacity == length) && workspace->plan_ready && workspace->tmp_spec1 &&
                 workspace->tmp_phys1 && workspace->tmp_real1;
    if (ready) {
        return SIM_RESULT_OK;
    }

    double complex* new_spec = (double complex*) calloc(length, sizeof(double complex));
    double complex* new_phys = (double complex*) calloc(length, sizeof(double complex));
    double*         new_real = (double*) calloc(length, sizeof(double));
    FFTPlan         new_plan;

    if (!new_spec || !new_phys || !new_real) {
        free(new_spec);
        free(new_phys);
        free(new_real);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    fft_plan_reset(&new_plan);
    SimResult plan_rc = fft_plan_init(&new_plan, length);
    if (plan_rc != SIM_RESULT_OK) {
        free(new_spec);
        free(new_phys);
        free(new_real);
        return plan_rc;
    }

    size_t new_bytes = 0U;
    if (!flux_lens_workspace_bytes(length, &new_bytes)) {
        fft_plan_destroy(&new_plan);
        free(new_spec);
        free(new_phys);
        free(new_real);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (context && new_bytes > workspace->scratch_bytes) {
        SimResult rc = sim_context_reserve_scratch(context, new_bytes - workspace->scratch_bytes);
        if (rc != SIM_RESULT_OK) {
            fft_plan_destroy(&new_plan);
            free(new_spec);
            free(new_phys);
            free(new_real);
            return rc;
        }
    }

    fft_plan_destroy(&workspace->plan);
    free(workspace->tmp_spec1);
    free(workspace->tmp_phys1);
    free(workspace->tmp_real1);

    if (context && new_bytes < workspace->scratch_bytes) {
        sim_context_release_scratch(context, workspace->scratch_bytes - new_bytes);
    }

    workspace->tmp_spec1     = new_spec;
    workspace->tmp_phys1     = new_phys;
    workspace->tmp_real1     = new_real;
    workspace->plan          = new_plan;
    workspace->capacity      = length;
    workspace->plan_ready    = true;
    workspace->scratch_bytes = new_bytes;
    return SIM_RESULT_OK;
}

SimResult flux_lens_ensure_capacity(FluxLensState*     lens,
                                    FluxLensWorkspace* workspace,
                                    SimContext*        context,
                                    size_t             length) {
    SimResult rc;

    if (!lens || !workspace) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rc = flux_lens_resize_state(lens, context, length);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    rc = flux_lens_resize_workspace(workspace, context, length);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    return SIM_RESULT_OK;
}

double flux_total_work(const double* S, size_t length) {
    if (!S || length == 0U) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0U; i < length; ++i) {
        sum += S[i];
    }
    return sum;
}

double flux_total_energy(const double complex* u_hat, size_t length) {
    if (!u_hat || length == 0U) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0U; i < length; ++i) {
        double re = creal(u_hat[i]);
        double im = cimag(u_hat[i]);
        sum += re * re + im * im;
    }
    return sum;
}

static void flux_lens_cumulative(const double* S, double* Pi, size_t length) {
    if (!S || !Pi || length == 0U) {
        return;
    }

    double sum  = 0.0;
    size_t half = length / 2U;
    for (size_t k = 0U; k <= half; ++k) {
        size_t idx_pos = k;
        bool   has_neg = !(k == 0U || (length % 2U == 0U && k == half));
        size_t idx_neg = has_neg ? (length - k) : 0U;

        sum += S[idx_pos];
        if (has_neg) {
            sum += S[idx_neg];
        }

        Pi[idx_pos] = sum;
        if (has_neg) {
            Pi[idx_neg] = sum;
        }
    }
}

static void flux_lens_clamp_band(double* lo, double* hi, double min_width, double max_k) {
    if (!lo || !hi) {
        return;
    }

    double band_lo = *lo;
    double band_hi = *hi;

    if (band_lo < 0.0) {
        band_lo = 0.0;
    }
    if (band_hi > max_k) {
        band_hi = max_k;
    }

    double width = band_hi - band_lo;
    if (width < min_width) {
        double center = 0.5 * (band_lo + band_hi);
        double half   = 0.5 * min_width;
        band_lo       = center - half;
        band_hi       = center + half;

        if (band_lo < 0.0) {
            band_hi -= band_lo;
            band_lo = 0.0;
        }
        if (band_hi > max_k) {
            double over = band_hi - max_k;
            band_lo -= over;
            band_hi = max_k;
            if (band_lo < 0.0) {
                band_lo = 0.0;
            }
        }
    }

    *lo = band_lo;
    *hi = band_hi;
}

bool flux_compute(const SimField*    field,
                  FluxLensState*     lens,
                  FluxLensWorkspace* workspace,
                  SimContext*        context) {
    if (!field || !lens || !workspace) {
        return false;
    }

    lens->marks.valid = false;

    if (field->layout.rank != 1U) {
        return false;
    }

    size_t length = sim_field_element_count(&field->layout);
    if (length == 0U) {
        return false;
    }

    SimResult rc = flux_lens_ensure_capacity(lens, workspace, context, length);
    if (rc != SIM_RESULT_OK) {
        return false;
    }

    SimFieldRepresentation repr        = sim_field_representation(field);
    bool                   is_spectral = (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL);
    bool                   is_complex  = sim_field_is_complex(field);
    const double*          data        = NULL;
    if (!is_complex) {
        data = sim_field_real_data_const(field);
        if (!data) {
            return false;
        }
    }

    if (is_spectral) {
        if (is_complex) {
            const SimComplexDouble* src = sim_field_complex_data_const(field);
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_spec1[i] = CMPLX(src[i].re, src[i].im);
            }
        } else {
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_spec1[i] = CMPLX(data[i], 0.0);
            }
        }
    } else {
        if (is_complex) {
            const SimComplexDouble* src = sim_field_complex_data_const(field);
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_phys1[i] = CMPLX(src[i].re, src[i].im);
            }
        } else {
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_phys1[i] = CMPLX(data[i], 0.0);
            }
        }

        rc = fft_plan_forward(&workspace->plan, workspace->tmp_phys1, workspace->tmp_spec1);
        if (rc != SIM_RESULT_OK) {
            return false;
        }
    }

    if (workspace->use_dealias) {
        double max_k  = (lens->max_k > 0.0) ? lens->max_k : (double) (length / 2U);
        double cutoff = (2.0 / 3.0) * max_k;
        for (size_t i = 0U; i < length; ++i) {
            double k = flux_lens_frequency_bin(i, length);
            if (fabs(k) > cutoff) {
                workspace->tmp_spec1[i] = CMPLX(0.0, 0.0);
            }
        }
    }

    if (is_spectral || workspace->use_dealias) {
        rc = fft_plan_inverse(&workspace->plan, workspace->tmp_spec1, workspace->tmp_phys1);
        if (rc != SIM_RESULT_OK) {
            return false;
        }
    }

    for (size_t i = 0U; i < length; ++i) {
        double complex u        = workspace->tmp_phys1[i];
        workspace->tmp_phys1[i] = u * u;
    }

    rc = fft_plan_forward(&workspace->plan, workspace->tmp_phys1, workspace->tmp_phys1);
    if (rc != SIM_RESULT_OK) {
        return false;
    }

    for (size_t i = 0U; i < length; ++i) {
        double         k     = flux_lens_frequency_bin(i, length);
        double complex n_hat = (-0.5 * I * k) * workspace->tmp_phys1[i];
        double complex prod  = conj(workspace->tmp_spec1[i]) * n_hat;
        lens->S[i]           = creal(prod);
    }

    flux_lens_cumulative(lens->S, lens->Pi, length);
    lens->marks.valid = true;
    return true;
}

void flux_lens_after_flux(FluxLensState* lens) {
    if (!lens) {
        return;
    }

    size_t length = lens->band_capacity;
    if (length == 0U || !lens->S || !lens->Pi || !lens->bucket_k || !lens->bucket_S ||
        !lens->bucket_absS || !lens->bucket_pi) {
        lens->marks.valid = false;
        return;
    }

    const double tol          = 1.0e-6;
    size_t       bucket_count = length / 2U + 1U;
    if (lens->bucket_capacity < bucket_count) {
        lens->marks.valid = false;
        return;
    }

    lens->bucket_count = bucket_count;
    for (size_t i = 0U; i < bucket_count; ++i) {
        lens->bucket_k[i]    = (double) i;
        lens->bucket_S[i]    = 0.0;
        lens->bucket_absS[i] = 0.0;
        lens->bucket_pi[i]   = 0.0;
    }

    for (size_t i = 0U; i < length; ++i) {
        double abs_k = fabs(flux_lens_frequency_bin(i, length));
        size_t idx   = (size_t) llround(abs_k);
        if (idx >= bucket_count || fabs(abs_k - (double) idx) > tol) {
            size_t match = bucket_count;
            for (size_t j = 0U; j < bucket_count; ++j) {
                if (fabs(lens->bucket_k[j] - abs_k) <= tol) {
                    match = j;
                    break;
                }
            }
            if (match == bucket_count) {
                continue;
            }
            idx = match;
        }

        double s = lens->S[i];
        lens->bucket_S[idx] += s;
        lens->bucket_absS[idx] += fabs(s);
        lens->bucket_pi[idx] = lens->Pi[i];
    }

    double pi_min     = 0.0;
    double pi_max     = 0.0;
    double absS_total = 0.0;
    double max_abs_pi = -1.0;
    size_t kc_index   = 0U;
    bool   have_pi    = false;

    for (size_t i = 0U; i < bucket_count; ++i) {
        double pi = lens->bucket_pi[i];
        if (!have_pi) {
            pi_min  = pi;
            pi_max  = pi;
            have_pi = true;
        } else {
            if (pi < pi_min) {
                pi_min = pi;
            }
            if (pi > pi_max) {
                pi_max = pi;
            }
        }

        double abs_pi = fabs(pi);
        if (abs_pi > max_abs_pi) {
            max_abs_pi = abs_pi;
            kc_index   = i;
        }

        absS_total += lens->bucket_absS[i];
    }

    double k50 = 0.0;
    double k90 = 0.0;
    if (absS_total > 0.0) {
        double target50 = 0.5 * absS_total;
        double target90 = 0.9 * absS_total;
        double running  = 0.0;
        bool   set50    = false;
        bool   set90    = false;
        for (size_t i = 0U; i < bucket_count; ++i) {
            running += lens->bucket_absS[i];
            if (!set50 && running >= target50) {
                k50   = lens->bucket_k[i];
                set50 = true;
            }
            if (!set90 && running >= target90) {
                k90   = lens->bucket_k[i];
                set90 = true;
                break;
            }
        }
        if (!set50) {
            k50 = lens->bucket_k[bucket_count - 1U];
        }
        if (!set90) {
            k90 = lens->bucket_k[bucket_count - 1U];
        }
    }

    lens->pi_min     = pi_min;
    lens->pi_max     = pi_max;
    lens->absS_total = absS_total;

    lens->marks.valid          = (bucket_count > 0U);
    lens->marks.kc             = lens->bucket_k[kc_index];
    lens->marks.k50            = k50;
    lens->marks.k90            = k90;
    lens->marks.pi_at_kc       = lens->bucket_pi[kc_index];
    lens->marks.total_abs_work = absS_total;
    lens->marks.pi_min         = pi_min;
    lens->marks.pi_max         = pi_max;
    lens->marks.kmax           = lens->bucket_k[bucket_count - 1U];
}

void flux_lens_update_targets_from_marks(FluxLensState* lens) {
    if (!lens || !lens->marks.valid) {
        return;
    }

    double width = lens->marks.k90 - lens->marks.k50;
    if (width < 0.0) {
        width = 0.0;
    }

    double padding      = fmax(1.0, 0.25 * width);
    double target_width = width + 2.0 * padding;
    if (target_width < lens->min_bandwidth) {
        target_width = lens->min_bandwidth;
    }

    double half          = 0.5 * target_width;
    lens->target_band_lo = lens->marks.kc - half;
    lens->target_band_hi = lens->marks.kc + half;

    flux_lens_clamp_band(
        &lens->target_band_lo, &lens->target_band_hi, lens->min_bandwidth, lens->max_k);
}

void flux_lens_relax_band(FluxLensState* lens) {
    if (!lens) {
        return;
    }

    double alpha = lens->smoothing;
    if (alpha < 0.0) {
        alpha = 0.0;
    }
    if (alpha > 1.0) {
        alpha = 1.0;
    }

    if (!lens->band_ready || !(lens->band_hi > lens->band_lo)) {
        lens->band_lo = lens->target_band_lo;
        lens->band_hi = lens->target_band_hi;
    } else {
        lens->band_lo += alpha * (lens->target_band_lo - lens->band_lo);
        lens->band_hi += alpha * (lens->target_band_hi - lens->band_hi);
    }

    flux_lens_clamp_band(&lens->band_lo, &lens->band_hi, lens->min_bandwidth, lens->max_k);
}

bool flux_lens_reconstruct(const SimField*    field,
                           FluxLensState*     lens,
                           FluxLensWorkspace* workspace) {
    if (!field || !lens || !workspace) {
        return false;
    }

    size_t length = sim_field_element_count(&field->layout);
    if (length == 0U || lens->band_capacity < length || workspace->capacity < length) {
        lens->band_ready = false;
        return false;
    }

    if (!(lens->band_hi > lens->band_lo)) {
        lens->band_ready = false;
        return false;
    }

    SimFieldRepresentation repr        = sim_field_representation(field);
    bool                   is_spectral = (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL);
    bool                   is_complex  = sim_field_is_complex(field);
    const double*          data        = NULL;
    if (!is_complex) {
        data = sim_field_real_data_const(field);
        if (!data) {
            lens->band_ready = false;
            return false;
        }
    }

    if (is_spectral) {
        if (is_complex) {
            const SimComplexDouble* src = sim_field_complex_data_const(field);
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_spec1[i] = CMPLX(src[i].re, src[i].im);
            }
        } else {
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_spec1[i] = CMPLX(data[i], 0.0);
            }
        }
    } else {
        if (is_complex) {
            const SimComplexDouble* src = sim_field_complex_data_const(field);
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_phys1[i] = CMPLX(src[i].re, src[i].im);
            }
        } else {
            for (size_t i = 0U; i < length; ++i) {
                workspace->tmp_phys1[i] = CMPLX(data[i], 0.0);
            }
        }

        if (fft_plan_forward(&workspace->plan, workspace->tmp_phys1, workspace->tmp_spec1) !=
            SIM_RESULT_OK) {
            lens->band_ready = false;
            return false;
        }
    }

    double width = lens->band_hi - lens->band_lo;
    double taper = fmax(1.0, 0.2 * width);

    for (size_t i = 0U; i < length; ++i) {
        double k      = fabs(flux_lens_frequency_bin(i, length));
        double weight = 0.0;

        if (k >= lens->band_lo && k <= lens->band_hi) {
            weight = 1.0;
        } else if (k > lens->band_lo - taper && k < lens->band_lo) {
            weight = (k - (lens->band_lo - taper)) / taper;
        } else if (k > lens->band_hi && k < lens->band_hi + taper) {
            weight = ((lens->band_hi + taper) - k) / taper;
        }

        if (weight < 0.0) {
            weight = 0.0;
        } else if (weight > 1.0) {
            weight = 1.0;
        }

        lens->band_spec[i] = workspace->tmp_spec1[i] * weight;
    }

    if (fft_plan_inverse(&workspace->plan, lens->band_spec, lens->band_phys) != SIM_RESULT_OK) {
        lens->band_ready = false;
        return false;
    }

    double max_component = 0.0;
    double max_magnitude = 0.0;
    for (size_t i = 0U; i < length; ++i) {
        double re   = creal(lens->band_phys[i]);
        double im   = cimag(lens->band_phys[i]);
        double comp = fmax(fabs(re), fabs(im));
        double mag  = hypot(re, im);
        if (comp > max_component) {
            max_component = comp;
        }
        if (mag > max_magnitude) {
            max_magnitude = mag;
        }
    }

    lens->band_max_component = max_component;
    lens->band_max_magnitude = max_magnitude;
    lens->band_ready         = true;
    return true;
}

void flux_lens_update(SimContext* context) {
    if (!context || sim_context_in_drift(context)) {
        return;
    }

    FluxLensState*     lens      = &context->runtime.flux_lens;
    FluxLensWorkspace* workspace = &context->runtime.flux_workspace;

    if (!lens->enabled) {
        flux_lens_release(lens, workspace, context);
        return;
    }

    SimField* field = sim_context_field(context, lens->field_index);
    if (!field) {
        lens->marks.valid = false;
        return;
    }

    size_t step_index = sim_context_step_index(context);
    bool   should_update =
        lens->force_update || (step_index >= lens->last_update_step + lens->update_period);
    if (!should_update) {
        return;
    }

    lens->force_update     = false;
    lens->last_update_step = step_index;

    if (!flux_compute(field, lens, workspace, context)) {
        lens->marks.valid = false;
        return;
    }

    flux_lens_after_flux(lens);
    if (!lens->marks.valid) {
        return;
    }

    flux_lens_update_targets_from_marks(lens);
    if (!lens->locked) {
        flux_lens_relax_band(lens);
    } else if (!(lens->band_hi > lens->band_lo)) {
        /* Locked mode still needs an initial valid band before reconstruction can run. */
        lens->band_lo = lens->target_band_lo;
        lens->band_hi = lens->target_band_hi;
        flux_lens_clamp_band(&lens->band_lo, &lens->band_hi, lens->min_bandwidth, lens->max_k);
    }
    flux_lens_reconstruct(field, lens, workspace);
}
