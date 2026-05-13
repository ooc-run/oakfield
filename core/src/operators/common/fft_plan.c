#include "oakfield/operators/common/fft_plan.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static bool fft_is_power_of_two(size_t n) {
    return n > 0U && (n & (n - 1U)) == 0U;
}

static size_t fft_next_power_of_two(size_t n) {
    size_t value = 1U;
    while (value < n) {
        value <<= 1U;
    }
    return value;
}

static size_t fft_reverse_bits(size_t value, size_t bits) {
    size_t reversed = 0U;
    for (size_t i = 0U; i < bits; ++i) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

void fft_pow2_plan_reset(FFTPow2Plan* plan) {
    if (!plan) {
        return;
    }
    plan->n              = 0U;
    plan->levels         = 0U;
    plan->bit_reverse    = NULL;
    plan->stage_twiddles = NULL;
#if defined(SIM_HAVE_VDSP)
    plan->setup    = NULL;
    plan->tmp_real = NULL;
    plan->tmp_imag = NULL;
#endif
}

void fft_pow2_plan_destroy(FFTPow2Plan* plan) {
    if (!plan) {
        return;
    }
#if defined(SIM_HAVE_VDSP)
    if (plan->setup) {
        vDSP_destroy_fftsetupD(plan->setup);
        plan->setup = NULL;
    }
    free(plan->tmp_real);
    free(plan->tmp_imag);
#endif
    free(plan->bit_reverse);
    free(plan->stage_twiddles);
    fft_pow2_plan_reset(plan);
}

SimResult fft_pow2_plan_init(FFTPow2Plan* plan, size_t n) {
    size_t          levels         = 0U;
    size_t*         bit_reverse    = NULL;
    double complex* stage_twiddles = NULL;

    if (!plan || n == 0U || !fft_is_power_of_two(n)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

#if defined(SIM_HAVE_VDSP)
    {
        vDSP_Length log2n = 0;
        size_t      tmp   = n;
        while (tmp > 1U) {
            log2n += 1;
            tmp >>= 1U;
        }
        FFTSetupD setup = vDSP_create_fftsetupD(log2n, kFFTRadix2);
        if (setup == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        plan->setup  = setup;
        plan->levels = (size_t) log2n;
        if (posix_memalign((void**) &plan->tmp_real, 64U, n * sizeof(double)) != 0 ||
            posix_memalign((void**) &plan->tmp_imag, 64U, n * sizeof(double)) != 0) {
            fft_pow2_plan_destroy(plan);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        plan->n = n;
        return SIM_RESULT_OK;
    }
#else
    while ((n >> levels) > 1U) {
        levels += 1U;
    }

    bit_reverse = (size_t*) calloc(n, sizeof(size_t));
    if (!bit_reverse) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    stage_twiddles = (double complex*) calloc(levels, sizeof(double complex));
    if (!stage_twiddles) {
        free(bit_reverse);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    for (size_t i = 0U; i < n; ++i) {
        bit_reverse[i] = fft_reverse_bits(i, levels);
    }

    for (size_t level = 0U; level < levels; ++level) {
        size_t m              = 1U << (level + 1U);
        double theta          = -2.0 * M_PI / (double) m;
        stage_twiddles[level] = CMPLX(cos(theta), sin(theta));
    }

    plan->n              = n;
    plan->levels         = levels;
    plan->bit_reverse    = bit_reverse;
    plan->stage_twiddles = stage_twiddles;
    return SIM_RESULT_OK;
#endif
}

void fft_pow2_execute(const FFTPow2Plan* plan, double complex* data, bool inverse) {
    size_t n;

    if (!plan || !data) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    DSPDoubleSplitComplex split = { .realp = plan->tmp_real, .imagp = plan->tmp_imag };
    vDSP_ctozD((const DSPDoubleComplex*) data, 2, &split, 1, plan->n);
    vDSP_fft_zipD(
        plan->setup, &split, 1, (vDSP_Length) plan->levels, inverse ? FFT_INVERSE : FFT_FORWARD);
    if (inverse) {
        double scale = 1.0 / (double) plan->n;
        vDSP_vsmulD(split.realp, 1, &scale, split.realp, 1, plan->n);
        vDSP_vsmulD(split.imagp, 1, &scale, split.imagp, 1, plan->n);
    }
    vDSP_ztocD(&split, 1, (DSPDoubleComplex*) data, 2, plan->n);
    return;
#endif

    n = plan->n;
    for (size_t i = 0U; i < n; ++i) {
        size_t j = plan->bit_reverse[i];
        if (j > i) {
            double complex temp = data[i];
            data[i]             = data[j];
            data[j]             = temp;
        }
    }

    for (size_t level = 0U; level < plan->levels; ++level) {
        size_t         m    = 1U << (level + 1U);
        size_t         half = m >> 1U;
        double complex w_m  = plan->stage_twiddles[level];
        if (inverse) {
            w_m = conj(w_m);
        }

        for (size_t k = 0U; k < n; k += m) {
            double complex w = CMPLX(1.0, 0.0);
            for (size_t j = 0U; j < half; ++j) {
                size_t         index  = k + j;
                size_t         offset = index + half;
                double complex t      = w * data[offset];
                double complex u      = data[index];
                data[index]           = u + t;
                data[offset]          = u - t;
                w *= w_m;
            }
        }
    }

    if (inverse) {
        double scale = 1.0 / (double) n;
        for (size_t i = 0U; i < n; ++i) {
            data[i] *= scale;
        }
    }
}

void fft_plan_reset(FFTPlan* plan) {
    if (!plan) {
        return;
    }
    plan->n              = 0U;
    plan->use_bluestein  = false;
    plan->bluestein_size = 0U;
    plan->chirp          = NULL;
    plan->chirp_conj     = NULL;
    plan->b_fft          = NULL;
    plan->a_buffer       = NULL;
    plan->scratch_n      = NULL;
    fft_pow2_plan_reset(&plan->pow2_plan);
    fft_pow2_plan_reset(&plan->bluestein_plan);
}

void fft_plan_destroy(FFTPlan* plan) {
    if (!plan) {
        return;
    }
    fft_pow2_plan_destroy(&plan->pow2_plan);
    fft_pow2_plan_destroy(&plan->bluestein_plan);
    free(plan->chirp);
    free(plan->chirp_conj);
    free(plan->b_fft);
    free(plan->a_buffer);
    free(plan->scratch_n);
    fft_plan_reset(plan);
}

SimResult fft_plan_init(FFTPlan* plan, size_t n) {
    SimResult result = SIM_RESULT_OK;

    if (!plan || n == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    fft_plan_reset(plan);
    plan->n = n;

    if (fft_is_power_of_two(n)) {
        plan->use_bluestein = false;
        result              = fft_pow2_plan_init(&plan->pow2_plan, n);
        if (result != SIM_RESULT_OK) {
            fft_plan_destroy(plan);
            return result;
        }
        plan->scratch_n = (double complex*) calloc(n, sizeof(double complex));
        if (!plan->scratch_n) {
            fft_plan_destroy(plan);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        return SIM_RESULT_OK;
    }

    plan->use_bluestein  = true;
    plan->bluestein_size = fft_next_power_of_two(2U * n - 1U);

    result = fft_pow2_plan_init(&plan->bluestein_plan, plan->bluestein_size);
    if (result != SIM_RESULT_OK) {
        fft_plan_destroy(plan);
        return result;
    }

    plan->chirp      = (double complex*) calloc(n, sizeof(double complex));
    plan->chirp_conj = (double complex*) calloc(n, sizeof(double complex));
    plan->b_fft      = (double complex*) calloc(plan->bluestein_size, sizeof(double complex));
    plan->a_buffer   = (double complex*) calloc(plan->bluestein_size, sizeof(double complex));
    plan->scratch_n  = (double complex*) calloc(n, sizeof(double complex));

    if (!plan->chirp || !plan->chirp_conj || !plan->b_fft || !plan->a_buffer || !plan->scratch_n) {
        fft_plan_destroy(plan);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    for (size_t i = 0U; i < n; ++i) {
        double         angle = M_PI * (double) (i * i) / (double) n;
        double complex value = CMPLX(cos(angle), sin(angle));
        plan->chirp[i]       = value;
        plan->chirp_conj[i]  = conj(value);
    }

    for (size_t i = 0U; i < plan->bluestein_size; ++i) {
        plan->b_fft[i] = CMPLX(0.0, 0.0);
    }

    for (size_t i = 0U; i < n; ++i) {
        plan->b_fft[i] = plan->chirp[i];
    }
    for (size_t i = 1U; i < n; ++i) {
        plan->b_fft[plan->bluestein_size - i] = plan->chirp[i];
    }

    fft_pow2_execute(&plan->bluestein_plan, plan->b_fft, false);
    return SIM_RESULT_OK;
}

SimResult
fft_plan_forward(const FFTPlan* plan, const double complex* input, double complex* output) {
    if (!plan || !input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!plan->use_bluestein) {
        (void) memcpy(output, input, plan->n * sizeof(double complex));
        fft_pow2_execute(&plan->pow2_plan, output, false);
        return SIM_RESULT_OK;
    }

    double complex* buffer = plan->a_buffer;
    size_t          n      = plan->n;
    size_t          m      = plan->bluestein_size;

    for (size_t i = 0U; i < n; ++i) {
        buffer[i] = input[i] * plan->chirp_conj[i];
    }
    for (size_t i = n; i < m; ++i) {
        buffer[i] = CMPLX(0.0, 0.0);
    }

    fft_pow2_execute(&plan->bluestein_plan, buffer, false);

    for (size_t i = 0U; i < m; ++i) {
        buffer[i] *= plan->b_fft[i];
    }

    fft_pow2_execute(&plan->bluestein_plan, buffer, true);

    for (size_t i = 0U; i < n; ++i) {
        output[i] = buffer[i] * plan->chirp_conj[i];
    }

    return SIM_RESULT_OK;
}

SimResult
fft_plan_inverse(const FFTPlan* plan, const double complex* input, double complex* output) {
    double complex* scratch;
    SimResult       result;

    if (!plan || !input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    scratch = plan->scratch_n;
    if (!scratch) {
        return SIM_RESULT_INVALID_STATE;
    }

    for (size_t i = 0U; i < plan->n; ++i) {
        scratch[i] = conj(input[i]);
    }

    result = fft_plan_forward(plan, scratch, output);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    for (size_t i = 0U; i < plan->n; ++i) {
        output[i] = conj(output[i]) / (double) plan->n;
    }

    return SIM_RESULT_OK;
}

void fft_plan2d_reset(FFTPlan2D* plan) {
    if (!plan) {
        return;
    }
    plan->rows        = 0U;
    plan->cols        = 0U;
    plan->row_stride  = 0U;
    plan->col_stride  = 0U;
    plan->scratch_len = 0U;
    plan->scratch_in  = NULL;
    plan->scratch_out = NULL;
    fft_plan_reset(&plan->row_plan);
    fft_plan_reset(&plan->col_plan);
}

void fft_plan2d_destroy(FFTPlan2D* plan) {
    if (!plan) {
        return;
    }
    fft_plan_destroy(&plan->row_plan);
    fft_plan_destroy(&plan->col_plan);
    free(plan->scratch_in);
    free(plan->scratch_out);
    fft_plan2d_reset(plan);
}

SimResult
fft_plan2d_init(FFTPlan2D* plan, size_t rows, size_t cols, size_t row_stride, size_t col_stride) {
    if (!plan || rows == 0U || cols == 0U || row_stride == 0U || col_stride == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    fft_plan2d_reset(plan);
    plan->rows       = rows;
    plan->cols       = cols;
    plan->row_stride = row_stride;
    plan->col_stride = col_stride;

    SimResult rc = fft_plan_init(&plan->row_plan, cols);
    if (rc != SIM_RESULT_OK) {
        fft_plan2d_destroy(plan);
        return rc;
    }
    rc = fft_plan_init(&plan->col_plan, rows);
    if (rc != SIM_RESULT_OK) {
        fft_plan2d_destroy(plan);
        return rc;
    }

    plan->scratch_len = (rows > cols) ? rows : cols;
    plan->scratch_in  = (double complex*) calloc(plan->scratch_len, sizeof(double complex));
    plan->scratch_out = (double complex*) calloc(plan->scratch_len, sizeof(double complex));
    if (!plan->scratch_in || !plan->scratch_out) {
        fft_plan2d_destroy(plan);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    return SIM_RESULT_OK;
}

static SimResult fft_plan2d_execute(const FFTPlan2D*      plan,
                                    const double complex* input,
                                    double complex*       output,
                                    bool                  inverse) {
    if (!plan || !input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (plan->rows == 0U || plan->cols == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!plan->scratch_in || !plan->scratch_out) {
        return SIM_RESULT_INVALID_STATE;
    }

    const size_t rows       = plan->rows;
    const size_t cols       = plan->cols;
    const size_t row_stride = plan->row_stride;
    const size_t col_stride = plan->col_stride;
    SimResult    rc         = SIM_RESULT_OK;

    for (size_t row = 0U; row < rows; ++row) {
        const size_t base = row * row_stride;
        if (col_stride == 1U) {
            const double complex* row_in  = input + base;
            double complex*       row_out = output + base;
            rc = inverse ? fft_plan_inverse(&plan->row_plan, row_in, row_out)
                         : fft_plan_forward(&plan->row_plan, row_in, row_out);
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
        } else {
            for (size_t col = 0U; col < cols; ++col) {
                plan->scratch_in[col] = input[base + col * col_stride];
            }
            rc = inverse ? fft_plan_inverse(&plan->row_plan, plan->scratch_in, plan->scratch_out)
                         : fft_plan_forward(&plan->row_plan, plan->scratch_in, plan->scratch_out);
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
            for (size_t col = 0U; col < cols; ++col) {
                output[base + col * col_stride] = plan->scratch_out[col];
            }
        }
    }

    for (size_t col = 0U; col < cols; ++col) {
        const size_t base = col * col_stride;
        for (size_t row = 0U; row < rows; ++row) {
            plan->scratch_in[row] = output[row * row_stride + base];
        }
        rc = inverse ? fft_plan_inverse(&plan->col_plan, plan->scratch_in, plan->scratch_out)
                     : fft_plan_forward(&plan->col_plan, plan->scratch_in, plan->scratch_out);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
        for (size_t row = 0U; row < rows; ++row) {
            output[row * row_stride + base] = plan->scratch_out[row];
        }
    }

    return SIM_RESULT_OK;
}

SimResult
fft_plan2d_forward(const FFTPlan2D* plan, const double complex* input, double complex* output) {
    return fft_plan2d_execute(plan, input, output, false);
}

SimResult
fft_plan2d_inverse(const FFTPlan2D* plan, const double complex* input, double complex* output) {
    return fft_plan2d_execute(plan, input, output, true);
}
