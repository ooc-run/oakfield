/**
 * @file fft_plan.h
 * @brief Reusable 1D and 2D complex FFT plans for operator implementations.
 *
 * Power-of-two sizes use a radix-2 plan, while non-power-of-two 1D sizes use
 * Bluestein convolution over an internal power-of-two plan. The 2D plan applies
 * separable row and column transforms using caller-provided strides.
 */
#ifndef OAKFIELD_FFT_PLAN_H
#define OAKFIELD_FFT_PLAN_H

#include "oakfield/field.h" /* for SimResult */
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reusable radix-2 complex FFT plan for power-of-two transform lengths.
 *
 * An initialized plan owns its bit-reversal table, twiddle storage, and optional
 * vDSP scratch/setup resources until fft_pow2_plan_destroy() is called.
 */
typedef struct FFTPow2Plan {
    size_t n;                       /**< Transform length. */
    size_t levels;                  /**< Number of radix-2 stages. */
    size_t *bit_reverse;            /**< Owned bit-reversal index table. */
    double complex *stage_twiddles; /**< Owned per-stage twiddle-factor table. */
#if defined(SIM_HAVE_VDSP)
    FFTSetupD setup;  /**< Owned vDSP FFT setup handle. */
    double *tmp_real; /**< Owned temporary real lane for vDSP conversion. */
    double *tmp_imag; /**< Owned temporary imaginary lane for vDSP conversion. */
#endif
} FFTPow2Plan;

/**
 * @brief Reusable 1D complex FFT plan for arbitrary nonzero transform lengths.
 *
 * Power-of-two lengths use @ref FFTPow2Plan directly; other lengths own the
 * Bluestein chirp/convolution scratch storage needed by fft_plan_forward() and
 * fft_plan_inverse().
 */
typedef struct FFTPlan {
    size_t n;                   /**< Transform length. */
    bool use_bluestein;         /**< True when non-power-of-two convolution path is active. */
    FFTPow2Plan pow2_plan;      /**< Direct radix-2 plan for power-of-two lengths. */
    FFTPow2Plan bluestein_plan; /**< Convolution FFT plan for Bluestein transforms. */
    size_t bluestein_size;      /**< Power-of-two convolution length for Bluestein path. */
    double complex *chirp;      /**< Owned Bluestein chirp factors. */
    double complex *chirp_conj; /**< Owned conjugated chirp factors. */
    double complex *b_fft;      /**< Owned FFT of the Bluestein chirp kernel. */
    double complex *a_buffer;   /**< Owned Bluestein input work buffer. */
    double complex *scratch_n;  /**< Owned output-length scratch buffer. */
} FFTPlan;

/**
 * @brief Separable 2D complex FFT plan built from row and column 1D plans.
 *
 * The plan owns its nested row/column plans and temporary row/column scratch
 * buffers until fft_plan2d_destroy() is called.
 */
typedef struct FFTPlan2D {
    size_t rows;                 /**< Number of rows in the logical transform. */
    size_t cols;                 /**< Number of columns in the logical transform. */
    size_t row_stride;           /**< Element stride between adjacent rows. */
    size_t col_stride;           /**< Element stride between adjacent columns. */
    size_t scratch_len;          /**< Element capacity of row/column scratch buffers. */
    FFTPlan row_plan;            /**< Reusable row transform plan. */
    FFTPlan col_plan;            /**< Reusable column transform plan. */
    double complex *scratch_in;  /**< Owned temporary input line buffer. */
    double complex *scratch_out; /**< Owned temporary output line buffer. */
} FFTPlan2D;

/**
 * @brief Reset a power-of-two FFT plan to an empty non-owning state.
 *
 * @param plan Plan to reset; NULL is ignored.
 */
void fft_pow2_plan_reset(FFTPow2Plan *plan);

/**
 * @brief Free storage owned by a power-of-two FFT plan and reset it.
 *
 * @param plan Plan to destroy; NULL is ignored.
 */
void fft_pow2_plan_destroy(FFTPow2Plan *plan);

/**
 * @brief Initialize a radix-2 FFT plan.
 *
 * @param plan Plan object to initialize.
 * @param n Transform length; must be nonzero and a power of two.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for invalid
 *         inputs, or #SIM_RESULT_OUT_OF_MEMORY on allocation/setup failure.
 */
SimResult fft_pow2_plan_init(FFTPow2Plan *plan, size_t n);

/**
 * @brief Execute an initialized radix-2 complex FFT in place.
 *
 * @param plan Initialized power-of-two plan.
 * @param data Complex buffer with plan->n entries; transformed in place.
 * @param inverse When true, compute the normalized inverse transform.
 */
void fft_pow2_execute(const FFTPow2Plan *plan, double complex *data, bool inverse);

/**
 * @brief Reset a 1D FFT plan to an empty non-owning state.
 *
 * @param plan Plan to reset; NULL is ignored.
 */
void fft_plan_reset(FFTPlan *plan);

/**
 * @brief Free storage owned by a 1D FFT plan and reset it.
 *
 * @param plan Plan to destroy; NULL is ignored.
 */
void fft_plan_destroy(FFTPlan *plan);

/**
 * @brief Initialize a 1D complex FFT plan for any nonzero length.
 *
 * @param plan Plan object to initialize.
 * @param n Transform length; must be greater than zero.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for invalid
 *         inputs, or #SIM_RESULT_OUT_OF_MEMORY on allocation failure.
 */
SimResult fft_plan_init(FFTPlan *plan, size_t n);

/**
 * @brief Execute a 1D forward complex FFT.
 *
 * @param plan Initialized 1D FFT plan.
 * @param input Input buffer with plan->n entries.
 * @param[out] output Output buffer with plan->n entries.
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_INVALID_ARGUMENT for NULL inputs.
 */
SimResult fft_plan_forward(const FFTPlan *plan, const double complex *input,
                           double complex *output);

/**
 * @brief Execute a normalized 1D inverse complex FFT.
 *
 * @param plan Initialized 1D FFT plan.
 * @param input Input buffer with plan->n entries.
 * @param[out] output Output buffer with plan->n entries.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         inputs, or #SIM_RESULT_INVALID_STATE for an incomplete plan.
 */
SimResult fft_plan_inverse(const FFTPlan *plan, const double complex *input,
                           double complex *output);

/**
 * @brief Reset a 2D FFT plan to an empty non-owning state.
 *
 * @param plan Plan to reset; NULL is ignored.
 */
void fft_plan2d_reset(FFTPlan2D *plan);

/**
 * @brief Free storage owned by a 2D FFT plan and reset it.
 *
 * @param plan Plan to destroy; NULL is ignored.
 */
void fft_plan2d_destroy(FFTPlan2D *plan);

/**
 * @brief Initialize a separable 2D complex FFT plan.
 *
 * @param plan Plan object to initialize.
 * @param rows Number of rows; must be greater than zero.
 * @param cols Number of columns; must be greater than zero.
 * @param row_stride Element stride between adjacent rows.
 * @param col_stride Element stride between adjacent columns.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for invalid
 *         inputs, or #SIM_RESULT_OUT_OF_MEMORY on allocation failure.
 */
SimResult fft_plan2d_init(FFTPlan2D *plan, size_t rows, size_t cols, size_t row_stride,
                          size_t col_stride);

/**
 * @brief Execute a separable 2D forward complex FFT.
 *
 * @param plan Initialized 2D FFT plan.
 * @param input Input buffer addressed by the plan strides.
 * @param[out] output Output buffer addressed by the plan strides.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL or
 *         invalid plan data, or #SIM_RESULT_INVALID_STATE for missing scratch storage.
 */
SimResult fft_plan2d_forward(const FFTPlan2D *plan, const double complex *input,
                             double complex *output);

/**
 * @brief Execute a normalized separable 2D inverse complex FFT.
 *
 * @param plan Initialized 2D FFT plan.
 * @param input Input buffer addressed by the plan strides.
 * @param[out] output Output buffer addressed by the plan strides.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL or
 *         invalid plan data, or #SIM_RESULT_INVALID_STATE for missing scratch storage.
 */
SimResult fft_plan2d_inverse(const FFTPlan2D *plan, const double complex *input,
                             double complex *output);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_FFT_PLAN_H */
