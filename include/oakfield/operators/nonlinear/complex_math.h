/**
 * @file complex_math.h
 * @brief Elementwise complex math operator.
 */
#ifndef OAKFIELD_COMPLEX_MATH_H
#define OAKFIELD_COMPLEX_MATH_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Elementwise complex math operation.
 */
typedef enum SimComplexMathMode {
    SIM_COMPLEX_MATH_ADD = 0, /**< Complex addition. */
    SIM_COMPLEX_MATH_SUB,     /**< Complex subtraction. */
    SIM_COMPLEX_MATH_MUL,     /**< Complex multiplication. */
    SIM_COMPLEX_MATH_DIV,     /**< Complex division. */
    SIM_COMPLEX_MATH_POW,     /**< Complex power. */
    SIM_COMPLEX_MATH_EXP,     /**< Complex exponential. */
    SIM_COMPLEX_MATH_LOG,     /**< Principal complex logarithm. */
    SIM_COMPLEX_MATH_SIN,     /**< Complex sine. */
    SIM_COMPLEX_MATH_COS,     /**< Complex cosine. */
    SIM_COMPLEX_MATH_TAN,     /**< Complex tangent. */
    SIM_COMPLEX_MATH_SINH,    /**< Complex hyperbolic sine. */
    SIM_COMPLEX_MATH_COSH,    /**< Complex hyperbolic cosine. */
    SIM_COMPLEX_MATH_TANH,    /**< Complex hyperbolic tangent. */
    SIM_COMPLEX_MATH_SQRT,    /**< Principal complex square root. */
    SIM_COMPLEX_MATH_CONJ,    /**< Complex conjugate. */
    SIM_COMPLEX_MATH_ABS,     /**< Complex magnitude. */
    SIM_COMPLEX_MATH_ARG,     /**< Complex phase angle. */
    SIM_COMPLEX_MATH_REAL,    /**< Real component extraction. */
    SIM_COMPLEX_MATH_IMAG,    /**< Imaginary component extraction. */
    SIM_COMPLEX_MATH_NEG      /**< Complex negation. */
} SimComplexMathMode;

/**
 * @brief Right-hand side source for binary ops.
 */
typedef enum SimComplexMathRhsSource {
    SIM_COMPLEX_MATH_RHS_FIELD = 0, /**< Read RHS values from rhs_field. */
    SIM_COMPLEX_MATH_RHS_CONSTANT   /**< Use rhs_constant_* as RHS values. */
} SimComplexMathRhsSource;

/**
 * @brief Output component selection when writing into real fields.
 */
typedef enum SimComplexMathOutputComponent {
    SIM_COMPLEX_MATH_OUTPUT_REAL = 0,  /**< Write real component. */
    SIM_COMPLEX_MATH_OUTPUT_IMAG,      /**< Write imaginary component. */
    SIM_COMPLEX_MATH_OUTPUT_MAGNITUDE, /**< Write magnitude. */
    SIM_COMPLEX_MATH_OUTPUT_PHASE      /**< Write phase angle. */
} SimComplexMathOutputComponent;

/**
 * @brief Phase wrap policy for phase outputs.
 */
typedef enum SimComplexMathPhaseWrap {
    SIM_COMPLEX_MATH_PHASE_NONE = 0, /**< Leave phase unwrapped. */
    SIM_COMPLEX_MATH_PHASE_SIGNED,   /**< Wrap phase to a signed angular interval. */
    SIM_COMPLEX_MATH_PHASE_UNSIGNED, /**< Wrap phase to an unsigned angular interval. */
    SIM_COMPLEX_MATH_PHASE_UNIT      /**< Normalize phase to the unit interval. */
} SimComplexMathPhaseWrap;

/**
 * @brief Configuration for the complex math operator.
 */
typedef struct SimComplexMathOperatorConfig {
    size_t lhs_field;                               /**< LHS field index. */
    size_t rhs_field;                               /**< RHS field index (ignored for constants). */
    size_t output_field;                            /**< Output field index. */
    SimComplexMathMode mode;                        /**< Complex math operation. */
    SimComplexMathRhsSource rhs_source;             /**< RHS source selection. */
    double rhs_constant_re;                         /**< RHS constant real component. */
    double rhs_constant_im;                         /**< RHS constant imaginary component. */
    double lhs_scale_re;                            /**< LHS pre-scale real component. */
    double lhs_scale_im;                            /**< LHS pre-scale imaginary component. */
    double rhs_scale_re;                            /**< RHS pre-scale real component. */
    double rhs_scale_im;                            /**< RHS pre-scale imaginary component. */
    double bias_re;                                 /**< Bias real component. */
    double bias_im;                                 /**< Bias imaginary component. */
    double epsilon;                                 /**< Guard for log/division singularities. */
    SimComplexMathOutputComponent output_component; /**< Real output selection. */
    SimComplexMathPhaseWrap phase_wrap;             /**< Phase wrap policy. */
    bool accumulate;                                /**< Add into output when true. */
    bool scale_by_dt;                               /**< Scale accumulated writes by substep dt. */
} SimComplexMathOperatorConfig;

/**
 * @brief Register a complex math operator with the provided configuration.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional complex math configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, or split registration.
 */
SimResult sim_add_complex_math_operator(struct SimContext *context,
                                        const SimComplexMathOperatorConfig *config,
                                        size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a complex math operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_complex_math_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no complex-math state.
 */
SimResult sim_complex_math_config(struct SimContext *context, size_t operator_index,
                                  SimComplexMathOperatorConfig *out_config);

/**
 * @brief Update an existing complex math operator in-place.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic metadata.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the complex math operator to update.
 * @param config Optional replacement complex math configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for a NULL
 *         context, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no complex-math state.
 */
SimResult sim_complex_math_update(struct SimContext *context, size_t operator_index,
                                  const SimComplexMathOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_COMPLEX_MATH_H */
