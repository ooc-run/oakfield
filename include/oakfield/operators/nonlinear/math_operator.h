/**
 * @file math_operator.h
 * @brief Elementwise math operator with discrete-friendly operations.
 */
#ifndef OAKFIELD_MATH_OPERATOR_H
#define OAKFIELD_MATH_OPERATOR_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Elementwise math operations.
 */
typedef enum SimElementwiseMathMode {
    SIM_ELEMENTWISE_MATH_FLOOR = 0, /**< Floor operation. */
    SIM_ELEMENTWISE_MATH_FRACT,     /**< Fractional-part operation. */
    SIM_ELEMENTWISE_MATH_MOD,       /**< Modulo operation. */
    SIM_ELEMENTWISE_MATH_STEP,      /**< Threshold step operation. */
    SIM_ELEMENTWISE_MATH_EQ,        /**< Equality comparison. */
    SIM_ELEMENTWISE_MATH_LT,        /**< Less-than comparison. */
    SIM_ELEMENTWISE_MATH_GT,        /**< Greater-than comparison. */
    SIM_ELEMENTWISE_MATH_SELECT     /**< Select true/false output based on condition. */
} SimElementwiseMathMode;

/**
 * @brief RHS source for binary operations.
 */
typedef enum SimElementwiseMathRhsSource {
    SIM_ELEMENTWISE_MATH_RHS_FIELD = 0, /**< Read RHS values from rhs_field. */
    SIM_ELEMENTWISE_MATH_RHS_CONSTANT   /**< Use rhs_constant as RHS value. */
} SimElementwiseMathRhsSource;

/**
 * @brief Configuration parameters for the elementwise math operator.
 */
typedef struct SimElementwiseMathOperatorConfig {
    size_t lhs_field;                       /**< Primary input field. */
    size_t rhs_field;                       /**< Secondary input field (optional). */
    size_t output_field;                    /**< Field receiving the result. */
    SimElementwiseMathMode mode;            /**< Operation selector. */
    SimElementwiseMathRhsSource rhs_source; /**< RHS source for binary ops/select. */
    double rhs_constant;                    /**< Constant RHS when rhs_source=constant. */
    double threshold;                       /**< Threshold for step/select. */
    double epsilon;                         /**< Equality tolerance for eq comparisons. */
    double true_value;                      /**< Value emitted when condition is true. */
    double false_value;                     /**< Value emitted when condition is false. */
    bool accumulate;                        /**< Add into output when true. */
    bool scale_by_dt;                       /**< Scale accumulated writes by substep dt. */
} SimElementwiseMathOperatorConfig;

/**
 * @brief Register an elementwise math operator with the provided configuration.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional elementwise math configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field/type compatibility checks, allocation, or split registration.
 */
SimResult sim_add_elementwise_math_operator(struct SimContext *context,
                                            const SimElementwiseMathOperatorConfig *config,
                                            size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a math operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_elementwise_math_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no math state.
 */
SimResult sim_elementwise_math_config(struct SimContext *context, size_t operator_index,
                                      SimElementwiseMathOperatorConfig *out_config);

/**
 * @brief Update an existing math operator in-place.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update validates field/type compatibility,
 * refreshes symbolic metadata, and invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the elementwise math operator to update.
 * @param config Optional replacement elementwise math configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field/type compatibility checks, or state validation.
 */
SimResult sim_elementwise_math_update(struct SimContext *context, size_t operator_index,
                                      const SimElementwiseMathOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_OPERATOR_H */
