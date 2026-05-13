/**
 * @file remainder.h
 * @brief Remainder operator measuring f(warped) - f(reference) with optional accumulation.
 *
 * Supports nonlinear transforms (abs, log_abs, power, tanh, identity) applied prior to differencing
 * and works with real or complex fields (component-wise for complex). Weight and bias are applied
 * after computing the residue. Accumulation optionally adds into the existing output field.
 *
 * Complex field support: component-wise or polar mode (magnitude residue along warped phase).
 */
#ifndef OAKFIELD_REMAINDER_H
#define OAKFIELD_REMAINDER_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Available nonlinearities applied before differencing.
 */
typedef enum SimRemainderNonlinearity {
    SIM_REMAINDER_NONLINEARITY_IDENTITY = 0, /**< Identity transform. */
    SIM_REMAINDER_NONLINEARITY_ABS,          /**< Absolute-value transform. */
    SIM_REMAINDER_NONLINEARITY_LOG_ABS,      /**< Log-magnitude transform. */
    SIM_REMAINDER_NONLINEARITY_POWER,        /**< Power-law transform. */
    SIM_REMAINDER_NONLINEARITY_TANH          /**< Hyperbolic tangent transform. */
} SimRemainderNonlinearity;

/**
 * @brief Complex processing mode for remainder operator.
 */
typedef enum SimRemainderComplexMode {
    SIM_REMAINDER_COMPLEX_MODE_COMPONENT =
        0, /**< Process real and imaginary parts independently. */
    SIM_REMAINDER_COMPLEX_MODE_POLAR =
        1 /**< Magnitude residue written along warped phase direction. */
} SimRemainderComplexMode;

/**
 * @brief Configuration for the remainder operator.
 */
typedef struct SimRemainderOperatorConfig {
    size_t warped_field;    /**< Field containing the warped signal. */
    size_t reference_field; /**< Field containing the reference signal u. */
    size_t output_field;    /**< Field receiving f(warped) - f(reference). */
    double weight;          /**< Gain applied to the remainder. */
    double bias;            /**< Constant offset added after weighting. */
    double exponent;        /**< Exponent used by POWER nonlinearity. */
    double epsilon;         /**< Positive guard preventing singularities in LOG_ABS/POWER. */
    SimRemainderNonlinearity nonlinearity; /**< Analytic function f applied to inputs. */
    bool accumulate;  /**< When true, adds into output instead of overwriting. */
    bool scale_by_dt; /**< When true, scale accumulated writes by substep dt. */
    SimRemainderComplexMode complex_mode; /**< Complex processing mode (component-wise or polar). */
} SimRemainderOperatorConfig;

/**
 * @brief Register a remainder operator that measures f(warped) - f(reference).
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional remainder configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field/type compatibility checks, allocation, kernel registration, or
 *         split registration.
 */
SimResult sim_add_remainder_operator(struct SimContext *context,
                                     const SimRemainderOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current remainder configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_remainder_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no remainder state.
 */
SimResult sim_remainder_config(struct SimContext *context, size_t operator_index,
                               SimRemainderOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered remainder configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic metadata and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the remainder operator to update.
 * @param config Optional replacement remainder configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for a NULL
 *         context, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no remainder state.
 */
SimResult sim_remainder_update(struct SimContext *context, size_t operator_index,
                               const SimRemainderOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_REMAINDER_H */
