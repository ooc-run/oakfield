/**
 * @file spatial_derivative.h
 * @brief 1D spatial derivative operator (finite differences with periodic boundary).
 */
#ifndef OAKFIELD_SPATIAL_DERIVATIVE_H
#define OAKFIELD_SPATIAL_DERIVATIVE_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Finite-difference stencil used by the spatial derivative operator.
 */
typedef enum SimSpatialDerivativeMethod {
    SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL = 0, /**< Central difference stencil. */
    SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD = 1, /**< Forward difference stencil. */
    SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD = 2 /**< Backward difference stencil. */
} SimSpatialDerivativeMethod;

/**
 * @brief Configuration for the spatial derivative operator.
 */
typedef struct SimSpatialDerivativeOperatorConfig {
    size_t input_field;                /**< Source field index. */
    size_t output_field;               /**< Destination field index. */
    double spacing;                    /**< Grid spacing dx (> 0). */
    SimSpatialDerivativeMethod method; /**< Finite difference stencil. */
    size_t axis;                       /**< Axis along which derivative is taken (0=x). */
    bool skew_forward; /**< When true, bias to a forward stencil instead of symmetric central. */
    bool accumulate;   /**< When true, adds into output instead of overwriting. */
    SimIRBoundaryPolicy boundary; /**< Boundary handling policy. */
} SimSpatialDerivativeOperatorConfig;

/**
 * @brief Return the schema name for a spatial-derivative stencil.
 *
 * @param method Stencil enum value.
 * @return Stable lowercase method name, or "central" for unknown values.
 */
const char *sim_spatial_derivative_method_name(SimSpatialDerivativeMethod method);

/**
 * @brief Parse a spatial-derivative stencil name.
 *
 * @param name Lowercase schema name such as "central", "forward", or "backward".
 * @param[out] out_method Receives the parsed method on success.
 * @return true when @p name maps to a known method; false otherwise.
 */
bool sim_spatial_derivative_method_from_string(const char *name,
                                               SimSpatialDerivativeMethod *out_method);

/**
 * @brief Register a 1D spatial derivative operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional derivative configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_spatial_derivative_operator(struct SimContext *context,
                                              const SimSpatialDerivativeOperatorConfig *config,
                                              size_t *out_index);

/**
 * @brief Copy the current spatial-derivative configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_spatial_derivative_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no derivative state.
 */
SimResult sim_spatial_derivative_config(struct SimContext *context, size_t operator_index,
                                        SimSpatialDerivativeOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered spatial-derivative configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the spatial-derivative operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         compatibility checks, or state validation.
 */
SimResult sim_spatial_derivative_update(struct SimContext *context, size_t operator_index,
                                        const SimSpatialDerivativeOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SPATIAL_DERIVATIVE_H */
