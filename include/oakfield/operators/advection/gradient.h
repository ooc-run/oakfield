/**
 * @file gradient.h
 * @brief Finite-difference gradient operator writing X/Y derivative fields.
 */
#ifndef OAKFIELD_GRADIENT_H
#define OAKFIELD_GRADIENT_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

#define SIM_GRADIENT_AXIS_AUTO ((size_t)SIZE_MAX)

/**
 * @brief Finite-difference stencils available for gradient derivatives.
 */
typedef enum SimGradientStencil {
    SIM_GRADIENT_STENCIL_CENTRAL_2 = 0, /**< Second-order central difference. */
    SIM_GRADIENT_STENCIL_CENTRAL_4,     /**< Fourth-order central difference. */
    SIM_GRADIENT_STENCIL_FORWARD_1,     /**< First-order forward difference. */
    SIM_GRADIENT_STENCIL_BACKWARD_1,    /**< First-order backward difference. */
    SIM_GRADIENT_STENCIL_FORWARD_2,     /**< Second-order forward difference. */
    SIM_GRADIENT_STENCIL_BACKWARD_2     /**< Second-order backward difference. */
} SimGradientStencil;

/**
 * @brief Configuration for writing finite-difference gradient components.
 */
typedef struct SimGradientOperatorConfig {
    size_t input_field;           /**< Source scalar field. */
    size_t output_field_x;        /**< Field receiving derivative along axis_x. */
    size_t output_field_y;        /**< Field receiving derivative along axis_y. */
    double spacing_x;             /**< Grid spacing along axis_x; non-positive values normalize. */
    double spacing_y;             /**< Grid spacing along axis_y; non-positive values normalize. */
    size_t axis_x;                /**< X derivative axis, or SIM_GRADIENT_AXIS_AUTO. */
    size_t axis_y;                /**< Y derivative axis, or SIM_GRADIENT_AXIS_AUTO. */
    SimGradientStencil stencil;   /**< Finite-difference stencil. */
    SimIRBoundaryPolicy boundary; /**< Boundary policy for out-of-range neighbors. */
    bool accumulate;              /**< Add into output fields when true. */
    bool scale_by_dt;             /**< Scale writes by substep dt when true. */
} SimGradientOperatorConfig;

/**
 * @brief Register a finite-difference gradient operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional gradient configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_gradient_operator(struct SimContext *context,
                                    const SimGradientOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current gradient configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_gradient_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no gradient state.
 */
SimResult sim_gradient_config(struct SimContext *context, size_t operator_index,
                              SimGradientOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered gradient operator.
 *
 * @p config is required. The replacement is normalized and field compatibility
 * is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the gradient operator to update.
 * @param config Replacement gradient configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, state validation, or field compatibility checks.
 */
SimResult sim_gradient_update(struct SimContext *context, size_t operator_index,
                              const SimGradientOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_GRADIENT_H */
