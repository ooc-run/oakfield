/**
 * @file divergence.h
 * @brief Finite-difference divergence operator for two vector components.
 */
#ifndef OAKFIELD_DIVERGENCE_H
#define OAKFIELD_DIVERGENCE_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

#define SIM_DIVERGENCE_AXIS_AUTO ((size_t)SIZE_MAX)

/**
 * @brief Finite-difference stencils available for divergence derivatives.
 */
typedef enum SimDivergenceStencil {
    SIM_DIVERGENCE_STENCIL_CENTRAL_2 = 0, /**< Second-order central difference. */
    SIM_DIVERGENCE_STENCIL_CENTRAL_4,     /**< Fourth-order central difference. */
    SIM_DIVERGENCE_STENCIL_FORWARD_1,     /**< First-order forward difference. */
    SIM_DIVERGENCE_STENCIL_BACKWARD_1,    /**< First-order backward difference. */
    SIM_DIVERGENCE_STENCIL_FORWARD_2,     /**< Second-order forward difference. */
    SIM_DIVERGENCE_STENCIL_BACKWARD_2     /**< Second-order backward difference. */
} SimDivergenceStencil;

/**
 * @brief Configuration for a finite-difference vector divergence operator.
 */
typedef struct SimDivergenceOperatorConfig {
    size_t input_field_x;         /**< Field supplying the X component. */
    size_t input_field_y;         /**< Field supplying the Y component. */
    size_t output_field;          /**< Field receiving dX/dx + dY/dy. */
    double spacing_x;             /**< Grid spacing along axis_x; non-positive values normalize. */
    double spacing_y;             /**< Grid spacing along axis_y; non-positive values normalize. */
    size_t axis_x;                /**< Axis used for X derivatives, or SIM_DIVERGENCE_AXIS_AUTO. */
    size_t axis_y;                /**< Axis used for Y derivatives, or SIM_DIVERGENCE_AXIS_AUTO. */
    SimDivergenceStencil stencil; /**< Finite-difference stencil. */
    SimIRBoundaryPolicy boundary; /**< Boundary policy for out-of-range neighbors. */
    bool accumulate;              /**< Add into the output when true. */
    bool scale_by_dt;             /**< Scale writes by substep dt when true. */
} SimDivergenceOperatorConfig;

/**
 * @brief Register a finite-difference divergence operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional divergence configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_divergence_operator(struct SimContext *context,
                                      const SimDivergenceOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current divergence configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_divergence_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no divergence state.
 */
SimResult sim_divergence_config(struct SimContext *context, size_t operator_index,
                                SimDivergenceOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered divergence operator.
 *
 * @p config is required. The replacement is normalized and field compatibility
 * is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the divergence operator to update.
 * @param config Replacement divergence configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, state validation, or field compatibility checks.
 */
SimResult sim_divergence_update(struct SimContext *context, size_t operator_index,
                                const SimDivergenceOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_DIVERGENCE_H */
