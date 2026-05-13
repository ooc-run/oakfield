/**
 * @file laplacian.h
 * @brief Finite-difference Laplacian operator for 1D/2D fields.
 */
#ifndef OAKFIELD_LAPLACIAN_H
#define OAKFIELD_LAPLACIAN_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

#define SIM_LAPLACIAN_AXIS_AUTO ((size_t)SIZE_MAX)

/**
 * @brief Finite-difference stencils available for Laplacian evaluation.
 */
typedef enum SimLaplacianStencil {
    SIM_LAPLACIAN_STENCIL_CROSS_2 = 0, /**< Second-order cross stencil. */
    SIM_LAPLACIAN_STENCIL_CROSS_4,     /**< Fourth-order cross stencil. */
    SIM_LAPLACIAN_STENCIL_ISOTROPIC_9  /**< Nine-point isotropic stencil. */
} SimLaplacianStencil;

/**
 * @brief Configuration for a finite-difference Laplacian operator.
 */
typedef struct SimLaplacianOperatorConfig {
    size_t input_field;           /**< Source field index. */
    size_t output_field;          /**< Field receiving the Laplacian. */
    double spacing_x;             /**< Grid spacing along axis_x; non-positive values normalize. */
    double spacing_y;             /**< Grid spacing along axis_y; non-positive values normalize. */
    size_t axis_x;                /**< First derivative axis, or SIM_LAPLACIAN_AXIS_AUTO. */
    size_t axis_y;                /**< Second derivative axis, or SIM_LAPLACIAN_AXIS_AUTO. */
    SimLaplacianStencil stencil;  /**< Finite-difference stencil family. */
    SimIRBoundaryPolicy boundary; /**< Boundary policy for out-of-range neighbors. */
    bool accumulate;              /**< Add into the output when true. */
    bool scale_by_dt;             /**< Scale writes by substep dt when true. */
} SimLaplacianOperatorConfig;

/**
 * @brief Register a finite-difference Laplacian operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Laplacian configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_laplacian_operator(struct SimContext *context,
                                     const SimLaplacianOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current Laplacian configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_laplacian_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no Laplacian state.
 */
SimResult sim_laplacian_config(struct SimContext *context, size_t operator_index,
                               SimLaplacianOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered Laplacian operator.
 *
 * @p config is required. The replacement is normalized and field compatibility
 * is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Laplacian operator to update.
 * @param config Replacement Laplacian configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, state validation, or field compatibility checks.
 */
SimResult sim_laplacian_update(struct SimContext *context, size_t operator_index,
                               const SimLaplacianOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_LAPLACIAN_H */
