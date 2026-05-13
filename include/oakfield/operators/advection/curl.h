/**
 * @file curl.h
 * @brief Finite-difference curl-like scalar operator for two vector components.
 */
#ifndef OAKFIELD_CURL_H
#define OAKFIELD_CURL_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

#define SIM_CURL_AXIS_AUTO ((size_t)SIZE_MAX)

/**
 * @brief Finite-difference stencils available for curl derivatives.
 */
typedef enum SimCurlStencil {
    SIM_CURL_STENCIL_CENTRAL_2 = 0, /**< Second-order central difference. */
    SIM_CURL_STENCIL_CENTRAL_4,     /**< Fourth-order central difference. */
    SIM_CURL_STENCIL_FORWARD_1,     /**< First-order forward difference. */
    SIM_CURL_STENCIL_BACKWARD_1,    /**< First-order backward difference. */
    SIM_CURL_STENCIL_FORWARD_2,     /**< Second-order forward difference. */
    SIM_CURL_STENCIL_BACKWARD_2     /**< Second-order backward difference. */
} SimCurlStencil;

/**
 * @brief Configuration for a finite-difference scalar curl operator.
 */
typedef struct SimCurlOperatorConfig {
    size_t input_field_x;         /**< Field supplying the X component. */
    size_t input_field_y;         /**< Field supplying the Y component. */
    size_t output_field;          /**< Field receiving dY/dx - dX/dy. */
    double spacing_x;             /**< Grid spacing along axis_x; non-positive values normalize. */
    double spacing_y;             /**< Grid spacing along axis_y; non-positive values normalize. */
    size_t axis_x;                /**< Axis used for X derivatives, or SIM_CURL_AXIS_AUTO. */
    size_t axis_y;                /**< Axis used for Y derivatives, or SIM_CURL_AXIS_AUTO. */
    SimCurlStencil stencil;       /**< Finite-difference stencil. */
    SimIRBoundaryPolicy boundary; /**< Boundary policy for out-of-range neighbors. */
    bool accumulate;              /**< Add into the output when true. */
    bool scale_by_dt;             /**< Scale writes by substep dt when true. */
} SimCurlOperatorConfig;

/**
 * @brief Register a finite-difference curl operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional curl configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_curl_operator(struct SimContext *context, const SimCurlOperatorConfig *config,
                                size_t *out_index);

/**
 * @brief Copy the current curl configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_curl_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no curl state.
 */
SimResult sim_curl_config(struct SimContext *context, size_t operator_index,
                          SimCurlOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered curl operator.
 *
 * @p config is required. The replacement is normalized and field compatibility
 * is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the curl operator to update.
 * @param config Replacement curl configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, state validation, or field compatibility checks.
 */
SimResult sim_curl_update(struct SimContext *context, size_t operator_index,
                          const SimCurlOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_CURL_H */
