/**
 * @file zero_field.h
 * @brief Utility operator that clears a real or complex field in place.
 */
#ifndef OAKFIELD_ZERO_FIELD_H
#define OAKFIELD_ZERO_FIELD_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for clearing a target field in place.
 */
typedef struct ZeroFieldOperatorConfig {
    size_t field_index; /**< Field index to clear. */
} ZeroFieldOperatorConfig;

/**
 * @brief Register an in-place zero-field utility operator.
 *
 * The operator clears all elements of the configured field during its split step.
 * The target field must exist at registration time.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional zero-field configuration; NULL selects field index 0.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context
 *         or missing field, #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or a
 *         registration error.
 */
SimResult sim_add_zero_field_operator(struct SimContext *context,
                                      const ZeroFieldOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current configuration from a registered zero-field operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_zero_field_operator().
 * @param[out] out_config Receives the operator configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no zero-field state.
 */
SimResult sim_zero_field_config(struct SimContext *context, size_t operator_index,
                                ZeroFieldOperatorConfig *out_config);

/**
 * @brief Update a registered zero-field operator without retargeting it.
 *
 * @p config is required and its field_index must match the registered target.
 * A successful update refreshes symbolic state and invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the zero-field operator to update.
 * @param config Replacement configuration with the same field_index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL inputs
 *         or field retargeting, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no zero-field state.
 */
SimResult sim_zero_field_update(struct SimContext *context, size_t operator_index,
                                const ZeroFieldOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_ZERO_FIELD_H */
