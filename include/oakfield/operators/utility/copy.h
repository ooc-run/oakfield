/**
 * @file copy.h
 * @brief Utility operator that copies one field into another compatible field.
 *
 * The copy operator supports real double, complex double, and exact-integer
 * fields. Exact integers are copied byte-for-byte and do not support accumulation
 * or dt scaling.
 */
#ifndef OAKFIELD_COPY_H
#define OAKFIELD_COPY_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for copying one compatible field into another.
 */
typedef struct SimCopyOperatorConfig {
    size_t input_field;  /**< Source field index. */
    size_t output_field; /**< Destination field index; layout and scalar domain must match input. */
    bool accumulate;     /**< Add into the destination when true; otherwise overwrite it. */
    bool scale_by_dt;    /**< Scale copied values by max(dt, 0) when true. */
} SimCopyOperatorConfig;

/**
 * @brief Register a field copy utility operator.
 *
 * The implementation copies and normalizes @p config, resolves the default
 * scale-by-dt policy, and validates that source and destination fields have the
 * same element count, element size, and scalar domain.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional copy configuration; NULL selects zero-initialized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context
 *         or invalid fields, #SIM_RESULT_TYPE_MISMATCH for incompatible domains,
 *         #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or a registration error.
 */
SimResult sim_add_copy_operator(struct SimContext *context, const SimCopyOperatorConfig *config,
                                size_t *out_index);

/**
 * @brief Copy the current configuration from a registered copy operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_copy_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no copy state.
 */
SimResult sim_copy_config(struct SimContext *context, size_t operator_index,
                          SimCopyOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered copy operator.
 *
 * @p config is required. The replacement is normalized and the referenced fields
 * are validated before it is stored.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the copy operator to update.
 * @param config Replacement copy configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL inputs
 *         or invalid fields, #SIM_RESULT_NOT_FOUND for a missing operator,
 *         #SIM_RESULT_INVALID_STATE for missing state, or #SIM_RESULT_TYPE_MISMATCH
 *         for incompatible field domains.
 */
SimResult sim_copy_update(struct SimContext *context, size_t operator_index,
                          const SimCopyOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_COPY_H */
