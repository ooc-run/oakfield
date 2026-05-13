/**
 * @file scale.h
 * @brief Utility operator that writes a scaled copy of one field to another field.
 *
 * The scale operator supports real double and complex double fields with matching
 * layouts and scalar domains. It can overwrite or accumulate into the destination.
 */
#ifndef OAKFIELD_SCALE_H
#define OAKFIELD_SCALE_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for scaling one compatible field into another.
 */
typedef struct SimScaleOperatorConfig {
    size_t input_field;  /**< Source field index. */
    size_t output_field; /**< Destination field index; layout and scalar kind must match input. */
    double scale;        /**< Finite scalar multiplier; non-finite values normalize to 0. */
    bool accumulate;     /**< Add into the destination when true; otherwise overwrite it. */
    bool scale_by_dt;    /**< Multiply @ref scale by max(dt, 0) when true. */
} SimScaleOperatorConfig;

/**
 * @brief Register a scale utility operator.
 *
 * The implementation copies and normalizes @p config, resolves the default
 * scale-by-dt policy, and validates matching real-double or complex-double fields.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional scale configuration; NULL selects zero-initialized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context
 *         or invalid fields, #SIM_RESULT_TYPE_MISMATCH for incompatible fields,
 *         #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or a registration error.
 */
SimResult sim_add_scale_operator(struct SimContext *context, const SimScaleOperatorConfig *config,
                                 size_t *out_index);

/**
 * @brief Copy the current configuration from a registered scale operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_scale_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no scale state.
 */
SimResult sim_scale_config(struct SimContext *context, size_t operator_index,
                           SimScaleOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered scale operator.
 *
 * @p config is required. The replacement is normalized and the referenced fields
 * are validated before it is stored.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the scale operator to update.
 * @param config Replacement scale configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL inputs
 *         or invalid fields, #SIM_RESULT_NOT_FOUND for a missing operator,
 *         #SIM_RESULT_INVALID_STATE for missing state, or #SIM_RESULT_TYPE_MISMATCH
 *         for incompatible field domains.
 */
SimResult sim_scale_update(struct SimContext *context, size_t operator_index,
                           const SimScaleOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SCALE_H */
