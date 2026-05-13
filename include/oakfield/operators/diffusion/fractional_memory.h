/**
 * @file fractional_memory.h
 * @brief History-based fractional memory operator applying a fractional derivative of order q.
 *
 * Maintains a rolling memory of past states to approximate fractional dynamics. Supports real and
 * complex fields through component-wise accumulation.
 */
#ifndef OAKFIELD_FRACTIONAL_MEMORY_H
#define OAKFIELD_FRACTIONAL_MEMORY_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration parameters for the fractional memory split operator.
 */
typedef struct FractionalMemoryOperatorConfig {
    size_t field_index;  /**< Target field index within the simulation context. */
    double order;        /**< Fractional derivative order (0..1).
                                  Values outside the range are clamped by the implementation. */
    double gain;         /**< Scaling factor applied to the memory term. */
    size_t memory_steps; /**< Number of historical steps retained in the memory kernel. */
} FractionalMemoryOperatorConfig;

/**
 * @brief Register a fractional memory operator with the provided configuration.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional operator configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         target-field validation, allocation, coefficient setup, or registration.
 */
SimResult sim_add_fractional_memory_operator(struct SimContext *context,
                                             const FractionalMemoryOperatorConfig *config,
                                             size_t *out_index);

/**
 * @brief Copy the current fractional memory configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_fractional_memory_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no memory state.
 */
SimResult sim_fractional_memory_config(struct SimContext *context, size_t operator_index,
                                       FractionalMemoryOperatorConfig *out_config);

/**
 * @brief Replace or renormalize an existing fractional memory operator.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. If order or memory_steps change, coefficients/history are marked
 * for rebuild before the next apply.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the fractional memory operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         validation, coefficient setup, or state validation.
 */
SimResult sim_fractional_memory_update(struct SimContext *context, size_t operator_index,
                                       const FractionalMemoryOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_FRACTIONAL_MEMORY_H */
