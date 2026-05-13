/**
 * @file segmented_sieve_mark_batch.h
 * @brief Batched exact integer segmented-sieve marking operator.
 */
#ifndef OAKFIELD_SEGMENTED_SIEVE_MARK_BATCH_H
#define OAKFIELD_SEGMENTED_SIEVE_MARK_BATCH_H

#include "oakfield/operator_split.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for batched segmented-sieve marking over exact integer fields.
 */
typedef struct SimSegmentedSieveMarkBatchOperatorConfig {
    size_t candidate_field; /**< Exact integer field holding candidate values. */
    size_t primes_field;    /**< Exact integer field containing primes to mark with. */
    size_t flags_field;     /**< In-place real/integer flag field cleared for composites. */
} SimSegmentedSieveMarkBatchOperatorConfig;

/**
 * @brief Register a batched segmented-sieve marking operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional batch mark configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         exact-integer field validation, allocation, or split registration.
 */
SimResult
sim_add_segmented_sieve_mark_batch_operator(struct SimContext *context,
                                            const SimSegmentedSieveMarkBatchOperatorConfig *config,
                                            size_t *out_index);

/**
 * @brief Copy the current batched sieve-mark configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_segmented_sieve_mark_batch_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no batch-mark state.
 */
SimResult
sim_segmented_sieve_mark_batch_config(struct SimContext *context, size_t operator_index,
                                      SimSegmentedSieveMarkBatchOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a batched segmented-sieve mark configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the batch mark operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         validation, or state validation.
 */
SimResult
sim_segmented_sieve_mark_batch_update(struct SimContext *context, size_t operator_index,
                                      const SimSegmentedSieveMarkBatchOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SEGMENTED_SIEVE_MARK_BATCH_H */
