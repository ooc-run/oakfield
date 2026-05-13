/**
 * @file segmented_sieve_mark.h
 * @brief Exact integer segmented-sieve marking operator.
 */
#ifndef OAKFIELD_SEGMENTED_SIEVE_MARK_H
#define OAKFIELD_SEGMENTED_SIEVE_MARK_H

#include "oakfield/operator_split.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for in-place segmented sieve marking.
 *
 * The operator reads an exact integer candidate field and clears entries in the
 * flag field when `candidate > prime_value && candidate % prime_value == 0`.
 * The flag field may be exact integer or `f64`; in both cases active entries
 * are preserved until a composite hit clears them to zero.
 */
typedef struct SimSegmentedSieveMarkOperatorConfig {
    size_t candidate_field; /**< Exact integer field holding candidate values. */
    size_t flags_field;     /**< In-place real/integer flag field cleared for composites. */
    uint64_t prime;         /**< Positive base prime used for the current marking pass. */
} SimSegmentedSieveMarkOperatorConfig;

/**
 * @brief Register an in-place segmented-sieve marking operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional mark configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         exact-integer field validation, allocation, or split registration.
 */
SimResult sim_add_segmented_sieve_mark_operator(struct SimContext *context,
                                                const SimSegmentedSieveMarkOperatorConfig *config,
                                                size_t *out_index);

/**
 * @brief Copy the current segmented-sieve mark configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_segmented_sieve_mark_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no mark state.
 */
SimResult sim_segmented_sieve_mark_config(struct SimContext *context, size_t operator_index,
                                          SimSegmentedSieveMarkOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a segmented-sieve mark configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the mark operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         validation, or state validation.
 */
SimResult sim_segmented_sieve_mark_update(struct SimContext *context, size_t operator_index,
                                          const SimSegmentedSieveMarkOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SEGMENTED_SIEVE_MARK_H */
