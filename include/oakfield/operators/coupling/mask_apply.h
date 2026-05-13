/**
 * @file mask_apply.h
 * @brief Mask/apply operator to gate a field by a mask.
 */
#ifndef OAKFIELD_MASK_APPLY_H
#define OAKFIELD_MASK_APPLY_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Mask application mode.
 */
typedef enum SimMaskMode { SIM_MASK_MODE_APPLY = 0, SIM_MASK_MODE_INVERT } SimMaskMode;

/**
 * @brief Configuration for the mask/apply operator.
 */
typedef struct SimMaskOperatorConfig {
    size_t input_field;   /**< Field supplying values to be masked. */
    size_t mask_field;    /**< Field supplying mask values. */
    size_t output_field;  /**< Field receiving the masked result. */
    SimMaskMode mode;     /**< Apply vs invert mask. */
    double threshold;     /**< Threshold for binary masking. */
    double feather;       /**< Soft transition half-width (0 for hard mask). */
    double fill_value;    /**< Fill value used when mask is inactive. */
    double fill_value_im; /**< Imaginary fill value when output is complex. */
    bool accumulate;      /**< Add into output when true. */
    bool scale_by_dt;     /**< Scale accumulated writes by substep dt. */
} SimMaskOperatorConfig;

/**
 * @brief Register a mask/apply operator with the provided configuration.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional mask configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_mask_operator(struct SimContext *context, const SimMaskOperatorConfig *config,
                                size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a mask operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_mask_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no mask state.
 */
SimResult sim_mask_config(struct SimContext *context, size_t operator_index,
                          SimMaskOperatorConfig *out_config);

/**
 * @brief Update an existing mask operator in-place.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the mask operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         compatibility checks, or state validation.
 */
SimResult sim_mask_update(struct SimContext *context, size_t operator_index,
                          const SimMaskOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MASK_APPLY_H */
