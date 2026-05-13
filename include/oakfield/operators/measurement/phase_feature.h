/**
 * @file phase_feature.h
 * @brief Phase-feature extraction for real and complex fields.
 */
#ifndef OAKFIELD_PHASE_FEATURE_H
#define OAKFIELD_PHASE_FEATURE_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for phase-feature extraction.
 */
typedef struct SimPhaseFeatureOperatorConfig {
    size_t input_field;  /**< Source field supplying samples. */
    size_t output_field; /**< Field receiving phase features. */
    double threshold;    /**< Magnitude gate below which samples are suppressed. */
    double exponent;     /**< Exponent applied to (|x| - threshold) when weighting features. */
    bool accumulate;     /**< When true, adds extracted features into the output. */
    bool scale_by_dt;    /**< When true, scale accumulated writes by substep dt. */
} SimPhaseFeatureOperatorConfig;

/**
 * @brief Register a phase-feature extraction operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional phase-feature configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_phase_feature_operator(struct SimContext *context,
                                         const SimPhaseFeatureOperatorConfig *config,
                                         size_t *out_index);

/**
 * @brief Copy the current phase-feature configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_phase_feature_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no phase-feature state.
 */
SimResult sim_phase_feature_config(struct SimContext *context, size_t operator_index,
                                   SimPhaseFeatureOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered phase-feature configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the phase-feature operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         compatibility checks, or state validation.
 */
SimResult sim_phase_feature_update(struct SimContext *context, size_t operator_index,
                                   const SimPhaseFeatureOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_PHASE_FEATURE_H */
