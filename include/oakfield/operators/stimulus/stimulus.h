/**
 * @file stimulus.h
 * @brief Legacy simple sinusoidal forcing operator (single-mode driver).
 *
 * For richer stimuli (chirp, traveling Gaussian), prefer the operators in sinusoidal.h
 * and gaussian.h. Supports real and complex fields; complex writes can include a phase
 * rotation.
 */
#ifndef OAKFIELD_STIMULUS_H
#define OAKFIELD_STIMULUS_H

#include "coords.h"
#include "oakfield/operator_split.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;
struct SimOperator;

/**
 * @brief Configuration parameters for the legacy sinusoidal stimulus operator.
 */
typedef struct StimulusOperatorConfig {
    size_t field_index;           /**< Field index receiving the forcing term. */
    double amplitude;             /**< Signal amplitude. */
    double wavenumber;            /**< Spatial wavenumber (rad / unit). */
    double omega;                 /**< Angular frequency (rad / s). */
    double phase;                 /**< Phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double time_offset; /**< Time offset applied before evaluating the driving function. */
    bool scale_by_dt; /**< When true, scale writes by substep dt; false = dt-independent signal. */
} StimulusOperatorConfig;

/**
 * @brief Register the legacy single-mode sinusoidal stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that adds a real sinusoidal forcing term to the target field. Complex
 * targets receive the forcing in the real component.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional stimulus configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_operator(struct SimContext *context,
                                    const StimulusOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current legacy stimulus configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_config(struct SimContext *context, size_t operator_index,
                              StimulusOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered legacy stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the legacy stimulus operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_update(struct SimContext *context, size_t operator_index,
                              const StimulusOperatorConfig *config);
#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_H */
