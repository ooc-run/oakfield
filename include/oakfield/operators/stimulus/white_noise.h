/**
 * @file white_noise.h
 * @brief True white-noise stimulus with seeded RNG and complex support.
 *
 * Injects independent Gaussian white noise into a target field each step.
 * For complex fields, independent noise is applied to real and imaginary
 * components. The operator is deterministic given its seed.
 */
#ifndef OAKFIELD_STIMULUS_WHITE_NOISE_H
#define OAKFIELD_STIMULUS_WHITE_NOISE_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for seeded Gaussian white-noise stimulus injection.
 */
typedef struct SimStimulusWhiteNoiseConfig {
    size_t field_index; /**< Target field index. */
    double sigma;       /**< Standard deviation of the white noise. */
    double mean;        /**< Mean offset applied to the noise. */
    double nominal_dt;  /**< Nominal dt when fixed_clock is enabled (for scaling). */
    uint64_t seed;      /**< Seed for reproducible random streams. */
    bool fixed_clock;   /**< Lock scaling to nominal_dt instead of adaptive dt. */
    bool scale_by_dt;   /**< When true, scale noise by sqrt(dt); else dt-independent. */
} SimStimulusWhiteNoiseConfig;

/**
 * @brief Register a seeded white-noise stimulus operator.
 *
 * The implementation copies and normalizes @p config, derives a deterministic
 * stream when the seed is zero, and registers the noise operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional white-noise configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_white_noise_operator(struct SimContext *context,
                                                const SimStimulusWhiteNoiseConfig *config,
                                                size_t *out_index);

/**
 * @brief Copy the current white-noise configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_white_noise_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_white_noise_config(struct SimContext *context, size_t operator_index,
                                          SimStimulusWhiteNoiseConfig *out_config);

/**
 * @brief Replace or renormalize a registered white-noise stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes RNG state when needed and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the white-noise operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_white_noise_update(struct SimContext *context, size_t operator_index,
                                          const SimStimulusWhiteNoiseConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_WHITE_NOISE_H */
