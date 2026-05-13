/**
 * @file gaussian.h
 * @brief Traveling Gaussian envelope stimulus for modulating fields.
 *
 * Can target real or complex fields; complex writes support phase rotation.
 */
#ifndef OAKFIELD_STIMULUS_GAUSSIAN_H
#define OAKFIELD_STIMULUS_GAUSSIAN_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Parameterization for Gaussian-envelope stimulus operators.
 */
typedef struct SimStimulusGaussianConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Peak amplitude of the envelope. */
    double sigma_x;               /**< Gaussian standard deviation along X (units). */
    double sigma_y;               /**< Gaussian standard deviation along Y (units). */
    double time_offset;           /**< Time offset applied before evaluating the envelope. */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double rotation;   /**< Phase rotation applied when writing complex outputs (radians). */
    double nominal_dt; /**< Nominal dt used when @ref fixed_clock is true. */
    bool fixed_clock;  /**< Hold the driving clock to @ref nominal_dt when true. */
    bool scale_by_dt;  /**< When true, scale writes by substep dt; false = dt-independent signal. */
} SimStimulusGaussianConfig;

/**
 * @brief Register a traveling Gaussian-envelope stimulus operator.
 *
 * The implementation copies and normalizes @p config, prepares any required
 * buffers, and registers the operator on the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Gaussian configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_gaussian_operator(struct SimContext *context,
                                             const SimStimulusGaussianConfig *config,
                                             size_t *out_index);

/**
 * @brief Copy the current Gaussian configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_gaussian_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_gaussian_config(struct SimContext *context, size_t operator_index,
                                       SimStimulusGaussianConfig *out_config);

/**
 * @brief Replace or renormalize a registered Gaussian stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes runtime state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Gaussian operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, buffer setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_gaussian_update(struct SimContext *context, size_t operator_index,
                                       const SimStimulusGaussianConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_GAUSSIAN_H */
