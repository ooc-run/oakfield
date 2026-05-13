/**
 * @file gabor.h
 * @brief Gabor kernel stimulus (Gaussian-windowed sinusoid) for 1D/2D fields.
 *
 * Supports real and complex fields. For complex fields, the kernel is written
 * with an additional global phase rotation.
 */
#ifndef OAKFIELD_STIMULUS_GABOR_H
#define OAKFIELD_STIMULUS_GABOR_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a Gaussian-windowed Gabor kernel stimulus.
 */
typedef struct SimStimulusGaborConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Peak amplitude of the Gabor kernel. */
    double wavenumber;            /**< Spatial wavenumber of the carrier (rad / unit). */
    double kx;                    /**< Optional wavevector X component (rad / unit). */
    double ky;                    /**< Optional wavevector Y component (rad / unit). */
    double omega;                 /**< Temporal frequency of the carrier (rad / s). */
    double phase;                 /**< Phase offset of the carrier (radians). */
    double sigma_x;               /**< Gaussian sigma along X (units). */
    double sigma_y;               /**< Gaussian sigma along Y (units). */
    double time_offset;           /**< Time offset applied before evaluation. */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double rotation;              /**< Additional phase rotation for complex outputs (radians). */
    double nominal_dt;            /**< Nominal dt when fixed_clock is enabled. */
    bool use_wavevector;          /**< When true, use (kx,ky) instead of wavenumber+coord. */
    bool fixed_clock;             /**< Lock evolution to nominal_dt instead of adaptive dt. */
    bool scale_by_dt;             /**< Scale writes by dt when true; else dt-independent. */
} SimStimulusGaborConfig;

/**
 * @brief Register a Gaussian-windowed Gabor stimulus operator.
 *
 * The implementation copies and normalizes @p config, prepares envelope storage,
 * and registers the operator on the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Gabor configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_gabor_operator(struct SimContext *context,
                                          const SimStimulusGaborConfig *config, size_t *out_index);

/**
 * @brief Copy the current Gabor configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_gabor_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_gabor_config(struct SimContext *context, size_t operator_index,
                                    SimStimulusGaborConfig *out_config);

/**
 * @brief Replace or renormalize a registered Gabor stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes envelope and symbolic state, then
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Gabor operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, buffer setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_gabor_update(struct SimContext *context, size_t operator_index,
                                    const SimStimulusGaborConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_GABOR_H */
