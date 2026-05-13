/**
 * @file moire.h
 * @brief Moire interference stimulus for 1D/2D coordinate systems.
 *
 * The operator adds the interference of two nearby gratings:
 *   m = 0.5 * (cos(theta_a) + cos(theta_b))
 * with either scalar coordinate mapping (`coord`) or explicit 2D wavevectors.
 */
#ifndef OAKFIELD_STIMULUS_MOIRE_H
#define OAKFIELD_STIMULUS_MOIRE_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for two-grating moire interference stimuli.
 */
typedef struct SimStimulusMoireConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double wavenumber_a;          /**< First scalar wavenumber (rad / unit). */
    double wavenumber_b;          /**< Second scalar wavenumber (rad / unit). */
    double k1x;                   /**< First wavevector X component (rad / unit). */
    double k1y;                   /**< First wavevector Y component (rad / unit). */
    double k2x;                   /**< Second wavevector X component (rad / unit). */
    double k2y;                   /**< Second wavevector Y component (rad / unit). */
    double omega_a;               /**< First angular frequency (rad / s). */
    double omega_b;               /**< Second angular frequency (rad / s). */
    double phase_a;               /**< First phase offset (radians). */
    double phase_b;               /**< Second phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping used when not in wavevector mode. */
    double time_offset;           /**< Additional time offset applied before evaluation. */
    double rotation;              /**< Complex-output rotation (radians, complex fields only). */
    bool use_wavevectors; /**< True: use (k1x,k1y)/(k2x,k2y); false: use wavenumber_a/b + coord. */
    bool scale_by_dt;     /**< True: scale writes by dt; false: dt-independent writes. */
} SimStimulusMoireConfig;

/**
 * @brief Register a moire interference stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the two-grating interference pattern.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional moire configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_moire_operator(struct SimContext *context,
                                          const SimStimulusMoireConfig *config, size_t *out_index);

/**
 * @brief Copy the current moire configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_moire_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_moire_config(struct SimContext *context, size_t operator_index,
                                    SimStimulusMoireConfig *out_config);

/**
 * @brief Replace or renormalize a registered moire stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the moire operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_moire_update(struct SimContext *context, size_t operator_index,
                                    const SimStimulusMoireConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_MOIRE_H */
