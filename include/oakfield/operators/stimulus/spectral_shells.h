/**
 * @file spectral_shells.h
 * @brief Spectral-shell stimulus: random annular bands in k-space.
 *
 * Synthesizes spatial fields from seeded Fourier modes sampled within
 * concentric k-space shells (annuli). For 1D fields, shell radii map to
 * signed 1D wavenumbers; for 2D fields, shell radii map to random angles.
 */
#ifndef OAKFIELD_STIMULUS_SPECTRAL_SHELLS_H
#define OAKFIELD_STIMULUS_SPECTRAL_SHELLS_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for seeded random modes sampled from spectral shells.
 */
typedef struct SimStimulusSpectralShellsConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Overall amplitude scale. */
    double k_min;                 /**< Minimum shell radius in k-space (rad / unit). */
    double k_max;                 /**< Maximum shell radius in k-space (rad / unit). */
    double shell_width;           /**< Width of each shell (rad / unit); 0 = auto. */
    double omega;                 /**< Temporal angular frequency Ω (rad / s). */
    SimStimulusCoordConfig coord; /**< Sampling lattice origin/spacing. */
    double time_offset;           /**< Additional time shift applied before evaluation. */
    double nominal_dt;            /**< Nominal dt when fixed_clock is enabled. */
    double spectral_slope;        /**< Spectral slope exponent β, PSD ∝ |k|^{-β}. */
    unsigned int shell_count;     /**< Number of concentric shells. */
    unsigned int modes_per_shell; /**< Random Fourier modes per shell. */
    uint64_t seed;                /**< RNG seed for deterministic shell realization. */
    bool fixed_clock;             /**< Lock evolution to nominal_dt instead of adaptive dt. */
    bool scale_by_dt;             /**< Scale writes by dt when true; else dt-independent. */
} SimStimulusSpectralShellsConfig;

/**
 * @brief Register a random spectral-shell stimulus operator.
 *
 * The implementation copies and normalizes @p config, builds deterministic shell
 * modes from the seed, and registers the operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional spectral-shells configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         mode setup, or split-operator registration.
 */
SimResult sim_add_stimulus_spectral_shells_operator(struct SimContext *context,
                                                    const SimStimulusSpectralShellsConfig *config,
                                                    size_t *out_index);

/**
 * @brief Copy the current spectral-shells configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_spectral_shells_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_spectral_shells_config(struct SimContext *context, size_t operator_index,
                                              SimStimulusSpectralShellsConfig *out_config);

/**
 * @brief Replace or renormalize a registered spectral-shells configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update rebuilds shell modes as needed and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the spectral-shells operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, mode setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_spectral_shells_update(struct SimContext *context, size_t operator_index,
                                              const SimStimulusSpectralShellsConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_SPECTRAL_SHELLS_H */
