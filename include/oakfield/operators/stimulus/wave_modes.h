/**
 * @file wave_modes.h
 * @brief Rectangular wave-equation standing modes with fixed boundaries.
 *
 * Evaluates a local chart (u, v) and injects
 *   A * Phi_{m,n}(u, v) * exp(i * (-omega_{m,n} * (t + t_0) + phi)),
 * where
 *   Phi_{m,n}(u, v) = sin(m * pi * (u / L_u + 1/2)) * sin(n * pi * (v / L_v + 1/2))
 * and
 *   omega_{m,n} = c * pi * sqrt((m / L_u)^2 + (n / L_v)^2).
 *
 * For rank-1 fields, only the u-mode is used and omega_m = c * pi * m / L_u.
 * Real fields receive the real component. Complex fields receive the full complex
 * mode with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_WAVE_MODES_H
#define OAKFIELD_STIMULUS_WAVE_MODES_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for rectangular standing wave-equation mode stimuli.
 */
typedef struct SimStimulusWaveModesConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    unsigned int mode_u;          /**< Standing-mode index along local u. */
    unsigned int mode_v;          /**< Standing-mode index along local v. */
    double extent_u;              /**< Rectangular extent along local u. */
    double extent_v;              /**< Rectangular extent along local v. */
    double wave_speed;            /**< Wave speed c used to derive omega_{m,n}. */
    double phase;                 /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local chart. */
    double time_offset;           /**< Additional time shift before evaluation. */
    double rotation;              /**< Complex-output rotation (radians). */
    bool scale_by_dt;             /**< When true, scale writes by dt. */
} SimStimulusWaveModesConfig;

/**
 * @brief Register a rectangular wave-equation standing-mode stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that evaluates the fixed-boundary mode on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional wave-modes configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_wave_modes_operator(struct SimContext *context,
                                               const SimStimulusWaveModesConfig *config,
                                               size_t *out_index);

/**
 * @brief Copy the current wave-modes configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_wave_modes_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_wave_modes_config(struct SimContext *context, size_t operator_index,
                                         SimStimulusWaveModesConfig *out_config);

/**
 * @brief Replace or renormalize a registered wave-modes stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the wave-modes operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_wave_modes_update(struct SimContext *context, size_t operator_index,
                                         const SimStimulusWaveModesConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_WAVE_MODES_H */
