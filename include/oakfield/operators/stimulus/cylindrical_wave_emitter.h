/**
 * @file cylindrical_wave_emitter.h
 * @brief Regularized cylindrical wave-emitter stimulus.
 *
 * Evaluates a local chart (u, v) and injects
 *   A * exp(-alpha * r_a) / sqrt(r_a) * exp(i * (k_r * r_a - omega * t + phi)),
 * where r_a = sqrt(rho^2 + a^2) regularizes the source core and
 * rho = |u - u_c| in rank-1 or hypot(u - u_c, v - v_c) in rank-2.
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * wave with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_CYLINDRICAL_WAVE_EMITTER_H
#define OAKFIELD_STIMULUS_CYLINDRICAL_WAVE_EMITTER_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a regularized cylindrical wave-emitter stimulus.
 */
typedef struct SimStimulusCylindricalWaveEmitterConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double radial_wavenumber;     /**< Radial wavenumber k_r. */
    double attenuation;           /**< Exponential attenuation alpha. */
    double softening_radius;      /**< Core radius a used to regularize r_a. */
    double center_u;              /**< Emitter center in local u. */
    double center_v;              /**< Emitter center in local v. */
    double velocity_u;            /**< Emitter-center drift in local u. */
    double velocity_v;            /**< Emitter-center drift in local v. */
    double omega;                 /**< Temporal angular frequency. */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local chart. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusCylindricalWaveEmitterConfig;

/**
 * @brief Register a regularized cylindrical wave-emitter stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the emitter wave into the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional emitter configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_cylindrical_wave_emitter_operator(
    struct SimContext *context, const SimStimulusCylindricalWaveEmitterConfig *config,
    size_t *out_index);

/**
 * @brief Copy the current cylindrical emitter configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_cylindrical_wave_emitter_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult
sim_stimulus_cylindrical_wave_emitter_config(struct SimContext *context, size_t operator_index,
                                             SimStimulusCylindricalWaveEmitterConfig *out_config);

/**
 * @brief Replace or renormalize a registered cylindrical emitter configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes derived state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the cylindrical emitter operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, allocation, or
 *         state validation fails.
 */
SimResult
sim_stimulus_cylindrical_wave_emitter_update(struct SimContext *context, size_t operator_index,
                                             const SimStimulusCylindricalWaveEmitterConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_CYLINDRICAL_WAVE_EMITTER_H */
