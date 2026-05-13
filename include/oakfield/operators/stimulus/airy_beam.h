/**
 * @file airy_beam.h
 * @brief Finite-energy separable Airy beam stimulus.
 *
 * Evaluates a local beam frame (u, v) and injects
 *   A * Ai(u / s_u) * Ai(v / s_v) * exp(a_u * u / s_u + a_v * v / s_v)
 *     * exp(i * (k_u * u + k_v * v - omega * t + phi)),
 * where the local coordinates can drift and rotate over time.
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * beam with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_AIRY_BEAM_H
#define OAKFIELD_STIMULUS_AIRY_BEAM_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a finite-energy separable Airy beam stimulus.
 */
typedef struct SimStimulusAiryBeamConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double scale_u;               /**< Airy scale along local u. */
    double scale_v;               /**< Airy scale along local v. */
    double apodization_u;         /**< Finite-energy taper along local u. */
    double apodization_v;         /**< Finite-energy taper along local v. */
    double center_u;              /**< Beam center in local u. */
    double center_v;              /**< Beam center in local v. */
    double velocity_u;            /**< Beam center drift in local u. */
    double velocity_v;            /**< Beam center drift in local v. */
    double orientation;           /**< Local beam orientation angle. */
    double orientation_rate;      /**< Orientation drift rate. */
    double carrier_u;             /**< Local carrier tilt along u. */
    double carrier_v;             /**< Local carrier tilt along v. */
    double omega;                 /**< Temporal angular frequency. */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local beam frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusAiryBeamConfig;

/**
 * @brief Register a finite-energy Airy beam stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the Airy beam into the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Airy beam configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_airy_beam_operator(struct SimContext *context,
                                              const SimStimulusAiryBeamConfig *config,
                                              size_t *out_index);

/**
 * @brief Copy the current Airy beam configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_airy_beam_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_airy_beam_config(struct SimContext *context, size_t operator_index,
                                        SimStimulusAiryBeamConfig *out_config);

/**
 * @brief Replace or renormalize a registered Airy beam stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes derived state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Airy beam operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, allocation, or
 *         state validation fails.
 */
SimResult sim_stimulus_airy_beam_update(struct SimContext *context, size_t operator_index,
                                        const SimStimulusAiryBeamConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_AIRY_BEAM_H */
