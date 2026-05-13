/**
 * @file hermite_gaussian_beam.h
 * @brief Hermite-Gaussian beam stimulus with separable transverse modes.
 *
 * Evaluates a local beam frame (u, v) and injects
 *   A * H_m(sqrt(2) * u / w_u) * H_n(sqrt(2) * v / w_v)
 *     * exp(-(u^2 / w_u^2 + v^2 / w_v^2))
 *     * exp(i * (k_u * u + k_v * v - omega * t + phi)),
 * where the local coordinates can drift and rotate over time.
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * beam with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_HERMITE_GAUSSIAN_BEAM_H
#define OAKFIELD_STIMULUS_HERMITE_GAUSSIAN_BEAM_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a separable Hermite-Gaussian beam stimulus.
 */
typedef struct SimStimulusHermiteGaussianBeamConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    unsigned int mode_u;          /**< Hermite mode index along local u. */
    unsigned int mode_v;          /**< Hermite mode index along local v. */
    double waist_u;               /**< Beam waist along local u. */
    double waist_v;               /**< Beam waist along local v. */
    double center_u;              /**< Beam center in local u. */
    double center_v;              /**< Beam center in local v. */
    double velocity_u;            /**< Beam center drift in local u. */
    double velocity_v;            /**< Beam center drift in local v. */
    double orientation;           /**< Local beam orientation angle. */
    double orientation_rate;      /**< Orientation drift rate. */
    double carrier_u;             /**< Carrier tilt along local u. */
    double carrier_v;             /**< Carrier tilt along local v. */
    double omega;                 /**< Temporal angular frequency. */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local beam frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusHermiteGaussianBeamConfig;

/**
 * @brief Register a Hermite-Gaussian beam stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the separable transverse mode into the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Hermite-Gaussian configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult
sim_add_stimulus_hermite_gaussian_beam_operator(struct SimContext *context,
                                                const SimStimulusHermiteGaussianBeamConfig *config,
                                                size_t *out_index);

/**
 * @brief Copy the current Hermite-Gaussian configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_hermite_gaussian_beam_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult
sim_stimulus_hermite_gaussian_beam_config(struct SimContext *context, size_t operator_index,
                                          SimStimulusHermiteGaussianBeamConfig *out_config);

/**
 * @brief Replace or renormalize a registered Hermite-Gaussian beam configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Hermite-Gaussian beam operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult
sim_stimulus_hermite_gaussian_beam_update(struct SimContext *context, size_t operator_index,
                                          const SimStimulusHermiteGaussianBeamConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_HERMITE_GAUSSIAN_BEAM_H */
