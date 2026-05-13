/**
 * @file bessel_beam.h
 * @brief Integer-order cylindrical Bessel beam stimulus.
 *
 * Evaluates a local beam frame (u, v) and injects
 *   A * J_n(k_r * rho) * exp(i * (n * theta - omega * t + phi)),
 * where rho = hypot(u / s_u, v / s_v) and theta = atan2(v / s_v, u / s_u).
 * The local frame can drift and rotate over time.
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * beam with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_BESSEL_BEAM_H
#define OAKFIELD_STIMULUS_BESSEL_BEAM_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for an integer-order cylindrical Bessel beam stimulus.
 */
typedef struct SimStimulusBesselBeamConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    int order;                    /**< Integer Bessel order n. */
    double radial_wavenumber;     /**< Radial wavenumber k_r. */
    double scale_u;               /**< Radial scale along local u. */
    double scale_v;               /**< Radial scale along local v. */
    double center_u;              /**< Beam center in local u. */
    double center_v;              /**< Beam center in local v. */
    double velocity_u;            /**< Beam center drift in local u. */
    double velocity_v;            /**< Beam center drift in local v. */
    double orientation;           /**< Local beam orientation angle. */
    double orientation_rate;      /**< Orientation drift rate. */
    double omega;                 /**< Temporal angular frequency. */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local beam frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusBesselBeamConfig;

/**
 * @brief Register an integer-order cylindrical Bessel beam stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the Bessel beam into the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Bessel beam configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_bessel_beam_operator(struct SimContext *context,
                                                const SimStimulusBesselBeamConfig *config,
                                                size_t *out_index);

/**
 * @brief Copy the current Bessel beam configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_bessel_beam_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_bessel_beam_config(struct SimContext *context, size_t operator_index,
                                          SimStimulusBesselBeamConfig *out_config);

/**
 * @brief Replace or renormalize a registered Bessel beam stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes derived state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Bessel beam operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, allocation, or
 *         state validation fails.
 */
SimResult sim_stimulus_bessel_beam_update(struct SimContext *context, size_t operator_index,
                                          const SimStimulusBesselBeamConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_BESSEL_BEAM_H */
