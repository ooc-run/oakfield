/**
 * @file zone_plate.h
 * @brief Fresnel-style zone plate stimulus with quadratic radial phase.
 *
 * Evaluates a local plate frame (u, v) and injects
 *   A * exp(-0.5 * ((u/a_u)^2 + (v/a_v)^2))
 *     * exp(i * (kappa * ((u/s_u)^2 + (v/s_v)^2) - omega * (t + t_0) + phi)),
 * where the local frame can drift and rotate over time.
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * pattern with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_ZONE_PLATE_H
#define OAKFIELD_STIMULUS_ZONE_PLATE_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a Fresnel-style zone plate stimulus.
 */
typedef struct SimStimulusZonePlateConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double radial_chirp;          /**< Quadratic radial phase coefficient. */
    double scale_u;               /**< Radius scale along local u. */
    double scale_v;               /**< Radius scale along local v. */
    double aperture_u;            /**< Gaussian aperture width along local u. */
    double aperture_v;            /**< Gaussian aperture width along local v. */
    double center_u;              /**< Plate center in local u. */
    double center_v;              /**< Plate center in local v. */
    double velocity_u;            /**< Plate-center drift in local u. */
    double velocity_v;            /**< Plate-center drift in local v. */
    double orientation;           /**< Local plate orientation angle. */
    double orientation_rate;      /**< Orientation drift rate. */
    double omega;                 /**< Temporal angular frequency (rad/s). */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local plate frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusZonePlateConfig;

/**
 * @brief Register a Fresnel-style zone plate stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the quadratic radial phase pattern into the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional zone-plate configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_zone_plate_operator(struct SimContext *context,
                                               const SimStimulusZonePlateConfig *config,
                                               size_t *out_index);

/**
 * @brief Copy the current zone-plate configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_zone_plate_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_zone_plate_config(struct SimContext *context, size_t operator_index,
                                         SimStimulusZonePlateConfig *out_config);

/**
 * @brief Replace or renormalize a registered zone-plate stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the zone-plate operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_zone_plate_update(struct SimContext *context, size_t operator_index,
                                         const SimStimulusZonePlateConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_ZONE_PLATE_H */
