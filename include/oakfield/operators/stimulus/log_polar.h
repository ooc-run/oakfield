/**
 * @file log_polar.h
 * @brief Log-polar interference stimulus with optional spiral phase drift.
 *
 * Builds a phase field from log-radius and polar angle:
 *   phi = k_r * log(r + eps) + m * theta - omega * (t + t_0) + phi_0
 * and writes A * cos(phi) to real fields or A * exp(i * phi) to complex fields.
 */
#ifndef OAKFIELD_STIMULUS_LOG_POLAR_H
#define OAKFIELD_STIMULUS_LOG_POLAR_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for log-polar interference stimulus fields.
 */
typedef struct SimStimulusLogPolarConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double radial_frequency;      /**< Log-radius phase coefficient. */
    double angular_frequency;     /**< Polar-angle phase coefficient. */
    double orientation;           /**< Output frame rotation angle (radians). */
    double orientation_rate;      /**< Output frame angular drift (rad/s). */
    double omega;                 /**< Temporal angular frequency (rad/s). */
    double phase;                 /**< Global phase offset (radians). */
    double radius_floor;          /**< Positive epsilon added before log(radius + eps). */
    double time_offset;           /**< Additional time shift applied before evaluation. */
    SimStimulusCoordConfig coord; /**< Coordinate sampling and radial-center config. */
    double rotation;              /**< Complex-output rotation (radians). */
    bool scale_by_dt;             /**< When true, scale writes by dt. */
} SimStimulusLogPolarConfig;

/**
 * @brief Register a log-polar interference stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that evaluates the log-radius and angular phase field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional log-polar configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_log_polar_operator(struct SimContext *context,
                                              const SimStimulusLogPolarConfig *config,
                                              size_t *out_index);

/**
 * @brief Copy the current log-polar configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_log_polar_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_log_polar_config(struct SimContext *context, size_t operator_index,
                                        SimStimulusLogPolarConfig *out_config);

/**
 * @brief Replace or renormalize a registered log-polar stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the log-polar operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_log_polar_update(struct SimContext *context, size_t operator_index,
                                        const SimStimulusLogPolarConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_LOG_POLAR_H */
