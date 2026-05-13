/**
 * @file lissajous.h
 * @brief Lissajous ridge stimulus with Gaussian band shaping.
 *
 * Builds a smooth band around the implicit relation
 *   sin(theta_x) = coupling * sin(theta_y) + bias
 * and modulates that band by a carrier phase 0.5 * (theta_x + theta_y).
 *
 * In separable coord mode, theta_x and theta_y sample the X/Y axes
 * independently. Other coord modes collapse to a shared scalar coordinate.
 */
#ifndef OAKFIELD_STIMULUS_LISSAJOUS_H
#define OAKFIELD_STIMULUS_LISSAJOUS_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a Gaussian-band Lissajous ridge stimulus.
 */
typedef struct SimStimulusLissajousConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Overall write amplitude. */
    double wavenumber_x;          /**< X oscillator spatial frequency (rad/unit). */
    double wavenumber_y;          /**< Y oscillator spatial frequency (rad/unit). */
    double omega_x;               /**< X oscillator angular frequency (rad/s). */
    double omega_y;               /**< Y oscillator angular frequency (rad/s). */
    double phase_x;               /**< X oscillator phase offset (radians). */
    double phase_y;               /**< Y oscillator phase offset (radians). */
    double coupling;              /**< Y oscillator coupling multiplier. */
    double bias;                  /**< Additive band offset applied to the implicit curve. */
    double line_width;            /**< Gaussian band width in implicit-curve space. */
    double time_offset;           /**< Additional time offset applied before evaluation. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping for scalar/separable evaluation. */
    double rotation;              /**< Complex-output rotation (radians). */
    bool scale_by_dt;             /**< When true, scale writes by dt. */
} SimStimulusLissajousConfig;

/**
 * @brief Register a Lissajous ridge stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the Gaussian band around the implicit Lissajous relation.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Lissajous configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_lissajous_operator(struct SimContext *context,
                                              const SimStimulusLissajousConfig *config,
                                              size_t *out_index);

/**
 * @brief Copy the current Lissajous configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_lissajous_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_lissajous_config(struct SimContext *context, size_t operator_index,
                                        SimStimulusLissajousConfig *out_config);

/**
 * @brief Replace or renormalize a registered Lissajous stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Lissajous operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_lissajous_update(struct SimContext *context, size_t operator_index,
                                        const SimStimulusLissajousConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_LISSAJOUS_H */
