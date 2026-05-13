/**
 * @file chladni.h
 * @brief Chladni nodal-line stimulus for vibrating rectangular plate modes.
 *
 * Builds a Gaussian band around the nodal set of an antisymmetrized rectangular-plate mode pair:
 *   base(u,v) = cos(m*pi*u/Lx) cos(n*pi*v/Ly) - mix * cos(n*pi*u/Lx) cos(m*pi*v/Ly)
 * and writes
 *   A * exp(-base(u,v)^2 / (2*sigma^2)) * exp(i * (-omega*t + phi)).
 *
 * Real fields receive the real component of the pattern; complex fields receive the full
 * complex pattern with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_CHLADNI_H
#define OAKFIELD_STIMULUS_CHLADNI_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a Chladni nodal-line stimulus on a rectangular plate.
 */
typedef struct SimStimulusChladniConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Overall write amplitude. */
    unsigned int mode_x;          /**< Plate mode number m along local u. */
    unsigned int mode_y;          /**< Plate mode number n along local v. */
    double plate_width;           /**< Plate width Lx in local u coordinates. */
    double plate_height;          /**< Plate height Ly in local v coordinates. */
    double mix;                   /**< Antisymmetric blend factor between swapped modes. */
    double line_width;            /**< Gaussian band width around the nodal set. */
    double omega;                 /**< Temporal angular frequency (rad/s). */
    double phase;                 /**< Phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the plate frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Complex-output rotation (radians). */
    bool scale_by_dt;             /**< When true, scale writes by dt. */
} SimStimulusChladniConfig;

/**
 * @brief Register a Chladni nodal-line stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the modal nodal-band pattern into the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Chladni configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_chladni_operator(struct SimContext *context,
                                            const SimStimulusChladniConfig *config,
                                            size_t *out_index);

/**
 * @brief Copy the current Chladni configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_chladni_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_chladni_config(struct SimContext *context, size_t operator_index,
                                      SimStimulusChladniConfig *out_config);

/**
 * @brief Replace or renormalize a registered Chladni stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes derived state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Chladni operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, allocation, or
 *         state validation fails.
 */
SimResult sim_stimulus_chladni_update(struct SimContext *context, size_t operator_index,
                                      const SimStimulusChladniConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_CHLADNI_H */
