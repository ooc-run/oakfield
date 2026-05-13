/**
 * @file optical_vortex.h
 * @brief Optical vortex beam stimulus with Gaussian envelope and phase singularity.
 *
 * Synthesizes a vortex beam of the form
 *   V(u,v,t) = A * rho^{|l|} * exp(-rho^2 / 2) * exp(i * (l * theta - omega * t + phi)),
 * where rho is the beam radius normalized by the configured waists and l is the
 * topological charge.
 *
 * For real fields, the real component of the beam is injected. For complex fields,
 * the full complex beam is written with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_OPTICAL_VORTEX_H
#define OAKFIELD_STIMULUS_OPTICAL_VORTEX_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for an optical vortex beam with phase winding.
 */
typedef struct SimStimulusOpticalVortexConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    int charge;                   /**< Topological charge l controlling phase winding. */
    double waist_x;               /**< Beam waist along local u. */
    double waist_y;               /**< Beam waist along local v. */
    double center_u;              /**< Beam center in local u. */
    double center_v;              /**< Beam center in local v. */
    double velocity_u;            /**< Beam center drift velocity in u. */
    double velocity_v;            /**< Beam center drift velocity in v. */
    double orientation;           /**< Local beam orientation angle. */
    double orientation_rate;      /**< Beam orientation drift (rad/s). */
    double omega;                 /**< Temporal angular frequency. */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local beam frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusOpticalVortexConfig;

/**
 * @brief Register an optical vortex beam stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the vortex envelope and phase winding into the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional optical-vortex configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_optical_vortex_operator(struct SimContext *context,
                                                   const SimStimulusOpticalVortexConfig *config,
                                                   size_t *out_index);

/**
 * @brief Copy the current optical-vortex configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_optical_vortex_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_optical_vortex_config(struct SimContext *context, size_t operator_index,
                                             SimStimulusOpticalVortexConfig *out_config);

/**
 * @brief Replace or renormalize a registered optical-vortex configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the optical-vortex operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_optical_vortex_update(struct SimContext *context, size_t operator_index,
                                             const SimStimulusOpticalVortexConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_OPTICAL_VORTEX_H */
