/**
 * @file laplace_beltrami.h
 * @brief Analytic Laplace-Beltrami eigenmode stimulus on simple manifolds.
 *
 * Evaluates a local chart (u, v) and injects a manifold eigenmode
 *   A * Phi_{m,n}(u,v; M) * exp(i * (-omega * (t + t_0) + phi)),
 * where M is one of:
 *   - rectangle: centered Dirichlet chart on [-L_u/2, L_u/2] x [-L_v/2, L_v/2]
 *   - flat_torus: periodic chart with periods L_u and L_v
 *   - cylinder: periodic in u with period L_u, Dirichlet in v on [-L_v/2, L_v/2]
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * mode with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_LAPLACE_BELTRAMI_H
#define OAKFIELD_STIMULUS_LAPLACE_BELTRAMI_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Analytic manifold families supported by Laplace-Beltrami stimuli.
 */
typedef enum SimStimulusLaplaceBeltramiManifold {
    SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE = 0, /**< Rectangular Dirichlet chart. */
    SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS,    /**< Periodic flat torus chart. */
    SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER       /**< Cylinder chart. */
} SimStimulusLaplaceBeltramiManifold;

/**
 * @brief Configuration for analytic Laplace-Beltrami eigenmode stimuli.
 */
typedef struct SimStimulusLaplaceBeltramiConfig {
    size_t field_index;                          /**< Target field index. */
    double amplitude;                            /**< Output amplitude scale. */
    SimStimulusLaplaceBeltramiManifold manifold; /**< Analytic manifold family. */
    int mode_u;                                  /**< Eigenmode index along local u. */
    int mode_v;                                  /**< Eigenmode index along local v. */
    double extent_u;                             /**< Chart extent or period along u. */
    double extent_v;                             /**< Chart extent or period along v. */
    double omega;                                /**< Temporal angular frequency (rad/s). */
    double phase;                                /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord;                /**< Coordinate mapping into the local chart. */
    double time_offset;                          /**< Additional time shift before evaluation. */
    double rotation;                             /**< Complex-output rotation (radians). */
    bool scale_by_dt;                            /**< When true, scale writes by dt. */
} SimStimulusLaplaceBeltramiConfig;

/**
 * @brief Register an analytic Laplace-Beltrami eigenmode stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that evaluates the selected manifold eigenmode on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Laplace-Beltrami configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_laplace_beltrami_operator(struct SimContext *context,
                                                     const SimStimulusLaplaceBeltramiConfig *config,
                                                     size_t *out_index);

/**
 * @brief Copy the current Laplace-Beltrami configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_laplace_beltrami_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_laplace_beltrami_config(struct SimContext *context, size_t operator_index,
                                               SimStimulusLaplaceBeltramiConfig *out_config);

/**
 * @brief Replace or renormalize a registered Laplace-Beltrami configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Laplace-Beltrami operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_laplace_beltrami_update(struct SimContext *context, size_t operator_index,
                                               const SimStimulusLaplaceBeltramiConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_LAPLACE_BELTRAMI_H */
