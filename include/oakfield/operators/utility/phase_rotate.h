/**
 * @file phase_rotate.h
 * @brief Utility operator that advances field phase by a rate times the timestep.
 *
 * Complex fields are rotated by exp(i * phase_rate * dt). Real fields, or fields
 * constrained to real spectral subspaces, receive the projected cos(theta) scale.
 */
#ifndef OAKFIELD_PHASE_ROTATE_H
#define OAKFIELD_PHASE_ROTATE_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for applying a uniform phase rotation to a target field.
 */
typedef struct PhaseRotateOperatorConfig {
    size_t field_index; /**< Field index to rotate or projectively scale. */
    double phase_rate;  /**< Angular phase rate in radians per second. */
} PhaseRotateOperatorConfig;

/**
 * @brief Register a phase-rotation utility operator.
 *
 * The implementation copies and normalizes @p config, validates that the target
 * field is real double or complex double, and may register a symbolic kernel for
 * unconstrained complex rotations.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional phase-rotation configuration; NULL selects field 0 and zero rate.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context
 *         or invalid field, #SIM_RESULT_TYPE_MISMATCH for unsupported field
 *         storage, #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or a
 *         registration error.
 */
SimResult sim_add_phase_rotate_operator(struct SimContext *context,
                                        const PhaseRotateOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current configuration from a registered phase-rotation operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_phase_rotate_operator().
 * @param[out] out_config Receives the operator configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no phase-rotation state.
 */
SimResult sim_phase_rotate_config(struct SimContext *context, size_t operator_index,
                                  PhaseRotateOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered phase-rotation operator.
 *
 * @p config is required. A non-finite replacement phase_rate preserves the
 * previous rate. A successful update refreshes operator info, patches the kernel
 * constant when one is registered, and invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the phase-rotation operator to update.
 * @param config Replacement phase-rotation configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL inputs
 *         or invalid field, #SIM_RESULT_NOT_FOUND for a missing operator,
 *         #SIM_RESULT_INVALID_STATE for missing state, or #SIM_RESULT_TYPE_MISMATCH
 *         for unsupported field storage.
 */
SimResult sim_phase_rotate_update(struct SimContext *context, size_t operator_index,
                                  const PhaseRotateOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_PHASE_ROTATE_H */
