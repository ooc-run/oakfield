/**
 * @file linear_dissipative.h
 * @brief Fractional Laplacian dissipative operator (|k|^alpha spectral damping).
 *
 * Applies dissipative dynamics via a fractional Laplacian term on matching real or complex fields.
 */
#ifndef OAKFIELD_LINEAR_DISSIPATIVE_H
#define OAKFIELD_LINEAR_DISSIPATIVE_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for the fractional Laplacian dissipative operator.
 */
typedef struct LinearDissipativeOperatorConfig {
    size_t field_index; /**< Field to which the dissipative term is applied. */
    double viscosity;   /**< Viscosity-like gain controlling dissipation strength. */
    double alpha;       /**< Fractional Laplacian exponent (0..2). */
    double spacing;     /**< Grid spacing used for discretizing the Laplacian. */
} LinearDissipativeOperatorConfig;

/**
 * @brief Add a fractional Laplacian dissipative operator to the context.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional operator configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, FFT plan setup, or registration.
 */
SimResult sim_add_linear_dissipative_operator(struct SimContext *context,
                                              const LinearDissipativeOperatorConfig *config,
                                              size_t *out_index);

/**
 * @brief Copy the current dissipative configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_linear_dissipative_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no dissipative state.
 */
SimResult sim_linear_dissipative_config(struct SimContext *context, size_t operator_index,
                                        LinearDissipativeOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered dissipative operator configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. Changes to alpha or spacing mark spectral coefficients dirty.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the dissipative operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         validation, cache setup, or state validation.
 */
SimResult sim_linear_dissipative_update(struct SimContext *context, size_t operator_index,
                                        const LinearDissipativeOperatorConfig *config);

/**
 * @brief Apply the dissipative split step for an already registered operator state.
 *
 * This is exposed for split descriptors and related operator glue; callers must
 * pass the state object created by sim_add_linear_dissipative_operator().
 *
 * @param state_ptr Internal LinearDissipativeOperatorState pointer.
 * @param context Simulation context containing the target field.
 * @param self Operator instance invoking the step.
 * @param dt Substep duration.
 * @return #SIM_RESULT_OK on success or an error code from state, field, or FFT evaluation.
 */
SimResult linear_dissipative_apply(void *state_ptr, struct SimContext *context,
                                   struct SimOperator *self, double dt);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_LINEAR_DISSIPATIVE_H */
