/**
 * @file neural_hybrid.h
 * @brief Analytic+residual neural hybrid operator scaffolding.
 */
#ifndef OAKFIELD_NEURAL_HYBRID_H
#define OAKFIELD_NEURAL_HYBRID_H

#include "neural_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for analytic + neural residual hybrid operator.
 */
typedef struct SimNeuralHybridOperatorConfig {
    SimNeuralBaseConfig base; /**< Shared neural input/output and backend configuration. */
    double analytic_gain;     /**< Gain applied to the analytic input path. */
    double residual_gain;     /**< Gain applied to the neural residual prediction. */
} SimNeuralHybridOperatorConfig;

/**
 * @brief Register an analytic-plus-neural residual hybrid operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional hybrid configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_neural_hybrid_operator(struct SimContext *context,
                                         const SimNeuralHybridOperatorConfig *config,
                                         size_t *out_index);

/**
 * @brief Copy the current neural hybrid configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_neural_hybrid_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no hybrid state.
 */
SimResult sim_neural_hybrid_config(struct SimContext *context, size_t operator_index,
                                   SimNeuralHybridOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered neural hybrid operator.
 *
 * @p config is required. The replacement is normalized and field/shape
 * compatibility is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the neural hybrid operator to update.
 * @param config Replacement hybrid configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field compatibility checks, or state validation.
 */
SimResult sim_neural_hybrid_update(struct SimContext *context, size_t operator_index,
                                   const SimNeuralHybridOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_NEURAL_HYBRID_H */
