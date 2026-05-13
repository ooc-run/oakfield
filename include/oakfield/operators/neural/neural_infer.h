/**
 * @file neural_infer.h
 * @brief Neural inference operator scaffolding.
 */
#ifndef OAKFIELD_NEURAL_INFER_H
#define OAKFIELD_NEURAL_INFER_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/** Maximum model identifier length (excluding null terminator). */
#define SIM_NEURAL_MODEL_ID_MAX 63U

/**
 * @brief Inference request metadata passed to neural callbacks.
 */
typedef struct SimNeuralInferenceRequest {
    const char *model_id;                          /**< Null-terminated model identifier. */
    SimNeuralDeterminismPolicy determinism_policy; /**< Requested determinism policy. */
    SimNeuralDeviceRequirement device_requirement; /**< Requested backend device class. */
    SimNeuralPrecisionMode precision_mode;         /**< Requested numeric precision. */
    SimNeuralShapeConstraints shape_constraints;   /**< Shape constraints advertised to backend. */
    bool normalize_input; /**< Apply input normalization before inference. */
    double input_scale;   /**< Scale applied to input samples. */
    double input_bias;    /**< Bias applied to input samples. */
    double output_scale;  /**< Scale applied to model outputs. */
    double output_bias;   /**< Bias applied to model outputs. */
    size_t step_index;    /**< Current simulation step. */
    double dt;            /**< Current substep duration in seconds. */
    double sim_time;      /**< Simulation time at the request. */
} SimNeuralInferenceRequest;

/**
 * @brief Callback surface for pluggable neural inference backends.
 *
 * Implementations write model outputs into @p output using the same shape and
 * element type as configured fields.
 *
 * @param userdata Opaque backend pointer from the operator configuration.
 * @param input Source field snapshot for the inference request.
 * @param[out] output Field storage that receives model predictions.
 * @param request Normalized inference request metadata.
 * @return #SIM_RESULT_OK on success or an error code reported by the backend.
 */
typedef SimResult (*SimNeuralInferenceFn)(void *userdata, const SimField *input, SimField *output,
                                          const SimNeuralInferenceRequest *request);

/**
 * @brief Shared base configuration for neural operators.
 */
typedef struct SimNeuralBaseConfig {
    size_t input_field;                          /**< Source field index. */
    size_t output_field;                         /**< Destination field index. */
    char model_id[SIM_NEURAL_MODEL_ID_MAX + 1U]; /**< Backend model identifier. */
    bool accumulate;                             /**< Add predictions into output when true. */
    bool scale_by_dt;                            /**< Scale writes by substep dt when true. */
    bool normalize_input; /**< Apply input_scale and input_bias before inference. */
    double input_scale;   /**< Input normalization scale. */
    double input_bias;    /**< Input normalization bias. */
    double output_scale;  /**< Output denormalization scale. */
    double output_bias;   /**< Output denormalization bias. */
    SimNeuralDeterminismPolicy determinism_policy; /**< Determinism requested from backend. */
    SimNeuralDeviceRequirement device_requirement; /**< Device requested from backend. */
    SimNeuralPrecisionMode precision_mode;         /**< Precision requested from backend. */
    SimNeuralShapeConstraints shape_constraints;   /**< Accepted input shape constraints. */
    SimNeuralInferenceFn inference_fn;             /**< Optional backend inference callback. */
    void *inference_userdata;                      /**< Opaque pointer passed to inference_fn. */
} SimNeuralBaseConfig;

/**
 * @brief Configuration for inference-only neural operator.
 */
typedef SimNeuralBaseConfig SimNeuralInferOperatorConfig;

/**
 * @brief Register an inference-only neural operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional neural inference configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_neural_infer_operator(struct SimContext *context,
                                        const SimNeuralInferOperatorConfig *config,
                                        size_t *out_index);

/**
 * @brief Copy the current neural inference configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_neural_infer_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no neural state.
 */
SimResult sim_neural_infer_config(struct SimContext *context, size_t operator_index,
                                  SimNeuralInferOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered neural inference operator.
 *
 * @p config is required. The replacement is normalized and field/shape
 * compatibility is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the neural inference operator to update.
 * @param config Replacement neural inference configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field compatibility checks, or state validation.
 */
SimResult sim_neural_infer_update(struct SimContext *context, size_t operator_index,
                                  const SimNeuralInferOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_NEURAL_INFER_H */
