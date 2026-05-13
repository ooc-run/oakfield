/**
 * @file neural_models.h
 * @brief Context-level neural model registry and execution helpers.
 *
 * Neural model registrations bind a stable model id to either an in-process
 * callback, an external process command, or a compiled backend placeholder.
 * Runtime helpers choose devices, apply determinism policy, invoke inference,
 * and keep per-model statistics inside SimContext.
 */
#ifndef OAKFIELD_NEURAL_MODELS_H
#define OAKFIELD_NEURAL_MODELS_H

#include "operators/neural/neural_infer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/** Maximum external-process command length (excluding null terminator). */
#define SIM_NEURAL_MODEL_COMMAND_MAX 255U
/** Maximum free-form model note length (excluding null terminator). */
#define SIM_NEURAL_MODEL_NOTE_MAX 127U
/** Maximum stored runtime error length (excluding null terminator). */
#define SIM_NEURAL_MODEL_ERROR_MAX 191U

/**
 * @brief Supported neural inference backend families.
 */
typedef enum SimNeuralBackendKind {
    SIM_NEURAL_BACKEND_CALLBACK = 0,           /**< In-process SimNeuralInferenceFn callback. */
    SIM_NEURAL_BACKEND_EXTERNAL_PROCESS,       /**< External executable using the process schema. */
    SIM_NEURAL_BACKEND_INPROCESS_ONNX_RUNTIME, /**< ONNX Runtime backend, when compiled in. */
    SIM_NEURAL_BACKEND_INPROCESS_LIBTORCH      /**< LibTorch backend, when compiled in. */
} SimNeuralBackendKind;

/**
 * @brief Execution devices that can be requested for inference.
 */
typedef enum SimNeuralExecutionDevice {
    SIM_NEURAL_EXEC_DEVICE_CPU = 0, /**< CPU execution. */
    SIM_NEURAL_EXEC_DEVICE_CUDA,    /**< CUDA-capable GPU execution. */
    SIM_NEURAL_EXEC_DEVICE_MPS      /**< Apple Metal Performance Shaders execution. */
} SimNeuralExecutionDevice;

#define SIM_NEURAL_DEVICE_MASK_CPU (1U << 0)
#define SIM_NEURAL_DEVICE_MASK_CUDA (1U << 1)
#define SIM_NEURAL_DEVICE_MASK_MPS (1U << 2)

/**
 * @brief Registration-time model metadata and backend configuration.
 */
typedef struct SimNeuralModelConfig {
    char model_id[SIM_NEURAL_MODEL_ID_MAX + 1U]; /**< Required stable model id. */
    SimNeuralBackendKind backend_kind;           /**< Backend route used by inference. */
    uint32_t supported_device_mask; /**< SIM_NEURAL_DEVICE_MASK_* bitset; zero means CPU. */
    SimNeuralPrecisionMode
        default_precision; /**< Precision used when a request does not override it. */
    bool deterministic;    /**< Whether strict determinism requests are allowed. */
    char external_command[SIM_NEURAL_MODEL_COMMAND_MAX + 1U]; /**< Process command. */
    char note[SIM_NEURAL_MODEL_NOTE_MAX + 1U];                /**< Optional free-form note. */
    SimNeuralInferenceFn callback_fn;                         /**< Required for callback backend. */
    void *callback_userdata; /**< User data passed to callback_fn. */
} SimNeuralModelConfig;

/**
 * @brief Accumulated runtime counters for a registered model.
 */
typedef struct SimNeuralModelRuntimeStats {
    uint64_t invocation_count;            /**< Number of attempted inference calls. */
    uint64_t success_count;               /**< Calls returning #SIM_RESULT_OK. */
    uint64_t failure_count;               /**< Calls returning a non-OK result. */
    uint64_t total_elements;              /**< Total input elements observed. */
    double last_wall_ms;                  /**< Wall-clock duration of the latest call. */
    double total_wall_ms;                 /**< Accumulated wall-clock duration. */
    size_t last_step_index;               /**< Context step index at latest call. */
    double last_sim_time;                 /**< Context simulation time at latest call. */
    SimResult last_result;                /**< Result from the latest call. */
    SimNeuralExecutionDevice last_device; /**< Device routed for the latest call. */
    char last_error[SIM_NEURAL_MODEL_ERROR_MAX + 1U]; /**< Latest error text. */
} SimNeuralModelRuntimeStats;

/**
 * @brief Stored registry entry pairing model config with runtime stats.
 */
typedef struct SimNeuralModelEntry {
    SimNeuralModelConfig config;      /**< Registration metadata. */
    SimNeuralModelRuntimeStats stats; /**< Mutable runtime counters. */
} SimNeuralModelEntry;

/**
 * @brief Owning dynamic array of neural model entries.
 */
typedef struct SimNeuralModelRegistry {
    SimNeuralModelEntry *entries; /**< Owned entry storage. */
    size_t count;                 /**< Number of active entries. */
    size_t capacity;              /**< Allocated entry capacity. */
} SimNeuralModelRegistry;

/**
 * @brief Return a default callback/CPU/deterministic model config.
 *
 * @return Zero-initialized config with callback backend, CPU support, default
 *         precision, and deterministic=true.
 */
SimNeuralModelConfig sim_neural_model_config_defaults(void);

/**
 * @brief Reset runtime stats to their initial successful CPU state.
 *
 * @param stats Stats object to reset; NULL is ignored.
 */
void sim_neural_model_runtime_stats_reset(SimNeuralModelRuntimeStats *stats);

/**
 * @brief Return a stable name for a backend kind.
 *
 * Unknown values return "callback" for compatibility with the default backend.
 *
 * @param kind Backend enum value.
 * @return Static lowercase backend name.
 */
const char *sim_neural_backend_kind_name(SimNeuralBackendKind kind);

/**
 * @brief Parse a backend kind name.
 *
 * Accepted aliases include "external-process", "process", "onnx", and
 * "torch". Matching is case-insensitive.
 *
 * @param text Text to parse.
 * @param[out] out_kind Receives the parsed backend kind.
 * @return true when parsing succeeds.
 */
bool sim_neural_backend_kind_from_string(const char *text, SimNeuralBackendKind *out_kind);

/**
 * @brief Return a stable name for an execution device.
 *
 * Unknown values return "cpu" for compatibility with default routing.
 *
 * @param device Device enum value.
 * @return Static lowercase device name.
 */
const char *sim_neural_execution_device_name(SimNeuralExecutionDevice device);

/**
 * @brief Parse an execution device name.
 *
 * Accepted aliases include "gpu" for CUDA and "metal" for MPS. Matching is
 * case-insensitive.
 *
 * @param text Text to parse.
 * @param[out] out_device Receives the parsed device.
 * @return true when parsing succeeds.
 */
bool sim_neural_execution_device_from_string(const char *text,
                                             SimNeuralExecutionDevice *out_device);

/**
 * @brief Test whether a model configuration supports a device.
 *
 * A zero supported_device_mask is treated as CPU-only.
 *
 * @param config Model configuration to inspect.
 * @param device Requested device.
 * @return true when the device bit is supported.
 */
bool sim_neural_model_supports_device(const SimNeuralModelConfig *config,
                                      SimNeuralExecutionDevice device);

/**
 * @brief Initialize an empty model registry.
 *
 * @param registry Registry storage to initialize.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_neural_model_registry_init(SimNeuralModelRegistry *registry);

/**
 * @brief Destroy a model registry and release entry storage.
 *
 * @param registry Registry to destroy; NULL is ignored.
 */
void sim_neural_model_registry_destroy(SimNeuralModelRegistry *registry);

/**
 * @brief Register or replace a neural model in a simulation context.
 *
 * model_id must be non-empty. Callback configs require callback_fn; external
 * process configs require external_command. Re-registering an existing model id
 * replaces its config and resets its stats.
 *
 * @param context Target simulation context.
 * @param config Model configuration to store.
 * @param[out] out_index Optional receiver for the registry index.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_neural_model_register(struct SimContext *context, const SimNeuralModelConfig *config,
                                    size_t *out_index);

/**
 * @brief Return the number of registered models.
 *
 * @param context Context to inspect.
 * @return Model count, or 0 when @p context is NULL.
 */
size_t sim_neural_model_count(const struct SimContext *context);

/**
 * @brief Copy model config by registry index.
 *
 * @param context Context to inspect.
 * @param model_index Registry index in [0, sim_neural_model_count()).
 * @param[out] out_config Receives the stored config.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_neural_model_config_at(const struct SimContext *context, size_t model_index,
                                     SimNeuralModelConfig *out_config);

/**
 * @brief Copy model runtime stats by registry index.
 *
 * @param context Context to inspect.
 * @param model_index Registry index in [0, sim_neural_model_count()).
 * @param[out] out_stats Receives the stored stats.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_neural_model_stats_at(const struct SimContext *context, size_t model_index,
                                    SimNeuralModelRuntimeStats *out_stats);

/**
 * @brief Copy model config by model id.
 *
 * @param context Context to inspect.
 * @param model_id Registered model id.
 * @param[out] out_config Receives the stored config.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_NOT_FOUND.
 */
SimResult sim_neural_model_config(const struct SimContext *context, const char *model_id,
                                  SimNeuralModelConfig *out_config);

/**
 * @brief Copy model runtime stats by model id.
 *
 * @param context Context to inspect.
 * @param model_id Registered model id.
 * @param[out] out_stats Receives the stored stats.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_NOT_FOUND.
 */
SimResult sim_neural_model_stats(const struct SimContext *context, const char *model_id,
                                 SimNeuralModelRuntimeStats *out_stats);

/**
 * @brief Run inference for a registered model.
 *
 * The effective request inherits the stored model id, default precision, and
 * routeable device. Unsupported devices or strict determinism requests against
 * non-deterministic models fail before backend invocation but still update
 * model stats.
 *
 * @param context Context containing the model registry and runtime clock.
 * @param model_id Registered model id.
 * @param input Input field passed to the backend; caller-owned.
 * @param output Output field passed to the backend; caller-owned and mutable.
 * @param request Optional inference request overrides.
 * @return Backend result, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_FOUND,
 *         or #SIM_RESULT_NOT_SUPPORTED for unavailable/unsupported routes.
 */
SimResult sim_neural_model_infer(struct SimContext *context, const char *model_id,
                                 const SimField *input, SimField *output,
                                 const SimNeuralInferenceRequest *request);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_NEURAL_MODELS_H */
