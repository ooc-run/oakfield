/**
 * @file sim_context.h
 * @brief Runtime container coordinating fields, operators, and execution.
 * @ingroup oakfield_contexts
 *
 * @details A SimContext owns registered fields, operator descriptors, scheduler
 * plan state, diagnostics, integrator bindings, and runtime counters. Fields
 * added with sim_context_add_field() transfer ownership to the context; backend
 * and integrator pointers remain caller-owned and must outlive any context step
 * that uses them.
 */
#ifndef OAKFIELD_SIM_CONTEXT_H
#define OAKFIELD_SIM_CONTEXT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "async_logger.h"
#include "field.h"
#include "kernel_ir.h"
#include "math/special_functions.h"
#include "neural_models.h"
#include "operator.h"
#include "sim_diagnostics.h"
#include "sim_integrator_state.h"
#include "sim_profiler.h"
#include "sim_runtime_state.h"
#include "sim_scheduler_state.h"
#include "sim_seed.h"
#include "sim_world.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Integrator;
struct SimBackend;
struct SimOperator;

/** Default maximum number of elements allowed in a single field (0 disables). */
#define SIM_CONTEXT_MAX_FIELD_ELEMENTS_DEFAULT (64U * 1024U * 1024U)
/** Default maximum bytes allowed in a single field (0 disables). */
#define SIM_CONTEXT_MAX_FIELD_BYTES_DEFAULT (512U * 1024U * 1024U)
/** Default maximum number of fields in a context (0 disables). */
#define SIM_CONTEXT_MAX_FIELDS_DEFAULT 256U
/** Default maximum total bytes across all fields in a context (0 disables). */
#define SIM_CONTEXT_MAX_TOTAL_FIELD_BYTES_DEFAULT (2ULL * 1024ULL * 1024ULL * 1024ULL)
/** Number of preferred visual field selection bits retained in the context. */
#define SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY 256U
/** Default maximum scratch bytes per operator (0 disables). */
#define SIM_CONTEXT_MAX_SCRATCH_BYTES_PER_OPERATOR_DEFAULT (256U * 1024U * 1024U)

/**
 * @brief Memory limits enforced by the simulation context.
 *
 * Set values to 0 to disable a particular limit.
 */
typedef struct SimContextMemoryLimits {
    size_t max_field_elements;             /**< Max element count per field. */
    size_t max_field_bytes;                /**< Max bytes per field. */
    size_t max_fields;                     /**< Max field count per context. */
    size_t max_total_field_bytes;          /**< Max total bytes across all fields. */
    size_t max_scratch_bytes_per_operator; /**< Max scratch bytes per operator. */
} SimContextMemoryLimits;

/**
 * @brief Simulation runtime state.
 */
typedef struct SimContext {
    SimWorld world;                 /**< Mostly-static world state. */
    SimRuntimeState runtime;        /**< Dynamic runtime state. */
    SimSchedulerPlan scheduler;     /**< Execution plan cache and backend binding. */
    SimIntegratorState integrators; /**< Integrator registry and active pointer. */
    SimDiagnostics diag;            /**< Diagnostics and fault handling. */
    SimProfiler profiler;           /**< Single-thread profiler for direct execution path. */
    bool profiler_ready;            /**< True when profiler counters are configured. */
    uint64_t base_seed;             /**< Base seed used for deterministic RNG streams. */
    SimRepresentationMode representation_mode; /**< Default representation mode for operators. */
    SimContextMemoryLimits memory_limits;      /**< Memory limits for fields/scratch. */
    size_t bytes_fields_in_use;                /**< Field bytes currently counted. */
    size_t bytes_scratch_in_use;               /**< Scratch bytes currently counted. */
    size_t bytes_total_in_use;                 /**< Total bytes currently counted. */

    bool continuity_override_enabled;      /**< True when global continuity override is active. */
    SimOperatorConfig continuity_override; /**< Override applied to newly registered operators. */

    /* Optional GUI preference: preferred visual and phase modes exposed to UI. -1 when unspecified.
     */
    int preferred_gui_visual_mode;        /**< Preferred visual rendering mode, or -1 unset. */
    int preferred_gui_phase_mode;         /**< Preferred phase rendering mode, or -1 unset. */
    int preferred_gui_visual_auto_scale;  /**< Preferred auto-scale toggle, or -1 unset. */
    double preferred_gui_visual_scale;    /**< Preferred manual visual scale. */
    int preferred_gui_visual_field_index; /**< Preferred visual field index, or -1 unset. */
    bool preferred_gui_visual_field_selected
        [SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY]; /**< Preferred per-field selection bits. */

    void (*log_fn)(SimLogLevel level, const char *message,
                   void *userdata); /**< Optional log hook. */

    void *log_userdata;                   /**< Userdata forwarded to log hook. */
    SimNeuralModelRegistry neural_models; /**< Registered neural models and runtime stats. */
} SimContext;

/**
 * @brief Set the default representation mode for the context.
 */
void sim_context_set_representation_mode(SimContext *context, SimRepresentationMode mode);

/**
 * @brief Get the default representation mode for the context.
 */
SimRepresentationMode sim_context_representation_mode(const SimContext *context);

/**
 * @brief Attach a logger callback to the context (optional).
 */
void sim_context_set_logger(SimContext *context,
                            void (*log_fn)(SimLogLevel level, const char *message, void *userdata),
                            void *userdata);

/**
 * @brief Emit a warning through the context logger (or stderr fallback).
 */
void sim_context_log_warning(const SimContext *context, const char *fmt, ...);

/**
 * @brief Return true when the current representation mode permits the given determinism flags.
 */
bool sim_context_allows_determinism(const SimContext *context, SimDeterminismFlags flags);

/**
 * @brief Return true when a kernel-backed operator is allowed under the current policy.
 */
bool sim_context_kernel_allowed(const SimContext *context, uint64_t required_features,
                                SimDeterminismFlags determinism_flags);

/**
 * @brief Return true when a kernel-backed operator is allowed under an explicit mode.
 */
bool sim_context_kernel_allowed_mode(const SimContext *context, SimRepresentationMode mode,
                                     uint64_t required_features,
                                     SimDeterminismFlags determinism_flags);

/**
 * @brief Initialize a simulation context with default universe parameters.
 *
 * @param[out] context Context to initialize.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_context_init(SimContext *context);

/**
 * @brief Initialize a simulation context with universe specification.
 *
 * @param[out] context Context to initialize.
 * @param universe_spec Universe specification (copied into context); may be NULL for defaults.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_context_init_with_universe(SimContext *context, const SimUniverseSpec *universe_spec);

/**
 * @brief Set the base seed used for deterministic RNG streams.
 */
void sim_context_set_seed(SimContext *context, uint64_t seed);

/**
 * @brief Get the base seed used for deterministic RNG streams.
 */
uint64_t sim_context_seed(const SimContext *context);

/**
 * @brief Configure memory limits for fields and scratch allocations.
 */
void sim_context_set_memory_limits(SimContext *context, const SimContextMemoryLimits *limits);

/**
 * @brief Fetch current memory limit settings.
 */
void sim_context_get_memory_limits(const SimContext *context, SimContextMemoryLimits *out_limits);

/**
 * @brief Report current memory usage counters.
 */
void sim_context_memory_usage(const SimContext *context, size_t *out_fields, size_t *out_scratch,
                              size_t *out_total);

/**
 * @brief Check if a field allocation would violate configured limits.
 */
SimResult sim_context_check_field_limits(const SimContext *context, size_t element_count,
                                         size_t field_bytes);

/**
 * @brief Reserve scratch bytes against the context counters.
 */
SimResult sim_context_reserve_scratch(SimContext *context, size_t bytes);

/**
 * @brief Release previously reserved scratch bytes.
 */
void sim_context_release_scratch(SimContext *context, size_t bytes);

/**
 * @brief Destroy a simulation context and free owned resources.
 *
 * @param context Context to destroy; may be NULL.
 */
void sim_context_destroy(SimContext *context);

/**
 * @brief Add a field to the context, transferring ownership.
 *
 * @param context Target context.
 * @param field Field instance; must be initialized. Ownership moves on success.
 * @param[out] out_index Optional pointer receiving the field index.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_context_add_field(SimContext *context, SimField *field, size_t *out_index);

/**
 * @brief Access a field by index.
 *
 * @param context Context instance.
 * @param index Field index.
 * @return Pointer to the field or NULL if out of range.
 */
SimField *sim_context_field(SimContext *context, size_t index);

/**
 * @brief Returns the number of fields currently owned by the context.
 *
 * Safe to call with NULL; returns 0 in that case.
 */
size_t sim_context_field_count(const SimContext *context);

/**
 * @brief Register a new operator within the context.
 *
 * @param context Context instance.
 * @param descriptor Operator descriptor.
 * @param[out] out_index Optional pointer receiving the assigned index.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_context_register_operator(SimContext *context,
                                        const SimOperatorDescriptor *descriptor, size_t *out_index);

/**
 * @brief Add an explicit dependency edge between two registered operators.
 *
 * Adds @p dependency_index to the dependency list of @p operator_index.
 * Duplicate edges are ignored.
 */
SimResult sim_context_add_operator_dependency(SimContext *context, size_t operator_index,
                                              size_t dependency_index);

/**
 * @brief Ensure the execution plan is up to date.
 *
 * @param context Context instance.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_context_prepare_plan(SimContext *context);

/**
 * @brief Execute the operator plan sequentially.
 *
 * @param context Context instance.
 * @return #SIM_RESULT_OK on success; propagates operator errors.
 */
SimResult sim_context_execute(SimContext *context);

/**
 * @brief Execute the current operator plan without re-running plan preparation.
 *
 * Callers must ensure the context plan is already valid via @ref sim_context_prepare_plan.
 */
SimResult sim_context_execute_prepared(SimContext *context);

/**
 * @brief Access the shared IR builder.
 *
 * @param context Context instance.
 * @return Pointer to the IR builder or NULL if @p context is NULL.
 */
SimIRBuilder *sim_context_ir_builder(SimContext *context);

/**
 * @brief Assigns the integrator responsible for advancing the context.
 *
 * Ownership remains with the caller; the context only stores the pointer.
 */
void sim_context_set_integrator(SimContext *context, struct Integrator *integrator);

/**
 * @brief Retrieves the currently bound integrator.
 */
struct Integrator *sim_context_integrator(SimContext *context);

/**
 * @brief Assign an ordered sequence of integrators for opt-in multi-integrator stepping.
 *
 * Passing @p count = 0 clears the sequence and detaches the active integrator.
 */
SimResult sim_context_set_integrator_sequence(SimContext *context,
                                              struct Integrator *const *integrators, size_t count);

/**
 * @brief Returns the number of integrators in the optional sequence.
 */
size_t sim_context_integrator_sequence_count(const SimContext *context);

/**
 * @brief Returns the integrator at @p index in the optional sequence.
 */
struct Integrator *sim_context_integrator_sequence_at(const SimContext *context, size_t index);

/**
 * @brief Assigns the backend used for kernel execution.
 */
void sim_context_set_backend(SimContext *context, struct SimBackend *backend);

/**
 * @brief Returns the active compute backend.
 */
struct SimBackend *sim_context_backend(SimContext *context);

/**
 * @brief Overrides the integration timestep stored in the context.
 */
void sim_context_set_timestep(SimContext *context, double dt);

/**
 * @brief Queries the integration timestep currently used by the context.
 */
double sim_context_timestep(const SimContext *context);

/**
 * @brief Configure optional field-stat feature families.
 */
void sim_context_set_field_stats_features(SimContext *context, uint32_t feature_mask);
uint32_t sim_context_field_stats_features(const SimContext *context);
void sim_context_reset_field_stats_profile(SimContext *context);
bool sim_context_field_stats_profile(const SimContext *context,
                                     SimFieldStatsRuntimeProfile *out_profile);

/**
 * @brief Set the time evolution model for the simulation (continuous vs map).
 *
 * Changing after steps have executed will emit a warning but still apply.
 */
void sim_context_set_time_model(SimContext *context, SimTimeModel model);

/**
 * @brief Enter/exit drift mode to suppress side effects during derivative evaluation.
 */
void sim_context_begin_drift(SimContext *context);
void sim_context_end_drift(SimContext *context);
bool sim_context_in_drift(const SimContext *context);
void sim_context_set_drift_time_override(SimContext *context, double time_value);
void sim_context_clear_drift_time_override(SimContext *context);

/**
 * @brief Returns the number of completed integration steps.
 */
size_t sim_context_step_index(const SimContext *context);

/**
 * @brief Returns the simulation time accumulated so far (seconds).
 */
double sim_context_time(const SimContext *context);

/**
 * @brief Capture per-step metrics (invoked by scheduler/integrators).
 */
void sim_context_record_step_metrics(SimContext *context, double requested_dt, double accepted_dt,
                                     double rms_error);

/**
 * @brief Capture per-step metrics with explicit wall-time attribution.
 */
void sim_context_record_step_metrics_with_timing(SimContext *context, double requested_dt,
                                                 double accepted_dt, double rms_error,
                                                 uint64_t step_wall_ns, uint64_t integrator_wall_ns,
                                                 uint64_t operator_wall_ns);

/**
 * @brief Advance runtime counters after an accepted step.
 *
 * Allows reuse across scheduler and scripting entry points.
 *
 * @param context Context whose runtime counters are updated.
 * @param accepted_dt Positive timestep accepted by the integrator.
 */
void sim_context_accept_step(SimContext *context, double accepted_dt);

/**
 * @brief Fetch the most recent step metrics sample, if available.
 *
 * @param context Context to inspect.
 * @param[out] out_metrics Receives the latest metrics sample.
 * @return true when a metrics sample was available.
 */
bool sim_context_latest_step_metrics(const SimContext *context, SimStepMetrics *out_metrics);

/**
 * @brief Copy a slice of the step-metrics history ordered oldest to newest.
 *
 * @param context Context to inspect.
 * @param[out] dest Destination array for copied samples.
 * @param capacity Number of entries available in @p dest.
 * @return Number of samples written.
 */
size_t sim_context_step_metrics_history(const SimContext *context, SimStepMetrics *dest,
                                        size_t capacity);

/** Retrieve profiler counters for the direct-execution profiler (non-scheduler). */
bool sim_context_profiler_counters(SimContext *context, SimProfilerCounter *out_counters,
                                   size_t capacity, size_t *out_count);

/** Snapshot the direct-execution profiler (non-scheduler). */
bool sim_context_profiler_snapshot(SimContext *context, SimProfilerSnapshot *out_snapshot);

/**
 * @brief Internal hook used by split operators to relay substep feedback.
 */
void sim_split_notify_integrator(struct SimContext *context, double dt_sub, double error_estimate);
/**
 * @brief Get hyperexponential truncation level from universe.
 *
 * @param context Context instance.
 * @return K parameter; returns 0 if context is NULL.
 */
size_t sim_context_truncation_level(const SimContext *context);

/**
 * @brief Get epsilon parameter from universe.
 */
double sim_context_epsilon(const SimContext *context);

const SimPole *sim_context_poles(const SimContext *context, size_t *out_count);
SimPoleFieldOptions sim_pole_field_options_default(void);
SimResult sim_context_synthesize_pole_field(struct SimContext *context, size_t field_index,
                                            const SimPoleFieldOptions *options);

/** @brief Override the fallback invoked when special functions fail. */
void sim_context_set_special_fallback(SimContext *context, SimSpecialFallbackFn fallback,
                                      void *userdata);

/** Set preferred GUI visual mode on the context (negative = unset). */
void sim_context_set_preferred_visual_mode(SimContext *context, int mode);

/** Get preferred GUI visual mode; returns negative when unset. */
int sim_context_preferred_visual_mode(const SimContext *context);

/* Set preferred Phase Mode (negative = unset) */
void sim_context_set_preferred_phase_mode(SimContext *context, int mode);

/* Get preferred Phase Mode; returns negative when unset. */
int sim_context_preferred_phase_mode(const SimContext *context);

/* Set preferred visual auto-scale state (-1 unset, 0 disabled, 1 enabled). */
void sim_context_set_preferred_visual_auto_scale(SimContext *context, int enabled);

/* Get preferred visual auto-scale state; returns negative when unset. */
int sim_context_preferred_visual_auto_scale(const SimContext *context);

/* Set preferred visual scale (positive value, negative to unset). */
void sim_context_set_preferred_visual_scale(SimContext *context, double scale);

/* Get preferred visual scale; returns negative when unset. */
double sim_context_preferred_visual_scale(const SimContext *context);

/* Set preferred visual field selection state for a given field index. */
void sim_context_set_preferred_visual_field_enabled(SimContext *context, size_t field_index,
                                                    bool enabled);

/* Get preferred visual field selection state for a given field index. */
bool sim_context_preferred_visual_field_enabled(const SimContext *context, size_t field_index);

/** @brief Retrieve the context-managed fallback hook for safe helpers. */
void sim_context_special_fallback_hook(const SimContext *context,
                                       SimSpecialFallbackFn *out_fallback, void **out_userdata);

/** @brief Total number of special-function faults observed so far. */
uint64_t sim_context_special_fault_count(const SimContext *context);

/** @brief Copy the most recent special-function fault report, if any. */
bool sim_context_last_special_fault(const SimContext *context, SimSpecialEvalReport *out_report);

/** @brief Reset accumulated special-function fault statistics. */
void sim_context_clear_special_faults(SimContext *context);

/**
 * @brief Access diagnostics container.
 */
SimDiagnostics *sim_context_diagnostics(SimContext *context);
const SimDiagnostics *sim_context_diagnostics_const(const SimContext *context);

#if SIM_DIAGNOSTICS
/**
 * @brief Flush per-thread special-function diagnostics into the context accumulator.
 *
 * Intended to be called once per completed integration step to avoid hot-loop logging.
 */
void sim_context_flush_special_diagnostics(SimContext *context);
#endif

SimResult sim_context_apply_operator(SimContext *context, struct SimOperator *op);
void sim_context_reset_continuity_counters(SimContext *context);
void sim_context_set_continuity_override(SimContext *context, bool enabled,
                                         const SimOperatorConfig *config);

/**
 * @brief Retrieve the dirty/stable continuity counters for a field.
 */
void sim_context_field_continuity_counts(const SimContext *context, size_t field_index,
                                         uint64_t *out_dirty, uint64_t *out_stable);

/* Optional accessors for counts to reduce internal coupling */
size_t sim_context_operator_count(const SimContext *context);
size_t sim_context_plan_operator_count(const SimContext *context);
bool sim_context_plan_is_valid(const SimContext *context);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_CONTEXT_H */
