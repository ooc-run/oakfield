/**
 * @file sim_runtime_state.h
 * @brief Dynamic runtime state for simulations.
 */
#ifndef OAKFIELD_SIM_RUNTIME_STATE_H
#define OAKFIELD_SIM_RUNTIME_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "field.h"
#include "sim_field_stats_runtime.h"
#include "sim_field_topology_runtime.h"
#include "sim_flux_lens.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SIM_RUNTIME_DEFAULT_DT
#define SIM_RUNTIME_DEFAULT_DT 0.0166667f
#endif

#define SIM_RUNTIME_VIS_DOWNSAMPLE_TARGET_DEFAULT 4096U
#define SIM_RUNTIME_VIS_DOWNSAMPLE_MAX_DEFAULT 16384U
#define SIM_RUNTIME_VIS_DOWNSAMPLE_TARGET_MIN 64U
#define SIM_RUNTIME_VIS_DOWNSAMPLE_TARGET_MAX 65536U

/**
 * @brief Complex sample stored in runtime visualization and drift caches.
 */
typedef struct SimRuntimeComplexSample {
    double re; /**< Real component. */
    double im; /**< Imaginary component. */
} SimRuntimeComplexSample;

/**
 * @brief Bounds and aggregate radii for phase-portrait visualization samples.
 */
typedef struct SimRuntimePhasePortraitMetrics {
    double mean_re;    /**< Mean real coordinate of sampled points. */
    double mean_im;    /**< Mean imaginary coordinate of sampled points. */
    double max_dev_x;  /**< Maximum absolute horizontal deviation from the mean. */
    double max_dev_y;  /**< Maximum absolute vertical deviation from the mean. */
    double max_dev;    /**< Maximum radial deviation from the mean. */
    double min_radius; /**< Minimum sample radius. */
    double max_radius; /**< Maximum sample radius. */
    double min_r2;     /**< Minimum squared sample radius. */
    double max_r2;     /**< Maximum squared sample radius. */
    double rms_radius; /**< Root-mean-square sample radius. */
    size_t count;      /**< Number of samples included in the metrics. */
} SimRuntimePhasePortraitMetrics;

/**
 * @brief Min/max summary for downsampled waveform visualization data.
 */
typedef struct SimRuntimeWaveformSampleStats {
    double min_value;     /**< Minimum scalar waveform value. */
    double max_value;     /**< Maximum scalar waveform value. */
    double max_magnitude; /**< Maximum absolute waveform value. */
    size_t count;         /**< Number of waveform samples included. */
} SimRuntimeWaveformSampleStats;

/**
 * @brief Time evolution model for the simulation.
 *
 * SIM_TIME_MODEL_CONTINUOUS: dt refinement should converge (ODE/PDE flows).
 * SIM_TIME_MODEL_MAP: dt is part of the discrete map; refinement changes the model.
 */
typedef enum SimTimeModel {
    SIM_TIME_MODEL_CONTINUOUS =
        0,                 /**< Continuous-time model where dt refinement should converge. */
    SIM_TIME_MODEL_MAP = 1 /**< Discrete map where dt is part of the model definition. */
} SimTimeModel;

/** Number of per-step samples preserved in the runtime ring buffer. */
#define SIM_STEP_METRIC_HISTORY 128U

/**
 * @brief Aggregated metrics captured once per completed integration step.
 */
typedef struct SimStepMetrics {
    size_t step_index;           /**< Completed step identifier. */
    double requested_dt;         /**< Integrator-requested dt prior to the step. */
    double accepted_dt;          /**< dt accepted by the integrator for this step. */
    double next_dt;              /**< dt scheduled for the following step (context->runtime.dt). */
    double rms_error;            /**< RMS error reported by the active integrator. */
    uint64_t step_wall_ns;       /**< Total wall time spent executing this step. */
    uint64_t integrator_wall_ns; /**< Wall time spent in integrator dispatch for this step. */
    uint64_t operator_wall_ns;   /**< Wall time spent in operator execution for this step. */
    uint64_t dirty_write_count;  /**< Sum of continuity_dirty_ops across all fields. */
    uint64_t stable_write_count; /**< Sum of continuity_stable_ops across all fields. */
    uint64_t integrator_workspace_bytes;     /**< Scratch workspace footprint owned by the active
                                                integrator. */
    uint64_t integrator_drift_scratch_bytes; /**< Drift snapshot/scratch bytes owned by the active
                                                integrator. */
    uint32_t integrator_attempt_count; /**< Attempts used by the active integrator for this step. */
    uint32_t integrator_rejection_count; /**< Rejected adaptive attempts before acceptance. */
    uint32_t active_warp_mask;           /**< Bit-mask of warp levels touched during the step. */
} SimStepMetrics;

/**
 * @brief Last observed terminal/non-running reason for a managed loop.
 */
typedef enum SimRuntimeLoopStopReason {
    SIM_RUNTIME_LOOP_STOP_REASON_NONE = 0,  /**< No terminal reason has been recorded. */
    SIM_RUNTIME_LOOP_STOP_REASON_PAUSED,    /**< Loop stopped because execution was paused. */
    SIM_RUNTIME_LOOP_STOP_REASON_CANCELLED, /**< Loop stopped because cancellation was requested. */
    SIM_RUNTIME_LOOP_STOP_REASON_STEPS_EXHAUSTED,      /**< Loop consumed its step budget. */
    SIM_RUNTIME_LOOP_STOP_REASON_MAX_SIM_TIME_REACHED, /**< Simulation-time budget reached. */
    SIM_RUNTIME_LOOP_STOP_REASON_MAX_WALL_MS_REACHED,  /**< Wall-clock budget reached. */
    SIM_RUNTIME_LOOP_STOP_REASON_RUNTIME_ERROR         /**< Loop stopped after a runtime error. */
} SimRuntimeLoopStopReason;

/**
 * @brief Origin of the most recent loop/runtime failure.
 */
typedef enum SimRuntimeLoopErrorSource {
    SIM_RUNTIME_LOOP_ERROR_SOURCE_NONE = 0,            /**< No error source has been recorded. */
    SIM_RUNTIME_LOOP_ERROR_SOURCE_SCHEDULER_STEP,      /**< Scheduler step dispatch failed. */
    SIM_RUNTIME_LOOP_ERROR_SOURCE_EXECUTION_FRAME,     /**< Execution-frame setup failed. */
    SIM_RUNTIME_LOOP_ERROR_SOURCE_OPERATOR_EXECUTION,  /**< Operator execution failed. */
    SIM_RUNTIME_LOOP_ERROR_SOURCE_INTEGRATOR_BRIDGE,   /**< Integrator bridge failed. */
    SIM_RUNTIME_LOOP_ERROR_SOURCE_INTEGRATOR_SEQUENCE, /**< Integrator sequence failed. */
    SIM_RUNTIME_LOOP_ERROR_SOURCE_LEGACY_STEP          /**< Legacy stepping path failed. */
} SimRuntimeLoopErrorSource;

/**
 * @brief Structured record of the most recent loop/runtime failure.
 */
typedef struct SimRuntimeLoopError {
    bool valid;                       /**< True when this record contains an error. */
    SimResult code;                   /**< Error code returned by the failing path. */
    SimRuntimeLoopErrorSource source; /**< Component that reported the error. */
    size_t step_index;                /**< Step index associated with the error. */
} SimRuntimeLoopError;

/**
 * @brief Most recent caller-owned stepping primitive observed on the context.
 *
 * This stays intentionally local to the runtime/runner and does not imply a
 * managed scheduler-owned loop. It is used only for descriptive driver-state
 * reporting.
 */
typedef enum SimRuntimeCallerStepMode {
    SIM_RUNTIME_CALLER_STEP_MODE_NONE = 0,       /**< No caller-owned step observed. */
    SIM_RUNTIME_CALLER_STEP_MODE_MANUAL_STEP,    /**< Context manual-step path observed. */
    SIM_RUNTIME_CALLER_STEP_MODE_INTEGRATOR_STEP /**< Integrator-owned step path observed. */
} SimRuntimeCallerStepMode;

/**
 * @brief Kind of caller-owned bounded loop currently being observed.
 */
typedef enum SimRuntimeLoopProgressKind {
    SIM_RUNTIME_LOOP_PROGRESS_NONE = 0,                  /**< No bounded loop is active. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_STEPS,                 /**< sim_context_run_steps-style loop. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_UNTIL,                 /**< sim_context_run_until-style loop. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_INTEGRATOR_STEPS,      /**< Integrator-step-count loop. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_UNTIL_INTEGRATOR,      /**< Integrator run-until loop. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_PARAMETER_SWEEP,       /**< Parameter sweep loop. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_PARAMETER_GRID,        /**< Parameter grid loop. */
    SIM_RUNTIME_LOOP_PROGRESS_RUN_PARAMETER_CONTINUATION /**< Parameter continuation loop. */
} SimRuntimeLoopProgressKind;

/**
 * @brief Snapshot of caller-owned loop progress and declared budgets.
 */
typedef struct SimRuntimeLoopProgress {
    bool active;                     /**< True while a caller-owned loop is active. */
    SimRuntimeLoopProgressKind kind; /**< Active bounded-loop family. */
    uint64_t steps_completed;        /**< Number of completed simulation steps. */
    uint64_t checks_completed;       /**< Number of loop-budget checks performed. */
    uint64_t step_chunk;             /**< Declared per-iteration step chunk. */
    uint64_t step_budget;            /**< Total allowed steps. */
    uint64_t check_budget;           /**< Total allowed loop checks. */
    double sim_time_advanced;        /**< Simulation time advanced by the loop. */
    double wall_time_ms;             /**< Wall-clock time consumed by the loop. */
    double sim_time_budget;          /**< Simulation-time budget in seconds. */
    double wall_time_budget;         /**< Wall-clock budget in milliseconds. */
    double requested_dt;             /**< Requested dt used by the loop. */
    size_t trace_limit;              /**< Maximum retained trace entries. */
    bool has_step_chunk;             /**< True when @ref step_chunk is meaningful. */
    bool has_step_budget;            /**< True when @ref step_budget is meaningful. */
    bool has_check_budget;           /**< True when @ref check_budget is meaningful. */
    bool has_sim_time_budget;        /**< True when @ref sim_time_budget is meaningful. */
    bool has_wall_time_budget;       /**< True when @ref wall_time_budget is meaningful. */
    bool has_requested_dt;           /**< True when @ref requested_dt is meaningful. */
    bool has_trace_limit;            /**< True when @ref trace_limit is meaningful. */
} SimRuntimeLoopProgress;

/**
 * @brief Runtime parameters and integration state.
 */
typedef struct SimRuntimeState {
    double dt;                  /**< Integration timestep. */
    SimTimeModel time_model;    /**< Time evolution semantics (continuous vs map). */
    size_t step_index;          /**< Monotonic step counter for diagnostics. */
    double time_accumulated;    /**< Simulation time accumulated from accepted steps (seconds). */
    uint32_t drift_mode_depth;  /**< >0 when executing drift-only evaluations (no side effects). */
    double drift_time_override; /**< Optional stage time override while in drift. */
    bool drift_time_override_active;     /**< True when sim_context_time should return the drift
                                            override. */
    uint64_t *field_dirty_counts;        /**< Per-field continuity counters (dirty). */
    uint64_t *field_stable_counts;       /**< Per-field continuity counters (stable). */
    double *field_phase_ema;             /**< Per-field phase coherence EMA. */
    double *field_phase_last_time;       /**< Per-field last timestamp used for EMA (seconds). */
    uint8_t *field_phase_lock_state;     /**< Per-field hysteresis lock flag. */
    uint8_t *field_phase_initialized;    /**< Per-field EMA init marker. */
    SimFieldStats *drift_field_stats;    /**< Per-field stats cached from drift evaluation. */
    size_t drift_field_stats_step_index; /**< Step index associated with cached drift stats. */
    bool *drift_field_stats_valid;       /**< Per-field validity bits for cached drift stats. */
    bool *drift_field_stats_requested;   /**< Per-field request flags for drift stats capture. */
    SimComplexDouble **drift_field_snapshots; /**< Per-field snapshots captured during drift. */
    size_t *drift_field_snapshot_capacity;    /**< Per-field allocated element capacity. */
    size_t *drift_field_snapshot_count;       /**< Per-field element count stored. */
    size_t *drift_field_snapshot_step_index;  /**< Per-field step index for snapshot. */
    bool *drift_field_snapshot_valid;         /**< Per-field snapshot validity. */
    bool *drift_field_snapshot_requested;     /**< Per-field snapshot request flags. */
    SimFieldStats *field_stats_cache;         /**< Per-field stats cache (current step). */
    size_t *field_stats_cache_step_index;     /**< Step index associated with cached stats. */
    SimFieldStatsComputeConfig
        field_stats_config; /**< Feature mask for optional field stats work. */
    SimFieldStatsRuntimeProfile
        field_stats_profile; /**< Rolling field stats timing/cache profile. */
    SimFieldTopologyRuntimeState *field_topology_cache; /**< Per-field topology cache/state. */
    size_t *field_health_cache_step_index; /**< Step index associated with cached health counts. */
    size_t *field_health_nan_counts;       /**< Per-field NaN counts (cached). */
    size_t *field_health_inf_counts;       /**< Per-field Inf counts (cached). */
    size_t measurement_cache_step_index; /**< Step index associated with cached measurement taps. */
    bool measurement_cache_valid;        /**< True when cached measurement taps are current. */
    double measurement_energy;           /**< Cached global energy measurement. */
    double measurement_dissipation;      /**< Cached global dissipation measurement. */
    double measurement_remainder;        /**< Cached remainder norm measurement. */
    size_t measurement_remainder_sources;          /**< Cached remainder source count. */
    bool measurement_energy_valid;                 /**< Cached energy tap validity. */
    bool measurement_dissipation_valid;            /**< Cached dissipation tap validity. */
    bool measurement_remainder_valid;              /**< Cached remainder tap validity. */
    SimRuntimeComplexSample **visual_sample_cache; /**< Per-field downsampled sample cache. */
    size_t *visual_sample_cache_capacity;          /**< Per-field allocated sample capacity. */
    size_t *visual_sample_cache_count;             /**< Per-field cached sample count. */
    size_t *visual_sample_cache_step_index;   /**< Step index associated with cached samples. */
    size_t *visual_sample_cache_source_count; /**< Source element count used for cached samples. */
    size_t *visual_sample_cache_stride;       /**< Stride used while producing cached samples. */
    bool *visual_sample_cache_valid;          /**< Per-field validity marker for cached samples. */
    SimRuntimePhasePortraitMetrics
        *phase_portrait_metrics_cache;               /**< Per-field phase-portrait metrics cache. */
    size_t *phase_portrait_metrics_cache_step_index; /**< Step index associated with cached
                                                        phase-portrait metrics. */
    size_t *phase_portrait_metrics_cache_sample_count; /**< Downsampled sample count used for cached
                                                          phase-portrait metrics. */
    bool *phase_portrait_metrics_cache_valid;          /**< Per-field validity marker for cached
                                                          phase-portrait metrics. */
    SimRuntimeWaveformSampleStats
        *waveform_sample_stats_cache;               /**< Per-field waveform sample stats cache. */
    size_t *waveform_sample_stats_cache_step_index; /**< Step index associated with cached waveform
                                                       sample stats. */
    size_t *waveform_sample_stats_cache_sample_count; /**< Sample count used for cached waveform
                                                         sample stats. */
    bool *waveform_sample_stats_cache_valid; /**< Per-field validity marker for cached waveform
                                                sample stats. */
    size_t
        visual_sample_target_samples; /**< Canonical target sample count for visual downsample. */
    size_t visual_sample_max_samples; /**< Upper bound for canonical target sample count. */
    size_t continuity_capacity;       /**< Allocated length for continuity counter arrays. */
    uint32_t current_step_warp_mask;  /**< Warp levels observed during the active step. */
    FluxLensState flux_lens;          /**< Flux lens state (single lens per context). */
    FluxLensWorkspace flux_workspace; /**< Scratch workspace for flux computations. */
    SimStepMetrics
        step_metrics[SIM_STEP_METRIC_HISTORY]; /**< Circular buffer of recent step metrics. */
    size_t step_metrics_head;                  /**< Next write index inside the metrics buffer. */
    size_t step_metrics_count;                 /**< Number of populated entries (<= history). */
    bool step_metrics_valid; /**< True once at least one metrics sample has been recorded. */
    SimStepMetrics step_metrics_latest;        /**< Most recent metrics sample. */
    bool pending_cancel;                       /**< True while a cancel request is being applied. */
    SimRuntimeLoopStopReason last_stop_reason; /**< Most recent terminal/non-running reason. */
    SimRuntimeLoopError last_loop_error;       /**< Most recent runtime/loop error. */
    SimRuntimeCallerStepMode
        last_caller_step_mode; /**< Most recent caller-owned stepping primitive observed. */
    size_t
        caller_step_streak; /**< Consecutive caller-owned steps observed with the same primitive. */
    uint32_t external_driver_depth; /**< Nested caller-owned external-driver claims. */
    SimRuntimeLoopProgress
        loop_progress; /**< Current caller-owned bounded-loop progress snapshot. */
    SimRuntimeLoopProgress
        last_loop_progress;        /**< Most recent completed caller-owned bounded-loop snapshot. */
    bool last_loop_progress_valid; /**< True when last_loop_progress contains a retained final
                                      snapshot. */
} SimRuntimeState;

/**
 * @brief Initialize runtime state with defaults.
 *
 * @param state Runtime state to initialize.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_runtime_state_init(SimRuntimeState *state);

/**
 * @brief Release continuity buffer allocations.
 *
 * @param state Runtime state to update; NULL is ignored.
 */
void sim_runtime_state_release_continuity_buffers(SimRuntimeState *state);

/**
 * @brief Ensure continuity counters can index at least @p required fields.
 *
 * @param state Runtime state to grow.
 * @param required Required number of field entries.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_runtime_state_ensure_continuity_capacity(SimRuntimeState *state, size_t required);

/**
 * @brief Destroy runtime state and release resources.
 *
 * @param state Runtime state to destroy; NULL is ignored.
 */
void sim_runtime_state_destroy(SimRuntimeState *state);

/**
 * @brief Append a step metrics sample to the runtime ring buffer.
 *
 * @param state Runtime state to update.
 * @param metrics Metrics sample to copy; NULL is ignored.
 */
void sim_runtime_state_record_step_metrics(SimRuntimeState *state, const SimStepMetrics *metrics);

/**
 * @brief Copy the most recent step metrics sample.
 *
 * @param state Runtime state to inspect.
 * @param[out] out_metrics Receives the latest sample.
 * @return true when a sample was available.
 */
bool sim_runtime_state_latest_step_metrics(const SimRuntimeState *state,
                                           SimStepMetrics *out_metrics);

/**
 * @brief Copy up to @p capacity samples into @p dest ordered from oldest to newest.
 *
 * @param state Runtime state to inspect.
 * @param[out] dest Destination array for copied samples.
 * @param capacity Number of entries available in @p dest.
 * @return Number of samples written.
 */
size_t sim_runtime_state_copy_step_metrics(const SimRuntimeState *state, SimStepMetrics *dest,
                                           size_t capacity);

/**
 * @brief Mark whether a scheduler-owned cancel is in progress.
 *
 * @param state Runtime state to update.
 * @param pending Pending-cancel flag value.
 */
void sim_runtime_state_set_pending_cancel(SimRuntimeState *state, bool pending);

/**
 * @brief Read the current pending-cancel flag.
 *
 * @param state Runtime state to inspect.
 * @return true when a cancel request is pending.
 */
bool sim_runtime_state_pending_cancel(const SimRuntimeState *state);

/**
 * @brief Record the most recent loop stop reason.
 *
 * @param state Runtime state to update.
 * @param reason Stop reason to store.
 */
void sim_runtime_state_set_last_stop_reason(SimRuntimeState *state,
                                            SimRuntimeLoopStopReason reason);

/**
 * @brief Fetch the most recent loop stop reason.
 *
 * @param state Runtime state to inspect.
 * @return Stored reason, or `SIM_RUNTIME_LOOP_STOP_REASON_NONE` for NULL state.
 */
SimRuntimeLoopStopReason sim_runtime_state_last_stop_reason(const SimRuntimeState *state);

/**
 * @brief Clear the most recent loop/runtime error record.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_clear_loop_error(SimRuntimeState *state);

/**
 * @brief Record the most recent loop/runtime error.
 *
 * @param state Runtime state to update.
 * @param code Error result code.
 * @param source Subsystem that produced the error.
 * @param step_index Step index associated with the error.
 */
void sim_runtime_state_record_loop_error(SimRuntimeState *state, SimResult code,
                                         SimRuntimeLoopErrorSource source, size_t step_index);

/**
 * @brief Fetch the most recent loop/runtime error.
 *
 * @param state Runtime state to inspect.
 * @param[out] out_error Receives the error record.
 * @return true when a valid error record was available.
 */
bool sim_runtime_state_last_loop_error(const SimRuntimeState *state,
                                       SimRuntimeLoopError *out_error);

/**
 * @brief Record a successful caller-owned step primitive.
 *
 * @param state Runtime state to update.
 * @param mode Step primitive to record; NONE is ignored.
 */
void sim_runtime_state_note_caller_step(SimRuntimeState *state, SimRuntimeCallerStepMode mode);

/**
 * @brief Clear descriptive caller-owned step history.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_clear_caller_step_history(SimRuntimeState *state);

/**
 * @brief Fetch the last caller-owned step primitive observed.
 *
 * @param state Runtime state to inspect.
 * @return Last mode, or `SIM_RUNTIME_CALLER_STEP_MODE_NONE` for NULL state.
 */
SimRuntimeCallerStepMode sim_runtime_state_last_caller_step_mode(const SimRuntimeState *state);

/**
 * @brief Fetch the consecutive caller-owned step streak.
 *
 * @param state Runtime state to inspect.
 * @return Consecutive streak count, or 0 for NULL state.
 */
size_t sim_runtime_state_caller_step_streak(const SimRuntimeState *state);

/**
 * @brief Acquire a caller-owned external-driver claim.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_acquire_external_driver(SimRuntimeState *state);

/**
 * @brief Release one caller-owned external-driver claim.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_release_external_driver(SimRuntimeState *state);

/**
 * @brief Clear all caller-owned external-driver claims.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_clear_external_driver(SimRuntimeState *state);

/**
 * @brief Report the current nested external-driver claim depth.
 *
 * @param state Runtime state to inspect.
 * @return Current claim depth, or 0 for NULL state.
 */
size_t sim_runtime_state_external_driver_depth(const SimRuntimeState *state);

/**
 * @brief Replace the current caller-owned loop progress snapshot.
 *
 * @param state Runtime state to update.
 * @param progress Progress snapshot to copy; NULL clears active progress.
 */
void sim_runtime_state_set_loop_progress(SimRuntimeState *state,
                                         const SimRuntimeLoopProgress *progress);

/**
 * @brief Clear the current caller-owned loop progress snapshot.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_clear_loop_progress(SimRuntimeState *state);

/**
 * @brief Fetch the current caller-owned loop progress snapshot.
 *
 * @param state Runtime state to inspect.
 * @param[out] out_progress Receives the active progress snapshot.
 * @return true when a progress snapshot is active.
 */
bool sim_runtime_state_loop_progress(const SimRuntimeState *state,
                                     SimRuntimeLoopProgress *out_progress);

/**
 * @brief Fetch the most recent completed caller-owned loop progress snapshot.
 *
 * @param state Runtime state to inspect.
 * @param[out] out_progress Receives the retained final progress snapshot.
 * @return true when a retained final progress snapshot is available.
 */
bool sim_runtime_state_last_loop_progress(const SimRuntimeState *state,
                                          SimRuntimeLoopProgress *out_progress);

/**
 * @brief Clear any retained completed caller-owned loop progress snapshot.
 *
 * @param state Runtime state to update.
 */
void sim_runtime_state_clear_last_loop_progress(SimRuntimeState *state);

/**
 * @brief Convert a loop stop reason to a stable machine-readable name.
 *
 * @param reason Stop reason enum value.
 * @return Static lowercase name.
 */
const char *sim_runtime_loop_stop_reason_name(SimRuntimeLoopStopReason reason);

/**
 * @brief Convert a loop error source to a stable machine-readable name.
 *
 * @param source Error source enum value.
 * @return Static lowercase name.
 */
const char *sim_runtime_loop_error_source_name(SimRuntimeLoopErrorSource source);

/**
 * @brief Convert a caller-owned step mode to a stable machine-readable name.
 *
 * @param mode Caller step mode enum value.
 * @return Static lowercase name.
 */
const char *sim_runtime_caller_step_mode_name(SimRuntimeCallerStepMode mode);

/**
 * @brief Convert a loop-progress kind to a stable machine-readable name.
 *
 * @param kind Loop-progress kind enum value.
 * @return Static lowercase name.
 */
const char *sim_runtime_loop_progress_kind_name(SimRuntimeLoopProgressKind kind);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_RUNTIME_STATE_H */
