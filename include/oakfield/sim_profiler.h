/**
 * @file sim_profiler.h
 * @brief Lightweight performance profiler for libsimcore runtime execution.
 */
#ifndef OAKFIELD_SIM_PROFILER_H
#define OAKFIELD_SIM_PROFILER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Snapshot of accumulated frame statistics.
 */
typedef struct SimProfilerSnapshot {
    uint64_t frame_start_ns;    /**< Monotonic timestamp when the frame began. */
    uint64_t frame_end_ns;      /**< Monotonic timestamp when the frame finished. */
    uint64_t total_ns;          /**< Total duration of the frame in nanoseconds. */
    double average_operator_ns; /**< Average operator execution time. */
} SimProfilerSnapshot;

/**
 * @brief Per-operator statistics tracked during profiling.
 */
typedef struct SimProfilerCounter {
    uint64_t inclusive_ns;       /**< Time accumulated for the operator within the frame. */
    uint64_t invocations;        /**< Number of times the operator was executed. */
    double delta_rms_sum;        /**< Sum of squared RMS deltas recorded (for averaging). */
    uint64_t delta_sample_count; /**< Number of samples aggregated for delta RMS. */
} SimProfilerCounter;

/**
 * @brief Profiler state used by the runtime scheduler.
 */
typedef struct SimProfiler {
    SimProfilerCounter *counters; /**< Per-operator counters. */
    size_t counter_count;         /**< Number of counters allocated. */
    bool frame_active;            /**< Indicates the profiler has an active frame. */
    uint64_t frame_start_ns;      /**< Frame start timestamp. */
    uint64_t frame_end_ns;        /**< Frame end timestamp. */
} SimProfiler;

/**
 * @brief Initialize the profiler for @p operators tracked counters.
 *
 * @param[out] profiler Profiler instance to initialize.
 * @param operators Number of operators to track.
 * @return #SIM_RESULT_OK on success or an error code otherwise.
 */
SimResult sim_profiler_init(SimProfiler *profiler, size_t operators);

/**
 * @brief Release resources held by the profiler.
 *
 * @param profiler Profiler instance; may be NULL.
 */
void sim_profiler_destroy(SimProfiler *profiler);

/**
 * @brief Begin a new profiling frame.
 *
 * @param profiler Profiler instance.
 */
void sim_profiler_begin_frame(SimProfiler *profiler);

/**
 * @brief Complete the active profiling frame.
 *
 * @param profiler Profiler instance.
 */
void sim_profiler_end_frame(SimProfiler *profiler);

/**
 * @brief Record the execution time for an operator.
 *
 * @param profiler Profiler instance.
 * @param operator_index Index of the operator within the execution plan.
 * @param duration_ns Execution duration in nanoseconds.
 */
void sim_profiler_record_operator(SimProfiler *profiler, size_t operator_index,
                                  uint64_t duration_ns);

/**
 * @brief Record a delta RMS contribution for an operator.
 *
 * @param profiler Profiler instance.
 * @param operator_index Index of the operator within the plan.
 * @param delta_rms Root-mean-square of the operator's field change (single sample or aggregated).
 * @param sample_count Number of samples used to estimate delta_rms.
 */
void sim_profiler_record_operator_delta(SimProfiler *profiler, size_t operator_index,
                                        double delta_rms, uint64_t sample_count);

/**
 * @brief Populate a snapshot of the most recent frame statistics.
 *
 * @param profiler Profiler instance.
 * @param[out] out_snapshot Destination snapshot structure.
 * @return #SIM_RESULT_OK when data is available, otherwise #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_profiler_snapshot(const SimProfiler *profiler, SimProfilerSnapshot *out_snapshot);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_PROFILER_H */
