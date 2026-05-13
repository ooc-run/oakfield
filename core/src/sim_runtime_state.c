/**
 * @file sim_runtime_state.c
 * @brief Dynamic runtime state implementation.
 */
#include "oakfield/sim_runtime_state.h"

#include <stdlib.h>
#include <string.h>

static void sim_runtime_state_reset_loop_metadata(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    state->pending_cancel = false;
    state->last_stop_reason = SIM_RUNTIME_LOOP_STOP_REASON_NONE;
    state->last_loop_error.valid = false;
    state->last_loop_error.code = SIM_RESULT_OK;
    state->last_loop_error.source = SIM_RUNTIME_LOOP_ERROR_SOURCE_NONE;
    state->last_loop_error.step_index = 0U;
    state->last_caller_step_mode = SIM_RUNTIME_CALLER_STEP_MODE_NONE;
    state->caller_step_streak = 0U;
    state->external_driver_depth = 0U;
    (void) memset(&state->loop_progress, 0, sizeof(state->loop_progress));
    state->loop_progress.kind = SIM_RUNTIME_LOOP_PROGRESS_NONE;
    (void) memset(&state->last_loop_progress, 0, sizeof(state->last_loop_progress));
    state->last_loop_progress.kind = SIM_RUNTIME_LOOP_PROGRESS_NONE;
    state->last_loop_progress_valid = false;
}

SimResult sim_runtime_state_init(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->dt = SIM_RUNTIME_DEFAULT_DT;
    state->time_model = SIM_TIME_MODEL_CONTINUOUS;
    state->step_index = 0U;
    state->time_accumulated = 0.0;
    state->drift_mode_depth = 0U;
    state->drift_time_override = 0.0;
    state->drift_time_override_active = false;
    state->field_dirty_counts = NULL;
    state->field_stable_counts = NULL;
    state->field_phase_ema = NULL;
    state->field_phase_last_time = NULL;
    state->field_phase_lock_state = NULL;
    state->field_phase_initialized = NULL;
    state->drift_field_stats = NULL;
    state->drift_field_stats_step_index = SIZE_MAX;
    state->drift_field_stats_valid = NULL;
    state->drift_field_stats_requested = NULL;
    state->drift_field_snapshots = NULL;
    state->drift_field_snapshot_capacity = NULL;
    state->drift_field_snapshot_count = NULL;
    state->drift_field_snapshot_step_index = NULL;
    state->drift_field_snapshot_valid = NULL;
    state->drift_field_snapshot_requested = NULL;
    state->field_stats_cache = NULL;
    state->field_stats_cache_step_index = NULL;
    state->field_stats_config = sim_field_stats_default_compute_config();
    sim_field_stats_profile_reset(&state->field_stats_profile, &state->field_stats_config);
    state->field_topology_cache = NULL;
    state->field_health_cache_step_index = NULL;
    state->field_health_nan_counts = NULL;
    state->field_health_inf_counts = NULL;
    state->measurement_cache_step_index = SIZE_MAX;
    state->measurement_cache_valid = false;
    state->measurement_energy = 0.0;
    state->measurement_dissipation = 0.0;
    state->measurement_remainder = 0.0;
    state->measurement_remainder_sources = 0U;
    state->measurement_energy_valid = false;
    state->measurement_dissipation_valid = false;
    state->measurement_remainder_valid = false;
    state->visual_sample_cache = NULL;
    state->visual_sample_cache_capacity = NULL;
    state->visual_sample_cache_count = NULL;
    state->visual_sample_cache_step_index = NULL;
    state->visual_sample_cache_source_count = NULL;
    state->visual_sample_cache_stride = NULL;
    state->visual_sample_cache_valid = NULL;
    state->phase_portrait_metrics_cache = NULL;
    state->phase_portrait_metrics_cache_step_index = NULL;
    state->phase_portrait_metrics_cache_sample_count = NULL;
    state->phase_portrait_metrics_cache_valid = NULL;
    state->waveform_sample_stats_cache = NULL;
    state->waveform_sample_stats_cache_step_index = NULL;
    state->waveform_sample_stats_cache_sample_count = NULL;
    state->waveform_sample_stats_cache_valid = NULL;
    state->visual_sample_target_samples = SIM_RUNTIME_VIS_DOWNSAMPLE_TARGET_DEFAULT;
    state->visual_sample_max_samples = SIM_RUNTIME_VIS_DOWNSAMPLE_MAX_DEFAULT;
    state->continuity_capacity = 0U;
    state->current_step_warp_mask = 0U;
    state->step_metrics_head = 0U;
    state->step_metrics_count = 0U;
    state->step_metrics_valid = false;
    memset(&state->step_metrics_latest, 0, sizeof(state->step_metrics_latest));
    memset(state->step_metrics, 0, sizeof(state->step_metrics));
    sim_runtime_state_reset_loop_metadata(state);
    return SIM_RESULT_OK;
}

void sim_runtime_state_release_continuity_buffers(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    free(state->field_dirty_counts);
    free(state->field_stable_counts);
    free(state->field_phase_ema);
    free(state->field_phase_last_time);
    free(state->field_phase_lock_state);
    free(state->field_phase_initialized);
    free(state->drift_field_stats);
    free(state->drift_field_stats_valid);
    free(state->drift_field_stats_requested);
    if (state->drift_field_snapshots != NULL) {
        for (size_t i = 0U; i < state->continuity_capacity; ++i) {
            free(state->drift_field_snapshots[i]);
        }
    }
    free(state->drift_field_snapshots);
    free(state->drift_field_snapshot_capacity);
    free(state->drift_field_snapshot_count);
    free(state->drift_field_snapshot_step_index);
    free(state->drift_field_snapshot_valid);
    free(state->drift_field_snapshot_requested);
    if (state->visual_sample_cache != NULL) {
        for (size_t i = 0U; i < state->continuity_capacity; ++i) {
            free(state->visual_sample_cache[i]);
        }
    }
    free(state->visual_sample_cache);
    free(state->visual_sample_cache_capacity);
    free(state->visual_sample_cache_count);
    free(state->visual_sample_cache_step_index);
    free(state->visual_sample_cache_source_count);
    free(state->visual_sample_cache_stride);
    free(state->visual_sample_cache_valid);
    free(state->phase_portrait_metrics_cache);
    free(state->phase_portrait_metrics_cache_step_index);
    free(state->phase_portrait_metrics_cache_sample_count);
    free(state->phase_portrait_metrics_cache_valid);
    free(state->waveform_sample_stats_cache);
    free(state->waveform_sample_stats_cache_step_index);
    free(state->waveform_sample_stats_cache_sample_count);
    free(state->waveform_sample_stats_cache_valid);
    free(state->field_stats_cache);
    free(state->field_stats_cache_step_index);
    if (state->field_topology_cache != NULL) {
        for (size_t i = 0U; i < state->continuity_capacity; ++i) {
            sim_field_topology_runtime_free(&state->field_topology_cache[i]);
        }
    }
    free(state->field_topology_cache);
    free(state->field_health_cache_step_index);
    free(state->field_health_nan_counts);
    free(state->field_health_inf_counts);
    state->field_dirty_counts = NULL;
    state->field_stable_counts = NULL;
    state->field_phase_ema = NULL;
    state->field_phase_last_time = NULL;
    state->field_phase_lock_state = NULL;
    state->field_phase_initialized = NULL;
    state->drift_field_stats = NULL;
    state->drift_field_stats_step_index = SIZE_MAX;
    state->drift_field_stats_valid = NULL;
    state->drift_field_stats_requested = NULL;
    state->drift_field_snapshots = NULL;
    state->drift_field_snapshot_capacity = NULL;
    state->drift_field_snapshot_count = NULL;
    state->drift_field_snapshot_step_index = NULL;
    state->drift_field_snapshot_valid = NULL;
    state->drift_field_snapshot_requested = NULL;
    state->field_stats_cache = NULL;
    state->field_stats_cache_step_index = NULL;
    state->field_stats_config = sim_field_stats_default_compute_config();
    sim_field_stats_profile_reset(&state->field_stats_profile, &state->field_stats_config);
    state->field_topology_cache = NULL;
    state->field_health_cache_step_index = NULL;
    state->field_health_nan_counts = NULL;
    state->field_health_inf_counts = NULL;
    state->measurement_cache_step_index = SIZE_MAX;
    state->measurement_cache_valid = false;
    state->measurement_energy = 0.0;
    state->measurement_dissipation = 0.0;
    state->measurement_remainder = 0.0;
    state->measurement_remainder_sources = 0U;
    state->measurement_energy_valid = false;
    state->measurement_dissipation_valid = false;
    state->measurement_remainder_valid = false;
    state->visual_sample_cache = NULL;
    state->visual_sample_cache_capacity = NULL;
    state->visual_sample_cache_count = NULL;
    state->visual_sample_cache_step_index = NULL;
    state->visual_sample_cache_source_count = NULL;
    state->visual_sample_cache_stride = NULL;
    state->visual_sample_cache_valid = NULL;
    state->phase_portrait_metrics_cache = NULL;
    state->phase_portrait_metrics_cache_step_index = NULL;
    state->phase_portrait_metrics_cache_sample_count = NULL;
    state->phase_portrait_metrics_cache_valid = NULL;
    state->waveform_sample_stats_cache = NULL;
    state->waveform_sample_stats_cache_step_index = NULL;
    state->waveform_sample_stats_cache_sample_count = NULL;
    state->waveform_sample_stats_cache_valid = NULL;
    state->continuity_capacity = 0U;
    sim_runtime_state_reset_loop_metadata(state);
}

void sim_runtime_state_destroy(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    sim_runtime_state_release_continuity_buffers(state);
    state->dt = SIM_RUNTIME_DEFAULT_DT;
    state->time_model = SIM_TIME_MODEL_CONTINUOUS;
    state->step_index = 0U;
    state->time_accumulated = 0.0;
    state->drift_mode_depth = 0U;
    state->drift_time_override = 0.0;
    state->drift_time_override_active = false;
    state->current_step_warp_mask = 0U;
    state->field_stats_config = sim_field_stats_default_compute_config();
    sim_field_stats_profile_reset(&state->field_stats_profile, &state->field_stats_config);
    state->step_metrics_head = 0U;
    state->step_metrics_count = 0U;
    state->step_metrics_valid = false;
    memset(&state->step_metrics_latest, 0, sizeof(state->step_metrics_latest));
    state->measurement_cache_step_index = SIZE_MAX;
    state->measurement_cache_valid = false;
    state->measurement_energy = 0.0;
    state->measurement_dissipation = 0.0;
    state->measurement_remainder = 0.0;
    state->measurement_remainder_sources = 0U;
    state->measurement_energy_valid = false;
    state->measurement_dissipation_valid = false;
    state->measurement_remainder_valid = false;
    state->visual_sample_target_samples = SIM_RUNTIME_VIS_DOWNSAMPLE_TARGET_DEFAULT;
    state->visual_sample_max_samples = SIM_RUNTIME_VIS_DOWNSAMPLE_MAX_DEFAULT;
    sim_runtime_state_reset_loop_metadata(state);
}

SimResult sim_runtime_state_ensure_continuity_capacity(SimRuntimeState *state,
                                                       size_t required)
{
    uint64_t *new_dirty = NULL;
    uint64_t *new_stable = NULL;
    double *new_phase_ema = NULL;
    double *new_phase_last_time = NULL;
    uint8_t *new_phase_lock = NULL;
    uint8_t *new_phase_init = NULL;
    SimFieldStats *new_stats = NULL;
    bool *new_drift_stats_valid = NULL;
    bool *new_drift_stats_requested = NULL;
    SimFieldStats *new_stats_cache = NULL;
    size_t *new_stats_cache_steps = NULL;
    SimFieldTopologyRuntimeState* new_topology_cache = NULL;
    SimComplexDouble **new_snapshots = NULL;
    size_t *new_snapshot_capacity = NULL;
    size_t *new_snapshot_count = NULL;
    size_t *new_snapshot_steps = NULL;
    bool *new_snapshot_valid = NULL;
    bool *new_snapshot_requested = NULL;
    size_t *new_health_steps = NULL;
    size_t *new_health_nan = NULL;
    size_t *new_health_inf = NULL;
    SimRuntimeComplexSample **new_visual_cache = NULL;
    size_t *new_visual_cache_capacity = NULL;
    size_t *new_visual_cache_count = NULL;
    size_t *new_visual_cache_steps = NULL;
    size_t *new_visual_cache_source_count = NULL;
    size_t *new_visual_cache_stride = NULL;
    bool *new_visual_cache_valid = NULL;
    SimRuntimePhasePortraitMetrics *new_phase_metrics = NULL;
    size_t *new_phase_metrics_steps = NULL;
    size_t *new_phase_metrics_counts = NULL;
    bool *new_phase_metrics_valid = NULL;
    SimRuntimeWaveformSampleStats *new_waveform_stats = NULL;
    size_t *new_waveform_stats_steps = NULL;
    size_t *new_waveform_stats_counts = NULL;
    bool *new_waveform_stats_valid = NULL;
    size_t new_capacity;
    size_t copy_bytes;

    if (state == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (required == 0U || required <= state->continuity_capacity)
    {
        return SIM_RESULT_OK;
    }

    new_capacity = (state->continuity_capacity == 0U) ? 4U : state->continuity_capacity;
    while (new_capacity < required)
    {
        new_capacity *= 2U;
    }

    new_dirty = (uint64_t *)calloc(new_capacity, sizeof(uint64_t));
    new_stable = (uint64_t *)calloc(new_capacity, sizeof(uint64_t));
    new_phase_ema = (double *)calloc(new_capacity, sizeof(double));
    new_phase_last_time = (double *)calloc(new_capacity, sizeof(double));
    new_phase_lock = (uint8_t *)calloc(new_capacity, sizeof(uint8_t));
    new_phase_init = (uint8_t *)calloc(new_capacity, sizeof(uint8_t));
    new_stats = (SimFieldStats *)calloc(new_capacity, sizeof(SimFieldStats));
    new_drift_stats_valid = (bool *)calloc(new_capacity, sizeof(bool));
    new_drift_stats_requested = (bool *)calloc(new_capacity, sizeof(bool));
    new_stats_cache = (SimFieldStats *)calloc(new_capacity, sizeof(SimFieldStats));
    new_stats_cache_steps = (size_t *)malloc(new_capacity * sizeof(size_t));
    new_topology_cache =
        (SimFieldTopologyRuntimeState*) calloc(new_capacity, sizeof(SimFieldTopologyRuntimeState));
    new_snapshots = (SimComplexDouble **)calloc(new_capacity, sizeof(SimComplexDouble *));
    new_snapshot_capacity = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_snapshot_count = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_snapshot_steps = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_snapshot_valid = (bool *)calloc(new_capacity, sizeof(bool));
    new_snapshot_requested = (bool *)calloc(new_capacity, sizeof(bool));
    new_health_steps = (size_t *)malloc(new_capacity * sizeof(size_t));
    new_health_nan = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_health_inf = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_visual_cache = (SimRuntimeComplexSample **)calloc(new_capacity,
                                                          sizeof(SimRuntimeComplexSample *));
    new_visual_cache_capacity = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_visual_cache_count = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_visual_cache_steps = (size_t *)malloc(new_capacity * sizeof(size_t));
    new_visual_cache_source_count = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_visual_cache_stride = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_visual_cache_valid = (bool *)calloc(new_capacity, sizeof(bool));
    new_phase_metrics = (SimRuntimePhasePortraitMetrics *)calloc(new_capacity,
                                                                 sizeof(SimRuntimePhasePortraitMetrics));
    new_phase_metrics_steps = (size_t *)malloc(new_capacity * sizeof(size_t));
    new_phase_metrics_counts = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_phase_metrics_valid = (bool *)calloc(new_capacity, sizeof(bool));
    new_waveform_stats = (SimRuntimeWaveformSampleStats *)calloc(new_capacity,
                                                                 sizeof(SimRuntimeWaveformSampleStats));
    new_waveform_stats_steps = (size_t *)malloc(new_capacity * sizeof(size_t));
    new_waveform_stats_counts = (size_t *)calloc(new_capacity, sizeof(size_t));
    new_waveform_stats_valid = (bool *)calloc(new_capacity, sizeof(bool));
    if (new_dirty == NULL || new_stable == NULL || new_phase_ema == NULL || new_phase_last_time == NULL ||
        new_phase_lock == NULL || new_phase_init == NULL || new_stats == NULL ||
        new_drift_stats_valid == NULL || new_drift_stats_requested == NULL ||
        new_stats_cache == NULL || new_stats_cache_steps == NULL || new_topology_cache == NULL ||
        new_snapshots == NULL ||
        new_snapshot_capacity == NULL || new_snapshot_count == NULL || new_snapshot_steps == NULL ||
        new_snapshot_valid == NULL || new_snapshot_requested == NULL || new_health_steps == NULL ||
        new_health_nan == NULL || new_health_inf == NULL || new_visual_cache == NULL ||
        new_visual_cache_capacity == NULL || new_visual_cache_count == NULL ||
        new_visual_cache_steps == NULL || new_visual_cache_source_count == NULL ||
        new_visual_cache_stride == NULL || new_visual_cache_valid == NULL ||
        new_phase_metrics == NULL || new_phase_metrics_steps == NULL ||
        new_phase_metrics_counts == NULL || new_phase_metrics_valid == NULL ||
        new_waveform_stats == NULL || new_waveform_stats_steps == NULL ||
        new_waveform_stats_counts == NULL || new_waveform_stats_valid == NULL)
    {
        free(new_dirty);
        free(new_stable);
        free(new_phase_ema);
        free(new_phase_last_time);
        free(new_phase_lock);
        free(new_phase_init);
        free(new_stats);
        free(new_drift_stats_valid);
        free(new_drift_stats_requested);
        free(new_stats_cache);
        free(new_stats_cache_steps);
        free(new_topology_cache);
        free(new_snapshots);
        free(new_snapshot_capacity);
        free(new_snapshot_count);
        free(new_snapshot_steps);
        free(new_snapshot_valid);
        free(new_snapshot_requested);
        free(new_health_steps);
        free(new_health_nan);
        free(new_health_inf);
        free(new_visual_cache);
        free(new_visual_cache_capacity);
        free(new_visual_cache_count);
        free(new_visual_cache_steps);
        free(new_visual_cache_source_count);
        free(new_visual_cache_stride);
        free(new_visual_cache_valid);
        free(new_phase_metrics);
        free(new_phase_metrics_steps);
        free(new_phase_metrics_counts);
        free(new_phase_metrics_valid);
        free(new_waveform_stats);
        free(new_waveform_stats_steps);
        free(new_waveform_stats_counts);
        free(new_waveform_stats_valid);
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    for (size_t i = 0U; i < new_capacity; ++i) {
        sim_field_topology_runtime_init(&new_topology_cache[i]);
    }
    (void)memset(new_stats_cache_steps, 0xFF, new_capacity * sizeof(size_t));
    (void)memset(new_health_steps, 0xFF, new_capacity * sizeof(size_t));
    (void)memset(new_visual_cache_steps, 0xFF, new_capacity * sizeof(size_t));
    (void)memset(new_phase_metrics_steps, 0xFF, new_capacity * sizeof(size_t));
    (void)memset(new_waveform_stats_steps, 0xFF, new_capacity * sizeof(size_t));

    copy_bytes = state->continuity_capacity * sizeof(uint64_t);
    if (copy_bytes > 0U)
    {
        memcpy(new_dirty, state->field_dirty_counts, copy_bytes);
        memcpy(new_stable, state->field_stable_counts, copy_bytes);
    }

    copy_bytes = state->continuity_capacity * sizeof(double);
    if (copy_bytes > 0U)
    {
        memcpy(new_phase_ema, state->field_phase_ema, copy_bytes);
        memcpy(new_phase_last_time, state->field_phase_last_time, copy_bytes);
    }

    copy_bytes = state->continuity_capacity * sizeof(uint8_t);
    if (copy_bytes > 0U)
    {
        memcpy(new_phase_lock, state->field_phase_lock_state, copy_bytes);
        memcpy(new_phase_init, state->field_phase_initialized, copy_bytes);
    }

    copy_bytes = state->continuity_capacity * sizeof(SimFieldStats);
    if (copy_bytes > 0U && state->drift_field_stats != NULL)
    {
        memcpy(new_stats, state->drift_field_stats, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->drift_field_stats_valid != NULL)
    {
        memcpy(new_drift_stats_valid, state->drift_field_stats_valid, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->drift_field_stats_requested != NULL)
    {
        memcpy(new_drift_stats_requested, state->drift_field_stats_requested, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(SimFieldStats);
    if (copy_bytes > 0U && state->field_stats_cache != NULL)
    {
        memcpy(new_stats_cache, state->field_stats_cache, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->field_stats_cache_step_index != NULL)
    {
        memcpy(new_stats_cache_steps, state->field_stats_cache_step_index, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(SimFieldTopologyRuntimeState);
    if (copy_bytes > 0U && state->field_topology_cache != NULL)
    {
        memcpy(new_topology_cache, state->field_topology_cache, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(SimComplexDouble *);
    if (copy_bytes > 0U && state->drift_field_snapshots != NULL)
    {
        memcpy(new_snapshots, state->drift_field_snapshots, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->drift_field_snapshot_capacity != NULL)
    {
        memcpy(new_snapshot_capacity, state->drift_field_snapshot_capacity, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->drift_field_snapshot_count != NULL)
    {
        memcpy(new_snapshot_count, state->drift_field_snapshot_count, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->drift_field_snapshot_step_index != NULL)
    {
        memcpy(new_snapshot_steps, state->drift_field_snapshot_step_index, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->drift_field_snapshot_valid != NULL)
    {
        memcpy(new_snapshot_valid, state->drift_field_snapshot_valid, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->drift_field_snapshot_requested != NULL)
    {
        memcpy(new_snapshot_requested, state->drift_field_snapshot_requested, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->field_health_cache_step_index != NULL)
    {
        memcpy(new_health_steps, state->field_health_cache_step_index, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->field_health_nan_counts != NULL)
    {
        memcpy(new_health_nan, state->field_health_nan_counts, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->field_health_inf_counts != NULL)
    {
        memcpy(new_health_inf, state->field_health_inf_counts, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(SimRuntimeComplexSample *);
    if (copy_bytes > 0U && state->visual_sample_cache != NULL)
    {
        memcpy(new_visual_cache, state->visual_sample_cache, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->visual_sample_cache_capacity != NULL)
    {
        memcpy(new_visual_cache_capacity, state->visual_sample_cache_capacity, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->visual_sample_cache_count != NULL)
    {
        memcpy(new_visual_cache_count, state->visual_sample_cache_count, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->visual_sample_cache_step_index != NULL)
    {
        memcpy(new_visual_cache_steps, state->visual_sample_cache_step_index, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->visual_sample_cache_source_count != NULL)
    {
        memcpy(new_visual_cache_source_count, state->visual_sample_cache_source_count, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->visual_sample_cache_stride != NULL)
    {
        memcpy(new_visual_cache_stride, state->visual_sample_cache_stride, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->visual_sample_cache_valid != NULL)
    {
        memcpy(new_visual_cache_valid, state->visual_sample_cache_valid, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(SimRuntimePhasePortraitMetrics);
    if (copy_bytes > 0U && state->phase_portrait_metrics_cache != NULL)
    {
        memcpy(new_phase_metrics, state->phase_portrait_metrics_cache, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->phase_portrait_metrics_cache_step_index != NULL)
    {
        memcpy(new_phase_metrics_steps, state->phase_portrait_metrics_cache_step_index, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->phase_portrait_metrics_cache_sample_count != NULL)
    {
        memcpy(new_phase_metrics_counts, state->phase_portrait_metrics_cache_sample_count, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->phase_portrait_metrics_cache_valid != NULL)
    {
        memcpy(new_phase_metrics_valid, state->phase_portrait_metrics_cache_valid, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(SimRuntimeWaveformSampleStats);
    if (copy_bytes > 0U && state->waveform_sample_stats_cache != NULL)
    {
        memcpy(new_waveform_stats, state->waveform_sample_stats_cache, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->waveform_sample_stats_cache_step_index != NULL)
    {
        memcpy(new_waveform_stats_steps, state->waveform_sample_stats_cache_step_index, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(size_t);
    if (copy_bytes > 0U && state->waveform_sample_stats_cache_sample_count != NULL)
    {
        memcpy(new_waveform_stats_counts, state->waveform_sample_stats_cache_sample_count, copy_bytes);
    }
    copy_bytes = state->continuity_capacity * sizeof(bool);
    if (copy_bytes > 0U && state->waveform_sample_stats_cache_valid != NULL)
    {
        memcpy(new_waveform_stats_valid, state->waveform_sample_stats_cache_valid, copy_bytes);
    }

    free(state->field_dirty_counts);
    free(state->field_stable_counts);
    free(state->field_phase_ema);
    free(state->field_phase_last_time);
    free(state->field_phase_lock_state);
    free(state->field_phase_initialized);
    free(state->drift_field_stats);
    free(state->drift_field_stats_valid);
    free(state->drift_field_stats_requested);
    free(state->field_stats_cache);
    free(state->field_stats_cache_step_index);
    free(state->field_topology_cache);
    free(state->field_health_cache_step_index);
    free(state->field_health_nan_counts);
    free(state->field_health_inf_counts);
    free(state->drift_field_snapshots);
    free(state->drift_field_snapshot_capacity);
    free(state->drift_field_snapshot_count);
    free(state->drift_field_snapshot_step_index);
    free(state->drift_field_snapshot_valid);
    free(state->drift_field_snapshot_requested);
    free(state->visual_sample_cache);
    free(state->visual_sample_cache_capacity);
    free(state->visual_sample_cache_count);
    free(state->visual_sample_cache_step_index);
    free(state->visual_sample_cache_source_count);
    free(state->visual_sample_cache_stride);
    free(state->visual_sample_cache_valid);
    free(state->phase_portrait_metrics_cache);
    free(state->phase_portrait_metrics_cache_step_index);
    free(state->phase_portrait_metrics_cache_sample_count);
    free(state->phase_portrait_metrics_cache_valid);
    free(state->waveform_sample_stats_cache);
    free(state->waveform_sample_stats_cache_step_index);
    free(state->waveform_sample_stats_cache_sample_count);
    free(state->waveform_sample_stats_cache_valid);
    state->field_dirty_counts = new_dirty;
    state->field_stable_counts = new_stable;
    state->field_phase_ema = new_phase_ema;
    state->field_phase_last_time = new_phase_last_time;
    state->field_phase_lock_state = new_phase_lock;
    state->field_phase_initialized = new_phase_init;
    state->drift_field_stats = new_stats;
    state->drift_field_stats_valid = new_drift_stats_valid;
    state->drift_field_stats_requested = new_drift_stats_requested;
    state->field_stats_cache = new_stats_cache;
    state->field_stats_cache_step_index = new_stats_cache_steps;
    state->field_topology_cache = new_topology_cache;
    state->field_health_cache_step_index = new_health_steps;
    state->field_health_nan_counts = new_health_nan;
    state->field_health_inf_counts = new_health_inf;
    state->drift_field_snapshots = new_snapshots;
    state->drift_field_snapshot_capacity = new_snapshot_capacity;
    state->drift_field_snapshot_count = new_snapshot_count;
    state->drift_field_snapshot_step_index = new_snapshot_steps;
    state->drift_field_snapshot_valid = new_snapshot_valid;
    state->drift_field_snapshot_requested = new_snapshot_requested;
    state->visual_sample_cache = new_visual_cache;
    state->visual_sample_cache_capacity = new_visual_cache_capacity;
    state->visual_sample_cache_count = new_visual_cache_count;
    state->visual_sample_cache_step_index = new_visual_cache_steps;
    state->visual_sample_cache_source_count = new_visual_cache_source_count;
    state->visual_sample_cache_stride = new_visual_cache_stride;
    state->visual_sample_cache_valid = new_visual_cache_valid;
    state->phase_portrait_metrics_cache = new_phase_metrics;
    state->phase_portrait_metrics_cache_step_index = new_phase_metrics_steps;
    state->phase_portrait_metrics_cache_sample_count = new_phase_metrics_counts;
    state->phase_portrait_metrics_cache_valid = new_phase_metrics_valid;
    state->waveform_sample_stats_cache = new_waveform_stats;
    state->waveform_sample_stats_cache_step_index = new_waveform_stats_steps;
    state->waveform_sample_stats_cache_sample_count = new_waveform_stats_counts;
    state->waveform_sample_stats_cache_valid = new_waveform_stats_valid;
    if (new_capacity > 0U) {
        (void) memset(state->drift_field_stats_valid, 0, new_capacity * sizeof(bool));
    }
    state->drift_field_stats_step_index = SIZE_MAX;
    state->continuity_capacity = new_capacity;

    return SIM_RESULT_OK;
}

void sim_runtime_state_record_step_metrics(SimRuntimeState *state,
                                           const SimStepMetrics *metrics)
{
    if (state == NULL || metrics == NULL)
    {
        return;
    }

    state->step_metrics[state->step_metrics_head] = *metrics;
    state->step_metrics_head = (state->step_metrics_head + 1U) % SIM_STEP_METRIC_HISTORY;
    if (state->step_metrics_count < SIM_STEP_METRIC_HISTORY)
    {
        state->step_metrics_count += 1U;
    }
    state->step_metrics_latest = *metrics;
    state->step_metrics_valid = true;
}

bool sim_runtime_state_latest_step_metrics(const SimRuntimeState *state,
                                           SimStepMetrics *out_metrics)
{
    if (state == NULL || out_metrics == NULL || !state->step_metrics_valid)
    {
        return false;
    }

    *out_metrics = state->step_metrics_latest;
    return true;
}

size_t sim_runtime_state_copy_step_metrics(const SimRuntimeState *state,
                                           SimStepMetrics *dest,
                                           size_t capacity)
{
    size_t count;
    size_t to_copy;
    size_t start_index;
    size_t i;

    if (state == NULL || dest == NULL || capacity == 0U || state->step_metrics_count == 0U)
    {
        return 0U;
    }

    count = state->step_metrics_count;
    to_copy = (capacity < count) ? capacity : count;
    start_index = (state->step_metrics_head + SIM_STEP_METRIC_HISTORY - count) % SIM_STEP_METRIC_HISTORY;

    for (i = 0U; i < to_copy; ++i)
    {
        size_t index = (start_index + i) % SIM_STEP_METRIC_HISTORY;
        dest[i] = state->step_metrics[index];
    }

    return to_copy;
}

void sim_runtime_state_set_pending_cancel(SimRuntimeState *state, bool pending)
{
    if (state == NULL)
    {
        return;
    }

    state->pending_cancel = pending;
}

bool sim_runtime_state_pending_cancel(const SimRuntimeState *state)
{
    return (state != NULL) ? state->pending_cancel : false;
}

void sim_runtime_state_set_last_stop_reason(SimRuntimeState *state,
                                            SimRuntimeLoopStopReason reason)
{
    if (state == NULL)
    {
        return;
    }

    state->last_stop_reason = reason;
}

SimRuntimeLoopStopReason sim_runtime_state_last_stop_reason(const SimRuntimeState *state)
{
    if (state == NULL)
    {
        return SIM_RUNTIME_LOOP_STOP_REASON_NONE;
    }

    return state->last_stop_reason;
}

void sim_runtime_state_clear_loop_error(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    state->last_loop_error.valid = false;
    state->last_loop_error.code = SIM_RESULT_OK;
    state->last_loop_error.source = SIM_RUNTIME_LOOP_ERROR_SOURCE_NONE;
    state->last_loop_error.step_index = 0U;
}

void sim_runtime_state_record_loop_error(SimRuntimeState *state,
                                         SimResult code,
                                         SimRuntimeLoopErrorSource source,
                                         size_t step_index)
{
    if (state == NULL)
    {
        return;
    }

    state->last_loop_error.valid = true;
    state->last_loop_error.code = code;
    state->last_loop_error.source = source;
    state->last_loop_error.step_index = step_index;
}

bool sim_runtime_state_last_loop_error(const SimRuntimeState *state,
                                       SimRuntimeLoopError *out_error)
{
    if (state == NULL || out_error == NULL || !state->last_loop_error.valid)
    {
        return false;
    }

    *out_error = state->last_loop_error;
    return true;
}

void sim_runtime_state_note_caller_step(SimRuntimeState *state,
                                        SimRuntimeCallerStepMode mode)
{
    if (state == NULL || mode == SIM_RUNTIME_CALLER_STEP_MODE_NONE)
    {
        return;
    }

    if (state->last_caller_step_mode == mode)
    {
        if (state->caller_step_streak < SIZE_MAX)
        {
            state->caller_step_streak += 1U;
        }
        return;
    }

    state->last_caller_step_mode = mode;
    state->caller_step_streak = 1U;
}

void sim_runtime_state_clear_caller_step_history(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    state->last_caller_step_mode = SIM_RUNTIME_CALLER_STEP_MODE_NONE;
    state->caller_step_streak = 0U;
}

SimRuntimeCallerStepMode sim_runtime_state_last_caller_step_mode(const SimRuntimeState *state)
{
    if (state == NULL)
    {
        return SIM_RUNTIME_CALLER_STEP_MODE_NONE;
    }

    return state->last_caller_step_mode;
}

size_t sim_runtime_state_caller_step_streak(const SimRuntimeState *state)
{
    if (state == NULL)
    {
        return 0U;
    }

    return state->caller_step_streak;
}

void sim_runtime_state_acquire_external_driver(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->external_driver_depth < UINT32_MAX)
    {
        state->external_driver_depth += 1U;
    }
}

void sim_runtime_state_release_external_driver(SimRuntimeState *state)
{
    if (state == NULL || state->external_driver_depth == 0U)
    {
        return;
    }

    state->external_driver_depth -= 1U;
}

void sim_runtime_state_clear_external_driver(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    state->external_driver_depth = 0U;
}

size_t sim_runtime_state_external_driver_depth(const SimRuntimeState *state)
{
    if (state == NULL)
    {
        return 0U;
    }

    return (size_t) state->external_driver_depth;
}

void sim_runtime_state_set_loop_progress(SimRuntimeState *state,
                                         const SimRuntimeLoopProgress *progress)
{
    if (state == NULL)
    {
        return;
    }

    if (progress == NULL)
    {
        sim_runtime_state_clear_loop_progress(state);
        return;
    }

    state->loop_progress = *progress;
}

void sim_runtime_state_clear_loop_progress(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->loop_progress.kind != SIM_RUNTIME_LOOP_PROGRESS_NONE)
    {
        state->last_loop_progress = state->loop_progress;
        state->last_loop_progress.active = false;
        state->last_loop_progress_valid = true;
    }

    (void) memset(&state->loop_progress, 0, sizeof(state->loop_progress));
    state->loop_progress.kind = SIM_RUNTIME_LOOP_PROGRESS_NONE;
}

bool sim_runtime_state_loop_progress(const SimRuntimeState *state,
                                     SimRuntimeLoopProgress *out_progress)
{
    if (state == NULL || out_progress == NULL || !state->loop_progress.active)
    {
        return false;
    }

    *out_progress = state->loop_progress;
    return true;
}

bool sim_runtime_state_last_loop_progress(const SimRuntimeState *state,
                                          SimRuntimeLoopProgress *out_progress)
{
    if (state == NULL || out_progress == NULL || !state->last_loop_progress_valid)
    {
        return false;
    }

    *out_progress = state->last_loop_progress;
    return true;
}

void sim_runtime_state_clear_last_loop_progress(SimRuntimeState *state)
{
    if (state == NULL)
    {
        return;
    }

    (void) memset(&state->last_loop_progress, 0, sizeof(state->last_loop_progress));
    state->last_loop_progress.kind = SIM_RUNTIME_LOOP_PROGRESS_NONE;
    state->last_loop_progress_valid = false;
}

const char *sim_runtime_loop_stop_reason_name(SimRuntimeLoopStopReason reason)
{
    switch (reason)
    {
        case SIM_RUNTIME_LOOP_STOP_REASON_PAUSED:
            return "paused";
        case SIM_RUNTIME_LOOP_STOP_REASON_CANCELLED:
            return "cancelled";
        case SIM_RUNTIME_LOOP_STOP_REASON_STEPS_EXHAUSTED:
            return "steps_exhausted";
        case SIM_RUNTIME_LOOP_STOP_REASON_MAX_SIM_TIME_REACHED:
            return "max_sim_time_reached";
        case SIM_RUNTIME_LOOP_STOP_REASON_MAX_WALL_MS_REACHED:
            return "max_wall_ms_reached";
        case SIM_RUNTIME_LOOP_STOP_REASON_RUNTIME_ERROR:
            return "runtime_error";
        case SIM_RUNTIME_LOOP_STOP_REASON_NONE:
        default:
            break;
    }
    return "none";
}

const char *sim_runtime_loop_error_source_name(SimRuntimeLoopErrorSource source)
{
    switch (source)
    {
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_SCHEDULER_STEP:
            return "scheduler_step";
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_EXECUTION_FRAME:
            return "execution_frame";
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_OPERATOR_EXECUTION:
            return "operator_execution";
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_INTEGRATOR_BRIDGE:
            return "integrator_bridge";
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_INTEGRATOR_SEQUENCE:
            return "integrator_sequence";
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_LEGACY_STEP:
            return "legacy_step";
        case SIM_RUNTIME_LOOP_ERROR_SOURCE_NONE:
        default:
            break;
    }
    return "none";
}

const char *sim_runtime_caller_step_mode_name(SimRuntimeCallerStepMode mode)
{
    switch (mode)
    {
        case SIM_RUNTIME_CALLER_STEP_MODE_MANUAL_STEP:
            return "manual_step";
        case SIM_RUNTIME_CALLER_STEP_MODE_INTEGRATOR_STEP:
            return "integrator_step";
        case SIM_RUNTIME_CALLER_STEP_MODE_NONE:
        default:
            break;
    }
    return "none";
}

const char *sim_runtime_loop_progress_kind_name(SimRuntimeLoopProgressKind kind)
{
    switch (kind)
    {
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_STEPS:
            return "run_steps";
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_UNTIL:
            return "run_until";
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_INTEGRATOR_STEPS:
            return "run_integrator_steps";
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_UNTIL_INTEGRATOR:
            return "run_until_integrator";
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_PARAMETER_SWEEP:
            return "run_parameter_sweep";
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_PARAMETER_GRID:
            return "run_parameter_grid";
        case SIM_RUNTIME_LOOP_PROGRESS_RUN_PARAMETER_CONTINUATION:
            return "run_parameter_continuation";
        case SIM_RUNTIME_LOOP_PROGRESS_NONE:
        default:
            break;
    }
    return "none";
}
