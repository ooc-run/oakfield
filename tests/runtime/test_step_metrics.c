#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static bool nearly_equal(float a, float b) {
    float diff = fabsf(a - b);
    float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return diff <= 1e-6f * scale;
}

static bool setup_scalar_field(SimContext *ctx, size_t *out_index) {
    size_t shape[1] = {8U};
    SimField field = {0};
    SimResult result =
        sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_field_init failed (%d)\n", (int)result);
        return false;
    }

    double *raw = sim_field_real_data(&field);
    if (raw == NULL) {
        fprintf(stderr, "[FAIL] sim_field_real_data returned NULL\n");
        sim_field_destroy(&field);
        return false;
    }
    memset(raw, 0, sim_field_bytes(&field));

    result = sim_context_add_field(ctx, &field, out_index);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_add_field failed (%d)\n", (int)result);
        sim_field_destroy(&field);
        return false;
    }

    return true;
}

static bool test_latest_metrics_capture(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_init failed\n");
        return false;
    }

    bool ok = false;
    size_t field_index = 0U;
    SimStepMetrics latest = {0};

    if (!setup_scalar_field(&ctx, &field_index)) {
        goto cleanup;
    }

    if (sim_context_latest_step_metrics(&ctx, &latest)) {
        fprintf(stderr, "[FAIL] expected no metrics before first sample\n");
        goto cleanup;
    }

    if (ctx.runtime.field_dirty_counts == NULL || ctx.runtime.field_stable_counts == NULL) {
        fprintf(stderr, "[FAIL] continuity counters not allocated\n");
        goto cleanup;
    }

    ctx.runtime.step_index = 5U;
    ctx.runtime.field_dirty_counts[field_index] = 3ULL;
    ctx.runtime.field_stable_counts[field_index] = 7ULL;

    sim_context_record_step_metrics(&ctx, 0.01f, 0.02f, 1.5e-4f);
    if (!sim_context_latest_step_metrics(&ctx, &latest)) {
        fprintf(stderr, "[FAIL] latest metrics missing after sample\n");
        goto cleanup;
    }
    if (latest.step_index != 4U) {
        fprintf(stderr, "[FAIL] expected step_index=4, got %zu\n", latest.step_index);
        goto cleanup;
    }
    if (!nearly_equal(latest.requested_dt, 0.01f)) {
        fprintf(stderr, "[FAIL] requested_dt mismatch: %.9g\n", latest.requested_dt);
        goto cleanup;
    }
    if (!nearly_equal(latest.accepted_dt, 0.02f)) {
        fprintf(stderr, "[FAIL] accepted_dt mismatch: %.9g\n", latest.accepted_dt);
        goto cleanup;
    }
    if (!nearly_equal(latest.next_dt, sim_context_timestep(&ctx))) {
        fprintf(stderr, "[FAIL] next_dt mismatch: %.9g\n", latest.next_dt);
        goto cleanup;
    }
    if (!nearly_equal(latest.rms_error, 1.5e-4f)) {
        fprintf(stderr, "[FAIL] rms_error mismatch: %.9g\n", latest.rms_error);
        goto cleanup;
    }
    if (latest.dirty_write_count != 3ULL || latest.stable_write_count != 7ULL) {
        fprintf(stderr, "[FAIL] continuity totals mismatch: dirty=%llu stable=%llu\n",
                (unsigned long long)latest.dirty_write_count,
                (unsigned long long)latest.stable_write_count);
        goto cleanup;
    }
    if (latest.integrator_attempt_count != 0U || latest.integrator_rejection_count != 0U ||
        latest.integrator_workspace_bytes != 0ULL ||
        latest.integrator_drift_scratch_bytes != 0ULL) {
        fprintf(stderr,
                "[FAIL] default integrator metrics should remain zero without integrator\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool test_history_ring_buffer(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_init failed\n");
        return false;
    }

    bool ok = false;
    size_t field_index = 0U;
    const size_t total_samples = SIM_STEP_METRIC_HISTORY + 5U;
    SimStepMetrics history[SIM_STEP_METRIC_HISTORY] = {0};

    if (!setup_scalar_field(&ctx, &field_index)) {
        goto cleanup;
    }
    if (ctx.runtime.field_dirty_counts == NULL || ctx.runtime.field_stable_counts == NULL) {
        fprintf(stderr, "[FAIL] continuity counters not allocated\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < total_samples; ++i) {
        float dt = 0.01f + (float)i * 0.0001f;
        ctx.runtime.step_index = i + 1U;
        ctx.runtime.field_dirty_counts[field_index] = (uint64_t)(i + 1U);
        ctx.runtime.field_stable_counts[field_index] = (uint64_t)(2U * (i + 1U));
        sim_context_set_timestep(&ctx, dt);
        sim_context_record_step_metrics(&ctx, dt, dt, (float)i * 1e-5f);
    }

    size_t copied = sim_context_step_metrics_history(&ctx, history, SIM_STEP_METRIC_HISTORY);
    if (copied != SIM_STEP_METRIC_HISTORY) {
        fprintf(stderr, "[FAIL] expected %u history samples, got %zu\n",
                (unsigned int)SIM_STEP_METRIC_HISTORY, copied);
        goto cleanup;
    }

    size_t expected_start = total_samples - SIM_STEP_METRIC_HISTORY;
    if (history[0].step_index != expected_start) {
        fprintf(stderr, "[FAIL] first history index mismatch: expected %zu got %zu\n",
                expected_start, history[0].step_index);
        goto cleanup;
    }

    size_t expected_last = (ctx.runtime.step_index > 0U) ? (ctx.runtime.step_index - 1U) : 0U;
    if (history[copied - 1U].step_index != expected_last) {
        fprintf(stderr, "[FAIL] last history index mismatch: expected %zu got %zu\n", expected_last,
                history[copied - 1U].step_index);
        goto cleanup;
    }
    if (history[copied - 1U].dirty_write_count != ctx.runtime.field_dirty_counts[field_index] ||
        history[copied - 1U].stable_write_count != ctx.runtime.field_stable_counts[field_index]) {
        fprintf(stderr, "[FAIL] last history continuity totals mismatch\n");
        goto cleanup;
    }
    if (history[copied - 1U].integrator_attempt_count != 0U ||
        history[copied - 1U].integrator_rejection_count != 0U) {
        fprintf(stderr,
                "[FAIL] history should preserve zero integrator attempt/rejection counts\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    bool ok = true;
    ok &= test_latest_metrics_capture();
    ok &= test_history_ring_buffer();
    return ok ? 0 : 1;
}
