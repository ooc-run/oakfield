#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool nearly_equal(double a, double b, double abs_tol) { return fabs(a - b) <= abs_tol; }

static double max_abs_error(const double *a, const double *b, size_t count) {
    double max_err = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double err = fabs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

static bool run_steps(SimContext *ctx, double dt, unsigned int steps) {
    for (unsigned int i = 0U; i < steps; ++i) {
        if (sim_context_execute(ctx) != SIM_RESULT_OK) {
            return false;
        }
        sim_context_accept_step(ctx, dt);
    }
    return true;
}

static bool setup_feedback_remainder(SimContext *ctx, size_t count, double dt, double k,
                                     const double *initial, size_t *out_field_index) {
    size_t shape[1] = {count};
    SimField state = {0};
    SimField reference = {0};
    bool state_ready = false;
    bool reference_ready = false;

    if (sim_context_init(ctx) != SIM_RESULT_OK) {
        return false;
    }

    sim_context_set_timestep(ctx, dt);
    sim_context_set_time_model(ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&state, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&reference, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto fail;
    }
    state_ready = true;
    reference_ready = true;

    double *s = (double *)sim_field_data(&state);
    double *r = (double *)sim_field_data(&reference);
    if (s == NULL || r == NULL) {
        goto fail;
    }
    memcpy(s, initial, count * sizeof(double));
    memset(r, 0, count * sizeof(double));

    size_t state_index = 0U;
    size_t ref_index = 0U;
    if (sim_context_add_field(ctx, &state, &state_index) != SIM_RESULT_OK ||
        sim_context_add_field(ctx, &reference, &ref_index) != SIM_RESULT_OK) {
        goto fail;
    }
    state_ready = false;
    reference_ready = false;

    SimRemainderOperatorConfig cfg = {0};
    cfg.warped_field = state_index;
    cfg.reference_field = ref_index;
    cfg.output_field = state_index;
    cfg.weight = k;
    cfg.bias = 0.0;
    cfg.exponent = 1.0;
    cfg.epsilon = 1.0e-6;
    cfg.nonlinearity = SIM_REMAINDER_NONLINEARITY_IDENTITY;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;
    cfg.complex_mode = SIM_REMAINDER_COMPLEX_MODE_COMPONENT;

    if (sim_add_remainder_operator(ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto fail;
    }

    if (sim_context_prepare_plan(ctx) != SIM_RESULT_OK) {
        goto fail;
    }

    if (out_field_index != NULL) {
        *out_field_index = state_index;
    }
    return true;

fail:
    if (state_ready) {
        sim_field_destroy(&state);
    }
    if (reference_ready) {
        sim_field_destroy(&reference);
    }
    sim_context_destroy(ctx);
    return false;
}

static bool setup_feedback_mixer(SimContext *ctx, size_t count, double dt, double k,
                                 const double *initial, size_t *out_field_index) {
    size_t shape[1] = {count};
    SimField state = {0};
    SimField reference = {0};
    bool state_ready = false;
    bool reference_ready = false;

    if (sim_context_init(ctx) != SIM_RESULT_OK) {
        return false;
    }

    sim_context_set_timestep(ctx, dt);
    sim_context_set_time_model(ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&state, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&reference, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto fail;
    }
    state_ready = true;
    reference_ready = true;

    double *s = (double *)sim_field_data(&state);
    double *r = (double *)sim_field_data(&reference);
    if (s == NULL || r == NULL) {
        goto fail;
    }
    memcpy(s, initial, count * sizeof(double));
    memset(r, 0, count * sizeof(double));

    size_t state_index = 0U;
    size_t ref_index = 0U;
    if (sim_context_add_field(ctx, &state, &state_index) != SIM_RESULT_OK ||
        sim_context_add_field(ctx, &reference, &ref_index) != SIM_RESULT_OK) {
        goto fail;
    }
    state_ready = false;
    reference_ready = false;

    SimMixerOperatorConfig cfg = {0};
    cfg.lhs_field = state_index;
    cfg.rhs_field = ref_index;
    cfg.output_field = state_index;
    cfg.lhs_gain = k;
    cfg.rhs_gain = 0.0;
    cfg.mix = 0.0;
    cfg.bias = 0.0;
    cfg.mode = SIM_MIXER_MODE_SUM;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;

    if (sim_add_mixer_operator(ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto fail;
    }

    if (sim_context_prepare_plan(ctx) != SIM_RESULT_OK) {
        goto fail;
    }

    if (out_field_index != NULL) {
        *out_field_index = state_index;
    }
    return true;

fail:
    if (state_ready) {
        sim_field_destroy(&state);
    }
    if (reference_ready) {
        sim_field_destroy(&reference);
    }
    sim_context_destroy(ctx);
    return false;
}

static bool run_convergence_case(const char *label,
                                 bool (*setup)(SimContext *, size_t, double, double, const double *,
                                               size_t *)) {
    enum { kCount = 8 };
    const double u0[kCount] = {1.0, -0.5, 0.25, 1.25, -2.0, 0.75, -1.5, 0.9};
    const double k = -0.8;

    const double dt0 = 0.1;
    const double dt1 = 0.05;
    const double dt2 = 0.025;
    const double T = 1.0;

    const unsigned int steps0 = (unsigned int)lrint(T / dt0);
    const unsigned int steps1 = (unsigned int)lrint(T / dt1);
    const unsigned int steps2 = (unsigned int)lrint(T / dt2);

    if (!nearly_equal((double)steps0 * dt0, T, 1e-12) ||
        !nearly_equal((double)steps1 * dt1, T, 1e-12) ||
        !nearly_equal((double)steps2 * dt2, T, 1e-12)) {
        fprintf(stderr, "[%s] internal: step counts do not match T\n", label);
        return false;
    }

    double exact[kCount];
    for (size_t i = 0U; i < kCount; ++i) {
        exact[i] = u0[i] * exp(k * T);
    }

    SimContext ctx0 = {0};
    SimContext ctx1 = {0};
    SimContext ctx2 = {0};
    bool ctx0_ready = false;
    bool ctx1_ready = false;
    bool ctx2_ready = false;
    size_t field0 = 0U;
    size_t field1 = 0U;
    size_t field2 = 0U;

    double out0[kCount];
    double out1[kCount];
    double out2[kCount];
    bool ok = false;

    if (!setup(&ctx0, kCount, dt0, k, u0, &field0)) {
        fprintf(stderr, "[%s] setup dt0 failed\n", label);
        goto cleanup;
    }
    ctx0_ready = true;
    if (!setup(&ctx1, kCount, dt1, k, u0, &field1)) {
        fprintf(stderr, "[%s] setup dt1 failed\n", label);
        goto cleanup;
    }
    ctx1_ready = true;
    if (!setup(&ctx2, kCount, dt2, k, u0, &field2)) {
        fprintf(stderr, "[%s] setup dt2 failed\n", label);
        goto cleanup;
    }
    ctx2_ready = true;

    if (!run_steps(&ctx0, dt0, steps0) || !run_steps(&ctx1, dt1, steps1) ||
        !run_steps(&ctx2, dt2, steps2)) {
        fprintf(stderr, "[%s] step execution failed\n", label);
        goto cleanup;
    }

    if (!nearly_equal(sim_context_time(&ctx0), T, 1e-12) ||
        !nearly_equal(sim_context_time(&ctx1), T, 1e-12) ||
        !nearly_equal(sim_context_time(&ctx2), T, 1e-12)) {
        fprintf(stderr, "[%s] time did not advance as expected\n", label);
        goto cleanup;
    }

    const double *d0 = sim_field_real_data_const(sim_context_field(&ctx0, field0));
    const double *d1 = sim_field_real_data_const(sim_context_field(&ctx1, field1));
    const double *d2 = sim_field_real_data_const(sim_context_field(&ctx2, field2));
    if (d0 == NULL || d1 == NULL || d2 == NULL) {
        fprintf(stderr, "[%s] output field pointer NULL\n", label);
        goto cleanup;
    }

    memcpy(out0, d0, sizeof(out0));
    memcpy(out1, d1, sizeof(out1));
    memcpy(out2, d2, sizeof(out2));

    double err0 = max_abs_error(out0, exact, kCount);
    double err1 = max_abs_error(out1, exact, kCount);
    double err2 = max_abs_error(out2, exact, kCount);

    /* Explicit Euler should converge at first order: error shrinks ~2x when dt halves. */
    if (!(err1 < err0 * 0.85) || !(err2 < err1 * 0.85)) {
        fprintf(stderr,
                "[%s] expected dt refinement convergence, got err0=%.3e err1=%.3e err2=%.3e\n",
                label, err0, err1, err2);
        goto cleanup;
    }

    if (err2 > 1.0e-2) {
        fprintf(stderr, "[%s] refined dt error too large: %.3e\n", label, err2);
        goto cleanup;
    }

    ok = true;

cleanup:
    if (ctx0_ready) {
        sim_context_destroy(&ctx0);
    }
    if (ctx1_ready) {
        sim_context_destroy(&ctx1);
    }
    if (ctx2_ready) {
        sim_context_destroy(&ctx2);
    }
    return ok;
}

int main(void) {
    bool ok_remainder = run_convergence_case("remainder_feedback", setup_feedback_remainder);
    bool ok_mixer = run_convergence_case("mixer_feedback", setup_feedback_mixer);

    return (ok_remainder && ok_mixer) ? 0 : 1;
}
