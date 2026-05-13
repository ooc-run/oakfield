#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static bool nearly_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    double scale = fmax(fabs(a), fabs(b));
    if (scale < 1.0) {
        scale = 1.0;
    }
    return diff <= tol * scale;
}

static bool run_copy_case(double dt, double *out_value) {
    const size_t N = 4U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField input = {0};
    SimField output = {0};
    bool ctx_ready = false;
    bool input_ready = false;
    bool output_ready = false;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }
    ctx_ready = true;
    sim_context_set_timestep(&ctx, dt);
    sim_context_set_time_model(&ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto cleanup;
    }
    input_ready = true;
    output_ready = true;

    double *src = (double *)sim_field_data(&input);
    double *dst = (double *)sim_field_data(&output);
    for (size_t i = 0U; i < N; ++i) {
        src[i] = 2.0;
        dst[i] = 10.0;
    }

    size_t input_index = 0U;
    size_t output_index = 0U;
    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &output_index) != SIM_RESULT_OK) {
        goto cleanup;
    }
    input_ready = false;
    output_ready = false;

    SimCopyOperatorConfig cfg = {0};
    cfg.input_field = input_index;
    cfg.output_field = output_index;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;

    if (sim_add_copy_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, output_index);
    const double *out = (const double *)sim_field_data_const(out_field);
    if (out == NULL) {
        goto cleanup;
    }

    *out_value = out[0];
    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return ok;
}

static bool run_scale_case(double dt, double *out_value) {
    const size_t N = 4U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField input = {0};
    SimField output = {0};
    bool ctx_ready = false;
    bool input_ready = false;
    bool output_ready = false;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }
    ctx_ready = true;
    sim_context_set_timestep(&ctx, dt);
    sim_context_set_time_model(&ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto cleanup;
    }
    input_ready = true;
    output_ready = true;

    double *src = (double *)sim_field_data(&input);
    double *dst = (double *)sim_field_data(&output);
    for (size_t i = 0U; i < N; ++i) {
        src[i] = 2.0;
        dst[i] = 10.0;
    }

    size_t input_index = 0U;
    size_t output_index = 0U;
    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &output_index) != SIM_RESULT_OK) {
        goto cleanup;
    }
    input_ready = false;
    output_ready = false;

    SimScaleOperatorConfig cfg = {0};
    cfg.input_field = input_index;
    cfg.output_field = output_index;
    cfg.scale = 3.0;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;

    if (sim_add_scale_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, output_index);
    const double *out = (const double *)sim_field_data_const(out_field);
    if (out == NULL) {
        goto cleanup;
    }

    *out_value = out[0];
    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return ok;
}

static bool run_remainder_case(double dt, double *out_value) {
    const size_t N = 4U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField warped = {0};
    SimField reference = {0};
    SimField output = {0};
    bool ctx_ready = false;
    bool warped_ready = false;
    bool reference_ready = false;
    bool output_ready = false;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }
    ctx_ready = true;
    sim_context_set_timestep(&ctx, dt);
    sim_context_set_time_model(&ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&warped, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&reference, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto cleanup;
    }
    warped_ready = true;
    reference_ready = true;
    output_ready = true;

    double *w = (double *)sim_field_data(&warped);
    double *r = (double *)sim_field_data(&reference);
    double *o = (double *)sim_field_data(&output);
    for (size_t i = 0U; i < N; ++i) {
        w[i] = 2.0;
        r[i] = 1.0;
        o[i] = 10.0;
    }

    size_t wi = 0U, ri = 0U, oi = 0U;
    if (sim_context_add_field(&ctx, &warped, &wi) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &reference, &ri) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &oi) != SIM_RESULT_OK) {
        goto cleanup;
    }
    warped_ready = false;
    reference_ready = false;
    output_ready = false;

    SimRemainderOperatorConfig cfg = {0};
    cfg.warped_field = wi;
    cfg.reference_field = ri;
    cfg.output_field = oi;
    cfg.weight = 2.0;
    cfg.bias = 0.5;
    cfg.exponent = 1.0;
    cfg.epsilon = 1.0e-6;
    cfg.nonlinearity = SIM_REMAINDER_NONLINEARITY_IDENTITY;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;
    cfg.complex_mode = SIM_REMAINDER_COMPLEX_MODE_COMPONENT;

    if (sim_add_remainder_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, oi);
    const double *out = (const double *)sim_field_data_const(out_field);
    if (!out) {
        goto cleanup;
    }

    *out_value = out[0];
    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (warped_ready) {
        sim_field_destroy(&warped);
    }
    if (reference_ready) {
        sim_field_destroy(&reference);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return ok;
}

static bool run_sieve_case(double dt, double *out_value) {
    const size_t N = 9U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField input = {0};
    SimField output = {0};
    bool ctx_ready = false;
    bool input_ready = false;
    bool output_ready = false;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }
    ctx_ready = true;
    sim_context_set_timestep(&ctx, dt);
    sim_context_set_time_model(&ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto cleanup;
    }
    input_ready = true;
    output_ready = true;

    double *x = (double *)sim_field_data(&input);
    double *y = (double *)sim_field_data(&output);
    for (size_t i = 0U; i < N; ++i) {
        x[i] = 3.0;
        y[i] = 1.0;
    }

    size_t xi = 0U, yi = 0U;
    if (sim_context_add_field(&ctx, &input, &xi) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &yi) != SIM_RESULT_OK) {
        goto cleanup;
    }
    input_ready = false;
    output_ready = false;

    SimSieveOperatorConfig cfg = {0};
    cfg.input_field = xi;
    cfg.output_field = yi;
    cfg.taps = 5U;
    cfg.sigma = 1.0;
    cfg.gain = 2.0;
    cfg.mode = SIM_SIEVE_MODE_LOW_PASS;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;

    if (sim_add_sieve_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, yi);
    const double *out = (const double *)sim_field_data_const(out_field);
    if (!out) {
        goto cleanup;
    }

    *out_value = out[0];
    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return ok;
}

static bool run_mixer_case(double dt, double *out_value) {
    const size_t N = 4U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField lhs = {0};
    SimField rhs = {0};
    SimField out_field = {0};
    bool ctx_ready = false;
    bool lhs_ready = false;
    bool rhs_ready = false;
    bool out_ready = false;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }
    ctx_ready = true;
    sim_context_set_timestep(&ctx, dt);
    sim_context_set_time_model(&ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&lhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&rhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&out_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto cleanup;
    }
    lhs_ready = true;
    rhs_ready = true;
    out_ready = true;

    double *a = (double *)sim_field_data(&lhs);
    double *b = (double *)sim_field_data(&rhs);
    double *o = (double *)sim_field_data(&out_field);
    for (size_t i = 0U; i < N; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        o[i] = 0.25;
    }

    size_t ai = 0U, bi = 0U, oi = 0U;
    if (sim_context_add_field(&ctx, &lhs, &ai) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &rhs, &bi) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &out_field, &oi) != SIM_RESULT_OK) {
        goto cleanup;
    }
    lhs_ready = false;
    rhs_ready = false;
    out_ready = false;

    SimMixerOperatorConfig cfg = {0};
    cfg.lhs_field = ai;
    cfg.rhs_field = bi;
    cfg.output_field = oi;
    cfg.lhs_gain = 1.0;
    cfg.rhs_gain = 1.0;
    cfg.mix = 0.5;
    cfg.bias = 0.5;
    cfg.mode = SIM_MIXER_MODE_SUM;
    cfg.accumulate = true;
    cfg.scale_by_dt = true;

    if (sim_add_mixer_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimField *out = sim_context_field(&ctx, oi);
    const double *values = (const double *)sim_field_data_const(out);
    if (!values) {
        goto cleanup;
    }

    *out_value = values[0];
    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (lhs_ready) {
        sim_field_destroy(&lhs);
    }
    if (rhs_ready) {
        sim_field_destroy(&rhs);
    }
    if (out_ready) {
        sim_field_destroy(&out_field);
    }
    return ok;
}

int main(void) {
    const double dt1 = 0.1;
    const double dt2 = 0.2;

    double c1 = 0.0, c2 = 0.0;
    if (!run_copy_case(dt1, &c1) || !run_copy_case(dt2, &c2)) {
        fprintf(stderr, "[FAIL] copy setup/execute\n");
        return 1;
    }
    {
        const double expected1 = 10.0 + dt1 * 2.0;
        const double expected2 = 10.0 + dt2 * 2.0;
        if (!nearly_equal(c1, expected1, 1.0e-12) || !nearly_equal(c2, expected2, 1.0e-12)) {
            fprintf(stderr, "[FAIL] copy dt scaling got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                    c1, c2, expected1, expected2);
            return 1;
        }
    }

    double sc1 = 0.0, sc2 = 0.0;
    if (!run_scale_case(dt1, &sc1) || !run_scale_case(dt2, &sc2)) {
        fprintf(stderr, "[FAIL] scale setup/execute\n");
        return 1;
    }
    {
        const double expected1 = 10.0 + dt1 * 6.0;
        const double expected2 = 10.0 + dt2 * 6.0;
        if (!nearly_equal(sc1, expected1, 1.0e-12) || !nearly_equal(sc2, expected2, 1.0e-12)) {
            fprintf(stderr, "[FAIL] scale dt scaling got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                    sc1, sc2, expected1, expected2);
            return 1;
        }
    }

    double r1 = 0.0, r2 = 0.0;
    if (!run_remainder_case(dt1, &r1) || !run_remainder_case(dt2, &r2)) {
        fprintf(stderr, "[FAIL] remainder setup/execute\n");
        return 1;
    }
    {
        const double residue = (2.0 - 1.0) * 2.0 + 0.5;
        const double expected1 = 10.0 + dt1 * residue;
        const double expected2 = 10.0 + dt2 * residue;
        if (!nearly_equal(r1, expected1, 1.0e-12) || !nearly_equal(r2, expected2, 1.0e-12)) {
            fprintf(stderr,
                    "[FAIL] remainder dt scaling got=(%.12g, %.12g) expected=(%.12g, %.12g)\n", r1,
                    r2, expected1, expected2);
            return 1;
        }
    }

    double s1 = 0.0, s2 = 0.0;
    if (!run_sieve_case(dt1, &s1) || !run_sieve_case(dt2, &s2)) {
        fprintf(stderr, "[FAIL] sieve setup/execute\n");
        return 1;
    }
    {
        const double value = 3.0 * 2.0;
        const double expected1 = 1.0 + dt1 * value;
        const double expected2 = 1.0 + dt2 * value;
        if (!nearly_equal(s1, expected1, 1.0e-12) || !nearly_equal(s2, expected2, 1.0e-12)) {
            fprintf(stderr, "[FAIL] sieve dt scaling got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                    s1, s2, expected1, expected2);
            return 1;
        }
    }

    double m1 = 0.0, m2 = 0.0;
    if (!run_mixer_case(dt1, &m1) || !run_mixer_case(dt2, &m2)) {
        fprintf(stderr, "[FAIL] mixer setup/execute\n");
        return 1;
    }
    {
        const double mixed = (1.0 + 2.0) + 0.5;
        const double expected1 = 0.25 + dt1 * mixed;
        const double expected2 = 0.25 + dt2 * mixed;
        if (!nearly_equal(m1, expected1, 1.0e-12) || !nearly_equal(m2, expected2, 1.0e-12)) {
            fprintf(stderr, "[FAIL] mixer dt scaling got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                    m1, m2, expected1, expected2);
            return 1;
        }
    }

    printf("[PASS] nonstimulus dt scaling (accumulate + scale_by_dt)\n");
    return 0;
}
