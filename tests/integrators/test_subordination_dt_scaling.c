#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    double alpha;
    size_t quadrature_n;
    double truncation;
    struct SimContext *context;
} SubordinationState;

static bool nearly_equal(double a, double b, double rel_tol) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= rel_tol * scale;
}

static bool run_case(void) {
    enum { kCount = 8 };
    size_t shape[1] = {kCount};
    const double dt = 0.05;

    SimContext ctx = {0};
    SimField field = {0};
    bool ctx_ready = false;
    bool field_ready = false;
    bool ok = false;

    Integrator integrator = {0};
    IntegratorRegistry registry = {0};

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_init failed\n");
        goto cleanup;
    }
    ctx_ready = true;

    sim_context_set_timestep(&ctx, dt);
    sim_context_set_time_model(&ctx, SIM_TIME_MODEL_CONTINUOUS);

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_field_init failed\n");
        goto cleanup;
    }
    field_ready = true;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "field data pointer NULL\n");
        goto cleanup;
    }
    for (size_t i = 0U; i < kCount; ++i) {
        data[i].re = sin(0.5 * (double)i) + 0.25 * cos(1.25 * (double)i);
        data[i].im = 0.5 * cos(0.3 * (double)i);
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_add_field failed\n");
        goto cleanup;
    }
    field_ready = false;

    LinearDissipativeOperatorConfig op_cfg = {0};
    op_cfg.field_index = field_index;
    op_cfg.alpha = 2.0;
    op_cfg.viscosity = 0.5;
    op_cfg.spacing = 1.0;
    if (sim_add_linear_dissipative_operator(&ctx, &op_cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_add_linear_dissipative_operator failed\n");
        goto cleanup;
    }

    if (sim_context_prepare_plan(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_prepare_plan failed\n");
        goto cleanup;
    }

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "integrator_registry_init failed\n");
        goto cleanup;
    }

    IntegratorCreateFn factory = integrator_registry_lookup(&registry, "subordination");
    if (factory == NULL) {
        fprintf(stderr, "subordination integrator factory missing\n");
        goto cleanup;
    }

    IntegratorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.userdata = &ctx; /* required for subordination drift evaluation */

    if (factory(&cfg, &integrator) != SIM_RESULT_OK) {
        fprintf(stderr, "failed to create subordination integrator\n");
        goto cleanup;
    }

    SubordinationState *sstate = (SubordinationState *)integrator.userdata;
    if (sstate == NULL || sstate->context != &ctx) {
        fprintf(stderr, "unexpected subordination userdata\n");
        goto cleanup;
    }
    sstate->alpha = 0.75;
    sstate->quadrature_n = 16U;
    sstate->truncation = 2.0;

    SimField *ctx_field = sim_context_field(&ctx, field_index);
    if (ctx_field == NULL) {
        fprintf(stderr, "context field lookup failed\n");
        goto cleanup;
    }

    SimComplexDouble initial[kCount];
    const SimComplexDouble *ctx_data0 = sim_field_complex_data_const(ctx_field);
    if (ctx_data0 == NULL) {
        fprintf(stderr, "context field data pointer NULL\n");
        goto cleanup;
    }
    memcpy(initial, ctx_data0, sizeof(initial));

    /* Manual quadrature using the intended s-dependent timestep. */
    SimComplexDouble accum[kCount];
    SimComplexDouble drift[kCount];
    for (size_t i = 0U; i < kCount; ++i) {
        accum[i].re = 0.0;
        accum[i].im = 0.0;
    }

    double alpha = sstate->alpha;
    size_t N = sstate->quadrature_n;
    double Lambda = sstate->truncation;
    double h = Lambda / (double)N;

    for (size_t i = 0U; i < N; ++i) {
        double s = (i + 0.5) * h;
        double weight = exp(-pow(s, alpha)) * h;

        sim_context_set_timestep(&ctx, dt * s);
        SimResult drift_result = integrator.drift(&integrator, ctx_field, (const double *)initial,
                                                  (double *)drift, kCount);
        sim_context_set_timestep(&ctx, dt);

        if (drift_result != SIM_RESULT_OK) {
            fprintf(stderr, "subordination drift failed (%d)\n", (int)drift_result);
            goto cleanup;
        }

        for (size_t j = 0U; j < kCount; ++j) {
            accum[j].re += weight * drift[j].re;
            accum[j].im += weight * drift[j].im;
        }
    }

    SimComplexDouble expected[kCount];
    for (size_t i = 0U; i < kCount; ++i) {
        expected[i].re = initial[i].re + dt * accum[i].re;
        expected[i].im = initial[i].im + dt * accum[i].im;
    }

    /* Reset the field and run the actual step. */
    SimComplexDouble *ctx_data = sim_field_complex_data(ctx_field);
    if (ctx_data == NULL) {
        fprintf(stderr, "context field data pointer NULL (mutable)\n");
        goto cleanup;
    }
    memcpy(ctx_data, initial, sizeof(initial));
    sim_context_set_timestep(&ctx, dt);
    integrator.step(&integrator, ctx_field, dt);

    const SimComplexDouble *out = sim_field_complex_data_const(ctx_field);
    if (out == NULL) {
        fprintf(stderr, "context field data pointer NULL (after step)\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < kCount; ++i) {
        if (!nearly_equal(out[i].re, expected[i].re, 5e-11) ||
            !nearly_equal(out[i].im, expected[i].im, 5e-11)) {
            fprintf(stderr, "mismatch at %zu: got (%.17g, %.17g) expected (%.17g, %.17g)\n", i,
                    out[i].re, out[i].im, expected[i].re, expected[i].im);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    integrator_registry_destroy(&registry);
    integrator_destroy(&integrator);
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (field_ready) {
        sim_field_destroy(&field);
    }
    return ok;
}

int main(void) {
    if (!run_case()) {
        return 1;
    }
    return 0;
}
