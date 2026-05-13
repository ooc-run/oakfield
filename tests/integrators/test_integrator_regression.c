#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct LinearDriftParams {
    double lambda_re;
    double lambda_im;
} LinearDriftParams;

static const double kInitialValues[] = {1.0, -0.5, 0.25, 1.25};
static const size_t kElementCount = sizeof(kInitialValues) / sizeof(kInitialValues[0]);

static SimResult linear_drift(Integrator *integrator, const Field *field, const double *state,
                              double *out_derivative, size_t count) {
    const LinearDriftParams *params = (const LinearDriftParams *)integrator->userdata;
    if (params == NULL || state == NULL || out_derivative == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_is_complex(field)) {
        const SimComplexDouble *src = (const SimComplexDouble *)state;
        SimComplexDouble *dst = (SimComplexDouble *)out_derivative;
        for (size_t i = 0; i < count; ++i) {
            double sr = src[i].re;
            double si = src[i].im;
            dst[i].re = params->lambda_re * sr - params->lambda_im * si;
            dst[i].im = params->lambda_re * si + params->lambda_im * sr;
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out_derivative[i] = params->lambda_re * state[i];
        }
    }

    return SIM_RESULT_OK;
}

static bool init_field(SimField *field, bool complex_case) {
    size_t shape[1] = {kElementCount};
    SimResult result =
        sim_field_init(field, 1U, shape, complex_case ? sizeof(SimComplexDouble) : sizeof(double),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] field init failed (%d)\n", result);
        return false;
    }

    if (complex_case) {
        SimComplexDouble *data = sim_field_complex_data(field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] complex field data pointer NULL\n");
            return false;
        }
        for (size_t i = 0; i < kElementCount; ++i) {
            data[i].re = kInitialValues[i];
            data[i].im = 0.0;
        }
    } else {
        double *data = sim_field_real_data(field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] real field data pointer NULL\n");
            return false;
        }
        for (size_t i = 0; i < kElementCount; ++i) {
            data[i] = kInitialValues[i];
        }
    }

    return true;
}

static double evaluate_real_error(const SimField *field, double lambda_re, double elapsed) {
    const double *data = sim_field_real_data_const(field);
    double scale = exp(lambda_re * elapsed);
    double max_error = 0.0;

    for (size_t i = 0; i < kElementCount; ++i) {
        double expected = kInitialValues[i] * scale;
        double diff = fabs(data[i] - expected);
        if (diff > max_error) {
            max_error = diff;
        }
    }

    return max_error;
}

static double evaluate_complex_error(const SimField *field, double lambda_re, double lambda_im,
                                     double elapsed) {
    const SimComplexDouble *data = sim_field_complex_data_const(field);
    double amplitude = exp(lambda_re * elapsed);
    double cos_term = cos(lambda_im * elapsed);
    double sin_term = sin(lambda_im * elapsed);
    double max_error = 0.0;

    for (size_t i = 0; i < kElementCount; ++i) {
        double expected_re = kInitialValues[i] * amplitude * cos_term;
        double expected_im = kInitialValues[i] * amplitude * sin_term;
        double dr = data[i].re - expected_re;
        double di = data[i].im - expected_im;
        double err = sqrt(dr * dr + di * di);
        if (err > max_error) {
            max_error = err;
        }
    }

    return max_error;
}

typedef struct FixedCaseSpec {
    const char *name;
    double lambda_re;
    double lambda_im;
    double dt;
    unsigned int steps;
    double allowed_error;
    bool complex_case;
} FixedCaseSpec;

typedef struct AdaptiveCaseSpec {
    const char *name;
    double lambda_re;
    double lambda_im;
    double target_time;
    double allowed_error;
    bool complex_case;
    float initial_dt;
    float min_dt;
    float max_dt;
    unsigned int max_iterations;
    float tolerance;
} AdaptiveCaseSpec;

static bool run_fixed_case(const FixedCaseSpec *spec) {
    SimField field = {0};
    IntegratorRegistry registry = {0};
    Integrator integrator = {0};
    LinearDriftParams params = {.lambda_re = spec->lambda_re, .lambda_im = spec->lambda_im};
    bool ok = false;

    if (!init_field(&field, spec->complex_case)) {
        goto cleanup;
    }

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] failed to init integrator registry\n");
        goto cleanup;
    }

    IntegratorCreateFn factory = integrator_registry_lookup(&registry, spec->name);
    if (factory == NULL) {
        fprintf(stderr, "[FAIL] integrator '%s' not found\n", spec->name);
        goto cleanup;
    }

    IntegratorConfig config;
    memset(&config, 0, sizeof(config));
    config.drift = linear_drift;
    config.userdata = &params;
    config.initial_dt = (float)spec->dt;
    config.min_dt = (float)spec->dt;
    config.max_dt = (float)spec->dt;
    config.tolerance = (float)fmax(1e-6, spec->allowed_error * 0.5);
    config.adaptive = false;
    config.safety = 0.9f;

    if (factory(&config, &integrator) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integrator '%s' configuration failed\n", spec->name);
        goto cleanup;
    }

    double elapsed = 0.0;
    for (unsigned int i = 0; i < spec->steps; ++i) {
        integrator.step(&integrator, &field, (float)spec->dt);
        elapsed += (double)integrator_last_step(&integrator);
    }

    double error = spec->complex_case
                       ? evaluate_complex_error(&field, spec->lambda_re, spec->lambda_im, elapsed)
                       : evaluate_real_error(&field, spec->lambda_re, elapsed);

    if (error > spec->allowed_error) {
        fprintf(stderr, "[FAIL] %s (%s) error %.4e exceeds %.4e\n", spec->name,
                spec->complex_case ? "complex" : "real", error, spec->allowed_error);
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&integrator);
    integrator_registry_destroy(&registry);
    sim_field_destroy(&field);
    return ok;
}

static bool run_adaptive_case(const AdaptiveCaseSpec *spec) {
    SimField field = {0};
    IntegratorRegistry registry = {0};
    Integrator integrator = {0};
    LinearDriftParams params = {.lambda_re = spec->lambda_re, .lambda_im = spec->lambda_im};
    bool ok = false;

    if (!init_field(&field, spec->complex_case)) {
        goto cleanup;
    }

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] failed to init integrator registry\n");
        goto cleanup;
    }

    IntegratorCreateFn factory = integrator_registry_lookup(&registry, spec->name);
    if (factory == NULL) {
        fprintf(stderr, "[FAIL] integrator '%s' not found\n", spec->name);
        goto cleanup;
    }

    IntegratorConfig config;
    memset(&config, 0, sizeof(config));
    config.drift = linear_drift;
    config.userdata = &params;
    config.initial_dt = spec->initial_dt;
    config.min_dt = spec->min_dt;
    config.max_dt = spec->max_dt;
    config.tolerance = spec->tolerance;
    config.adaptive = true;
    config.safety = 0.9f;

    if (factory(&config, &integrator) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integrator '%s' configuration failed\n", spec->name);
        goto cleanup;
    }

    double elapsed = 0.0;
    unsigned int iterations = 0U;
    while (elapsed < spec->target_time - 1e-6 && iterations < spec->max_iterations) {
        integrator.step(&integrator, &field, 0.0f);
        double last = (double)integrator_last_step(&integrator);
        if (!(last > 0.0)) {
            last = spec->initial_dt;
        }
        elapsed += last;
        iterations += 1U;
    }

    if (iterations >= spec->max_iterations) {
        fprintf(stderr, "[FAIL] integrator '%s' adaptive loop exceeded %u iterations\n", spec->name,
                spec->max_iterations);
        goto cleanup;
    }

    double error = spec->complex_case
                       ? evaluate_complex_error(&field, spec->lambda_re, spec->lambda_im, elapsed)
                       : evaluate_real_error(&field, spec->lambda_re, elapsed);

    if (error > spec->allowed_error) {
        fprintf(stderr, "[FAIL] %s (%s adaptive) error %.4e exceeds %.4e\n", spec->name,
                spec->complex_case ? "complex" : "real", error, spec->allowed_error);
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&integrator);
    integrator_registry_destroy(&registry);
    sim_field_destroy(&field);
    return ok;
}

static bool test_heun_adaptive_rejection_shrinks_dt(void) {
    SimField field = {0};
    IntegratorRegistry registry = {0};
    Integrator integrator = {0};
    LinearDriftParams params = {.lambda_re = -20.0, .lambda_im = 0.0};
    bool ok = false;

    if (!init_field(&field, false)) {
        goto cleanup;
    }

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] failed to init integrator registry for Heun adaptive test\n");
        goto cleanup;
    }

    IntegratorCreateFn factory = integrator_registry_lookup(&registry, "heun");
    if (factory == NULL) {
        fprintf(stderr, "[FAIL] heun integrator not found\n");
        goto cleanup;
    }

    IntegratorConfig config;
    memset(&config, 0, sizeof(config));
    config.drift = linear_drift;
    config.userdata = &params;
    config.initial_dt = 0.5f;
    config.min_dt = 1.0e-4f;
    config.max_dt = 1.0f;
    config.tolerance = 1.0e-6f;
    config.adaptive = true;
    config.safety = 0.9f;

    if (factory(&config, &integrator) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heun integrator configuration failed\n");
        goto cleanup;
    }

    double requested_dt = integrator_next_step(&integrator);
    integrator.step(&integrator, &field, 0.0f);

    double accepted_dt = integrator_last_step(&integrator);
    if (!(accepted_dt > 0.0) || !(accepted_dt < requested_dt)) {
        fprintf(stderr,
                "[FAIL] heun adaptive rejection should shrink dt (requested=%.6f accepted=%.6f)\n",
                requested_dt, accepted_dt);
        goto cleanup;
    }

    double suggested_dt = integrator_next_step(&integrator);
    if (!(suggested_dt > 0.0) || suggested_dt < accepted_dt) {
        fprintf(
            stderr,
            "[FAIL] heun adaptive next dt should remain positive and nondecreasing after acceptance"
            " (accepted=%.6f next=%.6f)\n",
            accepted_dt, suggested_dt);
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&integrator);
    integrator_registry_destroy(&registry);
    sim_field_destroy(&field);
    return ok;
}

static bool test_rkf45_stops_retrying_at_min_dt(void) {
    SimField field = {0};
    IntegratorRegistry registry = {0};
    Integrator integrator = {0};
    LinearDriftParams params = {.lambda_re = -20.0, .lambda_im = 0.0};
    bool ok = false;

    if (!init_field(&field, false)) {
        goto cleanup;
    }

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] failed to init integrator registry for RKF45 min_dt test\n");
        goto cleanup;
    }

    IntegratorCreateFn factory = integrator_registry_lookup(&registry, "rkf45");
    if (factory == NULL) {
        fprintf(stderr, "[FAIL] rkf45 integrator not found\n");
        goto cleanup;
    }

    IntegratorConfig config;
    memset(&config, 0, sizeof(config));
    config.drift = linear_drift;
    config.userdata = &params;
    config.initial_dt = 0.5f;
    config.min_dt = 0.05f;
    config.max_dt = 1.0f;
    config.tolerance = 1.0e-15f;
    config.adaptive = true;
    config.safety = 0.9f;

    if (factory(&config, &integrator) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rkf45 integrator configuration failed\n");
        goto cleanup;
    }

    integrator.step(&integrator, &field, 0.0f);

    if (!(integrator.last_step <= integrator.min_dt * 1.000001)) {
        fprintf(stderr, "[FAIL] rkf45 expected min_dt floor acceptance (step=%.6f min_dt=%.6f)\n",
                integrator.last_step, integrator.min_dt);
        goto cleanup;
    }

    if (!(integrator.last_attempt_count < 8U)) {
        fprintf(stderr, "[FAIL] rkf45 should stop retrying after reaching min_dt (attempts=%u)\n",
                (unsigned int)integrator.last_attempt_count);
        goto cleanup;
    }

    if (integrator.last_rejection_count + 1U != integrator.last_attempt_count) {
        fprintf(stderr, "[FAIL] rkf45 rejection accounting mismatch (attempts=%u rejections=%u)\n",
                (unsigned int)integrator.last_attempt_count,
                (unsigned int)integrator.last_rejection_count);
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&integrator);
    integrator_registry_destroy(&registry);
    sim_field_destroy(&field);
    return ok;
}

int main(void) {
    const FixedCaseSpec fixed_specs[] = {
        {.name = "euler",
         .lambda_re = -1.0,
         .lambda_im = 0.0,
         .dt = 0.01,
         .steps = 100U,
         .allowed_error = 3.0e-3,
         .complex_case = false},
        {.name = "euler",
         .lambda_re = -1.0,
         .lambda_im = 2.0,
         .dt = 0.01,
         .steps = 100U,
         .allowed_error = 1.5e-2,
         .complex_case = true},
        {.name = "heun",
         .lambda_re = -1.0,
         .lambda_im = 0.0,
         .dt = 0.05,
         .steps = 20U,
         .allowed_error = 2.0e-4,
         .complex_case = false},
        {.name = "heun",
         .lambda_re = -1.0,
         .lambda_im = 2.0,
         .dt = 0.05,
         .steps = 20U,
         .allowed_error = 3.0e-3,
         .complex_case = true},
        {.name = "rk4",
         .lambda_re = -1.0,
         .lambda_im = 0.0,
         .dt = 0.1,
         .steps = 10U,
         .allowed_error = 2.0e-05,
         .complex_case = false},
        {.name = "rk4",
         .lambda_re = -1.0,
         .lambda_im = 2.0,
         .dt = 0.1,
         .steps = 10U,
         .allowed_error = 4.0e-05,
         .complex_case = true},
        {.name = "backward_euler",
         .lambda_re = -1.0,
         .lambda_im = 0.0,
         .dt = 0.0025,
         .steps = 400U,
         .allowed_error = 2.5e-3,
         .complex_case = false},
        {.name = "backward_euler",
         .lambda_re = -1.0,
         .lambda_im = 2.0,
         .dt = 0.0025,
         .steps = 400U,
         .allowed_error = 3.0e-3,
         .complex_case = true},
        {.name = "crank_nicolson",
         .lambda_re = -1.0,
         .lambda_im = 0.0,
         .dt = 0.01,
         .steps = 100U,
         .allowed_error = 5.0e-5,
         .complex_case = false},
        {.name = "crank_nicolson",
         .lambda_re = -1.0,
         .lambda_im = 2.0,
         .dt = 0.01,
         .steps = 100U,
         .allowed_error = 1.0e-4,
         .complex_case = true},
    };

    const AdaptiveCaseSpec adaptive_specs[] = {
        {.name = "rkf45",
         .lambda_re = -1.0,
         .lambda_im = 0.0,
         .target_time = 1.0,
         .allowed_error = 5.0e-6,
         .complex_case = false,
         .initial_dt = 0.05f,
         .min_dt = 1.0e-4f,
         .max_dt = 0.2f,
         .max_iterations = 256U,
         .tolerance = 1.0e-6f},
        {.name = "rkf45",
         .lambda_re = -1.0,
         .lambda_im = 2.0,
         .target_time = 1.0,
         .allowed_error = 8.0e-6,
         .complex_case = true,
         .initial_dt = 0.05f,
         .min_dt = 1.0e-4f,
         .max_dt = 0.2f,
         .max_iterations = 256U,
         .tolerance = 2.0e-6f},
    };

    bool ok = true;
    for (size_t i = 0; i < sizeof(fixed_specs) / sizeof(fixed_specs[0]); ++i) {
        ok &= run_fixed_case(&fixed_specs[i]);
    }

    for (size_t i = 0; i < sizeof(adaptive_specs) / sizeof(adaptive_specs[0]); ++i) {
        ok &= run_adaptive_case(&adaptive_specs[i]);
    }

    ok &= test_heun_adaptive_rejection_shrinks_dt();
    ok &= test_rkf45_stops_retrying_at_min_dt();

    return ok ? 0 : 1;
}
