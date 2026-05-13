#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>

static double expected_sample(const SimPole *poles, size_t pole_count, double x, double y,
                              double plane_z, double softening) {
    if (poles == NULL || pole_count == 0U) {
        return 0.0;
    }

    double soft = (softening > 0.0) ? softening : 1.0e-6;
    double soft_sq = soft * soft;
    double value = 0.0;

    for (size_t i = 0U; i < pole_count; ++i) {
        double dx = x - (double)poles[i].x;
        double dy = y - (double)poles[i].y;
        double dz = plane_z - (double)poles[i].z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double denom = sqrt(r2 + soft_sq);
        if (!(denom > 0.0)) {
            denom = soft;
        }
        value += (double)poles[i].residue / denom;
    }

    return value;
}

int main(void) {
    SimPole poles[2] = {{.x = 0.0f, .y = 0.0f, .z = 0.0f, .residue = 2.0f, .type = "digamma"},
                        {.x = 2.0f, .y = -1.0f, .z = 0.5f, .residue = -0.5f, .type = "trigamma"}};

    SimUniverseSpec spec = {.poles = poles,
                            .pole_count = 2U,
                            .q = 1.0f,
                            .K = 20U,
                            .epsilon = 0.5f,
                            .sieve_sigma = 0.1f};

    SimContext context;
    if (sim_context_init_with_universe(&context, &spec) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_init_with_universe failed\n");
        return 1;
    }

    size_t shape[1] = {8U};
    SimField real_field = {0};
    if (sim_field_init(&real_field, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_field_init real failed\n");
        sim_context_destroy(&context);
        return 1;
    }

    size_t real_index = 0U;
    if (sim_context_add_field(&context, &real_field, &real_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_add_field real failed\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimField complex_field = {0};
    if (sim_field_init(&complex_field, 1, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_field_init complex failed\n");
        sim_context_destroy(&context);
        return 1;
    }

    size_t complex_index = 0U;
    if (sim_context_add_field(&context, &complex_field, &complex_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_add_field complex failed\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimPoleFieldOptions opts = sim_pole_field_options_default();
    opts.spacing_x = 0.5;
    opts.softening = 0.25;
    opts.plane_z = 0.25;

    if (sim_context_synthesize_pole_field(&context, real_index, &opts) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] synthesize real field failed\n");
        sim_context_destroy(&context);
        return 1;
    }

    if (sim_context_synthesize_pole_field(&context, complex_index, &opts) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] synthesize complex field failed\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimField *real_ctx_field = sim_context_field(&context, real_index);
    const double *real_data =
        real_ctx_field ? (const double *)sim_field_data_const(real_ctx_field) : NULL;
    SimField *complex_ctx_field = sim_context_field(&context, complex_index);
    const SimComplexDouble *complex_data =
        complex_ctx_field ? sim_field_complex_data(complex_ctx_field) : NULL;

    if (real_data == NULL || complex_data == NULL) {
        fprintf(stderr, "[FAIL] synthesized data missing\n");
        sim_context_destroy(&context);
        return 1;
    }

    const double tol = 1.0e-6;
    for (size_t i = 0U; i < shape[0]; ++i) {
        double x = opts.origin_x + (double)i * opts.spacing_x;
        double expected =
            expected_sample(poles, 2U, x, opts.origin_y, opts.plane_z, opts.softening);
        if (fabs(real_data[i] - expected) > tol) {
            fprintf(stderr, "[FAIL] real sample mismatch at %zu: got %.9f expected %.9f\n", i,
                    real_data[i], expected);
            sim_context_destroy(&context);
            return 1;
        }
        if (fabs(complex_data[i].re - expected) > tol || fabs(complex_data[i].im) > tol) {
            fprintf(stderr,
                    "[FAIL] complex sample mismatch at %zu: got (%.9f, %.9f) expected %.9f\n", i,
                    complex_data[i].re, complex_data[i].im, expected);
            sim_context_destroy(&context);
            return 1;
        }
    }

    sim_context_destroy(&context);
    printf("test_pole_field ok\n");
    return 0;
}
