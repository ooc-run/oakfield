/*
 * Migrated coupling operator coverage for complex mixer contracts.
 */
#include <oakfield/sim.h>

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* Helper for small tolerance comparisons */
static int nearly_equal(double a, double b, double tol) { return fabs(a - b) < tol; }

static int validate_output_samples(const SimField *lhs, const SimField *rhs, const SimField *out,
                                   const SimMixerOperatorConfig *cfg, int make_complex) {
    size_t count = sim_field_element_count(&out->layout);
    if (count == 0U) {
        return 1;
    }

    switch (cfg->mode) {
    case SIM_MIXER_MODE_LINEAR:
    case SIM_MIXER_MODE_SUM:
    case SIM_MIXER_MODE_CROSSFADE:
    case SIM_MIXER_MODE_AVERAGE:
    case SIM_MIXER_MODE_DIFFERENCE:
    case SIM_MIXER_MODE_FM:
    case SIM_MIXER_MODE_PM:
    case SIM_MIXER_MODE_FEEDBACK:
        break;
    default:
        return 1;
    }

    if (make_complex) {
        if (cfg->mode == SIM_MIXER_MODE_FM || cfg->mode == SIM_MIXER_MODE_PM) {
            return 1;
        }

        const SimComplexDouble *lhs_data = sim_field_complex_data_const(lhs);
        const SimComplexDouble *rhs_data = sim_field_complex_data_const(rhs);
        const SimComplexDouble *out_data = sim_field_complex_data_const(out);
        if (!lhs_data || !rhs_data || !out_data) {
            fprintf(stderr, "[FAIL] validate_output_samples: complex data unavailable\n");
            return 0;
        }

        for (size_t i = 0U; i < count; ++i) {
            double complex L = lhs_data[i].re + I * lhs_data[i].im;
            double complex R = rhs_data[i].re + I * rhs_data[i].im;
            double complex expected;

            switch (cfg->mode) {
            case SIM_MIXER_MODE_CROSSFADE:
                expected = (1.0 - cfg->mix) * L + cfg->mix * R + cfg->bias;
                break;
            case SIM_MIXER_MODE_AVERAGE:
                expected = 0.5 * (L + R) + cfg->bias;
                break;
            case SIM_MIXER_MODE_DIFFERENCE:
                expected = (L - R) + cfg->bias;
                break;
            case SIM_MIXER_MODE_FEEDBACK:
                expected = L + R + cfg->bias;
                break;
            case SIM_MIXER_MODE_LINEAR:
            case SIM_MIXER_MODE_SUM:
            default:
                expected = L + R + cfg->bias;
                break;
            }

            if (!nearly_equal(out_data[i].re, creal(expected), 1.0e-12) ||
                !nearly_equal(out_data[i].im, cimag(expected), 1.0e-12)) {
                fprintf(
                    stderr,
                    "[FAIL] mode=%d complex index=%zu got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                    (int)cfg->mode, i, out_data[i].re, out_data[i].im, creal(expected),
                    cimag(expected));
                return 0;
            }
        }
        return 1;
    }

    {
        const double *lhs_data = (const double *)sim_field_data((SimField *)lhs);
        const double *rhs_data = (const double *)sim_field_data((SimField *)rhs);
        const double *out_data = (const double *)sim_field_data((SimField *)out);
        if (!lhs_data || !rhs_data || !out_data) {
            fprintf(stderr, "[FAIL] validate_output_samples: real data unavailable\n");
            return 0;
        }

        for (size_t i = 0U; i < count; ++i) {
            double expected;
            switch (cfg->mode) {
            case SIM_MIXER_MODE_CROSSFADE:
                expected = (1.0 - cfg->mix) * lhs_data[i] + cfg->mix * rhs_data[i] + cfg->bias;
                break;
            case SIM_MIXER_MODE_AVERAGE:
                expected = 0.5 * (lhs_data[i] + rhs_data[i]) + cfg->bias;
                break;
            case SIM_MIXER_MODE_DIFFERENCE:
                expected = (lhs_data[i] - rhs_data[i]) + cfg->bias;
                break;
            case SIM_MIXER_MODE_FM:
            case SIM_MIXER_MODE_PM:
                expected = lhs_data[i] * cos(rhs_data[i]) + cfg->bias;
                break;
            case SIM_MIXER_MODE_FEEDBACK:
                expected = lhs_data[i] + rhs_data[i] + cfg->bias;
                break;
            case SIM_MIXER_MODE_LINEAR:
            case SIM_MIXER_MODE_SUM:
            default:
                expected = lhs_data[i] + rhs_data[i] + cfg->bias;
                break;
            }

            if (!nearly_equal(out_data[i], expected, 1.0e-12)) {
                fprintf(stderr, "[FAIL] mode=%d real index=%zu got=%.12g expected=%.12g\n",
                        (int)cfg->mode, i, out_data[i], expected);
                return 0;
            }
        }
    }

    return 1;
}

/* Compute total "energy" (sum of squared magnitudes) */
static double field_energy(const SimField *field) {
    if (sim_field_is_complex(field)) {
        const SimComplexDouble *data = sim_field_complex_data_const(field);
        size_t n = sim_field_element_count(&field->layout);
        double e = 0.0;
        for (size_t i = 0; i < n; ++i)
            e += data[i].re * data[i].re + data[i].im * data[i].im;
        return e;
    } else {
        const double *data = (const double *)sim_field_data((SimField *)field);
        size_t n = sim_field_element_count(&field->layout);
        double e = 0.0;
        for (size_t i = 0; i < n; ++i)
            e += data[i] * data[i];
        return e;
    }
}

/* Initialize sinusoidal test data */
static void fill_fields(SimField *lhs, SimField *rhs, size_t count, int make_complex) {
    const double w1 = 2.0 * M_PI / (double)count;
    const double w2 = 3.0 * M_PI / (double)count;

    if (make_complex) {
        SimResult res;
        res = sim_field_promote_inplace_to_complex(lhs);
        if (res != SIM_RESULT_OK) {
            fprintf(stderr, "[ERROR] Failed to promote lhs to complex\n");
            return;
        }
        res = sim_field_promote_inplace_to_complex(rhs);
        if (res != SIM_RESULT_OK) {
            fprintf(stderr, "[ERROR] Failed to promote rhs to complex\n");
            return;
        }

        SimComplexDouble *L = sim_field_complex_data(lhs);
        SimComplexDouble *R = sim_field_complex_data(rhs);

        for (size_t i = 0; i < count; ++i) {
            L[i].re = cos(w1 * (double)i);
            L[i].im = sin(w1 * (double)i);
            R[i].re = cos(w2 * (double)i);
            R[i].im = sin(w2 * (double)i);
        }
    } else {
        double *L = (double *)sim_field_data(lhs);
        double *R = (double *)sim_field_data(rhs);

        for (size_t i = 0; i < count; ++i) {
            L[i] = cos(w1 * (double)i);
            R[i] = cos(w2 * (double)i);
        }
    }
}

/* Run one mixer test for given mode */
static int run_mixer_mode(SimContext *ctx, SimMixerMode mode, int make_complex) {
    size_t lhs_idx = SIZE_MAX, rhs_idx = SIZE_MAX, out_idx = SIZE_MAX;
    size_t count = 64;
    double step = 1.0;
    size_t shape[1] = {count};
    SimField lhs = {0}, rhs = {0}, out = {0};
    SimResult result;
    int success = 0;

    /* Initialize fields */
    result = sim_field_init(&lhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lhs field init failed\n");
        return 0;
    }

    result = sim_field_init(&rhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rhs field init failed\n");
        sim_field_destroy(&lhs);
        return 0;
    }

    result = sim_field_init(&out, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] out field init failed\n");
        sim_field_destroy(&lhs);
        sim_field_destroy(&rhs);
        return 0;
    }

    /* Fill with test data */
    fill_fields(&lhs, &rhs, count, make_complex);

    /* Ensure output representation matches complex mode when needed. */
    if (make_complex) {
        if (sim_field_promote_inplace_to_complex(&out) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] out complex promotion failed\n");
            sim_field_destroy(&lhs);
            sim_field_destroy(&rhs);
            sim_field_destroy(&out);
            return 0;
        }
        SimComplexDouble *out_data = sim_field_complex_data(&out);
        if (out_data != NULL) {
            for (size_t i = 0U; i < count; ++i) {
                out_data[i].re = 0.0;
                out_data[i].im = 0.0;
            }
        }
    } else {
        double *out_data = (double *)sim_field_data(&out);
        if (out_data != NULL) {
            for (size_t i = 0U; i < count; ++i) {
                out_data[i] = 0.0;
            }
        }
    }

    /* Add fields to context */
    result = sim_context_add_field(ctx, &lhs, &lhs_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add lhs field failed\n");
        sim_field_destroy(&lhs);
        sim_field_destroy(&rhs);
        sim_field_destroy(&out);
        return 0;
    }

    result = sim_context_add_field(ctx, &rhs, &rhs_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add rhs field failed\n");
        sim_field_destroy(&rhs);
        sim_field_destroy(&out);
        return 0;
    }

    result = sim_context_add_field(ctx, &out, &out_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add out field failed\n");
        sim_field_destroy(&out);
        return 0;
    }

    /* Get field pointers from context (ownership transferred) */
    SimField *lhs_ptr = sim_context_field(ctx, lhs_idx);
    SimField *rhs_ptr = sim_context_field(ctx, rhs_idx);
    SimField *out_ptr = sim_context_field(ctx, out_idx);

    if (!lhs_ptr || !rhs_ptr || !out_ptr) {
        fprintf(stderr, "[FAIL] failed to retrieve field pointers\n");
        return 0;
    }

    /* Configure mixer operator */
    SimMixerOperatorConfig cfg;
    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.lhs_field = lhs_idx;
    cfg.rhs_field = rhs_idx;
    cfg.output_field = out_idx;
    cfg.lhs_gain = 1.0;
    cfg.rhs_gain = 1.0;
    cfg.mix = 0.25;
    cfg.bias = 0.0;
    cfg.mode = mode;
    cfg.accumulate = false;
    if (mode == SIM_MIXER_MODE_FEEDBACK) {
        cfg.feedback_epsilon = 1.0;
    }

    size_t op_idx = SIZE_MAX;
    result = sim_add_mixer_operator(ctx, &cfg, &op_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_mixer_operator mode=%d result=%d\n", mode, result);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx->world.operators, op_idx);
    if (!op) {
        fprintf(stderr, "[FAIL] operator registry missing for mode=%d\n", mode);
        return 0;
    }

    /* Apply the operator */
    result = op->evaluate(ctx, op, op->userdata);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] operator apply failed mode=%d result=%d\n", mode, result);
        return 0;
    }

    if (!validate_output_samples(lhs_ptr, rhs_ptr, out_ptr, &cfg, make_complex)) {
        return 0;
    }

    /* Check output energy */
    double e = field_energy(out_ptr);
    if (!isfinite(e)) {
        fprintf(stderr, "[FAIL] invalid energy mode=%d (%.6f)\n", mode, e);
        return 0;
    }

    /* Some modes may produce zero output (e.g., MIN/MAX with certain inputs) */
    fprintf(stdout, "[PASS] mode=%-2d %-20s energy=%.6f%s\n", mode, mixer_mode_name(mode), e,
            make_complex ? " (complex)" : "");

    return 1;
}

/* Entry point */
int main(void) {
    SimContext ctx = {0};
    SimResult result;

    result = sim_context_init(&ctx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FATAL] failed to initialize SimContext\n");
        return EXIT_FAILURE;
    }

    int ok = 1;

    /* Test real-valued fields */
    fprintf(stdout, "=== Testing real-valued fields ===\n");
    for (int mode = SIM_MIXER_MODE_LINEAR; mode <= SIM_MIXER_MODE_FEEDBACK; ++mode) {
        ok &= run_mixer_mode(&ctx, (SimMixerMode)mode, 0);
    }

    /* Reset context for complex tests */
    sim_context_destroy(&ctx);
    (void)memset(&ctx, 0, sizeof(ctx));
    result = sim_context_init(&ctx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FATAL] failed to reinitialize SimContext for complex tests\n");
        return EXIT_FAILURE;
    }

    /* Test complex-valued fields */
    fprintf(stdout, "\n=== Testing complex-valued fields ===\n");
    for (int mode = SIM_MIXER_MODE_LINEAR; mode <= SIM_MIXER_MODE_FEEDBACK; ++mode) {
        ok &= run_mixer_mode(&ctx, (SimMixerMode)mode, 1);
    }

    sim_context_destroy(&ctx);

    fprintf(stdout, "\n=== Mixer comprehensive test: %s ===\n",
            ok ? "ALL PASSED" : "FAILURES DETECTED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
