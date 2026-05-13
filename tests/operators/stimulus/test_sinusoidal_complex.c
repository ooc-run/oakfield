/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <math.h>
#include <oakfield/sim.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const size_t kShape[] = {64};
static const double kTol = 1.0e-8;
static const int kStressSteps = 120000;
static const double kStressDt = 1.0e-3;

static int run_sinusoidal_case(const char *label, const SimStimulusSinusoidalConfig *cfg_input,
                               double dt, int steps, double tol) {
    SimContext ctx;
    int rc = 1;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, dt);

    size_t shape[1] = {kShape[0]};
    SimField field = {0};
    if (sim_field_init(&field, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "Failed to init field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
        fprintf(stderr, "Promote to complex failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to add field to context\n");
        sim_field_destroy(&field); /* only valid on failure */
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusSinusoidalConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_sine_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Failed to register sinusoidal stimulus\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (!op) {
        fprintf(stderr, "[%s] Operator lookup failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    /* Accumulate expected values in tandem with operator execution. */
    size_t count = shape[0];
    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (expected == NULL) {
        fprintf(stderr, "[%s] Out of memory\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    double cos_r = cos(cfg.rotation);
    double sin_r = sin(cfg.rotation);
    double spacing = (fabs(cfg.coord.spacing_x) > 0.0) ? cfg.coord.spacing_x : 1.0;
    double scale = cfg.scale_by_dt ? dt : 1.0;

    for (int s = 0; s < steps; ++s) {
        double drive_t = (double)s * dt + cfg.time_offset;
        for (size_t i = 0; i < count; ++i) {
            double x = cfg.coord.origin_x + i * spacing;
            double theta = cfg.wavenumber * x - cfg.omega * drive_t + cfg.phase;
            double value = cfg.amplitude * sin(theta);
            expected[i].re += scale * value * cos_r;
            expected[i].im += scale * value * sin_r;
        }

        if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
            fprintf(stderr, "[%s] Stimulus apply failed at step %d\n", label, s);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }
        sim_context_accept_step(&ctx, dt);
    }

    SimField *f2 = sim_context_field(&ctx, field_index);
    if (!f2) {
        fprintf(stderr, "[%s] Field retrieval failed\n", label);
        free(expected);
        sim_context_destroy(&ctx);
        return 0;
    }

    const SimComplexDouble *z = sim_field_complex_data_const(f2);
    double max_err = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double err_re = fabs(z[i].re - expected[i].re);
        double err_im = fabs(z[i].im - expected[i].im);
        double err = fmax(err_re, err_im);
        if (err > max_err) {
            max_err = err;
        }
        if (err > tol) {
            fprintf(stderr,
                    "[%s] Mismatch at i=%zu re=%.12g (exp=%.12g) im=%.12g (exp=%.12g) err=%.3g "
                    "tol=%.3g\n",
                    label, i, z[i].re, expected[i].re, z[i].im, expected[i].im, err, tol);
            rc = 0;
            break;
        }
    }

    if (rc && max_err > tol) {
        fprintf(stderr, "[%s] Max error %.3g exceeded tol %.3g\n", label, max_err, tol);
        rc = 0;
    }

    free(expected);
    sim_context_destroy(&ctx);

    if (!rc) {
        fprintf(stderr, "[%s] FAILED (max_err=%.3g tol=%.3g)\n", label, max_err, tol);
    } else {
        fprintf(stdout, "[%s] ok (max_err=%.3g tol=%.3g)\n", label, max_err, tol);
    }

    return rc;
}

int main(void) {
    SimStimulusSinusoidalConfig cfg1;
    memset(&cfg1, 0, sizeof(cfg1));
    cfg1.amplitude = 5.0e-4;
    cfg1.wavenumber = 1.5;
    cfg1.omega = 0.3;
    cfg1.phase = 0.1;
    cfg1.rotation = 0.35;
    cfg1.scale_by_dt = false;

    SimStimulusSinusoidalConfig cfg2 = cfg1;
    cfg2.scale_by_dt = true;
    cfg2.rotation = -0.2;
    cfg2.omega = 0.15;

    int ok = 1;
    ok &= run_sinusoidal_case("sinusoidal_complex_dt_independent", &cfg1, kStressDt, kStressSteps,
                              kTol);
    ok &= run_sinusoidal_case("sinusoidal_complex_dt_scaled", &cfg2, kStressDt, kStressSteps, kTol);
    return ok ? 0 : 1;
}
