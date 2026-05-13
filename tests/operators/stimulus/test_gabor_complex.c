/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const size_t kShape[] = {64};
static const double kTol = 1.0e-6;
static const int kStressSteps = 120000;
static const double kStressDt = 1.0e-3;

static int run_gabor_case(const char *label, const SimStimulusGaborConfig *cfg_input, double dt,
                          int steps, double tol) {
    SimContext ctx;
    int rc = 1;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

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
    memset(sim_field_complex_data(&field), 0, shape[0] * sizeof(SimComplexDouble));

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to add field to context\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusGaborConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_gabor_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Failed to register gabor stimulus\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (!op) {
        fprintf(stderr, "[%s] Operator lookup failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }
    if (op->info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
        !op->info.representation.requires_complex_input ||
        !op->info.representation.requires_complex_representation ||
        !op->info.representation.preserves_real_subspace) {
        fprintf(stderr, "[%s] Complex representation metadata mismatch\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    /* Accumulate expected values alongside operator execution. */
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
    double sigma = (cfg.sigma_x > 0.0) ? cfg.sigma_x : 1.0;
    double scale = cfg.scale_by_dt ? dt : 1.0;

    for (int s = 0; s < steps; ++s) {
        double drive_t = (double)s * dt + cfg.time_offset;
        double center = cfg.coord.center_x;
        double inv_two_sigma_sq = 0.5 / (sigma * sigma);

        for (size_t i = 0; i < count; ++i) {
            double x = cfg.coord.origin_x + (double)i * spacing;
            double sample_x = x - cfg.coord.velocity_x * drive_t;
            double diff = sample_x - center;
            double envelope = cfg.amplitude * exp(-diff * diff * inv_two_sigma_sq);
            double theta = cfg.wavenumber * sample_x - cfg.omega * drive_t + cfg.phase;
            double c = cos(theta);
            double sgn = sin(theta);
            double re = envelope * c;
            double im = envelope * sgn;
            double out_re = re * cos_r - im * sin_r;
            double out_im = re * sin_r + im * cos_r;
            expected[i].re += scale * out_re;
            expected[i].im += scale * out_im;
        }

        if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
            fprintf(stderr, "[%s] Stimulus apply failed at step %d\n", label, s);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }
        sim_context_accept_step(&ctx, (float)dt);
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
    SimStimulusGaborConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 4.0e-4;
    cfg.coord.center_x = 0.5;
    cfg.sigma_x = 0.8;
    cfg.wavenumber = 1.3;
    cfg.omega = 0.2;
    cfg.phase = 0.15;
    cfg.coord.velocity_x = 0.04;
    cfg.coord.origin_x = 0.0;
    cfg.coord.spacing_x = 0.05;
    cfg.rotation = 0.2;
    cfg.scale_by_dt = false;

    int ok = 1;
    ok &= run_gabor_case("gabor_complex_dt_independent", &cfg, kStressDt, kStressSteps, kTol);
    return ok ? 0 : 1;
}
