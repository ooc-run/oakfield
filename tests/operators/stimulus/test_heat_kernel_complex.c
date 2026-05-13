/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kShape[] = {64U};
static const double kTol = 1.0e-8;

static int run_heat_kernel_case(const SimStimulusHeatKernelConfig *cfg_input, double dt, int steps,
                                double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    size_t shape[1] = {kShape[0]};
    SimField field = {0};
    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    size_t count = shape[0];

    if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_promote_inplace_to_complex\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_complex_data(&field), 0, count * sizeof(SimComplexDouble));

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusHeatKernelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_heat_kernel_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stimulus_heat_kernel_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: operator lookup\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (expected == NULL) {
        fprintf(stderr, "FAIL: out of memory\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double sigma0 = (cfg.sigma_x > 0.0) ? cfg.sigma_x : 1.0;
    double spacing = (fabs(cfg.coord.spacing_x) > 0.0) ? cfg.coord.spacing_x : 1.0;
    double cos_r = cos(cfg.rotation);
    double sin_r = sin(cfg.rotation);
    double scale = cfg.scale_by_dt ? dt : 1.0;

    for (int step = 0; step < steps; ++step) {
        double drive_t = (double)step * dt + cfg.time_offset;
        double t_pos = (drive_t > 0.0) ? drive_t : 0.0;
        double sigma_sq = sigma0 * sigma0 + 2.0 * cfg.diffusivity * t_pos;
        double sigma_eff = sqrt(sigma_sq);
        double norm = cfg.preserve_mass ? (sigma0 / sigma_eff) : 1.0;
        double center = cfg.coord.center_x + cfg.coord.velocity_x * drive_t;

        for (size_t i = 0U; i < count; ++i) {
            double x = cfg.coord.origin_x + (double)i * spacing;
            double sample_x = x - cfg.coord.velocity_x * drive_t;
            double diff = sample_x - center;
            double value = cfg.amplitude * norm * exp(-0.5 * diff * diff / sigma_sq);
            expected[i].re += scale * value * cos_r;
            expected[i].im += scale * value * sin_r;
        }

        if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: operator evaluate at step %d\n", step);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }
        sim_context_accept_step(&ctx, (float)dt);
    }

    SimField *result_field = sim_context_field(&ctx, field_index);
    if (result_field == NULL) {
        fprintf(stderr, "FAIL: result field lookup\n");
        free(expected);
        sim_context_destroy(&ctx);
        return 0;
    }

    const SimComplexDouble *values = sim_field_complex_data_const(result_field);
    int ok = 1;
    for (size_t i = 0U; i < count; ++i) {
        double err_re = fabs(values[i].re - expected[i].re);
        double err_im = fabs(values[i].im - expected[i].im);
        if (err_re > tol || err_im > tol) {
            fprintf(stderr, "FAIL: mismatch at %zu got=(%.12g, %.12g) expected=(%.12g, %.12g)\n", i,
                    values[i].re, values[i].im, expected[i].re, expected[i].im);
            ok = 0;
            break;
        }
    }

    free(expected);
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    SimStimulusHeatKernelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 4.0e-4;
    cfg.diffusivity = 0.18;
    cfg.sigma_x = 0.35;
    cfg.coord.center_x = 0.6;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.origin_x = -0.5;
    cfg.coord.spacing_x = 0.05;
    cfg.rotation = 0.25;
    cfg.preserve_mass = true;
    cfg.scale_by_dt = false;

    return run_heat_kernel_case(&cfg, 1.0e-3, 3000, kTol) ? 0 : 1;
}
