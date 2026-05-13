/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int approx(double a, double b, double eps) { return fabs(a - b) <= eps; }

static int run_case(const char *label, double period_y, const double *expected, size_t count) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_context_init\n", label);
        return 0;
    }

    size_t shape[2] = {4U, 1U};
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_field_init\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_context_add_field\n", label);
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCheckerboardConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 1.0;
    cfg.period_x = 100.0;
    cfg.period_y = period_y;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.spacing_x = 1.0;
    cfg.coord.spacing_y = 0.25;

    size_t op_index = 0U;
    if (sim_add_stimulus_checkerboard_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_add_stimulus_checkerboard_operator\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate == NULL) {
        fprintf(stderr, "[%s] FAIL: operator evaluate unavailable\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: operator evaluate\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimField *result = sim_context_field(&ctx, field_index);
        if (result == NULL) {
            fprintf(stderr, "[%s] FAIL: sim_context_field\n", label);
            sim_context_destroy(&ctx);
            return 0;
        }

        const double *values = (const double *)sim_field_data(result);
        for (size_t i = 0U; i < count; ++i) {
            if (!approx(values[i], expected[i], 1.0e-12)) {
                fprintf(stderr, "[%s] FAIL: value[%zu] got=%.17g expected=%.17g\n", label, i,
                        values[i], expected[i]);
                sim_context_destroy(&ctx);
                return 0;
            }
        }
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int run_complex_wide_case(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: sim_context_init\n");
        return 0;
    }

    size_t shape[2] = {4U, 96U};
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: promote_inplace_to_complex\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_complex_data(&field), 0, shape[0] * shape[1] * sizeof(SimComplexDouble));

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: sim_context_add_field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCheckerboardConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.42;
    cfg.period_x = 1.15;
    cfg.period_y = 0.72;
    cfg.phase = 1.1;
    cfg.complex_phase = 0.27;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.37;
    cfg.coord.origin_x = -1.48;
    cfg.coord.origin_y = -0.42;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    size_t op_index = 0U;
    if (sim_add_stimulus_checkerboard_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate == NULL) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: operator evaluate unavailable\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: operator evaluate\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCheckerboardConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_checkerboard_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_complex_wide] FAIL: config fetch\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimField *result = sim_context_field(&ctx, field_index);
        if (result == NULL) {
            fprintf(stderr, "[checkerboard_complex_wide] FAIL: sim_context_field\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        const SimComplexDouble *values = sim_field_complex_data_const(result);
        double t = sim_context_time(&ctx);
        double s = sin(normalized.coord.angle);
        double c = cos(normalized.coord.angle);
        double cr = cos(normalized.complex_phase);
        double sr = sin(normalized.complex_phase);
        int64_t cell_y = (normalized.period_y > 0.0) ? (int64_t)floor(normalized.phase) : 0LL;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t;
                double sample_y = y - normalized.coord.velocity_y * t;
                double u = sample_x * c + sample_y * s;
                int64_t cell_x = (int64_t)floor(u / normalized.period_x + normalized.phase);
                double sign = (((cell_x + cell_y) & 1LL) == 0LL) ? 1.0 : -1.0;
                double expected_re = normalized.amplitude * sign * cr;
                double expected_im = normalized.amplitude * sign * sr;

                if (!approx(values[idx].re, expected_re, 1.0e-12) ||
                    !approx(values[idx].im, expected_im, 1.0e-12)) {
                    fprintf(stderr,
                            "[checkerboard_complex_wide] FAIL: value[%zu] got=(%.17g, %.17g) "
                            "expected=(%.17g, %.17g)\n",
                            idx, values[idx].re, values[idx].im, expected_re, expected_im);
                    sim_context_destroy(&ctx);
                    return 0;
                }
            }
        }
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int run_spiral_scalar_case(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: sim_context_init\n");
        return 0;
    }

    size_t shape[2] = {5U, 31U};
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCheckerboardConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.63;
    cfg.period_x = 0.95;
    cfg.period_y = 0.0;
    cfg.phase = 0.35;
    cfg.complex_phase = 0.0;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.origin_x = -0.8;
    cfg.coord.origin_y = -0.35;
    cfg.coord.spacing_x = 0.11;
    cfg.coord.spacing_y = 0.17;
    cfg.coord.center_x = 0.25;
    cfg.coord.center_y = -0.1;
    cfg.coord.velocity_x = 0.07;
    cfg.coord.velocity_y = -0.05;
    cfg.coord.spiral_arms = 1.4;
    cfg.coord.spiral_pitch = 0.85;
    cfg.coord.spiral_phase = 0.2;
    cfg.coord.spiral_angular_velocity = 0.3;

    size_t op_index = 0U;
    if (sim_add_stimulus_checkerboard_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate == NULL) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: operator evaluate unavailable\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCheckerboardConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_checkerboard_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: config fetch\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    const double eval_t = 0.0;
    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: operator evaluate\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimField *result = sim_context_field(&ctx, field_index);
        if (result == NULL) {
            fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: sim_context_field\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        const double *values = sim_field_real_data_const(result);
        if (values == NULL) {
            fprintf(stderr, "[checkerboard_spiral_scalar] FAIL: field data missing\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double u = sim_stimulus_coord_u(&normalized.coord, x, y, eval_t);
                int64_t cell_x = (int64_t)floor((u / normalized.period_x) + normalized.phase);
                double expected =
                    ((cell_x & 1LL) == 0LL) ? normalized.amplitude : -normalized.amplitude;

                if (!approx(values[idx], expected, 1.0e-12)) {
                    fprintf(
                        stderr,
                        "[checkerboard_spiral_scalar] FAIL: value[%zu] got=%.17g expected=%.17g\n",
                        idx, values[idx], expected);
                    sim_context_destroy(&ctx);
                    return 0;
                }
            }
        }
    }

    sim_context_destroy(&ctx);
    return 1;
}

int main(void) {
    static const double kExpectedFractionalY[] = {1.0, 1.0, -1.0, -1.0};
    static const double kExpectedStripes[] = {1.0, 1.0, 1.0, 1.0};

    int ok = 1;
    ok &= run_case("fractional_period_y", 0.5, kExpectedFractionalY, 4U);
    ok &= run_case("stripe_period_y", 0.0, kExpectedStripes, 4U);
    ok &= run_complex_wide_case();
    ok &= run_spiral_scalar_case();
    return ok ? 0 : 1;
}
