/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kShape[] = {5U, 6U};
static const size_t kShapeWide[] = {4U, 96U};
static const double kTol = 1.0e-8;

static void gaussian_eval_value(const SimStimulusGaussianConfig *cfg, double x, double y, double t,
                                double *out_value) {
    double sigma_x = (cfg->sigma_x > 1.0e-9) ? cfg->sigma_x : 1.0;
    double sigma_y = (cfg->sigma_y > 1.0e-9) ? cfg->sigma_y : sigma_x;

    if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double center_x = cfg->coord.center_x + cfg->coord.velocity_x * t;
        double center_y = cfg->coord.center_y + cfg->coord.velocity_y * t;
        double dx = x - center_x;
        double dy = y - center_y;
        double env_x = exp(-0.5 * (dx * dx) / (sigma_x * sigma_x));
        double env_y = exp(-0.5 * (dy * dy) / (sigma_y * sigma_y));

        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            *out_value = cfg->amplitude * (env_x + env_y);
        } else {
            *out_value = cfg->amplitude * (env_x * env_y);
        }
        return;
    }

    double sample_x = x;
    double sample_y = y;
    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

    double u = sample_x;
    if (cfg->coord.mode == SIM_STIMULUS_COORD_AXIS) {
        u = (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) ? sample_y : sample_x;
    } else if (cfg->coord.mode == SIM_STIMULUS_COORD_ANGLE) {
        double s = sin(cfg->coord.angle);
        double c = cos(cfg->coord.angle);
        u = sample_x * c + sample_y * s;
    } else {
        double dx = sample_x - cfg->coord.center_x;
        double dy = sample_y - cfg->coord.center_y;
        u = hypot(dx, dy);
    }

    double center = cfg->coord.center_x + cfg->coord.velocity_x * t;
    double diff = u;
    if (cfg->coord.mode != SIM_STIMULUS_COORD_RADIAL &&
        cfg->coord.mode != SIM_STIMULUS_COORD_POLAR &&
        cfg->coord.mode != SIM_STIMULUS_COORD_AZIMUTH &&
        cfg->coord.mode != SIM_STIMULUS_COORD_ELLIPTIC &&
        cfg->coord.mode != SIM_STIMULUS_COORD_SPIRAL) {
        diff = u - center;
    }

    *out_value = cfg->amplitude * exp(-0.5 * (diff * diff) / (sigma_x * sigma_x));
}

static size_t element_count(size_t rank, const size_t *shape) {
    size_t count = 1U;
    for (size_t axis = 0U; axis < rank; ++axis) {
        count *= shape[axis];
    }
    return count;
}

static int run_gaussian_case(const char *label, const SimStimulusGaussianConfig *cfg_input,
                             size_t rank, const size_t *shape, double dt, int steps, double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_context_init\n", label);
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimField field = {0};
    if (sim_field_init(&field, rank, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_field_init\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimComplexDouble *initial = sim_field_complex_data(&field);
        if (initial == NULL) {
            fprintf(stderr, "FAIL[%s]: field data init\n", label);
            sim_field_destroy(&field);
            sim_context_destroy(&ctx);
            return 0;
        }
        memset(initial, 0, sim_field_bytes(&field));
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_context_add_field\n", label);
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusGaussianConfig cfg = {0};
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_gaussian_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_add_stimulus_gaussian_operator\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusGaussianConfig normalized = {0};
    if (sim_stimulus_gaussian_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_stimulus_gaussian_config\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL[%s]: operator lookup\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }
    if (op->info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
        !op->info.representation.requires_complex_input ||
        !op->info.representation.requires_complex_representation ||
        !op->info.representation.preserves_real_subspace) {
        fprintf(stderr, "FAIL[%s]: complex representation metadata mismatch\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = element_count(rank, shape);
    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (expected == NULL) {
        fprintf(stderr, "FAIL[%s]: out of memory\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    double sin_r = sin(normalized.rotation);
    double cos_r = cos(normalized.rotation);
    for (int step = 0; step < steps; ++step) {
        double scale = normalized.scale_by_dt ? dt : 1.0;
        double t = (double)step * dt + normalized.time_offset;
        size_t width = sim_field_width(sim_context_field(&ctx, field_index));

        for (size_t i = 0U; i < count; ++i) {
            size_t row = i / width;
            size_t col = i % width;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double value = 0.0;
            double out_re;
            double out_im;

            gaussian_eval_value(&normalized, x, y, t, &value);
            out_re = value * cos_r;
            out_im = value * sin_r;
            expected[i].re += scale * out_re;
            expected[i].im += scale * out_im;
        }

        if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL[%s]: operator evaluate at step %d\n", label, step);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }
        sim_context_accept_step(&ctx, (float)dt);
    }

    SimField *result_field = sim_context_field(&ctx, field_index);
    if (result_field == NULL) {
        fprintf(stderr, "FAIL[%s]: result field lookup\n", label);
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
            fprintf(stderr,
                    "FAIL[%s]: mismatch at %zu got=(%.12g, %.12g) expected=(%.12g, %.12g)\n", label,
                    i, values[i].re, values[i].im, expected[i].re, expected[i].im);
            ok = 0;
            break;
        }
    }

    free(expected);
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    SimStimulusGaussianConfig cfg = {0};
    cfg.amplitude = 0.41;
    cfg.sigma_x = 0.82;
    cfg.sigma_y = 1.07;
    cfg.time_offset = 0.04;
    cfg.rotation = 0.23;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;
    cfg.coord.center_x = -0.16;
    cfg.coord.center_y = 0.09;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.coord.origin_x = -0.9;
    cfg.coord.origin_y = -0.7;
    cfg.coord.spacing_x = 0.34;
    cfg.coord.spacing_y = 0.28;

    if (!run_gaussian_case("separable_small", &cfg, 2U, kShape, 0.05, 3, kTol)) {
        return 1;
    }

    cfg.amplitude = 0.37;
    cfg.sigma_x = 0.74;
    cfg.sigma_y = 0.96;
    cfg.time_offset = 0.03;
    cfg.rotation = -0.18;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.38;
    cfg.coord.center_x = 0.11;
    cfg.coord.center_y = -0.08;
    cfg.coord.velocity_x = 0.025;
    cfg.coord.velocity_y = -0.015;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.41;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;

    if (!run_gaussian_case("angle_wide", &cfg, 2U, kShapeWide, 0.04, 2, kTol)) {
        return 1;
    }

    return 0;
}
