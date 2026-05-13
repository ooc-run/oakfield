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
static const double kTol = 1.0e-9;

static double coord_u(const SimStimulusCoordConfig *coord, double x, double y, double t) {
    if (coord == NULL) {
        return x;
    }

    switch (coord->mode) {
    case SIM_STIMULUS_COORD_AXIS: {
        double sample_x = x;
        double sample_y = y;
        sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);
        return (coord->axis == SIM_STIMULUS_AXIS_Y) ? sample_y : sample_x;
    }
    case SIM_STIMULUS_COORD_ANGLE: {
        double sample_x = x;
        double sample_y = y;
        double s = sin(coord->angle);
        double c = cos(coord->angle);
        sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);
        return sample_x * c + sample_y * s;
    }
    case SIM_STIMULUS_COORD_RADIAL: {
        double cx = coord->center_x + coord->velocity_x * t;
        double cy = coord->center_y + coord->velocity_y * t;
        double dx = x - cx;
        double dy = y - cy;
        return sqrt(dx * dx + dy * dy);
    }
    case SIM_STIMULUS_COORD_SPIRAL: {
        double cx = coord->center_x + coord->velocity_x * t;
        double cy = coord->center_y + coord->velocity_y * t;
        double dx = x - cx;
        double dy = y - cy;
        double r = sqrt(dx * dx + dy * dy);
        double theta = atan2(dy, dx);
        return coord->spiral_pitch * r + coord->spiral_arms * theta + coord->spiral_phase +
               coord->spiral_angular_velocity * t;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        return x;
    }
}

static void lissajous_eval(const SimStimulusLissajousConfig *cfg, double ux, double uy, double t,
                           double *out_re, double *out_im) {
    double theta_x = cfg->wavenumber_x * ux - cfg->omega_x * t + cfg->phase_x;
    double theta_y = cfg->wavenumber_y * uy - cfg->omega_y * t + cfg->phase_y;
    double delta = sin(theta_x) - cfg->coupling * sin(theta_y) - cfg->bias;
    double band = exp(-0.5 * (delta * delta) / (cfg->line_width * cfg->line_width));
    double carrier = 0.5 * (theta_x + theta_y);

    *out_re = band * cos(carrier);
    *out_im = band * sin(carrier);
}

static int run_lissajous_case(const SimStimulusLissajousConfig *cfg_input, const size_t *shape_in,
                              double dt, int steps, double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    size_t shape[2] = {kShape[0], kShape[1]};
    if (shape_in != NULL) {
        shape[0] = shape_in[0];
        shape[1] = shape_in[1];
    }
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_complex_data(&field), 0, shape[0] * shape[1] * sizeof(SimComplexDouble));

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusLissajousConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_lissajous_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stimulus_lissajous_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusLissajousConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_lissajous_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_stimulus_lissajous_config\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: operator lookup\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = shape[0] * shape[1];
    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (expected == NULL) {
        fprintf(stderr, "FAIL: out of memory\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double sin_r = sin(normalized.rotation);
    double cos_r = cos(normalized.rotation);

    for (int step = 0; step < steps; ++step) {
        double scale = normalized.scale_by_dt ? dt : 1.0;
        double t = (double)step * dt + normalized.time_offset;

        for (size_t i = 0U; i < count; ++i) {
            size_t row = i / shape[1];
            size_t col = i % shape[1];
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double sample_x = x;
            double sample_y = y;
            double ux = 0.0;
            double uy = 0.0;
            double re = 0.0;
            double im = 0.0;
            sim_stimulus_coord_sample_xy(&normalized.coord, x, y, t, &sample_x, &sample_y);

            if (normalized.coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
                ux = sample_x;
                uy = sample_y;
            } else {
                double u = coord_u(&normalized.coord, x, y, t);
                ux = u;
                uy = u;
            }

            lissajous_eval(&normalized, ux, uy, t, &re, &im);

            re *= normalized.amplitude;
            im *= normalized.amplitude;
            expected[i].re += scale * (re * cos_r - im * sin_r);
            expected[i].im += scale * (re * sin_r + im * cos_r);
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
    SimStimulusLissajousConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.43;
    cfg.wavenumber_x = 3.0;
    cfg.wavenumber_y = 2.0;
    cfg.omega_x = 0.18;
    cfg.omega_y = -0.07;
    cfg.phase_x = 0.25;
    cfg.phase_y = 1.1;
    cfg.coupling = 0.85;
    cfg.bias = 0.08;
    cfg.line_width = 0.18;
    cfg.time_offset = 0.04;
    cfg.rotation = 0.55;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.origin_x = -1.1;
    cfg.coord.origin_y = -0.9;
    cfg.coord.spacing_x = 0.42;
    cfg.coord.spacing_y = 0.35;

    if (!run_lissajous_case(&cfg, kShape, 0.05, 3, kTol)) {
        return 1;
    }

    cfg.scale_by_dt = false;
    cfg.rotation = -0.2;
    cfg.time_offset = 0.07;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.origin_x = -0.8;
    cfg.coord.origin_y = -0.7;
    cfg.coord.spacing_x = 0.31;
    cfg.coord.spacing_y = 0.27;
    cfg.coord.center_x = 0.1;
    cfg.coord.center_y = -0.15;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.coord.spiral_arms = 1.7;
    cfg.coord.spiral_pitch = 0.9;
    cfg.coord.spiral_phase = 0.2;
    cfg.coord.spiral_angular_velocity = 0.14;

    if (!run_lissajous_case(&cfg, kShape, 0.03, 2, kTol)) {
        return 1;
    }

    cfg.scale_by_dt = false;
    cfg.rotation = 0.31;
    cfg.time_offset = 0.06;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.39;
    cfg.coord.origin_x = -1.42;
    cfg.coord.origin_y = -0.37;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (!run_lissajous_case(&cfg, kShapeWide, 0.02, 2, 1.0e-8)) {
        return 1;
    }

    return 0;
}
