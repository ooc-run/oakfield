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

static void log_polar_map_frame(const SimStimulusLogPolarConfig *cfg, double x, double y, double t,
                                double *out_u, double *out_v, double *out_spiral_phase) {
    double u = x;
    double v = y;
    double spiral_phase = 0.0;

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &u, &v);

    switch (cfg->coord.mode) {
    case SIM_STIMULUS_COORD_AXIS:
        if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
            double tmp = u;
            u = v;
            v = tmp;
        }
        break;
    case SIM_STIMULUS_COORD_ANGLE: {
        double s = sin(cfg->coord.angle);
        double c = cos(cfg->coord.angle);
        double sample_x = u;
        double sample_y = v;
        u = sample_x * c + sample_y * s;
        v = -sample_x * s + sample_y * c;
        break;
    }
    case SIM_STIMULUS_COORD_RADIAL:
    case SIM_STIMULUS_COORD_AZIMUTH:
    case SIM_STIMULUS_COORD_ELLIPTIC:
    case SIM_STIMULUS_COORD_SPIRAL: {
        sim_stimulus_coord_centered_xy(&cfg->coord, x, y, t, &u, &v);
        if (cfg->coord.mode == SIM_STIMULUS_COORD_AZIMUTH) {
            spiral_phase = atan2(v, u);
        } else if (cfg->coord.mode == SIM_STIMULUS_COORD_ELLIPTIC) {
            sim_stimulus_coord_elliptic_local(&cfg->coord, u, v, &u, &v);
        } else if (cfg->coord.mode == SIM_STIMULUS_COORD_SPIRAL) {
            double radius = hypot(u, v);
            double angle = atan2(v, u);
            spiral_phase = cfg->coord.spiral_pitch * radius + cfg->coord.spiral_arms * angle +
                           cfg->coord.spiral_phase + cfg->coord.spiral_angular_velocity * t;
        }
        break;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        break;
    }

    *out_u = u;
    *out_v = v;
    *out_spiral_phase = spiral_phase;
}

static void log_polar_eval(const SimStimulusLogPolarConfig *cfg, double u, double v, double t,
                           double spiral_phase, double *out_re, double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);
    double ur = u * c + v * s;
    double vr = -u * s + v * c;
    double arg = cfg->radial_frequency * log(hypot(ur, vr) + cfg->radius_floor) +
                 cfg->angular_frequency * atan2(vr, ur) + spiral_phase - cfg->omega * t +
                 cfg->phase;

    *out_re = cos(arg);
    *out_im = sin(arg);
}

static int run_log_polar_case(const SimStimulusLogPolarConfig *cfg_input, const size_t *shape_in,
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

    SimStimulusLogPolarConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_log_polar_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stimulus_log_polar_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusLogPolarConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_log_polar_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_stimulus_log_polar_config\n");
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

    double scale = normalized.scale_by_dt ? dt : 1.0;
    double sin_r = sin(normalized.rotation);
    double cos_r = cos(normalized.rotation);

    for (int step = 0; step < steps; ++step) {
        double t = (double)step * dt + normalized.time_offset;

        for (size_t i = 0U; i < count; ++i) {
            size_t row = i / shape[1];
            size_t col = i % shape[1];
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double spiral_phase = 0.0;
            double re = 0.0;
            double im = 0.0;
            double out_re = 0.0;
            double out_im = 0.0;

            log_polar_map_frame(&normalized, x, y, t, &u, &v, &spiral_phase);
            log_polar_eval(&normalized, u, v, t, spiral_phase, &re, &im);

            re *= normalized.amplitude;
            im *= normalized.amplitude;
            out_re = re * cos_r - im * sin_r;
            out_im = re * sin_r + im * cos_r;

            expected[i].re += scale * out_re;
            expected[i].im += scale * out_im;
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
    int ok = 1;

    SimStimulusLogPolarConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.42;
    cfg.radial_frequency = 2.7;
    cfg.angular_frequency = -3.0;
    cfg.orientation = 0.33;
    cfg.orientation_rate = 0.15;
    cfg.omega = 0.4;
    cfg.phase = -0.2;
    cfg.radius_floor = 0.18;
    cfg.time_offset = 0.05;
    cfg.rotation = 0.6;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.origin_x = -1.0;
    cfg.coord.origin_y = -0.6;
    cfg.coord.spacing_x = 0.4;
    cfg.coord.spacing_y = 0.35;
    cfg.coord.center_x = 0.3;
    cfg.coord.center_y = -0.1;
    cfg.coord.velocity_x = 0.05;
    cfg.coord.velocity_y = -0.04;
    cfg.coord.spiral_arms = 1.5;
    cfg.coord.spiral_pitch = 0.9;
    cfg.coord.spiral_phase = 0.2;
    cfg.coord.spiral_angular_velocity = 0.3;

    ok &= run_log_polar_case(&cfg, kShape, 0.125, 4, kTol);

    cfg.coord.mode = SIM_STIMULUS_COORD_AZIMUTH;
    cfg.time_offset = 0.0;
    ok &= run_log_polar_case(&cfg, kShape, 0.125, 3, kTol);

    cfg.coord.mode = SIM_STIMULUS_COORD_ELLIPTIC;
    cfg.coord.angle = 0.47;
    cfg.coord.ellipse_u = 0.92;
    cfg.coord.ellipse_v = 0.58;
    ok &= run_log_polar_case(&cfg, kShape, 0.125, 3, kTol);

    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.37;
    cfg.time_offset = 0.04;
    cfg.rotation = -0.28;
    cfg.scale_by_dt = false;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.41;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    ok &= run_log_polar_case(&cfg, kShapeWide, 0.02, 2, 1.0e-8);

    return ok ? 0 : 1;
}
