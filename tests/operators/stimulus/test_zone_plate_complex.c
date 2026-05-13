/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kShapeSmall[] = {5U, 6U};
static const size_t kShapeWide[] = {4U, 96U};
static const double kTol = 5.0e-8;

static void zone_plate_map_coord(const SimStimulusZonePlateConfig *cfg, double x, double y,
                                 double t, double *out_u, double *out_v) {
    const SimStimulusCoordConfig *coord = &cfg->coord;
    double sample_x = x;
    double sample_y = y;
    sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);

    switch (coord->mode) {
    case SIM_STIMULUS_COORD_AXIS:
        if (coord->axis == SIM_STIMULUS_AXIS_Y) {
            *out_u = sample_y;
            *out_v = sample_x;
        } else {
            *out_u = sample_x;
            *out_v = sample_y;
        }
        return;
    case SIM_STIMULUS_COORD_ANGLE: {
        double s = sin(coord->angle);
        double c = cos(coord->angle);
        *out_u = sample_x * c + sample_y * s;
        *out_v = -sample_x * s + sample_y * c;
        return;
    }
    case SIM_STIMULUS_COORD_RADIAL:
    case SIM_STIMULUS_COORD_POLAR:
    case SIM_STIMULUS_COORD_AZIMUTH:
    case SIM_STIMULUS_COORD_ELLIPTIC:
    case SIM_STIMULUS_COORD_SPIRAL: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);

        if (coord->mode == SIM_STIMULUS_COORD_SPIRAL) {
            double r = hypot(dx, dy);
            double th = atan2(dy, dx);
            *out_u = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                     coord->spiral_angular_velocity * t;
            *out_v = th;
        } else if (coord->mode == SIM_STIMULUS_COORD_POLAR) {
            *out_u = hypot(dx, dy);
            *out_v = atan2(dy, dx);
        } else if (coord->mode == SIM_STIMULUS_COORD_AZIMUTH) {
            *out_u = atan2(dy, dx);
            *out_v = hypot(dx, dy);
        } else if (coord->mode == SIM_STIMULUS_COORD_ELLIPTIC) {
            sim_stimulus_coord_elliptic_local(coord, dx, dy, out_u, out_v);
        } else {
            *out_u = dx;
            *out_v = dy;
        }
        return;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        *out_u = sample_x;
        *out_v = sample_y;
        return;
    }
}

static void zone_plate_eval(const SimStimulusZonePlateConfig *cfg, double u, double v, double t,
                            double *out_re, double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;

    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;
    double su = ur / cfg->scale_u;
    double sv = vr / cfg->scale_v;
    double au = ur / cfg->aperture_u;
    double av = vr / cfg->aperture_v;
    double rho2 = su * su + sv * sv;
    double aperture = exp(-0.5 * (au * au + av * av));
    double phase = cfg->radial_chirp * rho2 - cfg->omega * t + cfg->phase;

    *out_re = aperture * cos(phase);
    *out_im = aperture * sin(phase);
}

static size_t element_count(const size_t *shape) { return shape[0] * shape[1]; }

static int run_zone_plate_case(const char *label, const SimStimulusZonePlateConfig *cfg_input,
                               const size_t *shape, double dt, int steps, double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_context_init\n", label);
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_field_init\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimComplexDouble *initial = sim_field_complex_data(&field);
    if (initial == NULL) {
        fprintf(stderr, "[%s] FAIL: sim_field_complex_data\n", label);
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(initial, 0, sim_field_bytes(&field));

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_context_add_field\n", label);
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusZonePlateConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_zone_plate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_add_stimulus_zone_plate_operator\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusZonePlateConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_zone_plate_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] FAIL: sim_stimulus_zone_plate_config\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "[%s] FAIL: operator lookup\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = element_count(shape);
    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (expected == NULL) {
        fprintf(stderr, "[%s] FAIL: out of memory\n", label);
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
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            zone_plate_map_coord(&normalized, x, y, t, &u, &v);
            zone_plate_eval(&normalized, u, v, t, &re, &im);

            re *= normalized.amplitude;
            im *= normalized.amplitude;
            expected[i].re += scale * (re * cos_r - im * sin_r);
            expected[i].im += scale * (re * sin_r + im * cos_r);
        }

        if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
            fprintf(stderr, "[%s] FAIL: operator evaluate at step %d\n", label, step);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }
        sim_context_accept_step(&ctx, (float)dt);
    }

    {
        SimField *result_field = sim_context_field(&ctx, field_index);
        if (result_field == NULL) {
            fprintf(stderr, "[%s] FAIL: result field lookup\n", label);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }

        const SimComplexDouble *values = sim_field_complex_data_const(result_field);
        for (size_t i = 0U; i < count; ++i) {
            double err_re = fabs(values[i].re - expected[i].re);
            double err_im = fabs(values[i].im - expected[i].im);
            if (err_re > tol || err_im > tol) {
                fprintf(stderr,
                        "[%s] FAIL: mismatch at %zu got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                        label, i, values[i].re, values[i].im, expected[i].re, expected[i].im);
                free(expected);
                sim_context_destroy(&ctx);
                return 0;
            }
        }
    }

    free(expected);
    sim_context_destroy(&ctx);
    return 1;
}

int main(void) {
    SimStimulusZonePlateConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.55;
    cfg.radial_chirp = 22.0;
    cfg.scale_u = 0.82;
    cfg.scale_v = 1.18;
    cfg.aperture_u = 1.35;
    cfg.aperture_v = 0.92;
    cfg.center_u = 0.14;
    cfg.center_v = -0.1;
    cfg.velocity_u = 0.06;
    cfg.velocity_v = -0.03;
    cfg.orientation = 0.31;
    cfg.orientation_rate = 0.11;
    cfg.omega = 0.26;
    cfg.phase = -0.28;
    cfg.time_offset = 0.05;
    cfg.rotation = 0.22;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.origin_x = -1.25;
    cfg.coord.origin_y = -1.0;
    cfg.coord.spacing_x = 0.44;
    cfg.coord.spacing_y = 0.37;
    cfg.coord.velocity_x = 0.06;
    cfg.coord.velocity_y = -0.05;

    if (!run_zone_plate_case("separable_small", &cfg, kShapeSmall, 0.05, 3, kTol)) {
        return 1;
    }

    cfg.scale_by_dt = false;
    cfg.radial_chirp = -17.5;
    cfg.scale_u = 0.63;
    cfg.scale_v = 0.94;
    cfg.aperture_u = 1.1;
    cfg.aperture_v = 1.45;
    cfg.center_u = -0.2;
    cfg.center_v = 0.16;
    cfg.velocity_u = -0.04;
    cfg.velocity_v = 0.09;
    cfg.orientation = -0.45;
    cfg.orientation_rate = 0.07;
    cfg.omega = -0.18;
    cfg.phase = 0.33;
    cfg.time_offset = -0.03;
    cfg.rotation = -0.17;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.center_x = 0.1;
    cfg.coord.center_y = -0.12;
    cfg.coord.velocity_x = 0.02;
    cfg.coord.velocity_y = -0.01;
    cfg.coord.spiral_arms = 1.4;
    cfg.coord.spiral_pitch = 0.85;
    cfg.coord.spiral_phase = 0.21;
    cfg.coord.spiral_angular_velocity = 0.16;
    cfg.coord.origin_x = -0.9;
    cfg.coord.origin_y = -0.75;
    cfg.coord.spacing_x = 0.36;
    cfg.coord.spacing_y = 0.33;

    if (!run_zone_plate_case("spiral_small", &cfg, kShapeSmall, 0.08, 2, kTol)) {
        return 1;
    }

    cfg.scale_by_dt = false;
    cfg.amplitude = 0.47;
    cfg.radial_chirp = 14.0;
    cfg.scale_u = 0.79;
    cfg.scale_v = 1.11;
    cfg.aperture_u = 1.28;
    cfg.aperture_v = 0.96;
    cfg.center_u = -0.12;
    cfg.center_v = 0.18;
    cfg.velocity_u = 0.05;
    cfg.velocity_v = -0.04;
    cfg.orientation = 0.26;
    cfg.orientation_rate = -0.09;
    cfg.omega = 0.31;
    cfg.phase = 0.14;
    cfg.time_offset = 0.07;
    cfg.rotation = -0.23;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.41;
    cfg.coord.origin_x = -1.5;
    cfg.coord.origin_y = -0.7;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.04;
    cfg.coord.velocity_y = -0.03;

    if (!run_zone_plate_case("angle_wide", &cfg, kShapeWide, 0.03, 2, 1.0e-7)) {
        return 1;
    }

    return 0;
}
