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
static const double kTol = 5.0e-8;
static const double kSqrt2 = 1.4142135623730950488;

static double hermite_polynomial(unsigned int mode, double x) {
    if (mode == 0U) {
        return 1.0;
    }
    if (mode == 1U) {
        return 2.0 * x;
    }

    double hm2 = 1.0;
    double hm1 = 2.0 * x;
    for (unsigned int n = 2U; n <= mode; ++n) {
        double h = 2.0 * x * hm1 - 2.0 * (double)(n - 1U) * hm2;
        hm2 = hm1;
        hm1 = h;
    }
    return hm1;
}

static void hermite_gaussian_beam_map_coord(const SimStimulusHermiteGaussianBeamConfig *cfg,
                                            double x, double y, double t, double *out_u,
                                            double *out_v) {
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

static void hermite_gaussian_beam_eval(const SimStimulusHermiteGaussianBeamConfig *cfg, double u,
                                       double v, double t, double *out_re, double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;

    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;

    double xi = kSqrt2 * ur / cfg->waist_u;
    double eta = kSqrt2 * vr / cfg->waist_v;
    double envelope = hermite_polynomial(cfg->mode_u, xi) * hermite_polynomial(cfg->mode_v, eta) *
                      exp(-((ur * ur) / (cfg->waist_u * cfg->waist_u) +
                            (vr * vr) / (cfg->waist_v * cfg->waist_v)));
    double phase = cfg->carrier_u * ur + cfg->carrier_v * vr - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(phase);
    *out_im = envelope * sin(phase);
}

static size_t element_count(size_t rank, const size_t *shape) {
    size_t count = 1U;
    for (size_t i = 0U; i < rank; ++i) {
        count *= shape[i];
    }
    return count;
}

static int run_hermite_gaussian_beam_case(const char *label,
                                          const SimStimulusHermiteGaussianBeamConfig *cfg_input,
                                          size_t rank, const size_t *shape, double dt, int steps,
                                          double tol) {
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
            fprintf(stderr, "FAIL[%s]: sim_field_complex_data\n", label);
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

    SimStimulusHermiteGaussianBeamConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_hermite_gaussian_beam_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_add_stimulus_hermite_gaussian_beam_operator\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusHermiteGaussianBeamConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_hermite_gaussian_beam_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_stimulus_hermite_gaussian_beam_config\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL[%s]: operator lookup\n", label);
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

        for (size_t i = 0U; i < count; ++i) {
            double x = 0.0;
            double y = 0.0;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            if (rank == 1U) {
                x = normalized.coord.origin_x + (double)i * normalized.coord.spacing_x;
                y = 0.0;
            } else {
                size_t row = i / shape[1];
                size_t col = i % shape[1];
                x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            }

            hermite_gaussian_beam_map_coord(&normalized, x, y, t, &u, &v);
            hermite_gaussian_beam_eval(&normalized, u, v, t, &re, &im);

            re *= normalized.amplitude;
            im *= normalized.amplitude;
            expected[i].re += scale * (re * cos_r - im * sin_r);
            expected[i].im += scale * (re * sin_r + im * cos_r);
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
        double scale_re = fmax(1.0, fmax(fabs(values[i].re), fabs(expected[i].re)));
        double scale_im = fmax(1.0, fmax(fabs(values[i].im), fabs(expected[i].im)));
        if (err_re > tol * scale_re || err_im > tol * scale_im) {
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
    {
        SimStimulusHermiteGaussianBeamConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.amplitude = 0.34;
        cfg.mode_u = 2U;
        cfg.mode_v = 1U;
        cfg.waist_u = 0.82;
        cfg.waist_v = 1.13;
        cfg.center_u = -0.14;
        cfg.center_v = 0.09;
        cfg.velocity_u = 0.05;
        cfg.velocity_v = -0.03;
        cfg.orientation = 0.28;
        cfg.orientation_rate = 0.12;
        cfg.carrier_u = 0.71;
        cfg.carrier_v = -0.22;
        cfg.omega = 0.19;
        cfg.phase = -0.17;
        cfg.time_offset = 0.04;
        cfg.rotation = 0.16;
        cfg.scale_by_dt = true;
        cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
        cfg.coord.origin_x = -0.94;
        cfg.coord.origin_y = -0.88;
        cfg.coord.spacing_x = 0.36;
        cfg.coord.spacing_y = 0.33;
        cfg.coord.velocity_x = 0.07;
        cfg.coord.velocity_y = -0.04;

        if (!run_hermite_gaussian_beam_case("separable_small", &cfg, 2U, kShape, 0.05, 3, kTol)) {
            return 1;
        }
    }

    {
        SimStimulusHermiteGaussianBeamConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.amplitude = 0.49;
        cfg.mode_u = 4U;
        cfg.mode_v = 3U;
        cfg.waist_u = 0.78;
        cfg.waist_v = 1.06;
        cfg.center_u = 0.16;
        cfg.center_v = -0.13;
        cfg.velocity_u = -0.04;
        cfg.velocity_v = 0.05;
        cfg.orientation = -0.29;
        cfg.orientation_rate = 0.11;
        cfg.carrier_u = -0.52;
        cfg.carrier_v = 0.31;
        cfg.omega = -0.24;
        cfg.phase = 0.27;
        cfg.time_offset = 0.05;
        cfg.rotation = -0.21;
        cfg.scale_by_dt = false;
        cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
        cfg.coord.angle = 0.37;
        cfg.coord.origin_x = -1.48;
        cfg.coord.origin_y = -0.46;
        cfg.coord.spacing_x = 0.05;
        cfg.coord.spacing_y = 0.08;
        cfg.coord.velocity_x = 0.03;
        cfg.coord.velocity_y = -0.02;

        if (!run_hermite_gaussian_beam_case("angle_wide", &cfg, 2U, kShapeWide, 0.03, 3, 1.0e-7)) {
            return 1;
        }
    }

    {
        SimStimulusHermiteGaussianBeamConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.scale_by_dt = false;
        cfg.mode_u = 5U;
        cfg.mode_v = 4U;
        cfg.waist_u = 0.0;
        cfg.waist_v = NAN;
        cfg.center_u = 0.18;
        cfg.center_v = -0.11;
        cfg.velocity_u = -0.06;
        cfg.velocity_v = 0.05;
        cfg.orientation = -0.31;
        cfg.orientation_rate = 0.08;
        cfg.carrier_u = -0.44;
        cfg.carrier_v = 0.36;
        cfg.omega = -0.21;
        cfg.phase = 0.29;
        cfg.time_offset = 0.03;
        cfg.rotation = -0.24;
        cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
        cfg.coord.origin_x = -0.7;
        cfg.coord.origin_y = -0.65;
        cfg.coord.spacing_x = 0.28;
        cfg.coord.spacing_y = 0.24;
        cfg.coord.center_x = 0.06;
        cfg.coord.center_y = -0.04;
        cfg.coord.velocity_x = 0.02;
        cfg.coord.velocity_y = -0.01;
        cfg.coord.spiral_arms = 1.4;
        cfg.coord.spiral_pitch = 0.58;
        cfg.coord.spiral_phase = 0.13;
        cfg.coord.spiral_angular_velocity = -0.07;

        if (!run_hermite_gaussian_beam_case("spiral_small", &cfg, 2U, kShape, 0.04, 2, kTol)) {
            return 1;
        }
    }

    return 0;
}
