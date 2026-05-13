/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kShapeSmall2D[] = {5U, 6U};
static const size_t kShapeWide2D[] = {4U, 96U};
static const size_t kShapeWide1D[] = {96U};
static const double kTol = 5.0e-9;

static void lb_map_coord(const SimStimulusLaplaceBeltramiConfig *cfg, double x, double y, double t,
                         double *out_u, double *out_v) {
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

static double lb_dirichlet_mode(int mode, double coord, double extent) {
    int n = mode;
    if (n < 0) {
        n = -n;
    }
    if (n == 0) {
        n = 1;
    }
    return sin(M_PI * (double)n * (coord / extent + 0.5));
}

static double lb_periodic_phase(int mode, double coord, double extent) {
    return 2.0 * M_PI * (double)mode * coord / extent;
}

static void lb_eval(const SimStimulusLaplaceBeltramiConfig *cfg, double u, double v, double t,
                    bool include_v, double *out_re, double *out_im) {
    double spatial_re = 0.0;
    double spatial_im = 0.0;
    double temporal = -cfg->omega * t + cfg->phase;

    switch (cfg->manifold) {
    case SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS: {
        double arg = lb_periodic_phase(cfg->mode_u, u, cfg->extent_u);
        if (include_v) {
            arg += lb_periodic_phase(cfg->mode_v, v, cfg->extent_v);
        }
        spatial_re = cos(arg);
        spatial_im = sin(arg);
        break;
    }
    case SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER: {
        double arg = lb_periodic_phase(cfg->mode_u, u, cfg->extent_u);
        double envelope = include_v ? lb_dirichlet_mode(cfg->mode_v, v, cfg->extent_v) : 1.0;
        spatial_re = envelope * cos(arg);
        spatial_im = envelope * sin(arg);
        break;
    }
    case SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE:
    default: {
        double value = lb_dirichlet_mode(cfg->mode_u, u, cfg->extent_u);
        if (include_v) {
            value *= lb_dirichlet_mode(cfg->mode_v, v, cfg->extent_v);
        }
        spatial_re = value;
        spatial_im = 0.0;
        break;
    }
    }

    *out_re = spatial_re * cos(temporal) - spatial_im * sin(temporal);
    *out_im = spatial_re * sin(temporal) + spatial_im * cos(temporal);
}

static size_t element_count(size_t rank, const size_t *shape) {
    size_t count = 1U;
    for (size_t i = 0U; i < rank; ++i) {
        count *= shape[i];
    }
    return count;
}

static int run_lb_case(const char *label, const SimStimulusLaplaceBeltramiConfig *cfg_input,
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

    SimStimulusLaplaceBeltramiConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_laplace_beltrami_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_add_stimulus_laplace_beltrami_operator\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusLaplaceBeltramiConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_laplace_beltrami_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL[%s]: sim_stimulus_laplace_beltrami_config\n", label);
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
            size_t row = 0U;
            size_t col = i;
            double x = 0.0;
            double y = normalized.coord.origin_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;
            double out_re = 0.0;
            double out_im = 0.0;

            if (rank == 1U) {
                x = normalized.coord.origin_x + (double)i * normalized.coord.spacing_x;
            } else {
                row = i / shape[1];
                col = i % shape[1];
                x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            }

            lb_map_coord(&normalized, x, y, t, &u, &v);
            lb_eval(&normalized, u, v, t, rank > 1U, &re, &im);

            re *= normalized.amplitude;
            im *= normalized.amplitude;
            out_re = re * cos_r - im * sin_r;
            out_im = re * sin_r + im * cos_r;
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
    SimStimulusLaplaceBeltramiConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.31;
    cfg.manifold = SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS;
    cfg.mode_u = 2;
    cfg.mode_v = -3;
    cfg.extent_u = 2.4;
    cfg.extent_v = 1.6;
    cfg.omega = 0.27;
    cfg.phase = -0.18;
    cfg.time_offset = 0.05;
    cfg.rotation = 0.22;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.31;
    cfg.coord.origin_x = -1.2;
    cfg.coord.origin_y = -0.8;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.33;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (!run_lb_case("flat_torus_wide", &cfg, 2U, kShapeWide2D, 0.07, 4, kTol)) {
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.43;
    cfg.manifold = SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE;
    cfg.mode_u = -4;
    cfg.mode_v = -5;
    cfg.extent_u = 2.0;
    cfg.extent_v = 1.7;
    cfg.omega = 0.13;
    cfg.phase = 0.24;
    cfg.time_offset = -0.03;
    cfg.rotation = -0.16;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.origin_x = -1.0;
    cfg.coord.origin_y = -0.9;
    cfg.coord.spacing_x = 0.38;
    cfg.coord.spacing_y = 0.34;
    cfg.coord.center_x = 0.15;
    cfg.coord.center_y = -0.1;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.coord.spiral_arms = 1.2;
    cfg.coord.spiral_pitch = 0.8;
    cfg.coord.spiral_phase = -0.1;
    cfg.coord.spiral_angular_velocity = 0.25;

    if (!run_lb_case("rectangle_spiral_fallback", &cfg, 2U, kShapeSmall2D, 0.05, 3, kTol)) {
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.37;
    cfg.manifold = SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER;
    cfg.mode_u = -3;
    cfg.mode_v = 5;
    cfg.extent_u = 2.8;
    cfg.extent_v = 1.9;
    cfg.omega = -0.16;
    cfg.phase = 0.09;
    cfg.time_offset = 0.04;
    cfg.rotation = 0.18;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -1.7;
    cfg.coord.origin_y = 0.21;
    cfg.coord.spacing_x = 0.04;
    cfg.coord.velocity_x = 0.02;
    cfg.coord.velocity_y = -0.01;

    return run_lb_case("cylinder_1d_axis", &cfg, 1U, kShapeWide1D, 0.05, 4, kTol) ? 0 : 1;
}
