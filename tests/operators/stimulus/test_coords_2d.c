/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static int approx(double a, double b, double eps) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= eps * scale;
}

static int test_patch_row_helper(void) {
    const double eps = 1.0e-12;
    SimField field = {0};
    SimField field_1d = {0};
    size_t shape[2] = {4U, 6U};
    size_t shape_1d[1] = {8U};
    SimFieldPatch patch = {0};
    SimFieldPatch patch_1d = {0};
    SimStimulusCoordConfig coord = {0};
    SimStimulusCoordRow row = {0};
    double t = 0.75;
    int ok = 1;

    coord.mode = SIM_STIMULUS_COORD_ANGLE;
    coord.origin_x = 0.3;
    coord.origin_y = -1.1;
    coord.spacing_x = 0.5;
    coord.spacing_y = 1.25;
    coord.velocity_x = 0.2;
    coord.velocity_y = -0.35;
    coord.angle = M_PI / 7.0;
    coord.center_x = 0.4;
    coord.center_y = -0.2;
    coord.spiral_angular_velocity = 0.6;
    sim_stimulus_coord_normalize(&coord);

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&field_1d, 1U, shape_1d, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_patch_from_xywh(6U, 4U, 2U, 1U, 3U, 2U, &patch) != SIM_RESULT_OK ||
        sim_field_patch_from_xywh(8U, 1U, 2U, 0U, 4U, 1U, &patch_1d) != SIM_RESULT_OK) {
        fprintf(stderr, "[coord_patch_row] setup failed\n");
        sim_field_destroy(&field);
        sim_field_destroy(&field_1d);
        return 0;
    }

    for (size_t row_offset = 0U; row_offset < patch.height; ++row_offset) {
        if (sim_stimulus_coord_patch_row(&coord, &patch, row_offset, t, &row) != SIM_RESULT_OK) {
            fprintf(stderr, "[coord_patch_row] row helper failed (2d)\n");
            ok = 0;
            break;
        }

        for (size_t col = 0U; col < row.width; ++col) {
            size_t idx = 0U;
            double x = row.x + (double)col * row.x_step;
            double y = row.y + (double)col * row.y_step;
            double sample_x = row.sample_x + (double)col * row.sample_x_step;
            double sample_y = row.sample_y + (double)col * row.sample_y_step;
            double expected_x = 0.0;
            double expected_y = 0.0;
            double expected_sample_x = 0.0;
            double expected_sample_y = 0.0;

            if (sim_field_xy_to_index(&field, patch.x0 + col, patch.y0 + row_offset, &idx) !=
                    SIM_RESULT_OK ||
                sim_stimulus_coord_xy(&coord, &field, idx, &expected_x, &expected_y) !=
                    SIM_RESULT_OK) {
                fprintf(stderr, "[coord_patch_row] expected coord resolution failed (2d)\n");
                ok = 0;
                break;
            }

            sim_stimulus_coord_sample_xy(&coord, expected_x, expected_y, t, &expected_sample_x,
                                         &expected_sample_y);
            if (!approx(x, expected_x, eps) || !approx(y, expected_y, eps) ||
                !approx(sample_x, expected_sample_x, eps) ||
                !approx(sample_y, expected_sample_y, eps)) {
                fprintf(stderr, "[coord_patch_row] mismatch in 2d row helper\n");
                ok = 0;
                break;
            }
        }

        if (!ok) {
            break;
        }
    }

    if (ok && sim_stimulus_coord_patch_row(&coord, &patch_1d, 0U, t, &row) != SIM_RESULT_OK) {
        fprintf(stderr, "[coord_patch_row] row helper failed (1d)\n");
        ok = 0;
    }

    for (size_t col = 0U; ok && col < patch_1d.width; ++col) {
        size_t idx = 0U;
        double x = row.x + (double)col * row.x_step;
        double y = row.y + (double)col * row.y_step;
        double expected_x = 0.0;
        double expected_y = 0.0;

        if (sim_field_xy_to_index(&field_1d, patch_1d.x0 + col, 0U, &idx) != SIM_RESULT_OK ||
            sim_stimulus_coord_xy(&coord, &field_1d, idx, &expected_x, &expected_y) !=
                SIM_RESULT_OK) {
            fprintf(stderr, "[coord_patch_row] expected coord resolution failed (1d)\n");
            ok = 0;
            break;
        }

        if (!approx(x, expected_x, eps) || !approx(y, expected_y, eps)) {
            fprintf(stderr, "[coord_patch_row] mismatch in 1d row helper\n");
            ok = 0;
            break;
        }
    }

    sim_field_destroy(&field);
    sim_field_destroy(&field_1d);
    return ok;
}

static double expected_sine_value(const SimStimulusSinusoidalConfig *cfg, double x, double y) {
    const double phase = cfg->phase;
    const double k = cfg->wavenumber;
    const bool separable = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool use_wavevector = cfg->use_wavevector;

    if (!use_wavevector) {
        if (fabs(cfg->kx) > 1.0e-12 || fabs(cfg->ky) > 1.0e-12) {
            use_wavevector = true;
        }
    }

    if (use_wavevector) {
        double dot = cfg->kx * x + cfg->ky * y;
        return cfg->amplitude * sin(dot + phase);
    }

    if (separable) {
        double fx = sin(k * x + phase);
        double fy = sin(k * y + phase);
        double base = (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) ? (fx + fy) : (fx * fy);
        return cfg->amplitude * base;
    }

    double u = 0.0;
    switch (cfg->coord.mode) {
    case SIM_STIMULUS_COORD_AXIS:
        u = (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) ? y : x;
        break;
    case SIM_STIMULUS_COORD_ANGLE:
        u = x * cos(cfg->coord.angle) + y * sin(cfg->coord.angle);
        break;
    case SIM_STIMULUS_COORD_RADIAL:
    case SIM_STIMULUS_COORD_POLAR: {
        double dx = x - cfg->coord.center_x;
        double dy = y - cfg->coord.center_y;
        u = sqrt(dx * dx + dy * dy);
        break;
    }
    case SIM_STIMULUS_COORD_AZIMUTH: {
        double dx = x - cfg->coord.center_x;
        double dy = y - cfg->coord.center_y;
        u = atan2(dy, dx);
        break;
    }
    case SIM_STIMULUS_COORD_ELLIPTIC: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(&cfg->coord, x, y, 0.0, &dx, &dy);
        sim_stimulus_coord_elliptic_polar(&cfg->coord, dx, dy, &u, NULL);
        break;
    }
    case SIM_STIMULUS_COORD_SPIRAL: {
        double dx = x - cfg->coord.center_x;
        double dy = y - cfg->coord.center_y;
        double r = sqrt(dx * dx + dy * dy);
        double th = atan2(dy, dx);
        u = cfg->coord.spiral_pitch * r + cfg->coord.spiral_arms * th + cfg->coord.spiral_phase;
        break;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        u = x;
        break;
    }

    return cfg->amplitude * sin(k * u + phase);
}

static int run_sine_2d_case(const char *label, const SimStimulusSinusoidalConfig *cfg_input,
                            size_t ny, size_t nx, double tol) {
    SimContext ctx;
    int rc = 1;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Failed to init context\n", label);
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t shape[2] = {ny, nx};
    SimField field = {0};
    if (sim_field_init(&field, 2, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Failed to init field\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    double *raw = (double *)sim_field_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Failed to add field\n", label);
        sim_field_destroy(&field);
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
        fprintf(stderr, "[%s] Failed to register stimulus\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusSinusoidalConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_sinusoidal_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Failed to fetch normalized config\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (!op) {
        fprintf(stderr, "[%s] Operator lookup failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] Stimulus evaluation failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    const double *data = out_field ? (const double *)sim_field_data(out_field) : NULL;
    if (data == NULL) {
        fprintf(stderr, "[%s] Output field data missing\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    double max_err = 0.0;
    for (size_t y = 0U; y < ny; ++y) {
        for (size_t x = 0U; x < nx; ++x) {
            size_t idx = y * nx + x;
            double x_coord = normalized.coord.origin_x + (double)x * normalized.coord.spacing_x;
            double y_coord = normalized.coord.origin_y + (double)y * normalized.coord.spacing_y;
            double expected = expected_sine_value(&normalized, x_coord, y_coord);
            double err = fabs(data[idx] - expected);
            if (err > max_err) {
                max_err = err;
            }
            if (err > tol) {
                fprintf(stderr,
                        "[%s] Mismatch at (x=%zu,y=%zu) idx=%zu got=%.12g exp=%.12g err=%.3g "
                        "tol=%.3g\n",
                        label, x, y, idx, data[idx], expected, err, tol);
                rc = 0;
                break;
            }
        }
        if (!rc) {
            break;
        }
    }

    sim_context_destroy(&ctx);

    if (!rc) {
        fprintf(stderr, "[%s] FAILED (max_err=%.3g tol=%.3g)\n", label, max_err, tol);
    } else {
        fprintf(stdout, "[%s] ok (max_err=%.3g tol=%.3g)\n", label, max_err, tol);
    }

    return rc;
}

int main(void) {
    const size_t ny = 3U;
    const size_t nx = 4U;
    const double tol = 1.0e-8;

    SimStimulusSinusoidalConfig base;
    memset(&base, 0, sizeof(base));
    base.amplitude = 0.75;
    base.wavenumber = 1.1;
    base.phase = 0.2;
    base.omega = 0.0;
    base.scale_by_dt = false;
    base.coord.origin_x = 0.3;
    base.coord.origin_y = -1.1;
    base.coord.spacing_x = 0.5;
    base.coord.spacing_y = 1.25;
    base.coord.axis = SIM_STIMULUS_AXIS_X;

    int ok = 1;

    ok &= test_patch_row_helper();

    SimStimulusSinusoidalConfig cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    ok &= run_sine_2d_case("coord_axis_x", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_Y;
    ok &= run_sine_2d_case("coord_axis_y", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = M_PI / 4.0;
    ok &= run_sine_2d_case("coord_angle", &cfg, ny, nx, tol);
    ok &= run_sine_2d_case("coord_angle_wide", &cfg, 4U, 96U, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_RADIAL;
    cfg.coord.center_x = 0.6;
    cfg.coord.center_y = -0.4;
    ok &= run_sine_2d_case("coord_radial", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_POLAR;
    cfg.coord.center_x = 0.2;
    cfg.coord.center_y = -0.3;
    ok &= run_sine_2d_case("coord_polar", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_AZIMUTH;
    cfg.coord.center_x = -0.1;
    cfg.coord.center_y = 0.2;
    ok &= run_sine_2d_case("coord_azimuth", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_ELLIPTIC;
    cfg.coord.angle = 0.35;
    cfg.coord.center_x = -0.1;
    cfg.coord.center_y = 0.15;
    cfg.coord.ellipse_u = 0.8;
    cfg.coord.ellipse_v = 0.45;
    ok &= run_sine_2d_case("coord_elliptic", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.center_x = 0.2;
    cfg.coord.center_y = -0.3;
    cfg.coord.spiral_arms = 2.0;
    cfg.coord.spiral_pitch = 1.3;
    cfg.coord.spiral_phase = 0.25;
    cfg.coord.spiral_angular_velocity = 0.0;
    ok &= run_sine_2d_case("coord_spiral", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;
    cfg.use_wavevector = true;
    cfg.kx = 0.5;
    cfg.ky = 0.25;
    ok &= run_sine_2d_case("coord_separable_multiply", &cfg, ny, nx, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.combine = SIM_STIMULUS_SEPARABLE_ADD;
    ok &= run_sine_2d_case("coord_separable_add", &cfg, ny, nx, tol);
    ok &= run_sine_2d_case("coord_separable_add_wide", &cfg, 4U, 96U, tol);

    cfg = base;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.use_wavevector = true;
    cfg.kx = 0.7;
    cfg.ky = -1.1;
    cfg.wavenumber = 0.0;
    ok &= run_sine_2d_case("coord_wavevector", &cfg, ny, nx, tol);
    ok &= run_sine_2d_case("coord_wavevector_wide", &cfg, 4U, 96U, tol);

    return ok ? 0 : 1;
}
