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

static void log_grid_map_coord(const SimStimulusLogSpectralGridConfig *cfg, double x, double y,
                               double t, double *out_u, double *out_v) {
    if (cfg->use_wavevector) {
        double norm = hypot(cfg->kx, cfg->ky);
        double ux = 1.0;
        double uy = 0.0;
        if (norm > 1.0e-12) {
            ux = cfg->kx / norm;
            uy = cfg->ky / norm;
        }
        *out_u = ux * x + uy * y;
        *out_v = -uy * x + ux * y;
        return;
    }

    switch (cfg->coord.mode) {
    case SIM_STIMULUS_COORD_AXIS:
        if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
            *out_u = y;
            *out_v = x;
        } else {
            *out_u = x;
            *out_v = y;
        }
        return;
    case SIM_STIMULUS_COORD_ANGLE: {
        double s = sin(cfg->coord.angle);
        double c = cos(cfg->coord.angle);
        *out_u = x * c + y * s;
        *out_v = -x * s + y * c;
        return;
    }
    case SIM_STIMULUS_COORD_RADIAL:
    case SIM_STIMULUS_COORD_POLAR: {
        double cx = cfg->coord.center_x + cfg->coord.velocity_x * t;
        double cy = cfg->coord.center_y + cfg->coord.velocity_y * t;
        double dx = x - cx;
        double dy = y - cy;
        *out_u = hypot(dx, dy);
        *out_v = atan2(dy, dx);
        return;
    }
    case SIM_STIMULUS_COORD_SPIRAL: {
        double cx = cfg->coord.center_x + cfg->coord.velocity_x * t;
        double cy = cfg->coord.center_y + cfg->coord.velocity_y * t;
        double dx = x - cx;
        double dy = y - cy;
        double r = hypot(dx, dy);
        double th = atan2(dy, dx);
        *out_u = cfg->coord.spiral_pitch * r + cfg->coord.spiral_arms * th +
                 cfg->coord.spiral_phase + cfg->coord.spiral_angular_velocity * t;
        *out_v = th;
        return;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        *out_u = x;
        *out_v = y;
        return;
    }
}

static void log_grid_eval_uv(const SimStimulusLogSpectralGridConfig *cfg, double u, double v,
                             double t, double *out_re, double *out_im) {
    double log_k_min = log(fmax(cfg->k_min, 1.0e-12));
    double log_k_max = log(fmax(cfg->k_max, 1.0e-12));

    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);
    double ur = u * c + v * s;
    double vr = -u * s + v * c;

    unsigned int radial_bins = cfg->radial_bins;
    unsigned int angular_bins = cfg->angular_bins;
    if (radial_bins == 0U) {
        radial_bins = 6U;
    }
    if (angular_bins == 0U) {
        angular_bins = 12U;
    }
    unsigned int mode_count = radial_bins * angular_bins;

    double re_sum = 0.0;
    double im_sum = 0.0;
    double sum_sq = 0.0;

    for (unsigned int r = 0U; r < radial_bins; ++r) {
        double k = 0.0;
        if (radial_bins <= 1U || fabs(log_k_max - log_k_min) <= 1.0e-12) {
            k = exp(0.5 * (log_k_min + log_k_max));
        } else {
            double frac = (double)r / (double)(radial_bins - 1U);
            k = exp(log_k_min + frac * (log_k_max - log_k_min));
        }

        for (unsigned int a = 0U; a < angular_bins; ++a) {
            double angle = 2.0 * M_PI * ((double)a / (double)angular_bins);
            double kx = k * cos(angle);
            double ky = k * sin(angle);
            double w = 1.0;
            if (cfg->spectral_slope != 0.0 && k > 1.0e-12) {
                w = pow(k, -0.5 * cfg->spectral_slope);
            }
            sum_sq += w * w;

            double arg = kx * ur + ky * vr - cfg->omega * t + cfg->phase;
            re_sum += w * cos(arg);
            im_sum += w * sin(arg);
        }
    }

    double mode_norm =
        (mode_count > 0U && sum_sq > 1.0e-12) ? sqrt(sum_sq / (double)mode_count) : 1.0;
    double norm = (mode_count > 0U) ? (1.0 / sqrt((double)mode_count)) : 1.0;
    *out_re = re_sum * norm / mode_norm;
    *out_im = im_sum * norm / mode_norm;
}

static double log_grid_eval_real(const SimStimulusLogSpectralGridConfig *cfg, double x, double y,
                                 double t) {
    double re = 0.0;
    double im = 0.0;

    if (!cfg->use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double rx = 0.0;
        double ix = 0.0;
        double ry = 0.0;
        double iy = 0.0;
        log_grid_eval_uv(cfg, x, 0.0, t, &rx, &ix);
        log_grid_eval_uv(cfg, y, 0.0, t, &ry, &iy);
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            re = rx + ry;
            im = ix + iy;
        } else {
            re = rx * ry - ix * iy;
            im = rx * iy + ix * ry;
        }
    } else {
        double u = 0.0;
        double v = 0.0;
        log_grid_map_coord(cfg, x, y, t, &u, &v);
        log_grid_eval_uv(cfg, u, v, t, &re, &im);
    }

    (void)im;
    return cfg->amplitude * re;
}

static int run_log_grid_case(const char *label, size_t rank, const size_t *shape,
                             const SimStimulusLogSpectralGridConfig cfg_input, double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_context_init failed\n", label);
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    SimField field = {0};
    if (sim_field_init(&field, rank, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_field_init failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    double *raw = (double *)sim_field_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_context_add_field failed\n", label);
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusLogSpectralGridConfig cfg = cfg_input;
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_log_spectral_grid_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_add_stimulus_log_spectral_grid_operator failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusLogSpectralGridConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_log_spectral_grid_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_stimulus_log_spectral_grid_config failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "[%s] operator lookup failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] evaluate failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    if (out_field == NULL) {
        fprintf(stderr, "[%s] output field lookup failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    const double *data = (const double *)sim_field_data(out_field);
    if (data == NULL) {
        fprintf(stderr, "[%s] output field data missing\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = sim_field_bytes(out_field) / sizeof(double);
    if (count == 0U) {
        fprintf(stderr, "[%s] empty field\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    int ok = 1;
    double max_err = 0.0;
    double eval_t = sim_context_time(&ctx) + normalized.time_offset;
    for (size_t i = 0U; i < count; ++i) {
        size_t ix = 0U;
        size_t iy = 0U;
        if (sim_field_index_to_xy(out_field, i, &ix, &iy) != SIM_RESULT_OK) {
            fprintf(stderr, "[%s] index_to_xy failed at %zu\n", label, i);
            ok = 0;
            break;
        }

        double x = normalized.coord.origin_x + (double)ix * normalized.coord.spacing_x;
        double y = normalized.coord.origin_y + (double)iy * normalized.coord.spacing_y;
        double expected = log_grid_eval_real(&normalized, x, y, eval_t);
        double err = fabs(data[i] - expected);
        if (err > max_err) {
            max_err = err;
        }
        if (err > tol) {
            fprintf(stderr,
                    "[%s] mismatch i=%zu (x=%zu,y=%zu) got=%.12g exp=%.12g err=%.3g tol=%.3g\n",
                    label, i, ix, iy, data[i], expected, err, tol);
            ok = 0;
            break;
        }
    }

    sim_context_destroy(&ctx);

    if (!ok) {
        fprintf(stderr, "[%s] FAILED (max_err=%.3g tol=%.3g)\n", label, max_err, tol);
    } else {
        fprintf(stdout, "[%s] ok (max_err=%.3g tol=%.3g)\n", label, max_err, tol);
    }
    return ok;
}

int main(void) {
    int ok = 1;

    SimStimulusLogSpectralGridConfig cfg_axis;
    memset(&cfg_axis, 0, sizeof(cfg_axis));
    cfg_axis.amplitude = 0.8;
    cfg_axis.k_min = 0.25;
    cfg_axis.k_max = 2.3;
    cfg_axis.radial_bins = 5U;
    cfg_axis.angular_bins = 9U;
    cfg_axis.spectral_slope = 0.7;
    cfg_axis.orientation = 0.3;
    cfg_axis.orientation_rate = 0.12;
    cfg_axis.omega = 0.2;
    cfg_axis.phase = -0.1;
    cfg_axis.time_offset = 0.05;
    cfg_axis.random_phase = false;
    cfg_axis.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_axis.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg_axis.coord.origin_x = -0.6;
    cfg_axis.coord.spacing_x = 0.04;

    {
        const size_t shape_1d[1] = {64U};
        ok &= run_log_grid_case("log_grid_1d_axis", 1U, shape_1d, cfg_axis, 1.0e-9);
    }

    SimStimulusLogSpectralGridConfig cfg_wave;
    memset(&cfg_wave, 0, sizeof(cfg_wave));
    cfg_wave.amplitude = 1.0;
    cfg_wave.k_min = 0.2;
    cfg_wave.k_max = 3.1;
    cfg_wave.radial_bins = 6U;
    cfg_wave.angular_bins = 12U;
    cfg_wave.spectral_slope = 1.1;
    cfg_wave.orientation = -0.25;
    cfg_wave.orientation_rate = 0.06;
    cfg_wave.kx = 1.3;
    cfg_wave.ky = -0.8;
    cfg_wave.use_wavevector = true;
    cfg_wave.omega = -0.18;
    cfg_wave.phase = 0.22;
    cfg_wave.time_offset = -0.11;
    cfg_wave.random_phase = false;
    cfg_wave.coord.origin_x = -0.3;
    cfg_wave.coord.origin_y = 0.4;
    cfg_wave.coord.spacing_x = 0.2;
    cfg_wave.coord.spacing_y = 0.15;

    {
        const size_t shape_2d[2] = {9U, 7U};
        ok &= run_log_grid_case("log_grid_2d_wavevector", 2U, shape_2d, cfg_wave, 1.0e-9);
    }

    SimStimulusLogSpectralGridConfig cfg_sep;
    memset(&cfg_sep, 0, sizeof(cfg_sep));
    cfg_sep.amplitude = 0.55;
    cfg_sep.k_min = 0.3;
    cfg_sep.k_max = 2.0;
    cfg_sep.radial_bins = 4U;
    cfg_sep.angular_bins = 8U;
    cfg_sep.spectral_slope = 0.5;
    cfg_sep.phase = 0.12;
    cfg_sep.random_phase = false;
    cfg_sep.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg_sep.coord.combine = SIM_STIMULUS_SEPARABLE_ADD;
    cfg_sep.coord.origin_x = -0.2;
    cfg_sep.coord.origin_y = 0.1;
    cfg_sep.coord.spacing_x = 0.11;
    cfg_sep.coord.spacing_y = 0.14;

    {
        const size_t shape_sep[2] = {6U, 5U};
        ok &= run_log_grid_case("log_grid_2d_separable", 2U, shape_sep, cfg_sep, 1.0e-9);
    }

    SimStimulusLogSpectralGridConfig cfg_polar;
    memset(&cfg_polar, 0, sizeof(cfg_polar));
    cfg_polar.amplitude = 0.73;
    cfg_polar.k_min = 0.24;
    cfg_polar.k_max = 2.7;
    cfg_polar.radial_bins = 5U;
    cfg_polar.angular_bins = 10U;
    cfg_polar.spectral_slope = 0.8;
    cfg_polar.orientation = 0.16;
    cfg_polar.orientation_rate = -0.07;
    cfg_polar.omega = 0.14;
    cfg_polar.phase = -0.09;
    cfg_polar.time_offset = 0.03;
    cfg_polar.random_phase = false;
    cfg_polar.coord.mode = SIM_STIMULUS_COORD_POLAR;
    cfg_polar.coord.origin_x = -0.42;
    cfg_polar.coord.origin_y = -0.36;
    cfg_polar.coord.spacing_x = 0.18;
    cfg_polar.coord.spacing_y = 0.16;
    cfg_polar.coord.center_x = 0.08;
    cfg_polar.coord.center_y = -0.11;
    cfg_polar.coord.velocity_x = 0.02;
    cfg_polar.coord.velocity_y = -0.03;

    {
        const size_t shape_po[2] = {7U, 6U};
        ok &= run_log_grid_case("log_grid_2d_polar", 2U, shape_po, cfg_polar, 1.0e-9);
    }

    return ok ? 0 : 1;
}
