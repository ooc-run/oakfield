/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static double moire_eval_real(const SimStimulusMoireConfig *cfg, double x, double y, double t) {
    double re = 0.0;
    double im = 0.0;
    double sample_x = x;
    double sample_y = y;

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

    if (cfg->use_wavevectors) {
        double theta_a =
            cfg->k1x * sample_x + cfg->k1y * sample_y - cfg->omega_a * t + cfg->phase_a;
        double theta_b =
            cfg->k2x * sample_x + cfg->k2y * sample_y - cfg->omega_b * t + cfg->phase_b;
        re = 0.5 * (cos(theta_a) + cos(theta_b));
        im = 0.5 * (sin(theta_a) + sin(theta_b));
    } else if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double theta_ax = cfg->wavenumber_a * sample_x - cfg->omega_a * t + cfg->phase_a;
        double theta_bx = cfg->wavenumber_b * sample_x - cfg->omega_b * t + cfg->phase_b;
        double theta_ay = cfg->wavenumber_a * sample_y - cfg->omega_a * t + cfg->phase_a;
        double theta_by = cfg->wavenumber_b * sample_y - cfg->omega_b * t + cfg->phase_b;
        double rx = 0.5 * (cos(theta_ax) + cos(theta_bx));
        double ix = 0.5 * (sin(theta_ax) + sin(theta_bx));
        double ry = 0.5 * (cos(theta_ay) + cos(theta_by));
        double iy = 0.5 * (sin(theta_ay) + sin(theta_by));
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            re = rx + ry;
            im = ix + iy;
        } else {
            re = rx * ry - ix * iy;
            im = rx * iy + ix * ry;
        }
    } else {
        double u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
        {
            double theta_a = cfg->wavenumber_a * u - cfg->omega_a * t + cfg->phase_a;
            double theta_b = cfg->wavenumber_b * u - cfg->omega_b * t + cfg->phase_b;
            re = 0.5 * (cos(theta_a) + cos(theta_b));
            im = 0.5 * (sin(theta_a) + sin(theta_b));
        }
    }

    (void)im;
    return cfg->amplitude * re;
}

static int run_moire_case(const char *label, size_t rank, const size_t *shape,
                          const SimStimulusMoireConfig cfg_input, double tol) {
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

    SimStimulusMoireConfig cfg = cfg_input;
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_moire_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_add_stimulus_moire_operator failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusMoireConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_moire_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_stimulus_moire_config failed\n", label);
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
        double expected = moire_eval_real(&normalized, x, y, eval_t);
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

    SimStimulusMoireConfig cfg_1d;
    memset(&cfg_1d, 0, sizeof(cfg_1d));
    cfg_1d.amplitude = 0.8;
    cfg_1d.wavenumber_a = 3.5;
    cfg_1d.wavenumber_b = 3.8;
    cfg_1d.phase_a = 0.1;
    cfg_1d.phase_b = -0.2;
    cfg_1d.omega_a = 0.0;
    cfg_1d.omega_b = 0.0;
    cfg_1d.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_1d.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg_1d.coord.origin_x = -0.4;
    cfg_1d.coord.spacing_x = 0.15;
    cfg_1d.coord.origin_y = 0.0;
    cfg_1d.coord.spacing_y = 1.0;
    cfg_1d.use_wavevectors = false;
    cfg_1d.scale_by_dt = false;
    cfg_1d.rotation = 0.0;
    cfg_1d.time_offset = 0.0;

    {
        const size_t shape_1d[1] = {31U};
        ok &= run_moire_case("moire_1d_axis", 1U, shape_1d, cfg_1d, 1.0e-9);
    }

    SimStimulusMoireConfig cfg_2d;
    memset(&cfg_2d, 0, sizeof(cfg_2d));
    cfg_2d.amplitude = 1.1;
    cfg_2d.wavenumber_a = 0.0;
    cfg_2d.wavenumber_b = 0.0;
    cfg_2d.k1x = 1.9;
    cfg_2d.k1y = 0.7;
    cfg_2d.k2x = 2.05;
    cfg_2d.k2y = 0.76;
    cfg_2d.phase_a = 0.3;
    cfg_2d.phase_b = -0.15;
    cfg_2d.omega_a = 0.0;
    cfg_2d.omega_b = 0.0;
    cfg_2d.coord.origin_x = -0.2;
    cfg_2d.coord.origin_y = 0.5;
    cfg_2d.coord.spacing_x = 0.4;
    cfg_2d.coord.spacing_y = 0.3;
    cfg_2d.use_wavevectors = true;
    cfg_2d.scale_by_dt = false;
    cfg_2d.rotation = 0.0;
    cfg_2d.time_offset = 0.0;

    {
        const size_t shape_2d[2] = {6U, 9U};
        ok &= run_moire_case("moire_2d_wavevector", 2U, shape_2d, cfg_2d, 1.0e-9);
    }

    SimStimulusMoireConfig cfg_2d_wavevector_wide = cfg_2d;
    cfg_2d_wavevector_wide.k1x = 1.65;
    cfg_2d_wavevector_wide.k1y = 0.58;
    cfg_2d_wavevector_wide.k2x = 1.79;
    cfg_2d_wavevector_wide.k2y = 0.64;
    cfg_2d_wavevector_wide.phase_a = -0.22;
    cfg_2d_wavevector_wide.phase_b = 0.18;
    cfg_2d_wavevector_wide.omega_a = 0.14;
    cfg_2d_wavevector_wide.omega_b = -0.09;
    cfg_2d_wavevector_wide.coord.origin_x = -1.3;
    cfg_2d_wavevector_wide.coord.origin_y = -0.45;
    cfg_2d_wavevector_wide.coord.spacing_x = 0.05;
    cfg_2d_wavevector_wide.coord.spacing_y = 0.08;
    cfg_2d_wavevector_wide.coord.velocity_x = 0.04;
    cfg_2d_wavevector_wide.coord.velocity_y = -0.03;
    cfg_2d_wavevector_wide.time_offset = 0.12;

    {
        const size_t shape_2d_wide[2] = {4U, 96U};
        ok &= run_moire_case("moire_2d_wavevector_wide", 2U, shape_2d_wide, cfg_2d_wavevector_wide,
                             5.0e-9);
    }

    SimStimulusMoireConfig cfg_2d_coord;
    memset(&cfg_2d_coord, 0, sizeof(cfg_2d_coord));
    cfg_2d_coord.amplitude = 0.65;
    cfg_2d_coord.wavenumber_a = 2.2;
    cfg_2d_coord.wavenumber_b = 2.35;
    cfg_2d_coord.phase_a = -0.1;
    cfg_2d_coord.phase_b = 0.25;
    cfg_2d_coord.omega_a = 0.0;
    cfg_2d_coord.omega_b = 0.0;
    cfg_2d_coord.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg_2d_coord.coord.angle = 0.7;
    cfg_2d_coord.coord.origin_x = -0.3;
    cfg_2d_coord.coord.origin_y = 0.2;
    cfg_2d_coord.coord.spacing_x = 0.25;
    cfg_2d_coord.coord.spacing_y = 0.4;
    cfg_2d_coord.use_wavevectors = false;
    cfg_2d_coord.scale_by_dt = false;
    cfg_2d_coord.rotation = 0.0;
    cfg_2d_coord.time_offset = 0.0;

    {
        const size_t shape_2d[2] = {6U, 9U};
        ok &= run_moire_case("moire_2d_angle_coord", 2U, shape_2d, cfg_2d_coord, 1.0e-9);
    }

    SimStimulusMoireConfig cfg_2d_separable = cfg_2d_coord;
    cfg_2d_separable.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg_2d_separable.coord.combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;
    cfg_2d_separable.coord.origin_x = -1.4;
    cfg_2d_separable.coord.origin_y = -0.6;
    cfg_2d_separable.coord.spacing_x = 0.05;
    cfg_2d_separable.coord.spacing_y = 0.07;
    cfg_2d_separable.coord.velocity_x = 0.03;
    cfg_2d_separable.coord.velocity_y = -0.02;
    cfg_2d_separable.omega_a = 0.11;
    cfg_2d_separable.omega_b = -0.07;
    cfg_2d_separable.time_offset = 0.09;

    {
        const size_t shape_2d_wide[2] = {4U, 96U};
        ok &=
            run_moire_case("moire_2d_separable_wide", 2U, shape_2d_wide, cfg_2d_separable, 5.0e-9);
    }

    SimStimulusMoireConfig cfg_2d_spiral = cfg_2d_coord;
    cfg_2d_spiral.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg_2d_spiral.coord.center_x = 0.2;
    cfg_2d_spiral.coord.center_y = -0.1;
    cfg_2d_spiral.coord.velocity_x = 0.05;
    cfg_2d_spiral.coord.velocity_y = -0.04;
    cfg_2d_spiral.coord.spiral_arms = 1.5;
    cfg_2d_spiral.coord.spiral_pitch = 0.9;
    cfg_2d_spiral.coord.spiral_phase = 0.4;
    cfg_2d_spiral.coord.spiral_angular_velocity = 0.2;
    cfg_2d_spiral.time_offset = 0.3;

    {
        const size_t shape_2d[2] = {6U, 9U};
        ok &= run_moire_case("moire_2d_spiral_coord", 2U, shape_2d, cfg_2d_spiral, 1.0e-9);
    }

    return ok ? 0 : 1;
}
