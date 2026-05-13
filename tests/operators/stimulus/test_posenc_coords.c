/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static void posenc_eval_scalar(const SimStimulusPosEncConfig *cfg, double u, double t,
                               double *out_re, double *out_im) {
    double re = cfg->include_identity ? u : 0.0;
    double im = 0.0;

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->band_count; ++i) {
        double k = cfg->base_wavenumber * band_scale;
        double theta = k * u - cfg->omega * t + cfg->phase;
        re += cos(theta);
        im += sin(theta);
        band_scale *= cfg->band_growth;
    }

    double count = (double)cfg->band_count + (cfg->include_identity ? 1.0 : 0.0);
    if (count > 1.0e-12) {
        double norm = 1.0 / sqrt(count);
        re *= norm;
        im *= norm;
    }

    *out_re = re;
    *out_im = im;
}

static void posenc_eval_wavevector(const SimStimulusPosEncConfig *cfg, double x, double y, double t,
                                   double *out_re, double *out_im) {
    double projection = cfg->kx * x + cfg->ky * y;
    double kv_norm = hypot(cfg->kx, cfg->ky);
    double u = (kv_norm > 1.0e-12) ? (projection / kv_norm) : x;

    double re = cfg->include_identity ? u : 0.0;
    double im = 0.0;

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->band_count; ++i) {
        double k = cfg->base_wavenumber * band_scale;
        double theta = k * projection - cfg->omega * t + cfg->phase;
        re += cos(theta);
        im += sin(theta);
        band_scale *= cfg->band_growth;
    }

    double count = (double)cfg->band_count + (cfg->include_identity ? 1.0 : 0.0);
    if (count > 1.0e-12) {
        double norm = 1.0 / sqrt(count);
        re *= norm;
        im *= norm;
    }

    *out_re = re;
    *out_im = im;
}

static double posenc_eval_real(const SimStimulusPosEncConfig *cfg, double x, double y, double t) {
    double re = 0.0;
    double im = 0.0;
    double sample_x = x;
    double sample_y = y;

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

    if (cfg->use_wavevector) {
        posenc_eval_wavevector(cfg, sample_x, sample_y, t, &re, &im);
    } else if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double rx = 0.0;
        double ix = 0.0;
        double ry = 0.0;
        double iy = 0.0;
        posenc_eval_scalar(cfg, sample_x, t, &rx, &ix);
        posenc_eval_scalar(cfg, sample_y, t, &ry, &iy);
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            re = rx + ry;
            im = ix + iy;
        } else {
            re = rx * ry - ix * iy;
            im = rx * iy + ix * ry;
        }
    } else {
        double u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
        posenc_eval_scalar(cfg, u, t, &re, &im);
    }

    (void)im;
    return cfg->amplitude * re;
}

static int run_posenc_case(const char *label, size_t rank, const size_t *shape,
                           const SimStimulusPosEncConfig cfg_input, double tol) {
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

    SimStimulusPosEncConfig cfg = cfg_input;
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_posenc_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_add_stimulus_posenc_operator failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusPosEncConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_posenc_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_stimulus_posenc_config failed\n", label);
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
        double expected = posenc_eval_real(&normalized, x, y, eval_t);
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

    SimStimulusPosEncConfig cfg_1d;
    memset(&cfg_1d, 0, sizeof(cfg_1d));
    cfg_1d.amplitude = 0.8;
    cfg_1d.base_wavenumber = 1.7;
    cfg_1d.band_growth = 2.0;
    cfg_1d.band_count = 5U;
    cfg_1d.omega = 0.35;
    cfg_1d.phase = 0.2;
    cfg_1d.time_offset = -0.1;
    cfg_1d.include_identity = false;
    cfg_1d.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_1d.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg_1d.coord.origin_x = -0.5;
    cfg_1d.coord.spacing_x = 0.1;
    cfg_1d.scale_by_dt = false;

    {
        const size_t shape_1d[1] = {37U};
        ok &= run_posenc_case("posenc_1d_axis", 1U, shape_1d, cfg_1d, 1.0e-9);
    }
    {
        const size_t shape_1d_wide[1] = {96U};
        ok &= run_posenc_case("posenc_1d_axis_wide", 1U, shape_1d_wide, cfg_1d, 1.0e-9);
    }

    SimStimulusPosEncConfig cfg_2d_wave;
    memset(&cfg_2d_wave, 0, sizeof(cfg_2d_wave));
    cfg_2d_wave.amplitude = 1.1;
    cfg_2d_wave.base_wavenumber = 0.9;
    cfg_2d_wave.band_growth = 1.6;
    cfg_2d_wave.band_count = 4U;
    cfg_2d_wave.kx = 1.3;
    cfg_2d_wave.ky = -0.7;
    cfg_2d_wave.omega = -0.22;
    cfg_2d_wave.phase = 0.4;
    cfg_2d_wave.time_offset = 0.3;
    cfg_2d_wave.include_identity = true;
    cfg_2d_wave.use_wavevector = true;
    cfg_2d_wave.coord.origin_x = -0.3;
    cfg_2d_wave.coord.origin_y = 0.2;
    cfg_2d_wave.coord.spacing_x = 0.25;
    cfg_2d_wave.coord.spacing_y = 0.18;
    cfg_2d_wave.scale_by_dt = false;

    {
        const size_t shape_2d[2] = {8U, 6U};
        ok &= run_posenc_case("posenc_2d_wavevector", 2U, shape_2d, cfg_2d_wave, 1.0e-9);
    }
    {
        const size_t shape_2d_wide[2] = {4U, 96U};
        ok &= run_posenc_case("posenc_2d_wavevector_wide", 2U, shape_2d_wide, cfg_2d_wave, 1.0e-9);
    }

    SimStimulusPosEncConfig cfg_angle = cfg_2d_wave;
    cfg_angle.use_wavevector = false;
    cfg_angle.include_identity = false;
    cfg_angle.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg_angle.coord.angle = 0.41;
    cfg_angle.coord.origin_x = -0.25;
    cfg_angle.coord.origin_y = 0.35;
    cfg_angle.coord.spacing_x = 0.08;
    cfg_angle.coord.spacing_y = 0.11;
    cfg_angle.coord.velocity_x = 0.03;
    cfg_angle.coord.velocity_y = -0.04;

    {
        const size_t shape_angle[2] = {4U, 96U};
        ok &= run_posenc_case("posenc_2d_angle_wide", 2U, shape_angle, cfg_angle, 1.0e-9);
    }

    SimStimulusPosEncConfig cfg_sep;
    memset(&cfg_sep, 0, sizeof(cfg_sep));
    cfg_sep.amplitude = 0.6;
    cfg_sep.base_wavenumber = 1.3;
    cfg_sep.band_growth = 2.0;
    cfg_sep.band_count = 3U;
    cfg_sep.phase = -0.15;
    cfg_sep.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg_sep.coord.combine = SIM_STIMULUS_SEPARABLE_ADD;
    cfg_sep.coord.origin_x = -0.2;
    cfg_sep.coord.origin_y = 0.4;
    cfg_sep.coord.spacing_x = 0.12;
    cfg_sep.coord.spacing_y = 0.15;
    cfg_sep.scale_by_dt = false;

    {
        const size_t shape_sep[2] = {5U, 7U};
        ok &= run_posenc_case("posenc_2d_separable", 2U, shape_sep, cfg_sep, 1.0e-9);
    }
    {
        const size_t shape_sep_wide[2] = {4U, 96U};
        ok &= run_posenc_case("posenc_2d_separable_wide", 2U, shape_sep_wide, cfg_sep, 1.0e-9);
    }

    cfg_sep.coord.combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;
    cfg_sep.include_identity = true;
    cfg_sep.omega = 0.27;
    cfg_sep.time_offset = 0.16;

    {
        const size_t shape_sep_mul[2] = {4U, 96U};
        ok &= run_posenc_case("posenc_2d_separable_multiply_wide", 2U, shape_sep_mul, cfg_sep,
                              1.0e-9);
    }

    return ok ? 0 : 1;
}
