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

static double steerable_wrap_pi(double angle) {
    const double two_pi = 2.0 * M_PI;
    double wrapped = fmod(angle + M_PI, two_pi);
    if (wrapped < 0.0) {
        wrapped += two_pi;
    }
    return wrapped - M_PI;
}

static void steerable_map_coord(const SimStimulusSteerableWaveletConfig *cfg, double x, double y,
                                double t, double *out_u, double *out_v) {
    double sample_x = x;
    double sample_y = y;
    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

    if (cfg->use_wavevector) {
        double norm = hypot(cfg->kx, cfg->ky);
        double ux = 1.0;
        double uy = 0.0;
        if (norm > 1.0e-12) {
            ux = cfg->kx / norm;
            uy = cfg->ky / norm;
        }
        *out_u = ux * sample_x + uy * sample_y;
        *out_v = -uy * sample_x + ux * sample_y;
        return;
    }

    switch (cfg->coord.mode) {
    case SIM_STIMULUS_COORD_AXIS:
        if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
            *out_u = sample_y;
            *out_v = sample_x;
        } else {
            *out_u = sample_x;
            *out_v = sample_y;
        }
        return;
    case SIM_STIMULUS_COORD_ANGLE: {
        double s = sin(cfg->coord.angle);
        double c = cos(cfg->coord.angle);
        *out_u = sample_x * c + sample_y * s;
        *out_v = -sample_x * s + sample_y * c;
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
    case SIM_STIMULUS_COORD_AZIMUTH: {
        double cx = cfg->coord.center_x + cfg->coord.velocity_x * t;
        double cy = cfg->coord.center_y + cfg->coord.velocity_y * t;
        double dx = x - cx;
        double dy = y - cy;
        *out_u = atan2(dy, dx);
        *out_v = hypot(dx, dy);
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
        *out_u = sample_x;
        *out_v = sample_y;
        return;
    }
}

static void steerable_eval_uv(const SimStimulusSteerableWaveletConfig *cfg, double u, double v,
                              double t, double *out_re, double *out_im) {
    double steer = cfg->orientation + cfg->orientation_rate * t;
    double r = hypot(u, v);
    double phi = atan2(v, u);
    double delta = steerable_wrap_pi(phi - steer);

    double re_sum = 0.0;
    double im_sum = 0.0;

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->scale_count; ++i) {
        double k = cfg->base_wavenumber * band_scale;
        double target_radius = fabs(k);
        if (target_radius <= 1.0e-12) {
            target_radius = 1.0;
        }

        double log_ratio = log((r + 1.0e-12) / (target_radius + 1.0e-12));
        double radial_u = log_ratio / cfg->radial_bandwidth;
        double radial_env = exp(-0.5 * radial_u * radial_u);

        double theta = k * r - cfg->omega * t + cfg->phase;
        double carrier_re = cos(theta);
        double carrier_im = sin(theta);

        if (cfg->family == SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI) {
            double c = cos(delta);
            if (c < 0.0) {
                c = 0.0;
            }
            double exponent = (double)cfg->order * cfg->angular_sharpness;
            if (exponent < 1.0e-6) {
                exponent = 1.0;
            }
            double angular_env = pow(c, exponent);
            re_sum += radial_env * angular_env * carrier_re;
            im_sum += radial_env * angular_env * carrier_im;
        } else {
            double ndelta = (double)cfg->order * delta;
            double ore = cos(ndelta);
            double oim = sin(ndelta);
            double angular_env = pow(fmax(fabs(cos(delta)), 1.0e-12), cfg->angular_sharpness);
            double mixed_re = carrier_re * ore - carrier_im * oim;
            double mixed_im = carrier_re * oim + carrier_im * ore;
            re_sum += radial_env * angular_env * mixed_re;
            im_sum += radial_env * angular_env * mixed_im;
        }

        band_scale *= cfg->scale_growth;
    }

    double norm = (cfg->scale_count > 0U) ? (1.0 / sqrt((double)cfg->scale_count)) : 1.0;
    *out_re = re_sum * norm;
    *out_im = im_sum * norm;
}

static double steerable_eval_real(const SimStimulusSteerableWaveletConfig *cfg, double x, double y,
                                  double t) {
    double re = 0.0;
    double im = 0.0;

    if (!cfg->use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double rx = 0.0;
        double ix = 0.0;
        double ry = 0.0;
        double iy = 0.0;
        steerable_eval_uv(cfg, x, 0.0, t, &rx, &ix);
        steerable_eval_uv(cfg, y, 0.0, t, &ry, &iy);
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
        steerable_map_coord(cfg, x, y, t, &u, &v);
        steerable_eval_uv(cfg, u, v, t, &re, &im);
    }

    (void)im;
    return cfg->amplitude * re;
}

static int run_steerable_case(const char *label, size_t rank, const size_t *shape,
                              const SimStimulusSteerableWaveletConfig cfg_input, double tol) {
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

    SimStimulusSteerableWaveletConfig cfg = cfg_input;
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_steerable_wavelet_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_add_stimulus_steerable_wavelet_operator failed\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusSteerableWaveletConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_steerable_wavelet_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] sim_stimulus_steerable_wavelet_config failed\n", label);
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
        double expected = steerable_eval_real(&normalized, x, y, eval_t);
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

    SimStimulusSteerableWaveletConfig cfg_axis;
    memset(&cfg_axis, 0, sizeof(cfg_axis));
    cfg_axis.amplitude = 0.7;
    cfg_axis.family = SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI;
    cfg_axis.order = 2U;
    cfg_axis.scale_count = 4U;
    cfg_axis.base_wavenumber = 1.1;
    cfg_axis.scale_growth = 1.8;
    cfg_axis.radial_bandwidth = 0.7;
    cfg_axis.angular_sharpness = 1.5;
    cfg_axis.orientation = 0.2;
    cfg_axis.orientation_rate = 0.1;
    cfg_axis.omega = 0.35;
    cfg_axis.phase = -0.2;
    cfg_axis.time_offset = 0.05;
    cfg_axis.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_axis.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg_axis.coord.origin_x = -0.8;
    cfg_axis.coord.spacing_x = 0.05;

    {
        const size_t shape_1d[1] = {61U};
        ok &= run_steerable_case("steerable_1d_axis", 1U, shape_1d, cfg_axis, 1.0e-9);
    }

    SimStimulusSteerableWaveletConfig cfg_wave;
    memset(&cfg_wave, 0, sizeof(cfg_wave));
    cfg_wave.amplitude = 1.0;
    cfg_wave.family = SIM_STIMULUS_STEERABLE_WAVELET_RIESZ;
    cfg_wave.order = 3U;
    cfg_wave.scale_count = 5U;
    cfg_wave.base_wavenumber = 0.9;
    cfg_wave.scale_growth = 2.0;
    cfg_wave.radial_bandwidth = 0.6;
    cfg_wave.angular_sharpness = 1.2;
    cfg_wave.orientation = -0.4;
    cfg_wave.kx = 1.4;
    cfg_wave.ky = -0.9;
    cfg_wave.use_wavevector = true;
    cfg_wave.omega = -0.15;
    cfg_wave.phase = 0.3;
    cfg_wave.time_offset = 0.2;
    cfg_wave.coord.origin_x = -0.3;
    cfg_wave.coord.origin_y = 0.4;
    cfg_wave.coord.spacing_x = 0.2;
    cfg_wave.coord.spacing_y = 0.17;

    {
        const size_t shape_2d[2] = {9U, 7U};
        ok &= run_steerable_case("steerable_2d_wavevector", 2U, shape_2d, cfg_wave, 1.0e-9);
    }

    SimStimulusSteerableWaveletConfig cfg_fast = cfg_axis;
    cfg_fast.coord.origin_x = -1.28;

    {
        const size_t shape_fast[1] = {96U};
        ok &= run_steerable_case("steerable_1d_axis_fastpath", 1U, shape_fast, cfg_fast, 1.0e-9);
    }

    SimStimulusSteerableWaveletConfig cfg_wave_fast = cfg_wave;
    cfg_wave_fast.coord.origin_x = -1.28;
    cfg_wave_fast.coord.origin_y = -0.34;
    cfg_wave_fast.coord.spacing_x = 0.05;
    cfg_wave_fast.coord.spacing_y = 0.08;

    {
        const size_t shape_fast_wave[2] = {4U, 96U};
        ok &= run_steerable_case("steerable_2d_wavevector_fastpath", 2U, shape_fast_wave,
                                 cfg_wave_fast, 1.0e-9);
    }

    SimStimulusSteerableWaveletConfig cfg_sep;
    memset(&cfg_sep, 0, sizeof(cfg_sep));
    cfg_sep.amplitude = 0.5;
    cfg_sep.family = SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI;
    cfg_sep.order = 1U;
    cfg_sep.scale_count = 3U;
    cfg_sep.base_wavenumber = 1.6;
    cfg_sep.scale_growth = 1.7;
    cfg_sep.radial_bandwidth = 0.8;
    cfg_sep.angular_sharpness = 2.0;
    cfg_sep.phase = 0.1;
    cfg_sep.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg_sep.coord.combine = SIM_STIMULUS_SEPARABLE_ADD;
    cfg_sep.coord.origin_x = -0.2;
    cfg_sep.coord.origin_y = 0.1;
    cfg_sep.coord.spacing_x = 0.11;
    cfg_sep.coord.spacing_y = 0.14;

    {
        const size_t shape_sep[2] = {6U, 5U};
        ok &= run_steerable_case("steerable_2d_separable", 2U, shape_sep, cfg_sep, 1.0e-9);
    }

    SimStimulusSteerableWaveletConfig cfg_azimuth;
    memset(&cfg_azimuth, 0, sizeof(cfg_azimuth));
    cfg_azimuth.amplitude = 0.63;
    cfg_azimuth.family = SIM_STIMULUS_STEERABLE_WAVELET_RIESZ;
    cfg_azimuth.order = 2U;
    cfg_azimuth.scale_count = 4U;
    cfg_azimuth.base_wavenumber = 1.0;
    cfg_azimuth.scale_growth = 1.65;
    cfg_azimuth.radial_bandwidth = 0.72;
    cfg_azimuth.angular_sharpness = 1.4;
    cfg_azimuth.orientation = 0.35;
    cfg_azimuth.orientation_rate = -0.08;
    cfg_azimuth.omega = 0.18;
    cfg_azimuth.phase = -0.16;
    cfg_azimuth.time_offset = 0.07;
    cfg_azimuth.coord.mode = SIM_STIMULUS_COORD_AZIMUTH;
    cfg_azimuth.coord.origin_x = -0.25;
    cfg_azimuth.coord.origin_y = -0.3;
    cfg_azimuth.coord.spacing_x = 0.17;
    cfg_azimuth.coord.spacing_y = 0.19;
    cfg_azimuth.coord.center_x = 0.12;
    cfg_azimuth.coord.center_y = -0.05;
    cfg_azimuth.coord.velocity_x = -0.04;
    cfg_azimuth.coord.velocity_y = 0.03;

    {
        const size_t shape_az[2] = {7U, 6U};
        ok &= run_steerable_case("steerable_2d_azimuth", 2U, shape_az, cfg_azimuth, 1.0e-9);
    }

    SimStimulusSteerableWaveletConfig cfg_polar = cfg_azimuth;
    cfg_polar.coord.mode = SIM_STIMULUS_COORD_POLAR;

    {
        const size_t shape_po[2] = {7U, 6U};
        ok &= run_steerable_case("steerable_2d_polar", 2U, shape_po, cfg_polar, 1.0e-9);
    }

    return ok ? 0 : 1;
}
