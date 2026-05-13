#include <oakfield/math/airy.h>
#include <oakfield/math/bessel.h>
#include <oakfield/math/fourier.h>
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define TEST_STIM_FOURIER_EPS 1.0e-12
#define TEST_STIM_FOURIER_TWO_PI (2.0 * M_PI)
#define TEST_STIM_FOURIER_MAX_HARMONICS 16384

static bool nearly_equal(double a, double b, double eps) {
    return fabs(a - b) <= eps * fmax(1.0, fmax(fabs(a), fabs(b)));
}

static bool check_hermitian_spectrum(const SimComplexDouble *data, size_t count, double eps) {
    if (data == NULL || count < 2U) {
        return false;
    }

    if (!nearly_equal(data[0].im, 0.0, eps)) {
        return false;
    }
    if ((count % 2U) == 0U) {
        size_t nyquist = count / 2U;
        if (!nearly_equal(data[nyquist].im, 0.0, eps)) {
            return false;
        }
    }

    for (size_t k = 1U; k < count / 2U; ++k) {
        size_t nk = count - k;
        if (!nearly_equal(data[k].re, data[nk].re, eps) ||
            !nearly_equal(data[k].im, -data[nk].im, eps)) {
            return false;
        }
    }

    return true;
}

static double sample_value(size_t index, size_t count) {
    double x = (double)index / (double)count;
    return 0.75 * sin(2.0 * M_PI * x) + 0.35 * cos(6.0 * M_PI * x);
}

static double test_stimulus_fourier_phase_increment(const SimFourierWaveformConfig *cfg,
                                                    double dt_sub) {
    if (cfg == NULL) {
        return 0.0;
    }

    double dt = cfg->fixed_clock
                    ? ((cfg->nominal_dt > TEST_STIM_FOURIER_EPS) ? cfg->nominal_dt : dt_sub)
                    : dt_sub;
    if (dt <= 0.0) {
        return 0.0;
    }
    return cfg->frequency * dt;
}

static double test_stimulus_fourier_saw_correction(double phase, double dphase, bool mini) {
    double corr = mini ? sim_fourier_miniblep(phase, dphase) : sim_fourier_polyblep(phase, dphase);
    double next = phase + dphase;
    if (next >= 1.0) {
        corr -= mini ? sim_fourier_miniblep(next - 1.0, dphase)
                     : sim_fourier_polyblep(next - 1.0, dphase);
    }
    return corr;
}

static double test_stimulus_fourier_square_correction(double phase, double dphase, double duty,
                                                      bool mini) {
    double rising =
        mini ? sim_fourier_miniblep(phase, dphase) : sim_fourier_polyblep(phase, dphase);
    double edge_phase = phase - duty;
    if (edge_phase < 0.0) {
        edge_phase += 1.0;
    }
    double falling =
        mini ? sim_fourier_miniblep(edge_phase, dphase) : sim_fourier_polyblep(edge_phase, dphase);
    return rising - falling;
}

static double test_stimulus_fourier_center_bipolar(double x) {
    double wrapped = fmod(x + 1.0, 2.0);
    if (wrapped < 0.0) {
        wrapped += 2.0;
    }
    return wrapped - 1.0;
}

static int test_stimulus_fourier_harmonics(double dphase) {
    if (dphase <= 0.0) {
        return 0;
    }

    double partials = floor(0.5 / dphase);
    if (partials < 1.0) {
        partials = 1.0;
    }
    if (partials > (double)TEST_STIM_FOURIER_MAX_HARMONICS) {
        partials = (double)TEST_STIM_FOURIER_MAX_HARMONICS;
    }
    return (int)partials;
}

static double test_stimulus_fourier_step_sample(const SimFourierWaveformConfig *cfg, double dt) {
    if (cfg == NULL || cfg->amplitude == 0.0 || cfg->frequency <= 0.0) {
        return 0.0;
    }

    double dphase = test_stimulus_fourier_phase_increment(cfg, dt);
    if (dphase <= 0.0) {
        return 0.0;
    }

    double phase = cfg->phase;
    double blit_state = 0.0;
    double tri_vel = 0.0;
    double tri_pos = 0.0;
    double sample = 0.0;

    switch (cfg->method) {
    case SIM_FOURIER_METHOD_BLIT: {
        int harmonics = test_stimulus_fourier_harmonics(dphase);
        if (harmonics <= 0) {
            return 0.0;
        }

        double phase_radians = TEST_STIM_FOURIER_TWO_PI * phase;
        double dphase_radians = TEST_STIM_FOURIER_TWO_PI * dphase;

        if (cfg->shape == SIM_FOURIER_WAVEFORM_SAW) {
            sample = sim_fourier_saw_blit(phase_radians, dphase_radians, harmonics, &blit_state);
        } else if (cfg->shape == SIM_FOURIER_WAVEFORM_SQUARE) {
            sample = sim_fourier_square_blit(phase_radians, dphase_radians, harmonics, cfg->duty,
                                             &blit_state);
        } else {
            sample = sim_fourier_triangle_blit(phase_radians, dphase_radians, harmonics, &tri_vel,
                                               &tri_pos);
        }
        break;
    }
    case SIM_FOURIER_METHOD_MINIBLEP:
    case SIM_FOURIER_METHOD_POLYBLEP:
    default: {
        bool mini = (cfg->method == SIM_FOURIER_METHOD_MINIBLEP);
        if (cfg->shape == SIM_FOURIER_WAVEFORM_SAW) {
            double saw = 2.0 * phase - 1.0;
            saw -= test_stimulus_fourier_saw_correction(phase, dphase, mini);
            sample = saw;
        } else if (cfg->shape == SIM_FOURIER_WAVEFORM_SQUARE) {
            double square = (phase < cfg->duty) ? 1.0 : -1.0;
            square += test_stimulus_fourier_square_correction(phase, dphase, cfg->duty, mini);
            sample = square;
        } else {
            double square = (phase < cfg->duty) ? 1.0 : -1.0;
            square += test_stimulus_fourier_square_correction(phase, dphase, cfg->duty, mini);
            tri_vel = square;
            tri_pos += square * (4.0 * dphase);
            tri_pos = test_stimulus_fourier_center_bipolar(tri_pos);
            sample = tri_pos;
        }
        break;
    }
    }

    if (!isfinite(sample)) {
        return 0.0;
    }

    sample *= cfg->amplitude;
    if (cfg->scale_by_dt) {
        sample *= dt;
    }
    return sample;
}

static void test_posenc_eval_wavevector(const SimStimulusPosEncConfig *cfg, double x, double y,
                                        double t, double *out_re, double *out_im) {
    double projection = cfg->kx * x + cfg->ky * y;
    double kv_norm = hypot(cfg->kx, cfg->ky);
    double identity_u = (kv_norm > 1.0e-12) ? (projection / kv_norm) : x;
    double re = cfg->include_identity ? identity_u : 0.0;
    double im = 0.0;

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->band_count; ++i) {
        double k = cfg->base_wavenumber * band_scale;
        double theta = k * projection - cfg->omega * t + cfg->phase;
        re += cos(theta);
        im += sin(theta);
        band_scale *= cfg->band_growth;
    }

    {
        double norm_count = (double)cfg->band_count + (cfg->include_identity ? 1.0 : 0.0);
        if (norm_count > 1.0e-12) {
            double norm = 1.0 / sqrt(norm_count);
            re *= norm;
            im *= norm;
        }
    }

    *out_re = re;
    *out_im = im;
}

static void test_chladni_map_coord(const SimStimulusChladniConfig *cfg, double x, double y,
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

static void test_chladni_eval(const SimStimulusChladniConfig *cfg, double u, double v, double t,
                              double *out_re, double *out_im) {
    double px = M_PI * u / cfg->plate_width;
    double py = M_PI * v / cfg->plate_height;
    double a = cos((double)cfg->mode_x * px) * cos((double)cfg->mode_y * py);
    double b = cos((double)cfg->mode_y * px) * cos((double)cfg->mode_x * py);
    double d = a - cfg->mix * b;
    double band = exp(-0.5 * (d * d) / (cfg->line_width * cfg->line_width));
    double carrier = -cfg->omega * t + cfg->phase;

    *out_re = band * cos(carrier);
    *out_im = band * sin(carrier);
}

static void test_moire_eval_1d(const SimStimulusMoireConfig *cfg, double u, double t,
                               double *out_re, double *out_im) {
    double theta_a = cfg->wavenumber_a * u - cfg->omega_a * t + cfg->phase_a;
    double theta_b = cfg->wavenumber_b * u - cfg->omega_b * t + cfg->phase_b;
    *out_re = 0.5 * (cos(theta_a) + cos(theta_b));
    *out_im = 0.5 * (sin(theta_a) + sin(theta_b));
}

static void test_moire_eval(const SimStimulusMoireConfig *cfg, double x, double y, double t,
                            double *out_re, double *out_im) {
    double sample_x = x;
    double sample_y = y;

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

    if (cfg->use_wavevectors) {
        double theta_a =
            cfg->k1x * sample_x + cfg->k1y * sample_y - cfg->omega_a * t + cfg->phase_a;
        double theta_b =
            cfg->k2x * sample_x + cfg->k2y * sample_y - cfg->omega_b * t + cfg->phase_b;
        *out_re = 0.5 * (cos(theta_a) + cos(theta_b));
        *out_im = 0.5 * (sin(theta_a) + sin(theta_b));
        return;
    }

    if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        double rx = 0.0;
        double ix = 0.0;
        double ry = 0.0;
        double iy = 0.0;
        test_moire_eval_1d(cfg, sample_x, t, &rx, &ix);
        test_moire_eval_1d(cfg, sample_y, t, &ry, &iy);
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            *out_re = rx + ry;
            *out_im = ix + iy;
        } else {
            *out_re = rx * ry - ix * iy;
            *out_im = rx * iy + ix * ry;
        }
        return;
    }

    {
        double u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
        test_moire_eval_1d(cfg, u, t, out_re, out_im);
    }
}

static void test_zone_plate_map_coord(const SimStimulusZonePlateConfig *cfg, double x, double y,
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

static void test_zone_plate_eval(const SimStimulusZonePlateConfig *cfg, double u, double v,
                                 double t, double *out_re, double *out_im) {
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

static void test_traveling_wave_packet_map_coord(const SimStimulusTravelingWavePacketConfig *cfg,
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

static void test_traveling_wave_packet_eval(const SimStimulusTravelingWavePacketConfig *cfg,
                                            double u, double v, double t, double *out_re,
                                            double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;
    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;
    double exponent = -0.5 * ((ur * ur) / (cfg->sigma_u * cfg->sigma_u) +
                              (vr * vr) / (cfg->sigma_v * cfg->sigma_v));
    double envelope = exp(exponent);
    double carrier = cfg->carrier_u * ur + cfg->carrier_v * vr - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(carrier);
    *out_im = envelope * sin(carrier);
}

static void
test_cylindrical_wave_emitter_map_coord(const SimStimulusCylindricalWaveEmitterConfig *cfg,
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

static void test_cylindrical_wave_emitter_eval(const SimStimulusCylindricalWaveEmitterConfig *cfg,
                                               size_t rank, double u, double v, double t,
                                               double *out_re, double *out_im) {
    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;
    double rho = (rank > 1U) ? hypot(du, dv) : fabs(du);
    double radius = hypot(rho, cfg->softening_radius);
    double envelope = exp(-cfg->attenuation * radius) / sqrt(radius);
    double phase = cfg->radial_wavenumber * radius - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(phase);
    *out_im = envelope * sin(phase);
}

static void test_laplace_beltrami_map_coord(const SimStimulusLaplaceBeltramiConfig *cfg, double x,
                                            double y, double t, double *out_u, double *out_v) {
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

static double test_laplace_beltrami_dirichlet_mode(int mode, double coord, double extent) {
    int n = mode;
    if (n < 0) {
        n = -n;
    }
    if (n == 0) {
        n = 1;
    }
    return sin(M_PI * (double)n * (coord / extent + 0.5));
}

static double test_laplace_beltrami_periodic_phase(int mode, double coord, double extent) {
    return 2.0 * M_PI * (double)mode * coord / extent;
}

static void test_laplace_beltrami_eval(const SimStimulusLaplaceBeltramiConfig *cfg, double u,
                                       double v, double t, bool include_v, double *out_re,
                                       double *out_im) {
    double spatial_re = 0.0;
    double spatial_im = 0.0;
    double temporal = -cfg->omega * t + cfg->phase;

    switch (cfg->manifold) {
    case SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS: {
        double arg = test_laplace_beltrami_periodic_phase(cfg->mode_u, u, cfg->extent_u);
        if (include_v) {
            arg += test_laplace_beltrami_periodic_phase(cfg->mode_v, v, cfg->extent_v);
        }
        spatial_re = cos(arg);
        spatial_im = sin(arg);
        break;
    }
    case SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER: {
        double arg = test_laplace_beltrami_periodic_phase(cfg->mode_u, u, cfg->extent_u);
        double envelope =
            include_v ? test_laplace_beltrami_dirichlet_mode(cfg->mode_v, v, cfg->extent_v) : 1.0;
        spatial_re = envelope * cos(arg);
        spatial_im = envelope * sin(arg);
        break;
    }
    case SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE:
    default: {
        double value = test_laplace_beltrami_dirichlet_mode(cfg->mode_u, u, cfg->extent_u);
        if (include_v) {
            value *= test_laplace_beltrami_dirichlet_mode(cfg->mode_v, v, cfg->extent_v);
        }
        spatial_re = value;
        spatial_im = 0.0;
        break;
    }
    }

    *out_re = spatial_re * cos(temporal) - spatial_im * sin(temporal);
    *out_im = spatial_re * sin(temporal) + spatial_im * cos(temporal);
}

static void test_optical_vortex_map_coord(const SimStimulusOpticalVortexConfig *cfg, double x,
                                          double y, double t, double *out_u, double *out_v) {
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

static void test_optical_vortex_eval(const SimStimulusOpticalVortexConfig *cfg, double u, double v,
                                     double t, double *out_re, double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;
    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;
    double xu = ur / cfg->waist_x;
    double yv = vr / cfg->waist_y;
    double rho = hypot(xu, yv);
    double envelope = exp(-0.5 * rho * rho);
    int abs_l = (cfg->charge < 0) ? -cfg->charge : cfg->charge;

    if (abs_l > 0) {
        envelope *= pow(rho, (double)abs_l);
    }

    theta = atan2(yv, xu);
    *out_re = envelope * cos((double)cfg->charge * theta - cfg->omega * t + cfg->phase);
    *out_im = envelope * sin((double)cfg->charge * theta - cfg->omega * t + cfg->phase);
}

static void test_bessel_beam_map_coord(const SimStimulusBesselBeamConfig *cfg, double x, double y,
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

static void test_bessel_beam_eval(const SimStimulusBesselBeamConfig *cfg, double u, double v,
                                  double t, double *out_re, double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;
    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;
    double xu = ur / cfg->scale_u;
    double yv = vr / cfg->scale_v;
    double rho = hypot(xu, yv);
    double azimuth = atan2(yv, xu);
    double value = sim_bessel_jn_f64(cfg->order, cfg->radial_wavenumber * rho);
    double phase = (double)cfg->order * azimuth - cfg->omega * t + cfg->phase;

    *out_re = value * cos(phase);
    *out_im = value * sin(phase);
}

static void test_airy_beam_map_coord(const SimStimulusAiryBeamConfig *cfg, double x, double y,
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

static void test_airy_beam_eval(const SimStimulusAiryBeamConfig *cfg, double u, double v, double t,
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
    double expo_arg = cfg->apodization_u * su + cfg->apodization_v * sv;
    if (expo_arg > 60.0) {
        expo_arg = 60.0;
    } else if (expo_arg < -60.0) {
        expo_arg = -60.0;
    }
    double envelope = sim_airy_ai_f64(su) * sim_airy_ai_f64(sv) * exp(expo_arg);
    double phase = cfg->carrier_u * ur + cfg->carrier_v * vr - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(phase);
    *out_im = envelope * sin(phase);
}

static double test_hg_polynomial(unsigned int mode, double x) {
    if (!isfinite(x)) {
        return 0.0;
    }
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

static void test_hermite_gaussian_beam_map_coord(const SimStimulusHermiteGaussianBeamConfig *cfg,
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

static void test_hermite_gaussian_beam_eval(const SimStimulusHermiteGaussianBeamConfig *cfg,
                                            double u, double v, double t, double *out_re,
                                            double *out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s = sin(theta);
    double c = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du = u - center_u;
    double dv = v - center_v;
    double ur = du * c + dv * s;
    double vr = -du * s + dv * c;
    double xi = 1.4142135623730950488 * ur / cfg->waist_u;
    double eta = 1.4142135623730950488 * vr / cfg->waist_v;
    double envelope = test_hg_polynomial(cfg->mode_u, xi) * test_hg_polynomial(cfg->mode_v, eta) *
                      exp(-((ur * ur) / (cfg->waist_u * cfg->waist_u) +
                            (vr * vr) / (cfg->waist_v * cfg->waist_v)));
    double phase = cfg->carrier_u * ur + cfg->carrier_v * vr - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(phase);
    *out_im = envelope * sin(phase);
}

typedef struct {
    uint64_t state;
    uint64_t inc;
} test_rff_pcg32_t;

static uint32_t test_rff_pcg32_random(test_rff_pcg32_t *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void test_rff_pcg32_seed(test_rff_pcg32_t *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    (void)test_rff_pcg32_random(rng);
    rng->state += initstate;
    (void)test_rff_pcg32_random(rng);
}

static double test_rff_uniform(test_rff_pcg32_t *rng) {
    return ldexp(test_rff_pcg32_random(rng), -32);
}

static bool run_phase_rotate_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.25;
    const double phase_rate = 1.6;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double expected[8];
    double *data = sim_field_real_data(&field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        expected[i] = sample_value(i, shape[0]);
        data[i] = expected[i];
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    PhaseRotateOperatorConfig cfg = {.field_index = field_index, .phase_rate = phase_rate};
    if (sim_add_phase_rotate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real: add operator failed\n");
        goto cleanup;
    }

    SimOperator *phase_op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (phase_op == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_real: operator missing\n");
        goto cleanup;
    }
    SimOperatorInfo info = sim_operator_info(phase_op);
    if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
        info.representation.requires_complex_input) {
        fprintf(stderr, "[FAIL] phase_rotate_real: metadata mismatch\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_real: output missing\n");
        goto cleanup;
    }

    const double scale = cos(dt * phase_rate);
    for (size_t i = 0U; i < shape[0]; ++i) {
        if (!nearly_equal(data[i], expected[i] * scale, 1.0e-9)) {
            fprintf(stderr, "[FAIL] phase_rotate_real: value mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_phase_rotate_real_constrained_spectral_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.2;
    const double phase_rate = 0.7;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;

    SimComplexDouble *data = sim_field_complex_data(&field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    data[0].re = 1.1;
    data[1].re = 0.4;
    data[1].im = 0.25;
    data[shape[0] - 1U].re = 0.4;
    data[shape[0] - 1U].im = -0.25;
    data[shape[0] / 2U].re = -0.3;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    PhaseRotateOperatorConfig cfg = {.field_index = field_index, .phase_rate = phase_rate};
    if (sim_add_phase_rotate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: add operator failed\n");
        goto cleanup;
    }

    SimOperator *phase_op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (phase_op == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: operator missing\n");
        goto cleanup;
    }
    SimOperatorInfo info = sim_operator_info(phase_op);
    if (!info.preserves_real ||
        info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT ||
        !info.representation.requires_complex_input) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: metadata mismatch\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: execute failed\n");
        goto cleanup;
    }

    data = sim_field_complex_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: output missing\n");
        goto cleanup;
    }

    const double scale = cos(dt * phase_rate);
    if (!nearly_equal(data[1].re, 0.4 * scale, 1.0e-9) ||
        !nearly_equal(data[1].im, 0.25 * scale, 1.0e-9) ||
        !nearly_equal(data[shape[0] - 1U].re, 0.4 * scale, 1.0e-9) ||
        !nearly_equal(data[shape[0] - 1U].im, -0.25 * scale, 1.0e-9) ||
        !check_hermitian_spectrum(data, shape[0], 1.0e-9)) {
        fprintf(stderr, "[FAIL] phase_rotate_real_constrained: projected real scaling mismatch\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_phase_rotate_complex_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.125;
    const double phase_rate = 1.75;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_complex: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_complex: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble expected[8];
    SimComplexDouble *data = sim_field_complex_data(&field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        expected[i].re = sample_value(i, shape[0]);
        expected[i].im = 0.25 * sample_value((i + 3U) % shape[0], shape[0]);
        data[i] = expected[i];
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_complex: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    PhaseRotateOperatorConfig cfg = {.field_index = field_index, .phase_rate = phase_rate};
    if (sim_add_phase_rotate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_complex: add operator failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_complex: execute failed\n");
        goto cleanup;
    }

    data = sim_field_complex_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_complex: output missing\n");
        goto cleanup;
    }

    const double theta = dt * phase_rate;
    const double s = sin(theta);
    const double c = cos(theta);
    for (size_t i = 0U; i < shape[0]; ++i) {
        double expect_re = expected[i].re * c - expected[i].im * s;
        double expect_im = expected[i].re * s + expected[i].im * c;
        if (!nearly_equal(data[i].re, expect_re, 1.0e-9) ||
            !nearly_equal(data[i].im, expect_im, 1.0e-9)) {
            fprintf(stderr, "[FAIL] phase_rotate_complex: value mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_phase_rotate_imag_zero_constraint_preserved_when_inactive(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: field data missing\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = sample_value(i, shape[0]);
        data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, 0.2f);

    {
        PhaseRotateOperatorConfig cfg = {.field_index = field_index, .phase_rate = 0.0};
        if (sim_add_phase_rotate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: add operator failed\n");
            goto cleanup;
        }
    }

    {
        SimOperator *phase_op = sim_operator_registry_get(&ctx.world.operators, op_index);
        if (phase_op == NULL) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(phase_op);
        if (!info.preserves_real ||
            info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT ||
            !info.representation.requires_complex_input) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: metadata mismatch\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: execute failed\n");
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&ctx, field_index);
        if (out_field == NULL) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: output missing\n");
            goto cleanup;
        }
        SimFieldRepresentation repr = sim_field_representation(out_field);
        if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
            repr.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT) {
            fprintf(stderr,
                    "[FAIL] phase_rotate_imag_zero_inactive: representation mismatch (%d, %d)\n",
                    (int)repr.domain, (int)repr.value_kind);
            goto cleanup;
        }
        data = sim_field_complex_data(out_field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: output data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (!nearly_equal(data[i].re, sample_value(i, shape[0]), 1.0e-12) ||
                !nearly_equal(data[i].im, 0.0, 1.0e-12)) {
                fprintf(stderr, "[FAIL] phase_rotate_imag_zero_inactive: value mismatch at %zu\n",
                        i);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_phase_rotate_imag_zero_constraint_demotes_with_phase(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.125;
    const double phase_rate = 1.75;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: field data missing\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    data[1].re = 1.0;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    {
        PhaseRotateOperatorConfig cfg = {.field_index = field_index, .phase_rate = phase_rate};
        if (sim_add_phase_rotate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: add operator failed\n");
            goto cleanup;
        }
    }

    {
        SimOperator *phase_op = sim_operator_registry_get(&ctx.world.operators, op_index);
        if (phase_op == NULL) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(phase_op);
        if (info.preserves_real ||
            info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
            !info.representation.requires_complex_input) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: metadata mismatch\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: execute failed\n");
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&ctx, field_index);
        if (out_field == NULL) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: output missing\n");
            goto cleanup;
        }
        SimFieldRepresentation repr = sim_field_representation(out_field);
        if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
            repr.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR) {
            fprintf(stderr,
                    "[FAIL] phase_rotate_imag_zero_demote: representation mismatch (%d, %d)\n",
                    (int)repr.domain, (int)repr.value_kind);
            goto cleanup;
        }
        data = sim_field_complex_data(out_field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: output data missing\n");
            goto cleanup;
        }
        {
            const double theta = dt * phase_rate;
            if (!nearly_equal(data[1].re, cos(theta), 1.0e-9) ||
                !nearly_equal(data[1].im, sin(theta), 1.0e-9)) {
                fprintf(stderr, "[FAIL] phase_rotate_imag_zero_demote: value mismatch\n");
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_random_fourier_complex_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] random_fourier_complex: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] random_fourier_complex: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] random_fourier_complex: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] random_fourier_complex: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    SimStimulusRandomFourierConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.63;
    cfg.k_min = 0.3;
    cfg.k_max = 1.9;
    cfg.omega = -0.2;
    cfg.spectral_slope = 0.5;
    cfg.feature_count = 48U;
    cfg.seed = 98765U;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -0.4;
    cfg.coord.spacing_x = 0.07;
    cfg.coord.velocity_x = 0.05;
    cfg.time_offset = 0.13;

    if (sim_add_stimulus_random_fourier_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] random_fourier_complex: add operator failed\n");
        goto cleanup;
    }

    SimStimulusRandomFourierConfig normalized = {0};
    if (sim_stimulus_random_fourier_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] random_fourier_complex: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] random_fourier_complex: execute failed\n");
        goto cleanup;
    }

    data = sim_field_complex_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] random_fourier_complex: output missing\n");
        goto cleanup;
    }

    test_rff_pcg32_t rng;
    test_rff_pcg32_seed(&rng, normalized.seed, normalized.seed ^ 0x9E3779B97F4A7C15ULL);

    double k_values[48];
    double phi_values[48];
    double weight_values[48];
    double dk = normalized.k_max - normalized.k_min;
    double sum_sq = 0.0;

    for (unsigned int i = 0U; i < normalized.feature_count; ++i) {
        double u = test_rff_uniform(&rng);
        double v = test_rff_uniform(&rng);
        double k = (dk > 0.0) ? (normalized.k_min + dk * u) : normalized.k_min;
        double w = 1.0;
        if (normalized.spectral_slope != 0.0) {
            double freq = fabs(k);
            if (freq > 1.0e-9) {
                w = pow(freq, -0.5 * normalized.spectral_slope);
            }
        }
        k_values[i] = k;
        phi_values[i] = 2.0 * M_PI * v;
        weight_values[i] = w;
        sum_sq += w * w;
    }

    double feature_norm = (sum_sq > 1.0e-9) ? sqrt(sum_sq / (double)normalized.feature_count) : 1.0;
    double weight_base = 1.0 / sqrt((double)normalized.feature_count);
    double t = normalized.time_offset;

    for (size_t i = 0U; i < shape[0]; ++i) {
        double x = normalized.coord.origin_x + (double)i * normalized.coord.spacing_x;
        double sample_x = x - normalized.coord.velocity_x * t;
        double expect_re = 0.0;
        double expect_im = 0.0;

        for (unsigned int m = 0U; m < normalized.feature_count; ++m) {
            double theta = k_values[m] * sample_x - normalized.omega * t + phi_values[m];
            double w = weight_base;
            if (normalized.spectral_slope != 0.0) {
                w *= weight_values[m] / feature_norm;
            }
            expect_re += w * cos(theta);
            expect_im += w * sin(theta);
        }

        expect_re *= normalized.amplitude;
        expect_im *= normalized.amplitude;
        if (!nearly_equal(data[i].re, expect_re, 1.0e-9) ||
            !nearly_equal(data[i].im, expect_im, 1.0e-9)) {
            fprintf(stderr, "[FAIL] random_fourier_complex: mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_stimulus_fourier_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double expected_initial[96];
    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        expected_initial[i] = 0.15 * sample_value(i, shape[0]);
        data[i] = expected_initial[i];
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimFourierWaveformConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.8;
    cfg.frequency = 2.3;
    cfg.phase = 0.17;
    cfg.duty = 0.36;
    cfg.rotation = 0.0;
    cfg.nominal_dt = 0.0;
    cfg.shape = SIM_FOURIER_WAVEFORM_SQUARE;
    cfg.method = SIM_FOURIER_METHOD_POLYBLEP;
    cfg.fixed_clock = false;
    cfg.scale_by_dt = false;

    if (sim_add_stimulus_fourier_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: add operator failed\n");
        goto cleanup;
    }

    SimFourierWaveformConfig normalized = {0};
    if (sim_stimulus_fourier_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] stimulus_fourier_real: output missing\n");
        goto cleanup;
    }

    double expected_delta = test_stimulus_fourier_step_sample(&normalized, dt);
    for (size_t i = 0U; i < shape[0]; ++i) {
        double expected = expected_initial[i] + expected_delta;
        if (!nearly_equal(data[i], expected, 1.0e-9)) {
            fprintf(stderr, "[FAIL] stimulus_fourier_real: mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_stimulus_fourier_complex_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.015;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble expected_initial[96];
    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        expected_initial[i].re = 0.08 * sample_value(i, shape[0]);
        expected_initial[i].im = -0.06 * sample_value((i + 11U) % shape[0], shape[0]);
        data[i] = expected_initial[i];
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimFourierWaveformConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.52;
    cfg.frequency = 3.1;
    cfg.phase = 0.42;
    cfg.duty = 0.61;
    cfg.rotation = 0.37;
    cfg.nominal_dt = 0.0;
    cfg.shape = SIM_FOURIER_WAVEFORM_TRIANGLE;
    cfg.method = SIM_FOURIER_METHOD_BLIT;
    cfg.fixed_clock = false;
    cfg.scale_by_dt = true;

    if (sim_add_stimulus_fourier_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: add operator failed\n");
        goto cleanup;
    }

    SimFourierWaveformConfig normalized = {0};
    if (sim_stimulus_fourier_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: execute failed\n");
        goto cleanup;
    }

    data = sim_field_complex_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] stimulus_fourier_complex: output missing\n");
        goto cleanup;
    }

    double expected_delta = test_stimulus_fourier_step_sample(&normalized, dt);
    double expected_re = expected_delta * cos(normalized.rotation);
    double expected_im = expected_delta * sin(normalized.rotation);

    for (size_t i = 0U; i < shape[0]; ++i) {
        double want_re = expected_initial[i].re + expected_re;
        double want_im = expected_initial[i].im + expected_im;
        if (!nearly_equal(data[i].re, want_re, 1.0e-9) ||
            !nearly_equal(data[i].im, want_im, 1.0e-9)) {
            fprintf(stderr, "[FAIL] stimulus_fourier_complex: mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_posenc_complex_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.025;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] posenc_complex: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] posenc_complex: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] posenc_complex: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    size_t count = sim_field_element_count(&field.layout);
    SimComplexDouble expected_initial[384];
    for (size_t i = 0U; i < count; ++i) {
        expected_initial[i].re = 0.04 * sample_value(i % shape[1], shape[1]);
        expected_initial[i].im = -0.03 * sample_value((i + 7U) % shape[1], shape[1]);
        data[i] = expected_initial[i];
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] posenc_complex: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusPosEncConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.72;
    cfg.base_wavenumber = 1.05;
    cfg.band_growth = 1.9;
    cfg.band_count = 5U;
    cfg.kx = 1.2;
    cfg.ky = -0.65;
    cfg.omega = -0.18;
    cfg.phase = 0.29;
    cfg.time_offset = 0.11;
    cfg.rotation = 0.46;
    cfg.include_identity = true;
    cfg.use_wavevector = true;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.origin_x = -0.35;
    cfg.coord.origin_y = 0.18;
    cfg.coord.spacing_x = 0.07;
    cfg.coord.spacing_y = 0.13;
    cfg.coord.velocity_x = 0.05;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_posenc_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] posenc_complex: add operator failed\n");
        goto cleanup;
    }

    SimStimulusPosEncConfig normalized = {0};
    if (sim_stimulus_posenc_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] posenc_complex: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] posenc_complex: execute failed\n");
        goto cleanup;
    }

    data = sim_field_complex_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] posenc_complex: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double scale = normalized.scale_by_dt ? dt : 1.0;
        double sin_r = sin(normalized.rotation);
        double cos_r = cos(normalized.rotation);

        for (size_t i = 0U; i < count; ++i) {
            size_t ix = 0U;
            size_t iy = 0U;
            double re = 0.0;
            double im = 0.0;
            double x;
            double y;
            double sample_x;
            double sample_y;
            double rotated_re;
            double rotated_im;

            if (sim_field_index_to_xy(sim_context_field(&ctx, field_index), i, &ix, &iy) !=
                SIM_RESULT_OK) {
                fprintf(stderr, "[FAIL] posenc_complex: index_to_xy failed at %zu\n", i);
                goto cleanup;
            }

            x = normalized.coord.origin_x + (double)ix * normalized.coord.spacing_x;
            y = normalized.coord.origin_y + (double)iy * normalized.coord.spacing_y;
            sample_x = x - normalized.coord.velocity_x * t_eval;
            sample_y = y - normalized.coord.velocity_y * t_eval;

            test_posenc_eval_wavevector(&normalized, sample_x, sample_y, t_eval, &re, &im);
            re *= normalized.amplitude;
            im *= normalized.amplitude;
            rotated_re = re * cos_r - im * sin_r;
            rotated_im = re * sin_r + im * cos_r;

            if (!nearly_equal(data[i].re, expected_initial[i].re + scale * rotated_re, 1.0e-9) ||
                !nearly_equal(data[i].im, expected_initial[i].im + scale * rotated_im, 1.0e-9)) {
                fprintf(stderr, "[FAIL] posenc_complex: mismatch at %zu\n", i);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_wave_modes_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.05;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] wave_modes_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] wave_modes_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] wave_modes_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] wave_modes_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusWaveModesConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.44;
    cfg.mode_u = 4U;
    cfg.mode_v = 3U;
    cfg.extent_u = 2.8;
    cfg.extent_v = 1.7;
    cfg.wave_speed = 1.12;
    cfg.phase = -0.21;
    cfg.time_offset = 0.07;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -1.2;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.velocity_x = 0.03;

    if (sim_add_stimulus_wave_modes_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] wave_modes_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusWaveModesConfig normalized = {0};
    if (sim_stimulus_wave_modes_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] wave_modes_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] wave_modes_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] wave_modes_real: output missing\n");
        goto cleanup;
    }

    double mu = (double)normalized.mode_u / normalized.extent_u;
    double omega = normalized.wave_speed * M_PI * mu;
    double carrier = -omega * normalized.time_offset + normalized.phase;
    double carrier_re = normalized.amplitude * cos(carrier);

    for (size_t i = 0U; i < shape[0]; ++i) {
        double x = normalized.coord.origin_x + (double)i * normalized.coord.spacing_x;
        double sample_x = x - normalized.coord.velocity_x * normalized.time_offset;
        double spatial =
            sin(M_PI * (double)normalized.mode_u * (sample_x / normalized.extent_u + 0.5));
        double expected = carrier_re * spatial;
        if (!nearly_equal(data[i], expected, 1.0e-9)) {
            fprintf(stderr, "[FAIL] wave_modes_real: mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_chladni_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chladni_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chladni_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] chladni_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chladni_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, 0.02f);

    SimStimulusChladniConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.52;
    cfg.mode_x = 5U;
    cfg.mode_y = 8U;
    cfg.plate_width = 2.6;
    cfg.plate_height = 1.9;
    cfg.mix = 0.81;
    cfg.line_width = 0.14;
    cfg.omega = -0.23;
    cfg.phase = 0.27;
    cfg.time_offset = 0.06;
    cfg.rotation = 0.41;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.29;
    cfg.coord.origin_x = -1.5;
    cfg.coord.origin_y = -0.75;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.04;
    cfg.coord.velocity_y = -0.03;

    if (sim_add_stimulus_chladni_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chladni_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusChladniConfig normalized = {0};
    if (sim_stimulus_chladni_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chladni_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chladni_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] chladni_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_chladni_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_chladni_eval(&normalized, u, v, normalized.time_offset, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] chladni_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_moire_complex_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.03;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] moire_complex: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] moire_complex: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] moire_complex: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    size_t count = sim_field_element_count(&field.layout);
    SimComplexDouble expected_initial[384];
    for (size_t i = 0U; i < count; ++i) {
        expected_initial[i].re = 0.05 * sample_value(i % shape[1], shape[1]);
        expected_initial[i].im = -0.04 * sample_value((i + 13U) % shape[1], shape[1]);
        data[i] = expected_initial[i];
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] moire_complex: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusMoireConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.63;
    cfg.k1x = 1.72;
    cfg.k1y = 0.46;
    cfg.k2x = 1.86;
    cfg.k2y = 0.52;
    cfg.omega_a = 0.24;
    cfg.omega_b = -0.19;
    cfg.phase_a = -0.11;
    cfg.phase_b = 0.32;
    cfg.time_offset = 0.08;
    cfg.rotation = 0.37;
    cfg.use_wavevectors = true;
    cfg.scale_by_dt = true;
    cfg.coord.origin_x = -1.35;
    cfg.coord.origin_y = 0.42;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.07;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_moire_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] moire_complex: add operator failed\n");
        goto cleanup;
    }

    SimStimulusMoireConfig normalized = {0};
    if (sim_stimulus_moire_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] moire_complex: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] moire_complex: execute failed\n");
        goto cleanup;
    }

    data = sim_field_complex_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] moire_complex: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double scale = normalized.scale_by_dt ? dt : 1.0;
        double sin_r = sin(normalized.rotation);
        double cos_r = cos(normalized.rotation);

        for (size_t i = 0U; i < count; ++i) {
            size_t ix = 0U;
            size_t iy = 0U;
            double re = 0.0;
            double im = 0.0;
            double x;
            double y;
            double rotated_re;
            double rotated_im;

            if (sim_field_index_to_xy(sim_context_field(&ctx, field_index), i, &ix, &iy) !=
                SIM_RESULT_OK) {
                fprintf(stderr, "[FAIL] moire_complex: index_to_xy failed at %zu\n", i);
                goto cleanup;
            }

            x = normalized.coord.origin_x + (double)ix * normalized.coord.spacing_x;
            y = normalized.coord.origin_y + (double)iy * normalized.coord.spacing_y;

            test_moire_eval(&normalized, x, y, t_eval, &re, &im);
            re *= normalized.amplitude;
            im *= normalized.amplitude;
            rotated_re = re * cos_r - im * sin_r;
            rotated_im = re * sin_r + im * cos_r;

            if (!nearly_equal(data[i].re, expected_initial[i].re + scale * rotated_re, 1.0e-9) ||
                !nearly_equal(data[i].im, expected_initial[i].im + scale * rotated_im, 1.0e-9)) {
                fprintf(stderr, "[FAIL] moire_complex: mismatch at %zu\n", i);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_zone_plate_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] zone_plate_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] zone_plate_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] zone_plate_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] zone_plate_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, 0.025f);

    SimStimulusZonePlateConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.58;
    cfg.radial_chirp = 16.5;
    cfg.scale_u = 0.81;
    cfg.scale_v = 1.09;
    cfg.aperture_u = 1.24;
    cfg.aperture_v = 0.93;
    cfg.center_u = -0.14;
    cfg.center_v = 0.17;
    cfg.velocity_u = 0.05;
    cfg.velocity_v = -0.04;
    cfg.orientation = 0.34;
    cfg.orientation_rate = -0.12;
    cfg.omega = 0.22;
    cfg.phase = -0.18;
    cfg.time_offset = 0.09;
    cfg.rotation = 0.29;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.38;
    cfg.coord.origin_x = -1.45;
    cfg.coord.origin_y = -0.62;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.04;
    cfg.coord.velocity_y = -0.03;

    if (sim_add_stimulus_zone_plate_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] zone_plate_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusZonePlateConfig normalized = {0};
    if (sim_stimulus_zone_plate_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] zone_plate_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] zone_plate_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] zone_plate_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_zone_plate_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_zone_plate_eval(&normalized, u, v, normalized.time_offset, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] zone_plate_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_traveling_wave_packet_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.025;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusTravelingWavePacketConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.61;
    cfg.sigma_u = 0.73;
    cfg.sigma_v = 1.17;
    cfg.center_u = -0.15;
    cfg.center_v = 0.12;
    cfg.velocity_u = 0.08;
    cfg.velocity_v = -0.03;
    cfg.orientation = 0.27;
    cfg.orientation_rate = 0.16;
    cfg.carrier_u = 3.45;
    cfg.carrier_v = -0.92;
    cfg.omega = 0.31;
    cfg.phase = -0.14;
    cfg.time_offset = 0.07;
    cfg.rotation = 0.33;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.44;
    cfg.coord.origin_x = -1.52;
    cfg.coord.origin_y = -0.36;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.085;
    cfg.coord.velocity_x = 0.035;
    cfg.coord.velocity_y = -0.025;

    if (sim_add_stimulus_traveling_wave_packet_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusTravelingWavePacketConfig normalized = {0};
    if (sim_stimulus_traveling_wave_packet_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] traveling_wave_packet_real: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double scale = normalized.scale_by_dt ? dt : 1.0;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double u = 0.0;
                double v = 0.0;
                double re = 0.0;
                double im = 0.0;

                test_traveling_wave_packet_map_coord(&normalized, x, y, t_eval, &u, &v);
                test_traveling_wave_packet_eval(&normalized, u, v, t_eval, &re, &im);

                if (!nearly_equal(data[idx], scale * normalized.amplitude * re, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] traveling_wave_packet_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_optical_vortex_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] optical_vortex_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] optical_vortex_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] optical_vortex_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] optical_vortex_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusOpticalVortexConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.54;
    cfg.charge = 4;
    cfg.waist_x = 0.79;
    cfg.waist_y = 0.58;
    cfg.center_u = 0.13;
    cfg.center_v = -0.17;
    cfg.velocity_u = 0.05;
    cfg.velocity_v = -0.04;
    cfg.orientation = 0.26;
    cfg.orientation_rate = -0.11;
    cfg.omega = 0.23;
    cfg.phase = -0.27;
    cfg.time_offset = 0.09;
    cfg.rotation = 0.18;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.39;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.52;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_optical_vortex_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] optical_vortex_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusOpticalVortexConfig normalized = {0};
    if (sim_stimulus_optical_vortex_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] optical_vortex_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] optical_vortex_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] optical_vortex_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_optical_vortex_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_optical_vortex_eval(&normalized, u, v, normalized.time_offset, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] optical_vortex_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_cylindrical_wave_emitter_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusCylindricalWaveEmitterConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.57;
    cfg.radial_wavenumber = 7.2;
    cfg.attenuation = 0.19;
    cfg.softening_radius = 0.14;
    cfg.center_u = -0.11;
    cfg.center_v = 0.16;
    cfg.velocity_u = 0.07;
    cfg.velocity_v = -0.05;
    cfg.omega = 0.24;
    cfg.phase = -0.13;
    cfg.time_offset = 0.05;
    cfg.rotation = 0.23;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.41;
    cfg.coord.origin_x = -1.43;
    cfg.coord.origin_y = -0.34;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_cylindrical_wave_emitter_operator(&ctx, &cfg, &op_index) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusCylindricalWaveEmitterConfig normalized = {0};
    if (sim_stimulus_cylindrical_wave_emitter_config(&ctx, op_index, &normalized) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_cylindrical_wave_emitter_map_coord(&normalized, x, y, normalized.time_offset, &u,
                                                    &v);
            test_cylindrical_wave_emitter_eval(&normalized, 2U, u, v, normalized.time_offset, &re,
                                               &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] cylindrical_wave_emitter_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_laplace_beltrami_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusLaplaceBeltramiConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.46;
    cfg.manifold = SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER;
    cfg.mode_u = -4;
    cfg.mode_v = 3;
    cfg.extent_u = 2.7;
    cfg.extent_v = 1.8;
    cfg.omega = 0.21;
    cfg.phase = -0.17;
    cfg.time_offset = 0.06;
    cfg.rotation = 0.24;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.36;
    cfg.coord.origin_x = -1.48;
    cfg.coord.origin_y = -0.37;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_laplace_beltrami_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusLaplaceBeltramiConfig normalized = {0};
    if (sim_stimulus_laplace_beltrami_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] laplace_beltrami_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_laplace_beltrami_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_laplace_beltrami_eval(&normalized, u, v, normalized.time_offset, true, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] laplace_beltrami_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_digamma_square_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] digamma_square_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] digamma_square_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    {
        double *data = sim_field_real_data(&field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] digamma_square_real: data init failed\n");
            sim_field_destroy(&field);
            sim_context_destroy(&ctx);
            return false;
        }
        memset(data, 0, sim_field_bytes(&field));
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] digamma_square_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    SimStimulusDigammaSquareConfig cfg = {0};
    cfg.amplitude = 0.33;
    cfg.wavenumber = 1.14;
    cfg.omega = 0.19;
    cfg.phase = -0.24;
    cfg.harmonics = 3.8;
    cfg.a = 0.29;
    cfg.backend = SIM_DIGAMMA_BACKEND_12_TAIL;
    cfg.tolerance = 1.0e-12;
    cfg.shape = SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH;
    cfg.rotation = 0.0;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.combine = SIM_STIMULUS_SEPARABLE_ADD;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.39;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.time_offset = 0.07;
    cfg.field_index = field_index;

    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] digamma_square_real: add operator failed\n");
        goto cleanup;
    }

    {
        SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
        if (op == NULL) {
            fprintf(stderr, "[FAIL] digamma_square_real: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(op);
        if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
            info.representation.requires_complex_input) {
            fprintf(stderr, "[FAIL] digamma_square_real: metadata mismatch\n");
            goto cleanup;
        }
    }

    SimStimulusDigammaSquareConfig normalized = {0};
    if (sim_stimulus_digamma_square_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] digamma_square_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] digamma_square_real: execute failed\n");
        goto cleanup;
    }

    {
        const double *data = sim_field_real_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "[FAIL] digamma_square_real: output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * normalized.time_offset;
                double sample_y = y - normalized.coord.velocity_y * normalized.time_offset;
                double phase_x = normalized.wavenumber * sample_x + normalized.phase -
                                 normalized.omega * normalized.time_offset;
                double phase_y = normalized.wavenumber * sample_y + normalized.phase -
                                 normalized.omega * normalized.time_offset;
                double base_x;
                double base_y;
                double expected;

                if (fabs(normalized.a - 0.25) < 1.0e-6) {
                    base_x = sim_digamma_square_base_real(normalized.amplitude,
                                                          normalized.harmonics * cos(phase_x),
                                                          normalized.backend, normalized.tolerance);
                    base_y = sim_digamma_square_base_real(normalized.amplitude,
                                                          normalized.harmonics * cos(phase_y),
                                                          normalized.backend, normalized.tolerance);
                } else {
                    base_x = sim_digamma_square_base_deformed_real(
                        normalized.amplitude, normalized.a, normalized.harmonics * cos(phase_x),
                        normalized.backend, normalized.tolerance);
                    base_y = sim_digamma_square_base_deformed_real(
                        normalized.amplitude, normalized.a, normalized.harmonics * cos(phase_y),
                        normalized.backend, normalized.tolerance);
                }

                expected = base_x * sin(phase_x) + base_y * sin(phase_y);
                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] digamma_square_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_rd_seed_real_case(void) {
    SimContext ctx_real = {0};
    SimContext ctx_complex = {0};
    SimField real_field = {0};
    SimField complex_field = {0};
    size_t shape[2] = {4U, 96U};
    size_t real_field_index = 0U;
    size_t complex_field_index = 0U;
    size_t real_op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx_real) != SIM_RESULT_OK ||
        sim_context_init(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rd_seed_real: context init failed\n");
        goto cleanup;
    }

    if (sim_field_init(&real_field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&complex_field, 2U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rd_seed_real: field init failed\n");
        goto cleanup;
    }

    {
        double *real_data = sim_field_real_data(&real_field);
        SimComplexDouble *complex_data = sim_field_complex_data(&complex_field);
        if (real_data == NULL || complex_data == NULL) {
            fprintf(stderr, "[FAIL] rd_seed_real: data init failed\n");
            goto cleanup;
        }
        memset(real_data, 0, sim_field_bytes(&real_field));
        memset(complex_data, 0, sim_field_bytes(&complex_field));
    }

    if (sim_context_add_field(&ctx_real, &real_field, &real_field_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_complex, &complex_field, &complex_field_index) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rd_seed_real: add field failed\n");
        goto cleanup;
    }

    SimStimulusRDSeedConfig cfg = {0};
    cfg.amplitude = 0.68;
    cfg.bias = -0.11;
    cfg.scale = 8.3;
    cfg.threshold = 0.51;
    cfg.sharpness = 16.0;
    cfg.seed_count = 26U;
    cfg.seed = 29U;
    cfg.mode = SIM_STIMULUS_RD_SEED_LABYRINTH;
    cfg.omega = 0.42;
    cfg.phase = -0.18;
    cfg.rotation = 0.0;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.37;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.39;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    SimStimulusRDSeedConfig cfg_real = cfg;
    SimStimulusRDSeedConfig cfg_complex = cfg;
    cfg_real.field_index = real_field_index;
    cfg_complex.field_index = complex_field_index;

    if (sim_add_stimulus_rd_seed_operator(&ctx_real, &cfg_real, &real_op_index) != SIM_RESULT_OK ||
        sim_add_stimulus_rd_seed_operator(&ctx_complex, &cfg_complex, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rd_seed_real: add operator failed\n");
        goto cleanup;
    }

    {
        SimOperator *real_op = sim_operator_registry_get(&ctx_real.world.operators, real_op_index);
        if (real_op == NULL) {
            fprintf(stderr, "[FAIL] rd_seed_real: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(real_op);
        if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
            info.representation.requires_complex_input) {
            fprintf(stderr, "[FAIL] rd_seed_real: metadata mismatch\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx_real) != SIM_RESULT_OK ||
        sim_context_execute(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rd_seed_real: execute failed\n");
        goto cleanup;
    }

    {
        const double *real_data =
            sim_field_real_data_const(sim_context_field(&ctx_real, real_field_index));
        const SimComplexDouble *complex_data =
            sim_field_complex_data_const(sim_context_field(&ctx_complex, complex_field_index));
        size_t count = shape[0] * shape[1];
        if (real_data == NULL || complex_data == NULL) {
            fprintf(stderr, "[FAIL] rd_seed_real: output missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < count; ++i) {
            if (!nearly_equal(real_data[i], complex_data[i].re, 1.0e-9) ||
                !nearly_equal(complex_data[i].im, 0.0, 1.0e-9)) {
                fprintf(stderr, "[FAIL] rd_seed_real: projected mismatch at %zu\n", i);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_real);
    sim_context_destroy(&ctx_complex);
    return ok;
}

static bool run_checkerboard_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] checkerboard_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] checkerboard_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] checkerboard_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] checkerboard_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusCheckerboardConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.31;
    cfg.period_x = 1.15;
    cfg.period_y = 0.72;
    cfg.phase = 1.1;
    cfg.complex_phase = 0.23;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.37;
    cfg.coord.origin_x = -1.48;
    cfg.coord.origin_y = -0.42;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_checkerboard_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] checkerboard_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusCheckerboardConfig normalized = {0};
    if (sim_stimulus_checkerboard_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] checkerboard_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] checkerboard_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] checkerboard_real: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = sim_context_time(&ctx);
        double s = sin(normalized.coord.angle);
        double c = cos(normalized.coord.angle);
        int64_t cell_y = (normalized.period_y > 0.0) ? (int64_t)floor(normalized.phase) : 0LL;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * c + sample_y * s;
                int64_t cell_x = (int64_t)floor(u / normalized.period_x + normalized.phase);
                double expected = (((cell_x + cell_y) & 1LL) == 0LL ? normalized.amplitude
                                                                    : -normalized.amplitude);

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] checkerboard_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_lissajous_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lissajous_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lissajous_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] lissajous_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lissajous_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusLissajousConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.41;
    cfg.wavenumber_x = 3.0;
    cfg.wavenumber_y = 2.0;
    cfg.omega_x = 0.18;
    cfg.omega_y = -0.07;
    cfg.phase_x = 0.25;
    cfg.phase_y = 1.1;
    cfg.coupling = 0.85;
    cfg.bias = 0.08;
    cfg.line_width = 0.18;
    cfg.time_offset = 0.06;
    cfg.rotation = 0.31;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.39;
    cfg.coord.origin_x = -1.42;
    cfg.coord.origin_y = -0.37;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_lissajous_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lissajous_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusLissajousConfig normalized = {0};
    if (sim_stimulus_lissajous_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lissajous_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] lissajous_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] lissajous_real: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double s = sin(normalized.coord.angle);
        double c = cos(normalized.coord.angle);
        double width = normalized.line_width;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * c + sample_y * s;
                double theta_x =
                    normalized.wavenumber_x * u - normalized.omega_x * t_eval + normalized.phase_x;
                double theta_y =
                    normalized.wavenumber_y * u - normalized.omega_y * t_eval + normalized.phase_y;
                double delta = sin(theta_x) - normalized.coupling * sin(theta_y) - normalized.bias;
                double band = exp(-0.5 * (delta * delta) / (width * width));
                double carrier = 0.5 * (theta_x + theta_y);
                double expected = normalized.amplitude * band * cos(carrier);

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] lissajous_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_log_polar_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] log_polar_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] log_polar_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] log_polar_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] log_polar_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusLogPolarConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.39;
    cfg.radial_frequency = 2.7;
    cfg.angular_frequency = -3.0;
    cfg.orientation = 0.33;
    cfg.orientation_rate = 0.15;
    cfg.omega = 0.4;
    cfg.phase = -0.2;
    cfg.radius_floor = 0.18;
    cfg.time_offset = 0.04;
    cfg.rotation = -0.28;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.37;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.41;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_log_polar_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] log_polar_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusLogPolarConfig normalized = {0};
    if (sim_stimulus_log_polar_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] log_polar_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] log_polar_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] log_polar_real: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double frame_s = sin(normalized.coord.angle);
        double frame_c = cos(normalized.coord.angle);
        double rot_t = normalized.orientation + normalized.orientation_rate * t_eval;
        double rot_s = sin(rot_t);
        double rot_c = cos(rot_t);

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * frame_c + sample_y * frame_s;
                double v = -sample_x * frame_s + sample_y * frame_c;
                double ur = u * rot_c + v * rot_s;
                double vr = -u * rot_s + v * rot_c;
                double arg =
                    normalized.radial_frequency * log(hypot(ur, vr) + normalized.radius_floor) +
                    normalized.angular_frequency * atan2(vr, ur) - normalized.omega * t_eval +
                    normalized.phase;
                double expected = normalized.amplitude * cos(arg);

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] log_polar_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_morlet_field_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] morlet_field_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] morlet_field_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] morlet_field_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] morlet_field_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusMorletFieldConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.58;
    cfg.scale_count = 5U;
    cfg.base_wavenumber = 0.95;
    cfg.scale_growth = 1.75;
    cfg.sigma_base = 0.65;
    cfg.sigma_growth = 1.4;
    cfg.center_u = 0.2;
    cfg.center_v = -0.15;
    cfg.velocity_v = 0.06;
    cfg.orientation = -0.3;
    cfg.omega = -0.17;
    cfg.phase = 0.31;
    cfg.time_offset = 0.07;
    cfg.zero_mean = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.39;
    cfg.coord.origin_x = -1.42;
    cfg.coord.origin_y = -0.38;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_morlet_field_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] morlet_field_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusMorletFieldConfig normalized = {0};
    if (sim_stimulus_morlet_field_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] morlet_field_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] morlet_field_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] morlet_field_real: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double frame_s = sin(normalized.coord.angle);
        double frame_c = cos(normalized.coord.angle);
        double orient_t = normalized.orientation + normalized.orientation_rate * t_eval;
        double orient_s = sin(orient_t);
        double orient_c = cos(orient_t);
        double center_u_t = normalized.center_u + normalized.velocity_u * t_eval;
        double center_v_t = normalized.center_v + normalized.velocity_v * t_eval;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * frame_c + sample_y * frame_s;
                double v = -sample_x * frame_s + sample_y * frame_c;
                double du = u - center_u_t;
                double dv = v - center_v_t;
                double ur = du * orient_c + dv * orient_s;
                double vr = -du * orient_s + dv * orient_c;
                double r2 = ur * ur + vr * vr;
                double re_sum = 0.0;
                double k_scale = 1.0;
                double s_scale = 1.0;

                for (unsigned int scale_i = 0U; scale_i < normalized.scale_count; ++scale_i) {
                    double k = normalized.base_wavenumber * k_scale;
                    double sigma = normalized.sigma_base * s_scale;
                    if (sigma <= 1.0e-12) {
                        sigma = 1.0;
                    }
                    double envelope = exp(-0.5 * r2 / (sigma * sigma));
                    double phase = k * ur - normalized.omega * t_eval + normalized.phase;
                    re_sum += envelope * cos(phase);
                    k_scale *= normalized.scale_growth;
                    s_scale *= normalized.sigma_growth;
                }

                if (normalized.scale_count > 0U) {
                    re_sum *= 1.0 / sqrt((double)normalized.scale_count);
                }
                if (!normalized.zero_mean) {
                    /* no correction */
                }
                double expected = normalized.amplitude * re_sum;

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] morlet_field_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_gaussian_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gaussian_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gaussian_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] gaussian_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gaussian_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusGaussianConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.37;
    cfg.sigma_x = 0.74;
    cfg.sigma_y = 0.96;
    cfg.time_offset = 0.03;
    cfg.rotation = -0.18;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.38;
    cfg.coord.center_x = 0.11;
    cfg.coord.center_y = -0.08;
    cfg.coord.velocity_x = 0.025;
    cfg.coord.velocity_y = -0.015;
    cfg.coord.origin_x = -1.46;
    cfg.coord.origin_y = -0.41;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;

    if (sim_add_stimulus_gaussian_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gaussian_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusGaussianConfig normalized = {0};
    if (sim_stimulus_gaussian_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gaussian_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gaussian_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] gaussian_real: output missing\n");
        goto cleanup;
    }

    {
        double sigma_x = (normalized.sigma_x > 1.0e-9) ? normalized.sigma_x : 1.0;
        double t_eval = normalized.time_offset;
        double s = sin(normalized.coord.angle);
        double c = cos(normalized.coord.angle);
        double center = normalized.coord.center_x + normalized.coord.velocity_x * t_eval;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * c + sample_y * s;
                double diff = u - center;
                double expected =
                    normalized.amplitude * exp(-0.5 * (diff * diff) / (sigma_x * sigma_x));

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] gaussian_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_heat_kernel_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heat_kernel_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heat_kernel_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] heat_kernel_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heat_kernel_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusHeatKernelConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.45;
    cfg.diffusivity = 0.18;
    cfg.sigma_x = 0.53;
    cfg.sigma_y = 0.77;
    cfg.time_offset = 0.06;
    cfg.rotation = 0.19;
    cfg.preserve_mass = true;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.34;
    cfg.coord.center_x = 0.08;
    cfg.coord.center_y = -0.11;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.coord.origin_x = -1.45;
    cfg.coord.origin_y = -0.39;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;

    if (sim_add_stimulus_heat_kernel_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heat_kernel_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusHeatKernelConfig normalized = {0};
    if (sim_stimulus_heat_kernel_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heat_kernel_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] heat_kernel_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] heat_kernel_real: output missing\n");
        goto cleanup;
    }

    {
        double t_eval = normalized.time_offset;
        double t_pos = (t_eval > 0.0) ? t_eval : 0.0;
        double sigma0 = (normalized.sigma_x > 1.0e-9) ? normalized.sigma_x : 1.0;
        double sigma_sq = sigma0 * sigma0 + 2.0 * normalized.diffusivity * t_pos;
        double sigma_eff = sqrt(sigma_sq);
        double norm = normalized.preserve_mass ? (sigma0 / sigma_eff) : 1.0;
        double s = sin(normalized.coord.angle);
        double c = cos(normalized.coord.angle);
        double center = normalized.coord.center_x + normalized.coord.velocity_x * t_eval;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * c + sample_y * s;
                double diff = u - center;
                double expected =
                    normalized.amplitude * norm * exp(-0.5 * (diff * diff) / sigma_sq);

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] heat_kernel_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_gabor_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gabor_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gabor_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] gabor_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gabor_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusGaborConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.33;
    cfg.wavenumber = 1.7;
    cfg.omega = 0.22;
    cfg.phase = -0.14;
    cfg.sigma_x = 0.71;
    cfg.sigma_y = 0.94;
    cfg.time_offset = 0.04;
    cfg.rotation = 0.17;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.41;
    cfg.coord.center_x = 0.09;
    cfg.coord.center_y = -0.12;
    cfg.coord.velocity_x = 0.028;
    cfg.coord.velocity_y = -0.018;
    cfg.coord.origin_x = -1.42;
    cfg.coord.origin_y = -0.37;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;

    if (sim_add_stimulus_gabor_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gabor_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusGaborConfig normalized = {0};
    if (sim_stimulus_gabor_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gabor_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] gabor_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] gabor_real: output missing\n");
        goto cleanup;
    }

    {
        double sigma_x = (normalized.sigma_x > 1.0e-9) ? normalized.sigma_x : 1.0;
        double sigma_y = (normalized.sigma_y > 1.0e-9) ? normalized.sigma_y : sigma_x;
        double inv_x = 0.5 / (sigma_x * sigma_x);
        double inv_y = 0.5 / (sigma_y * sigma_y);
        double t_eval = normalized.time_offset;
        double s = sin(normalized.coord.angle);
        double c = cos(normalized.coord.angle);
        double center_x = normalized.coord.center_x + normalized.coord.velocity_x * t_eval;
        double center_y = normalized.coord.center_y + normalized.coord.velocity_y * t_eval;

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double sample_x = x - normalized.coord.velocity_x * t_eval;
                double sample_y = y - normalized.coord.velocity_y * t_eval;
                double u = sample_x * c + sample_y * s;
                double dx = x - center_x;
                double dy = y - center_y;
                double xr = c * dx + s * dy;
                double yr = -s * dx + c * dy;
                double envelope = normalized.amplitude * exp(-(xr * xr * inv_x + yr * yr * inv_y));
                double theta =
                    normalized.wavenumber * u - normalized.omega * t_eval + normalized.phase;
                double expected = envelope * cos(theta);

                if (!nearly_equal(data[idx], expected, 1.0e-9)) {
                    fprintf(stderr, "[FAIL] gabor_real: mismatch at %zu\n", idx);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_airy_beam_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] airy_beam_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] airy_beam_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] airy_beam_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] airy_beam_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusAiryBeamConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.39;
    cfg.scale_u = 0.88;
    cfg.scale_v = 1.12;
    cfg.apodization_u = 0.11;
    cfg.apodization_v = 0.07;
    cfg.center_u = 0.09;
    cfg.center_v = -0.12;
    cfg.velocity_u = -0.05;
    cfg.velocity_v = 0.04;
    cfg.orientation = -0.24;
    cfg.orientation_rate = 0.08;
    cfg.carrier_u = -0.61;
    cfg.carrier_v = 0.36;
    cfg.omega = -0.16;
    cfg.phase = 0.22;
    cfg.time_offset = 0.03;
    cfg.rotation = -0.15;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.35;
    cfg.coord.origin_x = -1.43;
    cfg.coord.origin_y = -0.47;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_airy_beam_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] airy_beam_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusAiryBeamConfig normalized = {0};
    if (sim_stimulus_airy_beam_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] airy_beam_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] airy_beam_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] airy_beam_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_airy_beam_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_airy_beam_eval(&normalized, u, v, normalized.time_offset, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] airy_beam_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_bessel_beam_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] bessel_beam_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] bessel_beam_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] bessel_beam_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] bessel_beam_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusBesselBeamConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.43;
    cfg.order = 4;
    cfg.radial_wavenumber = 4.3;
    cfg.scale_u = 0.91;
    cfg.scale_v = 1.17;
    cfg.center_u = 0.14;
    cfg.center_v = -0.11;
    cfg.velocity_u = -0.04;
    cfg.velocity_v = 0.05;
    cfg.orientation = -0.23;
    cfg.orientation_rate = 0.09;
    cfg.omega = -0.21;
    cfg.phase = 0.29;
    cfg.time_offset = 0.03;
    cfg.rotation = -0.18;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.36;
    cfg.coord.origin_x = -1.42;
    cfg.coord.origin_y = -0.44;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_bessel_beam_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] bessel_beam_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusBesselBeamConfig normalized = {0};
    if (sim_stimulus_bessel_beam_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] bessel_beam_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] bessel_beam_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] bessel_beam_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_bessel_beam_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_bessel_beam_eval(&normalized, u, v, normalized.time_offset, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] bessel_beam_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_hermite_gaussian_beam_real_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {4U, 96U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.025;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double *data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: data init failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        data[i] = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimStimulusHermiteGaussianBeamConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.46;
    cfg.mode_u = 3U;
    cfg.mode_v = 2U;
    cfg.waist_u = 0.81;
    cfg.waist_v = 1.09;
    cfg.center_u = -0.15;
    cfg.center_v = 0.14;
    cfg.velocity_u = 0.06;
    cfg.velocity_v = -0.05;
    cfg.orientation = 0.25;
    cfg.orientation_rate = -0.12;
    cfg.carrier_u = 0.63;
    cfg.carrier_v = -0.27;
    cfg.omega = 0.21;
    cfg.phase = -0.19;
    cfg.time_offset = 0.08;
    cfg.rotation = 0.23;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.41;
    cfg.coord.origin_x = -1.5;
    cfg.coord.origin_y = -0.48;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    if (sim_add_stimulus_hermite_gaussian_beam_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: add operator failed\n");
        goto cleanup;
    }

    SimStimulusHermiteGaussianBeamConfig normalized = {0};
    if (sim_stimulus_hermite_gaussian_beam_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: config fetch failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: execute failed\n");
        goto cleanup;
    }

    data = sim_field_real_data(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: output missing\n");
        goto cleanup;
    }

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t idx = row * shape[1] + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double u = 0.0;
            double v = 0.0;
            double re = 0.0;
            double im = 0.0;

            test_hermite_gaussian_beam_map_coord(&normalized, x, y, normalized.time_offset, &u, &v);
            test_hermite_gaussian_beam_eval(&normalized, u, v, normalized.time_offset, &re, &im);

            if (!nearly_equal(data[idx], normalized.amplitude * re, 1.0e-9)) {
                fprintf(stderr, "[FAIL] hermite_gaussian_beam_real: mismatch at %zu\n", idx);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_linear_dissipative_real_matches_complex(void) {
    SimContext ctx_real = {0};
    SimContext ctx_complex = {0};
    SimField real_field = {0};
    SimField complex_field = {0};
    size_t shape[1] = {16U};
    size_t real_index = 0U;
    size_t complex_index = 0U;
    size_t real_op_index = 0U;
    const double dt = 0.03;
    const size_t steps = 3U;
    bool ok = false;

    if (sim_context_init(&ctx_real) != SIM_RESULT_OK ||
        sim_context_init(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: context init failed\n");
        goto cleanup;
    }

    if (sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&complex_field, 1U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: field init failed\n");
        goto cleanup;
    }

    double *real_data = sim_field_real_data(&real_field);
    SimComplexDouble *complex_data = sim_field_complex_data(&complex_field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        double value = sample_value(i, shape[0]);
        real_data[i] = value;
        complex_data[i].re = value;
        complex_data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx_real, &real_field, &real_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_complex, &complex_field, &complex_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: add field failed\n");
        goto cleanup;
    }

    sim_context_set_timestep(&ctx_real, (float)dt);
    sim_context_set_timestep(&ctx_complex, (float)dt);

    LinearDissipativeOperatorConfig cfg_real = {
        .field_index = real_index, .viscosity = 0.45, .alpha = 1.5, .spacing = 0.2};
    LinearDissipativeOperatorConfig cfg_complex = {
        .field_index = complex_index, .viscosity = 0.45, .alpha = 1.5, .spacing = 0.2};

    if (sim_add_linear_dissipative_operator(&ctx_real, &cfg_real, &real_op_index) !=
            SIM_RESULT_OK ||
        sim_add_linear_dissipative_operator(&ctx_complex, &cfg_complex, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: add operator failed\n");
        goto cleanup;
    }

    SimOperator *real_op = sim_operator_registry_get(&ctx_real.world.operators, real_op_index);
    if (real_op == NULL) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: operator missing\n");
        goto cleanup;
    }
    SimOperatorInfo info = sim_operator_info(real_op);
    if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
        info.representation.requires_complex_input) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: metadata mismatch\n");
        goto cleanup;
    }

    for (size_t step = 0U; step < steps; ++step) {
        if (sim_context_execute(&ctx_real) != SIM_RESULT_OK ||
            sim_context_execute(&ctx_complex) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] linear_dissipative_real: execute failed\n");
            goto cleanup;
        }
    }

    real_data = sim_field_real_data(sim_context_field(&ctx_real, real_index));
    complex_data = sim_field_complex_data(sim_context_field(&ctx_complex, complex_index));
    if (real_data == NULL || complex_data == NULL) {
        fprintf(stderr, "[FAIL] linear_dissipative_real: output missing\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < shape[0]; ++i) {
        if (!nearly_equal(real_data[i], complex_data[i].re, 1.0e-7) ||
            !nearly_equal(complex_data[i].im, 0.0, 1.0e-7)) {
            fprintf(stderr, "[FAIL] linear_dissipative_real: mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_real);
    sim_context_destroy(&ctx_complex);
    return ok;
}

static bool run_dispersion_real_matches_complex_projection(void) {
    SimContext ctx_real = {0};
    SimContext ctx_complex = {0};
    SimField real_field = {0};
    SimField complex_field = {0};
    size_t shape[1] = {16U};
    size_t real_index = 0U;
    size_t complex_index = 0U;
    size_t real_op_index = 0U;
    const double dt = 0.05;
    bool ok = false;

    if (sim_context_init(&ctx_real) != SIM_RESULT_OK ||
        sim_context_init(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real: context init failed\n");
        goto cleanup;
    }

    if (sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&complex_field, 1U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real: field init failed\n");
        goto cleanup;
    }

    double *real_data = sim_field_real_data(&real_field);
    SimComplexDouble *complex_data = sim_field_complex_data(&complex_field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        double value = sample_value(i, shape[0]);
        real_data[i] = value;
        complex_data[i].re = value;
        complex_data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx_real, &real_field, &real_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_complex, &complex_field, &complex_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real: add field failed\n");
        goto cleanup;
    }

    sim_context_set_timestep(&ctx_real, (float)dt);
    sim_context_set_timestep(&ctx_complex, (float)dt);

    DispersionOperatorConfig cfg_real = {.field_index = real_index,
                                         .coefficient = 0.8,
                                         .order = 2.0,
                                         .spacing = 0.15,
                                         .reference_k = 0.0};
    DispersionOperatorConfig cfg_complex = {.field_index = complex_index,
                                            .coefficient = 0.8,
                                            .order = 2.0,
                                            .spacing = 0.15,
                                            .reference_k = 0.0};

    if (sim_add_dispersion_operator(&ctx_real, &cfg_real, &real_op_index) != SIM_RESULT_OK ||
        sim_add_dispersion_operator(&ctx_complex, &cfg_complex, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real: add operator failed\n");
        goto cleanup;
    }

    SimOperator *real_op = sim_operator_registry_get(&ctx_real.world.operators, real_op_index);
    if (real_op == NULL) {
        fprintf(stderr, "[FAIL] dispersion_real: operator missing\n");
        goto cleanup;
    }
    SimOperatorInfo info = sim_operator_info(real_op);
    if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
        info.representation.requires_complex_input) {
        fprintf(stderr, "[FAIL] dispersion_real: metadata mismatch\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx_real) != SIM_RESULT_OK ||
        sim_context_execute(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real: execute failed\n");
        goto cleanup;
    }

    real_data = sim_field_real_data(sim_context_field(&ctx_real, real_index));
    complex_data = sim_field_complex_data(sim_context_field(&ctx_complex, complex_index));
    if (real_data == NULL || complex_data == NULL) {
        fprintf(stderr, "[FAIL] dispersion_real: output missing\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < shape[0]; ++i) {
        if (!nearly_equal(real_data[i], complex_data[i].re, 1.0e-7)) {
            fprintf(stderr, "[FAIL] dispersion_real: projected mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_real);
    sim_context_destroy(&ctx_complex);
    return ok;
}

static bool run_linear_spectral_fusion_real_matches_complex_projection(void) {
    SimContext ctx_real = {0};
    SimContext ctx_complex = {0};
    SimField real_field = {0};
    SimField complex_field = {0};
    size_t shape[1] = {16U};
    size_t real_index = 0U;
    size_t complex_index = 0U;
    const double dt = 0.02;
    bool ok = false;

    if (sim_context_init(&ctx_real) != SIM_RESULT_OK ||
        sim_context_init(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_spectral_real: context init failed\n");
        goto cleanup;
    }

    if (sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&complex_field, 1U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_spectral_real: field init failed\n");
        goto cleanup;
    }

    double *real_data = sim_field_real_data(&real_field);
    SimComplexDouble *complex_data = sim_field_complex_data(&complex_field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        double value = sample_value(i, shape[0]);
        real_data[i] = value;
        complex_data[i].re = value;
        complex_data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx_real, &real_field, &real_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_complex, &complex_field, &complex_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_spectral_real: add field failed\n");
        goto cleanup;
    }

    sim_context_set_timestep(&ctx_real, (float)dt);
    sim_context_set_timestep(&ctx_complex, (float)dt);

    LinearSpectralFusionOperatorConfig cfg_real = {.field_index = real_index,
                                                   .viscosity = 0.25,
                                                   .alpha = 2.0,
                                                   .dissipation_spacing = 0.15,
                                                   .dispersion_coefficient = 0.9,
                                                   .dispersion_order = 2.0,
                                                   .dispersion_reference_k = 0.0,
                                                   .dispersion_spacing = 0.15,
                                                   .phase_rate = 0.35};
    LinearSpectralFusionOperatorConfig cfg_complex = cfg_real;
    cfg_complex.field_index = complex_index;

    if (sim_add_linear_spectral_fusion_operator(&ctx_real, &cfg_real, NULL) != SIM_RESULT_OK ||
        sim_add_linear_spectral_fusion_operator(&ctx_complex, &cfg_complex, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_spectral_real: add operator failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx_real) != SIM_RESULT_OK ||
        sim_context_execute(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] linear_spectral_real: execute failed\n");
        goto cleanup;
    }

    real_data = sim_field_real_data(sim_context_field(&ctx_real, real_index));
    complex_data = sim_field_complex_data(sim_context_field(&ctx_complex, complex_index));
    if (real_data == NULL || complex_data == NULL) {
        fprintf(stderr, "[FAIL] linear_spectral_real: output missing\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < shape[0]; ++i) {
        if (!nearly_equal(real_data[i], complex_data[i].re, 1.0e-7)) {
            fprintf(stderr, "[FAIL] linear_spectral_real: projected mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_real);
    sim_context_destroy(&ctx_complex);
    return ok;
}

static bool run_chaos_map_real_matches_complex_projection(void) {
    SimContext ctx_real = {0};
    SimContext ctx_complex = {0};
    SimField real_input = {0};
    SimField real_output = {0};
    SimField complex_input = {0};
    SimField complex_output = {0};
    size_t shape[1] = {6U};
    size_t real_input_index = 0U;
    size_t real_output_index = 0U;
    size_t complex_input_index = 0U;
    size_t complex_output_index = 0U;
    size_t real_op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx_real) != SIM_RESULT_OK ||
        sim_context_init(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chaos_map_real: context init failed\n");
        goto cleanup;
    }

    if (sim_field_init(&real_input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&real_output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&complex_input, 1U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init(&complex_output, 1U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chaos_map_real: field init failed\n");
        goto cleanup;
    }

    double *real_in_data = sim_field_real_data(&real_input);
    double *real_out_data = sim_field_real_data(&real_output);
    SimComplexDouble *complex_in_data = sim_field_complex_data(&complex_input);
    SimComplexDouble *complex_out_data = sim_field_complex_data(&complex_output);
    for (size_t i = 0U; i < shape[0]; ++i) {
        double value = -0.6 + 1.2 * ((double)i / (double)(shape[0] - 1U));
        real_in_data[i] = value;
        real_out_data[i] = 0.0;
        complex_in_data[i].re = value;
        complex_in_data[i].im = 0.0;
        complex_out_data[i].re = 0.0;
        complex_out_data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx_real, &real_input, &real_input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_real, &real_output, &real_output_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_complex, &complex_input, &complex_input_index) !=
            SIM_RESULT_OK ||
        sim_context_add_field(&ctx_complex, &complex_output, &complex_output_index) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chaos_map_real: add field failed\n");
        goto cleanup;
    }

    SimChaosMapOperatorConfig cfg_real = {0};
    cfg_real.input_field = real_input_index;
    cfg_real.output_field = real_output_index;
    cfg_real.map_type = SIM_CHAOS_MAP_STANDARD;
    cfg_real.kick_mode = SIM_CHAOS_KICK_DRIFT;
    cfg_real.iterations_per_step = 2U;
    cfg_real.blend = 1.0;
    cfg_real.k = 0.75;
    cfg_real.angle_scale = 1.1;
    cfg_real.k_field = SIZE_MAX;
    cfg_real.u_field = SIZE_MAX;
    cfg_real.a_field = SIZE_MAX;
    cfg_real.b_field = SIZE_MAX;
    cfg_real.c_field = SIZE_MAX;
    cfg_real.d_field = SIZE_MAX;

    SimChaosMapOperatorConfig cfg_complex = cfg_real;
    cfg_complex.input_field = complex_input_index;
    cfg_complex.output_field = complex_output_index;

    if (sim_add_chaos_map_operator(&ctx_real, &cfg_real, &real_op_index) != SIM_RESULT_OK ||
        sim_add_chaos_map_operator(&ctx_complex, &cfg_complex, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chaos_map_real: add operator failed\n");
        goto cleanup;
    }

    SimOperator *real_op = sim_operator_registry_get(&ctx_real.world.operators, real_op_index);
    if (real_op == NULL) {
        fprintf(stderr, "[FAIL] chaos_map_real: operator missing\n");
        goto cleanup;
    }
    SimOperatorInfo info = sim_operator_info(real_op);
    if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
        info.representation.requires_complex_input) {
        fprintf(stderr, "[FAIL] chaos_map_real: metadata mismatch\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx_real) != SIM_RESULT_OK ||
        sim_context_execute(&ctx_complex) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] chaos_map_real: execute failed\n");
        goto cleanup;
    }

    real_out_data = sim_field_real_data(sim_context_field(&ctx_real, real_output_index));
    complex_out_data =
        sim_field_complex_data(sim_context_field(&ctx_complex, complex_output_index));
    if (real_out_data == NULL || complex_out_data == NULL) {
        fprintf(stderr, "[FAIL] chaos_map_real: output missing\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < shape[0]; ++i) {
        if (!nearly_equal(real_out_data[i], complex_out_data[i].re, 1.0e-9)) {
            fprintf(stderr, "[FAIL] chaos_map_real: projected mismatch at %zu\n", i);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_real);
    sim_context_destroy(&ctx_complex);
    return ok;
}

int main(void) {
    if (!run_phase_rotate_real_case()) {
        return 1;
    }
    if (!run_phase_rotate_real_constrained_spectral_case()) {
        return 1;
    }
    if (!run_phase_rotate_complex_case()) {
        return 1;
    }
    if (!run_phase_rotate_imag_zero_constraint_preserved_when_inactive()) {
        return 1;
    }
    if (!run_phase_rotate_imag_zero_constraint_demotes_with_phase()) {
        return 1;
    }
    if (!run_stimulus_fourier_real_case()) {
        return 1;
    }
    if (!run_stimulus_fourier_complex_case()) {
        return 1;
    }
    if (!run_posenc_complex_case()) {
        return 1;
    }
    if (!run_random_fourier_complex_case()) {
        return 1;
    }
    if (!run_wave_modes_real_case()) {
        return 1;
    }
    if (!run_chladni_real_case()) {
        return 1;
    }
    if (!run_moire_complex_case()) {
        return 1;
    }
    if (!run_zone_plate_real_case()) {
        return 1;
    }
    if (!run_traveling_wave_packet_real_case()) {
        return 1;
    }
    if (!run_optical_vortex_real_case()) {
        return 1;
    }
    if (!run_cylindrical_wave_emitter_real_case()) {
        return 1;
    }
    if (!run_laplace_beltrami_real_case()) {
        return 1;
    }
    if (!run_digamma_square_real_case()) {
        return 1;
    }
    if (!run_rd_seed_real_case()) {
        return 1;
    }
    if (!run_checkerboard_real_case()) {
        return 1;
    }
    if (!run_lissajous_real_case()) {
        return 1;
    }
    if (!run_log_polar_real_case()) {
        return 1;
    }
    if (!run_morlet_field_real_case()) {
        return 1;
    }
    if (!run_gaussian_real_case()) {
        return 1;
    }
    if (!run_heat_kernel_real_case()) {
        return 1;
    }
    if (!run_gabor_real_case()) {
        return 1;
    }
    if (!run_airy_beam_real_case()) {
        return 1;
    }
    if (!run_bessel_beam_real_case()) {
        return 1;
    }
    if (!run_hermite_gaussian_beam_real_case()) {
        return 1;
    }
    if (!run_linear_dissipative_real_matches_complex()) {
        return 1;
    }
    if (!run_dispersion_real_matches_complex_projection()) {
        return 1;
    }
    if (!run_linear_spectral_fusion_real_matches_complex_projection()) {
        return 1;
    }
    if (!run_chaos_map_real_matches_complex_projection()) {
        return 1;
    }
    return 0;
}
