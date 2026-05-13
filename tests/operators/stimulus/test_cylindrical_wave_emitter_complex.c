/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double kTol2D = 1.0e-8;
static const double kTol1D = 1.0e-8;
static const size_t kShape2DWide[] = {4U, 96U};
static const size_t kShape1DWide[] = {96U};

static void cylindrical_wave_emitter_map_coord(const SimStimulusCylindricalWaveEmitterConfig *cfg,
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

static void cylindrical_wave_emitter_eval(const SimStimulusCylindricalWaveEmitterConfig *cfg,
                                          size_t rank, double u, double v, double t, double *out_re,
                                          double *out_im) {
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

static size_t element_count(size_t rank, const size_t *shape) {
    size_t count = 1U;
    for (size_t i = 0U; i < rank; ++i) {
        count *= shape[i];
    }
    return count;
}

static int run_case(const SimStimulusCylindricalWaveEmitterConfig *cfg_input, size_t rank,
                    const size_t *shape, double dt, int steps, double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    SimField field = {0};
    if (sim_field_init(&field, rank, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimComplexDouble *initial = sim_field_complex_data(&field);
        if (initial == NULL) {
            fprintf(stderr, "FAIL: sim_field_complex_data\n");
            sim_field_destroy(&field);
            sim_context_destroy(&ctx);
            return 0;
        }
        memset(initial, 0, sim_field_bytes(&field));
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCylindricalWaveEmitterConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_cylindrical_wave_emitter_operator(&ctx, &cfg, &op_index) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stimulus_cylindrical_wave_emitter_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusCylindricalWaveEmitterConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_cylindrical_wave_emitter_config(&ctx, op_index, &normalized) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_stimulus_cylindrical_wave_emitter_config\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: operator lookup\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = element_count(rank, shape);
    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (expected == NULL) {
        fprintf(stderr, "FAIL: out of memory\n");
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
            double out_re = 0.0;
            double out_im = 0.0;

            if (rank == 1U) {
                x = normalized.coord.origin_x + (double)i * normalized.coord.spacing_x;
                y = normalized.coord.origin_y;
            } else {
                size_t row = i / shape[1];
                size_t col = i % shape[1];
                x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            }

            cylindrical_wave_emitter_map_coord(&normalized, x, y, t, &u, &v);
            cylindrical_wave_emitter_eval(&normalized, rank, u, v, t, &re, &im);

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
    {
        SimStimulusCylindricalWaveEmitterConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.amplitude = 0.61;
        cfg.radial_wavenumber = 6.8;
        cfg.attenuation = 0.24;
        cfg.softening_radius = 0.11;
        cfg.center_u = -0.17;
        cfg.center_v = 0.12;
        cfg.velocity_u = 0.08;
        cfg.velocity_v = -0.05;
        cfg.omega = 0.21;
        cfg.phase = 0.14;
        cfg.time_offset = 0.03;
        cfg.rotation = -0.19;
        cfg.scale_by_dt = true;
        cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
        cfg.coord.angle = 0.37;
        cfg.coord.origin_x = -0.55;
        cfg.coord.origin_y = -0.42;
        cfg.coord.spacing_x = 0.05;
        cfg.coord.spacing_y = 0.19;
        cfg.coord.velocity_x = 0.03;
        cfg.coord.velocity_y = -0.02;
        if (!run_case(&cfg, 2U, kShape2DWide, 0.06, 3, kTol2D)) {
            return 1;
        }
    }

    {
        SimStimulusCylindricalWaveEmitterConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.amplitude = 0.48;
        cfg.radial_wavenumber = 4.6;
        cfg.attenuation = 0.12;
        cfg.softening_radius = 0.07;
        cfg.center_u = -0.09;
        cfg.velocity_u = 0.16;
        cfg.omega = -0.12;
        cfg.phase = -0.2;
        cfg.rotation = 0.17;
        cfg.scale_by_dt = false;
        cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
        cfg.coord.axis = SIM_STIMULUS_AXIS_X;
        cfg.coord.origin_x = -0.7;
        cfg.coord.origin_y = 0.12;
        cfg.coord.spacing_x = 0.04;
        cfg.coord.velocity_x = -0.03;
        cfg.coord.velocity_y = 0.02;
        if (!run_case(&cfg, 1U, kShape1DWide, 0.05, 4, kTol1D)) {
            return 1;
        }
    }

    return 0;
}
