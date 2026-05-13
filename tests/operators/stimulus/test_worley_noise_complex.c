/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kShape[] = {5U, 7U};
static const double kTol = 1.0e-9;

static uint64_t mix_u64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

static double uniform01(uint64_t value) {
    return (double)(mix_u64(value) >> 11U) * (1.0 / 9007199254740992.0);
}

static uint64_t hash_feature(uint64_t seed, int64_t cell_x, int64_t cell_y, unsigned channel,
                             unsigned axis) {
    uint64_t hash = mix_u64(seed ^ 0xD1B54A32D192ED03ULL);
    hash = mix_u64(hash ^ (uint64_t)(uint32_t)channel);
    hash = mix_u64(hash ^ (uint64_t)axis * 0x94D049BB133111EBULL);
    hash = mix_u64(hash ^ (uint64_t)cell_x);
    hash = mix_u64(hash ^ ((uint64_t)cell_y << 1U));
    return hash;
}

static void map_coord(const SimStimulusCoordConfig *coord, double x, double y, double t,
                      double *out_u, double *out_v) {
    double u = x;
    double v = y;

    if (coord != NULL) {
        u -= coord->velocity_x * t;
        v -= coord->velocity_y * t;
    }

    if (coord != NULL && coord->mode == SIM_STIMULUS_COORD_ANGLE) {
        double c = cos(coord->angle);
        double s = sin(coord->angle);
        double rotated_u = u * c + v * s;
        double rotated_v = -u * s + v * c;
        u = rotated_u;
        v = rotated_v;
    }

    if (out_u != NULL) {
        *out_u = u;
    }
    if (out_v != NULL) {
        *out_v = v;
    }
}

static double distance_metric(double dx, double dy, SimStimulusWorleyDistanceMetric metric,
                              double exponent) {
    switch (metric) {
    case SIM_STIMULUS_WORLEY_MANHATTAN:
        return fabs(dx) + fabs(dy);
    case SIM_STIMULUS_WORLEY_CHEBYSHEV:
        return fmax(fabs(dx), fabs(dy));
    case SIM_STIMULUS_WORLEY_MINKOWSKI:
        return pow(pow(fabs(dx), exponent) + pow(fabs(dy), exponent), 1.0 / exponent);
    case SIM_STIMULUS_WORLEY_EUCLIDEAN:
    default:
        return hypot(dx, dy);
    }
}

static double output_value(double f1, double f2, SimStimulusWorleyOutputMode mode) {
    if (!isfinite(f2)) {
        f2 = f1;
    }
    switch (mode) {
    case SIM_STIMULUS_WORLEY_F1:
        return f1;
    case SIM_STIMULUS_WORLEY_F2:
        return f2;
    case SIM_STIMULUS_WORLEY_F2_MINUS_F1:
    default:
        return fmax(0.0, f2 - f1);
    }
}

static double component_value(const SimStimulusWorleyNoiseConfig *cfg, double x, double y, double t,
                              unsigned channel) {
    double u = 0.0;
    double v = 0.0;
    double f1 = DBL_MAX;
    double f2 = DBL_MAX;

    map_coord(&cfg->coord, x, y, t, &u, &v);
    u *= cfg->feature_frequency;
    v *= cfg->feature_frequency;

    {
        int64_t base_x = (int64_t)floor(u);
        int64_t base_y = (int64_t)floor(v);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int64_t cell_x = base_x + (int64_t)dx;
                int64_t cell_y = base_y + (int64_t)dy;
                double jitter_x = uniform01(hash_feature(cfg->seed, cell_x, cell_y, channel, 0U));
                double jitter_y = uniform01(hash_feature(cfg->seed, cell_x, cell_y, channel, 1U));
                double feature_x = (double)cell_x + 0.5 + cfg->jitter * (jitter_x - 0.5);
                double feature_y = (double)cell_y + 0.5 + cfg->jitter * (jitter_y - 0.5);
                double distance = distance_metric(u - feature_x, v - feature_y,
                                                  cfg->distance_metric, cfg->distance_exponent);

                if (distance < f1) {
                    f2 = f1;
                    f1 = distance;
                } else if (distance < f2) {
                    f2 = distance;
                }
            }
        }
    }

    return output_value(f1, f2, cfg->output_mode);
}

static int run_case(const SimStimulusWorleyNoiseConfig *cfg_input, double dt, int steps,
                    double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    {
        size_t shape[2] = {kShape[0], kShape[1]};
        SimField field = {0};
        size_t field_index = 0U;
        SimStimulusWorleyNoiseConfig cfg;
        size_t op_index = 0U;
        SimStimulusWorleyNoiseConfig normalized;
        SimOperator *op;
        size_t count;
        SimComplexDouble *expected;

        if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                           NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: sim_field_init\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: sim_context_add_field\n");
            sim_field_destroy(&field);
            sim_context_destroy(&ctx);
            return 0;
        }

        memset(&cfg, 0, sizeof(cfg));
        if (cfg_input != NULL) {
            cfg = *cfg_input;
        }
        cfg.field_index = field_index;

        if (sim_add_stimulus_worley_noise_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: sim_add_stimulus_worley_noise_operator\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        memset(&normalized, 0, sizeof(normalized));
        if (sim_stimulus_worley_noise_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: sim_stimulus_worley_noise_config\n");
            sim_context_destroy(&ctx);
            return 0;
        }
        if (normalized.coord.mode != SIM_STIMULUS_COORD_ANGLE) {
            fprintf(stderr, "FAIL: expected angle coordinate mode\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        op = sim_operator_registry_get(&ctx.world.operators, op_index);
        if (op == NULL) {
            fprintf(stderr, "FAIL: operator lookup\n");
            sim_context_destroy(&ctx);
            return 0;
        }
        if (op->info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
            !op->info.representation.requires_complex_input ||
            !op->info.representation.requires_complex_representation ||
            !op->info.representation.preserves_real_subspace) {
            fprintf(stderr, "FAIL: complex representation metadata mismatch\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        count = kShape[0] * kShape[1];
        expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
        if (expected == NULL) {
            fprintf(stderr, "FAIL: out of memory\n");
            sim_context_destroy(&ctx);
            return 0;
        }

        for (int step = 0; step < steps; ++step) {
            double scale = normalized.scale_by_dt ? dt : 1.0;
            double current_time = (double)step * dt;

            for (size_t i = 0U; i < count; ++i) {
                size_t row = i / kShape[1];
                size_t col = i % kShape[1];
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double re = component_value(&normalized, x, y, current_time, 0U);
                double im = component_value(&normalized, x, y, current_time, 1U);

                expected[i].re += scale * normalized.amplitude * re;
                expected[i].im += scale * normalized.amplitude * im;
            }

            if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
                fprintf(stderr, "FAIL: operator evaluate at step %d\n", step);
                free(expected);
                sim_context_destroy(&ctx);
                return 0;
            }
            sim_context_accept_step(&ctx, (float)dt);
        }

        {
            SimField *result_field = sim_context_field(&ctx, field_index);
            const SimComplexDouble *values;
            int ok = 1;
            if (result_field == NULL) {
                fprintf(stderr, "FAIL: result field lookup\n");
                free(expected);
                sim_context_destroy(&ctx);
                return 0;
            }

            values = sim_field_complex_data_const(result_field);
            for (size_t i = 0U; i < count; ++i) {
                double err_re = fabs(values[i].re - expected[i].re);
                double err_im = fabs(values[i].im - expected[i].im);
                if (err_re > tol || err_im > tol) {
                    fprintf(stderr,
                            "FAIL: mismatch at %zu got=(%.12g, %.12g) expected=(%.12g, %.12g)\n", i,
                            values[i].re, values[i].im, expected[i].re, expected[i].im);
                    ok = 0;
                    break;
                }
            }

            free(expected);
            sim_context_destroy(&ctx);
            return ok;
        }
    }
}

int main(void) {
    SimStimulusWorleyNoiseConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.42;
    cfg.feature_frequency = 1.75;
    cfg.jitter = 0.63;
    cfg.distance_metric = SIM_STIMULUS_WORLEY_MINKOWSKI;
    cfg.distance_exponent = 3.25;
    cfg.output_mode = SIM_STIMULUS_WORLEY_F2_MINUS_F1;
    cfg.seed = 2468ULL;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.37;
    cfg.coord.velocity_x = 0.19;
    cfg.coord.velocity_y = -0.13;
    cfg.coord.origin_x = -0.6;
    cfg.coord.origin_y = 0.3;
    cfg.coord.spacing_x = 0.35;
    cfg.coord.spacing_y = 0.22;

    if (!run_case(&cfg, 0.125, 4, kTol)) {
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.33;
    cfg.feature_frequency = 2.4;
    cfg.jitter = 0.91;
    cfg.distance_metric = SIM_STIMULUS_WORLEY_EUCLIDEAN;
    cfg.output_mode = SIM_STIMULUS_WORLEY_F2;
    cfg.seed = 97531ULL;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = -0.21;
    cfg.coord.origin_x = -0.45;
    cfg.coord.origin_y = 0.25;
    cfg.coord.spacing_x = 0.18;
    cfg.coord.spacing_y = 0.16;

    return run_case(&cfg, 0.0625, 3, kTol) ? 0 : 1;
}
