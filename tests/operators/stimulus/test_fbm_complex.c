/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const size_t kShape[] = {5U, 96U};
static const double kTol = 1.0e-9;

typedef struct {
    uint64_t state;
    uint64_t inc;
} Pcg32State;

static uint32_t pcg32_random(Pcg32State *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void pcg32_seed(Pcg32State *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    (void)pcg32_random(rng);
    rng->state += initstate;
    (void)pcg32_random(rng);
}

static double pcg32_uniform(Pcg32State *rng) { return ldexp(pcg32_random(rng), -32); }

static void fill_phases(double *phases, unsigned int octaves, uint64_t seed) {
    Pcg32State rng;

    if (phases == NULL || octaves == 0U) {
        return;
    }

    pcg32_seed(&rng, seed, seed ^ 0x51ED2705B4C3A59DULL);
    for (unsigned int o = 0U; o < octaves; ++o) {
        phases[o] = 2.0 * M_PI * pcg32_uniform(&rng);
    }
}

static void eval_base(const double *phases, double u, double hurst, double lacunarity,
                      unsigned int octaves, double *out_re, double *out_im) {
    double amp_decay = pow(lacunarity, -hurst);
    double amp = 1.0;
    double freq = 1.0;
    double sum_re = 0.0;
    double sum_im = 0.0;

    for (unsigned int o = 0U; o < octaves; ++o) {
        double theta = 2.0 * M_PI * freq * u + phases[o];
        sum_re += amp * cos(theta);
        sum_im += amp * sin(theta);
        freq *= lacunarity;
        amp *= amp_decay;
    }

    if (out_re != NULL) {
        *out_re = sum_re;
    }
    if (out_im != NULL) {
        *out_im = sum_im;
    }
}

static int run_fbm_case(const SimStimulusFbmConfig *cfg_input, double dt, int steps, double tol) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    size_t shape[2] = {kShape[0], kShape[1]};
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusFbmConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (cfg_input != NULL) {
        cfg = *cfg_input;
    }
    cfg.field_index = field_index;

    size_t op_index = 0U;
    if (sim_add_stimulus_fbm_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stimulus_fbm_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusFbmConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_fbm_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_stimulus_fbm_config\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    if (normalized.coord.mode != SIM_STIMULUS_COORD_SEPARABLE) {
        fprintf(stderr, "FAIL: expected separable coordinate mode\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: operator lookup\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = kShape[0] * kShape[1];
    double *phases = (double *)calloc(normalized.octaves, sizeof(double));
    SimComplexDouble *expected = (SimComplexDouble *)calloc(count, sizeof(SimComplexDouble));
    if (phases == NULL || expected == NULL) {
        fprintf(stderr, "FAIL: out of memory\n");
        free(phases);
        free(expected);
        sim_context_destroy(&ctx);
        return 0;
    }

    fill_phases(phases, normalized.octaves, normalized.seed);

    for (int step = 0; step < steps; ++step) {
        double scale = normalized.scale_by_dt ? dt : 1.0;
        double current_time = (double)step * dt;

        for (size_t i = 0U; i < count; ++i) {
            size_t row = i / kShape[1];
            size_t col = i % kShape[1];
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double sample_x = x - normalized.coord.velocity_x * current_time;
            double sample_y = y - normalized.coord.velocity_y * current_time;

            double fx_re = 0.0;
            double fx_im = 0.0;
            double fy_re = 0.0;
            double fy_im = 0.0;
            double base_re;
            double base_im;

            eval_base(phases, sample_x, normalized.hurst, normalized.lacunarity, normalized.octaves,
                      &fx_re, &fx_im);
            eval_base(phases, sample_y, normalized.hurst, normalized.lacunarity, normalized.octaves,
                      &fy_re, &fy_im);

            if (normalized.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }

            expected[i].re += scale * normalized.amplitude * base_re;
            expected[i].im += scale * normalized.amplitude * base_im;
        }

        if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: operator evaluate at step %d\n", step);
            free(phases);
            free(expected);
            sim_context_destroy(&ctx);
            return 0;
        }
        sim_context_accept_step(&ctx, (float)dt);
    }

    SimField *result_field = sim_context_field(&ctx, field_index);
    if (result_field == NULL) {
        fprintf(stderr, "FAIL: result field lookup\n");
        free(phases);
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

    free(phases);
    free(expected);
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    SimStimulusFbmConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.31;
    cfg.hurst = 0.54;
    cfg.lacunarity = 1.88;
    cfg.octaves = 5U;
    cfg.seed = 24680ULL;
    cfg.scale_by_dt = true;
    cfg.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg.coord.combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;
    cfg.coord.origin_x = -0.72;
    cfg.coord.origin_y = 0.18;
    cfg.coord.velocity_x = 0.11;
    cfg.coord.velocity_y = -0.07;
    cfg.coord.spacing_x = 0.19;
    cfg.coord.spacing_y = 0.23;

    return run_fbm_case(&cfg, 0.125, 4, kTol) ? 0 : 1;
}
