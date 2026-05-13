/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t state;
    uint64_t inc;
} ref_shells_pcg32_t;

static uint32_t ref_shells_pcg32_random(ref_shells_pcg32_t *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void ref_shells_pcg32_seed(ref_shells_pcg32_t *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    (void)ref_shells_pcg32_random(rng);
    rng->state += initstate;
    (void)ref_shells_pcg32_random(rng);
}

static double ref_shells_uniform(ref_shells_pcg32_t *rng) {
    return ldexp(ref_shells_pcg32_random(rng), -32);
}

static int add_real_field(SimContext *ctx, size_t rank, const size_t *shape,
                          size_t *out_field_index) {
    SimField field = {0};
    if (sim_field_init(&field, rank, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        return 0;
    }

    double *raw = (double *)sim_field_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    if (sim_context_add_field(ctx, &field, out_field_index) != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return 0;
    }

    return 1;
}

static int compare_spectral_shells_reference(SimContext *ctx, size_t field_index,
                                             const SimStimulusSpectralShellsConfig *cfg, double tol,
                                             const char *label) {
    SimField *field = sim_context_field(ctx, field_index);
    if (field == NULL || cfg == NULL || field->element_size != sizeof(double)) {
        fprintf(stderr, "[%s] invalid reference inputs\n", label);
        return 0;
    }

    const double *values = (const double *)sim_field_data(field);
    if (values == NULL) {
        fprintf(stderr, "[%s] missing field data\n", label);
        return 0;
    }

    unsigned int desired = cfg->shell_count * cfg->modes_per_shell;
    double *kx = (double *)malloc((size_t)desired * sizeof(double));
    double *ky = (double *)malloc((size_t)desired * sizeof(double));
    double *phi = (double *)malloc((size_t)desired * sizeof(double));
    double *weight = (double *)malloc((size_t)desired * sizeof(double));
    if (kx == NULL || ky == NULL || phi == NULL || weight == NULL) {
        fprintf(stderr, "[%s] reference allocation failed\n", label);
        free(kx);
        free(ky);
        free(phi);
        free(weight);
        return 0;
    }

    const int layout_1d = (field->layout.rank == 1U);
    ref_shells_pcg32_t rng;
    ref_shells_pcg32_seed(&rng, cfg->seed, cfg->seed ^ 0xC2B2AE3D27D4EB4FULL);

    double k_min = cfg->k_min;
    double k_max = cfg->k_max;
    double range = k_max - k_min;
    double shell_step =
        (cfg->shell_count > 1U && range > 1.0e-9) ? (range / (double)(cfg->shell_count - 1U)) : 0.0;
    double shell_width = cfg->shell_width;
    if (shell_width <= 1.0e-9 && range > 1.0e-9) {
        shell_width = range / (double)cfg->shell_count;
    }

    unsigned int active_modes = 0U;
    double sum_sq = 0.0;
    for (unsigned int shell = 0U; shell < cfg->shell_count && active_modes < desired; ++shell) {
        double center = (cfg->shell_count > 1U && range > 1.0e-9)
                            ? (k_min + shell_step * (double)shell)
                            : ((range > 1.0e-9) ? (0.5 * (k_min + k_max)) : k_min);
        double shell_lo = center;
        double shell_hi = center;

        if (range > 1.0e-9 && shell_width > 1.0e-9) {
            shell_lo = center - 0.5 * shell_width;
            shell_hi = center + 0.5 * shell_width;
            if (shell_lo < k_min) {
                shell_lo = k_min;
            }
            if (shell_hi > k_max) {
                shell_hi = k_max;
            }
            if (shell_lo > shell_hi) {
                shell_lo = center;
                shell_hi = center;
            }
        }

        for (unsigned int mode = 0U; mode < cfg->modes_per_shell && active_modes < desired;
             ++mode) {
            double u = ref_shells_uniform(&rng);
            double v = ref_shells_uniform(&rng);
            double w = ref_shells_uniform(&rng);

            double radius = center;
            if (shell_hi > shell_lo + 1.0e-9) {
                radius = shell_lo + (shell_hi - shell_lo) * u;
            }
            if (!isfinite(radius)) {
                radius = 0.0;
            }
            if (radius < 0.0) {
                radius = -radius;
            }

            if (layout_1d) {
                double sign = (v < 0.5) ? -1.0 : 1.0;
                kx[active_modes] = sign * radius;
                ky[active_modes] = 0.0;
            } else {
                double angle = 2.0 * M_PI * v;
                kx[active_modes] = radius * cos(angle);
                ky[active_modes] = radius * sin(angle);
            }

            phi[active_modes] = 2.0 * M_PI * w;
            weight[active_modes] = 1.0;
            if (cfg->spectral_slope != 0.0 && radius > 1.0e-9) {
                weight[active_modes] = pow(radius, -0.5 * cfg->spectral_slope);
            }
            sum_sq += weight[active_modes] * weight[active_modes];
            ++active_modes;
        }
    }

    double mode_norm =
        (sum_sq > 1.0e-9 && active_modes > 0U) ? sqrt(sum_sq / (double)active_modes) : 1.0;
    double inv_sqrtM = (active_modes > 0U) ? (1.0 / sqrt((double)active_modes)) : 1.0;
    double t = sim_context_time(ctx) + cfg->time_offset;
    size_t count = sim_field_bytes(field) / sizeof(double);
    int ok = 1;

    for (size_t i = 0U; i < count; ++i) {
        size_t ix = 0U;
        size_t iy = 0U;
        if (sim_field_index_to_xy(field, i, &ix, &iy) != SIM_RESULT_OK) {
            fprintf(stderr, "[%s] index_to_xy failed at %zu\n", label, i);
            ok = 0;
            break;
        }

        double x = cfg->coord.origin_x + (double)ix * cfg->coord.spacing_x;
        double y = layout_1d ? 0.0 : (cfg->coord.origin_y + (double)iy * cfg->coord.spacing_y);
        double re_sum = 0.0;

        for (unsigned int m = 0U; m < active_modes; ++m) {
            double theta = kx[m] * x + ky[m] * y - cfg->omega * t + phi[m];
            re_sum += inv_sqrtM * (weight[m] / mode_norm) * cos(theta);
        }

        double expected = cfg->amplitude * re_sum;
        double err = fabs(values[i] - expected);
        if (err > tol) {
            fprintf(stderr, "[%s] ref mismatch idx=%zu got=%.12g exp=%.12g err=%.3g tol=%.3g\n",
                    label, i, values[i], expected, err, tol);
            ok = 0;
            break;
        }
    }

    free(kx);
    free(ky);
    free(phi);
    free(weight);
    return ok;
}

static int eval_operator(SimContext *ctx, size_t op_index, const char *label) {
    SimOperator *op = sim_operator_registry_get(&ctx->world.operators, op_index);
    if (op == NULL || op->evaluate == NULL) {
        fprintf(stderr, "[%s] operator evaluate hook unavailable\n", label);
        return 0;
    }
    if (op->evaluate(ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] operator evaluation failed\n", label);
        return 0;
    }
    return 1;
}

static int compare_real_fields(SimContext *ctx, size_t field_a, size_t field_b, double tol,
                               const char *label) {
    SimField *fa = sim_context_field(ctx, field_a);
    SimField *fb = sim_context_field(ctx, field_b);
    if (fa == NULL || fb == NULL) {
        fprintf(stderr, "[%s] missing comparison fields\n", label);
        return 0;
    }

    size_t bytes_a = sim_field_bytes(fa);
    size_t bytes_b = sim_field_bytes(fb);
    if (bytes_a != bytes_b || fa->element_size != sizeof(double) ||
        fb->element_size != sizeof(double)) {
        fprintf(stderr, "[%s] incompatible field layouts\n", label);
        return 0;
    }

    const double *a = (const double *)sim_field_data(fa);
    const double *b = (const double *)sim_field_data(fb);
    if (a == NULL || b == NULL) {
        fprintf(stderr, "[%s] missing comparison data\n", label);
        return 0;
    }

    size_t count = bytes_a / sizeof(double);
    for (size_t i = 0U; i < count; ++i) {
        double err = fabs(a[i] - b[i]);
        if (err > tol) {
            fprintf(stderr, "[%s] mismatch idx=%zu a=%.12g b=%.12g err=%.3g tol=%.3g\n", label, i,
                    a[i], b[i], err, tol);
            return 0;
        }
    }

    return 1;
}

static int check_signal_variation(SimContext *ctx, size_t field_index, const char *label) {
    SimField *field = sim_context_field(ctx, field_index);
    if (field == NULL || field->element_size != sizeof(double)) {
        fprintf(stderr, "[%s] invalid field\n", label);
        return 0;
    }

    const double *values = (const double *)sim_field_data(field);
    if (values == NULL) {
        fprintf(stderr, "[%s] missing field data\n", label);
        return 0;
    }

    size_t count = sim_field_bytes(field) / sizeof(double);
    if (count == 0U) {
        fprintf(stderr, "[%s] empty field\n", label);
        return 0;
    }

    double min_v = values[0];
    double max_v = values[0];
    double sum_abs = 0.0;

    for (size_t i = 0U; i < count; ++i) {
        double v = values[i];
        if (!isfinite(v)) {
            fprintf(stderr, "[%s] non-finite value at idx=%zu\n", label, i);
            return 0;
        }
        if (v < min_v) {
            min_v = v;
        }
        if (v > max_v) {
            max_v = v;
        }
        sum_abs += fabs(v);
    }

    if ((max_v - min_v) <= 1.0e-9 || sum_abs <= 1.0e-9) {
        fprintf(stderr, "[%s] insufficient variation (range=%.3g, sum_abs=%.3g)\n", label,
                max_v - min_v, sum_abs);
        return 0;
    }

    return 1;
}

static int run_case(const char *label, size_t rank, const size_t *shape) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] failed to init context\n", label);
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_a = 0U;
    size_t field_b = 0U;
    if (!add_real_field(&ctx, rank, shape, &field_a) ||
        !add_real_field(&ctx, rank, shape, &field_b)) {
        fprintf(stderr, "[%s] failed to add fields\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusSpectralShellsConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.75;
    cfg.k_min = 0.35;
    cfg.k_max = 1.9;
    cfg.shell_width = 0.2;
    cfg.omega = 0.0;
    cfg.time_offset = 0.0;
    cfg.nominal_dt = 0.0;
    cfg.spectral_slope = 0.7;
    cfg.shell_count = 5U;
    cfg.modes_per_shell = 12U;
    cfg.seed = 1234567ULL;
    cfg.fixed_clock = false;
    cfg.scale_by_dt = false;
    cfg.coord.origin_x = -0.3;
    cfg.coord.origin_y = 0.2;
    cfg.coord.spacing_x = 0.18;
    cfg.coord.spacing_y = 0.27;

    SimStimulusSpectralShellsConfig cfg_a = cfg;
    cfg_a.field_index = field_a;
    SimStimulusSpectralShellsConfig cfg_b = cfg;
    cfg_b.field_index = field_b;
    SimStimulusSpectralShellsConfig normalized;
    memset(&normalized, 0, sizeof(normalized));

    size_t op_a = 0U;
    size_t op_b = 0U;
    if (sim_add_stimulus_spectral_shells_operator(&ctx, &cfg_a, &op_a) != SIM_RESULT_OK ||
        sim_add_stimulus_spectral_shells_operator(&ctx, &cfg_b, &op_b) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] failed to add operators\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    if (sim_stimulus_spectral_shells_config(&ctx, op_a, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] failed to fetch normalized config\n", label);
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_a, label) || !eval_operator(&ctx, op_b, label)) {
        sim_context_destroy(&ctx);
        return 0;
    }

    int ok = 1;
    ok &= check_signal_variation(&ctx, field_a, label);
    ok &= compare_real_fields(&ctx, field_a, field_b, 1.0e-12, label);
    ok &= compare_spectral_shells_reference(&ctx, field_a, &normalized, 1.0e-9, label);

    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    int ok = 1;

    {
        const size_t shape_1d[1] = {96U};
        ok &= run_case("spectral_shells_1d", 1U, shape_1d);
    }

    {
        const size_t shape_2d[2] = {9U, 13U};
        ok &= run_case("spectral_shells_2d", 2U, shape_2d);
    }

    return ok ? 0 : 1;
}
