#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kCount = 17U;
static const double kTol = 1.0e-12;

static int run_white_noise_step(const SimStimulusWhiteNoiseConfig *cfg_input, double dt,
                                double *out_values, size_t count) {
    SimContext ctx = {0};
    size_t shape[1] = {count};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    SimOperator *op = NULL;
    SimStimulusWhiteNoiseConfig cfg;

    if (out_values == NULL || count == 0U) {
        return 0;
    }
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);
    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
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

    if (sim_add_stimulus_white_noise_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stimulus_white_noise_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: evaluate white noise\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        const double *values = sim_field_data_const(sim_context_field(&ctx, field_index));
        if (values == NULL) {
            fprintf(stderr, "FAIL: white-noise field data\n");
            sim_context_destroy(&ctx);
            return 0;
        }
        memcpy(out_values, values, count * sizeof(double));
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int check_scaled_ratio(const double *a, const double *b, size_t count, double factor,
                              double tol, const char *label) {
    for (size_t i = 0U; i < count; ++i) {
        double err = fabs(a[i] - factor * b[i]);
        if (err > tol) {
            fprintf(stderr, "FAIL: %s mismatch at %zu got=%.17g expected=%.17g\n", label, i, a[i],
                    factor * b[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    double a[kCount];
    double b[kCount];
    SimStimulusWhiteNoiseConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sigma = 0.35;
    cfg.mean = 0.0;
    cfg.seed = 123456ULL;
    cfg.nominal_dt = 0.25;
    cfg.fixed_clock = false;
    cfg.scale_by_dt = true;

    if (!run_white_noise_step(&cfg, 0.25, a, kCount) ||
        !run_white_noise_step(&cfg, 0.0625, b, kCount) ||
        !check_scaled_ratio(a, b, kCount, 2.0, kTol, "scale_by_dt")) {
        return 1;
    }

    cfg.fixed_clock = true;
    if (!run_white_noise_step(&cfg, 0.25, a, kCount) ||
        !run_white_noise_step(&cfg, 0.0625, b, kCount) ||
        !check_scaled_ratio(a, b, kCount, 1.0, kTol, "fixed_clock")) {
        return 1;
    }

    cfg.fixed_clock = false;
    cfg.scale_by_dt = false;
    if (!run_white_noise_step(&cfg, 0.25, a, kCount) ||
        !run_white_noise_step(&cfg, 0.0625, b, kCount) ||
        !check_scaled_ratio(a, b, kCount, 1.0, kTol, "dt_independent")) {
        return 1;
    }

    return 0;
}
