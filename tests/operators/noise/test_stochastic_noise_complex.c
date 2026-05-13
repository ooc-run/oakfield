/*
 * Migrated noise operator coverage for stochastic complex-output contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t kCount = 17U;
static const double kTol = 1.0e-12;

static int run_stochastic_noise_step(const StochasticNoiseOperatorConfig *cfg_input, double dt,
                                     SimComplexDouble *out_values, size_t count) {
    SimContext ctx = {0};
    size_t shape[1] = {count};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    SimOperator *op = NULL;
    StochasticNoiseOperatorConfig cfg;

    if (out_values == NULL || count == 0U) {
        return 0;
    }
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, (float)dt);
    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
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

    if (sim_add_stochastic_noise_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_stochastic_noise_operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: evaluate stochastic_noise\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        const SimComplexDouble *values =
            sim_field_complex_data_const(sim_context_field(&ctx, field_index));
        if (values == NULL) {
            fprintf(stderr, "FAIL: stochastic-noise complex field data\n");
            sim_context_destroy(&ctx);
            return 0;
        }
        memcpy(out_values, values, count * sizeof(SimComplexDouble));
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int check_complex_ratio(const SimComplexDouble *a, const SimComplexDouble *b, size_t count,
                               double factor, double tol, const char *label) {
    for (size_t i = 0U; i < count; ++i) {
        double err_re = fabs(a[i].re - factor * b[i].re);
        double err_im = fabs(a[i].im - factor * b[i].im);
        if (err_re > tol || err_im > tol) {
            fprintf(stderr, "FAIL: %s mismatch at %zu got=(%.17g, %.17g) expected=(%.17g, %.17g)\n",
                    label, i, a[i].re, a[i].im, factor * b[i].re, factor * b[i].im);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    SimComplexDouble a[kCount];
    SimComplexDouble b[kCount];
    StochasticNoiseOperatorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sigma = 0.28;
    cfg.tau = 0.05;
    cfg.alpha = 1.0;
    cfg.seed = 424242ULL;
    cfg.law = SIM_IR_NOISE_LAW_ITO;

    if (!run_stochastic_noise_step(&cfg, 0.25, a, kCount) ||
        !run_stochastic_noise_step(&cfg, 0.0625, b, kCount) ||
        !check_complex_ratio(a, b, kCount, 2.0, kTol, "ito_scaling")) {
        return 1;
    }

    cfg.law = SIM_IR_NOISE_LAW_STRATONOVICH;
    if (!run_stochastic_noise_step(&cfg, 0.125, a, kCount) ||
        !run_stochastic_noise_step(&cfg, 0.125, b, kCount) ||
        !check_complex_ratio(a, b, kCount, 1.0, kTol, "stratonovich_determinism")) {
        return 1;
    }

    return 0;
}
