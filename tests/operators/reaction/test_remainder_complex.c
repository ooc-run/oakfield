/*
 * Migrated reaction operator coverage for complex remainder contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool nearly_equal(double a, double b, double eps) {
    double diff = fabs(a - b);
    double scale = fmax(fabs(a), fabs(b));
    if (scale < eps)
        scale = 1.0;
    return diff <= eps * scale;
}

int main(void) {
    const size_t N = 16U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField warped = {0}, reference = {0}, output = {0};
    size_t i;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] ctx init\n");
        return 1;
    }

    if (sim_field_init(&warped, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&reference, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] field init\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_field_promote_inplace_to_complex(&warped) != SIM_RESULT_OK ||
        sim_field_promote_inplace_to_complex(&reference) != SIM_RESULT_OK ||
        sim_field_promote_inplace_to_complex(&output) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promote to complex\n");
        sim_field_destroy(&warped);
        sim_field_destroy(&reference);
        sim_field_destroy(&output);
        sim_context_destroy(&ctx);
        return 1;
    }

    SimComplexDouble *wz = sim_field_complex_data(&warped);
    SimComplexDouble *rz = sim_field_complex_data(&reference);
    SimComplexDouble *oz = sim_field_complex_data(&output);
    if (!wz || !rz || !oz) {
        fprintf(stderr, "[FAIL] complex ptrs\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (i = 0; i < N; ++i) {
        wz[i].re = 0.1 * (double)i;
        wz[i].im = 0.2 * (double)i;
        rz[i].re = 0.05 * (double)i;
        rz[i].im = -0.1 * (double)i;
        oz[i].re = -123.0;
        oz[i].im = -123.0;
    }

    size_t wi, ri, oi;
    if (sim_context_add_field(&ctx, &warped, &wi) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &reference, &ri) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &oi) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add fields\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimRemainderOperatorConfig cfg = {0};
    cfg.warped_field = wi;
    cfg.reference_field = ri;
    cfg.output_field = oi;
    cfg.weight = 1.0;
    cfg.bias = 0.0;
    cfg.exponent = 1.0;
    cfg.epsilon = 1.0e-6;
    cfg.nonlinearity = SIM_REMAINDER_NONLINEARITY_IDENTITY;
    cfg.accumulate = false;

    if (sim_add_remainder_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add operator\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute 1\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    /* Check identity remainder per component */
    for (i = 0; i < N; ++i) {
        double er = (0.1 * (double)i) - (0.05 * (double)i);
        double ei = (0.2 * (double)i) - (-0.1 * (double)i);
        if (!nearly_equal(oz[i].re, er, 1.0e-12) || !nearly_equal(oz[i].im, ei, 1.0e-12)) {
            fprintf(stderr, "[FAIL] identity i=%zu got=(%.12g, %.12g) expect=(%.12g, %.12g)\n", i,
                    oz[i].re, oz[i].im, er, ei);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    /* Accumulate with weight/bias */
    for (i = 0; i < N; ++i) {
        oz[i].re = 1.0;
        oz[i].im = 1.0;
    }
    cfg.accumulate = true;
    cfg.weight = 0.5;
    cfg.bias = 0.25;
    if (sim_remainder_update(&ctx, 0U, &cfg) != SIM_RESULT_OK) /* single operator at index 0 */
    {
        fprintf(stderr, "[FAIL] update operator\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute 2\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (i = 0; i < N; ++i) {
        double base_r = (0.1 * (double)i) - (0.05 * (double)i);
        double base_i = (0.2 * (double)i) - (-0.1 * (double)i);
        double add_r = base_r * cfg.weight + cfg.bias;
        double add_i = base_i * cfg.weight + cfg.bias;
        double er = 1.0 + add_r;
        double ei = 1.0 + add_i;
        if (!nearly_equal(oz[i].re, er, 1.0e-12) || !nearly_equal(oz[i].im, ei, 1.0e-12)) {
            fprintf(stderr, "[FAIL] accumulate i=%zu got=(%.12g, %.12g) expect=(%.12g, %.12g)\n", i,
                    oz[i].re, oz[i].im, er, ei);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    sim_context_destroy(&ctx);
    printf("[PASS] remainder complex identity/accumulate\n");
    return 0;
}
