/*
 * Migrated reaction operator coverage for polar complex remainder contracts.
 */
#include <math.h>
#include <oakfield/sim.h>
#include <stdio.h>
#include <string.h>

static int nearly(double a, double b, double eps) {
    double d = fabs(a - b);
    double s = fmax(fabs(a), fabs(b));
    if (s < eps)
        s = 1.0;
    return d <= eps * s;
}

int main(void) {
    const size_t N = 6U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField warped = {0}, reference = {0}, output = {0};
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
        fprintf(stderr, "[FAIL] promote complex\n");
        sim_field_destroy(&warped);
        sim_field_destroy(&reference);
        sim_field_destroy(&output);
        sim_context_destroy(&ctx);
        return 1;
    }

    SimComplexDouble *wz = sim_field_complex_data(&warped);
    SimComplexDouble *rz = sim_field_complex_data(&reference);
    SimComplexDouble *oz = sim_field_complex_data(&output);
    for (size_t i = 0; i < N; ++i) {
        double th = 0.3 * (double)i;        /* different phases */
        double rw = 1.0 + 0.1 * (double)i;  /* warped magnitude */
        double rr = 0.8 + 0.05 * (double)i; /* reference magnitude */
        wz[i].re = rw * cos(th);
        wz[i].im = rw * sin(th);
        rz[i].re = rr * cos(th);
        rz[i].im = rr * sin(th);
        oz[i].re = 0.0;
        oz[i].im = 0.0;
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
    cfg.complex_mode = 1U; /* polar */

    if (sim_add_remainder_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add remainder polar\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (size_t i = 0; i < N; ++i) {
        double th = 0.3 * (double)i;
        double rw = 1.0 + 0.1 * (double)i;
        double rr = 0.8 + 0.05 * (double)i;
        double residue = (rw - rr); /* identity nonlinearity */
        double expre = residue * cos(th);
        double expim = residue * sin(th);
        if (!nearly(oz[i].re, expre, 1e-12) || !nearly(oz[i].im, expim, 1e-12)) {
            fprintf(stderr, "[FAIL] polar remainder i=%zu got=(%.12g,%.12g) exp=(%.12g,%.12g)\n", i,
                    oz[i].re, oz[i].im, expre, expim);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    sim_context_destroy(&ctx);
    printf("[PASS] remainder complex polar mode\n");
    return 0;
}
