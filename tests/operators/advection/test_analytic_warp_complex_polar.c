/*
 * Migrated advection operator coverage for analytic warp polar-complex contracts.
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
    const size_t N = 8U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField field = {0};
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] ctx init\n");
        return 1;
    }
    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] field init\n");
        sim_context_destroy(&ctx);
        return 1;
    }
    if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promote\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 1;
    }
    SimComplexDouble *z = sim_field_complex_data(&field);
    for (size_t i = 0; i < N; ++i) {
        double th = 0.2 * (double)i;       /* distinct angles */
        double r = 0.1 + 0.05 * (double)i; /* increasing magnitudes */
        z[i].re = r * cos(th);
        z[i].im = r * sin(th);
    }
    size_t fi;
    if (sim_context_add_field(&ctx, &field, &fi) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add field\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    sim_context_set_timestep(&ctx, 0.05f);

    AnalyticWarpOperatorConfig cfg = {0};
    cfg.field_index = fi;
    cfg.profile = ANALYTIC_WARP_PROFILE_TANH; /* smooth bounded gradient */
    cfg.delta = 0.25;
    cfg.lambda = 0.4;
    cfg.bias = 0.1;
    cfg.exponent = 2.0;
    cfg.complex_mode = 1U; /* polar mode */

    if (sim_add_analytic_warp_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] register warp polar\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    /* Validate phase preserved (argument difference ~0) and magnitude changed in expected
     * direction. */
    for (size_t i = 0; i < N; ++i) {
        double new_r = sqrt(z[i].re * z[i].re + z[i].im * z[i].im);
        double new_th = atan2(z[i].im, z[i].re);
        double orig_th = 0.2 * (double)i;
        if (!nearly(new_th, orig_th, 1e-6)) {
            fprintf(stderr, "[FAIL] polar phase drift i=%zu orig=%.9g new=%.9g\n", i, orig_th,
                    new_th);
            sim_context_destroy(&ctx);
            return 1;
        }
        /* Expect magnitude perturbation: not equal to original r */
        double orig_r = 0.1 + 0.05 * (double)i;
        if (fabs(new_r - orig_r) < 1e-12) {
            fprintf(stderr, "[FAIL] polar magnitude unchanged i=%zu r=%.12g\n", i, new_r);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    sim_context_destroy(&ctx);
    printf("[PASS] analytic warp polar complex mode\n");
    return 0;
}
