/*
 * Migrated advection operator coverage for analytic warp complex contracts.
 */
#include <oakfield/sim.h>
#include <oakfield/math/special_functions.h>

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double approx_profile_derivative(AnalyticWarpProfile profile, double x) {
    switch (profile) {
    case ANALYTIC_WARP_PROFILE_TRIGAMMA:
        /* derivative of trigamma ≈ finite difference */
        return (sim_special_trigamma(x + 1e-6) - sim_special_trigamma(x - 1e-6)) / (2e-6);
    case ANALYTIC_WARP_PROFILE_DIGAMMA:
        /* derivative of digamma is trigamma */
        return sim_special_trigamma(x);
    case ANALYTIC_WARP_PROFILE_POWER: {
        double magnitude = fabs(x);
        double exponent = 2.0;
        if (magnitude < 1.0e-6)
            magnitude = 1.0e-6;
        return exponent * pow(magnitude, exponent - 1.0);
    }
    case ANALYTIC_WARP_PROFILE_TANH: {
        double t = tanh(x);
        return 1.0 - t * t;
    }
    case ANALYTIC_WARP_PROFILE_HYPEREXP:
        return sim_hyperexp_phi_deriv(x, 0.75, 6);
    default:
        return NAN;
    }
}

static bool run_complex_case(AnalyticWarpProfile profile) {
    SimContext ctx = {0};
    SimField field = {0};
    const size_t N = 16U;
    size_t shape[1] = {N};
    const double dt = 0.05;
    const double delta = 0.25;
    const double lambda = 0.2;
    const double bias = 0.5;
    bool attached = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] context init\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] field init\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promote to complex\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    attached = true; /* ownership transferred */

    SimField *f = sim_context_field(&ctx, field_index);
    SimComplexDouble *z = sim_field_complex_data(f);
    if (!z) {
        fprintf(stderr, "[FAIL] complex data ptr\n");
        sim_context_destroy(&ctx);
        return false;
    }

    /* initialize distinct real/imag sequences to ensure independent component handling */
    for (size_t i = 0; i < N; ++i) {
        z[i].re = 0.1 * (double)i; /* 0.0, 0.1, ... */
        z[i].im = 0.2 * (double)i; /* 0.0, 0.2, ... */
    }

    sim_context_set_timestep(&ctx, (float)dt);

    AnalyticWarpOperatorConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.profile = profile;
    cfg.delta = delta;
    cfg.lambda = lambda;
    cfg.bias = bias;
    if (profile == ANALYTIC_WARP_PROFILE_POWER) {
        cfg.exponent = 2.0; /* keep in sync with approx_profile_derivative */
    } else if (profile == ANALYTIC_WARP_PROFILE_HYPEREXP) {
        cfg.hyperexp_epsilon = 0.75;
        cfg.hyperexp_depth = 6;
    }

    if (sim_add_analytic_warp_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] register operator\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_context_prepare_plan(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] prepare plan\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute\n");
        sim_context_destroy(&ctx);
        return false;
    }

    /* verify per-component update: x <- x + dt * lambda * (2*delta) * gradient(x + bias) */
    for (size_t i = 0; i < N; ++i) {
        double xr0 = 0.1 * (double)i; /* initial real */
        double xi0 = 0.2 * (double)i; /* initial imag */
        double gr = approx_profile_derivative(profile, xr0 + bias);
        double gi = approx_profile_derivative(profile, xi0 + bias);
        double xr_expect = xr0;
        double xi_expect = xi0;
        if (isfinite(gr))
            xr_expect += dt * lambda * (2.0 * delta) * gr;
        if (isfinite(gi))
            xi_expect += dt * lambda * (2.0 * delta) * gi;

        double xr = z[i].re;
        double xi = z[i].im;
        double tol = 1.0e-8;
        double scale_r = fmax(1.0, fabs(xr_expect));
        double scale_i = fmax(1.0, fabs(xi_expect));
        if (fabs(xr - xr_expect) > tol * scale_r) {
            fprintf(stderr, "[FAIL] real mismatch i=%zu got=%.12g expect=%.12g\n", i, xr,
                    xr_expect);
            sim_context_destroy(&ctx);
            return false;
        }
        if (fabs(xi - xi_expect) > tol * scale_i) {
            fprintf(stderr, "[FAIL] imag mismatch i=%zu got=%.12g expect=%.12g\n", i, xi,
                    xi_expect);
            sim_context_destroy(&ctx);
            return false;
        }
    }

    sim_context_destroy(&ctx);
    return true;
}

int main(void) {
    bool ok_digamma = run_complex_case(ANALYTIC_WARP_PROFILE_DIGAMMA);
    bool ok_trigamma = run_complex_case(ANALYTIC_WARP_PROFILE_TRIGAMMA);
    bool ok_power = run_complex_case(ANALYTIC_WARP_PROFILE_POWER);
    bool ok_tanh = run_complex_case(ANALYTIC_WARP_PROFILE_TANH);
    bool ok_hyperexp = run_complex_case(ANALYTIC_WARP_PROFILE_HYPEREXP);

    if (!ok_digamma || !ok_trigamma || !ok_power || !ok_tanh || !ok_hyperexp)
        return 1;

    printf("[PASS] analytic warp complex digamma/trigamma/power/tanh/hyperexp\n");
    return 0;
}
