/*
 * Migrated advection operator coverage for analytic warp real-valued contracts.
 */
#include <oakfield/sim.h>

#include <float.h>
#include <math.h>
#include <oakfield/math/special_functions.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const double kHyperexpEpsilon = 0.75;
static const int kHyperexpDepth = 6;
static const double kQHyperexpQ = 0.9;

static double approx_digamma(double x) {
    double result = 0.0;
    double value = x;

    if (!isfinite(value)) {
        return NAN;
    }

    if (value <= 0.0) {
        double frac = value - floor(value);
        if (fabs(frac) < DBL_EPSILON) {
            return NAN;
        }
        double sin_px = sin(M_PI * value);
        double cos_px = cos(M_PI * value);
        if (fabs(sin_px) < DBL_EPSILON) {
            return NAN;
        }
        return approx_digamma(1.0 - value) - M_PI * (cos_px / sin_px);
    }

    while (value < 8.0) {
        result -= 1.0 / value;
        value += 1.0;
    }

    {
        double inv = 1.0 / value;
        double inv2 = inv * inv;
        double series =
            inv2 * (-1.0 / 12.0 +
                    inv2 * (1.0 / 120.0 +
                            inv2 * (-1.0 / 252.0 +
                                    inv2 * (1.0 / 240.0 +
                                            inv2 * (-1.0 / 132.0 + inv2 * (691.0 / 32760.0))))));

        result += log(value) - 0.5 * inv + series;
    }

    return result;
}

static double approx_trigamma(double x) {
    double result = 0.0;
    double value = x;

    if (!isfinite(value)) {
        return NAN;
    }

    if (value <= 0.0) {
        double sin_px = sin(M_PI * value);
        if (fabs(sin_px) < DBL_EPSILON) {
            return NAN;
        }
        double csc_sq = (M_PI / sin_px);
        csc_sq *= csc_sq;
        return csc_sq - approx_trigamma(1.0 - value);
    }

    while (value < 8.0) {
        result += 1.0 / (value * value);
        value += 1.0;
    }

    {
        double inv = 1.0 / value;
        double inv2 = inv * inv;
        double inv3 = inv2 * inv;
        double inv5 = inv3 * inv2;
        double inv7 = inv5 * inv2;
        double inv9 = inv7 * inv2;
        double inv11 = inv9 * inv2;
        double inv13 = inv11 * inv2;

        result += inv + 0.5 * inv2 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0 +
                  (5.0 * inv11) / 66.0 - (691.0 * inv13) / 2730.0;
    }

    return result;
}

static double approx_profile(AnalyticWarpProfile profile, double x) {
    switch (profile) {
    case ANALYTIC_WARP_PROFILE_TRIGAMMA:
        return sim_special_trigamma(x);
    case ANALYTIC_WARP_PROFILE_DIGAMMA:
    default:
        return sim_special_digamma(x);
    }
}

static double approx_profile_derivative(AnalyticWarpProfile profile, double x) {
    switch (profile) {
    case ANALYTIC_WARP_PROFILE_TRIGAMMA:
        /* derivative of trigamma = polygamma(2, x), if available;
       fall back to finite difference if not implemented yet */
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
        return sim_hyperexp_phi_deriv(x, kHyperexpEpsilon, kHyperexpDepth);
    case ANALYTIC_WARP_PROFILE_QHYPEREXP:
        return sim_qhyperexp_phi_deriv(x, kHyperexpEpsilon, kHyperexpDepth, kQHyperexpQ);

    default:
        return NAN;
    }
}

static bool run_profile_case(AnalyticWarpProfile profile) {
    SimContext context = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    double initial_data[8] = {0};
    double *data = NULL;
    const double dt = 0.05;
    const double delta = 0.25;
    const double lambda = 0.2;
    const double bias = 0.5;
    SimResult result;
    bool field_attached = false;
    bool ok = false;
    size_t i;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_field_init failed\n");
        goto cleanup;
    }

    data = (double *)sim_field_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] sim_field_data NULL\n");
        goto cleanup;
    }

    for (i = 0U; i < shape[0]; ++i) {
        initial_data[i] = 0.1 * (double)i;
        data[i] = initial_data[i];
    }

    if (sim_context_add_field(&context, &field, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_add_field failed\n");
        goto cleanup;
    }
    field_attached = true;
    field = (SimField){0};

    data = (double *)sim_field_data(sim_context_field(&context, 0U));
    if (data == NULL) {
        fprintf(stderr, "[FAIL] sim_field_data NULL after attach\n");
        goto cleanup;
    }

    sim_context_set_timestep(&context, (float)dt);

    {
        AnalyticWarpOperatorConfig cfg = {0};
        cfg.field_index = 0U;
        cfg.profile = profile;
        cfg.delta = delta;
        cfg.lambda = lambda;
        cfg.bias = bias;
        if (profile == ANALYTIC_WARP_PROFILE_HYPEREXP ||
            profile == ANALYTIC_WARP_PROFILE_QHYPEREXP) {
            cfg.hyperexp_epsilon = kHyperexpEpsilon;
            cfg.hyperexp_depth = kHyperexpDepth;
            cfg.hyperexp_q = kQHyperexpQ;
        }
        result = sim_add_analytic_warp_operator(&context, &cfg, NULL);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] sim_add_analytic_warp_operator failed (%d)\n", (int)result);
            goto cleanup;
        }
    }

    result = sim_context_prepare_plan(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_prepare_plan failed (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_execute failed (%d)\n", (int)result);
        goto cleanup;
    }

    for (i = 0U; i < shape[0]; ++i) {
        double sample = initial_data[i] + bias;
        double expected = initial_data[i];
        double gradient = approx_profile_derivative(profile, sample);
        if (isfinite(gradient)) {
            expected += dt * lambda * (2.0 * delta) * gradient;
        }
        if (fabs(data[i] - expected) > 1.0e-8 * fmax(1.0, fabs(expected))) {
            fprintf(stderr, "[FAIL] profile %d mismatch at %zu: got %.12g expected %.12g\n",
                    (int)profile, i, data[i], expected);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (field_attached) {
        sim_context_destroy(&context);
    } else {
        if (field.data != NULL) {
            sim_field_destroy(&field);
        }
        sim_context_destroy(&context);
    }

    return ok;
}

int main(void) {
    bool ok_digamma = run_profile_case(ANALYTIC_WARP_PROFILE_DIGAMMA);
    bool ok_trigamma = run_profile_case(ANALYTIC_WARP_PROFILE_TRIGAMMA);
    bool ok_hyperexp = run_profile_case(ANALYTIC_WARP_PROFILE_HYPEREXP);
    bool ok_qhyperexp = run_profile_case(ANALYTIC_WARP_PROFILE_QHYPEREXP);

    if (!ok_digamma || !ok_trigamma || !ok_hyperexp || !ok_qhyperexp) {
        return 1;
    }

    printf("[PASS] analytic warp operator digamma/trigamma/hyperexp/qhyperexp\n");
    return 0;
}
