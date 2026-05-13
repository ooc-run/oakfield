/*
 * Migrated reaction operator coverage for real-to-complex remainder contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>

static double remainder_eval(SimRemainderNonlinearity mode, double value, double exponent,
                             double epsilon) {
    double magnitude;
    switch (mode) {
    case SIM_REMAINDER_NONLINEARITY_ABS:
        return fabs(value);
    case SIM_REMAINDER_NONLINEARITY_LOG_ABS:
        magnitude = fabs(value);
        return log1p(magnitude / epsilon);
    case SIM_REMAINDER_NONLINEARITY_POWER:
        magnitude = pow(fabs(value) + epsilon, exponent);
        return copysign(magnitude, value);
    case SIM_REMAINDER_NONLINEARITY_TANH:
        return tanh(value);
    case SIM_REMAINDER_NONLINEARITY_IDENTITY:
    default:
        return value;
    }
}

static int nearly(double a, double b, double eps) {
    double d = fabs(a - b);
    double s = fmax(fabs(a), fabs(b));
    if (s < eps)
        s = 1.0;
    return d <= eps * s;
}

int main(void) {
    const size_t N = 10U;
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

    /* Only promote output: exercising the real->complex remainder branch. */
    if (sim_field_promote_inplace_to_complex(&output) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promote output\n");
        sim_field_destroy(&warped);
        sim_field_destroy(&reference);
        sim_field_destroy(&output);
        sim_context_destroy(&ctx);
        return 1;
    }

    double *wdata = (double *)sim_field_data(&warped);
    double *rdata = (double *)sim_field_data(&reference);
    for (size_t i = 0; i < N; ++i) {
        wdata[i] = 0.1 * (double)(i + 1);
        rdata[i] = 0.05 * (double)(i + 1);
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
    cfg.weight = 0.75;
    cfg.bias = -0.1;
    cfg.exponent = 1.5;
    cfg.epsilon = 1.0e-6;
    cfg.nonlinearity = SIM_REMAINDER_NONLINEARITY_POWER;
    cfg.accumulate = false;
    cfg.complex_mode = SIM_REMAINDER_COMPLEX_MODE_COMPONENT;

    if (sim_add_remainder_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add remainder\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimField *out_field = sim_context_field(&ctx, oi);
    const SimComplexDouble *out = sim_field_complex_data_const(out_field);
    if (!out) {
        fprintf(stderr, "[FAIL] output data\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    double baseline[10] = {0};
    for (size_t i = 0; i < N; ++i) {
        double ew = remainder_eval(cfg.nonlinearity, wdata[i], cfg.exponent, cfg.epsilon);
        double er = remainder_eval(cfg.nonlinearity, rdata[i], cfg.exponent, cfg.epsilon);
        double expect = (ew - er) * cfg.weight + cfg.bias;
        baseline[i] = expect;
        if (!nearly(out[i].re, expect, 1e-12) || !nearly(out[i].im, 0.0, 1e-12)) {
            fprintf(stderr, "[FAIL] overwrite i=%zu got=(%.12g,%.12g) exp=(%.12g,0)\n", i,
                    out[i].re, out[i].im, expect);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    /* Now accumulate onto existing output to exercise accumulate=true path. */
    cfg.accumulate = true;
    cfg.weight = 1.25;
    cfg.bias = 0.2;
    if (sim_remainder_update(&ctx, 0U, &cfg) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] update remainder\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute 2\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (size_t i = 0; i < N; ++i) {
        double ew = remainder_eval(cfg.nonlinearity, wdata[i], cfg.exponent, cfg.epsilon);
        double er = remainder_eval(cfg.nonlinearity, rdata[i], cfg.exponent, cfg.epsilon);
        double incr = (ew - er) * cfg.weight + cfg.bias;
        double expect = baseline[i] + incr;
        if (!nearly(out[i].re, expect, 1e-12)) {
            fprintf(stderr, "[FAIL] accumulate i=%zu got=%.12g exp=%.12g incr=%.12g\n", i,
                    out[i].re, expect, incr);
            sim_context_destroy(&ctx);
            return 1;
        }
        if (!nearly(out[i].im, 0.0, 1e-12)) {
            fprintf(stderr, "[FAIL] accumulate imag i=%zu got=%.12g\n", i, out[i].im);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    sim_context_destroy(&ctx);
    printf("[PASS] remainder real->complex power + accumulate\n");
    return 0;
}
