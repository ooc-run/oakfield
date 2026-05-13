/*
 * Migrated measurement operator coverage for complex sieve contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>

static int nearly(double a, double b, double eps) {
    double d = fabs(a - b);
    double s = fmax(fabs(a), fabs(b));
    if (s < eps)
        s = 1.0;
    return d <= eps * s;
}

static void build_kernel(double *kernel, unsigned int taps, double sigma) {
    unsigned int radius = taps / 2U;
    double sum = 0.0;
    for (unsigned int k = 0U; k < taps; ++k) {
        int offset = (int)k - (int)radius;
        double x = (double)offset;
        double value = exp(-(x * x) / (2.0 * sigma * sigma));
        kernel[k] = value;
        sum += value;
    }
    if (sum <= 0.0) {
        for (unsigned int k = 0U; k < taps; ++k) {
            kernel[k] = 1.0 / (double)taps;
        }
    } else {
        double inv = 1.0 / sum;
        for (unsigned int k = 0U; k < taps; ++k) {
            kernel[k] *= inv;
        }
    }
}

int main(void) {
    const size_t N = 8U;
    size_t shape[1] = {N};
    SimContext ctx = {0};
    SimField input = {0}, output = {0};
    size_t input_idx = SIZE_MAX, output_idx = SIZE_MAX;
    size_t op_idx = SIZE_MAX;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] ctx init\n");
        return 1;
    }

    if (sim_field_init(&input, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] field init\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimComplexDouble *idata = sim_field_complex_data(&input);
    for (size_t i = 0; i < N; ++i) {
        idata[i].re = 1.0 + (double)i;
        idata[i].im = 0.5 * (double)i;
    }

    SimComplexDouble *odata_init = sim_field_complex_data(&output);
    if (odata_init != NULL) {
        for (size_t i = 0; i < N; ++i) {
            odata_init[i].re = 0.0;
            odata_init[i].im = 0.0;
        }
    }

    if (sim_context_add_field(&ctx, &input, &input_idx) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &output_idx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add fields\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimSieveOperatorConfig cfg = {0};
    cfg.input_field = input_idx;
    cfg.output_field = output_idx;
    cfg.taps = 5U;
    cfg.sigma = 1.0;
    cfg.gain = 1.0;
    cfg.mode = SIM_SIEVE_MODE_LOW_PASS;
    cfg.accumulate = false;

    if (sim_add_sieve_operator(&ctx, &cfg, &op_idx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add sieve\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimField *out_field = sim_context_field(&ctx, output_idx);
    if (!sim_field_is_complex(out_field)) {
        fprintf(stderr, "[FAIL] output not complex\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    const SimComplexDouble *odata = sim_field_complex_data_const(out_field);
    if (!odata) {
        fprintf(stderr, "[FAIL] output data\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    double kernel[5] = {0};
    build_kernel(kernel, cfg.taps, cfg.sigma);

    /* Validate convolution matches the kernel used by the operator (clamped edges). */
    for (size_t i = 0; i < N; ++i) {
        double accum_re = 0.0, accum_im = 0.0;
        unsigned int radius = cfg.taps / 2U;
        for (unsigned int k = 0; k < cfg.taps; ++k) {
            int offset = (int)i + (int)k - (int)radius;
            if (offset < 0)
                offset = 0;
            else if (offset >= (int)N)
                offset = (int)N - 1;
            accum_re += kernel[k] * idata[(size_t)offset].re;
            accum_im += kernel[k] * idata[(size_t)offset].im;
        }

        if (!nearly(odata[i].re, accum_re, 1e-9) || !nearly(odata[i].im, accum_im, 1e-9)) {
            fprintf(stderr,
                    "[FAIL] convolution mismatch i=%zu got=(%.12g,%.12g) exp=(%.12g,%.12g)\n", i,
                    odata[i].re, odata[i].im, accum_re, accum_im);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    sim_context_destroy(&ctx);
    printf("[PASS] sieve complex promotion + convolution\n");
    return 0;
}
