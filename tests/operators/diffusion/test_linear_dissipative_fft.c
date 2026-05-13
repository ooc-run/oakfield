/*
 * Migrated diffusion operator coverage for FFT-backed dissipative contracts.
 */
#include <oakfield/sim.h>

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static void forward_dft(const double *data, size_t count, double complex *out) {
    for (size_t k = 0U; k < count; ++k) {
        double complex sum = 0.0 + 0.0 * I;
        for (size_t n = 0U; n < count; ++n) {
            double angle = -2.0 * M_PI * (double)k * (double)n / (double)count;
            double complex w = cos(angle) + I * sin(angle);
            sum += data[n] * w;
        }
        out[k] = sum;
    }
}

static void inverse_dft(const double complex *freq, size_t count, double *out) {
    for (size_t n = 0U; n < count; ++n) {
        double complex sum = 0.0 + 0.0 * I;
        for (size_t k = 0U; k < count; ++k) {
            double angle = 2.0 * M_PI * (double)k * (double)n / (double)count;
            double complex w = cos(angle) + I * sin(angle);
            sum += freq[k] * w;
        }
        out[n] = creal(sum) / (double)count;
    }
}

static void compute_lambdas(double alpha, double viscosity, double spacing, size_t count,
                            double *out) {
    double length = spacing * (double)count;
    double base_k = (count > 0U && length > 0.0) ? (2.0 * M_PI / length) : 0.0;

    for (size_t i = 0U; i < count; ++i) {
        double freq_index = (i <= count / 2U) ? (double)i : -((double)(count - i));
        double k_abs = fabs(freq_index * base_k);
        double value = 0.0;

        if (k_abs > 0.0 && alpha > 0.0) {
            value = -pow(k_abs, alpha);
        }

        out[i] = viscosity * value;
    }
}

static double energy_l2(const double *data, size_t count) {
    double sum = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        sum += data[i] * data[i];
    }
    return sum;
}

static bool setup_linear_dissipative_context(SimContext *ctx, SimField *field, const size_t *shape,
                                             size_t count, const double *initial, double dt) {
    if (sim_context_init(ctx) != SIM_RESULT_OK)
        return false;
    if (sim_field_init(field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK)
        return false;
    SimComplexDouble *data = sim_field_complex_data(field);
    if (!data)
        return false;
    for (size_t i = 0U; i < count; ++i) {
        data[i].re = initial[i];
        data[i].im = 0.0;
    }
    if (sim_context_add_field(ctx, field, NULL) != SIM_RESULT_OK)
        return false;
    *field = (SimField){0}; /* ownership moved */
    sim_context_set_timestep(ctx, dt);
    LinearDissipativeOperatorConfig cfg = {
        .field_index = 0U, .alpha = 2.0, .viscosity = 0.5, .spacing = 1.0};
    if (sim_add_linear_dissipative_operator(ctx, &cfg, NULL) != SIM_RESULT_OK)
        return false;
    if (sim_context_prepare_plan(ctx) != SIM_RESULT_OK)
        return false;
    return true;
}

static bool run_linear_dissipative_rate_consistency(size_t count) {
    /* Compare solutions at fixed physical time for dt vs dt/2 */
    const double dt0 = 0.01;
    const double total_time = 0.64; /* 64 steps at dt0 */
    const size_t steps0 = (size_t)(total_time / dt0);
    const double dt1 = dt0 * 0.5;
    const size_t steps1 = (size_t)(total_time / dt1);

    SimComplexDouble *out0 = NULL;
    SimComplexDouble *out1 = NULL;
    bool ok = false;

    SimContext ctx0 = {0}, ctx1 = {0};
    SimField f0 = {0}, f1 = {0};
    size_t shape[1] = {count};

    double *init = (double *)calloc(count, sizeof(double));
    if (!init)
        goto cleanup;

    for (size_t i = 0U; i < count; ++i) {
        double x = (double)i / (double)count;
        init[i] = sin(2.0 * M_PI * x) + 0.5 * cos(4.0 * M_PI * x);
    }

    if (!setup_linear_dissipative_context(&ctx0, &f0, shape, count, init, dt0))
        goto cleanup;
    if (!setup_linear_dissipative_context(&ctx1, &f1, shape, count, init, dt1))
        goto cleanup;

    for (size_t s = 0U; s < steps0; ++s) {
        if (sim_context_execute(&ctx0) != SIM_RESULT_OK)
            goto cleanup;
    }
    for (size_t s = 0U; s < steps1; ++s) {
        if (sim_context_execute(&ctx1) != SIM_RESULT_OK)
            goto cleanup;
    }

    SimField *c0 = sim_context_field(&ctx0, 0U);
    SimField *c1 = sim_context_field(&ctx1, 0U);
    if (!c0 || !c1)
        goto cleanup;
    out0 = sim_field_complex_data(c0);
    out1 = sim_field_complex_data(c1);
    if (!out0 || !out1)
        goto cleanup;

    double max_err = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double err_re = fabs(out0[i].re - out1[i].re);
        double err_im = fabs(out0[i].im - out1[i].im);
        double err = fmax(err_re, err_im);
        if (err > max_err)
            max_err = err;
    }

    if (max_err > 1e-6) {
        fprintf(stderr, "[FAIL] rate consistency mismatch (dt=%.3g vs %.3g): max_err=%.6g\n", dt0,
                dt1, max_err);
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx0);
    sim_context_destroy(&ctx1);
    free(init);
    return ok;
}

static bool run_linear_dissipative_case(size_t count) {
    SimContext context = {0};
    SimField field = {0};
    size_t shape[1] = {count};
    double *initial = NULL;
    double *expected = NULL;
    SimComplexDouble *data = NULL;
    double complex *freq = NULL;
    double *lambda = NULL;
    const double dt = 0.01;
    const size_t steps = 64U;
    bool field_attached = false;
    bool ok = false;

    if (count == 0U) {
        return false;
    }

    initial = (double *)calloc(count, sizeof(double));
    expected = (double *)calloc(count, sizeof(double));
    freq = (double complex *)calloc(count, sizeof(double complex));
    lambda = (double *)calloc(count, sizeof(double));
    if (!initial || !expected || !freq || !lambda) {
        goto cleanup;
    }

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        goto cleanup;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }

    data = sim_field_complex_data(&field);
    if (data == NULL) {
        goto cleanup;
    }

    for (size_t i = 0U; i < count; ++i) {
        double x = (double)i / (double)count;
        double value =
            sin(2.0 * M_PI * x) + 0.25 * cos(4.0 * M_PI * x) + 0.125 * sin(6.0 * M_PI * x);
        data[i].re = value;
        data[i].im = 0.0;
        initial[i] = value;
    }

    if (sim_context_add_field(&context, &field, NULL) != SIM_RESULT_OK) {
        goto cleanup;
    }
    field_attached = true;
    field = (SimField){0};

    data = sim_field_complex_data(sim_context_field(&context, 0U));
    if (!data) {
        goto cleanup;
    }

    sim_context_set_timestep(&context, (float)dt);

    {
        LinearDissipativeOperatorConfig cfg = {0};
        cfg.field_index = 0U;
        cfg.alpha = 2.0;
        cfg.viscosity = 0.5;
        cfg.spacing = 1.0;

        if (sim_add_linear_dissipative_operator(&context, &cfg, NULL) != SIM_RESULT_OK) {
            goto cleanup;
        }
    }

    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK) {
        goto cleanup;
    }

    double initial_energy = energy_l2(initial, count);

    for (size_t step = 0U; step < steps; ++step) {
        if (sim_context_execute(&context) != SIM_RESULT_OK) {
            goto cleanup;
        }
    }

    /* Field is now complex after operator applied; recompute final energy */
    SimField *field_after = sim_context_field(&context, 0U);
    if (!field_after) {
        goto cleanup;
    }

    double final_energy = 0.0;
    if (field_after->element_size == sizeof(SimComplexDouble)) {
        SimComplexDouble *cdata_after = sim_field_complex_data(field_after);
        for (size_t i = 0U; i < count; ++i) {
            final_energy +=
                cdata_after[i].re * cdata_after[i].re + cdata_after[i].im * cdata_after[i].im;
        }
    } else {
        double *rdata_after = sim_field_real_data(field_after);
        if (rdata_after == NULL) {
            goto cleanup;
        }
        final_energy = energy_l2(rdata_after, count);
    }
    if (!(final_energy <= initial_energy + 1.0e-9)) {
        fprintf(stderr, "[FAIL] energy increased for count=%zu (%.12g -> %.12g)\n", count,
                initial_energy, final_energy);
        goto cleanup;
    }

    forward_dft(initial, count, freq);
    compute_lambdas(2.0, 0.5, 1.0, count, lambda);

    for (size_t k = 0U; k < count; ++k) {
        double factor = exp(dt * lambda[k] * (double)steps);
        freq[k] *= factor;
    }

    inverse_dft(freq, count, expected);

    double max_error = 0.0;
    if (field_after->element_size == sizeof(SimComplexDouble)) {
        SimComplexDouble *cdata_after = sim_field_complex_data(field_after);
        for (size_t i = 0U; i < count; ++i) {
            /* Expected is real-valued after dissipation of a real initial condition */
            double err_re = fabs(cdata_after[i].re - expected[i]);
            double err_im = fabs(cdata_after[i].im); /* Should be near zero */
            double err = fmax(err_re, err_im);
            if (err > max_error) {
                max_error = err;
            }
        }
    } else {
        double *rdata_after = sim_field_real_data(field_after);
        if (rdata_after == NULL) {
            goto cleanup;
        }
        for (size_t i = 0U; i < count; ++i) {
            double err = fabs(rdata_after[i] - expected[i]);
            if (err > max_error) {
                max_error = err;
            }
        }
    }

    if (max_error > 1.0e-7 * fmax(1.0, initial_energy)) {
        fprintf(stderr, "[FAIL] spectral mismatch for count=%zu (max_error=%.12g)\n", count,
                max_error);
        goto cleanup;
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

    free(initial);
    free(expected);
    free(freq);
    free(lambda);

    return ok;
}

static bool run_linear_dissipative_2d_case(bool spectral) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {6U, 5U};
    size_t field_index = 0U;
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    const size_t kx = 1U;
    const size_t ky = 1U;
    const double dt = 0.1;
    const double viscosity = 0.5;
    const double alpha = 2.0;
    const double spacing = 1.0;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dissipative_2d: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dissipative_2d: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] dissipative_2d: data missing\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    if (spectral) {
        field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
        field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        for (size_t i = 0U; i < rows * cols; ++i) {
            data[i].re = 0.0;
            data[i].im = 0.0;
        }
        size_t idx = ky * cols + kx;
        data[idx].re = 1.0;
        data[idx].im = 0.0;
    } else {
        for (size_t y = 0U; y < rows; ++y) {
            for (size_t x = 0U; x < cols; ++x) {
                double angle =
                    2.0 * M_PI *
                    ((double)kx * (double)x / (double)cols + (double)ky * (double)y / (double)rows);
                size_t idx = y * cols + x;
                data[idx].re = cos(angle);
                data[idx].im = sin(angle);
            }
        }
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dissipative_2d: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    LinearDissipativeOperatorConfig cfg = {
        .field_index = field_index, .alpha = alpha, .viscosity = viscosity, .spacing = spacing};
    if (sim_add_linear_dissipative_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dissipative_2d: add operator failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dissipative_2d: execute failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    SimComplexDouble *out = sim_field_complex_data(out_field);
    if (out == NULL) {
        fprintf(stderr, "[FAIL] dissipative_2d: output missing\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double base_kx = (2.0 * M_PI) / ((double)cols * spacing);
    double base_ky = (2.0 * M_PI) / ((double)rows * spacing);
    double kx_val = base_kx * (double)kx;
    double ky_val = base_ky * (double)ky;
    double k_abs = sqrt(kx_val * kx_val + ky_val * ky_val);
    double lambda = viscosity * -pow(k_abs, alpha);
    double factor = exp(dt * lambda);

    if (spectral) {
        size_t idx = ky * cols + kx;
        SimComplexDouble expected = {factor, 0.0};
        if (fabs(out[idx].re - expected.re) > 1.0e-6 || fabs(out[idx].im - expected.im) > 1.0e-6) {
            fprintf(stderr, "[FAIL] dissipative_2d_spectral: mismatch got (%.9f, %.9f)\n",
                    out[idx].re, out[idx].im);
            goto cleanup;
        }
    } else {
        for (size_t y = 0U; y < rows; ++y) {
            for (size_t x = 0U; x < cols; ++x) {
                size_t idx = y * cols + x;
                double angle =
                    2.0 * M_PI *
                    ((double)kx * (double)x / (double)cols + (double)ky * (double)y / (double)rows);
                double exp_re = cos(angle) * factor;
                double exp_im = sin(angle) * factor;
                if (fabs(out[idx].re - exp_re) > 1.0e-6 || fabs(out[idx].im - exp_im) > 1.0e-6) {
                    fprintf(stderr, "[FAIL] dissipative_2d_physical: mismatch at (%zu,%zu)\n", y,
                            x);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_linear_dissipative_imag_zero_constraint_case(void) {
    SimContext context = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    double initial[8];
    double expected[8];
    double complex freq[8];
    double lambda[8];
    const double dt = 0.02;
    const size_t steps = 5U;
    bool ok = false;

    memset(expected, 0, sizeof(expected));
    memset(freq, 0, sizeof(freq));
    memset(lambda, 0, sizeof(lambda));

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: field init failed\n");
        sim_context_destroy(&context);
        return false;
    }

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: field data missing\n");
        sim_field_destroy(&field);
        sim_context_destroy(&context);
        return false;
    }

    for (size_t i = 0U; i < shape[0]; ++i) {
        double x = (double)i / (double)shape[0];
        initial[i] = cos(2.0 * M_PI * x) + 0.4 * sin(4.0 * M_PI * x);
        data[i].re = initial[i];
        data[i].im = 0.0;
    }
    field.repr.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    if (sim_context_add_field(&context, &field, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&context);
        return false;
    }

    sim_context_set_timestep(&context, (float)dt);

    {
        LinearDissipativeOperatorConfig cfg = {
            .field_index = 0U, .alpha = 2.0, .viscosity = 0.5, .spacing = 1.0};
        if (sim_add_linear_dissipative_operator(&context, &cfg, NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] imag_zero_dissipative: add operator failed\n");
            sim_context_destroy(&context);
            return false;
        }
    }

    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: prepare plan failed\n");
        sim_context_destroy(&context);
        return false;
    }

    for (size_t step = 0U; step < steps; ++step) {
        if (sim_context_execute(&context) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] imag_zero_dissipative: execute failed at step %zu\n", step);
            sim_context_destroy(&context);
            return false;
        }
    }

    SimField *field_after = sim_context_field(&context, 0U);
    if (field_after == NULL) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: output field missing\n");
        sim_context_destroy(&context);
        return false;
    }

    SimFieldRepresentation repr = sim_field_representation(field_after);
    if (repr.domain != SIM_FIELD_DOMAIN_PHYSICAL ||
        repr.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: representation not preserved (%d, %d)\n",
                (int)repr.domain, (int)repr.value_kind);
        sim_context_destroy(&context);
        return false;
    }

    data = sim_field_complex_data(field_after);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] imag_zero_dissipative: output data missing\n");
        sim_context_destroy(&context);
        return false;
    }

    forward_dft(initial, shape[0], freq);
    compute_lambdas(2.0, 0.5, 1.0, shape[0], lambda);
    for (size_t k = 0U; k < shape[0]; ++k) {
        double factor = exp(dt * lambda[k] * (double)steps);
        freq[k] *= factor;
    }
    inverse_dft(freq, shape[0], expected);

    for (size_t i = 0U; i < shape[0]; ++i) {
        if (fabs(data[i].re - expected[i]) > 1.0e-7) {
            fprintf(
                stderr,
                "[FAIL] imag_zero_dissipative: real mismatch at %zu (got %.12g expected %.12g)\n",
                i, data[i].re, expected[i]);
            sim_context_destroy(&context);
            return false;
        }
        if (data[i].im != 0.0) {
            fprintf(stderr,
                    "[FAIL] imag_zero_dissipative: imaginary lane should remain zero at %zu (got "
                    "%.12g)\n",
                    i, data[i].im);
            sim_context_destroy(&context);
            return false;
        }
    }

    ok = true;
    sim_context_destroy(&context);
    return ok;
}

int main(void) {
    bool ok_pow2 = run_linear_dissipative_case(8U);
    bool ok_generic = run_linear_dissipative_case(6U);
    bool ok_rate = run_linear_dissipative_rate_consistency(8U);
    bool ok_2d_spec = run_linear_dissipative_2d_case(true);
    bool ok_2d_phys = run_linear_dissipative_2d_case(false);
    bool ok_imag_zero = run_linear_dissipative_imag_zero_constraint_case();

    if (!ok_pow2 || !ok_generic || !ok_rate || !ok_2d_spec || !ok_2d_phys || !ok_imag_zero) {
        return 1;
    }

    printf("[PASS] linear dissipative FFT spectral accuracy and rate consistency\n");
    return 0;
}
