#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool nearly_equal(double a, double b, double rel_tol, double abs_tol) {
    double diff = fabs(a - b);
    double scale = fmax(fabs(a), fabs(b));
    if (scale < 1.0) {
        scale = 1.0;
    }
    return diff <= fmax(abs_tol, rel_tol * scale);
}

static bool complex_nearly_equal(SimComplexDouble a, SimComplexDouble b, double rel_tol,
                                 double abs_tol) {
    return nearly_equal(a.re, b.re, rel_tol, abs_tol) && nearly_equal(a.im, b.im, rel_tol, abs_tol);
}

static double energy_real(const double *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        sum += data[i] * data[i];
    }
    return sum;
}

static double energy_complex(const SimComplexDouble *data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        sum += data[i].re * data[i].re + data[i].im * data[i].im;
    }
    return sum;
}

static bool run_forward_hermitian_case(size_t n) {
    SimContext ctx = {0};
    SimField input = {0};
    bool input_ready = false;
    bool ctx_ready = false;
    size_t shape[1] = {n};
    size_t input_index = 0U;
    size_t spec_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: ctx init failed\n");
        goto cleanup;
    }
    ctx_ready = true;

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: field init failed\n");
        goto cleanup;
    }
    input_ready = true;

    double *x = sim_field_real_data(&input);
    if (x == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: input data NULL\n");
        goto cleanup;
    }

    /* Deterministic real signal. */
    for (size_t i = 0U; i < n; ++i) {
        double t = (double)i / (double)n;
        x[i] = 0.75 * cos(2.0 * M_PI * t) + 0.25 * sin(4.0 * M_PI * t) + 0.1 * cos(6.0 * M_PI * t);
    }

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: add field failed\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_add_fft_convert(&ctx, input_index, SIM_FFT_CONVERT_FORWARD, false, &spec_index,
                            &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: add fft_convert failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: execute failed\n");
        goto cleanup;
    }

    SimField *spec_field = sim_context_field(&ctx, spec_index);
    if (spec_field == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: spectral field missing\n");
        goto cleanup;
    }

    SimFieldRepresentation repr = sim_field_representation(spec_field);
    if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
        repr.value_kind != SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT) {
        fprintf(stderr,
                "[FAIL] fft_forward_hermitian: expected spectral+real-constraint, got domain=%d "
                "kind=%d\n",
                (int)repr.domain, (int)repr.value_kind);
        goto cleanup;
    }

    const SimComplexDouble *X = sim_field_complex_data_const(spec_field);
    if (X == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: spectral data NULL\n");
        goto cleanup;
    }

    /* Hermitian symmetry: X[n-k] == conj(X[k]). */
    const double rel_tol = 5e-12;
    const double abs_tol = 5e-12;
    for (size_t k = 1U; k < n; ++k) {
        size_t nk = (n - k) % n;
        SimComplexDouble lhs = X[nk];
        SimComplexDouble rhs = (SimComplexDouble){.re = X[k].re, .im = -X[k].im};
        if (!complex_nearly_equal(lhs, rhs, rel_tol, abs_tol)) {
            fprintf(stderr,
                    "[FAIL] fft_forward_hermitian: symmetry mismatch n=%zu k=%zu got(%.17g,%.17g) "
                    "expected(%.17g,%.17g)\n",
                    n, k, lhs.re, lhs.im, rhs.re, rhs.im);
            goto cleanup;
        }
    }

    /* DC component should be (near) real for real input. */
    if (!nearly_equal(X[0].im, 0.0, 1e-12, 1e-12)) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: DC imag too large (%.17g)\n", X[0].im);
        goto cleanup;
    }
    if ((n % 2U) == 0U) {
        size_t k = n / 2U;
        if (!nearly_equal(X[k].im, 0.0, 1e-12, 1e-12)) {
            fprintf(stderr, "[FAIL] fft_forward_hermitian: Nyquist imag too large (%.17g)\n",
                    X[k].im);
            goto cleanup;
        }
    }

    /* Parseval check for the normalization convention:
     * With forward unnormalized and inverse scaled by 1/n:
     *   sum_k |X_k|^2 ≈ n * sum_i |x_i|^2
     */
    const SimField *input_field = sim_context_field(&ctx, input_index);
    const double *x_after = input_field ? sim_field_real_data_const(input_field) : NULL;
    if (x_after == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian: input field data missing\n");
        goto cleanup;
    }
    double e_time = energy_real(x_after, n);
    double e_freq = energy_complex(X, n);
    double expected = (double)n * e_time;
    if (!nearly_equal(e_freq, expected, 5e-9, 1e-12)) {
        fprintf(stderr,
                "[FAIL] fft_forward_hermitian: parseval mismatch n=%zu time=%.17g freq=%.17g "
                "expected=%.17g\n",
                n, e_time, e_freq, expected);
        goto cleanup;
    }

    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    return ok;
}

static bool run_forward_hermitian_case_2d(size_t rows, size_t cols) {
    SimContext ctx = {0};
    SimField input = {0};
    bool input_ready = false;
    bool ctx_ready = false;
    size_t shape[2] = {rows, cols};
    size_t input_index = 0U;
    size_t spec_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: ctx init failed\n");
        goto cleanup;
    }
    ctx_ready = true;

    if (sim_field_init(&input, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: field init failed\n");
        goto cleanup;
    }
    input_ready = true;

    double *x = sim_field_real_data(&input);
    if (x == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: input data NULL\n");
        goto cleanup;
    }

    for (size_t y = 0U; y < rows; ++y) {
        for (size_t x_idx = 0U; x_idx < cols; ++x_idx) {
            double fx = (double)x_idx / (double)cols;
            double fy = (double)y / (double)rows;
            size_t idx = y * cols + x_idx;
            x[idx] = 0.5 + 0.25 * cos(2.0 * M_PI * fx) + 0.15 * sin(2.0 * M_PI * fy);
        }
    }

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: add field failed\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_add_fft_convert(&ctx, input_index, SIM_FFT_CONVERT_FORWARD, false, &spec_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: add fft_convert failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: execute failed\n");
        goto cleanup;
    }

    SimField *spec_field = sim_context_field(&ctx, spec_index);
    if (spec_field == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: spectral field missing\n");
        goto cleanup;
    }

    SimFieldRepresentation repr = sim_field_representation(spec_field);
    if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
        repr.value_kind != SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT) {
        fprintf(stderr,
                "[FAIL] fft_forward_hermitian_2d: expected spectral+real-constraint, got domain=%d "
                "kind=%d\n",
                (int)repr.domain, (int)repr.value_kind);
        goto cleanup;
    }

    const SimComplexDouble *X = sim_field_complex_data_const(spec_field);
    if (X == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: spectral data NULL\n");
        goto cleanup;
    }

    const double rel_tol = 5e-11;
    const double abs_tol = 5e-11;
    for (size_t y = 0U; y < rows; ++y) {
        size_t y_conj = (y == 0U) ? 0U : (rows - y);
        for (size_t x_idx = 0U; x_idx < cols; ++x_idx) {
            size_t x_conj = (x_idx == 0U) ? 0U : (cols - x_idx);
            size_t idx = y * cols + x_idx;
            size_t idx_conj = y_conj * cols + x_conj;
            SimComplexDouble lhs = X[idx_conj];
            SimComplexDouble rhs = (SimComplexDouble){.re = X[idx].re, .im = -X[idx].im};
            if (!complex_nearly_equal(lhs, rhs, rel_tol, abs_tol)) {
                fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: symmetry mismatch (%zu,%zu)\n", y,
                        x_idx);
                goto cleanup;
            }
        }
    }

    for (size_t y = 0U; y < rows; ++y) {
        size_t y_is_self = (y == 0U) || ((rows % 2U == 0U) && (y == rows / 2U));
        for (size_t x_idx = 0U; x_idx < cols; ++x_idx) {
            size_t x_is_self = (x_idx == 0U) || ((cols % 2U == 0U) && (x_idx == cols / 2U));
            if (y_is_self && x_is_self) {
                size_t idx = y * cols + x_idx;
                if (!nearly_equal(X[idx].im, 0.0, 1e-10, 1e-10)) {
                    fprintf(stderr,
                            "[FAIL] fft_forward_hermitian_2d: self-conjugate imag too large\n");
                    goto cleanup;
                }
            }
        }
    }

    const SimField *input_field = sim_context_field(&ctx, input_index);
    const double *x_after = input_field ? sim_field_real_data_const(input_field) : NULL;
    if (x_after == NULL) {
        fprintf(stderr, "[FAIL] fft_forward_hermitian_2d: input data missing\n");
        goto cleanup;
    }
    double e_time = energy_real(x_after, rows * cols);
    double e_freq = energy_complex(X, rows * cols);
    double expected = (double)(rows * cols) * e_time;
    if (!nearly_equal(e_freq, expected, 5e-8, 1e-10)) {
        fprintf(stderr,
                "[FAIL] fft_forward_hermitian_2d: parseval mismatch time=%.17g freq=%.17g "
                "expected=%.17g\n",
                e_time, e_freq, expected);
        goto cleanup;
    }

    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    return ok;
}

static bool run_roundtrip_case(size_t n) {
    SimContext ctx = {0};
    SimField input = {0};
    bool input_ready = false;
    bool ctx_ready = false;
    size_t shape[1] = {n};
    size_t input_index = 0U;
    size_t spec_index = 0U;
    size_t out_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip: ctx init failed\n");
        goto cleanup;
    }
    ctx_ready = true;

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip: input init failed\n");
        goto cleanup;
    }
    input_ready = true;

    double *x = sim_field_real_data(&input);
    if (x == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip: input data NULL\n");
        goto cleanup;
    }
    for (size_t i = 0U; i < n; ++i) {
        x[i] = 0.2 + 0.001 * (double)i + 0.3 * sin(2.0 * M_PI * (double)i / (double)n);
    }

    double baseline[64];
    if (n > sizeof(baseline) / sizeof(baseline[0])) {
        fprintf(stderr, "[FAIL] fft_roundtrip: test only supports n<=64\n");
        goto cleanup;
    }
    memcpy(baseline, x, n * sizeof(double));

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip: add input failed\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_add_fft_convert(&ctx, input_index, SIM_FFT_CONVERT_FORWARD, false, &spec_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip: add forward failed\n");
        goto cleanup;
    }
    if (sim_add_fft_convert(&ctx, spec_index, SIM_FFT_CONVERT_INVERSE, false, &out_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip: add inverse failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip: execute failed\n");
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, out_index);
    if (out_field == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip: output field missing\n");
        goto cleanup;
    }
    SimFieldRepresentation repr = sim_field_representation(out_field);
    if (repr.domain != SIM_FIELD_DOMAIN_PHYSICAL || out_field->element_size != sizeof(double)) {
        fprintf(stderr, "[FAIL] fft_roundtrip: output not physical real\n");
        goto cleanup;
    }

    const double *y = sim_field_real_data_const(out_field);
    if (y == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip: output data NULL\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < n; ++i) {
        if (!nearly_equal(y[i], baseline[i], 1e-10, 1e-12)) {
            fprintf(stderr, "[FAIL] fft_roundtrip: mismatch n=%zu i=%zu got=%.17g expected=%.17g\n",
                    n, i, y[i], baseline[i]);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    return ok;
}

static bool run_roundtrip_case_2d(size_t rows, size_t cols) {
    SimContext ctx = {0};
    SimField input = {0};
    bool input_ready = false;
    bool ctx_ready = false;
    size_t shape[2] = {rows, cols};
    size_t input_index = 0U;
    size_t spec_index = 0U;
    size_t out_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: ctx init failed\n");
        goto cleanup;
    }
    ctx_ready = true;

    if (sim_field_init(&input, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: input init failed\n");
        goto cleanup;
    }
    input_ready = true;

    double *x = sim_field_real_data(&input);
    if (x == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: input data NULL\n");
        goto cleanup;
    }

    size_t count = rows * cols;
    double *baseline = (double *)calloc(count, sizeof(double));
    if (baseline == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: baseline alloc failed\n");
        goto cleanup;
    }

    for (size_t y = 0U; y < rows; ++y) {
        for (size_t x_idx = 0U; x_idx < cols; ++x_idx) {
            double fx = (double)x_idx / (double)cols;
            double fy = (double)y / (double)rows;
            size_t idx = y * cols + x_idx;
            x[idx] = 0.1 + 0.2 * cos(2.0 * M_PI * fx) + 0.15 * sin(2.0 * M_PI * fy);
            baseline[idx] = x[idx];
        }
    }

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: add input failed\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_add_fft_convert(&ctx, input_index, SIM_FFT_CONVERT_FORWARD, false, &spec_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: add forward failed\n");
        goto cleanup;
    }
    if (sim_add_fft_convert(&ctx, spec_index, SIM_FFT_CONVERT_INVERSE, false, &out_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: add inverse failed\n");
        goto cleanup;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: execute failed\n");
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, out_index);
    if (out_field == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: output field missing\n");
        goto cleanup;
    }
    SimFieldRepresentation repr = sim_field_representation(out_field);
    if (repr.domain != SIM_FIELD_DOMAIN_PHYSICAL || out_field->element_size != sizeof(double)) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: output not physical real\n");
        goto cleanup;
    }

    const double *y = sim_field_real_data_const(out_field);
    if (y == NULL) {
        fprintf(stderr, "[FAIL] fft_roundtrip_2d: output data NULL\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < count; ++i) {
        if (!nearly_equal(y[i], baseline[i], 1e-10, 1e-12)) {
            fprintf(stderr, "[FAIL] fft_roundtrip_2d: mismatch i=%zu got=%.17g expected=%.17g\n", i,
                    y[i], baseline[i]);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    free(baseline);
    return ok;
}

static bool run_inverse_constraint_violation_case(bool exploration_mode) {
    SimContext ctx = {0};
    SimField input = {0};
    bool input_ready = false;
    bool ctx_ready = false;
    size_t shape[1] = {8U};
    size_t input_index = 0U;
    size_t out_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint: ctx init failed\n");
        return false;
    }
    ctx_ready = true;

    sim_context_set_representation_mode(&ctx, exploration_mode ? SIM_REPRESENTATION_MODE_EXPLORATION
                                                               : SIM_REPRESENTATION_MODE_STRICT);

    if (sim_field_init(&input, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint: field init failed\n");
        goto cleanup;
    }
    input_ready = true;

    input.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    input.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;

    SimComplexDouble *X = sim_field_complex_data(&input);
    if (X == NULL) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint: input data NULL\n");
        goto cleanup;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        X[i].re = 0.0;
        X[i].im = 0.0;
    }
    X[1].re = 1.0;
    X[1].im = 0.25;
    X[shape[0] - 1U].re = 1.0;
    X[shape[0] - 1U].im = 0.25;

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint: add field failed\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_add_fft_convert(&ctx, input_index, SIM_FFT_CONVERT_INVERSE, false, &out_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint: add fft_convert failed\n");
        goto cleanup;
    }

    SimResult exec_rc = sim_context_execute(&ctx);
    if (!exploration_mode) {
        if (exec_rc != SIM_RESULT_INVALID_STATE) {
            fprintf(stderr,
                    "[FAIL] fft_inverse_constraint: strict mode expected INVALID_STATE, got %d\n",
                    exec_rc);
            goto cleanup;
        }
    } else {
        if (exec_rc != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] fft_inverse_constraint: exploration execute failed (%d)\n",
                    exec_rc);
            goto cleanup;
        }
        SimField *out_field = sim_context_field(&ctx, out_index);
        if (out_field == NULL || out_field->element_size != sizeof(double)) {
            fprintf(stderr, "[FAIL] fft_inverse_constraint: exploration output missing/complex\n");
            goto cleanup;
        }
        const double *y = sim_field_real_data_const(out_field);
        if (y == NULL) {
            fprintf(stderr, "[FAIL] fft_inverse_constraint: exploration output data NULL\n");
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    return ok;
}

static bool run_inverse_constraint_violation_case_2d(bool exploration_mode) {
    SimContext ctx = {0};
    SimField input = {0};
    bool input_ready = false;
    bool ctx_ready = false;
    size_t shape[2] = {5U, 6U};
    size_t input_index = 0U;
    size_t out_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint_2d: ctx init failed\n");
        return false;
    }
    ctx_ready = true;

    sim_context_set_representation_mode(&ctx, exploration_mode ? SIM_REPRESENTATION_MODE_EXPLORATION
                                                               : SIM_REPRESENTATION_MODE_STRICT);

    if (sim_field_init(&input, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint_2d: field init failed\n");
        goto cleanup;
    }
    input_ready = true;

    input.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    input.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;

    SimComplexDouble *X = sim_field_complex_data(&input);
    if (X == NULL) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint_2d: input data NULL\n");
        goto cleanup;
    }
    for (size_t i = 0U; i < shape[0] * shape[1]; ++i) {
        X[i].re = 0.0;
        X[i].im = 0.0;
    }
    size_t idx = 1U * shape[1] + 2U;
    X[idx].re = 1.0;
    X[idx].im = 0.25;

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint_2d: add field failed\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_add_fft_convert(&ctx, input_index, SIM_FFT_CONVERT_INVERSE, false, &out_index, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fft_inverse_constraint_2d: add fft_convert failed\n");
        goto cleanup;
    }

    SimResult exec_rc = sim_context_execute(&ctx);
    if (!exploration_mode) {
        if (exec_rc != SIM_RESULT_INVALID_STATE) {
            fprintf(
                stderr,
                "[FAIL] fft_inverse_constraint_2d: strict mode expected INVALID_STATE, got %d\n",
                exec_rc);
            goto cleanup;
        }
    } else {
        if (exec_rc != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] fft_inverse_constraint_2d: exploration execute failed (%d)\n",
                    exec_rc);
            goto cleanup;
        }
        SimField *out_field = sim_context_field(&ctx, out_index);
        if (out_field == NULL || out_field->element_size != sizeof(double)) {
            fprintf(stderr,
                    "[FAIL] fft_inverse_constraint_2d: exploration output missing/complex\n");
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    return ok;
}

int main(void) {
    bool ok = true;
    ok = ok && run_forward_hermitian_case(8U);
    ok = ok && run_forward_hermitian_case(6U);
    ok = ok && run_forward_hermitian_case(7U);
    ok = ok && run_forward_hermitian_case_2d(5U, 6U);
    ok = ok && run_forward_hermitian_case_2d(6U, 10U);
    ok = ok && run_roundtrip_case(8U);
    ok = ok && run_roundtrip_case(6U);
    ok = ok && run_roundtrip_case(9U);
    ok = ok && run_roundtrip_case_2d(4U, 8U);
    ok = ok && run_roundtrip_case_2d(6U, 10U);
    ok = ok && run_inverse_constraint_violation_case(false);
    ok = ok && run_inverse_constraint_violation_case(true);
    ok = ok && run_inverse_constraint_violation_case_2d(false);
    ok = ok && run_inverse_constraint_violation_case_2d(true);
    return ok ? 0 : 1;
}
