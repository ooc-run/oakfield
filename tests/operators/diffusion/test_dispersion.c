/*
 * Migrated diffusion operator coverage for dispersion contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static bool nearly_equal(double a, double b, double eps) {
    return fabs(a - b) <= eps * fmax(1.0, fmax(fabs(a), fabs(b)));
}

static bool complex_nearly_equal(SimComplexDouble a, SimComplexDouble b, double eps) {
    return nearly_equal(a.re, b.re, eps) && nearly_equal(a.im, b.im, eps);
}

static bool check_hermitian_spectrum(const SimComplexDouble *X, size_t n, double tol) {
    if (!X || n < 2U)
        return false;

    if (!nearly_equal(X[0].im, 0.0, tol))
        return false;

    if ((n % 2U) == 0U) {
        size_t k = n / 2U;
        if (!nearly_equal(X[k].im, 0.0, tol))
            return false;
    }

    for (size_t k = 1U; k < n / 2U; ++k) {
        size_t nk = n - k;
        if (!nearly_equal(X[nk].re, X[k].re, tol) || !nearly_equal(X[nk].im, -X[k].im, tol))
            return false;
    }

    return true;
}

static bool run_dispersion_2d_spectral_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {6U, 8U};
    size_t field_index = 0U;
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    const size_t kx = 1U;
    const size_t ky = 2U;
    const double dt = 0.1;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_spectral: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_spectral: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;

    SimComplexDouble *data = (SimComplexDouble *)sim_field_data(&field);
    size_t count = rows * cols;
    for (size_t i = 0U; i < count; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    size_t idx = ky * cols + kx;
    data[idx].re = 1.0;
    data[idx].im = 0.0;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_spectral: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    DispersionOperatorConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.coefficient = 1.0;
    cfg.order = 2.0;
    cfg.spacing = 1.0;
    cfg.reference_k = 0.0;

    if (sim_add_dispersion_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_spectral: add dispersion failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_spectral: execute failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    SimComplexDouble *out = sim_field_complex_data(out_field);
    if (out == NULL) {
        fprintf(stderr, "[FAIL] dispersion_2d_spectral: output missing\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double base_kx = (2.0 * M_PI) / ((double)cols * cfg.spacing);
    double base_ky = (2.0 * M_PI) / ((double)rows * cfg.spacing);
    double kx_val = base_kx * (double)kx;
    double ky_val = base_ky * (double)ky;
    double k_abs = sqrt(kx_val * kx_val + ky_val * ky_val);
    double phase = dt * cfg.coefficient * pow(k_abs, cfg.order);
    SimComplexDouble expected = {cos(phase), sin(phase)};

    if (!complex_nearly_equal(out[idx], expected, 1.0e-6)) {
        fprintf(stderr,
                "[FAIL] dispersion_2d_spectral: mismatch got (%.9f, %.9f) expected (%.9f, %.9f)\n",
                out[idx].re, out[idx].im, expected.re, expected.im);
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_dispersion_2d_physical_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {6U, 8U};
    size_t field_index = 0U;
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    const size_t kx = 1U;
    const size_t ky = 2U;
    const double dt = 0.1;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_physical: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_physical: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimComplexDouble *data = (SimComplexDouble *)sim_field_data(&field);
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

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_physical: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    DispersionOperatorConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.coefficient = 1.0;
    cfg.order = 2.0;
    cfg.spacing = 1.0;
    cfg.reference_k = 0.0;

    if (sim_add_dispersion_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_physical: add dispersion failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_2d_physical: execute failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    SimComplexDouble *out = sim_field_complex_data(out_field);
    if (out == NULL) {
        fprintf(stderr, "[FAIL] dispersion_2d_physical: output missing\n");
        sim_context_destroy(&ctx);
        return false;
    }

    double base_kx = (2.0 * M_PI) / ((double)cols * cfg.spacing);
    double base_ky = (2.0 * M_PI) / ((double)rows * cfg.spacing);
    double kx_val = base_kx * (double)kx;
    double ky_val = base_ky * (double)ky;
    double k_abs = sqrt(kx_val * kx_val + ky_val * ky_val);
    double phase = dt * cfg.coefficient * pow(k_abs, cfg.order);
    double exp_re = cos(phase);
    double exp_im = sin(phase);

    for (size_t y = 0U; y < rows; ++y) {
        for (size_t x = 0U; x < cols; ++x) {
            double angle =
                2.0 * M_PI *
                ((double)kx * (double)x / (double)cols + (double)ky * (double)y / (double)rows);
            double orig_re = cos(angle);
            double orig_im = sin(angle);
            SimComplexDouble expected = {orig_re * exp_re - orig_im * exp_im,
                                         orig_re * exp_im + orig_im * exp_re};
            size_t idx = y * cols + x;
            if (!complex_nearly_equal(out[idx], expected, 1.0e-6)) {
                fprintf(stderr,
                        "[FAIL] dispersion_2d_physical: mismatch at (%zu,%zu) got (%.9f, %.9f)\n",
                        y, x, out[idx].re, out[idx].im);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_dispersion_real_constrained_spectral_preserves_conjugate_symmetry(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;

    SimComplexDouble *data = (SimComplexDouble *)sim_field_data(&field);
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    data[1].re = 0.8;
    data[1].im = 0.3;
    data[shape[0] - 1U].re = 0.8;
    data[shape[0] - 1U].im = -0.3;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    DispersionOperatorConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.coefficient = 1.0;
    cfg.order = 2.0;
    cfg.spacing = 1.0;
    cfg.reference_k = 0.0;

    if (sim_add_dispersion_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: add dispersion failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimResult execute_rc = sim_context_execute(&ctx);
    if (execute_rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: expected success, got %d\n",
                (int)execute_rc);
        goto cleanup;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    if (out_field == NULL) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: output missing\n");
        goto cleanup;
    }

    const SimComplexDouble *out = sim_field_complex_data_const(out_field);
    if (out == NULL) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: complex output missing\n");
        goto cleanup;
    }

    if (!check_hermitian_spectrum(out, shape[0], 1.0e-9)) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: Hermitian symmetry violated\n");
        goto cleanup;
    }

    double base_k = (2.0 * M_PI) / ((double)shape[0] * cfg.spacing);
    double phase = 0.1 * cfg.coefficient * pow(fabs(base_k * 1.0 - cfg.reference_k), cfg.order);
    double scale = cos(phase);
    if (!nearly_equal(out[1].re, 0.8 * scale, 1.0e-9) ||
        !nearly_equal(out[1].im, 0.3 * scale, 1.0e-9) ||
        !nearly_equal(out[shape[0] - 1U].re, 0.8 * scale, 1.0e-9) ||
        !nearly_equal(out[shape[0] - 1U].im, -0.3 * scale, 1.0e-9)) {
        fprintf(stderr, "[FAIL] dispersion_real_constrained: projected scale mismatch\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_dispersion_unknown_value_kind_rejected_preplan(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_unknown_value_kind: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_unknown_value_kind: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_unknown_value_kind: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    SimField *ctx_field = sim_context_field(&ctx, field_index);
    if (ctx_field == NULL) {
        fprintf(stderr, "[FAIL] dispersion_unknown_value_kind: context field missing\n");
        sim_context_destroy(&ctx);
        return false;
    }
    ctx_field->repr.value_kind = SIM_FIELD_VALUE_UNKNOWN;

    DispersionOperatorConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.coefficient = 1.0;
    cfg.order = 2.0;
    cfg.spacing = 1.0;
    cfg.reference_k = 0.0;

    if (sim_add_dispersion_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_unknown_value_kind: add dispersion failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    SimResult execute_rc = sim_context_execute(&ctx);
    if (execute_rc != SIM_RESULT_INVALID_ARGUMENT) {
        fprintf(stderr, "[FAIL] dispersion_unknown_value_kind: expected invalid argument, got %d\n",
                (int)execute_rc);
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_dispersion_imag_zero_constraint_preserved_when_inactive(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: field data missing\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = cos(2.0 * M_PI * (double)i / (double)shape[0]);
        data[i].im = 0.0;
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    {
        DispersionOperatorConfig cfg = {.field_index = field_index,
                                        .coefficient = 0.0,
                                        .order = 2.0,
                                        .spacing = 1.0,
                                        .reference_k = 0.0};
        if (sim_add_dispersion_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: add operator failed\n");
            goto cleanup;
        }
    }

    {
        SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
        if (op == NULL) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(op);
        if (!info.preserves_real ||
            info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT ||
            !info.representation.requires_complex_input) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: metadata mismatch\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: execute failed\n");
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&ctx, field_index);
        if (out_field == NULL) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: output missing\n");
            goto cleanup;
        }
        SimFieldRepresentation repr = sim_field_representation(out_field);
        if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
            repr.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT) {
            fprintf(stderr,
                    "[FAIL] dispersion_imag_zero_inactive: representation mismatch (%d, %d)\n",
                    (int)repr.domain, (int)repr.value_kind);
            goto cleanup;
        }
        data = sim_field_complex_data(out_field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: output data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            double expected = cos(2.0 * M_PI * (double)i / (double)shape[0]);
            if (!nearly_equal(data[i].re, expected, 1.0e-12) ||
                !nearly_equal(data[i].im, 0.0, 1.0e-12)) {
                fprintf(stderr, "[FAIL] dispersion_imag_zero_inactive: value mismatch at %zu\n", i);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool run_dispersion_imag_zero_constraint_demotes_with_phase(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.1;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: field init failed\n");
        sim_context_destroy(&ctx);
        return false;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: field data missing\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    data[1].re = 1.0;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return false;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    {
        DispersionOperatorConfig cfg = {.field_index = field_index,
                                        .coefficient = 1.0,
                                        .order = 2.0,
                                        .spacing = 1.0,
                                        .reference_k = 0.0};
        if (sim_add_dispersion_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: add operator failed\n");
            goto cleanup;
        }
    }

    {
        SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
        if (op == NULL) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(op);
        if (info.preserves_real ||
            info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
            !info.representation.requires_complex_input) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: metadata mismatch\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: execute failed\n");
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&ctx, field_index);
        if (out_field == NULL) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: output missing\n");
            goto cleanup;
        }
        SimFieldRepresentation repr = sim_field_representation(out_field);
        if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
            repr.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR) {
            fprintf(stderr,
                    "[FAIL] dispersion_imag_zero_demote: representation mismatch (%d, %d)\n",
                    (int)repr.domain, (int)repr.value_kind);
            goto cleanup;
        }

        data = sim_field_complex_data(out_field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: output data missing\n");
            goto cleanup;
        }

        {
            const double base_k = (2.0 * M_PI) / ((double)shape[0]);
            const double phase = dt * pow(base_k, 2.0);
            if (!nearly_equal(data[1].re, cos(phase), 1.0e-9) ||
                !nearly_equal(data[1].im, sin(phase), 1.0e-9)) {
                fprintf(stderr, "[FAIL] dispersion_imag_zero_demote: value mismatch\n");
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    const size_t N = shape[0];
    const size_t k = 1U; /* single-tone bin */
    const double dt = 0.1;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "field init failed\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimComplexDouble *data = (SimComplexDouble *)sim_field_data(&field);
    for (size_t n = 0U; n < N; ++n) {
        double angle = 2.0 * M_PI * (double)k * (double)n / (double)N;
        data[n].re = cos(angle);
        data[n].im = sin(angle);
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "add field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 1;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    DispersionOperatorConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.coefficient = 1.0;
    cfg.order = 2.0;
    cfg.spacing = 1.0;
    cfg.reference_k = 0.0;

    if (sim_add_dispersion_operator(&ctx, &cfg, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "add dispersion failed\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "execute failed\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    /* Expect the entire waveform to be multiplied by exp(i * phase) */
    double base_k = (2.0 * M_PI) / ((double)N * cfg.spacing);
    double k_abs = base_k * (double)k;
    double expected_phase = dt * cfg.coefficient * pow(k_abs, cfg.order);
    double exp_re = cos(expected_phase);
    double exp_im = sin(expected_phase);

    SimComplexDouble *out = sim_field_complex_data(sim_context_field(&ctx, field_index));
    for (size_t n = 0U; n < N; ++n) {
        double re = out[n].re;
        double im = out[n].im;
        /* original sample was cos + i sin; after multiplication, compare rotated sample */
        double orig_re = cos(2.0 * M_PI * (double)k * (double)n / (double)N);
        double orig_im = sin(2.0 * M_PI * (double)k * (double)n / (double)N);
        double exp_re_n = orig_re * exp_re - orig_im * exp_im;
        double exp_im_n = orig_re * exp_im + orig_im * exp_re;
        if (!nearly_equal(re, exp_re_n, 1.0e-6) || !nearly_equal(im, exp_im_n, 1.0e-6)) {
            fprintf(stderr, "mismatch at %zu got (%.9f, %.9f) expected (%.9f, %.9f)\n", n, re, im,
                    exp_re_n, exp_im_n);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    if (!ok) {
        return 1;
    }
    if (!run_dispersion_2d_spectral_case()) {
        return 1;
    }
    if (!run_dispersion_2d_physical_case()) {
        return 1;
    }
    if (!run_dispersion_real_constrained_spectral_preserves_conjugate_symmetry()) {
        return 1;
    }
    if (!run_dispersion_unknown_value_kind_rejected_preplan()) {
        return 1;
    }
    if (!run_dispersion_imag_zero_constraint_preserved_when_inactive()) {
        return 1;
    }
    if (!run_dispersion_imag_zero_constraint_demotes_with_phase()) {
        return 1;
    }
    return 0;
}
