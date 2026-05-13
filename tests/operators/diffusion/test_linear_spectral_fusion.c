/*
 * Migrated diffusion operator coverage for linear spectral-fusion contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static bool nearly_equal(double a, double b, double eps) {
    return fabs(a - b) <= eps * fmax(1.0, fmax(fabs(a), fabs(b)));
}

static bool complex_nearly_equal(SimComplexDouble a, SimComplexDouble b, double eps) {
    return nearly_equal(a.re, b.re, eps) && nearly_equal(a.im, b.im, eps);
}

static bool setup_complex_field_context(SimContext *ctx, size_t count, double dt,
                                        size_t *out_field_index) {
    SimField field = {0};
    size_t shape[1] = {count};
    size_t field_index = 0U;

    if (sim_context_init(ctx) != SIM_RESULT_OK) {
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        sim_context_destroy(ctx);
        return false;
    }

    SimComplexDouble *data = sim_field_complex_data(&field);
    for (size_t i = 0U; i < count; ++i) {
        double x = (double)i / (double)count;
        data[i].re = 0.8 * sin(2.0 * M_PI * x) + 0.4 * cos(6.0 * M_PI * x);
        data[i].im = 0.25 * cos(4.0 * M_PI * x) - 0.15 * sin(8.0 * M_PI * x);
    }

    if (sim_context_add_field(ctx, &field, &field_index) != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        sim_context_destroy(ctx);
        return false;
    }

    sim_context_set_timestep(ctx, (float)dt);
    if (out_field_index) {
        *out_field_index = field_index;
    }
    return true;
}

static bool run_fusion_equivalence_physical(size_t count) {
    const double dt = 0.02;
    const size_t steps = 4U;
    const double spacing = 0.1;

    /* params chosen to exercise both damping and phase */
    const double viscosity = 0.5;
    const double alpha = 1.5;
    const double dispersion_coeff = 0.7;
    const double dispersion_order = 2.0;
    const double reference_k = 0.0;
    const double phase_rate = 1.3;

    SimContext ctx_sep = {0};
    SimContext ctx_fused = {0};
    size_t field_sep = 0U;
    size_t field_fused = 0U;

    bool ok = false;

    if (!setup_complex_field_context(&ctx_sep, count, dt, &field_sep))
        goto cleanup;
    if (!setup_complex_field_context(&ctx_fused, count, dt, &field_fused))
        goto cleanup;

    LinearDissipativeOperatorConfig dissip = {
        .field_index = field_sep, .viscosity = viscosity, .alpha = alpha, .spacing = spacing};
    DispersionOperatorConfig disp = {.field_index = field_sep,
                                     .coefficient = dispersion_coeff,
                                     .order = dispersion_order,
                                     .spacing = spacing,
                                     .reference_k = reference_k};
    PhaseRotateOperatorConfig phase = {.field_index = field_sep, .phase_rate = phase_rate};

    if (sim_add_linear_dissipative_operator(&ctx_sep, &dissip, NULL) != SIM_RESULT_OK)
        goto cleanup;
    if (sim_add_dispersion_operator(&ctx_sep, &disp, NULL) != SIM_RESULT_OK)
        goto cleanup;
    if (sim_add_phase_rotate_operator(&ctx_sep, &phase, NULL) != SIM_RESULT_OK)
        goto cleanup;

    LinearSpectralFusionOperatorConfig fused = {.field_index = field_fused,
                                                .viscosity = viscosity,
                                                .alpha = alpha,
                                                .dissipation_spacing = spacing,
                                                .dispersion_coefficient = dispersion_coeff,
                                                .dispersion_order = dispersion_order,
                                                .dispersion_reference_k = reference_k,
                                                .dispersion_spacing = spacing,
                                                .phase_rate = phase_rate};

    if (sim_add_linear_spectral_fusion_operator(&ctx_fused, &fused, NULL) != SIM_RESULT_OK)
        goto cleanup;

    if (sim_context_prepare_plan(&ctx_sep) != SIM_RESULT_OK)
        goto cleanup;
    if (sim_context_prepare_plan(&ctx_fused) != SIM_RESULT_OK)
        goto cleanup;

    for (size_t s = 0U; s < steps; ++s) {
        if (sim_context_execute(&ctx_sep) != SIM_RESULT_OK)
            goto cleanup;
        if (sim_context_execute(&ctx_fused) != SIM_RESULT_OK)
            goto cleanup;
    }

    SimField *f_sep = sim_context_field(&ctx_sep, field_sep);
    SimField *f_fused = sim_context_field(&ctx_fused, field_fused);
    if (!f_sep || !f_fused)
        goto cleanup;

    SimComplexDouble *a = sim_field_complex_data(f_sep);
    SimComplexDouble *b = sim_field_complex_data(f_fused);
    if (!a || !b)
        goto cleanup;

    double max_err = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double err_re = fabs(a[i].re - b[i].re);
        double err_im = fabs(a[i].im - b[i].im);
        double err = fmax(err_re, err_im);
        if (err > max_err)
            max_err = err;
    }

    if (max_err > 5.0e-8) {
        fprintf(stderr, "[FAIL] fusion equivalence: max_err=%.12g\n", max_err);
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_sep);
    sim_context_destroy(&ctx_fused);
    return ok;
}

static bool check_hermitian_spectrum(const SimComplexDouble *X, size_t n, double tol) {
    if (!X || n < 2U)
        return false;

    /* DC */
    if (!nearly_equal(X[0].im, 0.0, tol))
        return false;

    /* Nyquist for even n */
    if ((n % 2U) == 0U) {
        size_t k = n / 2U;
        if (!nearly_equal(X[k].im, 0.0, tol))
            return false;
    }

    for (size_t k = 1U; k < n / 2U; ++k) {
        size_t nk = n - k;
        double exp_re = X[k].re;
        double exp_im = -X[k].im;
        if (!nearly_equal(X[nk].re, exp_re, tol) || !nearly_equal(X[nk].im, exp_im, tol))
            return false;
    }

    return true;
}

static bool run_fusion_preserves_conjugate_symmetry(size_t count) {
    const double dt = 0.01;
    const double spacing = 1.0;

    SimContext ctx = {0};
    SimField real_field = {0};
    size_t shape[1] = {count};
    size_t in_field = 0U;
    size_t spec_field = 0U;

    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK)
        goto cleanup;

    if (sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK)
        goto cleanup;

    double *x = sim_field_real_data(&real_field);
    for (size_t i = 0U; i < count; ++i) {
        double t = (double)i / (double)count;
        x[i] = sin(2.0 * M_PI * t) + 0.25 * cos(6.0 * M_PI * t);
    }

    if (sim_context_add_field(&ctx, &real_field, &in_field) != SIM_RESULT_OK)
        goto cleanup;
    real_field = (SimField){0};

    sim_context_set_timestep(&ctx, (float)dt);

    if (sim_add_fft_convert(&ctx, in_field, SIM_FFT_CONVERT_FORWARD, false, &spec_field, NULL) !=
        SIM_RESULT_OK)
        goto cleanup;

    LinearSpectralFusionOperatorConfig fused = {.field_index = spec_field,
                                                .viscosity = 0.2,
                                                .alpha = 2.0,
                                                .dissipation_spacing = spacing,
                                                .dispersion_coefficient = 0.0,
                                                .dispersion_order = 2.0,
                                                .dispersion_reference_k = 0.0,
                                                .dispersion_spacing = spacing,
                                                .phase_rate = 0.0};
    size_t fusion_op = 0U;
    if (sim_add_linear_spectral_fusion_operator(&ctx, &fused, &fusion_op) != SIM_RESULT_OK)
        goto cleanup;

    if (sim_context_prepare_plan(&ctx) != SIM_RESULT_OK)
        goto cleanup;

    if (sim_context_execute(&ctx) != SIM_RESULT_OK)
        goto cleanup;

    SimField *spec = sim_context_field(&ctx, spec_field);
    if (!spec)
        goto cleanup;

    SimFieldRepresentation repr = sim_field_representation(spec);
    if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
        repr.value_kind != SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT) {
        fprintf(stderr, "[FAIL] expected spectral real-constraint, got domain=%d kind=%d\n",
                repr.domain, repr.value_kind);
        goto cleanup;
    }

    const SimComplexDouble *X = sim_field_complex_data_const(spec);
    if (!check_hermitian_spectrum(X, count, 5.0e-10)) {
        fprintf(stderr, "[FAIL] hermitian symmetry violated after fused dissipation\n");
        goto cleanup;
    }

    /* Now enable a phase term; projected-real support should preserve Hermitian symmetry. */
    fused.phase_rate = 1.0;
    if (sim_linear_spectral_fusion_update(&ctx, fusion_op, &fused) != SIM_RESULT_OK)
        goto cleanup;
    SimResult rc = sim_context_execute(&ctx);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] expected success for projected real phase term, got %d\n", (int)rc);
        goto cleanup;
    }

    X = sim_field_complex_data_const(spec);
    if (!check_hermitian_spectrum(X, count, 5.0e-10)) {
        fprintf(stderr, "[FAIL] hermitian symmetry violated after projected real phase update\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    sim_field_destroy(&real_field);
    return ok;
}

static bool run_fusion_2d_spectral_case(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[2] = {6U, 5U};
    size_t field_index = 0U;
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    const size_t kx = 1U;
    const size_t ky = 2U;
    const double dt = 0.05;

    const double viscosity = 0.4;
    const double alpha = 1.5;
    const double dispersion_coeff = 0.9;
    const double dispersion_order = 2.0;
    const double reference_k = 0.0;
    const double phase_rate = 0.3;
    const double spacing = 1.0;

    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK)
        return false;

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK)
        goto cleanup;

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (!data)
        goto cleanup;
    for (size_t i = 0U; i < rows * cols; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    size_t idx = ky * cols + kx;
    data[idx].re = 1.0;
    data[idx].im = 0.0;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK)
        goto cleanup;

    sim_context_set_timestep(&ctx, (float)dt);

    LinearSpectralFusionOperatorConfig fused = {.field_index = field_index,
                                                .viscosity = viscosity,
                                                .alpha = alpha,
                                                .dissipation_spacing = spacing,
                                                .dispersion_coefficient = dispersion_coeff,
                                                .dispersion_order = dispersion_order,
                                                .dispersion_reference_k = reference_k,
                                                .dispersion_spacing = spacing,
                                                .phase_rate = phase_rate};

    if (sim_add_linear_spectral_fusion_operator(&ctx, &fused, NULL) != SIM_RESULT_OK)
        goto cleanup;

    if (sim_context_execute(&ctx) != SIM_RESULT_OK)
        goto cleanup;

    SimField *out_field = sim_context_field(&ctx, field_index);
    SimComplexDouble *out = sim_field_complex_data(out_field);
    if (!out)
        goto cleanup;

    double base_kx = (2.0 * M_PI) / ((double)cols * spacing);
    double base_ky = (2.0 * M_PI) / ((double)rows * spacing);
    double kx_val = base_kx * (double)kx;
    double ky_val = base_ky * (double)ky;
    double k_abs = sqrt(kx_val * kx_val + ky_val * ky_val);
    double lambda = viscosity * -pow(k_abs, alpha);
    double omega = dispersion_coeff * pow(fabs(k_abs - reference_k), dispersion_order);
    double factor = exp(dt * lambda);
    double theta = dt * omega + dt * phase_rate;
    SimComplexDouble expected = {factor * cos(theta), factor * sin(theta)};

    if (!complex_nearly_equal(out[idx], expected, 1.0e-6)) {
        fprintf(stderr, "[FAIL] fusion_2d_spectral: mismatch got (%.9f, %.9f)\n", out[idx].re,
                out[idx].im);
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    sim_field_destroy(&field);
    return ok;
}

static bool run_fusion_imag_zero_constraint_preserved(void) {
    SimContext ctx_ref = {0};
    SimContext ctx_fused = {0};
    SimField ref_field = {0};
    SimField fused_field = {0};
    size_t shape[1] = {8U};
    size_t ref_index = 0U;
    size_t fused_index = 0U;
    size_t fusion_op = 0U;
    const double dt = 0.02;
    const size_t steps = 5U;
    const double spacing = 1.0;
    bool ok = false;

    if (sim_context_init(&ctx_ref) != SIM_RESULT_OK ||
        sim_context_init(&ctx_fused) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: context init failed\n");
        goto cleanup;
    }

    if (sim_field_init(&ref_field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&fused_field, 1U, shape, sizeof(SimComplexDouble),
                       SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: field init failed\n");
        goto cleanup;
    }

    SimComplexDouble *ref_data = sim_field_complex_data(&ref_field);
    SimComplexDouble *fused_data = sim_field_complex_data(&fused_field);
    if (ref_data == NULL || fused_data == NULL) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: field data missing\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < shape[0]; ++i) {
        double x = (double)i / (double)shape[0];
        double value = cos(2.0 * M_PI * x) + 0.35 * sin(4.0 * M_PI * x);
        ref_data[i].re = value;
        ref_data[i].im = 0.0;
        fused_data[i].re = value;
        fused_data[i].im = 0.0;
    }
    ref_field.repr.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    ref_field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;
    fused_field.repr.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    fused_field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    if (sim_context_add_field(&ctx_ref, &ref_field, &ref_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx_fused, &fused_field, &fused_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: add field failed\n");
        goto cleanup;
    }

    sim_context_set_timestep(&ctx_ref, (float)dt);
    sim_context_set_timestep(&ctx_fused, (float)dt);

    {
        LinearDissipativeOperatorConfig dissip = {
            .field_index = ref_index, .viscosity = 0.3, .alpha = 2.0, .spacing = spacing};
        LinearSpectralFusionOperatorConfig fused = {.field_index = fused_index,
                                                    .viscosity = 0.3,
                                                    .alpha = 2.0,
                                                    .dissipation_spacing = spacing,
                                                    .dispersion_coefficient = 0.0,
                                                    .dispersion_order = 2.0,
                                                    .dispersion_reference_k = 0.0,
                                                    .dispersion_spacing = spacing,
                                                    .phase_rate = 0.0};
        if (sim_add_linear_dissipative_operator(&ctx_ref, &dissip, NULL) != SIM_RESULT_OK ||
            sim_add_linear_spectral_fusion_operator(&ctx_fused, &fused, &fusion_op) !=
                SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: add operator failed\n");
            goto cleanup;
        }
    }

    {
        SimOperator *op = sim_operator_registry_get(&ctx_fused.world.operators, fusion_op);
        if (op == NULL) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(op);
        if (info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT ||
            !info.preserves_real) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: metadata mismatch\n");
            goto cleanup;
        }
    }

    for (size_t step = 0U; step < steps; ++step) {
        if (sim_context_execute(&ctx_ref) != SIM_RESULT_OK ||
            sim_context_execute(&ctx_fused) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: execute failed at step %zu\n",
                    step);
            goto cleanup;
        }
    }

    {
        SimField *ref_out = sim_context_field(&ctx_ref, ref_index);
        SimField *fused_out = sim_context_field(&ctx_fused, fused_index);
        if (ref_out == NULL || fused_out == NULL) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: output field missing\n");
            goto cleanup;
        }

        SimFieldRepresentation repr = sim_field_representation(fused_out);
        if (repr.domain != SIM_FIELD_DOMAIN_PHYSICAL ||
            repr.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT) {
            fprintf(stderr,
                    "[FAIL] fusion_imag_zero_preserved: representation not preserved (%d, %d)\n",
                    (int)repr.domain, (int)repr.value_kind);
            goto cleanup;
        }

        ref_data = sim_field_complex_data(ref_out);
        fused_data = sim_field_complex_data(fused_out);
        if (ref_data == NULL || fused_data == NULL) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: output data missing\n");
            goto cleanup;
        }

        for (size_t i = 0U; i < shape[0]; ++i) {
            if (!nearly_equal(fused_data[i].re, ref_data[i].re, 1.0e-7)) {
                fprintf(stderr, "[FAIL] fusion_imag_zero_preserved: real mismatch at %zu\n", i);
                goto cleanup;
            }
            if (fused_data[i].im != 0.0) {
                fprintf(
                    stderr,
                    "[FAIL] fusion_imag_zero_preserved: imaginary lane should remain zero at %zu\n",
                    i);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx_ref);
    sim_context_destroy(&ctx_fused);
    sim_field_destroy(&ref_field);
    sim_field_destroy(&fused_field);
    return ok;
}

static bool run_fusion_imag_zero_constraint_demotes_with_phase(void) {
    SimContext ctx = {0};
    SimField field = {0};
    size_t shape[1] = {8U};
    size_t field_index = 0U;
    size_t fusion_op = 0U;
    const double dt = 0.1;
    const double phase_rate = 0.6;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_demote: context init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_demote: field init failed\n");
        goto cleanup;
    }

    field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT;

    SimComplexDouble *data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_demote: field data missing\n");
        goto cleanup;
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i].re = 0.0;
        data[i].im = 0.0;
    }
    data[1].re = 1.0;

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_demote: add field failed\n");
        goto cleanup;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    {
        LinearSpectralFusionOperatorConfig fused = {.field_index = field_index,
                                                    .viscosity = 0.0,
                                                    .alpha = 2.0,
                                                    .dissipation_spacing = 1.0,
                                                    .dispersion_coefficient = 0.0,
                                                    .dispersion_order = 2.0,
                                                    .dispersion_reference_k = 0.0,
                                                    .dispersion_spacing = 1.0,
                                                    .phase_rate = phase_rate};
        if (sim_add_linear_spectral_fusion_operator(&ctx, &fused, &fusion_op) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_demote: add operator failed\n");
            goto cleanup;
        }
    }

    {
        SimOperator *op = sim_operator_registry_get(&ctx.world.operators, fusion_op);
        if (op == NULL) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_demote: operator missing\n");
            goto cleanup;
        }
        SimOperatorInfo info = sim_operator_info(op);
        if (info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
            info.preserves_real) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_demote: metadata mismatch\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] fusion_imag_zero_demote: execute failed\n");
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&ctx, field_index);
        if (out_field == NULL) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_demote: output field missing\n");
            goto cleanup;
        }
        SimFieldRepresentation repr = sim_field_representation(out_field);
        if (repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
            repr.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_demote: representation not demoted (%d, %d)\n",
                    (int)repr.domain, (int)repr.value_kind);
            goto cleanup;
        }

        data = sim_field_complex_data(out_field);
        if (data == NULL) {
            fprintf(stderr, "[FAIL] fusion_imag_zero_demote: output data missing\n");
            goto cleanup;
        }

        {
            const double theta = dt * phase_rate;
            SimComplexDouble expected = {cos(theta), sin(theta)};
            if (!complex_nearly_equal(data[1], expected, 1.0e-9)) {
                fprintf(stderr,
                        "[FAIL] fusion_imag_zero_demote: value mismatch got (%.12g, %.12g)\n",
                        data[1].re, data[1].im);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    sim_field_destroy(&field);
    return ok;
}

int main(void) {
    bool ok1 = run_fusion_equivalence_physical(64U);
    bool ok2 = run_fusion_preserves_conjugate_symmetry(64U);
    bool ok3 = run_fusion_2d_spectral_case();
    bool ok4 = run_fusion_imag_zero_constraint_preserved();
    bool ok5 = run_fusion_imag_zero_constraint_demotes_with_phase();

    if (!ok1 || !ok2 || !ok3 || !ok4 || !ok5) {
        return 1;
    }

    printf("[PASS] linear spectral fusion equivalence + conjugate symmetry\n");
    return 0;
}
