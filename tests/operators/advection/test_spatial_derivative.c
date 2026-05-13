/*
 * Migrated advection operator coverage for spatial-derivative contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static void fill_real_field(SimField *field) {
    size_t count = sim_field_element_count(&field->layout);
    double *data = sim_field_real_data(field);
    for (size_t i = 0; i < count; ++i) {
        data[i] = sin((2.0 * M_PI * (double)i) / (double)count);
    }
}

static void fill_complex_field(SimField *field) {
    size_t count = sim_field_element_count(&field->layout);
    SimComplexDouble *data = sim_field_complex_data(field);
    for (size_t i = 0; i < count; ++i) {
        double phase = (2.0 * M_PI * (double)i) / (double)count;
        data[i].re = cos(phase);
        data[i].im = sin(phase);
    }
}

static int compare_real_fields(const SimField *a, const SimField *b, double tol) {
    const double *lhs = sim_field_real_data_const(a);
    const double *rhs = sim_field_real_data_const(b);
    size_t count = sim_field_element_count(&a->layout);
    for (size_t i = 0; i < count; ++i) {
        if (fabs(lhs[i] - rhs[i]) > tol) {
            fprintf(stderr, "mismatch at %zu: %.10e vs %.10e\n", i, lhs[i], rhs[i]);
            return 0;
        }
    }
    return 1;
}

static int compare_complex_fields(const SimField *a, const SimField *b, double tol) {
    const SimComplexDouble *lhs = sim_field_complex_data_const(a);
    const SimComplexDouble *rhs = sim_field_complex_data_const(b);
    size_t count = sim_field_element_count(&a->layout);
    for (size_t i = 0; i < count; ++i) {
        double dr = fabs(lhs[i].re - rhs[i].re);
        double di = fabs(lhs[i].im - rhs[i].im);
        if (dr > tol || di > tol) {
            fprintf(stderr, "complex mismatch at %zu: (%.10e, %.10e) vs (%.10e, %.10e)\n", i,
                    lhs[i].re, lhs[i].im, rhs[i].re, rhs[i].im);
            return 0;
        }
    }
    return 1;
}

static int run_spatial_derivative_case(int make_complex) {
    const size_t count = 32U;
    const size_t shape[1] = {count};
    SimContext ctx;
    SimField input = {0}, skew_out = {0}, forward_out = {0};
    size_t input_idx = SIZE_MAX;
    size_t skew_idx = SIZE_MAX;
    size_t forward_idx = SIZE_MAX;
    SimResult result;
    int success = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "context init failed\n");
        return 0;
    }

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&skew_out, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&forward_out, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "field init failed\n");
        goto cleanup;
    }

    if (make_complex) {
        if (sim_field_promote_inplace_to_complex(&input) != SIM_RESULT_OK ||
            sim_field_promote_inplace_to_complex(&skew_out) != SIM_RESULT_OK ||
            sim_field_promote_inplace_to_complex(&forward_out) != SIM_RESULT_OK) {
            fprintf(stderr, "complex promotion failed\n");
            goto cleanup;
        }
        fill_complex_field(&input);
    } else {
        fill_real_field(&input);
    }

    result = sim_context_add_field(&ctx, &input, &input_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add input failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_add_field(&ctx, &skew_out, &skew_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add skew field failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_add_field(&ctx, &forward_out, &forward_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add forward field failed (%d)\n", (int)result);
        goto cleanup;
    }

    SimSpatialDerivativeOperatorConfig skew_cfg;
    memset(&skew_cfg, 0, sizeof(skew_cfg));
    skew_cfg.input_field = input_idx;
    skew_cfg.output_field = skew_idx;
    skew_cfg.spacing = 1.0;
    skew_cfg.method = SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL;
    skew_cfg.skew_forward = true;
    skew_cfg.accumulate = false;

    SimSpatialDerivativeOperatorConfig forward_cfg = skew_cfg;
    forward_cfg.output_field = forward_idx;
    forward_cfg.method = SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD;
    forward_cfg.skew_forward = false;

    size_t skew_op = SIZE_MAX;
    size_t forward_op = SIZE_MAX;
    result = sim_add_spatial_derivative_operator(&ctx, &skew_cfg, &skew_op);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add skew operator failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_add_spatial_derivative_operator(&ctx, &forward_cfg, &forward_op);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add forward operator failed (%d)\n", (int)result);
        goto cleanup;
    }

    SimSpatialDerivativeOperatorConfig fetched = {0};
    result = sim_spatial_derivative_config(&ctx, forward_op, &fetched);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "config query failed (%d)\n", (int)result);
        goto cleanup;
    }
    if (fetched.method != forward_cfg.method ||
        fabs(fetched.spacing - forward_cfg.spacing) > 1.0e-12) {
        fprintf(stderr, "config mismatch after query\n");
        goto cleanup;
    }

    result = sim_spatial_derivative_update(&ctx, forward_op, &fetched);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "config update failed (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_prepare_plan(&ctx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "prepare plan failed (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_execute(&ctx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "execute failed (%d)\n", (int)result);
        goto cleanup;
    }

    SimField *skew_field = sim_context_field(&ctx, skew_idx);
    SimField *forward_field = sim_context_field(&ctx, forward_idx);
    if (!skew_field || !forward_field) {
        fprintf(stderr, "field lookup failed\n");
        goto cleanup;
    }

    if (!make_complex) {
        success = compare_real_fields(skew_field, forward_field, 1.0e-10);
    } else {
        success = compare_complex_fields(skew_field, forward_field, 1.0e-10);
    }

cleanup:
    sim_context_destroy(&ctx);
    return success;
}

static int run_spatial_derivative_kernel_equivalence_case(SimSpatialDerivativeMethod method,
                                                          SimIRBoundaryPolicy boundary) {
    const size_t count = 16U;
    const size_t shape[1] = {count};
    SimContext ctx_kernel;
    SimContext ctx_split;
    SimBackend backend = {0};
    SimField input_kernel = {0};
    SimField output_kernel = {0};
    SimField input_split = {0};
    SimField output_split = {0};
    size_t input_kernel_idx = SIZE_MAX;
    size_t output_kernel_idx = SIZE_MAX;
    size_t input_split_idx = SIZE_MAX;
    size_t output_split_idx = SIZE_MAX;
    size_t op_kernel = SIZE_MAX;
    size_t op_split = SIZE_MAX;
    SimResult result;
    int success = 0;

    if (sim_context_init(&ctx_kernel) != SIM_RESULT_OK ||
        sim_context_init(&ctx_split) != SIM_RESULT_OK) {
        fprintf(stderr, "context init failed\n");
        goto cleanup;
    }

    backend_init(&backend);
    sim_context_set_backend(&ctx_kernel, &backend);

    if (sim_field_init(&input_kernel, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&output_kernel, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&input_split, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&output_split, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "field init failed\n");
        goto cleanup;
    }

    fill_real_field(&input_kernel);
    fill_real_field(&input_split);

    result = sim_context_add_field(&ctx_kernel, &input_kernel, &input_kernel_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add kernel input failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_add_field(&ctx_kernel, &output_kernel, &output_kernel_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add kernel output failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_add_field(&ctx_split, &input_split, &input_split_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add split input failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_add_field(&ctx_split, &output_split, &output_split_idx);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add split output failed (%d)\n", (int)result);
        goto cleanup;
    }

    SimSpatialDerivativeOperatorConfig cfg_kernel;
    memset(&cfg_kernel, 0, sizeof(cfg_kernel));
    cfg_kernel.input_field = input_kernel_idx;
    cfg_kernel.output_field = output_kernel_idx;
    cfg_kernel.spacing = 1.0;
    cfg_kernel.method = method;
    cfg_kernel.axis = 0U;
    cfg_kernel.skew_forward = false;
    cfg_kernel.accumulate = false;
    cfg_kernel.boundary = boundary;

    SimSpatialDerivativeOperatorConfig cfg_split = cfg_kernel;
    cfg_split.input_field = input_split_idx;
    cfg_split.output_field = output_split_idx;

    result = sim_add_spatial_derivative_operator(&ctx_kernel, &cfg_kernel, &op_kernel);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add kernel operator failed (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_add_spatial_derivative_operator(&ctx_split, &cfg_split, &op_split);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "add split operator failed (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_prepare_plan(&ctx_kernel);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "kernel prepare plan failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_execute(&ctx_kernel);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "kernel execute failed (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_prepare_plan(&ctx_split);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "split prepare plan failed (%d)\n", (int)result);
        goto cleanup;
    }
    result = sim_context_execute(&ctx_split);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "split execute failed (%d)\n", (int)result);
        goto cleanup;
    }

    SimField *kernel_field = sim_context_field(&ctx_kernel, output_kernel_idx);
    SimField *split_field = sim_context_field(&ctx_split, output_split_idx);
    if (kernel_field == NULL || split_field == NULL) {
        fprintf(stderr, "output lookup failed\n");
        goto cleanup;
    }

    success = compare_real_fields(kernel_field, split_field, 1.0e-10);

cleanup:
    sim_context_destroy(&ctx_kernel);
    sim_context_destroy(&ctx_split);
    backend_destroy(&backend);
    return success;
}

static int run_spatial_derivative_kernel_equivalence(void) {
    const SimSpatialDerivativeMethod methods[] = {SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL,
                                                  SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD,
                                                  SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD};
    const SimIRBoundaryPolicy boundaries[] = {SIM_IR_BOUNDARY_NEUMANN, SIM_IR_BOUNDARY_DIRICHLET,
                                              SIM_IR_BOUNDARY_PERIODIC, SIM_IR_BOUNDARY_REFLECTIVE};

    for (size_t i = 0; i < (sizeof(methods) / sizeof(methods[0])); ++i) {
        for (size_t j = 0; j < (sizeof(boundaries) / sizeof(boundaries[0])); ++j) {
            if (!run_spatial_derivative_kernel_equivalence_case(methods[i], boundaries[j])) {
                fprintf(stderr, "kernel equivalence failed (method=%s boundary=%s)\n",
                        sim_spatial_derivative_method_name(methods[i]),
                        sim_boundary_policy_name(boundaries[j]));
                return 0;
            }
        }
    }
    return 1;
}

int main(void) {
    if (!run_spatial_derivative_case(0)) {
        fprintf(stderr, "real skew vs forward comparison failed\n");
        return EXIT_FAILURE;
    }

    if (!run_spatial_derivative_case(1)) {
        fprintf(stderr, "complex skew vs forward comparison failed\n");
        return EXIT_FAILURE;
    }

    if (!run_spatial_derivative_kernel_equivalence()) {
        fprintf(stderr, "kernel equivalence comparison failed\n");
        return EXIT_FAILURE;
    }

    printf("test_spatial_derivative ok\n");
    return EXIT_SUCCESS;
}
