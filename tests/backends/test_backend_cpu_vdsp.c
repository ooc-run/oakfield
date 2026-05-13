/*
 * Checks that optional Apple/vDSP CPU fast paths match the scalar CPU backend
 * for representative real binary ops and complex rotation kernels.
 */
#include <oakfield/backend.h>
#include <oakfield/field.h>
#include <oakfield/kernel_ir.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(SIM_HAVE_VDSP)
int main(void) {
    fprintf(stdout, "SIM_HAVE_VDSP disabled; skipping backend CPU vDSP test.\n");
    return 0;
}
#else

#define TEST_ELEMENT_COUNT 256U
#define MAX_DIFF 1.0e-12

static void populate_inputs(double *lhs, double *rhs, size_t count, SimIRNodeType op) {
    for (size_t i = 0; i < count; ++i) {
        double x = (double)(i + 1U) * 0.017;
        lhs[i] = sin(x) * 3.0 + cos(0.5 * x) * 0.25 + 0.1 * (double)i;
        rhs[i] = cos(0.75 * x) * 2.0 + sin(0.33 * x) * 0.2 + 1.5;
        if (op == SIM_IR_NODE_DIV && fabs(rhs[i]) < 1.0e-3) {
            rhs[i] += 0.5;
        }
    }
}

static bool verify_case(SimIRNodeType op) {
    SimField lhs = {0};
    SimField rhs = {0};
    SimField dst = {0};
    SimIRBuilder builder;
    memset(&builder, 0, sizeof(builder));
    bool builder_ready = false;
    SimBackend backend = {0};
    size_t shape[1] = {TEST_ELEMENT_COUNT};
    SimResult rc;
    bool ok = false;

    rc = sim_field_init(&lhs, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init lhs field: %d\n", (int)rc);
        goto cleanup;
    }
    rc = sim_field_init(&rhs, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init rhs field: %d\n", (int)rc);
        goto cleanup;
    }
    rc = sim_field_init(&dst, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init dst field: %d\n", (int)rc);
        goto cleanup;
    }

    populate_inputs((double *)sim_field_data(&lhs), (double *)sim_field_data(&rhs),
                    TEST_ELEMENT_COUNT, op);

    rc = sim_ir_builder_init(&builder);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init IR builder: %d\n", (int)rc);
        goto cleanup;
    }
    builder_ready = true;

    SimIRNodeId lhs_node = sim_ir_builder_field_ref(&builder, 0);
    SimIRNodeId rhs_node = sim_ir_builder_field_ref(&builder, 1);
    SimIRNodeId expr = sim_ir_builder_binary(&builder, op, lhs_node, rhs_node);
    if (lhs_node == SIM_IR_INVALID_NODE || rhs_node == SIM_IR_INVALID_NODE ||
        expr == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "failed to build IR nodes for op %d\n", (int)op);
        goto cleanup;
    }

    SimKernelIRBinding bindings[3] = {{.field_index = 0, .field = &lhs},
                                      {.field_index = 1, .field = &rhs},
                                      {.field_index = 2, .field = &dst}};
    SimKernelIROutput output = {.field_index = 2, .expression = expr};
    KernelIR kernel = {.builder = &builder,
                       .bindings = bindings,
                       .binding_count = 3U,
                       .outputs = &output,
                       .output_count = 1U,
                       .required_features = SIM_BACKEND_FEATURE_NONE};

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);

    double *dst_data = (double *)sim_field_data(&dst);
    double vdsp_out[TEST_ELEMENT_COUNT];

    sim_backend_cpu_set_vdsp_enabled(&backend, true);
    backend_launch(&backend, &kernel);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "vDSP launch failed for op %d: %d\n", (int)op, (int)backend.last_error);
        goto cleanup;
    }
    memcpy(vdsp_out, dst_data, sizeof(vdsp_out));

    memset(dst_data, 0, sizeof(vdsp_out));
    sim_backend_cpu_set_vdsp_enabled(&backend, false);
    backend_launch(&backend, &kernel);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "scalar launch failed for op %d: %d\n", (int)op, (int)backend.last_error);
        goto cleanup;
    }

    for (size_t i = 0; i < TEST_ELEMENT_COUNT; ++i) {
        double diff = fabs(vdsp_out[i] - dst_data[i]);
        if (diff > MAX_DIFF || isnan(vdsp_out[i]) || isnan(dst_data[i])) {
            fprintf(stderr,
                    "mismatch for op %d at element %zu: vdsp=%0.15f scalar=%0.15f diff=%g\n",
                    (int)op, i, vdsp_out[i], dst_data[i], diff);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    backend_destroy(&backend);
    if (builder_ready) {
        sim_ir_builder_destroy(&builder);
    }
    sim_field_destroy(&lhs);
    sim_field_destroy(&rhs);
    sim_field_destroy(&dst);
    return ok;
}

static bool verify_complex_rotate_case(void) {
    SimField src = {0};
    SimField dst = {0};
    SimIRBuilder builder;
    memset(&builder, 0, sizeof(builder));
    bool builder_ready = false;
    SimBackend backend = {0};
    size_t shape[1] = {TEST_ELEMENT_COUNT};
    SimResult rc;
    bool ok = false;
    const double dt = 0.125;
    const double rate = 1.3;
    double params[SIM_IR_PARAM_TIME + 1U] = {0};

    rc =
        sim_field_init(&src, 1, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init complex src field: %d\n", (int)rc);
        goto cleanup;
    }
    rc =
        sim_field_init(&dst, 1, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init complex dst field: %d\n", (int)rc);
        goto cleanup;
    }

    SimComplexDouble *src_data = sim_field_complex_data(&src);
    SimComplexDouble *dst_data = sim_field_complex_data(&dst);
    for (size_t i = 0U; i < TEST_ELEMENT_COUNT; ++i) {
        double x = (double)(i + 1U) * 0.013;
        src_data[i].re = sin(x) * 1.5 + cos(0.2 * x);
        src_data[i].im = cos(0.7 * x) * 0.75 - sin(0.5 * x);
        dst_data[i].re = 0.0;
        dst_data[i].im = 0.0;
    }

    rc = sim_ir_builder_init(&builder);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init IR builder for complex rotate: %d\n", (int)rc);
        goto cleanup;
    }
    builder_ready = true;

    SimIRNodeId field_node = sim_ir_builder_field_ref_typed(&builder, 0U, sim_ir_type_complex());
    SimIRNodeId dt_node = sim_ir_builder_param(&builder, SIM_IR_PARAM_DT);
    SimIRNodeId rate_node = sim_ir_builder_constant(&builder, rate);
    SimIRNodeId angle = sim_ir_builder_binary(&builder, SIM_IR_NODE_MUL, dt_node, rate_node);
    SimIRNodeId expr = sim_ir_builder_complex_rotate(&builder, field_node, angle);
    if (field_node == SIM_IR_INVALID_NODE || dt_node == SIM_IR_INVALID_NODE ||
        rate_node == SIM_IR_INVALID_NODE || angle == SIM_IR_INVALID_NODE ||
        expr == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "failed to build complex rotate IR\n");
        goto cleanup;
    }

    SimKernelIRBinding bindings[2] = {{.field_index = 0, .field = &src},
                                      {.field_index = 1, .field = &dst}};
    SimKernelIROutput output = {.field_index = 1, .expression = expr};
    params[SIM_IR_PARAM_DT] = dt;
    KernelIR kernel = {.builder = &builder,
                       .bindings = bindings,
                       .binding_count = 2U,
                       .outputs = &output,
                       .output_count = 1U,
                       .params = params,
                       .param_count = SIM_IR_PARAM_TIME + 1U,
                       .required_features = SIM_BACKEND_FEATURE_NONE};

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);

    SimComplexDouble vdsp_out[TEST_ELEMENT_COUNT];

    sim_backend_cpu_set_vdsp_enabled(&backend, true);
    backend_launch(&backend, &kernel);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "vDSP complex rotate launch failed: %d\n", (int)backend.last_error);
        goto cleanup;
    }
    memcpy(vdsp_out, dst_data, sizeof(vdsp_out));

    memset(dst_data, 0, TEST_ELEMENT_COUNT * sizeof(SimComplexDouble));
    sim_backend_cpu_set_vdsp_enabled(&backend, false);
    backend_launch(&backend, &kernel);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "scalar complex rotate launch failed: %d\n", (int)backend.last_error);
        goto cleanup;
    }

    for (size_t i = 0U; i < TEST_ELEMENT_COUNT; ++i) {
        double diff_re = fabs(vdsp_out[i].re - dst_data[i].re);
        double diff_im = fabs(vdsp_out[i].im - dst_data[i].im);
        if (diff_re > MAX_DIFF || diff_im > MAX_DIFF || isnan(vdsp_out[i].re) ||
            isnan(vdsp_out[i].im) || isnan(dst_data[i].re) || isnan(dst_data[i].im)) {
            fprintf(stderr,
                    "complex rotate mismatch at element %zu: vdsp=(%0.15f,%0.15f) "
                    "scalar=(%0.15f,%0.15f)\n",
                    i, vdsp_out[i].re, vdsp_out[i].im, dst_data[i].re, dst_data[i].im);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    backend_destroy(&backend);
    if (builder_ready) {
        sim_ir_builder_destroy(&builder);
    }
    sim_field_destroy(&src);
    sim_field_destroy(&dst);
    return ok;
}

int main(void) {
    SimIRNodeType ops[] = {SIM_IR_NODE_ADD, SIM_IR_NODE_SUB, SIM_IR_NODE_MUL, SIM_IR_NODE_DIV};

    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i) {
        if (!verify_case(ops[i])) {
            return EXIT_FAILURE;
        }
    }
    if (!verify_complex_rotate_case()) {
        return EXIT_FAILURE;
    }

    fprintf(stdout, "backend_cpu_vdsp parity checks passed.\n");
    return EXIT_SUCCESS;
}

#endif /* SIM_HAVE_VDSP */
