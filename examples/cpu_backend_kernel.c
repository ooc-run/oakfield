#include <oakfield/backend.h>

#include <stdio.h>

static int fail_result(const char* label, SimResult result) {
    fprintf(stderr, "%s failed (%d)\n", label, result);
    return 1;
}

int main(void) {
    SimField src      = { 0 };
    SimField dst      = { 0 };
    size_t   shape[1] = { 8U };
    SimResult result;

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        return fail_result("src init", result);
    }
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&src);
        return fail_result("dst init", result);
    }

    double* src_data = sim_field_real_data(&src);
    if (src_data == NULL) {
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        return fail_result("src data", SIM_RESULT_INVALID_STATE);
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        src_data[i] = (double) i;
    }

    SimIRBuilder builder = { 0 };
    result = sim_ir_builder_init(&builder);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        return fail_result("IR builder init", result);
    }

    SimIRNodeId src_node = sim_ir_builder_field_ref(&builder, 0U);
    SimIRNodeId scale    = sim_ir_builder_constant(&builder, 2.0);
    SimIRNodeId bias     = sim_ir_builder_constant(&builder, 1.0);
    SimIRNodeId product  = sim_ir_builder_binary(&builder, SIM_IR_NODE_MUL, src_node, scale);
    SimIRNodeId affine   = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, product, bias);
    if (src_node == SIM_IR_INVALID_NODE || scale == SIM_IR_INVALID_NODE ||
        bias == SIM_IR_INVALID_NODE || product == SIM_IR_INVALID_NODE ||
        affine == SIM_IR_INVALID_NODE) {
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        return fail_result("IR node creation", SIM_RESULT_OUT_OF_MEMORY);
    }

    SimKernelIRBinding bindings[2] = {
        { 0U, &src, sim_field_shape(&src), sim_field_strides(&src), sim_field_rank(&src) },
        { 1U, &dst, sim_field_shape(&dst), sim_field_strides(&dst), sim_field_rank(&dst) },
    };
    SimKernelIROutput output = {
        .field_index = 1U,
        .expression = affine,
    };
    KernelIR kernel = {
        .builder = &builder,
        .bindings = bindings,
        .binding_count = 2U,
        .outputs = &output,
        .output_count = 1U,
        .params = NULL,
        .param_count = 0U,
        .required_features = 0U,
        .complex_semantics = SIM_KERNEL_COMPLEX_SEMANTICS_TRUE_COMPLEX,
    };

    SimBackend backend = { .type = SIM_BACKEND_TYPE_CPU };
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        result = backend.last_error;
        backend_destroy(&backend);
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        return fail_result("backend_init", result);
    }

    backend_launch(&backend, &kernel);
    if (backend.last_error != SIM_RESULT_OK) {
        result = backend.last_error;
        backend_destroy(&backend);
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        return fail_result("backend_launch", result);
    }

    const double* dst_data = sim_field_real_data_const(&dst);
    printf("dst = 2 * src + 1:");
    for (size_t i = 0U; i < shape[0]; ++i) {
        printf(" %.1f", dst_data[i]);
    }
    printf("\n");

    backend_destroy(&backend);
    sim_ir_builder_destroy(&builder);
    sim_field_destroy(&dst);
    sim_field_destroy(&src);
    return 0;
}
