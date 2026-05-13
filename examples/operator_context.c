#include <oakfield/core.h>

#include <stdio.h>

typedef struct BiasOperatorState {
    size_t input_field;
    size_t output_field;
    double bias;
} BiasOperatorState;

static int fail_result(const char* label, SimResult result) {
    fprintf(stderr, "%s failed (%d)\n", label, result);
    return 1;
}

static SimResult bias_operator_eval(SimContext* context, SimOperator* self, void* userdata) {
    (void) self;
    BiasOperatorState* state = (BiasOperatorState*) userdata;
    if (context == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* src = sim_context_field(context, state->input_field);
    SimField* dst = sim_context_field(context, state->output_field);
    const double* src_data = (src != NULL) ? sim_field_real_data_const(src) : NULL;
    double*       dst_data = (dst != NULL) ? sim_field_real_data(dst) : NULL;
    if (src_data == NULL || dst_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t src_count = sim_field_element_count(&src->layout);
    size_t dst_count = sim_field_element_count(&dst->layout);
    if (src_count != dst_count) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    for (size_t i = 0U; i < src_count; ++i) {
        dst_data[i] = src_data[i] + state->bias;
    }
    return SIM_RESULT_OK;
}

int main(void) {
    SimContext context = { 0 };
    SimField   src = { 0 };
    SimField   dst = { 0 };
    size_t     shape[1] = { 8U };
    size_t     src_index = 0U;
    size_t     dst_index = 0U;
    SimResult  result;

    result = sim_context_init(&context);
    if (result != SIM_RESULT_OK) {
        return fail_result("sim_context_init", result);
    }

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return fail_result("src field init", result);
    }
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&src);
        sim_context_destroy(&context);
        return fail_result("dst field init", result);
    }

    double* src_data = sim_field_real_data(&src);
    if (src_data == NULL) {
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        sim_context_destroy(&context);
        return fail_result("src data", SIM_RESULT_INVALID_STATE);
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        src_data[i] = (double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        sim_context_destroy(&context);
        return fail_result("add src field", result);
    }
    result = sim_context_add_field(&context, &dst, &dst_index);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&dst);
        sim_context_destroy(&context);
        return fail_result("add dst field", result);
    }

    BiasOperatorState state = {
        .input_field = src_index,
        .output_field = dst_index,
        .bias = 1.5,
    };
    size_t reads[1] = { src_index };
    size_t writes[1] = { dst_index };
    SimOperatorInfo info = sim_operator_info_defaults();
    info.category = SIM_OPERATOR_CATEGORY_UTILITY;
    info.is_local = true;
    info.is_linear = false;
    info.preserves_real = true;
    sim_operator_info_set_schema_identity(&info, "example_bias");

    SimOperatorDescriptor descriptor = {
        .name = "example_bias",
        .evaluate = bias_operator_eval,
        .userdata = &state,
        .info = info,
        .read_indices = reads,
        .read_index_count = 1U,
        .write_indices = writes,
        .write_index_count = 1U,
    };
    size_t operator_index = 0U;
    result = sim_context_register_operator(&context, &descriptor, &operator_index);
    if (result != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return fail_result("sim_context_register_operator", result);
    }

    sim_context_set_timestep(&context, 0.25);
    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return fail_result("sim_context_execute", result);
    }
    sim_context_accept_step(&context, sim_context_timestep(&context));

    const SimField* executed_dst = sim_context_field(&context, dst_index);
    const double*   dst_data =
        (executed_dst != NULL) ? sim_field_real_data_const(executed_dst) : NULL;
    if (dst_data == NULL) {
        sim_context_destroy(&context);
        return fail_result("dst data", SIM_RESULT_INVALID_STATE);
    }

    printf("operator[%zu] context step=%zu time=%.2f dst:",
           operator_index,
           sim_context_step_index(&context),
           sim_context_time(&context));
    for (size_t i = 0U; i < shape[0]; ++i) {
        printf(" %.1f", dst_data[i]);
    }
    printf("\n");

    sim_context_destroy(&context);
    return 0;
}
