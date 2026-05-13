#include <oakfield/sim.h>

#include "graph_ir.h"

#include <stdbool.h>
#include <stdio.h>

static bool run_graph_ir_validation_case(SimGraphIRTimeSource time_source, bool has_state,
                                         bool reset_on_rewind, SimRepresentationMode mode,
                                         bool expect_ok) {
    SimContext ctx = {0};
    SimField input = {0};
    SimField output = {0};
    bool ctx_ready = false;
    bool input_ready = false;
    bool output_ready = false;
    size_t shape[1] = {4U};
    size_t input_index = 0U;
    size_t output_index = 0U;
    SimGraphIR graph = {0};
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] graph_ir_policy: ctx init failed\n");
        return false;
    }
    ctx_ready = true;
    sim_context_set_representation_mode(&ctx, mode);

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] graph_ir_policy: field init failed\n");
        goto cleanup;
    }
    input_ready = true;
    output_ready = true;

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] graph_ir_policy: add field failed\n");
        goto cleanup;
    }
    input_ready = false;
    output_ready = false;

    sim_graph_ir_init(&graph);

    SimGraphIRNodeDesc desc = {0};
    desc.kind = SIM_GRAPH_IR_NODE_CAST_COPY;
    desc.input.kind = SIM_GRAPH_IR_INPUT_FIELD;
    desc.input.field_index = input_index;
    desc.output.kind = SIM_GRAPH_IR_OUTPUT_FIELD;
    desc.output.field_index = output_index;
    desc.contract.input.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    desc.contract.input.value_kind = SIM_FIELD_VALUE_REAL_SCALAR;
    desc.contract.output.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    desc.contract.output.value_kind = SIM_FIELD_VALUE_REAL_SCALAR;
    desc.contract.output.preserves_real_subspace = true;
    desc.contract.output.preserves_real_constraint = true;
    desc.contract.purity.time_source = time_source;
    desc.contract.purity.dt_source = SIM_GRAPH_IR_DT_NONE;
    desc.contract.purity.has_state = has_state;
    desc.contract.purity.reset_on_rewind = reset_on_rewind;

    SimResult rc = sim_graph_ir_add_node(&graph, &desc, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] graph_ir_policy: add node failed (%d)\n", rc);
        goto cleanup;
    }

    SimGraphIRCompileContext graph_ctx = {0};
    sim_graph_ir_compile_context_init(&graph_ctx, &ctx);
    rc = sim_graph_ir_validate(&graph, &graph_ctx);

    if (expect_ok && rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] graph_ir_policy: expected OK, got %d\n", rc);
        goto cleanup;
    }
    if (!expect_ok && rc == SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] graph_ir_policy: expected failure, got OK\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_graph_ir_destroy(&graph);
    if (ctx_ready) {
        sim_context_destroy(&ctx);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return ok;
}

int main(void) {
    bool ok = true;

    ok = ok && run_graph_ir_validation_case(SIM_GRAPH_IR_TIME_STEP_PURE, false, true,
                                            SIM_REPRESENTATION_MODE_STRICT, true);
    ok = ok && run_graph_ir_validation_case(SIM_GRAPH_IR_TIME_ACCUMULATED, false, true,
                                            SIM_REPRESENTATION_MODE_STRICT, false);
    ok = ok && run_graph_ir_validation_case(SIM_GRAPH_IR_TIME_NONE, true, false,
                                            SIM_REPRESENTATION_MODE_STRICT, false);
    ok = ok && run_graph_ir_validation_case(SIM_GRAPH_IR_TIME_ACCUMULATED, false, true,
                                            SIM_REPRESENTATION_MODE_EXPLORATION, true);

    return ok ? 0 : 1;
}
