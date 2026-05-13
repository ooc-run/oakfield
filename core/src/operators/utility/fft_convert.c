#include "oakfield/operators/utility/fft_convert.h"
#include "oakfield/sim_context.h"
#include "oakfield/field.h"
#include "graph_ir.h"

#include <stdlib.h>
#include <limits.h>

typedef struct SimFFTConvertState {
    size_t     input_field;
    size_t     output_field;
    bool       inverse;
    bool       input_was_real;
    bool       inverse_output_real;
    SimGraphIR graph;
} SimFFTConvertState;

static void sim_fft_convert_state_release(SimFFTConvertState* state) {
    if (state == NULL) {
        return;
    }
    sim_graph_ir_destroy(&state->graph);
}

static void sim_fft_convert_state_destroy(void* userdata) {
    SimFFTConvertState* state = (SimFFTConvertState*) userdata;
    if (state == NULL) {
        return;
    }
    sim_fft_convert_state_release(state);
    free(state);
}

static const SimGraphIR* sim_fft_convert_graph_ir_view(const struct SimOperator* self,
                                                       void*                     userdata) {
    const SimFFTConvertState* state = (const SimFFTConvertState*) userdata;
    (void) self;
    return (state != NULL) ? &state->graph : NULL;
}

static SimResult
sim_fft_convert_eval(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) self;
    SimFFTConvertState* state = (SimFFTConvertState*) userdata;
    if (context == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimGraphIRCompileContext graph_ctx = { 0 };
    sim_graph_ir_compile_context_init(&graph_ctx, context);
    return sim_graph_ir_execute(&state->graph, &graph_ctx);
}

SimResult sim_add_fft_convert(struct SimContext*     context,
                              size_t                 input_field,
                              SimFFTConvertDirection direction,
                              bool                   in_place,
                              size_t*                out_field_index,
                              size_t*                out_operator_index) {
    SimFFTConvertState*    state = NULL;
    SimField*              input = NULL;
    SimFieldRepresentation input_repr;
    SimField               new_field;
    size_t                 output_field = input_field;
    SimOperatorDescriptor  desc         = { 0 };
    SimResult              rc;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* Domain transitions are explicit: this operator always writes into a dedicated output field. */
    if (in_place) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    input = sim_context_field(context, input_field);
    if (input == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    input_repr = sim_field_representation(input);

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimFFTConvertState*) calloc(1U, sizeof(SimFFTConvertState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->inverse        = (direction == SIM_FFT_CONVERT_INVERSE);
    state->input_field    = input_field;
    state->input_was_real = (input_repr.value_kind == SIM_FIELD_VALUE_REAL_SCALAR);
    state->inverse_output_real =
        state->inverse && (input_repr.value_kind == SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT);

    if (state->inverse && input_repr.value_kind == SIM_FIELD_VALUE_REAL_SCALAR) {
        free(state);
        return SIM_RESULT_TYPE_MISMATCH;
    }

    /* Allocate output field. */
    size_t element_size;
    if (state->inverse) {
        element_size = state->inverse_output_real ? sizeof(double) : sizeof(SimComplexDouble);
    } else {
        element_size = sizeof(SimComplexDouble);
    }

    size_t element_count = sim_field_element_count(&input->layout);
    if (element_count == 0U || element_size == 0U || element_count > (SIZE_MAX / element_size)) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    size_t output_bytes = element_count * element_size;
    rc                  = sim_context_check_field_limits(context, element_count, output_bytes);
    if (rc != SIM_RESULT_OK) {
        free(state);
        return rc;
    }

    rc = sim_field_init(
        &new_field, input->layout.rank, input->layout.shape, element_size, input->storage, NULL);
    if (rc != SIM_RESULT_OK) {
        free(state);
        return rc;
    }

    if (state->inverse) {
        new_field.repr.domain     = SIM_FIELD_DOMAIN_PHYSICAL;
        new_field.repr.value_kind = state->inverse_output_real ? SIM_FIELD_VALUE_REAL_SCALAR
                                                               : SIM_FIELD_VALUE_COMPLEX_SCALAR;
    } else {
        new_field.repr.domain     = SIM_FIELD_DOMAIN_SPECTRAL;
        new_field.repr.value_kind = state->input_was_real ? SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT
                                                          : SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }

    rc = sim_context_add_field(context, &new_field, &output_field);
    if (rc != SIM_RESULT_OK) {
        sim_field_destroy(&new_field);
        free(state);
        return rc;
    }
    state->output_field = output_field;

    sim_graph_ir_init(&state->graph);

    SimGraphIRNodeDesc node_desc              = { 0 };
    node_desc.contract.purity.time_source     = SIM_GRAPH_IR_TIME_NONE;
    node_desc.contract.purity.dt_source       = SIM_GRAPH_IR_DT_NONE;
    node_desc.contract.purity.has_state       = false;
    node_desc.contract.purity.reset_on_rewind = true;

    SimGraphIRNodeId canon_node = SIM_GRAPH_IR_INVALID_NODE;
    if (!state->inverse) {
        node_desc.kind                       = SIM_GRAPH_IR_NODE_FFT_FORWARD;
        node_desc.input.kind                 = SIM_GRAPH_IR_INPUT_FIELD;
        node_desc.input.field_index          = input_field;
        node_desc.output.kind                = SIM_GRAPH_IR_OUTPUT_FIELD;
        node_desc.output.field_index         = output_field;
        node_desc.contract.input.domain      = SIM_FIELD_DOMAIN_PHYSICAL;
        node_desc.contract.input.value_kind  = input_repr.value_kind;
        node_desc.contract.output.domain     = SIM_FIELD_DOMAIN_SPECTRAL;
        node_desc.contract.output.value_kind = state->input_was_real
                                                   ? SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT
                                                   : SIM_FIELD_VALUE_COMPLEX_SCALAR;
        node_desc.contract.output.preserves_real_subspace   = state->input_was_real;
        node_desc.contract.output.preserves_real_constraint = state->input_was_real;

        rc = sim_graph_ir_add_node(&state->graph, &node_desc, NULL);
        if (rc != SIM_RESULT_OK) {
            sim_fft_convert_state_release(state);
            free(state);
            return rc;
        }
    } else {
        if (state->inverse_output_real) {
            SimGraphIRNodeDesc canon_desc         = node_desc;
            canon_desc.kind                       = SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT;
            canon_desc.input.kind                 = SIM_GRAPH_IR_INPUT_FIELD;
            canon_desc.input.field_index          = input_field;
            canon_desc.output.kind                = SIM_GRAPH_IR_OUTPUT_TEMP;
            canon_desc.contract.input.domain      = SIM_FIELD_DOMAIN_SPECTRAL;
            canon_desc.contract.input.value_kind  = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;
            canon_desc.contract.output.domain     = SIM_FIELD_DOMAIN_SPECTRAL;
            canon_desc.contract.output.value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;
            canon_desc.contract.output.preserves_real_subspace   = true;
            canon_desc.contract.output.preserves_real_constraint = true;
            canon_desc.config.canonicalize.tolerance             = 1e-10;

            rc = sim_graph_ir_add_node(&state->graph, &canon_desc, &canon_node);
            if (rc != SIM_RESULT_OK) {
                sim_fft_convert_state_release(state);
                free(state);
                return rc;
            }
        }

        node_desc.kind = SIM_GRAPH_IR_NODE_FFT_INVERSE;
        if (canon_node != SIM_GRAPH_IR_INVALID_NODE) {
            node_desc.input.kind    = SIM_GRAPH_IR_INPUT_NODE;
            node_desc.input.node_id = canon_node;
        } else {
            node_desc.input.kind        = SIM_GRAPH_IR_INPUT_FIELD;
            node_desc.input.field_index = input_field;
        }
        node_desc.output.kind                = SIM_GRAPH_IR_OUTPUT_FIELD;
        node_desc.output.field_index         = output_field;
        node_desc.contract.input.domain      = SIM_FIELD_DOMAIN_SPECTRAL;
        node_desc.contract.input.value_kind  = state->inverse_output_real
                                                   ? SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT
                                                   : SIM_FIELD_VALUE_COMPLEX_SCALAR;
        node_desc.contract.output.domain     = SIM_FIELD_DOMAIN_PHYSICAL;
        node_desc.contract.output.value_kind = state->inverse_output_real
                                                   ? SIM_FIELD_VALUE_REAL_SCALAR
                                                   : SIM_FIELD_VALUE_COMPLEX_SCALAR;
        node_desc.contract.output.preserves_real_subspace   = state->inverse_output_real;
        node_desc.contract.output.preserves_real_constraint = false;

        SimGraphIRNodeId fft_node = SIM_GRAPH_IR_INVALID_NODE;
        rc                        = sim_graph_ir_add_node(&state->graph, &node_desc, &fft_node);
        if (rc != SIM_RESULT_OK) {
            sim_fft_convert_state_release(state);
            free(state);
            return rc;
        }

        if (canon_node != SIM_GRAPH_IR_INVALID_NODE) {
            rc = sim_graph_ir_add_edge(&state->graph, canon_node, fft_node);
            if (rc != SIM_RESULT_OK) {
                sim_fft_convert_state_release(state);
                free(state);
                return rc;
            }
        }
    }

    SimGraphIRCompileContext graph_ctx = { 0 };
    sim_graph_ir_compile_context_init(&graph_ctx, context);
    rc = sim_graph_ir_validate(&state->graph, &graph_ctx);
    if (rc != SIM_RESULT_OK) {
        sim_fft_convert_state_release(state);
        free(state);
        return rc;
    }

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_UTILITY;
    info.is_linear         = true;
    info.is_local          = false;
    info.is_nonlocal       = true;
    info.is_differentiable = true;
    info.preserves_real    = state->inverse_output_real || state->input_was_real;
    info.abstract_id       = state->inverse ? "fft_inverse" : "fft_forward";
    info.representation.domain =
        state->inverse ? SIM_FIELD_DOMAIN_PHYSICAL : SIM_FIELD_DOMAIN_SPECTRAL;
    info.representation.value_kind =
        state->inverse ? (state->inverse_output_real ? SIM_FIELD_VALUE_REAL_SCALAR
                                                     : SIM_FIELD_VALUE_COMPLEX_SCALAR)
                       : (state->input_was_real ? SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT
                                                : SIM_FIELD_VALUE_COMPLEX_SCALAR);
    info.representation.requires_complex_input = state->inverse ? true : !state->input_was_real;
    info.representation.requires_complex_representation =
        state->inverse ? !state->inverse_output_real : true;
    info.representation.preserves_real_subspace =
        state->inverse_output_real || state->input_was_real;

    desc.name          = state->inverse ? "fft_inverse" : "fft_forward";
    desc.evaluate      = sim_fft_convert_eval;
    desc.destroy       = (SimOperatorDestroyFn) sim_fft_convert_state_destroy;
    desc.userdata      = state;
    desc.info          = info;
    desc.graph_ir_view = sim_fft_convert_graph_ir_view;
    desc.read_mask     = (input_field < 64U) ? (1ULL << input_field) : 0ULL;
    desc.write_mask    = (output_field < 64U) ? (1ULL << output_field) : 0ULL;

    rc = sim_context_register_operator(context, &desc, out_operator_index);
    if (rc != SIM_RESULT_OK) {
        sim_fft_convert_state_release(state);
        free(state);
        return rc;
    }

    if (out_field_index != NULL) {
        *out_field_index = output_field;
    }

    return SIM_RESULT_OK;
}
