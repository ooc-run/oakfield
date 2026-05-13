/**
 * @file kernel_ir_eval.c
 * @brief Real-valued KernelIR evaluator for CPU fallback execution.
 *
 * The evaluator walks KernelIR nodes with memoized scalar results, using caller
 * callbacks for field reads, differential terms, noise, and stateful nodes. It
 * implements the real-domain arithmetic path and reports invalid nodes,
 * unsupported callbacks, or allocation failure through SimResult status codes.
 */
#include "internal/kernel_ir_internal.h"

bool sim_ir_constant_component(const SimIRBuilder* builder,
                                      const SimIRNode*    node,
                                      size_t              component,
                                      double*             out_value) {
    if (builder == NULL || node == NULL || out_value == NULL) {
        return false;
    }

    if (node->type != SIM_IR_NODE_CONSTANT) {
        return false;
    }

    if (node->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX) {
        if (node->value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY &&
            !sim_ir_type_is_scalar(node->value_type)) {
            if (component >= node->value_type.components) {
                return false;
            }
            *out_value = node->data.constant.small[component];
            return true;
        }

        *out_value = node->data.constant.scalar;
        return true;
    }

    if (builder->constants_data == NULL ||
        node->data.constant.constant_index >= builder->constants_count) {
        return false;
    }

    size_t idx        = node->data.constant.constant_index;
    size_t offset     = builder->constants_offsets[idx];
    size_t components = builder->constants_components[idx];
    if (component >= components) {
        return false;
    }
    *out_value = builder->constants_data[offset + component];
    return true;
}

static SimResult
sim_ir_evaluate_impl(SimIREvalState* state, SimIRNodeId node_id, double* out_value) {
    const SimIRNode* node;
    double           value  = 0.0;
    SimResult        result = SIM_RESULT_OK;

    if (state == NULL || out_value == NULL || state->builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= state->builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->flags != NULL) {
        if (state->flags[node_id] == 2U) {
            *out_value = state->cache[node_id];
            return SIM_RESULT_OK;
        }
        if (state->flags[node_id] == 1U) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }
        state->flags[node_id] = 1U;
    }

    node = &state->builder->nodes[node_id];

    SimIRType node_type = sim_ir_type_normalize(node->value_type);
    if (!sim_ir_type_is_scalar(node_type) ||
        sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(node_type)) ||
        sim_ir_domain_is_exact_integer(sim_ir_type_scalar_domain(node_type))) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto done;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
            if (!sim_ir_constant_component(state->builder, node, 0U, &value)) {
                result = SIM_RESULT_INVALID_ARGUMENT;
            }
            break;

        case SIM_IR_NODE_FIELD_REF:
            if (state->evaluator == NULL || state->evaluator->field_value == NULL) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            result = state->evaluator->field_value(
                state->evaluator->userdata, node->data.field, node->value_type, &value);
            break;

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV:
        case SIM_IR_NODE_POW: {
            double lhs;
            double rhs;

            result = sim_ir_evaluate_impl(state, node->data.binary.lhs, &lhs);
            if (result != SIM_RESULT_OK) {
                break;
            }

            result = sim_ir_evaluate_impl(state, node->data.binary.rhs, &rhs);
            if (result != SIM_RESULT_OK) {
                break;
            }

            switch (node->type) {
                case SIM_IR_NODE_ADD:
                    value = lhs + rhs;
                    break;
                case SIM_IR_NODE_SUB:
                    value = lhs - rhs;
                    break;
                case SIM_IR_NODE_MUL:
                    value = lhs * rhs;
                    break;
                case SIM_IR_NODE_DIV:
                    value = lhs / rhs;
                    break;
                case SIM_IR_NODE_POW:
                    value = pow(lhs, rhs);
                    break;
                default:
                    break;
            }
            break;
        }

        case SIM_IR_NODE_DIFF: {
            double operand_value = 0.0;

            result = sim_ir_evaluate_impl(state, node->data.diff.operand, &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            if (state->evaluator == NULL || state->evaluator->differential == NULL) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }

            result = state->evaluator->differential(state->evaluator->userdata,
                                                    state->builder,
                                                    node,
                                                    node->data.diff.operand,
                                                    operand_value,
                                                    &value);
            break;
        }

        case SIM_IR_NODE_NOISE:
            if (state->evaluator == NULL || state->evaluator->noise == NULL) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            result = state->evaluator->noise(state->evaluator->userdata, node, &value);
            break;

        case SIM_IR_NODE_PARAM:
            if (state->evaluator == NULL || state->evaluator->param_value == NULL) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            result = state->evaluator->param_value(
                state->evaluator->userdata, node->data.param.param, &value);
            break;

        case SIM_IR_NODE_INDEX:
        case SIM_IR_NODE_STATEFUL:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;

        case SIM_IR_NODE_CALL: {
            double operand_value = 0.0;
            result = sim_ir_evaluate_impl(state, node->data.call.operand, &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            switch (node->data.call.kind) {
                case SIM_IR_CALL_SIN:
                    value = sin(operand_value);
                    break;
                case SIM_IR_CALL_COS:
                    value = cos(operand_value);
                    break;
                case SIM_IR_CALL_EXP:
                    value = exp(operand_value);
                    break;
                case SIM_IR_CALL_ABS:
                    value = fabs(operand_value);
                    break;
                case SIM_IR_CALL_LOG:
                    value = log(operand_value);
                    break;
                case SIM_IR_CALL_TANH:
                    value = tanh(operand_value);
                    break;
                case SIM_IR_CALL_SINH:
                    value = sinh(operand_value);
                    break;
                case SIM_IR_CALL_SIGN:
                    value = copysign(1.0, operand_value);
                    break;
                default:
                    result = SIM_RESULT_INVALID_ARGUMENT;
                    break;
            }
            break;
        }

        case SIM_IR_NODE_FLOOR: {
            double operand_value = 0.0;
            result = sim_ir_evaluate_impl(state, node->data.unary.operand, &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            value = floor(operand_value);
            break;
        }

        case SIM_IR_NODE_MOD: {
            double lhs = 0.0;
            double rhs = 0.0;
            result     = sim_ir_evaluate_impl(state, node->data.binary.lhs, &lhs);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_evaluate_impl(state, node->data.binary.rhs, &rhs);
            if (result != SIM_RESULT_OK) {
                break;
            }
            if (rhs == 0.0) {
                value = 0.0;
            } else {
                value = fmod(lhs, rhs);
            }
            break;
        }

        case SIM_IR_NODE_COMPLEX_ROTATE:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;

        case SIM_IR_NODE_WARP: {
            double operand_value = 0.0;

            result = sim_ir_evaluate_impl(state, node->data.warp.operand, &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            SimWarpGuard      guard       = sim_ir_guard_to_runtime(&node->data.warp.guard);
            SimWarpSampleSpec sample_spec = { .sample = operand_value,
                                              .bias   = node->data.warp.bias,
                                              .delta  = node->data.warp.delta,
                                              .lambda = node->data.warp.lambda,
                                              .guard  = guard };

            double    response = 0.0;
            SimResult warp_rc  = sim_ir_warp_sample_response(&sample_spec,
                                                            node->data.warp.profile,
                                                            node->data.warp.tolerance,
                                                            NULL,
                                                            NULL,
                                                            &response);
            if (warp_rc != SIM_RESULT_OK) {
                value  = 0.0;
                result = warp_rc;
                break;
            }

            value = response;
            break;
        }

        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

done:
    if (state->flags != NULL) {
        if (result == SIM_RESULT_OK) {
            state->cache[node_id] = value;
            state->flags[node_id] = 2U;
        } else {
            state->flags[node_id] = 0U;
        }
    }

    if (result == SIM_RESULT_OK) {
        *out_value = value;
    }

    return result;
}

SimResult sim_ir_evaluate(const SimIRBuilder*   builder,
                          SimIRNodeId           root,
                          const SimIREvaluator* evaluator,
                          double*               out_value) {
    SimIREvalState state = { 0 };
    SimResult      result;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (builder == NULL || root == SIM_IR_INVALID_NODE || root >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimIRType root_type = sim_ir_type_normalize(builder->nodes[root].value_type);
    if (!sim_ir_type_is_scalar(root_type) ||
        sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(root_type)) ||
        sim_ir_domain_is_exact_integer(sim_ir_type_scalar_domain(root_type))) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state.builder           = builder;
    state.evaluator         = evaluator;
    state.evaluator_complex = NULL;
    state.cache_complex     = NULL;

    if (builder->count > 0U) {
        state.cache = (double*) calloc(builder->count, sizeof(double));
        state.flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
        if ((state.cache == NULL || state.flags == NULL)) {
            free(state.cache);
            free(state.flags);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    result = sim_ir_evaluate_impl(&state, root, out_value);

    free(state.cache);
    free(state.flags);
    return result;
}
