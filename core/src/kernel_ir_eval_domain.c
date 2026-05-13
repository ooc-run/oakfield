/**
 * @file kernel_ir_eval_domain.c
 * @brief Scalar-domain-aware KernelIR evaluator for real, complex, and integer nodes.
 *
 * This evaluator preserves exact integer payloads where possible, converts field
 * and callback values through SimScalarDomain descriptors, and falls back to
 * real or complex arithmetic for domains that require floating-point semantics.
 * It is the contract-enforcing path for typed constants, modulo, floor, and
 * domain-specific field accesses.
 */
#include "internal/kernel_ir_internal.h"

static bool sim_ir_constant_integer_raw(const SimIRNode* node, uint64_t* out_raw) {
    SimScalarDomain domain;

    if (node == NULL || out_raw == NULL || node->type != SIM_IR_NODE_CONSTANT) {
        return false;
    }

    domain = sim_ir_type_scalar_domain(node->value_type);
    if (!sim_ir_domain_is_exact_integer(domain)) {
        return false;
    }

    if (node->data.constant.exact_integer) {
        *out_raw = sim_ir_integer_truncate(node->data.constant.unsigned_scalar, domain);
        return true;
    }

    return sim_ir_integer_raw_from_double(node->data.constant.scalar, domain, out_raw);
}

typedef struct SimIRIntegerEvalState {
    const SimIRBuilder*         builder;
    const SimIREvaluatorDomain* evaluator;
    uint64_t*                   cache;
    unsigned char*              flags;
} SimIRIntegerEvalState;

static SimResult sim_ir_evaluate_impl_integer(SimIRIntegerEvalState* state,
                                              SimIRNodeId            node_id,
                                              uint64_t*              out_raw) {
    const SimIRNode* node;
    SimScalarDomain  domain;
    uint64_t         raw    = 0U;
    SimResult        result = SIM_RESULT_OK;

    if (state == NULL || out_raw == NULL || state->builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (node_id == SIM_IR_INVALID_NODE || node_id >= state->builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->flags != NULL) {
        if (state->flags[node_id] == 2U) {
            *out_raw = state->cache[node_id];
            return SIM_RESULT_OK;
        }
        if (state->flags[node_id] == 1U) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }
        state->flags[node_id] = 1U;
    }

    node   = &state->builder->nodes[node_id];
    domain = sim_ir_type_scalar_domain(node->value_type);
    if (!sim_ir_type_is_scalar(node->value_type) || !sim_ir_domain_is_exact_integer(domain)) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto done;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
            if (!sim_ir_constant_integer_raw(node, &raw)) {
                result = SIM_RESULT_INVALID_ARGUMENT;
            }
            break;

        case SIM_IR_NODE_FIELD_REF: {
            SimIRDomainValue value;
            if (state->evaluator == NULL || state->evaluator->field_value == NULL) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            value  = sim_ir_domain_value_zero(domain);
            result = state->evaluator->field_value(
                state->evaluator->userdata, node->data.field, node->value_type, domain, &value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            if (!sim_ir_domain_value_to_integer_raw(value, domain, &raw)) {
                result = SIM_RESULT_TYPE_MISMATCH;
            }
            break;
        }

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV:
        case SIM_IR_NODE_MOD:
        case SIM_IR_NODE_POW: {
            uint64_t lhs_raw = 0U;
            uint64_t rhs_raw = 0U;

            result = sim_ir_evaluate_impl_integer(state, node->data.binary.lhs, &lhs_raw);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_evaluate_impl_integer(state, node->data.binary.rhs, &rhs_raw);
            if (result != SIM_RESULT_OK) {
                break;
            }

            switch (node->type) {
                case SIM_IR_NODE_ADD:
                    raw = sim_ir_integer_truncate(lhs_raw + rhs_raw, domain);
                    break;
                case SIM_IR_NODE_SUB:
                    raw = sim_ir_integer_truncate(lhs_raw - rhs_raw, domain);
                    break;
                case SIM_IR_NODE_MUL:
                    raw = sim_ir_integer_truncate(lhs_raw * rhs_raw, domain);
                    break;
                case SIM_IR_NODE_DIV:
                case SIM_IR_NODE_MOD:
                    if (sim_ir_integer_truncate(rhs_raw, domain) == 0U) {
                        result = SIM_RESULT_INVALID_ARGUMENT;
                        break;
                    }
                    if (domain.is_signed) {
                        int64_t lhs = sim_ir_integer_as_i64(lhs_raw, domain);
                        int64_t rhs = sim_ir_integer_as_i64(rhs_raw, domain);
                        if (node->type == SIM_IR_NODE_DIV) {
                            if (lhs == INT64_MIN && rhs == -1 && domain.bit_width == 64U) {
                                raw = sim_ir_integer_from_i64(INT64_MIN, domain);
                            } else {
                                raw = sim_ir_integer_from_i64(lhs / rhs, domain);
                            }
                        } else {
                            raw = sim_ir_integer_from_i64((rhs == -1) ? 0 : (lhs % rhs), domain);
                        }
                    } else {
                        uint64_t lhs = sim_ir_integer_truncate(lhs_raw, domain);
                        uint64_t rhs = sim_ir_integer_truncate(rhs_raw, domain);
                        raw          = (node->type == SIM_IR_NODE_DIV) ? (lhs / rhs) : (lhs % rhs);
                    }
                    break;
                case SIM_IR_NODE_POW: {
                    uint64_t exponent = 0U;
                    uint64_t base     = sim_ir_integer_truncate(lhs_raw, domain);
                    uint64_t accum    = sim_ir_integer_truncate(1U, domain);
                    if (domain.is_signed) {
                        int64_t signed_exp = sim_ir_integer_as_i64(rhs_raw, domain);
                        if (signed_exp < 0) {
                            result = SIM_RESULT_INVALID_ARGUMENT;
                            break;
                        }
                        exponent = (uint64_t) signed_exp;
                    } else {
                        exponent = sim_ir_integer_truncate(rhs_raw, domain);
                    }
                    while (exponent > 0U) {
                        if ((exponent & 1U) != 0U) {
                            accum = sim_ir_integer_truncate(accum * base, domain);
                        }
                        exponent >>= 1U;
                        if (exponent > 0U) {
                            base = sim_ir_integer_truncate(base * base, domain);
                        }
                    }
                    raw = accum;
                    break;
                }
                default:
                    break;
            }
            break;
        }

        case SIM_IR_NODE_FLOOR:
            result = sim_ir_evaluate_impl_integer(state, node->data.unary.operand, &raw);
            break;

        case SIM_IR_NODE_PARAM: {
            SimIRDomainValue value;
            if (state->evaluator == NULL || state->evaluator->param_value == NULL) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            value  = sim_ir_domain_value_zero(domain);
            result = state->evaluator->param_value(
                state->evaluator->userdata, node->data.param.param, domain, &value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            if (!sim_ir_domain_value_to_integer_raw(value, domain, &raw)) {
                result = SIM_RESULT_TYPE_MISMATCH;
            }
            break;
        }

        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

done:
    if (state->flags != NULL) {
        if (result == SIM_RESULT_OK) {
            state->cache[node_id] = sim_ir_integer_truncate(raw, domain);
            state->flags[node_id] = 2U;
        } else {
            state->flags[node_id] = 0U;
        }
    }

    if (result == SIM_RESULT_OK) {
        *out_raw = sim_ir_integer_truncate(raw, domain);
    }
    return result;
}

typedef struct SimIREvalDomainAdapterState {
    const SimIREvaluatorDomain* evaluator;
} SimIREvalDomainAdapterState;

static SimResult sim_ir_eval_domain_field_real(void*     userdata,
                                               size_t    field_id,
                                               SimIRType type,
                                               double*   out_value) {
    SimIREvalDomainAdapterState* state;
    SimIRDomainValue             value;
    SimScalarDomain              domain = sim_ir_type_scalar_domain(type);
    SimResult                    rc;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->field_value == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    value = sim_ir_domain_value_zero(domain);
    rc = state->evaluator->field_value(
        state->evaluator->userdata, field_id, type, domain, &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = domain;
    }
    if (!sim_scalar_domain_equal(value.domain, domain) || sim_scalar_domain_is_complex(domain) ||
        sim_ir_domain_is_exact_integer(domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    *out_value = value.value.as_f64;
    return SIM_RESULT_OK;
}

static SimResult sim_ir_eval_domain_diff_real(void*               userdata,
                                              const SimIRBuilder* builder,
                                              const SimIRNode*    node,
                                              SimIRNodeId         operand,
                                              double              operand_value,
                                              double*             out_value) {
    SimIREvalDomainAdapterState* state;
    SimIRDomainValue             operand_domain_value;
    SimIRDomainValue             value;
    SimScalarDomain              domain;
    SimResult                    rc;

    if (builder == NULL || node == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->differential == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    domain                              = sim_ir_type_scalar_domain(node->value_type);
    operand_domain_value                = sim_ir_domain_value_zero(domain);
    operand_domain_value.domain         = domain;
    operand_domain_value.value.as_f64   = operand_value;
    value                               = sim_ir_domain_value_zero(domain);
    rc     = state->evaluator->differential(state->evaluator->userdata,
                                        builder,
                                        node,
                                        operand,
                                        domain,
                                        operand_domain_value,
                                        &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = domain;
    }
    if (!sim_scalar_domain_equal(value.domain, domain) || sim_scalar_domain_is_complex(domain) ||
        sim_ir_domain_is_exact_integer(domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    *out_value = value.value.as_f64;
    return SIM_RESULT_OK;
}

static SimResult sim_ir_eval_domain_noise_real(void* userdata, const SimIRNode* node, double* out_value) {
    SimIREvalDomainAdapterState* state;
    SimIRDomainValue             value;
    SimScalarDomain              domain;
    SimResult                    rc;

    if (node == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->noise == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    domain = sim_ir_type_scalar_domain(node->value_type);
    value  = sim_ir_domain_value_zero(domain);
    rc     = state->evaluator->noise(state->evaluator->userdata, node, domain, &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = domain;
    }
    if (!sim_scalar_domain_equal(value.domain, domain) || sim_scalar_domain_is_complex(domain) ||
        sim_ir_domain_is_exact_integer(domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    *out_value = value.value.as_f64;
    return SIM_RESULT_OK;
}

static SimResult
sim_ir_eval_domain_param_real(void* userdata, SimIRParamKind param, double* out_value) {
    SimIREvalDomainAdapterState* state;
    SimIRDomainValue             value;
    SimResult                    rc;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->param_value == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    value = sim_ir_domain_value_zero(sim_scalar_domain_f64());
    rc = state->evaluator->param_value(
        state->evaluator->userdata, param, sim_scalar_domain_f64(), &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = sim_scalar_domain_f64();
    }
    if (!sim_scalar_domain_equal(value.domain, sim_scalar_domain_f64())) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    *out_value = value.value.as_f64;
    return SIM_RESULT_OK;
}

static SimResult sim_ir_eval_domain_field_complex(void*             userdata,
                                                  size_t            field_id,
                                                  SimIRType         type,
                                                  SimComplexDouble* out_value) {
    SimIREvalDomainAdapterState* state;
    SimScalarDomain              domain;
    SimIRDomainValue             value;
    SimResult                    rc;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->field_value == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    domain = sim_ir_type_scalar_domain(type);
    value  = sim_ir_domain_value_zero(domain);
    rc     = state->evaluator->field_value(
        state->evaluator->userdata, field_id, type, domain, &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = domain;
    }
    if (!sim_scalar_domain_equal(value.domain, domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    *out_value = value.value.as_complex;
    return SIM_RESULT_OK;
}

static SimResult sim_ir_eval_domain_diff_complex(void*               userdata,
                                                 const SimIRBuilder* builder,
                                                 const SimIRNode*    node,
                                                 SimIRNodeId         operand,
                                                 SimComplexDouble    operand_value,
                                                 SimComplexDouble*   out_value) {
    SimIREvalDomainAdapterState* state;
    SimScalarDomain              domain;
    SimIRDomainValue             operand_domain_value;
    SimIRDomainValue             value;
    SimResult                    rc;

    if (builder == NULL || node == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->differential == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    domain = sim_ir_type_scalar_domain(node->value_type);
    operand_domain_value                  = sim_ir_domain_value_zero(domain);
    operand_domain_value.domain           = domain;
    operand_domain_value.value.as_complex = operand_value;
    value                                 = sim_ir_domain_value_zero(domain);
    rc                                    = state->evaluator->differential(state->evaluator->userdata,
                                         builder,
                                         node,
                                         operand,
                                         domain,
                                         operand_domain_value,
                                         &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = domain;
    }
    if (!sim_scalar_domain_equal(value.domain, domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    *out_value = value.value.as_complex;
    return SIM_RESULT_OK;
}

static SimResult
sim_ir_eval_domain_noise_complex(void* userdata, const SimIRNode* node, SimComplexDouble* out_value) {
    SimIREvalDomainAdapterState* state;
    SimScalarDomain              domain;
    SimIRDomainValue             value;
    SimResult                    rc;

    if (node == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->noise == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    domain = sim_ir_type_scalar_domain(node->value_type);
    value  = sim_ir_domain_value_zero(domain);
    rc     = state->evaluator->noise(state->evaluator->userdata, node, domain, &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = domain;
    }
    if (!sim_scalar_domain_equal(value.domain, domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    *out_value = value.value.as_complex;
    return SIM_RESULT_OK;
}

static SimResult sim_ir_eval_domain_param_complex(void*             userdata,
                                                  SimIRParamKind    param,
                                                  double*           out_value) {
    SimIREvalDomainAdapterState* state;
    SimIRDomainValue             value;
    SimResult                    rc;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimIREvalDomainAdapterState*) userdata;
    if (state == NULL || state->evaluator == NULL || state->evaluator->param_value == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    value = sim_ir_domain_value_zero(sim_scalar_domain_f64());
    rc = state->evaluator->param_value(
        state->evaluator->userdata, param, sim_scalar_domain_f64(), &value);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = sim_scalar_domain_f64();
    }
    if (!sim_scalar_domain_equal(value.domain, sim_scalar_domain_f64())) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    *out_value = value.value.as_f64;
    return SIM_RESULT_OK;
}

SimResult sim_ir_evaluate_domain(const SimIRBuilder*         builder,
                                 SimIRNodeId                 root,
                                 const SimIREvaluatorDomain* evaluator,
                                 SimIRDomainValue*           out_value) {
    SimIRType      root_type;
    SimScalarDomain root_domain;
    SimResult       rc;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_value = sim_ir_domain_value_zero(sim_scalar_domain_unknown());

    if (builder == NULL || root == SIM_IR_INVALID_NODE || root >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    root_type = sim_ir_type_normalize(builder->nodes[root].value_type);
    if (!sim_ir_type_is_scalar(root_type) &&
        !sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(root_type))) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    root_domain       = sim_ir_type_scalar_domain(root_type);
    out_value->domain = root_domain;

    if (sim_ir_domain_is_exact_integer(root_domain)) {
        SimIRIntegerEvalState integer_state = { 0 };
        uint64_t              raw           = 0U;

        integer_state.builder   = builder;
        integer_state.evaluator = evaluator;
        if (builder->count > 0U) {
            integer_state.cache = (uint64_t*) calloc(builder->count, sizeof(uint64_t));
            integer_state.flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
            if (integer_state.cache == NULL || integer_state.flags == NULL) {
                free(integer_state.cache);
                free(integer_state.flags);
                return SIM_RESULT_OUT_OF_MEMORY;
            }
        }

        rc = sim_ir_evaluate_impl_integer(&integer_state, root, &raw);
        if (rc == SIM_RESULT_OK) {
            *out_value = sim_ir_domain_value_from_integer_raw(root_domain, raw);
        }

        free(integer_state.cache);
        free(integer_state.flags);
        return rc;
    }

    if (sim_scalar_domain_is_complex(root_domain)) {
        SimComplexDouble complex_value = { 0.0, 0.0 };
        if (evaluator == NULL) {
            rc = sim_ir_evaluate_complex(builder, root, NULL, &complex_value);
        } else {
            SimIREvalDomainAdapterState adapter_state = { .evaluator = evaluator };
            SimIREvaluatorComplex       legacy_eval   = {
                      .field_value_c = sim_ir_eval_domain_field_complex,
                      .differential_c = sim_ir_eval_domain_diff_complex,
                      .noise_c = sim_ir_eval_domain_noise_complex,
                      .param_value = sim_ir_eval_domain_param_complex,
                      .userdata = &adapter_state
            };
            rc = sim_ir_evaluate_complex(builder, root, &legacy_eval, &complex_value);
        }
        if (rc == SIM_RESULT_OK) {
            out_value->value.as_complex = complex_value;
        }
        return rc;
    }

    {
        double real_value = 0.0;
        if (evaluator == NULL) {
            rc = sim_ir_evaluate(builder, root, NULL, &real_value);
        } else {
            SimIREvalDomainAdapterState adapter_state = { .evaluator = evaluator };
            SimIREvaluator              legacy_eval   = { .field_value = sim_ir_eval_domain_field_real,
                                                          .differential = sim_ir_eval_domain_diff_real,
                                                          .noise = sim_ir_eval_domain_noise_real,
                                                          .param_value = sim_ir_eval_domain_param_real,
                                                          .userdata = &adapter_state };
            rc = sim_ir_evaluate(builder, root, &legacy_eval, &real_value);
        }
        if (rc == SIM_RESULT_OK) {
            out_value->value.as_f64 = real_value;
        }
        return rc;
    }
}
