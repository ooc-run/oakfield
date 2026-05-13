/**
 * @file kernel_ir_eval_complex.c
 * @brief Complex-valued KernelIR evaluator for CPU fallback execution.
 *
 * Complex evaluation uses cartesian SimComplexDouble values and principal
 * libm complex-function semantics. The path mirrors the real evaluator while
 * preserving real/imaginary output layout and returning status codes for
 * invalid inputs, unsupported callbacks, and temporary cache allocation failure.
 */
#include "internal/kernel_ir_internal.h"

#if defined(__APPLE__)
static inline void sim_ir_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void sim_ir_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void sim_ir_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

static SimResult
sim_ir_evaluate_impl_complex(SimIREvalState* state, SimIRNodeId node_id, SimComplexDouble* out) {
    const SimIRNode* node;
    SimComplexDouble value  = { 0.0, 0.0 };
    SimResult        result = SIM_RESULT_OK;

    if (state == NULL || out == NULL || state->builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= state->builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->flags != NULL && state->cache_complex != NULL) {
        if (state->flags[node_id] == 2U) {
            *out = state->cache_complex[node_id];
            return SIM_RESULT_OK;
        }
        if (state->flags[node_id] == 1U) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }
        state->flags[node_id] = 1U;
    }

    node = &state->builder->nodes[node_id];

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
            if (node->value_type.components >= 2U) {
                double re = 0.0;
                double im = 0.0;
                if (!sim_ir_constant_component(state->builder, node, 0U, &re)) {
                    re = node->data.constant.scalar;
                }
                if (!sim_ir_constant_component(state->builder, node, 1U, &im)) {
                    im = 0.0;
                }
                value.re = re;
                value.im = im;
            } else {
                value.re = node->data.constant.scalar;
                value.im = 0.0;
            }
            break;

        case SIM_IR_NODE_FIELD_REF:
            if (!state->evaluator_complex || !state->evaluator_complex->field_value_c)
                return SIM_RESULT_NOT_FOUND;

            result = state->evaluator_complex->field_value_c(
                state->evaluator_complex->userdata, node->data.field, node->value_type, &value);

            break;

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV:
        case SIM_IR_NODE_POW: {
            SimComplexDouble lhs, rhs;
            result = sim_ir_evaluate_impl_complex(state, node->data.binary.lhs, &lhs);
            if (result != SIM_RESULT_OK)
                break;
            result = sim_ir_evaluate_impl_complex(state, node->data.binary.rhs, &rhs);
            if (result != SIM_RESULT_OK)
                break;

            switch (node->type) {
                case SIM_IR_NODE_ADD:
                    value.re = lhs.re + rhs.re;
                    value.im = lhs.im + rhs.im;
                    break;
                case SIM_IR_NODE_SUB:
                    value.re = lhs.re - rhs.re;
                    value.im = lhs.im - rhs.im;
                    break;
                case SIM_IR_NODE_MUL:
                    value.re = lhs.re * rhs.re - lhs.im * rhs.im;
                    value.im = lhs.re * rhs.im + lhs.im * rhs.re;
                    break;
                case SIM_IR_NODE_DIV: {
                    double denom = rhs.re * rhs.re + rhs.im * rhs.im;
                    value.re     = (lhs.re * rhs.re + lhs.im * rhs.im) / denom;
                    value.im     = (lhs.im * rhs.re - lhs.re * rhs.im) / denom;
                    break;
                }
                case SIM_IR_NODE_POW: {
                    double complex l = lhs.re + I * lhs.im;
                    double complex r = rhs.re + I * rhs.im;
                    double complex p = cpow(l, r);
                    value.re         = creal(p);
                    value.im         = cimag(p);
                    break;
                }
                default:
                    break;
            }
            break;
        }

        case SIM_IR_NODE_DIFF: {
            SimComplexDouble operand_value = { 0.0, 0.0 };

            result = sim_ir_evaluate_impl_complex(state, node->data.diff.operand, &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            if (!state->evaluator_complex || !state->evaluator_complex->differential_c) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }

            result = state->evaluator_complex->differential_c(state->evaluator_complex->userdata,
                                                              state->builder,
                                                              node,
                                                              node->data.diff.operand,
                                                              operand_value,
                                                              &value);
            break;
        }

        case SIM_IR_NODE_NOISE:
            if (!state->evaluator_complex || !state->evaluator_complex->noise_c) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            result =
                state->evaluator_complex->noise_c(state->evaluator_complex->userdata, node, &value);
            break;

        case SIM_IR_NODE_WARP: {
            SimComplexDouble op;
            result = sim_ir_evaluate_impl_complex(state, node->data.warp.operand, &op);
            if (result != SIM_RESULT_OK)
                break;

            SimWarpGuard      guard       = sim_ir_guard_to_runtime(&node->data.warp.guard);
            SimWarpSampleSpec sample_spec = { .sample = 0.0,
                                              .bias   = node->data.warp.bias,
                                              .delta  = node->data.warp.delta,
                                              .lambda = node->data.warp.lambda,
                                              .guard  = guard };

            double    response = 0.0;
            SimResult rc_re;
            SimResult rc_im;

            sample_spec.sample = op.re;
            rc_re              = sim_ir_warp_sample_response(&sample_spec,
                                                node->data.warp.profile,
                                                node->data.warp.tolerance,
                                                NULL,
                                                NULL,
                                                &response);
            if (rc_re == SIM_RESULT_OK && isfinite(response)) {
                value.re = response;
            } else {
                value.re = 0.0;
                if (result == SIM_RESULT_OK) {
                    result = rc_re;
                }
            }

            sample_spec.sample = op.im;
            rc_im              = sim_ir_warp_sample_response(&sample_spec,
                                                node->data.warp.profile,
                                                node->data.warp.tolerance,
                                                NULL,
                                                NULL,
                                                &response);
            if (rc_im == SIM_RESULT_OK && isfinite(response)) {
                value.im = response;
            } else {
                value.im = 0.0;
                if (result == SIM_RESULT_OK) {
                    result = rc_im;
                }
            }
            break;
        }

        case SIM_IR_NODE_PARAM:
            if (!state->evaluator_complex || !state->evaluator_complex->param_value) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            result = state->evaluator_complex->param_value(
                state->evaluator_complex->userdata, node->data.param.param, &value.re);
            value.im = 0.0;
            break;

        case SIM_IR_NODE_INDEX:
        case SIM_IR_NODE_CALL:
        case SIM_IR_NODE_FLOOR:
        case SIM_IR_NODE_MOD:
        case SIM_IR_NODE_COORD:
        case SIM_IR_NODE_STATEFUL:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;

        case SIM_IR_NODE_COMPLEX_PACK: {
            SimComplexDouble real_value = { 0.0, 0.0 };
            SimComplexDouble imag_value = { 0.0, 0.0 };

            result = sim_ir_evaluate_impl_complex(state, node->data.complex_pack.real, &real_value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_evaluate_impl_complex(state, node->data.complex_pack.imag, &imag_value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            value.re = real_value.re;
            value.im = imag_value.re;
            break;
        }

        case SIM_IR_NODE_COMPLEX_ROTATE: {
            SimComplexDouble op          = { 0.0, 0.0 };
            SimComplexDouble angle_value = { 0.0, 0.0 };
            double           theta       = 0.0;
            double           s           = 0.0;
            double           c           = 1.0;

            result = sim_ir_evaluate_impl_complex(state, node->data.complex_rotate.operand, &op);
            if (result != SIM_RESULT_OK) {
                break;
            }

            if (node->data.complex_rotate.angle == SIM_IR_INVALID_NODE) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                break;
            }

            result =
                sim_ir_evaluate_impl_complex(state, node->data.complex_rotate.angle, &angle_value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            theta = angle_value.re;

            sim_ir_sincos(theta, &s, &c);
            value.re = op.re * c - op.im * s;
            value.im = op.re * s + op.im * c;
            break;
        }

        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

    if (state->flags != NULL && state->cache_complex != NULL) {
        if (result == SIM_RESULT_OK) {
            state->cache_complex[node_id] = value;
            state->flags[node_id]         = 2U;
        } else {
            state->flags[node_id] = 0U;
        }
    }

    if (result == SIM_RESULT_OK) {
        *out = value;
    }
    return result;
}

SimResult sim_ir_evaluate_complex(const SimIRBuilder*          builder,
                                  SimIRNodeId                  root,
                                  const SimIREvaluatorComplex* evaluator,
                                  SimComplexDouble*            out_value) {
    SimIREvalState state = { 0 };
    SimResult      result;

    if (out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (builder == NULL || root == SIM_IR_INVALID_NODE || root >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state.builder           = builder;
    state.evaluator         = NULL;
    state.evaluator_complex = evaluator;

    if (builder->count > 0U) {
        state.cache_complex = (SimComplexDouble*) calloc(builder->count, sizeof(SimComplexDouble));
        state.flags         = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
        if (state.cache_complex == NULL || state.flags == NULL) {
            free(state.cache_complex);
            free(state.flags);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    result = sim_ir_evaluate_impl_complex(&state, root, out_value);

    free(state.cache_complex);
    free(state.flags);
    return result;
}
