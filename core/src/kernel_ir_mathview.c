/**
 * @file kernel_ir_mathview.c
 * @brief Canonical string, JSON, LaTeX, and hash rendering for KernelIR graphs.
 *
 * MathView rendering exposes KernelIR expressions to tooling with explicit
 * schema, scalar-domain, complex-cartesian, and branch-convention metadata. The
 * implementation uses bounded append helpers so callers can query required
 * lengths, render into fixed buffers, and hash the canonical representation.
 */
#include "oakfield/kernel_ir_mathview.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static SimResult
sim_ir_mathview_append(char* buffer, size_t capacity, size_t* offset, const char* fmt, ...) {
    if (offset == NULL || fmt == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);

    int needed = vsnprintf(NULL, 0, fmt, args_copy);
    va_end(args_copy);
    if (needed < 0) {
        va_end(args);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (buffer != NULL && capacity > 0U) {
        size_t remaining = (capacity > *offset) ? (capacity - *offset) : 0U;
        if (remaining > 0U) {
            (void) vsnprintf(buffer + *offset, remaining, fmt, args);
        }
    }
    va_end(args);

    *offset += (size_t) needed;
    if (buffer != NULL && capacity > 0U && *offset >= capacity) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    return SIM_RESULT_OK;
}

static SimResult sim_ir_mathview_append_raw(char*       buffer,
                                            size_t      capacity,
                                            size_t*     offset,
                                            const char* text,
                                            size_t      length) {
    if (offset == NULL || (text == NULL && length > 0U)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (buffer != NULL && capacity > 0U && length > 0U) {
        size_t remaining = (capacity > *offset) ? (capacity - *offset) : 0U;
        if (remaining > 0U) {
            size_t to_copy = (length < remaining) ? length : remaining;
            (void) memcpy(buffer + *offset, text, to_copy);
        }
    }

    *offset += length;
    if (buffer != NULL && capacity > 0U && *offset >= capacity) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    return SIM_RESULT_OK;
}

static SimResult sim_ir_mathview_append_json_escaped(char*       buffer,
                                                     size_t      capacity,
                                                     size_t*     offset,
                                                     const char* text) {
    if (offset == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (text == NULL) {
        text = "";
    }

    while (*text != '\0') {
        unsigned char ch     = (unsigned char) (*text++);
        SimResult     result = SIM_RESULT_OK;
        switch (ch) {
            case '\"':
                result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\\\"", 2U);
                break;
            case '\\':
                result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\\\\", 2U);
                break;
            case '\n':
                result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\\n", 2U);
                break;
            case '\r':
                result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\\r", 2U);
                break;
            case '\t':
                result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\\t", 2U);
                break;
            default:
                if (ch < 0x20U) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, "\\u%04x", ch);
                } else {
                    result =
                        sim_ir_mathview_append_raw(buffer, capacity, offset, (const char*) &ch, 1U);
                }
                break;
        }
        if (result != SIM_RESULT_OK) {
            return result;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult sim_ir_mathview_append_json_string(char*       buffer,
                                                    size_t      capacity,
                                                    size_t*     offset,
                                                    const char* text) {
    SimResult result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\"", 1U);
    if (result != SIM_RESULT_OK) {
        return result;
    }
    result = sim_ir_mathview_append_json_escaped(buffer, capacity, offset, text);
    if (result != SIM_RESULT_OK) {
        return result;
    }
    return sim_ir_mathview_append_raw(buffer, capacity, offset, "\"", 1U);
}

static SimResult sim_ir_mathview_append_scalar_constant(char*             buffer,
                                                        size_t            capacity,
                                                        size_t*           offset,
                                                        const SimIRNode*  node) {
    SimScalarDomain domain;

    if (offset == NULL || node == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    domain = sim_ir_type_scalar_domain(node->value_type);
    if (sim_scalar_domain_is_integer(domain) && node->data.constant.exact_integer) {
        if (domain.is_signed) {
            return sim_ir_mathview_append(
                buffer, capacity, offset, "%" PRId64, node->data.constant.signed_scalar);
        }
        return sim_ir_mathview_append(
            buffer, capacity, offset, "%" PRIu64, node->data.constant.unsigned_scalar);
    }

    return sim_ir_mathview_append(buffer, capacity, offset, "%.17g", node->data.constant.scalar);
}

static void sim_ir_mathview_type_info(SimIRType    type,
                                      const char** out_kind,
                                      size_t*      out_lanes,
                                      bool*        out_complex,
                                      const char** out_scalar_domain) {
    size_t lanes  = (type.components == 0U) ? 1U : type.components;
    bool   scalar = (type.kind == SIM_IR_VALUE_SCALAR) || (lanes <= 1U);
    SimScalarDomain scalar_domain = sim_ir_type_scalar_domain(type);

    if (out_kind != NULL) {
        *out_kind = scalar ? "scalar" : "vector";
    }
    if (out_lanes != NULL) {
        *out_lanes = lanes;
    }
    if (out_complex != NULL) {
        *out_complex = sim_scalar_domain_is_complex(scalar_domain);
    }
    if (out_scalar_domain != NULL) {
        *out_scalar_domain = sim_scalar_domain_name(scalar_domain);
    }
}

static SimResult
sim_ir_mathview_append_type_block(char* buffer, size_t capacity, size_t* offset, SimIRType type) {
    const char* kind       = NULL;
    const char* scalar_domain = NULL;
    size_t      lanes      = 1U;
    bool        is_complex = false;

    sim_ir_mathview_type_info(type, &kind, &lanes, &is_complex, &scalar_domain);

    return sim_ir_mathview_append(buffer,
                                  capacity,
                                  offset,
                                  "{kind=%s,lanes=%zu,complex=%s,scalar_domain=%s}",
                                  kind ? kind : "scalar",
                                  lanes,
                                  is_complex ? "true" : "false",
                                  scalar_domain ? scalar_domain : "unknown");
}

static SimResult
sim_ir_mathview_append_type_json(char* buffer, size_t capacity, size_t* offset, SimIRType type) {
    const char* kind       = NULL;
    const char* scalar_domain = NULL;
    size_t      lanes      = 1U;
    bool        is_complex = false;

    sim_ir_mathview_type_info(type, &kind, &lanes, &is_complex, &scalar_domain);

    return sim_ir_mathview_append(buffer,
                                  capacity,
                                  offset,
                                  "{\"kind\":\"%s\",\"lanes\":%zu,\"complex\":%s,\"scalar_domain\":\"%s\"}",
                                  kind ? kind : "scalar",
                                  lanes,
                                  is_complex ? "true" : "false",
                                  scalar_domain ? scalar_domain : "unknown");
}

static const char* sim_ir_mathview_param_name(SimIRParamKind param) {
    switch (param) {
        case SIM_IR_PARAM_DT:
            return "dt";
        case SIM_IR_PARAM_STEP_INDEX:
            return "step_index";
        case SIM_IR_PARAM_SQRT_DT:
            return "sqrt_dt";
        case SIM_IR_PARAM_TIME:
            return "time";
        default:
            return NULL;
    }
}

static const char* sim_ir_mathview_noise_law_name(SimIRNoiseLaw law) {
    switch (law) {
        case SIM_IR_NOISE_LAW_STRATONOVICH:
            return "stratonovich";
        case SIM_IR_NOISE_LAW_ITO:
        default:
            return "ito";
    }
}

static const char* sim_ir_mathview_noise_distribution_name(SimIRNoiseDistribution dist) {
    switch (dist) {
        case SIM_IR_NOISE_DISTRIBUTION_GAUSSIAN:
            return "gaussian";
        case SIM_IR_NOISE_DISTRIBUTION_UNIFORM:
        default:
            return "uniform";
    }
}

static const char* sim_ir_mathview_warp_profile_name(SimIRWarpProfile profile) {
    switch (profile) {
        case SIM_IR_WARP_PROFILE_TRIGAMMA:
            return "trigamma";
        case SIM_IR_WARP_PROFILE_DIGAMMA_7_TAIL:
            return "digamma_7_tail";
        case SIM_IR_WARP_PROFILE_DIGAMMA_5_TAIL:
            return "digamma_5_tail";
        case SIM_IR_WARP_PROFILE_DIGAMMA_ADAPTIVE:
            return "digamma_adaptive";
        case SIM_IR_WARP_PROFILE_DIGAMMA_MORTICI:
            return "digamma_mortici";
        case SIM_IR_WARP_PROFILE_DIGAMMA:
        default:
            return "digamma";
    }
}

static const char* sim_ir_mathview_call_name(SimIRCallKind kind) {
    switch (kind) {
        case SIM_IR_CALL_SIN:
            return "sin";
        case SIM_IR_CALL_COS:
            return "cos";
        case SIM_IR_CALL_EXP:
            return "exp";
        case SIM_IR_CALL_ABS:
            return "abs";
        case SIM_IR_CALL_LOG:
            return "log";
        case SIM_IR_CALL_TANH:
            return "tanh";
        case SIM_IR_CALL_SINH:
            return "sinh";
        case SIM_IR_CALL_SIGN:
            return "sign";
        default:
            return "unknown";
    }
}

static const char* sim_ir_mathview_boundary_name(SimIRBoundaryPolicy boundary) {
    switch (boundary) {
        case SIM_IR_BOUNDARY_DIRICHLET:
            return "dirichlet";
        case SIM_IR_BOUNDARY_PERIODIC:
            return "periodic";
        case SIM_IR_BOUNDARY_REFLECTIVE:
            return "reflective";
        case SIM_IR_BOUNDARY_NEUMANN:
        default:
            return "neumann";
    }
}

static const char* sim_ir_mathview_diff_method_name(SimIRDiffMethod method) {
    switch (method) {
        case SIM_IR_DIFF_METHOD_FORWARD:
            return "forward";
        case SIM_IR_DIFF_METHOD_BACKWARD:
            return "backward";
        case SIM_IR_DIFF_METHOD_CENTRAL:
            return "central";
        case SIM_IR_DIFF_METHOD_AUTO:
        default:
            return "auto";
    }
}

static bool sim_ir_mathview_constant_value(const SimIRBuilder* builder,
                                           const SimIRNode*    node,
                                           size_t              component,
                                           double*             out_value) {
    if (builder == NULL || node == NULL || out_value == NULL) {
        return false;
    }

    size_t components = node->value_type.components == 0U ? 1U : node->value_type.components;
    if (components <= 1U) {
        *out_value = node->data.constant.scalar;
        return true;
    }

    if (node->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX) {
        size_t idx = node->data.constant.constant_index;
        if (idx < builder->constants_count && component < builder->constants_components[idx]) {
            size_t offset = builder->constants_offsets[idx];
            if (builder->constants_data != NULL) {
                *out_value = builder->constants_data[offset + component];
                return true;
            }
        }
        return false;
    }

    if (component < SIM_IR_SMALL_CONSTANT_CAPACITY) {
        *out_value = node->data.constant.small[component];
        return true;
    }

    return false;
}

static SimResult sim_ir_mathview_render_node(const SimIRBuilder* builder,
                                             SimIRNodeId         node_id,
                                             char*               buffer,
                                             size_t              capacity,
                                             size_t*             offset,
                                             unsigned char*      stack_flags) {
    const SimIRNode* node   = NULL;
    SimResult        result = SIM_RESULT_OK;

    if (builder == NULL || offset == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (stack_flags != NULL) {
        if (stack_flags[node_id] != 0U) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }
        stack_flags[node_id] = 1U;
    }

    node = sim_ir_builder_get(builder, node_id);
    if (node == NULL) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto done;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT: {
            size_t components =
                node->value_type.components == 0U ? 1U : node->value_type.components;
            result = sim_ir_mathview_append(buffer, capacity, offset, "Const");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            if (components <= 1U) {
                result = sim_ir_mathview_append_scalar_constant(
                    buffer, capacity, offset, node);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, ")");
                break;
            }

            result = sim_ir_mathview_append(buffer, capacity, offset, "[");
            if (result != SIM_RESULT_OK) {
                break;
            }
            for (size_t i = 0U; i < components; ++i) {
                double value = 0.0;
                if (!sim_ir_mathview_constant_value(builder, node, i, &value)) {
                    value = 0.0;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "%.17g", value);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                if (i + 1U < components) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",");
                    if (result != SIM_RESULT_OK) {
                        break;
                    }
                }
            }
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "])");
            break;
        }
        case SIM_IR_NODE_FIELD_REF: {
            char symbol[64];
            (void) snprintf(symbol, sizeof(symbol), "field:%zu", node->data.field);
            result = sim_ir_mathview_append(buffer, capacity, offset, "Symbol");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, symbol);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        case SIM_IR_NODE_PARAM: {
            const char* param_name = sim_ir_mathview_param_name(node->data.param.param);
            char        symbol[64];
            if (param_name != NULL) {
                (void) snprintf(symbol, sizeof(symbol), "param:%s", param_name);
            } else {
                (void) snprintf(
                    symbol, sizeof(symbol), "param:%u", (unsigned) node->data.param.param);
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "Symbol");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, symbol);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        case SIM_IR_NODE_INDEX:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Symbol");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "index");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_CALL:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(
                buffer, capacity, offset, sim_ir_mathview_call_name(node->data.call.kind));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.call.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_FLOOR:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "floor");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.unary.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_MOD:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "mod");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_POW:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "pow");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_COORD: {
            char symbol[64];
            (void) snprintf(symbol, sizeof(symbol), "field:%zu", node->data.coord.field);
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "coord");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "Symbol");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result =
                sim_ir_mathview_append_type_block(buffer, capacity, offset, sim_ir_type_scalar());
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, symbol);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result =
                sim_ir_mathview_append(buffer, capacity, offset, ",%zu)", node->data.coord.axis);
            break;
        }

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV: {
            const char* op = (node->type == SIM_IR_NODE_ADD)   ? "+"
                             : (node->type == SIM_IR_NODE_SUB) ? "-"
                             : (node->type == SIM_IR_NODE_MUL) ? "*"
                                                               : "/";
            result         = sim_ir_mathview_append(buffer, capacity, offset, "Binary");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, op);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        case SIM_IR_NODE_DIFF:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "diff");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.diff.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            ",%zu,%.17g,%.17g,%zu,%zu,",
                                            node->data.diff.axis,
                                            node->data.diff.dx,
                                            node->data.diff.scale,
                                            node->data.diff.order,
                                            node->data.diff.stencil_order);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(
                buffer, capacity, offset, sim_ir_mathview_diff_method_name(node->data.diff.method));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(
                buffer, capacity, offset, ",%.17g,", node->data.diff.consistency_constant);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(
                buffer, capacity, offset, sim_ir_mathview_boundary_name(node->data.diff.boundary));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_NOISE:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "noise");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            ",%u,%.17g,%.17g,",
                                            node->data.noise.seed,
                                            node->data.noise.amplitude,
                                            node->data.noise.variance);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(
                buffer, capacity, offset, sim_ir_mathview_noise_law_name(node->data.noise.law));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(
                buffer,
                capacity,
                offset,
                sim_ir_mathview_noise_distribution_name(node->data.noise.distribution));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_WARP:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "warp");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(
                buffer,
                capacity,
                offset,
                sim_ir_mathview_warp_profile_name(node->data.warp.profile));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.warp.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            ",%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g)",
                                            node->data.warp.bias,
                                            node->data.warp.delta,
                                            node->data.warp.lambda,
                                            node->data.warp.tolerance,
                                            node->data.warp.guard.mode,
                                            node->data.warp.guard.clamp_min,
                                            node->data.warp.guard.clamp_max,
                                            node->data.warp.guard.tolerance);
            break;

        case SIM_IR_NODE_COMPLEX_ROTATE:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "complex_rotate");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.complex_rotate.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.complex_rotate.angle, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_COMPLEX_PACK:
            result = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "complex_pack");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.complex_pack.real, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node(
                builder, node->data.complex_pack.imag, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_STATEFUL: {
            const char* label = node->data.stateful.label ? node->data.stateful.label : "stateful";
            result            = sim_ir_mathview_append(buffer, capacity, offset, "Call");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_block(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, "stateful");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, label);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        default:
            result = SIM_RESULT_NOT_SUPPORTED;
            break;
    }

done:
    if (stack_flags != NULL) {
        stack_flags[node_id] = 0U;
    }
    return result;
}

SimResult sim_ir_mathview_render(const SimIRBuilder* builder,
                                 SimIRNodeId         root,
                                 char*               buffer,
                                 size_t              capacity,
                                 size_t*             out_length) {
    size_t         offset      = 0U;
    unsigned char* stack_flags = NULL;
    if (builder != NULL && builder->count > 0U) {
        stack_flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
        if (stack_flags == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    SimResult result = sim_ir_mathview_append(buffer,
                                              capacity,
                                              &offset,
                                              "math:v%s;complex=%s;branch=%s;",
                                              SIM_IR_MATHVIEW_SCHEMA_VERSION,
                                              SIM_IR_MATHVIEW_COMPLEX_SEMANTICS,
                                              SIM_IR_MATHVIEW_COMPLEX_BRANCH);
    if (result == SIM_RESULT_OK) {
        result = sim_ir_mathview_render_node(builder, root, buffer, capacity, &offset, stack_flags);
    }
    free(stack_flags);

    if (out_length != NULL) {
        *out_length = offset;
    }

    if (buffer != NULL && capacity > 0U) {
        if (offset >= capacity) {
            buffer[capacity - 1U] = '\0';
            return (result == SIM_RESULT_OK) ? SIM_RESULT_OUT_OF_MEMORY : result;
        }
        buffer[offset] = '\0';
    }

    return result;
}

static SimResult sim_ir_mathview_render_node_json(const SimIRBuilder* builder,
                                                  SimIRNodeId         node_id,
                                                  char*               buffer,
                                                  size_t              capacity,
                                                  size_t*             offset,
                                                  unsigned char*      stack_flags) {
    const SimIRNode* node   = NULL;
    SimResult        result = SIM_RESULT_OK;

    if (builder == NULL || offset == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (stack_flags != NULL) {
        if (stack_flags[node_id] != 0U) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }
        stack_flags[node_id] = 1U;
    }

    node = sim_ir_builder_get(builder, node_id);
    if (node == NULL) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto done;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT: {
            size_t components =
                node->value_type.components == 0U ? 1U : node->value_type.components;
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "{\"node\":\"Const\",\"type\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_json(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"value\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            if (components <= 1U) {
                result = sim_ir_mathview_append_scalar_constant(
                    buffer, capacity, offset, node);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}");
                break;
            }

            result = sim_ir_mathview_append(buffer, capacity, offset, "[");
            if (result != SIM_RESULT_OK) {
                break;
            }
            for (size_t i = 0U; i < components; ++i) {
                double value = 0.0;
                if (!sim_ir_mathview_constant_value(builder, node, i, &value)) {
                    value = 0.0;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "%.17g", value);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                if (i + 1U < components) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",");
                    if (result != SIM_RESULT_OK) {
                        break;
                    }
                }
            }
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "]}");
            break;
        }
        case SIM_IR_NODE_FIELD_REF: {
            char symbol[64];
            (void) snprintf(symbol, sizeof(symbol), "field:%zu", node->data.field);
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "{\"node\":\"Symbol\",\"type\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_json(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"name\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, symbol);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "}");
            break;
        }
        case SIM_IR_NODE_PARAM: {
            const char* param_name = sim_ir_mathview_param_name(node->data.param.param);
            char        symbol[64];
            if (param_name != NULL) {
                (void) snprintf(symbol, sizeof(symbol), "param:%s", param_name);
            } else {
                (void) snprintf(
                    symbol, sizeof(symbol), "param:%u", (unsigned) node->data.param.param);
            }
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "{\"node\":\"Symbol\",\"type\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_json(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"name\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, symbol);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "}");
            break;
        }
        case SIM_IR_NODE_INDEX:
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "{\"node\":\"Symbol\",\"type\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_json(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"name\":\"index\"}");
            break;

        case SIM_IR_NODE_CALL:
        case SIM_IR_NODE_FLOOR:
        case SIM_IR_NODE_MOD:
        case SIM_IR_NODE_POW:
        case SIM_IR_NODE_DIFF:
        case SIM_IR_NODE_NOISE:
        case SIM_IR_NODE_WARP:
        case SIM_IR_NODE_COMPLEX_ROTATE:
        case SIM_IR_NODE_COMPLEX_PACK:
        case SIM_IR_NODE_COORD:
        case SIM_IR_NODE_STATEFUL: {
            const char* call_name = NULL;
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "{\"node\":\"Call\",\"type\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_json(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"name\":");
            if (result != SIM_RESULT_OK) {
                break;
            }

            switch (node->type) {
                case SIM_IR_NODE_CALL:
                    call_name = sim_ir_mathview_call_name(node->data.call.kind);
                    break;
                case SIM_IR_NODE_FLOOR:
                    call_name = "floor";
                    break;
                case SIM_IR_NODE_MOD:
                    call_name = "mod";
                    break;
                case SIM_IR_NODE_POW:
                    call_name = "pow";
                    break;
                case SIM_IR_NODE_DIFF:
                    call_name = "diff";
                    break;
                case SIM_IR_NODE_NOISE:
                    call_name = "noise";
                    break;
                case SIM_IR_NODE_WARP:
                    call_name = "warp";
                    break;
                case SIM_IR_NODE_COMPLEX_ROTATE:
                    call_name = "complex_rotate";
                    break;
                case SIM_IR_NODE_COMPLEX_PACK:
                    call_name = "complex_pack";
                    break;
                case SIM_IR_NODE_COORD:
                    call_name = "coord";
                    break;
                case SIM_IR_NODE_STATEFUL:
                    call_name = "stateful";
                    break;
                default:
                    call_name = "unknown";
                    break;
            }

            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, call_name);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"args\":[");
            if (result != SIM_RESULT_OK) {
                break;
            }

            if (node->type == SIM_IR_NODE_CALL) {
                result = sim_ir_mathview_render_node_json(
                    builder, node->data.call.operand, buffer, capacity, offset, stack_flags);
            } else if (node->type == SIM_IR_NODE_FLOOR) {
                result = sim_ir_mathview_render_node_json(
                    builder, node->data.unary.operand, buffer, capacity, offset, stack_flags);
            } else if (node->type == SIM_IR_NODE_MOD || node->type == SIM_IR_NODE_POW) {
                result = sim_ir_mathview_render_node_json(
                    builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",");
                }
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_render_node_json(
                        builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
                }
            } else if (node->type == SIM_IR_NODE_DIFF) {
                result = sim_ir_mathview_render_node_json(
                    builder, node->data.diff.operand, buffer, capacity, offset, stack_flags);
            } else if (node->type == SIM_IR_NODE_WARP) {
                result = sim_ir_mathview_render_node_json(
                    builder, node->data.warp.operand, buffer, capacity, offset, stack_flags);
            } else if (node->type == SIM_IR_NODE_COMPLEX_ROTATE) {
                result = sim_ir_mathview_render_node_json(builder,
                                                          node->data.complex_rotate.operand,
                                                          buffer,
                                                          capacity,
                                                          offset,
                                                          stack_flags);
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",");
                }
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_render_node_json(builder,
                                                              node->data.complex_rotate.angle,
                                                              buffer,
                                                              capacity,
                                                              offset,
                                                              stack_flags);
                }
            } else if (node->type == SIM_IR_NODE_COMPLEX_PACK) {
                result = sim_ir_mathview_render_node_json(
                    builder, node->data.complex_pack.real, buffer, capacity, offset, stack_flags);
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",");
                }
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_render_node_json(builder,
                                                              node->data.complex_pack.imag,
                                                              buffer,
                                                              capacity,
                                                              offset,
                                                              stack_flags);
                }
            } else if (node->type == SIM_IR_NODE_COORD) {
                char symbol[64];
                (void) snprintf(symbol, sizeof(symbol), "field:%zu", node->data.coord.field);
                result = sim_ir_mathview_append(
                    buffer, capacity, offset, "{\"node\":\"Symbol\",\"type\":");
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append_type_json(
                        buffer, capacity, offset, sim_ir_type_scalar());
                }
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",\"name\":");
                }
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append_json_string(buffer, capacity, offset, symbol);
                }
                if (result == SIM_RESULT_OK) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, "}");
                }
            }

            if (result != SIM_RESULT_OK) {
                break;
            }

            result = sim_ir_mathview_append(buffer, capacity, offset, "]");
            if (result != SIM_RESULT_OK) {
                break;
            }

            if (node->type == SIM_IR_NODE_DIFF) {
                result = sim_ir_mathview_append(buffer,
                                                capacity,
                                                offset,
                                                ",\"params\":{\"axis\":%zu,\"dx\":%.17g,\"scale\":%"
                                                ".17g,\"order\":%zu,\"stencil\":%zu,\"method\":",
                                                node->data.diff.axis,
                                                node->data.diff.dx,
                                                node->data.diff.scale,
                                                node->data.diff.order,
                                                node->data.diff.stencil_order);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_json_string(
                    buffer,
                    capacity,
                    offset,
                    sim_ir_mathview_diff_method_name(node->data.diff.method));
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer,
                                                capacity,
                                                offset,
                                                ",\"C_p\":%.17g,\"boundary\":",
                                                node->data.diff.consistency_constant);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_json_string(
                    buffer,
                    capacity,
                    offset,
                    sim_ir_mathview_boundary_name(node->data.diff.boundary));
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}");
                if (result != SIM_RESULT_OK) {
                    break;
                }
            } else if (node->type == SIM_IR_NODE_NOISE) {
                result = sim_ir_mathview_append(
                    buffer,
                    capacity,
                    offset,
                    ",\"params\":{\"seed\":%u,\"amp\":%.17g,\"var\":%.17g,\"law\":",
                    node->data.noise.seed,
                    node->data.noise.amplitude,
                    node->data.noise.variance);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_json_string(
                    buffer, capacity, offset, sim_ir_mathview_noise_law_name(node->data.noise.law));
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, ",\"dist\":");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_json_string(
                    buffer,
                    capacity,
                    offset,
                    sim_ir_mathview_noise_distribution_name(node->data.noise.distribution));
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}");
                if (result != SIM_RESULT_OK) {
                    break;
                }
            } else if (node->type == SIM_IR_NODE_WARP) {
                result =
                    sim_ir_mathview_append(buffer, capacity, offset, ",\"params\":{\"profile\":");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_json_string(
                    buffer,
                    capacity,
                    offset,
                    sim_ir_mathview_warp_profile_name(node->data.warp.profile));
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(
                    buffer,
                    capacity,
                    offset,
                    ",\"bias\":%.17g,\"delta\":%.17g,\"lambda\":%.17g,\"radius\":%.17g,"
                    "\"tolerance\":%.17g,\"guard\":{\"mode\":%d,\"clamp_min\":%.17g,\"clamp_max\":%"
                    ".17g,\"tolerance\":%.17g}}",
                    node->data.warp.bias,
                    node->data.warp.delta,
                    node->data.warp.lambda,
                    node->data.warp.tolerance,
                    node->data.warp.guard.mode,
                    node->data.warp.guard.clamp_min,
                    node->data.warp.guard.clamp_max,
                    node->data.warp.guard.tolerance);
                if (result != SIM_RESULT_OK) {
                    break;
                }
            } else if (node->type == SIM_IR_NODE_COORD) {
                result = sim_ir_mathview_append(
                    buffer, capacity, offset, ",\"params\":{\"axis\":%zu}", node->data.coord.axis);
                if (result != SIM_RESULT_OK) {
                    break;
                }
            } else if (node->type == SIM_IR_NODE_STATEFUL) {
                const char* label =
                    node->data.stateful.label ? node->data.stateful.label : "stateful";
                result =
                    sim_ir_mathview_append(buffer, capacity, offset, ",\"params\":{\"label\":");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_json_string(buffer, capacity, offset, label);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}");
                if (result != SIM_RESULT_OK) {
                    break;
                }
            }

            result = sim_ir_mathview_append(buffer, capacity, offset, "}");
            break;
        }

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV: {
            const char* op = (node->type == SIM_IR_NODE_ADD)   ? "+"
                             : (node->type == SIM_IR_NODE_SUB) ? "-"
                             : (node->type == SIM_IR_NODE_MUL) ? "*"
                                                               : "/";
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "{\"node\":\"Binary\",\"type\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_type_json(buffer, capacity, offset, node->value_type);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"op\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_json_string(buffer, capacity, offset, op);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"lhs\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_json(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",\"rhs\":");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_json(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "}");
            break;
        }

        default:
            result = SIM_RESULT_NOT_SUPPORTED;
            break;
    }

done:
    if (stack_flags != NULL) {
        stack_flags[node_id] = 0U;
    }
    return result;
}

SimResult sim_ir_mathview_render_json(const SimIRBuilder* builder,
                                      SimIRNodeId         root,
                                      char*               buffer,
                                      size_t              capacity,
                                      size_t*             out_length) {
    size_t         offset      = 0U;
    unsigned char* stack_flags = NULL;
    if (builder != NULL && builder->count > 0U) {
        stack_flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
        if (stack_flags == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    SimResult result =
        sim_ir_mathview_render_node_json(builder, root, buffer, capacity, &offset, stack_flags);
    free(stack_flags);

    if (out_length != NULL) {
        *out_length = offset;
    }

    if (buffer != NULL && capacity > 0U) {
        if (offset >= capacity) {
            buffer[capacity - 1U] = '\0';
            return (result == SIM_RESULT_OK) ? SIM_RESULT_OUT_OF_MEMORY : result;
        }
        buffer[offset] = '\0';
    }

    return result;
}

static SimResult
sim_ir_mathview_append_latex_text(char* buffer, size_t capacity, size_t* offset, const char* text) {
    if (offset == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (text == NULL) {
        text = "";
    }

    while (*text != '\0') {
        char      ch     = *text++;
        SimResult result = SIM_RESULT_OK;
        if (ch == '_' || ch == '\\' || ch == '{' || ch == '}') {
            result = sim_ir_mathview_append_raw(buffer, capacity, offset, "\\", 1U);
            if (result != SIM_RESULT_OK) {
                return result;
            }
        }
        result = sim_ir_mathview_append_raw(buffer, capacity, offset, &ch, 1U);
        if (result != SIM_RESULT_OK) {
            return result;
        }
    }
    return SIM_RESULT_OK;
}

static SimResult sim_ir_mathview_render_node_latex(const SimIRBuilder* builder,
                                                   SimIRNodeId         node_id,
                                                   char*               buffer,
                                                   size_t              capacity,
                                                   size_t*             offset,
                                                   unsigned char*      stack_flags) {
    const SimIRNode* node   = NULL;
    SimResult        result = SIM_RESULT_OK;

    if (builder == NULL || offset == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (stack_flags != NULL) {
        if (stack_flags[node_id] != 0U) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }
        stack_flags[node_id] = 1U;
    }

    node = sim_ir_builder_get(builder, node_id);
    if (node == NULL) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto done;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT: {
            size_t components =
                node->value_type.components == 0U ? 1U : node->value_type.components;
            if (components <= 1U) {
                result = sim_ir_mathview_append_scalar_constant(
                    buffer, capacity, offset, node);
                break;
            }

            result = sim_ir_mathview_append(buffer, capacity, offset, "\\left[");
            if (result != SIM_RESULT_OK) {
                break;
            }
            for (size_t i = 0U; i < components; ++i) {
                double value = 0.0;
                if (!sim_ir_mathview_constant_value(builder, node, i, &value)) {
                    value = 0.0;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "%.17g", value);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                if (i + 1U < components) {
                    result = sim_ir_mathview_append(buffer, capacity, offset, ",");
                    if (result != SIM_RESULT_OK) {
                        break;
                    }
                }
            }
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "\\right]");
            break;
        }
        case SIM_IR_NODE_FIELD_REF:
            result = sim_ir_mathview_append(buffer, capacity, offset, "f_{%zu}", node->data.field);
            break;

        case SIM_IR_NODE_PARAM: {
            switch (node->data.param.param) {
                case SIM_IR_PARAM_DT:
                    result = sim_ir_mathview_append(buffer, capacity, offset, "\\mathrm{dt}");
                    break;
                case SIM_IR_PARAM_STEP_INDEX:
                    result = sim_ir_mathview_append(buffer, capacity, offset, "n");
                    break;
                case SIM_IR_PARAM_SQRT_DT:
                    result =
                        sim_ir_mathview_append(buffer, capacity, offset, "\\sqrt{\\mathrm{dt}}");
                    break;
                case SIM_IR_PARAM_TIME:
                    result = sim_ir_mathview_append(buffer, capacity, offset, "t");
                    break;
                default:
                    result = sim_ir_mathview_append(
                        buffer, capacity, offset, "p_{%u}", (unsigned) node->data.param.param);
                    break;
            }
            break;
        }

        case SIM_IR_NODE_INDEX:
            result = sim_ir_mathview_append(buffer, capacity, offset, "i");
            break;

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV: {
            const char* op = (node->type == SIM_IR_NODE_ADD)   ? "+"
                             : (node->type == SIM_IR_NODE_SUB) ? "-"
                             : (node->type == SIM_IR_NODE_MUL) ? "\\cdot"
                                                               : "/";
            if (node->type == SIM_IR_NODE_DIV) {
                result = sim_ir_mathview_append(buffer, capacity, offset, "\\frac{");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_render_node_latex(
                    builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}{");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_render_node_latex(
                    builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}");
                break;
            }

            result = sim_ir_mathview_append(buffer, capacity, offset, "(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, " %s ", op);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        case SIM_IR_NODE_CALL: {
            const char* name = sim_ir_mathview_call_name(node->data.call.kind);
            if (node->data.call.kind == SIM_IR_CALL_SIN ||
                node->data.call.kind == SIM_IR_CALL_COS ||
                node->data.call.kind == SIM_IR_CALL_EXP ||
                node->data.call.kind == SIM_IR_CALL_LOG ||
                node->data.call.kind == SIM_IR_CALL_TANH ||
                node->data.call.kind == SIM_IR_CALL_SINH) {
                result = sim_ir_mathview_append(buffer, capacity, offset, "\\%s(", name);
            } else if (node->data.call.kind == SIM_IR_CALL_ABS) {
                result = sim_ir_mathview_append(buffer, capacity, offset, "\\left|");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_render_node_latex(
                    builder, node->data.call.operand, buffer, capacity, offset, stack_flags);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "\\right|");
                break;
            } else {
                result = sim_ir_mathview_append(buffer, capacity, offset, "\\operatorname{");
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append_latex_text(buffer, capacity, offset, name);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = sim_ir_mathview_append(buffer, capacity, offset, "}(");
            }
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.call.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        case SIM_IR_NODE_FLOOR:
            result = sim_ir_mathview_append(buffer, capacity, offset, "\\left\\lfloor ");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.unary.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, " \\right\\rfloor");
            break;

        case SIM_IR_NODE_MOD:
            result = sim_ir_mathview_append(buffer, capacity, offset, "\\operatorname{mod}(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_POW:
            result = sim_ir_mathview_append(buffer, capacity, offset, "{");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.binary.lhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "}^{");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.binary.rhs, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, "}");
            break;

        case SIM_IR_NODE_COORD:
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            "\\operatorname{coord}(f_{%zu},%zu)",
                                            node->data.coord.field,
                                            node->data.coord.axis);
            break;

        case SIM_IR_NODE_DIFF:
            result = sim_ir_mathview_append(buffer, capacity, offset, "\\operatorname{diff}(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.diff.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            ",%zu,%.17g,%.17g,%zu,%zu,",
                                            node->data.diff.axis,
                                            node->data.diff.dx,
                                            node->data.diff.scale,
                                            node->data.diff.order,
                                            node->data.diff.stencil_order);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_latex_text(
                buffer, capacity, offset, sim_ir_mathview_diff_method_name(node->data.diff.method));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(
                buffer, capacity, offset, ",%.17g,", node->data.diff.consistency_constant);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_latex_text(
                buffer, capacity, offset, sim_ir_mathview_boundary_name(node->data.diff.boundary));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_NOISE:
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            "\\operatorname{noise}(%u,%.17g,%.17g,",
                                            node->data.noise.seed,
                                            node->data.noise.amplitude,
                                            node->data.noise.variance);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_latex_text(
                buffer, capacity, offset, sim_ir_mathview_noise_law_name(node->data.noise.law));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_latex_text(
                buffer,
                capacity,
                offset,
                sim_ir_mathview_noise_distribution_name(node->data.noise.distribution));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_WARP:
            result = sim_ir_mathview_append(buffer, capacity, offset, "\\operatorname{warp}(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_latex_text(
                buffer,
                capacity,
                offset,
                sim_ir_mathview_warp_profile_name(node->data.warp.profile));
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.warp.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer,
                                            capacity,
                                            offset,
                                            ",%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g)",
                                            node->data.warp.bias,
                                            node->data.warp.delta,
                                            node->data.warp.lambda,
                                            node->data.warp.tolerance,
                                            node->data.warp.guard.mode,
                                            node->data.warp.guard.clamp_min,
                                            node->data.warp.guard.clamp_max,
                                            node->data.warp.guard.tolerance);
            break;

        case SIM_IR_NODE_COMPLEX_ROTATE:
            result = sim_ir_mathview_append(
                buffer, capacity, offset, "\\operatorname{complex\\_rotate}(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.complex_rotate.operand, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.complex_rotate.angle, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_COMPLEX_PACK:
            result =
                sim_ir_mathview_append(buffer, capacity, offset, "\\operatorname{complex\\_pack}(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.complex_pack.real, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ",");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_render_node_latex(
                builder, node->data.complex_pack.imag, buffer, capacity, offset, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;

        case SIM_IR_NODE_STATEFUL: {
            const char* label = node->data.stateful.label ? node->data.stateful.label : "stateful";
            result = sim_ir_mathview_append(buffer, capacity, offset, "\\operatorname{stateful}(");
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append_latex_text(buffer, capacity, offset, label);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_mathview_append(buffer, capacity, offset, ")");
            break;
        }

        default:
            result = SIM_RESULT_NOT_SUPPORTED;
            break;
    }

done:
    if (stack_flags != NULL) {
        stack_flags[node_id] = 0U;
    }
    return result;
}

SimResult sim_ir_mathview_render_latex(const SimIRBuilder* builder,
                                       SimIRNodeId         root,
                                       char*               buffer,
                                       size_t              capacity,
                                       size_t*             out_length) {
    size_t         offset      = 0U;
    unsigned char* stack_flags = NULL;
    if (builder != NULL && builder->count > 0U) {
        stack_flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
        if (stack_flags == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    SimResult result =
        sim_ir_mathview_render_node_latex(builder, root, buffer, capacity, &offset, stack_flags);
    free(stack_flags);

    if (out_length != NULL) {
        *out_length = offset;
    }

    if (buffer != NULL && capacity > 0U) {
        if (offset >= capacity) {
            buffer[capacity - 1U] = '\0';
            return (result == SIM_RESULT_OK) ? SIM_RESULT_OUT_OF_MEMORY : result;
        }
        buffer[offset] = '\0';
    }

    return result;
}

static uint64_t sim_ir_mathview_hash_bytes(uint64_t hash, const char* data, size_t length) {
    const unsigned char* bytes = (const unsigned char*) data;
    for (size_t i = 0U; i < length; ++i) {
        hash ^= (uint64_t) bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

SimResult sim_ir_mathview_hash(const SimIRBuilder* builder, SimIRNodeId root, uint64_t* out_hash) {
    if (out_hash == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t    length = 0U;
    SimResult result = sim_ir_mathview_render(builder, root, NULL, 0U, &length);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    char* buffer = (char*) malloc(length + 1U);
    if (buffer == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    result = sim_ir_mathview_render(builder, root, buffer, length + 1U, NULL);
    if (result != SIM_RESULT_OK) {
        free(buffer);
        return result;
    }

    uint64_t    hash        = 14695981039346656037ULL;
    const char* version_tag = "mathview:v2:";
    hash                    = sim_ir_mathview_hash_bytes(hash, version_tag, strlen(version_tag));
    hash                    = sim_ir_mathview_hash_bytes(hash, buffer, length);

    free(buffer);
    *out_hash = hash;
    return SIM_RESULT_OK;
}

typedef struct SimSHA256 {
    uint32_t      state[8];
    uint64_t      length;
    unsigned char buffer[64];
    size_t        buffer_len;
} SimSHA256;

static uint32_t sim_sha256_rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32U - bits));
}

static void sim_sha256_transform(SimSHA256* ctx, const unsigned char data[64]) {
    static const uint32_t k[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U
    };
    uint32_t w[64];
    for (size_t i = 0U; i < 16U; ++i) {
        size_t idx = i * 4U;
        w[i]       = ((uint32_t) data[idx] << 24U) | ((uint32_t) data[idx + 1U] << 16U) |
               ((uint32_t) data[idx + 2U] << 8U) | ((uint32_t) data[idx + 3U]);
    }
    for (size_t i = 16U; i < 64U; ++i) {
        uint32_t s0 =
            sim_sha256_rotr(w[i - 15U], 7U) ^ sim_sha256_rotr(w[i - 15U], 18U) ^ (w[i - 15U] >> 3U);
        uint32_t s1 =
            sim_sha256_rotr(w[i - 2U], 17U) ^ sim_sha256_rotr(w[i - 2U], 19U) ^ (w[i - 2U] >> 10U);
        w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];

    for (size_t i = 0U; i < 64U; ++i) {
        uint32_t S1    = sim_sha256_rotr(e, 6U) ^ sim_sha256_rotr(e, 11U) ^ sim_sha256_rotr(e, 25U);
        uint32_t ch    = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0    = sim_sha256_rotr(a, 2U) ^ sim_sha256_rotr(a, 13U) ^ sim_sha256_rotr(a, 22U);
        uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sim_sha256_init(SimSHA256* ctx) {
    if (ctx == NULL) {
        return;
    }
    ctx->state[0]   = 0x6a09e667U;
    ctx->state[1]   = 0xbb67ae85U;
    ctx->state[2]   = 0x3c6ef372U;
    ctx->state[3]   = 0xa54ff53aU;
    ctx->state[4]   = 0x510e527fU;
    ctx->state[5]   = 0x9b05688cU;
    ctx->state[6]   = 0x1f83d9abU;
    ctx->state[7]   = 0x5be0cd19U;
    ctx->length     = 0U;
    ctx->buffer_len = 0U;
}

static void sim_sha256_update(SimSHA256* ctx, const unsigned char* data, size_t length) {
    if (ctx == NULL || data == NULL || length == 0U) {
        return;
    }

    ctx->length += (uint64_t) length;
    size_t offset = 0U;

    if (ctx->buffer_len > 0U) {
        size_t needed  = 64U - ctx->buffer_len;
        size_t to_copy = (length < needed) ? length : needed;
        memcpy(ctx->buffer + ctx->buffer_len, data, to_copy);
        ctx->buffer_len += to_copy;
        offset += to_copy;
        if (ctx->buffer_len == 64U) {
            sim_sha256_transform(ctx, ctx->buffer);
            ctx->buffer_len = 0U;
        }
    }

    while (offset + 64U <= length) {
        sim_sha256_transform(ctx, data + offset);
        offset += 64U;
    }

    if (offset < length) {
        size_t remaining = length - offset;
        memcpy(ctx->buffer, data + offset, remaining);
        ctx->buffer_len = remaining;
    }
}

static void sim_sha256_final(SimSHA256* ctx, unsigned char out_hash[32]) {
    if (ctx == NULL || out_hash == NULL) {
        return;
    }

    uint64_t bit_len               = ctx->length * 8U;
    ctx->buffer[ctx->buffer_len++] = 0x80U;

    if (ctx->buffer_len > 56U) {
        while (ctx->buffer_len < 64U) {
            ctx->buffer[ctx->buffer_len++] = 0U;
        }
        sim_sha256_transform(ctx, ctx->buffer);
        ctx->buffer_len = 0U;
    }

    while (ctx->buffer_len < 56U) {
        ctx->buffer[ctx->buffer_len++] = 0U;
    }

    for (int i = 7; i >= 0; --i) {
        ctx->buffer[ctx->buffer_len++] = (unsigned char) ((bit_len >> (i * 8)) & 0xFFU);
    }

    sim_sha256_transform(ctx, ctx->buffer);

    for (size_t i = 0U; i < 8U; ++i) {
        out_hash[i * 4U]      = (unsigned char) ((ctx->state[i] >> 24U) & 0xFFU);
        out_hash[i * 4U + 1U] = (unsigned char) ((ctx->state[i] >> 16U) & 0xFFU);
        out_hash[i * 4U + 2U] = (unsigned char) ((ctx->state[i] >> 8U) & 0xFFU);
        out_hash[i * 4U + 3U] = (unsigned char) (ctx->state[i] & 0xFFU);
    }
}

SimResult sim_ir_mathview_hash_sha256(const SimIRBuilder* builder,
                                      SimIRNodeId         root,
                                      unsigned char       out_hash[32]) {
    if (out_hash == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t    length = 0U;
    SimResult result = sim_ir_mathview_render(builder, root, NULL, 0U, &length);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    char* buffer = (char*) malloc(length + 1U);
    if (buffer == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    result = sim_ir_mathview_render(builder, root, buffer, length + 1U, NULL);
    if (result != SIM_RESULT_OK) {
        free(buffer);
        return result;
    }

    SimSHA256 ctx;
    sim_sha256_init(&ctx);
    sim_sha256_update(&ctx, (const unsigned char*) buffer, length);
    sim_sha256_final(&ctx, out_hash);

    free(buffer);
    return SIM_RESULT_OK;
}
