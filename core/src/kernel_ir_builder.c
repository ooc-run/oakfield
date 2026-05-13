/**
 * @file kernel_ir_builder.c
 * @brief KernelIR builder allocation, node construction, and reachability tagging.
 *
 * The builder records typed expression nodes in append-only arrays, normalizes
 * constants and warp guards, and returns SIM_IR_INVALID_NODE when allocation or
 * input validation fails. Helper entry points build scalar, vector, integer,
 * complex, differential, noise, stateful, and analytic warp nodes for fused
 * operator kernels.
 */
#include "internal/kernel_ir_internal.h"

static SimIRWarpGuard sim_ir_guard_normalize(SimIRWarpGuard guard) {
    SimContinuityMode mode = (SimContinuityMode) guard.mode;
    switch (mode) {
        case SIM_CONTINUITY_NONE:
        case SIM_CONTINUITY_STRICT:
        case SIM_CONTINUITY_CLAMPED:
        case SIM_CONTINUITY_LIMITED:
            break;
        default:
            mode = SIM_CONTINUITY_NONE;
            break;
    }

    guard.mode = (int) mode;

    if (mode == SIM_CONTINUITY_CLAMPED || mode == SIM_CONTINUITY_LIMITED) {
        if (!isfinite(guard.clamp_min)) {
            guard.clamp_min = -1.0e6;
        }
        if (!isfinite(guard.clamp_max)) {
            guard.clamp_max = 1.0e6;
        }
        if (guard.clamp_min > guard.clamp_max) {
            double tmp      = guard.clamp_min;
            guard.clamp_min = guard.clamp_max;
            guard.clamp_max = tmp;
        }
        guard.tolerance = sim_ir_positive_or_default(guard.tolerance, 1.0e-6);
    } else {
        guard.clamp_min = 0.0;
        guard.clamp_max = 0.0;
        guard.tolerance = 0.0;
    }

    return guard;
}

static SimIROpcode sim_ir_opcode_sanitize(SimIROpcode opcode) {
    switch (opcode) {
        case OAK_OP_DIFF:
        case OAK_OP_CONV:
        case OAK_OP_DISP:
        case OAK_OP_DIFFUSE:
        case OAK_OP_WARP:
        case OAK_OP_NOISE:
        case OAK_OP_FLOW:
        case OAK_OP_MISC:
        case OAK_OP_CORE:
            return opcode;
        default:
            return OAK_OP_MISC;
    }
}

static bool sim_ir_param_valid(SimIRParamKind param) {
    switch (param) {
        case SIM_IR_PARAM_DT:
        case SIM_IR_PARAM_STEP_INDEX:
        case SIM_IR_PARAM_SQRT_DT:
        case SIM_IR_PARAM_TIME:
            return true;
        default:
            return false;
    }
}

static SimWarpLevel sim_ir_warp_level_from_profile(SimIRWarpProfile profile) {
    switch (profile) {
        case SIM_IR_WARP_PROFILE_DIGAMMA:
        case SIM_IR_WARP_PROFILE_DIGAMMA_7_TAIL:
        case SIM_IR_WARP_PROFILE_DIGAMMA_5_TAIL:
        case SIM_IR_WARP_PROFILE_DIGAMMA_ADAPTIVE:
        case SIM_IR_WARP_PROFILE_DIGAMMA_MORTICI:
        case SIM_IR_WARP_PROFILE_TRIGAMMA:
        default:
            return SIM_WARP_LEVEL_LEVEL2;
    }
}

static SimResult sim_ir_builder_reserve(SimIRBuilder* builder, size_t additional) {
    size_t     required;
    size_t     new_capacity;
    SimIRNode* nodes;

    if (builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (additional == 0U) {
        return SIM_RESULT_OK;
    }

    if (builder->count + additional <= builder->capacity) {
        return SIM_RESULT_OK;
    }

    required     = builder->count + additional;
    new_capacity = (builder->capacity == 0U) ? 8U : builder->capacity;

    while (new_capacity < required) {
        new_capacity *= 2U;
    }

    nodes = (SimIRNode*) realloc(builder->nodes, new_capacity * sizeof(SimIRNode));
    if (nodes == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    builder->nodes    = nodes;
    builder->capacity = new_capacity;
    return SIM_RESULT_OK;
}

SimResult sim_ir_builder_init(SimIRBuilder* builder) {
    if (builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    builder->nodes                   = NULL;
    builder->count                   = 0U;
    builder->capacity                = 0U;
    builder->constants_data          = NULL;
    builder->constants_offsets       = NULL;
    builder->constants_components    = NULL;
    builder->constants_count         = 0U;
    builder->constants_capacity      = 0U;
    builder->constants_data_capacity = 0U;
    builder->constants_data_used     = 0U;
    builder->default_boundary        = SIM_IR_BOUNDARY_NEUMANN;
    return SIM_RESULT_OK;
}

void sim_ir_diff_spec_init(SimIRDiffSpec* spec, const SimIRBuilder* builder) {
    if (spec == NULL) {
        return;
    }
    spec->operand              = SIM_IR_INVALID_NODE;
    spec->axis                 = 0U;
    spec->dx                   = 1.0;
    spec->scale                = 1.0;
    spec->order                = 1U;
    spec->stencil_order        = 1U;
    spec->method               = SIM_IR_DIFF_METHOD_AUTO;
    spec->consistency_constant = 0.0;
    spec->boundary             = SIM_IR_BOUNDARY_NEUMANN;
    spec->result_type          = sim_ir_type_scalar();
    if (builder != NULL) {
        spec->boundary = builder->default_boundary;
    }
}

void sim_ir_builder_destroy(SimIRBuilder* builder) {
    if (builder == NULL) {
        return;
    }

    free(builder->nodes);
    builder->nodes    = NULL;
    builder->count    = 0U;
    builder->capacity = 0U;
    free(builder->constants_data);
    builder->constants_data = NULL;
    free(builder->constants_offsets);
    builder->constants_offsets = NULL;
    free(builder->constants_components);
    builder->constants_components    = NULL;
    builder->constants_count         = 0U;
    builder->constants_capacity      = 0U;
    builder->constants_data_capacity = 0U;
    builder->constants_data_used     = 0U;
    builder->default_boundary        = SIM_IR_BOUNDARY_NEUMANN;
}

void sim_ir_builder_set_default_boundary(SimIRBuilder* builder, SimIRBoundaryPolicy boundary) {
    if (builder == NULL) {
        return;
    }
    switch (boundary) {
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_PERIODIC:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            builder->default_boundary = boundary;
            break;
        default:
            builder->default_boundary = SIM_IR_BOUNDARY_NEUMANN;
            break;
    }
}

void sim_ir_builder_reset(SimIRBuilder* builder) {
    if (builder == NULL) {
        return;
    }

    builder->count               = 0U;
    builder->constants_count     = 0U;
    builder->constants_data_used = 0U;
}

void sim_ir_builder_apply_opcode(SimIRBuilder* builder,
                                 SimIROpcode   opcode,
                                 bool          preserve_existing) {
    if (builder == NULL || builder->nodes == NULL) {
        return;
    }

    SimIROpcode resolved = sim_ir_opcode_sanitize(opcode);
    for (size_t i = 0U; i < builder->count; ++i) {
        SimIRNode* node = &builder->nodes[i];
        if (!preserve_existing || node->opcode == OAK_OP_MISC) {
            node->opcode = resolved;
        }
    }
}

static SimResult sim_ir_collect_reachable_node(const SimIRBuilder* builder,
                                               SimIRNodeId         node_id,
                                               unsigned char*      reachable,
                                               unsigned char*      stack_flags) {
    if (builder == NULL || reachable == NULL || stack_flags == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (stack_flags[node_id] != 0U) {
        return SIM_RESULT_DEPENDENCY_ERROR;
    }

    if (reachable[node_id] != 0U) {
        return SIM_RESULT_OK;
    }

    reachable[node_id]   = 1U;
    stack_flags[node_id] = 1U;

    const SimIRNode* node   = &builder->nodes[node_id];
    SimResult        result = SIM_RESULT_OK;

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
        case SIM_IR_NODE_FIELD_REF:
        case SIM_IR_NODE_PARAM:
        case SIM_IR_NODE_INDEX:
        case SIM_IR_NODE_NOISE:
        case SIM_IR_NODE_COORD:
        case SIM_IR_NODE_STATEFUL:
            break;

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV:
        case SIM_IR_NODE_POW:
        case SIM_IR_NODE_MOD:
            result = sim_ir_collect_reachable_node(
                builder, node->data.binary.lhs, reachable, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_collect_reachable_node(
                builder, node->data.binary.rhs, reachable, stack_flags);
            break;

        case SIM_IR_NODE_DIFF:
            result = sim_ir_collect_reachable_node(
                builder, node->data.diff.operand, reachable, stack_flags);
            break;

        case SIM_IR_NODE_CALL:
            result = sim_ir_collect_reachable_node(
                builder, node->data.call.operand, reachable, stack_flags);
            break;

        case SIM_IR_NODE_FLOOR:
            result = sim_ir_collect_reachable_node(
                builder, node->data.unary.operand, reachable, stack_flags);
            break;

        case SIM_IR_NODE_WARP:
            result = sim_ir_collect_reachable_node(
                builder, node->data.warp.operand, reachable, stack_flags);
            break;

        case SIM_IR_NODE_COMPLEX_PACK:
            result = sim_ir_collect_reachable_node(
                builder, node->data.complex_pack.real, reachable, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_collect_reachable_node(
                builder, node->data.complex_pack.imag, reachable, stack_flags);
            break;

        case SIM_IR_NODE_COMPLEX_ROTATE:
            result = sim_ir_collect_reachable_node(
                builder, node->data.complex_rotate.operand, reachable, stack_flags);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = sim_ir_collect_reachable_node(
                builder, node->data.complex_rotate.angle, reachable, stack_flags);
            break;

        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

    stack_flags[node_id] = 0U;
    return result;
}

SimResult sim_ir_collect_reachable(const SimIRBuilder* builder,
                                   const SimIRNodeId*  roots,
                                   size_t              root_count,
                                   unsigned char*      out_reachable,
                                   size_t              reachable_count) {
    if (builder == NULL || out_reachable == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (builder->count > reachable_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (root_count > 0U && roots == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (builder->count == 0U) {
        return SIM_RESULT_OK;
    }

    (void) memset(out_reachable, 0, builder->count * sizeof(unsigned char));

    unsigned char* stack_flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
    if (stack_flags == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimResult result = SIM_RESULT_OK;
    for (size_t i = 0U; i < root_count; ++i) {
        result = sim_ir_collect_reachable_node(builder, roots[i], out_reachable, stack_flags);
        if (result != SIM_RESULT_OK) {
            break;
        }
    }

    free(stack_flags);
    return result;
}

SimResult sim_ir_builder_apply_opcode_reachable(SimIRBuilder*      builder,
                                                SimIROpcode        opcode,
                                                bool               preserve_existing,
                                                const SimIRNodeId* roots,
                                                size_t             root_count) {
    if (builder == NULL || builder->nodes == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (builder->count == 0U) {
        return SIM_RESULT_OK;
    }

    unsigned char* reachable = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
    if (reachable == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimResult result =
        sim_ir_collect_reachable(builder, roots, root_count, reachable, builder->count);
    if (result == SIM_RESULT_OK) {
        SimIROpcode resolved = sim_ir_opcode_sanitize(opcode);
        for (size_t i = 0U; i < builder->count; ++i) {
            if (reachable[i] == 0U) {
                continue;
            }
            SimIRNode* node = &builder->nodes[i];
            if (!preserve_existing || node->opcode == OAK_OP_MISC) {
                node->opcode = resolved;
            }
        }
    }

    free(reachable);
    return result;
}

const SimIRNode* sim_ir_builder_get(const SimIRBuilder* builder, SimIRNodeId id) {
    if (builder == NULL || id == SIM_IR_INVALID_NODE || id >= builder->count) {
        return NULL;
    }
    return &builder->nodes[id];
}

SimIRType sim_ir_builder_node_type(const SimIRBuilder* builder, SimIRNodeId id) {
    const SimIRNode* node = sim_ir_builder_get(builder, id);
    if (node == NULL) {
        return sim_ir_type_scalar();
    }
    return sim_ir_type_normalize(node->value_type);
}

SimWarpLevel sim_ir_builder_node_warp_class(const SimIRBuilder* builder, SimIRNodeId id) {
    const SimIRNode* node = sim_ir_builder_get(builder, id);
    if (node == NULL) {
        return SIM_WARP_LEVEL_NONE;
    }
    return node->warp_class;
}

SimResult
sim_ir_builder_set_node_warp_class(SimIRBuilder* builder, SimIRNodeId id, SimWarpLevel warp_class) {
    SimIRNode* node;

    if (builder == NULL || id == SIM_IR_INVALID_NODE || id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    switch (warp_class) {
        case SIM_WARP_LEVEL_NONE:
        case SIM_WARP_LEVEL_LEVEL0:
        case SIM_WARP_LEVEL_LEVEL1:
        case SIM_WARP_LEVEL_LEVEL2:
            break;
        default:
            return SIM_RESULT_INVALID_ARGUMENT;
    }

    node             = &builder->nodes[id];
    node->warp_class = warp_class;
    return SIM_RESULT_OK;
}

static SimIRNodeId sim_ir_builder_append(SimIRBuilder* builder) {
    SimResult result;

    result = sim_ir_builder_reserve(builder, 1U);
    if (result != SIM_RESULT_OK) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId id   = builder->count++;
    SimIRNode*  node = &builder->nodes[id];
    (void) memset(node, 0, sizeof(*node));
    node->opcode     = OAK_OP_MISC;
    node->warp_class = SIM_WARP_LEVEL_NONE;
    return id;
}

static SimIRNodeId
sim_ir_builder_constant_integer_raw_typed(SimIRBuilder* builder, uint64_t raw, SimIRType type) {
    SimIRNodeId    id;
    SimIRNode*     node;
    SimIRType      normalized;
    SimScalarDomain domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    normalized = sim_ir_type_normalize(type);
    domain     = normalized.scalar_domain;
    if (!sim_ir_type_is_scalar(normalized) || !sim_ir_domain_is_exact_integer(domain)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                               = &builder->nodes[id];
    node->type                         = SIM_IR_NODE_CONSTANT;
    node->id                           = id;
    node->value_type                   = normalized;
    node->is_local                     = true;
    node->data.constant.scalar         = sim_ir_integer_raw_to_double(raw, domain);
    node->data.constant.signed_scalar  = sim_ir_integer_as_i64(raw, domain);
    node->data.constant.unsigned_scalar = sim_ir_integer_truncate(raw, domain);
    node->data.constant.exact_integer  = true;
    node->data.constant.constant_index = SIM_IR_INVALID_CONSTANT_INDEX;
    return id;
}

SimIRNodeId sim_ir_builder_constant_typed(SimIRBuilder* builder, double value, SimIRType type) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   normalized;
    SimScalarDomain domain;
    uint64_t    raw = 0U;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    normalized = sim_ir_type_normalize(type);
    domain     = normalized.scalar_domain;
    if (sim_ir_domain_is_exact_integer(domain)) {
        if (!sim_ir_integer_raw_from_double(value, domain, &raw)) {
            return SIM_IR_INVALID_NODE;
        }
        return sim_ir_builder_constant_integer_raw_typed(builder, raw, normalized);
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                               = &builder->nodes[id];
    node->type                         = SIM_IR_NODE_CONSTANT;
    node->id                           = id;
    node->value_type                   = normalized;
    node->is_local                     = true;
    node->data.constant.scalar         = sim_ir_canonicalize_constant(value);
    node->data.constant.signed_scalar  = 0;
    node->data.constant.unsigned_scalar = 0U;
    node->data.constant.exact_integer  = false;
    node->data.constant.constant_index = SIM_IR_INVALID_CONSTANT_INDEX; /* mark no pool entry */
    return id;
}

SimIRNodeId sim_ir_builder_constant_i64_typed(SimIRBuilder* builder, int64_t value, SimIRType type) {
    SimIRType      normalized;
    SimScalarDomain domain;
    uint64_t       raw = 0U;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    normalized = sim_ir_type_normalize(type);
    domain     = normalized.scalar_domain;
    if (!sim_ir_integer_raw_from_signed(value, domain, &raw)) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_constant_integer_raw_typed(builder, raw, normalized);
}

SimIRNodeId
sim_ir_builder_constant_u64_typed(SimIRBuilder* builder, uint64_t value, SimIRType type) {
    SimIRType      normalized;
    SimScalarDomain domain;
    uint64_t       raw = 0U;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    normalized = sim_ir_type_normalize(type);
    domain     = normalized.scalar_domain;
    if (!sim_ir_integer_raw_from_unsigned(value, domain, &raw)) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_constant_integer_raw_typed(builder, raw, normalized);
}

static SimResult sim_ir_builder_constants_reserve(SimIRBuilder* builder, size_t additional) {
    size_t  required;
    size_t  new_capacity;
    size_t* new_offsets;
    size_t* new_components;

    if (builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (additional == 0U) {
        return SIM_RESULT_OK;
    }

    required = builder->constants_count + additional;
    if (required <= builder->constants_capacity) {
        return SIM_RESULT_OK;
    }

    new_capacity = (builder->constants_capacity == 0U) ? 4U : builder->constants_capacity;
    while (new_capacity < required) {
        new_capacity *= 2U;
    }

    new_offsets    = (size_t*) malloc(new_capacity * sizeof(size_t));
    new_components = (size_t*) malloc(new_capacity * sizeof(size_t));
    if (new_offsets == NULL || new_components == NULL) {
        free(new_offsets);
        free(new_components);
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (builder->constants_count > 0U) {
        memcpy(new_offsets, builder->constants_offsets, builder->constants_count * sizeof(size_t));
        memcpy(new_components,
               builder->constants_components,
               builder->constants_count * sizeof(size_t));
    }
    free(builder->constants_offsets);
    free(builder->constants_components);

    builder->constants_offsets    = new_offsets;
    builder->constants_components = new_components;
    builder->constants_capacity   = new_capacity;
    return SIM_RESULT_OK;
}

SimIRNodeId sim_ir_builder_constant_vector_typed(SimIRBuilder* builder,
                                                 const double* values,
                                                 size_t        components,
                                                 SimIRType     type) {
    SimIRNodeId id;
    SimIRNode*  node;
    size_t      idx;
    size_t      i;
    SimResult   res;
    SimIRType   normalized;

    if (builder == NULL || values == NULL || components == 0U) {
        return SIM_IR_INVALID_NODE;
    }

    normalized = sim_ir_type_normalize(type);
    if (normalized.components != components) {
        return SIM_IR_INVALID_NODE;
    }
    if (sim_ir_domain_is_exact_integer(normalized.scalar_domain)) {
        return SIM_IR_INVALID_NODE;
    }

    /* Skip sizing till we know this will be added to the pool */

    /* For small vectors, pack inline into the node to avoid heap lookups */
    if (components <= SIM_IR_SMALL_CONSTANT_CAPACITY) {
        id = sim_ir_builder_append(builder);
        if (id == SIM_IR_INVALID_NODE) {
            return SIM_IR_INVALID_NODE;
        }
        node                               = &builder->nodes[id];
        node->type                         = SIM_IR_NODE_CONSTANT;
        node->id                           = id;
        node->value_type                   = normalized;
        node->is_local                     = true;
        node->data.constant.constant_index = SIM_IR_INVALID_CONSTANT_INDEX;
        for (i = 0U; i < components; ++i) {
            node->data.constant.small[i] = sim_ir_canonicalize_constant(values[i]);
        }
        return id;
    }

    /* Add a new constants pool entry (for larger vectors) */
    /* Deduplicate identical vectors: if an existing pool entry equals values, reuse its index */
    for (size_t k = 0U; k < builder->constants_count; ++k) {
        if (builder->constants_components[k] == components) {
            size_t coff = builder->constants_offsets[k];
            if (builder->constants_data == NULL) {
                continue;
            }
            bool match = true;
            for (size_t c = 0U; c < components; ++c) {
                if (builder->constants_data[coff + c] != sim_ir_canonicalize_constant(values[c])) {
                    match = false;
                    break;
                }
            }
            if (match) {
                /* Found identical constant vector, reuse pool index */
                id = sim_ir_builder_append(builder);
                if (id == SIM_IR_INVALID_NODE) {
                    return SIM_IR_INVALID_NODE;
                }
                node                               = &builder->nodes[id];
                node->type                         = SIM_IR_NODE_CONSTANT;
                node->id                           = id;
                node->value_type                   = normalized;
                node->is_local                     = true;
                node->data.constant.constant_index = k;
                return id;
            }
        }
    }

    res = sim_ir_builder_constants_reserve(builder, 1U);
    if (res != SIM_RESULT_OK) {
        return SIM_IR_INVALID_NODE;
    }

    idx = builder->constants_count;
    /* compute next offset: store contiguous values */
    size_t offset = builder->constants_data_used;
    size_t needed = offset + components;
    if (needed > builder->constants_data_capacity) {
        /* grow capacity to fit */
        size_t new_capacity =
            builder->constants_data_capacity == 0U ? needed : builder->constants_data_capacity;
        while (needed > new_capacity) {
            new_capacity *= 2U;
        }
        double* new_data =
            (double*) realloc(builder->constants_data, new_capacity * sizeof(double));
        if (new_data == NULL) {
            return SIM_IR_INVALID_NODE;
        }
        builder->constants_data          = new_data;
        builder->constants_data_capacity = new_capacity;
    }
    if (builder->constants_data == NULL) {
        return SIM_IR_INVALID_NODE;
    }
    builder->constants_offsets[idx]    = offset;
    builder->constants_components[idx] = components;
    for (i = 0U; i < components; ++i) {
        builder->constants_data[offset + i] = sim_ir_canonicalize_constant(values[i]);
    }
    builder->constants_count += 1U;
    builder->constants_data_used = needed;

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    node                               = &builder->nodes[id];
    node->type                         = SIM_IR_NODE_CONSTANT;
    node->id                           = id;
    node->value_type                   = normalized;
    node->is_local                     = true;
    node->data.constant.constant_index = idx;
    return id;
}

SimIRNodeId
sim_ir_builder_constant_vector(SimIRBuilder* builder, const double* values, size_t components) {
    SimIRType type = sim_ir_type_vector(components);
    return sim_ir_builder_constant_vector_typed(builder, values, components, type);
}

SimIRNodeId sim_ir_builder_constant(SimIRBuilder* builder, double value) {
    return sim_ir_builder_constant_typed(builder, value, sim_ir_type_scalar());
}

SimIRNodeId sim_ir_builder_constant_complex(SimIRBuilder* builder, double real, double imag) {
    double values[2] = { real, imag };
    return sim_ir_builder_constant_vector_typed(builder, values, 2U, sim_ir_type_complex());
}

SimIRNodeId sim_ir_builder_param(SimIRBuilder* builder, SimIRParamKind param) {
    SimIRNodeId id;
    SimIRNode*  node;

    if (builder == NULL || !sim_ir_param_valid(param)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node             = &builder->nodes[id];
    node->type       = SIM_IR_NODE_PARAM;
    node->id         = id;
    node->value_type = (param == SIM_IR_PARAM_STEP_INDEX)
                           ? sim_ir_type_scalar_domain_typed(sim_scalar_domain_u64())
                           : sim_ir_type_scalar();
    node->is_local         = true;
    node->data.param.param = param;
    return id;
}

SimIRNodeId sim_ir_builder_index(SimIRBuilder* builder) {
    SimIRNodeId id;
    SimIRNode*  node;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node             = &builder->nodes[id];
    node->type       = SIM_IR_NODE_INDEX;
    node->id         = id;
    node->value_type = sim_ir_type_scalar_domain_typed(sim_scalar_domain_u64());
    node->is_local   = true;
    return id;
}

SimIRNodeId sim_ir_builder_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   operand_type;
    SimScalarDomain operand_domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (operand == SIM_IR_INVALID_NODE || operand >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    switch (kind) {
        case SIM_IR_CALL_SIN:
        case SIM_IR_CALL_COS:
        case SIM_IR_CALL_EXP:
        case SIM_IR_CALL_ABS:
        case SIM_IR_CALL_LOG:
        case SIM_IR_CALL_TANH:
        case SIM_IR_CALL_SINH:
        case SIM_IR_CALL_SIGN:
            break;
        default:
            return SIM_IR_INVALID_NODE;
    }

    operand_type   = sim_ir_type_normalize(builder->nodes[operand].value_type);
    operand_domain = sim_ir_type_scalar_domain(operand_type);
    if (!sim_scalar_domain_supports(operand_domain, SIM_SCALAR_CAP_ANALYTIC_CALL)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                    = &builder->nodes[id];
    node->type              = SIM_IR_NODE_CALL;
    node->id                = id;
    node->value_type        = operand_type;
    node->is_local          = builder->nodes[operand].is_local;
    node->data.call.kind    = kind;
    node->data.call.operand = operand;
    return id;
}

SimIRNodeId sim_ir_builder_floor(SimIRBuilder* builder, SimIRNodeId operand) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   operand_type;
    SimScalarDomain operand_domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (operand == SIM_IR_INVALID_NODE || operand >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    operand_type   = sim_ir_type_normalize(builder->nodes[operand].value_type);
    operand_domain = sim_ir_type_scalar_domain(operand_type);
    if (!sim_ir_type_is_scalar(operand_type) ||
        !sim_scalar_domain_supports(operand_domain, SIM_SCALAR_CAP_FLOOR)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                     = &builder->nodes[id];
    node->type               = SIM_IR_NODE_FLOOR;
    node->id                 = id;
    node->value_type         = operand_type;
    node->is_local           = builder->nodes[operand].is_local;
    node->data.unary.operand = operand;
    return id;
}

SimIRNodeId sim_ir_builder_mod(SimIRBuilder* builder, SimIRNodeId lhs, SimIRNodeId rhs) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   lhs_type;
    SimIRType   rhs_type;
    SimScalarDomain lhs_domain;
    SimScalarDomain rhs_domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE || lhs >= builder->count ||
        rhs >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    lhs_type   = sim_ir_type_normalize(builder->nodes[lhs].value_type);
    rhs_type   = sim_ir_type_normalize(builder->nodes[rhs].value_type);
    lhs_domain = sim_ir_type_scalar_domain(lhs_type);
    rhs_domain = sim_ir_type_scalar_domain(rhs_type);

    if (!sim_ir_type_is_scalar(lhs_type) || !sim_ir_type_equal(lhs_type, rhs_type) ||
        !sim_scalar_domain_supports(lhs_domain, SIM_SCALAR_CAP_MODULO) ||
        !sim_ir_type_is_scalar(rhs_type) ||
        !sim_scalar_domain_supports(rhs_domain, SIM_SCALAR_CAP_MODULO)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                  = &builder->nodes[id];
    node->type            = SIM_IR_NODE_MOD;
    node->id              = id;
    node->value_type      = lhs_type;
    node->data.binary.lhs = lhs;
    node->data.binary.rhs = rhs;
    node->is_local        = builder->nodes[lhs].is_local && builder->nodes[rhs].is_local;
    return id;
}

SimIRNodeId sim_ir_builder_coord(SimIRBuilder* builder, size_t field_id, size_t axis) {
    SimIRNodeId id;
    SimIRNode*  node;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                   = &builder->nodes[id];
    node->type             = SIM_IR_NODE_COORD;
    node->id               = id;
    node->value_type       = sim_ir_type_scalar_domain_typed(sim_scalar_domain_u64());
    node->is_local         = true;
    node->data.coord.field = field_id;
    node->data.coord.axis  = axis;
    return id;
}

SimIRNodeId sim_ir_builder_field_ref_typed(SimIRBuilder* builder, size_t field_id, SimIRType type) {
    SimIRNodeId id;
    SimIRNode*  node;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node             = &builder->nodes[id];
    node->type       = SIM_IR_NODE_FIELD_REF;
    node->id         = id;
    node->value_type = sim_ir_type_normalize(type);
    node->is_local   = true;
    node->data.field = field_id;
    return id;
}

SimIRNodeId sim_ir_builder_field_ref(SimIRBuilder* builder, size_t field_id) {
    return sim_ir_builder_field_ref_typed(builder, field_id, sim_ir_type_scalar());
}

SimIRNodeId
sim_ir_builder_binary(SimIRBuilder* builder, SimIRNodeType type, SimIRNodeId lhs, SimIRNodeId rhs) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   value_type;
    SimScalarDomain value_domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (!(type == SIM_IR_NODE_ADD || type == SIM_IR_NODE_SUB || type == SIM_IR_NODE_MUL ||
          type == SIM_IR_NODE_DIV || type == SIM_IR_NODE_POW)) {
        return SIM_IR_INVALID_NODE;
    }

    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE || lhs >= builder->count ||
        rhs >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    if (!sim_ir_type_equal(builder->nodes[lhs].value_type, builder->nodes[rhs].value_type)) {
        return SIM_IR_INVALID_NODE;
    }

    value_type   = sim_ir_type_normalize(builder->nodes[lhs].value_type);
    value_domain = sim_ir_type_scalar_domain(value_type);
    if (((type == SIM_IR_NODE_ADD || type == SIM_IR_NODE_SUB) &&
         !sim_scalar_domain_supports(value_domain, SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC)) ||
        (type == SIM_IR_NODE_MUL &&
         !sim_scalar_domain_supports(value_domain, SIM_SCALAR_CAP_MULTIPLICATIVE_ARITHMETIC)) ||
        (type == SIM_IR_NODE_DIV &&
         !sim_scalar_domain_supports(value_domain, SIM_SCALAR_CAP_DIVISION)) ||
        (type == SIM_IR_NODE_POW &&
         !sim_scalar_domain_supports(value_domain, SIM_SCALAR_CAP_POWER))) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                  = &builder->nodes[id];
    node->type            = type;
    node->id              = id;
    node->value_type      = value_type;
    node->data.binary.lhs = lhs;
    node->data.binary.rhs = rhs;
    node->is_local        = builder->nodes[lhs].is_local && builder->nodes[rhs].is_local;
    return id;
}

SimIRNodeId sim_ir_builder_pow(SimIRBuilder* builder, SimIRNodeId lhs, SimIRNodeId rhs) {
    return sim_ir_builder_binary(builder, SIM_IR_NODE_POW, lhs, rhs);
}

SimIRNodeId sim_ir_builder_diff_spec(SimIRBuilder* builder, const SimIRDiffSpec* spec) {
    SimIRNodeId         id;
    SimIRNode*          node;
    SimIRDiffSpec       local_spec;
    SimIRType           operand_type;
    SimIRType           result_type;
    SimIRBoundaryPolicy fallback_boundary = SIM_IR_BOUNDARY_NEUMANN;

    if (builder == NULL || spec == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (spec->operand == SIM_IR_INVALID_NODE || spec->operand >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    if (spec->dx <= 0.0) {
        return SIM_IR_INVALID_NODE;
    }

    local_spec = *spec;
    if (local_spec.order == 0U) {
        local_spec.order = 1U;
    }
    if (local_spec.stencil_order == 0U) {
        local_spec.stencil_order = local_spec.order;
    }
    switch (local_spec.method) {
        case SIM_IR_DIFF_METHOD_AUTO:
        case SIM_IR_DIFF_METHOD_CENTRAL:
        case SIM_IR_DIFF_METHOD_FORWARD:
        case SIM_IR_DIFF_METHOD_BACKWARD:
            break;
        default:
            local_spec.method = SIM_IR_DIFF_METHOD_AUTO;
            break;
    }
    switch (local_spec.boundary) {
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_PERIODIC:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            break;
        default:
            break;
    }
    if (builder != NULL) {
        fallback_boundary = builder->default_boundary;
    }
    if (local_spec.boundary < SIM_IR_BOUNDARY_NEUMANN ||
        local_spec.boundary > SIM_IR_BOUNDARY_REFLECTIVE) {
        local_spec.boundary = fallback_boundary;
    }
    if (!isfinite(local_spec.consistency_constant) || local_spec.consistency_constant < 0.0) {
        local_spec.consistency_constant = 0.0;
    }
    operand_type = builder->nodes[local_spec.operand].value_type;
    result_type  = local_spec.result_type.components == 0U ? operand_type : local_spec.result_type;
    result_type  = sim_ir_type_normalize(result_type);

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                                 = &builder->nodes[id];
    node->type                           = SIM_IR_NODE_DIFF;
    node->id                             = id;
    node->value_type                     = result_type;
    node->data.diff.operand              = local_spec.operand;
    node->data.diff.axis                 = local_spec.axis;
    node->data.diff.dx                   = local_spec.dx;
    node->data.diff.scale                = local_spec.scale;
    node->data.diff.order                = local_spec.order;
    node->data.diff.stencil_order        = local_spec.stencil_order;
    node->data.diff.method               = local_spec.method;
    node->data.diff.consistency_constant = local_spec.consistency_constant;
    node->data.diff.boundary             = local_spec.boundary;
    node->is_local                       = false;
    return id;
}

SimIRNodeId sim_ir_builder_diff(SimIRBuilder* builder,
                                SimIRNodeId   operand,
                                size_t        axis,
                                double        dx,
                                double        scale) {
    SimIRDiffSpec spec;
    SimIRType     operand_type;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (operand == SIM_IR_INVALID_NODE || operand >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    operand_type              = builder->nodes[operand].value_type;
    spec.operand              = operand;
    spec.axis                 = axis;
    spec.dx                   = dx;
    spec.scale                = scale;
    spec.order                = 1U;
    spec.stencil_order        = 1U;
    spec.method               = SIM_IR_DIFF_METHOD_AUTO;
    spec.consistency_constant = 0.0;
    spec.boundary             = builder->default_boundary;
    spec.result_type          = operand_type;
    return sim_ir_builder_diff_spec(builder, &spec);
}
SimIRNodeId sim_ir_builder_noise_spec(SimIRBuilder* builder, const SimIRNoiseSpec* spec) {
    SimIRNodeId    id;
    SimIRNode*     node;
    SimIRNoiseSpec local_spec;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    local_spec.seed         = 0U;
    local_spec.amplitude    = 1.0;
    local_spec.variance     = 1.0;
    local_spec.law          = SIM_IR_NOISE_LAW_ITO;
    local_spec.distribution = SIM_IR_NOISE_DISTRIBUTION_UNIFORM;
    local_spec.value_type   = sim_ir_type_scalar();

    if (spec != NULL) {
        local_spec = *spec;
    }

    if (local_spec.amplitude < 0.0) {
        return SIM_IR_INVALID_NODE;
    }

    if (local_spec.variance < 0.0 || !isfinite(local_spec.variance)) {
        local_spec.variance = local_spec.amplitude * local_spec.amplitude;
    }

    local_spec.value_type = sim_ir_type_normalize(local_spec.value_type);

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                          = &builder->nodes[id];
    node->type                    = SIM_IR_NODE_NOISE;
    node->id                      = id;
    node->value_type              = local_spec.value_type;
    node->data.noise.seed         = local_spec.seed;
    node->data.noise.amplitude    = local_spec.amplitude;
    node->data.noise.variance     = local_spec.variance;
    node->data.noise.law          = local_spec.law;
    node->data.noise.distribution = local_spec.distribution;
    node->is_local                = true;
    return id;
}

SimIRNodeId sim_ir_builder_noise(SimIRBuilder* builder, uint32_t seed, double amplitude) {
    SimIRNoiseSpec spec;
    spec.seed         = seed;
    spec.amplitude    = amplitude;
    spec.variance     = amplitude * amplitude;
    spec.law          = SIM_IR_NOISE_LAW_ITO;
    spec.distribution = SIM_IR_NOISE_DISTRIBUTION_UNIFORM;
    spec.value_type   = sim_ir_type_scalar();
    return sim_ir_builder_noise_spec(builder, &spec);
}

SimIRNodeId sim_ir_builder_stateful_spec(SimIRBuilder* builder, const SimIRStatefulSpec* spec) {
    SimIRStatefulSpec local_spec;
    SimIRNodeId       id;
    SimIRNode*        node;

    if (builder == NULL || spec == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    local_spec = *spec;
    if (local_spec.eval == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRType result_type =
        local_spec.value_type.components == 0U ? sim_ir_type_scalar() : local_spec.value_type;
    result_type = sim_ir_type_normalize(result_type);

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                         = &builder->nodes[id];
    node->type                   = SIM_IR_NODE_STATEFUL;
    node->id                     = id;
    node->value_type             = result_type;
    node->is_local               = false;
    node->data.stateful.eval     = local_spec.eval;
    node->data.stateful.userdata = local_spec.userdata;
    node->data.stateful.label    = local_spec.label;
    return id;
}

SimIRNodeId sim_ir_builder_stateful(SimIRBuilder*       builder,
                                    SimIRStatefulEvalFn eval,
                                    void*               userdata,
                                    const char*         label) {
    SimIRStatefulSpec spec;
    spec.eval       = eval;
    spec.userdata   = userdata;
    spec.label      = label;
    spec.value_type = sim_ir_type_scalar();
    return sim_ir_builder_stateful_spec(builder, &spec);
}

SimIRNodeId sim_ir_builder_warp_spec(SimIRBuilder* builder, const SimIRWarpSpec* spec) {
    SimIRWarpSpec local_spec;
    SimIRNodeId   id;
    SimIRNode*    node;
    SimIRType     operand_type;
    SimIRType     result_type;

    if (builder == NULL || spec == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (spec->operand == SIM_IR_INVALID_NODE || spec->operand >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    local_spec = *spec;

    if (!isfinite(local_spec.bias)) {
        local_spec.bias = 0.0;
    }

    if (!isfinite(local_spec.lambda)) {
        local_spec.lambda = 1.0;
    }

    if (!isfinite(local_spec.delta)) {
        local_spec.delta = 1.0e-6;
    } else {
        local_spec.delta = fabs(local_spec.delta);
        if (local_spec.delta <= 0.0) {
            local_spec.delta = 1.0e-6;
        }
    }

    if (!(local_spec.profile == SIM_IR_WARP_PROFILE_DIGAMMA ||
          local_spec.profile == SIM_IR_WARP_PROFILE_TRIGAMMA ||
          local_spec.profile == SIM_IR_WARP_PROFILE_DIGAMMA_7_TAIL ||
          local_spec.profile == SIM_IR_WARP_PROFILE_DIGAMMA_5_TAIL ||
          local_spec.profile == SIM_IR_WARP_PROFILE_DIGAMMA_ADAPTIVE ||
          local_spec.profile == SIM_IR_WARP_PROFILE_DIGAMMA_MORTICI)) {
        local_spec.profile = SIM_IR_WARP_PROFILE_DIGAMMA;
    }

    if (!isfinite(local_spec.tolerance) || local_spec.tolerance <= 0.0) {
        local_spec.tolerance = 0.0;
    }

    local_spec.guard = sim_ir_guard_normalize(local_spec.guard);

    operand_type = builder->nodes[local_spec.operand].value_type;
    if (!sim_ir_type_is_scalar(operand_type)) {
        return SIM_IR_INVALID_NODE;
    }

    result_type = local_spec.result_type.components == 0U ? operand_type : local_spec.result_type;
    result_type = sim_ir_type_normalize(result_type);
    if (!sim_ir_type_is_scalar(result_type)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                      = &builder->nodes[id];
    node->type                = SIM_IR_NODE_WARP;
    node->id                  = id;
    node->value_type          = result_type;
    node->data.warp.operand   = local_spec.operand;
    node->data.warp.bias      = local_spec.bias;
    node->data.warp.delta     = local_spec.delta;
    node->data.warp.lambda    = local_spec.lambda;
    node->data.warp.tolerance = local_spec.tolerance;
    node->data.warp.profile   = local_spec.profile;
    node->data.warp.guard     = local_spec.guard;
    SimWarpLevel warp_class   = local_spec.warp_class;
    switch (warp_class) {
        case SIM_WARP_LEVEL_NONE:
        case SIM_WARP_LEVEL_LEVEL0:
        case SIM_WARP_LEVEL_LEVEL1:
        case SIM_WARP_LEVEL_LEVEL2:
            break;
        default:
            warp_class = SIM_WARP_LEVEL_NONE;
            break;
    }
    if (warp_class == SIM_WARP_LEVEL_NONE) {
        warp_class = sim_ir_warp_level_from_profile(local_spec.profile);
    }
    node->warp_class = warp_class;
    node->is_local   = true;
    return id;
}

SimIRNodeId sim_ir_builder_warp(SimIRBuilder*    builder,
                                SimIRNodeId      operand,
                                SimIRWarpProfile profile,
                                double           bias,
                                double           delta,
                                double           lambda) {
    SimIRWarpSpec spec;

    spec.operand         = operand;
    spec.bias            = bias;
    spec.delta           = delta;
    spec.lambda          = lambda;
    spec.tolerance       = 0.0;
    spec.profile         = profile;
    spec.warp_class      = SIM_WARP_LEVEL_NONE;
    spec.guard.mode      = (int) SIM_CONTINUITY_NONE;
    spec.guard.clamp_min = 0.0;
    spec.guard.clamp_max = 0.0;
    spec.guard.tolerance = 0.0;
    spec.result_type     = sim_ir_type_scalar();

    return sim_ir_builder_warp_spec(builder, &spec);
}

SimIRNodeId
sim_ir_builder_complex_rotate(SimIRBuilder* builder, SimIRNodeId operand, SimIRNodeId angle) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   operand_type;
    SimIRType   angle_type;
    SimScalarDomain operand_domain;
    SimScalarDomain angle_domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (operand == SIM_IR_INVALID_NODE || operand >= builder->count ||
        angle == SIM_IR_INVALID_NODE || angle >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    operand_type = sim_ir_type_normalize(builder->nodes[operand].value_type);
    angle_type   = sim_ir_type_normalize(builder->nodes[angle].value_type);
    operand_domain = sim_ir_type_scalar_domain(operand_type);
    angle_domain   = sim_ir_type_scalar_domain(angle_type);

    if (operand_type.components != 2U ||
        !sim_scalar_domain_supports(operand_domain, SIM_SCALAR_CAP_COMPLEX_ROTATION)) {
        return SIM_IR_INVALID_NODE;
    }

    if (!sim_ir_type_is_scalar(angle_type) ||
        !sim_scalar_domain_supports(angle_domain, SIM_SCALAR_CAP_ORDERING)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                              = &builder->nodes[id];
    node->type                        = SIM_IR_NODE_COMPLEX_ROTATE;
    node->id                          = id;
    node->value_type                  = operand_type;
    node->data.complex_rotate.operand = operand;
    node->data.complex_rotate.angle   = angle;
    node->is_local                    = builder->nodes[operand].is_local;
    return id;
}

SimIRNodeId sim_ir_builder_complex_pack(SimIRBuilder* builder, SimIRNodeId real, SimIRNodeId imag) {
    SimIRNodeId id;
    SimIRNode*  node;
    SimIRType   real_type;
    SimIRType   imag_type;
    SimScalarDomain real_domain;
    SimScalarDomain imag_domain;

    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (real == SIM_IR_INVALID_NODE || imag == SIM_IR_INVALID_NODE || real >= builder->count ||
        imag >= builder->count) {
        return SIM_IR_INVALID_NODE;
    }

    real_type = sim_ir_type_normalize(builder->nodes[real].value_type);
    imag_type = sim_ir_type_normalize(builder->nodes[imag].value_type);
    real_domain = sim_ir_type_scalar_domain(real_type);
    imag_domain = sim_ir_type_scalar_domain(imag_type);

    if (!sim_ir_type_is_scalar(real_type) || sim_scalar_domain_is_complex(real_domain) ||
        !sim_scalar_domain_supports(real_domain, SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC) ||
        !sim_ir_type_is_scalar(imag_type) || sim_scalar_domain_is_complex(imag_domain) ||
        !sim_scalar_domain_supports(imag_domain, SIM_SCALAR_CAP_ADDITIVE_ARITHMETIC)) {
        return SIM_IR_INVALID_NODE;
    }

    id = sim_ir_builder_append(builder);
    if (id == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    node                         = &builder->nodes[id];
    node->type                   = SIM_IR_NODE_COMPLEX_PACK;
    node->id                     = id;
    node->value_type             = sim_ir_type_complex();
    node->data.complex_pack.real = real;
    node->data.complex_pack.imag = imag;
    node->is_local               = builder->nodes[real].is_local && builder->nodes[imag].is_local;
    return id;
}
