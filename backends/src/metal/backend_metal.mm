/**
 * @file backend_metal.mm
 * @brief Metal backend skeleton that prepares device resources and shader cache.
 *
 * The Metal backend emits a float-based MSL pointwise kernel for a supported
 * KernelIR subset, caches compiled compute pipelines by generated source hash,
 * and keeps per-kernel Metal buffers for reused field, constant, and output
 * storage. Unsupported IR shapes return #SIM_RESULT_NOT_SUPPORTED so callers
 * can use the CPU backend.
 */

#include "oakfield/backend.h"

#ifdef SIM_HAVE_METAL

extern "C" {
SimResult sim_backend_metal_init(SimBackend *backend);
SimResult sim_backend_metal_launch(SimBackend *backend, KernelIR *kernel);
void sim_backend_metal_destroy(SimBackend *backend);
}

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "internal/backend_metal_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

/**
 * @brief Compute a stable FNV-1a hash for generated MSL source.
 */
static size_t backend_metal_hash_string(const std::string &text)
{
    /* 64-bit FNV-1a hash for stable pipeline source keys without libc++ internals. */
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : text)
    {
        hash ^= (uint64_t)ch;
        hash *= 1099511628211ULL;
    }
    return (size_t)hash;
}

/**
 * @brief Format a double as an MSL float literal.
 *
 * Integer-looking decimal text receives `.0f` so generated shader source keeps
 * float semantics.
 */
static std::string backend_metal_float_literal(double value)
{
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.17g", value);
    std::string literal(buffer);
    if (literal.find_first_of(".eE") == std::string::npos)
    {
        literal += ".0";
    }
    literal += "f";
    return literal;
}

/**
 * @brief Drop one reference from a cached Metal pipeline source hash.
 *
 * When the reference count reaches zero, the cached pipeline is released in
 * non-ARC builds and removed from the cache map.
 */
static void backend_metal_unrefer_pipeline(SimBackendMetalState *state, size_t src_hash)
{
    if (state == NULL) return;
    auto it = state->pipeline_cache_by_hash.find(src_hash);
    if (it == state->pipeline_cache_by_hash.end()) return;
    if (it->second.refcount > 0) it->second.refcount--;
    if (it->second.refcount == 0)
    {
#if !__has_feature(objc_arc)
        if (it->second.pipeline != nil) [it->second.pipeline release];
#endif
        state->pipeline_cache_by_hash.erase(it);
    }
}

/**
 * @brief Destroy Metal backend state and release pipeline/buffer resources.
 */
static void backend_metal_state_destroy(SimBackendMetalState *state)
{
    size_t i;

    if (state == NULL)
    {
        return;
    }

    for (i = 0; i < state->pipeline_count; ++i)
    {
#if !__has_feature(objc_arc)
    [state->pipelines[i].constants_buffer release];
#endif
    if (state->pipelines[i].field_buffers != NULL)
    {
        for (size_t j = 0; j < state->pipelines[i].field_buffer_count; ++j)
        {
            if (state->pipelines[i].field_buffers[j] != nil)
            {
#if !__has_feature(objc_arc)
                [state->pipelines[i].field_buffers[j] release];
#endif
                state->pipelines[i].field_buffers[j] = nil;
            }
        }
        free(state->pipelines[i].field_buffers);
        state->pipelines[i].field_buffers = NULL;
        state->pipelines[i].field_buffer_count = 0U;
    }
    state->pipelines[i].constants_buffer = nil;
    state->pipelines[i].constants_bytes = 0U;
    state->pipelines[i].kernel = NULL;
    /* Decrement the cached pipeline reference through the entry source hash. */
        /* Unreference cached pipeline if this entry referenced one. */
        if (state->pipelines[i].src_hash != 0)
        {
            backend_metal_unrefer_pipeline(state, state->pipelines[i].src_hash);
            state->pipelines[i].src_hash = 0;
        }
        state->pipelines[i].pipeline = nil;
    }

    free(state->pipelines);
    state->pipelines = NULL;
    state->pipeline_count = 0U;
    state->pipeline_capacity = 0U;

#if !__has_feature(objc_arc)
    [state->queue release];
    [state->device release];
#else
    state->queue = nil;
    state->device = nil;
#endif

    /* Destroy pipeline cache by hash. Release any remaining pipelines. */
    for (auto &kv : state->pipeline_cache_by_hash)
    {
#if !__has_feature(objc_arc)
    if (kv.second.pipeline != nil) [kv.second.pipeline release];
#endif
    (void)kv.second.pipeline; /* Silence unused in ARC. */
    }
    state->pipeline_cache_by_hash.clear();

    delete state;
}

/**
 * @brief Add one reference to an existing cached Metal pipeline source hash.
 */
static void backend_metal_refer_pipeline(SimBackendMetalState *state, size_t src_hash)
{
    if (state == NULL) return;
    auto it = state->pipeline_cache_by_hash.find(src_hash);
    if (it == state->pipeline_cache_by_hash.end()) return;
    it->second.refcount++;
}

/* Debug helpers are implemented in backend_metal_debug.mm. */

/**
 * @brief Locate a field binding by KernelIR field index.
 */
static SimField *backend_metal_find_field(const KernelIR *kernel, size_t field_index)
{
    if (kernel == NULL || kernel->bindings == NULL)
    {
        return NULL;
    }
    for (size_t i = 0; i < kernel->binding_count; ++i)
    {
        if (kernel->bindings[i].field_index == field_index)
        {
            return kernel->bindings[i].field;
        }
    }
    return NULL;
}

/**
 * @brief Find or allocate a per-kernel Metal cache entry.
 *
 * The returned entry is owned by the backend state and persists until backend
 * destruction.
 */
static SimBackendMetalPipeline *
backend_metal_find_or_create_pipeline_entry(SimBackendMetalState *state, const KernelIR *kernel)
{
    if (state == NULL || kernel == NULL)
    {
        return NULL;
    }

    for (size_t i = 0; i < state->pipeline_count; ++i)
    {
        if (state->pipelines[i].kernel == kernel)
        {
            return &state->pipelines[i];
        }
    }

    if (state->pipeline_count >= state->pipeline_capacity)
    {
        size_t new_capacity = (state->pipeline_capacity > 0U) ? (state->pipeline_capacity * 2U) : 8U;
        SimBackendMetalPipeline *new_entries = (SimBackendMetalPipeline *)realloc(
            state->pipelines, new_capacity * sizeof(SimBackendMetalPipeline));
        if (new_entries == NULL)
        {
            return NULL;
        }
        if (new_capacity > state->pipeline_capacity)
        {
            size_t old_capacity = state->pipeline_capacity;
            memset(new_entries + old_capacity,
                   0,
                   (new_capacity - old_capacity) * sizeof(SimBackendMetalPipeline));
        }
        state->pipelines = new_entries;
        state->pipeline_capacity = new_capacity;
    }

    SimBackendMetalPipeline *entry = &state->pipelines[state->pipeline_count++];
    memset(entry, 0, sizeof(*entry));
    entry->kernel = kernel;
    entry->pipeline = nil;
    entry->constants_buffer = nil;
    entry->field_buffers = NULL;
    entry->out_buffer = nil;
    entry->src_hash = 0U;
    return entry;
}

/**
 * @brief Initialize a Metal backend instance.
 *
 * @param backend Backend handle to initialize.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_FOUND,
 * or #SIM_RESULT_OUT_OF_MEMORY.
 */
extern "C" SimResult sim_backend_metal_init(SimBackend *backend)
{
    SimBackendMetalState *state;
    id<MTLDevice> device;

    if (backend == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);

    device = MTLCreateSystemDefaultDevice();
    if (device == nil)
    {
        return SIM_RESULT_NOT_FOUND;
    }

    state = new (std::nothrow) SimBackendMetalState{};
    if (state == NULL)
    {
#if !__has_feature(objc_arc)
        [device release];
#endif
        return SIM_RESULT_OUT_OF_MEMORY;
    }

#if __has_feature(objc_arc)
    state->device = device;
    state->queue = [state->device newCommandQueue];
#else
    state->device = [device retain];
    state->queue = [state->device newCommandQueue];
    [device release];
#endif

    if (state->queue == nil)
    {
        backend_metal_state_destroy(state);
        return SIM_RESULT_NOT_FOUND;
    }

    backend->impl = state;
    backend->type = SIM_BACKEND_TYPE_METAL;
    backend_enable_feature(backend, SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS);
    return SIM_RESULT_OK;
}

/**
 * @brief Launch a supported KernelIR pointwise kernel through Metal.
 *
 * The generated MSL path uses 32-bit float buffers for compatibility with
 * Apple GPU execution. Host double data is converted to float before dispatch
 * and converted back to double after completion. Integer-domain and unsupported
 * stateful kernels are rejected with #SIM_RESULT_NOT_SUPPORTED.
 *
 * @param backend Initialized Metal backend.
 * @param kernel Kernel package to execute.
 * @return #SIM_RESULT_OK or an error describing unsupported IR, missing fields,
 * allocation failure, shader compilation failure, or command execution failure.
 */
extern "C" SimResult sim_backend_metal_launch(SimBackend *backend, KernelIR *kernel)
{
    SimBackendMetalState *state;

    if (backend == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimBackendMetalState *)backend->impl;
    if (state == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (kernel == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* Locate or create a persistent per-kernel entry to reuse pipeline and buffers. */
    SimBackendMetalPipeline *entry = backend_metal_find_or_create_pipeline_entry(state, kernel);
    if (entry == NULL)
    {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

        /*
         * Attempt specialized pointwise kernel emission. For Apple GPU
         * compatibility this path uses 32-bit floats and converts double host
         * data to float before uploading.
         */
        if (kernel == NULL || kernel->builder == NULL || kernel->output_count != 1U)
        {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        const SimIRBuilder *builder = kernel->builder;
        SimIRNodeId root_id = kernel->outputs[0].expression;
        if (root_id == SIM_IR_INVALID_NODE || root_id >= builder->count)
        {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        const SimIRNode *root = &builder->nodes[root_id];
        if (sim_scalar_domain_is_integer(sim_ir_type_scalar_domain(root->value_type)))
        {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        if (sim_ir_kernel_has_unsupported_complex_semantics(kernel))
        {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        /* Validate output field and element counts */
        const SimKernelIROutput *output = &kernel->outputs[0];
        SimField *out_field = backend_metal_find_field(kernel, output->field_index);
        if (out_field == nil)
        {
            return SIM_RESULT_NOT_FOUND;
        }
        if (sim_scalar_domain_is_integer(sim_scalar_domain_from_field(out_field)))
        {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        size_t comps = out_field->element_size / sizeof(double);
        size_t element_count = sim_field_bytes(out_field) / out_field->element_size;
        if (element_count == 0U || comps == 0U)
        {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        id<MTLBuffer> out_buf = nil;

        /*
         * Build an MSL expression from the supported KernelIR subset. This is
         * similar to CUDA expression generation but emits float expressions and
         * Metal buffer indexing.
         */
        std::set<size_t> field_refs;
        std::map<size_t, size_t> pool_const_offsets;
        std::vector<float> scalar_consts;
        bool supported_nodes = true;

        std::function<void(SimIRNodeId)> collect = [&](SimIRNodeId nid) {
            if (nid == SIM_IR_INVALID_NODE || nid >= builder->count) return;
            const SimIRNode *n = &builder->nodes[nid];
            switch (n->type)
            {
            case SIM_IR_NODE_FIELD_REF:
                field_refs.insert(n->data.field);
                break;
            case SIM_IR_NODE_CONSTANT:
                if (n->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX)
                {
                    size_t idx = n->data.constant.constant_index;
                    pool_const_offsets[idx] = builder->constants_offsets[idx];
                }
                else if (n->value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY)
                {
                    /* nothing to collect */
                }
                else
                {
                    scalar_consts.push_back((float)n->data.constant.scalar);
                }
                break;
            case SIM_IR_NODE_ADD:
            case SIM_IR_NODE_SUB:
            case SIM_IR_NODE_MUL:
            case SIM_IR_NODE_DIV:
            case SIM_IR_NODE_POW:
                collect(n->data.binary.lhs);
                collect(n->data.binary.rhs);
                break;
            case SIM_IR_NODE_DIFF:
                collect(n->data.diff.operand);
                break;
            case SIM_IR_NODE_CALL:
                collect(n->data.call.operand);
                break;
            case SIM_IR_NODE_FLOOR:
                collect(n->data.unary.operand);
                break;
            case SIM_IR_NODE_MOD:
                collect(n->data.binary.lhs);
                collect(n->data.binary.rhs);
                break;
            case SIM_IR_NODE_WARP:
                collect(n->data.warp.operand);
                break;
            case SIM_IR_NODE_COMPLEX_ROTATE:
                collect(n->data.complex_rotate.operand);
                collect(n->data.complex_rotate.angle);
                break;
            case SIM_IR_NODE_COMPLEX_PACK:
                collect(n->data.complex_pack.real);
                collect(n->data.complex_pack.imag);
                break;
            case SIM_IR_NODE_PARAM:
            case SIM_IR_NODE_INDEX:
            case SIM_IR_NODE_COORD:
            case SIM_IR_NODE_NOISE:
                /* leaf nodes */
                break;
            case SIM_IR_NODE_STATEFUL:
                supported_nodes = false;
                break;
            default:
                supported_nodes = false;
                break;
            }
        };
        collect(root_id);
        if (!supported_nodes)
        {
            return SIM_RESULT_NOT_SUPPORTED;
        }

        size_t out_bytes = element_count * comps * sizeof(float);
        id<MTLBuffer> out_buf_local = nil;
        if (entry != NULL && entry->out_buffer != nil && entry->out_buffer_bytes == out_bytes)
        {
            out_buf_local = entry->out_buffer;
        }
        else
        {
            out_buf_local = [state->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            if (out_buf_local == nil)
            {
                /* Fall back to returning allocation failure if buffer creation fails. */
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            if (entry != NULL)
            {
                if (entry->out_buffer != nil)
                {
#if !__has_feature(objc_arc)
                    [entry->out_buffer release];
#endif
                    entry->out_buffer = nil;
                    entry->out_buffer_bytes = 0U;
                }
                entry->out_buffer = out_buf_local;
                entry->out_buffer_bytes = out_bytes;
            }
        }
        out_buf = out_buf_local;

    std::map<size_t,int> field_arg_idx;
    int idx = 0;
    for (auto fid : field_refs) field_arg_idx[fid] = idx++;
    std::vector<size_t> fields_by_slot((int)field_arg_idx.size(), 0);
    for (auto &kv : field_arg_idx) fields_by_slot[kv.second] = kv.first;
    std::map<size_t,const SimKernelIRBinding*> binding_by_field;
    if (kernel->bindings != NULL)
    {
        for (size_t i = 0; i < kernel->binding_count; ++i)
        {
            const SimKernelIRBinding *b = &kernel->bindings[i];
            binding_by_field[b->field_index] = b;
        }
    }

        const bool has_constants = (kernel->builder != NULL && kernel->builder->constants_count > 0U);
        const bool has_params    = (kernel->params != NULL && kernel->param_count > 0U);

        std::function<std::string(SimIRNodeId, const std::string &, const std::string &)> emit_expr =
            [&](SimIRNodeId nid, const std::string &idx_expr, const std::string &comp_expr) -> std::string {
                const SimIRNode *n = &builder->nodes[nid];
                switch (n->type)
                {
                case SIM_IR_NODE_FIELD_REF:
                {
                    int fi = field_arg_idx[n->data.field];
                    char b[96];
                    snprintf(b, sizeof(b), "f%d[%s*comps + (%s)]", fi, idx_expr.c_str(), comp_expr.c_str());
                    return std::string(b);
                }
                case SIM_IR_NODE_CONSTANT:
                {
                    if (n->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX)
                    {
                        size_t off = builder->constants_offsets[n->data.constant.constant_index];
                        char b[96];
                        snprintf(b, sizeof(b), "(constants[%zu + (%s)])", off, comp_expr.c_str());
                        return std::string(b);
                    }
                    if (n->value_type.components <= 1U)
                    {
                        /* Scalar constants live in constant.scalar, not in the inline lane array. */
                        return backend_metal_float_literal((double)n->data.constant.scalar);
                    }
                    if (n->value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY)
                    {
                        size_t component_count = n->value_type.components;
                        if (component_count == 0U)
                        {
                            return std::string("0.0f");
                        }
                        std::string s;
                        for (size_t k = 0; k < component_count; ++k)
                        {
                            s += "((" + comp_expr + ")==" + std::to_string(k) + "u?";
                            s += backend_metal_float_literal((double)n->data.constant.small[k]);
                            s += ":";
                        }
                        s += "0.0f";
                        for (size_t k = 0; k < component_count; ++k)
                        {
                            s += ")";
                        }
                        return s;
                    }
                    return backend_metal_float_literal((double)n->data.constant.scalar);
                }
                case SIM_IR_NODE_PARAM:
                    if (has_params && n->data.param.param < kernel->param_count)
                    {
                        return "params[" + std::to_string((size_t)n->data.param.param) + "]";
                    }
                    return std::string("0.0f");
                case SIM_IR_NODE_INDEX:
                    return "(float)(" + idx_expr + ")";
                case SIM_IR_NODE_CALL:
                {
                    std::string operand = emit_expr(n->data.call.operand, idx_expr, comp_expr);
                    switch (n->data.call.kind)
                    {
                    case SIM_IR_CALL_SIN:  return "sin(" + operand + ")";
                    case SIM_IR_CALL_COS:  return "cos(" + operand + ")";
                    case SIM_IR_CALL_EXP:  return "exp(" + operand + ")";
                    case SIM_IR_CALL_ABS:  return "fabs(" + operand + ")";
                    case SIM_IR_CALL_LOG:  return "log(" + operand + ")";
                    case SIM_IR_CALL_TANH: return "tanh(" + operand + ")";
                    case SIM_IR_CALL_SIGN: return "copysign(1.0f, " + operand + ")";
                    default:               return std::string("0.0f");
                    }
                }
                case SIM_IR_NODE_FLOOR:
                    return "floor(" + emit_expr(n->data.unary.operand, idx_expr, comp_expr) + ")";
                case SIM_IR_NODE_COORD:
                {
                    size_t extent = 0U;
                    size_t stride = 0U;
                    auto it = binding_by_field.find(n->data.coord.field);
                    if (it != binding_by_field.end())
                    {
                        const SimKernelIRBinding *binding = it->second;
                        if (binding != NULL && binding->shape != NULL && binding->strides != NULL &&
                            n->data.coord.axis < binding->rank)
                        {
                            extent = binding->shape[n->data.coord.axis];
                            stride = binding->strides[n->data.coord.axis];
                        }
                    }
                    if (extent == 0U || stride == 0U)
                    {
                        if (n->data.coord.axis == 0U)
                        {
                            return "(float)(" + idx_expr + ")";
                        }
                        return std::string("0.0f");
                    }
                    return "(float)(((" + idx_expr + " / " + std::to_string(stride) + "u) % " +
                           std::to_string(extent) + "u))";
                }
                case SIM_IR_NODE_ADD:
                    return "(" + emit_expr(n->data.binary.lhs, idx_expr, comp_expr) + ") + (" +
                           emit_expr(n->data.binary.rhs, idx_expr, comp_expr) + ")";
                case SIM_IR_NODE_SUB:
                    return "(" + emit_expr(n->data.binary.lhs, idx_expr, comp_expr) + ") - (" +
                           emit_expr(n->data.binary.rhs, idx_expr, comp_expr) + ")";
                case SIM_IR_NODE_MUL:
                    return "(" + emit_expr(n->data.binary.lhs, idx_expr, comp_expr) + ") * (" +
                           emit_expr(n->data.binary.rhs, idx_expr, comp_expr) + ")";
                case SIM_IR_NODE_DIV:
                    return "(" + emit_expr(n->data.binary.lhs, idx_expr, comp_expr) + ") / (" +
                           emit_expr(n->data.binary.rhs, idx_expr, comp_expr) + ")";
                case SIM_IR_NODE_POW:
                    return "pow((" + emit_expr(n->data.binary.lhs, idx_expr, comp_expr) + "), (" +
                           emit_expr(n->data.binary.rhs, idx_expr, comp_expr) + "))";
                case SIM_IR_NODE_MOD:
                {
                    std::string lhs = emit_expr(n->data.binary.lhs, idx_expr, comp_expr);
                    std::string rhs = emit_expr(n->data.binary.rhs, idx_expr, comp_expr);
                    return "((" + rhs + ")==0.0f ? 0.0f : fmod((" + lhs + "), (" + rhs + ")))";
                }
                case SIM_IR_NODE_COMPLEX_PACK:
                {
                    std::string re = emit_expr(n->data.complex_pack.real, idx_expr, "0u");
                    std::string im = emit_expr(n->data.complex_pack.imag, idx_expr, "0u");
                    return "((" + comp_expr + ")==0u ? (" + re + ") : (" + im + "))";
                }
                case SIM_IR_NODE_COMPLEX_ROTATE:
                {
                    std::string re    = emit_expr(n->data.complex_rotate.operand, idx_expr, "0u");
                    std::string im    = emit_expr(n->data.complex_rotate.operand, idx_expr, "1u");
                    std::string theta = emit_expr(n->data.complex_rotate.angle, idx_expr, "0u");
                    std::string s     = "sin(" + theta + ")";
                    std::string cval  = "cos(" + theta + ")";
                    return "((" + comp_expr + ")==0u ? ((" + re + ")*(" + cval + ") - (" + im + ")*(" + s +
                           ")) : ((" + re + ")*(" + s + ") + (" + im + ")*(" + cval + ")))";
                }
                case SIM_IR_NODE_WARP:
                {
                    std::string operand = emit_expr(n->data.warp.operand, idx_expr, comp_expr);
                    return "(" + backend_metal_float_literal((double)n->data.warp.lambda) + " * warp_diff_f(" +
                           std::to_string((int)n->data.warp.profile) + ", (" + operand + " + " +
                           backend_metal_float_literal((double)n->data.warp.bias) + "), " +
                           backend_metal_float_literal((double)n->data.warp.delta) + "))";
                }
                case SIM_IR_NODE_DIFF:
                {
                    /* Emit finite difference with boundary policy awareness for 1D layouts. */
                    size_t axis = n->data.diff.axis;
                    size_t extent = 0U;
                    size_t stride = 1U;
                    if (n->data.diff.operand < builder->count)
                    {
                        const SimIRNode *opnode = &builder->nodes[n->data.diff.operand];
                        if (opnode->type == SIM_IR_NODE_FIELD_REF)
                        {
                            auto it_binding = binding_by_field.find(opnode->data.field);
                            if (it_binding != binding_by_field.end())
                            {
                                const SimKernelIRBinding *binding = it_binding->second;
                                if (binding != NULL && binding->shape != NULL && binding->strides != NULL &&
                                    axis < binding->rank)
                                {
                                    extent = binding->shape[axis];
                                    stride = binding->strides[axis];
                                }
                            }
                        }
                    }

                    if (extent == 0U || stride == 0U)
                    {
                        extent = 0U;
                        stride = 1U;
                    }

                    std::string coord = "(" + idx_expr + " / " + std::to_string(stride) + "u)";
                    std::string coord_mod =
                        "(" + coord + " % " + std::to_string((extent > 0U) ? extent : 1U) + "u)";
                    std::string center_expr = emit_expr(n->data.diff.operand, idx_expr, comp_expr);
                    std::string forward_idx = idx_expr;
                    std::string backward_idx = idx_expr;
                    std::string has_forward =
                        (extent > 0U) ? "((" + coord_mod + " + 1u) < " + std::to_string(extent) + "u)" : "false";
                    std::string has_backward = (extent > 0U) ? "(" + coord_mod + " > 0u)" : "false";

                    switch (n->data.diff.boundary)
                    {
                    case SIM_IR_BOUNDARY_PERIODIC:
                        if (extent > 0U)
                        {
                            has_forward = (extent > 1U) ? "true" : "false";
                            has_backward = (extent > 1U) ? "true" : "false";
                            forward_idx = "((" + coord_mod + " + 1u < " + std::to_string(extent) +
                                          "u) ? (" + idx_expr + " + " + std::to_string(stride) + "u) : (" + idx_expr +
                                          " - (" + std::to_string(extent) + "u - 1u) * " + std::to_string(stride) +
                                          "u))";
                            backward_idx = "(" + coord_mod + " > 0u ? (" + idx_expr + " - " + std::to_string(stride) +
                                           "u) : (" + idx_expr + " + (" + std::to_string(extent) + "u - 1u) * " +
                                           std::to_string(stride) + "u))";
                        }
                        break;
                    case SIM_IR_BOUNDARY_REFLECTIVE:
                        if (extent > 0U)
                        {
                            has_forward = (extent > 1U) ? "true" : "false";
                            has_backward = (extent > 1U) ? "true" : "false";
                            forward_idx = "((" + coord_mod + " + 1u < " + std::to_string(extent) +
                                          "u) ? (" + idx_expr + " + " + std::to_string(stride) + "u) : (" + idx_expr +
                                          " >= " + std::to_string(stride) + "u ? " + idx_expr + " - " +
                                          std::to_string(stride) + "u : " + idx_expr + "))";
                            backward_idx = "(" + coord_mod + " > 0u ? (" + idx_expr + " - " + std::to_string(stride) +
                                           "u) : (" + idx_expr + " + " + std::to_string(stride) + "u))";
                        }
                        break;
                    case SIM_IR_BOUNDARY_DIRICHLET:
                    case SIM_IR_BOUNDARY_NEUMANN:
                    default:
                        if (extent > 0U)
                        {
                            forward_idx = "(" + idx_expr + " + " + std::to_string(stride) + "u)";
                            backward_idx = "(" + idx_expr + " >= " + std::to_string(stride) + "u ? " + idx_expr + " - " +
                                           std::to_string(stride) + "u : " + idx_expr + ")";
                        }
                        break;
                    }

                    std::string forward_expr = emit_expr(n->data.diff.operand, forward_idx, comp_expr);
                    std::string backward_expr = emit_expr(n->data.diff.operand, backward_idx, comp_expr);

                    switch (n->data.diff.boundary)
                    {
                    case SIM_IR_BOUNDARY_DIRICHLET:
                        forward_expr = "(" + has_forward + " ? (" + forward_expr + ") : 0.0f)";
                        backward_expr = "(" + has_backward + " ? (" + backward_expr + ") : 0.0f)";
                        break;
                    case SIM_IR_BOUNDARY_NEUMANN:
                        forward_expr = "(" + has_forward + " ? (" + forward_expr + ") : (" + center_expr + "))";
                        backward_expr = "(" + has_backward + " ? (" + backward_expr + ") : (" + center_expr + "))";
                        break;
                    case SIM_IR_BOUNDARY_PERIODIC:
                    case SIM_IR_BOUNDARY_REFLECTIVE:
                    default:
                        break;
                    }

                    std::string dx_str = std::to_string((double)n->data.diff.dx);
                    std::string scale_str = std::to_string((double)n->data.diff.scale);
                    std::string derivative;

                    switch (n->data.diff.method)
                    {
                    case SIM_IR_DIFF_METHOD_FORWARD:
                        derivative = "((" + forward_expr + ") - (" + center_expr + ")) / (" + dx_str + "f)";
                        break;
                    case SIM_IR_DIFF_METHOD_BACKWARD:
                        derivative = "((" + center_expr + ") - (" + backward_expr + ")) / (" + dx_str + "f)";
                        break;
                    case SIM_IR_DIFF_METHOD_CENTRAL:
                        derivative = "((" + forward_expr + ") - (" + backward_expr + ")) / (2.0f * " + dx_str + "f)";
                        break;
                    case SIM_IR_DIFF_METHOD_AUTO:
                    default:
                        derivative =
                            "((" + has_forward + " && " + has_backward + ") ? ((" + forward_expr + ") - (" +
                            backward_expr + ")) / (2.0f * " + dx_str + "f) : (" + has_forward + " ? ((" + forward_expr +
                            ") - (" + center_expr + ")) / (" + dx_str + "f) : (" + has_backward + " ? ((" + center_expr +
                            ") - (" + backward_expr + ")) / (" + dx_str + "f) : 0.0f)))";
                        break;
                    }

                    return "(" + derivative + " * (" + scale_str + "f))";
                }
                case SIM_IR_NODE_NOISE:
                {
                    uint32_t seed = n->data.noise.seed;
                    double amp = n->data.noise.amplitude;
                    std::string amp_lit = backend_metal_float_literal((double)amp);
                    char buf[384];
                    snprintf(buf, sizeof(buf),
                             "(((((float)((((uint)(%s)*1664525u + %uu) ^ (((uint)(%s)*1664525u + %uu) >> 16)) & 0xFFFFFFu))"
                             " / 16777216.0f) * 2.0f - 1.0f) * %s)",
                             idx_expr.c_str(),
                             seed,
                             idx_expr.c_str(),
                             seed,
                             amp_lit.c_str());
                    return std::string(buf);
                }
                case SIM_IR_NODE_STATEFUL:
                default:
                    return std::string("0.0f");
                }
            };

        std::string expr = emit_expr(root_id, std::string("gid"), std::string("c"));

        /* Compose MSL kernel source, including helper functions used by warp nodes. */
    std::string msl_helpers = "#include <metal_stdlib>\nusing namespace metal;\n"
                "float digamma_f(float x) {\n"
                "    if (!isfinite(x)) return NAN;\n"
                                "    if (x <= 0.0f) { float frac = x - floor(x); if (fabs(frac) < 1e-6f) return NAN; return digamma_f(1.0f - x) - 3.14159265358979323846f * cos(3.14159265358979323846f * x) / sin(3.14159265358979323846f * x); }\n"
                "    float value = x; float result = 0.0f; while (value < 8.0f) { result -= 1.0f / value; value += 1.0f; } float inv = 1.0f / value; float inv2 = inv * inv; float series = inv2 * (-1.0f/12.0f + inv2*(1.0f/120.0f + inv2*(-1.0f/252.0f + inv2*(1.0f/240.0f + inv2*(-1.0f/132.0f + inv2*(691.0f/32760.0f)))))); result += log(value) - 0.5f*inv + series; return result; }\n"
                "float trigamma_f(float x) {\n"
                "    if (!isfinite(x)) return NAN;\n"
                "    if (x <= 0.0f) { float csc_sq = 1.0f / (sin(3.14159265358979323846f * x) * sin(3.14159265358979323846f * x)); if (!isfinite(csc_sq)) return NAN; return csc_sq - trigamma_f(1.0f - x); }\n"
                "    float value = x; float result = 0.0f; while (value < 8.0f) { result += 1.0f / (value * value); value += 1.0f; } float inv = 1.0f / value; float inv2 = inv * inv; float inv3 = inv2 * inv; float inv5 = inv3 * inv2; float inv7 = inv5 * inv2; float inv9 = inv7 * inv2; float inv11 = inv9 * inv2; float inv13 = inv11 * inv2; result += inv + 0.5f * inv2 + inv3/6.0f - inv5/30.0f + inv7/42.0f - inv9/30.0f + (5.0f*inv11)/66.0f - (691.0f*inv13)/2730.0f; return result; }\n"
                "float warp_diff_f(int prof, float sample, float delta) { if (prof == 1) { return trigamma_f(sample + delta) - trigamma_f(sample - delta); } else { return digamma_f(sample + delta) - digamma_f(sample - delta); } }\n";
    std::string msl = msl_helpers + "kernel void kernel_func(";
    for (size_t slot = 0; slot < fields_by_slot.size(); ++slot)
    {
        msl += "device const float* f" + std::to_string(slot) + ", ";
    }
    if (has_constants)
    {
        msl += "device const float* constants, ";
    }
    if (has_params)
    {
        msl += "device const float* params, ";
    }
    msl += "device float* out, uint gid [[thread_position_in_grid]]) { const uint count = " +
           std::to_string((uint32_t)element_count) + "u; const uint comps = " +
           std::to_string((uint32_t)comps) +
           "u; if (gid >= count) return; for (uint c = 0; c < comps; ++c) { float r = ";
    msl += expr + "; out[gid*comps + c] = r; } }\n";

        NSError *error = nil;
        /* Compute source hash for MSL to support pipeline reuse across identical sources. */
        size_t src_hash = backend_metal_hash_string(msl);
        id<MTLComputePipelineState> compiled_pipeline = nil;
        auto it = state->pipeline_cache_by_hash.find(src_hash);
        if (it != state->pipeline_cache_by_hash.end())
        {
            compiled_pipeline = it->second.pipeline;
            /* If the entry already references a different pipeline, unref it first. */
            if (entry->src_hash != 0 && entry->src_hash != src_hash)
            {
                backend_metal_unrefer_pipeline(state, entry->src_hash);
            }
            /* Set entry pipeline if not already set and increment refcount. */
            if (entry->src_hash != src_hash)
            {
                entry->pipeline = compiled_pipeline;
                entry->src_hash = src_hash;
                backend_metal_refer_pipeline(state, src_hash);
            }
        }
        else
        {
            NSString *msl_src_str = [NSString stringWithUTF8String:msl.c_str()];
            id<MTLLibrary> lib = [state->device newLibraryWithSource:msl_src_str options:nil error:&error];
            if (lib == nil)
            {
                if (error) fprintf(stderr, "[DEBUG] Metal library compile error: %s\n", [[error localizedDescription] UTF8String]);
                return SIM_RESULT_INVALID_STATE;
            }
            id<MTLFunction> func = [lib newFunctionWithName:@"kernel_func"];
            if (func == nil)
            {
#if !__has_feature(objc_arc)
                [lib release];
#endif
                return SIM_RESULT_INVALID_STATE;
            }
            compiled_pipeline = [state->device newComputePipelineStateWithFunction:func error:&error];
#if !__has_feature(objc_arc)
            [func release];
            [lib release];
#endif
            if (entry->src_hash != 0 && entry->src_hash != src_hash)
            {
                backend_metal_unrefer_pipeline(state, entry->src_hash);
            }
            entry->pipeline = compiled_pipeline;
            entry->src_hash = src_hash;
            /* We just created a new cache entry with refcount 1 so no need to increment further. */
            if (compiled_pipeline == nil)
            {
                if (error) fprintf(stderr, "[DEBUG] Metal pipeline error: %s\n", [[error localizedDescription] UTF8String]);
                return SIM_RESULT_INVALID_STATE;
            }
            /* Store compiled pipeline in cache map with refcount initialized to 1. */
            state->pipeline_cache_by_hash[src_hash] = { compiled_pipeline, 1 };
        }
        if (compiled_pipeline == nil)
        {
            if (error) fprintf(stderr, "[DEBUG] Metal pipeline error: %s\n", [[error localizedDescription] UTF8String]);
            return SIM_RESULT_INVALID_STATE;
        }
        id<MTLComputePipelineState> pipeline = compiled_pipeline;

        /* Encode and dispatch: bind field buffers, constants, params, and output. */
        id<MTLCommandBuffer> cb = [state->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        /* build field buffer array in order of field_arg_idx mapping (slot order) */
        std::vector<id<MTLBuffer>> field_bufs(fields_by_slot.size(), nil);
        /* Reuse persistent host-to-device buffers per field slot when possible. */
        if (entry != NULL)
        {
            if (entry->field_buffers == NULL || entry->field_buffer_count != fields_by_slot.size())
            {
                /* allocate and initialize the array of field buffer pointers */
                if (entry->field_buffers != NULL)
                {
                    free(entry->field_buffers);
                }
                entry->field_buffer_count = fields_by_slot.size();
                entry->field_buffers = (id<MTLBuffer> *)calloc(entry->field_buffer_count, sizeof(id<MTLBuffer>));
                for (size_t i = 0; i < entry->field_buffer_count; ++i)
                    entry->field_buffers[i] = nil;
            }
        }
        for (size_t slot = 0; slot < fields_by_slot.size(); ++slot)
        {
            size_t fid = fields_by_slot[slot];
            SimField *f = backend_metal_find_field(kernel, fid);
            if (f == NULL)
            {
                /* Leave missing fields as nil; validation above should reject required fields. */
            }
            else
            {
                double *data = (double *)sim_field_data(f);
                size_t total = element_count * comps;
                size_t bytes = total * sizeof(float);
                if (entry->field_buffers != NULL && entry->field_buffers[slot] != nil)
                {
                    id<MTLBuffer> b = entry->field_buffers[slot];
                    if (b.length != bytes)
                    {
#if !__has_feature(objc_arc)
                        [b release];
#endif
                        b = [state->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                        if (b == nil)
                        {
                            return SIM_RESULT_OUT_OF_MEMORY;
                        }
                        entry->field_buffers[slot] = b;
                    }
                    float *dst = (float *)b.contents;
                    for (size_t ii = 0; ii < total; ++ii)
                    {
                        dst[ii] = (float)data[ii];
                    }
                    field_bufs[slot] = b;
                }
                else
                {
                    id<MTLBuffer> b = [state->device newBufferWithLength:bytes
                                                                 options:MTLResourceStorageModeShared];
                    if (b == nil)
                    {
                        return SIM_RESULT_OUT_OF_MEMORY;
                    }
                    float *dst = (float *)b.contents;
                    for (size_t ii = 0; ii < total; ++ii)
                    {
                        dst[ii] = (float)data[ii];
                    }
                    field_bufs[slot] = b;
                    if (entry->field_buffers != NULL)
                    {
                        entry->field_buffers[slot] = b;
                    }
                }
            }
        }
        int buffer_index = 0;
        for (size_t i = 0; i < field_bufs.size(); ++i)
        {
            [enc setBuffer:field_bufs[i] offset:0 atIndex:buffer_index++];
        }
        /* If kernel contains constants, reuse/create a persistent constants buffer. */
        id<MTLBuffer> local_constants_buf = nil;
        if (has_constants && kernel->builder != NULL && kernel->builder->constants_count > 0U &&
            kernel->builder->constants_data != NULL)
        {
            size_t total_values = 0U;
            size_t last = kernel->builder->constants_count - 1U;
            total_values = kernel->builder->constants_offsets[last] + kernel->builder->constants_components[last];
            size_t bytes = total_values * sizeof(float);
            if (bytes > 0U)
            {
                if (entry->constants_buffer == nil || entry->constants_bytes != bytes)
                {
                    if (entry->constants_buffer != nil)
                    {
#if !__has_feature(objc_arc)
                        [entry->constants_buffer release];
#endif
                        entry->constants_buffer = nil;
                        entry->constants_bytes = 0U;
                    }
                    entry->constants_buffer = [state->device newBufferWithLength:bytes
                                                                           options:MTLResourceStorageModeShared];
                    if (entry->constants_buffer == nil)
                    {
                        return SIM_RESULT_OUT_OF_MEMORY;
                    }
                    entry->constants_bytes = bytes;
                }
                float *fdata = (float *)entry->constants_buffer.contents;
                for (size_t k = 0; k < total_values; ++k)
                {
                    fdata[k] = (float)kernel->builder->constants_data[k];
                }
                local_constants_buf = entry->constants_buffer;
            }
        }
        if (local_constants_buf != nil)
        {
            [enc setBuffer:local_constants_buf offset:0 atIndex:buffer_index++];
        }
        id<MTLBuffer> local_params_buf = nil;
        if (has_params)
        {
            size_t count = kernel->param_count;
            size_t bytes = count * sizeof(float);
            if (bytes > 0U)
            {
                local_params_buf = [state->device newBufferWithLength:bytes
                                                               options:MTLResourceStorageModeShared];
                if (local_params_buf == nil)
                {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                float *pdata = (float *)local_params_buf.contents;
                for (size_t k = 0; k < count; ++k)
                {
                    pdata[k] = (float)kernel->params[k];
                }
                [enc setBuffer:local_params_buf offset:0 atIndex:buffer_index++];
            }
        }
        if (out_buf != nil)
        {
            [enc setBuffer:out_buf offset:0 atIndex:buffer_index++];
        }
        MTLSize threadsPerThreadgroup = MTLSizeMake((pipeline.maxTotalThreadsPerThreadgroup > 0 ? pipeline.maxTotalThreadsPerThreadgroup : 64), 1, 1);
        MTLSize gridSize = MTLSizeMake(element_count, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        /* Copy back results to host double array */
        float *res = (float *)out_buf.contents;
        double *out_host = (double *)sim_field_data(out_field);
        size_t total = element_count * comps;
        for (size_t i = 0; i < total; ++i) out_host[i] = (double)res[i];

        /* Release temporary buffers and resources created for this launch. */
    #if !__has_feature(objc_arc)
        for (size_t i = 0; i < field_bufs.size(); ++i)
        {
            if (field_bufs[i] != nil)
                {
                    if (entry == NULL || entry->field_buffers == NULL || entry->field_buffers[i] != field_bufs[i])
                    {
                        [field_bufs[i] release];
                    }
                }
        }
        if (out_buf != nil)
        {
            if (entry == NULL || entry->out_buffer != out_buf)
            {
                [out_buf release];
            }
        }
            if (local_constants_buf != nil && local_constants_buf != (entry != NULL ? entry->constants_buffer : nil))
                [local_constants_buf release];
            if (local_params_buf != nil)
                [local_params_buf release];
        /* compiled_pipeline is cache-owned (state->pipeline_cache_by_hash) and must
         * only be released when the cache entry is unreferenced/erased. Releasing it
         * here can leave a dangling pointer in the cache and crash on the next launch.
         */
#endif
            (void)local_constants_buf;
            (void)local_params_buf;
        return SIM_RESULT_OK;
}

/**
 * @brief Destroy a Metal backend instance.
 *
 * @param backend Backend to destroy; NULL is ignored.
 */
extern "C" void sim_backend_metal_destroy(SimBackend *backend)
{
    if (backend == NULL)
    {
        return;
    }

    backend_metal_state_destroy((SimBackendMetalState *)backend->impl);
    backend->impl = NULL;
    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);
}

#else /* SIM_HAVE_METAL */

/**
 * @brief Metal-disabled initialization stub.
 */
extern "C" SimResult sim_backend_metal_init(SimBackend *backend)
{
    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);
    return SIM_RESULT_NOT_FOUND;
}

/**
 * @brief Metal-disabled launch stub.
 */
extern "C" SimResult sim_backend_metal_launch(SimBackend *backend, KernelIR *kernel)
{
    (void)backend;
    (void)kernel;
    return SIM_RESULT_NOT_SUPPORTED;
}

/**
 * @brief Metal-disabled destroy stub.
 */
extern "C" void sim_backend_metal_destroy(SimBackend *backend)
{
    (void)backend;
}

#endif /* SIM_HAVE_METAL */
