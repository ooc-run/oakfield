/**
 * @file backend_cuda.cu
 * @brief CUDA backend skeleton leveraging NVRTC for runtime compilation.
 *
 * The CUDA backend owns a CUDA context, uploads field and constant buffers, and
 * emits a small NVRTC-compiled pointwise kernel for the supported KernelIR
 * subset. Unsupported IR shapes return #SIM_RESULT_NOT_SUPPORTED so callers can
 * fall back to the CPU backend.
 */

#include "oakfield/backend.h"

#ifdef SIM_HAVE_CUDA

#include <cuda.h>
#include <nvrtc.h>

#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

/**
 * @brief Cached CUDA kernel compiled from a KernelIR instance.
 *
 * Entries cache per-kernel device buffers and the module/function derived from
 * emitted source. CUDA modules are shared through the backend state's hash map
 * and unloaded with the backend state.
 */
typedef struct SimBackendCUDAKernel {
    const KernelIR* kernel;          /**< KernelIR instance associated with the compiled program. */
    char*           ptx;             /**< Generated PTX string. */
    size_t          ptx_size;        /**< Length of the PTX buffer. */
    CUmodule        module;          /**< Loaded CUDA module. */
    CUfunction      function;        /**< Kernel entry point. */
    CUdeviceptr     constants_dev;   /**< Device pointer to constants pool, if uploaded. */
    size_t          constants_bytes; /**< Size in bytes of constants_dev. */
    CUdeviceptr*    field_dev_ptrs;  /**< Per-slot device pointers for fields. */
    size_t          field_dev_ptrs_count; /**< Number of slots in field_dev_ptrs. */
    CUdeviceptr     out_dev;              /**< Cached output device buffer for this kernel. */
    size_t          out_dev_bytes;        /**< Byte size of cached out_dev. */
    size_t          src_hash; /**< Hash of the emitted kernel source used for caching. */
} SimBackendCUDAKernel;

/**
 * @brief CUDA backend state tracking compiled kernels and device context.
 *
 * `module_cache_by_hash` maps emitted source hashes to loaded modules so
 * equivalent generated kernels share driver resources.
 */
typedef struct SimBackendCUDAState {
    CUdevice                             device;          /**< CUDA device handle. */
    CUcontext                            context;         /**< CUDA context for launches. */
    SimBackendCUDAKernel*                kernels;         /**< Dynamic cache of compiled kernels. */
    size_t                               kernel_count;    /**< Number of cached kernel entries. */
    size_t                               kernel_capacity; /**< Allocated capacity of the cache. */
    std::unordered_map<size_t, CUmodule> module_cache_by_hash; /**< Map src_hash -> loaded module */
} SimBackendCUDAState;

/**
 * @brief Release device buffers owned by one cached CUDA kernel entry.
 *
 * The loaded module is not unloaded here because module lifetime is owned by
 * `SimBackendCUDAState::module_cache_by_hash`.
 */
static void backend_cuda_release_kernel(SimBackendCUDAKernel* entry) {
    if (entry == NULL) {
        return;
    }

    /* entry->module is not unloaded here; CUDA modules are cached in state->module_cache_by_hash
     * and will be unreferenced in backend_cuda_state_destroy to avoid double free when shared. */
    entry->module = NULL;
    /* Note: We do not unreference cached modules here; cached modules are managed by state and
     * will be unloaded when the backend is destroyed. */
    /* entry->module is not set to NULL here to avoid dangling pointers; consumer should reset entry->module */
    entry->module   = NULL;
    entry->ptx_size = 0U;
    entry->function = NULL;
    entry->kernel   = NULL;
    if (entry->constants_dev != 0ULL) {
        (void) cuMemFree(entry->constants_dev);
        entry->constants_dev   = 0ULL;
        entry->constants_bytes = 0U;
    }
    if (entry->field_dev_ptrs != NULL) {
        for (size_t i = 0; i < entry->field_dev_ptrs_count; ++i) {
            if (entry->field_dev_ptrs[i] != 0ULL) {
                (void) cuMemFree(entry->field_dev_ptrs[i]);
                entry->field_dev_ptrs[i] = 0ULL;
            }
        }
        free(entry->field_dev_ptrs);
        entry->field_dev_ptrs       = NULL;
        entry->field_dev_ptrs_count = 0U;
    }
    if (entry->out_dev != 0ULL) {
        (void) cuMemFree(entry->out_dev);
        entry->out_dev       = 0ULL;
        entry->out_dev_bytes = 0U;
    }
    entry->src_hash = 0U;
}

/* No module refcounting helpers for now; CUDA refcounting is disabled to focus on Metal only. */

/**
 * @brief Destroy CUDA backend state, cached kernel entries, modules, and context.
 */
static void backend_cuda_state_destroy(SimBackendCUDAState* state) {
    size_t i;

    if (state == NULL) {
        return;
    }

    for (i = 0; i < state->kernel_count; ++i) {
        backend_cuda_release_kernel(&state->kernels[i]);
    }

    free(state->kernels);
    state->kernels         = NULL;
    state->kernel_count    = 0U;
    state->kernel_capacity = 0U;

    if (state->context != NULL) {
        (void) cuCtxDestroy(state->context);
        state->context = NULL;
    }
    /* Unload any cached modules put in module_cache_by_hash */
    for (auto& kv : state->module_cache_by_hash) {
        if (kv.second != NULL)
            cuModuleUnload(kv.second);
    }
    state->module_cache_by_hash.clear();

    free(state);
}

/**
 * @brief Allocate CUDA backend state and create the primary execution context.
 *
 * @param[out] out_state Receives initialized state on success.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_OUT_OF_MEMORY,
 * or #SIM_RESULT_NOT_FOUND when CUDA initialization/device/context creation fails.
 */
static SimResult backend_cuda_state_init(SimBackendCUDAState** out_state) {
    SimBackendCUDAState* state;
    CUresult             cu_result;

    if (out_state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimBackendCUDAState*) calloc(1U, sizeof(SimBackendCUDAState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        backend_cuda_state_destroy(state);
        return SIM_RESULT_NOT_FOUND;
    }

    cu_result = cuDeviceGet(&state->device, 0);
    if (cu_result != CUDA_SUCCESS) {
        backend_cuda_state_destroy(state);
        return SIM_RESULT_NOT_FOUND;
    }

    cu_result = cuCtxCreate(&state->context, NULL, 0, state->device);
    if (cu_result != CUDA_SUCCESS) {
        backend_cuda_state_destroy(state);
        return SIM_RESULT_NOT_FOUND;
    }

    *out_state = state;
    return SIM_RESULT_OK;
}

/**
 * @brief Locate a field binding by KernelIR field index.
 */
static SimField* backend_cuda_find_field(const KernelIR* kernel, size_t field_index) {
    size_t i;
    if (kernel == NULL || kernel->bindings == NULL) {
        return NULL;
    }
    for (i = 0; i < kernel->binding_count; ++i) {
        if (kernel->bindings[i].field_index == field_index) {
            return kernel->bindings[i].field;
        }
    }
    return NULL;
}

/**
 * @brief Return the number of logical elements in a field.
 */
static size_t backend_cuda_element_count(const SimField* field) {
    if (field == NULL || field->element_size == 0U) {
        return 0U;
    }
    return sim_field_bytes(field) / field->element_size;
}

/**
 * @brief Initialize a CUDA backend instance.
 *
 * @param backend Backend handle to initialize.
 * @return #SIM_RESULT_OK or an initialization/allocation error.
 */
extern "C" SimResult sim_backend_cuda_init(SimBackend* backend) {
    SimBackendCUDAState* state;
    SimResult            result;

    if (backend == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);

    result = backend_cuda_state_init(&state);
    if (result != SIM_RESULT_OK) {
        backend->impl = NULL;
        return result;
    }

    backend->impl = state;
    backend->type = SIM_BACKEND_TYPE_CUDA;
    backend_enable_feature(backend, SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS);
    return SIM_RESULT_OK;
}

/**
 * @brief Launch a supported KernelIR pointwise kernel through CUDA.
 *
 * Supported kernels have one output, non-integer scalar/vector storage, no
 * unsupported complex semantics, and an expression tree composed of field refs,
 * constants, arithmetic, finite differences, warp nodes, and noise. Other
 * kernels return #SIM_RESULT_NOT_SUPPORTED.
 *
 * @param backend Initialized CUDA backend.
 * @param kernel Kernel package to execute.
 * @return #SIM_RESULT_OK or an error describing unsupported IR, missing fields,
 * CUDA allocation/copy failures, NVRTC compilation failures, or launch failures.
 */
extern "C" SimResult sim_backend_cuda_launch(SimBackend* backend, KernelIR* kernel) {
    SimBackendCUDAState* state;
    CUresult             cuRes = CUDA_SUCCESS;

    if (backend == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimBackendCUDAState*) backend->impl;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /*
     * Full CUDA code generation is not yet implemented. Mark the request as
     * unsupported so callers can fall back to the CPU backend.
     */
    /* Upload constants to the device and cache a kernel entry so future
     * launches or compilation steps can reuse them, though full runtime
     * codegen and launch is still unsupported. */
    (void) state;
    (void) kernel;
    /* find cache entry for kernel */
    SimBackendCUDAKernel* entry = NULL;
    for (size_t i = 0; i < state->kernel_count; ++i) {
        if (state->kernels[i].kernel == kernel) {
            entry = &state->kernels[i];
            break;
        }
    }
    if (entry == NULL) {
        size_t new_capacity = (state->kernel_capacity == 0U) ? 4U : state->kernel_capacity * 2U;
        if (state->kernel_capacity == 0U) {
            state->kernels =
                (SimBackendCUDAKernel*) calloc(new_capacity, sizeof(SimBackendCUDAKernel));
            state->kernel_capacity = new_capacity;
        } else if (state->kernel_count + 1U > state->kernel_capacity) {
            SimBackendCUDAKernel* new_arr = (SimBackendCUDAKernel*) realloc(
                state->kernels, new_capacity * sizeof(SimBackendCUDAKernel));
            if (new_arr == NULL) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            state->kernels         = new_arr;
            state->kernel_capacity = new_capacity;
        }
        entry                       = &state->kernels[state->kernel_count++];
        entry->kernel               = kernel;
        entry->ptx                  = NULL;
        entry->ptx_size             = 0U;
        entry->module               = NULL;
        entry->function             = NULL;
        entry->constants_dev        = 0ULL;
        entry->constants_bytes      = 0U;
        entry->field_dev_ptrs       = NULL;
        entry->field_dev_ptrs_count = 0;
        entry->out_dev              = 0ULL;
        entry->out_dev_bytes        = 0U;
        entry->src_hash             = 0U;
    }

    /* Upload constant pool if present */
    if (kernel->builder != NULL && kernel->builder->constants_count > 0U &&
        kernel->builder->constants_data != NULL) {
        size_t last = kernel->builder->constants_count - 1U;
        size_t total_values =
            kernel->builder->constants_offsets[last] + kernel->builder->constants_components[last];
        size_t bytes = total_values * sizeof(double);
        if (bytes > 0U) {
            CUdeviceptr devptr;
            CUresult    cu_res = cuMemAlloc(&devptr, bytes);
            if (cu_res != CUDA_SUCCESS) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            cu_res = cuMemcpyHtoD(devptr, kernel->builder->constants_data, bytes);
            if (cu_res != CUDA_SUCCESS) {
                (void) cuMemFree(devptr);
                return SIM_RESULT_INVALID_STATE;
            }
            /* free previous if present */
            if (entry->constants_dev != 0ULL) {
                (void) cuMemFree(entry->constants_dev);
            }
            entry->constants_dev   = devptr;
            entry->constants_bytes = bytes;
            std::fprintf(stderr,
                         "[DEBUG] sim_backend_cuda_launch: uploaded %zu constants bytes to device\n",
                         bytes);
        }
    }
    /* Try to generate and launch specialized device kernel for simple
     * pointwise operations; emit a device kernel for the expression tree
     * and launch it. Only a small subset of IR node types are supported here. */
    if (kernel == NULL || kernel->builder == NULL || kernel->output_count != 1U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    const SimIRBuilder* builder = kernel->builder;
    SimIRNodeId         root_id = kernel->outputs[0].expression;
    if (root_id == SIM_IR_INVALID_NODE || root_id >= builder->count) {
        return SIM_RESULT_NOT_SUPPORTED;
    }
    const SimIRNode* root = &builder->nodes[root_id];
    if (sim_scalar_domain_is_integer(sim_ir_type_scalar_domain(root->value_type))) {
        return SIM_RESULT_NOT_SUPPORTED;
    }
    if (sim_ir_kernel_has_unsupported_complex_semantics(kernel)) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    /* Determine output field and element count */
    size_t    out_field_index = kernel->outputs[0].field_index;
    SimField* out_field       = backend_cuda_find_field(kernel, out_field_index);
    if (out_field == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }
    if (sim_scalar_domain_is_integer(sim_scalar_domain_from_field(out_field))) {
        return SIM_RESULT_NOT_SUPPORTED;
    }
    size_t comps         = out_field->element_size / sizeof(double);
    size_t element_count = backend_cuda_element_count(out_field);
    if (element_count == 0U || comps == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* Collect field refs and constants used by the expression. */
    std::set<size_t>         field_refs;
    std::map<size_t, size_t> pool_const_offsets; /* const_index -> offset */
    std::vector<double>      scalar_consts;
    bool                     supported_nodes = true;

    std::function<void(SimIRNodeId)> collect = [&](SimIRNodeId nid) {
        if (nid == SIM_IR_INVALID_NODE || nid >= builder->count)
            return;
        const SimIRNode* n = &builder->nodes[nid];
        switch (n->type) {
            case SIM_IR_NODE_FIELD_REF:
                field_refs.insert(n->data.field);
                break;
            case SIM_IR_NODE_CONSTANT:
                if (n->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX) {
                    size_t idx              = n->data.constant.constant_index;
                    pool_const_offsets[idx] = builder->constants_offsets[idx];
                } else if (n->value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY) {
                    /* inline - nothing to collect */
                } else {
                    /* scalar fallback - add to scalar_consts list (maybe duplicated) */
                    scalar_consts.push_back(n->data.constant.scalar);
                }
                break;
            case SIM_IR_NODE_ADD:
            case SIM_IR_NODE_SUB:
            case SIM_IR_NODE_MUL:
            case SIM_IR_NODE_DIV:
                collect(n->data.binary.lhs);
                collect(n->data.binary.rhs);
                break;
            case SIM_IR_NODE_DIFF:
                collect(n->data.diff.operand);
                break;
            case SIM_IR_NODE_NOISE:
                /* noise nodes do not reference additional fields/constants */
                break;
            case SIM_IR_NODE_WARP:
                collect(n->data.warp.operand);
                break;
            case SIM_IR_NODE_PARAM:
            case SIM_IR_NODE_COMPLEX_ROTATE:
            case SIM_IR_NODE_INDEX:
            case SIM_IR_NODE_CALL:
            case SIM_IR_NODE_FLOOR:
            case SIM_IR_NODE_MOD:
            case SIM_IR_NODE_COORD:
            case SIM_IR_NODE_STATEFUL:
                supported_nodes = false;
                break;
            default:
                supported_nodes = false;
                break;
        }
    };
    collect(root_id);

    if (!supported_nodes) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    /* Build mapping of field_index -> arg position */
    std::map<size_t, int> field_arg_idx;
    int                   arg_pos = 0;
    for (auto fid : field_refs) {
        field_arg_idx[fid] = arg_pos++;
    }

    std::map<size_t, const SimKernelIRBinding*> binding_by_field;
    if (kernel->bindings != NULL) {
        for (size_t i = 0; i < kernel->binding_count; ++i) {
            const SimKernelIRBinding* binding      = &kernel->bindings[i];
            binding_by_field[binding->field_index] = binding;
        }
    }

    /* Validate component counts compatibility */
    size_t required_comps = comps;
    for (auto fid : field_refs) {
        SimField* f = backend_cuda_find_field(kernel, fid);
        if (f == NULL) {
            return SIM_RESULT_NOT_FOUND;
        }
        if ((f->element_size / sizeof(double)) != required_comps) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    /*
     * Expression code generation helper. It emits a componentwise CUDA
     * expression for the supported KernelIR subset and mirrors the CPU boundary
     * policy for finite differences where binding layout metadata is available.
     */
    std::function<std::string(SimIRNodeId, const std::string&)> emit_expr =
        [&](SimIRNodeId nid, const std::string& idx_expr) -> std::string {
        const SimIRNode* n = &builder->nodes[nid];
        switch (n->type) {
            case SIM_IR_NODE_FIELD_REF: {
                int  idx = field_arg_idx[n->data.field];
                char buf[64];
                std::snprintf(buf, sizeof(buf), "f%d[%s*comps + c]", idx, idx_expr.c_str());
                return std::string(buf);
            }
            case SIM_IR_NODE_CONSTANT: {
                if (n->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX) {
                    size_t idx = n->data.constant.constant_index;
                    size_t off = builder->constants_offsets[idx];
                    char   buf[128];
                    std::snprintf(buf, sizeof(buf), "(constants[%zu + c])", off);
                    return std::string(buf);
                } else if (n->value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY) {
                    /* inline small constant -> emit ternary chain */
                    std::string s = "(c==0?";
                    for (size_t i = 0; i < n->value_type.components; ++i) {
                        char tmp[64];
                        std::snprintf(tmp, sizeof(tmp), "%g", n->data.constant.small[i]);
                        s += tmp;
                        if (i + 1 < n->value_type.components) {
                            s += ":(c==" + std::to_string(i + 1) + "?";
                        }
                    }
                    for (size_t i = 0; i + 1 < n->value_type.components; ++i)
                        s += ")";
                    return s;
                } else {
                    char tmp[64];
                    std::snprintf(tmp, sizeof(tmp), "%g", n->data.constant.scalar);
                    return std::string(tmp);
                }
            }
            case SIM_IR_NODE_ADD:
                return std::string("(") + emit_expr(n->data.binary.lhs, idx_expr) + ") + (" +
                       emit_expr(n->data.binary.rhs, idx_expr) + ")";
            case SIM_IR_NODE_SUB:
                return std::string("(") + emit_expr(n->data.binary.lhs, idx_expr) + ") - (" +
                       emit_expr(n->data.binary.rhs, idx_expr) + ")";
            case SIM_IR_NODE_MUL:
                return std::string("(") + emit_expr(n->data.binary.lhs, idx_expr) + ") * (" +
                       emit_expr(n->data.binary.rhs, idx_expr) + ")";
            case SIM_IR_NODE_DIV:
                return std::string("(") + emit_expr(n->data.binary.lhs, idx_expr) + ") / (" +
                       emit_expr(n->data.binary.rhs, idx_expr) + ")";
            case SIM_IR_NODE_WARP: {
                std::string operand = emit_expr(n->data.warp.operand, idx_expr);
                char        buf[256];
                std::snprintf(buf,
                              sizeof(buf),
                              "warp_response_d(%d, (%s), %.17g, %.17g, %.17g, %d, %.17g, %.17g, %.17g)",
                              (int) n->data.warp.profile,
                              operand.c_str(),
                              (double) n->data.warp.bias,
                              (double) n->data.warp.delta,
                              (double) n->data.warp.lambda,
                              (int) n->data.warp.guard.mode,
                              (double) n->data.warp.guard.clamp_min,
                              (double) n->data.warp.guard.clamp_max,
                              (double) n->data.warp.guard.tolerance);
                return std::string(buf);
            }
            case SIM_IR_NODE_DIFF: {
                size_t axis   = n->data.diff.axis;
                size_t extent = 0U;
                size_t stride = 1U;
                if (n->data.diff.operand < builder->count) {
                    const SimIRNode* opnode = &builder->nodes[n->data.diff.operand];
                    if (opnode->type == SIM_IR_NODE_FIELD_REF) {
                        size_t field_id   = opnode->data.field;
                        auto   it_binding = binding_by_field.find(field_id);
                        if (it_binding != binding_by_field.end()) {
                            const SimKernelIRBinding* binding = it_binding->second;
                            if (binding != NULL && binding->shape != NULL &&
                                binding->strides != NULL && axis < binding->rank) {
                                extent = binding->shape[axis];
                                stride = binding->strides[axis];
                            }
                        }
                    }
                }

                std::string coord = "(" + idx_expr + " / " + std::to_string(stride) + ")";
                std::string coord_mod =
                    "(" + coord + " % " + std::to_string((extent > 0U) ? extent : 1U) + ")";
                std::string center_expr  = emit_expr(n->data.diff.operand, idx_expr);
                std::string forward_idx  = idx_expr;
                std::string backward_idx = idx_expr;
                std::string has_forward =
                    (extent > 0U) ? "((" + coord_mod + " + 1) < " + std::to_string(extent) + ")"
                                  : "false";
                std::string has_backward = (extent > 0U) ? "(" + coord_mod + " > 0)" : "false";

                switch (n->data.diff.boundary) {
                    case SIM_IR_BOUNDARY_PERIODIC:
                        if (extent > 0U) {
                            has_forward  = (extent > 1U) ? "true" : "false";
                            has_backward = (extent > 1U) ? "true" : "false";
                            forward_idx  = "((" + coord_mod + " + 1 < " + std::to_string(extent) +
                                          ") ? (" + idx_expr + " + " + std::to_string(stride) +
                                          ") : (" + idx_expr + " - (" + std::to_string(extent) +
                                          " - 1) * " + std::to_string(stride) + "))";
                            backward_idx = "(" + coord_mod + " > 0 ? (" + idx_expr + " - " +
                                           std::to_string(stride) + ") : (" + idx_expr + " + (" +
                                           std::to_string(extent) + " - 1) * " +
                                           std::to_string(stride) + "))";
                        }
                        break;
                    case SIM_IR_BOUNDARY_REFLECTIVE:
                        if (extent > 0U) {
                            has_forward  = (extent > 1U) ? "true" : "false";
                            has_backward = (extent > 1U) ? "true" : "false";
                            forward_idx  = "((" + coord_mod + " + 1 < " + std::to_string(extent) +
                                          ") ? (" + idx_expr + " + " + std::to_string(stride) +
                                          ") : (" + idx_expr + " >= " + std::to_string(stride) +
                                          " ? " + idx_expr + " - " + std::to_string(stride) +
                                          " : " + idx_expr + "))";
                            backward_idx = "(" + coord_mod + " > 0 ? (" + idx_expr + " - " +
                                           std::to_string(stride) + ") : (" + idx_expr + " + " +
                                           std::to_string(stride) + "))";
                        }
                        break;
                    case SIM_IR_BOUNDARY_DIRICHLET:
                    case SIM_IR_BOUNDARY_NEUMANN:
                    default:
                        if (extent > 0U) {
                            forward_idx  = "(" + idx_expr + " + " + std::to_string(stride) + ")";
                            backward_idx = "(" + idx_expr + " >= " + std::to_string(stride) +
                                           " ? " + idx_expr + " - " + std::to_string(stride) +
                                           " : " + idx_expr + ")";
                        }
                        break;
                }

                std::string forward_expr  = emit_expr(n->data.diff.operand, forward_idx);
                std::string backward_expr = emit_expr(n->data.diff.operand, backward_idx);

                switch (n->data.diff.boundary) {
                    case SIM_IR_BOUNDARY_DIRICHLET:
                        forward_expr  = "(" + has_forward + " ? (" + forward_expr + ") : 0.0)";
                        backward_expr = "(" + has_backward + " ? (" + backward_expr + ") : 0.0)";
                        break;
                    case SIM_IR_BOUNDARY_NEUMANN:
                        forward_expr = "(" + has_forward + " ? (" + forward_expr + ") : (" +
                                       center_expr + "))";
                        backward_expr = "(" + has_backward + " ? (" + backward_expr + ") : (" +
                                        center_expr + "))";
                        break;
                    case SIM_IR_BOUNDARY_PERIODIC:
                    case SIM_IR_BOUNDARY_REFLECTIVE:
                    default:
                        break;
                }

                std::string dx_str    = std::to_string((double) n->data.diff.dx);
                std::string scale_str = std::to_string((double) n->data.diff.scale);
                std::string derivative;

                switch (n->data.diff.method) {
                    case SIM_IR_DIFF_METHOD_FORWARD:
                        derivative =
                            "((" + forward_expr + ") - (" + center_expr + ")) / (" + dx_str + ")";
                        break;
                    case SIM_IR_DIFF_METHOD_BACKWARD:
                        derivative =
                            "((" + center_expr + ") - (" + backward_expr + ")) / (" + dx_str + ")";
                        break;
                    case SIM_IR_DIFF_METHOD_CENTRAL:
                        derivative = "((" + forward_expr + ") - (" + backward_expr +
                                     ")) / (2.0 * " + dx_str + ")";
                        break;
                    case SIM_IR_DIFF_METHOD_AUTO:
                    default:
                        derivative = "((" + has_forward + " && " + has_backward + ") ? (((" +
                                     forward_expr + ") - (" + backward_expr + ")) / (2.0 * " +
                                     dx_str + ")) : (" + has_forward + " ? (((" + forward_expr +
                                     ") - (" + center_expr + ")) / (" + dx_str + ")) : (" +
                                     has_backward + " ? (((" + center_expr + ") - (" +
                                     backward_expr + ")) / (" + dx_str + ")) : 0.0)))";
                        break;
                }

                return "((" + derivative + ") * (" + scale_str + "))";
            }
            case SIM_IR_NODE_NOISE: {
                uint32_t seed = n->data.noise.seed;
                double   amp  = n->data.noise.amplitude;
                char     buf[256];
                std::snprintf(buf,
                              sizeof(buf),
                              "(((float)(((uint)(%s)*1664525u + %uu) ^ (((uint)(%s)*1664525u + %uu) >> "
                              "16)) & 0xFFFFFFu)/16777216.0f*2.0f - 1.0f) * %g",
                              idx_expr.c_str(),
                              seed,
                              idx_expr.c_str(),
                              seed,
                              amp);
                return std::string(buf);
            }
            default:
                return std::string("0.0");
        }
    };

    std::string expr = emit_expr(root_id, std::string("i"));

    /* Ensure we have PTX for emitted op; generate NVRTC code. */
    /* Compose kernel signature: field args, constants pointer, output pointer, count, components. */
    std::vector<size_t> fields_by_slot((int) field_arg_idx.size(), 0);
    for (auto& kv : field_arg_idx) {
        int slot             = kv.second;
        fields_by_slot[slot] = kv.first;
    }
    std::string src_helpers;
    src_helpers += "#include <math.h>\n";
    src_helpers +=
        "__device__ double digamma_d(double x) { if (!isfinite(x)) return NAN; if (x <= 0.0) { "
        "double frac = x - floor(x); if (fabs(frac) < 1e-16) return NAN; return digamma_d(1.0 - x) "
        "- M_PI * cos(M_PI * x) / sin(M_PI * x); } double value = x; double result = 0.0; while "
        "(value < 8.0) { result -= 1.0 / value; value += 1.0; } double inv = 1.0 / value; double "
        "inv2 = inv * inv; double series = inv2 * (-1.0/12.0 + inv2*(1.0/120.0 + inv2*(-1.0/252.0 "
        "+ inv2*(1.0/240.0 + inv2*(-1.0/132.0 + inv2*(691.0/32760.0)))))); result += log(value) - "
        "0.5 * inv + series; return result; }\n";
    src_helpers += "__device__ double trigamma_d(double x) { if (!isfinite(x)) return NAN; if (x "
                   "<= 0.0) { double sinpx = sin(M_PI * x); double csc_sq = 1.0 / (sinpx * sinpx); "
                   "if (!isfinite(csc_sq)) return NAN; return csc_sq - trigamma_d(1.0 - x); } "
                   "double value = x; double result = 0.0; while (value < 8.0) { result += 1.0 / "
                   "(value * value); value += 1.0; } double inv = 1.0 / value; double inv2 = inv * "
                   "inv; double inv3 = inv2 * inv; double inv5 = inv3 * inv2; double inv7 = inv5 * "
                   "inv2; double inv9 = inv7 * inv2; double inv11 = inv9 * inv2; double inv13 = "
                   "inv11 * inv2; result += inv + 0.5 * inv2 + inv3/6.0 - inv5/30.0 + inv7/42.0 - "
                   "inv9/30.0 + (5.0*inv11)/66.0 - (691.0*inv13)/2730.0; return result; }\n";
    src_helpers +=
        "__device__ double warp_positive_or_default(double requested, double fallback) { if "
        "(isfinite(requested) && requested > 0.0) return requested; return fallback; }\n";
    src_helpers += "__device__ double warp_clamp_value(double value, double clamp_min, double "
                   "clamp_max) { if (!(isfinite(clamp_min) && isfinite(clamp_max) && clamp_min < "
                   "clamp_max)) return value; if (value < clamp_min) return clamp_min; if (value > "
                   "clamp_max) return clamp_max; return value; }\n";
    src_helpers += "__device__ double warp_profile_eval(int prof, double x) { return (prof == 1) ? "
                   "trigamma_d(x) : digamma_d(x); }\n";
    src_helpers +=
        "__device__ double warp_gradient_d(int prof, double sample, double offset) { if "
        "(!(isfinite(offset) && offset > 0.0)) return NAN; double f_plus = warp_profile_eval(prof, "
        "sample + offset); double f_minus = warp_profile_eval(prof, sample - offset); if "
        "(!isfinite(f_plus) || !isfinite(f_minus)) return NAN; double grad = (f_plus - f_minus) / "
        "(2.0 * offset); if (!isfinite(grad)) return NAN; return grad; }\n";
    src_helpers +=
        "__device__ double warp_response_d(int prof, double sample, double bias, double delta, "
        "double lambda, int guard_mode, double clamp_min, double clamp_max, double tolerance) { "
        "double offset = fabs(delta); if (!(offset > 0.0)) offset = 1.0e-6; double biased = sample "
        "+ bias; double grad = warp_gradient_d(prof, biased, offset); int guard_clamped = "
        "(guard_mode == 2) || (guard_mode == 3); if (guard_clamped && isfinite(grad)) { grad = "
        "warp_clamp_value(grad, clamp_min, clamp_max); } if (!isfinite(grad)) { if (guard_clamped) "
        "{ double eps = warp_positive_or_default(fabs(tolerance), offset); double gp = "
        "warp_gradient_d(prof, biased + eps, offset); double gm = warp_gradient_d(prof, biased - "
        "eps, offset); int have_gp = isfinite(gp); int have_gm = isfinite(gm); double candidate = "
        "NAN; if (guard_mode == 2) { if (have_gp) candidate = gp; else if (have_gm) candidate = "
        "gm; } else { if (have_gp && have_gm) candidate = 0.5 * (gp + gm); else if (have_gp) "
        "candidate = gp; else if (have_gm) candidate = gm; } if (!isfinite(candidate)) candidate = "
        "0.0; candidate = warp_clamp_value(candidate, clamp_min, clamp_max); grad = candidate; } "
        "else { grad = 0.0; } } if (!isfinite(lambda)) lambda = 0.0; if (!isfinite(grad)) grad = "
        "0.0; double response = lambda * (2.0 * offset) * grad; if (!isfinite(response)) response "
        "= 0.0; return response; }\n";

    std::string src;
    src += src_helpers;
    src += "extern \"C\" __global__ void kernel_func(";
    /* fields in slot order */
    int total_field_args = (int) fields_by_slot.size();
    for (int slot = 0; slot < total_field_args; ++slot) {
        src += "const double* f" + std::to_string(slot) + ", ";
    }
    /* constants pointer */
    bool has_constants_dev = (entry != NULL && entry->constants_dev != 0ULL);
    if (has_constants_dev) {
        src += "const double* constants, ";
    }
    src += "double* out, size_t count, size_t comps) { size_t i = blockIdx.x*blockDim.x + "
           "threadIdx.x; if (i >= count) return; for (size_t c=0; c<comps; ++c) { double res = ";
    src += expr;
    src += "; out[i*comps + c] = res; } }\n";

    size_t                   bytes = element_count * required_comps * sizeof(double);
    std::vector<CUdeviceptr> d_field_ptrs(field_arg_idx.size(), 0ULL);
    std::vector<char>        d_field_ptrs_new(field_arg_idx.size(), 0);
    for (auto& kv : field_arg_idx) {
        size_t    fid  = kv.first;
        int       slot = kv.second;
        SimField* f    = backend_cuda_find_field(kernel, fid);
        if (f == NULL) {
            d_field_ptrs[slot]     = 0ULL;
            d_field_ptrs_new[slot] = 0;
            continue;
        }
        double*     host_data = (double*) sim_field_data(f);
        CUdeviceptr dp        = 0ULL;
        if (entry != NULL && entry->field_dev_ptrs != NULL &&
            entry->field_dev_ptrs_count == d_field_ptrs.size() &&
            entry->field_dev_ptrs[slot] != 0ULL) {
            dp    = entry->field_dev_ptrs[slot];
            cuRes = cuMemcpyHtoD(dp, host_data, bytes);
            if (cuRes != CUDA_SUCCESS) {
                for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                    if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                        cuMemFree(d_field_ptrs[j]);
                    }
                }
                return SIM_RESULT_INVALID_STATE;
            }
        } else {
            cuRes = cuMemAlloc(&dp, bytes);
            if (cuRes != CUDA_SUCCESS) {
                for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                    if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                        cuMemFree(d_field_ptrs[j]);
                    }
                }
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            cuRes = cuMemcpyHtoD(dp, host_data, bytes);
            if (cuRes != CUDA_SUCCESS) {
                cuMemFree(dp);
                for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                    if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                        cuMemFree(d_field_ptrs[j]);
                    }
                }
                return SIM_RESULT_INVALID_STATE;
            }
            d_field_ptrs_new[slot] = 1;
            if (entry != NULL) {
                if (entry->field_dev_ptrs == NULL ||
                    entry->field_dev_ptrs_count != d_field_ptrs.size()) {
                    free(entry->field_dev_ptrs);
                    entry->field_dev_ptrs_count = d_field_ptrs.size();
                    entry->field_dev_ptrs =
                        (CUdeviceptr*) calloc(entry->field_dev_ptrs_count, sizeof(CUdeviceptr));
                    if (entry->field_dev_ptrs == NULL) {
                        cuMemFree(dp);
                        for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                            if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                                cuMemFree(d_field_ptrs[j]);
                            }
                        }
                        entry->field_dev_ptrs_count = 0U;
                        return SIM_RESULT_OUT_OF_MEMORY;
                    }
                }
                entry->field_dev_ptrs[slot] = dp;
                d_field_ptrs_new[slot]      = 0;
            }
        }
        d_field_ptrs[slot] = dp;
    }

    std::hash<std::string> hasher;
    size_t                 src_hash = hasher(src);
    CUmodule               module   = NULL;
    CUfunction             func     = NULL;

    auto it_mod = state->module_cache_by_hash.find(src_hash);
    if (it_mod != state->module_cache_by_hash.end()) {
        module = it_mod->second;
        cuRes  = cuModuleGetFunction(&func, module, "kernel_func");
        if (cuRes != CUDA_SUCCESS) {
            cuModuleUnload(module);
            state->module_cache_by_hash.erase(it_mod);
            module = NULL;
            func   = NULL;
        }
    }

    if (module == NULL) {
        nvrtcProgram prog;
        nvrtcResult  nvrt = nvrtcCreateProgram(&prog, src.c_str(), "kernel.cu", 0, NULL, NULL);
        if (nvrt != NVRTC_SUCCESS) {
            for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                    cuMemFree(d_field_ptrs[j]);
                }
            }
            return SIM_RESULT_INVALID_STATE;
        }
        const char* opts[] = { "--gpu-architecture=compute_52" };
        nvrt               = nvrtcCompileProgram(prog, 1, opts);
        if (nvrt != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                    cuMemFree(d_field_ptrs[j]);
                }
            }
            return SIM_RESULT_INVALID_STATE;
        }
        size_t ptx_size = 0U;
        nvrtcGetPTXSize(prog, &ptx_size);
        std::vector<char> ptx(ptx_size);
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);

        cuRes = cuModuleLoadData(&module, ptx.data());
        if (cuRes != CUDA_SUCCESS) {
            for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                    cuMemFree(d_field_ptrs[j]);
                }
            }
            return SIM_RESULT_INVALID_STATE;
        }

        cuRes = cuModuleGetFunction(&func, module, "kernel_func");
        if (cuRes != CUDA_SUCCESS) {
            cuModuleUnload(module);
            for (size_t j = 0; j < d_field_ptrs.size(); ++j) {
                if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL) {
                    cuMemFree(d_field_ptrs[j]);
                }
            }
            return SIM_RESULT_INVALID_STATE;
        }

        state->module_cache_by_hash[src_hash] = module;
    }

    if (entry != NULL) {
        entry->module   = module;
        entry->function = func;
        entry->src_hash = src_hash;
    }

    /* Ensure we have an output device buffer cached for the kernel */
    CUdeviceptr d_out_dev          = 0ULL;
    bool        free_d_out_dev_end = false;
    if (entry != NULL && entry->out_dev != 0ULL && entry->out_dev_bytes == bytes) {
        d_out_dev = entry->out_dev;
    } else {
        CUdeviceptr new_out = 0ULL;
        cuRes               = cuMemAlloc(&new_out, bytes);
        if (cuRes != CUDA_SUCCESS) {
            for (size_t j = 0; j < d_field_ptrs.size(); ++j)
                if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL)
                    cuMemFree(d_field_ptrs[j]);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        if (entry != NULL) {
            if (entry->out_dev != 0ULL) {
                cuMemFree(entry->out_dev);
            }
            entry->out_dev       = new_out;
            entry->out_dev_bytes = bytes;
            d_out_dev            = new_out;
        } else {
            d_out_dev          = new_out;
            free_d_out_dev_end = true;
        }
    }

    /* Prepare arguments as driver API device-pointer references. */
    std::vector<CUdeviceptr> dev_ptrs =
        d_field_ptrs; /* copy list of device pointers for use in args */
    std::vector<void*> args;
    for (int slot = 0; slot < total_field_args; ++slot) {
        args.push_back(&dev_ptrs[slot]);
    }
    if (has_constants_dev) {
        args.push_back(&entry->constants_dev);
    }
    args.push_back(&d_out_dev);
    args.push_back(&element_count);
    args.push_back(&required_comps);

    /* Launch and synchronize the generated kernel. */
    int block = 128;
    int grid  = (int) ((element_count + block - 1) / block);
    cuRes     = cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, 0, args.data(), NULL);
    cuCtxSynchronize();
    if (cuRes != CUDA_SUCCESS) {
        for (size_t j = 0; j < d_field_ptrs.size(); ++j)
            if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL)
                cuMemFree(d_field_ptrs[j]);
        if (free_d_out_dev_end)
            cuMemFree(d_out_dev);
        return SIM_RESULT_INVALID_STATE;
    }

    /* Copy generated results back into the host output field. */
    SimField* out_f    = out_field;
    double*   out_host = (double*) sim_field_data(out_f);
    cuRes              = cuMemcpyDtoH(out_host, d_out_dev, bytes);

    if (free_d_out_dev_end)
        cuMemFree(d_out_dev);
    for (size_t j = 0; j < d_field_ptrs.size(); ++j)
        if (d_field_ptrs_new[j] && d_field_ptrs[j] != 0ULL)
            cuMemFree(d_field_ptrs[j]);
    return (cuRes == CUDA_SUCCESS) ? SIM_RESULT_OK : SIM_RESULT_INVALID_STATE;
}

/**
 * @brief Destroy a CUDA backend instance.
 *
 * @param backend Backend to destroy; NULL is ignored.
 */
extern "C" void sim_backend_cuda_destroy(SimBackend* backend) {
    if (backend == NULL) {
        return;
    }

    backend_cuda_state_destroy((SimBackendCUDAState*) backend->impl);
    backend->impl = NULL;
    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);
}

#else /* SIM_HAVE_CUDA */

/**
 * @brief CUDA-disabled initialization stub.
 */
extern "C" SimResult sim_backend_cuda_init(SimBackend* backend) {
    (void) backend;
    return SIM_RESULT_NOT_FOUND;
}

/**
 * @brief CUDA-disabled launch stub.
 */
extern "C" SimResult sim_backend_cuda_launch(SimBackend* backend, KernelIR* kernel) {
    (void) backend;
    (void) kernel;
    return SIM_RESULT_NOT_SUPPORTED;
}

/**
 * @brief CUDA-disabled destroy stub.
 */
extern "C" void sim_backend_cuda_destroy(SimBackend* backend) {
    (void) backend;
}

#endif /* SIM_HAVE_CUDA */
