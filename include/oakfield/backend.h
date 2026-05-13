/**
 * @file backend.h
 * @brief Backend abstraction layer for libsimcore execution engines.
 * @ingroup oakfield_backends
 *
 * Backends execute KernelIR graphs against bound simulation fields. The public
 * contract is intentionally small: callers initialize a SimBackend, launch a
 * KernelIR package, inspect `last_error`, and destroy the backend when its
 * implementation-specific resources are no longer needed.
 *
 * In the standalone CMake package, the CPU backend is the reference
 * backend. CUDA and Metal are experimental optional backends; requests return
 * #SIM_RESULT_NOT_FOUND unless the matching optional backend support is
 * compiled in.
 */
#ifndef OAKFIELD_BACKEND_H
#define OAKFIELD_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Identifies the concrete backend implementation in use.
 */
typedef enum SimBackendType {
    SIM_BACKEND_TYPE_CPU = 0, /**< Scalar CPU reference implementation. */
    SIM_BACKEND_TYPE_CUDA,    /**< CUDA backend using runtime compilation. */
    SIM_BACKEND_TYPE_METAL    /**< Metal backend using shader templates. */
} SimBackendType;

/**
 * @brief Compute backend feature flags indicating supported capabilities.
 *
 * KernelIR packages can require features such as analytic warp nodes or
 * boundary-aware finite differences. A backend must advertise all required
 * bits before it is a valid execution target for that kernel.
 */
typedef enum SimBackendFeature {
    SIM_BACKEND_FEATURE_NONE                 = 0ULL,      /**< No optional capabilities. */
    SIM_BACKEND_FEATURE_ANALYTIC_WARP        = 1ULL << 0, /**< Analytic warp IR nodes. */
    SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS = 1ULL << 1  /**< Boundary-aware finite differences. */
} SimBackendFeature;

/**
 * @brief Field binding made available to a compiled kernel.
 *
 * Bindings connect KernelIR field identifiers to runtime field storage and
 * layout metadata. Shape and stride arrays are measured in elements, not bytes;
 * when they are NULL, helper code may fall back to the field layout.
 */
typedef struct SimKernelIRBinding {
    size_t        field_index; /**< Index used by the IR to reference the field. */
    SimField*     field;       /**< Pointer to the bound field instance. */
    const size_t* shape;       /**< Shape pointer (elements) for the bound field. */
    const size_t* strides;     /**< Strides pointer (elements) for the bound field. */
    size_t        rank;        /**< Rank of the bound field. */
} SimKernelIRBinding;

/**
 * @brief Output specification emitted by a kernel.
 *
 * Each output writes one expression root to one bound destination field. The
 * destination field must be present in the KernelIR binding table.
 */
typedef struct SimKernelIROutput {
    size_t      field_index; /**< Destination field that receives the result. */
    SimIRNodeId expression;  /**< Root node identifier describing the value. */
} SimKernelIROutput;

/**
 * @brief Declares how complex-valued IR lanes should be interpreted by backends.
 *
 * True-complex semantics apply complex algebra to two-lane values. Componentwise
 * semantics allow backends to treat each lane independently, which enables
 * simpler pointwise fast paths for selected kernels.
 */
typedef enum SimKernelComplexSemantics {
    SIM_KERNEL_COMPLEX_SEMANTICS_TRUE_COMPLEX = 0, /**< Use true complex algebra semantics. */
    SIM_KERNEL_COMPLEX_SEMANTICS_COMPONENTWISE     /**< Treat lanes independently/componentwise. */
} SimKernelComplexSemantics;

/**
 * @brief KernelIR package describing the executable graph for a backend.
 *
 * The package is caller-owned for the duration of backend_launch(). Backends may
 * cache compiled artifacts keyed by the builder and emitted source, but they do
 * not take ownership of fields, bindings, outputs, params, or builder storage.
 */
typedef struct KernelIR {
    const SimIRBuilder*       builder;           /**< Underlying IR builder storage. */
    const SimKernelIRBinding* bindings;          /**< Field bindings referenced by the IR. */
    size_t                    binding_count;     /**< Number of entries in @ref bindings. */
    const SimKernelIROutput*  outputs;           /**< Output field mapping for the kernel. */
    size_t                    output_count;      /**< Number of kernel outputs. */
    double*                   params;            /**< Optional runtime parameter array. */
    size_t                    param_count;       /**< Number of entries in @ref params. */
    uint64_t                  required_features; /**< Feature mask required by the kernel. */
    SimKernelComplexSemantics complex_semantics; /**< Complex-lane interpretation contract. */
} KernelIR;

/**
 * @brief Runtime backend handle shared across implementations.
 *
 * `impl` is private to the selected backend type. Callers should initialize the
 * handle with backend_init() before launch and release it with backend_destroy().
 */
typedef struct SimBackend {
    SimBackendType type;       /**< Selected backend implementation. */
    SimResult      last_error; /**< Result code from the most recent operation. */
    uint64_t       features;   /**< Capability bitmask advertised by the backend. */
    void*          impl;       /**< Backend-specific opaque state. */
} SimBackend;

/**
 * @brief Initialize a backend instance.
 *
 * The requested `backend->type` selects CPU, CUDA, or Metal. Unknown types are
 * normalized to CPU. On failure, @ref SimBackend::last_error stores the
 * encountered error code and backend-specific implementation storage is not
 * considered valid.
 *
 * @param[in,out] backend Backend instance to initialize.
 */
void backend_init(SimBackend* backend);

/**
 * @brief Launch a kernel on the selected backend implementation.
 *
 * The backend writes results into the fields named by `kernel->outputs`. On
 * failure, @ref SimBackend::last_error stores the encountered error code.
 *
 * @param[in,out] backend Backend instance executing the kernel.
 * @param[in,out] kernel Kernel description to execute; storage remains caller-owned.
 */
void backend_launch(SimBackend* backend, KernelIR* kernel);

/**
 * @brief Tear down a backend instance and release owned resources.
 *
 * The handle is left with no advertised features and `last_error` set to
 * #SIM_RESULT_OK. Passing NULL is a no-op.
 *
 * @param[in,out] backend Backend instance to destroy.
 */
void backend_destroy(SimBackend* backend);

/**
 * @brief Query whether the CPU backend currently uses vDSP fast paths.
 *
 * @param backend Backend to inspect.
 * @return True only for initialized CPU backends compiled with vDSP support and
 * currently configured to use those paths.
 */
bool sim_backend_cpu_vdsp_enabled(const SimBackend* backend);

/**
 * @brief Enable or disable vDSP fast paths on the CPU backend at runtime.
 *
 * Calls against non-CPU backends or builds without vDSP support are ignored.
 *
 * @param backend Backend to mutate.
 * @param enabled Whether vDSP fast paths should be attempted.
 */
void sim_backend_cpu_set_vdsp_enabled(SimBackend* backend, bool enabled);

#ifdef SIM_TESTING
/**
 * @brief Copy the active Metal device name into a caller-provided buffer.
 *
 * This test/debug helper is available only when SIM_TESTING is defined.
 *
 * @param backend Initialized Metal backend.
 * @param[out] out_name Destination character buffer.
 * @param capacity Capacity of @p out_name in bytes.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult
sim_backend_metal_debug_copy_device_name(SimBackend* backend, char* out_name, size_t capacity);

/**
 * @brief Return the number of cached Metal pipeline entries.
 *
 * @param backend Initialized Metal backend.
 * @param[out] out_count Receives the pipeline cache entry count.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_backend_metal_debug_get_pipeline_cache_count(SimBackend* backend, size_t* out_count);

/**
 * @brief Return the total reference count across cached Metal pipelines.
 *
 * @param backend Initialized Metal backend.
 * @param[out] out_total Receives the sum of cache entry reference counts.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_backend_metal_debug_get_total_pipeline_refcount(SimBackend* backend,
                                                              size_t*     out_total);
#endif

/**
 * @brief Test whether a backend advertises every feature in a mask.
 *
 * @param backend Backend to inspect.
 * @param mask Feature mask to require.
 * @return True when @p backend is non-NULL and all bits in @p mask are present.
 */
static inline bool backend_supports_features(const SimBackend* backend, uint64_t mask) {
    if (backend == NULL) {
        return false;
    }
    return (backend->features & mask) == mask;
}

/**
 * @brief Test whether a backend advertises a single feature.
 *
 * @param backend Backend to inspect.
 * @param feature Feature bit to require.
 * @return True when the backend advertises @p feature.
 */
static inline bool backend_supports_feature(const SimBackend* backend, SimBackendFeature feature) {
    return backend_supports_features(backend, (uint64_t) feature);
}

/**
 * @brief Add one advertised feature bit to a backend.
 *
 * @param backend Backend to mutate; NULL is ignored.
 * @param feature Feature bit to add.
 */
static inline void backend_enable_feature(SimBackend* backend, SimBackendFeature feature) {
    if (backend == NULL) {
        return;
    }
    backend->features |= (uint64_t) feature;
}

/**
 * @brief Remove one advertised feature bit from a backend.
 *
 * @param backend Backend to mutate; NULL is ignored.
 * @param feature Feature bit to clear.
 */
static inline void backend_disable_feature(SimBackend* backend, SimBackendFeature feature) {
    if (backend == NULL) {
        return;
    }
    backend->features &= ~((uint64_t) feature);
}

/**
 * @brief Replace a backend's advertised feature mask.
 *
 * @param backend Backend to mutate; NULL is ignored.
 * @param mask New feature mask.
 */
static inline void backend_set_features(SimBackend* backend, uint64_t mask) {
    if (backend == NULL) {
        return;
    }
    backend->features = mask;
}

/**
 * @brief Return the innermost extent of a kernel binding.
 *
 * @param binding Binding to inspect.
 * @return Width in elements, or 0 for invalid/empty layout metadata.
 */
static inline size_t sim_kernel_binding_width(const SimKernelIRBinding* binding) {
    if (binding == NULL || binding->shape == NULL || binding->rank == 0U) {
        return 0U;
    }
    return binding->shape[binding->rank - 1U];
}

/**
 * @brief Return the second-innermost extent of a kernel binding.
 *
 * One-dimensional bindings report height 1.
 *
 * @param binding Binding to inspect.
 * @return Height in elements, or 0 for invalid/empty layout metadata.
 */
static inline size_t sim_kernel_binding_height(const SimKernelIRBinding* binding) {
    if (binding == NULL || binding->shape == NULL || binding->rank == 0U) {
        return 0U;
    }
    if (binding->rank == 1U) {
        return 1U;
    }
    return binding->shape[binding->rank - 2U];
}

/**
 * @brief Convert a linear element index into x/y coordinates for a binding.
 *
 * The conversion uses the two innermost axes for rank two or higher and treats
 * rank-one bindings as `y = 0`. If explicit binding shape/stride metadata is
 * missing, the helper falls back to the bound field layout.
 *
 * @param binding Binding and optional field layout to inspect.
 * @param index Linear element index.
 * @param[out] out_x Receives the innermost-axis coordinate.
 * @param[out] out_y Receives the second-innermost-axis coordinate or 0.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
static inline SimResult sim_kernel_binding_index_to_xy(const SimKernelIRBinding* binding,
                                                       size_t                    index,
                                                       size_t*                   out_x,
                                                       size_t*                   out_y) {
    if (binding == NULL || out_x == NULL || out_y == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const size_t* shape   = binding->shape;
    const size_t* strides = binding->strides;
    size_t        rank    = binding->rank;
    if ((shape == NULL || strides == NULL || rank == 0U) && binding->field != NULL) {
        shape   = binding->field->layout.shape;
        strides = binding->field->layout.strides;
        rank    = binding->field->layout.rank;
    }

    if (shape == NULL || strides == NULL || rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = 1U;
    for (size_t i = 0U; i < rank; ++i) {
        if (shape[i] == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count *= shape[i];
    }
    if (index >= count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (rank == 1U) {
        *out_x = index;
        *out_y = 0U;
        return SIM_RESULT_OK;
    }

    size_t axis_x   = rank - 1U;
    size_t axis_y   = rank - 2U;
    size_t extent_x = shape[axis_x];
    size_t extent_y = shape[axis_y];
    size_t stride_x = strides[axis_x];
    size_t stride_y = strides[axis_y];
    if (extent_x == 0U || extent_y == 0U || stride_x == 0U || stride_y == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_x = (index / stride_x) % extent_x;
    *out_y = (index / stride_y) % extent_y;
    return SIM_RESULT_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_BACKEND_H */
