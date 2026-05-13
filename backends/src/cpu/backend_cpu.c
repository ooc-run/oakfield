/**
 * @file backend_cpu.c
 * @brief CPU backend implementation featuring scalar and optional vDSP paths.
 *
 * The CPU backend is the reference KernelIR evaluator. It evaluates each output
 * field element-by-element with memoized recursive node evaluation, supports
 * real, complex, vector-lane, and integer scalar domains, and opportunistically
 * dispatches simple pointwise kernels through vDSP when available.
 */

#include "oakfield/backend.h"

#include "oakfield/operator.h"
#include "sim_accel.h"
#include "operators/common/warp_safety.h"
#include "operators/common/nd_neighbors.h"

#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#if defined(SIM_HAVE_CUDA)
SimResult sim_backend_cuda_init(SimBackend* backend);
SimResult sim_backend_cuda_launch(SimBackend* backend, KernelIR* kernel);
void      sim_backend_cuda_destroy(SimBackend* backend);
#endif

#if defined(SIM_HAVE_METAL)
SimResult sim_backend_metal_init(SimBackend* backend);
SimResult sim_backend_metal_launch(SimBackend* backend, KernelIR* kernel);
void      sim_backend_metal_destroy(SimBackend* backend);
#endif

/**
 * @brief Per-launch recursive evaluation cache for one CPU output traversal.
 *
 * Scratch slots are addressed by `(node_id, component)`. Flags use a
 * generation-stamped tri-state protocol: clean, active, and done. The active
 * state catches dependency cycles during recursive IR evaluation.
 */
typedef struct SimBackendCPUFrame {
    double*        scratch;            /**< Cached node values for a given evaluation context. */
    unsigned char* flags;              /**< Evaluation state per node (0=clean,1=active,2=done). */
    uint32_t*      flag_generation;    /**< Generation-stamp per node for lazy clears. */
    uint32_t       generation_counter; /**< Monotonic generation counter. */
    size_t         capacity;           /**< Number of per-node slots allocated. */
    size_t         components;         /**< Number of components per node (1=scalar,2=complex) */
} SimBackendCPUFrame;

/**
 * @brief Internal scratch buffer allocator state for the CPU backend.
 *
 * Frames are kept as a stack so recursive neighbor/differential evaluations can
 * borrow nested evaluation caches without invalidating the outer traversal.
 */
typedef struct SimBackendCPUState {
    SimBackendCPUFrame** frames;             /**< Stack of reusable evaluation frames. */
    size_t               frame_capacity;     /**< Number of allocated frames at @ref frames. */
    size_t               frame_count;        /**< Active frame count (stack depth). */
    size_t               capacity;           /**< Per-frame node capacity. */
    size_t               component_capacity; /**< Per-frame component capacity. */
#if defined(SIM_HAVE_VDSP)
    bool                       use_vdsp; /**< Indicates vDSP is available at runtime. */
    SimAccelSplitComplexScratch vdsp_complex_scratch;
#endif
} SimBackendCPUState;

/**
 * @brief Neighbor indices for one finite-difference axis.
 */
typedef struct SimBackendCPUAxisNeighbors {
    bool   has_forward;
    bool   has_backward;
    size_t forward_index;
    size_t backward_index;
    bool   forward_wrapped;
    bool   backward_wrapped;
} SimBackendCPUAxisNeighbors;

/**
 * @brief Reduced operand form for simple pointwise field operations.
 *
 * The fast path accepts a field term, scalar constant term, and vector constant
 * term accumulated from affine expression trees. It deliberately rejects
 * expressions that would need general recursive evaluation.
 */
typedef struct BackendCPUOperand {
    const double* field_data;
    double        field_scale;
    bool          has_field;
    bool          has_scalar_constant;
    double        scalar_constant;
    const double* vector_constant;
    double        vector_scale;
    size_t        vector_components;
} BackendCPUOperand;

typedef struct BackendCPUComponentKernelDesc BackendCPUComponentKernelDesc;

/**
 * @brief Specialized component-row evaluator for small fixed lane counts.
 */
typedef SimResult (*BackendCPUComponentKernel)(const KernelIR*     kernel,
                                               size_t              element_index,
                                               SimBackendCPUState* state,
                                               SimBackendCPUFrame* frame,
                                               const SimField*     reference_field,
                                               SimIRNodeId         node_id,
                                               double*             row,
                                               size_t              component_limit);

/**
 * @brief Dispatch record for evaluating all components of one output element.
 */
struct BackendCPUComponentKernelDesc {
    BackendCPUComponentKernel eval;
    size_t                    component_limit;
};

static SimResult backend_cpu_init_impl(SimBackend* backend);
static SimResult backend_cpu_launch_impl(SimBackend* backend, KernelIR* kernel);
static void      backend_cpu_destroy_impl(SimBackend* backend);

/**
 * @brief Return mutable CPU-private state from a generic backend handle.
 */
static SimBackendCPUState* backend_cpu_state(SimBackend* backend) {
    if (backend == NULL) {
        return NULL;
    }
    return (SimBackendCPUState*) backend->impl;
}

/**
 * @brief Initialize CPU backend state and advertise CPU-supported features.
 *
 * Reinitializing an already initialized CPU backend refreshes its type and
 * feature bits without replacing the existing scratch cache.
 *
 * @param backend Backend handle to initialize.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
static SimResult backend_cpu_init_impl(SimBackend* backend) {
    SimBackendCPUState* state;

    if (backend == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (backend->impl != NULL) {
        backend->type = SIM_BACKEND_TYPE_CPU;
        backend_enable_feature(backend, SIM_BACKEND_FEATURE_ANALYTIC_WARP);
        return SIM_RESULT_OK;
    }

    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);

    state = (SimBackendCPUState*) calloc(1U, sizeof(SimBackendCPUState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

#if defined(SIM_HAVE_VDSP)
    state->use_vdsp = true;
#endif

    backend->impl = state;
    backend->type = SIM_BACKEND_TYPE_CPU;
    backend_enable_feature(backend, SIM_BACKEND_FEATURE_ANALYTIC_WARP);
    backend_enable_feature(backend, SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS);
    return SIM_RESULT_OK;
}

/**
 * @brief Locate a field binding by KernelIR field index.
 *
 * @param kernel Kernel package containing bindings.
 * @param field_index Field identifier referenced by IR nodes.
 * @return Bound field pointer, or NULL when absent.
 */
static SimField* backend_cpu_find_field(const KernelIR* kernel, size_t field_index) {
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
 *
 * @param field Field to inspect.
 * @return Field byte size divided by element size, or 0 for invalid metadata.
 */
static size_t backend_cpu_element_count(const SimField* field) {
    size_t bytes;

    if (field == NULL || field->element_size == 0U) {
        return 0U;
    }

    bytes = sim_field_bytes(field);
    return (bytes / field->element_size);
}

/**
 * @brief Read a scalar double field value for a KernelIR field reference.
 *
 * @param kernel Kernel package containing bindings.
 * @param element_index Element to read.
 * @param field_index Field identifier referenced by the IR.
 * @param[out] out_value Receives the field value.
 * @return #SIM_RESULT_OK, #SIM_RESULT_NOT_FOUND, or #SIM_RESULT_INVALID_ARGUMENT.
 */
static SimResult backend_cpu_field_value(const KernelIR* kernel,
                                         size_t          element_index,
                                         size_t          field_index,
                                         double*         out_value) {
    const SimField* field;
    const double*   data;
    size_t          count;

    if (kernel == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = backend_cpu_find_field(kernel, field_index);
    if (field == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    if (field->element_size != sizeof(double)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = backend_cpu_element_count(field);
    if (element_index >= count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    data = (const double*) sim_field_data_const(field);
    if (data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_value = data[element_index];
    return SIM_RESULT_OK;
}

/**
 * @brief Read one numeric component from a real, complex, or double-vector field.
 *
 * Complex fields expose component 0 as real and component 1 as imaginary.
 * Vector-like double storage exposes components by element-strided lane index.
 *
 * @param kernel Kernel package containing bindings.
 * @param element_index Element to read.
 * @param field_index Field identifier referenced by the IR.
 * @param component Component lane to read.
 * @param[out] out_value Receives the component value.
 * @return #SIM_RESULT_OK, #SIM_RESULT_NOT_FOUND, or #SIM_RESULT_INVALID_ARGUMENT.
 */
static SimResult backend_cpu_field_value_component(const KernelIR* kernel,
                                                   size_t          element_index,
                                                   size_t          field_index,
                                                   size_t          component,
                                                   double*         out_value) {
    const SimField*         field;
    const double*           data;
    const SimComplexDouble* cdata;
    size_t                  count;

    if (kernel == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = backend_cpu_find_field(kernel, field_index);
    if (field == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    if (field->element_size == sizeof(double)) {
        count = backend_cpu_element_count(field);
        if (element_index >= count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        data = (const double*) sim_field_data_const(field);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        *out_value = data[element_index];
        return SIM_RESULT_OK;
    } else if (field->element_size == sizeof(double) * 2U) {
        count = backend_cpu_element_count(field);
        if (element_index >= count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        cdata = (const SimComplexDouble*) sim_field_data_const(field);
        if (cdata == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (component == 0U) {
            *out_value = cdata[element_index].re;
        } else if (component == 1U) {
            *out_value = cdata[element_index].im;
        } else {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        return SIM_RESULT_OK;
    } else if (field->element_size % sizeof(double) == 0U) {
        size_t comps = field->element_size / sizeof(double);
        count        = backend_cpu_element_count(field);
        if (element_index >= count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        data = (const double*) sim_field_data_const(field);
        if (data == NULL) {
            fprintf(stderr,
                    "[DEBUG] backend_cpu_field_value_component: data NULL for field %p\n",
                    (void*) field);
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (component >= comps) {
            fprintf(stderr,
                    "[DEBUG] backend_cpu_field_value_component: component %zu >= comps %zu for "
                    "field %p element_size=%zu\n",
                    component,
                    comps,
                    (void*) field,
                    field->element_size);
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        *out_value = data[element_index * comps + component];
        return SIM_RESULT_OK;
    }

    fprintf(
        stderr,
        "[DEBUG] backend_cpu_field_value_component: unsupported element_size=%zu for field %p\n",
        field->element_size,
        (void*) field);
    return SIM_RESULT_INVALID_ARGUMENT;
}

/**
 * @brief Evaluate a coordinate node for one element.
 *
 * Coordinates are derived from binding shape/stride metadata. Missing layout
 * metadata falls back to `axis 0 == element_index` and zero for other axes.
 *
 * @param kernel Kernel package containing bindings.
 * @param element_index Linear element index.
 * @param field_index Field whose layout defines the coordinate.
 * @param axis Axis requested by the IR coordinate node.
 * @param[out] out_value Receives the coordinate as a double.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
static SimResult backend_cpu_coord_value(const KernelIR* kernel,
                                         size_t          element_index,
                                         size_t          field_index,
                                         size_t          axis,
                                         double*         out_value) {
    const SimKernelIRBinding* binding = NULL;

    if (kernel == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (kernel->bindings != NULL) {
        for (size_t i = 0U; i < kernel->binding_count; ++i) {
            if (kernel->bindings[i].field_index == field_index) {
                binding = &kernel->bindings[i];
                break;
            }
        }
    }

    if (binding == NULL || binding->shape == NULL || binding->strides == NULL ||
        axis >= binding->rank) {
        if (axis == 0U) {
            *out_value = (double) element_index;
            return SIM_RESULT_OK;
        }
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    size_t extent = binding->shape[axis];
    size_t stride = binding->strides[axis];
    if (extent == 0U || stride == 0U) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    size_t coord = (element_index / stride) % extent;
    *out_value   = (double) coord;
    return SIM_RESULT_OK;
}

/**
 * @brief Return true when a scalar-domain descriptor represents integer storage.
 */
static bool backend_cpu_domain_is_integer(SimScalarDomain domain) {
    return sim_scalar_domain_is_integer(domain);
}

/**
 * @brief Build the low-bit mask for an integer scalar domain.
 *
 * Domains with bit width 64 use all bits.
 */
static uint64_t backend_cpu_integer_mask(SimScalarDomain domain) {
    if (domain.bit_width >= 64U) {
        return UINT64_MAX;
    }
    return (UINT64_C(1) << domain.bit_width) - UINT64_C(1);
}

/**
 * @brief Truncate raw integer bits to the target scalar-domain width.
 */
static uint64_t backend_cpu_integer_truncate(uint64_t raw, SimScalarDomain domain) {
    if (domain.bit_width >= 64U) {
        return raw;
    }
    return raw & backend_cpu_integer_mask(domain);
}

/**
 * @brief Interpret raw integer-domain bits as a signed 64-bit value.
 *
 * Narrow signed domains are sign-extended from their declared bit width.
 */
static int64_t backend_cpu_integer_as_i64(uint64_t raw, SimScalarDomain domain) {
    raw = backend_cpu_integer_truncate(raw, domain);
    if (!domain.is_signed || domain.bit_width >= 64U) {
        return (int64_t) raw;
    }

    uint64_t mask     = backend_cpu_integer_mask(domain);
    uint64_t sign_bit = UINT64_C(1) << (domain.bit_width - 1U);
    if ((raw & sign_bit) != 0U) {
        raw |= ~mask;
    }
    return (int64_t) raw;
}

/**
 * @brief Convert an exact double value into raw integer-domain bits.
 *
 * Fractional, non-finite, out-of-range, and non-integer-domain inputs are
 * rejected. Successful conversions are truncated to the declared domain width.
 */
static bool
backend_cpu_integer_raw_from_double(double value, SimScalarDomain domain, uint64_t* out_raw) {
    if (out_raw == NULL || !backend_cpu_domain_is_integer(domain) || !isfinite(value) ||
        trunc(value) != value) {
        return false;
    }

    if (domain.is_signed) {
        if (domain.bit_width == 32U && (value < (double) INT32_MIN || value > (double) INT32_MAX)) {
            return false;
        }
        if (domain.bit_width == 64U && (value < (double) INT64_MIN || value > (double) INT64_MAX)) {
            return false;
        }
        *out_raw = backend_cpu_integer_truncate((uint64_t) ((int64_t) value), domain);
        return true;
    }

    if (value < 0.0) {
        return false;
    }
    if (domain.bit_width == 32U && value > (double) UINT32_MAX) {
        return false;
    }
    if (domain.bit_width == 64U && value > (double) UINT64_MAX) {
        return false;
    }
    *out_raw = backend_cpu_integer_truncate((uint64_t) value, domain);
    return true;
}

/**
 * @brief Read an integer field element as raw domain bits.
 *
 * The field's scalar-domain descriptor and storage width must match the
 * requested domain exactly.
 *
 * @return #SIM_RESULT_OK, #SIM_RESULT_NOT_FOUND, #SIM_RESULT_TYPE_MISMATCH, or
 * #SIM_RESULT_INVALID_ARGUMENT.
 */
static SimResult backend_cpu_field_value_integer(const KernelIR* kernel,
                                                 size_t          element_index,
                                                 size_t          field_index,
                                                 SimScalarDomain domain,
                                                 uint64_t*       out_raw) {
    const SimField* field;
    size_t          count;

    if (kernel == NULL || out_raw == NULL || !backend_cpu_domain_is_integer(domain)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = backend_cpu_find_field(kernel, field_index);
    if (field == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }
    if (!sim_scalar_domain_equal(sim_scalar_domain_from_field(field), domain) ||
        !sim_field_storage_matches_scalar_domain(field->element_size, domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    count = backend_cpu_element_count(field);
    if (element_index >= count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    switch (domain.bit_width) {
        case 32U:
            if (domain.is_signed) {
                const int32_t* data = sim_field_i32_data_const(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                *out_raw = backend_cpu_integer_truncate((uint64_t) data[element_index], domain);
            } else {
                const uint32_t* data = sim_field_u32_data_const(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                *out_raw = backend_cpu_integer_truncate((uint64_t) data[element_index], domain);
            }
            return SIM_RESULT_OK;
        case 64U:
            if (domain.is_signed) {
                const int64_t* data = sim_field_i64_data_const(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                *out_raw = (uint64_t) data[element_index];
            } else {
                const uint64_t* data = sim_field_u64_data_const(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                *out_raw = data[element_index];
            }
            return SIM_RESULT_OK;
        default:
            return SIM_RESULT_INVALID_ARGUMENT;
    }
}

/**
 * @brief Evaluate a coordinate node into an integer scalar domain.
 *
 * Coordinate values must be exactly representable in the requested integer
 * domain.
 */
static SimResult backend_cpu_coord_value_integer(const KernelIR* kernel,
                                                 size_t          element_index,
                                                 size_t          field_index,
                                                 size_t          axis,
                                                 SimScalarDomain domain,
                                                 uint64_t*       out_raw) {
    double coord = 0.0;
    if (out_raw == NULL || !backend_cpu_domain_is_integer(domain)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    SimResult rc = backend_cpu_coord_value(kernel, element_index, field_index, axis, &coord);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    if (!backend_cpu_integer_raw_from_double(coord, domain, out_raw)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    return SIM_RESULT_OK;
}

/**
 * @brief Store raw integer-domain bits into a destination integer field.
 *
 * The destination field must already use the same scalar-domain descriptor.
 */
static SimResult backend_cpu_store_integer_value(SimField*       field,
                                                 size_t          element_index,
                                                 SimScalarDomain domain,
                                                 uint64_t        raw) {
    if (field == NULL || !backend_cpu_domain_is_integer(domain) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(field), domain)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    raw = backend_cpu_integer_truncate(raw, domain);
    switch (domain.bit_width) {
        case 32U:
            if (domain.is_signed) {
                int32_t* data = sim_field_i32_data(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                data[element_index] = (int32_t) backend_cpu_integer_as_i64(raw, domain);
            } else {
                uint32_t* data = sim_field_u32_data(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                data[element_index] = (uint32_t) raw;
            }
            return SIM_RESULT_OK;
        case 64U:
            if (domain.is_signed) {
                int64_t* data = sim_field_i64_data(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                data[element_index] = backend_cpu_integer_as_i64(raw, domain);
            } else {
                uint64_t* data = sim_field_u64_data(field);
                if (data == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                data[element_index] = raw;
            }
            return SIM_RESULT_OK;
        default:
            return SIM_RESULT_INVALID_ARGUMENT;
    }
}

/**
 * @brief Advance the deterministic xorshift state used by CPU noise nodes.
 */
static uint32_t backend_cpu_noise_step(uint32_t state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

/**
 * @brief Evaluate a deterministic noise IR node for one element/component.
 *
 * The stream is keyed by the node seed, element index, component, and optional
 * `SIM_IR_PARAM_STEP_INDEX` runtime parameter. Stochastic calculus law handling
 * remains at the integrator level; this helper only produces the node sample.
 */
static double backend_cpu_noise_sample(const KernelIR*  kernel,
                                       const SimIRNode* node,
                                       size_t           index,
                                       size_t           component) {
    uint32_t state;
    double   amplitude;
    double   sample = 0.0;
    uint32_t step   = 0U;

    if (node == NULL) {
        return 0.0;
    }

    amplitude = node->data.noise.amplitude;
    if (kernel != NULL && kernel->params != NULL && kernel->param_count > SIM_IR_PARAM_STEP_INDEX) {
        double step_value = kernel->params[SIM_IR_PARAM_STEP_INDEX];
        if (isfinite(step_value) && step_value > 0.0) {
            step = (uint32_t) step_value;
        }
    }
    state = node->data.noise.seed ^ (uint32_t) index;
    state ^= (uint32_t) (component + 1U) * 0x9E3779B9u;
    state ^= step * 0x85EBCA6Bu;

    switch (node->data.noise.distribution) {
        case SIM_IR_NOISE_DISTRIBUTION_GAUSSIAN: {
            double u1;
            double u2;
            double r;
            state = backend_cpu_noise_step(state);
            u1    = ((double) state + 1.0) / ((double) UINT32_MAX + 1.0);
            state = backend_cpu_noise_step(state ^ 0xA511E9B3UL);
            u2    = ((double) state + 1.0) / ((double) UINT32_MAX + 1.0);
            if (u1 <= 1.0e-12) {
                u1 = 1.0e-12;
            }
            r      = sqrt(-2.0 * log(u1));
            sample = amplitude * r * cos(2.0 * (double) M_PI * u2);
            break;
        }

        case SIM_IR_NOISE_DISTRIBUTION_UNIFORM:
        default:
            state  = backend_cpu_noise_step(state);
            sample = amplitude * (((double) state / (double) UINT32_MAX) * 2.0 - 1.0);
            break;
    }

    (void) node->data.noise.law; /* Law selection handled at integrator level. */
    return sample;
}

/**
 * @brief Free all heap storage owned by a reusable evaluation frame.
 */
static void backend_cpu_frame_destroy(SimBackendCPUFrame* frame) {
    if (frame == NULL) {
        return;
    }
    free(frame->scratch);
    free(frame->flags);
    free(frame->flag_generation);
    frame->scratch            = NULL;
    frame->flags              = NULL;
    frame->flag_generation    = NULL;
    frame->capacity           = 0U;
    frame->components         = 0U;
    frame->generation_counter = 0U;
}

/**
 * @brief Resize a frame's node/component cache arrays.
 *
 * Existing cached values are discarded because frames are generation-based and
 * only valid within a launch traversal.
 *
 * @return True on success, false on allocation failure.
 */
static bool
backend_cpu_frame_resize(SimBackendCPUFrame* frame, size_t capacity, size_t components) {
    double*        scratch;
    unsigned char* flags;
    uint32_t*      flag_generation;
    size_t         slots;

    if (frame == NULL) {
        return false;
    }

    if (capacity == 0U) {
        backend_cpu_frame_destroy(frame);
        return true;
    }

    slots = capacity * components;

    /* allocate storage for capacity * components double values */
    scratch         = (double*) malloc(slots * sizeof(double));
    flags           = (unsigned char*) calloc(slots, sizeof(unsigned char));
    flag_generation = (uint32_t*) calloc(slots, sizeof(uint32_t));
    if (scratch == NULL || flags == NULL || flag_generation == NULL) {
        free(scratch);
        free(flags);
        free(flag_generation);
        return false;
    }

    free(frame->scratch);
    free(frame->flags);
    free(frame->flag_generation);
    frame->scratch            = scratch;
    frame->flags              = flags;
    frame->flag_generation    = flag_generation;
    frame->capacity           = capacity;
    frame->components         = components;
    frame->generation_counter = 1U;
    return true;
}

/**
 * @brief Ensure every allocated frame can hold the requested node/component grid.
 */
static bool
backend_cpu_ensure_capacity(SimBackendCPUState* state, size_t required, size_t components) {
    size_t i;

    if (state == NULL) {
        return false;
    }

    if (required <= state->capacity && components <= state->component_capacity) {
        return true;
    }

    for (i = 0; i < state->frame_capacity; ++i) {
        if (state->frames[i] != NULL) {
            if (!backend_cpu_frame_resize(state->frames[i], required, components)) {
                return false;
            }
        }
    }

    state->capacity           = required;
    state->component_capacity = components;
    return true;
}

/**
 * @brief Borrow a reusable evaluation frame from the CPU frame stack.
 *
 * Nested derivative evaluations use additional frames to avoid clobbering the
 * caller's recursion flags and scratch values.
 */
static SimBackendCPUFrame*
backend_cpu_frame_acquire(SimBackendCPUState* state, size_t required, size_t components) {
    SimBackendCPUFrame* frame;

    if (state == NULL) {
        return NULL;
    }

    if (!backend_cpu_ensure_capacity(state, required, components)) {
        return NULL;
    }

    if (state->frame_count == state->frame_capacity) {
        size_t               new_capacity = state->frame_capacity + 1U;
        SimBackendCPUFrame** frames;

        frames = (SimBackendCPUFrame**) realloc(state->frames,
                                                new_capacity * sizeof(SimBackendCPUFrame*));
        if (frames == NULL) {
            return NULL;
        }

        state->frames                    = frames;
        state->frames[new_capacity - 1U] = NULL;
        state->frame_capacity            = new_capacity;
    }

    frame = state->frames[state->frame_count];
    if (frame == NULL) {
        frame = (SimBackendCPUFrame*) calloc(1U, sizeof(SimBackendCPUFrame));
        if (frame == NULL) {
            return NULL;
        }
        state->frames[state->frame_count] = frame;
    }

    if (state->capacity > 0U && frame->scratch == NULL) {
        if (!backend_cpu_frame_resize(frame, state->capacity, state->component_capacity)) {
            return NULL;
        }
    }

    /* Ensure components match desired value; if not, resize appropriately */
    if (frame->components != components) {
        if (!backend_cpu_frame_resize(frame, state->capacity, components)) {
            return NULL;
        }
    }

    state->frame_count += 1U;
    return frame;
}

/**
 * @brief Determine the maximum component count needed by nodes in a KernelIR graph.
 */
static size_t backend_cpu_builder_max_components(const KernelIR* kernel) {
    const SimIRBuilder* builder;
    size_t              i;
    size_t              max = 1U;

    if (kernel == NULL || kernel->builder == NULL) {
        return 1U;
    }
    builder = kernel->builder;
    for (i = 0U; i < builder->count; ++i) {
        const SimIRNode* node = &builder->nodes[i];
        if (node->value_type.components > max) {
            max = node->value_type.components;
        }
    }
    return (max < 1U) ? 1U : max;
}

/**
 * @brief Return the topmost frame to the reusable frame stack.
 */
static void backend_cpu_frame_release(SimBackendCPUState* state) {
    if (state == NULL || state->frame_count == 0U) {
        return;
    }
    state->frame_count -= 1U;
}

/**
 * @brief Advance a frame generation and lazily invalidate per-node flags.
 *
 * When the 32-bit generation counter wraps, all stamps are cleared and the
 * counter restarts at one.
 */
static void backend_cpu_frame_next_generation(SimBackendCPUFrame* frame) {
    if (frame == NULL) {
        return;
    }

    if (frame->flag_generation == NULL) {
        if (frame->flags != NULL && frame->capacity > 0U && frame->components > 0U) {
            size_t slots = frame->capacity * frame->components;
            (void) memset(frame->flags, 0, slots * sizeof(unsigned char));
        }
        return;
    }

    frame->generation_counter += 1U;
    if (frame->generation_counter == 0U) {
        size_t slots = frame->capacity * frame->components;
        (void) memset(frame->flag_generation, 0, slots * sizeof(uint32_t));
        frame->generation_counter = 1U;
    }
}

/**
 * @brief Recursively evaluate one integer-domain IR node.
 *
 * Integer arithmetic follows fixed-width wrap/truncation semantics for the
 * node's scalar domain. Division by zero and negative integer exponents are
 * reported as invalid arguments. The per-node cache detects cycles and stores
 * raw domain bits for already evaluated nodes.
 *
 * @param kernel Kernel package containing the IR builder and field bindings.
 * @param element_index Element being evaluated.
 * @param node_id Node to evaluate.
 * @param cache Per-node raw-bit cache.
 * @param flags Per-node evaluation flags.
 * @param[out] out_raw Receives raw bits truncated to the node's domain.
 * @return #SIM_RESULT_OK or an error from validation, dependencies, or field reads.
 */
static SimResult backend_cpu_eval_integer_node(const KernelIR* kernel,
                                               size_t          element_index,
                                               SimIRNodeId     node_id,
                                               uint64_t*       cache,
                                               unsigned char*  flags,
                                               uint64_t*       out_raw) {
    const SimIRBuilder* builder;
    const SimIRNode*    node;
    SimScalarDomain     domain;
    uint64_t            raw    = 0U;
    SimResult           result = SIM_RESULT_OK;

    if (kernel == NULL || kernel->builder == NULL || out_raw == NULL || cache == NULL ||
        flags == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    builder = kernel->builder;
    if (node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (flags[node_id] == 2U) {
        *out_raw = cache[node_id];
        return SIM_RESULT_OK;
    }
    if (flags[node_id] == 1U) {
        return SIM_RESULT_DEPENDENCY_ERROR;
    }
    flags[node_id] = 1U;

    node   = &builder->nodes[node_id];
    domain = sim_ir_type_scalar_domain(node->value_type);
    if (!sim_ir_type_is_scalar(node->value_type) || !backend_cpu_domain_is_integer(domain)) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto done;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
            if (node->data.constant.exact_integer) {
                raw = backend_cpu_integer_truncate(node->data.constant.unsigned_scalar, domain);
            } else if (!backend_cpu_integer_raw_from_double(
                           node->data.constant.scalar, domain, &raw)) {
                result = SIM_RESULT_INVALID_ARGUMENT;
            }
            break;

        case SIM_IR_NODE_FIELD_REF:
            result = backend_cpu_field_value_integer(
                kernel, element_index, node->data.field, domain, &raw);
            break;

        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV:
        case SIM_IR_NODE_MOD:
        case SIM_IR_NODE_POW: {
            uint64_t lhs_raw = 0U;
            uint64_t rhs_raw = 0U;

            result = backend_cpu_eval_integer_node(
                kernel, element_index, node->data.binary.lhs, cache, flags, &lhs_raw);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = backend_cpu_eval_integer_node(
                kernel, element_index, node->data.binary.rhs, cache, flags, &rhs_raw);
            if (result != SIM_RESULT_OK) {
                break;
            }

            switch (node->type) {
                case SIM_IR_NODE_ADD:
                    raw = backend_cpu_integer_truncate(lhs_raw + rhs_raw, domain);
                    break;
                case SIM_IR_NODE_SUB:
                    raw = backend_cpu_integer_truncate(lhs_raw - rhs_raw, domain);
                    break;
                case SIM_IR_NODE_MUL:
                    raw = backend_cpu_integer_truncate(lhs_raw * rhs_raw, domain);
                    break;
                case SIM_IR_NODE_DIV:
                case SIM_IR_NODE_MOD:
                    if (backend_cpu_integer_truncate(rhs_raw, domain) == 0U) {
                        result = SIM_RESULT_INVALID_ARGUMENT;
                        break;
                    }
                    if (domain.is_signed) {
                        int64_t lhs = backend_cpu_integer_as_i64(lhs_raw, domain);
                        int64_t rhs = backend_cpu_integer_as_i64(rhs_raw, domain);
                        if (node->type == SIM_IR_NODE_DIV) {
                            if (lhs == INT64_MIN && rhs == -1 && domain.bit_width == 64U) {
                                raw = (uint64_t) INT64_MIN;
                            } else {
                                raw = backend_cpu_integer_truncate((uint64_t) (lhs / rhs), domain);
                            }
                        } else {
                            raw = backend_cpu_integer_truncate(
                                (uint64_t) ((rhs == -1) ? 0 : (lhs % rhs)), domain);
                        }
                    } else {
                        uint64_t lhs = backend_cpu_integer_truncate(lhs_raw, domain);
                        uint64_t rhs = backend_cpu_integer_truncate(rhs_raw, domain);
                        raw          = (node->type == SIM_IR_NODE_DIV) ? (lhs / rhs) : (lhs % rhs);
                    }
                    break;
                case SIM_IR_NODE_POW: {
                    uint64_t exponent = 0U;
                    uint64_t base     = backend_cpu_integer_truncate(lhs_raw, domain);
                    uint64_t accum    = backend_cpu_integer_truncate(1U, domain);
                    if (domain.is_signed) {
                        int64_t signed_exp = backend_cpu_integer_as_i64(rhs_raw, domain);
                        if (signed_exp < 0) {
                            result = SIM_RESULT_INVALID_ARGUMENT;
                            break;
                        }
                        exponent = (uint64_t) signed_exp;
                    } else {
                        exponent = backend_cpu_integer_truncate(rhs_raw, domain);
                    }
                    while (exponent > 0U) {
                        if ((exponent & 1U) != 0U) {
                            accum = backend_cpu_integer_truncate(accum * base, domain);
                        }
                        exponent >>= 1U;
                        if (exponent > 0U) {
                            base = backend_cpu_integer_truncate(base * base, domain);
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
            result = backend_cpu_eval_integer_node(
                kernel, element_index, node->data.unary.operand, cache, flags, &raw);
            break;

        case SIM_IR_NODE_PARAM: {
            if (kernel->params == NULL || node->data.param.param >= kernel->param_count) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            if (!backend_cpu_integer_raw_from_double(
                    kernel->params[node->data.param.param], domain, &raw)) {
                result = SIM_RESULT_TYPE_MISMATCH;
            }
            break;
        }

        case SIM_IR_NODE_INDEX:
            raw = backend_cpu_integer_truncate((uint64_t) element_index, domain);
            break;

        case SIM_IR_NODE_COORD:
            result = backend_cpu_coord_value_integer(
                kernel, element_index, node->data.coord.field, node->data.coord.axis, domain, &raw);
            break;

        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

done:
    if (result == SIM_RESULT_OK) {
        cache[node_id] = backend_cpu_integer_truncate(raw, domain);
        flags[node_id] = 2U;
        *out_raw       = cache[node_id];
    } else {
        flags[node_id] = 0U;
    }
    return result;
}

/**
 * @brief Execute one integer-domain KernelIR output over its destination field.
 *
 * This path is separate from floating evaluation so integer fields preserve
 * exact domain width, signedness, and wrap behavior.
 *
 * @param kernel Kernel package containing IR, bindings, and outputs.
 * @param output Output mapping to execute.
 * @param dest_field Destination integer field.
 * @return #SIM_RESULT_OK, #SIM_RESULT_TYPE_MISMATCH, #SIM_RESULT_OUT_OF_MEMORY,
 * or an error propagated by integer node evaluation.
 */
static SimResult backend_cpu_execute_integer_output(const KernelIR*          kernel,
                                                    const SimKernelIROutput* output,
                                                    SimField*                dest_field) {
    const SimIRBuilder* builder;
    SimScalarDomain     domain;
    size_t              element_count;
    uint64_t*           cache  = NULL;
    unsigned char*      flags  = NULL;
    SimResult           result = SIM_RESULT_OK;

    if (kernel == NULL || output == NULL || dest_field == NULL || kernel->builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    builder = kernel->builder;
    if (output->expression == SIM_IR_INVALID_NODE || output->expression >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    domain = sim_ir_type_scalar_domain(builder->nodes[output->expression].value_type);
    if (!backend_cpu_domain_is_integer(domain) ||
        !sim_scalar_domain_equal(sim_scalar_domain_from_field(dest_field), domain) ||
        !sim_field_storage_matches_scalar_domain(dest_field->element_size, domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    element_count = backend_cpu_element_count(dest_field);
    if (element_count == 0U) {
        return SIM_RESULT_OK;
    }

    cache = (uint64_t*) calloc(builder->count, sizeof(uint64_t));
    flags = (unsigned char*) calloc(builder->count, sizeof(unsigned char));
    if (cache == NULL || flags == NULL) {
        free(cache);
        free(flags);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    for (size_t element_index = 0U; element_index < element_count; ++element_index) {
        uint64_t value_raw = 0U;
        memset(flags, 0, builder->count * sizeof(unsigned char));
        result = backend_cpu_eval_integer_node(
            kernel, element_index, output->expression, cache, flags, &value_raw);
        if (result != SIM_RESULT_OK) {
            break;
        }
        result = backend_cpu_store_integer_value(dest_field, element_index, domain, value_raw);
        if (result != SIM_RESULT_OK) {
            break;
        }
    }

    free(cache);
    free(flags);
    return result;
}

static SimResult backend_cpu_eval_node(const KernelIR*     kernel,
                                       size_t              element_index,
                                       SimBackendCPUState* state,
                                       SimBackendCPUFrame* frame,
                                       const SimField*     reference_field,
                                       SimIRNodeId         node_id,
                                       size_t              component,
                                       double*             out_value);

static BackendCPUComponentKernelDesc backend_cpu_select_component_kernel(size_t component_limit);
static SimResult                     backend_cpu_eval_components_kernel1(const KernelIR*     kernel,
                                                                         size_t              element_index,
                                                                         SimBackendCPUState* state,
                                                                         SimBackendCPUFrame* frame,
                                                                         const SimField*     reference_field,
                                                                         SimIRNodeId         node_id,
                                                                         double*             row,
                                                                         size_t              component_limit);
static SimResult                     backend_cpu_eval_components_kernel2(const KernelIR*     kernel,
                                                                         size_t              element_index,
                                                                         SimBackendCPUState* state,
                                                                         SimBackendCPUFrame* frame,
                                                                         const SimField*     reference_field,
                                                                         SimIRNodeId         node_id,
                                                                         double*             row,
                                                                         size_t              component_limit);
static SimResult                     backend_cpu_eval_components_kernel3(const KernelIR*     kernel,
                                                                         size_t              element_index,
                                                                         SimBackendCPUState* state,
                                                                         SimBackendCPUFrame* frame,
                                                                         const SimField*     reference_field,
                                                                         SimIRNodeId         node_id,
                                                                         double*             row,
                                                                         size_t              component_limit);
static SimResult                     backend_cpu_eval_components_kernel4(const KernelIR*     kernel,
                                                                         size_t              element_index,
                                                                         SimBackendCPUState* state,
                                                                         SimBackendCPUFrame* frame,
                                                                         const SimField*     reference_field,
                                                                         SimIRNodeId         node_id,
                                                                         double*             row,
                                                                         size_t              component_limit);
static SimResult backend_cpu_eval_components_kernel_generic(const KernelIR*     kernel,
                                                            size_t              element_index,
                                                            SimBackendCPUState* state,
                                                            SimBackendCPUFrame* frame,
                                                            const SimField*     reference_field,
                                                            SimIRNodeId         node_id,
                                                            double*             row,
                                                            size_t              component_limit);

static bool          backend_cpu_constant_scalar(const SimIRNode* node, double* out_value);
static const double* backend_cpu_constant_vector_data(const SimIRBuilder* builder,
                                                      const SimIRNode*    node,
                                                      size_t*             components);
static bool          backend_cpu_build_operand(const KernelIR*     kernel,
                                               const SimIRBuilder* builder,
                                               const SimIRNode*    node,
                                               size_t              element_count,
                                               size_t              element_size,
                                               size_t              dest_components,
                                               BackendCPUOperand*  operand);
static double
backend_cpu_operand_value(const BackendCPUOperand* operand, size_t idx, size_t dest_components);
static bool backend_cpu_try_simple_field_op(const KernelIR*          kernel,
                                            const SimKernelIROutput* output,
                                            SimBackendCPUState*      state,
                                            SimField*                dest_field,
                                            double*                  dest_data,
                                            size_t                   element_count);

/**
 * @brief Convert generic n-dimensional neighbor metadata to CPU axis neighbors.
 */
static SimResult backend_cpu_axis_neighbors(const SimField*             field,
                                            size_t                      element_index,
                                            size_t                      axis,
                                            SimIRBoundaryPolicy         boundary,
                                            SimBackendCPUAxisNeighbors* neighbors) {
    if (field == NULL || neighbors == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimNdAxisNeighbors nd_neighbors = { 0 };
    SimResult result = sim_nd_axis_neighbors(field, element_index, axis, boundary, &nd_neighbors);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    neighbors->has_forward      = nd_neighbors.forward.valid;
    neighbors->has_backward     = nd_neighbors.backward.valid;
    neighbors->forward_index    = nd_neighbors.forward.index;
    neighbors->backward_index   = nd_neighbors.backward.index;
    neighbors->forward_wrapped  = nd_neighbors.forward.wrapped;
    neighbors->backward_wrapped = nd_neighbors.backward.wrapped;
    return SIM_RESULT_OK;
}

/**
 * @brief Evaluate a node at another element index using a nested frame.
 *
 * Differential nodes use this helper for forward/backward neighbor samples so
 * recursive cache state from the center element stays isolated.
 */
static SimResult backend_cpu_eval_node_at_index(const KernelIR*     kernel,
                                                size_t              element_index,
                                                SimBackendCPUState* state,
                                                const SimField*     reference_field,
                                                SimIRNodeId         node_id,
                                                size_t              component,
                                                double*             out_value) {
    SimBackendCPUFrame* frame;
    SimResult           result;

    if (kernel == NULL || state == NULL || out_value == NULL || kernel->builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t max_components = backend_cpu_builder_max_components(kernel);
    frame = backend_cpu_frame_acquire(state, kernel->builder->count, max_components);
    if (frame == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    backend_cpu_frame_next_generation(frame);

    result = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, component, out_value);

    backend_cpu_frame_release(state);
    return result;
}

/**
 * @brief Recursively evaluate one floating/vector/complex IR node component.
 *
 * The evaluator handles componentwise vector lanes, true complex arithmetic for
 * complex scalar domains, finite differences with boundary policy semantics,
 * analytic warp samples, deterministic noise nodes, and stateful callback nodes.
 * Results are cached per `(node_id, component)` for the current frame generation.
 *
 * @param kernel Kernel package containing IR, bindings, and params.
 * @param element_index Element being evaluated.
 * @param state CPU backend state for nested frame allocation.
 * @param frame Active evaluation cache frame.
 * @param reference_field Destination field used for differential layout.
 * @param node_id Node to evaluate.
 * @param component Component lane to evaluate.
 * @param[out] out_value Receives the computed component value.
 * @return #SIM_RESULT_OK or an error from validation, dependencies, field access,
 * callbacks, or numerical helper policies.
 */
static SimResult backend_cpu_eval_node(const KernelIR*     kernel,
                                       size_t              element_index,
                                       SimBackendCPUState* state,
                                       SimBackendCPUFrame* frame,
                                       const SimField*     reference_field,
                                       SimIRNodeId         node_id,
                                       size_t              component,
                                       double*             out_value) {
    const SimIRBuilder* builder;
    const SimIRNode*    node;
    double              value  = 0.0;
    SimResult           result = SIM_RESULT_OK;

    if (kernel == NULL || state == NULL || frame == NULL || out_value == NULL ||
        kernel->builder == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    builder = kernel->builder;

    if (builder->nodes == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (frame->flags == NULL || frame->scratch == NULL || frame->flag_generation == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    size_t   slot            = node_id * frame->components + component;
    uint32_t slot_generation = frame->flag_generation[slot];

    if (slot_generation == frame->generation_counter && frame->flags[slot] == 2U) {
        *out_value = frame->scratch[slot];
        return SIM_RESULT_OK;
    }

    if (slot_generation == frame->generation_counter && frame->flags[slot] == 1U) {
        return SIM_RESULT_DEPENDENCY_ERROR;
    }

    frame->flag_generation[slot] = frame->generation_counter;
    frame->flags[slot]           = 1U;
    node                         = &builder->nodes[node_id];

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
            if (!sim_ir_type_is_scalar(node->value_type)) {
                /* If small inline constant present (no pool index), use it */
                if (node->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX &&
                    node->value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY) {
                    value = node->data.constant.small[component];
                } else if (node->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX &&
                           builder->constants_data != NULL &&
                           node->data.constant.constant_index < builder->constants_count) {
                    size_t const_index = node->data.constant.constant_index;
                    size_t offset      = builder->constants_offsets[const_index];
                    value              = builder->constants_data[offset + component];
                } else /* fallback to scalar broadcast */
                {
                    value = node->data.constant.scalar;
                }
            } else {
                value = node->data.constant.scalar;
            }
            break;
        case SIM_IR_NODE_FIELD_REF:
            result = backend_cpu_field_value_component(
                kernel, element_index, node->data.field, component, &value);
            break;
        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_MUL:
        case SIM_IR_NODE_DIV: {
            if (sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(node->value_type))) {
                double lhs_re = 0.0;
                double lhs_im = 0.0;
                double rhs_re = 0.0;
                double rhs_im = 0.0;

                if (component > 1U) {
                    result = SIM_RESULT_INVALID_ARGUMENT;
                    break;
                }

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.lhs,
                                               0U,
                                               &lhs_re);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.lhs,
                                               1U,
                                               &lhs_im);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.rhs,
                                               0U,
                                               &rhs_re);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.rhs,
                                               1U,
                                               &rhs_im);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                switch (node->type) {
                    case SIM_IR_NODE_ADD:
                        value = (component == 0U) ? (lhs_re + rhs_re) : (lhs_im + rhs_im);
                        break;
                    case SIM_IR_NODE_SUB:
                        value = (component == 0U) ? (lhs_re - rhs_re) : (lhs_im - rhs_im);
                        break;
                    case SIM_IR_NODE_MUL:
                        if (component == 0U) {
                            value = lhs_re * rhs_re - lhs_im * rhs_im;
                        } else {
                            value = lhs_re * rhs_im + lhs_im * rhs_re;
                        }
                        break;
                    case SIM_IR_NODE_DIV: {
                        double denom = rhs_re * rhs_re + rhs_im * rhs_im;
                        if (component == 0U) {
                            value = (lhs_re * rhs_re + lhs_im * rhs_im) / denom;
                        } else {
                            value = (lhs_im * rhs_re - lhs_re * rhs_im) / denom;
                        }
                        break;
                    }
                    default:
                        break;
                }
            } else {
                double lhs;
                double rhs;

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.lhs,
                                               component,
                                               &lhs);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.rhs,
                                               component,
                                               &rhs);
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
                    default:
                        break;
                }
            }
            break;
        }
        case SIM_IR_NODE_POW: {
            if (sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(node->value_type))) {
                double         lhs_re = 0.0;
                double         lhs_im = 0.0;
                double         rhs_re = 0.0;
                double         rhs_im = 0.0;
                double complex base;
                double complex exponent;
                double complex pow_value;

                if (component > 1U) {
                    result = SIM_RESULT_INVALID_ARGUMENT;
                    break;
                }

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.lhs,
                                               0U,
                                               &lhs_re);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.lhs,
                                               1U,
                                               &lhs_im);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.rhs,
                                               0U,
                                               &rhs_re);
                if (result != SIM_RESULT_OK) {
                    break;
                }
                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.rhs,
                                               1U,
                                               &rhs_im);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                base      = lhs_re + I * lhs_im;
                exponent  = rhs_re + I * rhs_im;
                pow_value = cpow(base, exponent);
                value     = (component == 0U) ? creal(pow_value) : cimag(pow_value);
            } else {
                double lhs = 0.0;
                double rhs = 0.0;

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.lhs,
                                               component,
                                               &lhs);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                result = backend_cpu_eval_node(kernel,
                                               element_index,
                                               state,
                                               frame,
                                               reference_field,
                                               node->data.binary.rhs,
                                               component,
                                               &rhs);
                if (result != SIM_RESULT_OK) {
                    break;
                }

                value = pow(lhs, rhs);
            }
            break;
        }
        case SIM_IR_NODE_DIFF: {
            SimBackendCPUAxisNeighbors neighbors      = { 0 };
            double                     center_value   = 0.0;
            double                     forward_value  = 0.0;
            double                     backward_value = 0.0;
            double                     derivative     = 0.0;
            double                     dx             = node->data.diff.dx;
            SimIRDiffMethod            method         = node->data.diff.method;

            if (dx <= 0.0) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                break;
            }

            if (reference_field == NULL) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                break;
            }

            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.diff.operand,
                                           component,
                                           &center_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            result = backend_cpu_axis_neighbors(reference_field,
                                                element_index,
                                                node->data.diff.axis,
                                                node->data.diff.boundary,
                                                &neighbors);
            if (result != SIM_RESULT_OK) {
                break;
            }

            forward_value  = center_value;
            backward_value = center_value;

            if (neighbors.has_forward) {
                result = backend_cpu_eval_node_at_index(kernel,
                                                        neighbors.forward_index,
                                                        state,
                                                        reference_field,
                                                        node->data.diff.operand,
                                                        component,
                                                        &forward_value);
                if (result != SIM_RESULT_OK) {
                    break;
                }
            }

            if (neighbors.has_backward) {
                result = backend_cpu_eval_node_at_index(kernel,
                                                        neighbors.backward_index,
                                                        state,
                                                        reference_field,
                                                        node->data.diff.operand,
                                                        component,
                                                        &backward_value);
                if (result != SIM_RESULT_OK) {
                    break;
                }
            }

            if (!neighbors.has_forward) {
                if (node->data.diff.boundary == SIM_IR_BOUNDARY_DIRICHLET) {
                    forward_value = 0.0;
                } else if (node->data.diff.boundary == SIM_IR_BOUNDARY_NEUMANN) {
                    forward_value = center_value;
                }
            }

            if (!neighbors.has_backward) {
                if (node->data.diff.boundary == SIM_IR_BOUNDARY_DIRICHLET) {
                    backward_value = 0.0;
                } else if (node->data.diff.boundary == SIM_IR_BOUNDARY_NEUMANN) {
                    backward_value = center_value;
                }
            }

            switch (method) {
                case SIM_IR_DIFF_METHOD_FORWARD:
                    derivative = (forward_value - center_value) / dx;
                    break;
                case SIM_IR_DIFF_METHOD_BACKWARD:
                    derivative = (center_value - backward_value) / dx;
                    break;
                case SIM_IR_DIFF_METHOD_CENTRAL:
                    derivative = (forward_value - backward_value) / (2.0 * dx);
                    break;
                case SIM_IR_DIFF_METHOD_AUTO:
                default:
                    if (neighbors.has_forward && neighbors.has_backward) {
                        derivative = (forward_value - backward_value) / (2.0 * dx);
                    } else if (neighbors.has_forward) {
                        derivative = (forward_value - center_value) / dx;
                    } else if (neighbors.has_backward) {
                        derivative = (center_value - backward_value) / dx;
                    } else {
                        derivative = 0.0;
                    }
                    break;
            }

            value = derivative * node->data.diff.scale;
            break;
        }
        case SIM_IR_NODE_NOISE:
            value = backend_cpu_noise_sample(kernel, node, element_index, component);
            break;
        case SIM_IR_NODE_PARAM:
            if (kernel->params == NULL || node->data.param.param >= kernel->param_count) {
                result = SIM_RESULT_NOT_FOUND;
                break;
            }
            value = kernel->params[node->data.param.param];
            break;
        case SIM_IR_NODE_INDEX:
            value = (double) element_index;
            break;
        case SIM_IR_NODE_CALL: {
            double operand_value = 0.0;

            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.call.operand,
                                           component,
                                           &operand_value);
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
            result               = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.unary.operand,
                                           component,
                                           &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }
            value = floor(operand_value);
            break;
        }
        case SIM_IR_NODE_MOD: {
            double lhs = 0.0;
            double rhs = 0.0;
            result     = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.binary.lhs,
                                           component,
                                           &lhs);
            if (result != SIM_RESULT_OK) {
                break;
            }
            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.binary.rhs,
                                           component,
                                           &rhs);
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
        case SIM_IR_NODE_COORD:
            result = backend_cpu_coord_value(
                kernel, element_index, node->data.coord.field, node->data.coord.axis, &value);
            break;
        case SIM_IR_NODE_COMPLEX_PACK: {
            if (component > 1U) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                break;
            }
            SimIRNodeId src =
                (component == 0U) ? node->data.complex_pack.real : node->data.complex_pack.imag;
            result = backend_cpu_eval_node(
                kernel, element_index, state, frame, reference_field, src, 0U, &value);
            break;
        }
        case SIM_IR_NODE_COMPLEX_ROTATE: {
            double re_value = 0.0;
            double im_value = 0.0;
            double theta    = 0.0;
            double s        = 0.0;
            double c        = 1.0;

            if (component > 1U) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                break;
            }

            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.complex_rotate.operand,
                                           0U,
                                           &re_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.complex_rotate.operand,
                                           1U,
                                           &im_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.complex_rotate.angle,
                                           0U,
                                           &theta);
            if (result != SIM_RESULT_OK) {
                break;
            }

            s = sin(theta);
            c = cos(theta);

            if (component == 0U) {
                value = re_value * c - im_value * s;
            } else {
                value = re_value * s + im_value * c;
            }
            break;
        }
        case SIM_IR_NODE_WARP: {
            double operand_value = 0.0;

            result = backend_cpu_eval_node(kernel,
                                           element_index,
                                           state,
                                           frame,
                                           reference_field,
                                           node->data.warp.operand,
                                           component,
                                           &operand_value);
            if (result != SIM_RESULT_OK) {
                break;
            }

            SimWarpGuard guard = { .mode      = (SimContinuityMode) node->data.warp.guard.mode,
                                   .clamp_min = node->data.warp.guard.clamp_min,
                                   .clamp_max = node->data.warp.guard.clamp_max,
                                   .tolerance = node->data.warp.guard.tolerance };

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
        case SIM_IR_NODE_STATEFUL: {
            if (node->data.stateful.eval == NULL) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                break;
            }
            result = node->data.stateful.eval(
                node->data.stateful.userdata, kernel, element_index, component, &value);
            break;
        }
        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

    if (result == SIM_RESULT_OK) {
        frame->flags[slot]   = 2U;
        frame->scratch[slot] = value;
        *out_value           = value;
    } else {
        frame->flags[slot] = 0U;
    }

    return result;
}

/**
 * @brief Evaluate a one-component output row.
 */
static SimResult backend_cpu_eval_components_kernel1(const KernelIR*     kernel,
                                                     size_t              element_index,
                                                     SimBackendCPUState* state,
                                                     SimBackendCPUFrame* frame,
                                                     const SimField*     reference_field,
                                                     SimIRNodeId         node_id,
                                                     double*             row,
                                                     size_t              component_limit) {
    (void) component_limit;
    return backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 0U, row);
}

/**
 * @brief Evaluate a two-component output row.
 */
static SimResult backend_cpu_eval_components_kernel2(const KernelIR*     kernel,
                                                     size_t              element_index,
                                                     SimBackendCPUState* state,
                                                     SimBackendCPUFrame* frame,
                                                     const SimField*     reference_field,
                                                     SimIRNodeId         node_id,
                                                     double*             row,
                                                     size_t              component_limit) {
    (void) component_limit;
    SimResult rc = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 0U, &row[0]);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    return backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 1U, &row[1]);
}

/**
 * @brief Evaluate a three-component output row.
 */
static SimResult backend_cpu_eval_components_kernel3(const KernelIR*     kernel,
                                                     size_t              element_index,
                                                     SimBackendCPUState* state,
                                                     SimBackendCPUFrame* frame,
                                                     const SimField*     reference_field,
                                                     SimIRNodeId         node_id,
                                                     double*             row,
                                                     size_t              component_limit) {
    (void) component_limit;
    SimResult rc = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 0U, &row[0]);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    rc = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 1U, &row[1]);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    return backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 2U, &row[2]);
}

/**
 * @brief Evaluate a four-component output row.
 */
static SimResult backend_cpu_eval_components_kernel4(const KernelIR*     kernel,
                                                     size_t              element_index,
                                                     SimBackendCPUState* state,
                                                     SimBackendCPUFrame* frame,
                                                     const SimField*     reference_field,
                                                     SimIRNodeId         node_id,
                                                     double*             row,
                                                     size_t              component_limit) {
    (void) component_limit;
    SimResult rc = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 0U, &row[0]);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    rc = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 1U, &row[1]);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    rc = backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 2U, &row[2]);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    return backend_cpu_eval_node(
        kernel, element_index, state, frame, reference_field, node_id, 3U, &row[3]);
}

/**
 * @brief Evaluate an arbitrary-width component row.
 */
static SimResult backend_cpu_eval_components_kernel_generic(const KernelIR*     kernel,
                                                            size_t              element_index,
                                                            SimBackendCPUState* state,
                                                            SimBackendCPUFrame* frame,
                                                            const SimField*     reference_field,
                                                            SimIRNodeId         node_id,
                                                            double*             row,
                                                            size_t              component_limit) {
    if (component_limit == 0U) {
        return SIM_RESULT_OK;
    }
    for (size_t comp_idx = 0U; comp_idx < component_limit; ++comp_idx) {
        SimResult rc = backend_cpu_eval_node(kernel,
                                             element_index,
                                             state,
                                             frame,
                                             reference_field,
                                             node_id,
                                             comp_idx,
                                             &row[comp_idx]);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
    }
    return SIM_RESULT_OK;
}

/**
 * @brief Select a small fixed-width row evaluator when possible.
 */
static BackendCPUComponentKernelDesc backend_cpu_select_component_kernel(size_t component_limit) {
    BackendCPUComponentKernelDesc desc = { 0 };
    desc.component_limit               = component_limit;
    switch (component_limit) {
        case 0U:
            desc.eval = backend_cpu_eval_components_kernel_generic;
            break;
        case 1U:
            desc.eval = backend_cpu_eval_components_kernel1;
            break;
        case 2U:
            desc.eval = backend_cpu_eval_components_kernel2;
            break;
        case 3U:
            desc.eval = backend_cpu_eval_components_kernel3;
            break;
        case 4U:
            desc.eval = backend_cpu_eval_components_kernel4;
            break;
        default:
            desc.eval = backend_cpu_eval_components_kernel_generic;
            break;
    }
    return desc;
}

#if defined(SIM_HAVE_VDSP)
/**
 * @brief Try a vDSP fast path for binary real field operations.
 *
 * This path accepts direct field-ref binary roots over double fields of equal
 * length. Unsupported expressions return false so the scalar evaluator can run.
 */
static bool backend_cpu_try_vdsp(const KernelIR*          kernel,
                                 const SimKernelIROutput* output,
                                 SimBackendCPUState*      state) {
    const SimIRNode*    root;
    const SimIRBuilder* builder;
    const SimIRNode*    lhs_node;
    const SimIRNode*    rhs_node;
    SimField*           dst;
    SimField*           lhs_field;
    SimField*           rhs_field;
    double*             dst_data;
    const double*       lhs_data;
    const double*       rhs_data;
    size_t              element_count;
    vDSP_Length         length;

    (void) state;

    if (kernel == NULL || output == NULL || state == NULL || kernel->builder == NULL) {
        return false;
    }

    if (!state->use_vdsp) {
        return false;
    }

    builder = kernel->builder;

    if (builder->nodes == NULL) {
        return false;
    }

    if (output->expression == SIM_IR_INVALID_NODE || output->expression >= builder->count) {
        return false;
    }

    root = &builder->nodes[output->expression];
    if (!(root->type == SIM_IR_NODE_ADD || root->type == SIM_IR_NODE_SUB ||
          root->type == SIM_IR_NODE_MUL || root->type == SIM_IR_NODE_DIV)) {
        return false;
    }

    if (root->data.binary.lhs >= builder->count || root->data.binary.rhs >= builder->count) {
        return false;
    }

    lhs_node = &builder->nodes[root->data.binary.lhs];
    rhs_node = &builder->nodes[root->data.binary.rhs];

    if (lhs_node->type != SIM_IR_NODE_FIELD_REF || rhs_node->type != SIM_IR_NODE_FIELD_REF) {
        return false;
    }

    dst       = backend_cpu_find_field(kernel, output->field_index);
    lhs_field = backend_cpu_find_field(kernel, lhs_node->data.field);
    rhs_field = backend_cpu_find_field(kernel, rhs_node->data.field);

    if (dst == NULL || lhs_field == NULL || rhs_field == NULL) {
        return false;
    }

    if (dst->element_size != sizeof(double) || lhs_field->element_size != sizeof(double) ||
        rhs_field->element_size != sizeof(double)) {
        return false;
    }

    element_count = backend_cpu_element_count(dst);
    if (element_count == 0U || backend_cpu_element_count(lhs_field) != element_count ||
        backend_cpu_element_count(rhs_field) != element_count) {
        return false;
    }

    dst_data = (double*) sim_field_data(dst);
    lhs_data = (const double*) sim_field_data_const(lhs_field);
    rhs_data = (const double*) sim_field_data_const(rhs_field);

    if (dst_data == NULL || lhs_data == NULL || rhs_data == NULL) {
        return false;
    }

    length = (vDSP_Length) element_count;

    switch (root->type) {
        case SIM_IR_NODE_ADD:
            vDSP_vaddD(lhs_data, 1, rhs_data, 1, dst_data, 1, length);
            return true;
        case SIM_IR_NODE_SUB:
            vDSP_vsubD(rhs_data, 1, lhs_data, 1, dst_data, 1, length);
            return true;
        case SIM_IR_NODE_MUL:
            vDSP_vmulD(lhs_data, 1, rhs_data, 1, dst_data, 1, length);
            return true;
        case SIM_IR_NODE_DIV:
            vDSP_vdivD(rhs_data, 1, lhs_data, 1, dst_data, 1, length);
            return true;
        default:
            break;
    }

    return false;
}
#endif

/**
 * @brief Extract a scalar constant value from an IR node.
 */
static bool backend_cpu_constant_scalar(const SimIRNode* node, double* out_value) {
    if (node == NULL || out_value == NULL) {
        return false;
    }
    if (node->type != SIM_IR_NODE_CONSTANT || !sim_ir_type_is_scalar(node->value_type)) {
        return false;
    }
    *out_value = node->data.constant.scalar;
    return true;
}

/**
 * @brief Return the backing lane data for a vector constant node.
 *
 * Inline small constants return their embedded lane array; pooled constants
 * return an offset into the builder constant pool.
 */
static const double* backend_cpu_constant_vector_data(const SimIRBuilder* builder,
                                                      const SimIRNode*    node,
                                                      size_t*             components) {
    if (builder == NULL || node == NULL) {
        return NULL;
    }
    if (node->type != SIM_IR_NODE_CONSTANT || node->value_type.kind != SIM_IR_VALUE_VECTOR) {
        return NULL;
    }
    if (components != NULL) {
        *components = node->value_type.components;
    }
    if (node->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX) {
        return node->data.constant.small;
    }
    if (builder->constants_data == NULL || builder->constants_offsets == NULL ||
        builder->constants_components == NULL ||
        node->data.constant.constant_index >= builder->constants_count) {
        return NULL;
    }
    return builder->constants_data + builder->constants_offsets[node->data.constant.constant_index];
}

/**
 * @brief Accumulate an affine operand tree into a simple pointwise operand form.
 *
 * Accepted trees are field refs, scalar/vector constants, additions,
 * subtractions, and multiplication/division by scalar constants. Everything else
 * is rejected for the general recursive path.
 */
static bool backend_cpu_accumulate_operand(const KernelIR*     kernel,
                                           const SimIRBuilder* builder,
                                           const SimIRNode*    node,
                                           size_t              element_count,
                                           size_t              element_size,
                                           size_t              dest_components,
                                           double              weight,
                                           BackendCPUOperand*  operand) {
    if (kernel == NULL || builder == NULL || node == NULL || operand == NULL) {
        return false;
    }

    switch (node->type) {
        case SIM_IR_NODE_FIELD_REF: {
            SimField*     field = backend_cpu_find_field(kernel, node->data.field);
            const double* data;
            if (field == NULL || field->element_size != element_size) {
                return false;
            }
            if (backend_cpu_element_count(field) != element_count) {
                return false;
            }
            data = (const double*) sim_field_data_const(field);
            if (data == NULL) {
                return false;
            }
            if (operand->has_field && operand->field_data != data) {
                return false;
            }
            operand->field_data = data;
            operand->has_field  = true;
            operand->field_scale += weight;
            return true;
        }
        case SIM_IR_NODE_CONSTANT: {
            if (sim_ir_type_is_scalar(node->value_type)) {
                operand->scalar_constant += weight * node->data.constant.scalar;
                operand->has_scalar_constant = true;
                return true;
            } else if (node->value_type.kind == SIM_IR_VALUE_VECTOR &&
                       node->value_type.components == dest_components) {
                size_t        components = 0U;
                const double* values = backend_cpu_constant_vector_data(builder, node, &components);
                if (values == NULL || components == 0U) {
                    return false;
                }
                if (operand->vector_constant != NULL && operand->vector_constant != values) {
                    return false;
                }
                operand->vector_constant   = values;
                operand->vector_components = components;
                operand->vector_scale += weight;
                return true;
            }
            return false;
        }
        case SIM_IR_NODE_ADD:
            return backend_cpu_accumulate_operand(kernel,
                                                  builder,
                                                  &builder->nodes[node->data.binary.lhs],
                                                  element_count,
                                                  element_size,
                                                  dest_components,
                                                  weight,
                                                  operand) &&
                   backend_cpu_accumulate_operand(kernel,
                                                  builder,
                                                  &builder->nodes[node->data.binary.rhs],
                                                  element_count,
                                                  element_size,
                                                  dest_components,
                                                  weight,
                                                  operand);
        case SIM_IR_NODE_SUB:
            return backend_cpu_accumulate_operand(kernel,
                                                  builder,
                                                  &builder->nodes[node->data.binary.lhs],
                                                  element_count,
                                                  element_size,
                                                  dest_components,
                                                  weight,
                                                  operand) &&
                   backend_cpu_accumulate_operand(kernel,
                                                  builder,
                                                  &builder->nodes[node->data.binary.rhs],
                                                  element_count,
                                                  element_size,
                                                  dest_components,
                                                  -weight,
                                                  operand);
        case SIM_IR_NODE_MUL: {
            if (builder->nodes == NULL || node->data.binary.lhs >= builder->count ||
                node->data.binary.rhs >= builder->count) {
                return false;
            }
            const SimIRNode* lhs    = &builder->nodes[node->data.binary.lhs];
            const SimIRNode* rhs    = &builder->nodes[node->data.binary.rhs];
            double           scalar = 0.0;
            if (backend_cpu_constant_scalar(lhs, &scalar)) {
                return backend_cpu_accumulate_operand(kernel,
                                                      builder,
                                                      rhs,
                                                      element_count,
                                                      element_size,
                                                      dest_components,
                                                      weight * scalar,
                                                      operand);
            }
            if (backend_cpu_constant_scalar(rhs, &scalar)) {
                return backend_cpu_accumulate_operand(kernel,
                                                      builder,
                                                      lhs,
                                                      element_count,
                                                      element_size,
                                                      dest_components,
                                                      weight * scalar,
                                                      operand);
            }
            return false;
        }
        case SIM_IR_NODE_DIV: {
            if (builder->nodes == NULL || node->data.binary.lhs >= builder->count ||
                node->data.binary.rhs >= builder->count) {
                return false;
            }
            const SimIRNode* lhs    = &builder->nodes[node->data.binary.lhs];
            const SimIRNode* rhs    = &builder->nodes[node->data.binary.rhs];
            double           scalar = 0.0;
            if (!backend_cpu_constant_scalar(rhs, &scalar) || scalar == 0.0) {
                return false;
            }
            return backend_cpu_accumulate_operand(kernel,
                                                  builder,
                                                  lhs,
                                                  element_count,
                                                  element_size,
                                                  dest_components,
                                                  weight / scalar,
                                                  operand);
        }
        default:
            break;
    }

    return false;
}

/**
 * @brief Build a normalized simple operand from an IR subtree.
 */
static bool backend_cpu_build_operand(const KernelIR*     kernel,
                                      const SimIRBuilder* builder,
                                      const SimIRNode*    node,
                                      size_t              element_count,
                                      size_t              element_size,
                                      size_t              dest_components,
                                      BackendCPUOperand*  operand) {
    if (operand == NULL) {
        return false;
    }
    operand->field_data          = NULL;
    operand->field_scale         = 0.0;
    operand->has_field           = false;
    operand->has_scalar_constant = false;
    operand->scalar_constant     = 0.0;
    operand->vector_constant     = NULL;
    operand->vector_scale        = 0.0;
    operand->vector_components   = 0U;
    return backend_cpu_accumulate_operand(
        kernel, builder, node, element_count, element_size, dest_components, 1.0, operand);
}

/**
 * @brief Read one flattened value from a simple operand.
 */
static double
backend_cpu_operand_value(const BackendCPUOperand* operand, size_t idx, size_t dest_components) {
    double value = 0.0;
    if (operand == NULL) {
        return value;
    }
    if (operand->has_field && operand->field_data != NULL) {
        value += operand->field_scale * operand->field_data[idx];
    }
    if (operand->has_scalar_constant) {
        value += operand->scalar_constant;
    }
    if (operand->vector_constant != NULL && operand->vector_components > 0U) {
        size_t lane = idx % operand->vector_components;
        value += operand->vector_scale * operand->vector_constant[lane];
    }
    (void) dest_components;
    return value;
}

#if defined(SIM_HAVE_VDSP)
/**
 * @brief Evaluate a scalar expression that is uniform across all elements.
 *
 * The vDSP complex-rotate path uses this for constant or parameterized angles.
 */
static bool backend_cpu_eval_uniform_scalar_node(const KernelIR*     kernel,
                                                 const SimIRBuilder* builder,
                                                 SimIRNodeId         node_id,
                                                 double*             out_value) {
    const SimIRNode* node;
    double           lhs = 0.0;
    double           rhs = 0.0;

    if (kernel == NULL || builder == NULL || out_value == NULL || builder->nodes == NULL ||
        node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return false;
    }

    node = &builder->nodes[node_id];
    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
            if (!sim_ir_type_is_scalar(node->value_type)) {
                return false;
            }
            *out_value = node->data.constant.scalar;
            return true;
        case SIM_IR_NODE_PARAM:
            if (kernel->params == NULL || node->data.param.param >= kernel->param_count) {
                return false;
            }
            *out_value = kernel->params[node->data.param.param];
            return true;
        case SIM_IR_NODE_ADD:
            return backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.lhs, &lhs) &&
                   backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.rhs, &rhs) &&
                   ((*out_value = lhs + rhs), true);
        case SIM_IR_NODE_SUB:
            return backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.lhs, &lhs) &&
                   backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.rhs, &rhs) &&
                   ((*out_value = lhs - rhs), true);
        case SIM_IR_NODE_MUL:
            return backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.lhs, &lhs) &&
                   backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.rhs, &rhs) &&
                   ((*out_value = lhs * rhs), true);
        case SIM_IR_NODE_DIV:
            return backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.lhs, &lhs) &&
                   backend_cpu_eval_uniform_scalar_node(kernel, builder, node->data.binary.rhs, &rhs) &&
                   ((*out_value = lhs / rhs), true);
        default:
            return false;
    }
}

/**
 * @brief Try the vDSP complex rotation fast path.
 *
 * The path accepts `COMPLEX_ROTATE(field_ref, uniform_angle)` over complex
 * source/destination fields of equal element count.
 */
static bool backend_cpu_try_vdsp_complex_rotate(const KernelIR*          kernel,
                                                const SimKernelIROutput* output,
                                                SimBackendCPUState*      state) {
    const SimIRBuilder*    builder;
    const SimIRNode*       root;
    const SimIRNode*       operand_node;
    SimField*              dest_field;
    SimField*              src_field;
    SimComplexDouble*      dst_data;
    const SimComplexDouble* src_data;
    size_t                 count;
    double                 theta = 0.0;
    double                 s     = 0.0;
    double                 c     = 1.0;

    if (kernel == NULL || output == NULL || state == NULL || kernel->builder == NULL ||
        !state->use_vdsp) {
        return false;
    }

    builder = kernel->builder;
    if (builder->nodes == NULL || output->expression == SIM_IR_INVALID_NODE ||
        output->expression >= builder->count) {
        return false;
    }

    root = &builder->nodes[output->expression];
    if (root->type != SIM_IR_NODE_COMPLEX_ROTATE) {
        return false;
    }
    if (root->data.complex_rotate.operand >= builder->count) {
        return false;
    }

    operand_node = &builder->nodes[root->data.complex_rotate.operand];
    if (operand_node->type != SIM_IR_NODE_FIELD_REF) {
        return false;
    }
    if (!backend_cpu_eval_uniform_scalar_node(
            kernel, builder, root->data.complex_rotate.angle, &theta)) {
        return false;
    }

    dest_field = backend_cpu_find_field(kernel, output->field_index);
    src_field  = backend_cpu_find_field(kernel, operand_node->data.field);
    if (dest_field == NULL || src_field == NULL ||
        dest_field->element_size != sizeof(SimComplexDouble) ||
        src_field->element_size != sizeof(SimComplexDouble)) {
        return false;
    }

    count = backend_cpu_element_count(dest_field);
    if (count == 0U || backend_cpu_element_count(src_field) != count) {
        return false;
    }

    dst_data = sim_field_complex_data(dest_field);
    src_data = sim_field_complex_data_const(src_field);
    if (dst_data == NULL || src_data == NULL) {
        return false;
    }

    s = sin(theta);
    c = cos(theta);
    if (!sim_accel_rotate_complex(&state->vdsp_complex_scratch, src_data, dst_data, count, c, s)) {
        return false;
    }
    return true;
}
#endif

/**
 * @brief Try the CPU simple pointwise field operation path.
 *
 * Supports binary arithmetic over affine field/constant operands and selected
 * componentwise complex expressions. Returning false means the caller should
 * continue with the general recursive evaluator.
 */
static bool backend_cpu_try_simple_field_op(const KernelIR*          kernel,
                                            const SimKernelIROutput* output,
                                            SimBackendCPUState*      state,
                                            SimField*                dest_field,
                                            double*                  dest_data,
                                            size_t                   element_count) {
    const SimIRBuilder* builder;
    const SimIRNode*    root;
    BackendCPUOperand   lhs_operand = { 0 };
    BackendCPUOperand   rhs_operand = { 0 };
    size_t              dest_components;
    size_t              element_size;

    if (kernel == NULL || output == NULL || dest_field == NULL || dest_data == NULL ||
        kernel->builder == NULL) {
        return false;
    }

    if (dest_field->element_size == 0U || (dest_field->element_size % sizeof(double)) != 0U) {
        return false;
    }

    builder = kernel->builder;
    if (builder->nodes == NULL || output->expression == SIM_IR_INVALID_NODE ||
        output->expression >= builder->count) {
        return false;
    }

    root = &builder->nodes[output->expression];
#if defined(SIM_HAVE_VDSP)
    if (root->type == SIM_IR_NODE_COMPLEX_ROTATE) {
        return backend_cpu_try_vdsp_complex_rotate(kernel, output, state);
    }
#endif
    if (!(root->type == SIM_IR_NODE_ADD || root->type == SIM_IR_NODE_SUB ||
          root->type == SIM_IR_NODE_MUL || root->type == SIM_IR_NODE_DIV)) {
        return false;
    }

    if (sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(root->value_type)) &&
        kernel->complex_semantics != SIM_KERNEL_COMPLEX_SEMANTICS_COMPONENTWISE) {
        return false;
    }

    if (root->data.binary.lhs >= builder->count || root->data.binary.rhs >= builder->count) {
        return false;
    }

    dest_components = dest_field->element_size / sizeof(double);
    element_size    = dest_field->element_size;

    if (!backend_cpu_build_operand(kernel,
                                   builder,
                                   &builder->nodes[root->data.binary.lhs],
                                   element_count,
                                   element_size,
                                   dest_components,
                                   &lhs_operand)) {
        return false;
    }

    if (!backend_cpu_build_operand(kernel,
                                   builder,
                                   &builder->nodes[root->data.binary.rhs],
                                   element_count,
                                   element_size,
                                   dest_components,
                                   &rhs_operand)) {
        return false;
    }

    if (element_count == 0U || dest_components == 0U) {
        return false;
    }

    size_t total_values = element_count * dest_components;
    switch (root->type) {
        case SIM_IR_NODE_ADD:
            for (size_t idx = 0; idx < total_values; ++idx) {
                double lhs     = backend_cpu_operand_value(&lhs_operand, idx, dest_components);
                double rhs     = backend_cpu_operand_value(&rhs_operand, idx, dest_components);
                dest_data[idx] = lhs + rhs;
            }
            return true;
        case SIM_IR_NODE_SUB:
            for (size_t idx = 0; idx < total_values; ++idx) {
                double lhs     = backend_cpu_operand_value(&lhs_operand, idx, dest_components);
                double rhs     = backend_cpu_operand_value(&rhs_operand, idx, dest_components);
                dest_data[idx] = lhs - rhs;
            }
            return true;
        case SIM_IR_NODE_MUL:
            for (size_t idx = 0; idx < total_values; ++idx) {
                double lhs     = backend_cpu_operand_value(&lhs_operand, idx, dest_components);
                double rhs     = backend_cpu_operand_value(&rhs_operand, idx, dest_components);
                dest_data[idx] = lhs * rhs;
            }
            return true;
        case SIM_IR_NODE_DIV:
            for (size_t idx = 0; idx < total_values; ++idx) {
                double lhs     = backend_cpu_operand_value(&lhs_operand, idx, dest_components);
                double rhs     = backend_cpu_operand_value(&rhs_operand, idx, dest_components);
                dest_data[idx] = lhs / rhs;
            }
            return true;
        default:
            return false;
    }
}

/**
 * @brief Execute a KernelIR package on the CPU backend.
 *
 * The launch processes each output field in order. Integer-domain outputs use
 * exact integer evaluation; floating/vector/complex outputs first try vDSP and
 * simple pointwise paths before falling back to recursive node evaluation.
 *
 * @return #SIM_RESULT_OK or an error stored by backend_launch() in
 * SimBackend::last_error.
 */
static SimResult backend_cpu_launch_impl(SimBackend* backend, KernelIR* kernel) {
    SimBackendCPUState* state;
    SimBackendCPUFrame* base_frame = NULL;
    size_t              output_index;
    SimResult           result = SIM_RESULT_OK;

    if (backend == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = backend_cpu_state(backend);
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (kernel == NULL || kernel->builder == NULL || kernel->outputs == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* Debug: print kernel builder constants summary (diagnostics builds only). */
#if SIM_DIAGNOSTICS
    if (kernel->builder != NULL && kernel->builder->constants_count > 0U &&
        kernel->builder->constants_data != NULL) {
        size_t total_values = 0U;
        if (kernel->builder->constants_count > 0U) {
            size_t last  = kernel->builder->constants_count - 1U;
            total_values = kernel->builder->constants_offsets[last] +
                           kernel->builder->constants_components[last];
        }
        fprintf(stderr,
                "[DEBUG] backend_cpu_launch_impl: kernel.builder=%p constants_count=%zu "
                "total_values=%zu\n",
                (void*) kernel->builder,
                kernel->builder->constants_count,
                total_values);
        for (size_t k = 0U; k < total_values; ++k) {
            fprintf(stderr,
                    "[DEBUG] builder.const_data[%zu] = %g\n",
                    k,
                    kernel->builder->constants_data[k]);
        }
    }
#endif

    size_t max_components = backend_cpu_builder_max_components(kernel);
    if (!backend_cpu_ensure_capacity(state, kernel->builder->count, max_components)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    base_frame = backend_cpu_frame_acquire(state, kernel->builder->count, max_components);
    if (base_frame == NULL || base_frame->scratch == NULL || base_frame->flags == NULL ||
        base_frame->flag_generation == NULL) {
        result = SIM_RESULT_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (output_index = 0; output_index < kernel->output_count; ++output_index) {
        const SimKernelIROutput* output     = &kernel->outputs[output_index];
        SimField*                dest_field = backend_cpu_find_field(kernel, output->field_index);
        SimIRType                output_type;
        double*                  dest_data;
        size_t                   element_count;
        size_t                   element_index;

        if (dest_field == NULL) {
            result = SIM_RESULT_NOT_FOUND;
            goto cleanup;
        }

        if (output->expression == SIM_IR_INVALID_NODE ||
            output->expression >= kernel->builder->count) {
            result = SIM_RESULT_INVALID_ARGUMENT;
            goto cleanup;
        }

        output_type = sim_ir_builder_node_type(kernel->builder, output->expression);
        if (backend_cpu_domain_is_integer(sim_ir_type_scalar_domain(output_type))) {
            result = backend_cpu_execute_integer_output(kernel, output, dest_field);
            if (result != SIM_RESULT_OK) {
                goto cleanup;
            }
            continue;
        }

        if (dest_field->element_size == 0U || (dest_field->element_size % sizeof(double) != 0U)) {
            result = SIM_RESULT_INVALID_ARGUMENT;
            goto cleanup;
        }

        element_count = backend_cpu_element_count(dest_field);
        if (element_count == 0U) {
            continue;
        }

        dest_data = (double*) sim_field_data(dest_field);
        if (dest_data == NULL) {
            result = SIM_RESULT_INVALID_ARGUMENT;
            goto cleanup;
        }
        size_t  dest_components = dest_field->element_size / sizeof(double);
        double* dest_values     = dest_data;
        size_t  component_limit = dest_components;
        if (component_limit > base_frame->components) {
            component_limit = base_frame->components;
        }
        BackendCPUComponentKernelDesc kernel_desc =
            backend_cpu_select_component_kernel(component_limit);
        double* scratch_row = base_frame->scratch + output->expression * base_frame->components;

#if defined(SIM_HAVE_VDSP)
        if (backend_cpu_try_vdsp(kernel, output, state)) {
            continue;
        }
#endif
        if (backend_cpu_try_simple_field_op(
                kernel, output, state, dest_field, dest_values, element_count)) {
            continue;
        }

        for (element_index = 0; element_index < element_count; ++element_index) {
            backend_cpu_frame_next_generation(base_frame);

            result = kernel_desc.eval(kernel,
                                      element_index,
                                      state,
                                      base_frame,
                                      dest_field,
                                      output->expression,
                                      scratch_row,
                                      kernel_desc.component_limit);
            if (result != SIM_RESULT_OK) {
                fprintf(stderr,
                        "[DEBUG] backend_cpu_launch_impl: eval_components failed expr=%zu "
                        "element=%zu result=%d\n",
                        (size_t) output->expression,
                        element_index,
                        (int) result);
                goto cleanup;
            }

            double* element_base = dest_values + element_index * dest_components;
            for (size_t comp_idx = 0; comp_idx < component_limit; ++comp_idx) {
                element_base[comp_idx] = scratch_row[comp_idx];
            }
        }
    }

cleanup:
    if (base_frame != NULL) {
        backend_cpu_frame_release(state);
    }

    return result;
}

/**
 * @brief Destroy CPU backend private state and reset advertised features.
 */
static void backend_cpu_destroy_impl(SimBackend* backend) {
    SimBackendCPUState* state;

    if (backend == NULL) {
        return;
    }

    state = backend_cpu_state(backend);
    if (state != NULL) {
        size_t i;
        for (i = 0; i < state->frame_capacity; ++i) {
            if (state->frames[i] != NULL) {
                backend_cpu_frame_destroy(state->frames[i]);
                free(state->frames[i]);
            }
        }
        free(state->frames);
#if defined(SIM_HAVE_VDSP)
        sim_accel_split_release(&state->vdsp_complex_scratch);
#endif
        free(state);
    }

    backend->impl = NULL;
    backend->type = SIM_BACKEND_TYPE_CPU;
    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);
}

/**
 * @brief Query whether an initialized CPU backend will attempt vDSP paths.
 */
bool sim_backend_cpu_vdsp_enabled(const SimBackend* backend) {
#if defined(SIM_HAVE_VDSP)
    if (backend == NULL || backend->type != SIM_BACKEND_TYPE_CPU) {
        return false;
    }
    const SimBackendCPUState* state = backend_cpu_state((SimBackend*) backend);
    return (state != NULL) && state->use_vdsp;
#else
    (void) backend;
    return false;
#endif
}

/**
 * @brief Toggle runtime use of vDSP paths for an initialized CPU backend.
 */
void sim_backend_cpu_set_vdsp_enabled(SimBackend* backend, bool enabled) {
#if defined(SIM_HAVE_VDSP)
    if (backend == NULL || backend->type != SIM_BACKEND_TYPE_CPU) {
        return;
    }
    SimBackendCPUState* state = backend_cpu_state(backend);
    if (state != NULL) {
        state->use_vdsp = enabled;
    }
#else
    (void) backend;
    (void) enabled;
#endif
}

/**
 * @brief Initialize the selected backend implementation.
 *
 * Invalid backend types are treated as CPU requests. CUDA and Metal requests
 * return #SIM_RESULT_NOT_FOUND when support was not compiled in.
 */
void backend_init(SimBackend* backend) {
    SimResult result = SIM_RESULT_INVALID_ARGUMENT;

    if (backend == NULL) {
        return;
    }

    if (!(backend->type == SIM_BACKEND_TYPE_CPU || backend->type == SIM_BACKEND_TYPE_CUDA ||
          backend->type == SIM_BACKEND_TYPE_METAL)) {
        backend->type = SIM_BACKEND_TYPE_CPU;
    }

    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);

    switch (backend->type) {
        case SIM_BACKEND_TYPE_CPU:
            result = backend_cpu_init_impl(backend);
            break;
        case SIM_BACKEND_TYPE_CUDA:
#if defined(SIM_HAVE_CUDA)
            result = sim_backend_cuda_init(backend);
#else
            result = SIM_RESULT_NOT_FOUND;
#endif
            break;
        case SIM_BACKEND_TYPE_METAL:
#if defined(SIM_HAVE_METAL)
            result = sim_backend_metal_init(backend);
#else
            result = SIM_RESULT_NOT_FOUND;
#endif
            break;
        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

    backend->last_error = result;
}

/**
 * @brief Dispatch a KernelIR launch to the active backend implementation.
 */
void backend_launch(SimBackend* backend, KernelIR* kernel) {
    SimResult result = SIM_RESULT_INVALID_ARGUMENT;

    if (backend == NULL) {
        return;
    }

    switch (backend->type) {
        case SIM_BACKEND_TYPE_CPU:
            result = backend_cpu_launch_impl(backend, kernel);
            break;
        case SIM_BACKEND_TYPE_CUDA:
#if defined(SIM_HAVE_CUDA)
            result = sim_backend_cuda_launch(backend, kernel);
#else
            result = SIM_RESULT_NOT_FOUND;
#endif
            break;
        case SIM_BACKEND_TYPE_METAL:
#if defined(SIM_HAVE_METAL)
            result = sim_backend_metal_launch(backend, kernel);
#else
            result = SIM_RESULT_NOT_FOUND;
#endif
            break;
        default:
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }

    backend->last_error = result;
}

/**
 * @brief Destroy the active backend implementation and clear feature bits.
 */
void backend_destroy(SimBackend* backend) {
    if (backend == NULL) {
        return;
    }

    switch (backend->type) {
        case SIM_BACKEND_TYPE_CPU:
            backend_cpu_destroy_impl(backend);
            break;
        case SIM_BACKEND_TYPE_CUDA:
#if defined(SIM_HAVE_CUDA)
            sim_backend_cuda_destroy(backend);
#else
            backend->impl = NULL;
#endif
            break;
        case SIM_BACKEND_TYPE_METAL:
#if defined(SIM_HAVE_METAL)
            sim_backend_metal_destroy(backend);
#else
            backend->impl = NULL;
#endif
            break;
        default:
            backend_cpu_destroy_impl(backend);
            break;
    }

    backend->last_error = SIM_RESULT_OK;
    backend_set_features(backend, SIM_BACKEND_FEATURE_NONE);
}
