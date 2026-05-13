/**
 * @file graph_ir.c
 * @brief GraphIR storage, validation, fusion, and execution support.
 *
 * GraphIR composes pointwise KernelIR nodes with FFT, canonicalization, and
 * explicit copy nodes. The implementation owns node/edge arrays, checks
 * representation and time contracts, expands compatible pointwise kernels, and
 * dispatches execution through the active simulation context and backend.
 */
#include "graph_ir.h"

#include "oakfield/operators/common/fft_plan.h"
#include "oakfield/sim_context.h"

#include "oakfield/backend.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

typedef struct SimGraphIRKernel {
    KernelIR                            kernel;
    SimOperatorKernelBindingDescriptor* binding_map;
    SimKernelIRBinding*                 bindings;
    SimOperatorKernelOutputDescriptor*  output_map;
    SimKernelIROutput*                  outputs;
    double*                             params;
    size_t                              binding_count;
    size_t                              output_count;
    size_t                              param_count;
} SimGraphIRKernel;

typedef struct SimGraphIRFftState {
    FFTPlan         plan;
    FFTPlan2D       plan2d;
    size_t          length;
    size_t          rank;
    size_t          shape[2];
    double complex* scratch_in;
    double complex* scratch_out;
    size_t          scratch_capacity;
    bool            output_real;
} SimGraphIRFftState;

typedef struct SimGraphIRCanonicalizeState {
    double complex* buffer;
    size_t          capacity;
    size_t          length;
    size_t          rank;
    size_t          shape[2];
    double          tolerance;
} SimGraphIRCanonicalizeState;

typedef struct SimGraphIRNode {
    SimGraphIRNodeKind               kind;
    SimGraphIRInputRef               input;
    SimGraphIROutputRef              output;
    SimGraphIRRepresentationContract contract;

    union {
        SimGraphIRKernel            pointwise;
        SimGraphIRFftState          fft;
        SimGraphIRCanonicalizeState canonicalize;
    } data;
} SimGraphIRNode;

const char* sim_graph_ir_node_kind_name(SimGraphIRNodeKind kind) {
    switch (kind) {
        case SIM_GRAPH_IR_NODE_POINTWISE_KERNEL:
            return "pointwise_kernel";
        case SIM_GRAPH_IR_NODE_FFT_FORWARD:
            return "fft_forward";
        case SIM_GRAPH_IR_NODE_FFT_INVERSE:
            return "fft_inverse";
        case SIM_GRAPH_IR_NODE_PROMOTE_COMPLEX:
            return "promote_complex";
        case SIM_GRAPH_IR_NODE_CAST_COPY:
            return "cast_copy";
        case SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT:
            return "canonicalize_real_constraint";
        default:
            return "unknown";
    }
}

const char* sim_graph_ir_input_kind_name(SimGraphIRInputKind kind) {
    switch (kind) {
        case SIM_GRAPH_IR_INPUT_NONE:
            return "none";
        case SIM_GRAPH_IR_INPUT_FIELD:
            return "field";
        case SIM_GRAPH_IR_INPUT_NODE:
            return "node";
        default:
            return "unknown";
    }
}

const char* sim_graph_ir_output_kind_name(SimGraphIROutputKind kind) {
    switch (kind) {
        case SIM_GRAPH_IR_OUTPUT_NONE:
            return "none";
        case SIM_GRAPH_IR_OUTPUT_FIELD:
            return "field";
        case SIM_GRAPH_IR_OUTPUT_TEMP:
            return "temp";
        default:
            return "unknown";
    }
}

const char* sim_graph_ir_time_source_name(SimGraphIRTimeSource source) {
    switch (source) {
        case SIM_GRAPH_IR_TIME_NONE:
            return "none";
        case SIM_GRAPH_IR_TIME_PARAM:
            return "param";
        case SIM_GRAPH_IR_TIME_STEP_PURE:
            return "step_pure";
        case SIM_GRAPH_IR_TIME_ACCUMULATED:
            return "accumulated";
        default:
            return "unknown";
    }
}

const char* sim_graph_ir_dt_source_name(SimGraphIRDTSrc source) {
    switch (source) {
        case SIM_GRAPH_IR_DT_NONE:
            return "none";
        case SIM_GRAPH_IR_DT_PARAM:
            return "param";
        case SIM_GRAPH_IR_DT_NOMINAL:
            return "nominal";
        default:
            return "unknown";
    }
}

static bool sim_graph_ir_builder_equal(const SimIRBuilder* a, const SimIRBuilder* b) {
    if (a == b) {
        return true;
    }
    if (a == NULL || b == NULL) {
        return false;
    }
    if (a->count != b->count || a->default_boundary != b->default_boundary) {
        return false;
    }
    if (a->count > 0U && memcmp(a->nodes, b->nodes, a->count * sizeof(SimIRNode)) != 0) {
        return false;
    }
    if (a->constants_count != b->constants_count) {
        return false;
    }
    if (a->constants_count > 0U) {
        if (memcmp(a->constants_offsets,
                   b->constants_offsets,
                   a->constants_count * sizeof(size_t)) != 0) {
            return false;
        }
        if (memcmp(a->constants_components,
                   b->constants_components,
                   a->constants_count * sizeof(size_t)) != 0) {
            return false;
        }
        if (a->constants_data_used != b->constants_data_used) {
            return false;
        }
        if (a->constants_data_used > 0U &&
            memcmp(a->constants_data, b->constants_data, a->constants_data_used * sizeof(double)) !=
                0) {
            return false;
        }
    }
    return true;
}

static bool sim_graph_ir_kernel_bindings_equal(const SimGraphIRKernel* a,
                                               const SimGraphIRKernel* b) {
    if (a == NULL || b == NULL || a->binding_count != b->binding_count) {
        return false;
    }
    for (size_t i = 0U; i < a->binding_count; ++i) {
        if (a->binding_map[i].ir_field_index != b->binding_map[i].ir_field_index ||
            a->binding_map[i].context_field_index != b->binding_map[i].context_field_index) {
            return false;
        }
    }
    return true;
}

static bool sim_graph_ir_kernel_params_equal(const SimGraphIRKernel* a, const SimGraphIRKernel* b) {
    if (a == NULL || b == NULL || a->param_count != b->param_count) {
        return false;
    }
    if (a->param_count == 0U) {
        return true;
    }
    return memcmp(a->params, b->params, a->param_count * sizeof(double)) == 0;
}

static bool sim_graph_ir_kernel_outputs_disjoint(const SimGraphIRKernel* a,
                                                 const SimGraphIRKernel* b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    for (size_t i = 0U; i < a->output_count; ++i) {
        size_t a_field = a->output_map[i].ir_field_index;
        for (size_t j = 0U; j < b->output_count; ++j) {
            if (a_field == b->output_map[j].ir_field_index) {
                return false;
            }
        }
    }
    return true;
}

static bool sim_graph_ir_kernels_compatible(const SimGraphIRKernel* a, const SimGraphIRKernel* b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    if (!sim_graph_ir_builder_equal(a->kernel.builder, b->kernel.builder)) {
        return false;
    }
    if (!sim_graph_ir_kernel_bindings_equal(a, b)) {
        return false;
    }
    if (!sim_graph_ir_kernel_params_equal(a, b)) {
        return false;
    }
    if (a->kernel.required_features != b->kernel.required_features) {
        return false;
    }
    if (a->kernel.complex_semantics != b->kernel.complex_semantics) {
        return false;
    }
    if (!sim_graph_ir_kernel_outputs_disjoint(a, b)) {
        return false;
    }
    return true;
}

static void sim_graph_ir_builder_free(SimIRBuilder* builder) {
    if (builder == NULL) {
        return;
    }
    free(builder->nodes);
    free(builder->constants_data);
    free(builder->constants_offsets);
    free(builder->constants_components);
    free(builder);
}

static SimIRBuilder* sim_graph_ir_builder_clone(const SimIRBuilder* src) {
    if (src == NULL) {
        return NULL;
    }

    SimIRBuilder* copy = (SimIRBuilder*) calloc(1U, sizeof(SimIRBuilder));
    if (copy == NULL) {
        return NULL;
    }

    copy->count            = src->count;
    copy->capacity         = src->capacity;
    copy->default_boundary = src->default_boundary;

    if (src->capacity > 0U) {
        copy->nodes = (SimIRNode*) malloc(src->capacity * sizeof(SimIRNode));
        if (copy->nodes == NULL) {
            sim_graph_ir_builder_free(copy);
            return NULL;
        }
        memcpy(copy->nodes, src->nodes, src->count * sizeof(SimIRNode));
    }

    if (src->constants_count > 0U) {
        size_t entry_capacity = src->constants_capacity;
        copy->constants_count = src->constants_count;
        if (entry_capacity < copy->constants_count) {
            entry_capacity = copy->constants_count;
        }
        copy->constants_capacity   = entry_capacity;
        copy->constants_offsets    = (size_t*) malloc(entry_capacity * sizeof(size_t));
        copy->constants_components = (size_t*) malloc(entry_capacity * sizeof(size_t));
        if (copy->constants_offsets == NULL || copy->constants_components == NULL) {
            sim_graph_ir_builder_free(copy);
            return NULL;
        }
        memcpy(copy->constants_offsets,
               src->constants_offsets,
               copy->constants_count * sizeof(size_t));
        memcpy(copy->constants_components,
               src->constants_components,
               copy->constants_count * sizeof(size_t));

        size_t total_values = src->constants_data_used;
        if (total_values > 0U) {
            size_t data_capacity = src->constants_data_capacity;
            if (data_capacity < total_values) {
                data_capacity = total_values;
            }
            copy->constants_data_capacity = data_capacity;
            copy->constants_data_used     = total_values;
            copy->constants_data          = (double*) malloc(data_capacity * sizeof(double));
            if (copy->constants_data == NULL) {
                sim_graph_ir_builder_free(copy);
                return NULL;
            }
            memcpy(copy->constants_data, src->constants_data, total_values * sizeof(double));
        }
    }

    return copy;
}

static void sim_graph_ir_kernel_destroy(SimGraphIRKernel* kernel) {
    if (kernel == NULL) {
        return;
    }
    if (kernel->kernel.builder != NULL) {
        sim_graph_ir_builder_free((SimIRBuilder*) kernel->kernel.builder);
        kernel->kernel.builder = NULL;
    }
    free(kernel->binding_map);
    free(kernel->bindings);
    free(kernel->output_map);
    free(kernel->outputs);
    free(kernel->params);
    kernel->binding_map   = NULL;
    kernel->bindings      = NULL;
    kernel->output_map    = NULL;
    kernel->outputs       = NULL;
    kernel->params        = NULL;
    kernel->binding_count = 0U;
    kernel->output_count  = 0U;
    kernel->param_count   = 0U;
}

static SimResult sim_graph_ir_kernel_init(SimGraphIRKernel*                  kernel,
                                          const SimOperatorKernelDescriptor* descriptor) {
    if (kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    memset(kernel, 0, sizeof(*kernel));

    if (descriptor == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (descriptor->builder == NULL || descriptor->bindings == NULL ||
        descriptor->binding_count == 0U || descriptor->outputs == NULL ||
        descriptor->output_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    kernel->bindings =
        (SimKernelIRBinding*) calloc(descriptor->binding_count, sizeof(SimKernelIRBinding));
    kernel->binding_map = (SimOperatorKernelBindingDescriptor*) calloc(
        descriptor->binding_count, sizeof(SimOperatorKernelBindingDescriptor));
    kernel->outputs =
        (SimKernelIROutput*) calloc(descriptor->output_count, sizeof(SimKernelIROutput));
    kernel->output_map = (SimOperatorKernelOutputDescriptor*) calloc(
        descriptor->output_count, sizeof(SimOperatorKernelOutputDescriptor));
    if (kernel->bindings == NULL || kernel->binding_map == NULL || kernel->outputs == NULL ||
        kernel->output_map == NULL) {
        sim_graph_ir_kernel_destroy(kernel);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    kernel->binding_count = descriptor->binding_count;
    kernel->output_count  = descriptor->output_count;
    kernel->param_count   = descriptor->param_count;

    if (descriptor->param_count > 0U) {
        kernel->params = (double*) calloc(descriptor->param_count, sizeof(double));
        if (kernel->params == NULL) {
            sim_graph_ir_kernel_destroy(kernel);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        if (descriptor->params != NULL) {
            memcpy(kernel->params, descriptor->params, descriptor->param_count * sizeof(double));
        }
    }

    for (size_t i = 0U; i < descriptor->binding_count; ++i) {
        kernel->binding_map[i]          = descriptor->bindings[i];
        kernel->bindings[i].field_index = descriptor->bindings[i].ir_field_index;
        kernel->bindings[i].field       = NULL;
    }

    for (size_t i = 0U; i < descriptor->output_count; ++i) {
        kernel->output_map[i]          = descriptor->outputs[i];
        kernel->outputs[i].field_index = descriptor->outputs[i].ir_field_index;
        kernel->outputs[i].expression  = descriptor->outputs[i].expression;
    }

    SimIRBuilder* copy = sim_graph_ir_builder_clone(descriptor->builder);
    if (copy == NULL) {
        sim_graph_ir_kernel_destroy(kernel);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    kernel->kernel.builder           = copy;
    kernel->kernel.bindings          = kernel->bindings;
    kernel->kernel.binding_count     = kernel->binding_count;
    kernel->kernel.outputs           = kernel->outputs;
    kernel->kernel.output_count      = kernel->output_count;
    kernel->kernel.params            = kernel->params;
    kernel->kernel.param_count       = kernel->param_count;
    kernel->kernel.required_features = descriptor->required_features;
    kernel->kernel.complex_semantics = descriptor->complex_semantics;

    return SIM_RESULT_OK;
}

static bool sim_graph_ir_kernel_requires_boundary_feature(const KernelIR* kernel) {
    if (kernel == NULL || kernel->builder == NULL) {
        return false;
    }

    for (size_t i = 0; i < kernel->builder->count; ++i) {
        const SimIRNode* node = &kernel->builder->nodes[i];
        if (node != NULL && node->type == SIM_IR_NODE_DIFF) {
            if (node->data.diff.method != SIM_IR_DIFF_METHOD_AUTO) {
                return true;
            }
            switch (node->data.diff.boundary) {
                case SIM_IR_BOUNDARY_NEUMANN:
                    break;
                case SIM_IR_BOUNDARY_DIRICHLET:
                case SIM_IR_BOUNDARY_PERIODIC:
                case SIM_IR_BOUNDARY_REFLECTIVE:
                default:
                    return true;
            }
        }
    }
    return false;
}

static bool sim_graph_ir_kernel_requires_warp_profile_fallback(const KernelIR*   kernel,
                                                               const SimBackend* backend) {
    if (kernel == NULL || kernel->builder == NULL || backend == NULL) {
        return false;
    }

    if (backend->type == SIM_BACKEND_TYPE_CPU) {
        return false;
    }

    for (size_t i = 0; i < kernel->builder->count; ++i) {
        const SimIRNode* node = &kernel->builder->nodes[i];
        if (node != NULL && node->type == SIM_IR_NODE_WARP) {
            switch (node->data.warp.profile) {
                case SIM_IR_WARP_PROFILE_DIGAMMA:
                case SIM_IR_WARP_PROFILE_TRIGAMMA:
                    break;
                default:
                    return true;
            }
        }
    }

    return false;
}

static SimResult sim_graph_ir_launch_cpu_fallback(KernelIR* kernel) {
    if (kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimBackend fallback = { 0 };
    backend_init(&fallback);
    backend_launch(&fallback, kernel);
    SimResult rc = fallback.last_error;
    backend_destroy(&fallback);
    return rc;
}

static SimResult sim_graph_ir_execute_kernel(SimGraphIRKernel* kernel, SimContext* context) {
    if (kernel == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (context->scheduler.backend == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    KernelIR* ir = &kernel->kernel;

    if (ir->params != NULL && ir->param_count > SIM_IR_PARAM_DT) {
        double dt = sim_context_timestep(context);
        if (!isfinite(dt)) {
            dt = 0.0;
        }
        ir->params[SIM_IR_PARAM_DT] = dt;
        if (ir->param_count > SIM_IR_PARAM_STEP_INDEX) {
            ir->params[SIM_IR_PARAM_STEP_INDEX] = (double) sim_context_step_index(context);
        }
        if (ir->param_count > SIM_IR_PARAM_SQRT_DT) {
            ir->params[SIM_IR_PARAM_SQRT_DT] = (dt > 0.0) ? sqrt(dt) : 0.0;
        }
        if (ir->param_count > SIM_IR_PARAM_TIME) {
            ir->params[SIM_IR_PARAM_TIME] = sim_context_time(context);
        }
    }

    for (size_t i = 0U; i < kernel->binding_count; ++i) {
        size_t    field_index = kernel->binding_map[i].context_field_index;
        SimField* field       = sim_context_field(context, field_index);
        if (field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        kernel->bindings[i].field_index = kernel->binding_map[i].ir_field_index;
        kernel->bindings[i].field       = field;
        kernel->bindings[i].shape       = sim_field_shape(field);
        kernel->bindings[i].strides     = sim_field_strides(field);
        kernel->bindings[i].rank        = sim_field_rank(field);
    }

    if (sim_graph_ir_kernel_requires_boundary_feature(ir) &&
        !backend_supports_feature(context->scheduler.backend,
                                  SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS)) {
        return sim_graph_ir_launch_cpu_fallback(ir);
    }
    if (sim_graph_ir_kernel_requires_warp_profile_fallback(ir, context->scheduler.backend)) {
        return sim_graph_ir_launch_cpu_fallback(ir);
    }
    if (!backend_supports_features(context->scheduler.backend, ir->required_features)) {
        return sim_graph_ir_launch_cpu_fallback(ir);
    }

    backend_launch(context->scheduler.backend, ir);
    SimResult result = context->scheduler.backend->last_error;
    if (result == SIM_RESULT_NOT_SUPPORTED) {
        return sim_graph_ir_launch_cpu_fallback(ir);
    }

    return result;
}

static SimResult sim_graph_ir_fft_ensure_capacity(SimGraphIRFftState* state, size_t length) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (state->scratch_capacity >= length) {
        return SIM_RESULT_OK;
    }

    double complex* in_buf =
        (double complex*) realloc(state->scratch_in, length * sizeof(double complex));
    double complex* out_buf =
        (double complex*) realloc(state->scratch_out, length * sizeof(double complex));
    if (in_buf == NULL || out_buf == NULL) {
        free(in_buf);
        free(out_buf);
        state->scratch_in       = NULL;
        state->scratch_out      = NULL;
        state->scratch_capacity = 0U;
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->scratch_in       = in_buf;
    state->scratch_out      = out_buf;
    state->scratch_capacity = length;
    return SIM_RESULT_OK;
}

static void sim_graph_ir_fft_release(SimGraphIRFftState* state) {
    if (state == NULL) {
        return;
    }
    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->scratch_in);
    free(state->scratch_out);
    state->scratch_in       = NULL;
    state->scratch_out      = NULL;
    state->scratch_capacity = 0U;
    state->length           = 0U;
    state->rank             = 0U;
    state->shape[0]         = 0U;
    state->shape[1]         = 0U;
}

static double sim_graph_ir_hermitian_error_1d(const double complex* data, size_t n) {
    if (data == NULL || n == 0U) {
        return 0.0;
    }

    double max_err = fabs(cimag(data[0]));
    if ((n % 2U) == 0U) {
        size_t k           = n / 2U;
        double nyquist_err = fabs(cimag(data[k]));
        if (nyquist_err > max_err) {
            max_err = nyquist_err;
        }
    }

    if (n > 1U) {
        size_t pair_count = (n - 1U) / 2U;
        for (size_t k = 1U; k <= pair_count; ++k) {
            size_t         nk     = n - k;
            double complex diff   = data[nk] - conj(data[k]);
            double         err_re = fabs(creal(diff));
            double         err_im = fabs(cimag(diff));
            if (err_re > max_err) {
                max_err = err_re;
            }
            if (err_im > max_err) {
                max_err = err_im;
            }
        }
    }

    return max_err;
}

static double sim_graph_ir_hermitian_error_2d(const double complex* data,
                                              size_t rows,
                                              size_t cols) {
    if (data == NULL || rows == 0U || cols == 0U) {
        return 0.0;
    }

    double max_err = 0.0;
    for (size_t y = 0U; y < rows; ++y) {
        size_t y_conj = (y == 0U) ? 0U : (rows - y);
        for (size_t x = 0U; x < cols; ++x) {
            size_t x_conj = (x == 0U) ? 0U : (cols - x);
            size_t idx = y * cols + x;
            size_t idx_conj = y_conj * cols + x_conj;

            if (idx_conj == idx) {
                double err = fabs(cimag(data[idx]));
                if (err > max_err) {
                    max_err = err;
                }
            } else if (idx_conj > idx) {
                double complex diff = data[idx_conj] - conj(data[idx]);
                double err_re = fabs(creal(diff));
                double err_im = fabs(cimag(diff));
                if (err_re > max_err) {
                    max_err = err_re;
                }
                if (err_im > max_err) {
                    max_err = err_im;
                }
            }
        }
    }

    return max_err;
}

static double sim_graph_ir_hermitian_error(const double complex* data,
                                           size_t rank,
                                           const size_t* shape) {
    if (data == NULL || shape == NULL || rank == 0U) {
        return 0.0;
    }
    if (rank == 1U) {
        return sim_graph_ir_hermitian_error_1d(data, shape[0]);
    }
    if (rank == 2U) {
        return sim_graph_ir_hermitian_error_2d(data, shape[0], shape[1]);
    }
    return 0.0;
}

static void sim_graph_ir_canonicalize_hermitian_1d(double complex* data, size_t n) {
    if (data == NULL || n == 0U) {
        return;
    }

    if (n > 1U) {
        size_t pair_count = (n - 1U) / 2U;
        for (size_t k = 1U; k <= pair_count; ++k) {
            size_t         nk  = n - k;
            double complex avg = 0.5 * (data[k] + conj(data[nk]));
            data[k]            = avg;
            data[nk]           = conj(avg);
        }
    }

    data[0] = CMPLX(creal(data[0]), 0.0);
    if ((n % 2U) == 0U) {
        size_t k = n / 2U;
        data[k]  = CMPLX(creal(data[k]), 0.0);
    }
}

static void sim_graph_ir_canonicalize_hermitian_2d(double complex* data,
                                                   size_t rows,
                                                   size_t cols) {
    if (data == NULL || rows == 0U || cols == 0U) {
        return;
    }

    for (size_t y = 0U; y < rows; ++y) {
        size_t y_conj = (y == 0U) ? 0U : (rows - y);
        for (size_t x = 0U; x < cols; ++x) {
            size_t x_conj = (x == 0U) ? 0U : (cols - x);
            size_t idx = y * cols + x;
            size_t idx_conj = y_conj * cols + x_conj;

            if (idx_conj == idx) {
                data[idx] = CMPLX(creal(data[idx]), 0.0);
            } else if (idx_conj > idx) {
                double complex avg = 0.5 * (data[idx] + conj(data[idx_conj]));
                data[idx] = avg;
                data[idx_conj] = conj(avg);
            }
        }
    }
}

static void sim_graph_ir_canonicalize_hermitian(double complex* data,
                                                size_t rank,
                                                const size_t* shape) {
    if (data == NULL || shape == NULL || rank == 0U) {
        return;
    }
    if (rank == 1U) {
        sim_graph_ir_canonicalize_hermitian_1d(data, shape[0]);
    } else if (rank == 2U) {
        sim_graph_ir_canonicalize_hermitian_2d(data, shape[0], shape[1]);
    }
}

static SimResult sim_graph_ir_execute_canonicalize(SimGraphIRNode*                 node,
                                                   const SimGraphIRCompileContext* ctx) {
    if (node == NULL || ctx == NULL || ctx->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimGraphIRCanonicalizeState* state     = &node->data.canonicalize;
    const double                 tolerance = (state->tolerance > 0.0) ? state->tolerance : 1e-10;

    size_t count = 0U;

    if (node->input.kind != SIM_GRAPH_IR_INPUT_FIELD) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    SimField* input = sim_context_field(ctx->context, node->input.field_index);
    if (input == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    const SimFieldLayout* layout = &input->layout;
    count                        = sim_field_element_count(layout);
    const SimComplexDouble* src = sim_field_complex_data_const(input);
    if (src == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->rank == 0U || layout->rank > 2U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    if (state->capacity < count) {
        double complex* buf =
            (double complex*) realloc(state->buffer, count * sizeof(double complex));
        if (buf == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->buffer   = buf;
        state->capacity = count;
    }
    state->rank = layout->rank;
    state->shape[0] = layout->shape[0];
    state->shape[1] = (layout->rank == 2U) ? layout->shape[1] : 0U;
    if (layout->rank == 1U) {
        size_t stride0 = layout->strides[0];
        for (size_t i = 0U; i < layout->shape[0]; ++i) {
            size_t idx = i * stride0;
            state->buffer[i] = CMPLX(src[idx].re, src[idx].im);
        }
    } else {
        size_t rows = layout->shape[0];
        size_t cols = layout->shape[1];
        size_t stride0 = layout->strides[0];
        size_t stride1 = layout->strides[1];
        for (size_t y = 0U; y < rows; ++y) {
            size_t row_base = y * stride0;
            size_t out_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                size_t idx = row_base + x * stride1;
                state->buffer[out_base + x] = CMPLX(src[idx].re, src[idx].im);
            }
        }
    }
    if (count == 0U) {
        state->length = 0U;
        return SIM_RESULT_OK;
    }

    double max_err = sim_graph_ir_hermitian_error(state->buffer, state->rank, state->shape);
    if (max_err > tolerance) {
        if (!ctx->exploration_mode &&
            ctx->representation_mode != SIM_REPRESENTATION_MODE_EXPLORATION) {
            return SIM_RESULT_INVALID_STATE;
        }
        sim_context_log_warning(
            ctx->context,
            "GraphIR: projecting spectral data to Hermitian symmetry (err=%.3g tol=%.3g).",
            max_err,
            tolerance);
        sim_graph_ir_canonicalize_hermitian(state->buffer, state->rank, state->shape);
    }

    state->length = count;
    return SIM_RESULT_OK;
}

static SimResult sim_graph_ir_copy_real_to_buffer(const double* src,
                                                  const SimFieldLayout* layout,
                                                  double complex* dst) {
    if (src == NULL || layout == NULL || dst == NULL || layout->shape == NULL ||
        layout->strides == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->rank == 1U) {
        size_t stride0 = layout->strides[0];
        for (size_t i = 0U; i < layout->shape[0]; ++i) {
            size_t idx = i * stride0;
            dst[i] = CMPLX(src[idx], 0.0);
        }
        return SIM_RESULT_OK;
    }
    if (layout->rank == 2U) {
        size_t rows = layout->shape[0];
        size_t cols = layout->shape[1];
        size_t stride0 = layout->strides[0];
        size_t stride1 = layout->strides[1];
        for (size_t y = 0U; y < rows; ++y) {
            size_t row_base = y * stride0;
            size_t out_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                size_t idx = row_base + x * stride1;
                dst[out_base + x] = CMPLX(src[idx], 0.0);
            }
        }
        return SIM_RESULT_OK;
    }
    return SIM_RESULT_NOT_SUPPORTED;
}

static SimResult sim_graph_ir_copy_complex_to_buffer(const SimComplexDouble* src,
                                                     const SimFieldLayout* layout,
                                                     double complex* dst) {
    if (src == NULL || layout == NULL || dst == NULL || layout->shape == NULL ||
        layout->strides == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->rank == 1U) {
        size_t stride0 = layout->strides[0];
        for (size_t i = 0U; i < layout->shape[0]; ++i) {
            size_t idx = i * stride0;
            dst[i] = CMPLX(src[idx].re, src[idx].im);
        }
        return SIM_RESULT_OK;
    }
    if (layout->rank == 2U) {
        size_t rows = layout->shape[0];
        size_t cols = layout->shape[1];
        size_t stride0 = layout->strides[0];
        size_t stride1 = layout->strides[1];
        for (size_t y = 0U; y < rows; ++y) {
            size_t row_base = y * stride0;
            size_t out_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                size_t idx = row_base + x * stride1;
                dst[out_base + x] = CMPLX(src[idx].re, src[idx].im);
            }
        }
        return SIM_RESULT_OK;
    }
    return SIM_RESULT_NOT_SUPPORTED;
}

static SimResult sim_graph_ir_copy_buffer_to_real(const double complex* src,
                                                  const SimFieldLayout* layout,
                                                  double* dst) {
    if (src == NULL || layout == NULL || dst == NULL || layout->shape == NULL ||
        layout->strides == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->rank == 1U) {
        size_t stride0 = layout->strides[0];
        for (size_t i = 0U; i < layout->shape[0]; ++i) {
            size_t idx = i * stride0;
            dst[idx] = creal(src[i]);
        }
        return SIM_RESULT_OK;
    }
    if (layout->rank == 2U) {
        size_t rows = layout->shape[0];
        size_t cols = layout->shape[1];
        size_t stride0 = layout->strides[0];
        size_t stride1 = layout->strides[1];
        for (size_t y = 0U; y < rows; ++y) {
            size_t row_base = y * stride0;
            size_t in_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                size_t idx = row_base + x * stride1;
                dst[idx] = creal(src[in_base + x]);
            }
        }
        return SIM_RESULT_OK;
    }
    return SIM_RESULT_NOT_SUPPORTED;
}

static SimResult sim_graph_ir_copy_buffer_to_complex(const double complex* src,
                                                     const SimFieldLayout* layout,
                                                     SimComplexDouble* dst) {
    if (src == NULL || layout == NULL || dst == NULL || layout->shape == NULL ||
        layout->strides == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->rank == 1U) {
        size_t stride0 = layout->strides[0];
        for (size_t i = 0U; i < layout->shape[0]; ++i) {
            size_t idx = i * stride0;
            dst[idx].re = creal(src[i]);
            dst[idx].im = cimag(src[i]);
        }
        return SIM_RESULT_OK;
    }
    if (layout->rank == 2U) {
        size_t rows = layout->shape[0];
        size_t cols = layout->shape[1];
        size_t stride0 = layout->strides[0];
        size_t stride1 = layout->strides[1];
        for (size_t y = 0U; y < rows; ++y) {
            size_t row_base = y * stride0;
            size_t in_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                size_t idx = row_base + x * stride1;
                dst[idx].re = creal(src[in_base + x]);
                dst[idx].im = cimag(src[in_base + x]);
            }
        }
        return SIM_RESULT_OK;
    }
    return SIM_RESULT_NOT_SUPPORTED;
}

static SimResult sim_graph_ir_execute_fft(SimGraphIR*                     graph,
                                          SimGraphIRNode*                 node,
                                          const SimGraphIRCompileContext* ctx) {
    if (node == NULL || ctx == NULL || ctx->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimGraphIRFftState* state  = &node->data.fft;
    SimField*           output = NULL;
    size_t              count  = 0U;

    if (node->output.kind != SIM_GRAPH_IR_OUTPUT_FIELD) {
        return SIM_RESULT_INVALID_STATE;
    }

    output = sim_context_field(ctx->context, node->output.field_index);
    if (output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimFieldLayout* input_layout = NULL;
    const SimFieldLayout* output_layout = &output->layout;
    size_t                rank = 0U;
    size_t                shape[2] = { 0U, 0U };

    const double complex*   buffer_input   = NULL;
    const SimComplexDouble* field_complex  = NULL;
    const double*           field_real     = NULL;
    bool                    input_is_field = false;

    if (node->input.kind == SIM_GRAPH_IR_INPUT_FIELD) {
        SimField* input = sim_context_field(ctx->context, node->input.field_index);
        if (input == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        input_layout = &input->layout;
        rank = input_layout->rank;
        if (rank == 0U || rank > 2U) {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        if (output_layout->rank != rank) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        shape[0] = input_layout->shape[0];
        shape[1] = (rank == 2U) ? input_layout->shape[1] : 0U;
        if (output_layout->shape[0] != shape[0] ||
            (rank == 2U && output_layout->shape[1] != shape[1])) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_element_count(input_layout);
        if (count == 0U || sim_field_element_count(output_layout) != count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        input_is_field = true;
        if (sim_field_is_complex(input)) {
            field_complex = sim_field_complex_data_const(input);
        } else {
            field_real = sim_field_real_data_const(input);
        }
        if (field_complex == NULL && field_real == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else if (node->input.kind == SIM_GRAPH_IR_INPUT_NODE) {
        if (graph == NULL || node->input.node_id >= graph->node_count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        SimGraphIRNode* source = &graph->nodes[node->input.node_id];
        if (source->kind != SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT) {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        SimGraphIRCanonicalizeState* canon = &source->data.canonicalize;
        buffer_input                       = canon->buffer;
        count                              = canon->length;
        rank                               = canon->rank;
        shape[0]                           = canon->shape[0];
        shape[1]                           = canon->shape[1];
        if (buffer_input == NULL || count == 0U) {
            return SIM_RESULT_INVALID_STATE;
        }
        if (rank == 0U || rank > 2U) {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        if (output_layout->rank != rank) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (output_layout->shape[0] != shape[0] ||
            (rank == 2U && output_layout->shape[1] != shape[1])) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (sim_field_element_count(output_layout) != count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else {
        return SIM_RESULT_INVALID_STATE;
    }

    if (rank == 1U) {
        if (state->rank != 1U || state->plan.n != count) {
            fft_plan_destroy(&state->plan);
            fft_plan2d_destroy(&state->plan2d);
            SimResult plan_rc = fft_plan_init(&state->plan, count);
            if (plan_rc != SIM_RESULT_OK) {
                return plan_rc;
            }
            state->length = count;
        }
    } else {
        size_t rows = shape[0];
        size_t cols = shape[1];
        if (state->rank != 2U || state->shape[0] != rows || state->shape[1] != cols) {
            fft_plan_destroy(&state->plan);
            fft_plan2d_destroy(&state->plan2d);
            SimResult plan_rc = fft_plan2d_init(&state->plan2d, rows, cols, cols, 1U);
            if (plan_rc != SIM_RESULT_OK) {
                return plan_rc;
            }
            state->length = count;
        }
    }
    state->rank = rank;
    state->shape[0] = shape[0];
    state->shape[1] = shape[1];

    SimResult rc = sim_graph_ir_fft_ensure_capacity(state, count);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    if (node->kind == SIM_GRAPH_IR_NODE_FFT_FORWARD) {
        if (!input_is_field) {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        if (field_complex != NULL) {
            rc = sim_graph_ir_copy_complex_to_buffer(field_complex, input_layout, state->scratch_in);
        } else {
            rc = sim_graph_ir_copy_real_to_buffer(field_real, input_layout, state->scratch_in);
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        if (rank == 1U) {
            rc = fft_plan_forward(&state->plan, state->scratch_in, state->scratch_out);
        } else {
            rc = fft_plan2d_forward(&state->plan2d, state->scratch_in, state->scratch_out);
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        SimComplexDouble* dst = sim_field_complex_data(output);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        rc = sim_graph_ir_copy_buffer_to_complex(state->scratch_out, output_layout, dst);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
    } else {
        if (buffer_input != NULL) {
            (void) memcpy(state->scratch_in, buffer_input, count * sizeof(double complex));
        } else {
            if (field_complex == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            rc = sim_graph_ir_copy_complex_to_buffer(field_complex, input_layout, state->scratch_in);
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
        }

        if (rank == 1U) {
            rc = fft_plan_inverse(&state->plan, state->scratch_in, state->scratch_out);
        } else {
            rc = fft_plan2d_inverse(&state->plan2d, state->scratch_in, state->scratch_out);
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        if (state->output_real) {
            double* dst = sim_field_real_data(output);
            if (dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            rc = sim_graph_ir_copy_buffer_to_real(state->scratch_out, output_layout, dst);
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
        } else {
            SimComplexDouble* dst = sim_field_complex_data(output);
            if (dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            rc = sim_graph_ir_copy_buffer_to_complex(state->scratch_out, output_layout, dst);
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult sim_graph_ir_execute_promote(SimGraphIRNode*                 node,
                                              const SimGraphIRCompileContext* ctx) {
    if (node == NULL || ctx == NULL || ctx->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (node->input.kind != SIM_GRAPH_IR_INPUT_FIELD ||
        node->output.kind != SIM_GRAPH_IR_OUTPUT_FIELD) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimField* input  = sim_context_field(ctx->context, node->input.field_index);
    SimField* output = sim_context_field(ctx->context, node->output.field_index);
    SimScalarDomain input_domain;
    SimScalarDomain output_domain;
    if (input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input == output) {
        return sim_field_require_complex(output);
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U || sim_field_element_count(&output->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    input_domain  = sim_scalar_domain_from_field(input);
    output_domain = sim_scalar_domain_from_field(output);
    if (sim_scalar_domain_is_complex(input_domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (!sim_scalar_domain_is_complex(output_domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (input_domain.kind != SIM_SCALAR_DOMAIN_REAL &&
        input_domain.kind != SIM_SCALAR_DOMAIN_UNKNOWN) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    const double*     src = sim_field_real_data_const(input);
    SimComplexDouble* dst = sim_field_complex_data(output);
    if (src == NULL || dst == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < count; ++i) {
        dst[i].re = src[i];
        dst[i].im = 0.0;
    }

    return SIM_RESULT_OK;
}

static SimResult sim_graph_ir_execute_cast_copy(SimGraphIRNode*                 node,
                                                const SimGraphIRCompileContext* ctx) {
    if (node == NULL || ctx == NULL || ctx->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (node->input.kind != SIM_GRAPH_IR_INPUT_FIELD ||
        node->output.kind != SIM_GRAPH_IR_OUTPUT_FIELD) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimField* input  = sim_context_field(ctx->context, node->input.field_index);
    SimField* output = sim_context_field(ctx->context, node->output.field_index);
    if (input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!input->layout.contiguous || !output->layout.contiguous) {
        return SIM_RESULT_INVALID_STATE;
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U || sim_field_element_count(&output->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input->element_size != output->element_size) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    size_t bytes = count * input->element_size;
    memcpy(output->data, input->data, bytes);
    return SIM_RESULT_OK;
}

static bool sim_graph_ir_value_kind_compatible(SimFieldValueKind required,
                                               SimFieldValueKind actual) {
    if (required == actual) {
        return true;
    }
    if (required == SIM_FIELD_VALUE_COMPLEX_SCALAR &&
        sim_field_value_kind_is_complex_valued(actual)) {
        return true;
    }
    return false;
}

static SimResult sim_graph_ir_check_representation(const SimGraphIRInputRequirement* req,
                                                   SimFieldRepresentation            actual,
                                                   const SimGraphIRCompileContext*   ctx,
                                                   const char*                       label) {
    if (req == NULL || ctx == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (req->domain != SIM_FIELD_DOMAIN_UNKNOWN && req->domain != actual.domain) {
        if (ctx->exploration_mode ||
            ctx->representation_mode == SIM_REPRESENTATION_MODE_EXPLORATION) {
            sim_context_log_warning(ctx->context,
                                    "GraphIR: %s domain mismatch (req=%d got=%d).",
                                    label,
                                    (int) req->domain,
                                    (int) actual.domain);
        } else {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    }

    if (req->value_kind != SIM_FIELD_VALUE_REAL_SCALAR ||
        actual.value_kind != SIM_FIELD_VALUE_REAL_SCALAR) {
        if (!sim_graph_ir_value_kind_compatible(req->value_kind, actual.value_kind)) {
            if (ctx->exploration_mode ||
                ctx->representation_mode == SIM_REPRESENTATION_MODE_EXPLORATION) {
                sim_context_log_warning(ctx->context,
                                        "GraphIR: %s value kind mismatch (req=%d got=%d).",
                                        label,
                                        (int) req->value_kind,
                                        (int) actual.value_kind);
            } else {
                return SIM_RESULT_TYPE_MISMATCH;
            }
        }
    }

    return SIM_RESULT_OK;
}

void sim_graph_ir_init(SimGraphIR* graph) {
    if (graph == NULL) {
        return;
    }
    memset(graph, 0, sizeof(*graph));
}

void sim_graph_ir_destroy(SimGraphIR* graph) {
    if (graph == NULL) {
        return;
    }

    if (graph->nodes != NULL) {
        for (size_t i = 0U; i < graph->node_count; ++i) {
            SimGraphIRNode* node = &graph->nodes[i];
            if (node == NULL) {
                continue;
            }
            switch (node->kind) {
                case SIM_GRAPH_IR_NODE_POINTWISE_KERNEL:
                    sim_graph_ir_kernel_destroy(&node->data.pointwise);
                    break;
                case SIM_GRAPH_IR_NODE_FFT_FORWARD:
                case SIM_GRAPH_IR_NODE_FFT_INVERSE:
                    sim_graph_ir_fft_release(&node->data.fft);
                    break;
                case SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT:
                    free(node->data.canonicalize.buffer);
                    node->data.canonicalize.buffer   = NULL;
                    node->data.canonicalize.capacity = 0U;
                    node->data.canonicalize.length   = 0U;
                    node->data.canonicalize.rank     = 0U;
                    node->data.canonicalize.shape[0] = 0U;
                    node->data.canonicalize.shape[1] = 0U;
                    break;
                case SIM_GRAPH_IR_NODE_PROMOTE_COMPLEX:
                case SIM_GRAPH_IR_NODE_CAST_COPY:
                default:
                    break;
            }
        }
    }

    free(graph->nodes);
    free(graph->edges);
    graph->nodes           = NULL;
    graph->edges           = NULL;
    graph->node_count      = 0U;
    graph->node_capacity   = 0U;
    graph->edge_count      = 0U;
    graph->edge_capacity   = 0U;
    graph->rewind_fn       = NULL;
    graph->rewind_userdata = NULL;
}

static SimResult sim_graph_ir_reserve_nodes(SimGraphIR* graph, size_t count) {
    if (graph == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (graph->node_capacity >= count) {
        return SIM_RESULT_OK;
    }
    size_t new_capacity = graph->node_capacity > 0U ? graph->node_capacity : 4U;
    while (new_capacity < count) {
        new_capacity *= 2U;
    }
    SimGraphIRNode* nodes =
        (SimGraphIRNode*) realloc(graph->nodes, new_capacity * sizeof(SimGraphIRNode));
    if (nodes == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    graph->nodes         = nodes;
    graph->node_capacity = new_capacity;
    return SIM_RESULT_OK;
}

static SimResult sim_graph_ir_reserve_edges(SimGraphIR* graph, size_t count) {
    if (graph == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (graph->edge_capacity >= count) {
        return SIM_RESULT_OK;
    }
    size_t new_capacity = graph->edge_capacity > 0U ? graph->edge_capacity : 4U;
    while (new_capacity < count) {
        new_capacity *= 2U;
    }
    SimGraphIREdge* edges =
        (SimGraphIREdge*) realloc(graph->edges, new_capacity * sizeof(SimGraphIREdge));
    if (edges == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    graph->edges         = edges;
    graph->edge_capacity = new_capacity;
    return SIM_RESULT_OK;
}

SimResult sim_graph_ir_add_node(SimGraphIR*               graph,
                                const SimGraphIRNodeDesc* desc,
                                SimGraphIRNodeId*         out_node_id) {
    if (graph == NULL || desc == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimResult reserve = sim_graph_ir_reserve_nodes(graph, graph->node_count + 1U);
    if (reserve != SIM_RESULT_OK) {
        return reserve;
    }

    SimGraphIRNode node = { 0 };
    node.kind           = desc->kind;
    node.input          = desc->input;
    node.output         = desc->output;
    node.contract       = desc->contract;

    SimResult init_result = SIM_RESULT_OK;
    switch (desc->kind) {
        case SIM_GRAPH_IR_NODE_POINTWISE_KERNEL:
            if (desc->config.pointwise.kernel == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            init_result =
                sim_graph_ir_kernel_init(&node.data.pointwise, desc->config.pointwise.kernel);
            break;
        case SIM_GRAPH_IR_NODE_FFT_FORWARD:
        case SIM_GRAPH_IR_NODE_FFT_INVERSE:
            fft_plan_reset(&node.data.fft.plan);
            fft_plan2d_reset(&node.data.fft.plan2d);
            node.data.fft.length           = 0U;
            node.data.fft.rank             = 0U;
            node.data.fft.shape[0]         = 0U;
            node.data.fft.shape[1]         = 0U;
            node.data.fft.scratch_in       = NULL;
            node.data.fft.scratch_out      = NULL;
            node.data.fft.scratch_capacity = 0U;
            node.data.fft.output_real =
                (desc->contract.output.value_kind == SIM_FIELD_VALUE_REAL_SCALAR);
            break;
        case SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT:
            node.data.canonicalize.buffer    = NULL;
            node.data.canonicalize.capacity  = 0U;
            node.data.canonicalize.length    = 0U;
            node.data.canonicalize.rank      = 0U;
            node.data.canonicalize.shape[0]  = 0U;
            node.data.canonicalize.shape[1]  = 0U;
            node.data.canonicalize.tolerance = desc->config.canonicalize.tolerance;
            break;
        case SIM_GRAPH_IR_NODE_PROMOTE_COMPLEX:
        case SIM_GRAPH_IR_NODE_CAST_COPY:
        default:
            break;
    }

    if (init_result != SIM_RESULT_OK) {
        return init_result;
    }

    graph->nodes[graph->node_count] = node;
    if (out_node_id != NULL) {
        *out_node_id = graph->node_count;
    }
    graph->node_count += 1U;
    return SIM_RESULT_OK;
}

SimResult sim_graph_ir_add_edge(SimGraphIR* graph, SimGraphIRNodeId from, SimGraphIRNodeId to) {
    if (graph == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    SimResult reserve = sim_graph_ir_reserve_edges(graph, graph->edge_count + 1U);
    if (reserve != SIM_RESULT_OK) {
        return reserve;
    }
    graph->edges[graph->edge_count].from = from;
    graph->edges[graph->edge_count].to   = to;
    graph->edge_count += 1U;
    return SIM_RESULT_OK;
}

SimResult sim_graph_ir_fuse_pointwise(SimGraphIR* graph) {
    if (graph == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t i = 0U;
    while (i + 1U < graph->node_count) {
        SimGraphIRNode* left  = &graph->nodes[i];
        SimGraphIRNode* right = &graph->nodes[i + 1U];

        if (left->kind != SIM_GRAPH_IR_NODE_POINTWISE_KERNEL ||
            right->kind != SIM_GRAPH_IR_NODE_POINTWISE_KERNEL ||
            !sim_graph_ir_kernels_compatible(&left->data.pointwise, &right->data.pointwise)) {
            i += 1U;
            continue;
        }

        size_t new_count = left->data.pointwise.output_count + right->data.pointwise.output_count;
        SimKernelIROutput* new_outputs =
            (SimKernelIROutput*) calloc(new_count, sizeof(SimKernelIROutput));
        SimOperatorKernelOutputDescriptor* new_output_map =
            (SimOperatorKernelOutputDescriptor*) calloc(new_count,
                                                        sizeof(SimOperatorKernelOutputDescriptor));
        if (new_outputs == NULL || new_output_map == NULL) {
            free(new_outputs);
            free(new_output_map);
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        memcpy(new_outputs,
               left->data.pointwise.outputs,
               left->data.pointwise.output_count * sizeof(SimKernelIROutput));
        memcpy(new_output_map,
               left->data.pointwise.output_map,
               left->data.pointwise.output_count * sizeof(SimOperatorKernelOutputDescriptor));

        memcpy(new_outputs + left->data.pointwise.output_count,
               right->data.pointwise.outputs,
               right->data.pointwise.output_count * sizeof(SimKernelIROutput));
        memcpy(new_output_map + left->data.pointwise.output_count,
               right->data.pointwise.output_map,
               right->data.pointwise.output_count * sizeof(SimOperatorKernelOutputDescriptor));

        free(left->data.pointwise.outputs);
        free(left->data.pointwise.output_map);

        left->data.pointwise.outputs             = new_outputs;
        left->data.pointwise.output_map          = new_output_map;
        left->data.pointwise.output_count        = new_count;
        left->data.pointwise.kernel.outputs      = new_outputs;
        left->data.pointwise.kernel.output_count = new_count;

        if (left->data.pointwise.kernel.builder == right->data.pointwise.kernel.builder) {
            right->data.pointwise.kernel.builder = NULL;
        }
        sim_graph_ir_kernel_destroy(&right->data.pointwise);

        if (i + 2U < graph->node_count) {
            memmove(&graph->nodes[i + 1U],
                    &graph->nodes[i + 2U],
                    (graph->node_count - (i + 2U)) * sizeof(SimGraphIRNode));
        }
        graph->node_count -= 1U;

        for (size_t n = 0U; n < graph->node_count; ++n) {
            if (graph->nodes[n].input.kind == SIM_GRAPH_IR_INPUT_NODE) {
                if (graph->nodes[n].input.node_id == i + 1U) {
                    graph->nodes[n].input.node_id = i;
                } else if (graph->nodes[n].input.node_id > i + 1U) {
                    graph->nodes[n].input.node_id -= 1U;
                }
            }
        }

        for (size_t e = 0U; e < graph->edge_count; ++e) {
            if (graph->edges[e].from == i + 1U) {
                graph->edges[e].from = i;
            } else if (graph->edges[e].from > i + 1U) {
                graph->edges[e].from -= 1U;
            }

            if (graph->edges[e].to == i + 1U) {
                graph->edges[e].to = i;
            } else if (graph->edges[e].to > i + 1U) {
                graph->edges[e].to -= 1U;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_graph_ir_validate(SimGraphIR* graph, const SimGraphIRCompileContext* ctx) {
    if (graph == NULL || ctx == NULL || ctx->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimResult fuse_rc = sim_graph_ir_fuse_pointwise(graph);
    if (fuse_rc != SIM_RESULT_OK) {
        return fuse_rc;
    }

    for (size_t i = 0U; i < graph->edge_count; ++i) {
        if (graph->edges[i].from >= graph->node_count || graph->edges[i].to >= graph->node_count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    for (size_t i = 0U; i < graph->node_count; ++i) {
        const SimGraphIRNode* node = &graph->nodes[i];

        if (node->contract.purity.has_state && !node->contract.purity.reset_on_rewind) {
            if (ctx->representation_mode == SIM_REPRESENTATION_MODE_STRICT) {
                return SIM_RESULT_NOT_SUPPORTED;
            }
            sim_context_log_warning(
                ctx->context, "GraphIR: stateful node not rewind-safe; strict mode required.");
        }

        if (node->contract.purity.time_source == SIM_GRAPH_IR_TIME_ACCUMULATED &&
            ctx->representation_mode == SIM_REPRESENTATION_MODE_STRICT) {
            return SIM_RESULT_NOT_SUPPORTED;
        }

        if (node->input.kind == SIM_GRAPH_IR_INPUT_NODE &&
            node->input.node_id >= graph->node_count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (node->input.kind == SIM_GRAPH_IR_INPUT_NODE && node->input.node_id >= i) {
            return SIM_RESULT_DEPENDENCY_ERROR;
        }

        if (node->input.kind == SIM_GRAPH_IR_INPUT_FIELD) {
            SimField* input = sim_context_field(ctx->context, node->input.field_index);
            if (input == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            SimFieldRepresentation repr = sim_field_representation(input);
            SimResult              rep_rc =
                sim_graph_ir_check_representation(&node->contract.input, repr, ctx, "input");
            if (rep_rc != SIM_RESULT_OK) {
                return rep_rc;
            }
        } else if (node->input.kind == SIM_GRAPH_IR_INPUT_NODE) {
            const SimGraphIRNode*  source = &graph->nodes[node->input.node_id];
            SimFieldRepresentation repr   = { .domain     = source->contract.output.domain,
                                              .value_kind = source->contract.output.value_kind };
            SimResult              rep_rc =
                sim_graph_ir_check_representation(&node->contract.input, repr, ctx, "input");
            if (rep_rc != SIM_RESULT_OK) {
                return rep_rc;
            }
        }
        if (node->output.kind == SIM_GRAPH_IR_OUTPUT_FIELD) {
            SimField* output = sim_context_field(ctx->context, node->output.field_index);
            if (output == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            SimFieldRepresentation     repr    = sim_field_representation(output);
            SimGraphIRInputRequirement out_req = { .domain     = node->contract.output.domain,
                                                   .value_kind = node->contract.output.value_kind };
            SimResult rep_rc = sim_graph_ir_check_representation(&out_req, repr, ctx, "output");
            if (rep_rc != SIM_RESULT_OK) {
                return rep_rc;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_graph_ir_execute(SimGraphIR* graph, const SimGraphIRCompileContext* ctx) {
    if (graph == NULL || ctx == NULL || ctx->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < graph->node_count; ++i) {
        SimGraphIRNode* node = &graph->nodes[i];
        SimResult       rc   = SIM_RESULT_OK;
        switch (node->kind) {
            case SIM_GRAPH_IR_NODE_POINTWISE_KERNEL:
                rc = sim_graph_ir_execute_kernel(&node->data.pointwise, ctx->context);
                break;
            case SIM_GRAPH_IR_NODE_FFT_FORWARD:
            case SIM_GRAPH_IR_NODE_FFT_INVERSE:
                rc = sim_graph_ir_execute_fft(graph, node, ctx);
                break;
            case SIM_GRAPH_IR_NODE_PROMOTE_COMPLEX:
                rc = sim_graph_ir_execute_promote(node, ctx);
                break;
            case SIM_GRAPH_IR_NODE_CAST_COPY:
                rc = sim_graph_ir_execute_cast_copy(node, ctx);
                break;
            case SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT:
                rc = sim_graph_ir_execute_canonicalize(node, ctx);
                break;
            default:
                rc = SIM_RESULT_NOT_SUPPORTED;
                break;
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
    }

    return SIM_RESULT_OK;
}

void sim_graph_ir_notify_rewind(SimGraphIR* graph, size_t step_index) {
    if (graph == NULL) {
        return;
    }
    if (graph->rewind_fn != NULL) {
        graph->rewind_fn(graph, step_index, graph->rewind_userdata);
    }
    (void) step_index;
}

void sim_graph_ir_set_rewind_callback(SimGraphIR* graph,
                                      void (*rewind_fn)(SimGraphIR* graph,
                                                        size_t      step_index,
                                                        void*       userdata),
                                      void* userdata) {
    if (graph == NULL) {
        return;
    }
    graph->rewind_fn       = rewind_fn;
    graph->rewind_userdata = userdata;
}

void sim_graph_ir_compile_context_init(SimGraphIRCompileContext* ctx, SimContext* context) {
    if (ctx == NULL) {
        return;
    }
    ctx->context             = context;
    ctx->representation_mode = sim_context_representation_mode(context);
    ctx->exploration_mode    = (ctx->representation_mode == SIM_REPRESENTATION_MODE_EXPLORATION);
}

size_t sim_graph_ir_node_count(const SimGraphIR* graph) {
    return (graph != NULL) ? graph->node_count : 0U;
}

size_t sim_graph_ir_edge_count(const SimGraphIR* graph) {
    return (graph != NULL) ? graph->edge_count : 0U;
}

bool sim_graph_ir_node_view(const SimGraphIR* graph,
                            SimGraphIRNodeId  node_id,
                            SimGraphIRNodeView* out_view) {
    const SimGraphIRNode* node;

    if (out_view != NULL) {
        memset(out_view, 0, sizeof(*out_view));
        out_view->canonicalize_tolerance = 0.0;
        out_view->has_canonicalize_tolerance = false;
    }
    if (graph == NULL || out_view == NULL || node_id >= graph->node_count) {
        return false;
    }

    node = &graph->nodes[node_id];
    out_view->kind = node->kind;
    out_view->input = node->input;
    out_view->output = node->output;
    out_view->contract = node->contract;
    switch (node->kind) {
        case SIM_GRAPH_IR_NODE_POINTWISE_KERNEL:
            out_view->pointwise_binding_count = node->data.pointwise.binding_count;
            out_view->pointwise_output_count = node->data.pointwise.output_count;
            out_view->pointwise_param_count = node->data.pointwise.param_count;
            break;
        case SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT:
            out_view->has_canonicalize_tolerance = true;
            out_view->canonicalize_tolerance = node->data.canonicalize.tolerance;
            break;
        case SIM_GRAPH_IR_NODE_FFT_FORWARD:
        case SIM_GRAPH_IR_NODE_FFT_INVERSE:
        case SIM_GRAPH_IR_NODE_PROMOTE_COMPLEX:
        case SIM_GRAPH_IR_NODE_CAST_COPY:
        default:
            break;
    }
    return true;
}

bool sim_graph_ir_edge_view(const SimGraphIR* graph,
                            size_t            edge_index,
                            SimGraphIREdge*   out_edge) {
    if (out_edge != NULL) {
        memset(out_edge, 0, sizeof(*out_edge));
    }
    if (graph == NULL || out_edge == NULL || edge_index >= graph->edge_count) {
        return false;
    }
    *out_edge = graph->edges[edge_index];
    return true;
}
