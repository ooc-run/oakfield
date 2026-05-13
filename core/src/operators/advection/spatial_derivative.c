#include "oakfield/operators/advection/spatial_derivative.h"
#include "operators/common/operator_utils.h"

#include "oakfield/operator.h"
#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define SPATIAL_DERIVATIVE_SYMBOLIC_CAPACITY 96

typedef struct SimSpatialDerivativeOperatorState {
    SimSpatialDerivativeOperatorConfig config;
    double                             inv_spacing;
    char                               symbolic[SPATIAL_DERIVATIVE_SYMBOLIC_CAPACITY];
} SimSpatialDerivativeOperatorState;

static const char* spatial_derivative_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimSpatialDerivativeOperatorState* state =
        (const SimSpatialDerivativeOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimSpatialDerivativeMethod
spatial_derivative_effective_method(const SimSpatialDerivativeOperatorState* state) {
    SimSpatialDerivativeMethod method = state->config.method;
    if (state->config.skew_forward && method == SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL) {
        method = SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD;
    }
    return method;
}

static bool spatial_derivative_validate_fields(struct SimContext*                        context,
                                               const SimSpatialDerivativeOperatorConfig* cfg) {
    SimField* input;
    SimField* output;
    size_t    input_count;
    size_t    output_count;
    size_t    rank;

    if (!context || !cfg) {
        return false;
    }

    input  = sim_context_field(context, cfg->input_field);
    output = sim_context_field(context, cfg->output_field);
    if (!input || !output) {
        return false;
    }

    if (input == output) {
        return false;
    }

    rank = sim_field_rank(input);
    if (rank == 0U || rank != sim_field_rank(output)) {
        return false;
    }

    if (rank < 1U || rank > 2U) {
        return false;
    }

    if (cfg->axis >= rank) {
        return false;
    }

    input_count  = sim_field_element_count(&input->layout);
    output_count = sim_field_element_count(&output->layout);
    if (input_count != output_count || input_count == 0U) {
        return false;
    }

    if (input->element_size != output->element_size) {
        return false;
    }

    if (!(input->element_size == sizeof(double) ||
          input->element_size == sizeof(SimComplexDouble))) {
        return false;
    }

    return true;
}

static const char* spatial_derivative_method_label(SimSpatialDerivativeMethod method) {
    switch (method) {
        case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
            return "forward";
        case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
            return "backward";
        case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
        default:
            return "central";
    }
}

static void spatial_derivative_refresh_symbolic(SimSpatialDerivativeOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    (void) snprintf(
        state->symbolic,
        sizeof(state->symbolic),
        "spatial_derivative axis=%zu method=%s skew=%s dx=%.4g boundary=%s accumulate=%s",
        state->config.axis,
        spatial_derivative_method_label(state->config.method),
        state->config.skew_forward ? "forward" : "symmetric",
        state->config.spacing,
        sim_boundary_policy_name(state->config.boundary),
        state->config.accumulate ? "true" : "false");
#else
    (void) state;
#endif
}

typedef struct SpatialDerivativeNeighbors {
    bool   has_forward;
    bool   has_backward;
    size_t forward_index;
    size_t backward_index;
} SpatialDerivativeNeighbors;

static void spatial_derivative_axis_neighbors(size_t                      element_index,
                                              size_t                      axis,
                                              const size_t*               shape,
                                              const size_t*               strides,
                                              size_t                      rank,
                                              SimIRBoundaryPolicy         boundary,
                                              SpatialDerivativeNeighbors* out) {
    if (out == NULL) {
        return;
    }
    out->has_forward    = false;
    out->has_backward   = false;
    out->forward_index  = element_index;
    out->backward_index = element_index;

    if (shape == NULL || strides == NULL || axis >= rank) {
        return;
    }

    size_t stride = strides[axis];
    size_t extent = shape[axis];
    if (extent == 0U || stride == 0U) {
        return;
    }

    size_t coord      = (element_index / stride) % extent;
    out->has_forward  = (coord + 1U < extent);
    out->has_backward = (coord > 0U);
    if (out->has_forward) {
        out->forward_index = element_index + stride;
    }
    if (out->has_backward) {
        out->backward_index = element_index - stride;
    }

    if (!out->has_forward) {
        switch (boundary) {
            case SIM_IR_BOUNDARY_PERIODIC:
                out->has_forward = (extent > 1U);
                if (out->has_forward) {
                    out->forward_index = element_index - (extent - 1U) * stride;
                }
                break;
            case SIM_IR_BOUNDARY_REFLECTIVE:
                out->has_forward = (extent > 1U);
                if (out->has_forward) {
                    out->forward_index =
                        (element_index >= stride) ? element_index - stride : element_index;
                }
                break;
            case SIM_IR_BOUNDARY_DIRICHLET:
            case SIM_IR_BOUNDARY_NEUMANN:
            default:
                break;
        }
    }

    if (!out->has_backward) {
        switch (boundary) {
            case SIM_IR_BOUNDARY_PERIODIC:
                out->has_backward = (extent > 1U);
                if (out->has_backward) {
                    out->backward_index = element_index + (extent - 1U) * stride;
                }
                break;
            case SIM_IR_BOUNDARY_REFLECTIVE:
                out->has_backward = (extent > 1U);
                if (out->has_backward) {
                    out->backward_index = element_index + stride;
                }
                break;
            case SIM_IR_BOUNDARY_DIRICHLET:
            case SIM_IR_BOUNDARY_NEUMANN:
            default:
                break;
        }
    }
}

static void spatial_derivative_apply_real(const SimSpatialDerivativeOperatorState* state,
                                          const SimField*                          field,
                                          const double*                            src,
                                          double*                                  dst,
                                          size_t                                   count) {
    if (!state || !field || !src || !dst || count == 0U) {
        return;
    }

    const size_t* shape   = sim_field_shape(field);
    const size_t* strides = sim_field_strides(field);
    size_t        rank    = sim_field_rank(field);
    if (shape == NULL || strides == NULL || rank == 0U || state->config.axis >= rank) {
        return;
    }

    const double               inv_dx        = state->inv_spacing;
    const double               central_scale = 0.5 * inv_dx;
    SimSpatialDerivativeMethod method        = spatial_derivative_effective_method(state);
    SimIRBoundaryPolicy        boundary      = state->config.boundary;

    size_t stride = strides[state->config.axis];
    size_t extent = shape[state->config.axis];

    if (boundary == SIM_IR_BOUNDARY_PERIODIC) {
        for (size_t idx = 0U; idx < count; ++idx) {
            size_t coord = (idx / stride) % extent;
            size_t forward_index =
                (coord + 1U < extent) ? (idx + stride) : (idx - (extent - 1U) * stride);
            size_t backward_index = (coord > 0U) ? (idx - stride) : (idx + (extent - 1U) * stride);

            double center         = src[idx];
            double forward_value  = src[forward_index];
            double backward_value = src[backward_index];

            double value = 0.0;
            switch (method) {
                case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                    value = (forward_value - center) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                    value = (center - backward_value) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
                default:
                    value = (forward_value - backward_value) * central_scale;
                    break;
            }

            if (state->config.accumulate) {
                dst[idx] += value;
            } else {
                dst[idx] = value;
            }
        }
        return;
    }

    if (boundary == SIM_IR_BOUNDARY_REFLECTIVE) {
        for (size_t idx = 0U; idx < count; ++idx) {
            size_t coord = (idx / stride) % extent;
            size_t forward_index =
                (coord + 1U < extent) ? (idx + stride) : ((extent > 1U) ? (idx - stride) : idx);
            size_t backward_index =
                (coord > 0U) ? (idx - stride) : ((extent > 1U) ? (idx + stride) : idx);

            double center         = src[idx];
            double forward_value  = src[forward_index];
            double backward_value = src[backward_index];

            double value = 0.0;
            switch (method) {
                case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                    value = (forward_value - center) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                    value = (center - backward_value) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
                default:
                    value = (forward_value - backward_value) * central_scale;
                    break;
            }

            if (state->config.accumulate) {
                dst[idx] += value;
            } else {
                dst[idx] = value;
            }
        }
        return;
    }

    for (size_t idx = 0U; idx < count; ++idx) {
        SpatialDerivativeNeighbors nb = { 0 };
        spatial_derivative_axis_neighbors(
            idx, state->config.axis, shape, strides, rank, boundary, &nb);

        double center         = src[idx];
        double forward_value  = center;
        double backward_value = center;

        if (nb.has_forward) {
            forward_value = src[nb.forward_index];
        }
        if (nb.has_backward) {
            backward_value = src[nb.backward_index];
        }

        if (!nb.has_forward && boundary == SIM_IR_BOUNDARY_DIRICHLET) {
            forward_value = 0.0;
        }
        if (!nb.has_backward && boundary == SIM_IR_BOUNDARY_DIRICHLET) {
            backward_value = 0.0;
        }
        if (!nb.has_forward && boundary == SIM_IR_BOUNDARY_NEUMANN) {
            forward_value = center;
        }
        if (!nb.has_backward && boundary == SIM_IR_BOUNDARY_NEUMANN) {
            backward_value = center;
        }

        double value = 0.0;
        switch (method) {
            case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                value = (forward_value - center) * inv_dx;
                break;
            case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                value = (center - backward_value) * inv_dx;
                break;
            case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
            default:
                value = (forward_value - backward_value) * central_scale;
                break;
        }

        if (state->config.accumulate) {
            dst[idx] += value;
        } else {
            dst[idx] = value;
        }
    }
}

static void spatial_derivative_apply_complex(const SimSpatialDerivativeOperatorState* state,
                                             const SimField*                          field,
                                             const SimComplexDouble*                  src,
                                             SimComplexDouble*                        dst,
                                             size_t                                   count) {
    if (!state || !field || !src || !dst || count == 0U) {
        return;
    }

    const size_t* shape   = sim_field_shape(field);
    const size_t* strides = sim_field_strides(field);
    size_t        rank    = sim_field_rank(field);
    if (shape == NULL || strides == NULL || rank == 0U || state->config.axis >= rank) {
        return;
    }

    const double               inv_dx        = state->inv_spacing;
    const double               central_scale = 0.5 * inv_dx;
    SimSpatialDerivativeMethod method        = spatial_derivative_effective_method(state);
    SimIRBoundaryPolicy        boundary      = state->config.boundary;

    size_t stride = strides[state->config.axis];
    size_t extent = shape[state->config.axis];

    if (boundary == SIM_IR_BOUNDARY_PERIODIC) {
        for (size_t idx = 0U; idx < count; ++idx) {
            size_t coord = (idx / stride) % extent;
            size_t forward_index =
                (coord + 1U < extent) ? (idx + stride) : (idx - (extent - 1U) * stride);
            size_t backward_index = (coord > 0U) ? (idx - stride) : (idx + (extent - 1U) * stride);

            SimComplexDouble center         = src[idx];
            SimComplexDouble forward_value  = src[forward_index];
            SimComplexDouble backward_value = src[backward_index];

            double re_value = 0.0;
            double im_value = 0.0;

            switch (method) {
                case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                    re_value = (forward_value.re - center.re) * inv_dx;
                    im_value = (forward_value.im - center.im) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                    re_value = (center.re - backward_value.re) * inv_dx;
                    im_value = (center.im - backward_value.im) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
                default:
                    re_value = (forward_value.re - backward_value.re) * central_scale;
                    im_value = (forward_value.im - backward_value.im) * central_scale;
                    break;
            }

            if (state->config.accumulate) {
                dst[idx].re += re_value;
                dst[idx].im += im_value;
            } else {
                dst[idx].re = re_value;
                dst[idx].im = im_value;
            }
        }
        return;
    }

    if (boundary == SIM_IR_BOUNDARY_REFLECTIVE) {
        for (size_t idx = 0U; idx < count; ++idx) {
            size_t coord = (idx / stride) % extent;
            size_t forward_index =
                (coord + 1U < extent) ? (idx + stride) : ((extent > 1U) ? (idx - stride) : idx);
            size_t backward_index =
                (coord > 0U) ? (idx - stride) : ((extent > 1U) ? (idx + stride) : idx);

            SimComplexDouble center         = src[idx];
            SimComplexDouble forward_value  = src[forward_index];
            SimComplexDouble backward_value = src[backward_index];

            double re_value = 0.0;
            double im_value = 0.0;

            switch (method) {
                case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                    re_value = (forward_value.re - center.re) * inv_dx;
                    im_value = (forward_value.im - center.im) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                    re_value = (center.re - backward_value.re) * inv_dx;
                    im_value = (center.im - backward_value.im) * inv_dx;
                    break;
                case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
                default:
                    re_value = (forward_value.re - backward_value.re) * central_scale;
                    im_value = (forward_value.im - backward_value.im) * central_scale;
                    break;
            }

            if (state->config.accumulate) {
                dst[idx].re += re_value;
                dst[idx].im += im_value;
            } else {
                dst[idx].re = re_value;
                dst[idx].im = im_value;
            }
        }
        return;
    }

    for (size_t idx = 0U; idx < count; ++idx) {
        SpatialDerivativeNeighbors nb = { 0 };
        spatial_derivative_axis_neighbors(
            idx, state->config.axis, shape, strides, rank, boundary, &nb);

        SimComplexDouble center         = src[idx];
        SimComplexDouble forward_value  = center;
        SimComplexDouble backward_value = center;

        if (nb.has_forward) {
            forward_value = src[nb.forward_index];
        }
        if (nb.has_backward) {
            backward_value = src[nb.backward_index];
        }

        if (!nb.has_forward && boundary == SIM_IR_BOUNDARY_DIRICHLET) {
            forward_value.re = forward_value.im = 0.0;
        }
        if (!nb.has_backward && boundary == SIM_IR_BOUNDARY_DIRICHLET) {
            backward_value.re = backward_value.im = 0.0;
        }
        if (!nb.has_forward && boundary == SIM_IR_BOUNDARY_NEUMANN) {
            forward_value = center;
        }
        if (!nb.has_backward && boundary == SIM_IR_BOUNDARY_NEUMANN) {
            backward_value = center;
        }

        double re_value = 0.0;
        double im_value = 0.0;

        switch (method) {
            case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                re_value = (forward_value.re - center.re) * inv_dx;
                im_value = (forward_value.im - center.im) * inv_dx;
                break;
            case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                re_value = (center.re - backward_value.re) * inv_dx;
                im_value = (center.im - backward_value.im) * inv_dx;
                break;
            case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
            default:
                re_value = (forward_value.re - backward_value.re) * central_scale;
                im_value = (forward_value.im - backward_value.im) * central_scale;
                break;
        }

        if (state->config.accumulate) {
            dst[idx].re += re_value;
            dst[idx].im += im_value;
        } else {
            dst[idx].re = re_value;
            dst[idx].im = im_value;
        }
    }
}

static SimResult spatial_derivative_evaluate(SimSpatialDerivativeOperatorState* state,
                                             struct SimContext*                 context) {
    SimField* input;
    SimField* output;
    size_t    count;

    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    input  = sim_context_field(context, state->config.input_field);
    output = sim_context_field(context, state->config.output_field);
    if (!input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input == output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (count != sim_field_element_count(&output->layout)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input->element_size != output->element_size) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input->element_size == sizeof(double)) {
        const double* src = sim_field_real_data_const(input);
        double*       dst = sim_field_real_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        spatial_derivative_apply_real(state, input, src, dst, count);
    } else if (input->element_size == sizeof(SimComplexDouble)) {
        const SimComplexDouble* src = sim_field_complex_data_const(input);
        SimComplexDouble*       dst = sim_field_complex_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        spatial_derivative_apply_complex(state, input, src, dst, count);
    } else {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    return SIM_RESULT_OK;
}

static SimResult spatial_derivative_step(void*               state_ptr,
                                         struct SimContext*  context,
                                         struct SimOperator* self,
                                         size_t              substep_index,
                                         double              dt_sub,
                                         void*               scratch,
                                         size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) dt_sub;
    (void) scratch;
    (void) scratch_size;
    return spatial_derivative_evaluate((SimSpatialDerivativeOperatorState*) state_ptr, context);
}

static void spatial_derivative_normalize_config(SimSpatialDerivativeOperatorConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->spacing) || config->spacing <= 0.0) {
        config->spacing = 1.0;
    }

    switch (config->method) {
        case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
        case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
        case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
            break;
        default:
            config->method = SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL;
            break;
    }

    /* axis will be validated against field rank later; keep as-is */

    switch (config->boundary) {
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_PERIODIC:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            break;
        default:
            config->boundary = SIM_IR_BOUNDARY_PERIODIC;
            break;
    }

    config->skew_forward = config->skew_forward ? true : false;
    config->accumulate   = config->accumulate ? true : false;
}

static int spatial_derivative_iequals(char a, char b) {
    return tolower((unsigned char) a) == tolower((unsigned char) b);
}

static bool spatial_derivative_string_iequals(const char* lhs, const char* rhs) {
    size_t i;

    if (!lhs || !rhs) {
        return false;
    }

    for (i = 0U;; ++i) {
        char ca = lhs[i];
        char cb = rhs[i];
        if (ca == '\0' && cb == '\0') {
            return true;
        }
        if (ca == '\0' || cb == '\0') {
            return false;
        }
        if (!spatial_derivative_iequals(ca, cb)) {
            return false;
        }
    }
}

const char* sim_spatial_derivative_method_name(SimSpatialDerivativeMethod method) {
    return spatial_derivative_method_label(method);
}

bool sim_spatial_derivative_method_from_string(const char*                 name,
                                               SimSpatialDerivativeMethod* out_method) {
    if (!name || !out_method) {
        return false;
    }

    if (spatial_derivative_string_iequals(name, "central")) {
        *out_method = SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL;
        return true;
    }
    if (spatial_derivative_string_iequals(name, "forward")) {
        *out_method = SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD;
        return true;
    }
    if (spatial_derivative_string_iequals(name, "backward")) {
        *out_method = SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD;
        return true;
    }

    return false;
}

SimResult sim_add_spatial_derivative_operator(struct SimContext*                        context,
                                              const SimSpatialDerivativeOperatorConfig* config,
                                              size_t*                                   out_index) {
    SimSpatialDerivativeOperatorState* state;
    SimSpatialDerivativeOperatorConfig local     = { 0 };
    SimOperatorInfo                    info      = sim_operator_info_defaults();
    SimOperatorConfig                  op_config = sim_operator_config_defaults();
    SimSplitPort                       ports[2];
    SimSplitAccess                     accesses[2];
    SimSplitSubstep                    substep = { 0 };
    SimSplitDescriptor                 desc    = { 0 };
    char                               name[SIM_OPERATOR_NAME_MAX + 1U];
    SimResult                          result;
    bool                               registered_kernel = false;

    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state =
        (SimSpatialDerivativeOperatorState*) calloc(1U, sizeof(SimSpatialDerivativeOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (config) {
        local = *config;
    } else {
        local.input_field  = 0U;
        local.output_field = 0U;
        local.spacing      = 1.0;
        local.method       = SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL;
        local.axis         = 0U;
        local.skew_forward = false;
        local.accumulate   = false;
        local.boundary     = SIM_IR_BOUNDARY_PERIODIC;
    }

    spatial_derivative_normalize_config(&local);

    if (!spatial_derivative_validate_fields(context, &local)) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config      = local;
    state->inv_spacing = 1.0 / local.spacing;
    spatial_derivative_refresh_symbolic(state);

    SimField* input         = sim_context_field(context, local.input_field);
    SimField* output        = sim_context_field(context, local.output_field);
    bool      needs_complex = (input != NULL && sim_field_is_complex(input)) ||
                              (output != NULL && sim_field_is_complex(output));
    info.category           = SIM_OPERATOR_CATEGORY_ADVECTION;
    info.warp_level         = SIM_WARP_LEVEL_NONE;
    info.is_noise           = false;
    info.is_spectral        = false;
    info.is_local           = true;
    info.is_nonlocal        = false;
    info.is_linear          = true;
    info.is_warp            = false;
    info.is_differentiable  = true;
    info.preserves_real     = true;
    info.preferred_dt       = 0.0;
    info.abstract_id        = "spatial_derivative";
    sim_operator_info_set_schema_identity(&info, "spatial_derivative");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;
    info.representation.boundary                        = local.boundary;
    info.representation.spacing_hint_rank               = 1U;
    info.representation.spacing_hint[0]                 = local.spacing;
    info.approximation.spatial_order                    = 1.0;
    info.approximation.stencil_order =
        (spatial_derivative_effective_method(state) == SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL) ? 2.0
                                                                                              : 1.0;

    sim_operator_make_unique_name(name, sizeof(name), "spatial_derivative");

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "spatial_derivative")) {
        if (input != NULL && output != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[2];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                bool      is_complex = needs_complex;
                SimIRType value_type = is_complex ? sim_ir_type_vector(2U) : sim_ir_type_scalar();
                value_type.scalar_domain =
                    is_complex ? sim_scalar_domain_c64() : sim_scalar_domain_f64();

                SimIRNodeId input_node  = sim_ir_builder_field_ref_typed(builder, 0U, value_type);
                SimIRNodeId output_node = SIM_IR_INVALID_NODE;
                if (local.accumulate) {
                    output_node = sim_ir_builder_field_ref_typed(builder, 1U, value_type);
                }

                SimIRDiffSpec diff_spec;
                sim_ir_diff_spec_init(&diff_spec, builder);
                diff_spec.operand              = input_node;
                diff_spec.axis                 = local.axis;
                diff_spec.dx                   = local.spacing;
                diff_spec.scale                = 1.0;
                diff_spec.order                = 1U;
                diff_spec.stencil_order        = (spatial_derivative_effective_method(state) ==
                                                  SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL)
                                                     ? 2U
                                                     : 1U;
                diff_spec.consistency_constant = 0.0;
                diff_spec.boundary             = local.boundary;
                diff_spec.result_type          = value_type;

                SimSpatialDerivativeMethod effective_method =
                    spatial_derivative_effective_method(state);
                switch (effective_method) {
                    case SIM_SPATIAL_DERIVATIVE_METHOD_FORWARD:
                        diff_spec.method = SIM_IR_DIFF_METHOD_FORWARD;
                        break;
                    case SIM_SPATIAL_DERIVATIVE_METHOD_BACKWARD:
                        diff_spec.method = SIM_IR_DIFF_METHOD_BACKWARD;
                        break;
                    case SIM_SPATIAL_DERIVATIVE_METHOD_CENTRAL:
                    default:
                        diff_spec.method = SIM_IR_DIFF_METHOD_CENTRAL;
                        break;
                }

                SimIRNodeId diff_node  = sim_ir_builder_diff_spec(builder, &diff_spec);
                SimIRNodeId expression = diff_node;
                if (local.accumulate) {
                    expression =
                        sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, output_node, diff_node);
                }

                if (input_node != SIM_IR_INVALID_NODE && diff_node != SIM_IR_INVALID_NODE &&
                    expression != SIM_IR_INVALID_NODE &&
                    (!local.accumulate || output_node != SIM_IR_INVALID_NODE)) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.input_field;
                    bindings[1].ir_field_index      = 1U;
                    bindings[1].context_field_index = local.output_field;

                    outputs[0].ir_field_index = 1U;
                    outputs[0].expression     = expression;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 2U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = 0U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = free;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    kdesc.read_mask             = 0ULL;
                    kdesc.write_mask            = 0ULL;
                    if (local.input_field < 64U) {
                        kdesc.read_mask |= (1ULL << local.input_field);
                    }
                    if (local.output_field < 64U) {
                        kdesc.write_mask |= (1ULL << local.output_field);
                        if (local.accumulate) {
                            kdesc.read_mask |= (1ULL << local.output_field);
                        }
                    }

                    result = sim_context_register_operator(context, &kdesc, out_index);
                    if (result == SIM_RESULT_OK) {
                        registered_kernel = true;
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    ports[0].context_field_index = local.input_field;
    ports[0].require_complex     = needs_complex;
    ports[1].context_field_index = local.output_field;
    ports[1].require_complex     = needs_complex;

    accesses[0].port = 0U;
    accesses[0].mode = SIM_ACCESS_READ;
    accesses[1].port = 1U;
    accesses[1].mode = SIM_ACCESS_WRITE;

    substep.name              = NULL;
    substep.fn                = spatial_derivative_step;
    substep.accesses          = accesses;
    substep.access_count      = 2U;
    substep.dt_scale          = 1.0;
    substep.barrier_after     = false;
    substep.error_measure     = NULL;
    substep.required_features = 0U;

    desc.name                     = name;
    desc.ports                    = ports;
    desc.port_count               = 2U;
    desc.substeps                 = &substep;
    desc.substep_count            = 1U;
    desc.state                    = state;
    desc.symbolic                 = spatial_derivative_symbolic;
    desc.destroy                  = free;
    desc.info                     = info;
    desc.config                   = op_config;
    desc.scratch.bytes_per_worker = 0U;
    desc.scratch.alignment        = 0U;

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_spatial_derivative_config(struct SimContext*                  context,
                                        size_t                              operator_index,
                                        SimSpatialDerivativeOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSpatialDerivativeOperatorState* state =
        (SimSpatialDerivativeOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_spatial_derivative_update(struct SimContext*                        context,
                                        size_t                                    operator_index,
                                        const SimSpatialDerivativeOperatorConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSpatialDerivativeOperatorState* state =
        (SimSpatialDerivativeOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimSpatialDerivativeOperatorConfig local = state->config;
    if (config) {
        local = *config;
    }

    spatial_derivative_normalize_config(&local);
    if (!spatial_derivative_validate_fields(context, &local)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config      = local;
    state->inv_spacing = 1.0 / local.spacing;
    spatial_derivative_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
