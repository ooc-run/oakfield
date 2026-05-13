#include "oakfield/operators/advection/curl.h"
#include "operators/common/nd_neighbors.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CURL_EPS 1.0e-12

typedef struct SimCurlOperatorState {
    SimCurlOperatorConfig config;
    char                  symbolic[128];
} SimCurlOperatorState;

static const char* curl_stencil_name(SimCurlStencil stencil) {
    switch (stencil) {
        case SIM_CURL_STENCIL_CENTRAL_4:
            return "central4";
        case SIM_CURL_STENCIL_FORWARD_1:
            return "forward1";
        case SIM_CURL_STENCIL_BACKWARD_1:
            return "backward1";
        case SIM_CURL_STENCIL_FORWARD_2:
            return "forward2";
        case SIM_CURL_STENCIL_BACKWARD_2:
            return "backward2";
        case SIM_CURL_STENCIL_CENTRAL_2:
        default:
            return "central2";
    }
}

static void curl_refresh_symbolic(SimCurlOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "curl stencil=%s dx=%.4g dy=%.4g boundary=%s accum=%s scale_by_dt=%s",
                    curl_stencil_name(state->config.stencil),
                    state->config.spacing_x,
                    state->config.spacing_y,
                    sim_boundary_policy_name(state->config.boundary),
                    state->config.accumulate ? "true" : "false",
                    state->config.scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* curl_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimCurlOperatorState* state = (const SimCurlOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void curl_normalize_config(SimCurlOperatorConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->spacing_x) || fabs(config->spacing_x) <= CURL_EPS) {
        config->spacing_x = 1.0;
    }
    if (!isfinite(config->spacing_y) || fabs(config->spacing_y) <= CURL_EPS) {
        config->spacing_y = config->spacing_x;
    }

    switch (config->stencil) {
        case SIM_CURL_STENCIL_CENTRAL_2:
        case SIM_CURL_STENCIL_CENTRAL_4:
        case SIM_CURL_STENCIL_FORWARD_1:
        case SIM_CURL_STENCIL_BACKWARD_1:
        case SIM_CURL_STENCIL_FORWARD_2:
        case SIM_CURL_STENCIL_BACKWARD_2:
            break;
        default:
            config->stencil = SIM_CURL_STENCIL_CENTRAL_2;
            break;
    }

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

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static bool curl_layout_matches(const SimField* lhs, const SimField* rhs) {
    if (!lhs || !rhs || lhs->layout.rank != rhs->layout.rank) {
        return false;
    }

    for (size_t axis = 0U; axis < lhs->layout.rank; ++axis) {
        if (lhs->layout.shape == NULL || rhs->layout.shape == NULL ||
            lhs->layout.shape[axis] != rhs->layout.shape[axis]) {
            return false;
        }
    }

    return true;
}

static bool
curl_validate_fields(const SimField* input_x, const SimField* input_y, const SimField* output) {
    if (!input_x || !input_y || !output) {
        return false;
    }

    size_t count = sim_field_element_count(&input_x->layout);
    if (count == 0U || sim_field_element_count(&input_y->layout) != count ||
        sim_field_element_count(&output->layout) != count) {
        return false;
    }

    if (!curl_layout_matches(input_x, input_y) || !curl_layout_matches(input_x, output)) {
        return false;
    }

    if (input_x->element_size != input_y->element_size ||
        input_x->element_size != output->element_size) {
        return false;
    }

    if (!(input_x->element_size == sizeof(double) ||
          input_x->element_size == sizeof(SimComplexDouble))) {
        return false;
    }

    return true;
}

static bool curl_resolve_axes(const SimField*              field,
                              const SimCurlOperatorConfig* cfg,
                              size_t*                      out_axis_x,
                              size_t*                      out_axis_y) {
    if (!field || !cfg || !out_axis_x || !out_axis_y) {
        return false;
    }

    size_t rank = sim_field_rank(field);
    if (rank < 2U) {
        return false;
    }

    size_t axis_x = (cfg->axis_x == SIM_CURL_AXIS_AUTO) ? (rank - 1U) : cfg->axis_x;
    size_t axis_y = (cfg->axis_y == SIM_CURL_AXIS_AUTO) ? (rank - 2U) : cfg->axis_y;

    if (axis_x >= rank || axis_y >= rank || axis_x == axis_y) {
        return false;
    }

    *out_axis_x = axis_x;
    *out_axis_y = axis_y;
    return true;
}

static double curl_sample_real(const SimField*     field,
                               const double*       src,
                               size_t              index,
                               size_t              axis,
                               ptrdiff_t           offset,
                               SimIRBoundaryPolicy boundary,
                               double              center) {
    SimNdNeighbor neighbor = { 0 };
    SimResult     rc = sim_nd_axis_offset_neighbor(field, index, axis, offset, boundary, &neighbor);
    if (rc != SIM_RESULT_OK || !neighbor.valid) {
        return (boundary == SIM_IR_BOUNDARY_DIRICHLET) ? 0.0 : center;
    }
    return src[neighbor.index];
}

static void curl_sample_complex(const SimField*         field,
                                const SimComplexDouble* src,
                                size_t                  index,
                                size_t                  axis,
                                ptrdiff_t               offset,
                                SimIRBoundaryPolicy     boundary,
                                double                  center_re,
                                double                  center_im,
                                double*                 out_re,
                                double*                 out_im) {
    SimNdNeighbor neighbor = { 0 };
    SimResult     rc = sim_nd_axis_offset_neighbor(field, index, axis, offset, boundary, &neighbor);
    if (rc != SIM_RESULT_OK || !neighbor.valid) {
        if (boundary == SIM_IR_BOUNDARY_DIRICHLET) {
            *out_re = 0.0;
            *out_im = 0.0;
        } else {
            *out_re = center_re;
            *out_im = center_im;
        }
        return;
    }

    *out_re = src[neighbor.index].re;
    *out_im = src[neighbor.index].im;
}

static double curl_axis_real(const SimField*              field,
                             const double*                src,
                             size_t                       index,
                             size_t                       axis,
                             const SimCurlOperatorConfig* cfg,
                             double                       dx) {
    double f0 = src[index];
    switch (cfg->stencil) {
        case SIM_CURL_STENCIL_FORWARD_1: {
            double fp1 = curl_sample_real(field, src, index, axis, 1, cfg->boundary, f0);
            return (fp1 - f0) / dx;
        }
        case SIM_CURL_STENCIL_BACKWARD_1: {
            double fm1 = curl_sample_real(field, src, index, axis, -1, cfg->boundary, f0);
            return (f0 - fm1) / dx;
        }
        case SIM_CURL_STENCIL_FORWARD_2: {
            double fp1 = curl_sample_real(field, src, index, axis, 1, cfg->boundary, f0);
            double fp2 = curl_sample_real(field, src, index, axis, 2, cfg->boundary, f0);
            return (-3.0 * f0 + 4.0 * fp1 - fp2) / (2.0 * dx);
        }
        case SIM_CURL_STENCIL_BACKWARD_2: {
            double fm1 = curl_sample_real(field, src, index, axis, -1, cfg->boundary, f0);
            double fm2 = curl_sample_real(field, src, index, axis, -2, cfg->boundary, f0);
            return (3.0 * f0 - 4.0 * fm1 + fm2) / (2.0 * dx);
        }
        case SIM_CURL_STENCIL_CENTRAL_4: {
            double fp1 = curl_sample_real(field, src, index, axis, 1, cfg->boundary, f0);
            double fm1 = curl_sample_real(field, src, index, axis, -1, cfg->boundary, f0);
            double fp2 = curl_sample_real(field, src, index, axis, 2, cfg->boundary, f0);
            double fm2 = curl_sample_real(field, src, index, axis, -2, cfg->boundary, f0);
            return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * dx);
        }
        case SIM_CURL_STENCIL_CENTRAL_2:
        default: {
            double fp1 = curl_sample_real(field, src, index, axis, 1, cfg->boundary, f0);
            double fm1 = curl_sample_real(field, src, index, axis, -1, cfg->boundary, f0);
            return (fp1 - fm1) / (2.0 * dx);
        }
    }
}

static void curl_axis_complex(const SimField*              field,
                              const SimComplexDouble*      src,
                              size_t                       index,
                              size_t                       axis,
                              const SimCurlOperatorConfig* cfg,
                              double                       dx,
                              double*                      out_re,
                              double*                      out_im) {
    double f0_re = src[index].re;
    double f0_im = src[index].im;

    double p1_re = 0.0, p1_im = 0.0;
    double p2_re = 0.0, p2_im = 0.0;
    double m1_re = 0.0, m1_im = 0.0;
    double m2_re = 0.0, m2_im = 0.0;

    switch (cfg->stencil) {
        case SIM_CURL_STENCIL_FORWARD_1:
            curl_sample_complex(
                field, src, index, axis, 1, cfg->boundary, f0_re, f0_im, &p1_re, &p1_im);
            *out_re = (p1_re - f0_re) / dx;
            *out_im = (p1_im - f0_im) / dx;
            return;
        case SIM_CURL_STENCIL_BACKWARD_1:
            curl_sample_complex(
                field, src, index, axis, -1, cfg->boundary, f0_re, f0_im, &m1_re, &m1_im);
            *out_re = (f0_re - m1_re) / dx;
            *out_im = (f0_im - m1_im) / dx;
            return;
        case SIM_CURL_STENCIL_FORWARD_2:
            curl_sample_complex(
                field, src, index, axis, 1, cfg->boundary, f0_re, f0_im, &p1_re, &p1_im);
            curl_sample_complex(
                field, src, index, axis, 2, cfg->boundary, f0_re, f0_im, &p2_re, &p2_im);
            *out_re = (-3.0 * f0_re + 4.0 * p1_re - p2_re) / (2.0 * dx);
            *out_im = (-3.0 * f0_im + 4.0 * p1_im - p2_im) / (2.0 * dx);
            return;
        case SIM_CURL_STENCIL_BACKWARD_2:
            curl_sample_complex(
                field, src, index, axis, -1, cfg->boundary, f0_re, f0_im, &m1_re, &m1_im);
            curl_sample_complex(
                field, src, index, axis, -2, cfg->boundary, f0_re, f0_im, &m2_re, &m2_im);
            *out_re = (3.0 * f0_re - 4.0 * m1_re + m2_re) / (2.0 * dx);
            *out_im = (3.0 * f0_im - 4.0 * m1_im + m2_im) / (2.0 * dx);
            return;
        case SIM_CURL_STENCIL_CENTRAL_4:
            curl_sample_complex(
                field, src, index, axis, 1, cfg->boundary, f0_re, f0_im, &p1_re, &p1_im);
            curl_sample_complex(
                field, src, index, axis, -1, cfg->boundary, f0_re, f0_im, &m1_re, &m1_im);
            curl_sample_complex(
                field, src, index, axis, 2, cfg->boundary, f0_re, f0_im, &p2_re, &p2_im);
            curl_sample_complex(
                field, src, index, axis, -2, cfg->boundary, f0_re, f0_im, &m2_re, &m2_im);
            *out_re = (-p2_re + 8.0 * p1_re - 8.0 * m1_re + m2_re) / (12.0 * dx);
            *out_im = (-p2_im + 8.0 * p1_im - 8.0 * m1_im + m2_im) / (12.0 * dx);
            return;
        case SIM_CURL_STENCIL_CENTRAL_2:
        default:
            curl_sample_complex(
                field, src, index, axis, 1, cfg->boundary, f0_re, f0_im, &p1_re, &p1_im);
            curl_sample_complex(
                field, src, index, axis, -1, cfg->boundary, f0_re, f0_im, &m1_re, &m1_im);
            *out_re = (p1_re - m1_re) / (2.0 * dx);
            *out_im = (p1_im - m1_im) / (2.0 * dx);
            return;
    }
}

static SimResult
curl_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimCurlOperatorState* state = (SimCurlOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* input_x = sim_context_field(context, state->config.input_field_x);
    SimField* input_y = sim_context_field(context, state->config.input_field_y);
    SimField* output  = sim_context_field(context, state->config.output_field);
    if (!curl_validate_fields(input_x, input_y, output)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t axis_x = 0U;
    size_t axis_y = 0U;
    if (!curl_resolve_axes(input_x, &state->config, &axis_x, &axis_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&input_x->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    const double dx    = state->config.spacing_x;
    const double dy    = state->config.spacing_y;
    const double scale = state->config.scale_by_dt ? fmax(dt, 0.0) : 1.0;

    if (sim_field_is_complex(input_x)) {
        const SimComplexDouble* src_x = sim_field_complex_data_const(input_x);
        const SimComplexDouble* src_y = sim_field_complex_data_const(input_y);
        SimComplexDouble*       dst   = sim_field_complex_data(output);
        if (!src_x || !src_y || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double dvy_dx_re = 0.0, dvy_dx_im = 0.0;
            double dvx_dy_re = 0.0, dvx_dy_im = 0.0;
            double curl_re = 0.0;
            double curl_im = 0.0;

            curl_axis_complex(
                input_y, src_y, i, axis_x, &state->config, dx, &dvy_dx_re, &dvy_dx_im);
            curl_axis_complex(
                input_x, src_x, i, axis_y, &state->config, dy, &dvx_dy_re, &dvx_dy_im);

            curl_re = dvy_dx_re - dvx_dy_re;
            curl_im = dvy_dx_im - dvx_dy_im;

            if (state->config.accumulate) {
                dst[i].re += scale * curl_re;
                dst[i].im += scale * curl_im;
            } else {
                dst[i].re = scale * curl_re;
                dst[i].im = scale * curl_im;
            }
        }
    } else {
        const double* src_x = sim_field_real_data_const(input_x);
        const double* src_y = sim_field_real_data_const(input_y);
        double*       dst   = sim_field_real_data(output);
        if (!src_x || !src_y || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double dvy_dx = curl_axis_real(input_y, src_y, i, axis_x, &state->config, dx);
            double dvx_dy = curl_axis_real(input_x, src_x, i, axis_y, &state->config, dy);
            double value  = dvy_dx - dvx_dy;

            if (state->config.accumulate) {
                dst[i] += scale * value;
            } else {
                dst[i] = scale * value;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult curl_step(void*               state_ptr,
                           struct SimContext*  context,
                           struct SimOperator* self,
                           size_t              substep_index,
                           double              dt_sub,
                           void*               scratch,
                           size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return curl_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_curl_operator(struct SimContext*           context,
                                const SimCurlOperatorConfig* config,
                                size_t*                      out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimCurlOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.axis_x      = SIM_CURL_AXIS_AUTO;
        local.axis_y      = SIM_CURL_AXIS_AUTO;
        local.spacing_x   = 1.0;
        local.spacing_y   = 1.0;
        local.stencil     = SIM_CURL_STENCIL_CENTRAL_2;
        local.boundary    = SIM_IR_BOUNDARY_PERIODIC;
        local.accumulate  = false;
        local.scale_by_dt = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "curl", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    curl_normalize_config(&local);

    SimField* input_x = sim_context_field(context, local.input_field_x);
    SimField* input_y = sim_context_field(context, local.input_field_y);
    SimField* output  = sim_context_field(context, local.output_field);
    if (!curl_validate_fields(input_x, input_y, output)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (local.output_field == local.input_field_x || local.output_field == local.input_field_y) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!curl_resolve_axes(input_x, &local, &(size_t){ 0 }, &(size_t){ 0 })) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimCurlOperatorState* state = (SimCurlOperatorState*) calloc(1U, sizeof(*state));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    curl_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "curl");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_ADVECTION;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "curl";
    sim_operator_info_set_schema_identity(&info, "curl");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    info.representation.boundary                        = local.boundary;
    info.representation.spacing_hint_rank               = 2U;
    info.representation.spacing_hint[0]                 = local.spacing_x;
    info.representation.spacing_hint[1]                 = local.spacing_y;
    info.approximation.spatial_order                    = 1.0;

    switch (local.stencil) {
        case SIM_CURL_STENCIL_FORWARD_2:
        case SIM_CURL_STENCIL_BACKWARD_2:
        case SIM_CURL_STENCIL_CENTRAL_4:
            info.approximation.stencil_order = 4.0;
            break;
        case SIM_CURL_STENCIL_FORWARD_1:
        case SIM_CURL_STENCIL_BACKWARD_1:
            info.approximation.stencil_order = 1.0;
            break;
        case SIM_CURL_STENCIL_CENTRAL_2:
        default:
            info.approximation.stencil_order = 2.0;
            break;
    }

    bool needs_complex = sim_field_is_complex(input_x) || sim_field_is_complex(input_y) ||
                         sim_field_is_complex(output);
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimSplitPort ports[3] = {
        { .context_field_index = state->config.input_field_x, .require_complex = needs_complex },
        { .context_field_index = state->config.input_field_y, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_READ },
                                   { .port = 2, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = curl_step,
                                .accesses          = accesses,
                                .access_count      = 3U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 3U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = curl_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_curl_config(struct SimContext*     context,
                          size_t                 operator_index,
                          SimCurlOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimCurlOperatorState* state = (SimCurlOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_curl_update(struct SimContext*           context,
                          size_t                       operator_index,
                          const SimCurlOperatorConfig* config) {
    if (!context || !config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimCurlOperatorState* state = (SimCurlOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimCurlOperatorConfig local = *config;
    local.scale_by_dt           = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "curl"), true, local.scale_by_dt);
    curl_normalize_config(&local);

    SimField* input_x = sim_context_field(context, local.input_field_x);
    SimField* input_y = sim_context_field(context, local.input_field_y);
    SimField* output  = sim_context_field(context, local.output_field);
    if (!curl_validate_fields(input_x, input_y, output)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (local.output_field == local.input_field_x || local.output_field == local.input_field_y) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!curl_resolve_axes(input_x, &local, &(size_t){ 0 }, &(size_t){ 0 })) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config = local;
    curl_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
