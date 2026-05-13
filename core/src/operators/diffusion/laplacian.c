#include "oakfield/operators/diffusion/laplacian.h"
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

#define LAPLACIAN_EPS 1.0e-12

typedef struct SimLaplacianOperatorState {
    SimLaplacianOperatorConfig config;
    char                       symbolic[128];
} SimLaplacianOperatorState;

static const char* laplacian_stencil_name(SimLaplacianStencil stencil) {
    switch (stencil) {
        case SIM_LAPLACIAN_STENCIL_CROSS_4:
            return "cross4";
        case SIM_LAPLACIAN_STENCIL_ISOTROPIC_9:
            return "isotropic9";
        case SIM_LAPLACIAN_STENCIL_CROSS_2:
        default:
            return "cross2";
    }
}

static void laplacian_refresh_symbolic(SimLaplacianOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "laplacian stencil=%s dx=%.4g dy=%.4g boundary=%s accum=%s scale_by_dt=%s",
                    laplacian_stencil_name(state->config.stencil),
                    state->config.spacing_x,
                    state->config.spacing_y,
                    sim_boundary_policy_name(state->config.boundary),
                    state->config.accumulate ? "true" : "false",
                    state->config.scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* laplacian_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimLaplacianOperatorState* state = (const SimLaplacianOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void laplacian_normalize_config(SimLaplacianOperatorConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->spacing_x) || fabs(config->spacing_x) <= LAPLACIAN_EPS) {
        config->spacing_x = 1.0;
    }
    if (!isfinite(config->spacing_y) || fabs(config->spacing_y) <= LAPLACIAN_EPS) {
        config->spacing_y = config->spacing_x;
    }

    switch (config->stencil) {
        case SIM_LAPLACIAN_STENCIL_CROSS_2:
        case SIM_LAPLACIAN_STENCIL_CROSS_4:
        case SIM_LAPLACIAN_STENCIL_ISOTROPIC_9:
            break;
        default:
            config->stencil = SIM_LAPLACIAN_STENCIL_CROSS_2;
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

static bool laplacian_resolve_axes(const SimField*                   field,
                                   const SimLaplacianOperatorConfig* cfg,
                                   size_t*                           out_axis_x,
                                   size_t*                           out_axis_y) {
    if (!field || !cfg || !out_axis_x || !out_axis_y) {
        return false;
    }

    size_t rank = sim_field_rank(field);
    if (rank < 2U) {
        return false;
    }

    size_t axis_x = (cfg->axis_x == SIM_LAPLACIAN_AXIS_AUTO) ? (rank - 1U) : cfg->axis_x;
    size_t axis_y = (cfg->axis_y == SIM_LAPLACIAN_AXIS_AUTO) ? (rank - 2U) : cfg->axis_y;

    if (axis_x >= rank || axis_y >= rank || axis_x == axis_y) {
        return false;
    }

    *out_axis_x = axis_x;
    *out_axis_y = axis_y;
    return true;
}

static bool laplacian_validate_fields(const SimField* input, const SimField* output) {
    if (!input || !output) {
        return false;
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U || sim_field_element_count(&output->layout) != count) {
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

static double laplacian_sample_real_axis(const SimField*     field,
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

static void laplacian_sample_complex_axis(const SimField*         field,
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

static double laplacian_sample_real_offset(const SimField*     field,
                                           const double*       src,
                                           size_t              index,
                                           const ptrdiff_t*    offsets,
                                           size_t              offset_count,
                                           SimIRBoundaryPolicy boundary,
                                           double              center) {
    SimNdNeighbor neighbor = { 0 };
    SimResult rc = sim_nd_offset_neighbor(field, index, offsets, offset_count, boundary, &neighbor);
    if (rc != SIM_RESULT_OK || !neighbor.valid) {
        return (boundary == SIM_IR_BOUNDARY_DIRICHLET) ? 0.0 : center;
    }
    return src[neighbor.index];
}

static void laplacian_sample_complex_offset(const SimField*         field,
                                            const SimComplexDouble* src,
                                            size_t                  index,
                                            const ptrdiff_t*        offsets,
                                            size_t                  offset_count,
                                            SimIRBoundaryPolicy     boundary,
                                            double                  center_re,
                                            double                  center_im,
                                            double*                 out_re,
                                            double*                 out_im) {
    SimNdNeighbor neighbor = { 0 };
    SimResult rc = sim_nd_offset_neighbor(field, index, offsets, offset_count, boundary, &neighbor);
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

static double laplacian_second_real(const SimField*                   field,
                                    const double*                     src,
                                    size_t                            index,
                                    size_t                            axis,
                                    const SimLaplacianOperatorConfig* cfg,
                                    double                            dx) {
    double f0 = src[index];
    if (cfg->stencil == SIM_LAPLACIAN_STENCIL_CROSS_4) {
        double fp1 = laplacian_sample_real_axis(field, src, index, axis, 1, cfg->boundary, f0);
        double fm1 = laplacian_sample_real_axis(field, src, index, axis, -1, cfg->boundary, f0);
        double fp2 = laplacian_sample_real_axis(field, src, index, axis, 2, cfg->boundary, f0);
        double fm2 = laplacian_sample_real_axis(field, src, index, axis, -2, cfg->boundary, f0);
        return (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * dx * dx);
    }

    double fp1 = laplacian_sample_real_axis(field, src, index, axis, 1, cfg->boundary, f0);
    double fm1 = laplacian_sample_real_axis(field, src, index, axis, -1, cfg->boundary, f0);
    return (fp1 - 2.0 * f0 + fm1) / (dx * dx);
}

static void laplacian_second_complex(const SimField*                   field,
                                     const SimComplexDouble*           src,
                                     size_t                            index,
                                     size_t                            axis,
                                     const SimLaplacianOperatorConfig* cfg,
                                     double                            dx,
                                     double*                           out_re,
                                     double*                           out_im) {
    double f0_re = src[index].re;
    double f0_im = src[index].im;

    double p1_re = 0.0, p1_im = 0.0;
    double p2_re = 0.0, p2_im = 0.0;
    double m1_re = 0.0, m1_im = 0.0;
    double m2_re = 0.0, m2_im = 0.0;

    if (cfg->stencil == SIM_LAPLACIAN_STENCIL_CROSS_4) {
        laplacian_sample_complex_axis(
            field, src, index, axis, 1, cfg->boundary, f0_re, f0_im, &p1_re, &p1_im);
        laplacian_sample_complex_axis(
            field, src, index, axis, -1, cfg->boundary, f0_re, f0_im, &m1_re, &m1_im);
        laplacian_sample_complex_axis(
            field, src, index, axis, 2, cfg->boundary, f0_re, f0_im, &p2_re, &p2_im);
        laplacian_sample_complex_axis(
            field, src, index, axis, -2, cfg->boundary, f0_re, f0_im, &m2_re, &m2_im);
        *out_re = (-p2_re + 16.0 * p1_re - 30.0 * f0_re + 16.0 * m1_re - m2_re) / (12.0 * dx * dx);
        *out_im = (-p2_im + 16.0 * p1_im - 30.0 * f0_im + 16.0 * m1_im - m2_im) / (12.0 * dx * dx);
        return;
    }

    laplacian_sample_complex_axis(
        field, src, index, axis, 1, cfg->boundary, f0_re, f0_im, &p1_re, &p1_im);
    laplacian_sample_complex_axis(
        field, src, index, axis, -1, cfg->boundary, f0_re, f0_im, &m1_re, &m1_im);
    *out_re = (p1_re - 2.0 * f0_re + m1_re) / (dx * dx);
    *out_im = (p1_im - 2.0 * f0_im + m1_im) / (dx * dx);
}

static SimResult
laplacian_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimLaplacianOperatorState* state = (SimLaplacianOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* input  = sim_context_field(context, state->config.input_field);
    SimField* output = sim_context_field(context, state->config.output_field);
    if (!laplacian_validate_fields(input, output)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t axis_x = 0U;
    size_t axis_y = 0U;
    if (!laplacian_resolve_axes(input, &state->config, &axis_x, &axis_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    const double dx    = state->config.spacing_x;
    const double dy    = state->config.spacing_y;
    const double scale = state->config.scale_by_dt ? fmax(dt, 0.0) : 1.0;

    if (state->config.stencil == SIM_LAPLACIAN_STENCIL_ISOTROPIC_9) {
        if (sim_field_rank(input) != 2U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        double delta = fabs(dx - dy);
        if (delta > LAPLACIAN_EPS * fmax(fabs(dx), fabs(dy))) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    if (sim_field_is_complex(input)) {
        const SimComplexDouble* src = sim_field_complex_data_const(input);
        SimComplexDouble*       dst = sim_field_complex_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (state->config.stencil == SIM_LAPLACIAN_STENCIL_ISOTROPIC_9) {
            ptrdiff_t    offsets[2] = { 0, 0 };
            const double inv        = 1.0 / (dx * dx);

            for (size_t i = 0U; i < count; ++i) {
                double center_re = src[i].re;
                double center_im = src[i].im;
                double axial_re = 0.0, axial_im = 0.0;
                double diag_re = 0.0, diag_im = 0.0;

                offsets[0] = 1;
                offsets[1] = 0;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &axial_re,
                                                &axial_im);

                offsets[0] = -1;
                offsets[1] = 0;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                axial_re += diag_re;
                axial_im += diag_im;

                offsets[0] = 0;
                offsets[1] = 1;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                axial_re += diag_re;
                axial_im += diag_im;

                offsets[0] = 0;
                offsets[1] = -1;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                axial_re += diag_re;
                axial_im += diag_im;

                offsets[0] = 1;
                offsets[1] = 1;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                diag_re            = diag_re;
                diag_im            = diag_im;
                double diag_sum_re = diag_re;
                double diag_sum_im = diag_im;

                offsets[0] = 1;
                offsets[1] = -1;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                diag_sum_re += diag_re;
                diag_sum_im += diag_im;

                offsets[0] = -1;
                offsets[1] = 1;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                diag_sum_re += diag_re;
                diag_sum_im += diag_im;

                offsets[0] = -1;
                offsets[1] = -1;
                laplacian_sample_complex_offset(input,
                                                src,
                                                i,
                                                offsets,
                                                2U,
                                                state->config.boundary,
                                                center_re,
                                                center_im,
                                                &diag_re,
                                                &diag_im);
                diag_sum_re += diag_re;
                diag_sum_im += diag_im;

                double lap_re = (4.0 * axial_re + diag_sum_re - 20.0 * center_re) * (inv / 6.0);
                double lap_im = (4.0 * axial_im + diag_sum_im - 20.0 * center_im) * (inv / 6.0);

                if (state->config.accumulate) {
                    dst[i].re += scale * lap_re;
                    dst[i].im += scale * lap_im;
                } else {
                    dst[i].re = scale * lap_re;
                    dst[i].im = scale * lap_im;
                }
            }
            return SIM_RESULT_OK;
        }

        for (size_t i = 0U; i < count; ++i) {
            double d2x_re = 0.0, d2x_im = 0.0;
            double d2y_re = 0.0, d2y_im = 0.0;
            laplacian_second_complex(input, src, i, axis_x, &state->config, dx, &d2x_re, &d2x_im);
            laplacian_second_complex(input, src, i, axis_y, &state->config, dy, &d2y_re, &d2y_im);
            double lap_re = d2x_re + d2y_re;
            double lap_im = d2x_im + d2y_im;

            if (state->config.accumulate) {
                dst[i].re += scale * lap_re;
                dst[i].im += scale * lap_im;
            } else {
                dst[i].re = scale * lap_re;
                dst[i].im = scale * lap_im;
            }
        }
    } else {
        const double* src = sim_field_real_data_const(input);
        double*       dst = sim_field_real_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (state->config.stencil == SIM_LAPLACIAN_STENCIL_ISOTROPIC_9) {
            ptrdiff_t    offsets[2] = { 0, 0 };
            const double inv        = 1.0 / (dx * dx);

            for (size_t i = 0U; i < count; ++i) {
                double center    = src[i];
                double axial_sum = 0.0;
                double diag_sum  = 0.0;

                offsets[0] = 1;
                offsets[1] = 0;
                axial_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = -1;
                offsets[1] = 0;
                axial_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = 0;
                offsets[1] = 1;
                axial_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = 0;
                offsets[1] = -1;
                axial_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = 1;
                offsets[1] = 1;
                diag_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = 1;
                offsets[1] = -1;
                diag_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = -1;
                offsets[1] = 1;
                diag_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                offsets[0] = -1;
                offsets[1] = -1;
                diag_sum += laplacian_sample_real_offset(
                    input, src, i, offsets, 2U, state->config.boundary, center);

                double lap = (4.0 * axial_sum + diag_sum - 20.0 * center) * (inv / 6.0);
                if (state->config.accumulate) {
                    dst[i] += scale * lap;
                } else {
                    dst[i] = scale * lap;
                }
            }
            return SIM_RESULT_OK;
        }

        for (size_t i = 0U; i < count; ++i) {
            double d2x = laplacian_second_real(input, src, i, axis_x, &state->config, dx);
            double d2y = laplacian_second_real(input, src, i, axis_y, &state->config, dy);
            double lap = d2x + d2y;
            if (state->config.accumulate) {
                dst[i] += scale * lap;
            } else {
                dst[i] = scale * lap;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult laplacian_step(void*               state_ptr,
                                struct SimContext*  context,
                                struct SimOperator* self,
                                size_t              substep_index,
                                double              dt_sub,
                                void*               scratch,
                                size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return laplacian_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_laplacian_operator(struct SimContext*                context,
                                     const SimLaplacianOperatorConfig* config,
                                     size_t*                           out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimLaplacianOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.axis_x      = SIM_LAPLACIAN_AXIS_AUTO;
        local.axis_y      = SIM_LAPLACIAN_AXIS_AUTO;
        local.spacing_x   = 1.0;
        local.spacing_y   = 1.0;
        local.stencil     = SIM_LAPLACIAN_STENCIL_CROSS_2;
        local.boundary    = SIM_IR_BOUNDARY_PERIODIC;
        local.accumulate  = false;
        local.scale_by_dt = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "laplacian", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    laplacian_normalize_config(&local);

    SimField* input  = sim_context_field(context, local.input_field);
    SimField* output = sim_context_field(context, local.output_field);
    if (!laplacian_validate_fields(input, output)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (local.input_field == local.output_field) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t axis_x = 0U;
    size_t axis_y = 0U;
    if (!laplacian_resolve_axes(input, &local, &axis_x, &axis_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (local.stencil == SIM_LAPLACIAN_STENCIL_ISOTROPIC_9) {
        if (sim_field_rank(input) != 2U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        double delta = fabs(local.spacing_x - local.spacing_y);
        if (delta > LAPLACIAN_EPS * fmax(fabs(local.spacing_x), fabs(local.spacing_y))) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    SimLaplacianOperatorState* state =
        (SimLaplacianOperatorState*) calloc(1U, sizeof(SimLaplacianOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    laplacian_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "laplacian");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_DIFFUSION;
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
    info.abstract_id       = "laplacian";
    sim_operator_info_set_schema_identity(&info, "laplacian");
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
    info.approximation.spatial_order                    = 2.0;

    if (local.stencil == SIM_LAPLACIAN_STENCIL_CROSS_4) {
        info.approximation.stencil_order = 4.0;
    } else {
        info.approximation.stencil_order = 2.0;
    }

    bool needs_complex = sim_field_is_complex(input) || sim_field_is_complex(output);
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimSplitPort ports[2] = {
        { .context_field_index = state->config.input_field, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = laplacian_step,
                                .accesses          = accesses,
                                .access_count      = 2U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 2U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = laplacian_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_laplacian_config(struct SimContext*          context,
                               size_t                      operator_index,
                               SimLaplacianOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimLaplacianOperatorState* state = (SimLaplacianOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_laplacian_update(struct SimContext*                context,
                               size_t                            operator_index,
                               const SimLaplacianOperatorConfig* config) {
    if (!context || !config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimLaplacianOperatorState* state = (SimLaplacianOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimLaplacianOperatorConfig local = *config;
    local.scale_by_dt                = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "laplacian"), true, local.scale_by_dt);
    laplacian_normalize_config(&local);

    SimField* input  = sim_context_field(context, local.input_field);
    SimField* output = sim_context_field(context, local.output_field);
    if (!laplacian_validate_fields(input, output)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (local.input_field == local.output_field) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t axis_x = 0U;
    size_t axis_y = 0U;
    if (!laplacian_resolve_axes(input, &local, &axis_x, &axis_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (local.stencil == SIM_LAPLACIAN_STENCIL_ISOTROPIC_9) {
        if (sim_field_rank(input) != 2U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        double delta = fabs(local.spacing_x - local.spacing_y);
        if (delta > LAPLACIAN_EPS * fmax(fabs(local.spacing_x), fabs(local.spacing_y))) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    state->config = local;
    laplacian_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
