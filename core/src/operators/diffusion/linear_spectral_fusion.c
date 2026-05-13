#include "oakfield/operators/diffusion/linear_spectral_fusion.h"
#include "oakfield/operators/common/fft_plan.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct LinearSpectralState {
    LinearSpectralFusionOperatorConfig config;
    size_t                             capacity;

    FFTPlan         plan;
    FFTPlan2D       plan2d;
    size_t          plan_rank;
    size_t          plan_shape[2];
    double complex* time_buffer;
    double complex* freq_buffer;

    double* lambda; /* dissipation rate per bin (real) */
    double* omega;  /* dispersion phase rate per bin (real) */
    bool    lambda_dirty;
    bool    omega_dirty;

    struct {
        SimIRNodeId dissipation_scale;
        SimIRNodeId alpha;
        SimIRNodeId dissipation_base_kx;
        SimIRNodeId dissipation_base_ky;
        SimIRNodeId dispersion_coefficient;
        SimIRNodeId dispersion_order;
        SimIRNodeId dispersion_reference_k;
        SimIRNodeId dispersion_base_kx;
        SimIRNodeId dispersion_base_ky;
        SimIRNodeId phase_rate;
        size_t      rank;
        size_t      rows;
        size_t      cols;
        bool        valid;
    } kernel_nodes;

    char symbolic[192];
} LinearSpectralState;

static void linear_spectral_release(void* state_ptr) {
    LinearSpectralState* state = (LinearSpectralState*) state_ptr;
    if (!state) {
        return;
    }

    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->time_buffer);
    free(state->freq_buffer);
    free(state->lambda);
    free(state->omega);
    free(state);
}

static const char* linear_spectral_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const LinearSpectralState* state = (const LinearSpectralState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static double resolve_spacing(double spacing) {
    return (spacing > 0.0 && isfinite(spacing)) ? spacing : 1.0;
}

static void linear_spectral_normalize_config(const LinearSpectralFusionOperatorConfig* source,
                                             const LinearSpectralFusionOperatorConfig* fallback,
                                             LinearSpectralFusionOperatorConfig*       out) {
    LinearSpectralFusionOperatorConfig local = { 0 };
    if (fallback != NULL) {
        local = *fallback;
    }
    if (source != NULL) {
        local = *source;
    }

    if (!isfinite(local.viscosity))
        local.viscosity = (fallback != NULL) ? fallback->viscosity : 0.0;
    if (!isfinite(local.alpha) || local.alpha < 0.0)
        local.alpha = (fallback != NULL) ? fallback->alpha : 2.0;
    if (!isfinite(local.dissipation_spacing) || local.dissipation_spacing <= 0.0)
        local.dissipation_spacing = (fallback != NULL) ? fallback->dissipation_spacing : 1.0;

    if (!isfinite(local.dispersion_coefficient))
        local.dispersion_coefficient = (fallback != NULL) ? fallback->dispersion_coefficient : 0.0;
    if (!isfinite(local.dispersion_order) || local.dispersion_order < 0.0)
        local.dispersion_order = (fallback != NULL) ? fallback->dispersion_order : 1.0;
    if (!isfinite(local.dispersion_spacing) || local.dispersion_spacing <= 0.0)
        local.dispersion_spacing = (fallback != NULL) ? fallback->dispersion_spacing : 1.0;
    if (!isfinite(local.dispersion_reference_k) || local.dispersion_reference_k < 0.0)
        local.dispersion_reference_k = (fallback != NULL) ? fallback->dispersion_reference_k : 0.0;

    if (!isfinite(local.phase_rate))
        local.phase_rate = (fallback != NULL) ? fallback->phase_rate : 0.0;

    if (out != NULL) {
        *out = local;
    }
}

static SimResult linear_spectral_describe_field(const SimField*         field,
                                                SimFieldRepresentation* out_repr,
                                                bool*                   out_needs_complex,
                                                bool*                   out_project_real) {
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFieldRepresentation repr          = sim_field_representation(field);
    const bool             needs_complex = sim_field_is_complex(field);
    const bool             project_real =
        !needs_complex || sim_field_representation_has_spectral_real_constraint(repr);

    if (needs_complex) {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else if (field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (out_repr != NULL) {
        *out_repr = repr;
    }
    if (out_needs_complex != NULL) {
        *out_needs_complex = needs_complex;
    }
    if (out_project_real != NULL) {
        *out_project_real = project_real;
    }

    return SIM_RESULT_OK;
}

static bool linear_spectral_has_phase_term(const LinearSpectralFusionOperatorConfig* cfg) {
    return cfg != NULL && (cfg->dispersion_coefficient != 0.0 || cfg->phase_rate != 0.0);
}

static SimFieldValueKind
linear_spectral_output_value_kind(SimFieldRepresentation                    repr,
                                  bool                                      needs_complex,
                                  const LinearSpectralFusionOperatorConfig* cfg) {
    if (repr.value_kind == SIM_FIELD_VALUE_UNKNOWN) {
        return needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    }
    if (sim_field_representation_has_imag_zero_constraint(repr) &&
        linear_spectral_has_phase_term(cfg)) {
        return SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }
    return repr.value_kind;
}

static void linear_spectral_fill_info(SimOperatorInfo*                          info,
                                      SimFieldRepresentation                    repr,
                                      bool                                      needs_complex,
                                      bool                                      project_real,
                                      const LinearSpectralFusionOperatorConfig* cfg) {
    if (info == NULL || cfg == NULL) {
        return;
    }

    const bool              has_phase_term = linear_spectral_has_phase_term(cfg);
    const bool              preserves_real = project_real || !has_phase_term;
    const SimFieldValueKind value_kind =
        linear_spectral_output_value_kind(repr, needs_complex, cfg);

    *info                   = sim_operator_info_defaults();
    info->category          = SIM_OPERATOR_CATEGORY_DIFFUSION;
    info->warp_level        = SIM_WARP_LEVEL_NONE;
    info->is_noise          = false;
    info->is_spectral       = true;
    info->is_local          = false;
    info->is_nonlocal       = true;
    info->is_linear         = true;
    info->is_warp           = false;
    info->is_differentiable = true;
    info->preserves_real    = preserves_real;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "linear_spectral_fusion";
    sim_operator_info_set_schema_identity(info, "linear_spectral_fusion");
    info->algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info->representation.domain                          = SIM_FIELD_DOMAIN_SPECTRAL;
    info->representation.value_kind                      = value_kind;
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = preserves_real;
}

static double linear_spectral_axis_index(size_t index, size_t count) {
    return (index <= count / 2U) ? (double) index : -((double) (count - index));
}

static void linear_spectral_refresh_symbolic(LinearSpectralState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const LinearSpectralFusionOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "lin_spec diff=%.3g alpha=%.3g disp=%.3g order=%.3g k0=%.3g phase_rate=%.3g",
                    cfg->viscosity,
                    cfg->alpha,
                    cfg->dispersion_coefficient,
                    cfg->dispersion_order,
                    cfg->dispersion_reference_k,
                    cfg->phase_rate);
#else
    (void) state;
#endif
}

static SimResult linear_spectral_ensure_capacity(LinearSpectralState*  state,
                                                 const SimFieldLayout* layout) {
    if (!state)
        return SIM_RESULT_INVALID_ARGUMENT;

    if (!layout || layout->shape == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->rank == 0U || layout->rank > 2U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    size_t count = sim_field_element_count(layout);
    size_t rank  = layout->rank;
    size_t rows  = layout->shape[0];
    size_t cols  = (rank == 2U) ? layout->shape[1] : 0U;

    if (count == 0U) {
        fft_plan_destroy(&state->plan);
        fft_plan2d_destroy(&state->plan2d);
        free(state->time_buffer);
        free(state->freq_buffer);
        free(state->lambda);
        free(state->omega);
        state->time_buffer   = NULL;
        state->freq_buffer   = NULL;
        state->lambda        = NULL;
        state->omega         = NULL;
        state->capacity      = 0U;
        state->plan_rank     = 0U;
        state->plan_shape[0] = 0U;
        state->plan_shape[1] = 0U;
        state->lambda_dirty  = true;
        state->omega_dirty   = true;
        return SIM_RESULT_OK;
    }

    const bool shape_match =
        (state->plan_rank == rank) &&
        (rank == 1U ? state->plan_shape[0] == rows
                    : (state->plan_shape[0] == rows && state->plan_shape[1] == cols));

    if (state->capacity == count && shape_match) {
        return SIM_RESULT_OK;
    }

    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->time_buffer);
    free(state->freq_buffer);
    free(state->lambda);
    free(state->omega);
    state->time_buffer = NULL;
    state->freq_buffer = NULL;
    state->lambda      = NULL;
    state->omega       = NULL;
    state->capacity    = 0U;

    if (count == 0U) {
        state->lambda_dirty = true;
        state->omega_dirty  = true;
        return SIM_RESULT_OK;
    }

    SimResult plan_rc = SIM_RESULT_OK;
    if (rank == 1U) {
        plan_rc = fft_plan_init(&state->plan, count);
    } else {
        plan_rc = fft_plan2d_init(&state->plan2d, rows, cols, cols, 1U);
    }
    if (plan_rc != SIM_RESULT_OK) {
        return plan_rc;
    }

    state->time_buffer = (double complex*) calloc(count, sizeof(double complex));
    state->freq_buffer = (double complex*) calloc(count, sizeof(double complex));
    state->lambda      = (double*) calloc(count, sizeof(double));
    state->omega       = (double*) calloc(count, sizeof(double));
    if (!state->time_buffer || !state->freq_buffer || !state->lambda || !state->omega) {
        free(state->time_buffer);
        free(state->freq_buffer);
        free(state->lambda);
        free(state->omega);
        state->time_buffer = NULL;
        state->freq_buffer = NULL;
        state->lambda      = NULL;
        state->omega       = NULL;
        fft_plan_destroy(&state->plan);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->capacity      = count;
    state->plan_rank     = rank;
    state->plan_shape[0] = rows;
    state->plan_shape[1] = cols;
    state->lambda_dirty  = true;
    state->omega_dirty   = true;
    return SIM_RESULT_OK;
}

static void linear_spectral_update_lambda(LinearSpectralState*  state,
                                          const SimFieldLayout* layout) {
    if (!state || !layout || layout->shape == NULL) {
        return;
    }
    if (layout->rank == 0U || layout->rank > 2U) {
        return;
    }
    if (state->lambda == NULL) {
        return;
    }

    const double dissipation = state->config.viscosity;
    const double alpha       = state->config.alpha;
    const double spacing_x   = resolve_spacing(state->config.dissipation_spacing);
    const double spacing_y   = spacing_x;

    size_t rows = layout->shape[0];
    size_t cols = (layout->rank == 2U) ? layout->shape[1] : 0U;

    if (layout->rank == 1U) {
        double length = spacing_x * (double) rows;
        double base_k = (rows > 0U && length > 0.0) ? (2.0 * M_PI / length) : 0.0;
        for (size_t i = 0U; i < rows; ++i) {
            double freq_index = linear_spectral_axis_index(i, rows);
            double k_abs      = fabs(freq_index * base_k);
            double value      = 0.0;

            if (k_abs > 0.0 && alpha >= 0.0 && isfinite(alpha)) {
                value = -pow(k_abs, alpha);
            }

            state->lambda[i] = dissipation * value;
        }
    } else {
        double length_x = spacing_x * (double) cols;
        double length_y = spacing_y * (double) rows;
        double base_kx  = (cols > 0U && length_x > 0.0) ? (2.0 * M_PI / length_x) : 0.0;
        double base_ky  = (rows > 0U && length_y > 0.0) ? (2.0 * M_PI / length_y) : 0.0;
        for (size_t y = 0U; y < rows; ++y) {
            double ky_index = linear_spectral_axis_index(y, rows);
            double ky       = ky_index * base_ky;
            size_t row_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                double kx_index = linear_spectral_axis_index(x, cols);
                double kx       = kx_index * base_kx;
                double k_abs    = sqrt(kx * kx + ky * ky);
                double value    = 0.0;
                if (k_abs > 0.0 && alpha >= 0.0 && isfinite(alpha)) {
                    value = -pow(k_abs, alpha);
                }
                state->lambda[row_base + x] = dissipation * value;
            }
        }
    }

    state->lambda_dirty = false;
}

static void linear_spectral_update_omega(LinearSpectralState* state, const SimFieldLayout* layout) {
    if (!state || !layout || layout->shape == NULL) {
        return;
    }
    if (layout->rank == 0U || layout->rank > 2U) {
        return;
    }

    const double coefficient = state->config.dispersion_coefficient;
    const double order       = state->config.dispersion_order;
    const double spacing     = resolve_spacing(state->config.dispersion_spacing);
    const double reference_k = (isfinite(state->config.dispersion_reference_k) &&
                                state->config.dispersion_reference_k >= 0.0)
                                   ? state->config.dispersion_reference_k
                                   : 0.0;

    double spacing_x = spacing;
    double spacing_y = spacing;

    size_t rows = layout->shape[0];
    size_t cols = (layout->rank == 2U) ? layout->shape[1] : 0U;

    if (layout->rank == 1U) {
        double length = spacing_x * (double) rows;
        double base_k = (rows > 0U && length > 0.0) ? (2.0 * M_PI / length) : 0.0;

        for (size_t i = 0U; i < rows; ++i) {
            double freq_index = linear_spectral_axis_index(i, rows);
            double k_abs      = fabs(freq_index * base_k);
            double k_shift    = fabs(k_abs - reference_k);
            double omega      = coefficient * pow(k_shift, order);
            state->omega[i]   = isfinite(omega) ? omega : 0.0;
        }
    } else {
        double length_x = spacing_x * (double) cols;
        double length_y = spacing_y * (double) rows;
        double base_kx  = (cols > 0U && length_x > 0.0) ? (2.0 * M_PI / length_x) : 0.0;
        double base_ky  = (rows > 0U && length_y > 0.0) ? (2.0 * M_PI / length_y) : 0.0;

        for (size_t y = 0U; y < rows; ++y) {
            double ky_index = linear_spectral_axis_index(y, rows);
            double ky       = ky_index * base_ky;
            size_t row_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                double kx_index            = linear_spectral_axis_index(x, cols);
                double kx                  = kx_index * base_kx;
                double k_abs               = sqrt(kx * kx + ky * ky);
                double k_shift             = fabs(k_abs - reference_k);
                double omega               = coefficient * pow(k_shift, order);
                state->omega[row_base + x] = isfinite(omega) ? omega : 0.0;
            }
        }
    }

    state->omega_dirty = false;
}

static SimIRNodeId linear_spectral_const(SimIRBuilder* builder, double value) {
    return sim_ir_builder_constant_typed(builder, value, sim_ir_type_scalar());
}

static SimIRNodeId linear_spectral_signed_axis_index(SimIRBuilder* builder,
                                                     size_t        field_id,
                                                     size_t        axis,
                                                     size_t        extent) {
    if (builder == NULL || extent == 0U) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId coord       = sim_ir_builder_coord(builder, field_id, axis);
    SimIRNodeId extent_node = linear_spectral_const(builder, (double) extent);
    SimIRNodeId half_extent = linear_spectral_const(builder, 0.5 * (double) extent);
    if (coord == SIM_IR_INVALID_NODE || extent_node == SIM_IR_INVALID_NODE ||
        half_extent == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId shifted    = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, coord, half_extent);
    SimIRNodeId ratio      = sim_ir_builder_binary(builder, SIM_IR_NODE_DIV, shifted, extent_node);
    SimIRNodeId wraps      = sim_ir_builder_floor(builder, ratio);
    SimIRNodeId period     = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, extent_node, wraps);
    SimIRNodeId signed_idx = sim_ir_builder_binary(builder, SIM_IR_NODE_SUB, coord, period);
    return signed_idx;
}

static SimIRNodeId linear_spectral_k_abs_expr(SimIRBuilder*         builder,
                                              const SimFieldLayout* layout,
                                              double                spacing,
                                              SimIRNodeId*          out_base_kx,
                                              SimIRNodeId*          out_base_ky) {
    if (builder == NULL || layout == NULL || layout->shape == NULL || layout->rank == 0U ||
        layout->rank > 2U) {
        return SIM_IR_INVALID_NODE;
    }
    if (out_base_kx != NULL) {
        *out_base_kx = SIM_IR_INVALID_NODE;
    }
    if (out_base_ky != NULL) {
        *out_base_ky = SIM_IR_INVALID_NODE;
    }

    const double safe_spacing = resolve_spacing(spacing);
    const size_t rows         = layout->shape[0];
    const size_t cols         = (layout->rank == 2U) ? layout->shape[1] : 0U;
    const size_t field_id     = 0U;

    if (layout->rank == 1U) {
        SimIRNodeId idx = linear_spectral_signed_axis_index(builder, field_id, 0U, rows);
        if (idx == SIM_IR_INVALID_NODE) {
            return SIM_IR_INVALID_NODE;
        }
        const double base_k_value =
            (rows > 0U && safe_spacing > 0.0) ? (2.0 * M_PI / (safe_spacing * (double) rows)) : 0.0;
        SimIRNodeId base_k = linear_spectral_const(builder, base_k_value);
        if (out_base_kx != NULL) {
            *out_base_kx = base_k;
        }
        SimIRNodeId k = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, idx, base_k);
        return sim_ir_builder_call(builder, SIM_IR_CALL_ABS, k);
    }

    SimIRNodeId y_idx = linear_spectral_signed_axis_index(builder, field_id, 0U, rows);
    SimIRNodeId x_idx = linear_spectral_signed_axis_index(builder, field_id, 1U, cols);
    if (y_idx == SIM_IR_INVALID_NODE || x_idx == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    const double base_ky_value =
        (rows > 0U && safe_spacing > 0.0) ? (2.0 * M_PI / (safe_spacing * (double) rows)) : 0.0;
    const double base_kx_value =
        (cols > 0U && safe_spacing > 0.0) ? (2.0 * M_PI / (safe_spacing * (double) cols)) : 0.0;

    SimIRNodeId base_kx = linear_spectral_const(builder, base_kx_value);
    SimIRNodeId base_ky = linear_spectral_const(builder, base_ky_value);
    if (out_base_kx != NULL) {
        *out_base_kx = base_kx;
    }
    if (out_base_ky != NULL) {
        *out_base_ky = base_ky;
    }
    SimIRNodeId kx   = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, x_idx, base_kx);
    SimIRNodeId ky   = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, y_idx, base_ky);
    SimIRNodeId kx2  = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, kx, kx);
    SimIRNodeId ky2  = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, ky, ky);
    SimIRNodeId sum  = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, kx2, ky2);
    SimIRNodeId half = linear_spectral_const(builder, 0.5);
    return sim_ir_builder_pow(builder, sum, half);
}

static SimIRNodeId linear_spectral_build_kernel_expr(SimIRBuilder* builder,
                                                     const LinearSpectralFusionOperatorConfig* cfg,
                                                     const SimFieldLayout* layout,
                                                     LinearSpectralState*  state) {
    if (builder == NULL || cfg == NULL || layout == NULL || state == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    memset(&state->kernel_nodes, 0, sizeof(state->kernel_nodes));
    state->kernel_nodes.dissipation_scale      = SIM_IR_INVALID_NODE;
    state->kernel_nodes.alpha                  = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dissipation_base_kx    = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dissipation_base_ky    = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dispersion_coefficient = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dispersion_order       = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dispersion_reference_k = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dispersion_base_kx     = SIM_IR_INVALID_NODE;
    state->kernel_nodes.dispersion_base_ky     = SIM_IR_INVALID_NODE;
    state->kernel_nodes.phase_rate             = SIM_IR_INVALID_NODE;

    SimIRNodeId field = sim_ir_builder_field_ref_typed(builder, 0U, sim_ir_type_complex());
    SimIRNodeId dt    = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
    if (field == SIM_IR_INVALID_NODE || dt == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId k_abs_diss = linear_spectral_k_abs_expr(builder,
                                                        layout,
                                                        cfg->dissipation_spacing,
                                                        &state->kernel_nodes.dissipation_base_kx,
                                                        &state->kernel_nodes.dissipation_base_ky);
    SimIRNodeId k_abs_disp = linear_spectral_k_abs_expr(builder,
                                                        layout,
                                                        cfg->dispersion_spacing,
                                                        &state->kernel_nodes.dispersion_base_kx,
                                                        &state->kernel_nodes.dispersion_base_ky);
    if (k_abs_diss == SIM_IR_INVALID_NODE || k_abs_disp == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId alpha                     = linear_spectral_const(builder, cfg->alpha);
    SimIRNodeId diss                      = linear_spectral_const(builder, -cfg->viscosity);
    state->kernel_nodes.alpha             = alpha;
    state->kernel_nodes.dissipation_scale = diss;
    SimIRNodeId diss_pow                  = sim_ir_builder_pow(builder, k_abs_diss, alpha);
    SimIRNodeId lambda    = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, diss, diss_pow);
    SimIRNodeId dt_lambda = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, dt, lambda);
    SimIRNodeId factor    = sim_ir_builder_call(builder, SIM_IR_CALL_EXP, dt_lambda);

    SimIRNodeId ref_k = linear_spectral_const(builder, cfg->dispersion_reference_k);
    SimIRNodeId order = linear_spectral_const(builder, cfg->dispersion_order);
    SimIRNodeId beta  = linear_spectral_const(builder, cfg->dispersion_coefficient);
    SimIRNodeId phase = linear_spectral_const(builder, cfg->phase_rate);
    state->kernel_nodes.dispersion_reference_k = ref_k;
    state->kernel_nodes.dispersion_order       = order;
    state->kernel_nodes.dispersion_coefficient = beta;
    state->kernel_nodes.phase_rate             = phase;

    SimIRNodeId k_shift  = sim_ir_builder_binary(builder, SIM_IR_NODE_SUB, k_abs_disp, ref_k);
    k_shift              = sim_ir_builder_call(builder, SIM_IR_CALL_ABS, k_shift);
    SimIRNodeId disp_pow = sim_ir_builder_pow(builder, k_shift, order);
    SimIRNodeId omega    = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, beta, disp_pow);
    SimIRNodeId omega_plus_phase = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, omega, phase);
    SimIRNodeId theta = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, dt, omega_plus_phase);

    SimIRNodeId rotated    = sim_ir_builder_complex_rotate(builder, field, theta);
    SimIRNodeId factor_vec = sim_ir_builder_complex_pack(builder, factor, factor);
    SimIRNodeId root       = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, rotated, factor_vec);

    state->kernel_nodes.rank = layout->rank;
    state->kernel_nodes.rows = layout->shape[0];
    state->kernel_nodes.cols = (layout->rank == 2U) ? layout->shape[1] : 0U;
    state->kernel_nodes.valid =
        (root != SIM_IR_INVALID_NODE) && (state->kernel_nodes.alpha != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.dissipation_scale != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.dissipation_base_kx != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.dispersion_reference_k != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.dispersion_order != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.dispersion_coefficient != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.dispersion_base_kx != SIM_IR_INVALID_NODE) &&
        (state->kernel_nodes.phase_rate != SIM_IR_INVALID_NODE);
    if (state->kernel_nodes.rank == 2U) {
        state->kernel_nodes.valid =
            state->kernel_nodes.valid &&
            (state->kernel_nodes.dissipation_base_ky != SIM_IR_INVALID_NODE) &&
            (state->kernel_nodes.dispersion_base_ky != SIM_IR_INVALID_NODE);
    }

    return root;
}

static bool
linear_spectral_set_builder_scalar(SimIRBuilder* builder, SimIRNodeId node_id, double value) {
    if (builder == NULL || node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return false;
    }
    SimIRNode* node = &builder->nodes[node_id];
    if (node->type != SIM_IR_NODE_CONSTANT ||
        node->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX) {
        return false;
    }
    node->data.constant.scalar = value;
    return true;
}

static double linear_spectral_base_k(size_t extent, double spacing) {
    if (extent == 0U || !isfinite(spacing) || spacing <= 0.0) {
        return 0.0;
    }
    return 2.0 * M_PI / (spacing * (double) extent);
}

static void linear_spectral_refresh_kernel_constants(LinearSpectralState* state,
                                                     SimIRBuilder*        builder) {
    if (state == NULL || builder == NULL || !state->kernel_nodes.valid) {
        return;
    }

    const LinearSpectralFusionOperatorConfig* cfg  = &state->config;
    const size_t                              rows = state->kernel_nodes.rows;
    const size_t                              cols = state->kernel_nodes.cols;

    const double diss_spacing = resolve_spacing(cfg->dissipation_spacing);
    const double disp_spacing = resolve_spacing(cfg->dispersion_spacing);

    (void) linear_spectral_set_builder_scalar(
        builder, state->kernel_nodes.dissipation_scale, -cfg->viscosity);
    (void) linear_spectral_set_builder_scalar(builder, state->kernel_nodes.alpha, cfg->alpha);
    (void) linear_spectral_set_builder_scalar(
        builder,
        state->kernel_nodes.dissipation_base_kx,
        linear_spectral_base_k((state->kernel_nodes.rank == 2U) ? cols : rows, diss_spacing));
    if (state->kernel_nodes.rank == 2U) {
        (void) linear_spectral_set_builder_scalar(builder,
                                                  state->kernel_nodes.dissipation_base_ky,
                                                  linear_spectral_base_k(rows, diss_spacing));
    }

    (void) linear_spectral_set_builder_scalar(
        builder, state->kernel_nodes.dispersion_coefficient, cfg->dispersion_coefficient);
    (void) linear_spectral_set_builder_scalar(
        builder, state->kernel_nodes.dispersion_order, cfg->dispersion_order);
    (void) linear_spectral_set_builder_scalar(
        builder, state->kernel_nodes.dispersion_reference_k, cfg->dispersion_reference_k);
    (void) linear_spectral_set_builder_scalar(
        builder,
        state->kernel_nodes.dispersion_base_kx,
        linear_spectral_base_k((state->kernel_nodes.rank == 2U) ? cols : rows, disp_spacing));
    if (state->kernel_nodes.rank == 2U) {
        (void) linear_spectral_set_builder_scalar(builder,
                                                  state->kernel_nodes.dispersion_base_ky,
                                                  linear_spectral_base_k(rows, disp_spacing));
    }
    (void) linear_spectral_set_builder_scalar(
        builder, state->kernel_nodes.phase_rate, cfg->phase_rate);
}

static SimResult linear_spectral_apply(void*               state_ptr,
                                       struct SimContext*  context,
                                       struct SimOperator* self,
                                       double              dt) {
    (void) self;
    LinearSpectralState* state = (LinearSpectralState*) state_ptr;
    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult              field_rc =
        linear_spectral_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }

    const SimFieldLayout* layout = &field->layout;
    size_t                count  = sim_field_element_count(layout);
    if (count == 0U)
        return SIM_RESULT_OK;
    if (layout->rank == 0U || layout->rank > 2U)
        return SIM_RESULT_NOT_SUPPORTED;

    size_t rank    = layout->rank;
    size_t rows    = layout->shape[0];
    size_t cols    = (rank == 2U) ? layout->shape[1] : 0U;
    size_t stride0 = layout->strides[0];
    size_t stride1 = (rank == 2U) ? layout->strides[1] : 0U;

    const bool has_dispersion     = (state->config.dispersion_coefficient != 0.0);
    const bool has_phase          = (state->config.phase_rate != 0.0);
    const bool has_dissipation    = (state->config.viscosity != 0.0);
    const bool has_phase_term     = has_dispersion || has_phase;
    const bool imag_zero          = sim_field_representation_has_imag_zero_constraint(repr);
    const bool preserve_imag_zero = imag_zero && !has_phase_term;

    if (!has_dispersion && !has_phase && !has_dissipation) {
        return SIM_RESULT_OK;
    }

    if (imag_zero && has_phase_term) {
        SimFieldRepresentation updated_repr = repr;
        updated_repr.value_kind             = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        SimResult repr_rc                   = sim_field_set_representation(field, updated_repr);
        if (repr_rc != SIM_RESULT_OK) {
            return repr_rc;
        }
        repr = updated_repr;
    }

    SimResult rc = linear_spectral_ensure_capacity(state, layout);
    if (rc != SIM_RESULT_OK)
        return rc;
    if (state->time_buffer == NULL || state->freq_buffer == NULL || state->lambda == NULL ||
        state->omega == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (state->lambda_dirty)
        linear_spectral_update_lambda(state, layout);
    if (state->omega_dirty)
        linear_spectral_update_omega(state, layout);

    const double dt_safe  = isfinite(dt) ? dt : 0.0;
    const double phase_dt = dt_safe * state->config.phase_rate;

    if (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL) {
        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    double factor = exp(dt_safe * state->lambda[i]);
                    double re     = cdata[idx].re;
                    double im     = preserve_imag_zero ? 0.0 : cdata[idx].im;
                    if (project_real && has_phase_term) {
                        double theta  = dt_safe * state->omega[i] + phase_dt;
                        double scale  = factor * cos(theta);
                        cdata[idx].re = scale * re;
                        cdata[idx].im = scale * im;
                    } else if (has_phase_term) {
                        double theta  = dt_safe * state->omega[i] + phase_dt;
                        double s      = sin(theta);
                        double c      = cos(theta);
                        cdata[idx].re = factor * (re * c - im * s);
                        cdata[idx].im = factor * (re * s + im * c);
                    } else {
                        cdata[idx].re = factor * re;
                        cdata[idx].im = preserve_imag_zero ? 0.0 : (factor * im);
                    }
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base    = y * stride0;
                    size_t lambda_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx    = row_base + x * stride1;
                        double factor = exp(dt_safe * state->lambda[lambda_base + x]);
                        double re     = cdata[idx].re;
                        double im     = preserve_imag_zero ? 0.0 : cdata[idx].im;
                        if (project_real && has_phase_term) {
                            double theta  = dt_safe * state->omega[lambda_base + x] + phase_dt;
                            double scale  = factor * cos(theta);
                            cdata[idx].re = scale * re;
                            cdata[idx].im = scale * im;
                        } else if (has_phase_term) {
                            double theta  = dt_safe * state->omega[lambda_base + x] + phase_dt;
                            double s      = sin(theta);
                            double c      = cos(theta);
                            cdata[idx].re = factor * (re * c - im * s);
                            cdata[idx].im = factor * (re * s + im * c);
                        } else {
                            cdata[idx].re = factor * re;
                            cdata[idx].im = preserve_imag_zero ? 0.0 : (factor * im);
                        }
                    }
                }
            }
        } else {
            double* data = sim_field_real_data(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    double factor = exp(dt_safe * state->lambda[i]);
                    if (has_phase_term) {
                        double theta = dt_safe * state->omega[i] + phase_dt;
                        factor *= cos(theta);
                    }
                    data[idx] *= factor;
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base    = y * stride0;
                    size_t lambda_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx    = row_base + x * stride1;
                        double factor = exp(dt_safe * state->lambda[lambda_base + x]);
                        if (has_phase_term) {
                            double theta = dt_safe * state->omega[lambda_base + x] + phase_dt;
                            factor *= cos(theta);
                        }
                        data[idx] *= factor;
                    }
                }
            }
        }
        return SIM_RESULT_OK;
    }

    if (needs_complex) {
        SimComplexDouble* cdata = sim_field_complex_data(field);
        if (cdata == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (rank == 1U) {
            for (size_t i = 0U; i < rows; ++i) {
                size_t idx = i * stride0;
                state->time_buffer[i] =
                    CMPLX(cdata[idx].re, preserve_imag_zero ? 0.0 : cdata[idx].im);
            }
        } else {
            for (size_t y = 0U; y < rows; ++y) {
                size_t row_base = y * stride0;
                size_t out_base = y * cols;
                for (size_t x = 0U; x < cols; ++x) {
                    size_t idx = row_base + x * stride1;
                    state->time_buffer[out_base + x] =
                        CMPLX(cdata[idx].re, preserve_imag_zero ? 0.0 : cdata[idx].im);
                }
            }
        }
    } else {
        double* data = sim_field_real_data(field);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (rank == 1U) {
            for (size_t i = 0U; i < rows; ++i) {
                size_t idx            = i * stride0;
                state->time_buffer[i] = CMPLX(data[idx], 0.0);
            }
        } else {
            for (size_t y = 0U; y < rows; ++y) {
                size_t row_base = y * stride0;
                size_t out_base = y * cols;
                for (size_t x = 0U; x < cols; ++x) {
                    size_t idx                       = row_base + x * stride1;
                    state->time_buffer[out_base + x] = CMPLX(data[idx], 0.0);
                }
            }
        }
    }

    if (rank == 1U) {
        rc = fft_plan_forward(&state->plan, state->time_buffer, state->freq_buffer);
    } else {
        rc = fft_plan2d_forward(&state->plan2d, state->time_buffer, state->freq_buffer);
    }
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    for (size_t i = 0U; i < count; ++i) {
        double factor = exp(dt_safe * state->lambda[i]);
        if (project_real && has_phase_term) {
            double theta = dt_safe * state->omega[i] + phase_dt;
            state->freq_buffer[i] *= factor * cos(theta);
        } else if (has_phase_term) {
            double theta          = dt_safe * state->omega[i] + phase_dt;
            double s              = sin(theta);
            double c              = cos(theta);
            double re             = creal(state->freq_buffer[i]);
            double im             = cimag(state->freq_buffer[i]);
            state->freq_buffer[i] = CMPLX(factor * (re * c - im * s), factor * (re * s + im * c));
        } else {
            state->freq_buffer[i] *= factor;
        }
    }

    if (rank == 1U) {
        rc = fft_plan_inverse(&state->plan, state->freq_buffer, state->time_buffer);
    } else {
        rc = fft_plan2d_inverse(&state->plan2d, state->freq_buffer, state->time_buffer);
    }
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    if (needs_complex) {
        SimComplexDouble* cdata = sim_field_complex_data(field);
        if (cdata == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (rank == 1U) {
            for (size_t i = 0U; i < rows; ++i) {
                size_t idx    = i * stride0;
                cdata[idx].re = creal(state->time_buffer[i]);
                cdata[idx].im = preserve_imag_zero ? 0.0 : cimag(state->time_buffer[i]);
            }
        } else {
            for (size_t y = 0U; y < rows; ++y) {
                size_t row_base = y * stride0;
                size_t in_base  = y * cols;
                for (size_t x = 0U; x < cols; ++x) {
                    size_t idx    = row_base + x * stride1;
                    cdata[idx].re = creal(state->time_buffer[in_base + x]);
                    cdata[idx].im =
                        preserve_imag_zero ? 0.0 : cimag(state->time_buffer[in_base + x]);
                }
            }
        }
    } else {
        double* data = sim_field_real_data(field);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (rank == 1U) {
            for (size_t i = 0U; i < rows; ++i) {
                size_t idx = i * stride0;
                data[idx]  = creal(state->time_buffer[i]);
            }
        } else {
            for (size_t y = 0U; y < rows; ++y) {
                size_t row_base = y * stride0;
                size_t in_base  = y * cols;
                for (size_t x = 0U; x < cols; ++x) {
                    size_t idx = row_base + x * stride1;
                    data[idx]  = creal(state->time_buffer[in_base + x]);
                }
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult linear_spectral_step(void*               state_ptr,
                                      struct SimContext*  context,
                                      struct SimOperator* self,
                                      size_t              substep_index,
                                      double              dt_sub,
                                      void*               scratch,
                                      size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return linear_spectral_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_linear_spectral_fusion_operator(struct SimContext*                        context,
                                                  const LinearSpectralFusionOperatorConfig* config,
                                                  size_t* out_index) {
    if (!context)
        return SIM_RESULT_INVALID_ARGUMENT;

    LinearSpectralFusionOperatorConfig local = { 0 };
    linear_spectral_normalize_config(config, NULL, &local);

    LinearSpectralState* state = (LinearSpectralState*) calloc(1U, sizeof(LinearSpectralState));
    if (!state)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config             = local;
    state->capacity           = 0U;
    state->plan_rank          = 0U;
    state->plan_shape[0]      = 0U;
    state->plan_shape[1]      = 0U;
    state->lambda_dirty       = true;
    state->omega_dirty        = true;
    state->kernel_nodes.valid = false;
    linear_spectral_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "linear_spectral_fusion");

    SimField*              field         = sim_context_field(context, local.field_index);
    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult              field_rc =
        linear_spectral_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        linear_spectral_release(state);
        return field_rc;
    }

    SimOperatorInfo info = sim_operator_info_defaults();
    linear_spectral_fill_info(&info, repr, needs_complex, project_real, &local);

    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;
    SimOperatorConfig op_config         = sim_operator_config_defaults();

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "linear_spectral_fusion")) {
        const bool can_register_complex_kernel =
            needs_complex && !project_real && repr.domain == SIM_FIELD_DOMAIN_SPECTRAL &&
            !sim_field_representation_has_imag_zero_constraint(repr);
        if (field != NULL && can_register_complex_kernel &&
            field->element_size == sizeof(SimComplexDouble) && field->layout.rank > 0U &&
            field->layout.rank <= 2U) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimIRNodeId root =
                    linear_spectral_build_kernel_expr(builder, &local, &field->layout, state);
                if (root != SIM_IR_INVALID_NODE) {
                    SimOperatorKernelBindingDescriptor bindings[1];
                    SimOperatorKernelOutputDescriptor  outputs[1];
                    SimOperatorKernelDescriptor        kernel_desc = { 0 };

                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = root;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_DT + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = linear_spectral_release;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    if (local.field_index < 64U) {
                        kdesc.read_mask |= (1ULL << local.field_index);
                        kdesc.write_mask |= (1ULL << local.field_index);
                    }

                    result = sim_context_register_operator(context, &kdesc, out_index);
                    if (result == SIM_RESULT_OK) {
                        linear_spectral_refresh_kernel_constants(state, builder);
                        registered_kernel = true;
                    } else {
                        state->kernel_nodes.valid = false;
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = linear_spectral_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = linear_spectral_symbolic,
                                .destroy       = linear_spectral_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult rc = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (rc != SIM_RESULT_OK) {
        linear_spectral_release(state);
    }
    return rc;
}

SimResult sim_linear_spectral_fusion_config(struct SimContext*                  context,
                                            size_t                              operator_index,
                                            LinearSpectralFusionOperatorConfig* out_config) {
    if (!context || !out_config)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op)
        return SIM_RESULT_NOT_FOUND;
    LinearSpectralState* state = (LinearSpectralState*) sim_operator_state(op);
    if (!state)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_linear_spectral_fusion_update(struct SimContext* context,
                                            size_t             operator_index,
                                            const LinearSpectralFusionOperatorConfig* config) {
    if (!context || !config)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op)
        return SIM_RESULT_NOT_FOUND;

    LinearSpectralState* state = (LinearSpectralState*) sim_operator_state(op);
    if (!state)
        return SIM_RESULT_INVALID_STATE;

    LinearSpectralFusionOperatorConfig local = { 0 };
    linear_spectral_normalize_config(config, &state->config, &local);
    local.field_index = state->config.field_index;

    const bool lambda_changed = (local.viscosity != state->config.viscosity) ||
                                (local.alpha != state->config.alpha) ||
                                (local.dissipation_spacing != state->config.dissipation_spacing);
    const bool omega_changed =
        (local.dispersion_coefficient != state->config.dispersion_coefficient) ||
        (local.dispersion_order != state->config.dispersion_order) ||
        (local.dispersion_spacing != state->config.dispersion_spacing) ||
        (local.dispersion_reference_k != state->config.dispersion_reference_k);

    state->config = local;
    if (lambda_changed)
        state->lambda_dirty = true;
    if (omega_changed)
        state->omega_dirty = true;
    linear_spectral_refresh_symbolic(state);

    if (op->kernel != NULL && state->kernel_nodes.valid) {
        SimIRBuilder* builder = (SimIRBuilder*) op->kernel->kernel.builder;
        linear_spectral_refresh_kernel_constants(state, builder);
    }

    {
        SimFieldRepresentation repr          = { 0 };
        bool                   needs_complex = false;
        bool                   project_real  = false;
        SimResult              field_rc =
            linear_spectral_describe_field(sim_context_field(context, state->config.field_index),
                                           &repr,
                                           &needs_complex,
                                           &project_real);
        if (field_rc != SIM_RESULT_OK) {
            return field_rc;
        }
        linear_spectral_fill_info(&op->info, repr, needs_complex, project_real, &state->config);
    }

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
