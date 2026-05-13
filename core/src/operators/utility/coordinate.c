#include "oakfield/operators/utility/coordinate.h"
#include "operators/common/operator_utils.h"
#include "oakfield/plane_chart.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define COORD_SYMBOLIC_CAPACITY 144
#define COORD_PATCH_TILE_ROWS 32U

typedef struct SimCoordinateOperatorState {
    SimCoordinateOperatorConfig config;
    bool                        has_plane_chart;
    bool                        integer_affine_enabled;
    uint64_t                    integer_gain_raw;
    uint64_t                    integer_bias_raw;
    SimPlaneSamplingFrame       plane_frame;
    SimPlaneChartConfig         plane_chart;
    SimPlaneProjectionConfig    plane_projection;
    SimPlaneChartStatus         plane_chart_status;
    char                        symbolic[COORD_SYMBOLIC_CAPACITY];
} SimCoordinateOperatorState;

typedef struct SimCoordinateFloatEvalContext {
    const SimCoordinateOperatorState* state;
    SimField*                         output_field;
    bool                              is_complex;
    bool                              normalize;
    double                            t;
    double                            scale;
    double                            min_value;
    double                            max_value;
    double                            inv_range;
} SimCoordinateFloatEvalContext;

static const char* coordinate_mode_name(SimCoordinateMode mode) {
    switch (mode) {
        case SIM_COORD_MODE_COORD:
            return "coord";
        case SIM_COORD_MODE_INDEX:
        default:
            return "index";
    }
}

static const char* coordinate_normalize_name(SimCoordinateNormalizeMode mode) {
    switch (mode) {
        case SIM_COORD_NORMALIZE_UNIT:
            return "unit";
        case SIM_COORD_NORMALIZE_CENTERED:
            return "centered";
        case SIM_COORD_NORMALIZE_SIGNED:
            return "signed";
        case SIM_COORD_NORMALIZE_NONE:
        default:
            return "none";
    }
}

static void coordinate_normalize_config(SimCoordinateOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (config->mode != SIM_COORD_MODE_INDEX && config->mode != SIM_COORD_MODE_COORD) {
        config->mode = SIM_COORD_MODE_INDEX;
    }

    if (config->normalize < SIM_COORD_NORMALIZE_NONE ||
        config->normalize > SIM_COORD_NORMALIZE_SIGNED) {
        config->normalize = SIM_COORD_NORMALIZE_NONE;
    }

    if (!isfinite(config->gain)) {
        config->gain = 1.0;
    }
    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static void coordinate_refresh_runtime(SimCoordinateOperatorState* state) {
    if (state == NULL) {
        return;
    }

    state->has_plane_chart    = false;
    state->plane_chart_status = SIM_PLANE_CHART_STATUS_UNSUPPORTED;

    if (state->config.mode != SIM_COORD_MODE_COORD ||
        state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        return;
    }

    state->plane_chart_status = sim_plane_chart_from_stimulus_coord(
        &state->config.coord, &state->plane_frame, &state->plane_chart, &state->plane_projection);
    state->has_plane_chart = (state->plane_chart_status == SIM_PLANE_CHART_STATUS_OK);
}

static bool coordinate_supports_integer_output(const SimCoordinateOperatorConfig* config) {
    return config != NULL && config->mode == SIM_COORD_MODE_INDEX &&
           config->normalize == SIM_COORD_NORMALIZE_NONE && !config->scale_by_dt;
}

static bool coordinate_refresh_integer_state(SimCoordinateOperatorState* state,
                                             const SimField*             output_field) {
    SimScalarDomain domain;
    uint64_t        gain_raw = 0U;
    uint64_t        bias_raw = 0U;

    if (state == NULL) {
        return false;
    }

    state->integer_affine_enabled = false;
    state->integer_gain_raw       = 0U;
    state->integer_bias_raw       = 0U;

    if (output_field == NULL) {
        return false;
    }

    domain = sim_field_scalar_domain(output_field);
    if (!sim_operator_domain_is_exact_integer(domain) ||
        !coordinate_supports_integer_output(&state->config)) {
        return false;
    }

    if (state->config.exact_gain_enabled) {
        gain_raw = sim_operator_integer_truncate(state->config.exact_gain_raw, domain);
    } else if (!sim_operator_integer_raw_from_double(state->config.gain, domain, &gain_raw)) {
        return false;
    }

    if (state->config.exact_bias_enabled) {
        bias_raw = sim_operator_integer_truncate(state->config.exact_bias_raw, domain);
    } else if (!sim_operator_integer_raw_from_double(state->config.bias, domain, &bias_raw)) {
        return false;
    }

    state->integer_affine_enabled = true;
    state->integer_gain_raw       = gain_raw;
    state->integer_bias_raw       = bias_raw;
    return true;
}

static double coordinate_value_resolved(const SimCoordinateOperatorState* state,
                                        size_t                            index,
                                        double                            x,
                                        double                            y,
                                        double                            t) {
    if (state == NULL) {
        return 0.0;
    }

    const SimCoordinateOperatorConfig* cfg = &state->config;
    if (cfg->mode == SIM_COORD_MODE_INDEX) {
        return (double) index;
    }

    if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            return x + y;
        }
        return x * y;
    }

    if (state->has_plane_chart) {
        SimPlaneProjectionValue projected = { 0 };
        if (sim_plane_chart_eval_projected(&state->plane_frame,
                                           &state->plane_chart,
                                           &state->plane_projection,
                                           x,
                                           y,
                                           t,
                                           &projected) == SIM_PLANE_CHART_STATUS_OK &&
            isfinite(projected.primary)) {
            return projected.primary;
        }
    }

    return sim_stimulus_coord_u(&cfg->coord, x, y, t);
}

static double coordinate_value(const SimCoordinateOperatorState* state,
                               const SimField*                   field,
                               size_t                            index,
                               double                            t) {
    double x = 0.0;
    double y = 0.0;

    if (state == NULL || field == NULL) {
        return 0.0;
    }
    if (state->config.mode == SIM_COORD_MODE_INDEX) {
        return (double) index;
    }
    if (sim_stimulus_coord_xy(&state->config.coord, field, index, &x, &y) != SIM_RESULT_OK) {
        return 0.0;
    }

    return coordinate_value_resolved(state, index, x, y, t);
}

static bool coordinate_bounds(const SimCoordinateOperatorState* state,
                              const SimField*                   field,
                              size_t                            count,
                              double                            t,
                              double*                           out_min,
                              double*                           out_max) {
    if (state == NULL || field == NULL || out_min == NULL || out_max == NULL || count == 0U) {
        return false;
    }

    double min_value = HUGE_VAL;
    double max_value = -HUGE_VAL;
    for (size_t i = 0U; i < count; ++i) {
        double value = coordinate_value(state, field, i, t);
        if (!isfinite(value)) {
            continue;
        }
        if (value < min_value) {
            min_value = value;
        }
        if (value > max_value) {
            max_value = value;
        }
    }

    if (!isfinite(min_value) || !isfinite(max_value) || max_value <= min_value) {
        return false;
    }

    *out_min = min_value;
    *out_max = max_value;
    return true;
}

static double coordinate_transform_value(const SimCoordinateFloatEvalContext* eval, double value) {
    if (eval == NULL) {
        return value;
    }

    if (eval->normalize) {
        value = (value - eval->min_value) * eval->inv_range;
        if (eval->state->config.normalize == SIM_COORD_NORMALIZE_CENTERED) {
            value -= 0.5;
        } else if (eval->state->config.normalize == SIM_COORD_NORMALIZE_SIGNED) {
            value = value * 2.0 - 1.0;
        }
    }

    return value * eval->state->config.gain + eval->state->config.bias;
}

static SimResult coordinate_prepare_float(const SimCoordinateOperatorState* state,
                                          SimField*                         output_field,
                                          double                            base_time,
                                          double                            dt,
                                          SimCoordinateFloatEvalContext*    out_eval) {
    SimCoordinateFloatEvalContext eval  = { 0 };
    size_t                        count = 0U;

    if (state == NULL || output_field == NULL || out_eval == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count             = sim_field_element_count(&output_field->layout);
    eval.state        = state;
    eval.output_field = output_field;
    eval.is_complex   = sim_field_is_complex(output_field);
    eval.t            = base_time + state->config.time_offset;
    eval.scale        = state->config.scale_by_dt ? fmax(dt, 0.0) : 1.0;
    eval.normalize    = false;
    eval.min_value    = 0.0;
    eval.max_value    = 0.0;
    eval.inv_range    = 0.0;

    if (count == 0U || state->config.normalize == SIM_COORD_NORMALIZE_NONE) {
        *out_eval = eval;
        return SIM_RESULT_OK;
    }

    if (state->config.mode == SIM_COORD_MODE_INDEX) {
        if (count > 1U) {
            eval.normalize = true;
            eval.min_value = 0.0;
            eval.max_value = (double) (count - 1U);
            eval.inv_range = 1.0 / (eval.max_value - eval.min_value);
        }
    } else if (coordinate_bounds(
                   state, output_field, count, eval.t, &eval.min_value, &eval.max_value)) {
        eval.normalize = true;
        eval.inv_range = 1.0 / (eval.max_value - eval.min_value);
    }

    *out_eval = eval;
    return SIM_RESULT_OK;
}

static SimResult coordinate_eval_full_float(const SimCoordinateFloatEvalContext* eval) {
    size_t count = 0U;

    if (eval == NULL || eval->output_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = sim_field_element_count(&eval->output_field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (eval->is_complex) {
        SimComplexDouble* out = sim_field_complex_data(eval->output_field);
        if (out == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            double value = coordinate_transform_value(
                eval, coordinate_value(eval->state, eval->output_field, i, eval->t));
            if (!isfinite(value)) {
                continue;
            }

            if (eval->state->config.accumulate) {
                out[i].re += eval->scale * value;
            } else {
                out[i].re = value;
                out[i].im = 0.0;
            }
        }
    } else {
        double* out = (double*) sim_field_data(eval->output_field);
        if (out == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            double value = coordinate_transform_value(
                eval, coordinate_value(eval->state, eval->output_field, i, eval->t));
            if (!isfinite(value)) {
                continue;
            }

            if (eval->state->config.accumulate) {
                out[i] += eval->scale * value;
            } else {
                out[i] = value;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult coordinate_eval_patch_float(const SimCoordinateFloatEvalContext* eval,
                                             const SimFieldPatch*                 patch) {
    SimFieldPatchView patch_view = { 0 };
    SimResult         rc         = SIM_RESULT_OK;

    if (eval == NULL || !sim_field_patch_is_valid(patch)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rc = sim_field_patch_view_from_field(eval->output_field, patch, false, &patch_view);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    if (eval->is_complex) {
        SimComplexDouble* base = (SimComplexDouble*) patch_view.buffer_view.data;
        if (base == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t row_offset = 0U; row_offset < patch->height; ++row_offset) {
            SimStimulusCoordRow coord_row = { 0 };
            SimComplexDouble*   out_row   = base + row_offset * patch_view.row_stride;
            size_t              index = (patch->y0 + row_offset) * patch->field_width + patch->x0;
            double              x     = 0.0;
            double              y     = 0.0;

            if (eval->state->config.mode == SIM_COORD_MODE_COORD) {
                rc = sim_stimulus_coord_patch_row(
                    &eval->state->config.coord, patch, row_offset, eval->t, &coord_row);
                if (rc != SIM_RESULT_OK) {
                    return rc;
                }
                x = coord_row.x;
                y = coord_row.y;
            }

            for (size_t col = 0U; col < patch->width; ++col) {
                double value = coordinate_transform_value(
                    eval, coordinate_value_resolved(eval->state, index, x, y, eval->t));
                if (isfinite(value)) {
                    if (eval->state->config.accumulate) {
                        out_row[col].re += eval->scale * value;
                    } else {
                        out_row[col].re = value;
                        out_row[col].im = 0.0;
                    }
                }

                index += 1U;
                x += coord_row.x_step;
                y += coord_row.y_step;
            }
        }
    } else {
        double* base = (double*) patch_view.buffer_view.data;
        if (base == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t row_offset = 0U; row_offset < patch->height; ++row_offset) {
            SimStimulusCoordRow coord_row = { 0 };
            double*             out_row   = base + row_offset * patch_view.row_stride;
            size_t              index = (patch->y0 + row_offset) * patch->field_width + patch->x0;
            double              x     = 0.0;
            double              y     = 0.0;

            if (eval->state->config.mode == SIM_COORD_MODE_COORD) {
                rc = sim_stimulus_coord_patch_row(
                    &eval->state->config.coord, patch, row_offset, eval->t, &coord_row);
                if (rc != SIM_RESULT_OK) {
                    return rc;
                }
                x = coord_row.x;
                y = coord_row.y;
            }

            for (size_t col = 0U; col < patch->width; ++col) {
                double value = coordinate_transform_value(
                    eval, coordinate_value_resolved(eval->state, index, x, y, eval->t));
                if (isfinite(value)) {
                    if (eval->state->config.accumulate) {
                        out_row[col] += eval->scale * value;
                    } else {
                        out_row[col] = value;
                    }
                }

                index += 1U;
                x += coord_row.x_step;
                y += coord_row.y_step;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult coordinate_finalize_float(const SimCoordinateFloatEvalContext* eval) {
    (void) eval;
    return SIM_RESULT_OK;
}

static SimResult coordinate_eval_tiled_float(const SimCoordinateFloatEvalContext* eval) {
    SimFieldPatch full_patch = { 0 };
    SimResult     rc         = SIM_RESULT_OK;

    if (eval == NULL || eval->output_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rc = sim_field_patch_full(
        sim_field_width(eval->output_field), sim_field_height(eval->output_field), &full_patch);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    for (size_t tile_y = 0U; tile_y < full_patch.height; tile_y += COORD_PATCH_TILE_ROWS) {
        SimFieldPatch tile        = { 0 };
        size_t        tile_height = full_patch.height - tile_y;
        if (tile_height > COORD_PATCH_TILE_ROWS) {
            tile_height = COORD_PATCH_TILE_ROWS;
        }

        rc = sim_field_patch_from_xywh(full_patch.field_width,
                                       full_patch.field_height,
                                       full_patch.x0,
                                       full_patch.y0 + tile_y,
                                       full_patch.width,
                                       tile_height,
                                       &tile);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        rc = coordinate_eval_patch_float(eval, &tile);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
    }

    return coordinate_finalize_float(eval);
}

static void coordinate_refresh_symbolic(SimCoordinateOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimCoordinateOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "coordinate mode=%s norm=%s gain=%.3g bias=%.3g scale_by_dt=%s",
                    coordinate_mode_name(cfg->mode),
                    coordinate_normalize_name(cfg->normalize),
                    cfg->gain,
                    cfg->bias,
                    cfg->scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* coordinate_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimCoordinateOperatorState* state = (const SimCoordinateOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void coordinate_destroy(void* state_ptr) {
    SimCoordinateOperatorState* state = (SimCoordinateOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state);
}

static SimResult
coordinate_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimCoordinateOperatorState* state = (SimCoordinateOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField*       output_field = sim_context_field(context, state->config.output_field);
    SimScalarDomain output_domain;
    if (output_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    output_domain = sim_field_scalar_domain(output_field);
    if (sim_operator_domain_is_exact_integer(output_domain)) {
        void*  out   = sim_field_data(output_field);
        size_t count = sim_field_element_count(&output_field->layout);
        if (!state->integer_affine_enabled) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
        if (out == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            uint64_t value_raw = sim_operator_integer_truncate((uint64_t) i, output_domain);
            value_raw          = sim_operator_integer_truncate(
                value_raw * state->integer_gain_raw + state->integer_bias_raw, output_domain);
            if (state->config.accumulate) {
                uint64_t existing_raw = 0U;
                if (!sim_operator_integer_read(out, output_domain, i, &existing_raw)) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                value_raw = sim_operator_integer_truncate(existing_raw + value_raw, output_domain);
            }
            if (!sim_operator_integer_write(out, output_domain, i, value_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
        }
        return SIM_RESULT_OK;
    }
    if (!sim_operator_field_domain_is_f64_or_c64(output_field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    size_t count = sim_field_element_count(&output_field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    {
        SimCoordinateFloatEvalContext eval = { 0 };
        SimResult                     rc =
            coordinate_prepare_float(state, output_field, sim_context_time(context), dt, &eval);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        rc = coordinate_eval_tiled_float(&eval);
        if (rc == SIM_RESULT_NOT_SUPPORTED) {
            return coordinate_eval_full_float(&eval);
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult coordinate_step(void*               state_ptr,
                                 struct SimContext*  context,
                                 struct SimOperator* self,
                                 size_t              substep_index,
                                 double              dt_sub,
                                 void*               scratch,
                                 size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return coordinate_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_coordinate_operator(struct SimContext*                 context,
                                      const SimCoordinateOperatorConfig* config,
                                      size_t*                            out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimCoordinateOperatorState* state =
        (SimCoordinateOperatorState*) calloc(1U, sizeof(SimCoordinateOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimCoordinateOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.output_field = 0U;
        local.mode         = SIM_COORD_MODE_INDEX;
        local.normalize    = SIM_COORD_NORMALIZE_NONE;
        local.gain         = 1.0;
        local.bias         = 0.0;
        local.time_offset  = 0.0;
        local.accumulate   = false;
        local.scale_by_dt  = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "coordinate", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    coordinate_normalize_config(&local);
    state->config = local;
    coordinate_refresh_runtime(state);
    coordinate_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "coordinate");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_UTILITY;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "coordinate";
    sim_operator_info_set_schema_identity(&info, "coordinate");
    info.algebraic_flags       = SIM_OPERATOR_ALG_AFFINE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;

    SimField* output_field = sim_context_field(context, state->config.output_field);
    bool needs_complex = output_field != NULL &&
                         sim_scalar_domain_is_complex(sim_scalar_domain_from_field(output_field));
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    if (output_field == NULL) {
        coordinate_destroy(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (sim_operator_domain_is_exact_integer(sim_field_scalar_domain(output_field))) {
        if (!coordinate_refresh_integer_state(state, output_field)) {
            coordinate_destroy(state);
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else if (!sim_operator_field_domain_is_f64_or_c64(output_field)) {
        coordinate_destroy(state);
        return SIM_RESULT_TYPE_MISMATCH;
    }

    SimSplitPort   port   = { .context_field_index = state->config.output_field,
                              .require_complex     = needs_complex };
    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = coordinate_step,
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
                                .symbolic      = coordinate_symbolic,
                                .destroy       = coordinate_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        coordinate_destroy(state);
    }

    return result;
}

SimResult sim_coordinate_config(struct SimContext*           context,
                                size_t                       operator_index,
                                SimCoordinateOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimCoordinateOperatorState* state = (SimCoordinateOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_coordinate_update(struct SimContext*                 context,
                                size_t                             operator_index,
                                const SimCoordinateOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimCoordinateOperatorState* state = (SimCoordinateOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimCoordinateOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "coordinate", true, config->scale_by_dt);
    }

    coordinate_normalize_config(&local);
    state->config = local;
    {
        SimField* output_field = sim_context_field(context, state->config.output_field);
        if (output_field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (sim_operator_domain_is_exact_integer(sim_field_scalar_domain(output_field))) {
            if (!coordinate_refresh_integer_state(state, output_field)) {
                return SIM_RESULT_TYPE_MISMATCH;
            }
        } else {
            state->integer_affine_enabled = false;
            state->integer_gain_raw       = 0U;
            state->integer_bias_raw       = 0U;
            if (!sim_operator_field_domain_is_f64_or_c64(output_field)) {
                return SIM_RESULT_TYPE_MISMATCH;
            }
        }
    }
    coordinate_refresh_runtime(state);
    coordinate_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
