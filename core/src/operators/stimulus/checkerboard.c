#include "oakfield/operators/stimulus/checkerboard.h"
#include "operators/common/operator_utils.h"
#include "oakfield/plane_chart.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"

#include <limits.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_CHECKERBOARD_VDSP_MIN_LEN 64U

typedef struct SimStimulusCheckerboardState {
    SimStimulusCheckerboardConfig config;
    bool                          has_plane_chart;
    SimPlaneSamplingFrame         plane_frame;
    SimPlaneChartConfig           plane_chart;
    SimPlaneProjectionConfig      plane_projection;
    SimPlaneChartStatus           plane_chart_status;
    char                          symbolic[160];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_row;
    size_t  vdsp_capacity;
#endif
} SimStimulusCheckerboardState;

static void checkerboard_normalize(SimStimulusCheckerboardConfig* config) {
    if (config == NULL) {
        return;
    }
    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->period_x) || config->period_x < 1.0) {
        config->period_x = 2.0;
    }
    if (!isfinite(config->period_y)) {
        config->period_y = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }
    if (!isfinite(config->complex_phase)) {
        config->complex_phase = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void checkerboard_refresh_runtime(SimStimulusCheckerboardState* state) {
    if (state == NULL) {
        return;
    }

    state->has_plane_chart    = false;
    state->plane_chart_status = SIM_PLANE_CHART_STATUS_UNSUPPORTED;

    if (state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        return;
    }

    state->plane_chart_status = sim_plane_chart_from_stimulus_coord(
        &state->config.coord, &state->plane_frame, &state->plane_chart, &state->plane_projection);
    state->has_plane_chart = (state->plane_chart_status == SIM_PLANE_CHART_STATUS_OK);
}

static void checkerboard_refresh_symbolic(SimStimulusCheckerboardState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    const SimStimulusCheckerboardConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "checkerboard A=%.3g Px=%.3g Py=%.3g",
                    cfg->amplitude,
                    cfg->period_x,
                    cfg->period_y);
#else
    (void) state;
#endif
}

static const char* checkerboard_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusCheckerboardState* state = (const SimStimulusCheckerboardState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

typedef struct StimulusCheckerboardIRParams {
    bool needs_dt;
} StimulusCheckerboardIRParams;

static SimIRNodeId checkerboard_ir_binary(SimIRBuilder* builder,
                                          SimIRNodeType type,
                                          SimIRNodeId   lhs,
                                          SimIRNodeId   rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId checkerboard_build_ir(SimIRBuilder*                        builder,
                                         const SimStimulusCheckerboardConfig* config,
                                         StimulusCheckerboardIRParams*        params) {
    if (builder == NULL || config == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (config->coord.mode != SIM_STIMULUS_COORD_AXIS) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt = false;

    SimIRNodeId coord_x   = sim_ir_builder_coord(builder, 0U, 0U);
    SimIRNodeId coord_y   = sim_ir_builder_coord(builder, 0U, 1U);
    SimIRNodeId spacing_x = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin_x  = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId spacing_y = sim_ir_builder_constant(builder, config->coord.spacing_y);
    SimIRNodeId origin_y  = sim_ir_builder_constant(builder, config->coord.origin_y);
    SimIRNodeId x         = checkerboard_ir_binary(
        builder,
        SIM_IR_NODE_ADD,
        origin_x,
        checkerboard_ir_binary(builder, SIM_IR_NODE_MUL, spacing_x, coord_x));
    SimIRNodeId y = checkerboard_ir_binary(
        builder,
        SIM_IR_NODE_ADD,
        origin_y,
        checkerboard_ir_binary(builder, SIM_IR_NODE_MUL, spacing_y, coord_y));
    if (x == SIM_IR_INVALID_NODE || y == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId u        = (config->coord.axis == SIM_STIMULUS_AXIS_Y) ? y : x;
    SimIRNodeId v        = (config->coord.axis == SIM_STIMULUS_AXIS_Y) ? x : y;
    SimIRNodeId period_x = sim_ir_builder_constant(builder, config->period_x);
    SimIRNodeId phase    = sim_ir_builder_constant(builder, config->phase);
    SimIRNodeId cell_x   = sim_ir_builder_floor(
        builder,
        checkerboard_ir_binary(builder,
                               SIM_IR_NODE_ADD,
                               checkerboard_ir_binary(builder, SIM_IR_NODE_DIV, u, period_x),
                               phase));

    SimIRNodeId cell_y = sim_ir_builder_constant(builder, 0.0);
    if (config->period_y > 0.0) {
        SimIRNodeId period_y = sim_ir_builder_constant(builder, config->period_y);
        cell_y               = sim_ir_builder_floor(
            builder,
            checkerboard_ir_binary(builder,
                                   SIM_IR_NODE_ADD,
                                   checkerboard_ir_binary(builder, SIM_IR_NODE_DIV, v, period_y),
                                   phase));
    }

    if (cell_x == SIM_IR_INVALID_NODE || cell_y == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId cell_sum  = checkerboard_ir_binary(builder, SIM_IR_NODE_ADD, cell_x, cell_y);
    SimIRNodeId pi_node   = sim_ir_builder_constant(builder, M_PI);
    SimIRNodeId angle     = checkerboard_ir_binary(builder, SIM_IR_NODE_MUL, pi_node, cell_sum);
    SimIRNodeId sign      = sim_ir_builder_call(builder, SIM_IR_CALL_COS, angle);
    SimIRNodeId amplitude = sim_ir_builder_constant(builder, config->amplitude);
    SimIRNodeId value     = checkerboard_ir_binary(builder, SIM_IR_NODE_MUL, amplitude, sign);
    if (value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId scale = SIM_IR_INVALID_NODE;
    if (config->scale_by_dt) {
        params->needs_dt      = true;
        SimIRNodeId dt_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
        SimIRNodeId dt_scaled = dt_node;
        scale                 = dt_scaled;
    } else {
        scale = sim_ir_builder_constant(builder, 1.0);
    }
    return checkerboard_ir_binary(builder, SIM_IR_NODE_MUL, value, scale);
}

static bool checkerboard_eval_uv(const SimStimulusCheckerboardState* state,
                                 double                              x,
                                 double                              y,
                                 double                              sample_x,
                                 double                              sample_y,
                                 double                              time_value,
                                 double*                             out_u,
                                 double*                             out_v) {
    if (state == NULL || out_u == NULL || out_v == NULL || !isfinite(x) || !isfinite(y) ||
        !isfinite(sample_x) || !isfinite(sample_y) || !isfinite(time_value)) {
        return false;
    }

    const SimStimulusCheckerboardConfig* cfg = &state->config;

    if (state->has_plane_chart) {
        if (cfg->coord.mode == SIM_STIMULUS_COORD_AXIS) {
            SimPlaneChartCoord chart_coord = { 0 };
            if (sim_plane_chart_eval(
                    &state->plane_frame, &state->plane_chart, x, y, time_value, &chart_coord) ==
                    SIM_PLANE_CHART_STATUS_OK &&
                isfinite(chart_coord.primary) && isfinite(chart_coord.secondary)) {
                if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                    *out_u = chart_coord.secondary;
                    *out_v = chart_coord.primary;
                } else {
                    *out_u = chart_coord.primary;
                    *out_v = chart_coord.secondary;
                }
                return true;
            }
        } else {
            SimPlaneProjectionValue projected = { 0 };
            if (sim_plane_chart_eval_projected(&state->plane_frame,
                                               &state->plane_chart,
                                               &state->plane_projection,
                                               x,
                                               y,
                                               time_value,
                                               &projected) == SIM_PLANE_CHART_STATUS_OK &&
                isfinite(projected.primary)) {
                *out_u = projected.primary;
                *out_v = 0.0;
                return true;
            }
        }
    }

    if (cfg->coord.mode == SIM_STIMULUS_COORD_AXIS) {
        if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
            *out_u = sample_y;
            *out_v = sample_x;
        } else {
            *out_u = sample_x;
            *out_v = sample_y;
        }
        return true;
    }

    *out_u = sim_stimulus_coord_u(&cfg->coord, x, y, time_value);
    *out_v = 0.0;
    return isfinite(*out_u);
}

static void checkerboard_destroy(void* state_ptr) {
    SimStimulusCheckerboardState* state = (SimStimulusCheckerboardState*) state_ptr;
    if (state != NULL) {
#if defined(SIM_HAVE_VDSP)
        free(state->vdsp_row);
#endif
    }
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool checkerboard_vdsp_ensure_buffer(SimStimulusCheckerboardState* state, size_t width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_row != NULL) {
        return true;
    }

    double* resized = (double*) realloc(state->vdsp_row, width * sizeof(double));
    if (resized == NULL) {
        return false;
    }

    state->vdsp_row      = resized;
    state->vdsp_capacity = width;
    return true;
}

static void checkerboard_fill_constant(double* row, size_t offset, size_t count, double value) {
    if (row == NULL || count == 0U) {
        return;
    }

    const vDSP_Length len = (vDSP_Length) count;
    vDSP_vfillD(&value, row + offset, 1, len);
}

static bool checkerboard_fill_parity_row(double* row,
                                         size_t  width,
                                         double  q_start,
                                         double  q_step,
                                         int64_t parity_bias) {
    if (row == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (!isfinite(q_start) || !isfinite(q_step)) {
        return false;
    }

    size_t  run_start = 0U;
    int64_t cell      = (int64_t) floor(q_start);
    double  sign      = (((cell + parity_bias) & 1LL) == 0LL) ? 1.0 : -1.0;

    for (size_t col = 1U; col < width; ++col) {
        double  q         = q_start + q_step * (double) col;
        int64_t next_cell = (int64_t) floor(q);
        if (next_cell == cell) {
            continue;
        }

        checkerboard_fill_constant(row, run_start, col - run_start, sign);
        run_start = col;
        cell      = next_cell;
        sign      = (((cell + parity_bias) & 1LL) == 0LL) ? 1.0 : -1.0;
    }

    checkerboard_fill_constant(row, run_start, width - run_start, sign);
    return true;
}

static bool checkerboard_try_vdsp_rows(SimStimulusCheckerboardState* state,
                                       const SimField*               field,
                                       bool                          is_complex,
                                       double*                       dst_real,
                                       SimComplexDouble*             dst_complex,
                                       size_t                        count,
                                       double                        scale,
                                       double                        time_value) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t rank = field->layout.rank;
    if (rank == 0U || rank > 2U) {
        return false;
    }

    const SimStimulusCheckerboardConfig* cfg = &state->config;
    bool separable                           = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool axis_mode                           = (cfg->coord.mode == SIM_STIMULUS_COORD_AXIS);
    bool angle_mode                          = (cfg->coord.mode == SIM_STIMULUS_COORD_ANGLE);

    if (!separable && !axis_mode && !angle_mode) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_CHECKERBOARD_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!checkerboard_vdsp_ensure_buffer(state, width)) {
        return false;
    }

    double period_x    = (cfg->period_x >= 1.0) ? cfg->period_x : 2.0;
    double period_y    = (cfg->period_y > 0.0) ? cfg->period_y : 0.0;
    bool   use_y       = (period_y > 0.0);
    double phase_cells = isfinite(cfg->phase) ? cfg->phase : 0.0;
    double sample_x0   = cfg->coord.origin_x - cfg->coord.velocity_x * time_value;
    double sample_y0   = cfg->coord.origin_y - cfg->coord.velocity_y * time_value;
    double dx          = cfg->coord.spacing_x;
    double dy          = cfg->coord.spacing_y;
    double value_scale = cfg->amplitude * scale;
    double real_scale  = value_scale;
    double imag_scale  = 0.0;

    if (is_complex) {
        real_scale = value_scale * cos(cfg->complex_phase);
        imag_scale = value_scale * sin(cfg->complex_phase);
    }

    if (!isfinite(period_x) || !isfinite(period_y) || !isfinite(phase_cells) ||
        !isfinite(sample_x0) || !isfinite(sample_y0) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(real_scale) || !isfinite(imag_scale)) {
        return false;
    }
    if (real_scale == 0.0 && (!is_complex || imag_scale == 0.0)) {
        return true;
    }

    double  angle_s           = 0.0;
    double  angle_c           = 1.0;
    int64_t angle_parity_bias = 0LL;
    if (angle_mode) {
        angle_s = sin(cfg->coord.angle);
        angle_c = cos(cfg->coord.angle);
        if (!isfinite(angle_s) || !isfinite(angle_c)) {
            return false;
        }
        if (use_y) {
            angle_parity_bias = (int64_t) floor(phase_cells);
        }
    }

    for (size_t row = 0U; row < height; ++row) {
        double* row_values = state->vdsp_row;
        double  sample_y   = sample_y0 + (double) row * dy;

        if (!isfinite(sample_y)) {
            return false;
        }

        if (separable) {
            if (!checkerboard_fill_parity_row(
                    row_values, width, sample_x0 / period_x + phase_cells, dx / period_x, 0LL)) {
                return false;
            }

            if (use_y) {
                int64_t cell_y = (int64_t) floor(sample_y / period_y + phase_cells);
                double  sign_y = ((cell_y & 1LL) == 0LL) ? 1.0 : -1.0;
                if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    sim_accel_add_scalar_real(row_values, width, sign_y);
                } else {
                    sim_accel_scale_inplace_real(row_values, width, sign_y);
                }
            }
        } else if (axis_mode) {
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                int64_t parity_bias = (int64_t) floor(sample_y / period_x + phase_cells);
                if (use_y) {
                    if (!checkerboard_fill_parity_row(row_values,
                                                      width,
                                                      sample_x0 / period_y + phase_cells,
                                                      dx / period_y,
                                                      parity_bias)) {
                        return false;
                    }
                } else {
                    double sign = ((parity_bias & 1LL) == 0LL) ? 1.0 : -1.0;
                    checkerboard_fill_constant(row_values, 0U, width, sign);
                }
            } else {
                int64_t parity_bias = 0LL;
                if (use_y) {
                    parity_bias = (int64_t) floor(sample_y / period_y + phase_cells);
                }
                if (!checkerboard_fill_parity_row(row_values,
                                                  width,
                                                  sample_x0 / period_x + phase_cells,
                                                  dx / period_x,
                                                  parity_bias)) {
                    return false;
                }
            }
        } else {
            double q_start = (angle_c * sample_x0 + angle_s * sample_y) / period_x + phase_cells;
            double q_step  = (angle_c * dx) / period_x;
            if (!checkerboard_fill_parity_row(
                    row_values, width, q_start, q_step, angle_parity_bias)) {
                return false;
            }
        }

        size_t offset = row * width;
        if (!is_complex) {
            sim_accel_copy_scale_real(row_values, dst_real + offset, width, real_scale, true);
        } else {
            sim_accel_accumulate_real_to_complex(
                row_values, dst_complex + offset, width, real_scale, imag_scale);
        }
    }

    return true;
}
#endif

static SimResult checkerboard_step(void*               state_ptr,
                                   struct SimContext*  context,
                                   struct SimOperator* self,
                                   size_t              substep_index,
                                   double              dt_sub,
                                   void*               scratch,
                                   size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    SimStimulusCheckerboardState* state = (SimStimulusCheckerboardState*) state_ptr;
    if (state == NULL || context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusCheckerboardConfig* cfg = &state->config;
    if (cfg->amplitude == 0.0)
        return SIM_RESULT_OK;

    SimField* field = sim_context_field(context, cfg->field_index);
    if (field == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (is_complex) {
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    } else {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    }

    if (count == 0U)
        return SIM_RESULT_OK;

    double period_x    = (cfg->period_x >= 1.0) ? cfg->period_x : 2.0;
    double period_y    = (cfg->period_y > 0.0) ? cfg->period_y : 0.0;
    bool   use_y       = (period_y > 0.0);
    double phase       = cfg->phase;
    double phase_cells = phase;
    if (!isfinite(phase_cells))
        phase_cells = 0.0;

    double scale      = cfg->scale_by_dt ? dt_sub : 1.0;
    double time_value = sim_context_time(context);
    bool   separable  = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

#if defined(SIM_HAVE_VDSP)
    if (checkerboard_try_vdsp_rows(state,
                                   field,
                                   is_complex,
                                   is_complex ? NULL : (double*) sim_field_data(field),
                                   is_complex ? sim_field_complex_data(field) : NULL,
                                   count,
                                   scale,
                                   time_value)) {
        return SIM_RESULT_OK;
    }
#endif

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL)
            return SIM_RESULT_INVALID_ARGUMENT;

        for (size_t idx = 0U; idx < count; ++idx) {
            double x        = 0.0;
            double y        = 0.0;
            double sample_x = 0.0;
            double sample_y = 0.0;
            if (sim_stimulus_coord_xy(&cfg->coord, field, idx, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_stimulus_coord_sample_xy(&cfg->coord, x, y, time_value, &sample_x, &sample_y);

            double sign = 1.0;
            if (separable) {
                double cell_x = floor((sample_x / period_x) + phase_cells);
                double cell_y = use_y ? floor((sample_y / period_y) + phase_cells) : 0.0;
                double sign_x = ((((int) cell_x) & 1) == 0) ? 1.0 : -1.0;
                double sign_y = use_y ? ((((int) cell_y) & 1) == 0 ? 1.0 : -1.0) : 0.0;
                if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    sign = use_y ? (sign_x + sign_y) : sign_x;
                } else {
                    sign = use_y ? (sign_x * sign_y) : sign_x;
                }
            } else {
                double u = 0.0;
                double v = 0.0;
                if (!checkerboard_eval_uv(state, x, y, sample_x, sample_y, time_value, &u, &v)) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                double cell_x = floor((u / period_x) + phase_cells);
                double cell_y = use_y ? floor((v / period_y) + phase_cells) : 0.0;
                int    parity = ((int) cell_x + (int) cell_y) & 1;
                sign          = (parity == 0) ? 1.0 : -1.0;
            }

            double value = cfg->amplitude * sign;
            if (isfinite(value)) {
                dst[idx] += scale * value;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL)
            return SIM_RESULT_INVALID_ARGUMENT;

        double cr = cos(cfg->complex_phase);
        double sr = sin(cfg->complex_phase);

        for (size_t idx = 0U; idx < count; ++idx) {
            double x        = 0.0;
            double y        = 0.0;
            double sample_x = 0.0;
            double sample_y = 0.0;
            if (sim_stimulus_coord_xy(&cfg->coord, field, idx, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_stimulus_coord_sample_xy(&cfg->coord, x, y, time_value, &sample_x, &sample_y);

            double sign = 1.0;
            if (separable) {
                double cell_x = floor((sample_x / period_x) + phase_cells);
                double cell_y = use_y ? floor((sample_y / period_y) + phase_cells) : 0.0;
                double sign_x = ((((int) cell_x) & 1) == 0) ? 1.0 : -1.0;
                double sign_y = use_y ? ((((int) cell_y) & 1) == 0 ? 1.0 : -1.0) : 0.0;
                if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    sign = use_y ? (sign_x + sign_y) : sign_x;
                } else {
                    sign = use_y ? (sign_x * sign_y) : sign_x;
                }
            } else {
                double u = 0.0;
                double v = 0.0;
                if (!checkerboard_eval_uv(state, x, y, sample_x, sample_y, time_value, &u, &v)) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                double cell_x = floor((u / period_x) + phase_cells);
                double cell_y = use_y ? floor((v / period_y) + phase_cells) : 0.0;
                int    parity = ((int) cell_x + (int) cell_y) & 1;
                sign          = (parity == 0) ? 1.0 : -1.0;
            }

            double mag = cfg->amplitude * sign;
            double re  = mag * cr;
            double im  = mag * sr;
            if (isfinite(re) && isfinite(im)) {
                dst[idx].re += scale * re;
                dst[idx].im += scale * im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_checkerboard_operator(struct SimContext*                   context,
                                                 const SimStimulusCheckerboardConfig* config,
                                                 size_t*                              out_index) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusCheckerboardConfig local = { 0 };
    if (config != NULL)
        local = *config;

    checkerboard_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_checkerboard",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusCheckerboardState* state =
        (SimStimulusCheckerboardState*) calloc(1U, sizeof(SimStimulusCheckerboardState));
    if (state == NULL)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config = local;
    checkerboard_refresh_runtime(state);
    checkerboard_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_checkerboard");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
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
    info.abstract_id       = "stimulus_checkerboard";
    sim_operator_info_set_schema_identity(&info, "stimulus_checkerboard");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    bool needs_complex =
        sim_field_is_complex(sim_context_field(context, state->config.field_index));

    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_checkerboard")) {
        SimField* field = sim_context_field(context, local.field_index);
        if (field != NULL) {
            bool is_complex = sim_field_is_complex(field);
            if ((!is_complex && field->element_size != sizeof(double)) ||
                (is_complex && field->element_size != sizeof(SimComplexDouble))) {
                field = NULL;
            }
        }
        if (field != NULL && local.coord.mode != SIM_STIMULUS_COORD_AXIS) {
            field = NULL;
        }
        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };
                StimulusCheckerboardIRParams       ir_params   = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId delta      = checkerboard_build_ir(builder, &local, &ir_params);
                if (is_complex && delta != SIM_IR_INVALID_NODE) {
                    SimIRNodeId zero   = sim_ir_builder_constant(builder, 0.0);
                    SimIRNodeId packed = sim_ir_builder_complex_pack(builder, delta, zero);
                    if (packed != SIM_IR_INVALID_NODE && local.complex_phase != 0.0) {
                        SimIRNodeId angle = sim_ir_builder_constant(builder, local.complex_phase);
                        packed            = sim_ir_builder_complex_rotate(builder, packed, angle);
                    }
                    delta = packed;
                }
                SimIRNodeId sum =
                    checkerboard_ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

                if (field_node != SIM_IR_INVALID_NODE && delta != SIM_IR_INVALID_NODE &&
                    sum != SIM_IR_INVALID_NODE) {
                    size_t param_count = ir_params.needs_dt ? ((size_t) SIM_IR_PARAM_DT + 1U) : 0U;

                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = sum;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = param_count;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = checkerboard_destroy;
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
                        registered_kernel = true;
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = checkerboard_step,
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
                                .symbolic      = checkerboard_symbolic,
                                .destroy       = checkerboard_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        checkerboard_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_checkerboard_config(struct SimContext*             context,
                                           size_t                         operator_index,
                                           SimStimulusCheckerboardConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusCheckerboardState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusCheckerboardState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusCheckerboardState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_checkerboard_update(struct SimContext*                   context,
                                           size_t                               operator_index,
                                           const SimStimulusCheckerboardConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusCheckerboardState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusCheckerboardState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusCheckerboardState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    SimStimulusCheckerboardConfig local = state->config;
    if (config != NULL)
        local = *config;

    checkerboard_normalize(&local);
    state->config = local;
    checkerboard_refresh_runtime(state);
    checkerboard_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
