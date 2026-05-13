#include "oakfield/operators/stimulus/zeta_plane_slice.h"

#include "oakfield/field.h"
#include "oakfield/math/xi.h"
#include "oakfield/math/zeta.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "oakfield/plane_chart.h"
#include "oakfield/sim_context.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_ZETA_PLANE_SLICE_SPAN_EPS 1.0e-9

#if !OAKFIELD_ENABLE_ZETA_CORE

SimResult sim_add_stimulus_zeta_plane_slice_operator(struct SimContext*                     context,
                                                     const SimStimulusZetaPlaneSliceConfig* config,
                                                     size_t* out_index) {
    (void) context;
    (void) config;
    (void) out_index;
    return SIM_RESULT_NOT_SUPPORTED;
}

SimResult sim_stimulus_zeta_plane_slice_config(struct SimContext*               context,
                                               size_t                           operator_index,
                                               SimStimulusZetaPlaneSliceConfig* out_config) {
    (void) context;
    (void) operator_index;
    (void) out_config;
    return SIM_RESULT_NOT_SUPPORTED;
}

SimResult sim_stimulus_zeta_plane_slice_update(struct SimContext* context,
                                               size_t             operator_index,
                                               const SimStimulusZetaPlaneSliceConfig* config) {
    (void) context;
    (void) operator_index;
    (void) config;
    return SIM_RESULT_NOT_SUPPORTED;
}

#else

typedef struct SimStimulusZetaPlaneSliceState {
    SimStimulusZetaPlaneSliceConfig config;
    SimZetaContext                  zeta_context;
    SimXiContext                    xi_context;
    SimPlaneChartConfig             plane_chart;
    SimPlaneProjectionConfig        sigma_projection;
    SimPlaneProjectionConfig        t_projection;
    char                            symbolic[192];
} SimStimulusZetaPlaneSliceState;

typedef struct SimStimulusZetaPlaneSliceChartMap {
    SimPlaneSamplingFrame frame;
    double                primary_min;
    double                primary_max;
    double                secondary_min;
    double                secondary_max;
    bool                  valid;
} SimStimulusZetaPlaneSliceChartMap;

static inline double complex zeta_plane_slice_to_c64(SimComplexDouble z) {
    return z.re + z.im * I;
}

static const char* zeta_plane_slice_family_name(SimStimulusZetaPlaneSliceFamily family) {
    switch (family) {
        case SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI:
            return "xi";
        case SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA:
        default:
            return "zeta";
    }
}

static const char* zeta_plane_slice_view_name(SimStimulusZetaPlaneSliceViewMode view_mode) {
    switch (view_mode) {
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_IM:
            return "im";
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ABS:
            return "abs";
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS:
            return "log_abs";
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ARG:
            return "arg";
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_RE:
        default:
            return "re";
    }
}

static const char* zeta_plane_slice_render_name(SimStimulusZetaPlaneSliceRenderMode render_mode) {
    switch (render_mode) {
        case SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE:
            return "interactive";
        case SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT:
        default:
            return "exact";
    }
}

static const char* zeta_plane_slice_chart_name(SimPlaneChartKind chart_kind) {
    switch (chart_kind) {
        case SIM_PLANE_CHART_POLAR:
            return "polar";
        case SIM_PLANE_CHART_ELLIPTIC:
            return "elliptic";
        case SIM_PLANE_CHART_SPIRAL:
            return "spiral";
        case SIM_PLANE_CHART_CARTESIAN:
        default:
            return "cartesian";
    }
}

static const char* zeta_plane_slice_projection_name(SimPlaneProjectionKind projection_kind) {
    switch (projection_kind) {
        case SIM_PLANE_PROJECTION_SECONDARY:
            return "secondary";
        case SIM_PLANE_PROJECTION_PRIMARY:
        default:
            return "primary";
    }
}

static void zeta_plane_slice_normalize_axis_projection(SimPlaneProjectionKind* projection_kind,
                                                       SimPlaneProjectionKind  fallback) {
    SimPlaneProjectionConfig projection = { 0 };

    if (projection_kind == NULL) {
        return;
    }

    projection.kind = *projection_kind;
    sim_plane_projection_normalize(&projection);
    if (projection.kind == SIM_PLANE_PROJECTION_FULL) {
        projection.kind = fallback;
    }
    *projection_kind = projection.kind;
}

static void zeta_plane_slice_normalize(SimStimulusZetaPlaneSliceConfig* config) {
    SimPlaneChartConfig chart = { 0 };

    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->sigma_center)) {
        config->sigma_center = 0.5;
    }
    if (!isfinite(config->t_center)) {
        config->t_center = 0.0;
    }
    if (!isfinite(config->sigma_span) ||
        fabs(config->sigma_span) <= STIM_ZETA_PLANE_SLICE_SPAN_EPS) {
        config->sigma_span = 4.0;
    }
    config->sigma_span = fabs(config->sigma_span);
    if (!isfinite(config->t_span) || fabs(config->t_span) <= STIM_ZETA_PLANE_SLICE_SPAN_EPS) {
        config->t_span = 40.0;
    }
    config->t_span = fabs(config->t_span);
    if (!isfinite(config->log_floor) || config->log_floor <= DBL_MIN) {
        config->log_floor = 1.0e-12;
    }
    if (config->family != SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA &&
        config->family != SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI) {
        config->family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA;
    }
    if (config->view_mode < SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_RE ||
        config->view_mode > SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ARG) {
        config->view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS;
    }
    if (config->render_mode < SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT ||
        config->render_mode > SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE) {
        config->render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT;
    }
    zeta_plane_slice_normalize_axis_projection(&config->sigma_projection,
                                               SIM_PLANE_PROJECTION_PRIMARY);
    zeta_plane_slice_normalize_axis_projection(&config->t_projection,
                                               SIM_PLANE_PROJECTION_SECONDARY);

    if (!isfinite(config->chart_center_x)) {
        config->chart_center_x = 0.0;
    }
    if (!isfinite(config->chart_center_y)) {
        config->chart_center_y = 0.0;
    }

    chart.kind                    = config->chart_kind;
    chart.secondary_wrap          = (config->chart_kind == SIM_PLANE_CHART_CARTESIAN)
                                        ? SIM_PLANE_CHART_WRAP_NONE
                                        : SIM_PLANE_CHART_WRAP_SIGNED_ANGLE;
    chart.rotation                = config->chart_rotation;
    chart.ellipse_u               = config->chart_ellipse_u;
    chart.ellipse_v               = config->chart_ellipse_v;
    chart.spiral_arms             = config->chart_spiral_arms;
    chart.spiral_pitch            = config->chart_spiral_pitch;
    chart.spiral_phase            = config->chart_spiral_phase;
    chart.spiral_angular_velocity = config->chart_spiral_angular_velocity;
    sim_plane_chart_normalize(&chart);

    config->chart_kind                    = chart.kind;
    config->chart_rotation                = chart.rotation;
    config->chart_ellipse_u               = chart.ellipse_u;
    config->chart_ellipse_v               = chart.ellipse_v;
    config->chart_spiral_arms             = chart.spiral_arms;
    config->chart_spiral_pitch            = chart.spiral_pitch;
    config->chart_spiral_phase            = chart.spiral_phase;
    config->chart_spiral_angular_velocity = chart.spiral_angular_velocity;
}

static void zeta_plane_slice_refresh_runtime(SimStimulusZetaPlaneSliceState* state) {
    if (state == NULL) {
        return;
    }

    if (state->config.render_mode == SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT) {
        state->zeta_context = sim_zeta_context_default();
        state->xi_context   = sim_xi_context_default();
    } else {
        state->zeta_context = sim_zeta_context_interactive();
        state->xi_context   = sim_xi_context_interactive();
    }

    state->plane_chart = (SimPlaneChartConfig){
        .kind                    = state->config.chart_kind,
        .secondary_wrap          = (state->config.chart_kind == SIM_PLANE_CHART_CARTESIAN)
                                       ? SIM_PLANE_CHART_WRAP_NONE
                                       : SIM_PLANE_CHART_WRAP_SIGNED_ANGLE,
        .rotation                = state->config.chart_rotation,
        .ellipse_u               = state->config.chart_ellipse_u,
        .ellipse_v               = state->config.chart_ellipse_v,
        .spiral_arms             = state->config.chart_spiral_arms,
        .spiral_pitch            = state->config.chart_spiral_pitch,
        .spiral_phase            = state->config.chart_spiral_phase,
        .spiral_angular_velocity = state->config.chart_spiral_angular_velocity,
    };
    state->sigma_projection = (SimPlaneProjectionConfig){
        .kind = state->config.sigma_projection,
    };
    state->t_projection = (SimPlaneProjectionConfig){
        .kind = state->config.t_projection,
    };
    sim_plane_chart_normalize(&state->plane_chart);
    sim_plane_projection_normalize(&state->sigma_projection);
    sim_plane_projection_normalize(&state->t_projection);
}

static void zeta_plane_slice_refresh_symbolic(SimStimulusZetaPlaneSliceState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "zeta_plane %s %s %s %s sigma=%s%s t=%s%s sig=%.3g t=%.3g",
                    zeta_plane_slice_family_name(state->config.family),
                    zeta_plane_slice_render_name(state->config.render_mode),
                    zeta_plane_slice_view_name(state->config.view_mode),
                    zeta_plane_slice_chart_name(state->config.chart_kind),
                    zeta_plane_slice_projection_name(state->config.sigma_projection),
                    state->config.sigma_flip ? "(flip)" : "",
                    zeta_plane_slice_projection_name(state->config.t_projection),
                    state->config.t_flip ? "(flip)" : "",
                    state->config.sigma_center,
                    state->config.t_center);
#else
    (void) state;
#endif
}

static const char* zeta_plane_slice_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusZetaPlaneSliceState* state = (const SimStimulusZetaPlaneSliceState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void zeta_plane_slice_destroy(void* state_ptr) {
    free(state_ptr);
}

static bool zeta_plane_slice_build_frame(const SimField*                        field,
                                         const SimStimulusZetaPlaneSliceConfig* config,
                                         SimPlaneSamplingFrame*                 out_frame) {
    size_t width  = 0U;
    size_t height = 0U;

    if (field == NULL || config == NULL || out_frame == NULL) {
        return false;
    }

    if (field->layout.rank == 1U) {
        width  = field->layout.shape[0];
        height = 1U;
    } else if (field->layout.rank == 2U) {
        height = field->layout.shape[0];
        width  = field->layout.shape[1];
    } else {
        return false;
    }

    if (width == 0U || height == 0U) {
        return false;
    }

    *out_frame = (SimPlaneSamplingFrame){
        .origin_x   = (width > 1U) ? -0.5 : 0.0,
        .origin_y   = (height > 1U) ? 0.5 : 0.0,
        .spacing_x  = (width > 1U) ? (1.0 / (double) (width - 1U)) : 1.0,
        .spacing_y  = (height > 1U) ? (-1.0 / (double) (height - 1U)) : 1.0,
        .center_x   = config->chart_center_x,
        .center_y   = config->chart_center_y,
        .velocity_x = 0.0,
        .velocity_y = 0.0,
    };
    sim_plane_sampling_frame_normalize(out_frame);
    return true;
}

static double zeta_plane_slice_normalize_range(double value, double min_value, double max_value) {
    double span = max_value - min_value;

    if (!isfinite(value) || !isfinite(min_value) || !isfinite(max_value) ||
        fabs(span) <= STIM_ZETA_PLANE_SLICE_SPAN_EPS) {
        return 0.0;
    }

    return ((value - min_value) / span) - 0.5;
}

static bool zeta_plane_slice_eval_chart_coord(const SimField*                       field,
                                              size_t                                linear_index,
                                              const SimStimulusZetaPlaneSliceState* state,
                                              const SimPlaneSamplingFrame*          frame,
                                              SimPlaneChartCoord*                   out_coord) {
    size_t ix       = 0U;
    size_t iy       = 0U;
    double sample_x = 0.0;
    double sample_y = 0.0;

    if (field == NULL || state == NULL || frame == NULL || out_coord == NULL) {
        return false;
    }

    if (sim_field_index_to_xy(field, linear_index, &ix, &iy) != SIM_RESULT_OK) {
        return false;
    }

    sample_x = frame->origin_x + (double) ix * frame->spacing_x;
    sample_y = frame->origin_y + (double) iy * frame->spacing_y;

    return sim_plane_chart_eval(frame, &state->plane_chart, sample_x, sample_y, 0.0, out_coord) ==
               SIM_PLANE_CHART_STATUS_OK &&
           isfinite(out_coord->primary) && isfinite(out_coord->secondary);
}

static bool zeta_plane_slice_build_chart_map(const SimField*                       field,
                                             const SimStimulusZetaPlaneSliceState* state,
                                             SimStimulusZetaPlaneSliceChartMap*    out_map) {
    size_t count = 0U;

    if (field == NULL || state == NULL || out_map == NULL) {
        return false;
    }

    if (!zeta_plane_slice_build_frame(field, &state->config, &out_map->frame)) {
        return false;
    }

    out_map->primary_min   = HUGE_VAL;
    out_map->primary_max   = -HUGE_VAL;
    out_map->secondary_min = HUGE_VAL;
    out_map->secondary_max = -HUGE_VAL;
    out_map->valid         = false;

    count = sim_field_element_count(&field->layout);
    for (size_t i = 0U; i < count; ++i) {
        SimPlaneChartCoord coord = { 0.0, 0.0 };
        if (!zeta_plane_slice_eval_chart_coord(field, i, state, &out_map->frame, &coord)) {
            return false;
        }
        if (coord.primary < out_map->primary_min) {
            out_map->primary_min = coord.primary;
        }
        if (coord.primary > out_map->primary_max) {
            out_map->primary_max = coord.primary;
        }
        if (coord.secondary < out_map->secondary_min) {
            out_map->secondary_min = coord.secondary;
        }
        if (coord.secondary > out_map->secondary_max) {
            out_map->secondary_max = coord.secondary;
        }
        out_map->valid = true;
    }

    return out_map->valid;
}

static bool zeta_plane_slice_sample_coord(const SimField*                          field,
                                          size_t                                   linear_index,
                                          const SimStimulusZetaPlaneSliceState*    state,
                                          const SimStimulusZetaPlaneSliceChartMap* map,
                                          double*                                  out_sigma,
                                          double*                                  out_t) {
    SimPlaneChartCoord      coord           = { 0.0, 0.0 };
    SimPlaneProjectionValue sigma_projected = { 0.0, 0.0, false };
    SimPlaneProjectionValue t_projected     = { 0.0, 0.0, false };
    double                  sigma_min       = 0.0;
    double                  sigma_max       = 0.0;
    double                  t_min           = 0.0;
    double                  t_max           = 0.0;
    double                  sigma_u         = 0.0;
    double                  t_v             = 0.0;

    if (field == NULL || state == NULL || map == NULL || !map->valid || out_sigma == NULL ||
        out_t == NULL) {
        return false;
    }

    if (!zeta_plane_slice_eval_chart_coord(field, linear_index, state, &map->frame, &coord)) {
        return false;
    }

    if (sim_plane_projection_eval(&state->sigma_projection, coord, &sigma_projected) !=
            SIM_PLANE_CHART_STATUS_OK ||
        !isfinite(sigma_projected.primary)) {
        return false;
    }
    if (sim_plane_projection_eval(&state->t_projection, coord, &t_projected) !=
            SIM_PLANE_CHART_STATUS_OK ||
        !isfinite(t_projected.primary)) {
        return false;
    }

    if (state->sigma_projection.kind == SIM_PLANE_PROJECTION_SECONDARY) {
        sigma_min = map->secondary_min;
        sigma_max = map->secondary_max;
    } else {
        sigma_min = map->primary_min;
        sigma_max = map->primary_max;
    }
    if (state->t_projection.kind == SIM_PLANE_PROJECTION_SECONDARY) {
        t_min = map->secondary_min;
        t_max = map->secondary_max;
    } else {
        t_min = map->primary_min;
        t_max = map->primary_max;
    }

    sigma_u = zeta_plane_slice_normalize_range(sigma_projected.primary, sigma_min, sigma_max);
    t_v     = zeta_plane_slice_normalize_range(t_projected.primary, t_min, t_max);
    if (state->config.sigma_flip) {
        sigma_u = -sigma_u;
    }
    if (state->config.t_flip) {
        t_v = -t_v;
    }

    *out_sigma = state->config.sigma_center + state->config.sigma_span * sigma_u;
    *out_t     = state->config.t_center + state->config.t_span * t_v;
    return true;
}

static bool zeta_plane_slice_status_usable(SimZetaStatus status) {
    return status != SIM_ZETA_STATUS_INVALID_ARGUMENT && status != SIM_ZETA_STATUS_SINGULAR &&
           status != SIM_ZETA_STATUS_NUMERIC_FAILURE;
}

static bool zeta_plane_slice_eval_value(const SimStimulusZetaPlaneSliceState* state,
                                        double                                sigma,
                                        double                                t,
                                        SimComplexDouble*                     out_value) {
    SimComplexDouble s = { sigma, t };

    if (state == NULL || out_value == NULL) {
        return false;
    }

    if (state->config.family == SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI) {
        if (state->config.render_mode == SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT) {
            SimComplexBall ball = sim_xi_eval_ball(s, &state->xi_context);
            if (!zeta_plane_slice_status_usable(ball.status)) {
                return false;
            }
            *out_value = ball.center;
        } else {
            SimXiResult result = sim_xi_eval(s, &state->xi_context);
            if (!zeta_plane_slice_status_usable(result.status)) {
                return false;
            }
            *out_value = result.value;
        }
    } else {
        if (state->config.render_mode == SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT) {
            SimComplexBall ball = sim_zeta_eval_ball(s, &state->zeta_context);
            if (!zeta_plane_slice_status_usable(ball.status)) {
                return false;
            }
            *out_value = ball.center;
        } else {
            SimZetaResult result = sim_zeta_eval(s, &state->zeta_context);
            if (!zeta_plane_slice_status_usable(result.status)) {
                return false;
            }
            *out_value = result.value;
        }
    }

    return isfinite(out_value->re) && isfinite(out_value->im);
}

static double zeta_plane_slice_project(const SimStimulusZetaPlaneSliceConfig* config,
                                       SimComplexDouble                       value) {
    double complex z = zeta_plane_slice_to_c64(value);
    double         magnitude;

    if (config == NULL) {
        return 0.0;
    }

    magnitude = cabs(z);

    switch (config->view_mode) {
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_IM:
            return cimag(z);
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ABS:
            return magnitude;
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS:
            return log(fmax(magnitude, config->log_floor));
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ARG:
            return (magnitude <= config->log_floor) ? 0.0 : carg(z);
        case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_RE:
        default:
            return creal(z);
    }
}

static SimResult zeta_plane_slice_step(void*               state_ptr,
                                       struct SimContext*  context,
                                       struct SimOperator* self,
                                       size_t              substep_index,
                                       double              dt_sub,
                                       void*               scratch,
                                       size_t              scratch_size) {
    SimStimulusZetaPlaneSliceState*   state = (SimStimulusZetaPlaneSliceState*) state_ptr;
    SimField*                         field;
    bool                              is_complex;
    size_t                            count;
    SimStimulusZetaPlaneSliceChartMap chart_map = { 0 };

    (void) self;
    (void) substep_index;
    (void) dt_sub;
    (void) scratch;
    (void) scratch_size;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, state->config.field_index);
    if (field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    is_complex = sim_field_is_complex(field);
    if (!is_complex) {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    if (count == 0U || state->config.amplitude == 0.0) {
        return SIM_RESULT_OK;
    }

    if (!zeta_plane_slice_build_chart_map(field, state, &chart_map)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double           sigma = 0.0;
            double           t     = 0.0;
            SimComplexDouble value = { 0.0, 0.0 };
            double           sample;

            if (!zeta_plane_slice_sample_coord(field, i, state, &chart_map, &sigma, &t)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (!zeta_plane_slice_eval_value(state, sigma, t, &value)) {
                continue;
            }

            sample = state->config.amplitude * zeta_plane_slice_project(&state->config, value);
            if (isfinite(sample)) {
                dst[i] += sample;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double           sigma  = 0.0;
            double           t      = 0.0;
            SimComplexDouble value  = { 0.0, 0.0 };
            SimComplexDouble sample = { 0.0, 0.0 };

            if (!zeta_plane_slice_sample_coord(field, i, state, &chart_map, &sigma, &t)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (!zeta_plane_slice_eval_value(state, sigma, t, &value)) {
                continue;
            }

            sample.re = state->config.amplitude * zeta_plane_slice_project(&state->config, value);
            sample.im = 0.0;
            if (isfinite(sample.re) && isfinite(sample.im)) {
                dst[i].re += sample.re;
                dst[i].im += sample.im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_zeta_plane_slice_operator(struct SimContext*                     context,
                                                     const SimStimulusZetaPlaneSliceConfig* config,
                                                     size_t* out_index) {
    SimStimulusZetaPlaneSliceConfig local = { 0 };
    SimStimulusZetaPlaneSliceState* state;
    char                            name[SIM_OPERATOR_NAME_MAX + 1U];
    SimOperatorInfo                 info;
    SimOperatorConfig               op_config;
    SimSplitPort                    port;
    SimSplitAccess                  access;
    SimSplitSubstep                 substep;
    SimSplitDescriptor              desc = { 0 };
    bool                            needs_complex;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (config != NULL) {
        local = *config;
    }
    zeta_plane_slice_normalize(&local);

    state = (SimStimulusZetaPlaneSliceState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    zeta_plane_slice_refresh_runtime(state);
    zeta_plane_slice_refresh_symbolic(state);

    sim_operator_make_unique_name(name, sizeof(name), "stimulus_zeta_plane_slice");

    info                   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stimulus_zeta_plane_slice";
    sim_operator_info_set_schema_identity(&info, "stimulus_zeta_plane_slice");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;

    needs_complex = sim_field_is_complex(sim_context_field(context, state->config.field_index));
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = true;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    op_config = sim_operator_config_defaults();
    port      = (SimSplitPort){ .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    access    = (SimSplitAccess){ .port = 0U, .mode = SIM_ACCESS_RW };
    substep   = (SimSplitSubstep){ .name              = NULL,
                                   .fn                = zeta_plane_slice_step,
                                   .accesses          = &access,
                                   .access_count      = 1U,
                                   .dt_scale          = 1.0,
                                   .barrier_after     = false,
                                   .error_measure     = NULL,
                                   .required_features = 0U };
    desc      = (SimSplitDescriptor){
        .name          = name,
        .ports         = &port,
        .port_count    = 1U,
        .substeps      = &substep,
        .substep_count = 1U,
        .state         = state,
        .symbolic      = zeta_plane_slice_symbolic,
        .destroy       = zeta_plane_slice_destroy,
        .info          = info,
        .config        = op_config,
        .scratch       = { 0U, 0U },
    };

    {
        SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
        if (result != SIM_RESULT_OK) {
            zeta_plane_slice_destroy(state);
        }
        return result;
    }
}

SimResult sim_stimulus_zeta_plane_slice_config(struct SimContext*               context,
                                               size_t                           operator_index,
                                               SimStimulusZetaPlaneSliceConfig* out_config) {
    SimOperator*                    op;
    SimStimulusZetaPlaneSliceState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimStimulusZetaPlaneSliceState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_zeta_plane_slice_update(struct SimContext* context,
                                               size_t             operator_index,
                                               const SimStimulusZetaPlaneSliceConfig* config) {
    SimOperator*                    op;
    SimStimulusZetaPlaneSliceState* state;
    SimStimulusZetaPlaneSliceConfig local;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimStimulusZetaPlaneSliceState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local = state->config;
    if (config != NULL) {
        local = *config;
    }

    zeta_plane_slice_normalize(&local);
    state->config = local;
    zeta_plane_slice_refresh_runtime(state);
    zeta_plane_slice_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}

#endif
