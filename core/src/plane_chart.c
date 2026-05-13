/**
 * @file plane_chart.c
 * @brief Chart-aware 2D coordinate normalization and projection evaluation.
 *
 * Plane charts convert integer sample coordinates into cartesian, polar,
 * log-polar, screen, or user-defined parameter spaces with explicit wrapping,
 * rotation, scaling, and projection policies. Status values distinguish valid
 * samples from clipped, singular, or invalid chart evaluations.
 */
#include "oakfield/plane_chart.h"

#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static int sim_plane_chart_isfinite_value(double value) {
    return isfinite(value);
}

static double sim_plane_chart_wrap_value(double value, SimPlaneChartWrapPolicy wrap) {
    switch (wrap) {
        case SIM_PLANE_CHART_WRAP_UNSIGNED_ANGLE: {
            double wrapped = fmod(value, 2.0 * M_PI);
            if (wrapped < 0.0) {
                wrapped += 2.0 * M_PI;
            }
            return wrapped;
        }
        case SIM_PLANE_CHART_WRAP_UNIT_INTERVAL: {
            double wrapped = fmod(value, 1.0);
            if (wrapped < 0.0) {
                wrapped += 1.0;
            }
            return wrapped;
        }
        case SIM_PLANE_CHART_WRAP_SIGNED_ANGLE:
            return atan2(sin(value), cos(value));
        case SIM_PLANE_CHART_WRAP_NONE:
        default:
            return value;
    }
}

static void sim_plane_chart_rotate(double x, double y, double angle, double* out_u, double* out_v) {
    double s = sin(angle);
    double c = cos(angle);

    if (out_u == NULL || out_v == NULL) {
        return;
    }

    *out_u = x * c + y * s;
    *out_v = -x * s + y * c;
}

static void sim_plane_chart_sample_xy(const SimPlaneSamplingFrame* frame,
                                      double                       x,
                                      double                       y,
                                      double                       t,
                                      double*                      out_x,
                                      double*                      out_y) {
    double sample_x = x;
    double sample_y = y;

    if (frame != NULL) {
        sample_x -= frame->velocity_x * t;
        sample_y -= frame->velocity_y * t;
    }

    if (out_x != NULL) {
        *out_x = sample_x;
    }
    if (out_y != NULL) {
        *out_y = sample_y;
    }
}

static void sim_plane_chart_centered_xy(const SimPlaneSamplingFrame* frame,
                                        double                       x,
                                        double                       y,
                                        double                       t,
                                        double*                      out_dx,
                                        double*                      out_dy) {
    double sample_x = x;
    double sample_y = y;
    double center_x = 0.0;
    double center_y = 0.0;

    sim_plane_chart_sample_xy(frame, x, y, t, &sample_x, &sample_y);
    if (frame != NULL) {
        center_x = frame->center_x;
        center_y = frame->center_y;
    }

    if (out_dx != NULL) {
        *out_dx = sample_x - center_x;
    }
    if (out_dy != NULL) {
        *out_dy = sample_y - center_y;
    }
}

const char* sim_plane_chart_status_string(SimPlaneChartStatus status) {
    switch (status) {
        case SIM_PLANE_CHART_STATUS_OK:
            return "ok";
        case SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT:
            return "invalid-argument";
        case SIM_PLANE_CHART_STATUS_UNSUPPORTED:
            return "unsupported";
        case SIM_PLANE_CHART_STATUS_OUT_OF_DOMAIN:
            return "out-of-domain";
        case SIM_PLANE_CHART_STATUS_SINGULAR:
            return "singular";
        case SIM_PLANE_CHART_STATUS_NON_INVERTIBLE:
            return "non-invertible";
        case SIM_PLANE_CHART_STATUS_NUMERIC_FAILURE:
            return "numeric-failure";
        default:
            return "unknown";
    }
}

void sim_plane_sampling_frame_normalize(SimPlaneSamplingFrame* frame) {
    if (frame == NULL) {
        return;
    }

    if (!sim_plane_chart_isfinite_value(frame->origin_x)) {
        frame->origin_x = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(frame->origin_y)) {
        frame->origin_y = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(frame->spacing_x) ||
        fabs(frame->spacing_x) <= SIM_PLANE_CHART_EPS) {
        frame->spacing_x = 1.0;
    }
    if (!sim_plane_chart_isfinite_value(frame->spacing_y) ||
        fabs(frame->spacing_y) <= SIM_PLANE_CHART_EPS) {
        frame->spacing_y = frame->spacing_x;
    }
    if (!sim_plane_chart_isfinite_value(frame->center_x)) {
        frame->center_x = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(frame->center_y)) {
        frame->center_y = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(frame->velocity_x)) {
        frame->velocity_x = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(frame->velocity_y)) {
        frame->velocity_y = 0.0;
    }
}

void sim_plane_chart_normalize(SimPlaneChartConfig* chart) {
    if (chart == NULL) {
        return;
    }

    if (chart->kind < SIM_PLANE_CHART_CARTESIAN || chart->kind > SIM_PLANE_CHART_SPIRAL) {
        chart->kind = SIM_PLANE_CHART_CARTESIAN;
    }
    if (chart->secondary_wrap < SIM_PLANE_CHART_WRAP_NONE ||
        chart->secondary_wrap > SIM_PLANE_CHART_WRAP_UNIT_INTERVAL) {
        chart->secondary_wrap = (chart->kind == SIM_PLANE_CHART_CARTESIAN)
                                    ? SIM_PLANE_CHART_WRAP_NONE
                                    : SIM_PLANE_CHART_WRAP_SIGNED_ANGLE;
    }
    if (!sim_plane_chart_isfinite_value(chart->rotation)) {
        chart->rotation = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(chart->ellipse_u) ||
        fabs(chart->ellipse_u) <= SIM_PLANE_CHART_EPS) {
        chart->ellipse_u = 1.0;
    }
    chart->ellipse_u = fabs(chart->ellipse_u);
    if (!sim_plane_chart_isfinite_value(chart->ellipse_v) ||
        fabs(chart->ellipse_v) <= SIM_PLANE_CHART_EPS) {
        chart->ellipse_v = chart->ellipse_u;
    }
    chart->ellipse_v = fabs(chart->ellipse_v);
    if (!sim_plane_chart_isfinite_value(chart->spiral_arms)) {
        chart->spiral_arms = 1.0;
    }
    if (!sim_plane_chart_isfinite_value(chart->spiral_pitch)) {
        chart->spiral_pitch = 1.0;
    }
    if (!sim_plane_chart_isfinite_value(chart->spiral_phase)) {
        chart->spiral_phase = 0.0;
    }
    if (!sim_plane_chart_isfinite_value(chart->spiral_angular_velocity)) {
        chart->spiral_angular_velocity = 0.0;
    }
    if (chart->kind == SIM_PLANE_CHART_SPIRAL &&
        fabs(chart->spiral_arms) <= SIM_PLANE_CHART_EPS &&
        fabs(chart->spiral_pitch) <= SIM_PLANE_CHART_EPS) {
        chart->spiral_arms = 1.0;
        chart->spiral_pitch = 1.0;
    }
}

void sim_plane_projection_normalize(SimPlaneProjectionConfig* projection) {
    if (projection == NULL) {
        return;
    }

    if (projection->kind < SIM_PLANE_PROJECTION_FULL ||
        projection->kind > SIM_PLANE_PROJECTION_SECONDARY) {
        projection->kind = SIM_PLANE_PROJECTION_PRIMARY;
    }
}

unsigned int sim_plane_chart_semantic_flags(const SimPlaneChartConfig* chart) {
    SimPlaneChartConfig local = { 0 };

    if (chart == NULL) {
        return SIM_PLANE_CHART_SEMANTIC_NONE;
    }

    local = *chart;
    sim_plane_chart_normalize(&local);

    switch (local.kind) {
        case SIM_PLANE_CHART_POLAR:
        case SIM_PLANE_CHART_ELLIPTIC:
        case SIM_PLANE_CHART_SPIRAL:
            return SIM_PLANE_CHART_SEMANTIC_PERIODIC_SECONDARY |
                   SIM_PLANE_CHART_SEMANTIC_SINGULAR_CENTER |
                   SIM_PLANE_CHART_SEMANTIC_SECONDARY_BRANCH_CUT;
        case SIM_PLANE_CHART_CARTESIAN:
        default:
            return SIM_PLANE_CHART_SEMANTIC_NONE;
    }
}

unsigned int sim_plane_chart_capability_flags(const SimPlaneChartConfig* chart) {
    (void) chart;
    return SIM_PLANE_CHART_CAPABILITY_NONE;
}

SimPlaneChartStatus sim_plane_chart_eval(const SimPlaneSamplingFrame* frame,
                                         const SimPlaneChartConfig*   chart,
                                         double                       x,
                                         double                       y,
                                         double                       t,
                                         SimPlaneChartCoord*          out_coord) {
    SimPlaneSamplingFrame local_frame = { 0 };
    SimPlaneChartConfig   local_chart = { 0 };

    if (frame == NULL || chart == NULL || out_coord == NULL ||
        !sim_plane_chart_isfinite_value(x) || !sim_plane_chart_isfinite_value(y) ||
        !sim_plane_chart_isfinite_value(t)) {
        return SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT;
    }

    local_frame = *frame;
    local_chart = *chart;
    sim_plane_sampling_frame_normalize(&local_frame);
    sim_plane_chart_normalize(&local_chart);

    switch (local_chart.kind) {
        case SIM_PLANE_CHART_CARTESIAN: {
            double sample_x = 0.0;
            double sample_y = 0.0;
            sim_plane_chart_sample_xy(&local_frame, x, y, t, &sample_x, &sample_y);
            sim_plane_chart_rotate(sample_x, sample_y, local_chart.rotation, &out_coord->primary,
                                   &out_coord->secondary);
            return (sim_plane_chart_isfinite_value(out_coord->primary) &&
                    sim_plane_chart_isfinite_value(out_coord->secondary))
                       ? SIM_PLANE_CHART_STATUS_OK
                       : SIM_PLANE_CHART_STATUS_NUMERIC_FAILURE;
        }
        case SIM_PLANE_CHART_POLAR: {
            double dx = 0.0;
            double dy = 0.0;
            sim_plane_chart_centered_xy(&local_frame, x, y, t, &dx, &dy);
            out_coord->primary = hypot(dx, dy);
            out_coord->secondary =
                sim_plane_chart_wrap_value(atan2(dy, dx), local_chart.secondary_wrap);
            return (sim_plane_chart_isfinite_value(out_coord->primary) &&
                    sim_plane_chart_isfinite_value(out_coord->secondary))
                       ? SIM_PLANE_CHART_STATUS_OK
                       : SIM_PLANE_CHART_STATUS_NUMERIC_FAILURE;
        }
        case SIM_PLANE_CHART_ELLIPTIC: {
            double dx = 0.0;
            double dy = 0.0;
            double u  = 0.0;
            double v  = 0.0;
            sim_plane_chart_centered_xy(&local_frame, x, y, t, &dx, &dy);
            sim_plane_chart_rotate(dx, dy, local_chart.rotation, &u, &v);
            u /= local_chart.ellipse_u;
            v /= local_chart.ellipse_v;
            out_coord->primary = hypot(u, v);
            out_coord->secondary =
                sim_plane_chart_wrap_value(atan2(v, u), local_chart.secondary_wrap);
            return (sim_plane_chart_isfinite_value(out_coord->primary) &&
                    sim_plane_chart_isfinite_value(out_coord->secondary))
                       ? SIM_PLANE_CHART_STATUS_OK
                       : SIM_PLANE_CHART_STATUS_NUMERIC_FAILURE;
        }
        case SIM_PLANE_CHART_SPIRAL: {
            double dx = 0.0;
            double dy = 0.0;
            double r  = 0.0;
            double th = 0.0;
            sim_plane_chart_centered_xy(&local_frame, x, y, t, &dx, &dy);
            r = hypot(dx, dy);
            th = atan2(dy, dx);
            out_coord->primary = local_chart.spiral_pitch * r + local_chart.spiral_arms * th +
                                 local_chart.spiral_phase +
                                 local_chart.spiral_angular_velocity * t;
            out_coord->secondary =
                sim_plane_chart_wrap_value(th, local_chart.secondary_wrap);
            return (sim_plane_chart_isfinite_value(out_coord->primary) &&
                    sim_plane_chart_isfinite_value(out_coord->secondary))
                       ? SIM_PLANE_CHART_STATUS_OK
                       : SIM_PLANE_CHART_STATUS_NUMERIC_FAILURE;
        }
        default:
            return SIM_PLANE_CHART_STATUS_UNSUPPORTED;
    }
}

SimPlaneChartStatus sim_plane_projection_eval(const SimPlaneProjectionConfig* projection,
                                              SimPlaneChartCoord             coord,
                                              SimPlaneProjectionValue*       out_value) {
    SimPlaneProjectionConfig local = { 0 };

    if (projection == NULL || out_value == NULL ||
        !sim_plane_chart_isfinite_value(coord.primary) ||
        !sim_plane_chart_isfinite_value(coord.secondary)) {
        return SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT;
    }

    local = *projection;
    sim_plane_projection_normalize(&local);

    switch (local.kind) {
        case SIM_PLANE_PROJECTION_FULL:
            out_value->primary = coord.primary;
            out_value->secondary = coord.secondary;
            out_value->has_secondary = true;
            return SIM_PLANE_CHART_STATUS_OK;
        case SIM_PLANE_PROJECTION_SECONDARY:
            out_value->primary = coord.secondary;
            out_value->secondary = 0.0;
            out_value->has_secondary = false;
            return SIM_PLANE_CHART_STATUS_OK;
        case SIM_PLANE_PROJECTION_PRIMARY:
        default:
            out_value->primary = coord.primary;
            out_value->secondary = 0.0;
            out_value->has_secondary = false;
            return SIM_PLANE_CHART_STATUS_OK;
    }
}

SimPlaneChartStatus sim_plane_chart_eval_projected(const SimPlaneSamplingFrame* frame,
                                                   const SimPlaneChartConfig*   chart,
                                                   const SimPlaneProjectionConfig* projection,
                                                   double                       x,
                                                   double                       y,
                                                   double                       t,
                                                   SimPlaneProjectionValue*     out_value) {
    SimPlaneChartCoord   coord = { 0.0, 0.0 };
    SimPlaneChartStatus  status;

    status = sim_plane_chart_eval(frame, chart, x, y, t, &coord);
    if (status != SIM_PLANE_CHART_STATUS_OK) {
        return status;
    }
    return sim_plane_projection_eval(projection, coord, out_value);
}

SimPlaneChartStatus sim_plane_chart_from_stimulus_coord(
    const SimStimulusCoordConfig* coord,
    SimPlaneSamplingFrame*        out_frame,
    SimPlaneChartConfig*          out_chart,
    SimPlaneProjectionConfig*     out_projection) {
    SimStimulusCoordConfig local = { 0 };

    if (coord == NULL || out_frame == NULL || out_chart == NULL || out_projection == NULL) {
        return SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT;
    }

    local = *coord;
    sim_stimulus_coord_normalize(&local);

    *out_frame = (SimPlaneSamplingFrame) {
        .origin_x = local.origin_x,
        .origin_y = local.origin_y,
        .spacing_x = local.spacing_x,
        .spacing_y = local.spacing_y,
        .center_x = local.center_x,
        .center_y = local.center_y,
        .velocity_x = local.velocity_x,
        .velocity_y = local.velocity_y,
    };
    *out_chart = (SimPlaneChartConfig) {
        .kind = SIM_PLANE_CHART_CARTESIAN,
        .secondary_wrap = SIM_PLANE_CHART_WRAP_NONE,
        .rotation = 0.0,
        .ellipse_u = 1.0,
        .ellipse_v = 1.0,
        .spiral_arms = 1.0,
        .spiral_pitch = 1.0,
        .spiral_phase = 0.0,
        .spiral_angular_velocity = 0.0,
    };
    *out_projection = (SimPlaneProjectionConfig) {
        .kind = SIM_PLANE_PROJECTION_PRIMARY,
    };

    switch (local.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            out_projection->kind = (local.axis == SIM_STIMULUS_AXIS_Y)
                                       ? SIM_PLANE_PROJECTION_SECONDARY
                                       : SIM_PLANE_PROJECTION_PRIMARY;
            break;
        case SIM_STIMULUS_COORD_ANGLE:
            out_chart->kind = SIM_PLANE_CHART_CARTESIAN;
            out_chart->rotation = local.angle;
            out_projection->kind = SIM_PLANE_PROJECTION_PRIMARY;
            break;
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR:
            out_chart->kind = SIM_PLANE_CHART_POLAR;
            out_chart->secondary_wrap = SIM_PLANE_CHART_WRAP_SIGNED_ANGLE;
            out_projection->kind = SIM_PLANE_PROJECTION_PRIMARY;
            break;
        case SIM_STIMULUS_COORD_AZIMUTH:
            out_chart->kind = SIM_PLANE_CHART_POLAR;
            out_chart->secondary_wrap = SIM_PLANE_CHART_WRAP_SIGNED_ANGLE;
            out_projection->kind = SIM_PLANE_PROJECTION_SECONDARY;
            break;
        case SIM_STIMULUS_COORD_ELLIPTIC:
            out_chart->kind = SIM_PLANE_CHART_ELLIPTIC;
            out_chart->secondary_wrap = SIM_PLANE_CHART_WRAP_SIGNED_ANGLE;
            out_chart->rotation = local.angle;
            out_chart->ellipse_u = local.ellipse_u;
            out_chart->ellipse_v = local.ellipse_v;
            out_projection->kind = SIM_PLANE_PROJECTION_PRIMARY;
            break;
        case SIM_STIMULUS_COORD_SPIRAL:
            out_chart->kind = SIM_PLANE_CHART_SPIRAL;
            out_chart->secondary_wrap = SIM_PLANE_CHART_WRAP_SIGNED_ANGLE;
            out_chart->spiral_arms = local.spiral_arms;
            out_chart->spiral_pitch = local.spiral_pitch;
            out_chart->spiral_phase = local.spiral_phase;
            out_chart->spiral_angular_velocity = local.spiral_angular_velocity;
            out_projection->kind = SIM_PLANE_PROJECTION_PRIMARY;
            break;
        case SIM_STIMULUS_COORD_SEPARABLE:
            return SIM_PLANE_CHART_STATUS_UNSUPPORTED;
        default:
            return SIM_PLANE_CHART_STATUS_UNSUPPORTED;
    }

    sim_plane_sampling_frame_normalize(out_frame);
    sim_plane_chart_normalize(out_chart);
    sim_plane_projection_normalize(out_projection);
    return SIM_PLANE_CHART_STATUS_OK;
}
