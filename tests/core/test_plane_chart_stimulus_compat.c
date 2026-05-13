/*
 * Verifies that the newer plane-chart helpers preserve the legacy stimulus
 * coordinate interpretation used by existing operators.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static int approx(double a, double b, double eps) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= eps * scale;
}

static void rotate_xy(double x, double y, double angle, double *out_u, double *out_v) {
    double s = sin(angle);
    double c = cos(angle);
    *out_u = x * c + y * s;
    *out_v = -x * s + y * c;
}

static void expected_chart_coord(const SimStimulusCoordConfig *coord, double x, double y, double t,
                                 double *out_primary, double *out_secondary) {
    double sample_x = x;
    double sample_y = y;
    double dx = 0.0;
    double dy = 0.0;
    double u = 0.0;
    double v = 0.0;

    sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);

    switch (coord->mode) {
    case SIM_STIMULUS_COORD_AXIS:
        *out_primary = sample_x;
        *out_secondary = sample_y;
        return;
    case SIM_STIMULUS_COORD_ANGLE:
        rotate_xy(sample_x, sample_y, coord->angle, out_primary, out_secondary);
        return;
    case SIM_STIMULUS_COORD_RADIAL:
    case SIM_STIMULUS_COORD_POLAR:
    case SIM_STIMULUS_COORD_AZIMUTH:
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        *out_primary = hypot(dx, dy);
        *out_secondary = atan2(dy, dx);
        return;
    case SIM_STIMULUS_COORD_ELLIPTIC:
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        sim_stimulus_coord_elliptic_local(coord, dx, dy, &u, &v);
        *out_primary = hypot(u, v);
        *out_secondary = atan2(v, u);
        return;
    case SIM_STIMULUS_COORD_SPIRAL:
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        *out_secondary = atan2(dy, dx);
        *out_primary = coord->spiral_pitch * hypot(dx, dy) + coord->spiral_arms * (*out_secondary) +
                       coord->spiral_phase + coord->spiral_angular_velocity * t;
        return;
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        *out_primary = sample_x;
        *out_secondary = sample_y;
        return;
    }
}

static int run_supported_case(const char *label, SimStimulusCoordConfig coord, double x, double y,
                              double t, unsigned int expected_flags) {
    SimPlaneSamplingFrame frame = {0};
    SimPlaneChartConfig chart = {0};
    SimPlaneProjectionConfig projection = {0};
    SimPlaneChartCoord coord_value = {0.0, 0.0};
    SimPlaneProjectionValue projected = {0.0, 0.0, false};
    SimPlaneProjectionValue full_value = {0.0, 0.0, false};
    double expected_primary = 0.0;
    double expected_secondary = 0.0;
    double expected_legacy = 0.0;

    sim_stimulus_coord_normalize(&coord);
    expected_chart_coord(&coord, x, y, t, &expected_primary, &expected_secondary);
    expected_legacy = sim_stimulus_coord_u(&coord, x, y, t);

    if (sim_plane_chart_from_stimulus_coord(&coord, &frame, &chart, &projection) !=
        SIM_PLANE_CHART_STATUS_OK) {
        fprintf(stderr, "FAIL [%s]: adapter rejected supported mode\n", label);
        return 0;
    }
    if (sim_plane_chart_eval(&frame, &chart, x, y, t, &coord_value) != SIM_PLANE_CHART_STATUS_OK) {
        fprintf(stderr, "FAIL [%s]: chart eval failed\n", label);
        return 0;
    }
    if (sim_plane_chart_eval_projected(&frame, &chart, &projection, x, y, t, &projected) !=
        SIM_PLANE_CHART_STATUS_OK) {
        fprintf(stderr, "FAIL [%s]: projected eval failed\n", label);
        return 0;
    }

    if (!approx(coord_value.primary, expected_primary, 1.0e-12) ||
        !approx(coord_value.secondary, expected_secondary, 1.0e-12)) {
        fprintf(stderr,
                "FAIL [%s]: chart coord mismatch got=(%.17g, %.17g) "
                "expected=(%.17g, %.17g)\n",
                label, coord_value.primary, coord_value.secondary, expected_primary,
                expected_secondary);
        return 0;
    }
    if (!approx(projected.primary, expected_legacy, 1.0e-12)) {
        fprintf(stderr, "FAIL [%s]: projected primary mismatch got=%.17g expected=%.17g\n", label,
                projected.primary, expected_legacy);
        return 0;
    }
    if (projected.has_secondary) {
        fprintf(stderr, "FAIL [%s]: legacy projection should be scalar\n", label);
        return 0;
    }

    {
        SimPlaneProjectionConfig full_projection = {SIM_PLANE_PROJECTION_FULL};
        if (sim_plane_chart_eval_projected(&frame, &chart, &full_projection, x, y, t,
                                           &full_value) != SIM_PLANE_CHART_STATUS_OK ||
            !full_value.has_secondary || !approx(full_value.primary, expected_primary, 1.0e-12) ||
            !approx(full_value.secondary, expected_secondary, 1.0e-12)) {
            fprintf(stderr, "FAIL [%s]: full projection mismatch\n", label);
            return 0;
        }
    }

    if (sim_plane_chart_semantic_flags(&chart) != expected_flags) {
        fprintf(stderr, "FAIL [%s]: semantic flags mismatch got=0x%x expected=0x%x\n", label,
                sim_plane_chart_semantic_flags(&chart), expected_flags);
        return 0;
    }

    return 1;
}

static int test_supported_modes(void) {
    unsigned int polar_flags = SIM_PLANE_CHART_SEMANTIC_PERIODIC_SECONDARY |
                               SIM_PLANE_CHART_SEMANTIC_SINGULAR_CENTER |
                               SIM_PLANE_CHART_SEMANTIC_SECONDARY_BRANCH_CUT;

    SimStimulusCoordConfig axis_x = {0};
    axis_x.mode = SIM_STIMULUS_COORD_AXIS;
    axis_x.axis = SIM_STIMULUS_AXIS_X;
    axis_x.velocity_x = 0.3;
    axis_x.velocity_y = -0.2;

    SimStimulusCoordConfig axis_y = axis_x;
    axis_y.axis = SIM_STIMULUS_AXIS_Y;

    SimStimulusCoordConfig angle = axis_x;
    angle.mode = SIM_STIMULUS_COORD_ANGLE;
    angle.angle = 0.37;

    SimStimulusCoordConfig radial = axis_x;
    radial.mode = SIM_STIMULUS_COORD_RADIAL;
    radial.center_x = 1.25;
    radial.center_y = -0.5;

    SimStimulusCoordConfig azimuth = radial;
    azimuth.mode = SIM_STIMULUS_COORD_AZIMUTH;

    SimStimulusCoordConfig elliptic = radial;
    elliptic.mode = SIM_STIMULUS_COORD_ELLIPTIC;
    elliptic.angle = -0.41;
    elliptic.ellipse_u = 2.5;
    elliptic.ellipse_v = 0.75;

    SimStimulusCoordConfig spiral = radial;
    spiral.mode = SIM_STIMULUS_COORD_SPIRAL;
    spiral.spiral_arms = 2.25;
    spiral.spiral_pitch = -0.8;
    spiral.spiral_phase = 0.3;
    spiral.spiral_angular_velocity = 0.45;

    return run_supported_case("axis_x", axis_x, 3.4, -1.2, 0.75, SIM_PLANE_CHART_SEMANTIC_NONE) &&
           run_supported_case("axis_y", axis_y, 3.4, -1.2, 0.75, SIM_PLANE_CHART_SEMANTIC_NONE) &&
           run_supported_case("angle", angle, -0.8, 2.1, 0.5, SIM_PLANE_CHART_SEMANTIC_NONE) &&
           run_supported_case("radial", radial, 2.0, 1.25, 0.25, polar_flags) &&
           run_supported_case("azimuth", azimuth, 2.0, 1.25, 0.25, polar_flags) &&
           run_supported_case("elliptic", elliptic, -1.4, 0.9, 1.1, polar_flags) &&
           run_supported_case("spiral", spiral, -1.4, 0.9, 1.1, polar_flags);
}

static int test_unsupported_mode(void) {
    SimStimulusCoordConfig coord = {0};
    SimPlaneSamplingFrame frame = {0};
    SimPlaneChartConfig chart = {0};
    SimPlaneProjectionConfig projection = {0};
    SimPlaneChartStatus status;

    coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    coord.combine = SIM_STIMULUS_SEPARABLE_ADD;
    status = sim_plane_chart_from_stimulus_coord(&coord, &frame, &chart, &projection);
    if (status != SIM_PLANE_CHART_STATUS_UNSUPPORTED) {
        fprintf(stderr, "FAIL [separable]: expected unsupported status, got %s\n",
                sim_plane_chart_status_string(status));
        return 0;
    }
    return 1;
}

static int test_normalization_defaults(void) {
    SimPlaneSamplingFrame frame = {NAN, NAN, 0.0, NAN, NAN, NAN, NAN, NAN};
    SimPlaneChartConfig chart = {99, 99, NAN, 0.0, NAN, NAN, NAN, NAN, NAN};
    SimPlaneProjectionConfig projection = {99};

    sim_plane_sampling_frame_normalize(&frame);
    sim_plane_chart_normalize(&chart);
    sim_plane_projection_normalize(&projection);

    if (!approx(frame.origin_x, 0.0, 1.0e-12) || !approx(frame.origin_y, 0.0, 1.0e-12) ||
        !approx(frame.spacing_x, 1.0, 1.0e-12) || !approx(frame.spacing_y, 1.0, 1.0e-12) ||
        chart.kind != SIM_PLANE_CHART_CARTESIAN ||
        chart.secondary_wrap != SIM_PLANE_CHART_WRAP_NONE ||
        !approx(chart.ellipse_u, 1.0, 1.0e-12) || !approx(chart.ellipse_v, 1.0, 1.0e-12) ||
        projection.kind != SIM_PLANE_PROJECTION_PRIMARY) {
        fprintf(stderr,
                "FAIL [normalize]: defaults mismatch "
                "frame=(%.17g, %.17g, %.17g, %.17g) "
                "chart(kind=%d wrap=%d ellipse_u=%.17g ellipse_v=%.17g) "
                "projection=%d\n",
                frame.origin_x, frame.origin_y, frame.spacing_x, frame.spacing_y, (int)chart.kind,
                (int)chart.secondary_wrap, chart.ellipse_u, chart.ellipse_v, (int)projection.kind);
        return 0;
    }

    return 1;
}

static int test_status_strings(void) {
    return strcmp(sim_plane_chart_status_string(SIM_PLANE_CHART_STATUS_OK), "ok") == 0 &&
           strcmp(sim_plane_chart_status_string(SIM_PLANE_CHART_STATUS_UNSUPPORTED),
                  "unsupported") == 0;
}

int main(void) {
    if (!test_supported_modes()) {
        return 1;
    }
    if (!test_unsupported_mode()) {
        return 1;
    }
    if (!test_normalization_defaults()) {
        return 1;
    }
    if (!test_status_strings()) {
        fprintf(stderr, "FAIL [status_string]\n");
        return 1;
    }
    return 0;
}
