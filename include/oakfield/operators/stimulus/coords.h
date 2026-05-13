/**
 * @file coords.h
 * @brief Shared spatial coordinate helpers for stimulus operators.
 */
#ifndef OAKFIELD_STIMULUS_COORDS_H
#define OAKFIELD_STIMULUS_COORDS_H

#include "oakfield/field.h"
#include "oakfield/field_patch.h"

#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STIMULUS_COORD_EPS 1.0e-12

/**
 * @brief Spatial coordinate mappings shared by stimulus operators.
 */
typedef enum SimStimulusCoordMode {
    SIM_STIMULUS_COORD_AXIS = 0,  /**< Single Cartesian axis coordinate. */
    SIM_STIMULUS_COORD_ANGLE,     /**< Rotated linear coordinate. */
    SIM_STIMULUS_COORD_RADIAL,    /**< Radial distance from center. */
    SIM_STIMULUS_COORD_POLAR,     /**< Polar coordinate pair collapsed by projection. */
    SIM_STIMULUS_COORD_AZIMUTH,   /**< Azimuthal angle around center. */
    SIM_STIMULUS_COORD_ELLIPTIC,  /**< Elliptic radial coordinate. */
    SIM_STIMULUS_COORD_SEPARABLE, /**< Separable X/Y coordinate combination. */
    SIM_STIMULUS_COORD_SPIRAL     /**< Spiral phase coordinate. */
} SimStimulusCoordMode;

/**
 * @brief Cartesian axis selector for stimulus coordinates.
 */
typedef enum SimStimulusCoordAxis {
    SIM_STIMULUS_AXIS_X = 0, /**< X axis. */
    SIM_STIMULUS_AXIS_Y = 1  /**< Y axis. */
} SimStimulusCoordAxis;

/**
 * @brief Combination mode for separable X/Y stimulus coordinates.
 */
typedef enum SimStimulusSeparableMode {
    SIM_STIMULUS_SEPARABLE_MULTIPLY = 0, /**< Multiply separable components. */
    SIM_STIMULUS_SEPARABLE_ADD = 1       /**< Add separable components. */
} SimStimulusSeparableMode;

/**
 * @brief Shared coordinate mapping configuration used by stimulus operators.
 */
typedef struct SimStimulusCoordConfig {
    SimStimulusCoordMode mode;        /**< Coordinate mapping mode. */
    SimStimulusCoordAxis axis;        /**< Axis used for axis mode. */
    SimStimulusSeparableMode combine; /**< Separable combine rule. */
    double angle;                     /**< Angle for angled mode (radians). */
    double origin_x;                  /**< Spatial origin X (units). */
    double origin_y;                  /**< Spatial origin Y (units). */
    double spacing_x;                 /**< Spacing in X (units). */
    double spacing_y;                 /**< Spacing in Y (units). */
    double center_x;                  /**< Radial center X (units). */
    double center_y;                  /**< Radial center Y (units). */
    double velocity_x;                /**< Sampling-frame drift velocity X (units/s). */
    double velocity_y;                /**< Sampling-frame drift velocity Y (units/s). */
    double ellipse_u;                 /**< Elliptic semi-axis in the local U frame. */
    double ellipse_v;                 /**< Elliptic semi-axis in the local V frame. */
    double spiral_arms;               /**< Angular arm count multiplier for spiral mode. */
    double spiral_pitch;              /**< Radial pitch multiplier for spiral mode. */
    double spiral_phase;              /**< Additive phase offset for spiral mode (radians). */
    double spiral_angular_velocity;   /**< Angular drift rate for spiral mode. */
} SimStimulusCoordConfig;

/**
 * @brief Precomputed coordinate row state for efficient stimulus patch iteration.
 */
typedef struct SimStimulusCoordRow {
    size_t x0;            /**< Starting x index for the row segment. */
    size_t y0;            /**< Starting y index for the row segment. */
    size_t width;         /**< Number of samples in the row segment. */
    double x;             /**< Physical x coordinate at the first sample. */
    double y;             /**< Physical y coordinate at the first sample. */
    double x_step;        /**< Physical x increment between samples. */
    double y_step;        /**< Physical y increment between samples. */
    double sample_x;      /**< Drift-adjusted x sample coordinate at the first sample. */
    double sample_y;      /**< Drift-adjusted y sample coordinate at the first sample. */
    double sample_x_step; /**< Drift-adjusted x increment between samples. */
    double sample_y_step; /**< Drift-adjusted y increment between samples. */
} SimStimulusCoordRow;

/**
 * @brief Apply the stimulus-frame velocity offset to a physical coordinate.
 *
 * When @p coord is NULL the input coordinate is returned unchanged. NULL output
 * pointers are ignored, which lets callers request only the side effects they need.
 *
 * @param coord Optional coordinate configuration supplying velocity_x/y.
 * @param x Physical X coordinate in field units.
 * @param y Physical Y coordinate in field units.
 * @param t Evaluation time in seconds.
 * @param[out] out_x Receives the drifted X sample coordinate.
 * @param[out] out_y Receives the drifted Y sample coordinate.
 */
static inline void sim_stimulus_coord_sample_xy(const SimStimulusCoordConfig *coord, double x,
                                                double y, double t, double *out_x, double *out_y) {
    if (out_x == NULL || out_y == NULL) {
        return;
    }

    double sample_x = x;
    double sample_y = y;
    if (coord != NULL) {
        sample_x -= coord->velocity_x * t;
        sample_y -= coord->velocity_y * t;
    }

    *out_x = sample_x;
    *out_y = sample_y;
}

/**
 * @brief Convert a physical coordinate to a center-relative stimulus coordinate.
 *
 * The coordinate is first drifted by sim_stimulus_coord_sample_xy(), then the
 * configured center is subtracted. A NULL @p coord uses a zero center and no drift.
 *
 * @param coord Optional coordinate configuration.
 * @param x Physical X coordinate in field units.
 * @param y Physical Y coordinate in field units.
 * @param t Evaluation time in seconds.
 * @param[out] out_dx Receives sample_x - center_x.
 * @param[out] out_dy Receives sample_y - center_y.
 */
static inline void sim_stimulus_coord_centered_xy(const SimStimulusCoordConfig *coord, double x,
                                                  double y, double t, double *out_dx,
                                                  double *out_dy) {
    if (out_dx == NULL || out_dy == NULL) {
        return;
    }

    double sample_x = x;
    double sample_y = y;
    double cx = 0.0;
    double cy = 0.0;
    if (coord != NULL) {
        sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);
        cx = coord->center_x;
        cy = coord->center_y;
    }

    *out_dx = sample_x - cx;
    *out_dy = sample_y - cy;
}

/**
 * @brief Rotate a coordinate into the stimulus local frame.
 *
 * The convention is u = x*cos(angle) + y*sin(angle) and
 * v = -x*sin(angle) + y*cos(angle), matching the local-frame projections used by
 * angle, elliptic, and beam-like stimuli.
 *
 * @param x Input X coordinate or center-relative delta.
 * @param y Input Y coordinate or center-relative delta.
 * @param angle Rotation angle in radians.
 * @param[out] out_u Receives the rotated local U coordinate.
 * @param[out] out_v Receives the rotated local V coordinate.
 */
static inline void sim_stimulus_coord_rotate_xy(double x, double y, double angle, double *out_u,
                                                double *out_v) {
    if (out_u == NULL || out_v == NULL) {
        return;
    }

    double s = sin(angle);
    double c = cos(angle);
    *out_u = x * c + y * s;
    *out_v = -x * s + y * c;
}

/**
 * @brief Convert center-relative deltas into the normalized elliptic local frame.
 *
 * The deltas are rotated by coord->angle and divided by ellipse_u/ellipse_v.
 * Missing or near-zero axes fall back to 1.0 so callers do not divide by zero.
 *
 * @param coord Optional coordinate configuration supplying angle and ellipse axes.
 * @param dx Center-relative X delta.
 * @param dy Center-relative Y delta.
 * @param[out] out_u Receives the normalized local U coordinate.
 * @param[out] out_v Receives the normalized local V coordinate.
 */
static inline void sim_stimulus_coord_elliptic_local(const SimStimulusCoordConfig *coord, double dx,
                                                     double dy, double *out_u, double *out_v) {
    if (out_u == NULL || out_v == NULL) {
        return;
    }

    double ur = dx;
    double vr = dy;
    double au = 1.0;
    double av = 1.0;

    if (coord != NULL) {
        sim_stimulus_coord_rotate_xy(dx, dy, coord->angle, &ur, &vr);
        au = coord->ellipse_u;
        av = coord->ellipse_v;
    }

    if (fabs(au) <= STIMULUS_COORD_EPS) {
        au = 1.0;
    }
    if (fabs(av) <= STIMULUS_COORD_EPS) {
        av = 1.0;
    }

    *out_u = ur / au;
    *out_v = vr / av;
}

/**
 * @brief Evaluate radius and angle in the normalized elliptic local frame.
 *
 * The radius is hypot(u, v) and the angle is atan2(v, u) after
 * sim_stimulus_coord_elliptic_local() applies rotation and axis scaling.
 *
 * @param coord Optional coordinate configuration.
 * @param dx Center-relative X delta.
 * @param dy Center-relative Y delta.
 * @param[out] out_r Optional destination for the elliptic radius.
 * @param[out] out_theta Optional destination for the elliptic angle in radians.
 */
static inline void sim_stimulus_coord_elliptic_polar(const SimStimulusCoordConfig *coord, double dx,
                                                     double dy, double *out_r, double *out_theta) {
    double u = 0.0;
    double v = 0.0;

    sim_stimulus_coord_elliptic_local(coord, dx, dy, &u, &v);

    if (out_r != NULL) {
        *out_r = hypot(u, v);
    }
    if (out_theta != NULL) {
        *out_theta = atan2(v, u);
    }
}

/**
 * @brief Evaluate polar radius and angle around the configured moving center.
 *
 * The input coordinate is drifted by velocity_x/y at time @p t before subtracting
 * center_x/y. Radius is in field units and theta is returned in radians.
 *
 * @param coord Optional coordinate configuration.
 * @param x Physical X coordinate in field units.
 * @param y Physical Y coordinate in field units.
 * @param t Evaluation time in seconds.
 * @param[out] out_r Optional destination for hypot(dx, dy).
 * @param[out] out_theta Optional destination for atan2(dy, dx).
 */
static inline void sim_stimulus_coord_polar(const SimStimulusCoordConfig *coord, double x, double y,
                                            double t, double *out_r, double *out_theta) {
    double dx = 0.0;
    double dy = 0.0;

    sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);

    if (out_r != NULL) {
        *out_r = hypot(dx, dy);
    }
    if (out_theta != NULL) {
        *out_theta = atan2(dy, dx);
    }
}

/**
 * @brief Sanitize a coordinate configuration in place.
 *
 * Invalid enum values are reset to defaults, non-finite scalars are replaced by
 * stable defaults, spacing and ellipse axes are made finite and nonzero, and
 * degenerate spiral settings are repaired.
 *
 * @param coord Coordinate configuration to normalize; NULL is ignored.
 */
static inline void sim_stimulus_coord_normalize(SimStimulusCoordConfig *coord) {
    if (coord == NULL) {
        return;
    }
    if (coord->mode < SIM_STIMULUS_COORD_AXIS || coord->mode > SIM_STIMULUS_COORD_SPIRAL) {
        coord->mode = SIM_STIMULUS_COORD_AXIS;
    }
    if (coord->axis != SIM_STIMULUS_AXIS_X && coord->axis != SIM_STIMULUS_AXIS_Y) {
        coord->axis = SIM_STIMULUS_AXIS_X;
    }
    if (coord->combine != SIM_STIMULUS_SEPARABLE_MULTIPLY &&
        coord->combine != SIM_STIMULUS_SEPARABLE_ADD) {
        coord->combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;
    }
    if (!isfinite(coord->angle)) {
        coord->angle = 0.0;
    }
    if (!isfinite(coord->origin_x)) {
        coord->origin_x = 0.0;
    }
    if (!isfinite(coord->origin_y)) {
        coord->origin_y = 0.0;
    }
    if (!isfinite(coord->spacing_x) || fabs(coord->spacing_x) <= STIMULUS_COORD_EPS) {
        coord->spacing_x = 1.0;
    }
    if (!isfinite(coord->spacing_y) || fabs(coord->spacing_y) <= STIMULUS_COORD_EPS) {
        coord->spacing_y = coord->spacing_x;
    }
    if (!isfinite(coord->center_x)) {
        coord->center_x = 0.0;
    }
    if (!isfinite(coord->center_y)) {
        coord->center_y = 0.0;
    }
    if (!isfinite(coord->velocity_x)) {
        coord->velocity_x = 0.0;
    }
    if (!isfinite(coord->velocity_y)) {
        coord->velocity_y = 0.0;
    }
    if (!isfinite(coord->ellipse_u) || fabs(coord->ellipse_u) <= STIMULUS_COORD_EPS) {
        coord->ellipse_u = 1.0;
    }
    coord->ellipse_u = fabs(coord->ellipse_u);
    if (!isfinite(coord->ellipse_v) || fabs(coord->ellipse_v) <= STIMULUS_COORD_EPS) {
        coord->ellipse_v = coord->ellipse_u;
    }
    coord->ellipse_v = fabs(coord->ellipse_v);
    if (!isfinite(coord->spiral_arms)) {
        coord->spiral_arms = 1.0;
    }
    if (!isfinite(coord->spiral_pitch)) {
        coord->spiral_pitch = 1.0;
    }
    if (!isfinite(coord->spiral_phase)) {
        coord->spiral_phase = 0.0;
    }
    if (!isfinite(coord->spiral_angular_velocity)) {
        coord->spiral_angular_velocity = 0.0;
    }
    if (coord->mode == SIM_STIMULUS_COORD_SPIRAL &&
        fabs(coord->spiral_arms) <= STIMULUS_COORD_EPS &&
        fabs(coord->spiral_pitch) <= STIMULUS_COORD_EPS) {
        coord->spiral_arms = 1.0;
        coord->spiral_pitch = 1.0;
    }
}

/**
 * @brief Report whether a coordinate mapping is independent of evaluation time.
 *
 * A NULL coordinate is treated as invariant. Nonzero frame velocity or spiral
 * angular velocity makes the mapping time dependent.
 *
 * @param coord Optional coordinate configuration to inspect.
 * @return true when samples are independent of time; false otherwise.
 */
static inline bool sim_stimulus_coord_is_time_invariant(const SimStimulusCoordConfig *coord) {
    if (coord == NULL) {
        return true;
    }
    if (fabs(coord->velocity_x) > STIMULUS_COORD_EPS ||
        fabs(coord->velocity_y) > STIMULUS_COORD_EPS) {
        return false;
    }
    if (coord->mode == SIM_STIMULUS_COORD_SPIRAL &&
        fabs(coord->spiral_angular_velocity) > STIMULUS_COORD_EPS) {
        return false;
    }
    return true;
}

/**
 * @brief Convert integer field indices to physical stimulus coordinates.
 *
 * Coordinates are computed as origin + index * spacing along each axis. The
 * helper does not apply velocity, center, or mode-specific projection.
 *
 * @param coord Coordinate configuration supplying origin and spacing.
 * @param ix X lattice index.
 * @param iy Y lattice index.
 * @param[out] out_x Receives the physical X coordinate.
 * @param[out] out_y Receives the physical Y coordinate.
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_INVALID_ARGUMENT for NULL pointers.
 */
static inline SimResult sim_stimulus_coord_xy_at_indices(const SimStimulusCoordConfig *coord,
                                                         size_t ix, size_t iy, double *out_x,
                                                         double *out_y) {
    if (coord == NULL || out_x == NULL || out_y == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_x = coord->origin_x + (double)ix * coord->spacing_x;
    *out_y = coord->origin_y + (double)iy * coord->spacing_y;
    return SIM_RESULT_OK;
}

/**
 * @brief Build per-row coordinate increments for a field patch row.
 *
 * The returned row describes both physical x/y coordinates and velocity-adjusted
 * sample_x/sample_y coordinates for contiguous row evaluation.
 *
 * @param coord Coordinate configuration supplying origin, spacing, and velocity.
 * @param patch Valid field patch being traversed.
 * @param row_offset Row offset inside @p patch.
 * @param t Evaluation time in seconds.
 * @param[out] out_row Receives row extents, coordinates, and sample increments.
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_INVALID_ARGUMENT for invalid inputs.
 */
static inline SimResult sim_stimulus_coord_patch_row(const SimStimulusCoordConfig *coord,
                                                     const SimFieldPatch *patch, size_t row_offset,
                                                     double t, SimStimulusCoordRow *out_row) {
    SimStimulusCoordRow row = {0};

    if (coord == NULL || !sim_field_patch_is_valid(patch) || out_row == NULL ||
        row_offset >= patch->height) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    row.x0 = patch->x0;
    row.y0 = patch->y0 + row_offset;
    row.width = patch->width;
    row.x_step = coord->spacing_x;
    row.y_step = 0.0;
    row.sample_x_step = row.x_step;
    row.sample_y_step = row.y_step;

    if (sim_stimulus_coord_xy_at_indices(coord, row.x0, row.y0, &row.x, &row.y) != SIM_RESULT_OK) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    sim_stimulus_coord_sample_xy(coord, row.x, row.y, t, &row.sample_x, &row.sample_y);
    *out_row = row;
    return SIM_RESULT_OK;
}

/**
 * @brief Convert a field element index to physical stimulus coordinates.
 *
 * The field layout supplies the integer x/y location and @p coord supplies
 * origin and spacing. The helper does not apply velocity or projection.
 *
 * @param coord Coordinate configuration supplying origin and spacing.
 * @param field Field whose layout maps @p index to x/y indices.
 * @param index Linear element index in @p field.
 * @param[out] out_x Receives the physical X coordinate.
 * @param[out] out_y Receives the physical Y coordinate.
 * @return #SIM_RESULT_OK on success, a field-layout error, or
 *         #SIM_RESULT_INVALID_ARGUMENT for NULL pointers.
 */
static inline SimResult sim_stimulus_coord_xy(const SimStimulusCoordConfig *coord,
                                              const SimField *field, size_t index, double *out_x,
                                              double *out_y) {
    size_t ix = 0U;
    size_t iy = 0U;
    SimResult rc = SIM_RESULT_OK;

    if (coord == NULL || field == NULL || out_x == NULL || out_y == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rc = sim_field_index_to_xy(field, index, &ix, &iy);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    return sim_stimulus_coord_xy_at_indices(coord, ix, iy, out_x, out_y);
}

/**
 * @brief Evaluate the scalar coordinate used by one-dimensional stimulus phases.
 *
 * The selected mode maps (x, y, t) to an axis, angled projection, radius,
 * azimuth, elliptic radius, spiral phase coordinate, or separable X coordinate.
 * A NULL @p coord returns @p x.
 *
 * @param coord Optional coordinate configuration selecting the projection mode.
 * @param x Physical X coordinate in field units.
 * @param y Physical Y coordinate in field units.
 * @param t Evaluation time in seconds.
 * @return Scalar coordinate in field units or radians, depending on mode.
 */
static inline double sim_stimulus_coord_u(const SimStimulusCoordConfig *coord, double x, double y,
                                          double t) {
    if (coord == NULL) {
        return x;
    }
    double sample_x = x;
    double sample_y = y;
    sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);
    switch (coord->mode) {
    case SIM_STIMULUS_COORD_AXIS:
        return (coord->axis == SIM_STIMULUS_AXIS_Y) ? sample_y : sample_x;
    case SIM_STIMULUS_COORD_ANGLE: {
        double u = 0.0;
        double v = 0.0;
        sim_stimulus_coord_rotate_xy(sample_x, sample_y, coord->angle, &u, &v);
        return u;
    }
    case SIM_STIMULUS_COORD_RADIAL: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        return hypot(dx, dy);
    }
    case SIM_STIMULUS_COORD_POLAR: {
        double r = 0.0;
        sim_stimulus_coord_polar(coord, x, y, t, &r, NULL);
        return r;
    }
    case SIM_STIMULUS_COORD_AZIMUTH: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        return atan2(dy, dx);
    }
    case SIM_STIMULUS_COORD_ELLIPTIC: {
        double dx = 0.0;
        double dy = 0.0;
        double r = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        sim_stimulus_coord_elliptic_polar(coord, dx, dy, &r, NULL);
        return r;
    }
    case SIM_STIMULUS_COORD_SPIRAL: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        double r = hypot(dx, dy);
        double theta = atan2(dy, dx);
        return coord->spiral_pitch * r + coord->spiral_arms * theta + coord->spiral_phase +
               coord->spiral_angular_velocity * t;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        return sample_x;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_COORDS_H */
