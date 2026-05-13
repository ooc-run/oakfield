/**
 * @file plane_chart.h
 * @brief Shared plane-chart helpers for chart-aware 2D field parameterization.
 *
 * This Phase 1 slice is compatibility-first: it defines a reusable sampling
 * frame, chart, and projection surface, plus an adapter from the existing
 * `SimStimulusCoordConfig` for the supported non-separable modes.
 */
#ifndef OAKFIELD_PLANE_CHART_H
#define OAKFIELD_PLANE_CHART_H

#include "operators/stimulus/coords.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIM_PLANE_CHART_EPS 1.0e-12

/**
 * @brief Coordinate chart families available for 2D plane parameterization.
 */
typedef enum SimPlaneChartKind {
    SIM_PLANE_CHART_CARTESIAN = 0, /**< Cartesian x/y chart. */
    SIM_PLANE_CHART_POLAR,         /**< Polar radius/angle chart. */
    SIM_PLANE_CHART_ELLIPTIC,      /**< Elliptic radius/angle-like chart. */
    SIM_PLANE_CHART_SPIRAL         /**< Spiral phase/radius chart. */
} SimPlaneChartKind;

/**
 * @brief Projection modes used to reduce chart coordinates to output values.
 */
typedef enum SimPlaneProjectionKind {
    SIM_PLANE_PROJECTION_FULL = 0, /**< Preserve both primary and secondary coordinates. */
    SIM_PLANE_PROJECTION_PRIMARY,  /**< Emit only the primary coordinate. */
    SIM_PLANE_PROJECTION_SECONDARY /**< Emit only the secondary coordinate. */
} SimPlaneProjectionKind;

/**
 * @brief Wrapping policies for secondary chart coordinates.
 */
typedef enum SimPlaneChartWrapPolicy {
    SIM_PLANE_CHART_WRAP_NONE = 0,       /**< Leave the secondary coordinate unwrapped. */
    SIM_PLANE_CHART_WRAP_SIGNED_ANGLE,   /**< Wrap to a signed angular interval. */
    SIM_PLANE_CHART_WRAP_UNSIGNED_ANGLE, /**< Wrap to an unsigned angular interval. */
    SIM_PLANE_CHART_WRAP_UNIT_INTERVAL   /**< Wrap to the unit interval [0, 1). */
} SimPlaneChartWrapPolicy;

/**
 * @brief Status values returned by plane-chart helpers.
 */
typedef enum SimPlaneChartStatus {
    SIM_PLANE_CHART_STATUS_OK = 0,           /**< Chart operation completed successfully. */
    SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT, /**< NULL or invalid input was supplied. */
    SIM_PLANE_CHART_STATUS_UNSUPPORTED,      /**< Requested chart/projection is unsupported. */
    SIM_PLANE_CHART_STATUS_OUT_OF_DOMAIN,    /**< Coordinates lie outside the chart domain. */
    SIM_PLANE_CHART_STATUS_SINGULAR,         /**< Chart evaluation hit a singular point. */
    SIM_PLANE_CHART_STATUS_NON_INVERTIBLE,   /**< Requested transform is not invertible. */
    SIM_PLANE_CHART_STATUS_NUMERIC_FAILURE   /**< Non-finite or unstable numeric result. */
} SimPlaneChartStatus;

enum {
    SIM_PLANE_CHART_SEMANTIC_NONE = 0u,
    SIM_PLANE_CHART_SEMANTIC_PERIODIC_SECONDARY = 1u << 0,
    SIM_PLANE_CHART_SEMANTIC_SINGULAR_CENTER = 1u << 1,
    SIM_PLANE_CHART_SEMANTIC_SECONDARY_BRANCH_CUT = 1u << 2
};

enum { SIM_PLANE_CHART_CAPABILITY_NONE = 0u };

/**
 * @brief Sampling frame that maps field sample coordinates into chart input space.
 */
typedef struct SimPlaneSamplingFrame {
    double origin_x;   /**< World x coordinate at sample index zero. */
    double origin_y;   /**< World y coordinate at sample index zero. */
    double spacing_x;  /**< World-space spacing per x sample. */
    double spacing_y;  /**< World-space spacing per y sample. */
    double center_x;   /**< Chart center x coordinate at t=0. */
    double center_y;   /**< Chart center y coordinate at t=0. */
    double velocity_x; /**< Center drift velocity along x. */
    double velocity_y; /**< Center drift velocity along y. */
} SimPlaneSamplingFrame;

/**
 * @brief Parameters selecting and shaping a reusable 2D plane chart.
 */
typedef struct SimPlaneChartConfig {
    SimPlaneChartKind kind;                 /**< Chart family to evaluate. */
    SimPlaneChartWrapPolicy secondary_wrap; /**< Secondary-coordinate wrapping policy. */
    double rotation;                        /**< Chart rotation angle in radians. */
    double ellipse_u;                       /**< Elliptic chart scale along local u. */
    double ellipse_v;                       /**< Elliptic chart scale along local v. */
    double spiral_arms;                     /**< Number of spiral arms. */
    double spiral_pitch;                    /**< Radial pitch used by spiral charts. */
    double spiral_phase;                    /**< Static phase offset for spiral charts. */
    double spiral_angular_velocity;         /**< Spiral phase drift rate in radians/second. */
} SimPlaneChartConfig;

/**
 * @brief Projection policy for reducing chart coordinates to stimulus values.
 */
typedef struct SimPlaneProjectionConfig {
    SimPlaneProjectionKind kind; /**< Projection mode applied after chart evaluation. */
} SimPlaneProjectionConfig;

/**
 * @brief Pair of primary and secondary coordinates produced by a plane chart.
 */
typedef struct SimPlaneChartCoord {
    double primary;   /**< Primary chart coordinate. */
    double secondary; /**< Secondary chart coordinate. */
} SimPlaneChartCoord;

/**
 * @brief Projected chart value with an optional secondary coordinate.
 */
typedef struct SimPlaneProjectionValue {
    double primary;     /**< Projected primary value. */
    double secondary;   /**< Projected secondary value when @ref has_secondary is true. */
    bool has_secondary; /**< True when @ref secondary contains a meaningful value. */
} SimPlaneProjectionValue;

/**
 * @brief Human-readable description of a plane-chart status.
 *
 * @param status Status enum value.
 * @return Static machine-readable string such as "ok" or "numeric-failure".
 */
const char *sim_plane_chart_status_string(SimPlaneChartStatus status);

/**
 * @brief Normalize a sampling frame in-place.
 *
 * Non-finite origins, centers, and velocities are set to zero. Near-zero or
 * non-finite spacing falls back to finite positive defaults.
 *
 * @param frame Frame to normalize; NULL is ignored.
 */
void sim_plane_sampling_frame_normalize(SimPlaneSamplingFrame *frame);

/**
 * @brief Normalize a plane-chart config in-place.
 *
 * Invalid enum values and non-finite chart parameters are replaced with
 * cartesian/signed-angle-compatible defaults.
 *
 * @param chart Chart config to normalize; NULL is ignored.
 */
void sim_plane_chart_normalize(SimPlaneChartConfig *chart);

/**
 * @brief Normalize a projection config in-place.
 *
 * @param projection Projection config to normalize; NULL is ignored.
 */
void sim_plane_projection_normalize(SimPlaneProjectionConfig *projection);

/**
 * @brief Semantic metadata flags for a chart configuration.
 *
 * @param chart Chart config to inspect; NULL is treated as no semantics.
 * @return Bitmask of SIM_PLANE_CHART_SEMANTIC_* values.
 */
unsigned int sim_plane_chart_semantic_flags(const SimPlaneChartConfig *chart);

/**
 * @brief Numerical capability flags currently exposed for a chart configuration.
 *
 * @param chart Chart config to inspect.
 * @return Bitmask of SIM_PLANE_CHART_CAPABILITY_* values.
 */
unsigned int sim_plane_chart_capability_flags(const SimPlaneChartConfig *chart);

/**
 * @brief Evaluate the chart coordinates at the provided sample-space point.
 *
 * @param frame Sampling frame that maps sample indices to world coordinates.
 * @param chart Chart transform to evaluate.
 * @param x Sample-space x coordinate.
 * @param y Sample-space y coordinate.
 * @param t Simulation time used for frame velocity and spiral angular velocity.
 * @param[out] out_coord Receives primary/secondary chart coordinates.
 * @return Chart status indicating success, invalid input, unsupported chart, or numeric failure.
 */
SimPlaneChartStatus sim_plane_chart_eval(const SimPlaneSamplingFrame *frame,
                                         const SimPlaneChartConfig *chart, double x, double y,
                                         double t, SimPlaneChartCoord *out_coord);

/**
 * @brief Project chart coordinates into a value surface.
 *
 * @param projection Projection policy.
 * @param coord Chart coordinates to project; values must be finite.
 * @param[out] out_value Receives primary value and optional secondary value.
 * @return `SIM_PLANE_CHART_STATUS_OK` or `SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT`.
 */
SimPlaneChartStatus sim_plane_projection_eval(const SimPlaneProjectionConfig *projection,
                                              SimPlaneChartCoord coord,
                                              SimPlaneProjectionValue *out_value);

/**
 * @brief Convenience wrapper for chart evaluation followed by projection.
 *
 * @param frame Sampling frame that maps sample indices to world coordinates.
 * @param chart Chart transform to evaluate.
 * @param projection Projection policy applied after chart evaluation.
 * @param x Sample-space x coordinate.
 * @param y Sample-space y coordinate.
 * @param t Simulation time used by moving charts.
 * @param[out] out_value Receives the projected value.
 * @return First non-OK status from chart or projection evaluation.
 */
SimPlaneChartStatus sim_plane_chart_eval_projected(const SimPlaneSamplingFrame *frame,
                                                   const SimPlaneChartConfig *chart,
                                                   const SimPlaneProjectionConfig *projection,
                                                   double x, double y, double t,
                                                   SimPlaneProjectionValue *out_value);

/**
 * @brief Build a frame/chart/projection triple from a legacy stimulus coord config.
 *
 * Returns `SIM_PLANE_CHART_STATUS_UNSUPPORTED` for structural modes that do not
 * yet have a shared chart representation, such as `SEPARABLE`.
 *
 * @param coord Legacy stimulus coordinate config.
 * @param[out] out_frame Receives the normalized sampling frame.
 * @param[out] out_chart Receives the normalized chart config.
 * @param[out] out_projection Receives the normalized projection config.
 * @return `SIM_PLANE_CHART_STATUS_OK`, `SIM_PLANE_CHART_STATUS_INVALID_ARGUMENT`,
 *         or `SIM_PLANE_CHART_STATUS_UNSUPPORTED`.
 */
SimPlaneChartStatus sim_plane_chart_from_stimulus_coord(const SimStimulusCoordConfig *coord,
                                                        SimPlaneSamplingFrame *out_frame,
                                                        SimPlaneChartConfig *out_chart,
                                                        SimPlaneProjectionConfig *out_projection);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_PLANE_CHART_H */
