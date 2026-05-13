/**
 * @file zeta_plane_slice.h
 * @brief Complex-plane slice stimulus for Zeta/Xi visualization.
 *
 * Samples either zeta(s) or xi(s) over a rectangular window in the
 * complex plane and writes a scalar projection of the sampled value into
 * the target field. The window is parameterized by the center
 * `sigma_center + i t_center` and spans `sigma_span` by `t_span`.
 *
 * This operator uses the supported Zeta/Xi evaluators. Render-mode defaults,
 * sampling heuristics, and visual output may evolve as the visualization surface
 * is refined.
 */
#ifndef OAKFIELD_STIMULUS_ZETA_PLANE_SLICE_H
#define OAKFIELD_STIMULUS_ZETA_PLANE_SLICE_H

#include "oakfield/operator_split.h"
#include "oakfield/plane_chart.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Special-function family sampled by the complex-plane slice stimulus.
 */
typedef enum SimStimulusZetaPlaneSliceFamily {
    SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA = 0, /**< Sample the Riemann zeta function. */
    SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI = 1    /**< Sample the completed xi function. */
} SimStimulusZetaPlaneSliceFamily;

/**
 * @brief Scalar projection extracted from sampled Zeta/Xi values.
 */
typedef enum SimStimulusZetaPlaneSliceViewMode {
    SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_RE = 0,  /**< Real component. */
    SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_IM,      /**< Imaginary component. */
    SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ABS,     /**< Magnitude. */
    SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS, /**< Log magnitude. */
    SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ARG      /**< Phase angle. */
} SimStimulusZetaPlaneSliceViewMode;

/**
 * @brief Accuracy/performance mode for zeta-plane sampling.
 */
typedef enum SimStimulusZetaPlaneSliceRenderMode {
    SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT = 0,  /**< Use exact/high-accuracy context. */
    SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE /**< Use relaxed interactive context. */
} SimStimulusZetaPlaneSliceRenderMode;

/**
 * @brief Configuration for sampling zeta or xi over a charted complex-plane slice.
 */
typedef struct SimStimulusZetaPlaneSliceConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Scalar multiplier applied to the projected view. */
    double sigma_center;          /**< Real-part center of the sampled window. */
    double t_center;              /**< Imaginary-part center of the sampled window. */
    double sigma_span;            /**< Width of the sampled window in sigma. */
    double t_span;                /**< Height of the sampled window in t. */
    double log_floor;             /**< Positive floor used by log_abs to avoid -inf at zeros. */
    SimPlaneChartKind chart_kind; /**< Shared plane chart used to warp the sampled window. */
    SimPlaneProjectionKind sigma_projection; /**< Chart axis projected into sigma. */
    SimPlaneProjectionKind t_projection;     /**< Chart axis projected into t. */
    bool sigma_flip;       /**< When true, reflect the sigma projection across the window center. */
    bool t_flip;           /**< When true, reflect the t projection across the window center. */
    double chart_center_x; /**< Chart-space center offset in normalized sample space. */
    double chart_center_y; /**< Chart-space center offset in normalized sample space. */
    double chart_rotation; /**< Chart rotation in radians for cartesian/elliptic modes. */
    double chart_ellipse_u;                      /**< Elliptic semi-axis in local U coordinates. */
    double chart_ellipse_v;                      /**< Elliptic semi-axis in local V coordinates. */
    double chart_spiral_arms;                    /**< Spiral arm-count multiplier. */
    double chart_spiral_pitch;                   /**< Spiral radial pitch multiplier. */
    double chart_spiral_phase;                   /**< Spiral additive phase offset in radians. */
    double chart_spiral_angular_velocity;        /**< Spiral angular drift rate. */
    SimStimulusZetaPlaneSliceFamily family;      /**< Which analytic family to sample. */
    SimStimulusZetaPlaneSliceViewMode view_mode; /**< Scalar projection written into the field. */
    SimStimulusZetaPlaneSliceRenderMode render_mode; /**< Exact or interactive render behavior. */
} SimStimulusZetaPlaneSliceConfig;

/**
 * @brief Register a Zeta/Xi complex-plane slice stimulus operator.
 *
 * The implementation copies and normalizes @p config, prepares the plane chart,
 * and registers a scalar projection of the sampled analytic family.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional zeta-plane configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         runtime setup, or split-operator registration.
 */
SimResult sim_add_stimulus_zeta_plane_slice_operator(struct SimContext *context,
                                                     const SimStimulusZetaPlaneSliceConfig *config,
                                                     size_t *out_index);

/**
 * @brief Copy the current Zeta-plane slice configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_zeta_plane_slice_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_zeta_plane_slice_config(struct SimContext *context, size_t operator_index,
                                               SimStimulusZetaPlaneSliceConfig *out_config);

/**
 * @brief Replace or renormalize a registered Zeta-plane slice configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes runtime chart state and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the zeta-plane slice operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, runtime setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_zeta_plane_slice_update(struct SimContext *context, size_t operator_index,
                                               const SimStimulusZetaPlaneSliceConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_ZETA_PLANE_SLICE_H */
