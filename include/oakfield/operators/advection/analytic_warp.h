/**
 * @file analytic_warp.h
 * @brief Analytic warp operator (smooth nonlinear deformation) acting on a single field.
 *
 * Applies profile-specific analytic transforms (digamma, trigamma, power, tanh) by
 * sampling gradients and accumulating a response term scaled by lambda and delta.
 *
 * Complex field support: component-wise (Re/Im) or polar mode (magnitude gradient along phase),
 * selectable via config.
 */
#ifndef OAKFIELD_ANALYTIC_WARP_H
#define OAKFIELD_ANALYTIC_WARP_H

#include <stddef.h>

#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Analytic warp profile enumeration.
 */
typedef enum AnalyticWarpProfile {
    ANALYTIC_WARP_PROFILE_DIGAMMA = 0,  /**< Digamma warp (psi). */
    ANALYTIC_WARP_PROFILE_TRIGAMMA = 1, /**< Trigamma warp (psi_1). */
    ANALYTIC_WARP_PROFILE_POWER = 2,    /**< Power-law warp (|x|^p with sign preservation). */
    ANALYTIC_WARP_PROFILE_TANH = 3,     /**< Hyperbolic tangent warp. */
    ANALYTIC_WARP_PROFILE_HYPEREXP = 4, /**< Hyperexponential warp using phi(lambda, epsilon; K). */
    ANALYTIC_WARP_PROFILE_QHYPEREXP =
        5 /**< q-deformed hyperexponential warp using phi_q(lambda, epsilon; K, q). */
} AnalyticWarpProfile;

/**
 * @brief Complex processing mode for analytic warp.
 */
typedef enum AnalyticWarpComplexMode {
    ANALYTIC_WARP_COMPLEX_MODE_COMPONENT =
        0, /**< Process real and imaginary parts independently. */
    ANALYTIC_WARP_COMPLEX_MODE_POLAR =
        1 /**< Compute gradient at magnitude, apply along current phase. */
} AnalyticWarpComplexMode;

/**
 * @brief Configuration for the analytic warp operator.
 */
typedef struct AnalyticWarpOperatorConfig {
    size_t field_index;          /**< Target field index. */
    AnalyticWarpProfile profile; /**< Warp profile selector. */
    double delta;                /**< Symmetric evaluation offset for gradient estimation. */
    double lambda;               /**< Scaling applied to the warp response. */
    double bias;                 /**< Additive bias before evaluating profile. */
    double exponent;             /**< Exponent used by POWER profile (ignored otherwise). */
    AnalyticWarpComplexMode complex_mode; /**< Complex processing mode (component-wise or polar). */
    double hyperexp_epsilon;      /**< Hyperexponential epsilon offset (profile-specific). */
    int hyperexp_depth;           /**< Hyperexponential truncation depth K (>0). */
    double hyperexp_q;            /**< q parameter for q-hyperexponential profile. */
    SimContinuityMode continuity; /**< Continuity guard policy applied during evaluation. */
    double continuity_clamp_min;  /**< Lower clamp bound when continuity requires clamping. */
    double continuity_clamp_max;  /**< Upper clamp bound when continuity requires clamping. */
    double continuity_tolerance;  /**< Offset tolerance controlling limited continuity blending. */
} AnalyticWarpOperatorConfig;

/**
 * @brief Register an analytic warp operator on the provided context.
 *
 * The implementation copies and normalizes @p config, validates the target field,
 * and registers either a split operator or an eligible kernel-backed operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional operator configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, or operator registration.
 */
SimResult sim_add_analytic_warp_operator(struct SimContext *context,
                                         const AnalyticWarpOperatorConfig *config,
                                         size_t *out_index);

/**
 * @brief Retrieve the current configuration for an analytic warp operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Registry index of the target operator.
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no warp state.
 */
SimResult sim_analytic_warp_config(struct SimContext *context, size_t operator_index,
                                   AnalyticWarpOperatorConfig *out_config);

/**
 * @brief Update an existing analytic warp operator with a new configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state, operator metadata,
 * and any registered kernel constants, then invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Registry index of the operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         validation, state validation, or registration metadata refresh.
 */
SimResult sim_analytic_warp_update(struct SimContext *context, size_t operator_index,
                                   const AnalyticWarpOperatorConfig *config);

/**
 * @brief Compute the conservative warp-level classification for a config.
 *
 * This refines the static schema classification using profile-specific
 * parameters to inform scheduling and dt heuristics without mutating IR. NULL or
 * invalid configs fall back to the conservative high-risk level.
 *
 * @param config Optional analytic-warp configuration to classify.
 * @return Effective warp-level classification for scheduling metadata.
 */
SimWarpLevel sim_analytic_warp_effective_level(const AnalyticWarpOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_ANALYTIC_WARP_H */
