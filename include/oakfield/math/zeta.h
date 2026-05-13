/**
 * @file zeta.h
 * @brief Riemann zeta evaluation helpers with branch metadata and error estimates.
 */
#ifndef OAKFIELD_MATH_ZETA_H
#define OAKFIELD_MATH_ZETA_H

#include <stddef.h>

#include "oakfield/field.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status values returned by zeta evaluators.
 */
typedef enum SimZetaStatus {
    SIM_ZETA_STATUS_OK = 0,           /**< Evaluation completed successfully. */
    SIM_ZETA_STATUS_SINGULAR,         /**< Argument lies on a singularity or pole. */
    SIM_ZETA_STATUS_INVALID_ARGUMENT, /**< Input argument or output pointer was invalid. */
    SIM_ZETA_STATUS_NO_CONVERGENCE,   /**< Series/refinement did not converge within budget. */
    SIM_ZETA_STATUS_NUMERIC_FAILURE   /**< Evaluation produced an unstable numeric result. */
} SimZetaStatus;

/**
 * @brief Rigor level attached to complex-ball enclosures.
 */
typedef enum SimBallRigor {
    SIM_BALL_RIGOR_HEURISTIC = 0, /**< Heuristic error bounds only. */
    SIM_BALL_RIGOR_VALIDATED,     /**< Bounds validated by runtime checks. */
    SIM_BALL_RIGOR_FORMAL         /**< Formally justified enclosure. */
} SimBallRigor;

/**
 * @brief Dispatcher branch used for a zeta evaluation.
 */
typedef enum SimZetaBranch {
    SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN = 0, /**< Direct Euler-Maclaurin summation branch. */
    SIM_ZETA_BRANCH_ETA_ACCELERATED,            /**< Eta-accelerated series branch. */
    SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION, /**< Approximate functional equation. */
    SIM_ZETA_BRANCH_RIEMANN_SIEGEL,                  /**< Riemann-Siegel branch. */
    SIM_ZETA_BRANCH_NEAR_ONE_LAURENT,                /**< Laurent expansion near s=1. */
    SIM_ZETA_BRANCH_LOCAL_EXPANSION,                 /**< Local expansion around a cached center. */
    SIM_ZETA_BRANCH_REFLECTION                       /**< Reflection/formula-transformed branch. */
} SimZetaBranch;

#define SIM_ZETA_FLAG_USED_REFLECTION 0x1u
#define SIM_ZETA_FLAG_NEAR_POLE 0x2u
#define SIM_ZETA_FLAG_TRIVIAL_ZERO 0x4u
#define SIM_ZETA_FLAG_USED_ETA 0x8u
#define SIM_ZETA_FLAG_USED_NEAR_ONE_LAURENT 0x10u
#define SIM_ZETA_FLAG_USED_AFE 0x20u
#define SIM_ZETA_FLAG_USED_ADAPTIVE_REFINEMENT 0x40u
#define SIM_ZETA_FLAG_USED_RIEMANN_SIEGEL 0x80u
#define SIM_ZETA_FLAG_USED_LOCAL_EXPANSION 0x100u
#define SIM_ZETA_FLAG_ZERO_PROXIMITY 0x200u

/**
 * @brief Complex ball enclosure with evaluator status, rigor, and refinement metadata.
 */
typedef struct SimComplexBall {
    SimComplexDouble center;             /**< Center of the complex enclosure. */
    double radius;                       /**< Nonnegative radius of the enclosure. */
    SimZetaStatus status;                /**< Status associated with the enclosed value. */
    unsigned int flags;                  /**< Bitmask of SIM_ZETA_FLAG_* values. */
    size_t refinement_rounds;            /**< Adaptive refinement rounds performed. */
    unsigned int working_precision_bits; /**< Effective working precision in bits. */
    SimBallRigor rigor;                  /**< Rigor level of the enclosure. */
    size_t validation_passes;            /**< Number of validation passes completed. */
} SimComplexBall;

/**
 * @brief Evaluation controls for the Riemann zeta dispatcher.
 */
typedef struct SimZetaContext {
    double abs_tol;                  /**< Absolute error tolerance. */
    double rel_tol;                  /**< Relative error tolerance. */
    size_t initial_terms;            /**< Initial series term budget. */
    size_t max_terms;                /**< Maximum series term budget. */
    size_t euler_maclaurin_terms;    /**< Euler-Maclaurin correction term count. */
    size_t eta_initial_terms;        /**< Initial eta-series term budget. */
    size_t eta_max_terms;            /**< Maximum eta-series term budget. */
    double eta_sigma_limit;          /**< Real-part threshold for eta acceleration. */
    double eta_imag_limit;           /**< Imaginary-part threshold for eta acceleration. */
    size_t afe_initial_cutoff;       /**< Initial approximate-functional-equation cutoff. */
    size_t afe_max_cutoff;           /**< Maximum approximate-functional-equation cutoff. */
    double afe_sigma_limit;          /**< Real-part threshold for AFE dispatch. */
    double afe_imag_min;             /**< Minimum imaginary magnitude for AFE dispatch. */
    double riemann_siegel_imag_min;  /**< Minimum imaginary magnitude for Riemann-Siegel. */
    double critical_line_tolerance;  /**< Tolerance for treating points as critical-line samples. */
    size_t riemann_siegel_max_terms; /**< Maximum Riemann-Siegel term budget. */
    double pole_radius;              /**< Radius treated as near the pole at s=1. */
    double near_one_radius;          /**< Radius selecting the near-one Laurent branch. */
    double local_expansion_radius;   /**< Radius selecting local expansion. */
    size_t ball_validation_rounds;   /**< Complex-ball validation pass budget. */
    double ball_validation_scale;    /**< Radius expansion factor for ball validation. */
    size_t adaptive_max_rounds;      /**< Maximum adaptive refinement rounds. */
    double adaptive_tightening_factor; /**< Tolerance tightening factor per refinement. */
} SimZetaContext;

/**
 * @brief Value, error estimates, branch metadata, and status from zeta evaluation.
 */
typedef struct SimZetaResult {
    SimComplexDouble value;              /**< Evaluated zeta value. */
    double abs_error;                    /**< Absolute error estimate. */
    double rel_error;                    /**< Relative error estimate. */
    size_t terms_used;                   /**< Main-series terms used. */
    size_t correction_terms;             /**< Correction/asymptotic terms used. */
    size_t refinement_rounds;            /**< Adaptive refinement rounds performed. */
    unsigned int working_precision_bits; /**< Effective working precision in bits. */
    SimZetaBranch branch;                /**< Dispatcher branch used. */
    SimZetaStatus status;                /**< Evaluation status. */
    unsigned int flags;                  /**< Bitmask of SIM_ZETA_FLAG_* values. */
} SimZetaResult;

/**
 * @brief Zeta value and first derivative with associated error and branch metadata.
 */
typedef struct SimZetaDerivativeResult {
    SimComplexDouble value;              /**< Evaluated zeta value. */
    SimComplexDouble derivative;         /**< Evaluated first derivative. */
    double abs_error;                    /**< Absolute error estimate for @ref value. */
    double rel_error;                    /**< Relative error estimate for @ref value. */
    double derivative_abs_error;         /**< Absolute error estimate for @ref derivative. */
    double derivative_rel_error;         /**< Relative error estimate for @ref derivative. */
    size_t refinement_rounds;            /**< Adaptive refinement rounds performed. */
    unsigned int working_precision_bits; /**< Effective working precision in bits. */
    SimZetaBranch branch;                /**< Dispatcher branch used. */
    SimZetaStatus status;                /**< Evaluation status. */
    unsigned int flags;                  /**< Bitmask of SIM_ZETA_FLAG_* values. */
} SimZetaDerivativeResult;

/**
 * @brief Human-readable description of a zeta status value.
 *
 * @param status Status returned by zeta/ball evaluators.
 * @return Static diagnostic string.
 */
const char *sim_zeta_status_string(SimZetaStatus status);

/**
 * @brief Human-readable description of a zeta branch value.
 *
 * @param branch Branch identifier from a zeta result.
 * @return Static diagnostic string.
 */
const char *sim_zeta_branch_string(SimZetaBranch branch);

/**
 * @brief Default tolerances and truncation limits for the current zeta dispatcher.
 *
 * @return Default zeta evaluation context.
 */
SimZetaContext sim_zeta_context_default(void);

/**
 * @brief Low-latency zeta context for interactive visual exploration.
 *
 * This keeps the existing dispatcher but relaxes tolerances, caps work budgets,
 * and disables validation so renderers can use `sim_zeta_eval()` for responsive
 * previews without changing the exact path.
 *
 * @return Relaxed low-latency zeta evaluation context.
 */
SimZetaContext sim_zeta_context_interactive(void);

/**
 * @brief Evaluate zeta(s) with the direct Euler-Maclaurin branch.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Structured zeta result for the direct branch.
 */
SimZetaResult sim_zeta_eval_direct_euler_maclaurin(SimComplexDouble s,
                                                   const SimZetaContext *context);

/**
 * @brief Evaluate zeta(s) with the Phase 2 accelerated eta/Hasse branch.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Structured zeta result for the eta/Hasse branch.
 */
SimZetaResult sim_zeta_eval_eta_accelerated(SimComplexDouble s, const SimZetaContext *context);

/**
 * @brief Evaluate zeta(s) with the Phase 3 approximate functional equation branch.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Structured zeta result for the approximate functional equation branch.
 */
SimZetaResult sim_zeta_eval_approximate_fe(SimComplexDouble s, const SimZetaContext *context);

/**
 * @brief Evaluate zeta(1/2 + i t) with the Phase 4 Riemann-Siegel branch.
 *
 * @param t Critical-line height.
 * @param context Optional evaluation context.
 * @return Structured zeta result on the critical line.
 */
SimZetaResult sim_zeta_eval_riemann_siegel(double t, const SimZetaContext *context);

/**
 * @brief Evaluate zeta(s) and its first complex derivative.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Structured value and derivative result.
 */
SimZetaDerivativeResult sim_zeta_eval_with_derivative(SimComplexDouble s,
                                                      const SimZetaContext *context);

/**
 * @brief Ball-style enclosure derived from the current zeta error model.
 *
 * The returned rigor metadata distinguishes heuristic, validated, and future
 * formal interval results. The current implementation provides heuristic and
 * cross-checked validated balls, plus formal enclosures for exact special
 * values and direct-Euler-Maclaurin right-half-plane paths, including a
 * genuinely complex family with Re(s) > 1.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Complex ball enclosure with status and rigor metadata.
 */
SimComplexBall sim_zeta_eval_ball(SimComplexDouble s, const SimZetaContext *context);

/**
 * @brief Evaluate zeta(s) using the Phase 3 dispatcher and adaptive refinement layer.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Structured zeta result from the adaptive dispatcher.
 */
SimZetaResult sim_zeta_eval(SimComplexDouble s, const SimZetaContext *context);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_ZETA_H */
