/**
 * @file xi.h
 * @brief Completed Riemann xi evaluation helpers.
 */
#ifndef OAKFIELD_MATH_XI_H
#define OAKFIELD_MATH_XI_H

#include <stddef.h>

#include "zeta.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SIM_XI_FLAG_USED_LOG_ASSEMBLY 0x1u
#define SIM_XI_FLAG_USED_REFLECTION 0x2u
#define SIM_XI_FLAG_EXACT_LIMIT 0x4u
#define SIM_XI_FLAG_USED_NEAR_ONE_EXPANSION 0x8u
#define SIM_XI_FLAG_USED_LOCAL_EXPANSION 0x10u
#define SIM_XI_FLAG_USED_ZERO_REFINEMENT 0x20u
#define SIM_XI_FLAG_ZERO_BRACKET_VALIDATED 0x40u
#define SIM_XI_FLAG_ZERO_BRACKET_FORMAL 0x80u

/**
 * @brief Evaluation controls for completed Riemann xi calculations.
 */
typedef struct SimXiContext {
    SimZetaContext zeta;        /**< Nested zeta dispatcher controls. */
    double abs_tol;             /**< Absolute error tolerance. */
    double rel_tol;             /**< Relative error tolerance. */
    double exact_limit_radius;  /**< Radius around removable limits treated with exact formulas. */
    double zero_tolerance;      /**< Critical-line zero refinement tolerance. */
    size_t zero_max_iterations; /**< Maximum zero-refinement iterations. */
} SimXiContext;

/**
 * @brief Value, error estimates, branch metadata, and status from xi evaluation.
 */
typedef struct SimXiResult {
    SimComplexDouble value;              /**< Evaluated xi value. */
    double abs_error;                    /**< Absolute error estimate. */
    double rel_error;                    /**< Relative error estimate. */
    size_t terms_used;                   /**< Zeta terms used by the underlying branch. */
    size_t refinement_rounds;            /**< Adaptive refinement rounds performed. */
    unsigned int working_precision_bits; /**< Effective working precision in bits. */
    SimZetaBranch zeta_branch;           /**< Zeta branch used while assembling xi. */
    SimZetaStatus status;                /**< Evaluation status. */
    unsigned int flags;                  /**< Bitmask of SIM_XI_FLAG_* values. */
} SimXiResult;

/**
 * @brief Xi value and first derivative with associated error and branch metadata.
 */
typedef struct SimXiDerivativeResult {
    SimComplexDouble value;              /**< Evaluated xi value. */
    SimComplexDouble derivative;         /**< Evaluated first derivative. */
    double abs_error;                    /**< Absolute error estimate for @ref value. */
    double rel_error;                    /**< Relative error estimate for @ref value. */
    double derivative_abs_error;         /**< Absolute error estimate for @ref derivative. */
    double derivative_rel_error;         /**< Relative error estimate for @ref derivative. */
    size_t refinement_rounds;            /**< Adaptive refinement rounds performed. */
    unsigned int working_precision_bits; /**< Effective working precision in bits. */
    SimZetaBranch zeta_branch;           /**< Zeta branch used while assembling xi. */
    SimZetaStatus status;                /**< Evaluation status. */
    unsigned int flags;                  /**< Bitmask of SIM_XI_FLAG_* values. */
} SimXiDerivativeResult;

/**
 * @brief Critical-line zero-search result for the completed xi function.
 */
typedef struct SimXiZeroResult {
    double t;                    /**< Critical-line ordinate of the zero candidate. */
    SimComplexBall xi_ball;      /**< Xi enclosure evaluated at @ref t. */
    SimBallRigor bracket_rigor;  /**< Rigor of the sign-changing bracket. */
    double derivative;           /**< Real critical-line derivative d/dt Xi(t), not d/ds xi(s). */
    double derivative_abs_error; /**< Absolute error estimate for @ref derivative. */
    size_t iterations;           /**< Number of refinement iterations performed. */
    SimZetaBranch zeta_branch;   /**< Zeta branch used during final evaluation. */
    SimZetaStatus status;        /**< Evaluation/search status. */
    unsigned int flags;          /**< Bitmask of SIM_XI_FLAG_* values. */
} SimXiZeroResult;

/**
 * @brief Default tolerances and limit-handling thresholds for the Phase 3 xi evaluator.
 *
 * @return Default xi evaluation context.
 */
SimXiContext sim_xi_context_default(void);

/**
 * @brief Low-latency xi context for interactive visual exploration.
 *
 * This pairs xi with `sim_zeta_context_interactive()` and relaxed tolerances so
 * renderers can use `sim_xi_eval()` for responsive previews while keeping the
 * exact/certified path separate.
 *
 * @return Relaxed low-latency xi evaluation context.
 */
SimXiContext sim_xi_context_interactive(void);

/**
 * @brief Human-readable description of an xi status value.
 *
 * @param status Status returned by xi evaluators.
 * @return Static diagnostic string.
 */
const char *sim_xi_status_string(SimZetaStatus status);

/**
 * @brief Evaluate the completed xi function.
 *
 * @param s Complex xi argument.
 * @param context Optional xi context.
 * @return Structured xi value, errors, zeta branch, status, and flags.
 */
SimXiResult sim_xi_eval(SimComplexDouble s, const SimXiContext *context);

/**
 * @brief Evaluate xi on the critical line.
 *
 * @param t Critical-line height in `s = 1/2 + i t`.
 * @param context Optional xi context.
 * @return Structured xi result on the critical line.
 */
SimXiResult sim_xi_eval_critical_line(double t, const SimXiContext *context);

/**
 * @brief Evaluate xi(s) and its first complex derivative.
 *
 * @param s Complex xi argument.
 * @param context Optional xi context.
 * @return Structured value and derivative result.
 */
SimXiDerivativeResult sim_xi_eval_with_derivative(SimComplexDouble s, const SimXiContext *context);

/**
 * @brief Ball-style enclosure derived from the current xi error model.
 *
 * The returned rigor metadata distinguishes heuristic, validated, and future
 * formal interval results. The current implementation provides heuristic and
 * cross-checked validated balls, plus formal enclosures for exact xi limits
 * and real integer arguments whose reflected zeta ball is formal.
 * Finite NO_CONVERGENCE evaluations are exposed as heuristic balls so callers
 * can still use decisive sign information without treating it as certification.
 *
 * @param s Complex xi argument.
 * @param context Optional xi context.
 * @return Complex ball enclosure with status and rigor metadata.
 */
SimComplexBall sim_xi_eval_ball(SimComplexDouble s, const SimXiContext *context);

/**
 * @brief Heuristically locate a critical-line zero of Xi(t) inside a bracket with a sign change.
 *
 * SIM_ZETA_STATUS_OK indicates a refined zero candidate supported by a
 * validated-or-better sign-changing bracket. `bracket_rigor` reports the
 * strength of that bracket explicitly, and `SIM_XI_FLAG_USED_ZERO_REFINEMENT`
 * only means a derivative-based refinement step ran; it is not itself a
 * certification flag. SIM_ZETA_STATUS_NO_CONVERGENCE may still carry a finite
 * best-effort candidate obtained from heuristic balls.
 *
 * @param t_lo Lower critical-line height.
 * @param t_hi Upper critical-line height; must exceed `t_lo`.
 * @param context Optional xi context.
 * @return Zero-search result with candidate, ball, derivative, and rigor.
 */
SimXiZeroResult sim_xi_find_critical_zero(double t_lo, double t_hi, const SimXiContext *context);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_XI_H */
