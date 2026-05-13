/**
 * @file theta.h
 * @brief Jacobi-theta-derived waveform helpers.
 */
#ifndef OAKFIELD_MATH_THETA_H
#define OAKFIELD_MATH_THETA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Research-grade evaluation of real Jacobi-theta-derived waveform families.
 *
 * Mathematical conventions
 * ------------------------
 * For a real phase z and real nome q with 0 <= q < 1, the classical Jacobi theta
 * functions are evaluated through their Fourier series,
 *
 *   theta1(z,q)   = 2 sum_{n>=0} (-1)^n q^{(n+1/2)^2} sin((2n+1) z),
 *   theta2(0,q)   = 2 sum_{n>=0}          q^{(n+1/2)^2},
 *   theta3(0,q)   = 1 + 2 sum_{n>=1}      q^{n^2},
 *   theta4(z,q)   = 1 + 2 sum_{n>=1} (-1)^n q^{n^2} cos(2n z).
 *
 * The functions below expose waveform-oriented combinations of these series while
 * also reporting numerical diagnostics. The API accepts precomputed s1 = sin(z)
 * and c1 = cos(z) so that a caller can reuse oscillator state or SIMD pipelines.
 * If the supplied pair is not close to the unit circle, the implementation falls
 * back to recomputing sin(z) and cos(z) internally.
 *
 * Numerical methodology
 * ---------------------
 * - Series are accumulated in long double with Neumaier compensated summation.
 * - Each result carries a conservative absolute truncation/error estimate.
 * - Domain failures, singular configurations, and non-convergence are surfaced via
 *   theta_status instead of being silently clipped.
 * - The legacy double-returning wrappers are kept for compatibility; on failure
 *   they return NaN.
 */

typedef enum theta_status {
    THETA_STATUS_OK = 0,        /**< Evaluation completed successfully. */
    THETA_STATUS_DOMAIN_ERROR,  /**< Inputs fall outside the supported theta domain. */
    THETA_STATUS_SINGULAR,      /**< Evaluation encountered a singular configuration. */
    THETA_STATUS_NO_CONVERGENCE /**< Series evaluation did not converge within budget. */
} theta_status;

/**
 * @brief Theta waveform value with error estimate and convergence metadata.
 */
typedef struct theta_eval {
    double value;        /**< Evaluated function value. */
    double abs_error;    /**< Conservative absolute error estimate. */
    size_t terms_used;   /**< Number of Fourier/Lambert terms accumulated. */
    theta_status status; /**< Numerical status code. */
} theta_eval;

/**
 * @brief Human-readable description of a theta_status value.
 */
const char *theta_status_string(theta_status status);

/**
 * @brief Evaluate theta1(z,q) / theta2(0,q).
 *
 * This is the odd, 2*pi-periodic theta quotient used as the base sinusoid-like
 * member of the family. For q -> 0 it satisfies the analytic limit
 *
 *   theta1(z,q) / theta2(0,q) -> sin(z).
 *
 * @param z   Real phase variable.
 * @param q   Real nome, required to satisfy 0 <= q < 1.
 * @param s1  Optional cached sin(z).
 * @param c1  Optional cached cos(z).
 * @return    Value, error estimate, number of terms, and convergence status.
 */
theta_eval theta1_norm_eval(double z, double q, double s1, double c1);

/**
 * @brief Evaluate the square-like theta quotient
 *        [theta3(0,q)/theta2(0,q)] * [theta1(z,q)/theta4(z,q)].
 *
 * Under the standard change of variable v = pi*u/(2*K(k)), this is the classical
 * theta-function quotient for Jacobi sn(u,k). In the small-nome limit q -> 0 it
 * reduces analytically to sin(z).
 *
 * @param z   Real phase variable.
 * @param q   Real nome, required to satisfy 0 <= q < 1.
 * @param s1  Optional cached sin(z).
 * @param c1  Optional cached cos(z).
 * @return    Value, error estimate, number of terms, and convergence status.
 */
theta_eval theta_square_eval(double z, double q, double s1, double c1);

/**
 * @brief Evaluate the phase-based saw map (1/pi) arg(1 - q exp(-i z)).
 *
 * This function is not itself a classical theta quotient. It is the principal-
 * branch argument of a first-order analytic phase map and serves as a smooth saw
 * surrogate. For q -> 1^- it approaches a centered sawtooth on the principal
 * branch, away from the branch point at z = 2*pi*m.
 *
 * @param z   Real phase variable. Only used when s1/c1 must be recomputed.
 * @param q   Real shaping parameter, required to satisfy 0 <= q < 1.
 * @param s1  Optional cached sin(z).
 * @param c1  Optional cached cos(z).
 * @return    Value, error estimate, number of terms, and convergence status.
 */
theta_eval theta_saw_unit_eval(double z, double q, double s1, double c1);

/**
 * @brief Evaluate the triangle-like odd-harmonic Lambert/theta series
 *
 *   T(z,q) = (8/pi^2) * [(-log q)/sqrt(q)]
 *            * sum_{n>=0} (-1)^n
 *              [ q^{n+1/2} / ((2n+1)^2 (1 - q^{2n+1})) ] sin((2n+1) z).
 *
 * Despite the historical function name, the chosen normalization does not recover
 * the classical triangle-wave series as q -> 1^-. Instead, the coefficients tend
 * to the smoother cubic odd-harmonic law 8/(pi^2 (2n+1)^3), so the limit is a
 * twice-integrated square-wave family rather than an exact triangle. The
 * expression is singular as q -> 0^+ and therefore this routine is defined only
 * for 0 < q < 1.
 *
 * @param z   Real phase variable.
 * @param q   Real nome/shaping parameter, required to satisfy 0 < q < 1.
 * @param s1  Optional cached sin(z).
 * @param c1  Optional cached cos(z).
 * @return    Value, error estimate, number of terms, and convergence status.
 */
theta_eval theta_triangle_eval(double z, double q, double s1, double c1);

/* Backward-compatible scalar wrappers. These return NaN if evaluation fails. */
double theta1_norm(double z, double q, double s1, double c1);
double theta_square(double z, double q, double s1, double c1);
double theta_saw_unit(double z, double q, double s1, double c1);
double theta_triangle(double z, double q, double s1, double c1);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_THETA_H */
