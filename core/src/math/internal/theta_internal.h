/**
 * @file theta_internal.h
 * @brief Shared private helpers for the split theta math modules.
 *
 * This header is included by several small waveform modules that need fast
 * trigonometric recurrences or reciprocal powers. Helpers are kept inline to
 * avoid introducing a separate private object file.
 */
#ifndef OAKFIELD_MATH_THETA_INTERNAL_H
#define OAKFIELD_MATH_THETA_INTERNAL_H

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SIM_THETA_TOL 1e-14
#define SIM_THETA_NMAX ((size_t) 512)

/**
 * @brief Advance sine/cosine values by one fixed angular step.
 *
 * Given `sin(kx)` and `cos(kx)`, plus `sin(x)` and `cos(x)`, this computes
 * `sin((k+1)x)` and `cos((k+1)x)` using angle-addition formulas.
 *
 * @param sin_k Current sine value.
 * @param cos_k Current cosine value.
 * @param s1 Sine of the step angle.
 * @param c1 Cosine of the step angle.
 * @param out_sin Receives the advanced sine value.
 * @param out_cos Receives the advanced cosine value.
 */
static inline void sim_theta_sincos_step(double  sin_k,
                                         double  cos_k,
                                         double  s1,
                                         double  c1,
                                         double *out_sin,
                                         double *out_cos) {
    *out_sin = fma(sin_k, c1, cos_k * s1);
    *out_cos = fma(cos_k, c1, -sin_k * s1);
}

/**
 * @brief Recognize small integer exponents that have reciprocal-power fast paths.
 *
 * @param s Candidate real exponent.
 * @return Integer exponent in [1, 4] when recognized, otherwise 0.
 */
static inline int sim_theta_is_small_int_s(double s) {
    int si = (int) llround(s);
    if (fabs(s - (double) si) <= 1e-12 && si >= 1 && si <= 4) {
        return si;
    }
    return 0;
}

/**
 * @brief Compute `n^-si` for small positive integer exponents without `pow()`.
 *
 * @param n Positive base.
 * @param si Integer exponent, usually in [1, 4].
 * @return Reciprocal power `1 / n^si`, falling back to `pow()` outside the fast set.
 */
static inline double sim_theta_inv_pow_small_int(double n, int si) {
    switch (si) {
        case 1:
            return 1.0 / n;
        case 2: {
            double n2 = n * n;
            return 1.0 / n2;
        }
        case 3: {
            double n2 = n * n;
            return 1.0 / (n2 * n);
        }
        case 4: {
            double n2 = n * n;
            return 1.0 / (n2 * n2);
        }
        default:
            return pow(n, -(double) si);
    }
}

/**
 * @brief Compute `n^-p` for a runtime integer exponent.
 *
 * This helper mirrors `sim_theta_inv_pow_small_int()` but accepts arbitrary
 * integer exponents from callers that already have an integral power.
 *
 * @param p Integer exponent.
 * @param n Positive base.
 * @return Reciprocal power `1 / n^p`.
 */
static inline double sim_theta_inv_pow_const_runtime(int p, double n) {
    switch (p) {
        case 1:
            return 1.0 / n;
        case 2: {
            double n2 = n * n;
            return 1.0 / n2;
        }
        case 3: {
            double n2 = n * n;
            return 1.0 / (n2 * n);
        }
        case 4: {
            double n2 = n * n;
            return 1.0 / (n2 * n2);
        }
        default:
            return pow(n, -(double) p);
    }
}

#endif /* OAKFIELD_MATH_THETA_INTERNAL_H */
