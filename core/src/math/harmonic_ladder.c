/**
 * @file harmonic_ladder.c
 * @brief Generalized harmonic ladder helpers built from polygamma differences.
 *
 * For small `K`, direct finite sums avoid cancellation and overhead. For larger
 * values, the implementation uses digamma, trigamma, and tetragamma differences
 * to evaluate the ladder and its derivatives.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Asymptotic digamma difference psi(a + K) - psi(a).
 *
 * Used when `a` is large enough that subtracting two full digamma evaluations
 * would lose useful precision.
 *
 * @param a Positive starting offset.
 * @param K Number of ladder terms.
 * @param tail_terms Number of Bernoulli tail terms to include.
 * @return Approximation to `psi(a + K) - psi(a)`.
 */
static inline double harmonic_digamma_diff_asymp_f64(double a, int K, int tail_terms) {
    const double ap   = a + (double) K;
    const double inv  = 1.0 / a;
    const double invp = 1.0 / ap;
    double       res  = log1p(((double) K) * inv) + 0.5 * (inv - invp);

    const double inv2   = inv * inv;
    const double inv2p  = invp * invp;
    double       inv2n  = inv2;
    double       inv2np = inv2p;

    int N    = tail_terms;
    int maxN = (int) (sizeof(B2) / sizeof(B2[0]));
    if (N > maxN)
        N = maxN;

    for (int n = 1; n <= N; ++n) {
        const double c = -B2[n - 1] / (2.0 * (double) n);
        res += c * (inv2np - inv2n);
        inv2n *= inv2;
        inv2np *= inv2p;
    }
    return res;
}

/**
 * @brief Asymptotic trigamma difference psi_1(a + K) - psi_1(a).
 *
 * @param a Positive starting offset.
 * @param K Number of ladder terms.
 * @param tail_terms Number of Bernoulli tail terms to include.
 * @return Approximation to the first derivative of the harmonic ladder.
 */
static inline double harmonic_trigamma_diff_asymp_f64(double a, int K, int tail_terms) {
    const double ap    = a + (double) K;
    const double inv   = 1.0 / a;
    const double invp  = 1.0 / ap;
    const double inv2  = inv * inv;
    const double inv2p = invp * invp;

    double res = (invp - inv) + 0.5 * (inv2p - inv2);

    double inv_pow  = inv2 * inv;
    double inv_powp = inv2p * invp;

    int N    = tail_terms;
    int maxN = (int) (sizeof(B2) / sizeof(B2[0]));
    if (N > maxN)
        N = maxN;

    for (int n = 1; n <= N; ++n) {
        res += B2[n - 1] * (inv_powp - inv_pow);
        inv_pow *= inv2;
        inv_powp *= inv2p;
    }
    return res;
}

/**
 * @brief Evaluate the generalized harmonic ladder H_K(a).
 *
 * Computes `sum_{k=0}^{K-1} 1 / (a + k)` for `a > 0` and `K >= 0`.
 *
 * @param a Positive starting offset.
 * @param K Number of terms.
 * @return H_K(a), zero for `K == 0`, or NaN for invalid inputs.
 */
double sim_ghn_HK(double a, int K) {
    if (!(a > 0.0) || !isfinite(a))
        return NAN;
    if (K < 0)
        return NAN;
    if (K == 0)
        return 0.0;

    if (K <= 32) {
        long double s  = 0.0L;
        long double ak = (long double) a;
        for (int k = 0; k < K; ++k, ak += 1.0L)
            s += 1.0L / ak;
        return (double) s;
    }

    if (a > 64.0 && K <= 128)
        return harmonic_digamma_diff_asymp_f64(a, K, 12);

    return sim_digamma_f64_12(a + (double) K) - sim_digamma_f64_12(a);
}

/**
 * @brief Evaluate the first derivative of H_K(a) with respect to a.
 *
 * Computes `-sum 1 / (a + k)^2`, equivalent to a trigamma difference.
 *
 * @param a Positive starting offset.
 * @param K Number of terms.
 * @return First derivative, zero for `K == 0`, or NaN for invalid inputs.
 */
double sim_ghn_HK_d1(double a, int K) {
    if (!(a > 0.0) || !isfinite(a))
        return NAN;
    if (K < 0)
        return NAN;
    if (K == 0)
        return 0.0;

    if (K <= 32) {
        long double s  = 0.0L;
        long double ak = (long double) a;
        for (int k = 0; k < K; ++k, ak += 1.0L)
            s -= 1.0L / (ak * ak);
        return (double) s;
    }

    if (a > 64.0 && K <= 128)
        return harmonic_trigamma_diff_asymp_f64(a, K, 12);

    return sim_trigamma_f64_12(a + (double) K) - sim_trigamma_f64_12(a);
}

/**
 * @brief Evaluate the second derivative of H_K(a) with respect to a.
 *
 * Computes `2 * sum 1 / (a + k)^3`, equivalent to a tetragamma difference.
 *
 * @param a Positive starting offset.
 * @param K Number of terms.
 * @return Second derivative, zero for `K == 0`, or NaN for invalid inputs.
 */
double sim_ghn_HK_d2(double a, int K) {
    if (!(a > 0.0) || !isfinite(a))
        return NAN;
    if (K < 0)
        return NAN;
    if (K == 0)
        return 0.0;

    if (K <= 32) {
        long double s  = 0.0L;
        long double ak = (long double) a;
        for (int k = 0; k < K; ++k, ak += 1.0L)
            s += 2.0L / (ak * ak * ak);
        return (double) s;
    }

    return sim_tetragamma_f64_12(a + (double) K) - sim_tetragamma_f64_12(a);
}

/**
 * @brief Evaluate the complex generalized harmonic ladder H_K(a).
 *
 * @param a Complex starting offset.
 * @param K Number of terms.
 * @return Complex ladder value, zero for `K == 0`, or NaNs for invalid `K`.
 */
SimComplexDouble sim_ghn_HK_complex(SimComplexDouble a, int K) {
    if (K < 0)
        return (SimComplexDouble) { NAN, NAN };
    if (K == 0)
        return (SimComplexDouble) { 0.0, 0.0 };
    SimComplexDouble ap = { a.re + (double) K, a.im };
    SimComplexDouble p1 = sim_digamma_c64_12(ap);
    SimComplexDouble p0 = sim_digamma_c64_12(a);
    return (SimComplexDouble) { p1.re - p0.re, p1.im - p0.im };
}

/**
 * @brief Evaluate the complex first derivative of H_K(a).
 *
 * @param a Complex starting offset.
 * @param K Number of terms.
 * @return Complex first derivative, zero for `K == 0`, or NaNs for invalid `K`.
 */
SimComplexDouble sim_ghn_HK_d1_complex(SimComplexDouble a, int K) {
    if (K < 0)
        return (SimComplexDouble) { NAN, NAN };
    if (K == 0)
        return (SimComplexDouble) { 0.0, 0.0 };
    SimComplexDouble ap = { a.re + (double) K, a.im };
    SimComplexDouble p1 = sim_trigamma_c64_12(ap);
    SimComplexDouble p0 = sim_trigamma_c64_12(a);
    return (SimComplexDouble) { p1.re - p0.re, p1.im - p0.im };
}

/**
 * @brief Evaluate the complex second derivative of H_K(a).
 *
 * @param a Complex starting offset.
 * @param K Number of terms.
 * @return Complex second derivative, zero for `K == 0`, or NaNs for invalid `K`.
 */
SimComplexDouble sim_ghn_HK_d2_complex(SimComplexDouble a, int K) {
    if (K < 0)
        return (SimComplexDouble) { NAN, NAN };
    if (K == 0)
        return (SimComplexDouble) { 0.0, 0.0 };
    SimComplexDouble ap = { a.re + (double) K, a.im };
    SimComplexDouble p1 = sim_tetragamma_c64_12(ap);
    SimComplexDouble p0 = sim_tetragamma_c64_12(a);
    return (SimComplexDouble) { p1.re - p0.re, p1.im - p0.im };
}
