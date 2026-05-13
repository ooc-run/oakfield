/**
 * @file q_ladder.c
 * @brief q-deformed finite ladder phi helpers and derivatives.
 *
 * These functions evaluate finite sums of the form
 * `lambda / (lambda + epsilon * q^k)` and their lambda derivatives. The
 * implementation handles the classical `q == 1` case by delegating to the
 * non-q finite ladder and contains special handling for the degenerate `q == 0`
 * limit.
 *
 * @warning Experimental implementation. q-ladder domains, convergence behavior,
 * and precision expectations may change before release.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief q-deformed finite ladder phi_q(lambda, epsilon; K, q).
 *
 * Strategy:
 *   - Use direct summation for K ≤ 12.
 *   - Hybrid q^k:
 *         For k < threshold: qk *= q
 *         For k ≥ threshold: qk = exp(k * lnq)
 *   - Compensated summation to maintain accuracy.
 *   - Use first-order expansion when ε q^k << λ.
 */

/**
 * @brief Evaluate the real q-deformed finite ladder sum.
 *
 * For long sums, the implementation switches between multiplicative `qk *= q`
 * updates and `exp(k * log(q))` to limit drift/underflow. When the tail enters
 * a small-ratio regime, a geometric approximation is used to finish the sum.
 *
 * @param lambda Numerator and base denominator parameter.
 * @param epsilon q-scaled denominator perturbation.
 * @param K Number of finite ladder terms; must be positive.
 * @param q Deformation parameter. `q == 1` maps to the classical ladder.
 * @return Finite q-ladder value, or NaN for invalid inputs.
 */
double sim_qhyperexp_phi(double lambda, double epsilon, int K, double q) {
    if (!isfinite(lambda) || !isfinite(epsilon) || !isfinite(q) || K <= 0)
        return NAN;

    if (q == 1.0)
        return sim_hyperexp_phi(lambda, epsilon, K);
    if (q == 0.0) {
        double first = lambda / (lambda + epsilon);
        return first + (K - 1) * 1.0;
    }

    /* Small-K direct */
    if (K <= 12) {
        double sum = 0.0, c = 0.0;
        double qk = 1.0;
        for (int k = 0; k < K; k++) {
            double d    = lambda + epsilon * qk;
            double term = lambda / d;
            double y    = term - c;
            double t    = sum + y;
            c           = (t - sum) - y;
            sum         = t;
            qk *= q;
        }
        return sum;
    }

    /* Hybrid q^k strategy */
    double lnq      = log(q);
    int    k_switch = (lnq < -1e-4) ? (int) (-30.0 / lnq) : 100;

    double       sum = 0.0, c = 0.0;
    double       qk      = 1.0;
    const double ratio   = epsilon / lambda;
    const double tol     = 1e-8;
    const bool   q_decay = q < 1.0;
    const double qinv    = 1.0 / q;

    for (int k = 0; k < K; k++) {
        /* stable q^k generation */
        if (k == k_switch) {
            qk = exp(k * lnq);
        }

        /* denominator */
        double d = lambda + epsilon * qk;

        /* small-ratio expansion when ε q^k << λ */
        double term;
        if (fabs(epsilon * qk) < tol * fabs(lambda)) {
            /* φ ≈ 1 − (εq^k)/λ  */
            term = 1.0 - (epsilon * qk) / lambda;
        } else {
            term = lambda / d;
        }

        /* compensated sum */
        double y = term - c;
        double t = sum + y;
        c        = (t - sum) - y;
        sum      = t;

        /* tail acceleration when ratio stays small/large */
        int remaining = K - k - 1;
        if (remaining > 0) {
            if (q_decay && fabs(epsilon * qk * q) < tol * fabs(lambda)) {
                double tail_geom = (fabs(lnq) < 1e-6) ? ((double) remaining)
                                                      : (-expm1(remaining * lnq) / (1.0 - q));
                double tail_corr = ratio * qk * q * tail_geom;
                double tail_sum  = remaining - tail_corr;
                double ytail     = tail_sum - c;
                double tsum      = sum + ytail;
                sum              = tsum;
                break;
            } else if (!q_decay && fabs(lambda) < tol * fabs(epsilon * qk * q)) {
                double lnqinv    = log(qinv);
                double tail_geom = (fabs(lnqinv) < 1e-6)
                                       ? ((double) remaining)
                                       : (-expm1(remaining * lnqinv) / (1.0 - qinv));
                double lead      = (lambda / epsilon) * (qinv) / qk;
                double tail_sum  = lead * tail_geom;
                double ytail     = tail_sum - c;
                double tsum      = sum + ytail;
                sum              = tsum;
                break;
            }
        }

        /* next q^k */
        if (k < k_switch)
            qk *= q;
        else
            qk = exp((k + 1) * lnq);
    }

    return sum;
}

/**
 * @brief Evaluate the derivative of the real q-deformed finite ladder sum.
 *
 * Computes the derivative with respect to `lambda` of the finite q-ladder
 * terms. The same q-power generation and tail-acceleration strategy used by
 * `sim_qhyperexp_phi()` is applied here.
 *
 * @param lambda Numerator and base denominator parameter.
 * @param epsilon q-scaled denominator perturbation.
 * @param K Number of finite ladder terms; must be positive.
 * @param q Deformation parameter.
 * @return Lambda derivative of the finite q-ladder, or NaN for invalid inputs.
 */
double sim_qhyperexp_phi_deriv(double lambda, double epsilon, int K, double q) {
    if (!isfinite(lambda) || !isfinite(epsilon) || !isfinite(q) || K <= 0)
        return NAN;

    if (q == 1.0)
        return sim_hyperexp_phi_deriv(lambda, epsilon, K);
    if (q == 0.0)
        return epsilon / ((lambda + epsilon) * (lambda + epsilon));

    double lnq      = log(q);
    int    k_switch = (lnq < -1e-4) ? (int) (-30.0 / lnq) : 100;

    double       qk  = 1.0;
    double       sum = 0.0, c = 0.0;
    const double tol     = 1e-8;
    const bool   q_decay = q < 1.0;
    const double qinv    = 1.0 / q;

    for (int k = 0; k < K; k++) {
        if (k == k_switch)
            qk = exp(k * lnq);

        double d    = lambda + epsilon * qk;
        double inv  = 1.0 / d;
        double term = (epsilon * qk) * (inv * inv);

        double y = term - c;
        double t = sum + y;
        c        = (t - sum) - y;
        sum      = t;

        int remaining = K - k - 1;
        if (remaining > 0) {
            if (q_decay && fabs(epsilon * qk * q) < tol * fabs(lambda)) {
                double tail_geom = (fabs(lnq) < 1e-6) ? ((double) remaining)
                                                      : (-expm1(remaining * lnq) / (1.0 - q));
                double tail_sum  = (epsilon / (lambda * lambda)) * qk * q * tail_geom;
                double ytail     = tail_sum - c;
                double tsum      = sum + ytail;
                sum              = tsum;
                break;
            } else if (!q_decay && fabs(lambda) < tol * fabs(epsilon * qk * q)) {
                double lnqinv    = log(qinv);
                double tail_geom = (fabs(lnqinv) < 1e-6)
                                       ? ((double) remaining)
                                       : (-expm1(remaining * lnqinv) / (1.0 - qinv));
                double lead      = (1.0 / epsilon) * (qinv) / qk;
                double tail_sum  = lead * tail_geom;
                double ytail     = tail_sum - c;
                double tsum      = sum + ytail;
                sum              = tsum;
                break;
            }
        }

        if (k < k_switch)
            qk *= q;
        else
            qk = exp((k + 1) * lnq);
    }

    return sum;
}

/**
 * @brief Evaluate the complex q-deformed finite ladder sum.
 *
 * Complex `lambda` and `epsilon` follow the same finite sum as the real helper.
 * Degenerate `q == 1` and `q == 0` cases are handled explicitly; otherwise
 * terms are accumulated with compensated complex summation.
 *
 * @param lambda Complex numerator/base denominator parameter.
 * @param epsilon Complex q-scaled denominator perturbation.
 * @param K Number of finite ladder terms; must be positive.
 * @param q Real deformation parameter.
 * @return Complex q-ladder value, or `{NAN, NAN}` for invalid inputs.
 */
SimComplexDouble
sim_qhyperexp_phi_complex(SimComplexDouble lambda, SimComplexDouble epsilon, int K, double q) {
    SimComplexDouble out = { NAN, NAN };
    if (K <= 0)
        return out;

    double complex l = lambda.re + I * lambda.im;
    double complex e = epsilon.re + I * epsilon.im;

    if (q == 1.0) {
        SimComplexDouble v = sim_hyperexp_phi_complex(lambda, epsilon, K);
        return v;
    }
    if (q == 0.0) {
        double complex first = l / (l + e);
        double complex val   = first + (K - 1);
        return (SimComplexDouble) { creal(val), cimag(val) };
    }

    /* ---- Small-K direct ---- */
    if (K <= 12) {
        double complex sum = 0.0, c = 0.0;
        double         qk = 1.0;
        for (int k = 0; k < K; k++) {
            double complex d    = l + e * qk;
            double complex term = l / d;

            double complex y = term - c;
            double complex t = sum + y;
            c                = (t - sum) - y;
            sum              = t;

            qk *= q;
        }
        return (SimComplexDouble) { creal(sum), cimag(sum) };
    }

    /* hybrid q-power */
    double lnq      = log(q);
    int    k_switch = (lnq < -1e-4) ? (int) (-30.0 / lnq) : 100;

    double complex sum = 0.0, c = 0.0;
    double         qk      = 1.0;
    const double   tol     = 1e-8;
    const bool     q_decay = q < 1.0;
    const double   qinv    = 1.0 / q;

    for (int k = 0; k < K; k++) {
        if (k == k_switch) {
            qk = exp(k * lnq);
        }

        double complex d = l + e * qk;
        double complex term;

        /* small-ratio expansion: if |ε q^k| << |λ| */
        if (cabs(e * qk) < tol * cabs(l)) {
            term = 1.0 - (e * qk) / l; /* stable complex approx */
        } else {
            term = l / d;
        }

        double complex y = term - c;
        double complex t = sum + y;
        c                = (t - sum) - y;
        sum              = t;

        int remaining = K - k - 1;
        if (remaining > 0) {
            if (q_decay && cabs(e * qk * q) < tol * cabs(l)) {
                double         tail_geom = (fabs(lnq) < 1e-6) ? ((double) remaining)
                                                              : (-expm1(remaining * lnq) / (1.0 - q));
                double complex tail_corr = (e / l) * qk * q * tail_geom;
                double complex tail_sum  = remaining - tail_corr;
                double complex ytail     = tail_sum - c;
                double complex tsum      = sum + ytail;
                sum                      = tsum;
                break;
            } else if (!q_decay && cabs(l) < tol * cabs(e * qk * q)) {
                double         lnqinv    = log(qinv);
                double         tail_geom = (fabs(lnqinv) < 1e-6)
                                               ? ((double) remaining)
                                               : (-expm1(remaining * lnqinv) / (1.0 - qinv));
                double complex lead      = (l / e) * (qinv / qk);
                double complex tail_sum  = lead * tail_geom;
                double complex ytail     = tail_sum - c;
                double complex tsum      = sum + ytail;
                sum                      = tsum;
                break;
            }
        }

        if (k < k_switch)
            qk *= q;
        else
            qk = exp((k + 1) * lnq);
    }

    return (SimComplexDouble) { creal(sum), cimag(sum) };
}

/**
 * @brief Evaluate the complex lambda-derivative of the q-deformed finite ladder sum.
 *
 * Computes the complex derivative term-by-term, with the same short-sum,
 * q-degenerate, and tail-acceleration paths as the value evaluator.
 *
 * @param lambda Complex numerator/base denominator parameter.
 * @param epsilon Complex q-scaled denominator perturbation.
 * @param K Number of finite ladder terms; must be positive.
 * @param q Real deformation parameter.
 * @return Complex lambda derivative, or `{NAN, NAN}` for invalid inputs.
 */
SimComplexDouble sim_qhyperexp_phi_deriv_complex(SimComplexDouble lambda,
                                                 SimComplexDouble epsilon,
                                                 int              K,
                                                 double           q) {
    SimComplexDouble out = { NAN, NAN };
    if (K <= 0)
        return out;

    double complex l = lambda.re + I * lambda.im;
    double complex e = epsilon.re + I * epsilon.im;

    if (q == 1.0)
        return sim_hyperexp_phi_deriv_complex_opt(lambda, epsilon, K);
    if (q == 0.0) {
        double complex denom = l + e;
        double complex val   = e / (denom * denom);
        return (SimComplexDouble) { creal(val), cimag(val) };
    }

    double lnq      = log(q);
    int    k_switch = (lnq < -1e-4) ? (int) (-30.0 / lnq) : 100;

    double complex sum = 0.0, c = 0.0;
    double         qk      = 1.0;
    const double   tol     = 1e-8;
    const bool     q_decay = q < 1.0;
    const double   qinv    = 1.0 / q;

    for (int k = 0; k < K; k++) {
        if (k == k_switch)
            qk = exp(k * lnq);

        double complex d    = l + e * qk;
        double complex inv  = 1.0 / d;
        double complex term = (e * qk) * (inv * inv);

        double complex y = term - c;
        double complex t = sum + y;
        c                = (t - sum) - y;
        sum              = t;

        int remaining = K - k - 1;
        if (remaining > 0) {
            if (q_decay && cabs(e * qk * q) < tol * cabs(l)) {
                double         tail_geom = (fabs(lnq) < 1e-6) ? ((double) remaining)
                                                              : (-expm1(remaining * lnq) / (1.0 - q));
                double complex tail_sum  = (e / (l * l)) * qk * q * tail_geom;
                double complex ytail     = tail_sum - c;
                double complex tsum      = sum + ytail;
                sum                      = tsum;
                break;
            } else if (!q_decay && cabs(l) < tol * cabs(e * qk * q)) {
                double         lnqinv    = log(qinv);
                double         tail_geom = (fabs(lnqinv) < 1e-6)
                                               ? ((double) remaining)
                                               : (-expm1(remaining * lnqinv) / (1.0 - qinv));
                double complex lead      = (1.0 / e) * (qinv / qk);
                double complex tail_sum  = lead * tail_geom;
                double complex ytail     = tail_sum - c;
                double complex tsum      = sum + ytail;
                sum                      = tsum;
                break;
            }
        }

        if (k < k_switch)
            qk *= q;
        else
            qk = exp((k + 1) * lnq);
    }

    return (SimComplexDouble) { creal(sum), cimag(sum) };
}
