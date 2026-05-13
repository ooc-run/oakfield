/**
 * @file bessel.c
 * @brief Cylindrical Bessel J_n approximations for real scalar inputs.
 *
 * The evaluator supports integer orders, including negative orders via
 * `J_{-n}(x) = (-1)^n J_n(x)`. Small and moderate arguments use the defining
 * power series; large arguments use the leading asymptotic cosine form.
 */

#include "oakfield/math/bessel.h"

#include <math.h>
#include <stddef.h>

#define SIM_BESSEL_SERIES_MAX_TERMS 64U
#define SIM_BESSEL_ASYMPTOTIC_THRESHOLD 18.0
#define SIM_BESSEL_ASYMPTOTIC_TERMS 8U

/**
 * @brief Evaluate J_order(x) for nonnegative order and nonnegative x by series summation.
 *
 * The first term is `(x/2)^order / order!`; subsequent terms are generated
 * recursively to avoid repeated factorials. Summation stops when the next term
 * is tiny relative to the partial sum or when `SIM_BESSEL_SERIES_MAX_TERMS`
 * is reached.
 *
 * @param order Nonnegative integer Bessel order.
 * @param x Nonnegative real argument.
 * @return Series approximation to J_order(x).
 */
static double sim_bessel_jn_series_nonnegative_f64(unsigned int order, double x) {
    double half_x = 0.5 * x;
    double term   = 1.0;
    double sum    = 0.0;

    if (order == 0U) {
        term = 1.0;
    } else {
        for (unsigned int i = 1U; i <= order; ++i) {
            term *= half_x / (double) i;
        }
    }

    sum = term;
    for (size_t k = 1U; k < SIM_BESSEL_SERIES_MAX_TERMS; ++k) {
        double denom = (double) k * (double) (k + (size_t) order);
        term *= -(half_x * half_x) / denom;
        sum += term;
        if (fabs(term) <= 1.0e-16 * (fabs(sum) + 1.0)) {
            break;
        }
    }

    return sum;
}

/**
 * @brief Evaluate the large-argument expansion for integer-order J_n(x).
 *
 * Uses the classical fixed-order expansion
 * `sqrt(2/(pi*x)) * (cos(w) P - sin(w) Q)`, with enough terms to improve the
 * benchmark range without relying on platform-specific libm Bessel routines.
 *
 * @param order Nonnegative integer Bessel order.
 * @param x Positive real argument.
 * @return Large-argument approximation to J_order(x).
 */
static double sim_bessel_jn_asymptotic_nonnegative_f64(unsigned int order, double x) {
    double mu      = 4.0 * (double) order * (double) order;
    double inv     = 1.0 / x;
    double inv_pow = 1.0;
    double coeff   = 1.0;
    double p       = 1.0;
    double q       = 0.0;

    for (size_t m = 1U; m <= SIM_BESSEL_ASYMPTOTIC_TERMS; ++m) {
        double odd = 2.0 * (double) m - 1.0;
        coeff *= (mu - odd * odd) / (8.0 * (double) m);
        inv_pow *= inv;

        if ((m & 1U) != 0U) {
            size_t k = (m - 1U) / 2U;
            q += ((k & 1U) != 0U ? -coeff : coeff) * inv_pow;
        } else {
            size_t k = m / 2U;
            p += ((k & 1U) != 0U ? -coeff : coeff) * inv_pow;
        }
    }

    double phase = x - 0.5 * (double) order * M_PI - 0.25 * M_PI;
    return sqrt(2.0 / (M_PI * x)) * (cos(phase) * p - sin(phase) * q);
}

/**
 * @brief Evaluate the cylindrical Bessel function J_n(x) in double precision.
 *
 * Handles NaN and infinities explicitly, applies parity identities for negative
 * order and negative argument, then chooses between the power series and the
 * leading large-argument asymptotic approximation.
 *
 * @param order Integer Bessel order.
 * @param x Real argument.
 * @return Approximate value of J_order(x).
 */
double sim_bessel_jn_f64(int order, double x) {
    int    sign_factor = 1;
    int    nonnegative_order;
    double abs_x;
    double value;

    if (isnan(x)) {
        return x;
    }
    if (isinf(x)) {
        return 0.0;
    }

    if (order < 0) {
        nonnegative_order = -order;
        if ((nonnegative_order & 1) != 0) {
            sign_factor = -sign_factor;
        }
    } else {
        nonnegative_order = order;
    }

    abs_x = fabs(x);
    if (x < 0.0 && (nonnegative_order & 1) != 0) {
        sign_factor = -sign_factor;
    }

    if (abs_x == 0.0) {
        return (nonnegative_order == 0) ? (double) sign_factor : 0.0;
    }

    if (abs_x >= SIM_BESSEL_ASYMPTOTIC_THRESHOLD) {
        value = sim_bessel_jn_asymptotic_nonnegative_f64((unsigned int) nonnegative_order, abs_x);
    } else {
        value = sim_bessel_jn_series_nonnegative_f64((unsigned int) nonnegative_order, abs_x);
    }

    return (double) sign_factor * value;
}

/**
 * @brief Evaluate J_0(x) in double precision.
 *
 * @param x Real argument.
 * @return Approximate value of the zeroth-order cylindrical Bessel function.
 */
double sim_bessel_j0_f64(double x) {
    return sim_bessel_jn_f64(0, x);
}

/**
 * @brief Evaluate J_1(x) in double precision.
 *
 * @param x Real argument.
 * @return Approximate value of the first-order cylindrical Bessel function.
 */
double sim_bessel_j1_f64(double x) {
    return sim_bessel_jn_f64(1, x);
}

/**
 * @brief Evaluate the cylindrical Bessel function J_n(x) in float precision.
 *
 * The calculation is delegated to the double implementation and rounded back
 * to `float`, preserving the same order and argument conventions.
 *
 * @param order Integer Bessel order.
 * @param x Real single-precision argument.
 * @return Approximate value of J_order(x) rounded to float.
 */
float sim_bessel_jn_f32(int order, float x) {
    return (float) sim_bessel_jn_f64(order, (double) x);
}
