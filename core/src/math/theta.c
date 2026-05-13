/**
 * @file theta.c
 * @brief Research-grade Jacobi-theta-derived waveform evaluators.
 *
 * The evaluators return both a value and a conservative absolute error estimate.
 * Internally they use long-double accumulation, Neumaier compensation, and
 * explicit tail bounds for the q-series. Convenience wrappers at the end return
 * only the value and map hard domain/singularity failures to quiet NaN.
 */

#include "oakfield/math/theta.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

#ifndef THETA_RG_MAX_TERMS
#define THETA_RG_MAX_TERMS 16384u
#endif

#ifndef THETA_RG_ABS_TOL
#define THETA_RG_ABS_TOL (32.0L * LDBL_EPSILON)
#endif

#define THETA_PI ((long double) M_PI)

/**
 * @brief Accumulated odd half-integer theta components and error bounds.
 */
typedef struct theta_half_result {
    long double theta1;
    long double theta2_zero;
    long double err_theta1;
    long double err_theta2_zero;
    size_t      terms_used;
    bool        converged;
} theta_half_result;

/**
 * @brief Accumulated even theta-series components and error bounds.
 */
typedef struct theta_even_result {
    long double theta3_zero;
    long double theta4;
    long double err_theta3_zero;
    long double err_theta4;
    size_t      terms_used;
    bool        converged;
} theta_even_result;

/**
 * @brief Validate q for theta functions that allow the closed endpoint q == 0.
 *
 * @param q Nome-like theta parameter.
 * @return true when `q` is finite and in [0, 1).
 */
static inline bool theta_valid_q_closed(double q) {
    return isfinite(q) && q >= 0.0 && q < 1.0;
}

/**
 * @brief Validate q for theta functions that require a strictly positive nome.
 *
 * @param q Nome-like theta parameter.
 * @return true when `q` is finite and in (0, 1).
 */
static inline bool theta_valid_q_open(double q) {
    return isfinite(q) && q > 0.0 && q < 1.0;
}

/**
 * @brief Add one long-double term with Neumaier compensation.
 *
 * @param term Term to add.
 * @param sum Running sum.
 * @param comp Running compensation term.
 * @param abs_accum Running sum of absolute term magnitudes.
 */
static inline void
theta_neumaier_add(long double term, long double* sum, long double* comp, long double* abs_accum) {
    long double t = *sum + term;
    if (fabsl(*sum) >= fabsl(term)) {
        *comp += (*sum - t) + term;
    } else {
        *comp += (term - t) + *sum;
    }
    *sum = t;
    *abs_accum += fabsl(term);
}

/**
 * @brief Advance long-double sine/cosine recurrences by one fixed step.
 *
 * @param sin_k Current sine value.
 * @param cos_k Current cosine value.
 * @param s_step Sine of the step angle.
 * @param c_step Cosine of the step angle.
 * @param ns Receives the next sine value.
 * @param nc Receives the next cosine value.
 */
static inline void theta_sincos_step(long double  sin_k,
                                     long double  cos_k,
                                     long double  s_step,
                                     long double  c_step,
                                     long double* ns,
                                     long double* nc) {
    *ns = sin_k * c_step + cos_k * s_step;
    *nc = cos_k * c_step - sin_k * s_step;
}

/**
 * @brief Prepare normalized sine/cosine values for a theta phase.
 *
 * If caller-supplied `s1_in` and `c1_in` are finite and close to unit length,
 * they are normalized and reused. Otherwise the values are recomputed from `z`
 * after reducing it modulo 2*pi.
 *
 * @param z Phase in radians.
 * @param s1_in Optional sine of z.
 * @param c1_in Optional cosine of z.
 * @param sin_z Receives normalized sine.
 * @param cos_z Receives normalized cosine.
 */
static inline void
theta_prepare_angles(double z, double s1_in, double c1_in, long double* sin_z, long double* cos_z) {
    long double s  = (long double) s1_in;
    long double c  = (long double) c1_in;
    long double r2 = s * s + c * c;

    if (isfinite(s1_in) && isfinite(c1_in) && fabsl(r2 - 1.0L) <= 256.0L * LDBL_EPSILON) {
        long double norm = sqrtl(r2);
        *sin_z           = s / norm;
        *cos_z           = c / norm;
        return;
    }

    long double zr = remainderl((long double) z, 2.0L * THETA_PI);
    *sin_z         = sinl(zr);
    *cos_z         = cosl(zr);
}

/**
 * @brief Construct a theta evaluation result from long-double intermediates.
 *
 * @param value Long-double value to cast to double.
 * @param abs_error Conservative absolute error estimate.
 * @param terms_used Number of terms consumed by the dominant series.
 * @param status Numerical status.
 * @return Packed `theta_eval` result.
 */
static inline theta_eval
theta_make_eval(long double value, long double abs_error, size_t terms_used, theta_status status) {
    theta_eval out;
    out.value      = (double) value;
    out.abs_error  = (double) (abs_error >= 0.0L ? abs_error : 0.0L);
    out.terms_used = terms_used;
    out.status     = status;
    return out;
}

/**
 * @brief Construct an error result with zero value/error and a status code.
 *
 * @param terms_used Number of terms consumed before the error was detected.
 * @param status Error status to report.
 * @return Packed `theta_eval` error result.
 */
static inline theta_eval theta_make_error_eval(size_t terms_used, theta_status status) {
    return theta_make_eval(0.0L, 0.0L, terms_used, status);
}

/**
 * @brief Return a quiet NaN bit pattern without depending on optional libm helpers.
 *
 * @return Quiet NaN double.
 */
static inline double theta_quiet_nan(void) {
    union {
        uint64_t bits;
        double   value;
    } out = { UINT64_C(0x7ff8000000000000) };
    return out.value;
}

/**
 * @brief Divide two bounded long-double quantities and propagate an error bound.
 *
 * @param a Numerator center.
 * @param da Numerator absolute error.
 * @param b Denominator center.
 * @param db Denominator absolute error.
 * @param value Receives `a / b`.
 * @param err Receives propagated absolute error.
 * @return false when the denominator interval may contain zero.
 */
static inline bool theta_ratio_bound(long double  a,
                                     long double  da,
                                     long double  b,
                                     long double  db,
                                     long double* value,
                                     long double* err) {
    long double babs = fabsl(b);
    long double bmin = babs - db;
    if (!(bmin > 0.0L)) {
        return false;
    }

    *value = a / b;
    *err   = da / bmin + (fabsl(a) * db) / (babs * bmin);
    return true;
}

/**
 * @brief Multiply two bounded long-double quantities and propagate an error bound.
 *
 * @param x First factor center.
 * @param dx First factor absolute error.
 * @param y Second factor center.
 * @param dy Second factor absolute error.
 * @param value Receives `x * y`.
 * @param err Receives propagated absolute error.
 */
static inline void theta_product_bound(long double  x,
                                       long double  dx,
                                       long double  y,
                                       long double  dy,
                                       long double* value,
                                       long double* err) {
    *value = x * y;
    *err   = fabsl(x) * dy + fabsl(y) * dx + dx * dy;
}

/**
 * @brief Accumulate odd half-integer theta components.
 *
 * Produces the numerator for theta1-like odd sine series and theta2(0,q) used
 * as a normalizing denominator, with tail and roundoff estimates.
 *
 * @param q Nome in [0, 1).
 * @param sin_z Sine of the phase.
 * @param cos_z Cosine of the phase.
 * @return Odd-series accumulators and convergence metadata.
 */
static theta_half_result theta_half_series(long double q, long double sin_z, long double cos_z) {
    theta_half_result out;
    out.theta1          = 0.0L;
    out.theta2_zero     = 0.0L;
    out.err_theta1      = 0.0L;
    out.err_theta2_zero = 0.0L;
    out.terms_used      = 0;
    out.converged       = true;

    if (q == 0.0L) {
        return out;
    }

    long double s_step = 2.0L * sin_z * cos_z;
    long double c_step = cos_z * cos_z - sin_z * sin_z;

    long double sin_k = sin_z;
    long double cos_k = cos_z;
    long double q2    = q * q;
    long double p     = sqrtl(sqrtl(q));
    long double mult  = q2;
    long double sign  = 1.0L;

    long double sum     = 0.0L;
    long double sum_c   = 0.0L;
    long double sum_abs = 0.0L;

    long double t2     = 0.0L;
    long double t2_c   = 0.0L;
    long double t2_abs = 0.0L;

    long double tail = (long double) DBL_MAX;

    for (size_t n = 0; n < THETA_RG_MAX_TERMS; ++n) {
        long double coeff = 2.0L * p;
        theta_neumaier_add(sign * coeff * sin_k, &sum, &sum_c, &sum_abs);
        theta_neumaier_add(coeff, &t2, &t2_c, &t2_abs);
        out.terms_used = n + 1;

        long double next_p    = p * mult;
        long double next_mult = mult * q2;
        if (next_p == 0.0L) {
            tail          = 0.0L;
            out.converged = true;
            break;
        }

        if (next_mult < 1.0L) {
            tail = (2.0L * next_p) / (1.0L - next_mult);
        }

        if (tail <= THETA_RG_ABS_TOL) {
            out.converged = true;
            break;
        }

        long double ns, nc;
        theta_sincos_step(sin_k, cos_k, s_step, c_step, &ns, &nc);
        sin_k         = ns;
        cos_k         = nc;
        sign          = -sign;
        p             = next_p;
        mult          = next_mult;
        out.converged = false;
    }

    out.theta1          = sum + sum_c;
    out.theta2_zero     = t2 + t2_c;
    out.err_theta1      = tail + 8.0L * LDBL_EPSILON * sum_abs;
    out.err_theta2_zero = tail + 8.0L * LDBL_EPSILON * t2_abs;
    return out;
}

/**
 * @brief Accumulate even theta components at z and zero.
 *
 * Produces theta3(0,q)-like and theta4(z,q)-like components used by the
 * square-wave quotient.
 *
 * @param q Nome in [0, 1).
 * @param sin_z Sine of the phase.
 * @param cos_z Cosine of the phase.
 * @return Even-series accumulators and convergence metadata.
 */
static theta_even_result theta_even_series(long double q, long double sin_z, long double cos_z) {
    theta_even_result out;
    out.theta3_zero     = 1.0L;
    out.theta4          = 1.0L;
    out.err_theta3_zero = 0.0L;
    out.err_theta4      = 0.0L;
    out.terms_used      = 0;
    out.converged       = true;

    if (q == 0.0L) {
        return out;
    }

    long double s_step = 2.0L * sin_z * cos_z;
    long double c_step = cos_z * cos_z - sin_z * sin_z;

    long double sin_k = s_step;
    long double cos_k = c_step;
    long double q2    = q * q;
    long double p     = q;
    long double mult  = q * q2;
    long double sign  = -1.0L;

    long double sum     = 1.0L;
    long double sum_c   = 0.0L;
    long double sum_abs = 1.0L;

    long double t4     = 1.0L;
    long double t4_c   = 0.0L;
    long double t4_abs = 1.0L;

    long double tail = (long double) DBL_MAX;

    for (size_t n = 1; n <= THETA_RG_MAX_TERMS; ++n) {
        long double coeff = 2.0L * p;
        theta_neumaier_add(coeff, &sum, &sum_c, &sum_abs);
        theta_neumaier_add(sign * coeff * cos_k, &t4, &t4_c, &t4_abs);
        out.terms_used = n;

        long double next_p    = p * mult;
        long double next_mult = mult * q2;
        if (next_p == 0.0L) {
            tail          = 0.0L;
            out.converged = true;
            break;
        }

        if (next_mult < 1.0L) {
            tail = (2.0L * next_p) / (1.0L - next_mult);
        }

        if (tail <= THETA_RG_ABS_TOL) {
            out.converged = true;
            break;
        }

        long double ns, nc;
        theta_sincos_step(sin_k, cos_k, s_step, c_step, &ns, &nc);
        sin_k         = ns;
        cos_k         = nc;
        sign          = -sign;
        p             = next_p;
        mult          = next_mult;
        out.converged = false;
    }

    out.theta3_zero     = sum + sum_c;
    out.theta4          = t4 + t4_c;
    out.err_theta3_zero = tail + 8.0L * LDBL_EPSILON * sum_abs;
    out.err_theta4      = tail + 8.0L * LDBL_EPSILON * t4_abs;
    return out;
}

/**
 * @brief Convert a theta status enum to a human-readable string.
 *
 * @param status Theta evaluation status.
 * @return Static string describing the status.
 */
const char* theta_status_string(theta_status status) {
    switch (status) {
        case THETA_STATUS_OK:
            return "ok";
        case THETA_STATUS_DOMAIN_ERROR:
            return "domain error";
        case THETA_STATUS_SINGULAR:
            return "singular or denominator unresolved";
        case THETA_STATUS_NO_CONVERGENCE:
            return "series did not satisfy tolerance before THETA_RG_MAX_TERMS";
        default:
            return "unknown";
    }
}

/**
 * @brief Evaluate theta1(z,q) normalized by theta2(0,q).
 *
 * @param z Phase in radians.
 * @param q Nome in [0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, error estimate, terms, and status.
 */
theta_eval theta1_norm_eval(double z, double q, double s1, double c1) {
    if (!theta_valid_q_closed(q)) {
        return theta_make_error_eval(0, THETA_STATUS_DOMAIN_ERROR);
    }

    long double sin_z, cos_z;
    theta_prepare_angles(z, s1, c1, &sin_z, &cos_z);

    if (q == 0.0) {
        return theta_make_eval(sin_z, 0.0L, 0, THETA_STATUS_OK);
    }

    theta_half_result odd = theta_half_series((long double) q, sin_z, cos_z);

    long double value, err;
    if (!theta_ratio_bound(
            odd.theta1, odd.err_theta1, odd.theta2_zero, odd.err_theta2_zero, &value, &err)) {
        return theta_make_error_eval(odd.terms_used, THETA_STATUS_SINGULAR);
    }

    theta_status status = odd.converged ? THETA_STATUS_OK : THETA_STATUS_NO_CONVERGENCE;
    return theta_make_eval(value, err, odd.terms_used, status);
}

/**
 * @brief Evaluate the square-like theta quotient.
 *
 * Combines odd and even theta components as a product of two bounded ratios.
 *
 * @param z Phase in radians.
 * @param q Nome in [0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, error estimate, terms, and status.
 */
theta_eval theta_square_eval(double z, double q, double s1, double c1) {
    if (!theta_valid_q_closed(q)) {
        return theta_make_error_eval(0, THETA_STATUS_DOMAIN_ERROR);
    }

    long double sin_z, cos_z;
    theta_prepare_angles(z, s1, c1, &sin_z, &cos_z);

    if (q == 0.0) {
        return theta_make_eval(sin_z, 0.0L, 0, THETA_STATUS_OK);
    }

    theta_half_result odd  = theta_half_series((long double) q, sin_z, cos_z);
    theta_even_result even = theta_even_series((long double) q, sin_z, cos_z);

    long double ratio_a, err_a;
    if (!theta_ratio_bound(even.theta3_zero,
                           even.err_theta3_zero,
                           odd.theta2_zero,
                           odd.err_theta2_zero,
                           &ratio_a,
                           &err_a)) {
        return theta_make_error_eval((odd.terms_used > even.terms_used) ? odd.terms_used
                                                                        : even.terms_used,
                                     THETA_STATUS_SINGULAR);
    }

    long double ratio_b, err_b;
    if (!theta_ratio_bound(
            odd.theta1, odd.err_theta1, even.theta4, even.err_theta4, &ratio_b, &err_b)) {
        return theta_make_error_eval((odd.terms_used > even.terms_used) ? odd.terms_used
                                                                        : even.terms_used,
                                     THETA_STATUS_SINGULAR);
    }

    long double value, err;
    theta_product_bound(ratio_a, err_a, ratio_b, err_b, &value, &err);

    theta_status status = THETA_STATUS_OK;
    if (!odd.converged || !even.converged) {
        status = THETA_STATUS_NO_CONVERGENCE;
    }

    size_t terms = (odd.terms_used > even.terms_used) ? odd.terms_used : even.terms_used;
    return theta_make_eval(value, err, terms, status);
}

/**
 * @brief Evaluate the phase-based saw map `(1/pi) arg(1 - q exp(-i z))`.
 *
 * @param z Phase in radians.
 * @param q Nome in [0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, small roundoff estimate, and status.
 */
theta_eval theta_saw_unit_eval(double z, double q, double s1, double c1) {
    (void) z;

    if (!theta_valid_q_closed(q)) {
        return theta_make_error_eval(0, THETA_STATUS_DOMAIN_ERROR);
    }

    long double sin_z, cos_z;
    theta_prepare_angles(z, s1, c1, &sin_z, &cos_z);

    long double y     = (long double) q * sin_z;
    long double x     = 1.0L - (long double) q * cos_z;
    long double value = atan2l(y, x) / THETA_PI;

    return theta_make_eval(value, 8.0L * LDBL_EPSILON, 0, THETA_STATUS_OK);
}

/**
 * @brief Evaluate the triangle-like odd-harmonic theta/Lambert series.
 *
 * Requires `0 < q < 1` because the formula uses `log(q)` and the Lambert
 * denominator `1 - q^k`.
 *
 * @param z Phase in radians.
 * @param q Nome in (0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, error estimate, terms, and status.
 */
theta_eval theta_triangle_eval(double z, double q, double s1, double c1) {
    if (!theta_valid_q_open(q)) {
        return theta_make_error_eval(0, THETA_STATUS_DOMAIN_ERROR);
    }

    long double sin_z, cos_z;
    theta_prepare_angles(z, s1, c1, &sin_z, &cos_z);

    long double ql        = (long double) q;
    long double lnq       = logl(ql);
    long double scale     = (-lnq) / sqrtl(ql);
    long double prefactor = 8.0L / (THETA_PI * THETA_PI);

    long double s_step = 2.0L * sin_z * cos_z;
    long double c_step = cos_z * cos_z - sin_z * sin_z;

    long double sin_k = sin_z;
    long double cos_k = cos_z;
    long double sign  = 1.0L;
    long double qpow  = sqrtl(ql);
    long double k_lnq = lnq;

    long double sum        = 0.0L;
    long double sum_c      = 0.0L;
    long double sum_abs    = 0.0L;
    long double tail       = (long double) DBL_MAX;
    bool        converged  = false;
    size_t      terms_used = 0;

    for (size_t n = 0; n < THETA_RG_MAX_TERMS; ++n) {
        long double k     = (long double) (2u * n + 1u);
        long double denom = -expm1l(k_lnq); /* 1 - q^k */
        long double coeff = (qpow / (k * k)) / denom;
        long double term  = sign * sin_k * coeff;
        theta_neumaier_add(term, &sum, &sum_c, &sum_abs);
        terms_used = n + 1;

        long double next_qpow  = qpow * ql;
        long double next_k_lnq = k_lnq + 2.0L * lnq;
        long double next_k     = k + 2.0L;
        long double next_denom = -expm1l(next_k_lnq);
        long double next_coeff =
            (next_qpow == 0.0L) ? 0.0L : (next_qpow / (next_k * next_k)) / next_denom;
        tail = next_coeff / (1.0L - ql);

        if (prefactor * scale * tail <= THETA_RG_ABS_TOL) {
            converged = true;
            break;
        }

        long double ns, nc;
        theta_sincos_step(sin_k, cos_k, s_step, c_step, &ns, &nc);
        sin_k = ns;
        cos_k = nc;
        sign  = -sign;
        qpow  = next_qpow;
        k_lnq = next_k_lnq;
    }

    long double  value  = prefactor * scale * (sum + sum_c);
    long double  err    = prefactor * scale * (tail + 8.0L * LDBL_EPSILON * sum_abs);
    theta_status status = converged ? THETA_STATUS_OK : THETA_STATUS_NO_CONVERGENCE;
    return theta_make_eval(value, err, terms_used, status);
}

/**
 * @brief Convenience wrapper returning only normalized theta1 value.
 *
 * @param z Phase in radians.
 * @param q Nome in [0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, or quiet NaN for domain/singularity failures.
 */
double theta1_norm(double z, double q, double s1, double c1) {
    theta_eval out = theta1_norm_eval(z, q, s1, c1);
    return (out.status == THETA_STATUS_DOMAIN_ERROR || out.status == THETA_STATUS_SINGULAR)
               ? theta_quiet_nan()
               : out.value;
}

/**
 * @brief Convenience wrapper returning only the square-like theta quotient.
 *
 * @param z Phase in radians.
 * @param q Nome in [0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, or quiet NaN for domain/singularity failures.
 */
double theta_square(double z, double q, double s1, double c1) {
    theta_eval out = theta_square_eval(z, q, s1, c1);
    return (out.status == THETA_STATUS_DOMAIN_ERROR || out.status == THETA_STATUS_SINGULAR)
               ? theta_quiet_nan()
               : out.value;
}

/**
 * @brief Convenience wrapper returning only the saw-like theta value.
 *
 * @param z Phase in radians.
 * @param q Nome in [0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, or quiet NaN for domain/singularity failures.
 */
double theta_saw_unit(double z, double q, double s1, double c1) {
    theta_eval out = theta_saw_unit_eval(z, q, s1, c1);
    return (out.status == THETA_STATUS_DOMAIN_ERROR || out.status == THETA_STATUS_SINGULAR)
               ? theta_quiet_nan()
               : out.value;
}

/**
 * @brief Convenience wrapper returning only the triangle-like theta value.
 *
 * @param z Phase in radians.
 * @param q Nome in (0, 1).
 * @param s1 Optional precomputed sine of z.
 * @param c1 Optional precomputed cosine of z.
 * @return Value, or quiet NaN for domain/singularity failures.
 */
double theta_triangle(double z, double q, double s1, double c1) {
    theta_eval out = theta_triangle_eval(z, q, s1, c1);
    return (out.status == THETA_STATUS_DOMAIN_ERROR || out.status == THETA_STATUS_SINGULAR)
               ? theta_quiet_nan()
               : out.value;
}
