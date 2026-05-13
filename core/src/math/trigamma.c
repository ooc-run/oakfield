/**
 * @file trigamma.c
 * @brief Real and complex trigamma implementations with fixed, adaptive, and fast tails.
 *
 * Trigamma is the first derivative of digamma. The implementation mirrors the
 * digamma structure: reflection for negative-side arguments, recurrence shifts
 * into a stable asymptotic region, and Bernoulli/Stirling tails for evaluation.
 */

#include "internal/polygamma_internal.h"

static inline double trigamma_core_f64(double x, int tail_terms);
static inline float  trigamma_core_f32(float x, int tail_terms);
static SimComplexDouble trigamma_core_c64(SimComplexDouble z, int tail_terms);

/**
 * @brief Evaluate double-precision trigamma with a 12-term Stirling tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Approximation to psi_1(x).
 */
double sim_trigamma_f64_12(double x) {
    return trigamma_core_f64(x, 12);
}

/**
 * @brief Evaluate double-precision trigamma with a 7-term Stirling tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Approximation to psi_1(x) using fewer Bernoulli coefficients.
 */
double sim_trigamma_f64_7(double x) {
    return trigamma_core_f64(x, 7);
}

/**
 * @brief Evaluate double-precision trigamma with a 5-term Stirling tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Approximation to psi_1(x) optimized for lower tail cost.
 */
double sim_trigamma_f64_5(double x) {
    return trigamma_core_f64(x, 5);
}

/**
 * @brief Evaluate float trigamma with the default float tail length.
 *
 * @param x Real single-precision argument.
 * @return Approximation to psi_1(x), rounded to float.
 */
float sim_trigamma_f32_12(float x) {
    return trigamma_core_f32(x, 8);
}

/**
 * @brief Evaluate float trigamma with a 7-term tail.
 *
 * @param x Real single-precision argument.
 * @return Approximation to psi_1(x), rounded to float.
 */
float sim_trigamma_f32_7(float x) {
    return trigamma_core_f32(x, 7);
}

/**
 * @brief Evaluate float trigamma with a 5-term tail.
 *
 * @param x Real single-precision argument.
 * @return Approximation to psi_1(x), rounded to float.
 */
float sim_trigamma_f32_5(float x) {
    return trigamma_core_f32(x, 5);
}

/**
 * @brief Evaluate double-precision trigamma using a fast asymptotic approximation.
 *
 * This backend shifts positive inputs to at least 3, evaluates the leading
 * asymptotic terms, then unwinds recurrence shifts. It is intended for fast,
 * approximate paths.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Fast approximation to psi_1(x).
 */
double sim_trigamma_f64_mortici(double x) {
    if (!isfinite(x))
        return NAN;
    if (x <= 0.0) {
        if (sim_is_int(x))
            return NAN;
        return (M_PI * M_PI) * sim_cscpi_sq(x) - sim_trigamma_f64_mortici(1.0 - x);
    }

    int nshift = 0;
    while (x < 3.0) {
        x += 1.0;
        nshift++;
    }

    double inv  = 1.0 / x;
    double inv2 = inv * inv;
    double inv3 = inv2 * inv;
    double y    = inv + 0.5 * inv2 + (1.0 / 6.0) * inv3;
    while (nshift--) {
        x -= 1.0;
        y -= 1.0 / (x * x);
    }
    return y;
}

/**
 * @brief Evaluate float trigamma using a fast asymptotic approximation.
 *
 * @param x Real single-precision argument. Nonpositive integer poles return NaN.
 * @return Fast approximation to psi_1(x), rounded to float.
 */
float sim_trigamma_f32_mortici(float x) {
    if (!isfinite(x))
        return NAN;
    if (x <= 0.0f) {
        if (sim_is_intf(x))
            return NAN;
        return ((float) M_PI * (float) M_PI) * sim_cscpi_sq_f(x) -
               sim_trigamma_f32_mortici(1.0f - x);
    }

    int nshift = 0;
    while (x < 3.0f) {
        x += 1.0f;
        nshift++;
    }

    float inv  = 1.0f / x;
    float inv2 = inv * inv;
    float inv3 = inv2 * inv;
    float y    = inv + 0.5f * inv2 + (1.0f / 6.0f) * inv3;
    while (nshift--) {
        x -= 1.0f;
        y -= 1.0f / (x * x);
    }
    return y;
}

/**
 * @brief Evaluate double-precision trigamma with an adaptive Bernoulli tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @param tol Absolute stopping tolerance for the asymptotic tail.
 * @return Approximation to psi_1(x), or NaN for invalid inputs/poles.
 */
double sim_trigamma_f64_tail(double x, double tol) {
    if (!isfinite(x))
        return NAN;
    if (x <= 0.0) {
        if (sim_is_int(x))
            return NAN;
#if defined(SIM_TRIG_FREE)
        double z     = x;
        double shift = 0.0;
        int    iter  = 0;
        while (z <= 0.0) {
            shift += 1.0 / (z * z);
            z += 1.0;
            if (z <= 0.0 && sim_is_int(z))
                return NAN;
            if (++iter >= 1024)
                return NAN;
        }
        return shift + sim_trigamma_f64_tail(z, tol);
#else
        double nearest = nearbyint(x);
        double dist    = fabs(x - nearest);
        if (dist < DBL_EPSILON)
            return NAN;
#if SIM_DIAGNOSTICS
        sim_special_diag_track_reflection();
        sim_special_diag_track_pole_distance(dist);
#endif
        return (M_PI * M_PI) * sim_cscpi_sq(x) - sim_trigamma_f64_tail(1.0 - x, tol);
#endif
    }

    double y           = 0.0;
    double z           = x;
    double shift_accum = 0.0;
    while (z + 2.0 < SIM_STIRLING_SHIFT_THRESHOLD) {
        y += 1.0 / (z * z);
        shift_accum += 1.0;
        z += 1.0;
        y += 1.0 / (z * z);
        shift_accum += 1.0;
        z += 1.0;
    }
    while (z < SIM_STIRLING_SHIFT_THRESHOLD) {
        y += 1.0 / (z * z);
        shift_accum += 1.0;
        z += 1.0;
    }

    const double inv = 1.0 / z;
    const double inv2 = inv * inv;
    double       tail    = inv + 0.5 * inv2;
    double       inv_pow = inv2 * inv;
    for (int n = 1; n < 100; ++n) {
        double term = B2[n - 1] * inv_pow;
        tail += term;
        double bound = fabs(term * inv2 / (1.0 - inv2));
        if (bound < tol)
            break;
        inv_pow *= inv2;
    }
    y += tail;
#if SIM_DIAGNOSTICS
    sim_special_diag_track_recurrence(shift_accum);
    sim_special_diag_track_stirling_tail(tail);
#else
    (void) shift_accum;
#endif
    return y;
}

/**
 * @brief Core real trigamma evaluator with reflection, recurrence, and fixed tail terms.
 *
 * @param x Real argument.
 * @param tail_terms Requested number of Bernoulli-pair coefficients.
 * @return Approximation to psi_1(x), or NaN for poles/invalid inputs.
 */
static inline double trigamma_core_f64(double x, int tail_terms) {
    if (!isfinite(x))
        return NAN;

    double acc = 0.0;
    double z   = x;
#if !defined(SIM_TRIG_FREE)
    if (z <= 0.0) {
        if (sim_is_int(z))
            return NAN;
        double nearest = nearbyint(z);
        double dist    = fabs(z - nearest);
        if (dist < DBL_EPSILON)
            return NAN;
        return (M_PI * M_PI) * sim_cscpi_sq(z) - trigamma_core_f64(1.0 - z, tail_terms);
    }
#else
    int iter = 0;
    while (z <= 0.0) {
        if (sim_is_int(z))
            return NAN;
        acc += 1.0 / (z * z);
        z += 1.0;
        if (z <= 0.0 && sim_is_int(z))
            return NAN;
        if (++iter >= 1024)
            return NAN;
    }
#endif

    while (z + 2.0 < SIM_STIRLING_SHIFT_THRESHOLD) {
        acc += 1.0 / (z * z);
        z += 1.0;
        acc += 1.0 / (z * z);
        z += 1.0;
    }
    while (z < SIM_STIRLING_SHIFT_THRESHOLD) {
        acc += 1.0 / (z * z);
        z += 1.0;
    }

    const double inv = 1.0 / z;
    const double inv2 = inv * inv;
    double       series  = inv + 0.5 * inv2;
    double       inv_pow = inv2 * inv;
    int          N       = tail_terms;
    int          maxN    = (int) (sizeof(B2) / sizeof(B2[0]));
    if (N > maxN)
        N = maxN;
    for (int n = 1; n <= N; ++n) {
        series = FMA(B2[n - 1], inv_pow, series);
        inv_pow *= inv2;
    }
    return acc + series;
}

/**
 * @brief Core float trigamma evaluator with reflection, recurrence, and fixed tail terms.
 *
 * @param x Real single-precision argument.
 * @param tail_terms Requested number of float Bernoulli-pair coefficients.
 * @return Approximation to psi_1(x), or NaN for poles/invalid inputs.
 */
static inline float trigamma_core_f32(float x, int tail_terms) {
    if (!isfinite(x))
        return NAN;

    float acc = 0.0f;
    float z   = x;
#if !defined(SIM_TRIG_FREE)
    if (z <= 0.0f) {
        if (sim_is_intf(z))
            return NAN;
        float nearest = nearbyintf(z);
        float dist    = fabsf(z - nearest);
        if (dist < FLT_EPSILON)
            return NAN;
        return ((float) M_PI * (float) M_PI) * sim_cscpi_sq_f(z) -
               trigamma_core_f32(1.0f - z, tail_terms);
    }
#else
    int iter = 0;
    while (z <= 0.0f) {
        if (sim_is_intf(z))
            return NAN;
        acc += 1.0f / (z * z);
        z += 1.0f;
        if (z <= 0.0f && sim_is_intf(z))
            return NAN;
        if (++iter >= 1024)
            return NAN;
    }
#endif

    while (z + 2.0f < (float) SIM_STIRLING_SHIFT_THRESHOLD) {
        acc += 1.0f / (z * z);
        z += 1.0f;
        acc += 1.0f / (z * z);
        z += 1.0f;
    }
    while (z < (float) SIM_STIRLING_SHIFT_THRESHOLD) {
        acc += 1.0f / (z * z);
        z += 1.0f;
    }

    float inv     = 1.0f / z;
    float inv2    = inv * inv;
    float series  = inv + 0.5f * inv2;
    float inv_pow = inv2 * inv;
    int   N       = tail_terms;
    int   maxN    = (int) (sizeof(B2f) / sizeof(B2f[0]));
    if (N > maxN)
        N = maxN;
    for (int n = 1; n <= N; ++n) {
        series = FMA(B2f[n - 1], inv_pow, series);
        inv_pow *= inv2;
    }
    return acc + series;
}

/**
 * @brief Core complex trigamma evaluator with reflection and fixed tail terms.
 *
 * @param z Complex argument.
 * @param tail_terms Requested number of Bernoulli-pair coefficients.
 * @return Complex approximation to psi_1(z), or `{NAN, NAN}` for invalid inputs.
 */
static SimComplexDouble trigamma_core_c64(SimComplexDouble z, int tail_terms) {
    double complex cz = z.re + I * z.im;

    if (!isfinite(z.re) || !isfinite(z.im))
        return (SimComplexDouble) { NAN, NAN };

    if (creal(cz) < 0.5) {
#if SIM_DIAGNOSTICS
        sim_special_diag_track_reflection();
#endif
        SimComplexDouble one_minus = { 1.0 - z.re, -z.im };
        SimComplexDouble psi1_ref  = trigamma_core_c64(one_minus, tail_terms);

        double complex s;
        double complex c;
        sim_sincospi_complex(z.re, z.im, &s, &c);
        double complex inv_s = M_PI / s;
        double complex csc2  = inv_s * inv_s;
        double complex out   = csc2 - (psi1_ref.re + I * psi1_ref.im);
        return (SimComplexDouble) { creal(out), cimag(out) };
    }

    double complex acc         = 0.0 + 0.0 * I;
    double complex w           = cz;
    double         shift_accum = 0.0;
    while (creal(w) < SIM_STIRLING_SHIFT_THRESHOLD && cabs(w) < 64.0) {
        acc += 1.0 / (w * w);
        w += 1.0;
        shift_accum += 1.0;
    }

    double complex inv = 1.0 / w;
    double complex inv2 = inv * inv;
    double complex inv3 = inv2 * inv;
    double complex series = inv + 0.5 * inv2;
    double complex pow    = inv3;
    int            N      = tail_terms;
    int            maxN   = (int) (sizeof(B2) / sizeof(B2[0]));
    if (N > maxN)
        N = maxN;
    for (int n = 1; n <= N; ++n) {
        series += B2[n - 1] * pow;
        pow *= inv2;
    }

    double complex out = acc + series;
#if SIM_DIAGNOSTICS
    if (shift_accum > 0.0)
        sim_special_diag_track_recurrence(shift_accum);
    sim_special_diag_track_stirling_tail(creal(series));
#else
    (void) shift_accum;
#endif
    return (SimComplexDouble) { creal(out), cimag(out) };
}

/**
 * @brief Evaluate complex double trigamma with a 12-term Stirling tail.
 *
 * @param z Complex argument.
 * @return Complex approximation to psi_1(z).
 */
SimComplexDouble sim_trigamma_c64_12(SimComplexDouble z) {
    return trigamma_core_c64(z, 12);
}

/**
 * @brief Evaluate complex double trigamma with a 7-term Stirling tail.
 *
 * @param z Complex argument.
 * @return Complex approximation to psi_1(z).
 */
SimComplexDouble sim_trigamma_c64_7(SimComplexDouble z) {
    return trigamma_core_c64(z, 7);
}

/**
 * @brief Evaluate complex double trigamma with a 5-term Stirling tail.
 *
 * @param z Complex argument.
 * @return Complex approximation to psi_1(z).
 */
SimComplexDouble sim_trigamma_c64_5(SimComplexDouble z) {
    return trigamma_core_c64(z, 5);
}

/**
 * @brief Evaluate complex float trigamma with a 12-term double backend.
 *
 * @param zf Complex single-precision argument.
 * @return Complex approximation to psi_1(zf), rounded to float components.
 */
SimComplexFloat sim_trigamma_c32_12(SimComplexFloat zf) {
    SimComplexDouble zd = { (double) zf.re, (double) zf.im };
    return sim_complex_to_c32(trigamma_core_c64(zd, 12));
}

/**
 * @brief Evaluate complex float trigamma with a 7-term double backend.
 *
 * @param zf Complex single-precision argument.
 * @return Complex approximation to psi_1(zf), rounded to float components.
 */
SimComplexFloat sim_trigamma_c32_7(SimComplexFloat zf) {
    SimComplexDouble zd = { (double) zf.re, (double) zf.im };
    return sim_complex_to_c32(trigamma_core_c64(zd, 7));
}

/**
 * @brief Evaluate complex float trigamma with a 5-term double backend.
 *
 * @param zf Complex single-precision argument.
 * @return Complex approximation to psi_1(zf), rounded to float components.
 */
SimComplexFloat sim_trigamma_c32_5(SimComplexFloat zf) {
    SimComplexDouble zd = { (double) zf.re, (double) zf.im };
    return sim_complex_to_c32(trigamma_core_c64(zd, 5));
}

/**
 * @brief Evaluate complex double trigamma by increasing tail length until convergence.
 *
 * @param z Complex argument.
 * @param tol Absolute convergence threshold on successive approximations.
 * @return Best complex approximation found by the tail sweep.
 */
SimComplexDouble sim_trigamma_c64_tail(SimComplexDouble z, double tol) {
    int              N    = 3;
    SimComplexDouble last = trigamma_core_c64(z, N);
    for (; N < 60; ++N) {
        SimComplexDouble cur = trigamma_core_c64(z, N);
        if (fabs(cur.re - last.re) + fabs(cur.im - last.im) < tol)
            return cur;
        last = cur;
    }
    return last;
}
