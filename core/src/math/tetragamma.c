/**
 * @file tetragamma.c
 * @brief Real and complex tetragamma implementations with fixed Stirling tails.
 *
 * Tetragamma is the second derivative of digamma. These paths share the
 * polygamma reflection/recurrence machinery but use the differentiated
 * Stirling tail appropriate for psi_2.
 */

#include "internal/polygamma_internal.h"

static inline double tetragamma_core_f64(double x, int tail_terms);
static inline float  tetragamma_core_f32(float x, int tail_terms);
static SimComplexDouble tetragamma_core_c64(SimComplexDouble z, int tail_terms);

/**
 * @brief Evaluate double-precision tetragamma with a 12-term Stirling tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Approximation to psi_2(x).
 */
double sim_tetragamma_f64_12(double x) {
    return tetragamma_core_f64(x, 12);
}

/**
 * @brief Evaluate double-precision tetragamma with a 7-term Stirling tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Approximation to psi_2(x) using fewer Bernoulli coefficients.
 */
double sim_tetragamma_f64_7(double x) {
    return tetragamma_core_f64(x, 7);
}

/**
 * @brief Evaluate double-precision tetragamma with a 5-term Stirling tail.
 *
 * @param x Real argument. Nonpositive integer poles return NaN.
 * @return Approximation to psi_2(x) optimized for lower tail cost.
 */
double sim_tetragamma_f64_5(double x) {
    return tetragamma_core_f64(x, 5);
}

/**
 * @brief Evaluate float tetragamma with the default float tail length.
 *
 * @param x Real single-precision argument.
 * @return Approximation to psi_2(x), rounded to float.
 */
float sim_tetragamma_f32_12(float x) {
    return tetragamma_core_f32(x, 8);
}

/**
 * @brief Evaluate float tetragamma with a 7-term tail.
 *
 * @param x Real single-precision argument.
 * @return Approximation to psi_2(x), rounded to float.
 */
float sim_tetragamma_f32_7(float x) {
    return tetragamma_core_f32(x, 7);
}

/**
 * @brief Evaluate float tetragamma with a 5-term tail.
 *
 * @param x Real single-precision argument.
 * @return Approximation to psi_2(x), rounded to float.
 */
float sim_tetragamma_f32_5(float x) {
    return tetragamma_core_f32(x, 5);
}

/**
 * @brief Core real tetragamma evaluator with reflection, recurrence, and fixed tail terms.
 *
 * @param x Real argument.
 * @param tail_terms Requested number of Bernoulli-pair coefficients.
 * @return Approximation to psi_2(x), or NaN for poles/invalid inputs.
 */
static inline double tetragamma_core_f64(double x, int tail_terms) {
    if (!isfinite(x))
        return NAN;

    double z   = x;
    double acc = 0.0;

#if !defined(SIM_TRIG_FREE)
    if (z <= 0.0) {
        if (sim_is_int(z))
            return NAN;
        double nearest = nearbyint(z);
        double dist    = fabs(z - nearest);
        if (dist < DBL_EPSILON)
            return NAN;
        double reflection = 2.0 * M_PI * M_PI * M_PI * sim_cotpi(z) * sim_cscpi_sq(z);
        return tetragamma_core_f64(1.0 - z, tail_terms) - reflection;
    }
#else
    int iter = 0;
    while (z <= 0.0) {
        if (sim_is_int(z))
            return NAN;
        acc -= 2.0 / (z * z * z);
        z += 1.0;
        if (++iter >= 1024)
            return NAN;
    }
#endif

    while (z + 2.0 < SIM_STIRLING_SHIFT_THRESHOLD) {
        acc -= 2.0 / (z * z * z);
        z += 1.0;
        acc -= 2.0 / (z * z * z);
        z += 1.0;
    }
    while (z < SIM_STIRLING_SHIFT_THRESHOLD) {
        acc -= 2.0 / (z * z * z);
        z += 1.0;
    }

    const double inv  = 1.0 / z;
    const double inv2 = inv * inv;
    const double inv3 = inv2 * inv;
    const double inv4 = inv2 * inv2;

    double series = -inv2 - inv3;
    int    N      = tail_terms;
    int    maxN   = (int) (sizeof(B2) / sizeof(B2[0]));
    if (N > maxN)
        N = maxN;
    if (N >= 1) {
        series += -(2.0 * 1.0 + 1.0) * B2[0] * inv4;
        double inv_pow = inv4 * inv2;
        for (int n = 2; n <= N; ++n) {
            double coeff = -(2.0 * n + 1.0) * B2[n - 1];
            series += coeff * inv_pow;
            inv_pow *= inv2;
        }
    }

    return acc + series;
}

/**
 * @brief Core float tetragamma evaluator with reflection, recurrence, and fixed tail terms.
 *
 * @param x Real single-precision argument.
 * @param tail_terms Requested number of float Bernoulli-pair coefficients.
 * @return Approximation to psi_2(x), or NaN for poles/invalid inputs.
 */
static inline float tetragamma_core_f32(float x, int tail_terms) {
    if (!isfinite(x))
        return NAN;

    float z   = x;
    float acc = 0.0f;

#if !defined(SIM_TRIG_FREE)
    if (z <= 0.0f) {
        if (sim_is_intf(z))
            return NAN;
        float nearest = nearbyintf(z);
        float dist    = fabsf(z - nearest);
        if (dist < FLT_EPSILON)
            return NAN;
        float reflection =
            2.0f * (float) M_PI * (float) M_PI * (float) M_PI * sim_cotpi_f(z) * sim_cscpi_sq_f(z);
        return tetragamma_core_f32(1.0f - z, tail_terms) - reflection;
    }
#else
    int iter = 0;
    while (z <= 0.0f) {
        if (sim_is_intf(z))
            return NAN;
        acc -= 2.0f / (z * z * z);
        z += 1.0f;
        if (++iter >= 1024)
            return NAN;
    }
#endif

    while (z + 2.0f < (float) SIM_STIRLING_SHIFT_THRESHOLD) {
        acc -= 2.0f / (z * z * z);
        z += 1.0f;
        acc -= 2.0f / (z * z * z);
        z += 1.0f;
    }
    while (z < (float) SIM_STIRLING_SHIFT_THRESHOLD) {
        acc -= 2.0f / (z * z * z);
        z += 1.0f;
    }

    const float inv  = 1.0f / z;
    const float inv2 = inv * inv;
    const float inv3 = inv2 * inv;
    const float inv4 = inv2 * inv2;

    float series = -inv2 - inv3;
    int   N      = tail_terms;
    int   maxN   = (int) (sizeof(B2f) / sizeof(B2f[0]));
    if (N > maxN)
        N = maxN;
    if (N >= 1) {
        series += -(2.0f * 1.0f + 1.0f) * B2f[0] * inv4;
        float inv_pow = inv4 * inv2;
        for (int n = 2; n <= N; ++n) {
            float coeff = -(2.0f * n + 1.0f) * B2f[n - 1];
            series += coeff * inv_pow;
            inv_pow *= inv2;
        }
    }

    return acc + series;
}

/**
 * @brief Core complex tetragamma evaluator with reflection and fixed tail terms.
 *
 * @param z Complex argument.
 * @param tail_terms Requested number of Bernoulli-pair coefficients.
 * @return Complex approximation to psi_2(z), or `{NAN, NAN}` for invalid inputs.
 */
static SimComplexDouble tetragamma_core_c64(SimComplexDouble z, int tail_terms) {
    double complex cz = z.re + I * z.im;

    if (!isfinite(z.re) || !isfinite(z.im))
        return (SimComplexDouble) { NAN, NAN };

    if (creal(cz) < 0.5) {
#if SIM_DIAGNOSTICS
        sim_special_diag_track_reflection();
#endif
        SimComplexDouble one_minus = { 1.0 - z.re, -z.im };
        SimComplexDouble psi2_ref  = tetragamma_core_c64(one_minus, tail_terms);

        double complex s;
        double complex c;
        sim_sincospi_complex(z.re, z.im, &s, &c);
        double complex cot        = c / s;
        double complex inv_s      = M_PI / s;
        double complex csc2       = inv_s * inv_s;
        double complex reflection = 2.0 * M_PI * M_PI * M_PI * cot * csc2;

        double complex out = (psi2_ref.re + I * psi2_ref.im) - reflection;
        return (SimComplexDouble) { creal(out), cimag(out) };
    }

    double complex acc = 0.0 + 0.0 * I;
    double complex w   = cz;
    while (creal(w) < SIM_STIRLING_SHIFT_THRESHOLD && cabs(w) < 64.0) {
        acc -= 2.0 / (w * w * w);
        w += 1.0;
    }

    double complex inv  = 1.0 / w;
    double complex inv2 = inv * inv;
    double complex inv3 = inv2 * inv;
    double complex inv4 = inv2 * inv2;

    double complex series = -inv2 - inv3;
    int            N      = tail_terms;
    int            maxN   = (int) (sizeof(B2) / sizeof(B2[0]));
    if (N > maxN)
        N = maxN;
    if (N >= 1) {
        series += -(2.0 * 1.0 + 1.0) * B2[0] * inv4;
        double complex pow = inv4 * inv2;
        for (int n = 2; n <= N; ++n) {
            double coeff = -(2.0 * n + 1.0) * B2[n - 1];
            series += coeff * pow;
            pow *= inv2;
        }
    }

    double complex out = acc + series;
    return (SimComplexDouble) { creal(out), cimag(out) };
}

/**
 * @brief Evaluate complex double tetragamma with a 12-term Stirling tail.
 *
 * @param z Complex argument.
 * @return Complex approximation to psi_2(z).
 */
SimComplexDouble sim_tetragamma_c64_12(SimComplexDouble z) {
    return tetragamma_core_c64(z, 12);
}

/**
 * @brief Evaluate complex double tetragamma with a 7-term Stirling tail.
 *
 * @param z Complex argument.
 * @return Complex approximation to psi_2(z).
 */
SimComplexDouble sim_tetragamma_c64_7(SimComplexDouble z) {
    return tetragamma_core_c64(z, 7);
}

/**
 * @brief Evaluate complex double tetragamma with a 5-term Stirling tail.
 *
 * @param z Complex argument.
 * @return Complex approximation to psi_2(z).
 */
SimComplexDouble sim_tetragamma_c64_5(SimComplexDouble z) {
    return tetragamma_core_c64(z, 5);
}

/**
 * @brief Evaluate complex float tetragamma with a 12-term double backend.
 *
 * @param zf Complex single-precision argument.
 * @return Complex approximation to psi_2(zf), rounded to float components.
 */
SimComplexFloat sim_tetragamma_c32_12(SimComplexFloat zf) {
    SimComplexDouble zd = { (double) zf.re, (double) zf.im };
    return sim_complex_to_c32(tetragamma_core_c64(zd, 12));
}

/**
 * @brief Evaluate complex float tetragamma with a 7-term double backend.
 *
 * @param zf Complex single-precision argument.
 * @return Complex approximation to psi_2(zf), rounded to float components.
 */
SimComplexFloat sim_tetragamma_c32_7(SimComplexFloat zf) {
    SimComplexDouble zd = { (double) zf.re, (double) zf.im };
    return sim_complex_to_c32(tetragamma_core_c64(zd, 7));
}

/**
 * @brief Evaluate complex float tetragamma with a 5-term double backend.
 *
 * @param zf Complex single-precision argument.
 * @return Complex approximation to psi_2(zf), rounded to float components.
 */
SimComplexFloat sim_tetragamma_c32_5(SimComplexFloat zf) {
    SimComplexDouble zd = { (double) zf.re, (double) zf.im };
    return sim_complex_to_c32(tetragamma_core_c64(zd, 5));
}
