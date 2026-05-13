/**
 * @file polygamma_internal.h
 * @brief Shared private helpers for polygamma, finite ladder, and q-analog code.
 *
 * This header centralizes numeric constants, Bernoulli table declarations,
 * diagnostics hooks, fallback-report plumbing, and inline trigonometric helpers
 * used by the special-function implementation files.
 */
#ifndef SIM_POLYGAMMA_INTERNAL_H
#define SIM_POLYGAMMA_INTERNAL_H

#include <stdbool.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "oakfield/math/special_functions.h"

#ifndef DBL_TRUE_MIN
#define DBL_TRUE_MIN DBL_MIN
#endif

#ifndef FLT_TRUE_MIN
#define FLT_TRUE_MIN FLT_MIN
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_INV_PI
#define M_INV_PI 0.31830988618379067153776752674502872
#endif

#define SIM_POLYGAMMA_INV_TWO_PI (0.5 * M_INV_PI)
#define SIM_DIGAMMA_SQUARE_A_EPS 1.0e-6

#ifndef SIM_STIRLING_SHIFT_THRESHOLD
#define SIM_STIRLING_SHIFT_THRESHOLD 12.0 /* shift until |z| >= this before asymptotic */
#endif

#ifndef SIM_HAVE_SYSTEM_TRIGPI
#define SIM_HAVE_SYSTEM_TRIGPI 0 /* 0 = use polynomial sinpi/cospi for real; complex uses exp */
#endif

/*------------------------------------------------------------------------------
 * q-analogs configuration
 *----------------------------------------------------------------------------*/
#ifndef SIM_Q_ANALOG_DEFAULT_TOL
#define SIM_Q_ANALOG_DEFAULT_TOL 1e-12
#endif

#ifndef SIM_Q_ANALOG_MAX_TERMS
#define SIM_Q_ANALOG_MAX_TERMS 200000
#endif

/*------------------------------------------------------------------------------
 * FMA detection
 *----------------------------------------------------------------------------*/
#if defined(__has_builtin)
#if __has_builtin(__builtin_fma)
#undef FMA
#define FMA(a, b, c) __builtin_fma((a), (b), (c))
#endif
#endif

/*------------------------------------------------------------------------------
 * FMA helper
 *----------------------------------------------------------------------------*/
#ifndef FMA
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#if defined(__FMA__) || defined(__AVX2__) || defined(__FMA4__)
#define FMA(a, b, c) fma((a), (b), (c))
#else
#define FMA(a, b, c) ((a) * (b) + (c))
#endif
#else
#define FMA(a, b, c) ((a) * (b) + (c))
#endif
#endif

/*------------------------------------------------------------------------------
 * Compiler attributes & alignment helpers
 *----------------------------------------------------------------------------*/
#if defined(__GNUC__) || defined(__clang__)
#define SIM_ATTR_CONST __attribute__((const))
#define SIM_ALIGN64_PRE
#define SIM_ALIGN64_POST __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define SIM_ATTR_CONST
#define SIM_ALIGN64_PRE __declspec(align(64))
#define SIM_ALIGN64_POST
#else
#define SIM_ATTR_CONST
#define SIM_ALIGN64_PRE
#define SIM_ALIGN64_POST
#endif

#define SIM_POLYGAMMA_B2_COUNT 12
#define SIM_POLYGAMMA_B2F_COUNT 8
#define SIM_POLYGAMMA_C2_COUNT 12
#define SIM_POLYGAMMA_C2F_COUNT 8

extern const double sim_polygamma_B2[SIM_POLYGAMMA_B2_COUNT];
extern const float  sim_polygamma_B2f[SIM_POLYGAMMA_B2F_COUNT];
extern const double sim_polygamma_C2[SIM_POLYGAMMA_C2_COUNT];
extern const float  sim_polygamma_C2f[SIM_POLYGAMMA_C2F_COUNT];

#define B2 sim_polygamma_B2
#define B2f sim_polygamma_B2f
#define C2 sim_polygamma_C2
#define C2f sim_polygamma_C2f

#if SIM_DIAGNOSTICS
void sim_special_diag_track_pole_distance(double distance);
void sim_special_diag_track_recurrence(double shift_magnitude);
void sim_special_diag_track_stirling_tail(double tail);
void sim_special_diag_track_reflection(void);
#else
static inline void sim_special_diag_track_pole_distance(double distance) { (void) distance; }
static inline void sim_special_diag_track_recurrence(double shift_magnitude) { (void) shift_magnitude; }
static inline void sim_special_diag_track_stirling_tail(double tail) { (void) tail; }
static inline void sim_special_diag_track_reflection(void) {}
#endif

void sim_special_report_seed(SimSpecialEvalReport* report,
                             const char*           function,
                             SimComplexDouble      input,
                             double                q,
                             double                aux,
                             double                exponent,
                             double                tol);
void sim_special_report_update(SimSpecialEvalReport* report,
                               SimSpecialFault       fault,
                               int                   iteration,
                               double                residual);
SimResult sim_special_apply_fallback(SimSpecialFallbackFn        fallback,
                                     void*                       userdata,
                                     const SimSpecialEvalReport* report,
                                     SimComplexDouble*           value_out);

SimComplexDouble sim_hyperexp_phi_deriv_complex_opt(SimComplexDouble lambda,
                                                    SimComplexDouble epsilon,
                                                    int              K);

/**
 * @brief Test whether a double is an integer to local floating tolerance.
 *
 * @param x Candidate real value.
 * @return true when `x` is close enough to an integer for pole detection.
 */
static inline bool sim_is_int(double x) SIM_ATTR_CONST;
static inline bool sim_is_int(double x) {
    double ix;
    double frac = modf(x, &ix);
    return fabs(frac) <= DBL_EPSILON * fmax(1.0, fabs(ix));
}

/**
 * @brief Test whether a float is an integer to local floating tolerance.
 *
 * @param x Candidate real value.
 * @return true when `x` is close enough to an integer for pole detection.
 */
static inline bool sim_is_intf(float x) SIM_ATTR_CONST;
static inline bool sim_is_intf(float x) {
    float ix;
    float frac = modff(x, &ix);
    return fabsf(frac) <= FLT_EPSILON * fmaxf(1.0f, fabsf(ix));
}

/**
 * @brief Check whether both components of a complex ABI value are finite.
 *
 * @param z Complex value stored in `SimComplexDouble`.
 * @return true when real and imaginary components are finite.
 */
static inline bool sim_complex_isfinite(SimComplexDouble z) {
    return isfinite(z.re) && isfinite(z.im);
}

/*------------------------------------------------------------------------------
 * Real sinpi/cospi using quadrant reduction and polynomial kernel (no libm trig)
 * sin(pi x) and cos(pi x) with x reduced to r in [-0.5,0.5].
 *----------------------------------------------------------------------------*/
#if !SIM_HAVE_SYSTEM_TRIGPI
static inline double sim_sinpi(double x) SIM_ATTR_CONST;
static inline double sim_cospi(double x) SIM_ATTR_CONST;
static inline double sim_cotpi(double x) SIM_ATTR_CONST;
static inline double sim_cscpi_sq(double x) SIM_ATTR_CONST;
#else
static inline double sim_sinpi(double x) SIM_ATTR_CONST;
static inline double sim_cospi(double x) SIM_ATTR_CONST;
static inline double sim_cotpi(double x) SIM_ATTR_CONST;
static inline double sim_cscpi_sq(double x) SIM_ATTR_CONST;
#endif
#if !SIM_HAVE_SYSTEM_TRIGPI
/**
 * @brief Reduce a real argument for polynomial sin(pi*x)/cos(pi*x) evaluation.
 *
 * @param x Input argument.
 * @param r Receives the reduced residual in approximately [-0.5, 0.5].
 * @param parity Receives the parity of the rounded integer part.
 */
static inline void sim_reduce_half(double x, double* r, int* parity) {
    /* reduce to r in [-0.5,0.5] using round-to-nearest integer */
    double n = nearbyint(x);
    *r       = x - n;
    int64_t k;
    if (x >= 0)
        k = (int64_t) fmod(n, 2.0);
    else
        k = (int64_t) fmod(-n, 2.0);
    *parity = (int) (k & 1);
}
/**
 * @brief Evaluate sine and cosine of pi times a reduced residual.
 *
 * @param r Reduced residual in approximately [-0.5, 0.5].
 * @param s Receives `sin(pi*r)`.
 * @param c Receives `cos(pi*r)`.
 */
static inline void sim_sincospi_kernel(double r, double* s, double* c) {
    /* Evaluate sin(pi r) / cos(pi r) for |r|<=0.5 using fused Horner polynomials. */
    const double y  = M_PI * r;
    const double y2 = y * y;

    *s = ((((((-1.0 / 39916800.0) * y2 + 1.0 / 362880.0) * y2 - 1.0 / 5040.0) * y2 + 1.0 / 120.0) *
               y2 -
           1.0 / 6.0) *
              y2 +
          1.0) *
         y;

    *c = ((((((-1.0 / 3628800.0) * y2 + 1.0 / 40320.0) * y2 - 1.0 / 720.0) * y2 + 1.0 / 24.0) * y2 -
           0.5) *
              y2 +
          1.0);
}

/**
 * @brief Evaluate sin(pi*x) with argument reduction.
 *
 * @param x Real argument.
 * @return sin(pi*x).
 */
static inline double sim_sinpi(double x) {
    double r, s, c;
    int    parity;
    sim_reduce_half(x, &r, &parity);
    sim_sincospi_kernel(r, &s, &c);
    /* sin(pi*(n+r)) = (-1)^n sin(pi r) */
    return parity ? -s : s;
}

/**
 * @brief Evaluate cos(pi*x) with argument reduction.
 *
 * @param x Real argument.
 * @return cos(pi*x).
 */
static inline double sim_cospi(double x) {
    double r, s, c;
    int    parity;
    sim_reduce_half(x, &r, &parity);
    sim_sincospi_kernel(r, &s, &c);
    /* cos(pi*(n+r)) = (-1)^n cos(pi r) */
    return parity ? -c : c;
}
#else
/**
 * @brief Evaluate sin(pi*x) through the system libm path.
 *
 * @param x Real argument.
 * @return sin(pi*x).
 */
static inline double sim_sinpi(double x) {
    return sin(M_PI * x);
}
/**
 * @brief Evaluate cos(pi*x) through the system libm path.
 *
 * @param x Real argument.
 * @return cos(pi*x).
 */
static inline double sim_cospi(double x) {
    return cos(M_PI * x);
}
#endif

/* Reflection helpers (cot/csc) – optionally disabled for trig-free builds. */
#if defined(SIM_TRIG_FREE)
/**
 * @brief Placeholder cot(pi*x) used when trigonometric reflection is disabled.
 *
 * @param x Real argument, ignored.
 * @return NaN.
 */
static inline double sim_cotpi(double x) {
    (void) x;
    return NAN;
}

/**
 * @brief Placeholder csc^2(pi*x) used when trigonometric reflection is disabled.
 *
 * @param x Real argument, ignored.
 * @return NaN.
 */
static inline double sim_cscpi_sq(double x) {
    (void) x;
    return NAN;
}
#else
/**
 * @brief Evaluate cot(pi*x) using the local sinpi/cospi helpers.
 *
 * @param x Real argument.
 * @return cot(pi*x), with signed infinity near sine zeros.
 */
static inline double sim_cotpi(double x) {
    double s = sim_sinpi(x);
    double c = sim_cospi(x);
    if (fabs(s) <= DBL_TRUE_MIN)
        return copysign(INFINITY, c);
    return c / s;
}

/**
 * @brief Evaluate (pi / sin(pi*x))^2 for trigamma reflection formulas.
 *
 * @param x Real argument.
 * @return Squared scaled cosecant, or infinity near sine zeros.
 */
static inline double sim_cscpi_sq(double x) {
    double s = sim_sinpi(x);
    if (fabs(s) <= DBL_TRUE_MIN)
        return INFINITY;
    double inv = M_PI / s;
    return inv * inv;
}
#endif

/* float versions */
static inline float sim_sinpi_f(float x) SIM_ATTR_CONST;
static inline float sim_cospi_f(float x) SIM_ATTR_CONST;
static inline float sim_cotpi_f(float x) SIM_ATTR_CONST;
static inline float sim_cscpi_sq_f(float x) SIM_ATTR_CONST;
/**
 * @brief Evaluate sin(pi*x) for float inputs.
 *
 * @param x Real single-precision argument.
 * @return sin(pi*x), rounded to float.
 */
static inline float sim_sinpi_f(float x) {
#if !SIM_HAVE_SYSTEM_TRIGPI
    float       n      = nearbyintf(x);
    float       r      = x - n;
    int         parity = ((int) llroundf(n)) & 1;
    const float y      = (float) M_PI * r;
    const float y2     = y * y;
    float res = ((((((-1.0f / 39916800.0f) * y2 + 1.0f / 362880.0f) * y2 - 1.0f / 5040.0f) * y2 +
                   1.0f / 120.0f) *
                      y2 -
                  1.0f / 6.0f) *
                     y2 +
                 1.0f) *
                y;
    return parity ? -res : res;
#else
    return sinf((float) M_PI * x);
#endif
}
/**
 * @brief Evaluate cos(pi*x) for float inputs.
 *
 * @param x Real single-precision argument.
 * @return cos(pi*x), rounded to float.
 */
static inline float sim_cospi_f(float x) {
#if !SIM_HAVE_SYSTEM_TRIGPI
    float       n      = nearbyintf(x);
    float       r      = x - n;
    int         parity = ((int) llroundf(n)) & 1;
    const float y      = (float) M_PI * r;
    const float y2     = y * y;
    float       res = ((((((-1.0f / 3628800.0f) * y2 + 1.0f / 40320.0f) * y2 - 1.0f / 720.0f) * y2 +
                   1.0f / 24.0f) *
                      y2 -
                  0.5f) *
                     y2 +
                 1.0f);
    return parity ? -res : res;
#else
    return cosf((float) M_PI * x);
#endif
}
/**
 * @brief Evaluate cot(pi*x) for float inputs.
 *
 * @param x Real single-precision argument.
 * @return cot(pi*x), or infinity/NaN according to build mode.
 */
static inline float sim_cotpi_f(float x) {
#if defined(SIM_TRIG_FREE)
    (void) x;
    return NAN;
#else
    float s = sim_sinpi_f(x), c = sim_cospi_f(x);
    return (fabsf(s) <= FLT_TRUE_MIN) ? copysignf(INFINITY, c) : c / s;
#endif
}
/**
 * @brief Evaluate (pi / sin(pi*x))^2 for float inputs.
 *
 * @param x Real single-precision argument.
 * @return Squared scaled cosecant, or infinity/NaN according to build mode.
 */
static inline float sim_cscpi_sq_f(float x) {
#if defined(SIM_TRIG_FREE)
    (void) x;
    return NAN;
#else
    float s = sim_sinpi_f(x);
    if (fabsf(s) <= FLT_TRUE_MIN)
        return INFINITY;
    float inv = (float) M_PI / s;
    return inv * inv;
#endif
}

/**
 * @brief Evaluate complex sin(pi*z) and cos(pi*z) from real/imaginary parts.
 *
 * @param ar Real part of z.
 * @param ai Imaginary part of z.
 * @param s Receives complex sin(pi*z).
 * @param c Receives complex cos(pi*z).
 */
static inline void sim_sincospi_complex(double ar, double ai, double complex* s, double complex* c) {
    const double ch = cosh(M_PI * ai);
    const double sh = sinh(M_PI * ai);
    const double sr = sim_sinpi(ar);
    const double cr = sim_cospi(ar);
    *s              = (sr * ch) + I * (cr * sh);
    *c              = (cr * ch) - I * (sr * sh);
}

/**
 * @brief Narrow a complex double ABI value to a complex float ABI value.
 *
 * @param z Complex double value.
 * @return Component-wise float conversion.
 */
static inline SimComplexFloat sim_complex_to_c32(SimComplexDouble z) {
    SimComplexFloat out;
    out.re = (float) z.re;
    out.im = (float) z.im;
    return out;
}

#endif /* SIM_POLYGAMMA_INTERNAL_H */
