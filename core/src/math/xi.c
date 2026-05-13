/**
 * @file xi.c
 * @brief Completed Riemann xi evaluation, ball enclosures, and critical-zero search.
 *
 * The xi evaluator wraps the zeta implementation with the gamma/pi completion
 * factor, handles exact limits at s = 0 and s = 1, supports ball-style
 * enclosures, and provides a sign-bracketed critical-line zero finder.
 */

#include "oakfield/math/xi.h"

#include "oakfield/math/loggamma.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define SIM_EULER_MASCHERONI 0.57721566490153286060651209008240243
#define SIM_STIELTJES_GAMMA1 -0.07281584548367672486058637587490132
#define SIM_XI_ZERO_SIGN_SAFETY 8.0
#define SIM_PI_LO 3.1415926535897930L
#define SIM_PI_HI 3.1415926535897936L

/**
 * @brief Convert an ABI complex value to C99 `double complex`.
 *
 * @param z ABI-stable complex value.
 * @return C99 complex value with matching components.
 */
static inline double complex sim_to_c64(SimComplexDouble z) {
    return z.re + z.im * I;
}

/**
 * @brief Convert C99 `double complex` to the ABI complex structure.
 *
 * @param z C99 complex value.
 * @return ABI-stable complex value with matching components.
 */
static inline SimComplexDouble sim_from_c64(double complex z) {
    return (SimComplexDouble) { creal(z), cimag(z) };
}

/**
 * @brief Check whether both components of a xi argument are finite.
 *
 * @param z Complex argument.
 * @return true when real and imaginary components are finite.
 */
static inline bool sim_xi_isfinite(SimComplexDouble z) {
    return isfinite(z.re) && isfinite(z.im);
}

/**
 * @brief Return the local displacement from s = 1.
 *
 * @param z Complex argument.
 * @return `z - 1`.
 */
static inline double complex sim_xi_delta_from_one(double complex z) {
    return z - (1.0 + 0.0 * I);
}

/**
 * @brief Test whether a complex value is near a target on the real axis.
 *
 * @param z Complex value.
 * @param target Real target.
 * @param tol Absolute tolerance for real and imaginary components.
 * @return true when `z` lies within the target tolerance box.
 */
static bool sim_xi_is_near_real(double complex z, double target, double tol) {
    return fabs(cimag(z)) <= tol && fabs(creal(z) - target) <= tol;
}

/**
 * @brief Detect negative even integers for reflection-sensitive xi inputs.
 *
 * @param z Complex value.
 * @param tol Absolute tolerance for integer and imaginary checks.
 * @return true when `z` is near a negative even integer.
 */
static bool sim_xi_is_negative_even_integer(double complex z, double tol) {
    if (fabs(cimag(z)) > tol) {
        return false;
    }

    double nearest = nearbyint(creal(z));
    if (fabs(creal(z) - nearest) > tol || nearest >= 0.0) {
        return false;
    }

    long long n = (long long) nearest;
    return (n % 2LL) == 0LL;
}

/**
 * @brief Classify statuses that cannot be treated as usable approximate values.
 *
 * @param status Zeta/xi status.
 * @return true for invalid, singular, or numeric-failure statuses.
 */
static bool sim_xi_status_is_hard_failure(SimZetaStatus status) {
    return status == SIM_ZETA_STATUS_INVALID_ARGUMENT ||
           status == SIM_ZETA_STATUS_SINGULAR ||
           status == SIM_ZETA_STATUS_NUMERIC_FAILURE;
}

/**
 * @brief Check whether a xi result has finite value/error data.
 *
 * @param result Result to inspect.
 * @return true when status and numeric fields can seed a ball enclosure.
 */
static bool sim_xi_result_is_usable(const SimXiResult* result) {
    return result != NULL &&
           (result->status == SIM_ZETA_STATUS_OK ||
            result->status == SIM_ZETA_STATUS_NO_CONVERGENCE) &&
           sim_xi_isfinite(result->value) &&
           isfinite(result->abs_error) &&
           result->abs_error >= 0.0;
}

/**
 * @brief Check whether a complex ball can be used in xi validation/search.
 *
 * @param ball Candidate ball enclosure.
 * @return true when status, center, and radius are numerically usable.
 */
static bool sim_xi_ball_is_usable(SimComplexBall ball) {
    return (ball.status == SIM_ZETA_STATUS_OK ||
            ball.status == SIM_ZETA_STATUS_NO_CONVERGENCE) &&
           sim_xi_isfinite(ball.center) &&
           isfinite(ball.radius) &&
           ball.radius >= 0.0;
}

static SimXiContext sim_xi_context_make_default(void);
static void sim_xi_ball_update_radius(SimComplexBall* out);
static bool sim_xi_ball_is_validated(SimComplexBall ball);
static bool sim_xi_ball_real_sign(SimComplexBall ball, int* out_sign);

/**
 * @brief Step a long double downward for conservative interval endpoints.
 *
 * @param x Input endpoint.
 * @return Next representable value toward negative infinity.
 */
static long double sim_xi_next_down_ld(long double x) {
    return nextafterl(x, -INFINITY);
}

/**
 * @brief Step a long double upward for conservative interval endpoints.
 *
 * @param x Input endpoint.
 * @return Next representable value toward positive infinity.
 */
static long double sim_xi_next_up_ld(long double x) {
    return nextafterl(x, INFINITY);
}

/**
 * @brief Step a double upward for conservative radius casts.
 *
 * @param x Input value.
 * @return Next representable double toward positive infinity.
 */
static double sim_xi_next_up_double(double x) {
    return nextafter(x, INFINITY);
}

/**
 * @brief Convert a real-centered complex ball to a long-double interval.
 *
 * @param ball Ball with zero imaginary center.
 * @param out_lo Receives lower interval endpoint.
 * @param out_hi Receives upper interval endpoint.
 * @return true when a finite ordered interval was produced.
 */
static bool sim_xi_real_interval_from_ball(SimComplexBall ball,
                                           long double*   out_lo,
                                           long double*   out_hi) {
    if (!sim_xi_ball_is_usable(ball) || out_lo == NULL || out_hi == NULL) {
        return false;
    }
    if (fabs(ball.center.im) > 0.0) {
        return false;
    }

    long double center = (long double) ball.center.re;
    long double radius = (long double) ball.radius;
    long double lo     = sim_xi_next_down_ld(center - radius);
    long double hi     = sim_xi_next_up_ld(center + radius);
    if (!isfinite((double) lo) || !isfinite((double) hi) || !(lo <= hi)) {
        return false;
    }

    *out_lo = lo;
    *out_hi = hi;
    return true;
}

/**
 * @brief Scale a real interval by a positive factor with outward rounding.
 *
 * @param io_lo In/out lower endpoint.
 * @param io_hi In/out upper endpoint.
 * @param factor Positive scale factor.
 */
static void sim_xi_real_interval_scale_positive(long double* io_lo,
                                                long double* io_hi,
                                                long double  factor) {
    long double lo = *io_lo;
    long double hi = *io_hi;

    *io_lo = sim_xi_next_down_ld(lo * factor);
    *io_hi = sim_xi_next_up_ld(hi * factor);
}

/**
 * @brief Multiply two nonnegative real intervals with outward rounding.
 *
 * @param a_lo Lower endpoint of first interval.
 * @param a_hi Upper endpoint of first interval.
 * @param b_lo Lower endpoint of second interval.
 * @param b_hi Upper endpoint of second interval.
 * @param out_lo Receives product lower endpoint.
 * @param out_hi Receives product upper endpoint.
 * @return true when inputs are valid and the product interval is ordered.
 */
static bool sim_xi_real_interval_positive_product(long double  a_lo,
                                                  long double  a_hi,
                                                  long double  b_lo,
                                                  long double  b_hi,
                                                  long double* out_lo,
                                                  long double* out_hi) {
    if (!(a_lo >= 0.0L) || !(b_lo >= 0.0L) || out_lo == NULL || out_hi == NULL) {
        return false;
    }

    *out_lo = sim_xi_next_down_ld(a_lo * b_lo);
    *out_hi = sim_xi_next_up_ld(a_hi * b_hi);
    return *out_lo <= *out_hi;
}

/**
 * @brief Convert a real interval into a complex ball with real center.
 *
 * @param lo Lower interval endpoint.
 * @param hi Upper interval endpoint.
 * @param out_ball Receives ball center/radius.
 * @return true when conversion produced finite center and radius.
 */
static bool sim_xi_interval_to_ball(long double     lo,
                                    long double     hi,
                                    SimComplexBall* out_ball) {
    if (out_ball == NULL || !(lo <= hi)) {
        return false;
    }

    long double mid_ld    = 0.5L * (lo + hi);
    double      center_re = (double) mid_ld;
    long double radius_ld =
        fmaxl(fabsl((long double) center_re - lo), fabsl(hi - (long double) center_re));
    double radius = (double) radius_ld;
    if ((long double) radius < radius_ld) {
        radius = sim_xi_next_up_double(radius);
    }

    out_ball->center.re = center_re;
    out_ball->center.im = 0.0;
    out_ball->radius    = radius;
    return isfinite(out_ball->center.re) && isfinite(out_ball->radius);
}

/**
 * @brief Detect exact-real integer inputs for formal xi enclosures.
 *
 * @param z Complex value.
 * @param out_integer Receives nearest integer when detected.
 * @return true when `z` is close to a real integer.
 */
static bool sim_xi_is_exact_real_integer(double complex z, long long* out_integer) {
    if (fabs(cimag(z)) > 64.0 * DBL_EPSILON) {
        return false;
    }

    double nearest = nearbyint(creal(z));
    if (fabs(creal(z) - nearest) > 64.0 * DBL_EPSILON) {
        return false;
    }

    if (out_integer != NULL) {
        *out_integer = (long long) nearest;
    }
    return true;
}

/**
 * @brief Build outward-rounded bounds for 1/pi from hard-coded pi bounds.
 *
 * @param out_lo Receives lower bound.
 * @param out_hi Receives upper bound.
 * @return true when bounds are positive and ordered.
 */
static bool sim_xi_integer_inv_pi_bounds(long double* out_lo, long double* out_hi) {
    if (out_lo == NULL || out_hi == NULL || !(SIM_PI_LO > 0.0L) || !(SIM_PI_LO < SIM_PI_HI)) {
        return false;
    }

    *out_lo = sim_xi_next_down_ld(1.0L / SIM_PI_HI);
    *out_hi = sim_xi_next_up_ld(1.0L / SIM_PI_LO);
    return *out_lo > 0.0L && *out_lo <= *out_hi;
}

/**
 * @brief Bound the xi completion factor for positive integer arguments.
 *
 * @param n Integer argument, at least 2.
 * @param out_lo Receives lower bound.
 * @param out_hi Receives upper bound.
 * @return true when a positive interval bound was produced.
 */
static bool sim_xi_integer_completion_bounds(long long    n,
                                             long double* out_lo,
                                             long double* out_hi) {
    if (out_lo == NULL || out_hi == NULL || n < 2LL) {
        return false;
    }

    long double inv_pi_lo = 0.0L;
    long double inv_pi_hi = 0.0L;
    if (!sim_xi_integer_inv_pi_bounds(&inv_pi_lo, &inv_pi_hi)) {
        return false;
    }

    long double base_lo = 0.0L;
    long double base_hi = 0.0L;
    long long   current = 0LL;
    if ((n % 2LL) == 0LL) {
        base_lo = inv_pi_lo;
        base_hi = inv_pi_hi;
        current = 2LL;
    } else {
        base_lo = sim_xi_next_down_ld(0.5L * inv_pi_lo);
        base_hi = sim_xi_next_up_ld(0.5L * inv_pi_hi);
        current = 3LL;
    }

    while (current < n) {
        long double step = 0.5L * (long double) current;
        sim_xi_real_interval_scale_positive(&base_lo, &base_hi, step);
        if (!sim_xi_real_interval_positive_product(base_lo,
                                                   base_hi,
                                                   inv_pi_lo,
                                                   inv_pi_hi,
                                                   &base_lo,
                                                   &base_hi)) {
            return false;
        }
        current += 2LL;
    }

    long double poly = 0.5L * (long double) n * (long double) (n - 1LL);
    sim_xi_real_interval_scale_positive(&base_lo, &base_hi, poly);
    *out_lo = base_lo;
    *out_hi = base_hi;
    return *out_lo > 0.0L && *out_lo <= *out_hi;
}

/**
 * @brief Build a formal xi ball for exact real integer inputs.
 *
 * This uses a formal zeta ball at the reflected/effective positive integer and
 * interval bounds for the completion factor.
 *
 * @param s_in Complex input expected to be a real integer.
 * @param context Optional xi context.
 * @param out_ball Receives formal ball when available.
 * @return true when a formal integer enclosure was produced.
 */
static bool sim_xi_eval_ball_formal_real_integer(SimComplexDouble      s_in,
                                                 const SimXiContext*   context,
                                                 SimComplexBall*       out_ball) {
    const SimXiContext fallback = sim_xi_context_make_default();
    const SimXiContext* ctx     = (context != NULL) ? context : &fallback;
    long long           n       = 0LL;
    long long           effective_n = 0LL;
    unsigned int        flags       = 0U;
    SimComplexBall      zeta_ball   = { 0 };
    long double         completion_lo = 0.0L;
    long double         completion_hi = 0.0L;
    long double         zeta_lo       = 0.0L;
    long double         zeta_hi       = 0.0L;
    long double         xi_lo         = 0.0L;
    long double         xi_hi         = 0.0L;
    double              magnitude     = 0.0;
    double              target        = 0.0;

    if (out_ball == NULL || !sim_xi_isfinite(s_in) ||
        !sim_xi_is_exact_real_integer(sim_to_c64(s_in), &n) ||
        n == 0LL || n == 1LL) {
        return false;
    }

    effective_n = n;
    if (effective_n < 0LL) {
        effective_n = 1LL - effective_n;
        flags |= SIM_XI_FLAG_USED_REFLECTION;
    }
    if (effective_n < 2LL) {
        return false;
    }

    zeta_ball = sim_zeta_eval_ball((SimComplexDouble) { (double) effective_n, 0.0 }, &ctx->zeta);
    if (zeta_ball.rigor != SIM_BALL_RIGOR_FORMAL ||
        !sim_xi_real_interval_from_ball(zeta_ball, &zeta_lo, &zeta_hi) ||
        !sim_xi_integer_completion_bounds(effective_n, &completion_lo, &completion_hi) ||
        !sim_xi_real_interval_positive_product(completion_lo,
                                               completion_hi,
                                               zeta_lo,
                                               zeta_hi,
                                               &xi_lo,
                                               &xi_hi) ||
        !sim_xi_interval_to_ball(xi_lo, xi_hi, out_ball)) {
        return false;
    }

    magnitude = fabs(out_ball->center.re);
    target    = fmax(ctx->abs_tol, ctx->rel_tol * fmax(1.0, magnitude));

    out_ball->status = (out_ball->radius <= target) ? SIM_ZETA_STATUS_OK
                                                    : SIM_ZETA_STATUS_NO_CONVERGENCE;
    out_ball->flags  = flags;
    out_ball->refinement_rounds      = zeta_ball.refinement_rounds;
    out_ball->working_precision_bits = zeta_ball.working_precision_bits;
    out_ball->rigor                  = SIM_BALL_RIGOR_FORMAL;
    out_ball->validation_passes      = 0U;
    sim_xi_ball_update_radius(out_ball);
    return true;
}

/**
 * @brief Build the default xi evaluation context.
 *
 * @return Context populated with default xi and nested zeta tolerances.
 */
static SimXiContext sim_xi_context_make_default(void) {
    SimXiContext context = { 0 };
    context.zeta               = sim_zeta_context_default();
    context.abs_tol            = 1.0e-12;
    context.rel_tol            = 1.0e-12;
    context.exact_limit_radius = 1.0e-10;
    context.zero_tolerance     = 1.0e-10;
    context.zero_max_iterations = 64U;
    return context;
}

/**
 * @brief Return the default xi evaluation context.
 *
 * @return Copy of the default context.
 */
SimXiContext sim_xi_context_default(void) {
    return sim_xi_context_make_default();
}

/**
 * @brief Return a low-latency xi context for interactive exploration.
 *
 * @return Context with relaxed xi/zeta tolerances and smaller zero-search budget.
 */
SimXiContext sim_xi_context_interactive(void) {
    SimXiContext context = sim_xi_context_make_default();
    context.zeta               = sim_zeta_context_interactive();
    context.abs_tol            = 5.0e-5;
    context.rel_tol            = 5.0e-5;
    context.zero_tolerance     = 1.0e-6;
    context.zero_max_iterations = 24U;
    return context;
}

/**
 * @brief Convert a xi status enum to a diagnostic string.
 *
 * Xi reuses the zeta status enum, so this delegates to the zeta status string.
 *
 * @param status Status value.
 * @return Static status-name string.
 */
const char* sim_xi_status_string(SimZetaStatus status) {
    return sim_zeta_status_string(status);
}

/**
 * @brief Return the scale used when comparing primary and reflected xi balls.
 *
 * @param context Xi context with nested zeta validation settings.
 * @return Validation scale at least one.
 */
static double sim_xi_validation_scale(const SimXiContext* context) {
    return fmax(1.0, context->zeta.ball_validation_scale);
}

/**
 * @brief Enforce a finite, nonnegative xi ball radius with roundoff floor.
 *
 * @param out Ball enclosure to update; ignored when NULL.
 */
static void sim_xi_ball_update_radius(SimComplexBall* out) {
    if (out == NULL) {
        return;
    }

    if (!isfinite(out->radius) || out->radius < 0.0) {
        out->radius = INFINITY;
    } else {
        out->radius = fmax(out->radius, 64.0 * DBL_EPSILON *
                                           fmax(1.0, cabs(sim_to_c64(out->center))));
    }
}

/**
 * @brief Translate zeta implementation flags into xi implementation flags.
 *
 * @param zeta Zeta result whose flags should be mapped.
 * @return Bitset of xi flags.
 */
static unsigned int sim_xi_flags_from_zeta(const SimZetaResult* zeta) {
    unsigned int flags = 0U;
    if (zeta == NULL) {
        return flags;
    }

    if ((zeta->flags & SIM_ZETA_FLAG_USED_REFLECTION) != 0U) {
        flags |= SIM_XI_FLAG_USED_REFLECTION;
    }
    if ((zeta->flags & SIM_ZETA_FLAG_USED_LOCAL_EXPANSION) != 0U) {
        flags |= SIM_XI_FLAG_USED_LOCAL_EXPANSION;
    }
    return flags;
}

/**
 * @brief Build a relaxed context for ambiguous zero-bracket probes.
 *
 * @param context Base xi context.
 * @return Context with looser value tolerances for sign discovery.
 */
static SimXiContext sim_xi_context_relaxed_zero_probe(const SimXiContext* context) {
    SimXiContext relaxed = *context;

    relaxed.abs_tol      = fmax(relaxed.abs_tol, 1.0e-8);
    relaxed.rel_tol      = fmax(relaxed.rel_tol, 1.0e-8);
    relaxed.zeta.abs_tol = fmax(relaxed.zeta.abs_tol, 1.0e-8);
    relaxed.zeta.rel_tol = fmax(relaxed.zeta.rel_tol, 1.0e-8);
    return relaxed;
}

/**
 * @brief Evaluate a critical-line xi ball for zero-search sign probing.
 *
 * If the primary ball is inconclusive and unvalidated, a relaxed context is
 * tried to obtain a cheaper definite sign without discarding the primary when
 * it is stronger.
 *
 * @param t Critical-line height.
 * @param context Xi evaluation context.
 * @return Best ball available for sign probing at `1/2 + i t`.
 */
static SimComplexBall sim_xi_eval_ball_zero_probe(double t, const SimXiContext* context) {
    SimComplexDouble s       = { 0.5, t };
    SimComplexBall   primary = sim_xi_eval_ball(s, context);
    int              sign    = 0;

    if (!sim_xi_ball_is_usable(primary) || sim_xi_ball_real_sign(primary, &sign) ||
        sim_xi_ball_is_validated(primary)) {
        return primary;
    }

    SimXiContext relaxed_context = sim_xi_context_relaxed_zero_probe(context);
    SimComplexBall relaxed       = sim_xi_eval_ball(s, &relaxed_context);
    int            relaxed_sign  = 0;

    if (!sim_xi_ball_is_usable(relaxed)) {
        return primary;
    }
    if (sim_xi_ball_real_sign(relaxed, &relaxed_sign) &&
        !sim_xi_ball_real_sign(primary, NULL)) {
        return relaxed;
    }
    if (relaxed.rigor > primary.rigor ||
        (relaxed.status == SIM_ZETA_STATUS_OK && primary.status != SIM_ZETA_STATUS_OK) ||
        relaxed.radius < primary.radius) {
        return relaxed;
    }
    return primary;
}

/**
 * @brief Evaluate xi near s = 1 using a cancellation-safe local expansion.
 *
 * The zeta pole is removed analytically by expanding `(s - 1) zeta(s)` before
 * multiplying by the completion factor.
 *
 * @param s Complex argument near one.
 * @param context Xi evaluation context.
 * @param extra_flags Additional flags to OR into the result.
 * @return Local xi evaluation result.
 */
static SimXiResult sim_xi_eval_near_one_internal(double complex         s,
                                                 const SimXiContext*    context,
                                                 unsigned int           extra_flags) {
    SimXiResult out = { .value       = { NAN, NAN },
                        .abs_error   = NAN,
                        .rel_error   = NAN,
                        .terms_used  = 0U,
                        .refinement_rounds = 0U,
                        .working_precision_bits = DBL_MANT_DIG,
                        .zeta_branch = SIM_ZETA_BRANCH_NEAR_ONE_LAURENT,
                        .status      = SIM_ZETA_STATUS_OK,
                        .flags       = SIM_XI_FLAG_USED_LOG_ASSEMBLY |
                                  SIM_XI_FLAG_USED_NEAR_ONE_EXPANSION |
                                  SIM_XI_FLAG_USED_LOCAL_EXPANSION | extra_flags };

    double complex delta = sim_xi_delta_from_one(s);
    double complex scaled_zeta =
        1.0 + SIM_EULER_MASCHERONI * delta - SIM_STIELTJES_GAMMA1 * delta * delta;
    SimLogGammaResult lgamma =
        sim_log_gamma_eval((SimComplexDouble) { 0.5 * creal(s), 0.5 * cimag(s) });
    if (lgamma.status != SIM_LOG_GAMMA_STATUS_OK) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double complex log_completion = sim_to_c64(lgamma.value) - 0.5 * s * log(M_PI);
    if (!isfinite(creal(log_completion)) || !isfinite(cimag(log_completion))) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double complex completion = 0.5 * s * cexp(log_completion);
    double complex value      = completion * scaled_zeta;
    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double delta_abs       = cabs(delta);
    double scaled_error    = 2.0 * fabs(SIM_STIELTJES_GAMMA1) * delta_abs * delta_abs * delta_abs;
    double gamma_error     = lgamma.abs_error;
    double completion_abs  = cabs(completion);
    double scaled_abs      = cabs(scaled_zeta);
    double magnitude       = cabs(value);

    out.value      = sim_from_c64(value);
    out.abs_error  = completion_abs * scaled_error +
                    completion_abs * scaled_abs * gamma_error +
                    64.0 * DBL_EPSILON * fmax(1.0, magnitude);
    out.rel_error  = out.abs_error / fmax(magnitude, DBL_MIN);
    out.terms_used = 0U;

    if (out.abs_error > fmax(context->abs_tol, context->rel_tol * fmax(1.0, magnitude))) {
        out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
    }

    return out;
}

/**
 * @brief Compute the real-axis sign radius for a critical-line xi ball.
 *
 * @param ball Complex ball enclosure.
 * @return Inflated radius including imaginary leakage.
 */
static double sim_xi_ball_real_radius(SimComplexBall ball) {
    return SIM_XI_ZERO_SIGN_SAFETY * (ball.radius + fabs(ball.center.im));
}

/**
 * @brief Check whether a ball has validation rigor or better.
 *
 * @param ball Complex ball enclosure.
 * @return true when `ball.rigor >= SIM_BALL_RIGOR_VALIDATED`.
 */
static bool sim_xi_ball_is_validated(SimComplexBall ball) {
    return ball.rigor >= SIM_BALL_RIGOR_VALIDATED;
}

/**
 * @brief Determine a rigorous-enough real sign from a xi ball.
 *
 * @param ball Complex ball enclosure.
 * @param out_sign Receives +1 or -1 when sign is definite.
 * @return true when the real interval excludes zero.
 */
static bool sim_xi_ball_real_sign(SimComplexBall ball, int* out_sign) {
    double radius = sim_xi_ball_real_radius(ball);
    if (ball.center.re > radius) {
        if (out_sign != NULL) {
            *out_sign = 1;
        }
        return true;
    }
    if (ball.center.re < -radius) {
        if (out_sign != NULL) {
            *out_sign = -1;
        }
        return true;
    }
    return false;
}

/**
 * @brief Determine whether two endpoint balls form a sign-changing bracket.
 *
 * @param left Left endpoint ball.
 * @param right Right endpoint ball.
 * @param out_rigor Receives the weaker rigor of the two endpoints.
 * @return true when endpoint signs are definite and opposite.
 */
static bool sim_xi_zero_bracket_rigor(SimComplexBall left,
                                      SimComplexBall right,
                                      SimBallRigor*  out_rigor) {
    int sign_left  = 0;
    int sign_right = 0;

    if (out_rigor == NULL || !sim_xi_ball_is_usable(left) || !sim_xi_ball_is_usable(right) ||
        !sim_xi_ball_real_sign(left, &sign_left) || !sim_xi_ball_real_sign(right, &sign_right) ||
        sign_left == sign_right) {
        return false;
    }

    *out_rigor = (left.rigor < right.rigor) ? left.rigor : right.rigor;
    return true;
}

/**
 * @brief Check whether bracket rigor is strong enough to certify a zero result.
 *
 * @param rigor Bracket rigor classification.
 * @return true for validated or formal brackets.
 */
static bool sim_xi_zero_bracket_is_certified(SimBallRigor rigor) {
    return rigor >= SIM_BALL_RIGOR_VALIDATED;
}

/**
 * @brief Convert bracket rigor into result flags.
 *
 * @param rigor Bracket rigor classification.
 * @return Bitset of zero-bracket flags.
 */
static unsigned int sim_xi_zero_bracket_flags(SimBallRigor rigor) {
    unsigned int flags = 0U;
    if (rigor >= SIM_BALL_RIGOR_VALIDATED) {
        flags |= SIM_XI_FLAG_ZERO_BRACKET_VALIDATED;
    }
    if (rigor >= SIM_BALL_RIGOR_FORMAL) {
        flags |= SIM_XI_FLAG_ZERO_BRACKET_FORMAL;
    }
    return flags;
}

/**
 * @brief Evaluate the completed Riemann xi function.
 *
 * Handles exact limits at 0 and 1, near-one cancellation, reflection across
 * `s -> 1 - s`, zeta evaluation, and the gamma/pi completion factor. Error
 * estimates combine zeta error, log-gamma error, and roundoff.
 *
 * @param s_in Complex xi argument.
 * @param context Optional xi context; defaults are used when NULL.
 * @return Structured xi result with value, errors, branch, status, and flags.
 */
SimXiResult sim_xi_eval(SimComplexDouble s_in, const SimXiContext* context) {
    const SimXiContext fallback = sim_xi_context_make_default();
    const SimXiContext* ctx     = (context != NULL) ? context : &fallback;
    SimXiResult out             = { .value       = { NAN, NAN },
                        .abs_error   = NAN,
                        .rel_error   = NAN,
                        .terms_used  = 0U,
                        .refinement_rounds = 0U,
                        .working_precision_bits = DBL_MANT_DIG,
                        .zeta_branch = SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN,
                        .status      = SIM_ZETA_STATUS_OK,
                        .flags       = 0U };

    if (!sim_xi_isfinite(s_in)) {
        out.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return out;
    }

    double complex s                 = sim_to_c64(s_in);
    double         exact_limit_tol   = 64.0 * DBL_EPSILON;
    double         reflection_tol    = fmax(exact_limit_tol, ctx->exact_limit_radius);

    if (sim_xi_is_near_real(s, 0.0, exact_limit_tol) ||
        sim_xi_is_near_real(s, 1.0, exact_limit_tol)) {
        out.value       = (SimComplexDouble) { 0.5, 0.0 };
        out.abs_error   = 0.0;
        out.rel_error   = 0.0;
        out.terms_used  = 0U;
        out.refinement_rounds = 0U;
        out.working_precision_bits = DBL_MANT_DIG;
        out.zeta_branch = SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN;
        out.flags       = SIM_XI_FLAG_EXACT_LIMIT;
        return out;
    }

    if (cabs(sim_xi_delta_from_one(s)) <= ctx->zeta.near_one_radius) {
        return sim_xi_eval_near_one_internal(s, ctx, 0U);
    }

    if (cabs(s) <= ctx->zeta.near_one_radius) {
        SimXiResult reflected_result = sim_xi_eval_near_one_internal(1.0 - s, ctx, SIM_XI_FLAG_USED_REFLECTION);
        return reflected_result;
    }

    if (sim_xi_is_negative_even_integer(s, reflection_tol)) {
        SimComplexDouble reflected = { 1.0 - creal(s), -cimag(s) };
        SimXiResult      reflected_result = sim_xi_eval(reflected, ctx);
        reflected_result.flags |= SIM_XI_FLAG_USED_REFLECTION;
        return reflected_result;
    }

    SimZetaResult zeta = sim_zeta_eval(s_in, &ctx->zeta);
    out.status                 = zeta.status;
    out.zeta_branch            = zeta.branch;
    out.refinement_rounds      = zeta.refinement_rounds;
    out.working_precision_bits = zeta.working_precision_bits;
    out.flags                  = sim_xi_flags_from_zeta(&zeta);
    if (sim_xi_status_is_hard_failure(zeta.status)) {
        return out;
    }
    if (!isfinite(zeta.value.re) || !isfinite(zeta.value.im) ||
        !isfinite(zeta.abs_error) || zeta.abs_error < 0.0) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    SimLogGammaResult lgamma =
        sim_log_gamma_eval((SimComplexDouble) { 0.5 * creal(s), 0.5 * cimag(s) });
    if (lgamma.status != SIM_LOG_GAMMA_STATUS_OK) {
        out.status      = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double complex zeta_value      = sim_to_c64(zeta.value);
    double complex polynomial      = 0.5 * s * (s - 1.0);
    double complex log_completion  = sim_to_c64(lgamma.value) - 0.5 * s * log(M_PI);
    double complex completion      = cexp(log_completion);
    double complex value           = polynomial * completion * zeta_value;
    if ((!isfinite(creal(value)) || !isfinite(cimag(value))) && cabs(polynomial) > DBL_MIN &&
        cabs(zeta_value) > DBL_MIN) {
        value = cexp(clog(polynomial) + log_completion + clog(zeta_value));
    }
    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        out.status      = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        out.zeta_branch = zeta.branch;
        out.flags       = SIM_XI_FLAG_USED_LOG_ASSEMBLY | sim_xi_flags_from_zeta(&zeta);
        return out;
    }

    double scale           = fmax(1.0, cabs(value));
    double polynomial_abs  = cabs(polynomial);
    double completion_abs  = cabs(completion);
    double zeta_abs        = cabs(zeta_value);
    double zeta_component  = polynomial_abs * completion_abs * zeta.abs_error;
    double gamma_component = polynomial_abs * completion_abs * zeta_abs * lgamma.abs_error;
    double roundoff =
        64.0 * DBL_EPSILON * scale * (1.0 + cabs(s) + cabs(log_completion));

    out.value       = sim_from_c64(value);
    out.abs_error   = zeta_component + gamma_component + roundoff;
    out.rel_error   = out.abs_error / fmax(cabs(value), DBL_MIN);
    out.terms_used  = zeta.terms_used;
    out.refinement_rounds = zeta.refinement_rounds;
    out.working_precision_bits = zeta.working_precision_bits;
    out.zeta_branch = zeta.branch;
    out.flags       = SIM_XI_FLAG_USED_LOG_ASSEMBLY | sim_xi_flags_from_zeta(&zeta);

    if (!isfinite(out.abs_error) || !isfinite(out.rel_error)) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    if (zeta.status != SIM_ZETA_STATUS_OK ||
        out.abs_error > fmax(ctx->abs_tol, ctx->rel_tol * scale)) {
        out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
        return out;
    }

    return out;
}

/**
 * @brief Evaluate xi on the critical line `s = 1/2 + i t`.
 *
 * @param t Critical-line height.
 * @param context Optional xi context.
 * @return Structured xi result at `1/2 + i t`.
 */
SimXiResult sim_xi_eval_critical_line(double t, const SimXiContext* context) {
    return sim_xi_eval((SimComplexDouble) { 0.5, t }, context);
}

/**
 * @brief Evaluate xi and estimate its first complex derivative.
 *
 * Uses centered imaginary-direction finite differences at two step sizes with
 * Richardson extrapolation. The derivative status is downgraded when neighboring
 * evaluations fail or switch zeta branches.
 *
 * @param s Complex xi argument.
 * @param context Optional xi context.
 * @return Xi value, derivative estimate, errors, and status metadata.
 */
SimXiDerivativeResult sim_xi_eval_with_derivative(SimComplexDouble s,
                                                  const SimXiContext* context) {
    const SimXiContext fallback = sim_xi_context_make_default();
    const SimXiContext* ctx     = (context != NULL) ? context : &fallback;
    SimXiDerivativeResult out   = { 0 };

    SimXiResult value = sim_xi_eval(s, ctx);
    out.value                  = value.value;
    out.abs_error              = value.abs_error;
    out.rel_error              = value.rel_error;
    out.refinement_rounds      = value.refinement_rounds;
    out.working_precision_bits = value.working_precision_bits;
    out.zeta_branch            = value.zeta_branch;
    out.status                 = value.status;
    out.flags                  = value.flags;
    out.derivative             = (SimComplexDouble) { NAN, NAN };
    out.derivative_abs_error   = NAN;
    out.derivative_rel_error   = NAN;
    if (value.status != SIM_ZETA_STATUS_OK) {
        return out;
    }

    double complex cs = sim_to_c64(s);
    double scale      = fmax(1.0, cabs(cs));
    double h1         = fmax(8.0 * ctx->exact_limit_radius, cbrt(DBL_EPSILON) * scale);
    double h2         = 0.5 * h1;

    SimXiResult fp1 = sim_xi_eval((SimComplexDouble) { s.re, s.im + h1 }, ctx);
    SimXiResult fm1 = sim_xi_eval((SimComplexDouble) { s.re, s.im - h1 }, ctx);
    SimXiResult fp2 = sim_xi_eval((SimComplexDouble) { s.re, s.im + h2 }, ctx);
    SimXiResult fm2 = sim_xi_eval((SimComplexDouble) { s.re, s.im - h2 }, ctx);
    out.flags |= fp1.flags | fm1.flags | fp2.flags | fm2.flags;
    if (fp1.refinement_rounds > out.refinement_rounds) {
        out.refinement_rounds = fp1.refinement_rounds;
    }
    if (fm1.refinement_rounds > out.refinement_rounds) {
        out.refinement_rounds = fm1.refinement_rounds;
    }
    if (fp2.refinement_rounds > out.refinement_rounds) {
        out.refinement_rounds = fp2.refinement_rounds;
    }
    if (fm2.refinement_rounds > out.refinement_rounds) {
        out.refinement_rounds = fm2.refinement_rounds;
    }
    if (fp1.status != SIM_ZETA_STATUS_OK || fm1.status != SIM_ZETA_STATUS_OK ||
        fp2.status != SIM_ZETA_STATUS_OK || fm2.status != SIM_ZETA_STATUS_OK) {
        out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
        return out;
    }
    if (fp1.zeta_branch != value.zeta_branch || fm1.zeta_branch != value.zeta_branch ||
        fp2.zeta_branch != value.zeta_branch || fm2.zeta_branch != value.zeta_branch) {
        out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
        return out;
    }

    double complex d1 = (sim_to_c64(fp1.value) - sim_to_c64(fm1.value)) / (2.0 * I * h1);
    double complex d2 = (sim_to_c64(fp2.value) - sim_to_c64(fm2.value)) / (2.0 * I * h2);
    double complex derivative = (4.0 * d2 - d1) / 3.0;
    double estimate =
        cabs(d2 - d1) +
        (fp2.abs_error + fm2.abs_error) / fmax(2.0 * h2, DBL_MIN) +
        64.0 * DBL_EPSILON * fmax(1.0, cabs(derivative));

    out.derivative           = sim_from_c64(derivative);
    out.derivative_abs_error = estimate;
    out.derivative_rel_error = estimate / fmax(cabs(derivative), DBL_MIN);
    if (!isfinite(estimate) ||
        estimate >
            fmax(ctx->abs_tol / fmax(h2, DBL_MIN),
                 ctx->rel_tol * fmax(1.0, cabs(derivative)))) {
        out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
    }
    return out;
}

/**
 * @brief Build a ball-style enclosure for xi(s).
 *
 * Exact integer inputs may receive formal enclosures. Other inputs start from
 * the xi error model and may be upgraded through reflected-value validation.
 *
 * @param s Complex xi argument.
 * @param context Optional xi context.
 * @return Complex ball with center, radius, status, rigor, and validation metadata.
 */
SimComplexBall sim_xi_eval_ball(SimComplexDouble s, const SimXiContext* context) {
    const SimXiContext fallback = sim_xi_context_make_default();
    const SimXiContext* ctx     = (context != NULL) ? context : &fallback;
    SimComplexBall formal       = { 0 };

    if (sim_xi_eval_ball_formal_real_integer(s, ctx, &formal)) {
        return formal;
    }

    SimXiResult value = sim_xi_eval(s, ctx);
    SimComplexBall out = { .center                 = value.value,
                         .radius                 = value.abs_error,
                         .status                 = value.status,
                         .flags                  = value.flags,
                         .refinement_rounds      = value.refinement_rounds,
                         .working_precision_bits = value.working_precision_bits,
                         .rigor                  = SIM_BALL_RIGOR_HEURISTIC,
                         .validation_passes      = 0U };

    if (sim_xi_status_is_hard_failure(out.status) || !sim_xi_result_is_usable(&value)) {
        return out;
    }
    if (value.abs_error == 0.0 && (value.flags & SIM_XI_FLAG_EXACT_LIMIT) != 0U) {
        out.rigor = SIM_BALL_RIGOR_FORMAL;
        return out;
    }
    sim_xi_ball_update_radius(&out);
    if (out.status != SIM_ZETA_STATUS_OK) {
        return out;
    }
    if (ctx->zeta.ball_validation_rounds == 0U) {
        return out;
    }

    SimComplexBall zeta_ball = sim_zeta_eval_ball(s, &ctx->zeta);
    if (zeta_ball.status != SIM_ZETA_STATUS_OK || zeta_ball.rigor < SIM_BALL_RIGOR_VALIDATED) {
        return out;
    }

    SimComplexDouble reflected_s = { 1.0 - s.re, -s.im };
    SimXiResult reflected        = sim_xi_eval(reflected_s, ctx);
    if (reflected.status != SIM_ZETA_STATUS_OK) {
        return out;
    }

    SimComplexBall reflected_ball = { .center                 = reflected.value,
                                    .radius                 = reflected.abs_error,
                                    .status                 = reflected.status,
                                    .flags                  = reflected.flags,
                                    .refinement_rounds      = reflected.refinement_rounds,
                                    .working_precision_bits = reflected.working_precision_bits,
                                    .rigor                  = SIM_BALL_RIGOR_HEURISTIC,
                                    .validation_passes      = 0U };
    sim_xi_ball_update_radius(&reflected_ball);

    {
        double complex center_primary = sim_to_c64(out.center);
        double complex center_peer    = sim_to_c64(reflected_ball.center);
        double         disagreement   = cabs(center_primary - center_peer);
        double         scale          = sim_xi_validation_scale(ctx);
        if (disagreement <= scale * fmax(out.radius, reflected_ball.radius)) {
            out.center = sim_from_c64(0.5 * (center_primary + center_peer));
            out.radius = 0.5 * disagreement + fmax(out.radius, reflected_ball.radius);
            out.flags |= reflected.flags;
            if (reflected.refinement_rounds > out.refinement_rounds) {
                out.refinement_rounds = reflected.refinement_rounds;
            }
            if (reflected.working_precision_bits > out.working_precision_bits) {
                out.working_precision_bits = reflected.working_precision_bits;
            }
            out.rigor = SIM_BALL_RIGOR_VALIDATED;
            out.validation_passes = 1U + zeta_ball.validation_passes;
            sim_xi_ball_update_radius(&out);
        }
    }
    return out;
}

/**
 * @brief Locate a critical-line zero inside a sign-changing bracket.
 *
 * The search combines secant and bisection candidates, evaluates xi balls on
 * the critical line, and only reports full success when the bracket rigor and
 * derivative check meet the configured criteria.
 *
 * @param t_lo Lower critical-line height.
 * @param t_hi Upper critical-line height; must exceed `t_lo`.
 * @param context Optional xi context.
 * @return Zero candidate, enclosing ball, derivative estimate, status, and flags.
 */
SimXiZeroResult sim_xi_find_critical_zero(double t_lo,
                                          double t_hi,
                                          const SimXiContext* context) {
    const SimXiContext fallback = sim_xi_context_make_default();
    const SimXiContext* ctx     = (context != NULL) ? context : &fallback;
    SimXiZeroResult     out     = { 0 };

    out.t                    = NAN;
    out.xi_ball.center       = (SimComplexDouble) { NAN, NAN };
    out.xi_ball.radius       = INFINITY;
    out.xi_ball.status       = SIM_ZETA_STATUS_INVALID_ARGUMENT;
    out.bracket_rigor        = SIM_BALL_RIGOR_HEURISTIC;
    out.derivative           = NAN;
    out.derivative_abs_error = NAN;
    out.iterations           = 0U;
    out.zeta_branch          = SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION;
    out.status               = SIM_ZETA_STATUS_INVALID_ARGUMENT;
    out.flags                = 0U;

    if (!isfinite(t_lo) || !isfinite(t_hi) || !(t_lo < t_hi)) {
        return out;
    }

    double         a       = t_lo;
    double         b       = t_hi;
    SimComplexBall ball_a  = sim_xi_eval_ball_zero_probe(a, ctx);
    SimComplexBall ball_b  = sim_xi_eval_ball_zero_probe(b, ctx);
    SimBallRigor   bracket_rigor = SIM_BALL_RIGOR_HEURISTIC;
    int            sign_a  = 0;
    int            sign_b  = 0;
    if (!sim_xi_ball_is_usable(ball_a) || !sim_xi_ball_is_usable(ball_b) ||
        !sim_xi_ball_real_sign(ball_a, &sign_a) || !sim_xi_ball_real_sign(ball_b, &sign_b) ||
        sign_a == sign_b || !sim_xi_zero_bracket_rigor(ball_a, ball_b, &bracket_rigor)) {
        out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
        out.xi_ball = ball_a;
        return out;
    }

    for (size_t iter = 0U; iter < ctx->zero_max_iterations; ++iter) {
        double fa = ball_a.center.re;
        double fb = ball_b.center.re;
        double candidate = 0.5 * (a + b);
        if (isfinite(fa) && isfinite(fb) && fabs(fb - fa) > DBL_MIN) {
            double secant = b - fb * (b - a) / (fb - fa);
            if (secant > a && secant < b) {
                candidate = secant;
            }
        }

        SimComplexBall ball_c = sim_xi_eval_ball_zero_probe(candidate, ctx);
        out.iterations        = iter + 1U;
        if (!sim_xi_ball_is_usable(ball_c)) {
            out.status = ball_c.status;
            out.xi_ball = ball_c;
            return out;
        }

        double real_radius = sim_xi_ball_real_radius(ball_c);
        int    sign_c      = 0;
        bool   validated   = sim_xi_ball_is_validated(ball_c);
        bool   definite    = sim_xi_ball_real_sign(ball_c, &sign_c);

        if (validated &&
            ((fabs(ball_c.center.re) <= real_radius && (b - a) <= ctx->zero_tolerance) ||
             (real_radius <= ctx->zero_tolerance &&
              fabs(ball_c.center.re) <= 2.0 * real_radius))) {
            SimXiDerivativeResult derivative =
                sim_xi_eval_with_derivative((SimComplexDouble) { 0.5, candidate }, ctx);
            out.t                    = candidate;
            out.xi_ball              = ball_c;
            out.bracket_rigor        = bracket_rigor;
            out.derivative           = creal(I * sim_to_c64(derivative.derivative));
            out.derivative_abs_error = derivative.derivative_abs_error;
            out.zeta_branch          = derivative.zeta_branch;
            out.status               = (derivative.status == SIM_ZETA_STATUS_OK &&
                              sim_xi_zero_bracket_is_certified(bracket_rigor))
                                             ? SIM_ZETA_STATUS_OK
                                             : SIM_ZETA_STATUS_NO_CONVERGENCE;
            out.flags                = derivative.flags;
            if (derivative.status == SIM_ZETA_STATUS_OK) {
                out.flags |= SIM_XI_FLAG_USED_ZERO_REFINEMENT;
                out.flags |= sim_xi_zero_bracket_flags(bracket_rigor);
            }
            return out;
        }

        if (!definite) {
            if ((b - a) <= ctx->zero_tolerance) {
                SimXiDerivativeResult derivative =
                    sim_xi_eval_with_derivative((SimComplexDouble) { 0.5, candidate }, ctx);
                out.t                    = candidate;
                out.xi_ball              = ball_c;
                out.bracket_rigor        = bracket_rigor;
                out.derivative           = creal(I * sim_to_c64(derivative.derivative));
                out.derivative_abs_error = derivative.derivative_abs_error;
                out.zeta_branch          = derivative.zeta_branch;
                out.status               = (sim_xi_zero_bracket_is_certified(bracket_rigor) &&
                              derivative.status == SIM_ZETA_STATUS_OK)
                                               ? SIM_ZETA_STATUS_OK
                                               : SIM_ZETA_STATUS_NO_CONVERGENCE;
                out.flags                = derivative.flags;
                if (derivative.status == SIM_ZETA_STATUS_OK) {
                    out.flags |= SIM_XI_FLAG_USED_ZERO_REFINEMENT;
                    out.flags |= sim_xi_zero_bracket_flags(bracket_rigor);
                }
                return out;
            }
            candidate = 0.5 * (a + b);
            ball_c    = sim_xi_eval_ball_zero_probe(candidate, ctx);
            if (!sim_xi_ball_is_usable(ball_c)) {
                out.status = ball_c.status;
                out.xi_ball = ball_c;
                return out;
            }
            validated = sim_xi_ball_is_validated(ball_c);
            definite  = sim_xi_ball_real_sign(ball_c, &sign_c);
            if (!definite) {
                continue;
            }
        }

        if (sign_c == sign_a) {
            a      = candidate;
            ball_a = ball_c;
            sign_a = sign_c;
        } else {
            b      = candidate;
            ball_b = ball_c;
            sign_b = sign_c;
        }
        if (!sim_xi_zero_bracket_rigor(ball_a, ball_b, &bracket_rigor)) {
            out.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
            out.xi_ball = ball_c;
            return out;
        }
    }

    out.t       = 0.5 * (a + b);
    out.xi_ball = sim_xi_eval_ball_zero_probe(out.t, ctx);
    out.bracket_rigor = bracket_rigor;
    {
        SimXiDerivativeResult derivative =
            sim_xi_eval_with_derivative((SimComplexDouble) { 0.5, out.t }, ctx);
        out.derivative           = creal(I * sim_to_c64(derivative.derivative));
        out.derivative_abs_error = derivative.derivative_abs_error;
        out.zeta_branch          = derivative.zeta_branch;
        out.flags                = derivative.flags;
        if (derivative.status == SIM_ZETA_STATUS_OK) {
            out.flags |= SIM_XI_FLAG_USED_ZERO_REFINEMENT;
            out.flags |= sim_xi_zero_bracket_flags(bracket_rigor);
        }
    }
    out.status = (sim_xi_zero_bracket_is_certified(bracket_rigor) &&
                  out.xi_ball.status == SIM_ZETA_STATUS_OK &&
                  (out.flags & SIM_XI_FLAG_USED_ZERO_REFINEMENT) != 0U)
                     ? SIM_ZETA_STATUS_OK
                     : ((out.xi_ball.status == SIM_ZETA_STATUS_OK) ? SIM_ZETA_STATUS_NO_CONVERGENCE
                                                                   : out.xi_ball.status);
    return out;
}
