/**
 * @file zeta.c
 * @brief Adaptive complex Riemann zeta evaluation and ball-style error enclosures.
 *
 * The dispatcher combines direct Euler-Maclaurin summation, eta/Hasse
 * acceleration, an approximate functional equation, a Riemann-Siegel critical
 * line path, local expansions near special points, and reflection. Results
 * carry status, branch, error estimates, and flags describing the path used.
 */

#include "oakfield/math/zeta.h"

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
#define SIM_ZETA_EM_MAX_CORRECTIONS 9U

static const long double sim_zeta_em_coeffs[SIM_ZETA_EM_MAX_CORRECTIONS] = {
    1.0L / 12.0L,
    -1.0L / 720.0L,
    1.0L / 30240.0L,
    -1.0L / 1209600.0L,
    1.0L / 47900160.0L,
    -691.0L / 1307674368000.0L,
    1.0L / 74724249600.0L,
    -3617.0L / 10670622842880000.0L,
    43867.0L / 3265210599921280000.0L
};

typedef struct SimComplexKahan {
    double sum_re;
    double sum_im;
    double corr_re;
    double corr_im;
} SimComplexKahan;

/**
 * @brief Convert an ABI complex value to C99 `double complex`.
 *
 * @param z ABI-stable complex input.
 * @return C99 complex value with matching components.
 */
static inline double complex sim_to_c64(SimComplexDouble z) {
    return z.re + z.im * I;
}

/**
 * @brief Convert C99 `double complex` to the ABI complex structure.
 *
 * @param z C99 complex input.
 * @return ABI-stable complex value with matching components.
 */
static inline SimComplexDouble sim_from_c64(double complex z) {
    return (SimComplexDouble) { creal(z), cimag(z) };
}

/**
 * @brief Convert an ABI complex value to long-double complex precision.
 *
 * @param z ABI-stable complex input.
 * @return Long-double complex value with matching components.
 */
static inline long double complex sim_to_cld(SimComplexDouble z) {
    return (long double) z.re + (long double) z.im * I;
}

/**
 * @brief Check whether both components of a zeta argument are finite.
 *
 * @param z Complex input.
 * @return true when real and imaginary components are finite.
 */
static inline bool sim_zeta_isfinite(SimComplexDouble z) {
    return isfinite(z.re) && isfinite(z.im);
}

/**
 * @brief Return the local displacement from the pole center s = 1.
 *
 * @param s Complex argument.
 * @return `s - 1`.
 */
static inline double complex sim_zeta_delta_from_one(double complex s) {
    return s - (1.0 + 0.0 * I);
}

/**
 * @brief Test whether a complex argument is near a specified real integer.
 *
 * @param s Complex argument.
 * @param target Real target value.
 * @param tol Absolute tolerance for both real displacement and imaginary part.
 * @return true when `s` is within the tolerance box around `target`.
 */
static bool sim_zeta_is_near_real_integer(double complex s, double target, double tol) {
    return fabs(cimag(s)) <= tol && fabs(creal(s) - target) <= tol;
}

/**
 * @brief Detect trivial-zero locations at negative even integers.
 *
 * @param s Complex argument.
 * @param tol Absolute tolerance for real/integer and imaginary checks.
 * @return true when `s` is near a negative even integer.
 */
static bool sim_zeta_is_negative_even_integer(double complex s, double tol) {
    if (fabs(cimag(s)) > tol) {
        return false;
    }

    double nearest = nearbyint(creal(s));
    if (fabs(creal(s) - nearest) > tol || nearest >= 0.0) {
        return false;
    }

    long long n = (long long) nearest;
    return (n % 2LL) == 0LL;
}

/**
 * @brief Add a complex term using component-wise Kahan/Neumaier compensation.
 *
 * @param acc Mutable compensated accumulator.
 * @param value Complex term to add.
 */
static void sim_complex_kahan_add(SimComplexKahan* acc, double complex value) {
    double y_re = creal(value) - acc->corr_re;
    double t_re = acc->sum_re + y_re;
    acc->corr_re = (t_re - acc->sum_re) - y_re;
    acc->sum_re  = t_re;

    double y_im = cimag(value) - acc->corr_im;
    double t_im = acc->sum_im + y_im;
    acc->corr_im = (t_im - acc->sum_im) - y_im;
    acc->sum_im  = t_im;
}

/**
 * @brief Extract the current compensated complex sum.
 *
 * @param acc Compensated accumulator.
 * @return Complex sum value.
 */
static inline double complex sim_complex_kahan_value(const SimComplexKahan* acc) {
    return acc->sum_re + acc->sum_im * I;
}

/**
 * @brief Combine absolute and relative tolerances into an absolute target.
 *
 * @param abs_tol Absolute tolerance floor.
 * @param rel_tol Relative tolerance multiplier.
 * @param magnitude Current result magnitude.
 * @return Absolute target tolerance.
 */
static inline double sim_zeta_target_tol(double abs_tol, double rel_tol, double magnitude) {
    return fmax(abs_tol, rel_tol * fmax(1.0, magnitude));
}

/**
 * @brief Refresh relative error from the stored value and absolute error.
 *
 * @param out Result to update; ignored when NULL.
 */
static void sim_zeta_update_rel_error(SimZetaResult* out) {
    if (out == NULL) {
        return;
    }

    double magnitude = cabs(sim_to_c64(out->value));
    out->rel_error = out->abs_error / fmax(magnitude, DBL_MIN);
}

/**
 * @brief Build the default zeta evaluation context.
 *
 * @return Context populated with conservative default tolerances and budgets.
 */
static SimZetaContext sim_zeta_context_make_default(void) {
    SimZetaContext context = { 0 };
    context.abs_tol               = 1.0e-13;
    context.rel_tol               = 1.0e-12;
    context.initial_terms         = 16U;
    context.max_terms             = 32768U;
    context.euler_maclaurin_terms = 8U;
    context.eta_initial_terms     = 8U;
    context.eta_max_terms         = 192U;
    context.eta_sigma_limit       = 1.1;
    context.eta_imag_limit        = 48.0;
    context.afe_initial_cutoff    = 8U;
    context.afe_max_cutoff        = 2048U;
    context.afe_sigma_limit       = 1.5;
    context.afe_imag_min          = 10.0;
    context.riemann_siegel_imag_min = 32.0;
    context.critical_line_tolerance = 1.0e-10;
    context.riemann_siegel_max_terms = 4096U;
    context.pole_radius           = 256.0 * DBL_EPSILON;
    context.near_one_radius       = 1.0e-4;
    context.local_expansion_radius = 1.0e-4;
    context.ball_validation_rounds = 1U;
    context.ball_validation_scale = 8.0;
    context.adaptive_max_rounds   = 3U;
    context.adaptive_tightening_factor = 4.0;
    return context;
}

/**
 * @brief Return the default zeta evaluation context.
 *
 * @return Copy of the default context.
 */
SimZetaContext sim_zeta_context_default(void) {
    return sim_zeta_context_make_default();
}

/**
 * @brief Return a low-latency zeta context for interactive exploration.
 *
 * @return Context with relaxed tolerances and smaller work budgets.
 */
SimZetaContext sim_zeta_context_interactive(void) {
    SimZetaContext context = sim_zeta_context_make_default();
    context.abs_tol                    = 5.0e-5;
    context.rel_tol                    = 5.0e-5;
    context.initial_terms              = 8U;
    context.max_terms                  = 768U;
    context.euler_maclaurin_terms      = 4U;
    context.eta_initial_terms          = 4U;
    context.eta_max_terms              = 48U;
    context.eta_imag_limit             = 96.0;
    context.afe_initial_cutoff         = 4U;
    context.afe_max_cutoff             = 96U;
    context.riemann_siegel_imag_min    = 24.0;
    context.critical_line_tolerance    = 1.0e-6;
    context.riemann_siegel_max_terms   = 256U;
    context.ball_validation_rounds     = 0U;
    context.ball_validation_scale      = 1.0;
    context.adaptive_max_rounds        = 0U;
    context.adaptive_tightening_factor = 2.0;
    return context;
}

/**
 * @brief Convert a zeta status enum to a diagnostic string.
 *
 * @param status Status value.
 * @return Static status-name string.
 */
const char* sim_zeta_status_string(SimZetaStatus status) {
    switch (status) {
        case SIM_ZETA_STATUS_OK:
            return "ok";
        case SIM_ZETA_STATUS_SINGULAR:
            return "singular";
        case SIM_ZETA_STATUS_INVALID_ARGUMENT:
            return "invalid-argument";
        case SIM_ZETA_STATUS_NO_CONVERGENCE:
            return "no-convergence";
        case SIM_ZETA_STATUS_NUMERIC_FAILURE:
            return "numeric-failure";
        default:
            return "unknown";
    }
}

/**
 * @brief Convert a zeta branch enum to a diagnostic string.
 *
 * @param branch Branch value.
 * @return Static branch-name string.
 */
const char* sim_zeta_branch_string(SimZetaBranch branch) {
    switch (branch) {
        case SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN:
            return "direct-euler-maclaurin";
        case SIM_ZETA_BRANCH_ETA_ACCELERATED:
            return "eta-accelerated";
        case SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION:
            return "approximate-functional-equation";
        case SIM_ZETA_BRANCH_RIEMANN_SIEGEL:
            return "riemann-siegel";
        case SIM_ZETA_BRANCH_NEAR_ONE_LAURENT:
            return "near-one-laurent";
        case SIM_ZETA_BRANCH_LOCAL_EXPANSION:
            return "local-expansion";
        case SIM_ZETA_BRANCH_REFLECTION:
            return "reflection";
        default:
            return "unknown";
    }
}

/**
 * @brief Return the real-axis tolerance used for pole/special-point tests.
 *
 * @param context Optional zeta context.
 * @return Positive tolerance derived from the context or a fallback value.
 */
static double sim_zeta_real_tol(const SimZetaContext* context) {
    return fmax(64.0 * DBL_EPSILON, (context != NULL) ? context->pole_radius : 1.0e-8);
}

/**
 * @brief Return the configured radius for the Laurent expansion near s = 1.
 *
 * @param context Optional zeta context.
 * @return Positive near-one radius.
 */
static double sim_zeta_near_one_radius(const SimZetaContext* context) {
    return fmax(1.0e-12, (context != NULL) ? context->near_one_radius : 1.0e-4);
}

/**
 * @brief Test whether an argument lies on the zeta pole at s = 1.
 *
 * @param s Complex argument.
 * @param context Optional zeta context controlling tolerance.
 * @return true when `s` is close enough to the pole to report singular.
 */
static bool sim_zeta_has_pole(double complex s, const SimZetaContext* context) {
    double tol = sim_zeta_real_tol(context);
    return fabs(cimag(s)) <= tol && fabs(creal(s) - 1.0) <= tol;
}

/**
 * @brief Test whether an argument should use the near-one Laurent expansion.
 *
 * @param s Complex argument.
 * @param context Optional zeta context.
 * @return true when `|s - 1|` is inside the configured radius.
 */
static bool sim_zeta_is_near_one(double complex s, const SimZetaContext* context) {
    return cabs(sim_zeta_delta_from_one(s)) <= sim_zeta_near_one_radius(context);
}

/**
 * @brief Return the radius for local trivial-zero expansions.
 *
 * @param context Optional zeta context.
 * @return Positive local-expansion radius.
 */
static double sim_zeta_local_expansion_radius(const SimZetaContext* context) {
    return fmax(1.0e-10, (context != NULL) ? context->local_expansion_radius : 1.0e-4);
}

/**
 * @brief Test whether an argument is numerically on the critical line.
 *
 * @param s Complex argument.
 * @param context Optional zeta context.
 * @return true when `Re(s)` is close to 1/2.
 */
static bool sim_zeta_is_near_critical_line(double complex s, const SimZetaContext* context) {
    double tol = (context != NULL) ? context->critical_line_tolerance : 1.0e-10;
    return fabs(creal(s) - 0.5) <= fmax(tol, 64.0 * DBL_EPSILON);
}

/**
 * @brief Check whether a zeta result satisfies the requested tolerance.
 *
 * @param out Result to inspect.
 * @param context Optional context with target tolerances.
 * @return true when the result is finite and its absolute error meets target.
 */
static bool sim_zeta_result_meets_target(const SimZetaResult* out,
                                         const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;

    if (out == NULL || !isfinite(out->abs_error) || !isfinite(out->value.re) ||
        !isfinite(out->value.im)) {
        return false;
    }

    return out->abs_error <=
           sim_zeta_target_tol(ctx->abs_tol, ctx->rel_tol, cabs(sim_to_c64(out->value)));
}

/**
 * @brief Tighten tolerances and work budgets for an adaptive refinement round.
 *
 * @param base Base context to refine.
 * @param round Zero-based refinement round.
 * @return Refined context with scaled tolerances and term budgets.
 */
static SimZetaContext sim_zeta_context_refined(const SimZetaContext* base, size_t round) {
    SimZetaContext refined = *base;
    double         factor  = fmax(2.0, base->adaptive_tightening_factor);
    double         scale   = pow(factor, (double) round);
    size_t         term_scale =
        (size_t) llround(pow(2.0, (double) round));
    if (term_scale < 1U) {
        term_scale = 1U;
    }

    refined.abs_tol /= scale;
    refined.rel_tol /= scale;

    if (round > 0U) {
        refined.initial_terms     = base->initial_terms * term_scale;
        refined.max_terms         = base->max_terms * term_scale;
        refined.eta_initial_terms = base->eta_initial_terms * term_scale;
        refined.eta_max_terms     = base->eta_max_terms * term_scale;
        refined.afe_initial_cutoff = base->afe_initial_cutoff * term_scale;
        refined.afe_max_cutoff     = base->afe_max_cutoff * term_scale;
    }

    if (refined.initial_terms < 4U) {
        refined.initial_terms = 4U;
    }
    if (refined.max_terms < refined.initial_terms) {
        refined.max_terms = refined.initial_terms;
    }
    if (refined.eta_initial_terms < 4U) {
        refined.eta_initial_terms = 4U;
    }
    if (refined.eta_max_terms < refined.eta_initial_terms) {
        refined.eta_max_terms = refined.eta_initial_terms;
    }
    if (refined.afe_initial_cutoff < 4U) {
        refined.afe_initial_cutoff = 4U;
    }
    if (refined.afe_max_cutoff < refined.afe_initial_cutoff) {
        refined.afe_max_cutoff = refined.afe_initial_cutoff;
    }

    return refined;
}

/**
 * @brief Build a refinement context for validation/ball peer evaluations.
 *
 * Ball validation disables recursive ball-validation rounds in the peer context
 * to avoid unbounded nested validation work.
 *
 * @param base Base context to refine.
 * @param round Zero-based validation round.
 * @return Refined context suitable for peer evaluation.
 */
static SimZetaContext sim_zeta_context_ball_refined(const SimZetaContext* base, size_t round) {
    SimZetaContext refined = sim_zeta_context_refined(base, round + 1U);
    if (refined.ball_validation_rounds > 0U) {
        refined.ball_validation_rounds = 0U;
    }
    return refined;
}

/**
 * @brief Evaluate an independent peer branch for ball validation.
 *
 * @param s Complex argument.
 * @param primary_branch Branch used by the primary result.
 * @param context Base context.
 * @param round Validation round.
 * @param out_peer Receives the peer result when a peer branch is available.
 * @return true when a peer branch was evaluated.
 */
static bool sim_zeta_eval_validation_peer(SimComplexDouble      s,
                                          SimZetaBranch         primary_branch,
                                          const SimZetaContext* context,
                                          size_t                round,
                                          SimZetaResult*        out_peer) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaContext        peer_ctx = sim_zeta_context_ball_refined(ctx, round);

    if (out_peer == NULL) {
        return false;
    }

    switch (primary_branch) {
        case SIM_ZETA_BRANCH_RIEMANN_SIEGEL:
            *out_peer = sim_zeta_eval_approximate_fe(s, &peer_ctx);
            return true;
        case SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION:
            *out_peer = sim_zeta_eval_direct_euler_maclaurin(s, &peer_ctx);
            return true;
        case SIM_ZETA_BRANCH_ETA_ACCELERATED:
            *out_peer = sim_zeta_eval_direct_euler_maclaurin(s, &peer_ctx);
            return true;
        case SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN:
            if (s.re > 0.0 && s.re <= ctx->eta_sigma_limit && fabs(s.im) <= ctx->eta_imag_limit) {
                *out_peer = sim_zeta_eval_eta_accelerated(s, &peer_ctx);
                return true;
            }
            return false;
        default:
            return false;
    }
}

/**
 * @brief Enforce a finite, nonnegative ball radius with a roundoff floor.
 *
 * @param out Ball enclosure to update; ignored when NULL.
 */
static void sim_zeta_ball_update_radius(SimComplexBall* out) {
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
 * @brief Check whether a real right-half-plane argument can use formal EM bounds.
 *
 * @param s_in Complex argument with expected zero imaginary part.
 * @param context Context controlling pole/near-one tolerances.
 * @return true when a formal real Euler-Maclaurin enclosure is applicable.
 */
static bool sim_zeta_is_real_right_half_formal_candidate(SimComplexDouble      s_in,
                                                          const SimZetaContext* context) {
    double tol   = sim_zeta_real_tol(context);
    double sigma = s_in.re;
    return fabs(s_in.im) <= tol && sigma > 1.0 + sim_zeta_near_one_radius(context);
}

/**
 * @brief Check whether a complex right-half-plane argument can use formal EM bounds.
 *
 * @param s_in Complex argument.
 * @param context Context controlling pole/near-one tolerances.
 * @return true when a formal complex Euler-Maclaurin enclosure is applicable.
 */
static bool sim_zeta_is_complex_right_half_formal_candidate(SimComplexDouble      s_in,
                                                             const SimZetaContext* context) {
    double tol   = sim_zeta_real_tol(context);
    double sigma = s_in.re;
    return fabs(s_in.im) > tol && sigma > 1.0 + sim_zeta_near_one_radius(context);
}

/**
 * @brief Bound the real Euler-Maclaurin remainder for Re(s) > 1.
 *
 * @param sigma Real part/exponent of s.
 * @param n_cutoff Summation cutoff N.
 * @param corrections Number of Bernoulli correction terms used.
 * @return Long-double upper bound on the tail remainder.
 */
static long double sim_zeta_em_formal_remainder_bound(long double sigma,
                                                      long double n_cutoff,
                                                      size_t      corrections) {
    if (!(sigma > 1.0L) || !(n_cutoff >= 1.0L) || corrections == 0U) {
        return INFINITY;
    }

    long double rising = 1.0L;
    for (size_t j = 0U; j < (2U * corrections); ++j) {
        rising *= sigma + (long double) j;
    }

    long double exponent = 1.0L - sigma - 2.0L * (long double) corrections;
    long double decay    = powl(n_cutoff, exponent);
    long double denom    = sigma + 2.0L * (long double) corrections - 1.0L;
    long double pi2      = 2.0L * (long double) M_PI;
    long double bernoulli_sup =
        4.0L / powl(pi2, 2.0L * (long double) corrections);

    return bernoulli_sup * rising * decay / denom;
}

/**
 * @brief Bound the complex Euler-Maclaurin remainder for Re(s) > 1.
 *
 * @param s Complex argument in long-double precision.
 * @param sigma Real part of s.
 * @param n_cutoff Summation cutoff N.
 * @param corrections Number of Bernoulli correction terms used.
 * @return Long-double upper bound on the complex tail remainder.
 */
static long double sim_zeta_em_formal_remainder_bound_complex(long double complex s,
                                                              long double         sigma,
                                                              long double         n_cutoff,
                                                              size_t              corrections) {
    if (!(sigma > 1.0L) || !(n_cutoff >= 1.0L) || corrections == 0U) {
        return INFINITY;
    }

    long double complex rising = 1.0L + 0.0L * I;
    for (size_t j = 0U; j < (2U * corrections); ++j) {
        rising *= s + (long double) j;
    }

    long double exponent = 1.0L - sigma - 2.0L * (long double) corrections;
    long double decay    = powl(n_cutoff, exponent);
    long double denom    = sigma + 2.0L * (long double) corrections - 1.0L;
    long double pi2      = 2.0L * (long double) M_PI;
    long double bernoulli_sup =
        4.0L / powl(pi2, 2.0L * (long double) corrections);

    return bernoulli_sup * cabsl(rising) * decay / denom;
}

/**
 * @brief Build a formal ball enclosure from a real right-half-plane EM series.
 *
 * @param s_in Complex input expected to be real with Re(s) > 1.
 * @param context Optional evaluation context.
 * @param out_ball Receives the best formal enclosure found.
 * @return true when a formal enclosure was produced.
 */
static bool sim_zeta_eval_ball_formal_real_series(SimComplexDouble      s_in,
                                                  const SimZetaContext* context,
                                                  SimComplexBall*       out_ball) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    if (out_ball == NULL || !sim_zeta_is_real_right_half_formal_candidate(s_in, ctx)) {
        return false;
    }

    long double sigma = (long double) s_in.re;
    size_t      corrections = ctx->euler_maclaurin_terms;
    if (corrections == 0U) {
        corrections = 1U;
    }
    if (corrections >= SIM_ZETA_EM_MAX_CORRECTIONS) {
        corrections = SIM_ZETA_EM_MAX_CORRECTIONS - 1U;
    }

    size_t initial_terms = (ctx->initial_terms < 4U) ? 4U : ctx->initial_terms;
    size_t max_terms     = (ctx->max_terms < initial_terms) ? initial_terms : ctx->max_terms;
    long double target   = (long double) sim_zeta_target_tol(ctx->abs_tol, ctx->rel_tol, s_in.re);

    SimComplexBall best = { .center                 = { NAN, NAN },
                            .radius                 = INFINITY,
                            .status                 = SIM_ZETA_STATUS_NO_CONVERGENCE,
                            .flags                  = 0U,
                            .refinement_rounds      = 0U,
                            .working_precision_bits = DBL_MANT_DIG,
                            .rigor                  = SIM_BALL_RIGOR_FORMAL,
                            .validation_passes      = 0U };

    for (size_t n_cutoff = initial_terms; n_cutoff <= max_terms; ) {
        long double partial = 0.0L;
        for (size_t n = 1U; n < n_cutoff; ++n) {
            partial += powl((long double) n, -sigma);
        }

        long double n_ld  = (long double) n_cutoff;
        long double value = partial + powl(n_ld, 1.0L - sigma) / (sigma - 1.0L) +
                            0.5L * powl(n_ld, -sigma);

        long double pochhammer = sigma;
        for (size_t k = 1U; k <= corrections; ++k) {
            long double correction =
                sim_zeta_em_coeffs[k - 1U] * pochhammer *
                powl(n_ld, -sigma - (long double) (2U * k - 1U));
            value += correction;
            pochhammer *= (sigma + (long double) (2U * k - 1U)) *
                          (sigma + (long double) (2U * k));
        }

        long double remainder =
            sim_zeta_em_formal_remainder_bound(sigma, n_ld, corrections);
        long double roundoff =
            16.0L * LDBL_EPSILON *
            ((long double) (n_cutoff + corrections + 8U) * fmaxl(1.0L, fabsl(value)));
        double      center   = (double) value;
        long double cast_err = fabsl(value - (long double) center);
        long double radius   = remainder + roundoff + cast_err;

        if (radius < (long double) best.radius) {
            best.center = (SimComplexDouble) { center, 0.0 };
            best.radius = (double) radius;
            best.status = (radius <= target) ? SIM_ZETA_STATUS_OK : SIM_ZETA_STATUS_NO_CONVERGENCE;
            best.flags  = 0U;
        }
        if (radius <= target) {
            *out_ball = best;
            return true;
        }

        if (n_cutoff == max_terms || n_cutoff > (SIZE_MAX / 2U)) {
            break;
        }
        n_cutoff *= 2U;
    }

    if (isfinite(best.radius)) {
        *out_ball = best;
        return true;
    }
    return false;
}

/**
 * @brief Build a formal ball enclosure from a complex right-half-plane EM series.
 *
 * @param s_in Complex input with Re(s) > 1.
 * @param context Optional evaluation context.
 * @param out_ball Receives the best formal enclosure found.
 * @return true when a formal enclosure was produced.
 */
static bool sim_zeta_eval_ball_formal_complex_series(SimComplexDouble      s_in,
                                                     const SimZetaContext* context,
                                                     SimComplexBall*       out_ball) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    if (out_ball == NULL || !sim_zeta_is_complex_right_half_formal_candidate(s_in, ctx)) {
        return false;
    }

    long double sigma = (long double) s_in.re;
    long double complex s = sim_to_cld(s_in);
    size_t      corrections = ctx->euler_maclaurin_terms;
    if (corrections == 0U) {
        corrections = 1U;
    }
    if (corrections >= SIM_ZETA_EM_MAX_CORRECTIONS) {
        corrections = SIM_ZETA_EM_MAX_CORRECTIONS - 1U;
    }

    size_t initial_terms = (ctx->initial_terms < 4U) ? 4U : ctx->initial_terms;
    size_t max_terms     = (ctx->max_terms < initial_terms) ? initial_terms : ctx->max_terms;

    SimComplexBall best = { .center                 = { NAN, NAN },
                            .radius                 = INFINITY,
                            .status                 = SIM_ZETA_STATUS_NO_CONVERGENCE,
                            .flags                  = 0U,
                            .refinement_rounds      = 0U,
                            .working_precision_bits = DBL_MANT_DIG,
                            .rigor                  = SIM_BALL_RIGOR_FORMAL,
                            .validation_passes      = 0U };

    for (size_t n_cutoff = initial_terms; n_cutoff <= max_terms; ) {
        long double complex partial = 0.0L + 0.0L * I;
        long double         abs_sum = 0.0L;

        for (size_t n = 1U; n < n_cutoff; ++n) {
            long double log_n = logl((long double) n);
            long double complex term = cexpl(-s * log_n);
            partial += term;
            abs_sum += cabsl(term);
        }

        long double n_ld      = (long double) n_cutoff;
        long double log_n_ld  = logl(n_ld);
        long double complex integral_term = cexpl((1.0L - s) * log_n_ld) / (s - 1.0L);
        long double complex half_term     = 0.5L * cexpl(-s * log_n_ld);
        long double complex value         = partial + integral_term + half_term;
        abs_sum += cabsl(integral_term) + cabsl(half_term);

        long double complex pochhammer = s;
        for (size_t k = 1U; k <= corrections; ++k) {
            long double complex correction =
                sim_zeta_em_coeffs[k - 1U] * pochhammer *
                cexpl(-(s + (long double) (2U * k - 1U)) * log_n_ld);
            value += correction;
            abs_sum += cabsl(correction);
            pochhammer *= (s + (long double) (2U * k - 1U)) *
                          (s + (long double) (2U * k));
        }

        long double remainder =
            sim_zeta_em_formal_remainder_bound_complex(s, sigma, n_ld, corrections);
        long double magnitude = cabsl(value);
        long double roundoff =
            32.0L * LDBL_EPSILON *
            ((long double) (n_cutoff + corrections + 8U) *
             fmaxl(1.0L, abs_sum + magnitude));
        SimComplexDouble center = { (double) creall(value), (double) cimagl(value) };
        long double complex center_ld = sim_to_cld(center);
        long double cast_err = cabsl(value - center_ld);
        long double radius   = remainder + roundoff + cast_err;
        long double target   =
            (long double) sim_zeta_target_tol(ctx->abs_tol, ctx->rel_tol, cabs(sim_to_c64(center)));

        if (radius < (long double) best.radius) {
            best.center = center;
            best.radius = (double) radius;
            best.status = (radius <= target) ? SIM_ZETA_STATUS_OK : SIM_ZETA_STATUS_NO_CONVERGENCE;
            best.flags  = 0U;
        }
        if (radius <= target) {
            *out_ball = best;
            return true;
        }

        if (n_cutoff == max_terms || n_cutoff > (SIZE_MAX / 2U)) {
            break;
        }
        n_cutoff *= 2U;
    }

    if (isfinite(best.radius)) {
        *out_ball = best;
        return true;
    }
    return false;
}

/**
 * @brief Construct a zeta result initialized to a status and NaN value.
 *
 * @param branch Branch to record in the result.
 * @param status Status to record in the result.
 * @return Initialized result with NaN value/error fields.
 */
static SimZetaResult sim_zeta_make_status(SimZetaBranch branch, SimZetaStatus status) {
    SimZetaResult out = { 0 };
    out.value            = (SimComplexDouble) { NAN, NAN };
    out.abs_error        = NAN;
    out.rel_error        = NAN;
    out.terms_used       = 0U;
    out.correction_terms = 0U;
    out.refinement_rounds = 0U;
    out.working_precision_bits = DBL_MANT_DIG;
    out.branch           = branch;
    out.status           = status;
    out.flags            = 0U;
    return out;
}

/**
 * @brief Evaluate `base^(exponent_shift - s)` from a precomputed logarithm.
 *
 * @param exponent_shift Real exponent offset.
 * @param s Complex zeta argument.
 * @param log_base Natural log of the positive base.
 * @return Complex power value.
 */
static double complex sim_zeta_pow_real(double exponent_shift, double complex s, double log_base) {
    return cexp((exponent_shift - s) * log_base);
}

/**
 * @brief Evaluate zeta near s = 1 using a short Laurent expansion.
 *
 * @param s Complex argument near one.
 * @param context Optional context controlling pole radius.
 * @return Zeta result with near-pole flags and error estimate.
 */
static SimZetaResult sim_zeta_eval_near_one_internal(double complex s,
                                                     const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaResult out =
        sim_zeta_make_status(SIM_ZETA_BRANCH_NEAR_ONE_LAURENT, SIM_ZETA_STATUS_OK);
    double complex delta = sim_zeta_delta_from_one(s);

    if (sim_zeta_has_pole(s, ctx)) {
        out.status = SIM_ZETA_STATUS_SINGULAR;
        out.flags  = SIM_ZETA_FLAG_NEAR_POLE | SIM_ZETA_FLAG_USED_NEAR_ONE_LAURENT;
        return out;
    }

    double complex value = 1.0 / delta + SIM_EULER_MASCHERONI - SIM_STIELTJES_GAMMA1 * delta;
    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        out.flags  = SIM_ZETA_FLAG_NEAR_POLE | SIM_ZETA_FLAG_USED_NEAR_ONE_LAURENT;
        return out;
    }

    double abs_delta = cabs(delta);
    double magnitude = cabs(value);
    double estimate =
        2.0 * fabs(SIM_STIELTJES_GAMMA1) * abs_delta * abs_delta +
        64.0 * DBL_EPSILON * fmax(1.0, magnitude);

    out.value            = sim_from_c64(value);
    out.abs_error        = estimate;
    out.rel_error        = estimate / fmax(magnitude, DBL_MIN);
    out.terms_used       = 0U;
    out.correction_terms = 2U;
    out.flags            = SIM_ZETA_FLAG_NEAR_POLE | SIM_ZETA_FLAG_USED_NEAR_ONE_LAURENT;
    return out;
}

/**
 * @brief Locate the nearest negative-even trivial-zero center within local radius.
 *
 * @param s Complex argument.
 * @param context Optional context controlling local-expansion radius.
 * @param out_n Receives positive n for center `-2n`, when requested.
 * @param out_center Receives the exact center, when requested.
 * @return true when `s` is close enough to a trivial zero for local expansion.
 */
static bool sim_zeta_find_trivial_zero_center(double complex         s,
                                              const SimZetaContext* context,
                                              int*                  out_n,
                                              double complex*       out_center) {
    double radius = sim_zeta_local_expansion_radius(context);

    if (creal(s) >= 0.0) {
        return false;
    }

    double nearest_even = 2.0 * nearbyint(creal(s) * 0.5);
    if (nearest_even >= 0.0 || fabs(nearest_even) < 1.0) {
        return false;
    }

    double complex center = nearest_even + 0.0 * I;
    if (cabs(s - center) > radius) {
        return false;
    }

    long long n = llround(-0.5 * nearest_even);
    if (n <= 0LL) {
        return false;
    }

    if (out_n != NULL) {
        *out_n = (int) n;
    }
    if (out_center != NULL) {
        *out_center = center;
    }
    return true;
}

/**
 * @brief Estimate zeta derivative at the trivial zero s = -2n.
 *
 * Uses the functional-equation identity involving zeta(2n + 1).
 *
 * @param n Positive trivial-zero index.
 * @param context Context for evaluating zeta(2n + 1).
 * @param out_derivative Receives zeta'(-2n), when requested.
 * @param out_abs_error Receives an absolute error estimate, when requested.
 * @return true when the derivative estimate was produced.
 */
static bool sim_zeta_trivial_zero_derivative(int                   n,
                                             const SimZetaContext* context,
                                             double complex*       out_derivative,
                                             double*               out_abs_error) {
    if (n <= 0) {
        return false;
    }

    double        sign      = ((n & 1) != 0) ? -1.0 : 1.0;
    double        exponent  = 2.0 * (double) n;
    double        scale     = sign * tgamma(exponent + 1.0) /
                          (2.0 * pow(2.0 * M_PI, exponent));
    SimZetaResult zeta_odd =
        sim_zeta_eval_direct_euler_maclaurin((SimComplexDouble) { 2.0 * (double) n + 1.0, 0.0 },
                                             context);
    if (zeta_odd.status != SIM_ZETA_STATUS_OK) {
        return false;
    }

    double complex derivative = scale * zeta_odd.value.re;
    if (!isfinite(creal(derivative)) || !isfinite(cimag(derivative))) {
        return false;
    }

    if (out_derivative != NULL) {
        *out_derivative = derivative;
    }
    if (out_abs_error != NULL) {
        *out_abs_error =
            fabs(scale) * zeta_odd.abs_error + 64.0 * DBL_EPSILON * fmax(1.0, cabs(derivative));
    }
    return true;
}

/**
 * @brief Evaluate zeta near a trivial zero by first-order local expansion.
 *
 * @param s Complex argument near a negative even integer.
 * @param context Optional zeta context.
 * @return Local-expansion result or an error status.
 */
static SimZetaResult sim_zeta_eval_local_expansion_internal(double complex        s,
                                                            const SimZetaContext* context) {
    SimZetaResult out =
        sim_zeta_make_status(SIM_ZETA_BRANCH_LOCAL_EXPANSION, SIM_ZETA_STATUS_OK);
    int           n       = 0;
    double complex center = 0.0;

    if (!sim_zeta_find_trivial_zero_center(s, context, &n, &center)) {
        out.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return out;
    }

    double complex derivative     = 0.0;
    double         derivative_err = 0.0;
    if (!sim_zeta_trivial_zero_derivative(n, context, &derivative, &derivative_err)) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double complex delta = s - center;
    double complex value = derivative * delta;
    double        radius = cabs(delta);
    double        estimate =
        radius * derivative_err +
        2.0 * radius * radius * fmax(1.0, cabs(derivative)) +
        64.0 * DBL_EPSILON * fmax(1.0, cabs(value));

    out.value            = sim_from_c64(value);
    out.abs_error        = estimate;
    out.rel_error        = estimate / fmax(cabs(value), DBL_MIN);
    out.terms_used       = 1U;
    out.correction_terms = 1U;
    out.flags            = SIM_ZETA_FLAG_USED_LOCAL_EXPANSION | SIM_ZETA_FLAG_ZERO_PROXIMITY |
                SIM_ZETA_FLAG_TRIVIAL_ZERO;
    return out;
}

/**
 * @brief Evaluate the functional-equation chi(s) factor.
 *
 * @param s Complex argument.
 * @param out_chi Receives chi(s), when requested.
 * @param out_abs_error Receives propagated absolute error, when requested.
 * @return true when chi(s) was finite and evaluable.
 */
static bool sim_zeta_chi_eval(double complex s,
                              double complex* out_chi,
                              double*         out_abs_error) {
    SimLogGammaResult lgamma =
        sim_log_gamma_eval((SimComplexDouble) { 1.0 - creal(s), -cimag(s) });
    if (lgamma.status != SIM_LOG_GAMMA_STATUS_OK) {
        return false;
    }

    double complex log_chi =
        s * log(2.0) + (s - 1.0) * log(M_PI) + clog(csin(0.5 * M_PI * s)) +
        sim_to_c64(lgamma.value);
    double complex chi = cexp(log_chi);
    if (!isfinite(creal(chi)) || !isfinite(cimag(chi))) {
        return false;
    }

    if (out_chi != NULL) {
        *out_chi = chi;
    }
    if (out_abs_error != NULL) {
        *out_abs_error =
            cabs(chi) * (lgamma.abs_error + 64.0 * DBL_EPSILON * (1.0 + cabs(log_chi)));
    }
    return true;
}

/**
 * @brief Evaluate one Hasse eta-acceleration term with compensated inner sum.
 *
 * @param s Complex zeta argument.
 * @param n Hasse term index.
 * @return Complex accelerated term.
 */
static double complex sim_zeta_hasse_term(double complex s, size_t n) {
    SimComplexKahan inner = { 0.0, 0.0, 0.0, 0.0 };
    double          binom = 1.0;
    double          sign  = 1.0;

    for (size_t k = 0U; k <= n; ++k) {
        double complex basis = cexp(-s * log((double) (k + 1U)));
        sim_complex_kahan_add(&inner, sign * binom * basis);

        if (k < n) {
            binom *= (double) (n - k) / (double) (k + 1U);
            sign = -sign;
        }
    }

    return ldexp(1.0, -(int) (n + 1U)) * sim_complex_kahan_value(&inner);
}

/**
 * @brief Evaluate the Riemann-Siegel theta phase.
 *
 * @param t Critical-line height.
 * @param out_theta Receives theta(t), when requested.
 * @return true when log-gamma evaluation succeeded.
 */
static bool sim_zeta_riemann_siegel_theta(double t, double* out_theta) {
    SimLogGammaResult lgamma =
        sim_log_gamma_eval((SimComplexDouble) { 0.25, 0.5 * t });
    if (lgamma.status != SIM_LOG_GAMMA_STATUS_OK) {
        return false;
    }

    if (out_theta != NULL) {
        *out_theta = lgamma.value.im - 0.5 * t * log(M_PI);
    }
    return true;
}

/**
 * @brief Evaluate zeta through the functional-equation reflection path.
 *
 * @param s Complex argument, typically with negative real part.
 * @param context Optional zeta context.
 * @return Reflected zeta result with propagated error estimate.
 */
static SimZetaResult sim_zeta_eval_reflection_internal(double complex s,
                                                       const SimZetaContext* context) {
    SimZetaResult out = sim_zeta_make_status(SIM_ZETA_BRANCH_REFLECTION, SIM_ZETA_STATUS_OK);
    double        tol = sim_zeta_real_tol(context);

    if (sim_zeta_is_negative_even_integer(s, tol)) {
        out.value            = (SimComplexDouble) { 0.0, 0.0 };
        out.abs_error        = 0.0;
        out.rel_error        = 0.0;
        out.terms_used       = 0U;
        out.correction_terms = 0U;
        out.flags            = SIM_ZETA_FLAG_USED_REFLECTION | SIM_ZETA_FLAG_TRIVIAL_ZERO;
        return out;
    }

    SimComplexDouble reflected_arg = { 1.0 - creal(s), -cimag(s) };
    SimZetaResult reflected = sim_zeta_eval(reflected_arg, context);
    if (reflected.status != SIM_ZETA_STATUS_OK) {
        reflected.branch = SIM_ZETA_BRANCH_REFLECTION;
        reflected.flags |= SIM_ZETA_FLAG_USED_REFLECTION;
        return reflected;
    }

    SimLogGammaResult lgamma =
        sim_log_gamma_eval((SimComplexDouble) { 1.0 - creal(s), -cimag(s) });
    if (lgamma.status != SIM_LOG_GAMMA_STATUS_OK) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double complex log_prefactor =
        s * log(2.0) + (s - 1.0) * log(M_PI) + clog(csin(0.5 * M_PI * s)) +
        sim_to_c64(lgamma.value);
    double complex reflected_value = sim_to_c64(reflected.value);
    double complex prefactor       = cexp(log_prefactor);
    double complex value           = prefactor * reflected_value;
    if ((!isfinite(creal(value)) || !isfinite(cimag(value))) &&
        cabs(reflected_value) > DBL_MIN) {
        value = cexp(log_prefactor + clog(reflected_value));
    }
    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    out.value            = sim_from_c64(value);
    out.terms_used       = reflected.terms_used;
    out.correction_terms = reflected.correction_terms;
    out.flags            = reflected.flags | SIM_ZETA_FLAG_USED_REFLECTION;
    out.abs_error =
        cabs(prefactor) * reflected.abs_error +
        cabs(value) * (lgamma.abs_error + 64.0 * DBL_EPSILON * (1.0 + cabs(log_prefactor)));
    sim_zeta_update_rel_error(&out);
    return out;
}

/**
 * @brief Evaluate zeta with direct Euler-Maclaurin summation.
 *
 * This branch handles the pole, zero at s = 0, trivial zeros, and the near-one
 * Laurent expansion before running an adaptive cutoff loop with Bernoulli
 * corrections.
 *
 * @param s_in Complex zeta argument.
 * @param context Optional context controlling tolerances and term budgets.
 * @return Structured zeta result with branch, status, and error estimate.
 */
SimZetaResult sim_zeta_eval_direct_euler_maclaurin(SimComplexDouble s_in,
                                                   const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaResult         best =
        sim_zeta_make_status(SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN, SIM_ZETA_STATUS_NO_CONVERGENCE);

    if (!sim_zeta_isfinite(s_in)) {
        best.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return best;
    }

    double complex s = sim_to_c64(s_in);
    double         tol_near_pole = sim_zeta_real_tol(ctx);

    if (sim_zeta_has_pole(s, ctx)) {
        best.status = SIM_ZETA_STATUS_SINGULAR;
        best.flags  = SIM_ZETA_FLAG_NEAR_POLE;
        return best;
    }

    if (sim_zeta_is_near_one(s, ctx)) {
        return sim_zeta_eval_near_one_internal(s, ctx);
    }

    if (sim_zeta_is_near_real_integer(s, 0.0, tol_near_pole)) {
        best.value            = (SimComplexDouble) { -0.5, 0.0 };
        best.abs_error        = 0.0;
        best.rel_error        = 0.0;
        best.terms_used       = 0U;
        best.correction_terms = 0U;
        best.status           = SIM_ZETA_STATUS_OK;
        return best;
    }

    if (sim_zeta_is_negative_even_integer(s, tol_near_pole)) {
        best.value            = (SimComplexDouble) { 0.0, 0.0 };
        best.abs_error        = 0.0;
        best.rel_error        = 0.0;
        best.terms_used       = 0U;
        best.correction_terms = 0U;
        best.status           = SIM_ZETA_STATUS_OK;
        best.flags            = SIM_ZETA_FLAG_TRIVIAL_ZERO;
        return best;
    }

    size_t requested_corrections = ctx->euler_maclaurin_terms;
    if (requested_corrections == 0U) {
        requested_corrections = 1U;
    }
    if (requested_corrections >= SIM_ZETA_EM_MAX_CORRECTIONS) {
        requested_corrections = SIM_ZETA_EM_MAX_CORRECTIONS - 1U;
    }

    size_t initial_terms = (ctx->initial_terms < 4U) ? 4U : ctx->initial_terms;
    size_t max_terms     = (ctx->max_terms < initial_terms) ? initial_terms : ctx->max_terms;

    for (size_t n_cutoff = initial_terms; n_cutoff <= max_terms; ) {
        double log_n = log((double) n_cutoff);

        SimComplexKahan partial = { 0.0, 0.0, 0.0, 0.0 };
        for (size_t n = 1U; n < n_cutoff; ++n) {
            double log_term = log((double) n);
            sim_complex_kahan_add(&partial, cexp(-s * log_term));
        }

        double complex value = sim_complex_kahan_value(&partial);
        value += cexp((1.0 - s) * log_n) / (s - 1.0);
        value += 0.5 * cexp(-s * log_n);

        double complex last_correction = 0.0;
        double complex next_correction = 0.0;
        double complex pochhammer      = s;

        for (size_t k = 1U; k <= requested_corrections; ++k) {
            double complex correction =
                (double) sim_zeta_em_coeffs[k - 1U] * pochhammer * sim_zeta_pow_real(-(double) (2U * k - 1U), s, log_n);
            value += correction;
            last_correction = correction;

            double complex next_pochhammer =
                pochhammer * (s + (double) (2U * k - 1U)) * (s + (double) (2U * k));
            if (k < SIM_ZETA_EM_MAX_CORRECTIONS) {
                next_correction =
                    (double) sim_zeta_em_coeffs[k] * next_pochhammer *
                    sim_zeta_pow_real(-(double) (2U * (k + 1U) - 1U), s, log_n);
            }
            pochhammer = next_pochhammer;
        }

        if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
            best.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
            return best;
        }

        double magnitude = cabs(value);
        double roundoff  = 64.0 * DBL_EPSILON *
                          (double) (n_cutoff + requested_corrections + 4U) * fmax(1.0, magnitude);
        double estimate  = cabs(next_correction);
        if (estimate < cabs(last_correction)) {
            estimate += roundoff;
        } else {
            estimate = estimate + roundoff;
        }

        best.value            = sim_from_c64(value);
        best.abs_error        = estimate;
        best.rel_error        = estimate / fmax(magnitude, DBL_MIN);
        best.terms_used       = n_cutoff - 1U;
        best.correction_terms = requested_corrections;
        best.status           = SIM_ZETA_STATUS_OK;

        double tol = fmax(ctx->abs_tol, ctx->rel_tol * fmax(1.0, magnitude));
        if (estimate <= tol) {
            return best;
        }

        if (n_cutoff == max_terms || n_cutoff > (SIZE_MAX / 2U)) {
            break;
        }
        n_cutoff *= 2U;
    }

    return best;
}

/**
 * @brief Evaluate zeta with the eta/Hasse accelerated branch.
 *
 * The Hasse series is accumulated until the configured term budget or target
 * tolerance is reached, then converted from eta-like values by the standard
 * denominator `1 - 2^(1-s)`.
 *
 * @param s_in Complex zeta argument.
 * @param context Optional context controlling tolerances and eta budgets.
 * @return Structured zeta result for the accelerated branch.
 */
SimZetaResult sim_zeta_eval_eta_accelerated(SimComplexDouble s_in,
                                            const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaResult         best =
        sim_zeta_make_status(SIM_ZETA_BRANCH_ETA_ACCELERATED, SIM_ZETA_STATUS_NO_CONVERGENCE);

    if (!sim_zeta_isfinite(s_in)) {
        best.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return best;
    }

    double complex s = sim_to_c64(s_in);
    double         tol_near_pole = sim_zeta_real_tol(ctx);

    if (sim_zeta_has_pole(s, ctx)) {
        best.status = SIM_ZETA_STATUS_SINGULAR;
        best.flags  = SIM_ZETA_FLAG_NEAR_POLE;
        return best;
    }

    if (sim_zeta_is_near_one(s, ctx)) {
        return sim_zeta_eval_near_one_internal(s, ctx);
    }

    if (sim_zeta_is_near_real_integer(s, 0.0, tol_near_pole)) {
        best.value            = (SimComplexDouble) { -0.5, 0.0 };
        best.abs_error        = 0.0;
        best.rel_error        = 0.0;
        best.terms_used       = 0U;
        best.correction_terms = 0U;
        best.status           = SIM_ZETA_STATUS_OK;
        best.flags            = SIM_ZETA_FLAG_USED_ETA;
        return best;
    }

    size_t initial_terms = (ctx->eta_initial_terms < 4U) ? 4U : ctx->eta_initial_terms;
    size_t max_terms     = (ctx->eta_max_terms < initial_terms) ? initial_terms : ctx->eta_max_terms;

    double complex denominator = 1.0 - cexp((1.0 - s) * log(2.0));
    if (cabs(denominator) <= DBL_MIN) {
        best.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        best.flags  = SIM_ZETA_FLAG_USED_ETA | SIM_ZETA_FLAG_NEAR_POLE;
        return best;
    }

    SimComplexKahan numerator_acc = { 0.0, 0.0, 0.0, 0.0 };
    double complex  last_term     = 0.0;

    for (size_t n = 0U; n < max_terms; ++n) {
        double complex term = sim_zeta_hasse_term(s, n);
        sim_complex_kahan_add(&numerator_acc, term);
        last_term = term;

        double complex numerator = sim_complex_kahan_value(&numerator_acc);
        double complex value     = numerator / denominator;
        if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
            best.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
            best.flags  = SIM_ZETA_FLAG_USED_ETA;
            return best;
        }

        double magnitude = cabs(value);
        double estimate  = (cabs(last_term) +
                           64.0 * DBL_EPSILON * (double) (n + 2U) * fmax(1.0, cabs(numerator))) /
                          fmax(cabs(denominator), DBL_MIN);

        best.value            = sim_from_c64(value);
        best.abs_error        = estimate;
        best.rel_error        = estimate / fmax(magnitude, DBL_MIN);
        best.terms_used       = n + 1U;
        best.correction_terms = 0U;
        best.status           = SIM_ZETA_STATUS_OK;
        best.flags            = SIM_ZETA_FLAG_USED_ETA;

        if ((n + 1U) >= initial_terms) {
            double tol = fmax(ctx->abs_tol, ctx->rel_tol * fmax(1.0, magnitude));
            if (estimate <= tol) {
                return best;
            }
        }
    }

    return best;
}

/**
 * @brief Evaluate zeta with an approximate functional-equation cross-check.
 *
 * The branch evaluates both `zeta(s)` and the reflected companion through the
 * direct Euler-Maclaurin path, combines them with chi(s), and uses their
 * mismatch as part of the error estimate.
 *
 * @param s_in Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Structured zeta result for the approximate functional equation branch.
 */
SimZetaResult sim_zeta_eval_approximate_fe(SimComplexDouble s_in,
                                           const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaResult         best =
        sim_zeta_make_status(SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION,
                             SIM_ZETA_STATUS_NO_CONVERGENCE);
    if (!sim_zeta_isfinite(s_in)) {
        best.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return best;
    }

    double complex s = sim_to_c64(s_in);
    double         tol_near_pole = sim_zeta_real_tol(ctx);
    if (sim_zeta_has_pole(s, ctx)) {
        best.status = SIM_ZETA_STATUS_SINGULAR;
        best.flags  = SIM_ZETA_FLAG_NEAR_POLE;
        return best;
    }
    if (sim_zeta_is_near_one(s, ctx)) {
        return sim_zeta_eval_near_one_internal(s, ctx);
    }
    if (sim_zeta_is_near_real_integer(s, 0.0, tol_near_pole)) {
        best.value            = (SimComplexDouble) { -0.5, 0.0 };
        best.abs_error        = 0.0;
        best.rel_error        = 0.0;
        best.terms_used       = 0U;
        best.correction_terms = 0U;
        best.status           = SIM_ZETA_STATUS_OK;
        best.flags            = SIM_ZETA_FLAG_USED_AFE;
        return best;
    }

    double complex chi     = 0.0;
    double         chi_err = 0.0;
    if (!sim_zeta_chi_eval(s, &chi, &chi_err)) {
        best.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        best.flags  = SIM_ZETA_FLAG_USED_AFE;
        return best;
    }

    SimZetaResult forward = sim_zeta_eval_direct_euler_maclaurin(s_in, ctx);
    if (forward.status != SIM_ZETA_STATUS_OK) {
        forward.branch = SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION;
        forward.flags |= SIM_ZETA_FLAG_USED_AFE;
        return forward;
    }

    SimComplexDouble reflected_arg = { 1.0 - creal(s), -cimag(s) };
    SimZetaResult companion        = sim_zeta_eval_direct_euler_maclaurin(reflected_arg, ctx);
    if (companion.status != SIM_ZETA_STATUS_OK) {
        companion.branch = SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION;
        companion.flags |= SIM_ZETA_FLAG_USED_AFE;
        return companion;
    }

    double complex forward_value   = sim_to_c64(forward.value);
    double complex companion_value = sim_to_c64(companion.value);
    double complex reflected_value = chi * companion_value;
    double complex value           = 0.5 * (forward_value + reflected_value);
    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        best.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        best.flags  = SIM_ZETA_FLAG_USED_AFE;
        return best;
    }

    double mismatch = cabs(forward_value - reflected_value);
    double reflected_error =
        cabs(chi) * companion.abs_error + chi_err * cabs(companion_value);
    double estimate = 0.5 * (forward.abs_error + reflected_error + mismatch);
    double magnitude = cabs(value);

    best.value            = sim_from_c64(value);
    best.abs_error        = estimate;
    best.rel_error        = estimate / fmax(magnitude, DBL_MIN);
    best.terms_used       = forward.terms_used + companion.terms_used;
    best.correction_terms = forward.correction_terms + companion.correction_terms;
    best.status           = SIM_ZETA_STATUS_OK;
    best.flags            = SIM_ZETA_FLAG_USED_AFE;
    return best;
}

/**
 * @brief Evaluate zeta on the critical line with a Riemann-Siegel approximation.
 *
 * For small `|t|`, this function delegates to the approximate functional
 * equation branch. For larger heights, it evaluates the real Z-function main
 * sum and a compact correction term, then rotates back by the theta phase.
 *
 * @param t Critical-line height in `s = 1/2 + i t`.
 * @param context Optional evaluation context.
 * @return Structured zeta result on the critical line.
 */
SimZetaResult sim_zeta_eval_riemann_siegel(double t, const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaResult         out =
        sim_zeta_make_status(SIM_ZETA_BRANCH_RIEMANN_SIEGEL, SIM_ZETA_STATUS_OK);
    double                at = fabs(t);

    if (!isfinite(t)) {
        out.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return out;
    }

    if (at < ctx->riemann_siegel_imag_min) {
        SimZetaResult fallback_result =
            sim_zeta_eval_approximate_fe((SimComplexDouble) { 0.5, t }, ctx);
        fallback_result.flags |= SIM_ZETA_FLAG_USED_RIEMANN_SIEGEL;
        return fallback_result;
    }

    double tau = sqrt(at / (2.0 * M_PI));
    size_t n_terms = (size_t) floor(tau);
    if (n_terms < 1U) {
        n_terms = 1U;
    }
    if (n_terms > ctx->riemann_siegel_max_terms) {
        n_terms = ctx->riemann_siegel_max_terms;
    }

    double theta = 0.0;
    if (!sim_zeta_riemann_siegel_theta(at, &theta)) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    SimComplexKahan acc = { 0.0, 0.0, 0.0, 0.0 };
    for (size_t n = 1U; n <= n_terms; ++n) {
        double phase = theta - at * log((double) n);
        sim_complex_kahan_add(&acc, cos(phase) / sqrt((double) n));
    }

    double z_main = 2.0 * creal(sim_complex_kahan_value(&acc));
    double a      = tau - floor(tau);
    double correction = 0.0;
    double denom      = cos(2.0 * M_PI * a);
    if (fabs(denom) > 1.0e-8) {
        double phi =
            cos(2.0 * M_PI * (a * a - a - 0.0625)) / denom;
        double sign = ((n_terms & 1U) != 0U) ? 1.0 : -1.0;
        correction  = sign * pow(at / (2.0 * M_PI), -0.25) * phi;
    }

    double        z_value = z_main + correction;
    double complex value  = cexp(-I * theta) * z_value;
    if (t < 0.0) {
        value = conj(value);
    }

    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    double estimate =
        fabs(correction) * (pow(fmax(at, 1.0), -0.5) + 64.0 * DBL_EPSILON) +
        2.0 * pow(fmax(at, 1.0), -0.75);

    out.value            = sim_from_c64(value);
    out.abs_error        = estimate;
    out.rel_error        = estimate / fmax(cabs(value), DBL_MIN);
    out.terms_used       = n_terms;
    out.correction_terms = 1U;
    out.flags            = SIM_ZETA_FLAG_USED_RIEMANN_SIEGEL;
    return out;
}

/**
 * @brief Select and run one zeta branch without adaptive refinement.
 *
 * @param s Complex zeta argument.
 * @param context Refined context for this dispatch attempt.
 * @return Result from the chosen branch.
 */
static SimZetaResult sim_zeta_eval_dispatch_once(SimComplexDouble s,
                                                 const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;

    if (!sim_zeta_isfinite(s)) {
        SimZetaResult out =
            sim_zeta_make_status(SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN, SIM_ZETA_STATUS_INVALID_ARGUMENT);
        return out;
    }

    if (sim_zeta_is_negative_even_integer(sim_to_c64(s), sim_zeta_real_tol(ctx))) {
        return sim_zeta_eval_reflection_internal(sim_to_c64(s), ctx);
    }

    if (sim_zeta_find_trivial_zero_center(sim_to_c64(s), ctx, NULL, NULL)) {
        SimZetaResult local = sim_zeta_eval_local_expansion_internal(sim_to_c64(s), ctx);
        if (sim_zeta_result_meets_target(&local, ctx)) {
            return local;
        }
    }

    if (s.re < 0.0) {
        return sim_zeta_eval_reflection_internal(sim_to_c64(s), ctx);
    }

    if (sim_zeta_is_near_one(sim_to_c64(s), ctx)) {
        return sim_zeta_eval_near_one_internal(sim_to_c64(s), ctx);
    }

    if (sim_zeta_is_near_critical_line(sim_to_c64(s), ctx) &&
        fabs(s.im) >= ctx->riemann_siegel_imag_min) {
        return sim_zeta_eval_riemann_siegel(s.im, ctx);
    }

    if (s.re <= ctx->afe_sigma_limit && fabs(s.im) >= ctx->afe_imag_min) {
        return sim_zeta_eval_approximate_fe(s, ctx);
    }

    if (s.re <= ctx->eta_sigma_limit && fabs(s.im) <= ctx->eta_imag_limit) {
        return sim_zeta_eval_eta_accelerated(s, ctx);
    }

    return sim_zeta_eval_direct_euler_maclaurin(s, ctx);
}

/**
 * @brief Evaluate zeta using dispatcher selection and adaptive refinement.
 *
 * The dispatcher is rerun with tighter contexts until the result meets the base
 * tolerance, consecutive refinement rounds agree, or the maximum round count is
 * exhausted.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context; defaults are used when NULL.
 * @return Best structured zeta result found by the dispatcher.
 */
SimZetaResult sim_zeta_eval(SimComplexDouble s, const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* base    = (context != NULL) ? context : &fallback;
    SimZetaResult         previous = sim_zeta_make_status(
        SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN, SIM_ZETA_STATUS_INVALID_ARGUMENT);
    SimZetaResult last = previous;
    bool          have_previous = false;

    if (!sim_zeta_isfinite(s)) {
        last.status = SIM_ZETA_STATUS_INVALID_ARGUMENT;
        return last;
    }

    for (size_t round = 0U; round <= base->adaptive_max_rounds; ++round) {
        SimZetaContext refined = sim_zeta_context_refined(base, round);
        SimZetaResult  current = sim_zeta_eval_dispatch_once(s, &refined);
        double         magnitude = cabs(sim_to_c64(current.value));
        double         target    =
            sim_zeta_target_tol(base->abs_tol, base->rel_tol, magnitude);
        current.refinement_rounds   = round;
        current.working_precision_bits = DBL_MANT_DIG;
        if (round > 0U) {
            current.flags |= SIM_ZETA_FLAG_USED_ADAPTIVE_REFINEMENT;
        }

        if (current.status != SIM_ZETA_STATUS_OK && current.status != SIM_ZETA_STATUS_NO_CONVERGENCE) {
            return current;
        }

        if (have_previous && isfinite(current.value.re) && isfinite(current.value.im) &&
            isfinite(previous.value.re) && isfinite(previous.value.im)) {
            double disagreement = cabs(sim_to_c64(current.value) - sim_to_c64(previous.value));
            if (!isfinite(current.abs_error) || disagreement > current.abs_error) {
                current.abs_error = disagreement;
                sim_zeta_update_rel_error(&current);
            }

            if (disagreement <= target && sim_zeta_result_meets_target(&current, base)) {
                current.status = SIM_ZETA_STATUS_OK;
                return current;
            }
        }

        if (sim_zeta_result_meets_target(&current, base)) {
            current.status = SIM_ZETA_STATUS_OK;
            return current;
        }

        current.status = SIM_ZETA_STATUS_NO_CONVERGENCE;

        previous      = current;
        last          = current;
        have_previous = true;
    }

    last.status = SIM_ZETA_STATUS_NO_CONVERGENCE;
    return last;
}

/**
 * @brief Evaluate zeta and estimate its first complex derivative.
 *
 * The derivative is estimated by centered imaginary-direction differences at
 * two step sizes followed by Richardson extrapolation. The derivative status is
 * downgraded if neighboring evaluations fail or choose a different branch.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Zeta value, derivative estimate, errors, and status metadata.
 */
SimZetaDerivativeResult sim_zeta_eval_with_derivative(SimComplexDouble s,
                                                      const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaDerivativeResult out   = { 0 };

    SimZetaResult value = sim_zeta_eval(s, ctx);
    out.value                  = value.value;
    out.abs_error              = value.abs_error;
    out.rel_error              = value.rel_error;
    out.refinement_rounds      = value.refinement_rounds;
    out.working_precision_bits = value.working_precision_bits;
    out.branch                 = value.branch;
    out.status                 = value.status;
    out.flags                  = value.flags;
    out.derivative             = (SimComplexDouble) { NAN, NAN };
    out.derivative_abs_error   = NAN;
    out.derivative_rel_error   = NAN;
    if (value.status != SIM_ZETA_STATUS_OK) {
        return out;
    }

    double complex cs = sim_to_c64(s);
    if (sim_zeta_has_pole(cs, ctx)) {
        out.status = SIM_ZETA_STATUS_SINGULAR;
        return out;
    }

    double scale = fmax(1.0, cabs(cs));
    double h1    = fmax(8.0 * sim_zeta_real_tol(ctx), cbrt(DBL_EPSILON) * scale);
    if (value.branch == SIM_ZETA_BRANCH_LOCAL_EXPANSION) {
        h1 = fmin(h1, sqrt(fmax(ctx->abs_tol, 64.0 * DBL_EPSILON)));
    }
    double h2    = 0.5 * h1;

    SimZetaResult fp1 = sim_zeta_eval((SimComplexDouble) { s.re, s.im + h1 }, ctx);
    SimZetaResult fm1 = sim_zeta_eval((SimComplexDouble) { s.re, s.im - h1 }, ctx);
    SimZetaResult fp2 = sim_zeta_eval((SimComplexDouble) { s.re, s.im + h2 }, ctx);
    SimZetaResult fm2 = sim_zeta_eval((SimComplexDouble) { s.re, s.im - h2 }, ctx);
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
    if (fp1.branch != value.branch || fm1.branch != value.branch || fp2.branch != value.branch ||
        fm2.branch != value.branch) {
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
 * @brief Build a ball-style enclosure from zeta value/error estimates.
 *
 * The routine starts with the dispatcher error model, upgrades to formal
 * right-half-plane Euler-Maclaurin balls when possible, and optionally validates
 * the ball against an independent peer branch.
 *
 * @param s Complex zeta argument.
 * @param context Optional evaluation context.
 * @return Complex ball with center, radius, status, rigor, and validation metadata.
 */
SimComplexBall sim_zeta_eval_ball(SimComplexDouble s, const SimZetaContext* context) {
    const SimZetaContext fallback = sim_zeta_context_make_default();
    const SimZetaContext* ctx     = (context != NULL) ? context : &fallback;
    SimZetaResult result = sim_zeta_eval(s, ctx);
    SimComplexBall out   = { .center                 = result.value,
                           .radius                 = result.abs_error,
                           .status                 = result.status,
                           .flags                  = result.flags,
                           .refinement_rounds      = result.refinement_rounds,
                           .working_precision_bits = result.working_precision_bits,
                           .rigor                  = SIM_BALL_RIGOR_HEURISTIC,
                           .validation_passes      = 0U };

    if (out.status != SIM_ZETA_STATUS_OK && out.status != SIM_ZETA_STATUS_NO_CONVERGENCE) {
        return out;
    }
    if (result.status == SIM_ZETA_STATUS_OK && result.abs_error == 0.0) {
        out.rigor = SIM_BALL_RIGOR_FORMAL;
        return out;
    }
    if (!isfinite(out.center.re) || !isfinite(out.center.im) ||
        !isfinite(out.radius) || out.radius < 0.0) {
        out.status = SIM_ZETA_STATUS_NUMERIC_FAILURE;
        return out;
    }
    sim_zeta_ball_update_radius(&out);
    {
        SimComplexBall formal = { 0 };
        if (sim_zeta_eval_ball_formal_real_series(s, ctx, &formal) ||
            sim_zeta_eval_ball_formal_complex_series(s, ctx, &formal)) {
            sim_zeta_ball_update_radius(&formal);
            if (formal.status == SIM_ZETA_STATUS_OK || formal.radius <= out.radius) {
                return formal;
            }
        }
    }
    if (out.status != SIM_ZETA_STATUS_OK) {
        return out;
    }
    if (ctx->ball_validation_rounds == 0U) {
        return out;
    }

    for (size_t round = 0U; round < ctx->ball_validation_rounds; ++round) {
        SimZetaResult peer = { 0 };
        if (!sim_zeta_eval_validation_peer(s, result.branch, ctx, round, &peer) ||
            peer.status != SIM_ZETA_STATUS_OK) {
            break;
        }

        double complex center_primary = sim_to_c64(out.center);
        double complex center_peer    = sim_to_c64(peer.value);
        double        peer_radius     = peer.abs_error;
        double        disagreement    = cabs(center_primary - center_peer);
        double        scale           = fmax(1.0, ctx->ball_validation_scale);
        SimComplexBall peer_ball      = { .center                 = peer.value,
                                     .radius                 = peer_radius,
                                     .status                 = peer.status,
                                     .flags                  = peer.flags,
                                     .refinement_rounds      = peer.refinement_rounds,
                                     .working_precision_bits = peer.working_precision_bits,
                                     .rigor                  = SIM_BALL_RIGOR_HEURISTIC,
                                     .validation_passes      = 0U };

        sim_zeta_ball_update_radius(&peer_ball);
        if (disagreement > scale * fmax(out.radius, peer_ball.radius)) {
            break;
        }

        out.center = sim_from_c64(0.5 * (center_primary + center_peer));
        out.radius = 0.5 * disagreement + fmax(out.radius, peer_ball.radius);
        out.flags |= peer.flags;
        if (peer.refinement_rounds > out.refinement_rounds) {
            out.refinement_rounds = peer.refinement_rounds;
        }
        if (peer.working_precision_bits > out.working_precision_bits) {
            out.working_precision_bits = peer.working_precision_bits;
        }
        out.rigor = SIM_BALL_RIGOR_VALIDATED;
        out.validation_passes += 1U;
        sim_zeta_ball_update_radius(&out);
    }

    return out;
}
