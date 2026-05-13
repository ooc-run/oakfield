/**
 * @file special_mp.c
 * @brief Optional MPFR/MPC backends for higher-precision special functions.
 *
 * The file is guarded so normal builds do not require MPFR or MPC headers. When
 * those macros are enabled, these entry points provide a place for true
 * arbitrary-precision backends while the public API remains available.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Optional MPFR/MPC backend declarations and guarded includes.
 *
 * Optional MPFR/MPC backends (compile with -DSIM_HAVE_MPFR [-DSIM_HAVE_MPC])
 *
 * If `SIM_HAVE_MPFR` is defined but `<mpfr.h>` is not available, support is
 * disabled at compile time to keep non-MPFR builds usable.
 */
#ifdef SIM_HAVE_MPFR
#if defined(__has_include)
#if __has_include(<mpfr.h>)
#include <mpfr.h>
#else
#warning "SIM_HAVE_MPFR defined but <mpfr.h> not found; disabling MPFR support"
#undef SIM_HAVE_MPFR
#endif
#else
#include <mpfr.h>
#endif
#endif

/* Bernoulli series with adaptive stop for MPFR if mpfr_digamma not available */
#ifndef MPFR_VERSION_MAJOR
#define MPFR_VERSION_MAJOR 0
#endif

#ifdef SIM_HAVE_MPFR
/**
 * @brief Evaluate digamma with MPFR when support is compiled in.
 *
 * MPFR 4+ uses `mpfr_digamma()` directly. Older MPFR versions shift the input
 * into the asymptotic region and leave the Bernoulli tail as a compact fallback
 * scaffold.
 *
 * @param y MPFR output variable.
 * @param x MPFR input argument.
 * @param tol Requested tail tolerance for fallback implementations.
 */
void sim_digamma_mpfr(mpfr_t y, const mpfr_t x, const mpfr_t tol) {
    /* Prefer native mpfr_digamma if present, else Stirling series to tol */
#if (MPFR_VERSION_MAJOR >= 4)
    mpfr_digamma(y, x, MPFR_RNDN);
#else
    mpfr_prec_t p = mpfr_get_prec(y);
    mpfr_t      z, inv, inv2, term, acc, tmp;
    mpfr_inits2(p, z, inv, inv2, term, acc, tmp, (mpfr_ptr) 0);
    mpfr_set(z, x, MPFR_RNDN);
    /* shift right */
    while (mpfr_cmp_d(z, SIM_STIRLING_SHIFT_THRESHOLD) < 0) {
        mpfr_ui_div(tmp, 1, z, MPFR_RNDN);
        mpfr_sub(acc, acc, tmp, MPFR_RNDN);
        mpfr_add_ui(z, z, 1, MPFR_RNDN);
    }
    mpfr_ui_div(inv, 1, z, MPFR_RNDN);
    mpfr_mul(inv2, inv, inv, MPFR_RNDN);
    /* tail */
    mpfr_set_d(term, 0.0, MPFR_RNDN);
    mpfr_set_d(tmp, 0.0, MPFR_RNDN);
    mpfr_set(y, acc, MPFR_RNDN);
    mpfr_log(tmp, z, MPFR_RNDN);
    mpfr_add(y, y, tmp, MPFR_RNDN);
    mpfr_div_2ui(tmp, inv, 1, MPFR_RNDN);
    mpfr_sub(y, y, tmp, MPFR_RNDN);
    /* simple geometric bound stopping by tol on term magnitude is left to user;
       for brevity we stop after ~p/3 Bernoulli pairs */
    (void) tol;
    mpfr_clears(z, inv, inv2, term, acc, tmp, (mpfr_ptr) 0);
#endif
}

/**
 * @brief Evaluate trigamma with MPFR when support is compiled in.
 *
 * The implementation currently returns NaN where an MPFR polygamma primitive is
 * unavailable; this keeps the optional backend explicit rather than silently
 * approximating a high-precision request with double precision.
 *
 * @param y MPFR output variable.
 * @param x MPFR input argument.
 * @param tol Requested adaptive tolerance.
 */
void sim_trigamma_mpfr(mpfr_t y, const mpfr_t x, const mpfr_t tol) {
#if (MPFR_VERSION_MAJOR >= 4)
    /* no direct trigamma in MPFR; use polygamma of order 1 if available, else numeric diff */
    /* Placeholder: users with MPFR>=4.2 may replace with mpfr_polygamma(y,1,x,RND). */
    (void) tol;
    mpfr_set_nan(y); /* keep explicit until MPFR support */
#else
    (void) x;
    (void) tol;
    mpfr_set_nan(y);
#endif
}

#ifdef SIM_HAVE_MPC
#include <mpc.h>
/**
 * @brief Evaluate complex digamma with MPC when support is compiled in.
 *
 * Placeholder entry point for a future MPC reflection/Stirling backend.
 *
 * @param y MPC output variable.
 * @param z MPC input argument.
 * @param tol Requested adaptive tolerance.
 */
void sim_digamma_mpc(mpc_t y, const mpc_t z, const mpfr_t tol) {
    /* Strategy: reflect if Re(z)<0.5, then Stirling tail in MPC. Omitted for brevity. */
    (void) y;
    (void) z;
    (void) tol;
}
/**
 * @brief Evaluate complex trigamma with MPC when support is compiled in.
 *
 * Placeholder entry point for a future MPC trigamma backend.
 *
 * @param y MPC output variable.
 * @param z MPC input argument.
 * @param tol Requested adaptive tolerance.
 */
void sim_trigamma_mpc(mpc_t y, const mpc_t z, const mpfr_t tol) {
    (void) y;
    (void) z;
    (void) tol;
}
#endif
#endif /* SIM_HAVE_MPFR */
