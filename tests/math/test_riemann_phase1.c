/*
 * Broad regression coverage for the Zeta/Xi implementation: branch selection,
 * formal balls, reflection, and zero search.
 */
#include <oakfield/math/loggamma.h>
#include <oakfield/math/xi.h>
#include <oakfield/math/zeta.h>

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define TEST_EULER_MASCHERONI 0.57721566490153286060651209008240243
#define TEST_STIELTJES_GAMMA1 -0.07281584548367672486058637587490132
#define TEST_ZETA_HALF -1.46035450880958681288949915251529801
#define TEST_APERY 1.20205690315959428540
#define TEST_FIRST_RIEMANN_ZERO 14.134725141734693790
#define TEST_THIRD_RIEMANN_ZERO 25.010857580145688763
#define TEST_ZERO_SIGN_SAFETY 8.0

static double complex to_c64(SimComplexDouble z) { return z.re + z.im * I; }

static bool expect_close_real(const char *label, double value, double expected, double tol) {
    double err = fabs(value - expected);
    if (err > tol) {
        fprintf(stderr, "[FAIL] %s: value=%.17g expected=%.17g abs_err=%.3e tol=%.3e\n", label,
                value, expected, err, tol);
        return false;
    }
    return true;
}

static bool expect_close_complex(const char *label, double complex value, double complex expected,
                                 double tol) {
    double err = cabs(value - expected);
    if (err > tol) {
        fprintf(stderr,
                "[FAIL] %s: value=(%.17g, %.17g) expected=(%.17g, %.17g) abs_err=%.3e tol=%.3e\n",
                label, creal(value), cimag(value), creal(expected), cimag(expected), err, tol);
        return false;
    }
    return true;
}

int main(void) {
    bool ok = true;

    SimLogGammaResult lg_z = sim_log_gamma_eval((SimComplexDouble){2.5, 1.25});
    SimLogGammaResult lg_z1 = sim_log_gamma_eval((SimComplexDouble){3.5, 1.25});
    if (lg_z.status != SIM_LOG_GAMMA_STATUS_OK || lg_z1.status != SIM_LOG_GAMMA_STATUS_OK) {
        fprintf(stderr, "[FAIL] complex log-gamma returned status (%d, %d)\n", lg_z.status,
                lg_z1.status);
        ok = false;
    } else {
        double complex z = 2.5 + 1.25 * I;
        double complex test = cexp(to_c64(lg_z1.value) - to_c64(lg_z.value) - clog(z));
        ok = ok && expect_close_complex("loggamma recurrence", test, 1.0 + 0.0 * I, 2.0e-12);
    }

    SimZetaResult zeta_2 = sim_zeta_eval((SimComplexDouble){2.0, 0.0}, NULL);
    if (zeta_2.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] zeta(2) status=%d\n", zeta_2.status);
        ok = false;
    } else {
        ok = ok && expect_close_real("zeta(2)", zeta_2.value.re, (M_PI * M_PI) / 6.0, 2.0e-12);
        ok = ok && expect_close_real("zeta(2) imag", zeta_2.value.im, 0.0, 1.0e-13);
        ok = ok && (zeta_2.branch == SIM_ZETA_BRANCH_DIRECT_EULER_MACLAURIN);
    }

    {
        SimComplexBall zeta_2_ball = sim_zeta_eval_ball((SimComplexDouble){2.0, 0.0}, NULL);
        if (zeta_2_ball.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] zeta ball(2) status=%d\n", zeta_2_ball.status);
            ok = false;
        } else {
            ok = ok && (zeta_2_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && expect_close_real("zeta ball(2)", zeta_2_ball.center.re, (M_PI * M_PI) / 6.0,
                                         2.0e-12);
            ok = ok && expect_close_real("zeta ball(2) imag", zeta_2_ball.center.im, 0.0, 1.0e-15);
        }
    }

    {
        SimComplexBall zeta_complex_ball = sim_zeta_eval_ball((SimComplexDouble){2.0, 14.0}, NULL);
        SimComplexBall zeta_complex_conj_ball =
            sim_zeta_eval_ball((SimComplexDouble){2.0, -14.0}, NULL);
        SimZetaResult zeta_complex_em =
            sim_zeta_eval_direct_euler_maclaurin((SimComplexDouble){2.0, 14.0}, NULL);
        if (zeta_complex_ball.status != SIM_ZETA_STATUS_OK ||
            zeta_complex_conj_ball.status != SIM_ZETA_STATUS_OK ||
            zeta_complex_em.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] complex zeta ball statuses (%d, %d, %d)\n",
                    zeta_complex_ball.status, zeta_complex_conj_ball.status,
                    zeta_complex_em.status);
            ok = false;
        } else {
            ok = ok && (zeta_complex_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (zeta_complex_conj_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok =
                ok && expect_close_complex("zeta ball/em overlap", to_c64(zeta_complex_ball.center),
                                           to_c64(zeta_complex_em.value), 5.0e-12);
            ok = ok &&
                 expect_close_complex("zeta ball conjugation", to_c64(zeta_complex_ball.center),
                                      conj(to_c64(zeta_complex_conj_ball.center)), 5.0e-12);
        }
    }

    SimZetaResult zeta_4 = sim_zeta_eval((SimComplexDouble){4.0, 0.0}, NULL);
    if (zeta_4.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] zeta(4) status=%d\n", zeta_4.status);
        ok = false;
    } else {
        ok = ok && expect_close_real("zeta(4)", zeta_4.value.re, pow(M_PI, 4.0) / 90.0, 2.0e-12);
        ok = ok && expect_close_real("zeta(4) imag", zeta_4.value.im, 0.0, 1.0e-13);
    }

    SimZetaResult zeta_minus_1 = sim_zeta_eval((SimComplexDouble){-1.0, 0.0}, NULL);
    if (zeta_minus_1.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] zeta(-1) status=%d\n", zeta_minus_1.status);
        ok = false;
    } else {
        ok = ok && expect_close_real("zeta(-1)", zeta_minus_1.value.re, -1.0 / 12.0, 1.0e-11);
        ok = ok && ((zeta_minus_1.flags & SIM_ZETA_FLAG_USED_REFLECTION) != 0U);
        ok = ok && (zeta_minus_1.branch == SIM_ZETA_BRANCH_REFLECTION);
    }

    {
        SimComplexBall zeta_zero_ball = sim_zeta_eval_ball((SimComplexDouble){0.0, 0.0}, NULL);
        SimComplexBall zeta_trivial_ball = sim_zeta_eval_ball((SimComplexDouble){-2.0, 0.0}, NULL);
        if (zeta_zero_ball.status != SIM_ZETA_STATUS_OK ||
            zeta_trivial_ball.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] exact zeta balls statuses (%d, %d)\n", zeta_zero_ball.status,
                    zeta_trivial_ball.status);
            ok = false;
        } else {
            ok = ok && (zeta_zero_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (zeta_zero_ball.radius == 0.0);
            ok = ok && expect_close_real("zeta ball(0)", zeta_zero_ball.center.re, -0.5, 0.0);
            ok = ok && (zeta_trivial_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (zeta_trivial_ball.radius == 0.0);
            ok = ok && expect_close_real("zeta ball(-2)", zeta_trivial_ball.center.re, 0.0, 0.0);
        }
    }

    SimZetaResult zeta_half = sim_zeta_eval((SimComplexDouble){0.5, 0.0}, NULL);
    if (zeta_half.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] zeta(1/2) status=%d\n", zeta_half.status);
        ok = false;
    } else {
        ok = ok && expect_close_real("zeta(1/2)", zeta_half.value.re, TEST_ZETA_HALF, 5.0e-12);
        ok = ok && expect_close_real("zeta(1/2) imag", zeta_half.value.im, 0.0, 1.0e-13);
        ok = ok && (zeta_half.branch == SIM_ZETA_BRANCH_ETA_ACCELERATED);
        ok = ok && ((zeta_half.flags & SIM_ZETA_FLAG_USED_ETA) != 0U);
    }

    {
        SimComplexDouble overlap_arg = {1.25, 0.4};
        SimZetaResult eta_overlap = sim_zeta_eval_eta_accelerated(overlap_arg, NULL);
        SimZetaResult em_overlap = sim_zeta_eval_direct_euler_maclaurin(overlap_arg, NULL);
        if (eta_overlap.status != SIM_ZETA_STATUS_OK || em_overlap.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] eta/em overlap statuses (%d, %d)\n", eta_overlap.status,
                    em_overlap.status);
            ok = false;
        } else {
            ok = ok && expect_close_complex("eta/em overlap", to_c64(eta_overlap.value),
                                            to_c64(em_overlap.value), 5.0e-11);
        }
    }

    {
        SimComplexDouble afe_arg = {0.5, 14.0};
        SimZetaContext afe_ctx = sim_zeta_context_default();
        afe_ctx.afe_max_cutoff = 4096U;
        afe_ctx.adaptive_max_rounds = 0U;
        SimZetaResult afe = sim_zeta_eval_approximate_fe(afe_arg, &afe_ctx);
        SimZetaResult em = sim_zeta_eval_direct_euler_maclaurin(afe_arg, NULL);
        if (afe.status != SIM_ZETA_STATUS_OK || em.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] afe/em statuses (%d, %d)\n", afe.status, em.status);
            ok = false;
        } else {
            ok = ok && expect_close_complex("afe/em overlap", to_c64(afe.value), to_c64(em.value),
                                            5.0e-8);
            ok = ok && (afe.branch == SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION);
            ok = ok && ((afe.flags & SIM_ZETA_FLAG_USED_AFE) != 0U);
        }
    }

    {
        SimZetaResult large_t = sim_zeta_eval((SimComplexDouble){0.5, 14.0}, NULL);
        if (large_t.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] zeta(1/2+14i) status=%d\n", large_t.status);
            ok = false;
        } else {
            ok = ok && (large_t.branch == SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION);
            ok = ok && ((large_t.flags & SIM_ZETA_FLAG_USED_AFE) != 0U);
        }
    }

    {
        SimZetaResult rs = sim_zeta_eval_riemann_siegel(50.0, NULL);
        SimZetaResult afe = sim_zeta_eval_approximate_fe((SimComplexDouble){0.5, 50.0}, NULL);
        SimZetaResult strict_dispatch = sim_zeta_eval((SimComplexDouble){0.5, 50.0}, NULL);
        SimComplexBall strict_ball = sim_zeta_eval_ball((SimComplexDouble){0.5, 50.0}, NULL);
        SimZetaContext relaxed_ctx = sim_zeta_context_default();
        relaxed_ctx.abs_tol = 2.0e-1;
        relaxed_ctx.rel_tol = 2.0e-1;
        SimZetaResult relaxed_dispatch = sim_zeta_eval((SimComplexDouble){0.5, 50.0}, &relaxed_ctx);
        if (rs.status != SIM_ZETA_STATUS_OK || afe.status != SIM_ZETA_STATUS_OK ||
            strict_dispatch.status != SIM_ZETA_STATUS_NO_CONVERGENCE ||
            relaxed_dispatch.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] riemann-siegel statuses (%d, %d, %d, %d)\n", rs.status,
                    afe.status, strict_dispatch.status, relaxed_dispatch.status);
            ok = false;
        } else {
            ok = ok && expect_close_complex("riemann-siegel overlap", to_c64(rs.value),
                                            to_c64(afe.value), 1.5e-1);
            ok = ok && (strict_dispatch.branch == SIM_ZETA_BRANCH_RIEMANN_SIEGEL);
            ok = ok && ((strict_dispatch.flags & SIM_ZETA_FLAG_USED_RIEMANN_SIEGEL) != 0U);
            ok = ok && (strict_ball.status == SIM_ZETA_STATUS_NO_CONVERGENCE);
            ok = ok && (strict_ball.rigor == SIM_BALL_RIGOR_HEURISTIC);
            ok = ok && (relaxed_dispatch.branch == SIM_ZETA_BRANCH_RIEMANN_SIEGEL);
            ok = ok && ((relaxed_dispatch.flags & SIM_ZETA_FLAG_USED_RIEMANN_SIEGEL) != 0U);
        }
    }

    {
        const double delta = 1.0e-6;
        double s_real = 1.0 + delta;
        double actual_delta = s_real - 1.0;
        SimZetaResult near_one = sim_zeta_eval((SimComplexDouble){s_real, 0.0}, NULL);
        if (near_one.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] zeta(1+delta) status=%d\n", near_one.status);
            ok = false;
        } else {
            double expected =
                (1.0 / actual_delta) + TEST_EULER_MASCHERONI - TEST_STIELTJES_GAMMA1 * actual_delta;
            ok = ok && expect_close_real("zeta near one", near_one.value.re, expected, 1.0e-9);
            ok = ok && (near_one.branch == SIM_ZETA_BRANCH_NEAR_ONE_LAURENT);
            ok = ok && ((near_one.flags & SIM_ZETA_FLAG_USED_NEAR_ONE_LAURENT) != 0U);
        }
    }

    {
        const double delta = 1.0e-8;
        SimZetaResult near_trivial = sim_zeta_eval((SimComplexDouble){-2.0 + delta, 0.0}, NULL);
        double expected_derivative = -TEST_APERY / (4.0 * M_PI * M_PI);
        if (near_trivial.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] zeta(-2+delta) status=%d\n", near_trivial.status);
            ok = false;
        } else {
            ok = ok && expect_close_real("zeta near trivial zero", near_trivial.value.re,
                                         expected_derivative * delta, 1.0e-10);
            ok = ok && (near_trivial.branch == SIM_ZETA_BRANCH_LOCAL_EXPANSION);
            ok = ok && ((near_trivial.flags & SIM_ZETA_FLAG_USED_LOCAL_EXPANSION) != 0U);
        }
    }

    {
        SimZetaDerivativeResult derivative =
            sim_zeta_eval_with_derivative((SimComplexDouble){-2.0, 0.0}, NULL);
        double expected = -TEST_APERY / (4.0 * M_PI * M_PI);
        if (derivative.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] zeta'(-2) status=%d\n", derivative.status);
            ok = false;
        } else {
            ok = ok && expect_close_real("zeta'(-2)", derivative.derivative.re, expected, 5.0e-8);
            ok = ok && expect_close_real("zeta'(-2) imag", derivative.derivative.im, 0.0, 5.0e-8);
        }
    }

    {
        SimZetaDerivativeResult derivative =
            sim_zeta_eval_with_derivative((SimComplexDouble){0.5, 32.0}, NULL);
        ok = ok && (derivative.status == SIM_ZETA_STATUS_NO_CONVERGENCE);
    }

    {
        SimZetaContext adaptive_ctx = sim_zeta_context_default();
        adaptive_ctx.initial_terms = 4U;
        adaptive_ctx.max_terms = 4U;
        adaptive_ctx.adaptive_max_rounds = 3U;
        adaptive_ctx.adaptive_tightening_factor = 4.0;
        SimZetaResult adaptive = sim_zeta_eval((SimComplexDouble){2.0, 0.0}, &adaptive_ctx);
        if (adaptive.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] adaptive zeta(2) status=%d\n", adaptive.status);
            ok = false;
        } else {
            ok = ok && expect_close_real("adaptive zeta(2)", adaptive.value.re, (M_PI * M_PI) / 6.0,
                                         2.0e-12);
            ok = ok && (adaptive.refinement_rounds > 0U);
            ok = ok && ((adaptive.flags & SIM_ZETA_FLAG_USED_ADAPTIVE_REFINEMENT) != 0U);
        }
    }

    SimXiResult xi_0 = sim_xi_eval((SimComplexDouble){0.0, 0.0}, NULL);
    SimXiResult xi_1 = sim_xi_eval((SimComplexDouble){1.0, 0.0}, NULL);
    if (xi_0.status != SIM_ZETA_STATUS_OK || xi_1.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] xi exact limits returned status (%d, %d)\n", xi_0.status,
                xi_1.status);
        ok = false;
    } else {
        ok = ok && expect_close_real("xi(0)", xi_0.value.re, 0.5, 1.0e-15);
        ok = ok && expect_close_real("xi(1)", xi_1.value.re, 0.5, 1.0e-15);
        ok = ok && ((xi_0.flags & SIM_XI_FLAG_EXACT_LIMIT) != 0U);
        ok = ok && ((xi_1.flags & SIM_XI_FLAG_EXACT_LIMIT) != 0U);
    }

    {
        SimComplexBall xi_0_ball = sim_xi_eval_ball((SimComplexDouble){0.0, 0.0}, NULL);
        SimComplexBall xi_1_ball = sim_xi_eval_ball((SimComplexDouble){1.0, 0.0}, NULL);
        if (xi_0_ball.status != SIM_ZETA_STATUS_OK || xi_1_ball.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] xi exact balls statuses (%d, %d)\n", xi_0_ball.status,
                    xi_1_ball.status);
            ok = false;
        } else {
            ok = ok && (xi_0_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (xi_1_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (xi_0_ball.radius == 0.0);
            ok = ok && (xi_1_ball.radius == 0.0);
            ok = ok && expect_close_real("xi ball(0)", xi_0_ball.center.re, 0.5, 0.0);
            ok = ok && expect_close_real("xi ball(1)", xi_1_ball.center.re, 0.5, 0.0);
        }
    }

    {
        SimComplexBall xi_2_ball = sim_xi_eval_ball((SimComplexDouble){2.0, 0.0}, NULL);
        SimComplexBall xi_3_ball = sim_xi_eval_ball((SimComplexDouble){3.0, 0.0}, NULL);
        SimComplexBall xi_neg2_ball = sim_xi_eval_ball((SimComplexDouble){-2.0, 0.0}, NULL);
        double xi_2_expected = M_PI / 6.0;
        double xi_3_expected = (3.0 * TEST_APERY) / (2.0 * M_PI);
        if (xi_2_ball.status != SIM_ZETA_STATUS_OK || xi_3_ball.status != SIM_ZETA_STATUS_OK ||
            xi_neg2_ball.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] formal xi integer balls statuses (%d, %d, %d)\n",
                    xi_2_ball.status, xi_3_ball.status, xi_neg2_ball.status);
            ok = false;
        } else {
            ok = ok && (xi_2_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (xi_3_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && (xi_neg2_ball.rigor == SIM_BALL_RIGOR_FORMAL);
            ok = ok && expect_close_real("xi ball(2)", xi_2_ball.center.re, xi_2_expected, 5.0e-13);
            ok = ok && expect_close_real("xi ball(3)", xi_3_ball.center.re, xi_3_expected, 5.0e-13);
            ok = ok && expect_close_real("xi ball(-2)=xi(3)", xi_neg2_ball.center.re, xi_3_expected,
                                         5.0e-13);
            ok = ok && (xi_neg2_ball.flags & SIM_XI_FLAG_USED_REFLECTION);
        }
    }

    {
        SimXiResult near_limit = sim_xi_eval((SimComplexDouble){1.0e-10, 1.0e-10}, NULL);
        if (near_limit.status == SIM_ZETA_STATUS_INVALID_ARGUMENT ||
            near_limit.status == SIM_ZETA_STATUS_NUMERIC_FAILURE ||
            !isfinite(near_limit.value.re) || !isfinite(near_limit.value.im) ||
            !isfinite(near_limit.abs_error)) {
            fprintf(stderr, "[FAIL] xi near exact-limit neighborhood unusable status=%d\n",
                    near_limit.status);
            ok = false;
        } else {
            ok = ok && ((near_limit.flags & SIM_XI_FLAG_EXACT_LIMIT) == 0U);
            ok = ok && (near_limit.abs_error > 0.0);
            ok = ok && (cabs(to_c64(near_limit.value) - (0.5 + 0.0 * I)) > 1.0e-13);
        }
    }

    {
        SimXiResult xi_high_t = sim_xi_eval((SimComplexDouble){0.5, 50.0}, NULL);
        SimComplexBall xi_high_t_ball = sim_xi_eval_ball((SimComplexDouble){0.5, 50.0}, NULL);
        if (xi_high_t.status != SIM_ZETA_STATUS_NO_CONVERGENCE || !isfinite(xi_high_t.value.re) ||
            !isfinite(xi_high_t.value.im) || !isfinite(xi_high_t.abs_error) ||
            xi_high_t.abs_error <= 0.0 || xi_high_t_ball.status != SIM_ZETA_STATUS_NO_CONVERGENCE ||
            !isfinite(xi_high_t_ball.center.re) || !isfinite(xi_high_t_ball.center.im) ||
            !isfinite(xi_high_t_ball.radius) || xi_high_t_ball.radius <= 0.0) {
            fprintf(stderr,
                    "[FAIL] xi(1/2+50i) best-effort propagation failed (%d, %.3e, %d, %.3e)\n",
                    xi_high_t.status, xi_high_t.abs_error, xi_high_t_ball.status,
                    xi_high_t_ball.radius);
            ok = false;
        } else {
            ok = ok && (xi_high_t.zeta_branch == SIM_ZETA_BRANCH_RIEMANN_SIEGEL);
            ok = ok && (xi_high_t_ball.rigor == SIM_BALL_RIGOR_HEURISTIC);
        }
    }

    {
        SimXiResult xi_large_t = sim_xi_eval((SimComplexDouble){0.5, 14.0}, NULL);
        SimXiResult xi_large_t_sym = sim_xi_eval((SimComplexDouble){0.5, -14.0}, NULL);
        if (xi_large_t.status != SIM_ZETA_STATUS_OK ||
            xi_large_t_sym.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] xi large-t statuses (%d, %d)\n", xi_large_t.status,
                    xi_large_t_sym.status);
            ok = false;
        } else {
            ok = ok && (xi_large_t.zeta_branch == SIM_ZETA_BRANCH_APPROXIMATE_FUNCTIONAL_EQUATION);
            ok = ok && expect_close_complex("xi conjugation large-t", to_c64(xi_large_t.value),
                                            conj(to_c64(xi_large_t_sym.value)), 5.0e-8);
        }
    }

    {
        SimComplexBall ball_lo = sim_xi_eval_ball((SimComplexDouble){0.5, 14.0}, NULL);
        SimComplexBall ball_hi = sim_xi_eval_ball((SimComplexDouble){0.5, 14.3}, NULL);
        bool lo_negative =
            ball_lo.center.re < -TEST_ZERO_SIGN_SAFETY * (ball_lo.radius + fabs(ball_lo.center.im));
        bool lo_positive =
            ball_lo.center.re > TEST_ZERO_SIGN_SAFETY * (ball_lo.radius + fabs(ball_lo.center.im));
        bool hi_negative =
            ball_hi.center.re < -TEST_ZERO_SIGN_SAFETY * (ball_hi.radius + fabs(ball_hi.center.im));
        bool hi_positive =
            ball_hi.center.re > TEST_ZERO_SIGN_SAFETY * (ball_hi.radius + fabs(ball_hi.center.im));
        if (ball_lo.status != SIM_ZETA_STATUS_OK || ball_hi.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] xi ball statuses (%d, %d)\n", ball_lo.status, ball_hi.status);
            ok = false;
        } else {
            ok = ok && (ball_lo.rigor == SIM_BALL_RIGOR_VALIDATED);
            ok = ok && (ball_hi.rigor == SIM_BALL_RIGOR_VALIDATED);
            ok = ok && (ball_lo.validation_passes > 0U);
            ok = ok && (ball_hi.validation_passes > 0U);
            ok = ok && ((lo_negative && hi_positive) || (lo_positive && hi_negative));
        }
    }

    {
        SimXiZeroResult zero = sim_xi_find_critical_zero(14.0, 14.3, NULL);
        if (zero.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] critical zero search status=%d\n", zero.status);
            ok = false;
        } else {
            ok = ok &&
                 expect_close_real("first riemann zero", zero.t, TEST_FIRST_RIEMANN_ZERO, 1.0e-6);
            ok = ok && (fabs(zero.derivative) > zero.derivative_abs_error);
            ok = ok && (zero.bracket_rigor >= SIM_BALL_RIGOR_VALIDATED);
            ok = ok && ((zero.flags & SIM_XI_FLAG_USED_ZERO_REFINEMENT) != 0U);
            ok = ok && ((zero.flags & SIM_XI_FLAG_ZERO_BRACKET_VALIDATED) != 0U);
            ok = ok && ((zero.flags & SIM_XI_FLAG_ZERO_BRACKET_FORMAL) == 0U);
            ok = ok && (zero.xi_ball.rigor == SIM_BALL_RIGOR_VALIDATED);
        }
    }

    {
        SimXiZeroResult third_zero = sim_xi_find_critical_zero(25.0, 26.0, NULL);
        if ((third_zero.status != SIM_ZETA_STATUS_OK &&
             third_zero.status != SIM_ZETA_STATUS_NO_CONVERGENCE) ||
            !isfinite(third_zero.t)) {
            fprintf(stderr, "[FAIL] third critical zero search status=%d\n", third_zero.status);
            ok = false;
        } else {
            ok = ok && expect_close_real("third riemann zero", third_zero.t,
                                         TEST_THIRD_RIEMANN_ZERO, 1.0e-5);
            ok = ok && expect_close_real("third zero xi real", third_zero.xi_ball.center.re, 0.0,
                                         fmax(5.0e-8, 8.0 * third_zero.xi_ball.radius));
            ok = ok && (((third_zero.flags & SIM_XI_FLAG_ZERO_BRACKET_VALIDATED) != 0U) ==
                        (third_zero.bracket_rigor >= SIM_BALL_RIGOR_VALIDATED));
            ok = ok && (((third_zero.flags & SIM_XI_FLAG_ZERO_BRACKET_FORMAL) != 0U) ==
                        (third_zero.bracket_rigor >= SIM_BALL_RIGOR_FORMAL));
            if (third_zero.bracket_rigor < SIM_BALL_RIGOR_VALIDATED) {
                ok = ok && (third_zero.status == SIM_ZETA_STATUS_NO_CONVERGENCE);
                ok = ok && ((third_zero.flags & SIM_XI_FLAG_ZERO_BRACKET_VALIDATED) == 0U);
            }
            ok = ok && ((third_zero.flags & SIM_XI_FLAG_USED_ZERO_REFINEMENT) != 0U ||
                        third_zero.status == SIM_ZETA_STATUS_NO_CONVERGENCE);
        }
    }

    {
        const double delta = 1.0e-6;
        SimXiResult xi_near_one = sim_xi_eval((SimComplexDouble){1.0 + delta, 0.0}, NULL);
        SimXiResult xi_near_zero = sim_xi_eval((SimComplexDouble){-delta, 0.0}, NULL);
        if (xi_near_one.status != SIM_ZETA_STATUS_OK || xi_near_zero.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] xi near-pole statuses (%d, %d)\n", xi_near_one.status,
                    xi_near_zero.status);
            ok = false;
        } else {
            ok = ok && expect_close_complex("xi near one symmetry", to_c64(xi_near_one.value),
                                            to_c64(xi_near_zero.value), 1.0e-12);
            ok = ok && ((xi_near_one.flags & SIM_XI_FLAG_USED_NEAR_ONE_EXPANSION) != 0U);
            ok = ok && ((xi_near_zero.flags & SIM_XI_FLAG_USED_REFLECTION) != 0U);
        }
    }

    {
        SimXiDerivativeResult derivative =
            sim_xi_eval_with_derivative((SimComplexDouble){0.5, TEST_FIRST_RIEMANN_ZERO}, NULL);
        if (derivative.status != SIM_ZETA_STATUS_OK) {
            fprintf(stderr, "[FAIL] xi' at first zero status=%d\n", derivative.status);
            ok = false;
        } else {
            ok = ok && (cabs(to_c64(derivative.derivative)) > derivative.derivative_abs_error);
        }
    }

    {
        SimXiDerivativeResult derivative =
            sim_xi_eval_with_derivative((SimComplexDouble){0.5, 32.0}, NULL);
        ok = ok && (derivative.status == SIM_ZETA_STATUS_NO_CONVERGENCE);
    }

    SimXiResult xi_s = sim_xi_eval((SimComplexDouble){2.0, 3.0}, NULL);
    SimXiResult xi_sym = sim_xi_eval((SimComplexDouble){-1.0, -3.0}, NULL);
    if (xi_s.status != SIM_ZETA_STATUS_OK || xi_sym.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] xi symmetry statuses (%d, %d)\n", xi_s.status, xi_sym.status);
        ok = false;
    } else {
        ok = ok &&
             expect_close_complex("xi symmetry", to_c64(xi_s.value), to_c64(xi_sym.value), 2.0e-10);
        ok = ok && ((xi_sym.flags & SIM_XI_FLAG_USED_REFLECTION) != 0U);
    }

    SimXiResult xi_neg_even = sim_xi_eval((SimComplexDouble){-2.0, 0.0}, NULL);
    SimXiResult xi_pos_odd = sim_xi_eval((SimComplexDouble){3.0, 0.0}, NULL);
    if (xi_neg_even.status != SIM_ZETA_STATUS_OK || xi_pos_odd.status != SIM_ZETA_STATUS_OK) {
        fprintf(stderr, "[FAIL] xi(-2)/xi(3) statuses (%d, %d)\n", xi_neg_even.status,
                xi_pos_odd.status);
        ok = false;
    } else {
        ok = ok && expect_close_complex("xi(-2)=xi(3)", to_c64(xi_neg_even.value),
                                        to_c64(xi_pos_odd.value), 1.0e-12);
        ok = ok && ((xi_neg_even.flags & SIM_XI_FLAG_USED_REFLECTION) != 0U);
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
