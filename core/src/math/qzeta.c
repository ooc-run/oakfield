/**
 * @file qzeta.c
 * @brief Real q-zeta evaluator with safe fallback support.
 *
 * This module evaluates a Hurwitz-style q-zeta series for real `s`, shift `a`,
 * and deformation `q`. The safe API classifies invalid domains and convergence
 * failure before optionally asking a user fallback for a replacement value.
 *
 * @warning Experimental implementation. q-zeta domains, convergence limits,
 * fallback policy, and precision expectations may change before release.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Safely evaluate a real q-zeta/Hurwitz-style series.
 *
 * For `q` close to one, the routine evaluates the classical Hurwitz-like
 * series `sum (a+n)^(-s)`. For `0 < q < 1`, it uses Jackson q-numbers in the
 * denominator and q-exponential weights. The supported real domain is `a > 0`
 * and `s > 1`.
 *
 * @param s Real exponent; must be greater than 1.
 * @param a Positive Hurwitz shift.
 * @param q Real deformation parameter.
 * @param fallback Optional callback invoked on primary failure.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives the real result or NaN on failure.
 * @return `SIM_RESULT_OK` if the primary or fallback path produced a value.
 */
SimResult sim_q_zeta_safe(double                s,
                          double                a,
                          double                q,
                          SimSpecialFallbackFn  fallback,
                          void*                 userdata,
                          SimSpecialEvalReport* report,
                          double*               out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    const double     tol      = SIM_Q_ANALOG_DEFAULT_TOL;
    const double     tol_exit = tol * 0.5;
    SimComplexDouble primary  = { a, 0.0 };
    sim_special_report_seed(report, "sim_q_zeta", primary, q, a, s, tol);

    SimResult status        = SIM_RESULT_OK;
    double    result        = NAN;
    double    last_term_abs = 0.0;
    int       iter          = 0;

    if (!isfinite(s) || !isfinite(a) || !isfinite(q)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }
    if (a <= 0.0 || s <= 1.0) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    if (fabs(q - 1.0) <= 1e-12) {
        double sum = 0.0;
        for (iter = 0; iter < SIM_Q_ANALOG_MAX_TERMS; ++iter) {
            double term = pow(a + (double) iter, -s);
            sum += term;
            last_term_abs = fabs(term);
            if (last_term_abs < tol_exit)
                break;
        }
        if (iter == SIM_Q_ANALOG_MAX_TERMS) {
            sim_special_report_update(
                report, SIM_SPECIAL_FAULT_ITERATION_LIMIT, iter, last_term_abs);
            status = SIM_RESULT_INVALID_ARGUMENT;
            goto fail;
        }
        result = sum;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, iter + 1, last_term_abs);
        if (out_value)
            *out_value = result;
        return SIM_RESULT_OK;
    }

    if (!(q > 0.0 && q < 1.0)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    double lnq   = log(q);
    double denom = -expm1(lnq);
    if (denom == 0.0) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, 0.0);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    double sum = 0.0;
    for (iter = 0; iter < SIM_Q_ANALOG_MAX_TERMS; ++iter) {
        double xk    = a + (double) iter;
        double numer = -expm1(lnq * xk);
        double qnum  = numer / denom;
        if (!isfinite(qnum) || qnum <= 0.0) {
            sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, iter, qnum);
            status = SIM_RESULT_INVALID_ARGUMENT;
            goto fail;
        }

        double weight = exp((s - 1.0) * lnq * xk);
        double term   = weight / pow(qnum, s);
        if (!isfinite(term)) {
            sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, iter, term);
            status = SIM_RESULT_INVALID_ARGUMENT;
            goto fail;
        }

        sum += term;
        last_term_abs = fabs(term);
        if (last_term_abs < tol_exit)
            break;
    }

    if (iter == SIM_Q_ANALOG_MAX_TERMS) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_ITERATION_LIMIT, iter, last_term_abs);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    result = sum;
    sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, iter + 1, last_term_abs);
    if (out_value)
        *out_value = result;
    return SIM_RESULT_OK;

fail: {
    SimComplexDouble fallback_value = { NAN, NAN };
    SimResult        fb = sim_special_apply_fallback(fallback, userdata, report, &fallback_value);
    if (fb == SIM_RESULT_OK) {
        if (out_value)
            *out_value = fallback_value.re;
        return SIM_RESULT_OK;
    }
    if (out_value)
        *out_value = NAN;
    return (fb == SIM_RESULT_INVALID_ARGUMENT) ? status : fb;
}
}

/**
 * @brief Evaluate real q-zeta, returning NaN on failure.
 *
 * @param s Real exponent.
 * @param a Positive Hurwitz shift.
 * @param q Real deformation parameter.
 * @return q-zeta value, or NaN on invalid input/non-convergence.
 */
double sim_q_zeta(double s, double a, double q) {
    double    out;
    SimResult status = sim_q_zeta_safe(s, a, q, NULL, NULL, NULL, &out);
    if (status != SIM_RESULT_OK)
        return NAN;
    return out;
}
