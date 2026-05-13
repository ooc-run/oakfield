/**
 * @file qdigamma.c
 * @brief Jackson q-digamma evaluators with safe fallback support.
 *
 * The q-digamma implementation uses the Jackson series for `0 < q < 1` and
 * falls back to the classical digamma when `q` is sufficiently close to one.
 * Safe APIs report convergence, singularity, and domain faults explicitly.
 *
 * @warning Experimental implementation. q-digamma domains, convergence limits,
 * fallback policy, and precision expectations may change before release.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Sum the analytic q-digamma series for complex z and 0 < q < 1.
 *
 * Iterates terms `q^{z+n} / (1 - q^{z+n})` until the term magnitude is below
 * the configured tolerance or `SIM_Q_ANALOG_MAX_TERMS` is reached. A small
 * denominator is treated as a pole/singularity.
 *
 * @param z Complex input argument.
 * @param q Real deformation parameter in (0, 1).
 * @param tol Absolute stopping tolerance for the series term.
 * @param report Diagnostic report to update.
 * @param out_value Receives the q-digamma value on convergence.
 * @return `SIM_RESULT_OK` on convergence, otherwise an error status.
 */
static SimResult sim_q_digamma_series_eval(SimComplexDouble      z,
                                           double                q,
                                           double                tol,
                                           SimSpecialEvalReport* report,
                                           SimComplexDouble*     out_value) {
    SimResult      status = SIM_RESULT_OK;
    double complex cz     = z.re + I * z.im;
    double         lnq    = log(q);
    if (!isfinite(lnq)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double complex qpow          = cexp(lnq * cz);
    double complex sum           = 0.0 + 0.0 * I;
    double         last_term_abs = 0.0;
    const double   tol_exit      = tol * 0.5;

    for (int n = 0; n < SIM_Q_ANALOG_MAX_TERMS; ++n) {
        double complex denom     = 1.0 - qpow;
        double         denom_abs = cabs(denom);
        if (denom_abs < tol) {
            sim_special_report_update(report, SIM_SPECIAL_FAULT_SINGULARITY, n, denom_abs);
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        double complex term = qpow / denom;
        sum += term;
        last_term_abs = cabs(term);
        if (last_term_abs < tol_exit) {
            sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, n + 1, last_term_abs);
            double complex out = -log1p(-q) + lnq * sum;
            if (out_value)
                *out_value = (SimComplexDouble) { creal(out), cimag(out) };
            return SIM_RESULT_OK;
        }
        qpow *= q;
    }

    sim_special_report_update(
        report, SIM_SPECIAL_FAULT_ITERATION_LIMIT, SIM_Q_ANALOG_MAX_TERMS, last_term_abs);
    status = SIM_RESULT_INVALID_ARGUMENT;
    if (out_value)
        *out_value = (SimComplexDouble) { NAN, NAN };
    return status;
}

/**
 * @brief Safely evaluate complex q-digamma and apply fallback on failure.
 *
 * @param z Complex input argument.
 * @param q Real deformation parameter. Values near 1 use classical digamma.
 * @param fallback Optional callback invoked on primary failure.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives the complex result or NaNs on failure.
 * @return `SIM_RESULT_OK` if the primary or fallback path produced a value.
 */
SimResult sim_q_digamma_complex_safe(SimComplexDouble      z,
                                     double                q,
                                     SimSpecialFallbackFn  fallback,
                                     void*                 userdata,
                                     SimSpecialEvalReport* report,
                                     SimComplexDouble*     out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    const double tol = SIM_Q_ANALOG_DEFAULT_TOL;
    sim_special_report_seed(report, "sim_q_digamma", z, q, NAN, NAN, tol);

    SimResult status = SIM_RESULT_OK;

    if (!sim_complex_isfinite(z) || !isfinite(q)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    if (fabs(q - 1.0) <= 1e-12) {
        SimComplexDouble classic = sim_digamma_c64_12(z);
        if (out_value)
            *out_value = classic;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, 0.0);
        return SIM_RESULT_OK;
    }

    if (!(q > 0.0 && q < 1.0)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    status = sim_q_digamma_series_eval(z, q, tol, report, out_value);
    if (status == SIM_RESULT_OK)
        return SIM_RESULT_OK;

fail: {
    SimComplexDouble fallback_value = { NAN, NAN };
    SimResult        fb = sim_special_apply_fallback(fallback, userdata, report, &fallback_value);
    if (fb == SIM_RESULT_OK) {
        if (out_value)
            *out_value = fallback_value;
        return SIM_RESULT_OK;
    }
    if (out_value)
        *out_value = (SimComplexDouble) { NAN, NAN };
    return (fb == SIM_RESULT_INVALID_ARGUMENT) ? status : fb;
}
}

/**
 * @brief Safely evaluate real q-digamma and apply fallback on failure.
 *
 * @param x Real input argument.
 * @param q Real deformation parameter.
 * @param fallback Optional callback invoked on primary failure.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives the real result or NaN on failure.
 * @return `SIM_RESULT_OK` if a value was produced.
 */
SimResult sim_q_digamma_safe(double                x,
                             double                q,
                             SimSpecialFallbackFn  fallback,
                             void*                 userdata,
                             SimSpecialEvalReport* report,
                             double*               out_value) {
    SimComplexDouble out;
    SimResult        status = sim_q_digamma_complex_safe(
        (SimComplexDouble) { x, 0.0 }, q, fallback, userdata, report, &out);
    if (status == SIM_RESULT_OK && out_value)
        *out_value = out.re;
    else if (status != SIM_RESULT_OK && out_value)
        *out_value = NAN;
    return status;
}

/**
 * @brief Evaluate real q-digamma, returning NaN on failure.
 *
 * @param x Real input argument.
 * @param q Real deformation parameter.
 * @return q-digamma value, or NaN on failure.
 */
double sim_q_digamma(double x, double q) {
    double    out;
    SimResult status = sim_q_digamma_safe(x, q, NULL, NULL, NULL, &out);
    if (status != SIM_RESULT_OK)
        return NAN;
    return out;
}

/**
 * @brief Evaluate complex q-digamma, returning NaNs on failure.
 *
 * @param z Complex input argument.
 * @param q Real deformation parameter.
 * @return Complex q-digamma value, or `{NAN, NAN}` on failure.
 */
SimComplexDouble sim_q_digamma_complex(SimComplexDouble z, double q) {
    SimComplexDouble out;
    SimResult        status = sim_q_digamma_complex_safe(z, q, NULL, NULL, NULL, &out);
    if (status != SIM_RESULT_OK)
        return (SimComplexDouble) { NAN, NAN };
    return out;
}
