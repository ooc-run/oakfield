/**
 * @file qnumber.c
 * @brief Jackson q-number evaluators with optional safe fallback hooks.
 *
 * A Jackson q-number is `[z]_q = (1 - q^z) / (1 - q)`. This module handles
 * the stable `q -> 1` limit, reports domain/numeric faults through
 * `SimSpecialEvalReport`, and lets safe callers provide a fallback value.
 *
 * @warning Experimental implementation. q-number domains, fallback behavior,
 * and precision expectations may change before release.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Core complex Jackson q-number evaluator that populates a diagnostic report.
 *
 * The evaluator accepts positive real `q`, handles `q == 1` by returning `z`,
 * and treats `q == 0` as a real-domain special case. Complex powers are formed
 * as `exp(log(q) * z)`.
 *
 * @param z Complex input argument.
 * @param q Real deformation parameter.
 * @param report Optional diagnostic report updated on success or failure.
 * @param out_value Receives the evaluated q-number on success.
 * @return `SIM_RESULT_OK` on success, otherwise an error status.
 */
static SimResult sim_q_number_complex_eval(SimComplexDouble      z,
                                           double                q,
                                           SimSpecialEvalReport* report,
                                           SimComplexDouble*     out_value) {
    SimResult    status   = SIM_RESULT_OK;
    const double tol      = SIM_Q_ANALOG_DEFAULT_TOL;
    const double tol_exit = tol * 0.5;
    (void) tol_exit;
    sim_special_report_seed(report, "sim_q_number", z, q, NAN, NAN, tol);

    if (!sim_complex_isfinite(z) || !isfinite(q)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    if (fabs(q - 1.0) <= 1e-12) {
        if (out_value)
            *out_value = z;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, 0.0);
        return SIM_RESULT_OK;
    }

    if (q == 0.0) {
        if (fabs(z.im) > 0.0) {
            sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
            status = SIM_RESULT_INVALID_ARGUMENT;
            goto fail;
        }
        double result = 0.0;
        if (z.re == 0.0) {
            result = 0.0;
        } else if (z.re > 0.0) {
            result = 1.0;
        } else {
            sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
            status = SIM_RESULT_INVALID_ARGUMENT;
            goto fail;
        }
        if (out_value)
            *out_value = (SimComplexDouble) { result, 0.0 };
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, 0.0);
        return SIM_RESULT_OK;
    }

    if (q <= 0.0) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    double lnq = log(q);
    if (!isfinite(lnq)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    double denom = -expm1(lnq);
    if (denom == 0.0) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, 0.0);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    double complex cz    = z.re + I * z.im;
    double complex qpow  = cexp(lnq * cz);
    double complex numer = 1.0 - qpow;
    double complex out   = numer / denom;

    if (!isfinite(creal(out)) || !isfinite(cimag(out))) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        status = SIM_RESULT_INVALID_ARGUMENT;
        goto fail;
    }

    if (out_value)
        *out_value = (SimComplexDouble) { creal(out), cimag(out) };
    sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 1, cabs(out));
    return SIM_RESULT_OK;

fail:
    if (out_value)
        *out_value = (SimComplexDouble) { NAN, NAN };
    return status;
}

/**
 * @brief Safely evaluate a complex Jackson q-number and apply fallback on failure.
 *
 * @param z Complex input argument.
 * @param q Real deformation parameter.
 * @param fallback Optional callback invoked when the primary evaluator fails.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives the result or NaNs on failure without fallback.
 * @return `SIM_RESULT_OK` if the primary or fallback path produced a value.
 */
SimResult sim_q_number_complex_safe(SimComplexDouble      z,
                                    double                q,
                                    SimSpecialFallbackFn  fallback,
                                    void*                 userdata,
                                    SimSpecialEvalReport* report,
                                    SimComplexDouble*     out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    SimComplexDouble value;
    SimResult        status = sim_q_number_complex_eval(z, q, report, &value);
    if (status == SIM_RESULT_OK) {
        if (out_value)
            *out_value = value;
        return SIM_RESULT_OK;
    }

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

/**
 * @brief Safely evaluate a real Jackson q-number and apply fallback on failure.
 *
 * @param x Real input argument.
 * @param q Real deformation parameter.
 * @param fallback Optional callback invoked on primary failure.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives the real result or NaN on failure.
 * @return `SIM_RESULT_OK` if a value was produced.
 */
SimResult sim_q_number_safe(double                x,
                            double                q,
                            SimSpecialFallbackFn  fallback,
                            void*                 userdata,
                            SimSpecialEvalReport* report,
                            double*               out_value) {
    SimComplexDouble value;
    SimResult        status = sim_q_number_complex_safe(
        (SimComplexDouble) { x, 0.0 }, q, fallback, userdata, report, &value);
    if (status == SIM_RESULT_OK && out_value != NULL)
        *out_value = value.re;
    else if (status != SIM_RESULT_OK && out_value != NULL)
        *out_value = NAN;
    return status;
}

/**
 * @brief Evaluate a real Jackson q-number, returning NaN on failure.
 *
 * @param x Real input argument.
 * @param q Real deformation parameter.
 * @return `[x]_q`, or NaN when the safe evaluator rejects the inputs.
 */
double sim_q_number(double x, double q) {
    double    out;
    SimResult status = sim_q_number_safe(x, q, NULL, NULL, NULL, &out);
    if (status != SIM_RESULT_OK)
        return NAN;
    return out;
}

/**
 * @brief Evaluate a complex Jackson q-number, returning NaNs on failure.
 *
 * @param z Complex input argument.
 * @param q Real deformation parameter.
 * @return Complex q-number value, or `{NAN, NAN}` on failure.
 */
SimComplexDouble sim_q_number_complex(SimComplexDouble z, double q) {
    SimComplexDouble out;
    SimResult        status = sim_q_number_complex_safe(z, q, NULL, NULL, NULL, &out);
    if (status != SIM_RESULT_OK)
        return (SimComplexDouble) { NAN, NAN };
    return out;
}
