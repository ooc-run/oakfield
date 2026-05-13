/**
 * @file special_eval.c
 * @brief Safe wrappers for classical special functions with fallback hooks.
 *
 * The raw mathematical helpers generally return NaN on domain errors or
 * numerical failure. These safe wrappers classify those failures, populate a
 * `SimSpecialEvalReport`, and optionally let a caller-supplied fallback provide
 * an alternate value.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Classical helpers with fallback support.
 *
 * Classical helpers (digamma / trigamma / finite ladder) with fallback support
 *
 * Each wrapper follows the same shape: seed the report, validate inputs, run the
 * primary evaluator, classify failure, then call `sim_special_apply_fallback()`
 * if necessary.
 */

/**
 * @brief Safely evaluate real digamma and apply fallback on failure.
 *
 * @param x Real argument.
 * @param fallback Optional replacement-value callback.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report; a local temporary is used when NULL.
 * @param out_value Receives digamma value or NaN.
 * @return `SIM_RESULT_OK` when the primary or fallback path produced a value.
 */
SimResult sim_digamma_safe(double                x,
                           SimSpecialFallbackFn  fallback,
                           void*                 userdata,
                           SimSpecialEvalReport* report,
                           double*               out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    sim_special_report_seed(
        report, "sim_digamma", (SimComplexDouble) { x, 0.0 }, NAN, NAN, NAN, 0.0);

    SimResult status = SIM_RESULT_INVALID_ARGUMENT;

    if (!isfinite(x)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        goto fail;
    }

    double value = sim_digamma_f64_12(x);
    if (isfinite(value)) {
        if (out_value)
            *out_value = value;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, fabs(value));
        return SIM_RESULT_OK;
    }

    if (x <= 0.0 && sim_is_int(x))
        sim_special_report_update(report, SIM_SPECIAL_FAULT_SINGULARITY, 0, NAN);
    else
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, value);

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
 * @brief Safely evaluate real trigamma and apply fallback on failure.
 *
 * @param x Real argument.
 * @param fallback Optional replacement-value callback.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives trigamma value or NaN.
 * @return `SIM_RESULT_OK` when the primary or fallback path produced a value.
 */
SimResult sim_trigamma_safe(double                x,
                            SimSpecialFallbackFn  fallback,
                            void*                 userdata,
                            SimSpecialEvalReport* report,
                            double*               out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    sim_special_report_seed(
        report, "sim_trigamma", (SimComplexDouble) { x, 0.0 }, NAN, NAN, NAN, 0.0);

    SimResult status = SIM_RESULT_INVALID_ARGUMENT;

    if (!isfinite(x)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        goto fail;
    }

    double value = sim_trigamma_f64_12(x);
    if (isfinite(value)) {
        if (out_value)
            *out_value = value;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, value);
        return SIM_RESULT_OK;
    }

    if (x <= 0.0 && sim_is_int(x))
        sim_special_report_update(report, SIM_SPECIAL_FAULT_SINGULARITY, 0, NAN);
    else
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, value);

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
 * @brief Safely evaluate real tetragamma and apply fallback on failure.
 *
 * @param x Real argument.
 * @param fallback Optional replacement-value callback.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives tetragamma value or NaN.
 * @return `SIM_RESULT_OK` when the primary or fallback path produced a value.
 */
SimResult sim_tetragamma_safe(double                x,
                              SimSpecialFallbackFn  fallback,
                              void*                 userdata,
                              SimSpecialEvalReport* report,
                              double*               out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    sim_special_report_seed(
        report, "sim_tetragamma", (SimComplexDouble) { x, 0.0 }, NAN, NAN, NAN, 0.0);

    SimResult status = SIM_RESULT_INVALID_ARGUMENT;

    if (!isfinite(x)) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, NAN);
        goto fail;
    }

    double value = sim_tetragamma_f64_12(x);
    if (isfinite(value)) {
        if (out_value)
            *out_value = value;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, fabs(value));
        return SIM_RESULT_OK;
    }

    if (x <= 0.0 && sim_is_int(x))
        sim_special_report_update(report, SIM_SPECIAL_FAULT_SINGULARITY, 0, NAN);
    else
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, value);

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
 * @brief Safely evaluate the real finite ladder phi helper.
 *
 * @param lambda Numerator parameter in `lambda / (lambda + epsilon + k)`.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms; must be positive.
 * @param fallback Optional replacement-value callback.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives phi value or NaN.
 * @return `SIM_RESULT_OK` when the primary or fallback path produced a value.
 */
SimResult sim_hyperexp_phi_safe(double                lambda,
                                double                epsilon,
                                int                   K,
                                SimSpecialFallbackFn  fallback,
                                void*                 userdata,
                                SimSpecialEvalReport* report,
                                double*               out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    sim_special_report_seed(report,
                            "sim_hyperexp_phi",
                            (SimComplexDouble) { lambda, 0.0 },
                            NAN,
                            epsilon,
                            (double) K,
                            0.0);

    SimResult status = SIM_RESULT_INVALID_ARGUMENT;

    if (!isfinite(lambda) || !isfinite(epsilon) || K <= 0) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
        goto fail;
    }

    double value = sim_hyperexp_phi(lambda, epsilon, K);
    if (isfinite(value)) {
        if (out_value)
            *out_value = value;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, fabs(value));
        return SIM_RESULT_OK;
    }

    double a = lambda + epsilon;
    if ((a <= 0.0 && sim_is_int(a)) || ((a + (double) K) <= 0.0 && sim_is_int(a + (double) K)))
        sim_special_report_update(report, SIM_SPECIAL_FAULT_SINGULARITY, 0, NAN);
    else
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, value);

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
 * @brief Safely evaluate the lambda derivative of the real finite ladder phi helper.
 *
 * @param lambda Numerator parameter in the finite ladder.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms; must be positive.
 * @param fallback Optional replacement-value callback.
 * @param userdata Opaque pointer passed to `fallback`.
 * @param report Optional diagnostic report.
 * @param out_value Receives derivative value or NaN.
 * @return `SIM_RESULT_OK` when the primary or fallback path produced a value.
 */
SimResult sim_hyperexp_phi_deriv_safe(double                lambda,
                                      double                epsilon,
                                      int                   K,
                                      SimSpecialFallbackFn  fallback,
                                      void*                 userdata,
                                      SimSpecialEvalReport* report,
                                      double*               out_value) {
    SimSpecialEvalReport local_report;
    if (report == NULL)
        report = &local_report;

    sim_special_report_seed(report,
                            "sim_hyperexp_phi_deriv",
                            (SimComplexDouble) { lambda, 0.0 },
                            NAN,
                            epsilon,
                            (double) K,
                            0.0);

    SimResult status = SIM_RESULT_INVALID_ARGUMENT;

    if (!isfinite(lambda) || !isfinite(epsilon) || K <= 0) {
        sim_special_report_update(report, SIM_SPECIAL_FAULT_DOMAIN, 0, NAN);
        goto fail;
    }

    double value = sim_hyperexp_phi_deriv(lambda, epsilon, K);
    if (isfinite(value)) {
        if (out_value)
            *out_value = value;
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NONE, 0, fabs(value));
        return SIM_RESULT_OK;
    }

    double a = lambda + epsilon;
    if ((a <= 0.0 && sim_is_int(a)) || ((a + (double) K) <= 0.0 && sim_is_int(a + (double) K)))
        sim_special_report_update(report, SIM_SPECIAL_FAULT_SINGULARITY, 0, NAN);
    else
        sim_special_report_update(report, SIM_SPECIAL_FAULT_NUMERIC, 0, value);

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
