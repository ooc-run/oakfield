/**
 * @file special_common.c
 * @brief Shared diagnostics, fallback plumbing, and Bernoulli tables for special functions.
 *
 * The public safe wrappers share a common error-report structure and fallback
 * convention. This file also owns the Bernoulli tables used by the fixed and
 * adaptive Stirling tails in the polygamma family.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Special-function fault helpers.
 *
 * The strings are stable, compact labels intended for logs and test assertions.
 * They are not localized and should not be parsed as user-facing prose.
 */
static const char* const sim_special_fault_names[] = { "none",
                                                       "domain",
                                                       "singularity",
                                                       "iteration-limit",
                                                       "numeric" };

static const char kSimSpecialUnknownFunction[] = "<unknown>";

/**
 * @brief Convert a special-function fault enum to a stable diagnostic string.
 *
 * @param fault Fault category reported by a safe evaluator.
 * @return A static string naming `fault`, or `"unknown"` when out of range.
 */
const char* sim_special_fault_name(SimSpecialFault fault) {
    size_t index = (size_t) fault;
    if (index >= (sizeof(sim_special_fault_names) / sizeof(sim_special_fault_names[0])))
        return "unknown";
    return sim_special_fault_names[index];
}

/**
 * @brief Initialize a special-function evaluation report before computation.
 *
 * All fields are reset to a no-fault state while preserving metadata about the
 * function, primary argument, q parameter, auxiliary parameter, exponent, and
 * tolerance used by the evaluator.
 *
 * @param report Report to initialize; may be NULL.
 * @param function Static or long-lived function name string.
 * @param input Primary real or complex input.
 * @param q q-analog parameter, or NaN when not applicable.
 * @param aux Auxiliary real parameter, or NaN when not applicable.
 * @param exponent Exponent/order parameter, or NaN when not applicable.
 * @param tol Target tolerance recorded for diagnostics.
 */
void sim_special_report_seed(SimSpecialEvalReport* report,
                             const char*           function,
                             SimComplexDouble      input,
                             double                q,
                             double                aux,
                             double                exponent,
                             double                tol) {
    if (report == NULL)
        return;
    report->fault           = SIM_SPECIAL_FAULT_NONE;
    report->function        = (function != NULL) ? function : kSimSpecialUnknownFunction;
    report->input           = input;
    report->q_param         = q;
    report->aux_param       = aux;
    report->exponent_param  = exponent;
    report->iteration_count = 0;
    report->residual        = 0.0;
    report->tolerance       = tol;
}

/**
 * @brief Update a special-function report with the latest fault and residual data.
 *
 * @param report Report to mutate; may be NULL.
 * @param fault New fault classification.
 * @param iteration Iteration count or term index associated with the update.
 * @param residual Last term magnitude, residual, or diagnostic scalar.
 */
void sim_special_report_update(SimSpecialEvalReport* report,
                               SimSpecialFault       fault,
                               int                   iteration,
                               double                residual) {
    if (report == NULL)
        return;
    report->fault           = fault;
    report->iteration_count = iteration;
    report->residual        = residual;
}

/**
 * @brief Invoke a user fallback callback when a safe evaluator cannot complete.
 *
 * @param fallback User callback; NULL means no fallback is available.
 * @param userdata Opaque pointer forwarded to the callback.
 * @param report Read-only diagnostic report describing the primary failure.
 * @param value_out Receives a replacement value from the callback.
 * @return Callback status, or `SIM_RESULT_INVALID_ARGUMENT` when no callback exists.
 */
SimResult sim_special_apply_fallback(SimSpecialFallbackFn        fallback,
                                     void*                       userdata,
                                     const SimSpecialEvalReport* report,
                                     SimComplexDouble*           value_out) {
    if (fallback == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;
    return fallback(userdata, report, value_out);
}


/**
 * @brief Bernoulli numbers B_{2n} for Stirling tails.
 *
 * Digamma:  - Σ_{n>=1} B_{2n}/(2n z^{2n})
 * Trigamma: + Σ_{n>=1} B_{2n}/(   z^{2n+1})
 *
 * The arrays are aligned for cache-friendly scalar/vector access and are
 * intentionally short; callers clamp requested tail lengths to the available
 * table size.
 */
SIM_ALIGN64_PRE const double sim_polygamma_B2[SIM_POLYGAMMA_B2_COUNT] SIM_ALIGN64_POST = {
    1.0 / 6.0,       -1.0 / 30.0,       1.0 / 42.0,       -1.0 / 30.0,
    5.0 / 66.0,      -691.0 / 2730.0,   7.0 / 6.0,        -3617.0 / 510.0,
    43867.0 / 798.0, -174611.0 / 330.0, 854513.0 / 138.0, -236364091.0 / 2730.0
}; /* up to B_{24}; adaptively truncated by chosen tail length */

SIM_ALIGN64_PRE const float sim_polygamma_B2f[SIM_POLYGAMMA_B2F_COUNT] SIM_ALIGN64_POST = { 1.0f / 6.0f,  -1.0f / 30.0f,
                                                              1.0f / 42.0f, -1.0f / 30.0f,
                                                              5.0f / 66.0f, -691.0f / 2730.0f,
                                                              7.0f / 6.0f,  -3617.0f / 510.0f };

/**
 * @brief Precomputed digamma-tail coefficients derived from Bernoulli numbers.
 *
 * `C2[n-1] = -B_{2n} / (2n)` so fixed-tail digamma paths can use one
 * multiply-add per Bernoulli pair.
 */
SIM_ALIGN64_PRE const double sim_polygamma_C2[SIM_POLYGAMMA_C2_COUNT] SIM_ALIGN64_POST = {
    -B2[0] / (2.0 * 1.0), -B2[1] / (2.0 * 2.0),  -B2[2] / (2.0 * 3.0),   -B2[3] / (2.0 * 4.0),
    -B2[4] / (2.0 * 5.0), -B2[5] / (2.0 * 6.0),  -B2[6] / (2.0 * 7.0),   -B2[7] / (2.0 * 8.0),
    -B2[8] / (2.0 * 9.0), -B2[9] / (2.0 * 10.0), -B2[10] / (2.0 * 11.0), -B2[11] / (2.0 * 12.0)
};

SIM_ALIGN64_PRE const float sim_polygamma_C2f[SIM_POLYGAMMA_C2F_COUNT] SIM_ALIGN64_POST = {
    -B2f[0] / (2.0f * 1.0f), -B2f[1] / (2.0f * 2.0f), -B2f[2] / (2.0f * 3.0f),
    -B2f[3] / (2.0f * 4.0f), -B2f[4] / (2.0f * 5.0f), -B2f[5] / (2.0f * 6.0f),
    -B2f[6] / (2.0f * 7.0f), -B2f[7] / (2.0f * 8.0f)
};
