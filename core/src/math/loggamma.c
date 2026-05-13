/**
 * @file loggamma.c
 * @brief Principal-branch complex log-gamma evaluation via Lanczos approximation.
 *
 * The evaluator is primarily used by zeta and xi completion factors. It detects
 * poles on the nonpositive real axis, applies the reflection formula for
 * `Re(z) < 0.5`, and returns a structured status/error estimate.
 */

#include "oakfield/math/loggamma.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define SIM_LOG_GAMMA_LANCZOS_G 7.0
#define SIM_LOG_GAMMA_LANCZOS_TERM_COUNT 9U
#define SIM_LOG_SQRT_TWO_PI 0.91893853320467274178032973640562

static const double sim_log_gamma_lanczos_coeffs[SIM_LOG_GAMMA_LANCZOS_TERM_COUNT] = {
    0.99999999999980993227684700473478,
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7
};

/**
 * @brief Convert the ABI complex structure to C99 `double complex`.
 *
 * @param z ABI-stable complex input.
 * @return C99 complex value with matching components.
 */
static inline double complex sim_to_c64(SimComplexDouble z) {
    return z.re + z.im * I;
}

/**
 * @brief Convert a C99 `double complex` value to the ABI structure.
 *
 * @param z C99 complex input.
 * @return ABI-stable complex value with matching components.
 */
static inline SimComplexDouble sim_from_c64(double complex z) {
    return (SimComplexDouble) { creal(z), cimag(z) };
}

/**
 * @brief Check whether a log-gamma argument has finite components.
 *
 * @param z Complex argument.
 * @return true when both real and imaginary components are finite.
 */
static inline bool sim_log_gamma_isfinite(SimComplexDouble z) {
    return isfinite(z.re) && isfinite(z.im);
}

/**
 * @brief Test whether a real value is close to an integer for pole detection.
 *
 * @param value Real value to test.
 * @return true when `value` is within a scaled epsilon of an integer.
 */
static bool sim_log_gamma_is_near_integer(double value) {
    double nearest = nearbyint(value);
    double tol     = 32.0 * DBL_EPSILON * fmax(1.0, fabs(value));
    return fabs(value - nearest) <= tol;
}

/**
 * @brief Detect poles of Gamma on the nonpositive real axis.
 *
 * @param z Complex argument.
 * @return true when `z` is sufficiently close to 0, -1, -2, ...
 */
static bool sim_log_gamma_has_pole(SimComplexDouble z) {
    if (fabs(z.im) > 32.0 * DBL_EPSILON) {
        return false;
    }
    return z.re <= 0.0 && sim_log_gamma_is_near_integer(z.re);
}

/**
 * @brief Evaluate complex log-gamma with the Lanczos approximation.
 *
 * For `Re(z) < 0.5`, the reflection formula is used and
 * `SIM_LOG_GAMMA_FLAG_USED_REFLECTION` is set in `flags`.
 *
 * @param z Complex argument away from poles.
 * @param flags Optional bitset updated with implementation choices.
 * @return Principal-branch approximation to log Gamma(z).
 */
static double complex sim_log_gamma_lanczos(double complex z, unsigned int* flags) {
    if (creal(z) < 0.5) {
        if (flags != NULL) {
            *flags |= SIM_LOG_GAMMA_FLAG_USED_REFLECTION;
        }
        return log(M_PI) - clog(csin(M_PI * z)) - sim_log_gamma_lanczos(1.0 - z, flags);
    }

    double complex shifted = z - 1.0;
    double complex series  = sim_log_gamma_lanczos_coeffs[0];

    for (size_t i = 1U; i < SIM_LOG_GAMMA_LANCZOS_TERM_COUNT; ++i) {
        series += sim_log_gamma_lanczos_coeffs[i] / (shifted + (double) i);
    }

    double complex t = shifted + SIM_LOG_GAMMA_LANCZOS_G + 0.5;
    return SIM_LOG_SQRT_TWO_PI + (shifted + 0.5) * clog(t) - t + clog(series);
}

/**
 * @brief Convert a log-gamma status enum to a diagnostic string.
 *
 * @param status Status value returned by log-gamma evaluators.
 * @return Static string naming the status.
 */
const char* sim_log_gamma_status_string(SimLogGammaStatus status) {
    switch (status) {
        case SIM_LOG_GAMMA_STATUS_OK:
            return "ok";
        case SIM_LOG_GAMMA_STATUS_SINGULAR:
            return "singular";
        case SIM_LOG_GAMMA_STATUS_INVALID_ARGUMENT:
            return "invalid-argument";
        case SIM_LOG_GAMMA_STATUS_NUMERIC_FAILURE:
            return "numeric-failure";
        default:
            return "unknown";
    }
}

/**
 * @brief Evaluate principal-branch complex log-gamma with status metadata.
 *
 * @param z Complex argument.
 * @return Value, error estimate, Lanczos term count, status, and flags.
 */
SimLogGammaResult sim_log_gamma_eval(SimComplexDouble z) {
    SimLogGammaResult out = { .value         = { NAN, NAN },
                              .abs_error     = NAN,
                              .lanczos_terms = SIM_LOG_GAMMA_LANCZOS_TERM_COUNT,
                              .status        = SIM_LOG_GAMMA_STATUS_OK,
                              .flags         = 0U };

    if (!sim_log_gamma_isfinite(z)) {
        out.status = SIM_LOG_GAMMA_STATUS_INVALID_ARGUMENT;
        return out;
    }

    if (sim_log_gamma_has_pole(z)) {
        out.status = SIM_LOG_GAMMA_STATUS_SINGULAR;
        return out;
    }

    double complex value = sim_log_gamma_lanczos(sim_to_c64(z), &out.flags);
    if (!isfinite(creal(value)) || !isfinite(cimag(value))) {
        out.status = SIM_LOG_GAMMA_STATUS_NUMERIC_FAILURE;
        return out;
    }

    out.value = sim_from_c64(value);
    out.abs_error =
        (32.0 + ((out.flags & SIM_LOG_GAMMA_FLAG_USED_REFLECTION) != 0U ? 96.0 : 0.0)) *
        DBL_EPSILON * fmax(1.0, cabs(value));
    return out;
}

/**
 * @brief Convenience wrapper returning only the complex log-gamma value.
 *
 * @param z Complex argument.
 * @return log Gamma(z), or `{NAN, NAN}` when evaluation fails.
 */
SimComplexDouble sim_log_gamma_value(SimComplexDouble z) {
    SimLogGammaResult result = sim_log_gamma_eval(z);
    return (result.status == SIM_LOG_GAMMA_STATUS_OK) ? result.value
                                                      : (SimComplexDouble) { NAN, NAN };
}
