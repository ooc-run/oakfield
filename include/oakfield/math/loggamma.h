/**
 * @file loggamma.h
 * @brief Complex log-gamma helpers for Riemann zeta/xi evaluation.
 */
#ifndef OAKFIELD_MATH_LOGGAMMA_H
#define OAKFIELD_MATH_LOGGAMMA_H

#include <stddef.h>

#include "oakfield/field.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status values returned by complex log-gamma evaluators.
 */
typedef enum SimLogGammaStatus {
    SIM_LOG_GAMMA_STATUS_OK = 0,           /**< Evaluation completed successfully. */
    SIM_LOG_GAMMA_STATUS_SINGULAR,         /**< Argument lies on a log-gamma singularity. */
    SIM_LOG_GAMMA_STATUS_INVALID_ARGUMENT, /**< Input argument or output pointer was invalid. */
    SIM_LOG_GAMMA_STATUS_NUMERIC_FAILURE   /**< Evaluation produced an unstable numeric result. */
} SimLogGammaStatus;

#define SIM_LOG_GAMMA_FLAG_USED_REFLECTION 0x1u

/**
 * @brief Principal log-gamma value with error estimate and diagnostic metadata.
 */
typedef struct SimLogGammaResult {
    SimComplexDouble value;   /**< Principal log-gamma value. */
    double abs_error;         /**< Absolute error estimate. */
    size_t lanczos_terms;     /**< Number of Lanczos terms used. */
    SimLogGammaStatus status; /**< Evaluation status. */
    unsigned int flags;       /**< Bitmask of SIM_LOG_GAMMA_FLAG_* values. */
} SimLogGammaResult;

/**
 * @brief Human-readable description of a log-gamma status value.
 *
 * @param status Status returned by log-gamma evaluators.
 * @return Static diagnostic string.
 */
const char *sim_log_gamma_status_string(SimLogGammaStatus status);

/**
 * @brief Evaluate the principal-branch complex log-gamma function.
 *
 * @param z Complex argument.
 * @return Structured value, error estimate, status, and implementation flags.
 */
SimLogGammaResult sim_log_gamma_eval(SimComplexDouble z);

/**
 * @brief Convenience wrapper returning only the complex log-gamma value.
 *
 * On failure both components are returned as NaN.
 *
 * @param z Complex argument.
 * @return log Gamma(z), or `{NAN, NAN}` on failure.
 */
SimComplexDouble sim_log_gamma_value(SimComplexDouble z);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_LOGGAMMA_H */
