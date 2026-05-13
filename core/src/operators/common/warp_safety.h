/**
 * @file warp_safety.h
 * @brief Shared singularity guards for analytic warp-style gradient sampling.
 */
#ifndef OAKFIELD_WARP_SAFETY_H
#define OAKFIELD_WARP_SAFETY_H

#include "oakfield/field.h"
#include "oakfield/operator.h"
#include "oakfield/math/special_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Guard metadata describing how to handle singularities during warp sampling.
 */
typedef struct SimWarpGuard {
    SimContinuityMode mode;      /**< Continuity policy. */
    double            clamp_min; /**< Lower clamp bound when applicable. */
    double            clamp_max; /**< Upper clamp bound when applicable. */
    double            tolerance; /**< Offset tolerance used for limited continuity probes. */
} SimWarpGuard;

/**
 * @brief Populate a warp guard structure from an operator instance.
 *
 * When @p op is NULL the defaults are used. Invalid clamp ranges and non-positive
 * tolerances are sanitized before writing @p out_guard.
 *
 * @param op Optional operator whose continuity config should override defaults.
 * @param default_clamp_min Default lower clamp bound.
 * @param default_clamp_max Default upper clamp bound.
 * @param default_tolerance Default positive probe tolerance.
 * @param[out] out_guard Receives the sanitized guard; NULL is ignored.
 */
void sim_warp_guard_from_operator(const SimOperator* op,
                                  double             default_clamp_min,
                                  double             default_clamp_max,
                                  double             default_tolerance,
                                  SimWarpGuard*      out_guard);

/**
 * @brief Specification describing a single warp gradient sample request.
 */
typedef struct SimWarpSampleSpec {
    double       sample; /**< Raw sample value before bias. */
    double       bias;   /**< Bias applied prior to evaluating the gradient. */
    double       delta;  /**< Symmetric offset used for finite-difference style scaling. */
    double       lambda; /**< Multiplicative response scale. */
    SimWarpGuard guard;  /**< Continuity/clamp policy to enforce. */
} SimWarpSampleSpec;

/**
 * @brief Callback signature used to evaluate profile-specific gradients.
 *
 * @param userdata Caller-owned profile state.
 * @param sample Biased sample value at which to evaluate the gradient.
 * @param fallback Optional special-function fallback hook.
 * @param fallback_userdata Userdata passed to @p fallback.
 * @param[out] out_gradient Receives the profile gradient.
 * @return #SIM_RESULT_OK on success or an error code for failed evaluation.
 */
typedef SimResult (*SimWarpGradientFn)(void*                userdata,
                                       double               sample,
                                       SimSpecialFallbackFn fallback,
                                       void*                fallback_userdata,
                                       double*              out_gradient);

/**
 * @brief Sample a warp gradient using the provided specification and guard policy.
 *
 * Guard modes can clamp finite gradients or probe neighboring samples when the
 * primary evaluation fails or returns a non-finite value.
 *
 * @param spec Sample request and guard policy.
 * @param gradient_fn Profile-specific gradient callback.
 * @param gradient_userdata Userdata passed to @p gradient_fn.
 * @param fallback Optional special-function fallback hook.
 * @param fallback_userdata Userdata passed to @p fallback.
 * @param[out] out_gradient Receives the guarded gradient.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for missing
 *         required pointers, or the callback error when unguarded evaluation fails.
 */
SimResult sim_warp_sample_gradient(const SimWarpSampleSpec* spec,
                                   SimWarpGradientFn        gradient_fn,
                                   void*                    gradient_userdata,
                                   SimSpecialFallbackFn     fallback,
                                   void*                    fallback_userdata,
                                   double*                  out_gradient);

/**
 * @brief Sample a warp response (lambda * 2 * delta * gradient) with guard handling.
 *
 * @param spec Sample request and guard policy.
 * @param gradient_fn Profile-specific gradient callback.
 * @param gradient_userdata Userdata passed to @p gradient_fn.
 * @param fallback Optional special-function fallback hook.
 * @param fallback_userdata Userdata passed to @p fallback.
 * @param[out] out_response Receives the scaled response.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for missing or
 *         non-finite required values, or an error from sim_warp_sample_gradient().
 */
SimResult sim_warp_sample_response(const SimWarpSampleSpec* spec,
                                   SimWarpGradientFn        gradient_fn,
                                   void*                    gradient_userdata,
                                   SimSpecialFallbackFn     fallback,
                                   void*                    fallback_userdata,
                                   double*                  out_response);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_WARP_SAFETY_H */
