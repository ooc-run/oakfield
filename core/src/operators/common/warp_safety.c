#include "operators/common/warp_safety.h"

#include <math.h>
#include <stdbool.h>

static bool sim_warp_continuity_valid(SimContinuityMode mode)
{
    switch (mode)
    {
    case SIM_CONTINUITY_NONE:
    case SIM_CONTINUITY_STRICT:
    case SIM_CONTINUITY_CLAMPED:
    case SIM_CONTINUITY_LIMITED:
        return true;
    default:
        return false;
    }
}

static double sim_warp_clamp_value(double value, double clamp_min, double clamp_max)
{
    if (!(isfinite(clamp_min) && isfinite(clamp_max) && clamp_min < clamp_max))
    {
        return value;
    }
    if (value < clamp_min)
    {
        return clamp_min;
    }
    if (value > clamp_max)
    {
        return clamp_max;
    }
    return value;
}

static double sim_warp_positive_or_default(double requested, double fallback)
{
    if (isfinite(requested) && requested > 0.0)
    {
        return requested;
    }
    return fallback;
}

void sim_warp_guard_from_operator(const SimOperator *op,
                                  double default_clamp_min,
                                  double default_clamp_max,
                                  double default_tolerance,
                                  SimWarpGuard *out_guard)
{
    const double kFallbackTolerance = 1.0e-6;
    SimWarpGuard guard = {
        .mode = SIM_CONTINUITY_NONE,
        .clamp_min = default_clamp_min,
        .clamp_max = default_clamp_max,
        .tolerance = sim_warp_positive_or_default(default_tolerance, kFallbackTolerance)};

    if (!(isfinite(guard.clamp_min) && isfinite(guard.clamp_max) && guard.clamp_min < guard.clamp_max))
    {
        guard.clamp_min = 0.0;
        guard.clamp_max = 0.0;
    }

    if (op != NULL)
    {
        SimContinuityMode mode = op->config.continuity;
        guard.mode = sim_warp_continuity_valid(mode) ? mode : SIM_CONTINUITY_NONE;

        double clamp_min = (double)op->config.clamp_min;
        double clamp_max = (double)op->config.clamp_max;
        if (isfinite(clamp_min) && isfinite(clamp_max) && clamp_min < clamp_max)
        {
            guard.clamp_min = clamp_min;
            guard.clamp_max = clamp_max;
        }

        double tol = (double)op->config.continuity_tol;
        if (isfinite(tol) && tol > 0.0)
        {
            guard.tolerance = tol;
        }
    }

    guard.clamp_min = isfinite(guard.clamp_min) ? guard.clamp_min : 0.0;
    guard.clamp_max = isfinite(guard.clamp_max) ? guard.clamp_max : 0.0;
    if (!(guard.clamp_min < guard.clamp_max))
    {
        guard.clamp_min = default_clamp_min;
        guard.clamp_max = default_clamp_max;
    }

    guard.tolerance = sim_warp_positive_or_default(guard.tolerance, kFallbackTolerance);

    if (out_guard != NULL)
    {
        *out_guard = guard;
    }
}

SimResult sim_warp_sample_gradient(const SimWarpSampleSpec *spec,
                                   SimWarpGradientFn gradient_fn,
                                   void *gradient_userdata,
                                   SimSpecialFallbackFn fallback,
                                   void *fallback_userdata,
                                   double *out_gradient)
{
    if (spec == NULL || gradient_fn == NULL || out_gradient == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimWarpGuard *guard = &spec->guard;
    double biased_sample = spec->sample + spec->bias;
    double gradient = 0.0;
    SimResult rc = gradient_fn(gradient_userdata,
                               biased_sample,
                               fallback,
                               fallback_userdata,
                               &gradient);
    if (rc == SIM_RESULT_OK && isfinite(gradient))
    {
        if (guard->mode == SIM_CONTINUITY_CLAMPED || guard->mode == SIM_CONTINUITY_LIMITED)
        {
            gradient = sim_warp_clamp_value(gradient, guard->clamp_min, guard->clamp_max);
        }
        *out_gradient = gradient;
        return SIM_RESULT_OK;
    }

    if (guard->mode != SIM_CONTINUITY_CLAMPED && guard->mode != SIM_CONTINUITY_LIMITED)
    {
        return rc;
    }

    double fallback_tol = sim_warp_positive_or_default(fabs(spec->delta), 1.0e-6);
    double epsilon = sim_warp_positive_or_default(guard->tolerance, fallback_tol);

    double g_plus = NAN;
    double g_minus = NAN;
    SimResult rc_plus = gradient_fn(gradient_userdata,
                                    biased_sample + epsilon,
                                    fallback,
                                    fallback_userdata,
                                    &g_plus);
    SimResult rc_minus = gradient_fn(gradient_userdata,
                                     biased_sample - epsilon,
                                     fallback,
                                     fallback_userdata,
                                     &g_minus);

    bool have_plus = (rc_plus == SIM_RESULT_OK) && isfinite(g_plus);
    bool have_minus = (rc_minus == SIM_RESULT_OK) && isfinite(g_minus);
    double candidate = NAN;

    if (guard->mode == SIM_CONTINUITY_CLAMPED)
    {
        if (have_plus)
        {
            candidate = g_plus;
        }
        else if (have_minus)
        {
            candidate = g_minus;
        }
    }
    else
    {
        if (have_plus && have_minus)
        {
            candidate = 0.5 * (g_plus + g_minus);
        }
        else if (have_plus)
        {
            candidate = g_plus;
        }
        else if (have_minus)
        {
            candidate = g_minus;
        }
    }

    if (!isfinite(candidate))
    {
        candidate = 0.0;
    }

    candidate = sim_warp_clamp_value(candidate, guard->clamp_min, guard->clamp_max);
    *out_gradient = candidate;
    return SIM_RESULT_OK;
}

SimResult sim_warp_sample_response(const SimWarpSampleSpec *spec,
                                   SimWarpGradientFn gradient_fn,
                                   void *gradient_userdata,
                                   SimSpecialFallbackFn fallback,
                                   void *fallback_userdata,
                                   double *out_response)
{
    if (out_response == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double gradient = 0.0;
    SimResult rc = sim_warp_sample_gradient(spec,
                                            gradient_fn,
                                            gradient_userdata,
                                            fallback,
                                            fallback_userdata,
                                            &gradient);
    if (rc != SIM_RESULT_OK)
    {
        return rc;
    }

    if (!isfinite(gradient))
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double delta = fabs(spec->delta);
    delta = sim_warp_positive_or_default(delta, 1.0e-6);

    if (!isfinite(spec->lambda))
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double response = spec->lambda * (2.0 * delta) * gradient;
    if (!isfinite(response))
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_response = response;
    return SIM_RESULT_OK;
}
