/**
 * @file backward_euler.c
 * @brief Fixed-point backward Euler integrator for stiff drifts.
 *
 * Backward Euler advances `u_{n+1} = u_n + dt * f(u_{n+1})` for real and
 * complex fields. The implementation uses an explicit Euler prediction as the
 * initial guess, then iterates to the configured tolerance with a fixed
 * iteration cap. Adaptive mode reduces the timestep between failed convergence
 * attempts but does not compute a separate embedded accuracy estimate.
 */

#include "oakfield/integrator.h"

#include <math.h>
#include <string.h>

/**
 * @brief Advance one field with backward Euler fixed-point iteration.
 *
 * On each attempt, the residual is the normalized difference between successive
 * fixed-point guesses. If the residual reaches tolerance, the current guess is
 * accepted. If convergence fails and adaptivity is enabled, the timestep is
 * halved by the configured safety factor down to `min_dt`.
 *
 * @param integrator Configured backward Euler integrator.
 * @param field Field updated in place.
 * @param dt Requested timestep, or nonpositive to use the stored suggestion.
 */
static void integrator_step(Integrator *integrator, Field *field, double dt)
{
    const unsigned int max_attempts = 6U;
    const unsigned int max_iterations = 12U;
    unsigned int attempt;
    bool accepted = false;

    if (!integrator || !field || !integrator->drift)
        return;

    const bool is_complex = sim_field_domain_is_complex(field);
    integrator->is_complex = is_complex;

    size_t count = integrator_state_length(field);
    if (count == 0U)
        return;

    if (integrator_ensure_workspace(integrator, 5U, count) != SIM_RESULT_OK)
        return;

    double step = (dt > 0.0) ? dt : integrator_next_step(integrator);
    step = integrator_clamp_dt(integrator, step);

    /* --------------------------------------------------------------------- */
    /* REAL BRANCH                                                          */
    /* --------------------------------------------------------------------- */
    if (!is_complex)
    {
        double *state = (double *)sim_field_data(field);
        double *k0 = integrator_buffer(integrator, 0U);
        double *guess = integrator_buffer(integrator, 1U);
        double *k_guess = integrator_buffer(integrator, 2U);
        double *updated = integrator_buffer(integrator, 3U);
        double *noise_scratch = integrator_buffer(integrator, 4U);

        if (!state || !k0 || !guess || !k_guess || !updated || !noise_scratch)
            return;

        for (attempt = 0U; attempt < max_attempts; ++attempt)
        {
            double residual = 0.0;

            if (integrator->drift(integrator, field, state, k0, count) != SIM_RESULT_OK)
                return;

            for (size_t i = 0U; i < count; ++i)
                guess[i] = state[i] + (double)step * k0[i];

            for (unsigned int iter = 0U; iter < max_iterations; ++iter)
            {
                if (integrator->drift(integrator, field, guess, k_guess, count) != SIM_RESULT_OK)
                    return;

                for (size_t i = 0U; i < count; ++i)
                    updated[i] = state[i] + (double)step * k_guess[i];

                residual = integrator_measure_error(guess, updated, count);
                memcpy(guess, updated, count * sizeof(double));

                if (residual <= (double)integrator->tolerance)
                {
                    accepted = true;
                    break;
                }
            }

            if (accepted)
                break;

            if (!integrator->adaptive || step <= integrator->min_dt)
                break;

            step = fmax(integrator->min_dt, integrator->safety * 0.5 * step);
        }

        memcpy(state, guess, count * sizeof(double));
        integrator_apply_stochastic(integrator, field, noise_scratch, count, step);

        integrator->last_step = step;
        integrator->last_error = 0.0;
        integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
        integrator->last_rejection_count =
            integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
        integrator->current_dt = integrator_clamp_dt(integrator, step);
        return;
    }

    /* --------------------------------------------------------------------- */
    /* COMPLEX BRANCH                                                       */
    /* --------------------------------------------------------------------- */
    SimComplexDouble *state = sim_field_complex_data(field);
    SimComplexDouble *k0 = integrator_buffer_complex(integrator, 0U);
    SimComplexDouble *guess = integrator_buffer_complex(integrator, 1U);
    SimComplexDouble *k_guess = integrator_buffer_complex(integrator, 2U);
    SimComplexDouble *updated = integrator_buffer_complex(integrator, 3U);
    SimComplexDouble *noise_scratch = integrator_buffer_complex(integrator, 4U);

    if (!state || !k0 || !guess || !k_guess || !updated || !noise_scratch)
        return;

    for (attempt = 0U; attempt < max_attempts; ++attempt)
    {
        double residual = 0.0;

        if (integrator->drift(integrator, field, (double *)state, (double *)k0, count) != SIM_RESULT_OK)
            return;

        for (size_t i = 0U; i < count; ++i)
        {
            guess[i].re = state[i].re + step * k0[i].re;
            guess[i].im = state[i].im + step * k0[i].im;
        }

        for (unsigned int iter = 0U; iter < max_iterations; ++iter)
        {
            if (integrator->drift(integrator, field, (double *)guess, (double *)k_guess, count) != SIM_RESULT_OK)
                return;

            for (size_t i = 0U; i < count; ++i)
            {
                updated[i].re = state[i].re + step * k_guess[i].re;
                updated[i].im = state[i].im + step * k_guess[i].im;
            }

            residual = integrator_measure_error_complex(guess, updated, count);
            memcpy(guess, updated, count * sizeof(SimComplexDouble));

            if (residual <= (double)integrator->tolerance)
            {
                accepted = true;
                break;
            }
        }

        if (accepted)
            break;

        if (!integrator->adaptive || step <= integrator->min_dt)
            break;

        step = fmax(integrator->min_dt, integrator->safety * 0.5 * step);
    }

    memcpy(state, guess, count * sizeof(SimComplexDouble));
    integrator_apply_stochastic(integrator, field, (double *)noise_scratch, count, step);

    integrator->last_step = step;
    integrator->last_error = 0.0;
    integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
    integrator->last_rejection_count =
        integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
    integrator->current_dt = integrator_clamp_dt(integrator, step);
}

/**
 * @brief Create a backward Euler integrator.
 *
 * @param config Integrator configuration; drift callback is required.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK or an error from integrator_configure().
 */
SimResult integrator_backward_euler_create(const IntegratorConfig *config, Integrator *out)
{
    return integrator_configure(out, "backward_euler", integrator_step, config);
}
