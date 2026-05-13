/**
 * @file crank_nicolson.c
 * @brief Fixed-point Crank-Nicolson integrator for real and complex fields.
 *
 * The method advances the trapezoidal implicit update
 * `u_{n+1} = u_n + (dt / 2) * (f(u_n) + f(u_{n+1}))`. The implementation uses
 * an explicit Euler predictor followed by fixed-point midpoint iteration.
 * Adaptive mode only shrinks the timestep after failed convergence attempts;
 * no embedded accuracy estimate is produced.
 */

#include "oakfield/integrator.h"
#include <math.h>
#include <string.h>

/**
 * @brief Advance one field with Crank-Nicolson fixed-point iteration.
 *
 * Residuals compare successive midpoint guesses with the shared real or complex
 * error metric. Stochastic increments are applied after the converged or final
 * guess is copied back to the field.
 *
 * @param integrator Configured Crank-Nicolson integrator.
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

    /* Determine scalar element count for the field state */
    size_t count = integrator_state_length(field);
    if (count == 0U)
        return;

    if (integrator_ensure_workspace(integrator, 6U, count) != SIM_RESULT_OK)
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
        double *heun_state = integrator_buffer(integrator, 4U);
        double *noise_scratch = integrator_buffer(integrator, 5U);

        if (!state)
            return;

        for (attempt = 0U; attempt < max_attempts; ++attempt)
        {
            double residual = 0.0;
            bool heun_ready = false;

            if (integrator->drift(integrator, field, state, k0, count) != SIM_RESULT_OK)
                return;

            for (size_t i = 0; i < count; ++i)
                guess[i] = state[i] + (double)step * k0[i];

            for (unsigned int iter = 0U; iter < max_iterations; ++iter)
            {
                if (integrator->drift(integrator, field, guess, k_guess, count) != SIM_RESULT_OK)
                    return;

                if (!heun_ready)
                {
                    for (size_t i = 0; i < count; ++i)
                        heun_state[i] = state[i] + (double)step * 0.5 * (k0[i] + k_guess[i]);
                    heun_ready = true;
                }

                for (size_t i = 0; i < count; ++i)
                    updated[i] = state[i] + (double)step * 0.5 * (k0[i] + k_guess[i]);

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
    SimComplexDouble *heun_state = integrator_buffer_complex(integrator, 4U);
    SimComplexDouble *noise_scratch = integrator_buffer_complex(integrator, 5U);

    if (!state)
        return;

    for (attempt = 0U; attempt < max_attempts; ++attempt)
    {
        double residual = 0.0;
        bool heun_ready = false;

        if (integrator->drift(integrator, field, (double *)state, (double *)k0, count) != SIM_RESULT_OK)
            return;

        for (size_t i = 0; i < count; ++i)
        {
            guess[i].re = state[i].re + step * k0[i].re;
            guess[i].im = state[i].im + step * k0[i].im;
        }

        for (unsigned int iter = 0U; iter < max_iterations; ++iter)
        {
            if (integrator->drift(integrator, field, (double *)guess, (double *)k_guess, count) != SIM_RESULT_OK)
                return;

            if (!heun_ready)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    heun_state[i].re = state[i].re + step * 0.5 * (k0[i].re + k_guess[i].re);
                    heun_state[i].im = state[i].im + step * 0.5 * (k0[i].im + k_guess[i].im);
                }
                heun_ready = true;
            }

            for (size_t i = 0; i < count; ++i)
            {
                updated[i].re = state[i].re + step * 0.5 * (k0[i].re + k_guess[i].re);
                updated[i].im = state[i].im + step * 0.5 * (k0[i].im + k_guess[i].im);
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

/* ------------------------------------------------------------------------- */

/**
 * @brief Create a Crank-Nicolson integrator.
 *
 * @param config Integrator configuration; drift callback is required.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK or an error from integrator_configure().
 */
SimResult integrator_crank_nicolson_create(const IntegratorConfig *config, Integrator *out)
{
    return integrator_configure(out, "crank_nicolson", integrator_step, config);
}
