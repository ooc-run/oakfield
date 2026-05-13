/**
 * @file heun.c
 * @brief Explicit Heun predictor-corrector integrator.
 *
 * This module advances real and complex fields with the trapezoidal
 * predictor-corrector pair: an Euler predictor followed by averaging the start
 * and predicted-end drifts. The Euler prediction also provides the embedded
 * error estimate used by the shared adaptive timestep controller.
 */

#include "oakfield/integrator.h"
#include "sim_accel.h"
#include <math.h>
#include <string.h>

/**
 * @brief Advance one field with Heun's method and optional adaptive rejection.
 *
 * The deterministic candidate is the second-order Heun correction. For
 * adaptive runs, the difference between the Euler predictor and corrected state
 * is used as a normalized error estimate; fixed-step runs report zero error.
 * Stochastic increments are applied after the deterministic candidate is copied
 * back to the field.
 *
 * @param integrator Configured Heun integrator.
 * @param field Field updated in place.
 * @param dt Requested timestep, or nonpositive to use the stored suggestion.
 */
static void integrator_step(Integrator *integrator, Field *field, double dt)
{
    const unsigned int max_attempts = 8U;
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
        double *k1 = integrator_buffer(integrator, 0U);
        double *euler_state = integrator_buffer(integrator, 1U);
        double *k2 = integrator_buffer(integrator, 2U);
        double *heun_state = integrator_buffer(integrator, 3U);
        double *noise_scratch = integrator_buffer(integrator, 4U);

        if (!state)
            return;

        double error_norm = 0.0;

        for (attempt = 0U; attempt < max_attempts; ++attempt)
        {
            const double* weighted_sources[3];
            double        weighted_coeffs[3];

            if (integrator->drift(integrator, field, state, k1, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = step;
            sim_accel_weighted_sum_real(
                weighted_sources, weighted_coeffs, 2U, euler_state, count);

            if (integrator->drift(integrator, field, euler_state, k2, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_sources[2] = k2;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = step * 0.5;
            weighted_coeffs[2] = step * 0.5;
            sim_accel_weighted_sum_real(
                weighted_sources, weighted_coeffs, 3U, heun_state, count);

            error_norm = integrator->adaptive
                             ? integrator_measure_error(euler_state, heun_state, count)
                             : 0.0;

            if (!integrator->adaptive ||
                error_norm <= (double)integrator->tolerance ||
                step <= integrator->min_dt)
            {
                accepted = true;
                break;
            }

            step = integrator_reject_dt(integrator, step, error_norm, 2.0);
        }

        memcpy(state, heun_state, count * sizeof(double));
        integrator_apply_stochastic(integrator, field, noise_scratch, count, step);

        integrator->last_step = step;
        integrator->last_error = error_norm;
        integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
        integrator->last_rejection_count =
            integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
        integrator->current_dt = (integrator->adaptive && accepted)
                                     ? integrator_suggest_dt(integrator, step, error_norm, 2.0)
                                     : integrator_clamp_dt(integrator, step);
        return;
    }

    /* --------------------------------------------------------------------- */
    /* COMPLEX BRANCH                                                       */
    /* --------------------------------------------------------------------- */
    SimComplexDouble *state = sim_field_complex_data(field);
    SimComplexDouble *k1 = integrator_buffer_complex(integrator, 0U);
    SimComplexDouble *euler_state = integrator_buffer_complex(integrator, 1U);
    SimComplexDouble *k2 = integrator_buffer_complex(integrator, 2U);
    SimComplexDouble *heun_state = integrator_buffer_complex(integrator, 3U);
    SimComplexDouble *noise_scratch = integrator_buffer_complex(integrator, 4U);

    if (!state)
        return;

    double error_norm = 0.0;

    for (attempt = 0U; attempt < max_attempts; ++attempt)
    {
        const double* weighted_sources[3];
        double        weighted_coeffs[3];

        if (integrator->drift(integrator, field, (double *)state, (double *)k1, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = step;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 2U, (double*) euler_state, count * 2U);

        if (integrator->drift(integrator, field, (double *)euler_state, (double *)k2, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_sources[2] = (const double*) k2;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = step * 0.5;
        weighted_coeffs[2] = step * 0.5;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 3U, (double*) heun_state, count * 2U);

        error_norm = integrator->adaptive
                         ? integrator_measure_error_complex(euler_state, heun_state, count)
                         : 0.0;

        if (!integrator->adaptive ||
            error_norm <= (double)integrator->tolerance ||
            step <= integrator->min_dt)
        {
            accepted = true;
            break;
        }

        step = integrator_reject_dt(integrator, step, error_norm, 2.0);
    }

    memcpy(state, heun_state, count * sizeof(SimComplexDouble));
    integrator_apply_stochastic(integrator, field, (double *)noise_scratch, count, step);

    integrator->last_step = step;
    integrator->last_error = error_norm;
    integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
    integrator->last_rejection_count =
        integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
    integrator->current_dt = (integrator->adaptive && accepted)
                                 ? integrator_suggest_dt(integrator, step, error_norm, 2.0)
                                 : integrator_clamp_dt(integrator, step);
}

/* ------------------------------------------------------------------------- */

/**
 * @brief Create a Heun predictor-corrector integrator.
 *
 * @param config Integrator configuration; drift callback is required.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK or an error from integrator_configure().
 */
SimResult integrator_heun_create(const IntegratorConfig *config, Integrator *out)
{
    return integrator_configure(out, "heun", integrator_step, config);
}
