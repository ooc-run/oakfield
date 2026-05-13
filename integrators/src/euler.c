/**
 * @file euler.c
 * @brief Explicit Euler integrator with step-doubling error control.
 *
 * The module advances real and complex fields with a first-order explicit
 * Euler step. When adaptivity is enabled, it compares one full step against two
 * half steps and uses the shared timestep controller to reject or grow the next
 * timestep. Stochastic increments are applied after the deterministic candidate
 * is copied back to the field.
 */

#include "oakfield/integrator.h"
#include "sim_accel.h"

#include <math.h>
#include <string.h>

/**
 * @brief Advance one field with explicit Euler and optional step-doubling.
 *
 * Real fields use `double[count]` workspace. Complex fields use
 * `SimComplexDouble[count]` workspace and call the shared weighted-sum helper
 * over the underlying real/imag component storage. A failed drift evaluation or
 * workspace allocation leaves the field unchanged for that attempt and returns
 * through the void step contract.
 *
 * @param integrator Configured Euler integrator.
 * @param field Field updated in place.
 * @param dt Requested timestep, or nonpositive to use the stored suggestion.
 */
static void integrator_step(Integrator* integrator, Field* field, double dt) {
    bool is_complex                 = sim_field_domain_is_complex(field);
    integrator->is_complex          = is_complex;
    const unsigned int max_attempts = 8U;
    unsigned int       attempt;
    size_t             count;
    double             step;
    bool               accepted   = false;
    double             error_norm = 0.0;

    if (integrator == NULL || field == NULL || integrator->drift == NULL)
        return;

    count = integrator_state_length(field);
    if (count == 0U)
        return;

    if (integrator_ensure_workspace(integrator, 5U, count) != SIM_RESULT_OK)
        return;

    step = (dt > 0.0) ? dt : integrator_next_step(integrator);
    step = integrator_clamp_dt(integrator, step);

    /* --- COMPLEX / REAL branch --- */
    if (is_complex) {
        /* ==============================
         * Complex-field integrator path
         * ============================== */
        SimComplexDouble* state          = sim_field_complex_data(field);
        SimComplexDouble* k1             = integrator_buffer_complex(integrator, 0U);
        SimComplexDouble* state_full     = integrator_buffer_complex(integrator, 1U);
        SimComplexDouble* state_half     = integrator_buffer_complex(integrator, 2U);
        SimComplexDouble* k_half         = integrator_buffer_complex(integrator, 3U);
        SimComplexDouble* two_half_state = integrator_buffer_complex(integrator, 4U);

        if (!state || !k1 || !state_full || !state_half || !k_half || !two_half_state)
            return;

        for (attempt = 0U; attempt < max_attempts; ++attempt) {
            double half_step = 0.5 * step;
            const double* weighted_sources[2];
            double        weighted_coeffs[2];

            if (integrator->drift(integrator, field, (double*) state, (double*) k1, count) !=
                SIM_RESULT_OK)
                return;

            weighted_sources[0] = (const double*) state;
            weighted_sources[1] = (const double*) k1;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = step;
            sim_accel_weighted_sum_real(
                weighted_sources, weighted_coeffs, 2U, (double*) state_full, count * 2U);

            if (integrator->adaptive) {
                weighted_coeffs[1] = half_step;
                sim_accel_weighted_sum_real(
                    weighted_sources, weighted_coeffs, 2U, (double*) state_half, count * 2U);

                if (integrator->drift(
                        integrator, field, (double*) state_half, (double*) k_half, count) !=
                    SIM_RESULT_OK)
                    return;

                weighted_sources[0] = (const double*) state_half;
                weighted_sources[1] = (const double*) k_half;
                weighted_coeffs[0] = 1.0;
                weighted_coeffs[1] = half_step;
                sim_accel_weighted_sum_real(
                    weighted_sources, weighted_coeffs, 2U, (double*) two_half_state, count * 2U);

                error_norm = integrator_measure_error_complex(state_full, two_half_state, count);
            } else {
                memcpy(two_half_state, state_full, count * sizeof(SimComplexDouble));
                error_norm = 0.0;
            }

            if (!integrator->adaptive || error_norm <= (double) integrator->tolerance ||
                step <= integrator->min_dt) {
                accepted = true;
                break;
            }

            step = integrator_reject_dt(integrator, step, error_norm, 1.0);
        }

        memcpy(state, two_half_state, count * sizeof(SimComplexDouble));
        integrator_apply_stochastic(integrator, field, (double*) state_full, count, step);

        integrator->last_step  = step;
        integrator->last_error = error_norm;
        integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
        integrator->last_rejection_count =
            integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
        integrator->current_dt = integrator_suggest_dt(integrator, step, error_norm, 1.0);
    } else {
        /* ==============================
         * Real-field integrator path
         * ============================== */
        double* state = (double*) sim_field_data(field);
        if (state == NULL)
            return;

        double* k1             = integrator_buffer(integrator, 0U);
        double* state_full     = integrator_buffer(integrator, 1U);
        double* state_half     = integrator_buffer(integrator, 2U);
        double* k_half         = integrator_buffer(integrator, 3U);
        double* two_half_state = integrator_buffer(integrator, 4U);

        for (attempt = 0U; attempt < max_attempts; ++attempt) {
            double half_step = 0.5 * step;
            const double* weighted_sources[2];
            double        weighted_coeffs[2];

            if (integrator->drift(integrator, field, state, k1, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = step;
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, state_full, count);

            if (integrator->adaptive) {
                weighted_coeffs[1] = half_step;
                sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, state_half, count);

                if (integrator->drift(integrator, field, state_half, k_half, count) !=
                    SIM_RESULT_OK)
                    return;

                weighted_sources[0] = state_half;
                weighted_sources[1] = k_half;
                weighted_coeffs[0] = 1.0;
                weighted_coeffs[1] = half_step;
                sim_accel_weighted_sum_real(
                    weighted_sources, weighted_coeffs, 2U, two_half_state, count);

                error_norm = integrator_measure_error(state_full, two_half_state, count);
            } else {
                memcpy(two_half_state, state_full, count * sizeof(double));
                error_norm = 0.0;
            }

            if (!integrator->adaptive || error_norm <= (double) integrator->tolerance ||
                step <= integrator->min_dt) {
                accepted = true;
                break;
            }

            step = integrator_reject_dt(integrator, step, error_norm, 1.0);
        }

        memcpy(state, two_half_state, count * sizeof(double));
        integrator_apply_stochastic(integrator, field, state_full, count, step);

        integrator->last_step  = step;
        integrator->last_error = error_norm;
        integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
        integrator->last_rejection_count =
            integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
        integrator->current_dt = integrator_suggest_dt(integrator, step, error_norm, 1.0);
    }
}

/**
 * @brief Create an explicit Euler integrator.
 *
 * @param config Integrator configuration; drift callback is required.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK or an error from integrator_configure().
 */
SimResult integrator_euler_create(const IntegratorConfig* config, Integrator* out) {
    return integrator_configure(out, "euler", integrator_step, config);
}
