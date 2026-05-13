/**
 * @file rk4.c
 * @brief Classical fourth-order Runge-Kutta integrator.
 *
 * The implementation supports real and complex fields and uses the shared
 * weighted-sum acceleration layer for stage combinations. Adaptive mode uses a
 * lower-order Heun comparison constructed from the start and end drifts as a
 * practical local error estimate while the deterministic candidate remains the
 * RK4 update.
 */

#include "oakfield/integrator.h"
#include "sim_accel.h"

#include <math.h>
#include <string.h>

/**
 * @brief Compare the RK4 real update against an embedded Heun estimate.
 *
 * The estimate uses the initial drift @p k1 and an end-of-step drift @p k_end
 * to form a trapezoidal state, then reports the maximum component difference
 * normalized by `max(abs(rk4), abs(heun), 1)`.
 *
 * @param state Initial real state.
 * @param k1 Drift at the initial state.
 * @param k_end Drift evaluated at the full-step Euler trial.
 * @param rk4_state RK4 candidate state.
 * @param step Timestep used by the stage evaluations.
 * @param count Number of real entries.
 * @return Scaled infinity-norm discrepancy between RK4 and Heun states.
 */
static double rk4_measure_heun_error_real(const double* state,
                                          const double* k1,
                                          const double* k_end,
                                          const double* rk4_state,
                                          double        step,
                                          size_t        count) {
    double half = 0.5 * step;
    double max0 = 0.0;
    double max1 = 0.0;
    double max2 = 0.0;
    double max3 = 0.0;
    size_t i    = 0U;

    for (; i + 3U < count; i += 4U) {
        double heun0  = state[i] + half * (k1[i] + k_end[i]);
        double diff0  = fabs(rk4_state[i] - heun0);
        double scale0 = fmax(fmax(fabs(rk4_state[i]), fabs(heun0)), 1.0);
        double heun1  = state[i + 1U] + half * (k1[i + 1U] + k_end[i + 1U]);
        double diff1  = fabs(rk4_state[i + 1U] - heun1);
        double scale1 = fmax(fmax(fabs(rk4_state[i + 1U]), fabs(heun1)), 1.0);
        double heun2  = state[i + 2U] + half * (k1[i + 2U] + k_end[i + 2U]);
        double diff2  = fabs(rk4_state[i + 2U] - heun2);
        double scale2 = fmax(fmax(fabs(rk4_state[i + 2U]), fabs(heun2)), 1.0);
        double heun3  = state[i + 3U] + half * (k1[i + 3U] + k_end[i + 3U]);
        double diff3  = fabs(rk4_state[i + 3U] - heun3);
        double scale3 = fmax(fmax(fabs(rk4_state[i + 3U]), fabs(heun3)), 1.0);

        max0 = fmax(max0, diff0 / scale0);
        max1 = fmax(max1, diff1 / scale1);
        max2 = fmax(max2, diff2 / scale2);
        max3 = fmax(max3, diff3 / scale3);
    }

    for (; i < count; ++i) {
        double heun  = state[i] + half * (k1[i] + k_end[i]);
        double diff  = fabs(rk4_state[i] - heun);
        double scale = fmax(fmax(fabs(rk4_state[i]), fabs(heun)), 1.0);
        max0         = fmax(max0, diff / scale);
    }

    return fmax(fmax(max0, max1), fmax(max2, max3));
}

/**
 * @brief Compare the RK4 complex update against an embedded Heun estimate.
 *
 * Complex differences use squared magnitude and are normalized by
 * `max(|rk4|^2, |heun|^2, 1)` before the mean square root is returned.
 *
 * @param state Initial complex state.
 * @param k1 Drift at the initial state.
 * @param k_end Drift evaluated at the full-step Euler trial.
 * @param rk4_state RK4 candidate state.
 * @param step Timestep used by the stage evaluations.
 * @param count Number of complex entries.
 * @return Scaled RMS discrepancy between RK4 and Heun states.
 */
static double rk4_measure_heun_error_complex(const SimComplexDouble* state,
                                             const SimComplexDouble* k1,
                                             const SimComplexDouble* k_end,
                                             const SimComplexDouble* rk4_state,
                                             double                  step,
                                             size_t                  count) {
    double half = 0.5 * step;
    double sum0 = 0.0;
    double sum1 = 0.0;
    size_t i    = 0U;

    for (; i + 1U < count; i += 2U) {
        double heun0_re = state[i].re + half * (k1[i].re + k_end[i].re);
        double heun0_im = state[i].im + half * (k1[i].im + k_end[i].im);
        double dr0      = rk4_state[i].re - heun0_re;
        double di0      = rk4_state[i].im - heun0_im;
        double diff20   = dr0 * dr0 + di0 * di0;
        double an20 =
            rk4_state[i].re * rk4_state[i].re + rk4_state[i].im * rk4_state[i].im;
        double bn20 = heun0_re * heun0_re + heun0_im * heun0_im;
        double scale20 = fmax(fmax(an20, bn20), 1.0);

        double heun1_re = state[i + 1U].re + half * (k1[i + 1U].re + k_end[i + 1U].re);
        double heun1_im = state[i + 1U].im + half * (k1[i + 1U].im + k_end[i + 1U].im);
        double dr1      = rk4_state[i + 1U].re - heun1_re;
        double di1      = rk4_state[i + 1U].im - heun1_im;
        double diff21   = dr1 * dr1 + di1 * di1;
        double an21 = rk4_state[i + 1U].re * rk4_state[i + 1U].re +
                      rk4_state[i + 1U].im * rk4_state[i + 1U].im;
        double bn21    = heun1_re * heun1_re + heun1_im * heun1_im;
        double scale21 = fmax(fmax(an21, bn21), 1.0);

        sum0 += diff20 / scale20;
        sum1 += diff21 / scale21;
    }

    for (; i < count; ++i) {
        double heun_re = state[i].re + half * (k1[i].re + k_end[i].re);
        double heun_im = state[i].im + half * (k1[i].im + k_end[i].im);
        double dr      = rk4_state[i].re - heun_re;
        double di      = rk4_state[i].im - heun_im;
        double diff2   = dr * dr + di * di;
        double an2     = rk4_state[i].re * rk4_state[i].re + rk4_state[i].im * rk4_state[i].im;
        double bn2     = heun_re * heun_re + heun_im * heun_im;
        double scale2  = fmax(fmax(an2, bn2), 1.0);
        sum0 += diff2 / scale2;
    }

    return sqrt((sum0 + sum1) / (double) count);
}

/**
 * @brief Advance one field with the classical RK4 tableau.
 *
 * Adaptive mode retries with smaller timesteps when the embedded Heun estimate
 * exceeds tolerance. The function records the final step, error norm, and retry
 * counters on the integrator; stochastic increments are applied after the
 * deterministic RK4 candidate is copied back to the field.
 *
 * @param integrator Configured RK4 integrator.
 * @param field Field updated in place.
 * @param dt Requested timestep, or nonpositive to use the stored suggestion.
 */
static void integrator_step(Integrator *integrator, Field *field, double dt)
{
    const unsigned int max_attempts = 8U;
    unsigned int attempt;
    size_t count;
    double step;
    double error_norm = 0.0;
    bool accepted = false;

    if (!integrator || !field || !integrator->drift)
        return;

    bool is_complex = sim_field_domain_is_complex(field);
    integrator->is_complex = is_complex;

    count = integrator_state_length(field);
    if (count == 0U)
        return;

    if (integrator_ensure_workspace(integrator, 5U, count) != SIM_RESULT_OK)
        return;

    step = (dt > 0.0) ? dt : integrator_next_step(integrator);
    step = integrator_clamp_dt(integrator, step);
    integrator->current_dt = step;

    /* ------------------- REAL BRANCH ------------------- */
    if (!is_complex)
    {
        double *state = (double *)sim_field_data(field);
        if (!state)
            return;

        double *k1 = integrator_buffer(integrator, 0U);
        double *k2 = integrator_buffer(integrator, 1U);
        double *k3 = integrator_buffer(integrator, 2U);
        double *k4 = integrator_buffer(integrator, 3U);
        double *temp = integrator_buffer(integrator, 4U);

        for (attempt = 0U; attempt < max_attempts; ++attempt)
        {
            double sixth = (double)step / 6.0;
            double half = (double)step * 0.5;
            const double* weighted_sources[5];
            double        weighted_coeffs[5];

            if (integrator->drift(integrator, field, state, k1, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = half;
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, temp, count);
            if (integrator->drift(integrator, field, temp, k2, count) != SIM_RESULT_OK)
                return;

            weighted_sources[1] = k2;
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, temp, count);
            if (integrator->drift(integrator, field, temp, k3, count) != SIM_RESULT_OK)
                return;

            weighted_sources[1] = k3;
            weighted_coeffs[1] = step;
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, temp, count);
            if (integrator->drift(integrator, field, temp, k4, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_sources[2] = k2;
            weighted_sources[3] = k3;
            weighted_sources[4] = k4;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = sixth;
            weighted_coeffs[2] = 2.0 * sixth;
            weighted_coeffs[3] = 2.0 * sixth;
            weighted_coeffs[4] = sixth;
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 5U, temp, count);

            if (integrator->adaptive)
            {
                /* Full-step Euler trial reused for the embedded Heun estimate. */
                weighted_sources[0] = state;
                weighted_sources[1] = k1;
                weighted_coeffs[0] = 1.0;
                weighted_coeffs[1] = step;
                sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, k2, count);
                if (integrator->drift(integrator, field, k2, k4, count) != SIM_RESULT_OK)
                    return;

                error_norm = rk4_measure_heun_error_real(state, k1, k4, temp, step, count);
            }
            else
                error_norm = 0.0;

            if (!integrator->adaptive || error_norm <= integrator->tolerance || step <= integrator->min_dt)
            {
                accepted = true;
                break;
            }

            step = integrator_reject_dt(integrator, step, error_norm, 4.0);
            integrator->current_dt = step;
        }

        memcpy(state, temp, count * sizeof(double));
        integrator_apply_stochastic(integrator, field, k2, count, step);

        integrator->last_step = step;
        integrator->last_error = error_norm;
        integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
        integrator->last_rejection_count =
            integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
        integrator->current_dt = (integrator->adaptive && accepted)
                                     ? integrator_suggest_dt(integrator, step, error_norm, 4.0)
                                     : integrator_clamp_dt(integrator, step);
        return;
    }

    /* ------------------- COMPLEX BRANCH ------------------- */
    SimComplexDouble *state = sim_field_complex_data(field);
    if (!state)
        return;

    SimComplexDouble *k1 = integrator_buffer_complex(integrator, 0U);
    SimComplexDouble *k2 = integrator_buffer_complex(integrator, 1U);
    SimComplexDouble *k3 = integrator_buffer_complex(integrator, 2U);
    SimComplexDouble *k4 = integrator_buffer_complex(integrator, 3U);
    SimComplexDouble *temp = integrator_buffer_complex(integrator, 4U);

    for (attempt = 0U; attempt < max_attempts; ++attempt)
    {
        double sixth = (double)step / 6.0;
        double half = (double)step * 0.5;
        const double* weighted_sources[5];
        double        weighted_coeffs[5];

        if (integrator->drift(integrator, field, (double *)state, (double *)k1, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = half;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 2U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k2, count) != SIM_RESULT_OK)
            return;

        weighted_sources[1] = (const double*) k2;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 2U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k3, count) != SIM_RESULT_OK)
            return;

        weighted_sources[1] = (const double*) k3;
        weighted_coeffs[1] = step;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 2U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k4, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_sources[2] = (const double*) k2;
        weighted_sources[3] = (const double*) k3;
        weighted_sources[4] = (const double*) k4;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = sixth;
        weighted_coeffs[2] = 2.0 * sixth;
        weighted_coeffs[3] = 2.0 * sixth;
        weighted_coeffs[4] = sixth;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 5U, (double*) temp, count * 2U);

        if (integrator->adaptive)
        {
            weighted_sources[0] = (const double*) state;
            weighted_sources[1] = (const double*) k1;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = step;
            sim_accel_weighted_sum_real(
                weighted_sources, weighted_coeffs, 2U, (double*) k2, count * 2U);
            if (integrator->drift(integrator, field, (double *)k2, (double *)k4, count) != SIM_RESULT_OK)
                return;

            error_norm = rk4_measure_heun_error_complex(state, k1, k4, temp, step, count);
        }
        else
            error_norm = 0.0;

        if (!integrator->adaptive || error_norm <= integrator->tolerance || step <= integrator->min_dt)
        {
            accepted = true;
            break;
        }

        step = integrator_reject_dt(integrator, step, error_norm, 4.0);
        integrator->current_dt = step;
    }

    memcpy(state, temp, count * sizeof(SimComplexDouble));
    integrator_apply_stochastic(integrator, field, (double *)k2, count, step);

    integrator->last_step = step;
    integrator->last_error = error_norm;
    integrator->last_attempt_count   = accepted ? (attempt + 1U) : max_attempts;
    integrator->last_rejection_count =
        integrator->adaptive ? (integrator->last_attempt_count - 1U) : 0U;
    integrator->current_dt = (integrator->adaptive && accepted)
                                 ? integrator_suggest_dt(integrator, step, error_norm, 4.0)
                                 : integrator_clamp_dt(integrator, step);
}

/**
 * @brief Create a classical RK4 integrator.
 *
 * @param config Integrator configuration; drift callback is required.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK or an error from integrator_configure().
 */
SimResult integrator_rk4_create(const IntegratorConfig *config, Integrator *out)
{
    return integrator_configure(out, "rk4", integrator_step, config);
}
