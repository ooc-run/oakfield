/**
 * @file rkf45.c
 * @brief Runge-Kutta-Fehlberg 4(5) integrator with forced error control.
 *
 * The stepper evaluates the Fehlberg six-stage tableau and writes the
 * fifth-order state while using the fourth/fifth difference as the adaptive
 * error estimate. Real paths report a scaled infinity norm; complex paths use a
 * scaled RMS magnitude. The factory forces `adaptive = true` because this
 * method's contract is tied to embedded error control.
 */

#include "oakfield/integrator.h"
#include "sim_accel.h"

#include <math.h>
#include <string.h>

/**
 * @brief Finalize real RKF45 output and compute the embedded error estimate.
 *
 * The helper writes the fifth-order state to @p out_state and compares it to
 * the fourth-order Fehlberg companion using component scaling
 * `max(abs(fifth), abs(fourth), 1)`.
 *
 * @param state Initial real state.
 * @param k1 First stage derivative.
 * @param k3 Third stage derivative.
 * @param k4 Fourth stage derivative.
 * @param k5 Fifth stage derivative.
 * @param k6 Sixth stage derivative.
 * @param step Timestep used for the tableau.
 * @param[out] out_state Buffer receiving the fifth-order state.
 * @param count Number of real entries.
 * @return Scaled infinity-norm fourth/fifth discrepancy.
 */
static double rkf45_finalize_real(const double* state,
                                  const double* k1,
                                  const double* k3,
                                  const double* k4,
                                  const double* k5,
                                  const double* k6,
                                  double        step,
                                  double*       out_state,
                                  size_t        count) {
    const double c4_k1 = step * (25.0 / 216.0);
    const double c4_k3 = step * (1408.0 / 2565.0);
    const double c4_k4 = step * (2197.0 / 4104.0);
    const double c4_k5 = step * -(1.0 / 5.0);
    const double c5_k1 = step * (16.0 / 135.0);
    const double c5_k3 = step * (6656.0 / 12825.0);
    const double c5_k4 = step * (28561.0 / 56430.0);
    const double c5_k5 = step * -(9.0 / 50.0);
    const double c5_k6 = step * (2.0 / 55.0);
    double       max0  = 0.0;
    double       max1  = 0.0;
    double       max2  = 0.0;
    double       max3  = 0.0;
    size_t       i     = 0U;

    for (; i + 3U < count; i += 4U) {
        double fourth0 = state[i] + c4_k1 * k1[i] + c4_k3 * k3[i] + c4_k4 * k4[i] + c4_k5 * k5[i];
        double fifth0 =
            state[i] + c5_k1 * k1[i] + c5_k3 * k3[i] + c5_k4 * k4[i] + c5_k5 * k5[i] + c5_k6 * k6[i];
        double diff0  = fabs(fifth0 - fourth0);
        double scale0 = fmax(fmax(fabs(fifth0), fabs(fourth0)), 1.0);

        double fourth1 = state[i + 1U] + c4_k1 * k1[i + 1U] + c4_k3 * k3[i + 1U] +
                         c4_k4 * k4[i + 1U] + c4_k5 * k5[i + 1U];
        double fifth1 = state[i + 1U] + c5_k1 * k1[i + 1U] + c5_k3 * k3[i + 1U] +
                        c5_k4 * k4[i + 1U] + c5_k5 * k5[i + 1U] + c5_k6 * k6[i + 1U];
        double diff1  = fabs(fifth1 - fourth1);
        double scale1 = fmax(fmax(fabs(fifth1), fabs(fourth1)), 1.0);

        double fourth2 = state[i + 2U] + c4_k1 * k1[i + 2U] + c4_k3 * k3[i + 2U] +
                         c4_k4 * k4[i + 2U] + c4_k5 * k5[i + 2U];
        double fifth2 = state[i + 2U] + c5_k1 * k1[i + 2U] + c5_k3 * k3[i + 2U] +
                        c5_k4 * k4[i + 2U] + c5_k5 * k5[i + 2U] + c5_k6 * k6[i + 2U];
        double diff2  = fabs(fifth2 - fourth2);
        double scale2 = fmax(fmax(fabs(fifth2), fabs(fourth2)), 1.0);

        double fourth3 = state[i + 3U] + c4_k1 * k1[i + 3U] + c4_k3 * k3[i + 3U] +
                         c4_k4 * k4[i + 3U] + c4_k5 * k5[i + 3U];
        double fifth3 = state[i + 3U] + c5_k1 * k1[i + 3U] + c5_k3 * k3[i + 3U] +
                        c5_k4 * k4[i + 3U] + c5_k5 * k5[i + 3U] + c5_k6 * k6[i + 3U];
        double diff3  = fabs(fifth3 - fourth3);
        double scale3 = fmax(fmax(fabs(fifth3), fabs(fourth3)), 1.0);

        out_state[i]          = fifth0;
        out_state[i + 1U]     = fifth1;
        out_state[i + 2U]     = fifth2;
        out_state[i + 3U]     = fifth3;
        max0                  = fmax(max0, diff0 / scale0);
        max1                  = fmax(max1, diff1 / scale1);
        max2                  = fmax(max2, diff2 / scale2);
        max3                  = fmax(max3, diff3 / scale3);
    }

    for (; i < count; ++i) {
        double fourth = state[i] + c4_k1 * k1[i] + c4_k3 * k3[i] + c4_k4 * k4[i] + c4_k5 * k5[i];
        double fifth =
            state[i] + c5_k1 * k1[i] + c5_k3 * k3[i] + c5_k4 * k4[i] + c5_k5 * k5[i] + c5_k6 * k6[i];
        double diff  = fabs(fifth - fourth);
        double scale = fmax(fmax(fabs(fifth), fabs(fourth)), 1.0);
        out_state[i] = fifth;
        max0         = fmax(max0, diff / scale);
    }

    return fmax(fmax(max0, max1), fmax(max2, max3));
}

/**
 * @brief Finalize complex RKF45 output and compute the embedded error estimate.
 *
 * The helper writes the fifth-order complex state and returns an RMS
 * fourth/fifth discrepancy normalized by squared complex magnitudes.
 *
 * @param state Initial complex state.
 * @param k1 First stage derivative.
 * @param k3 Third stage derivative.
 * @param k4 Fourth stage derivative.
 * @param k5 Fifth stage derivative.
 * @param k6 Sixth stage derivative.
 * @param step Timestep used for the tableau.
 * @param[out] out_state Buffer receiving the fifth-order state.
 * @param count Number of complex entries.
 * @return Scaled RMS fourth/fifth discrepancy.
 */
static double rkf45_finalize_complex(const SimComplexDouble* state,
                                     const SimComplexDouble* k1,
                                     const SimComplexDouble* k3,
                                     const SimComplexDouble* k4,
                                     const SimComplexDouble* k5,
                                     const SimComplexDouble* k6,
                                     double                  step,
                                     SimComplexDouble*       out_state,
                                     size_t                  count) {
    const double c4_k1 = step * (25.0 / 216.0);
    const double c4_k3 = step * (1408.0 / 2565.0);
    const double c4_k4 = step * (2197.0 / 4104.0);
    const double c4_k5 = step * -(1.0 / 5.0);
    const double c5_k1 = step * (16.0 / 135.0);
    const double c5_k3 = step * (6656.0 / 12825.0);
    const double c5_k4 = step * (28561.0 / 56430.0);
    const double c5_k5 = step * -(9.0 / 50.0);
    const double c5_k6 = step * (2.0 / 55.0);
    double       sum0  = 0.0;
    double       sum1  = 0.0;
    size_t       i     = 0U;

    for (; i + 1U < count; i += 2U) {
        double fourth0_re = state[i].re + c4_k1 * k1[i].re + c4_k3 * k3[i].re + c4_k4 * k4[i].re +
                            c4_k5 * k5[i].re;
        double fourth0_im = state[i].im + c4_k1 * k1[i].im + c4_k3 * k3[i].im + c4_k4 * k4[i].im +
                            c4_k5 * k5[i].im;
        double fifth0_re = state[i].re + c5_k1 * k1[i].re + c5_k3 * k3[i].re + c5_k4 * k4[i].re +
                           c5_k5 * k5[i].re + c5_k6 * k6[i].re;
        double fifth0_im = state[i].im + c5_k1 * k1[i].im + c5_k3 * k3[i].im + c5_k4 * k4[i].im +
                           c5_k5 * k5[i].im + c5_k6 * k6[i].im;
        double dr0       = fifth0_re - fourth0_re;
        double di0       = fifth0_im - fourth0_im;
        double diff20    = dr0 * dr0 + di0 * di0;
        double an20      = fifth0_re * fifth0_re + fifth0_im * fifth0_im;
        double bn20      = fourth0_re * fourth0_re + fourth0_im * fourth0_im;
        double scale20   = fmax(fmax(an20, bn20), 1.0);

        double fourth1_re = state[i + 1U].re + c4_k1 * k1[i + 1U].re + c4_k3 * k3[i + 1U].re +
                            c4_k4 * k4[i + 1U].re + c4_k5 * k5[i + 1U].re;
        double fourth1_im = state[i + 1U].im + c4_k1 * k1[i + 1U].im + c4_k3 * k3[i + 1U].im +
                            c4_k4 * k4[i + 1U].im + c4_k5 * k5[i + 1U].im;
        double fifth1_re = state[i + 1U].re + c5_k1 * k1[i + 1U].re + c5_k3 * k3[i + 1U].re +
                           c5_k4 * k4[i + 1U].re + c5_k5 * k5[i + 1U].re + c5_k6 * k6[i + 1U].re;
        double fifth1_im = state[i + 1U].im + c5_k1 * k1[i + 1U].im + c5_k3 * k3[i + 1U].im +
                           c5_k4 * k4[i + 1U].im + c5_k5 * k5[i + 1U].im + c5_k6 * k6[i + 1U].im;
        double dr1       = fifth1_re - fourth1_re;
        double di1       = fifth1_im - fourth1_im;
        double diff21    = dr1 * dr1 + di1 * di1;
        double an21      = fifth1_re * fifth1_re + fifth1_im * fifth1_im;
        double bn21      = fourth1_re * fourth1_re + fourth1_im * fourth1_im;
        double scale21   = fmax(fmax(an21, bn21), 1.0);

        out_state[i].re      = fifth0_re;
        out_state[i].im      = fifth0_im;
        out_state[i + 1U].re = fifth1_re;
        out_state[i + 1U].im = fifth1_im;
        sum0 += diff20 / scale20;
        sum1 += diff21 / scale21;
    }

    for (; i < count; ++i) {
        double fourth_re =
            state[i].re + c4_k1 * k1[i].re + c4_k3 * k3[i].re + c4_k4 * k4[i].re + c4_k5 * k5[i].re;
        double fourth_im =
            state[i].im + c4_k1 * k1[i].im + c4_k3 * k3[i].im + c4_k4 * k4[i].im + c4_k5 * k5[i].im;
        double fifth_re = state[i].re + c5_k1 * k1[i].re + c5_k3 * k3[i].re + c5_k4 * k4[i].re +
                          c5_k5 * k5[i].re + c5_k6 * k6[i].re;
        double fifth_im = state[i].im + c5_k1 * k1[i].im + c5_k3 * k3[i].im + c5_k4 * k4[i].im +
                          c5_k5 * k5[i].im + c5_k6 * k6[i].im;
        double dr       = fifth_re - fourth_re;
        double di       = fifth_im - fourth_im;
        double diff2    = dr * dr + di * di;
        double an2      = fifth_re * fifth_re + fifth_im * fifth_im;
        double bn2      = fourth_re * fourth_re + fourth_im * fourth_im;
        double scale2   = fmax(fmax(an2, bn2), 1.0);
        out_state[i].re = fifth_re;
        out_state[i].im = fifth_im;
        sum0 += diff2 / scale2;
    }

    return sqrt((sum0 + sum1) / (double) count);
}

/**
 * @brief Advance one field with the RKF45 embedded pair.
 *
 * The deterministic candidate is the fifth-order Fehlberg estimate. Timesteps
 * are retried until the embedded error satisfies tolerance, the minimum
 * timestep is reached, or the fixed retry budget is exhausted. Stochastic
 * increments are added after the candidate is copied back to the field.
 *
 * @param integrator Configured RKF45 integrator.
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

    if (integrator_ensure_workspace(integrator, 7U, count) != SIM_RESULT_OK)
        return;

    step = (dt > 0.0) ? dt : integrator_next_step(integrator);
    step = integrator_clamp_dt(integrator, step);
    integrator->current_dt = step;

    if (!is_complex)
    {
        /* ------------------- REAL BRANCH ------------------- */
        double *state = (double *)sim_field_data(field);
        if (!state)
            return;

        double *k1 = integrator_buffer(integrator, 0U);
        double *k2 = integrator_buffer(integrator, 1U);
        double *k3 = integrator_buffer(integrator, 2U);
        double *k4 = integrator_buffer(integrator, 3U);
        double *k5 = integrator_buffer(integrator, 4U);
        double *k6 = integrator_buffer(integrator, 5U);
        double *temp = integrator_buffer(integrator, 6U);

        for (attempt = 0U; attempt < max_attempts; ++attempt)
        {
            double h = (double)step;
            const double* weighted_sources[6];
            double        weighted_coeffs[6];

            if (integrator->drift(integrator, field, state, k1, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = h * (1.0 / 4.0);
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 2U, temp, count);
            if (integrator->drift(integrator, field, temp, k2, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_sources[2] = k2;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = h * (3.0 / 32.0);
            weighted_coeffs[2] = h * (9.0 / 32.0);
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 3U, temp, count);
            if (integrator->drift(integrator, field, temp, k3, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_sources[2] = k2;
            weighted_sources[3] = k3;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = h * (1932.0 / 2197.0);
            weighted_coeffs[2] = h * -(7200.0 / 2197.0);
            weighted_coeffs[3] = h * (7296.0 / 2197.0);
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 4U, temp, count);
            if (integrator->drift(integrator, field, temp, k4, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_sources[2] = k2;
            weighted_sources[3] = k3;
            weighted_sources[4] = k4;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = h * (439.0 / 216.0);
            weighted_coeffs[2] = h * -8.0;
            weighted_coeffs[3] = h * (3680.0 / 513.0);
            weighted_coeffs[4] = h * -(845.0 / 4104.0);
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 5U, temp, count);
            if (integrator->drift(integrator, field, temp, k5, count) != SIM_RESULT_OK)
                return;

            weighted_sources[0] = state;
            weighted_sources[1] = k1;
            weighted_sources[2] = k2;
            weighted_sources[3] = k3;
            weighted_sources[4] = k4;
            weighted_sources[5] = k5;
            weighted_coeffs[0] = 1.0;
            weighted_coeffs[1] = h * -(8.0 / 27.0);
            weighted_coeffs[2] = h * 2.0;
            weighted_coeffs[3] = h * -(3544.0 / 2565.0);
            weighted_coeffs[4] = h * (1859.0 / 4104.0);
            weighted_coeffs[5] = h * -(11.0 / 40.0);
            sim_accel_weighted_sum_real(weighted_sources, weighted_coeffs, 6U, temp, count);
            if (integrator->drift(integrator, field, temp, k6, count) != SIM_RESULT_OK)
                return;

            error_norm = integrator->adaptive
                             ? rkf45_finalize_real(state, k1, k3, k4, k5, k6, h, temp, count)
                             : 0.0;
            if (!integrator->adaptive) {
                rkf45_finalize_real(state, k1, k3, k4, k5, k6, h, temp, count);
            }
            if (!integrator->adaptive || error_norm <= integrator->tolerance ||
                step <= integrator->min_dt)
            {
                accepted = true;
                break;
            }

            step = integrator_reject_dt(integrator, step, error_norm, 4.0);
            integrator->current_dt = step;
        }

        memcpy(state, temp, count * sizeof(double));
        integrator_apply_stochastic(integrator, field, k6, count, step);

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
    SimComplexDouble *k5 = integrator_buffer_complex(integrator, 4U);
    SimComplexDouble *k6 = integrator_buffer_complex(integrator, 5U);
    SimComplexDouble *temp = integrator_buffer_complex(integrator, 6U);

    for (attempt = 0U; attempt < max_attempts; ++attempt)
    {
        double h = (double)step;
        const double* weighted_sources[6];
        double        weighted_coeffs[6];

        if (integrator->drift(integrator, field, (double *)state, (double *)k1, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = h * 0.25;
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 2U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k2, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_sources[2] = (const double*) k2;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = h * (3.0 / 32.0);
        weighted_coeffs[2] = h * (9.0 / 32.0);
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 3U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k3, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_sources[2] = (const double*) k2;
        weighted_sources[3] = (const double*) k3;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = h * (1932.0 / 2197.0);
        weighted_coeffs[2] = h * -(7200.0 / 2197.0);
        weighted_coeffs[3] = h * (7296.0 / 2197.0);
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 4U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k4, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_sources[2] = (const double*) k2;
        weighted_sources[3] = (const double*) k3;
        weighted_sources[4] = (const double*) k4;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = h * (439.0 / 216.0);
        weighted_coeffs[2] = h * -8.0;
        weighted_coeffs[3] = h * (3680.0 / 513.0);
        weighted_coeffs[4] = h * -(845.0 / 4104.0);
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 5U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k5, count) != SIM_RESULT_OK)
            return;

        weighted_sources[0] = (const double*) state;
        weighted_sources[1] = (const double*) k1;
        weighted_sources[2] = (const double*) k2;
        weighted_sources[3] = (const double*) k3;
        weighted_sources[4] = (const double*) k4;
        weighted_sources[5] = (const double*) k5;
        weighted_coeffs[0] = 1.0;
        weighted_coeffs[1] = h * -(8.0 / 27.0);
        weighted_coeffs[2] = h * 2.0;
        weighted_coeffs[3] = h * -(3544.0 / 2565.0);
        weighted_coeffs[4] = h * (1859.0 / 4104.0);
        weighted_coeffs[5] = h * -(11.0 / 40.0);
        sim_accel_weighted_sum_real(
            weighted_sources, weighted_coeffs, 6U, (double*) temp, count * 2U);
        if (integrator->drift(integrator, field, (double *)temp, (double *)k6, count) != SIM_RESULT_OK)
            return;

        error_norm = integrator->adaptive
                         ? rkf45_finalize_complex(state, k1, k3, k4, k5, k6, h, temp, count)
                         : 0.0;
        if (!integrator->adaptive) {
            rkf45_finalize_complex(state, k1, k3, k4, k5, k6, h, temp, count);
        }
        if (!integrator->adaptive || error_norm <= integrator->tolerance ||
            step <= integrator->min_dt)
        {
            accepted = true;
            break;
        }
        step = integrator_reject_dt(integrator, step, error_norm, 4.0);
        integrator->current_dt = step;
    }

    memcpy(state, temp, count * sizeof(SimComplexDouble));
    integrator_apply_stochastic(integrator, field, (double *)k6, count, step);

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
 * @brief Create an RKF45 integrator with adaptive stepping forced on.
 *
 * @param config Integrator configuration; drift callback is required.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or an error from
 * integrator_configure().
 */
SimResult integrator_rkf45_create(const IntegratorConfig *config, Integrator *out)
{
    if (out == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* RKF45 is only well-behaved with error control; force adaptive on even if the
     * caller asked for a fixed step. Other config fields are honored. */
    if (config != NULL)
    {
        IntegratorConfig sanitized = *config;
        sanitized.adaptive = true;
        return integrator_configure(out, "rkf45", integrator_step, &sanitized);
    }

    return integrator_configure(out, "rkf45", integrator_step, NULL);
}
