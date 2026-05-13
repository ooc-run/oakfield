/**
 * @file sim_noise_source.c
 * @brief Deterministic random streams and noise update kernels.
 *
 * Noise sources use a small PCG-style generator, Box-Muller normal sampling,
 * and finite-checked gain parameters to provide reproducible white, temporal,
 * and mean-reverting updates for real and SimComplexDouble fields. The update
 * helpers mutate caller-owned destination/state arrays in place.
 */
#include "oakfield/sim_noise_source.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define SIM_NOISE_SOURCE_EPS 1.0e-12

static uint32_t sim_noise_source_random_u32(SimNoiseSourceRng* rng) {
    uint64_t old        = rng->state;
    rng->state          = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t) (old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

void sim_noise_source_seed(SimNoiseSourceRng* rng, uint64_t initstate, uint64_t initseq) {
    if (rng == NULL) {
        return;
    }

    rng->state            = 0U;
    rng->inc              = (initseq << 1u) | 1u;
    rng->normal_has_spare = false;
    rng->normal_spare     = 0.0;
    (void) sim_noise_source_random_u32(rng);
    rng->state += initstate;
    (void) sim_noise_source_random_u32(rng);
}

double sim_noise_source_uniform(SimNoiseSourceRng* rng) {
    if (rng == NULL) {
        return 0.0;
    }
    return ldexp(sim_noise_source_random_u32(rng), -32);
}

void sim_noise_source_normal_pair(SimNoiseSourceRng* rng, double* n0, double* n1) {
    double u1;
    double u2;
    double mag;
    double angle;
    double s = 0.0;
    double c = 0.0;

    if (rng == NULL) {
        if (n0 != NULL) {
            *n0 = 0.0;
        }
        if (n1 != NULL) {
            *n1 = 0.0;
        }
        return;
    }

    u1 = sim_noise_source_uniform(rng);
    u2 = sim_noise_source_uniform(rng);
    if (u1 <= SIM_NOISE_SOURCE_EPS) {
        u1 = SIM_NOISE_SOURCE_EPS;
    }

    mag   = sqrt(-2.0 * log(u1));
    angle = 2.0 * M_PI * u2;
#if defined(__APPLE__)
    __sincos(angle, &s, &c);
#elif defined(__clang__) || defined(__GNUC__)
    sincos(angle, &s, &c);
#else
    s = sin(angle);
    c = cos(angle);
#endif

    if (n0 != NULL) {
        *n0 = mag * c;
    }
    if (n1 != NULL) {
        *n1 = mag * s;
    }
}

double sim_noise_source_normal(SimNoiseSourceRng* rng) {
    double n0 = 0.0;
    double n1 = 0.0;

    if (rng == NULL) {
        return 0.0;
    }
    if (rng->normal_has_spare) {
        rng->normal_has_spare = false;
        return rng->normal_spare;
    }

    sim_noise_source_normal_pair(rng, &n0, &n1);
    rng->normal_spare     = n1;
    rng->normal_has_spare = true;
    return n0;
}

double sim_noise_source_gaussian_gain(double sigma, double dt, bool scale_by_dt) {
    double gain;

    if (!isfinite(sigma)) {
        sigma = 0.0;
    }
    if (!isfinite(dt)) {
        dt = 0.0;
    }

    gain = sigma;
    if (scale_by_dt) {
        gain *= sqrt(fabs(dt));
    }
    return isfinite(gain) ? gain : 0.0;
}

SimResult sim_noise_source_ensure_capacity(double**           real_state,
                                           SimComplexDouble** complex_state,
                                           size_t*            capacity,
                                           size_t             count,
                                           bool               is_complex) {
    size_t old_capacity;

    if (real_state == NULL || complex_state == NULL || capacity == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (count == 0U) {
        free(*real_state);
        free(*complex_state);
        *real_state   = NULL;
        *complex_state = NULL;
        *capacity     = 0U;
        return SIM_RESULT_OK;
    }

    old_capacity = *capacity;

    if (is_complex) {
        if (*complex_state == NULL || old_capacity < count) {
            size_t start = (*complex_state != NULL) ? old_capacity : 0U;
            SimComplexDouble* resized =
                (SimComplexDouble*) realloc(*complex_state, count * sizeof(SimComplexDouble));
            if (resized == NULL) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            for (size_t i = start; i < count; ++i) {
                resized[i].re = 0.0;
                resized[i].im = 0.0;
            }
            *complex_state = resized;
        }
    } else {
        if (*real_state == NULL || old_capacity < count) {
            size_t start = (*real_state != NULL) ? old_capacity : 0U;
            double* resized = (double*) realloc(*real_state, count * sizeof(double));
            if (resized == NULL) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            for (size_t i = start; i < count; ++i) {
                resized[i] = 0.0;
            }
            *real_state = resized;
        }
    }

    if (*capacity < count) {
        *capacity = count;
    }
    return SIM_RESULT_OK;
}

void sim_noise_source_apply_white_real(double*            dst,
                                       size_t             count,
                                       double             mean,
                                       double             gain,
                                       SimNoiseSourceRng* rng) {
    if (dst == NULL || rng == NULL || count == 0U) {
        return;
    }
    if (!isfinite(mean)) {
        mean = 0.0;
    }
    if (!isfinite(gain)) {
        gain = 0.0;
    }

    for (size_t i = 0U; i < count; ++i) {
        dst[i] += mean + gain * sim_noise_source_normal(rng);
    }
}

void sim_noise_source_apply_white_complex(SimComplexDouble* dst,
                                          size_t            count,
                                          double            mean_re,
                                          double            mean_im,
                                          double            gain_re,
                                          double            gain_im,
                                          SimNoiseSourceRng* rng) {
    if (dst == NULL || rng == NULL || count == 0U) {
        return;
    }
    if (!isfinite(mean_re)) {
        mean_re = 0.0;
    }
    if (!isfinite(mean_im)) {
        mean_im = 0.0;
    }
    if (!isfinite(gain_re)) {
        gain_re = 0.0;
    }
    if (!isfinite(gain_im)) {
        gain_im = 0.0;
    }

    for (size_t i = 0U; i < count; ++i) {
        dst[i].re += mean_re + gain_re * sim_noise_source_normal(rng);
        dst[i].im += mean_im + gain_im * sim_noise_source_normal(rng);
    }
}

void sim_noise_source_temporal_params(double  sigma,
                                      double  tau,
                                      double  alpha,
                                      double  dt,
                                      double* out_decay,
                                      double* out_variance) {
    double decay    = 1.0;
    double variance = 0.0;

    if (!isfinite(sigma)) {
        sigma = 0.0;
    }
    if (!isfinite(dt) || dt <= 0.0) {
        dt = 0.0;
    }
    if (!isfinite(tau) || tau <= 0.0) {
        tau = 1.0e-12;
    }
    if (!isfinite(alpha) || alpha < 0.1) {
        alpha = 0.1;
    }

    if (dt > 0.0) {
        decay    = exp(-pow(dt / tau, alpha));
        variance = sigma * sqrt(fmax(0.0, 1.0 - decay * decay));
    }

    if (out_decay != NULL) {
        *out_decay = decay;
    }
    if (out_variance != NULL) {
        *out_variance = variance;
    }
}

void sim_noise_source_apply_temporal_real(double*            dst,
                                          double*            noise_state,
                                          size_t             count,
                                          double             output_scale,
                                          double             decay,
                                          double             variance,
                                          SimNoiseSourceRng* rng) {
    if (dst == NULL || noise_state == NULL || rng == NULL || count == 0U) {
        return;
    }
    if (!isfinite(output_scale)) {
        output_scale = 0.0;
    }
    if (!isfinite(decay)) {
        decay = 1.0;
    }
    if (!isfinite(variance)) {
        variance = 0.0;
    }

    for (size_t i = 0U; i < count; ++i) {
        double next = decay * noise_state[i] + variance * sim_noise_source_normal(rng);
        noise_state[i] = next;
        dst[i] += output_scale * next;
    }
}

void sim_noise_source_apply_temporal_complex(SimComplexDouble* dst,
                                             SimComplexDouble* noise_state,
                                             size_t            count,
                                             double            output_scale,
                                             double            decay,
                                             double            variance,
                                             SimNoiseSourceRng* rng) {
    if (dst == NULL || noise_state == NULL || rng == NULL || count == 0U) {
        return;
    }
    if (!isfinite(output_scale)) {
        output_scale = 0.0;
    }
    if (!isfinite(decay)) {
        decay = 1.0;
    }
    if (!isfinite(variance)) {
        variance = 0.0;
    }

    for (size_t i = 0U; i < count; ++i) {
        double next_re = decay * noise_state[i].re + variance * sim_noise_source_normal(rng);
        double next_im = decay * noise_state[i].im + variance * sim_noise_source_normal(rng);
        noise_state[i].re = next_re;
        noise_state[i].im = next_im;
        dst[i].re += output_scale * next_re;
        dst[i].im += output_scale * next_im;
    }
}

void sim_noise_source_apply_mean_reverting_real(double*            dst,
                                                size_t             count,
                                                double             mean,
                                                double             decay,
                                                double             variance,
                                                SimNoiseSourceRng* rng) {
    if (dst == NULL || rng == NULL || count == 0U) {
        return;
    }
    if (!isfinite(mean)) {
        mean = 0.0;
    }
    if (!isfinite(decay)) {
        decay = 1.0;
    }
    if (!isfinite(variance)) {
        variance = 0.0;
    }

    for (size_t i = 0U; i < count; ++i) {
        double value = mean + decay * (dst[i] - mean);
        if (variance != 0.0) {
            value += variance * sim_noise_source_normal(rng);
        }
        dst[i] = value;
    }
}

void sim_noise_source_apply_mean_reverting_complex(SimComplexDouble* dst,
                                                   size_t            count,
                                                   double            mean,
                                                   double            decay,
                                                   double            variance,
                                                   SimNoiseSourceRng* rng) {
    if (dst == NULL || rng == NULL || count == 0U) {
        return;
    }
    if (!isfinite(mean)) {
        mean = 0.0;
    }
    if (!isfinite(decay)) {
        decay = 1.0;
    }
    if (!isfinite(variance)) {
        variance = 0.0;
    }

    for (size_t i = 0U; i < count; ++i) {
        double next_re = mean + decay * (dst[i].re - mean);
        double next_im = mean + decay * (dst[i].im - mean);
        if (variance != 0.0) {
            next_re += variance * sim_noise_source_normal(rng);
            next_im += variance * sim_noise_source_normal(rng);
        }
        dst[i].re = next_re;
        dst[i].im = next_im;
    }
}
