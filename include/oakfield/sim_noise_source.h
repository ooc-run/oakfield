/**
 * @file sim_noise_source.h
 * @brief Deterministic RNG streams and reusable noise update helpers.
 *
 * This header exposes the shared stochastic primitives used by stimulus and
 * noise operators. Streams are explicitly seeded, output buffers are
 * caller-owned, and helper functions clamp non-finite parameters to deterministic
 * fallbacks rather than propagating NaN through update loops.
 */
#ifndef OAKFIELD_SIM_NOISE_SOURCE_H
#define OAKFIELD_SIM_NOISE_SOURCE_H

#include "field.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIM_NOISE_SOURCE_STREAM_STIMULUS_WHITE 0x85EBCA77C2B2AE63ULL
#define SIM_NOISE_SOURCE_STREAM_STOCHASTIC 0x000000000BAD5EEDULL
#define SIM_NOISE_SOURCE_STREAM_ORNSTEIN 0xD8D12A6C9B4E41B7ULL

/**
 * @brief Mutable deterministic RNG state with cached normal sample.
 *
 * The state/inc pair implements the underlying uniform stream. The spare fields
 * cache the second Box-Muller output so repeated scalar-normal calls consume the
 * same sequence as paired normal generation.
 */
typedef struct SimNoiseSourceRng {
    uint64_t state;        /**< Current generator state. */
    uint64_t inc;          /**< Odd stream increment derived from the seed sequence. */
    bool normal_has_spare; /**< True when normal_spare holds a cached sample. */
    double normal_spare;   /**< Cached standard-normal sample. */
} SimNoiseSourceRng;

/**
 * @brief Seed a deterministic RNG stream.
 *
 * @param rng RNG state to initialize; NULL is ignored.
 * @param initstate Initial generator state value.
 * @param initseq Stream selector; converted to an odd increment internally.
 */
void sim_noise_source_seed(SimNoiseSourceRng *rng, uint64_t initstate, uint64_t initseq);

/**
 * @brief Draw a uniform variate in [0, 1).
 *
 * @param rng RNG state to advance.
 * @return Uniform double in [0, 1), or 0.0 when @p rng is NULL.
 */
double sim_noise_source_uniform(SimNoiseSourceRng *rng);

/**
 * @brief Draw a pair of independent standard-normal variates.
 *
 * Uses Box-Muller sampling with the first uniform clamped away from zero to
 * avoid log(0). Output pointers may be NULL.
 *
 * @param rng RNG state to advance; NULL writes zeroes to non-NULL outputs.
 * @param[out] n0 Receives the first N(0, 1) sample when non-NULL.
 * @param[out] n1 Receives the second N(0, 1) sample when non-NULL.
 */
void sim_noise_source_normal_pair(SimNoiseSourceRng *rng, double *n0, double *n1);

/**
 * @brief Draw one standard-normal variate.
 *
 * @param rng RNG state to advance.
 * @return A cached or newly sampled N(0, 1) value, or 0.0 when @p rng is NULL.
 */
double sim_noise_source_normal(SimNoiseSourceRng *rng);

/**
 * @brief Compute the finite gain used by Gaussian white-noise updates.
 *
 * Non-finite sigma or dt values are treated as zero. When @p scale_by_dt is
 * true, the gain is multiplied by sqrt(abs(dt)).
 *
 * @param sigma Noise amplitude.
 * @param dt Simulation timestep.
 * @param scale_by_dt Whether to apply sqrt(abs(dt)) scaling.
 * @return Finite gain value, or 0.0 if the computed gain is non-finite.
 */
double sim_noise_source_gaussian_gain(double sigma, double dt, bool scale_by_dt);

/**
 * @brief Ensure reusable temporal-noise state storage has at least @p count entries.
 *
 * The selected real or complex state array is allocated or grown with new slots
 * zero-initialized. Passing count zero frees both arrays and resets capacity.
 *
 * @param[in,out] real_state Pointer to caller-owned real state storage.
 * @param[in,out] complex_state Pointer to caller-owned complex state storage.
 * @param[in,out] capacity Current and resulting number of allocated entries.
 * @param count Required element count; zero releases both arrays.
 * @param is_complex Selects which state array must be grown for nonzero count.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_noise_source_ensure_capacity(double **real_state, SimComplexDouble **complex_state,
                                           size_t *capacity, size_t count, bool is_complex);

/**
 * @brief Add real-valued white Gaussian noise to an array in place.
 *
 * Non-finite mean and gain are treated as zero. The destination is unchanged
 * when dst, rng, or count is empty.
 *
 * @param[in,out] dst Real destination values to increment.
 * @param count Number of entries in @p dst.
 * @param mean Deterministic offset added to each element.
 * @param gain Multiplier for independent N(0, 1) samples.
 * @param rng RNG state to advance.
 */
void sim_noise_source_apply_white_real(double *dst, size_t count, double mean, double gain,
                                       SimNoiseSourceRng *rng);

/**
 * @brief Add complex white Gaussian noise to an array in place.
 *
 * Real and imaginary channels draw independent standard-normal samples. Non-
 * finite means or gains are treated as zero.
 *
 * @param[in,out] dst Complex destination values to increment.
 * @param count Number of entries in @p dst.
 * @param mean_re Real-channel deterministic offset.
 * @param mean_im Imaginary-channel deterministic offset.
 * @param gain_re Real-channel normal multiplier.
 * @param gain_im Imaginary-channel normal multiplier.
 * @param rng RNG state to advance.
 */
void sim_noise_source_apply_white_complex(SimComplexDouble *dst, size_t count, double mean_re,
                                          double mean_im, double gain_re, double gain_im,
                                          SimNoiseSourceRng *rng);

/**
 * @brief Derive decay and innovation scale for temporal correlated noise.
 *
 * The update convention is state = decay * state + variance * N(0, 1). Invalid
 * dt disables innovation, invalid tau is clamped to a tiny positive value, and
 * alpha is bounded below by 0.1.
 *
 * @param sigma Innovation amplitude before temporal normalization.
 * @param tau Positive correlation timescale.
 * @param alpha Positive stretch exponent, clamped to at least 0.1.
 * @param dt Simulation timestep.
 * @param[out] out_decay Receives decay factor when non-NULL.
 * @param[out] out_variance Receives normal multiplier when non-NULL.
 */
void sim_noise_source_temporal_params(double sigma, double tau, double alpha, double dt,
                                      double *out_decay, double *out_variance);

/**
 * @brief Advance real temporal-noise state and add it to a destination array.
 *
 * @param[in,out] dst Destination values incremented by output_scale * state.
 * @param[in,out] noise_state Persistent per-element real noise state.
 * @param count Number of entries in both arrays.
 * @param output_scale Scale applied to the updated state before adding to dst.
 * @param decay Autoregressive decay factor.
 * @param variance Multiplier for fresh standard-normal innovation.
 * @param rng RNG state to advance.
 */
void sim_noise_source_apply_temporal_real(double *dst, double *noise_state, size_t count,
                                          double output_scale, double decay, double variance,
                                          SimNoiseSourceRng *rng);

/**
 * @brief Advance complex temporal-noise state and add it to a destination array.
 *
 * Real and imaginary state channels use independent normal innovations.
 *
 * @param[in,out] dst Destination values incremented by output_scale * state.
 * @param[in,out] noise_state Persistent per-element complex noise state.
 * @param count Number of entries in both arrays.
 * @param output_scale Scale applied to updated channels before adding to dst.
 * @param decay Autoregressive decay factor.
 * @param variance Multiplier for fresh standard-normal innovations.
 * @param rng RNG state to advance.
 */
void sim_noise_source_apply_temporal_complex(SimComplexDouble *dst, SimComplexDouble *noise_state,
                                             size_t count, double output_scale, double decay,
                                             double variance, SimNoiseSourceRng *rng);

/**
 * @brief Apply a real mean-reverting update in place.
 *
 * Each element becomes mean + decay * (x - mean) plus optional Gaussian
 * innovation. Non-finite parameters fall back to mean=0, decay=1, variance=0.
 *
 * @param[in,out] dst Real values to update in place.
 * @param count Number of entries in @p dst.
 * @param mean Reversion target.
 * @param decay Distance-from-mean multiplier.
 * @param variance Optional standard-normal innovation multiplier.
 * @param rng RNG state to advance when variance is nonzero.
 */
void sim_noise_source_apply_mean_reverting_real(double *dst, size_t count, double mean,
                                                double decay, double variance,
                                                SimNoiseSourceRng *rng);

/**
 * @brief Apply a complex mean-reverting update in place.
 *
 * The same real mean is applied to both real and imaginary channels, with
 * independent Gaussian innovations when variance is nonzero.
 *
 * @param[in,out] dst Complex values to update in place.
 * @param count Number of entries in @p dst.
 * @param mean Reversion target for both channels.
 * @param decay Distance-from-mean multiplier.
 * @param variance Optional standard-normal innovation multiplier.
 * @param rng RNG state to advance when variance is nonzero.
 */
void sim_noise_source_apply_mean_reverting_complex(SimComplexDouble *dst, size_t count, double mean,
                                                   double decay, double variance,
                                                   SimNoiseSourceRng *rng);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_NOISE_SOURCE_H */
