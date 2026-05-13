/**
 * @file random_fourier.h
 * @brief Random Fourier feature stimulus fields.
 *
 * Synthesizes spatially structured fields via random Fourier features:
 *   f(x, t) = sum_k a_k cos(ω_k x - Ω t + φ_k)
 *
 * For complex fields, a complex-valued sum is written. For real fields,
 * the real part is written, yielding Hermitian-symmetric spectra.
 */
#ifndef OAKFIELD_STIMULUS_RANDOM_FOURIER_H
#define OAKFIELD_STIMULUS_RANDOM_FOURIER_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for seeded random Fourier feature stimulus fields.
 */
typedef struct SimStimulusRandomFourierConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Overall amplitude scale. */
    double k_min;                 /**< Minimum spatial wavenumber (rad / unit). */
    double k_max;                 /**< Maximum spatial wavenumber (rad / unit). */
    double kx;                    /**< Optional base wavevector X component (rad / unit). */
    double ky;                    /**< Optional base wavevector Y component (rad / unit). */
    double omega;                 /**< Temporal angular frequency Ω (rad / s). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double time_offset;           /**< Additional time shift applied before evaluation. */
    double nominal_dt;            /**< Nominal dt when fixed_clock is enabled. */
    double spectral_slope;        /**< Spectral slope exponent β with target PSD ∝ |k|^{-β}. */
    unsigned int feature_count;   /**< Number of random Fourier features. */
    uint64_t seed;                /**< RNG seed for reproducible features. */
    bool use_wavevector;          /**< Use (kx,ky)-projected features instead of coord mapping. */
    bool fixed_clock;             /**< Lock evolution to nominal_dt instead of adaptive dt. */
    bool scale_by_dt;             /**< Scale writes by dt when true; else dt-independent. */
} SimStimulusRandomFourierConfig;

/**
 * @brief Register a random Fourier feature stimulus operator.
 *
 * The implementation copies and normalizes @p config, derives deterministic
 * features from the seed, and registers the operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional random-Fourier configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         feature setup, or split-operator registration.
 */
SimResult sim_add_stimulus_random_fourier_operator(struct SimContext *context,
                                                   const SimStimulusRandomFourierConfig *config,
                                                   size_t *out_index);

/**
 * @brief Copy the current random-Fourier configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_random_fourier_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_random_fourier_config(struct SimContext *context, size_t operator_index,
                                             SimStimulusRandomFourierConfig *out_config);

/**
 * @brief Replace or renormalize a registered random-Fourier configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update rebuilds features as needed and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the random-Fourier operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, feature setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_random_fourier_update(struct SimContext *context, size_t operator_index,
                                             const SimStimulusRandomFourierConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_RANDOM_FOURIER_H */
