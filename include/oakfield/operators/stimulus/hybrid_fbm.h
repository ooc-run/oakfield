/**
 * @file hybrid_fbm.h
 * @brief Hybrid fBm stimulus with octave-to-octave cascade weighting.
 *
 * Synthesizes hybrid fractal noise using a seeded multi-octave cosine basis
 * where each octave modulates the next through a bounded weight cascade.
 *
 * For complex fields, the imaginary channel uses the corresponding sinusoidal
 * basis with its own bounded cascade.
 */
#ifndef OAKFIELD_STIMULUS_HYBRID_FBM_H
#define OAKFIELD_STIMULUS_HYBRID_FBM_H

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
 * @brief Configuration for seeded hybrid fractional Brownian motion stimulus noise.
 */
typedef struct SimStimulusHybridFbmConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Base amplitude of the coarsest octave. */
    double hurst;                 /**< Hurst exponent H (0 < H < 1). */
    double lacunarity;            /**< Frequency multiplier per octave lambda (>= 1). */
    unsigned int octaves;         /**< Number of octaves to sum (>= 1). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    uint64_t seed;                /**< RNG seed for reproducible phases. */
    bool scale_by_dt;             /**< When true, scale writes by dt; else dt-independent. */
} SimStimulusHybridFbmConfig;

/**
 * @brief Register a hybrid fBm stimulus operator.
 *
 * The implementation copies and normalizes @p config, initializes octave phases
 * from the configured seed, and registers the cascading fractal operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional hybrid fBm configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         octave setup, or split-operator registration.
 */
SimResult sim_add_stimulus_hybrid_fbm_operator(struct SimContext *context,
                                               const SimStimulusHybridFbmConfig *config,
                                               size_t *out_index);

/**
 * @brief Copy the current hybrid fBm configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_hybrid_fbm_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_hybrid_fbm_config(struct SimContext *context, size_t operator_index,
                                         SimStimulusHybridFbmConfig *out_config);

/**
 * @brief Replace or renormalize a registered hybrid fBm stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes octave data and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the hybrid fBm operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, octave setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_hybrid_fbm_update(struct SimContext *context, size_t operator_index,
                                         const SimStimulusHybridFbmConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_HYBRID_FBM_H */
