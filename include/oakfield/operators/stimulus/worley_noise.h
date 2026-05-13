/**
 * @file worley_noise.h
 * @brief Worley / cellular noise stimulus with selectable distance metrics.
 *
 * Synthesizes one deterministic feature point per lattice cell and evaluates
 * the distance to the nearest or second-nearest feature. The lattice is sampled
 * in a local coordinate frame controlled by `coord`, then scaled by
 * `feature_frequency`.
 *
 * For complex fields, the imaginary component is generated from an independent
 * deterministic feature lattice derived from the same seed.
 */
#ifndef OAKFIELD_STIMULUS_WORLEY_NOISE_H
#define OAKFIELD_STIMULUS_WORLEY_NOISE_H

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
 * @brief Distance metric used when evaluating Worley feature points.
 */
typedef enum SimStimulusWorleyDistanceMetric {
    SIM_STIMULUS_WORLEY_EUCLIDEAN = 0, /**< Euclidean distance. */
    SIM_STIMULUS_WORLEY_MANHATTAN = 1, /**< Manhattan distance. */
    SIM_STIMULUS_WORLEY_CHEBYSHEV = 2, /**< Chebyshev distance. */
    SIM_STIMULUS_WORLEY_MINKOWSKI = 3  /**< Minkowski distance with configured exponent. */
} SimStimulusWorleyDistanceMetric;

/**
 * @brief Feature-distance output selected by Worley noise.
 */
typedef enum SimStimulusWorleyOutputMode {
    SIM_STIMULUS_WORLEY_F1 = 0,         /**< Nearest feature distance. */
    SIM_STIMULUS_WORLEY_F2 = 1,         /**< Second-nearest feature distance. */
    SIM_STIMULUS_WORLEY_F2_MINUS_F1 = 2 /**< Difference between second and nearest distances. */
} SimStimulusWorleyOutputMode;

/**
 * @brief Configuration for seeded Worley cellular-noise stimulus fields.
 */
typedef struct SimStimulusWorleyNoiseConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double feature_frequency;     /**< Lattice cells per coordinate unit. */
    double jitter;                /**< Feature-point jitter within each cell [0, 1]. */
    double distance_exponent;     /**< Exponent used when distance_metric=minkowski. */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    uint64_t seed;                /**< RNG seed for reproducible cellular layouts. */
    SimStimulusWorleyDistanceMetric
        distance_metric; /**< Distance metric used between samples and features. */
    SimStimulusWorleyOutputMode output_mode; /**< F1/F2/F2-F1 selection. */
    bool scale_by_dt; /**< When true, scale writes by dt; else dt-independent. */
} SimStimulusWorleyNoiseConfig;

/**
 * @brief Register a Worley cellular-noise stimulus operator.
 *
 * The implementation copies and normalizes @p config, derives deterministic
 * hash bases from the seed, and registers the noise operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Worley-noise configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         cache setup, or split-operator registration.
 */
SimResult sim_add_stimulus_worley_noise_operator(struct SimContext *context,
                                                 const SimStimulusWorleyNoiseConfig *config,
                                                 size_t *out_index);

/**
 * @brief Copy the current Worley-noise configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_worley_noise_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_worley_noise_config(struct SimContext *context, size_t operator_index,
                                           SimStimulusWorleyNoiseConfig *out_config);

/**
 * @brief Replace or renormalize a registered Worley-noise stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes hash bases/caches as needed and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Worley-noise operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, cache setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_worley_noise_update(struct SimContext *context, size_t operator_index,
                                           const SimStimulusWorleyNoiseConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_WORLEY_NOISE_H */
