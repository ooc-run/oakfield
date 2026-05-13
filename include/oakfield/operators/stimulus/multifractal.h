/**
 * @file multifractal.h
 * @brief Multiplicative multifractal stimulus built from centered octave products.
 *
 * Synthesizes multiplicative fractal noise using a seeded multi-octave cosine basis:
 *   M(x) = prod_o (1 + 0.5 * lambda^{-H o} * cos(2*pi*lambda^o*x + phi_o)) - 1.
 *
 * For complex fields, the imaginary channel uses the corresponding centered
 * multiplicative product of the sine basis.
 */
#ifndef OAKFIELD_STIMULUS_MULTIFRACTAL_H
#define OAKFIELD_STIMULUS_MULTIFRACTAL_H

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
 * @brief Configuration for seeded multiplicative multifractal stimulus noise.
 */
typedef struct SimStimulusMultifractalConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Base amplitude of the coarsest octave. */
    double hurst;                 /**< Hurst exponent H (0 < H < 1). */
    double lacunarity;            /**< Frequency multiplier per octave lambda (>= 1). */
    unsigned int octaves;         /**< Number of octaves to combine (>= 1). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    uint64_t seed;                /**< RNG seed for reproducible phases. */
    bool scale_by_dt;             /**< When true, scale writes by dt; else dt-independent. */
} SimStimulusMultifractalConfig;

/**
 * @brief Register a multiplicative multifractal stimulus operator.
 *
 * The implementation copies and normalizes @p config, initializes octave phases
 * from the configured seed, and registers the multifractal operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional multifractal configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         octave setup, or split-operator registration.
 */
SimResult sim_add_stimulus_multifractal_operator(struct SimContext *context,
                                                 const SimStimulusMultifractalConfig *config,
                                                 size_t *out_index);

/**
 * @brief Copy the current multifractal configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_multifractal_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_multifractal_config(struct SimContext *context, size_t operator_index,
                                           SimStimulusMultifractalConfig *out_config);

/**
 * @brief Replace or renormalize a registered multifractal stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes octave data and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the multifractal operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, octave setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_multifractal_update(struct SimContext *context, size_t operator_index,
                                           const SimStimulusMultifractalConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_MULTIFRACTAL_H */
