/**
 * @file turbulence.h
 * @brief Turbulence stimulus built from a centered absolute-value fractal basis.
 *
 * Synthesizes billowy multi-octave turbulence using
 *   T(x) = sum_o A 2^{-H o} (|cos(2π λ^o x + φ_o)| - 2/π).
 *
 * For complex fields, the imaginary channel uses the corresponding centered
 * absolute sine basis.
 */
#ifndef OAKFIELD_STIMULUS_TURBULENCE_H
#define OAKFIELD_STIMULUS_TURBULENCE_H

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
 * @brief Configuration for seeded billowy turbulence stimulus noise.
 */
typedef struct SimStimulusTurbulenceConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Base amplitude of the coarsest octave. */
    double hurst;                 /**< Hurst exponent H (0 < H < 1). */
    double lacunarity;            /**< Frequency multiplier per octave λ (>= 1). */
    unsigned int octaves;         /**< Number of octaves to sum (>= 1). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    uint64_t seed;                /**< RNG seed for reproducible phases. */
    bool scale_by_dt;             /**< When true, scale writes by dt; else dt-independent. */
} SimStimulusTurbulenceConfig;

/**
 * @brief Register a billowy turbulence stimulus operator.
 *
 * The implementation copies and normalizes @p config, initializes octave phases
 * from the configured seed, and registers the turbulence operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional turbulence configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         octave setup, or split-operator registration.
 */
SimResult sim_add_stimulus_turbulence_operator(struct SimContext *context,
                                               const SimStimulusTurbulenceConfig *config,
                                               size_t *out_index);

/**
 * @brief Copy the current turbulence configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_turbulence_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_turbulence_config(struct SimContext *context, size_t operator_index,
                                         SimStimulusTurbulenceConfig *out_config);

/**
 * @brief Replace or renormalize a registered turbulence stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes octave data and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the turbulence operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, octave setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_turbulence_update(struct SimContext *context, size_t operator_index,
                                         const SimStimulusTurbulenceConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_TURBULENCE_H */
