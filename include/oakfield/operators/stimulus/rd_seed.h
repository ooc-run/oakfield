/**
 * @file rd_seed.h
 * @brief Reaction-diffusion seed-pattern stimulus.
 *
 * Applies seeded spatial templates (spots/stripes/labyrinth/rings) as a
 * persistent reaction-diffusion style stimulus.
 */
#ifndef OAKFIELD_STIMULUS_RD_SEED_H
#define OAKFIELD_STIMULUS_RD_SEED_H

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
 * @brief Seed-pattern templates for reaction-diffusion initialization.
 */
typedef enum SimStimulusRDSeedMode {
    SIM_STIMULUS_RD_SEED_SPOTS = 0, /**< Spot-like seed pattern. */
    SIM_STIMULUS_RD_SEED_STRIPES,   /**< Stripe-like seed pattern. */
    SIM_STIMULUS_RD_SEED_LABYRINTH, /**< Labyrinth-like seed pattern. */
    SIM_STIMULUS_RD_SEED_RINGS      /**< Ring-like seed pattern. */
} SimStimulusRDSeedMode;

/**
 * @brief Configuration for reaction-diffusion seed-pattern stimuli.
 */
typedef struct SimStimulusRDSeedConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Seed amplitude scale. */
    double bias;                  /**< Additive baseline offset. */
    double scale;                 /**< Spatial scale / wavenumber control. */
    double threshold;             /**< Logistic threshold in [0,1]. */
    double sharpness;             /**< Logistic sharpness. */
    double omega;                 /**< Temporal phase rate (rad/s). */
    double phase;                 /**< Global phase offset (rad). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping controls. */
    double time_offset;           /**< Extra time shift before evaluation. */
    double rotation;              /**< Complex output rotation (rad). */
    unsigned int seed_count;      /**< Number of randomized seed primitives. */
    uint64_t seed;                /**< RNG seed controlling layout. */
    SimStimulusRDSeedMode mode;   /**< Seed-pattern mode. */
    bool scale_by_dt;             /**< When true, scale writes by dt; else dt-independent. */
} SimStimulusRDSeedConfig;

/**
 * @brief Register a reaction-diffusion seed-pattern stimulus operator.
 *
 * The implementation copies and normalizes @p config, rebuilds the seeded
 * template table, and registers the operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional RD seed configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         table setup, or split-operator registration.
 */
SimResult sim_add_stimulus_rd_seed_operator(struct SimContext *context,
                                            const SimStimulusRDSeedConfig *config,
                                            size_t *out_index);

/**
 * @brief Copy the current RD seed configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_rd_seed_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_rd_seed_config(struct SimContext *context, size_t operator_index,
                                      SimStimulusRDSeedConfig *out_config);

/**
 * @brief Replace or renormalize a registered RD seed stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update rebuilds the seed tables as needed and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the RD seed operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, table setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_rd_seed_update(struct SimContext *context, size_t operator_index,
                                      const SimStimulusRDSeedConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_RD_SEED_H */
