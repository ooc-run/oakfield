/**
 * @file checkerboard.h
 * @brief Checkerboard / stripe pattern stimulus (complex-capable).
 *
 * Generates a static geometric pattern over 1D or 2D fields:
 *     f(i, j) = A · (-1)^{⌊i / Px⌋ + ⌊j / Py⌋}
 * For complex fields, an additional global phase rotation is applied.
 */
#ifndef OAKFIELD_STIMULUS_CHECKERBOARD_H
#define OAKFIELD_STIMULUS_CHECKERBOARD_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for checkerboard or stripe pattern stimulus writes.
 */
typedef struct SimStimulusCheckerboardConfig {
    size_t field_index; /**< Target field index. */
    double amplitude;   /**< Amplitude of the pattern. */
    double period_x;    /**< Period in coordinate units along X (>= 1). */
    double period_y;    /**< Period in coordinate units along Y (> 0; <= 0 → stripes in X only). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double phase;                 /**< Global phase offset in checker cells (0..1). */
    double complex_phase;         /**< Additional complex rotation for complex fields (radians). */
    bool scale_by_dt;             /**< When true, scale writes by dt; otherwise dt-independent. */
} SimStimulusCheckerboardConfig;

/**
 * @brief Register a checkerboard or stripe stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the geometric pattern into the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional checkerboard configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_checkerboard_operator(struct SimContext *context,
                                                 const SimStimulusCheckerboardConfig *config,
                                                 size_t *out_index);

/**
 * @brief Copy the current checkerboard configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_checkerboard_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_checkerboard_config(struct SimContext *context, size_t operator_index,
                                           SimStimulusCheckerboardConfig *out_config);

/**
 * @brief Replace or renormalize a registered checkerboard stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes derived state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the checkerboard operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, allocation, or
 *         state validation fails.
 */
SimResult sim_stimulus_checkerboard_update(struct SimContext *context, size_t operator_index,
                                           const SimStimulusCheckerboardConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_CHECKERBOARD_H */
