/**
 * @file metal_mix.h
 * @brief Metal-friendly mixer subset (linear + crossfade) with split fallback.
 */
#ifndef OAKFIELD_METAL_MIX_H
#define OAKFIELD_METAL_MIX_H

#include "mixer.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for the metal_mix operator.
 *
 * Supports only SIM_MIXER_MODE_LINEAR and SIM_MIXER_MODE_CROSSFADE.
 */
typedef struct SimMetalMixOperatorConfig {
    size_t lhs_field;    /**< First input field. */
    size_t rhs_field;    /**< Second input field. */
    size_t output_field; /**< Field receiving the mixed result. */
    double lhs_gain;     /**< Gain applied to lhs samples. */
    double rhs_gain;     /**< Gain applied to rhs samples. */
    double mix;          /**< Crossfade parameter. */
    double bias;         /**< Constant offset added to the result. */
    SimMixerMode mode;   /**< Supported mode: linear or crossfade. */
    bool accumulate;     /**< Add into output when true. */
    bool scale_by_dt;    /**< Scale writes by substep dt when true. */
} SimMetalMixOperatorConfig;

/**
 * @brief Return the schema name for a metal-mix mode.
 *
 * @param mode Mixer mode value.
 * @return Stable lowercase mode name.
 */
const char *metal_mix_mode_name(SimMixerMode mode);

/**
 * @brief Parse a metal-mix mode name.
 *
 * Only SIM_MIXER_MODE_LINEAR and SIM_MIXER_MODE_CROSSFADE are accepted.
 *
 * @param name Schema mode name.
 * @param[out] out_mode Receives the parsed mode on success.
 * @return true when @p name maps to a supported metal-mix mode.
 */
bool metal_mix_mode_from_name(const char *name, SimMixerMode *out_mode);

/**
 * @brief Register a metal-friendly mixer subset operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional metal-mix configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         mode validation, field compatibility checks, allocation, or registration.
 */
SimResult sim_add_metal_mix_operator(struct SimContext *context,
                                     const SimMetalMixOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current metal-mix configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_metal_mix_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no metal-mix state.
 */
SimResult sim_metal_mix_config(struct SimContext *context, size_t operator_index,
                               SimMetalMixOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered metal-mix configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the metal-mix operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, mode
 *         validation, field compatibility checks, or state validation.
 */
SimResult sim_metal_mix_update(struct SimContext *context, size_t operator_index,
                               const SimMetalMixOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_METAL_MIX_H */
