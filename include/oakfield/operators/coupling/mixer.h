/**
 * @file mixer.h
 * @brief Field mixing/coupling operators (linear, multiply, crossfade, AM/FM/PM, etc.).
 *
 * Supports real and complex fields; for complex, operations are applied component-wise unless
 * the mode explicitly defines a complex interaction (e.g., ring modulation).
 */
#ifndef OAKFIELD_MIXER_H
#define OAKFIELD_MIXER_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Mixing strategies supported by the mixer operator family.
 */
typedef enum SimMixerMode {
    SIM_MIXER_MODE_LINEAR = 0,          /**< Weighted linear blend. */
    SIM_MIXER_MODE_MULTIPLY,            /**< Pointwise multiplication. */
    SIM_MIXER_MODE_CROSSFADE,           /**< Crossfade between lhs and rhs. */
    SIM_MIXER_MODE_SUM,                 /**< Sum inputs after gain and bias. */
    SIM_MIXER_MODE_POWER,               /**< Raise lhs by rhs-derived exponent. */
    SIM_MIXER_MODE_AM,                  /**< Amplitude modulation. */
    SIM_MIXER_MODE_FM,                  /**< Frequency-modulation-style phase coupling. */
    SIM_MIXER_MODE_PM,                  /**< Phase modulation. */
    SIM_MIXER_MODE_RING_MOD,            /**< Ring modulation product. */
    SIM_MIXER_MODE_MAX,                 /**< Pointwise maximum. */
    SIM_MIXER_MODE_MIN,                 /**< Pointwise minimum. */
    SIM_MIXER_MODE_AVERAGE,             /**< Pointwise average. */
    SIM_MIXER_MODE_DIFFERENCE,          /**< Signed lhs-rhs difference. */
    SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE, /**< Absolute lhs-rhs difference. */
    SIM_MIXER_MODE_FEEDBACK             /**< Feedback coupling mode. */
} SimMixerMode;

/**
 * @brief Interpretation of feedback_epsilon in feedback mixer mode.
 */
typedef enum SimMixerFeedbackEpsilonMode {
    SIM_MIXER_FEEDBACK_EPS_INPUT = 0, /**< Epsilon scales the input injection term. */
    SIM_MIXER_FEEDBACK_EPS_FEEDBACK   /**< Epsilon scales the feedback (previous output) term. */
} SimMixerFeedbackEpsilonMode;

/**
 * @brief Operator splitting strategy for feedback mixer integration.
 */
typedef enum SimMixerFeedbackSplitMode {
    SIM_MIXER_FEEDBACK_SPLIT_NONE = 0, /**< No split integration for feedback. */
    SIM_MIXER_FEEDBACK_SPLIT_LIE,      /**< Lie splitting for feedback updates. */
    SIM_MIXER_FEEDBACK_SPLIT_STRANG    /**< Strang splitting for feedback updates. */
} SimMixerFeedbackSplitMode;

/**
 * @brief Configuration parameters for the mixer operator.
 */
typedef struct SimMixerOperatorConfig {
    size_t lhs_field;        /**< First input field. */
    size_t rhs_field;        /**< Second input field. */
    size_t output_field;     /**< Field receiving the mixed result. */
    double lhs_gain;         /**< Gain applied to lhs samples. */
    double rhs_gain;         /**< Gain applied to rhs samples. */
    double mix;              /**< Crossfade parameter used by CROSSFADE mode. */
    double bias;             /**< Constant offset added to the mixed output. */
    double feedback_epsilon; /**< Feedback strength epsilon (mode-specific). */
    SimMixerMode mode;       /**< Mixing strategy. */
    SimMixerFeedbackEpsilonMode feedback_epsilon_mode; /**< Interpretation of feedback_epsilon. */
    SimMixerFeedbackSplitMode feedback_split; /**< Optional split integration for feedback. */
    bool accumulate;                          /**< When true, adds into the output field. */
    bool scale_by_dt; /**< When true, scale accumulated writes by substep dt. */
} SimMixerOperatorConfig;

/**
 * @brief Return the schema name for a mixer mode.
 *
 * @param mode Mixer mode value.
 * @return Stable lowercase mode name.
 */
const char *mixer_mode_name(SimMixerMode mode);

/**
 * @brief Return the schema name for a feedback epsilon interpretation mode.
 *
 * @param mode Feedback epsilon mode value.
 * @return Stable lowercase mode name.
 */
const char *mixer_feedback_epsilon_mode_name(SimMixerFeedbackEpsilonMode mode);

/**
 * @brief Return the schema name for a feedback split mode.
 *
 * @param mode Feedback split mode value.
 * @return Stable lowercase mode name.
 */
const char *mixer_feedback_split_name(SimMixerFeedbackSplitMode mode);

/**
 * @brief Convert a schema/descriptor string value into a mixer mode enum.
 *
 * @param name Schema value string such as "linear" or "multiply".
 * @param[out] out_mode Filled with the corresponding enum on success.
 * @return true when the string maps to a known mixer mode.
 */
bool mixer_mode_from_name(const char *name, SimMixerMode *out_mode);

/**
 * @brief Parse a feedback epsilon mode name.
 *
 * @param name Schema value string.
 * @param[out] out_mode Filled with the corresponding enum on success.
 * @return true when @p name maps to a known feedback epsilon mode.
 */
bool mixer_feedback_epsilon_mode_from_name(const char *name, SimMixerFeedbackEpsilonMode *out_mode);

/**
 * @brief Parse a feedback split mode name.
 *
 * @param name Schema value string.
 * @param[out] out_mode Filled with the corresponding enum on success.
 * @return true when @p name maps to a known feedback split mode.
 */
bool mixer_feedback_split_from_name(const char *name, SimMixerFeedbackSplitMode *out_mode);

/**
 * @brief Register a mixer operator combining two fields.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional mixer configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         mode validation, field compatibility checks, allocation, or registration.
 */
SimResult sim_add_mixer_operator(struct SimContext *context, const SimMixerOperatorConfig *config,
                                 size_t *out_index);

/**
 * @brief Retrieve the configuration stored in a mixer operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_mixer_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no mixer state.
 */
SimResult sim_mixer_config(struct SimContext *context, size_t operator_index,
                           SimMixerOperatorConfig *out_config);

/**
 * @brief Update an existing mixer operator's configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes runtime/symbolic state and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the mixer operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, mode
 *         validation, field compatibility checks, or state validation.
 */
SimResult sim_mixer_update(struct SimContext *context, size_t operator_index,
                           const SimMixerOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MIXER_H */
