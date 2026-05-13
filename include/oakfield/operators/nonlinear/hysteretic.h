/**
 * @file hysteretic.h
 * @brief Hysteretic operator with Schmitt, play, and Bouc-Wen modes.
 *        Supports matching real or complex input/output fields.
 */
#ifndef OAKFIELD_HYSTERETIC_H
#define OAKFIELD_HYSTERETIC_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Hysteresis mode selection.
 */
typedef enum SimHystereticMode {
    SIM_HYSTERETIC_MODE_SCHMITT = 0, /**< Binary Schmitt trigger. */
    SIM_HYSTERETIC_MODE_PLAY,        /**< Continuous play/deadband operator. */
    SIM_HYSTERETIC_MODE_BOUC_WEN     /**< Dynamic Bouc-Wen hysteresis model. */
} SimHystereticMode;

/**
 * @brief Threshold specification strategy.
 */
typedef enum SimHystereticThresholdMode {
    SIM_HYSTERETIC_THRESHOLD_BOUNDS = 0,  /**< Use (threshold_low, threshold_high). */
    SIM_HYSTERETIC_THRESHOLD_CENTER_WIDTH /**< Use (threshold_center, threshold_width). */
} SimHystereticThresholdMode;

/**
 * @brief Input preprocessing options.
 */
typedef enum SimHystereticInputMode {
    SIM_HYSTERETIC_INPUT_DIRECT = 0, /**< Use input samples directly. */
    SIM_HYSTERETIC_INPUT_ABS,        /**< Use absolute value of input. */
    SIM_HYSTERETIC_INPUT_SQUARED     /**< Use squared input. */
} SimHystereticInputMode;

/**
 * @brief Configuration for hysteretic operators.
 */
typedef struct SimHystereticOperatorConfig {
    size_t input_field;                        /**< Input field index. */
    size_t output_field;                       /**< Output field index. */
    SimHystereticMode mode;                    /**< Hysteresis mode. */
    SimHystereticThresholdMode threshold_mode; /**< Threshold mode. */
    SimHystereticInputMode input_mode;         /**< Input preprocessing mode. */
    double input_gain;                         /**< Input scaling factor. */
    double input_bias;                         /**< Input bias. */
    double threshold_low;                      /**< Lower threshold (bounds mode). */
    double threshold_high;                     /**< Upper threshold (bounds mode). */
    double threshold_center;                   /**< Threshold center (center/width mode). */
    double threshold_width;                    /**< Threshold width (center/width mode). */
    double output_low;                         /**< Schmitt low output value. */
    double output_high;                        /**< Schmitt high output value. */
    double state_min;                          /**< Clamp min for internal state. */
    double state_max;                          /**< Clamp max for internal state. */
    double smooth;                             /**< Output smoothing factor [0,1]. */
    double rate_limit;                         /**< Max |delta| per second (0 disables). */
    bool accumulate;                           /**< Accumulate into output. */
    bool scale_by_dt;                          /**< Scale accumulation by dt. */
    bool initialize_from_input;                /**< Initialize state from input. */
    double initial_output;                     /**< Initial output state. */
    double initial_input;                      /**< Initial input history (Bouc-Wen). */
    double initial_z;                          /**< Initial Bouc-Wen internal state. */
    double play_radius;                        /**< Play/deadband radius (<=0 uses band). */
    double bw_alpha;                           /**< Bouc-Wen alpha (linear blend). */
    double bw_A;                               /**< Bouc-Wen A scale. */
    double bw_beta;                            /**< Bouc-Wen beta parameter. */
    double bw_gamma;                           /**< Bouc-Wen gamma parameter. */
    double bw_n;                               /**< Bouc-Wen exponent n (>=1). */
    double bw_z_clamp;                         /**< Clamp for Bouc-Wen z state (<=0 disables). */
    double bw_xdot_clamp;                      /**< Clamp for input derivative (<=0 disables). */
    double output_gain;                        /**< Output scaling factor. */
    double output_bias;                        /**< Output bias. */
} SimHystereticOperatorConfig;

/**
 * @brief Registers a hysteretic operator instance.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional hysteretic configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_hysteretic_operator(struct SimContext *context,
                                      const SimHystereticOperatorConfig *config, size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a hysteretic operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_hysteretic_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no hysteretic state.
 */
SimResult sim_hysteretic_config(struct SimContext *context, size_t operator_index,
                                SimHystereticOperatorConfig *out_config);

/**
 * @brief Update an existing hysteretic operator in-place.
 *
 * @p config is required. A successful update normalizes the replacement,
 * validates field compatibility, resets internal history initialization, and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the hysteretic operator to update.
 * @param config Replacement hysteretic configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field compatibility checks, or state validation.
 */
SimResult sim_hysteretic_update(struct SimContext *context, size_t operator_index,
                                const SimHystereticOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_HYSTERETIC_H */
