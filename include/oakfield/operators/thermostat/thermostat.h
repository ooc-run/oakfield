/**
 * @file thermostat.h
 * @brief Soft energy / lambda regulation (thermostat) operator.
 */
#ifndef OAKFIELD_THERMOSTAT_H
#define OAKFIELD_THERMOSTAT_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Thermostat regulation mode.
 *
 * THERMOSTAT_SOFT_LAMBDA adjusts lambda_eff from the measured mean energy.
 * THERMOSTAT_ADD and THERMOSTAT_MULT apply additive or multiplicative
 * relaxation toward either E_target or an optional per-point memory field.
 */
typedef enum ThermostatMode {
    THERMOSTAT_NONE = 0,        /**< Disable thermostat regulation. */
    THERMOSTAT_SOFT_LAMBDA = 1, /**< Adjust lambda_eff from measured mean energy. */
    THERMOSTAT_ADD = 2,         /**< Additive relaxation toward target or memory field. */
    THERMOSTAT_MULT = 3         /**< Multiplicative relaxation toward target or memory field. */
} ThermostatMode;

/**
 * @brief Configuration for the thermostat operator.
 *
 * The operator updates internal scalars such as lambda_eff and optionally uses a
 * memory target field for additive or multiplicative regulation modes.
 */
typedef struct ThermostatOperatorConfig {
    size_t field_index;           /**< Target field containing u. */
    ThermostatMode mode;          /**< Regulation mode. */
    double E_target;              /**< Desired mean energy averaged over |u|^2. */
    double lambda_base;           /**< Base (unregulated) lambda. */
    double lambda_soft_gain;      /**< Gain applied to (E - E_target) in soft lambda. */
    double lambda_min;            /**< Lower clamp; ignored if not finite. */
    double lambda_max;            /**< Upper clamp; ignored if not finite. */
    double lambda_smooth;         /**< Exponential smoothing factor [0,1]. */
    double lambda_rebuild_thresh; /**< Threshold for integrator rebuild trigger. */
    double softplus_k;            /**< Softness for smooth clamp (>=0). */
    double mu;                    /**< Relaxation strength for ADD/MULT modes (0..1). */
    bool auto_nu_guard;           /**< If true, compute minimal nu so Re(L) >= 0 when alpha < 0. */
    size_t memory_field;          /**< Index of M field; set to (size_t)-1 if unused. */
    bool use_memory_field;        /**< True if memory_field refers to a valid field. */
} ThermostatOperatorConfig;

/**
 * @brief Register a thermostat operator instance.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional thermostat configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         allocation, kernel registration, or split registration.
 */
SimResult sim_add_thermostat_operator(struct SimContext *context,
                                      const ThermostatOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current thermostat configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_thermostat_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no thermostat state.
 */
SimResult sim_thermostat_config(struct SimContext *context, size_t operator_index,
                                ThermostatOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered thermostat operator.
 *
 * @p config is required. A successful update normalizes the replacement, resets
 * lambda/cache state, refreshes symbolic metadata, and invalidates the scheduler
 * plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the thermostat operator to update.
 * @param config Replacement thermostat configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, or state validation.
 */
SimResult sim_thermostat_update(struct SimContext *context, size_t operator_index,
                                const ThermostatOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_THERMOSTAT_H */
