/**
 * @file operator_semilinear.h
 * @brief Semilinear role classification for ETDRK4-compatible operators.
 *
 * Public helpers in this header describe how a registered operator contributes
 * to a semilinear split: exact linear flow, dt-scaled increment, general
 * nonlinear term, or unsupported. The traits are advisory metadata used by
 * integrator planning and do not execute or mutate operator state.
 */
#ifndef OAKFIELD_OPERATOR_SEMILINEAR_H
#define OAKFIELD_OPERATOR_SEMILINEAR_H

#include <stddef.h>

#include "oakfield/operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Classifies how an operator participates in a semilinear ETDRK4 split.
 */
typedef enum SimSemilinearRole
{
    SIM_SEMILINEAR_ROLE_UNSUPPORTED = 0,       /**< Operator is excluded from semilinear planning. */
    SIM_SEMILINEAR_ROLE_EXACT_LINEAR_FLOW,     /**< Operator can be represented as an exact linear flow. */
    SIM_SEMILINEAR_ROLE_DT_SCALED_INCREMENT,   /**< Operator contributes an additive dt-scaled increment. */
    SIM_SEMILINEAR_ROLE_GENERAL_NONLINEAR      /**< Operator is nonlinear but may be handled explicitly. */
} SimSemilinearRole;

/**
 * @brief Instance-resolved semilinear traits used by integrator plan classification.
 */
typedef struct SimSemilinearTraits
{
    SimSemilinearRole role;   /**< Classification for semilinear splitting. */
    SimClockMode clock_mode;  /**< Best-effort clock classification for time-sensitive sources. */
    bool scale_by_dt;         /**< True when the operator contribution scales with dt. */
    bool accumulate;          /**< True when the operator adds into its destination. */
    size_t input_field;       /**< Source field when applicable, otherwise SIZE_MAX. */
    size_t output_field;      /**< Destination field when applicable, otherwise SIZE_MAX. */
} SimSemilinearTraits;

/**
 * @brief Return default semilinear traits for unsupported operators.
 *
 * @return Traits with role #SIM_SEMILINEAR_ROLE_UNSUPPORTED and field indices
 *         set to SIZE_MAX.
 */
SimSemilinearTraits sim_operator_semilinear_traits_defaults(void);

/**
 * @brief Classify an operator's role in a semilinear ETDRK4 split.
 *
 * The returned traits preserve current ETDRK4 support rules, including the
 * existing allowlist for exact linear flows and dt-scaled additive sources.
 * The function only inspects registry/config state and does not mutate the
 * context or operator.
 *
 * @param context Simulation context containing the registered operator.
 * @param operator_index Index of the operator in the context registry.
 * @param target_field_index Field index whose semilinear plan is being built.
 * @param[out] out_traits Receives the resolved traits on success.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_NOT_FOUND.
 */
SimResult sim_operator_classify_semilinear(struct SimContext*   context,
                                           size_t               operator_index,
                                           size_t               target_field_index,
                                           SimSemilinearTraits* out_traits);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_OPERATOR_SEMILINEAR_H */
