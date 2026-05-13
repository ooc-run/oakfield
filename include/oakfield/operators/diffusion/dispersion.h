/**
 * @file dispersion.h
 * @brief Spectral dispersion operator applying k-dependent phase rotation.
 *
 * Complex fields receive exp(i * theta(k)) evolution. Real fields use the
 * projected real multiplier cos(theta(k)).
 */
#ifndef OAKFIELD_DISPERSION_H
#define OAKFIELD_DISPERSION_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for k-dependent spectral phase dispersion.
 */
typedef struct DispersionOperatorConfig {
    size_t field_index; /**< Target field index. */
    double coefficient; /**< Dispersion coefficient multiplying the k-dependent phase. */
    double order;   /**< Power applied to abs(abs(k) - reference_k); negative normalizes to 1. */
    double spacing; /**< Grid spacing used for wave-number scaling; non-positive normalizes to 1. */
    double reference_k; /**< Non-negative reference wave number k0. */
} DispersionOperatorConfig;

/**
 * @brief Register a spectral dispersion operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional dispersion configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         target-field validation, allocation, FFT plan setup, or registration.
 */
SimResult sim_add_dispersion_operator(struct SimContext *context,
                                      const DispersionOperatorConfig *config, size_t *out_index);

/**
 * @brief Copy the current dispersion configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_dispersion_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no dispersion state.
 */
SimResult sim_dispersion_config(struct SimContext *context, size_t operator_index,
                                DispersionOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered dispersion operator.
 *
 * @p config is required. The target field cannot be retargeted by update.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the dispersion operator to update.
 * @param config Replacement dispersion configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, state validation, field validation, or cache refresh.
 */
SimResult sim_dispersion_update(struct SimContext *context, size_t operator_index,
                                const DispersionOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_DISPERSION_H */
