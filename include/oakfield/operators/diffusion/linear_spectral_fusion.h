/**
 * @file linear_spectral_fusion.h
 * @brief Fused spectral dissipation, dispersion, and global phase operator.
 */
#ifndef OAKFIELD_LINEAR_SPECTRAL_FUSION_H
#define OAKFIELD_LINEAR_SPECTRAL_FUSION_H

#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for the fused spectral dissipation, dispersion, and phase operator.
 */
typedef struct LinearSpectralFusionOperatorConfig {
    size_t field_index; /**< Target field index. */

    /* Dissipation: exp(dt * lambda(k)), lambda(k) = -viscosity * |k|^alpha. */
    double viscosity;           /**< Dissipative viscosity coefficient. */
    double alpha;               /**< Dissipation exponent applied to |k|. */
    double dissipation_spacing; /**< Grid spacing used for dissipation wavenumbers. */

    /* Dispersion: exp(i * dt * omega(k)), omega(k) = coefficient * | |k| - k0 |^order.
     * Real fields use the projected real multiplier cos(dt * omega(k)).
     */
    double dispersion_coefficient; /**< Dispersion angular-frequency coefficient. */
    double dispersion_order;       /**< Dispersion exponent applied around reference k. */
    double dispersion_reference_k; /**< Reference wavenumber k0 for dispersion. */
    double dispersion_spacing;     /**< Grid spacing used for dispersion wavenumbers. */

    /* Global phase: exp(i * dt * phase_rate). Real fields use cos(dt * phase_rate). */
    double phase_rate; /**< Global phase rotation rate in radians/second. */
} LinearSpectralFusionOperatorConfig;

/**
 * @brief Register a fused linear spectral operator.
 *
 * The operator combines dissipative damping, dispersive phase, and global phase
 * terms in one spectral pass when a split fallback is required.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional fusion configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, FFT/cache setup, or registration.
 */
SimResult sim_add_linear_spectral_fusion_operator(struct SimContext *context,
                                                  const LinearSpectralFusionOperatorConfig *config,
                                                  size_t *out_index);

/**
 * @brief Copy the current fused spectral configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_linear_spectral_fusion_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no fusion state.
 */
SimResult sim_linear_spectral_fusion_config(struct SimContext *context, size_t operator_index,
                                            LinearSpectralFusionOperatorConfig *out_config);

/**
 * @brief Replace the non-target parameters of a fused spectral operator.
 *
 * @p config is required. The registered target field is preserved even if
 * config->field_index differs. A successful update refreshes symbolic/kernel
 * constants, marks changed spectral coefficient caches dirty, and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the fused spectral operator to update.
 * @param config Replacement configuration values.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field validation, or state validation.
 */
SimResult sim_linear_spectral_fusion_update(struct SimContext *context, size_t operator_index,
                                            const LinearSpectralFusionOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_LINEAR_SPECTRAL_FUSION_H */
