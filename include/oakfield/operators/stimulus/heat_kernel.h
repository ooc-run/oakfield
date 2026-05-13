/**
 * @file heat_kernel.h
 * @brief Heat-kernel stimulus with diffusive Gaussian broadening.
 *
 * Evolves an initial Gaussian profile under the heat equation by expanding
 * its width as sigma(t)^2 = sigma_0^2 + 2 D t for t >= 0.
 */
#ifndef OAKFIELD_STIMULUS_HEAT_KERNEL_H
#define OAKFIELD_STIMULUS_HEAT_KERNEL_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a diffusive Gaussian heat-kernel stimulus.
 */
typedef struct SimStimulusHeatKernelConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Initial amplitude scale. */
    double diffusivity;           /**< Heat diffusivity D (units^2 / s). */
    double sigma_x;               /**< Initial sigma along X (units). */
    double sigma_y;               /**< Initial sigma along Y (units). */
    double time_offset;           /**< Extra time shift before evaluation. */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double rotation;              /**< Complex output rotation (radians). */
    double nominal_dt;            /**< Nominal dt used when fixed_clock is true. */
    bool fixed_clock;             /**< Hold the driving clock to nominal_dt when true. */
    bool scale_by_dt;             /**< When true, scale writes by dt. */
    bool preserve_mass;           /**< Preserve integrated mass as the profile broadens. */
} SimStimulusHeatKernelConfig;

/**
 * @brief Register a diffusive heat-kernel stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that evaluates the broadened Gaussian profile on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional heat-kernel configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_heat_kernel_operator(struct SimContext *context,
                                                const SimStimulusHeatKernelConfig *config,
                                                size_t *out_index);

/**
 * @brief Copy the current heat-kernel configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_heat_kernel_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_heat_kernel_config(struct SimContext *context, size_t operator_index,
                                          SimStimulusHeatKernelConfig *out_config);

/**
 * @brief Replace or renormalize a registered heat-kernel stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the heat-kernel operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_heat_kernel_update(struct SimContext *context, size_t operator_index,
                                          const SimStimulusHeatKernelConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_HEAT_KERNEL_H */
