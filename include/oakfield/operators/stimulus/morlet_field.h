/**
 * @file morlet_field.h
 * @brief Morlet wavelet field stimulus with explicit coordinate controls.
 */
#ifndef OAKFIELD_STIMULUS_MORLET_FIELD_H
#define OAKFIELD_STIMULUS_MORLET_FIELD_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a multi-scale Morlet wavelet field stimulus.
 */
typedef struct SimStimulusMorletFieldConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    unsigned int scale_count;     /**< Number of Morlet scales. */
    double base_wavenumber;       /**< Base carrier wavenumber (rad / unit). */
    double scale_growth;          /**< Geometric growth factor for wavenumber. */
    double sigma_base;            /**< Base Gaussian envelope width (units). */
    double sigma_growth;          /**< Geometric growth factor for envelope width. */
    double center_u;              /**< Center in local u coordinate. */
    double center_v;              /**< Center in local v coordinate. */
    double velocity_u;            /**< Center drift velocity in u (units/s). */
    double velocity_v;            /**< Center drift velocity in v (units/s). */
    double orientation;           /**< Wavelet orientation angle (radians). */
    double orientation_rate;      /**< Orientation angular drift (rad/s). */
    double kx;                    /**< Optional wavevector X component. */
    double ky;                    /**< Optional wavevector Y component. */
    double omega;                 /**< Temporal angular frequency (rad/s). */
    double phase;                 /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping when not wavevector mode. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Complex-output rotation (radians). */
    bool use_wavevector;          /**< Use (kx, ky) projection basis. */
    bool zero_mean;               /**< Apply Morlet zero-mean correction. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusMorletFieldConfig;

/**
 * @brief Register a Morlet wavelet field stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that sums the requested Morlet scales on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional Morlet-field configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_morlet_field_operator(struct SimContext *context,
                                                 const SimStimulusMorletFieldConfig *config,
                                                 size_t *out_index);

/**
 * @brief Copy the current Morlet-field configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_morlet_field_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_morlet_field_config(struct SimContext *context, size_t operator_index,
                                           SimStimulusMorletFieldConfig *out_config);

/**
 * @brief Replace or renormalize a registered Morlet-field configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Morlet-field operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_morlet_field_update(struct SimContext *context, size_t operator_index,
                                           const SimStimulusMorletFieldConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_MORLET_FIELD_H */
