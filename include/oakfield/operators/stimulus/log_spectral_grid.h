/**
 * @file log_spectral_grid.h
 * @brief Log-frequency spectral grid stimulus with explicit coordinate controls.
 */
#ifndef OAKFIELD_STIMULUS_LOG_SPECTRAL_GRID_H
#define OAKFIELD_STIMULUS_LOG_SPECTRAL_GRID_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a log-frequency spectral grid stimulus.
 */
typedef struct SimStimulusLogSpectralGridConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double k_min;                 /**< Minimum log-grid radius (rad / unit). */
    double k_max;                 /**< Maximum log-grid radius (rad / unit). */
    unsigned int radial_bins;     /**< Number of logarithmic radial bins. */
    unsigned int angular_bins;    /**< Number of angular bins per radial bin. */
    double spectral_slope;        /**< Slope exponent β with PSD ∝ |k|^{-β}. */
    double orientation;           /**< Base orientation rotation (radians). */
    double orientation_rate;      /**< Orientation drift rate (rad/s). */
    double kx;                    /**< Optional wavevector X basis component. */
    double ky;                    /**< Optional wavevector Y basis component. */
    double omega;                 /**< Temporal angular frequency (rad/s). */
    double phase;                 /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping when not wavevector mode. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Complex-output rotation (radians). */
    uint64_t seed;                /**< Seed for random phase initialization. */
    bool use_wavevector;          /**< Use (kx, ky) projection basis. */
    bool random_phase;            /**< Randomize per-mode phases when true. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusLogSpectralGridConfig;

/**
 * @brief Register a log-frequency spectral grid stimulus operator.
 *
 * The implementation copies and normalizes @p config, builds the modal grid and
 * phase table, and registers the operator on the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional log-spectral-grid configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         mode-table setup, or split-operator registration.
 */
SimResult sim_add_stimulus_log_spectral_grid_operator(
    struct SimContext *context, const SimStimulusLogSpectralGridConfig *config, size_t *out_index);

/**
 * @brief Copy the current log-spectral-grid configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_log_spectral_grid_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_log_spectral_grid_config(struct SimContext *context, size_t operator_index,
                                                SimStimulusLogSpectralGridConfig *out_config);

/**
 * @brief Replace or renormalize a registered log-spectral-grid configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update rebuilds modes as needed and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the log-spectral-grid operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, mode-table
 *         setup, or state validation fails.
 */
SimResult sim_stimulus_log_spectral_grid_update(struct SimContext *context, size_t operator_index,
                                                const SimStimulusLogSpectralGridConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_LOG_SPECTRAL_GRID_H */
