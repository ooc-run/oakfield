/**
 * @file steerable_wavelet.h
 * @brief Steerable wavelet stimulus (Simoncelli/Riesz-style) with explicit coord controls.
 */
#ifndef OAKFIELD_STIMULUS_STEERABLE_WAVELET_H
#define OAKFIELD_STIMULUS_STEERABLE_WAVELET_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Steerable wavelet families available to the stimulus.
 */
typedef enum SimStimulusSteerableWaveletFamily {
    SIM_STIMULUS_STEERABLE_WAVELET_SIMONCELLI = 0, /**< Simoncelli-style wavelet family. */
    SIM_STIMULUS_STEERABLE_WAVELET_RIESZ = 1       /**< Riesz-style wavelet family. */
} SimStimulusSteerableWaveletFamily;

/**
 * @brief Configuration for steerable wavelet stimulus families and scales.
 */
typedef struct SimStimulusSteerableWaveletConfig {
    size_t field_index;                       /**< Target field index. */
    double amplitude;                         /**< Output amplitude scale. */
    SimStimulusSteerableWaveletFamily family; /**< Wavelet family mode. */
    unsigned int order;                       /**< Steering order / angular order. */
    unsigned int scale_count;                 /**< Number of radial scales. */
    double base_wavenumber;                   /**< Base radial wavenumber (rad / unit). */
    double scale_growth;                      /**< Geometric growth factor between scales. */
    double radial_bandwidth;                  /**< Log-radius Gaussian width. */
    double angular_sharpness;                 /**< Angular envelope sharpness. */
    double orientation;                       /**< Steering angle (radians). */
    double orientation_rate;                  /**< Steering angular drift (rad/s). */
    double kx;                                /**< Optional wavevector X component. */
    double ky;                                /**< Optional wavevector Y component. */
    double omega;                             /**< Temporal angular frequency (rad/s). */
    double phase;                             /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord;             /**< Coordinate mapping when not wavevector mode. */
    double time_offset;                       /**< Additional time offset before evaluation. */
    double rotation;                          /**< Complex-output rotation (radians). */
    bool use_wavevector;                      /**< Use (kx, ky) projection basis. */
    bool scale_by_dt;                         /**< Scale writes by dt when true. */
} SimStimulusSteerableWaveletConfig;

/**
 * @brief Register a steerable wavelet stimulus operator.
 *
 * The implementation copies and normalizes @p config, prepares the requested
 * wavelet family, and registers the operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional steerable-wavelet configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_steerable_wavelet_operator(
    struct SimContext *context, const SimStimulusSteerableWaveletConfig *config, size_t *out_index);

/**
 * @brief Copy the current steerable-wavelet configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_steerable_wavelet_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_steerable_wavelet_config(struct SimContext *context, size_t operator_index,
                                                SimStimulusSteerableWaveletConfig *out_config);

/**
 * @brief Replace or renormalize a registered steerable-wavelet configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes runtime state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the steerable-wavelet operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, buffer setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_steerable_wavelet_update(struct SimContext *context, size_t operator_index,
                                                const SimStimulusSteerableWaveletConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_STEERABLE_WAVELET_H */
