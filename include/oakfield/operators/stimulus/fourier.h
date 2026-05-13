/**
 * @file fourier.h
 * @brief Bandlimited Fourier waveform stimulus (saw / square / triangle; BLIT, PolyBLEP, miniBLEP).
 */
#ifndef OAKFIELD_STIMULUS_FOURIER_H
#define OAKFIELD_STIMULUS_FOURIER_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Waveform shapes produced by the Fourier stimulus.
 */
typedef enum SimFourierWaveformShape {
    SIM_FOURIER_WAVEFORM_SAW = 0, /**< Sawtooth waveform. */
    SIM_FOURIER_WAVEFORM_SQUARE,  /**< Square waveform. */
    SIM_FOURIER_WAVEFORM_TRIANGLE /**< Triangle waveform. */
} SimFourierWaveformShape;

/**
 * @brief Bandlimiting methods used by Fourier waveform synthesis.
 */
typedef enum SimFourierWaveformMethod {
    SIM_FOURIER_METHOD_BLIT = 0, /**< Bandlimited impulse train method. */
    SIM_FOURIER_METHOD_POLYBLEP, /**< Polynomial bandlimited step method. */
    SIM_FOURIER_METHOD_MINIBLEP  /**< Windowed-sinc miniBLEP method. */
} SimFourierWaveformMethod;

/**
 * @brief Configuration for Fourier waveform stimulus.
 *
 * All frequencies are in Hz. Phase is normalized to [0,1).
 */
typedef struct SimFourierWaveformConfig {
    size_t field_index;              /**< Target field. */
    double amplitude;                /**< Output amplitude. */
    double frequency;                /**< Fundamental frequency (Hz). */
    double phase;                    /**< Initial phase in cycles [0,1). */
    double duty;                     /**< Duty cycle for square/triangle (0..1, default 0.5). */
    double rotation;                 /**< Rotation when writing complex fields (radians). */
    double nominal_dt;               /**< Optional fixed clock dt when @ref fixed_clock is true. */
    SimFourierWaveformShape shape;   /**< Waveform shape. */
    SimFourierWaveformMethod method; /**< Bandlimiting strategy. */
    bool fixed_clock;                /**< Use nominal_dt (or current dt) for phase increments. */
    bool scale_by_dt;                /**< Scale writes by substep dt (force-like semantics). */
} SimFourierWaveformConfig;

/**
 * @brief Register a bandlimited Fourier waveform stimulus operator.
 *
 * The implementation copies and normalizes @p config, initializes the requested
 * waveform state, and registers the operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional waveform configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult sim_add_stimulus_fourier_operator(struct SimContext *context,
                                            const SimFourierWaveformConfig *config,
                                            size_t *out_index);

/**
 * @brief Copy the current Fourier waveform configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_fourier_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_fourier_config(struct SimContext *context, size_t operator_index,
                                      SimFourierWaveformConfig *out_config);

/**
 * @brief Replace or renormalize a registered Fourier waveform configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes waveform state and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the Fourier waveform operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult sim_stimulus_fourier_update(struct SimContext *context, size_t operator_index,
                                      const SimFourierWaveformConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_FOURIER_H */
