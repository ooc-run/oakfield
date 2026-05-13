/**
 * @file sinusoidal.h
 * @brief Sinusoidal stimulus operators: traveling, standing, chirped, and Gaussian-envelope.
 *
 * Complex fields are driven by writing real/imag components with optional rotation.
 */
#ifndef OAKFIELD_STIMULUS_SINUSOIDAL_H
#define OAKFIELD_STIMULUS_SINUSOIDAL_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Shared configuration for sinusoidal stimulus variants.
 */
typedef struct SimStimulusSinusoidalConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Signal amplitude. */
    double wavenumber;            /**< Base spatial wavenumber (rad / unit). */
    double kx;                    /**< Optional wavevector X component (rad / unit). */
    double ky;                    /**< Optional wavevector Y component (rad / unit). */
    double omega;                 /**< Base angular frequency (rad / s). */
    double phase;                 /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double time_offset;           /**< Additional time offset applied before evaluation. */
    double nominal_dt; /**< Optional nominal dt when fixed_clock is enabled (<=0 uses actual dt). */
    double kdot;       /**< Wavenumber sweep rate for chirp (rad / unit / s). */
    double wdot;       /**< Frequency sweep rate for chirp (rad / s^2). */
    double rotation;   /**< Rotation applied when writing into complex fields (radians). */
    bool use_wavevector; /**< When true, use (kx,ky) instead of wavenumber+coord. */
    bool fixed_clock;    /**< Lock the driving clock to nominal_dt instead of adaptive dt. */
    bool scale_by_dt; /**< When true, scale writes by substep dt; false = dt-independent signal. */
} SimStimulusSinusoidalConfig;

/**
 * @brief Sinusoidal stimulus operation mode.
 */
typedef enum SimStimulusSinusoidalMode {
    SIM_STIMULUS_SINUSOIDAL_SINE = 0, /**< Traveling sine wave mode. */
    SIM_STIMULUS_SINUSOIDAL_STANDING, /**< Standing-wave mode. */
    SIM_STIMULUS_SINUSOIDAL_CHIRP     /**< Chirped sine wave mode. */
} SimStimulusSinusoidalMode;

/**
 * @brief Internal state for sinusoidal stimulus operators.
 */
typedef struct SimStimulusSinusoidalState {
    SimStimulusSinusoidalConfig config; /**< Normalized operator configuration. */
    SimStimulusSinusoidalMode mode;     /**< Active sinusoidal variant. */
    SimClockMode clock_mode;            /**< Clock mode used by the registered variant. */
    double locked_time;                 /**< Accumulated or locked clock time. */
    size_t last_step_index;             /**< Step index associated with locked_time. */
    bool clock_initialized;             /**< True once clock state has been initialized. */
    double snapshot_locked_time;        /**< Saved locked_time for drift restore. */
    size_t snapshot_last_step_index;    /**< Saved last_step_index for drift restore. */
    bool snapshot_clock_initialized;    /**< Saved clock initialization state for drift restore. */
    double *buffer;                     /**< Owned real-valued work buffer. */
    size_t buffer_capacity;             /**< Allocated element capacity for @ref buffer. */
    double *vdsp_block;                 /**< Owned vDSP block input buffer. */
    double *vdsp_theta;                 /**< Owned vDSP phase buffer. */
    double *vdsp_value;                 /**< Owned vDSP output buffer. */
    size_t vdsp_capacity;               /**< Allocated element capacity for vDSP buffers. */
    char symbolic[192];                 /**< Cached symbolic descriptor string. */
} SimStimulusSinusoidalState;

/**
 * @brief Register a traveling-wave sinusoidal stimulus operator.
 *
 * The implementation copies and normalizes @p config, selects the sine mode,
 * and registers the operator on the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional sinusoidal configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_sine_operator(struct SimContext *context,
                                         const SimStimulusSinusoidalConfig *config,
                                         size_t *out_index);

/**
 * @brief Register a standing-wave sinusoidal stimulus operator.
 *
 * The implementation copies and normalizes @p config, selects the standing-wave
 * mode, and registers the operator on the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional sinusoidal configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_standing_operator(struct SimContext *context,
                                             const SimStimulusSinusoidalConfig *config,
                                             size_t *out_index);

/**
 * @brief Register a chirped sinusoidal stimulus operator.
 *
 * The implementation copies and normalizes @p config, selects the chirp mode,
 * and registers the operator on the configured target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional sinusoidal configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_chirp_operator(struct SimContext *context,
                                          const SimStimulusSinusoidalConfig *config,
                                          size_t *out_index);

/**
 * @brief Copy the current sinusoidal configuration from a registered operator.
 *
 * This accessor is shared by the sine, standing-wave, and chirp variants.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by a sinusoidal registration call.
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_sinusoidal_config(struct SimContext *context, size_t operator_index,
                                         SimStimulusSinusoidalConfig *out_config);

/**
 * @brief Replace or renormalize a registered sinusoidal stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes waveform state and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the sinusoidal operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, buffer setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_sinusoidal_update(struct SimContext *context, size_t operator_index,
                                         const SimStimulusSinusoidalConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_SINUSOIDAL_H */
