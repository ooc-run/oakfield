/**
 * @file digamma_square.h
 * @brief Digamma square stimulus operators
 *
 * Complex fields are driven by writing real/imag components with optional rotation.
 */
#ifndef OAKFIELD_STIMULUS_DIGAMMA_SQUARE_H
#define OAKFIELD_STIMULUS_DIGAMMA_SQUARE_H

#include "coords.h"
#include "oakfield/math/special_functions.h"
#include "oakfield/operator_split.h"
#include "oakfield/operators/coupling/mixer.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Waveform shape for digamma square stimulus.
 */
typedef enum SimDigammaSquareWaveformShape {
    SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT = 0, /**< Default square-like digamma waveform. */
    SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE,    /**< Triangle-like waveform variant. */
    SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH     /**< Sawtooth-like waveform variant. */
} SimDigammaSquareWaveformShape;

/**
 * @brief Shared configuration for digamma square stimulus variants.
 */
typedef struct SimStimulusDigammaSquareConfig {
    size_t field_index;           /**< Target field index. */
    size_t warp_field_index;      /**< Optional warp field index for spatial modulation. */
    double amplitude;             /**< Signal amplitude. */
    double wavenumber;            /**< Base spatial wavenumber (rad / unit). */
    double kx;                    /**< Optional wavevector X component (rad / unit). */
    double ky;                    /**< Optional wavevector Y component (rad / unit). */
    double omega;                 /**< Base angular frequency (rad / s). */
    double phase;                 /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double time_offset;           /**< Additional time offset applied before evaluation. */
    double nominal_dt; /**< Optional nominal dt when fixed_clock is enabled (<=0 uses actual dt). */
    double velocity;   /**< Advection speed for traveling Gaussian envelope (units / s). */
    double harmonics;  /**< Number of harmonics for square wave approximation. */
    double a;          /**< Deformation shift parameter (default 0.25). */
    double rotation;   /**< Rotation applied when writing into complex fields (radians). */
    double tolerance;  /**< Absolute tolerance for adaptive digamma/trigamma (if used). */
    bool fixed_clock;  /**< Lock the driving clock to nominal_dt instead of adaptive dt. */
    bool scale_by_dt;  /**< When true, scale writes by substep dt; false = dt-independent signal. */
    bool use_wavevector;                 /**< When true, use (kx,ky) instead of wavenumber+coord. */
    bool use_warp;                       /**< When true, modulate bc by warp field sample. */
    double warp_mix;                     /**< Mixing factor for warp modulation. */
    double warp_bias;                    /**< Bias added to warp field sample before modulation. */
    SimMixerMode warp_mode;              /**< Warp mixing strategy (sum/multiply/crossfade). */
    SimDigammaBackend backend;           /**< Backend choice for digamma/trigamma evaluation. */
    SimDigammaSquareWaveformShape shape; /**< Waveform shape. */
} SimStimulusDigammaSquareConfig;

/**
 * @brief State for digamma square stimulus.
 */
typedef struct SimDigammaSquareState {
    SimStimulusDigammaSquareConfig config; /**< Normalized operator configuration. */
    SimClockMode clock_mode;               /**< Clock mode used by the registered variant. */
    double locked_time;                    /**< Accumulated or locked clock time. */
    size_t last_step_index;                /**< Step index associated with locked_time. */
    bool clock_initialized;                /**< True once clock state has been initialized. */
    double *buffer;                        /**< Owned real-valued work buffer. */
    size_t buffer_capacity;                /**< Allocated element capacity for @ref buffer. */
    SimComplexDouble *buffer_complex;      /**< Owned complex-valued work buffer. */
    size_t buffer_complex_capacity;        /**< Allocated element capacity for complex buffer. */
#if defined(SIM_HAVE_VDSP)
    double *vdsp_block;   /**< Owned vDSP block input buffer. */
    double *vdsp_phase;   /**< Owned vDSP phase buffer. */
    double *vdsp_sin;     /**< Owned vDSP sine output buffer. */
    double *vdsp_cos;     /**< Owned vDSP cosine output buffer. */
    size_t vdsp_capacity; /**< Allocated element capacity for vDSP buffers. */
#endif
    char symbolic[192]; /**< Cached symbolic descriptor string. */
} SimDigammaSquareState;

/**
 * @brief Register a digamma square stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers the
 * default digamma-square waveform as a split operator targeting the configured
 * field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional digamma-square configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_digamma_square_operator(struct SimContext *context,
                                                   const SimStimulusDigammaSquareConfig *config,
                                                   size_t *out_index);

/**
 * @brief Register a digamma-square operator while overriding the target field.
 *
 * This compatibility wrapper copies @p config when supplied, replaces its
 * field_index with @p field_index, and registers the legacy operator name.
 *
 * @param context Simulation context that will own the operator.
 * @param field_index Target field index written by the operator.
 * @param config Optional configuration values other than the target field.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_digamma_square_operator(struct SimContext *context, size_t field_index,
                                          const SimStimulusDigammaSquareConfig *config,
                                          size_t *out_index);

/**
 * @brief Copy the current digamma-square configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by a digamma-square registration call.
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_digamma_square_config(struct SimContext *context, size_t operator_index,
                                             SimStimulusDigammaSquareConfig *out_config);

/**
 * @brief Replace or renormalize a registered digamma-square stimulus configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic and cached runtime state
 * before invalidating the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the digamma-square operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, buffer setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_digamma_square_update(struct SimContext *context, size_t operator_index,
                                             const SimStimulusDigammaSquareConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_DIGAMMA_SQUARE_H */
