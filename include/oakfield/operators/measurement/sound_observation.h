/**
 * @file sound_observation.h
 * @brief Audio observation operator for mapping field measurements to audio controls.
 */
#ifndef OAKFIELD_SOUND_OBSERVATION_H
#define OAKFIELD_SOUND_OBSERVATION_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Field-derived source used to drive a sound control.
 */
typedef enum SimSoundTranslationSource {
    SIM_SOUND_TRANSLATION_SOURCE_OFF = 0,           /**< Control source disabled. */
    SIM_SOUND_TRANSLATION_SOURCE_AMPLITUDE,         /**< Use field amplitude. */
    SIM_SOUND_TRANSLATION_SOURCE_PHASE,             /**< Use field phase. */
    SIM_SOUND_TRANSLATION_SOURCE_SPECTRAL_CENTROID, /**< Use spectral centroid estimate. */
    SIM_SOUND_TRANSLATION_SOURCE_FIELD              /**< Use raw sampled field value. */
} SimSoundTranslationSource;

/**
 * @brief Optional modulation source for a translated control.
 */
typedef enum SimSoundModulatorSource {
    SIM_SOUND_MODULATOR_NONE = 0,      /**< No modulation source. */
    SIM_SOUND_MODULATOR_SELF_FIELD,    /**< Modulate from the observed field. */
    SIM_SOUND_MODULATOR_EXTERNAL_FIELD /**< Modulate from the configured modulator field. */
} SimSoundModulatorSource;

/**
 * @brief Reduction used when sampling a field for sound observation.
 */
typedef enum SimSoundSamplingMode {
    SIM_SOUND_SAMPLING_POINT = 0,   /**< Sample one point. */
    SIM_SOUND_SAMPLING_MEAN,        /**< Mean over the selected samples. */
    SIM_SOUND_SAMPLING_RMS,         /**< Root-mean-square over selected samples. */
    SIM_SOUND_SAMPLING_PEAK,        /**< Signed peak over selected samples. */
    SIM_SOUND_SAMPLING_ABS_MEAN,    /**< Mean absolute value over selected samples. */
    SIM_SOUND_SAMPLING_ABS_PEAK,    /**< Peak absolute value over selected samples. */
    SIM_SOUND_SAMPLING_WINDOWED_RMS /**< Windowed root-mean-square over selected samples. */
} SimSoundSamplingMode;

/**
 * @brief Domain sampled by the observer.
 */
typedef enum SimSoundSamplingDomain {
    SIM_SOUND_SAMPLING_DOMAIN_PHYSICAL = 0, /**< Sample the physical-domain field. */
    SIM_SOUND_SAMPLING_DOMAIN_SPECTRAL      /**< Sample spectral-domain values. */
} SimSoundSamplingDomain;

/**
 * @brief Window function applied to multi-sample reductions.
 */
typedef enum SimSoundWindowType {
    SIM_SOUND_WINDOW_RECT = 0, /**< Rectangular window. */
    SIM_SOUND_WINDOW_HANN,     /**< Hann window. */
    SIM_SOUND_WINDOW_HAMMING,  /**< Hamming window. */
    SIM_SOUND_WINDOW_BLACKMAN, /**< Blackman window. */
    SIM_SOUND_WINDOW_KAISER    /**< Kaiser window. */
} SimSoundWindowType;

/**
 * @brief Mapping from pan control value to channel gains.
 */
typedef enum SimSoundPanLaw {
    SIM_SOUND_PAN_LAW_LINEAR = 0, /**< Linear pan gain law. */
    SIM_SOUND_PAN_LAW_EQUAL_POWER /**< Equal-power pan gain law. */
} SimSoundPanLaw;

/**
 * @brief Pitch-control scale used to derive output frequency.
 */
typedef enum SimSoundPitchScale {
    SIM_SOUND_PITCH_SCALE_LINEAR_HZ = 0, /**< Interpret pitch linearly in hertz. */
    SIM_SOUND_PITCH_SCALE_LOG2_HZ,       /**< Interpret pitch logarithmically base 2. */
    SIM_SOUND_PITCH_SCALE_MIDI           /**< Interpret pitch using MIDI note semantics. */
} SimSoundPitchScale;

/**
 * @brief Output representation produced by the observation operator.
 */
typedef enum SimSoundOutputMode {
    SIM_SOUND_OUTPUT_CONTROLS = 0, /**< Emit derived audio-control values. */
    SIM_SOUND_OUTPUT_RAW_SAMPLES   /**< Emit raw sampled audio frames. */
} SimSoundOutputMode;

/**
 * @brief Component sampled when emitting raw audio samples.
 */
typedef enum SimSoundRawSampleSource {
    SIM_SOUND_RAW_SOURCE_REAL = 0,  /**< Emit real component. */
    SIM_SOUND_RAW_SOURCE_IMAG,      /**< Emit imaginary component. */
    SIM_SOUND_RAW_SOURCE_MAGNITUDE, /**< Emit magnitude. */
    SIM_SOUND_RAW_SOURCE_PHASE      /**< Emit phase. */
} SimSoundRawSampleSource;

/**
 * @brief Channel layout used for raw audio sample output.
 */
typedef enum SimSoundRawChannelMode {
    SIM_SOUND_RAW_CHANNEL_MONO = 0,         /**< Emit one mono channel. */
    SIM_SOUND_RAW_CHANNEL_STEREO_DUPLICATE, /**< Duplicate mono samples to stereo. */
    SIM_SOUND_RAW_CHANNEL_STEREO_REAL_IMAG  /**< Emit real/imaginary as stereo channels. */
} SimSoundRawChannelMode;

/**
 * @brief Resampling kernel for converting field samples to raw audio samples.
 */
typedef enum SimSoundResampleMode {
    SIM_SOUND_RESAMPLE_NEAREST = 0, /**< Nearest-neighbor resampling. */
    SIM_SOUND_RESAMPLE_LINEAR,      /**< Linear resampling. */
    SIM_SOUND_RESAMPLE_CUBIC        /**< Cubic resampling. */
} SimSoundResampleMode;

/**
 * @brief Configuration for sound observation and audio-control extraction.
 */
typedef struct SimSoundObservationConfig {
    size_t input_field;             /**< Source field supplying samples. */
    size_t modulator_field;         /**< Optional modulator field (SIZE_MAX uses input_field). */
    SimSoundOutputMode output_mode; /**< Whether to emit controls or raw samples. */
    size_t output_bus;              /**< Destination audio bus index. */
    double output_send;             /**< Send amount for the destination bus. */
    bool output_pre_fader;          /**< True when the send is pre-fader. */
    SimSoundSamplingMode sampling_mode;     /**< Reduction used to sample the field. */
    SimSoundSamplingDomain sampling_domain; /**< Physical or spectral sampling domain. */
    SimSoundWindowType window_type;         /**< Window function for windowed reductions. */
    size_t window_length;                   /**< Number of samples in the reduction window. */
    size_t window_offset;                  /**< Offset of the reduction window from sample_index. */
    size_t sample_index;                   /**< Anchor sample index for point/windowed sampling. */
    SimSoundTranslationSource gain_source; /**< Source driving gain. */
    SimSoundModulatorSource gain_modulator;  /**< Optional gain modulation source. */
    double gain_base;                        /**< Base gain before source scaling. */
    double gain_scale;                       /**< Scale applied to gain source value. */
    double gain_min;                         /**< Minimum clamped gain. */
    double gain_max;                         /**< Maximum clamped gain. */
    SimSoundTranslationSource pan_source;    /**< Source driving pan. */
    SimSoundModulatorSource pan_modulator;   /**< Optional pan modulation source. */
    double pan_center;                       /**< Pan center value. */
    double pan_width;                        /**< Source scale applied around pan_center. */
    SimSoundPanLaw pan_law;                  /**< Pan law used for stereo gain mapping. */
    SimSoundTranslationSource pitch_source;  /**< Source driving pitch. */
    SimSoundModulatorSource pitch_modulator; /**< Optional pitch modulation source. */
    double pitch_base_hz;                    /**< Base pitch frequency in Hz. */
    double pitch_range_octaves;              /**< Pitch modulation range in octaves. */
    SimSoundPitchScale pitch_scale;          /**< Pitch scale interpretation. */
    SimSoundTranslationSource fm_source;     /**< Source driving frequency modulation. */
    SimSoundModulatorSource fm_modulator;    /**< Optional FM modulation source. */
    double fm_depth;                         /**< FM depth. */
    double fm_center;                        /**< FM center value. */
    double fm_ratio;                         /**< FM ratio multiplier. */
    double fm_clip;                          /**< Absolute FM clamp; <=0 disables. */
    double attack_ms;                        /**< Envelope attack time in milliseconds. */
    double release_ms;                       /**< Envelope release time in milliseconds. */
    double smoothing_tau;                    /**< Additional smoothing time constant. */
    bool scale_by_dt;                        /**< Scale output controls by substep dt when true. */
    SimSoundRawSampleSource raw_source;      /**< Field component used for raw samples. */
    SimSoundRawChannelMode raw_channel_mode; /**< Raw sample channel layout. */
    double raw_gain;                         /**< Gain applied to raw samples. */
    double raw_clip;                         /**< Absolute raw sample clamp; <=0 disables. */
    SimSoundResampleMode raw_resample_mode;  /**< Resampling kernel for raw output. */
} SimSoundObservationConfig;

/**
 * @brief Latest sound-control values produced by an observation operator.
 */
typedef struct SimSoundObservationTap {
    double gain;                    /**< Latest gain control. */
    double pan;                     /**< Latest pan control. */
    double pitch_hz;                /**< Latest pitch in Hz. */
    double fm;                      /**< Latest frequency-modulation control. */
    SimSoundOutputMode output_mode; /**< Output mode that produced this tap. */
    bool valid;                     /**< True when the tap contains initialized values. */
    size_t step_index;              /**< Simulation step that produced this tap. */
} SimSoundObservationTap;

/**
 * @brief Register a sound observation operator.
 *
 * The operator samples a field into audio-control taps or raw sample output
 * according to the configured sampling, modulation, and smoothing modes.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional sound observation configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, or split registration.
 */
SimResult sim_add_sound_observation_operator(struct SimContext *context,
                                             const SimSoundObservationConfig *config,
                                             size_t *out_index);

/**
 * @brief Copy the current sound observation configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_sound_observation_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no sound state.
 */
SimResult sim_sound_observation_config(struct SimContext *context, size_t operator_index,
                                       SimSoundObservationConfig *out_config);

/**
 * @brief Replace or renormalize a registered sound observation configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes runtime state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the sound observation operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, field
 *         validation, allocation, or state validation.
 */
SimResult sim_sound_observation_update(struct SimContext *context, size_t operator_index,
                                       const SimSoundObservationConfig *config);

/**
 * @brief Copy the latest sound-control tap from a registered observer.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_sound_observation_operator().
 * @param[out] out_tap Receives the latest tap snapshot.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no sound state.
 */
SimResult sim_sound_observation_tap(struct SimContext *context, size_t operator_index,
                                    SimSoundObservationTap *out_tap);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SOUND_OBSERVATION_H */
