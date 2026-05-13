#include "oakfield/operators/measurement/sound_observation.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/common/fft_plan.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define SOUND_OBSERVATION_SYMBOLIC_CAPACITY 192

typedef struct SimSoundObservationOperatorState {
    SimSoundObservationConfig config;
    double                    gain;
    double                    pan;
    double                    pitch_hz;
    double                    fm;
    FFTPlan                   fft_plan;
    double complex*           fft_input;
    double complex*           fft_output;
    size_t                    fft_count;
    size_t                    step_index;
    bool                      valid;
    char                      symbolic[SOUND_OBSERVATION_SYMBOLIC_CAPACITY];
} SimSoundObservationOperatorState;

static double sound_observation_sanitize(double value, double fallback) {
    return isfinite(value) ? value : fallback;
}

static double sound_observation_clamp(double value, double min_value, double max_value) {
    if (!isfinite(value)) {
        return min_value;
    }
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static void sound_observation_normalize_config(SimSoundObservationConfig* config) {
    if (!config) {
        return;
    }

    if (config->modulator_field == SIZE_MAX) {
        config->modulator_field = config->input_field;
    }

    if (!isfinite(config->output_send)) {
        config->output_send = 0.0;
    }
    if (config->output_send < 0.0) {
        config->output_send = 0.0;
    } else if (config->output_send > 1.0) {
        config->output_send = 1.0;
    }
    config->output_pre_fader = config->output_pre_fader ? true : false;
    config->scale_by_dt      = config->scale_by_dt ? true : false;

    {
        int mode_value = (int) config->sampling_mode;
        if (mode_value < (int) SIM_SOUND_SAMPLING_POINT ||
            mode_value > (int) SIM_SOUND_SAMPLING_WINDOWED_RMS) {
            config->sampling_mode = SIM_SOUND_SAMPLING_MEAN;
        }
    }
    {
        int domain_value = (int) config->sampling_domain;
        if (domain_value < (int) SIM_SOUND_SAMPLING_DOMAIN_PHYSICAL ||
            domain_value > (int) SIM_SOUND_SAMPLING_DOMAIN_SPECTRAL) {
            config->sampling_domain = SIM_SOUND_SAMPLING_DOMAIN_PHYSICAL;
        }
    }
    {
        int window_value = (int) config->window_type;
        if (window_value < (int) SIM_SOUND_WINDOW_RECT ||
            window_value > (int) SIM_SOUND_WINDOW_KAISER) {
            config->window_type = SIM_SOUND_WINDOW_RECT;
        }
    }
    {
        int source_value = (int) config->gain_source;
        if (source_value < (int) SIM_SOUND_TRANSLATION_SOURCE_OFF ||
            source_value > (int) SIM_SOUND_TRANSLATION_SOURCE_FIELD) {
            config->gain_source = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        }
    }
    {
        int mod_value = (int) config->gain_modulator;
        if (mod_value < (int) SIM_SOUND_MODULATOR_NONE ||
            mod_value > (int) SIM_SOUND_MODULATOR_EXTERNAL_FIELD) {
            config->gain_modulator = SIM_SOUND_MODULATOR_NONE;
        }
    }
    {
        int source_value = (int) config->pan_source;
        if (source_value < (int) SIM_SOUND_TRANSLATION_SOURCE_OFF ||
            source_value > (int) SIM_SOUND_TRANSLATION_SOURCE_FIELD) {
            config->pan_source = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        }
    }
    {
        int mod_value = (int) config->pan_modulator;
        if (mod_value < (int) SIM_SOUND_MODULATOR_NONE ||
            mod_value > (int) SIM_SOUND_MODULATOR_EXTERNAL_FIELD) {
            config->pan_modulator = SIM_SOUND_MODULATOR_NONE;
        }
    }
    {
        int pan_value = (int) config->pan_law;
        if (pan_value < (int) SIM_SOUND_PAN_LAW_LINEAR ||
            pan_value > (int) SIM_SOUND_PAN_LAW_EQUAL_POWER) {
            config->pan_law = SIM_SOUND_PAN_LAW_LINEAR;
        }
    }
    {
        int source_value = (int) config->pitch_source;
        if (source_value < (int) SIM_SOUND_TRANSLATION_SOURCE_OFF ||
            source_value > (int) SIM_SOUND_TRANSLATION_SOURCE_FIELD) {
            config->pitch_source = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        }
    }
    {
        int mod_value = (int) config->pitch_modulator;
        if (mod_value < (int) SIM_SOUND_MODULATOR_NONE ||
            mod_value > (int) SIM_SOUND_MODULATOR_EXTERNAL_FIELD) {
            config->pitch_modulator = SIM_SOUND_MODULATOR_NONE;
        }
    }
    {
        int scale_value = (int) config->pitch_scale;
        if (scale_value < (int) SIM_SOUND_PITCH_SCALE_LINEAR_HZ ||
            scale_value > (int) SIM_SOUND_PITCH_SCALE_MIDI) {
            config->pitch_scale = SIM_SOUND_PITCH_SCALE_LINEAR_HZ;
        }
    }
    {
        int source_value = (int) config->fm_source;
        if (source_value < (int) SIM_SOUND_TRANSLATION_SOURCE_OFF ||
            source_value > (int) SIM_SOUND_TRANSLATION_SOURCE_FIELD) {
            config->fm_source = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        }
    }
    {
        int mod_value = (int) config->fm_modulator;
        if (mod_value < (int) SIM_SOUND_MODULATOR_NONE ||
            mod_value > (int) SIM_SOUND_MODULATOR_EXTERNAL_FIELD) {
            config->fm_modulator = SIM_SOUND_MODULATOR_NONE;
        }
    }
    {
        int output_value = (int) config->output_mode;
        if (output_value < (int) SIM_SOUND_OUTPUT_CONTROLS ||
            output_value > (int) SIM_SOUND_OUTPUT_RAW_SAMPLES) {
            config->output_mode = SIM_SOUND_OUTPUT_CONTROLS;
        }
    }
    {
        int raw_value = (int) config->raw_source;
        if (raw_value < (int) SIM_SOUND_RAW_SOURCE_REAL ||
            raw_value > (int) SIM_SOUND_RAW_SOURCE_PHASE) {
            config->raw_source = SIM_SOUND_RAW_SOURCE_REAL;
        }
    }
    {
        int channel_value = (int) config->raw_channel_mode;
        if (channel_value < (int) SIM_SOUND_RAW_CHANNEL_MONO ||
            channel_value > (int) SIM_SOUND_RAW_CHANNEL_STEREO_REAL_IMAG) {
            config->raw_channel_mode = SIM_SOUND_RAW_CHANNEL_MONO;
        }
    }
    {
        int resample_value = (int) config->raw_resample_mode;
        if (resample_value < (int) SIM_SOUND_RESAMPLE_NEAREST ||
            resample_value > (int) SIM_SOUND_RESAMPLE_CUBIC) {
            config->raw_resample_mode = SIM_SOUND_RESAMPLE_LINEAR;
        }
    }

    config->gain_base  = sound_observation_sanitize(config->gain_base, 1.0);
    config->gain_scale = sound_observation_sanitize(config->gain_scale, 1.0);
    config->gain_min   = sound_observation_sanitize(config->gain_min, 0.0);
    config->gain_max   = sound_observation_sanitize(config->gain_max, 1.0);
    if (config->gain_min > config->gain_max) {
        double tmp       = config->gain_min;
        config->gain_min = config->gain_max;
        config->gain_max = tmp;
    }

    config->pan_center = sound_observation_sanitize(config->pan_center, 0.0);
    config->pan_width  = sound_observation_sanitize(config->pan_width, 1.0);
    if (config->pan_width < 0.0) {
        config->pan_width = fabs(config->pan_width);
    }

    config->pitch_base_hz = sound_observation_sanitize(config->pitch_base_hz, 440.0);
    if (config->pitch_base_hz < 0.0) {
        config->pitch_base_hz = 0.0;
    }
    config->pitch_range_octaves = sound_observation_sanitize(config->pitch_range_octaves, 1.0);
    if (config->pitch_range_octaves < 0.0) {
        config->pitch_range_octaves = 0.0;
    }

    config->fm_depth  = sound_observation_sanitize(config->fm_depth, 0.0);
    config->fm_center = sound_observation_sanitize(config->fm_center, 0.0);
    config->fm_ratio  = sound_observation_sanitize(config->fm_ratio, 1.0);
    config->fm_clip   = sound_observation_sanitize(config->fm_clip, 0.0);
    if (config->fm_clip < 0.0) {
        config->fm_clip = 0.0;
    }

    config->attack_ms     = sound_observation_sanitize(config->attack_ms, 0.0);
    config->release_ms    = sound_observation_sanitize(config->release_ms, 0.0);
    config->smoothing_tau = sound_observation_sanitize(config->smoothing_tau, 0.0);
    if (config->attack_ms < 0.0) {
        config->attack_ms = 0.0;
    }
    if (config->release_ms < 0.0) {
        config->release_ms = 0.0;
    }
    if (config->smoothing_tau < 0.0) {
        config->smoothing_tau = 0.0;
    }

    config->raw_gain = sound_observation_sanitize(config->raw_gain, 1.0);
    config->raw_clip = sound_observation_sanitize(config->raw_clip, 1.0);
    if (config->raw_clip < 0.0) {
        config->raw_clip = 0.0;
    }
}

static void sound_observation_refresh_symbolic(SimSoundObservationOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimSoundObservationConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "sound_observation gain=%.3g pan=%.3g pitch=%.3g fm=%.3g mode=%d",
                    cfg->gain_base,
                    cfg->pan_center,
                    cfg->pitch_base_hz,
                    cfg->fm_center,
                    (int) cfg->sampling_mode);
#else
    (void) state;
#endif
}

static const char* sound_observation_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimSoundObservationOperatorState* state =
        (const SimSoundObservationOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void sound_observation_release(void* state_ptr) {
    SimSoundObservationOperatorState* state = (SimSoundObservationOperatorState*) state_ptr;
    if (!state) {
        return;
    }

    fft_plan_destroy(&state->fft_plan);
    free(state->fft_input);
    free(state->fft_output);
    free(state);
}

static SimResult sound_observation_fft_prepare(SimSoundObservationOperatorState* state,
                                               size_t                            count) {
    if (!state) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (count == 0U) {
        fft_plan_destroy(&state->fft_plan);
        free(state->fft_input);
        free(state->fft_output);
        state->fft_input  = NULL;
        state->fft_output = NULL;
        state->fft_count  = 0U;
        return SIM_RESULT_OK;
    }

    if (state->fft_count == count && state->fft_input && state->fft_output) {
        return SIM_RESULT_OK;
    }

    fft_plan_destroy(&state->fft_plan);
    free(state->fft_input);
    free(state->fft_output);
    state->fft_input  = NULL;
    state->fft_output = NULL;
    state->fft_count  = 0U;

    SimResult rc = fft_plan_init(&state->fft_plan, count);
    if (rc != SIM_RESULT_OK) {
        fft_plan_destroy(&state->fft_plan);
        return rc;
    }

    state->fft_input  = (double complex*) calloc(count, sizeof(double complex));
    state->fft_output = (double complex*) calloc(count, sizeof(double complex));
    if (!state->fft_input || !state->fft_output) {
        free(state->fft_input);
        free(state->fft_output);
        state->fft_input  = NULL;
        state->fft_output = NULL;
        fft_plan_destroy(&state->fft_plan);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->fft_count = count;
    return SIM_RESULT_OK;
}

static bool sound_observation_field_info(const SimField*          field,
                                         const double**           real_data,
                                         const SimComplexDouble** complex_data,
                                         size_t*                  count_out,
                                         bool*                    is_complex_out) {
    if (!field || !real_data || !complex_data || !count_out || !is_complex_out) {
        return false;
    }

    *real_data      = NULL;
    *complex_data   = NULL;
    *count_out      = 0U;
    *is_complex_out = sim_field_is_complex(field);

    size_t bytes = sim_field_bytes(field);
    if (bytes == 0U) {
        return true;
    }

    if (*is_complex_out) {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return false;
        }
        *complex_data = sim_field_complex_data_const((SimField*) field);
        if (!*complex_data) {
            return false;
        }
        *count_out = bytes / sizeof(SimComplexDouble);
        return true;
    }

    if (field->element_size != sizeof(double)) {
        return false;
    }
    *real_data = (const double*) sim_field_data_const((SimField*) field);
    if (!*real_data) {
        return false;
    }
    *count_out = bytes / sizeof(double);
    return true;
}

static void sound_observation_window(const SimSoundObservationConfig* cfg,
                                     size_t                           count,
                                     size_t*                          start_out,
                                     size_t*                          length_out) {
    size_t start  = 0U;
    size_t length = 0U;

    if (count > 0U) {
        start = cfg->window_offset;
        if (start >= count) {
            start = 0U;
        }
        length = (cfg->window_length > 0U) ? cfg->window_length : count;
        if (length > count) {
            length = count;
        }
        if (start + length > count) {
            length = count - start;
        }
    }

    *start_out  = start;
    *length_out = length;
}

static double sound_observation_i0(double x) {
    double ax = fabs(x);
    if (ax < 3.75) {
        double y = x / 3.75;
        y *= y;
        return 1.0 +
               y * (3.5156229 +
                    y * (3.0899424 +
                         y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }

    double y = 3.75 / ax;
    return (exp(ax) / sqrt(ax)) *
           (0.39894228 +
            y * (0.01328592 +
                 y * (0.00225319 +
                      y * (-0.00157565 +
                           y * (0.00916281 +
                                y * (-0.02057706 +
                                     y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
}

static double
sound_observation_window_weight(SimSoundWindowType type, size_t index, size_t length) {
    if (length <= 1U) {
        return 1.0;
    }

    double n     = (double) index;
    double d     = (double) (length - 1U);
    double phase = 2.0 * M_PI * n / d;

    switch (type) {
        case SIM_SOUND_WINDOW_RECT:
            return 1.0;
        case SIM_SOUND_WINDOW_HANN:
            return 0.5 - 0.5 * cos(phase);
        case SIM_SOUND_WINDOW_HAMMING:
            return 0.54 - 0.46 * cos(phase);
        case SIM_SOUND_WINDOW_BLACKMAN:
            return 0.42 - 0.5 * cos(phase) + 0.08 * cos(2.0 * phase);
        case SIM_SOUND_WINDOW_KAISER: {
            double beta  = 8.6;
            double t     = (2.0 * n / d) - 1.0;
            double denom = sound_observation_i0(beta);
            double arg   = beta * sqrt(fmax(0.0, 1.0 - t * t));
            double num   = sound_observation_i0(arg);
            return (denom > 0.0) ? (num / denom) : 1.0;
        }
        default:
            return 1.0;
    }
}

static double sound_observation_phase_sample(double re, double im, bool is_complex, bool* valid) {
    double phase = 0.0;
    if (is_complex) {
        if (!isfinite(re) || !isfinite(im)) {
            if (valid) {
                *valid = false;
            }
            return 0.0;
        }
        if (im == 0.0 && re < 0.0) {
            phase = -M_PI;
        } else {
            phase = atan2(im, re);
        }
    } else {
        if (!isfinite(re)) {
            if (valid) {
                *valid = false;
            }
            return 0.0;
        }
        phase = (re < 0.0) ? -M_PI : 0.0;
    }
    if (valid) {
        *valid = true;
    }
    return phase / M_PI;
}

static double sound_observation_sample_scalar(const SimSoundObservationConfig* cfg,
                                              const SimField*                  field,
                                              bool                             prefer_abs,
                                              bool*                            out_valid) {
    const double*           real_data    = NULL;
    const SimComplexDouble* complex_data = NULL;
    size_t                  count        = 0U;
    bool                    is_complex   = false;

    if (!sound_observation_field_info(field, &real_data, &complex_data, &count, &is_complex)) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    if (count == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    SimSoundSamplingMode mode = cfg->sampling_mode;
    if (mode == SIM_SOUND_SAMPLING_POINT) {
        size_t index = cfg->sample_index;
        if (index >= count) {
            index = count - 1U;
        }

        double value = 0.0;
        if (is_complex) {
            double re = complex_data[index].re;
            double im = complex_data[index].im;
            value     = hypot(re, im);
        } else {
            value = prefer_abs ? fabs(real_data[index]) : real_data[index];
        }

        if (!isfinite(value)) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }

        if (out_valid) {
            *out_valid = true;
        }
        return value;
    }

    size_t start  = 0U;
    size_t length = 0U;
    sound_observation_window(cfg, count, &start, &length);
    if (length == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    bool force_abs = (mode == SIM_SOUND_SAMPLING_ABS_MEAN || mode == SIM_SOUND_SAMPLING_ABS_PEAK);
    bool use_abs   = prefer_abs || force_abs;

    double sum         = 0.0;
    double sum_sq      = 0.0;
    double weight_sum  = 0.0;
    double max_abs     = 0.0;
    double peak_value  = 0.0;
    size_t valid_count = 0U;
    bool   have_peak   = false;

    for (size_t offset = 0U; offset < length; ++offset) {
        size_t index = start + offset;
        double value = 0.0;

        if (is_complex) {
            double re = complex_data[index].re;
            double im = complex_data[index].im;
            value     = hypot(re, im);
        } else {
            value = use_abs ? fabs(real_data[index]) : real_data[index];
        }

        if (!isfinite(value)) {
            continue;
        }

        switch (mode) {
            case SIM_SOUND_SAMPLING_MEAN:
            case SIM_SOUND_SAMPLING_ABS_MEAN:
                sum += value;
                valid_count += 1U;
                break;
            case SIM_SOUND_SAMPLING_RMS:
                sum_sq += value * value;
                valid_count += 1U;
                break;
            case SIM_SOUND_SAMPLING_PEAK:
            case SIM_SOUND_SAMPLING_ABS_PEAK: {
                double abs_value = fabs(value);
                if (!have_peak || abs_value > max_abs) {
                    max_abs = abs_value;
                    peak_value =
                        (mode == SIM_SOUND_SAMPLING_ABS_PEAK || use_abs) ? abs_value : value;
                    have_peak = true;
                }
                break;
            }
            case SIM_SOUND_SAMPLING_WINDOWED_RMS: {
                double weight = sound_observation_window_weight(cfg->window_type, offset, length);
                if (!isfinite(weight) || weight <= 0.0) {
                    break;
                }
                sum_sq += weight * value * value;
                weight_sum += weight;
                break;
            }
            default:
                break;
        }
    }

    if (mode == SIM_SOUND_SAMPLING_MEAN || mode == SIM_SOUND_SAMPLING_ABS_MEAN) {
        if (valid_count == 0U) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }
        if (out_valid) {
            *out_valid = true;
        }
        return sum / (double) valid_count;
    }

    if (mode == SIM_SOUND_SAMPLING_RMS) {
        if (valid_count == 0U) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }
        if (out_valid) {
            *out_valid = true;
        }
        return sqrt(sum_sq / (double) valid_count);
    }

    if (mode == SIM_SOUND_SAMPLING_WINDOWED_RMS) {
        if (weight_sum <= 0.0) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }
        if (out_valid) {
            *out_valid = true;
        }
        return sqrt(sum_sq / weight_sum);
    }

    if (out_valid) {
        *out_valid = have_peak;
    }
    return have_peak ? peak_value : 0.0;
}

static double sound_observation_sample_phase(const SimSoundObservationConfig* cfg,
                                             const SimField*                  field,
                                             bool*                            out_valid) {
    const double*           real_data    = NULL;
    const SimComplexDouble* complex_data = NULL;
    size_t                  count        = 0U;
    bool                    is_complex   = false;

    if (!sound_observation_field_info(field, &real_data, &complex_data, &count, &is_complex)) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    if (count == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    SimSoundSamplingMode mode = cfg->sampling_mode;
    if (mode == SIM_SOUND_SAMPLING_POINT) {
        size_t index = cfg->sample_index;
        if (index >= count) {
            index = count - 1U;
        }
        if (is_complex) {
            return sound_observation_phase_sample(
                complex_data[index].re, complex_data[index].im, true, out_valid);
        }
        return sound_observation_phase_sample(real_data[index], 0.0, false, out_valid);
    }

    size_t start  = 0U;
    size_t length = 0U;
    sound_observation_window(cfg, count, &start, &length);
    if (length == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    if (mode == SIM_SOUND_SAMPLING_PEAK || mode == SIM_SOUND_SAMPLING_ABS_PEAK) {
        double max_abs    = 0.0;
        bool   have_peak  = false;
        size_t peak_index = start;
        for (size_t offset = 0U; offset < length; ++offset) {
            size_t index     = start + offset;
            double magnitude = 0.0;
            if (is_complex) {
                double re = complex_data[index].re;
                double im = complex_data[index].im;
                magnitude = hypot(re, im);
            } else {
                magnitude = fabs(real_data[index]);
            }
            if (!isfinite(magnitude)) {
                continue;
            }
            if (!have_peak || magnitude > max_abs) {
                max_abs    = magnitude;
                peak_index = index;
                have_peak  = true;
            }
        }
        if (!have_peak) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }
        if (is_complex) {
            return sound_observation_phase_sample(
                complex_data[peak_index].re, complex_data[peak_index].im, true, out_valid);
        }
        return sound_observation_phase_sample(real_data[peak_index], 0.0, false, out_valid);
    }

    double sum_re      = 0.0;
    double sum_im      = 0.0;
    size_t valid_count = 0U;
    for (size_t offset = 0U; offset < length; ++offset) {
        size_t index = start + offset;
        double re    = 0.0;
        double im    = 0.0;

        if (is_complex) {
            re = complex_data[index].re;
            im = complex_data[index].im;
        } else {
            re = real_data[index];
        }
        if (!isfinite(re) || !isfinite(im)) {
            continue;
        }
        sum_re += re;
        sum_im += im;
        valid_count += 1U;
    }

    if (valid_count == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    return sound_observation_phase_sample(sum_re, sum_im, true, out_valid);
}

static double sound_observation_sample_spectral_centroid(SimSoundObservationOperatorState* state,
                                                         const SimField*                   field,
                                                         bool* out_valid) {
    if (!state) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    const SimSoundObservationConfig* cfg          = &state->config;
    const double*                    real_data    = NULL;
    const SimComplexDouble*          complex_data = NULL;
    size_t                           count        = 0U;
    bool                             is_complex   = false;

    if (!sound_observation_field_info(field, &real_data, &complex_data, &count, &is_complex)) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    if (count == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    size_t start  = 0U;
    size_t length = 0U;
    sound_observation_window(cfg, count, &start, &length);
    if (length == 0U) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    double sum_power    = 0.0;
    double sum_weighted = 0.0;

    if (cfg->sampling_domain == SIM_SOUND_SAMPLING_DOMAIN_PHYSICAL) {
        SimResult rc = sound_observation_fft_prepare(state, length);
        if (rc != SIM_RESULT_OK) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }

        for (size_t offset = 0U; offset < length; ++offset) {
            size_t index = start + offset;
            double re    = 0.0;
            double im    = 0.0;
            if (is_complex) {
                re = complex_data[index].re;
                im = complex_data[index].im;
            } else {
                re = real_data[index];
            }

            if (!isfinite(re) || !isfinite(im)) {
                re = 0.0;
                im = 0.0;
            }

            double weight = sound_observation_window_weight(cfg->window_type, offset, length);
            if (!isfinite(weight)) {
                weight = 0.0;
            }
            state->fft_input[offset] = CMPLX(re * weight, im * weight);
        }

        rc = fft_plan_forward(&state->fft_plan, state->fft_input, state->fft_output);
        if (rc != SIM_RESULT_OK) {
            if (out_valid) {
                *out_valid = false;
            }
            return 0.0;
        }

        size_t bins = length;
        if (!is_complex) {
            bins = length / 2U + 1U;
        }

        double denom = (bins > 1U) ? (double) (bins - 1U) : 1.0;
        for (size_t bin = 0U; bin < bins; ++bin) {
            double re    = creal(state->fft_output[bin]);
            double im    = cimag(state->fft_output[bin]);
            double power = re * re + im * im;
            if (!isfinite(power) || power <= 0.0) {
                continue;
            }
            double norm_index = (denom > 0.0) ? ((double) bin / denom) : 0.0;
            sum_power += power;
            sum_weighted += norm_index * power;
        }
    } else {
        double denom = (length > 1U) ? (double) (length - 1U) : 1.0;
        for (size_t offset = 0U; offset < length; ++offset) {
            size_t index     = start + offset;
            double magnitude = 0.0;
            if (is_complex) {
                double re = complex_data[index].re;
                double im = complex_data[index].im;
                magnitude = hypot(re, im);
            } else {
                magnitude = fabs(real_data[index]);
            }
            if (!isfinite(magnitude)) {
                continue;
            }
            double power      = magnitude * magnitude;
            double norm_index = (denom > 0.0) ? ((double) offset / denom) : 0.0;
            sum_power += power;
            sum_weighted += norm_index * power;
        }
    }

    if (sum_power <= 0.0) {
        if (out_valid) {
            *out_valid = false;
        }
        return 0.0;
    }

    if (out_valid) {
        *out_valid = true;
    }
    return sum_weighted / sum_power;
}

static const SimField* sound_observation_modulator_field(const SimField*         input,
                                                         const SimField*         modulator,
                                                         SimSoundModulatorSource source) {
    if (source == SIM_SOUND_MODULATOR_SELF_FIELD) {
        return input;
    }
    if (source == SIM_SOUND_MODULATOR_EXTERNAL_FIELD) {
        return modulator;
    }
    return NULL;
}

static double sound_observation_source_value(const SimSoundObservationConfig*  cfg,
                                             SimSoundObservationOperatorState* state,
                                             const SimField*                   field,
                                             SimSoundTranslationSource         source,
                                             bool*                             out_valid) {
    switch (source) {
        case SIM_SOUND_TRANSLATION_SOURCE_AMPLITUDE:
            return sound_observation_sample_scalar(cfg, field, true, out_valid);
        case SIM_SOUND_TRANSLATION_SOURCE_PHASE:
            return sound_observation_sample_phase(cfg, field, out_valid);
        case SIM_SOUND_TRANSLATION_SOURCE_SPECTRAL_CENTROID:
            return sound_observation_sample_spectral_centroid(state, field, out_valid);
        case SIM_SOUND_TRANSLATION_SOURCE_FIELD:
            return sound_observation_sample_scalar(cfg, field, false, out_valid);
        case SIM_SOUND_TRANSLATION_SOURCE_OFF:
        default:
            if (out_valid) {
                *out_valid = true;
            }
            return 0.0;
    }
}

static double sound_observation_modulator_value(const SimSoundObservationConfig* cfg,
                                                const SimField*                  input,
                                                const SimField*                  modulator,
                                                SimSoundModulatorSource          source) {
    if (source == SIM_SOUND_MODULATOR_NONE) {
        return 1.0;
    }

    const SimField* field = sound_observation_modulator_field(input, modulator, source);
    if (!field) {
        return 1.0;
    }

    bool   valid = false;
    double value = sound_observation_sample_scalar(cfg, field, false, &valid);
    if (!valid || !isfinite(value)) {
        return 1.0;
    }
    return value;
}

static double sound_observation_map_pitch(double             base_hz,
                                          double             range_octaves,
                                          SimSoundPitchScale scale,
                                          double             feature) {
    if (!isfinite(feature)) {
        feature = 0.0;
    }

    if (base_hz <= 0.0) {
        return 0.0;
    }

    if (scale == SIM_SOUND_PITCH_SCALE_LINEAR_HZ) {
        double span = base_hz * (pow(2.0, range_octaves) - 1.0);
        return base_hz + feature * span;
    }

    if (scale == SIM_SOUND_PITCH_SCALE_MIDI) {
        double base_midi = 69.0 + 12.0 * (log(base_hz / 440.0) / log(2.0));
        double midi      = base_midi + feature * range_octaves * 12.0;
        return 440.0 * pow(2.0, (midi - 69.0) / 12.0);
    }

    return base_hz * pow(2.0, feature * range_octaves);
}

static double sound_observation_one_pole(double prev, double target, double tau, double dt) {
    if (tau <= 0.0 || dt <= 0.0 || !isfinite(prev) || !isfinite(target)) {
        return target;
    }

    double alpha = exp(-dt / tau);
    return alpha * prev + (1.0 - alpha) * target;
}

static double sound_observation_attack_release(double prev,
                                               double target,
                                               double attack_ms,
                                               double release_ms,
                                               double dt) {
    if (dt <= 0.0 || !isfinite(prev) || !isfinite(target)) {
        return target;
    }

    double tau_ms = (target > prev) ? attack_ms : release_ms;
    if (tau_ms <= 0.0) {
        return target;
    }

    return sound_observation_one_pole(prev, target, tau_ms * 0.001, dt);
}

static double sound_observation_smooth_value(const SimSoundObservationConfig* cfg,
                                             double                           prev,
                                             double                           target,
                                             double                           dt) {
    if (cfg->smoothing_tau > 0.0) {
        return sound_observation_one_pole(prev, target, cfg->smoothing_tau, dt);
    }

    if (cfg->attack_ms > 0.0 || cfg->release_ms > 0.0) {
        return sound_observation_attack_release(prev, target, cfg->attack_ms, cfg->release_ms, dt);
    }

    return target;
}

static SimResult sound_observation_apply(void*               state_ptr,
                                         struct SimContext*  context,
                                         struct SimOperator* self,
                                         double              dt) {
    (void) self;
    (void) dt;

    SimSoundObservationOperatorState* state = (SimSoundObservationOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_context_in_drift(context)) {
        return SIM_RESULT_OK;
    }

    const SimSoundObservationConfig* cfg       = &state->config;
    SimField*                        input     = sim_context_field(context, cfg->input_field);
    SimField*                        modulator = sim_context_field(context, cfg->modulator_field);

    if (!input || !modulator) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   sample_valid = false;
    bool   any_valid    = false;
    bool   had_prev     = state->valid;
    double gain_feature =
        sound_observation_source_value(cfg, state, input, cfg->gain_source, &sample_valid);
    any_valid |= sample_valid;
    double gain_mod = sound_observation_modulator_value(cfg, input, modulator, cfg->gain_modulator);
    gain_feature *= gain_mod;
    double gain_target = cfg->gain_base + cfg->gain_scale * gain_feature;
    gain_target        = sound_observation_clamp(gain_target, cfg->gain_min, cfg->gain_max);

    double pan_feature =
        sound_observation_source_value(cfg, state, input, cfg->pan_source, &sample_valid);
    any_valid |= sample_valid;
    double pan_mod = sound_observation_modulator_value(cfg, input, modulator, cfg->pan_modulator);
    pan_feature *= pan_mod;
    double pan_target = cfg->pan_center + cfg->pan_width * pan_feature;

    double pitch_feature =
        sound_observation_source_value(cfg, state, input, cfg->pitch_source, &sample_valid);
    any_valid |= sample_valid;
    double pitch_mod =
        sound_observation_modulator_value(cfg, input, modulator, cfg->pitch_modulator);
    pitch_feature *= pitch_mod;
    double pitch_target = sound_observation_map_pitch(
        cfg->pitch_base_hz, cfg->pitch_range_octaves, cfg->pitch_scale, pitch_feature);

    double fm_feature =
        sound_observation_source_value(cfg, state, input, cfg->fm_source, &sample_valid);
    any_valid |= sample_valid;
    double fm_mod = sound_observation_modulator_value(cfg, input, modulator, cfg->fm_modulator);
    fm_feature *= fm_mod;
    double fm_target = cfg->fm_center + cfg->fm_depth * fm_feature;
    fm_target *= cfg->fm_ratio;
    if (cfg->fm_clip > 0.0) {
        fm_target = sound_observation_clamp(fm_target, -cfg->fm_clip, cfg->fm_clip);
    }

    if (!isfinite(pan_target)) {
        pan_target = 0.0;
    }
    if (!isfinite(pitch_target) || pitch_target < 0.0) {
        pitch_target = 0.0;
    }
    if (!isfinite(fm_target)) {
        fm_target = 0.0;
    }

    double gain     = gain_target;
    double pan      = pan_target;
    double pitch_hz = pitch_target;
    double fm       = fm_target;
    if (had_prev && any_valid) {
        gain     = sound_observation_smooth_value(cfg, state->gain, gain_target, dt);
        pan      = sound_observation_smooth_value(cfg, state->pan, pan_target, dt);
        pitch_hz = sound_observation_smooth_value(cfg, state->pitch_hz, pitch_target, dt);
        fm       = sound_observation_smooth_value(cfg, state->fm, fm_target, dt);
    } else if (had_prev && !any_valid) {
        gain     = state->gain;
        pan      = state->pan;
        pitch_hz = state->pitch_hz;
        fm       = state->fm;
    }

    state->gain       = gain;
    state->pan        = pan;
    state->pitch_hz   = pitch_hz;
    state->fm         = fm;
    state->step_index = sim_context_step_index(context);
    state->valid      = any_valid || had_prev;

    return SIM_RESULT_OK;
}

static SimResult sound_observation_step(void*               state_ptr,
                                        struct SimContext*  context,
                                        struct SimOperator* self,
                                        size_t              substep_index,
                                        double              dt_sub,
                                        void*               scratch,
                                        size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return sound_observation_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_sound_observation_operator(struct SimContext*               context,
                                             const SimSoundObservationConfig* config,
                                             size_t*                          out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimSoundObservationOperatorState* state =
        (SimSoundObservationOperatorState*) calloc(1U, sizeof(SimSoundObservationOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimSoundObservationConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.input_field         = 0U;
        local.modulator_field     = SIZE_MAX;
        local.output_mode         = SIM_SOUND_OUTPUT_CONTROLS;
        local.output_bus          = 0U;
        local.output_send         = 1.0;
        local.output_pre_fader    = false;
        local.sampling_mode       = SIM_SOUND_SAMPLING_MEAN;
        local.sampling_domain     = SIM_SOUND_SAMPLING_DOMAIN_PHYSICAL;
        local.window_type         = SIM_SOUND_WINDOW_RECT;
        local.window_length       = 0U;
        local.window_offset       = 0U;
        local.sample_index        = 0U;
        local.gain_source         = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        local.gain_modulator      = SIM_SOUND_MODULATOR_NONE;
        local.gain_base           = 1.0;
        local.gain_scale          = 1.0;
        local.gain_min            = 0.0;
        local.gain_max            = 1.0;
        local.pan_source          = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        local.pan_modulator       = SIM_SOUND_MODULATOR_NONE;
        local.pan_center          = 0.0;
        local.pan_width           = 1.0;
        local.pan_law             = SIM_SOUND_PAN_LAW_LINEAR;
        local.pitch_source        = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        local.pitch_modulator     = SIM_SOUND_MODULATOR_NONE;
        local.pitch_base_hz       = 440.0;
        local.pitch_range_octaves = 1.0;
        local.pitch_scale         = SIM_SOUND_PITCH_SCALE_LINEAR_HZ;
        local.fm_source           = SIM_SOUND_TRANSLATION_SOURCE_OFF;
        local.fm_modulator        = SIM_SOUND_MODULATOR_NONE;
        local.fm_depth            = 0.0;
        local.fm_center           = 0.0;
        local.fm_ratio            = 1.0;
        local.fm_clip             = 0.0;
        local.attack_ms           = 0.0;
        local.release_ms          = 0.0;
        local.smoothing_tau       = 0.0;
        local.scale_by_dt         = true;
        local.raw_source          = SIM_SOUND_RAW_SOURCE_REAL;
        local.raw_channel_mode    = SIM_SOUND_RAW_CHANNEL_MONO;
        local.raw_gain            = 1.0;
        local.raw_clip            = 1.0;
        local.raw_resample_mode   = SIM_SOUND_RESAMPLE_LINEAR;
    }

    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "sound_observation",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);
    sound_observation_normalize_config(&local);
    state->config = local;
    state->valid  = false;
    sound_observation_refresh_symbolic(state);

    SimField* input_field     = sim_context_field(context, state->config.input_field);
    SimField* modulator_field = sim_context_field(context, state->config.modulator_field);
    if (!input_field || !modulator_field) {
        sound_observation_release(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "sound_observation");

    bool            input_complex     = sim_field_is_complex(input_field);
    bool            modulator_complex = sim_field_is_complex(modulator_field);
    bool            needs_complex     = input_complex || modulator_complex;
    SimOperatorInfo info              = sim_operator_info_defaults();
    info.category                     = SIM_OPERATOR_CATEGORY_MEASUREMENT;
    info.warp_level                   = SIM_WARP_LEVEL_NONE;
    info.is_noise                     = false;
    info.is_spectral       = (state->config.sampling_domain == SIM_SOUND_SAMPLING_DOMAIN_SPECTRAL);
    info.is_local          = (state->config.sampling_domain == SIM_SOUND_SAMPLING_DOMAIN_PHYSICAL &&
                              state->config.sampling_mode == SIM_SOUND_SAMPLING_POINT);
    info.is_nonlocal       = !info.is_local;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "sound_observation";
    sim_operator_info_set_schema_identity(&info, "sound_observation");
    info.algebraic_flags = SIM_OPERATOR_ALG_NONE;
    info.representation.domain =
        info.is_spectral ? SIM_FIELD_DOMAIN_SPECTRAL : SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimSplitPort ports[2] = { { .context_field_index = state->config.input_field,
                                .require_complex     = input_complex },
                              { .context_field_index = state->config.modulator_field,
                                .require_complex     = modulator_complex } };

    SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_READ } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = sound_observation_step,
                                .accesses          = accesses,
                                .access_count      = 2U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 2U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = sound_observation_symbolic,
                                .destroy       = sound_observation_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        sound_observation_release(state);
    }

    return result;
}

SimResult sim_sound_observation_config(struct SimContext*         context,
                                       size_t                     operator_index,
                                       SimSoundObservationConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSoundObservationOperatorState* state =
        (SimSoundObservationOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_sound_observation_update(struct SimContext*               context,
                                       size_t                           operator_index,
                                       const SimSoundObservationConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSoundObservationOperatorState* state =
        (SimSoundObservationOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimSoundObservationConfig local = state->config;
    if (config) {
        local             = *config;
        local.scale_by_dt = sim_operator_resolve_scale_by_dt(
            context, "sound_observation", true, config->scale_by_dt);
    }

    sound_observation_normalize_config(&local);
    state->config = local;
    sound_observation_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}

SimResult sim_sound_observation_tap(struct SimContext*      context,
                                    size_t                  operator_index,
                                    SimSoundObservationTap* out_tap) {
    if (!context || !out_tap) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSoundObservationOperatorState* state =
        (SimSoundObservationOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    out_tap->gain        = state->gain;
    out_tap->pan         = state->pan;
    out_tap->pitch_hz    = state->pitch_hz;
    out_tap->fm          = state->fm;
    out_tap->output_mode = state->config.output_mode;
    out_tap->valid       = state->valid;
    out_tap->step_index  = state->step_index;
    return SIM_RESULT_OK;
}
