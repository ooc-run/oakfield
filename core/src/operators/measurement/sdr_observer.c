#include "oakfield/operators/measurement/sdr_observer.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define SDR_OBS_EPS 1.0e-12
#define SDR_OBS_NORM 127.5
#define SDR_GAIN_SCALE 10 /* rtlsdr gain is in tenths of dB */

#if OAKFIELD_ENABLE_RTLSDR
#if SIM_HAVE_RTL_SDR_H
#include <rtl-sdr.h>
#else
#include <rtlsdr.h>
#endif
#endif

typedef struct SimSdrObserverState {
    SimSdrObserverConfig config;
    SimSdrObserverStatus status;
    char                 symbolic[128];
    double               phase_accum;
    double               mix_phase_accum;
    SimComplexDouble     prev_sample;
    bool                 prev_sample_valid;
    bool                 device_open;
#if OAKFIELD_ENABLE_RTLSDR
    rtlsdr_dev_t* dev;
    uint8_t*      iq_buf;
    size_t        iq_buf_bytes;
#endif
} SimSdrObserverState;

const char* sim_sdr_observer_backend_mode_name(SimSdrObserverBackendMode mode) {
    switch (mode) {
        case SIM_SDR_OBSERVER_BACKEND_SYNTHETIC:
            return "synthetic";
        case SIM_SDR_OBSERVER_BACKEND_RTL_SDR:
            return "rtl_sdr";
        case SIM_SDR_OBSERVER_BACKEND_UNKNOWN:
        default:
            break;
    }
    return "unknown";
}

const char* sim_sdr_observer_fallback_reason_name(SimSdrObserverFallbackReason reason) {
    switch (reason) {
        case SIM_SDR_OBSERVER_FALLBACK_RTLSDR_DISABLED:
            return "rtlsdr_disabled";
        case SIM_SDR_OBSERVER_FALLBACK_DEVICE_OPEN_FAILED:
            return "device_open_failed";
        case SIM_SDR_OBSERVER_FALLBACK_DEVICE_CONFIG_FAILED:
            return "device_config_failed";
        case SIM_SDR_OBSERVER_FALLBACK_BUFFER_ALLOCATION_FAILED:
            return "buffer_allocation_failed";
        case SIM_SDR_OBSERVER_FALLBACK_READ_FAILED:
            return "read_failed";
        case SIM_SDR_OBSERVER_FALLBACK_NONE:
        default:
            break;
    }
    return "none";
}

static void sdr_observer_normalize_config(SimSdrObserverConfig* cfg) {
    if (cfg == NULL) {
        return;
    }
    if (!isfinite(cfg->center_freq) || cfg->center_freq < 0.0) {
        cfg->center_freq = 100.0e6;
    }
    if (!isfinite(cfg->sample_rate) || cfg->sample_rate < 1.0) {
        cfg->sample_rate = 2.048e6;
    }
    if (!isfinite(cfg->gain) || cfg->gain < 0.0) {
        cfg->gain = 0.0;
    }
    if (!isfinite(cfg->freq_offset)) {
        cfg->freq_offset = 0.0;
    }
    if (!isfinite(cfg->bandwidth) || cfg->bandwidth < 0.0) {
        cfg->bandwidth = 0.0;
    }
    if (!isfinite(cfg->amplitude)) {
        cfg->amplitude = 1.0;
    }
    if (cfg->device_index < 0) {
        cfg->device_index = 0;
    }
}

static double sdr_observer_effective_tuned_freq(const SimSdrObserverConfig* cfg) {
    if (cfg == NULL) {
        return 0.0;
    }
    return cfg->center_freq + cfg->freq_offset;
}

static void sdr_observer_status_init(SimSdrObserverStatus* status) {
    if (status == NULL) {
        return;
    }

    (void) memset(status, 0, sizeof(*status));
    status->active_backend       = SIM_SDR_OBSERVER_BACKEND_UNKNOWN;
    status->last_fallback_reason = SIM_SDR_OBSERVER_FALLBACK_NONE;
    status->last_error_code      = SIM_RESULT_OK;
#if OAKFIELD_ENABLE_RTLSDR
    status->rtl_sdr_enabled = true;
#else
    status->rtl_sdr_enabled = false;
#endif
}

static void sdr_observer_status_set_backend(SimSdrObserverState*      state,
                                            SimSdrObserverBackendMode backend) {
    if (state == NULL) {
        return;
    }

    state->status.active_backend  = backend;
    state->status.using_synthetic = (backend == SIM_SDR_OBSERVER_BACKEND_SYNTHETIC);
}

static void sdr_observer_status_record_error(SimSdrObserverState*         state,
                                             SimResult                    code,
                                             int                          backend_error,
                                             SimSdrObserverFallbackReason reason,
                                             const char*                  fmt,
                                             ...) {
    va_list args;

    if (state == NULL) {
        return;
    }

    state->status.has_last_error     = true;
    state->status.last_error_code    = code;
    state->status.last_backend_error = backend_error;
    if (reason != SIM_SDR_OBSERVER_FALLBACK_NONE) {
        state->status.last_fallback_reason = reason;
    }

    if (fmt == NULL) {
        state->status.last_error_message[0] = '\0';
        return;
    }

    va_start(args, fmt);
    (void) vsnprintf(
        state->status.last_error_message, sizeof(state->status.last_error_message), fmt, args);
    va_end(args);
    state->status.last_error_message[sizeof(state->status.last_error_message) - 1U] = '\0';
}

static void sdr_observer_status_note_fallback(SimSdrObserverState*         state,
                                              SimSdrObserverFallbackReason reason,
                                              SimResult                    code,
                                              int                          backend_error,
                                              bool                         device_open,
                                              const char*                  fmt,
                                              ...) {
    va_list args;

    if (state == NULL) {
        return;
    }

    state->status.fallback_count += 1U;
    sdr_observer_status_set_backend(state, SIM_SDR_OBSERVER_BACKEND_SYNTHETIC);
    state->status.last_fallback_reason = reason;
    state->status.device_open          = device_open;
    state->status.last_read_iq_bytes   = 0U;
    state->status.has_last_error       = true;
    state->status.last_error_code      = code;
    state->status.last_backend_error   = backend_error;

    if (fmt == NULL) {
        state->status.last_error_message[0] = '\0';
        return;
    }

    va_start(args, fmt);
    (void) vsnprintf(
        state->status.last_error_message, sizeof(state->status.last_error_message), fmt, args);
    va_end(args);
    state->status.last_error_message[sizeof(state->status.last_error_message) - 1U] = '\0';
}

static void sdr_observer_status_note_open_state(SimSdrObserverState* state, bool device_open) {
    if (state == NULL) {
        return;
    }

    state->status.device_open = device_open;
    if (device_open) {
        sdr_observer_status_set_backend(state, SIM_SDR_OBSERVER_BACKEND_RTL_SDR);
    } else if (state->status.active_backend == SIM_SDR_OBSERVER_BACKEND_RTL_SDR) {
        sdr_observer_status_set_backend(state, SIM_SDR_OBSERVER_BACKEND_SYNTHETIC);
    }
}

static void sdr_observer_status_note_successful_read(SimSdrObserverState* state, size_t iq_bytes) {
    if (state == NULL) {
        return;
    }

    state->status.has_successful_read = true;
    state->status.successful_read_count += 1U;
    state->status.last_read_iq_bytes = iq_bytes;
    state->status.device_open        = state->device_open;
    sdr_observer_status_set_backend(state, SIM_SDR_OBSERVER_BACKEND_RTL_SDR);
}

static void sdr_observer_refresh_symbolic(SimSdrObserverState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    const SimSdrObserverConfig* cfg           = &state->config;
    static const char* const    demod_names[] = { "raw", "am", "fm", "pm" };
    const char* demod_str = (cfg->demod >= 0 && cfg->demod <= SIM_SDR_OBSERVER_DEMOD_PM)
                                ? demod_names[cfg->demod]
                                : "raw";
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "sdr_observer freq=%.3gHz sr=%.3gHz demod=%s",
                    cfg->center_freq,
                    cfg->sample_rate,
                    demod_str);
#else
    (void) state;
#endif
}

static const char* sdr_observer_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimSdrObserverState* state = (const SimSdrObserverState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static double sdr_observer_output_scale(const SimSdrObserverConfig* cfg, double dt) {
    double scale = cfg->amplitude;
    if (cfg->scale_by_dt) {
        scale *= fmax(dt, 0.0);
    }
    return scale;
}

static double sdr_observer_wrap_phase(double phase) {
    double wrapped = fmod(phase + M_PI, 2.0 * M_PI);
    if (wrapped < 0.0) {
        wrapped += 2.0 * M_PI;
    }
    return wrapped - M_PI;
}

static double sdr_observer_mix_phase_increment(const SimSdrObserverConfig* cfg) {
    double sample_rate = fmax(cfg->sample_rate, 1.0);
    return -2.0 * M_PI * cfg->freq_offset / sample_rate;
}

static SimComplexDouble sdr_observer_rotate_sample(SimComplexDouble sample, double phase) {
    if (fabs(phase) <= SDR_OBS_EPS) {
        return sample;
    }

    double           cos_phase = cos(phase);
    double           sin_phase = sin(phase);
    SimComplexDouble rotated;
    rotated.re = sample.re * cos_phase - sample.im * sin_phase;
    rotated.im = sample.re * sin_phase + sample.im * cos_phase;
    return rotated;
}

static void sdr_observer_reset_demod_state(SimSdrObserverState* state) {
    if (state == NULL) {
        return;
    }

    state->mix_phase_accum   = 0.0;
    state->prev_sample.re    = 0.0;
    state->prev_sample.im    = 0.0;
    state->prev_sample_valid = false;
}

static void sdr_observer_close_device(SimSdrObserverState* state) {
#if OAKFIELD_ENABLE_RTLSDR
    if (state->dev != NULL) {
        rtlsdr_close(state->dev);
        state->dev = NULL;
    }
    free(state->iq_buf);
    state->iq_buf       = NULL;
    state->iq_buf_bytes = 0U;
#else
    (void) state;
#endif
    sdr_observer_reset_demod_state(state);
    state->device_open = false;
    sdr_observer_status_note_open_state(state, false);
}

static void sdr_observer_destroy(void* state_ptr) {
    SimSdrObserverState* state = (SimSdrObserverState*) state_ptr;
    if (state != NULL) {
        sdr_observer_close_device(state);
        free(state);
    }
}

static SimComplexDouble sdr_observer_iq_sample(const uint8_t* iq, size_t index) {
    SimComplexDouble sample;
    sample.re = (((double) iq[2U * index]) - SDR_OBS_NORM) / SDR_OBS_NORM;
    sample.im = (((double) iq[2U * index + 1U]) - SDR_OBS_NORM) / SDR_OBS_NORM;
    return sample;
}

static void sdr_observer_store_complex(SimComplexDouble* dst,
                                       size_t            index,
                                       bool              accumulate,
                                       SimComplexDouble  value) {
    if (accumulate) {
        dst[index].re += value.re;
        dst[index].im += value.im;
        return;
    }

    dst[index] = value;
}

static void
sdr_observer_store_real(SimComplexDouble* dst, size_t index, bool accumulate, double value) {
    if (accumulate) {
        dst[index].re += value;
        return;
    }

    dst[index].re = value;
    dst[index].im = 0.0;
}

static void
sdr_observer_zero_tail(SimComplexDouble* dst, size_t begin, size_t count, bool accumulate) {
    if (begin < count && !accumulate) {
        memset(&dst[begin], 0, (count - begin) * sizeof(SimComplexDouble));
    }
}

#if OAKFIELD_ENABLE_RTLSDR
static bool sdr_observer_rtlsdr_ok(SimSdrObserverState* state,
                                   struct SimContext*   context,
                                   const char*          op_name,
                                   const char*          action,
                                   int                  rc) {
    if (rc >= 0) {
        return true;
    }

    sdr_observer_status_record_error(state,
                                     SIM_RESULT_INVALID_STATE,
                                     rc,
                                     SIM_SDR_OBSERVER_FALLBACK_NONE,
                                     "%s failed (rc=%d)",
                                     action,
                                     rc);
    sim_context_log_warning(context, "%s: %s failed (rc=%d)", op_name, action, rc);
    return false;
}

static bool sdr_observer_configure_device(SimSdrObserverState*        state,
                                          struct SimContext*          context,
                                          const char*                 op_name,
                                          const SimSdrObserverConfig* cfg) {
    bool ok = true;

    ok = sdr_observer_rtlsdr_ok(state,
                                context,
                                op_name,
                                "set center frequency",
                                rtlsdr_set_center_freq(state->dev, (uint32_t) cfg->center_freq)) &&
         ok;
    ok = sdr_observer_rtlsdr_ok(state,
                                context,
                                op_name,
                                "set sample rate",
                                rtlsdr_set_sample_rate(state->dev, (uint32_t) cfg->sample_rate)) &&
         ok;

    if (cfg->gain <= SDR_OBS_EPS) {
        ok = sdr_observer_rtlsdr_ok(state,
                                    context,
                                    op_name,
                                    "disable manual tuner gain",
                                    rtlsdr_set_tuner_gain_mode(state->dev, 0)) &&
             ok;
    } else {
        ok = sdr_observer_rtlsdr_ok(state,
                                    context,
                                    op_name,
                                    "enable manual tuner gain",
                                    rtlsdr_set_tuner_gain_mode(state->dev, 1)) &&
             ok;
        ok = sdr_observer_rtlsdr_ok(
                 state,
                 context,
                 op_name,
                 "set tuner gain",
                 rtlsdr_set_tuner_gain(state->dev, (int) (cfg->gain * SDR_GAIN_SCALE))) &&
             ok;
    }

    ok =
        sdr_observer_rtlsdr_ok(state,
                               context,
                               op_name,
                               "set tuner bandwidth",
                               rtlsdr_set_tuner_bandwidth(state->dev, (uint32_t) cfg->bandwidth)) &&
        ok;

    ok = sdr_observer_rtlsdr_ok(
             state, context, op_name, "reset RTL-SDR buffer", rtlsdr_reset_buffer(state->dev)) &&
         ok;
    return ok;
}
#endif

static bool sdr_observer_open_device(SimSdrObserverState* state,
                                     struct SimContext*   context,
                                     const char*          op_name,
                                     size_t               sample_count) {
#if OAKFIELD_ENABLE_RTLSDR
    const SimSdrObserverConfig* cfg     = &state->config;
    int                         open_rc = 0;

    open_rc = rtlsdr_open(&state->dev, (uint32_t) cfg->device_index);
    if (open_rc < 0) {
        sim_context_log_warning(context,
                                "%s: failed to open RTL-SDR device %d; running in synthetic mode.",
                                op_name,
                                cfg->device_index);
        state->dev = NULL;
        sdr_observer_status_note_fallback(
            state,
            SIM_SDR_OBSERVER_FALLBACK_DEVICE_OPEN_FAILED,
            SIM_RESULT_NOT_FOUND,
            open_rc,
            false,
            "%s: failed to open RTL-SDR device %d; running in synthetic mode.",
            op_name,
            cfg->device_index);
        return false;
    }

    if (!sdr_observer_configure_device(state, context, op_name, cfg)) {
        sdr_observer_status_note_fallback(
            state,
            SIM_SDR_OBSERVER_FALLBACK_DEVICE_CONFIG_FAILED,
            SIM_RESULT_INVALID_STATE,
            state->status.last_backend_error,
            true,
            "%s: failed to configure RTL-SDR device %d; running in synthetic mode.",
            op_name,
            cfg->device_index);
        sdr_observer_close_device(state);
        return false;
    }

    size_t buf_bytes = sample_count * 2U * sizeof(uint8_t);
    /* rtlsdr requires buffer size to be a multiple of 512 */
    buf_bytes = ((buf_bytes + 511U) / 512U) * 512U;

    state->iq_buf = (uint8_t*) malloc(buf_bytes);
    if (state->iq_buf == NULL) {
        sdr_observer_status_note_fallback(
            state,
            SIM_SDR_OBSERVER_FALLBACK_BUFFER_ALLOCATION_FAILED,
            SIM_RESULT_OUT_OF_MEMORY,
            0,
            true,
            "%s: failed to allocate SDR IQ buffer; running in synthetic mode.",
            op_name);
        sdr_observer_close_device(state);
        return false;
    }
    state->iq_buf_bytes = buf_bytes;
    sdr_observer_status_note_open_state(state, true);
    return true;
#else
    sdr_observer_status_note_fallback(
        state,
        SIM_SDR_OBSERVER_FALLBACK_RTLSDR_DISABLED,
        SIM_RESULT_NOT_SUPPORTED,
        0,
        false,
        "%s: RTL-SDR support is disabled at build time; running in synthetic mode.",
        op_name != NULL ? op_name : "sdr_observer");
    (void) context;
    (void) sample_count;
    return false;
#endif
}

/* Synthetic IQ generator used when no hardware is available. */
static void
sdr_observer_synth(SimSdrObserverState* state, SimComplexDouble* dst, size_t count, double dt) {
    const SimSdrObserverConfig* cfg = &state->config;

    double test_freq = cfg->freq_offset;
    if (fabs(test_freq) < SDR_OBS_EPS) {
        test_freq = cfg->sample_rate / 64.0;
    }
    double phase_inc = 2.0 * M_PI * test_freq / cfg->sample_rate;
    double scale     = sdr_observer_output_scale(cfg, dt);

    double base_phase = state->phase_accum;

#if defined(_OPENMP)
#pragma omp parallel if (count > 1024)
    {
#pragma omp for
        for (size_t i = 0U; i < count; i++) {
            double           ph     = base_phase + phase_inc * (double) i;
            SimComplexDouble raw_iq = { .re = cos(ph), .im = sin(ph) };
            SimComplexDouble output = { 0.0, 0.0 };

            switch (cfg->demod) {
                case SIM_SDR_OBSERVER_DEMOD_AM:
                    output.re = scale;
                    break;
                case SIM_SDR_OBSERVER_DEMOD_FM:
                    output.re = scale * test_freq;
                    break;
                case SIM_SDR_OBSERVER_DEMOD_PM:
                    output.re = scale * sdr_observer_wrap_phase(ph);
                    break;
                case SIM_SDR_OBSERVER_DEMOD_RAW:
                default:
                    output.re = scale * raw_iq.re;
                    output.im = scale * raw_iq.im;
                    break;
            }

            sdr_observer_store_complex(dst, i, cfg->accumulate, output);
        }
    }
#else
    /* Use phasor rotation for single-threaded case to avoid repeated cos/sin */
    double           re_inc = cos(phase_inc);
    double           im_inc = sin(phase_inc);
    SimComplexDouble phasor;
    phasor.re = cos(base_phase);
    phasor.im = sin(base_phase);

    for (size_t i = 0U; i < count; i++) {
        double           ph     = base_phase + phase_inc * (double) i;
        SimComplexDouble output = { 0.0, 0.0 };

        switch (cfg->demod) {
            case SIM_SDR_OBSERVER_DEMOD_AM:
                output.re = scale;
                break;
            case SIM_SDR_OBSERVER_DEMOD_FM:
                output.re = scale * test_freq;
                break;
            case SIM_SDR_OBSERVER_DEMOD_PM:
                output.re = scale * sdr_observer_wrap_phase(ph);
                break;
            case SIM_SDR_OBSERVER_DEMOD_RAW:
            default:
                output.re = scale * phasor.re;
                output.im = scale * phasor.im;
                break;
        }

        sdr_observer_store_complex(dst, i, cfg->accumulate, output);

        /* Rotate phasor */
        double next_re = phasor.re * re_inc - phasor.im * im_inc;
        double next_im = phasor.re * im_inc + phasor.im * re_inc;
        phasor.re      = next_re;
        phasor.im      = next_im;

        /* Occasional re-normalization to prevent drift in long loops */
        if ((i & 0xFFU) == 0xFFU) {
            double mag_sq = phasor.re * phasor.re + phasor.im * phasor.im;
            double inv    = 1.0 / sqrt(mag_sq + 1e-20);
            phasor.re *= inv;
            phasor.im *= inv;
        }
    }
#endif

    state->phase_accum += phase_inc * (double) count;
    state->phase_accum = fmod(state->phase_accum, 2.0 * M_PI);
}

#if OAKFIELD_ENABLE_RTLSDR
static void sdr_observer_convert_raw(const uint8_t*              iq,
                                     size_t                      n_iq_bytes,
                                     SimComplexDouble*           dst,
                                     size_t                      count,
                                     const SimSdrObserverConfig* cfg,
                                     double                      dt,
                                     double*                     mix_phase_accum) {
    double scale      = sdr_observer_output_scale(cfg, dt);
    size_t available  = n_iq_bytes / 2U;
    size_t to_process = (count < available) ? count : available;
    double phase      = *mix_phase_accum;
    double phase_inc  = sdr_observer_mix_phase_increment(cfg);

    for (size_t i = 0U; i < to_process; i++) {
        SimComplexDouble sample = sdr_observer_iq_sample(iq, i);
        sample                  = sdr_observer_rotate_sample(sample, phase);
        sample.re *= scale;
        sample.im *= scale;
        sdr_observer_store_complex(dst, i, cfg->accumulate, sample);
        phase = sdr_observer_wrap_phase(phase + phase_inc);
    }

    *mix_phase_accum = phase;
    sdr_observer_zero_tail(dst, to_process, count, cfg->accumulate);
}

static void sdr_observer_demod_am(const uint8_t*              iq,
                                  size_t                      n_iq_bytes,
                                  SimComplexDouble*           dst,
                                  size_t                      count,
                                  const SimSdrObserverConfig* cfg,
                                  double                      dt,
                                  double*                     mix_phase_accum) {
    double scale      = sdr_observer_output_scale(cfg, dt);
    size_t available  = n_iq_bytes / 2U;
    size_t to_process = (count < available) ? count : available;
    double phase      = *mix_phase_accum;
    double phase_inc  = sdr_observer_mix_phase_increment(cfg);

    for (size_t i = 0U; i < to_process; i++) {
        SimComplexDouble sample = sdr_observer_iq_sample(iq, i);
        sample                  = sdr_observer_rotate_sample(sample, phase);
        sdr_observer_store_real(dst, i, cfg->accumulate, scale * hypot(sample.re, sample.im));
        phase = sdr_observer_wrap_phase(phase + phase_inc);
    }

    *mix_phase_accum = phase;
    sdr_observer_zero_tail(dst, to_process, count, cfg->accumulate);
}

static void sdr_observer_demod_fm(const uint8_t*              iq,
                                  size_t                      n_iq_bytes,
                                  SimComplexDouble*           dst,
                                  size_t                      count,
                                  const SimSdrObserverConfig* cfg,
                                  double                      dt,
                                  SimComplexDouble*           prev,
                                  bool*                       prev_valid,
                                  double*                     mix_phase_accum) {
    double scale = sdr_observer_output_scale(cfg, dt) * fmax(cfg->sample_rate, 1.0) / (2.0 * M_PI);
    size_t available               = n_iq_bytes / 2U;
    size_t to_process              = (count < available) ? count : available;
    double phase                   = *mix_phase_accum;
    double phase_inc               = sdr_observer_mix_phase_increment(cfg);
    SimComplexDouble previous      = *prev;
    bool             have_previous = *prev_valid;

    for (size_t i = 0U; i < to_process; i++) {
        SimComplexDouble sample   = sdr_observer_iq_sample(iq, i);
        double           freq_dev = 0.0;

        sample = sdr_observer_rotate_sample(sample, phase);
        if (have_previous) {
            double cross = sample.im * previous.re - sample.re * previous.im;
            double dot   = sample.re * previous.re + sample.im * previous.im;
            freq_dev     = scale * atan2(cross, dot);
        }

        previous      = sample;
        have_previous = true;
        sdr_observer_store_real(dst, i, cfg->accumulate, freq_dev);
        phase = sdr_observer_wrap_phase(phase + phase_inc);
    }

    *prev            = previous;
    *prev_valid      = have_previous;
    *mix_phase_accum = phase;
    sdr_observer_zero_tail(dst, to_process, count, cfg->accumulate);
}

static void sdr_observer_demod_pm(const uint8_t*              iq,
                                  size_t                      n_iq_bytes,
                                  SimComplexDouble*           dst,
                                  size_t                      count,
                                  const SimSdrObserverConfig* cfg,
                                  double                      dt,
                                  double*                     mix_phase_accum) {
    double scale      = sdr_observer_output_scale(cfg, dt);
    size_t available  = n_iq_bytes / 2U;
    size_t to_process = (count < available) ? count : available;
    double phase      = *mix_phase_accum;
    double phase_inc  = sdr_observer_mix_phase_increment(cfg);

    for (size_t i = 0U; i < to_process; i++) {
        SimComplexDouble sample = sdr_observer_iq_sample(iq, i);
        sample                  = sdr_observer_rotate_sample(sample, phase);
        sdr_observer_store_real(dst, i, cfg->accumulate, scale * atan2(sample.im, sample.re));
        phase = sdr_observer_wrap_phase(phase + phase_inc);
    }

    *mix_phase_accum = phase;
    sdr_observer_zero_tail(dst, to_process, count, cfg->accumulate);
}
#endif /* OAKFIELD_ENABLE_RTLSDR */

static SimResult sdr_observer_step(void*               state_ptr,
                                   struct SimContext*  context,
                                   struct SimOperator* self,
                                   size_t              substep_index,
                                   double              dt_sub,
                                   void*               scratch,
                                   size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    SimSdrObserverState* state = (SimSdrObserverState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimSdrObserverConfig* cfg   = &state->config;
    SimField*                   field = sim_context_field(context, cfg->field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_field_is_complex(field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    SimComplexDouble* dst = sim_field_complex_data(field);
    if (dst == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    size_t count = sim_field_element_count(&field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

#if OAKFIELD_ENABLE_RTLSDR
    if (state->device_open && state->dev != NULL) {
        size_t needed_bytes = count * 2U * sizeof(uint8_t);
        /* Ensure multiple of 512 for rtlsdr */
        size_t buf_size = ((needed_bytes + 511U) / 512U) * 512U;

        if (buf_size > state->iq_buf_bytes) {
            uint8_t* new_buf = (uint8_t*) realloc(state->iq_buf, buf_size);
            if (new_buf == NULL) {
                sdr_observer_status_note_fallback(
                    state,
                    SIM_SDR_OBSERVER_FALLBACK_BUFFER_ALLOCATION_FAILED,
                    SIM_RESULT_OUT_OF_MEMORY,
                    0,
                    true,
                    "%s: failed to grow SDR IQ buffer; using synthetic fallback for this step.",
                    self != NULL ? self->name : "sdr_observer");
                sdr_observer_reset_demod_state(state);
                goto synth_fallback;
            }
            state->iq_buf       = new_buf;
            state->iq_buf_bytes = buf_size;
        }

        int n_read = 0;
        int rc     = rtlsdr_read_sync(state->dev, state->iq_buf, (int) buf_size, &n_read);
        if (rc < 0 || n_read <= 0) {
            sdr_observer_status_note_fallback(state,
                                              SIM_SDR_OBSERVER_FALLBACK_READ_FAILED,
                                              SIM_RESULT_INVALID_STATE,
                                              rc,
                                              true,
                                              "%s: RTL-SDR read failed (rc=%d, bytes=%d); using "
                                              "synthetic fallback for this step.",
                                              self != NULL ? self->name : "sdr_observer",
                                              rc,
                                              n_read);
            sdr_observer_reset_demod_state(state);
            goto synth_fallback;
        }

        size_t n_iq_bytes = (size_t) n_read;
        sdr_observer_status_note_successful_read(state, n_iq_bytes);

        switch (cfg->demod) {
            case SIM_SDR_OBSERVER_DEMOD_AM:
                sdr_observer_demod_am(
                    state->iq_buf, n_iq_bytes, dst, count, cfg, dt_sub, &state->mix_phase_accum);
                break;
            case SIM_SDR_OBSERVER_DEMOD_FM:
                sdr_observer_demod_fm(state->iq_buf,
                                      n_iq_bytes,
                                      dst,
                                      count,
                                      cfg,
                                      dt_sub,
                                      &state->prev_sample,
                                      &state->prev_sample_valid,
                                      &state->mix_phase_accum);
                break;
            case SIM_SDR_OBSERVER_DEMOD_PM:
                sdr_observer_demod_pm(
                    state->iq_buf, n_iq_bytes, dst, count, cfg, dt_sub, &state->mix_phase_accum);
                break;
            case SIM_SDR_OBSERVER_DEMOD_RAW:
            default:
                sdr_observer_convert_raw(
                    state->iq_buf, n_iq_bytes, dst, count, cfg, dt_sub, &state->mix_phase_accum);
                break;
        }
        goto normalize_step;
    }

synth_fallback:
#endif /* OAKFIELD_ENABLE_RTLSDR */

    sdr_observer_status_set_backend(state, SIM_SDR_OBSERVER_BACKEND_SYNTHETIC);
    state->status.last_read_iq_bytes = 0U;
    sdr_observer_synth(state, dst, count, dt_sub);

normalize_step:
    if (cfg->normalize) {
        double max_mag = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(max : max_mag) if (count > 1024)
#endif
        for (size_t i = 0U; i < count; i++) {
            double mag = dst[i].re * dst[i].re + dst[i].im * dst[i].im;
            if (mag > max_mag) {
                max_mag = mag;
            }
        }
        if (max_mag > SDR_OBS_EPS) {
            double inv = 1.0 / sqrt(max_mag);
#if defined(_OPENMP)
#pragma omp parallel for if (count > 1024)
#endif
            for (size_t i = 0U; i < count; i++) {
                dst[i].re *= inv;
                dst[i].im *= inv;
            }
        }
    }
    return SIM_RESULT_OK;
}

SimResult sim_add_sdr_observer_operator(struct SimContext*          context,
                                        const SimSdrObserverConfig* config,
                                        size_t*                     out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimSdrObserverConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }
    if (local.amplitude == 0.0 && (config == NULL || config->amplitude == 0.0)) {
        local.amplitude = 1.0;
    }
    if (local.sample_rate == 0.0) {
        local.sample_rate = 2.048e6;
    }
    if (local.center_freq == 0.0) {
        local.center_freq = 100.0e6;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "sdr_observer", (config != NULL), (config != NULL) ? config->scale_by_dt : false);

    sdr_observer_normalize_config(&local);

    SimField* field = sim_context_field(context, local.field_index);
    if (field == NULL || !sim_field_is_complex(field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    SimSdrObserverState* state = (SimSdrObserverState*) calloc(1U, sizeof(SimSdrObserverState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    sdr_observer_status_init(&state->status);
    state->config            = local;
    state->phase_accum       = 0.0;
    state->mix_phase_accum   = 0.0;
    state->prev_sample.re    = 0.0;
    state->prev_sample.im    = 0.0;
    state->prev_sample_valid = false;
    state->device_open       = false;

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "sdr_observer");

    size_t count = sim_field_element_count(&field->layout);
    if (count > 0U) {
        state->device_open = sdr_observer_open_device(state, context, name, count);
    }

    sdr_observer_refresh_symbolic(state);

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_MEASUREMENT;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = false;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "sdr_observer";
    sim_operator_info_set_schema_identity(&info, "sdr_observer");
    info.algebraic_flags                                = 0U;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
    info.representation.requires_complex_input          = true;
    info.representation.requires_complex_representation = true;
    info.representation.preserves_real_subspace         = false;

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = true };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = sdr_observer_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = sdr_observer_symbolic,
                                .destroy       = sdr_observer_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        sdr_observer_destroy(state);
    }
    return result;
}

SimResult sim_sdr_observer_config(struct SimContext*    context,
                                  size_t                operator_index,
                                  SimSdrObserverConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSdrObserverState* state = (SimSdrObserverState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_sdr_observer_status(struct SimContext*    context,
                                  size_t                operator_index,
                                  SimSdrObserverStatus* out_status) {
    if (context == NULL || out_status == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSdrObserverState* state = (SimSdrObserverState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_status                      = state->status;
    out_status->effective_tuned_freq = sdr_observer_effective_tuned_freq(&state->config);
    return SIM_RESULT_OK;
}

SimResult sim_sdr_observer_update(struct SimContext*          context,
                                  size_t                      operator_index,
                                  const SimSdrObserverConfig* config) {
    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimSdrObserverState* state = (SimSdrObserverState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimSdrObserverConfig local = *config;
    local.scale_by_dt          = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "sdr_observer"), true, local.scale_by_dt);
    sdr_observer_normalize_config(&local);

    if (local.field_index != state->config.field_index) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool center_freq_changed = fabs(local.center_freq - state->config.center_freq) > SDR_OBS_EPS;
    bool sample_rate_changed = fabs(local.sample_rate - state->config.sample_rate) > SDR_OBS_EPS;
    bool gain_changed        = fabs(local.gain - state->config.gain) > SDR_OBS_EPS;
    bool bandwidth_changed   = fabs(local.bandwidth - state->config.bandwidth) > SDR_OBS_EPS;
    bool device_changed      = (local.device_index != state->config.device_index);
    bool freq_offset_changed = fabs(local.freq_offset - state->config.freq_offset) > SDR_OBS_EPS;
    bool demod_changed       = (local.demod != state->config.demod);
    bool reconfigure_needed =
        center_freq_changed || sample_rate_changed || gain_changed || bandwidth_changed;
    bool demod_state_changed =
        reconfigure_needed || device_changed || freq_offset_changed || demod_changed;

    state->config = local;

#if OAKFIELD_ENABLE_RTLSDR
    SimField* field = sim_context_field(context, state->config.field_index);
    size_t    count = (field != NULL) ? sim_field_element_count(&field->layout) : 0U;

    if (device_changed) {
        sdr_observer_close_device(state);
        if (count > 0U) {
            state->device_open = sdr_observer_open_device(state, context, op->name, count);
        }
    } else if (reconfigure_needed) {
        if (state->device_open && state->dev != NULL) {
            if (!sdr_observer_configure_device(state, context, op->name, &local)) {
                sdr_observer_close_device(state);
            }
        } else if (count > 0U) {
            state->device_open = sdr_observer_open_device(state, context, op->name, count);
        }
    }
#else
    if (device_changed || reconfigure_needed) {
        state->device_open = false;
        sdr_observer_status_note_fallback(
            state,
            SIM_SDR_OBSERVER_FALLBACK_RTLSDR_DISABLED,
            SIM_RESULT_NOT_SUPPORTED,
            0,
            false,
            "%s: RTL-SDR support is disabled at build time; running in synthetic mode.",
            op->name[0] != '\0' ? op->name : "sdr_observer");
    }
    (void) center_freq_changed;
    (void) sample_rate_changed;
    (void) gain_changed;
    (void) bandwidth_changed;
    (void) device_changed;
    (void) freq_offset_changed;
    (void) demod_changed;
    (void) reconfigure_needed;
#endif

    if (demod_state_changed) {
        sdr_observer_reset_demod_state(state);
    }

    sdr_observer_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
