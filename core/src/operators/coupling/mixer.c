#include "oakfield/operators/coupling/mixer.h"
#include "operators/common/operator_utils.h"
#include "sim_accel.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIXER_SYMBOLIC_CAPACITY 160
#define MIXER_ACCEL_MIN_LEN 64U

typedef struct SimMixerOperatorState {
    SimMixerOperatorConfig    config;
    SimMixerFeedbackSplitMode split_layout;
    double*                   vdsp_block;
    size_t                    vdsp_capacity;
    char                      symbolic[MIXER_SYMBOLIC_CAPACITY];
} SimMixerOperatorState;

static void mixer_destroy(void* userdata);
static void mixer_feedback_decay_complex(const SimMixerOperatorConfig* cfg,
                                         double                        dt,
                                         SimComplexDouble*             out,
                                         size_t                        count);
static void
mixer_feedback_decay_real(const SimMixerOperatorConfig* cfg, double dt, double* out, size_t count);
static void mixer_feedback_inject_complex(const SimMixerOperatorConfig* cfg,
                                          const SimComplexDouble*       lhs_c,
                                          const double*                 lhs_r,
                                          const SimComplexDouble*       rhs_c,
                                          const double*                 rhs_r,
                                          SimComplexDouble*             out,
                                          size_t                        count);
static void mixer_feedback_inject_real(const SimMixerOperatorConfig* cfg,
                                       const double*                 lhs,
                                       const double*                 rhs,
                                       double*                       out,
                                       size_t                        count);
static bool mixer_reserve_real_fm_pm_buffers(SimMixerOperatorState* state,
                                             size_t                 count,
                                             double**               phase,
                                             double**               value);
static bool mixer_try_apply_real_fm_pm_accel(SimMixerOperatorState*        state,
                                             const SimMixerOperatorConfig* cfg,
                                             double                        dt,
                                             const double*                 lhs,
                                             const double*                 rhs,
                                             double*                       out,
                                             size_t                        count);

const char* mixer_mode_name(SimMixerMode mode) {
    switch (mode) {
        case SIM_MIXER_MODE_MULTIPLY:
            return "multiply";
        case SIM_MIXER_MODE_CROSSFADE:
            return "crossfade";
        case SIM_MIXER_MODE_SUM:
            return "sum";
        case SIM_MIXER_MODE_POWER:
            return "power";
        case SIM_MIXER_MODE_AM:
            return "am";
        case SIM_MIXER_MODE_FM:
            return "fm";
        case SIM_MIXER_MODE_PM:
            return "pm";
        case SIM_MIXER_MODE_RING_MOD:
            return "ring_mod";
        case SIM_MIXER_MODE_MAX:
            return "max";
        case SIM_MIXER_MODE_MIN:
            return "min";
        case SIM_MIXER_MODE_AVERAGE:
            return "average";
        case SIM_MIXER_MODE_DIFFERENCE:
            return "difference";
        case SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE:
            return "abs_diff";
        case SIM_MIXER_MODE_FEEDBACK:
            return "feedback";
        case SIM_MIXER_MODE_LINEAR:
        default:
            return "linear";
    }
}

const char* mixer_feedback_epsilon_mode_name(SimMixerFeedbackEpsilonMode mode) {
    switch (mode) {
        case SIM_MIXER_FEEDBACK_EPS_FEEDBACK:
            return "feedback";
        case SIM_MIXER_FEEDBACK_EPS_INPUT:
        default:
            return "input";
    }
}

const char* mixer_feedback_split_name(SimMixerFeedbackSplitMode mode) {
    switch (mode) {
        case SIM_MIXER_FEEDBACK_SPLIT_LIE:
            return "lie";
        case SIM_MIXER_FEEDBACK_SPLIT_STRANG:
            return "strang";
        case SIM_MIXER_FEEDBACK_SPLIT_NONE:
        default:
            return "none";
    }
}

bool mixer_mode_from_name(const char* name, SimMixerMode* out_mode) {
    if (name == NULL || out_mode == NULL) {
        return false;
    }

#define MATCH_MODE(value, enum_value)                                                              \
    if (strcmp(name, value) == 0) {                                                                \
        *out_mode = enum_value;                                                                    \
        return true;                                                                               \
    }

    MATCH_MODE("linear", SIM_MIXER_MODE_LINEAR);
    MATCH_MODE("multiply", SIM_MIXER_MODE_MULTIPLY);
    MATCH_MODE("crossfade", SIM_MIXER_MODE_CROSSFADE);
    MATCH_MODE("sum", SIM_MIXER_MODE_SUM);
    MATCH_MODE("power", SIM_MIXER_MODE_POWER);
    MATCH_MODE("am", SIM_MIXER_MODE_AM);
    MATCH_MODE("fm", SIM_MIXER_MODE_FM);
    MATCH_MODE("pm", SIM_MIXER_MODE_PM);
    MATCH_MODE("ring_mod", SIM_MIXER_MODE_RING_MOD);
    MATCH_MODE("max", SIM_MIXER_MODE_MAX);
    MATCH_MODE("min", SIM_MIXER_MODE_MIN);
    MATCH_MODE("average", SIM_MIXER_MODE_AVERAGE);
    MATCH_MODE("difference", SIM_MIXER_MODE_DIFFERENCE);
    MATCH_MODE("abs_diff", SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE);
    MATCH_MODE("feedback", SIM_MIXER_MODE_FEEDBACK);

#undef MATCH_MODE

    return false;
}

bool mixer_feedback_epsilon_mode_from_name(const char*                  name,
                                           SimMixerFeedbackEpsilonMode* out_mode) {
    if (name == NULL || out_mode == NULL) {
        return false;
    }

#define MATCH_FB_EPS_MODE(value, enum_value)                                                       \
    if (strcmp(name, value) == 0) {                                                                \
        *out_mode = enum_value;                                                                    \
        return true;                                                                               \
    }

    MATCH_FB_EPS_MODE("input", SIM_MIXER_FEEDBACK_EPS_INPUT);
    MATCH_FB_EPS_MODE("feedback", SIM_MIXER_FEEDBACK_EPS_FEEDBACK);

#undef MATCH_FB_EPS_MODE

    return false;
}

bool mixer_feedback_split_from_name(const char* name, SimMixerFeedbackSplitMode* out_mode) {
    if (name == NULL || out_mode == NULL) {
        return false;
    }

#define MATCH_FB_SPLIT(value, enum_value)                                                          \
    if (strcmp(name, value) == 0) {                                                                \
        *out_mode = enum_value;                                                                    \
        return true;                                                                               \
    }

    MATCH_FB_SPLIT("none", SIM_MIXER_FEEDBACK_SPLIT_NONE);
    MATCH_FB_SPLIT("lie", SIM_MIXER_FEEDBACK_SPLIT_LIE);
    MATCH_FB_SPLIT("strang", SIM_MIXER_FEEDBACK_SPLIT_STRANG);

#undef MATCH_FB_SPLIT

    return false;
}

static void mixer_normalize_config(SimMixerOperatorConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->lhs_gain)) {
        config->lhs_gain = 1.0;
    }
    if (!isfinite(config->rhs_gain)) {
        config->rhs_gain = 1.0;
    }
    if (!isfinite(config->mix)) {
        config->mix = 0.5;
    }
    if (config->mix < 0.0) {
        config->mix = 0.0;
    } else if (config->mix > 1.0) {
        config->mix = 1.0;
    }
    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }
    if (!isfinite(config->feedback_epsilon)) {
        config->feedback_epsilon = 1.0;
    }

    {
        int mode_value = (int) config->mode;
        if (mode_value < (int) SIM_MIXER_MODE_LINEAR ||
            mode_value > (int) SIM_MIXER_MODE_FEEDBACK) {
            config->mode = SIM_MIXER_MODE_LINEAR;
        }
    }

    {
        int eps_mode = (int) config->feedback_epsilon_mode;
        if (eps_mode < (int) SIM_MIXER_FEEDBACK_EPS_INPUT ||
            eps_mode > (int) SIM_MIXER_FEEDBACK_EPS_FEEDBACK) {
            config->feedback_epsilon_mode = SIM_MIXER_FEEDBACK_EPS_INPUT;
        }
    }
    {
        int split_mode = (int) config->feedback_split;
        if (split_mode < (int) SIM_MIXER_FEEDBACK_SPLIT_NONE ||
            split_mode > (int) SIM_MIXER_FEEDBACK_SPLIT_STRANG) {
            config->feedback_split = SIM_MIXER_FEEDBACK_SPLIT_NONE;
        }
    }

    if (config->mode == SIM_MIXER_MODE_FEEDBACK) {
        /* Feedback produces the full output; ignore accumulate requests. */
        config->accumulate = false;
    }

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static SimResult mixer_validate_fields(const SimField* lhs_field,
                                       const SimField* rhs_field,
                                       const SimField* out_field) {
    size_t lhs_count   = 0U;
    size_t rhs_count   = 0U;
    size_t out_count   = 0U;
    bool   lhs_complex = false;
    bool   rhs_complex = false;
    bool   out_complex = false;
    bool   any_complex = false;

    if (lhs_field == NULL || rhs_field == NULL || out_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    lhs_count = sim_field_element_count(&lhs_field->layout);
    rhs_count = sim_field_element_count(&rhs_field->layout);
    out_count = sim_field_element_count(&out_field->layout);
    if (lhs_count == 0U || lhs_count != rhs_count || lhs_count != out_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    lhs_complex = sim_field_is_complex(lhs_field);
    rhs_complex = sim_field_is_complex(rhs_field);
    out_complex = sim_field_is_complex(out_field);
    any_complex = lhs_complex || rhs_complex || out_complex;

    if ((!lhs_complex && !sim_operator_field_domain_is_f64(lhs_field)) ||
        (!rhs_complex && !sim_operator_field_domain_is_f64(rhs_field)) ||
        (!out_complex && !sim_operator_field_domain_is_f64(out_field))) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (any_complex && !out_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    return SIM_RESULT_OK;
}

static void mixer_refresh_symbolic(SimMixerOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimMixerOperatorConfig* cfg = &state->config;
    if (cfg->mode == SIM_MIXER_MODE_FEEDBACK) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "mixer mode=feedback decay=%.3g lhs=%.3g rhs=%.3g eps=%.3g eps_mode=%s "
                        "split=%s bias=%.3g",
                        cfg->lhs_gain,
                        cfg->lhs_gain,
                        cfg->rhs_gain,
                        cfg->feedback_epsilon,
                        mixer_feedback_epsilon_mode_name(cfg->feedback_epsilon_mode),
                        mixer_feedback_split_name(cfg->feedback_split),
                        cfg->bias);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "mixer mode=%s lhs=%.3g rhs=%.3g mix=%.3g bias=%.3g scale_by_dt=%s",
                        mixer_mode_name(cfg->mode),
                        cfg->lhs_gain,
                        cfg->rhs_gain,
                        cfg->mix,
                        cfg->bias,
                        cfg->scale_by_dt ? "true" : "false");
    }
#else
    (void) state;
#endif
}

static const char* mixer_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimMixerOperatorState* state = (const SimMixerOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static inline bool mixer_sample_finite(double complex v) {
    return isfinite(creal(v)) && isfinite(cimag(v));
}

static void mixer_destroy(void* userdata) {
    SimMixerOperatorState* state = (SimMixerOperatorState*) userdata;
    if (state == NULL) {
        return;
    }
    free(state->vdsp_block);
    free(state);
}

static bool mixer_reserve_real_fm_pm_buffers(SimMixerOperatorState* state,
                                             size_t                 count,
                                             double**               phase,
                                             double**               value) {
    if (phase != NULL) {
        *phase = NULL;
    }
    if (value != NULL) {
        *value = NULL;
    }
    if (state == NULL) {
        return false;
    }
    if (count == 0U) {
        return true;
    }

    if (state->vdsp_block == NULL || state->vdsp_capacity < count) {
        double* block = (double*) realloc(state->vdsp_block, count * 2U * sizeof(double));
        if (block == NULL) {
            return false;
        }
        state->vdsp_block    = block;
        state->vdsp_capacity = count;
    }

    if (phase != NULL) {
        *phase = state->vdsp_block;
    }
    if (value != NULL) {
        *value = state->vdsp_block + count;
    }
    return true;
}

static bool mixer_try_apply_real_fm_pm_accel(SimMixerOperatorState*        state,
                                             const SimMixerOperatorConfig* cfg,
                                             double                        dt,
                                             const double*                 lhs,
                                             const double*                 rhs,
                                             double*                       out,
                                             size_t                        count) {
    if (state == NULL || cfg == NULL || lhs == NULL || rhs == NULL || out == NULL ||
        count < MIXER_ACCEL_MIN_LEN) {
        return false;
    }
    if (cfg->mode != SIM_MIXER_MODE_FM && cfg->mode != SIM_MIXER_MODE_PM) {
        return false;
    }

#if defined(SIM_HAVE_VDSP)
    double  lhs_max = 0.0;
    double  rhs_max = 0.0;
    double* phase   = NULL;
    double* value   = NULL;

    if (!sim_accel_scan_real_finite_maxabs(lhs, count, &lhs_max) ||
        !sim_accel_scan_real_finite_maxabs(rhs, count, &rhs_max)) {
        return false;
    }

    {
        const double result_scale =
            cfg->accumulate ? (cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0) : 1.0;
        const double max_term =
            fabs(result_scale * cfg->lhs_gain) * lhs_max + fabs(result_scale * cfg->bias);
        (void) rhs_max;
        if (!isfinite(max_term) || max_term > DBL_MAX) {
            return false;
        }
    }

    if (!mixer_reserve_real_fm_pm_buffers(state, count, &phase, &value)) {
        return false;
    }

    {
        const vDSP_Length len        = (vDSP_Length) count;
        const int         vforce_len = (int) count;
        const double      add_scale  = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

        if (cfg->rhs_gain == 1.0) {
            if (phase != rhs) {
                memmove(phase, rhs, count * sizeof(double));
            }
        } else {
            vDSP_vsmulD(rhs, 1, &cfg->rhs_gain, phase, 1, len);
        }
        vvcos(value, phase, &vforce_len);

        if (cfg->lhs_gain == 1.0) {
            if (phase != lhs) {
                memmove(phase, lhs, count * sizeof(double));
            }
        } else {
            vDSP_vsmulD(lhs, 1, &cfg->lhs_gain, phase, 1, len);
        }
        vDSP_vmulD(phase, 1, value, 1, phase, 1, len);

        if (cfg->bias != 0.0) {
            vDSP_vsaddD(phase, 1, &cfg->bias, phase, 1, len);
        }

        if (cfg->accumulate) {
            if (add_scale == 0.0) {
                return true;
            }
            if (add_scale != 1.0) {
                vDSP_vsmulD(phase, 1, &add_scale, phase, 1, len);
            }
            vDSP_vaddD(out, 1, phase, 1, out, 1, len);
        } else if (out != phase) {
            memmove(out, phase, count * sizeof(double));
        }
    }

    return true;
#else
    (void) state;
    (void) cfg;
    (void) dt;
    (void) lhs;
    (void) rhs;
    (void) out;
    (void) count;
    return false;
#endif
}

static void mixer_apply_complex_mode(const SimMixerOperatorConfig* cfg,
                                     double                        dt,
                                     const SimComplexDouble*       lhs_c,
                                     const double*                 lhs_r,
                                     const SimComplexDouble*       rhs_c,
                                     const double*                 rhs_r,
                                     SimComplexDouble*             out,
                                     size_t                        count) {
    if (cfg == NULL || out == NULL || count == 0U) {
        return;
    }

    const double add_scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;
    const double mix       = cfg->mix;
    const double lhs_gain  = cfg->lhs_gain;
    const double rhs_gain  = cfg->rhs_gain;
    const double bias      = cfg->bias;

    switch (cfg->mode) {
        case SIM_MIXER_MODE_MULTIPLY:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = L * R + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_CROSSFADE:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = ((1.0 - mix) * L + mix * R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_SUM:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = (L + R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_POWER:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = cpow(L, R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_AM:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = (1.0 + R) * L + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_FM:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = L * cexp(I * R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_PM:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = cabs(L) * cexp(I * (carg(L) + R)) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_RING_MOD:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = (L * R) - (creal(L) * creal(R)) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_MAX:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = (cabs(L) > cabs(R)) ? L : R;
                M += bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_MIN:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = (cabs(L) < cabs(R)) ? L : R;
                M += bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_AVERAGE:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = 0.5 * (L + R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_DIFFERENCE:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = (L - R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex D = L - R;
                double complex M = CMPLX(fabs(creal(D)), fabs(cimag(D))) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;

        case SIM_MIXER_MODE_FEEDBACK: {
            double epsilon     = cfg->feedback_epsilon;
            double decay_scale = exp(
                -lhs_gain * fmax(dt, 0.0) *
                ((cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_FEEDBACK) ? epsilon : 1.0));
            double rhs_scale =
                rhs_gain *
                ((cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_INPUT) ? epsilon : 1.0);
            for (size_t i = 0; i < count; ++i) {
                double complex P = out[i].re + I * out[i].im;
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = decay_scale * P + rhs_scale * (L + R) + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;
        }

        case SIM_MIXER_MODE_LINEAR:
        default:
            for (size_t i = 0; i < count; ++i) {
                double complex L =
                    lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
                double complex R =
                    rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
                double complex M = L + R + bias;
                if (!mixer_sample_finite(M))
                    continue;
                if (cfg->accumulate) {
                    out[i].re += add_scale * creal(M);
                    out[i].im += add_scale * cimag(M);
                } else {
                    out[i].re = creal(M);
                    out[i].im = cimag(M);
                }
            }
            break;
    }
}

static void mixer_apply_real_mode(const SimMixerOperatorConfig* cfg,
                                  double                        dt,
                                  const double*                 lhs,
                                  const double*                 rhs,
                                  double*                       out,
                                  size_t                        count) {
    if (cfg == NULL || lhs == NULL || rhs == NULL || out == NULL || count == 0U) {
        return;
    }

    const double add_scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;
    const double mix       = cfg->mix;
    const double lhs_gain  = cfg->lhs_gain;
    const double rhs_gain  = cfg->rhs_gain;
    const double bias      = cfg->bias;

    switch (cfg->mode) {
        case SIM_MIXER_MODE_MULTIPLY:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = L * R + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_CROSSFADE:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = ((1.0 - mix) * L + mix * R) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_SUM:
            for (size_t i = 0; i < count; ++i) {
                double M = lhs_gain * lhs[i] + rhs_gain * rhs[i] + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_POWER:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = pow(fabs(L), R) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_AM:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = (1.0 + R) * L + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_FM:
        case SIM_MIXER_MODE_PM:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = L * cos(R) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_RING_MOD:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = (L * R) - (L * R * 0.5) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_MAX:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = fmax(L, R) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_MIN:
            for (size_t i = 0; i < count; ++i) {
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = fmin(L, R) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_AVERAGE:
            for (size_t i = 0; i < count; ++i) {
                double M = 0.5 * (lhs_gain * lhs[i] + rhs_gain * rhs[i]) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_DIFFERENCE:
            for (size_t i = 0; i < count; ++i) {
                double M = (lhs_gain * lhs[i] - rhs_gain * rhs[i]) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE:
            for (size_t i = 0; i < count; ++i) {
                double M = fabs(lhs_gain * lhs[i] - rhs_gain * rhs[i]) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;

        case SIM_MIXER_MODE_FEEDBACK: {
            double epsilon     = cfg->feedback_epsilon;
            double decay_scale = exp(
                -lhs_gain * fmax(dt, 0.0) *
                ((cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_FEEDBACK) ? epsilon : 1.0));
            double rhs_scale =
                rhs_gain *
                ((cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_INPUT) ? epsilon : 1.0);
            for (size_t i = 0; i < count; ++i) {
                double P = out[i];
                double L = lhs_gain * lhs[i];
                double R = rhs_gain * rhs[i];
                double M = decay_scale * P + rhs_scale * (L + R) + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;
        }

        case SIM_MIXER_MODE_LINEAR:
        default:
            for (size_t i = 0; i < count; ++i) {
                double M = lhs_gain * lhs[i] + rhs_gain * rhs[i] + bias;
                if (!isfinite(M))
                    continue;
                if (cfg->accumulate)
                    out[i] += add_scale * M;
                else
                    out[i] = M;
            }
            break;
    }
}

static SimResult
mixer_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimMixerOperatorState* state = (SimMixerOperatorState*) state_ptr;
    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField* out_field = sim_context_field(context, state->config.output_field);
    {
        SimResult validation = mixer_validate_fields(lhs_field, rhs_field, out_field);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    bool lhs_complex = sim_field_is_complex(lhs_field);
    bool rhs_complex = sim_field_is_complex(rhs_field);
    bool out_complex = sim_field_is_complex(out_field);
    bool any_complex = lhs_complex || rhs_complex || out_complex;

    size_t lhs_count = sim_field_element_count(&lhs_field->layout);
    size_t rhs_count = sim_field_element_count(&rhs_field->layout);
    size_t out_count = sim_field_element_count(&out_field->layout);

    if (any_complex) {
        SimComplexDouble*       out = sim_field_complex_data(out_field);
        const SimComplexDouble* lhs_c =
            lhs_complex ? sim_field_complex_data_const(lhs_field) : NULL;
        const SimComplexDouble* rhs_c =
            rhs_complex ? sim_field_complex_data_const(rhs_field) : NULL;
        const double* lhs_r =
            (!lhs_complex) ? (const double*) sim_field_data_const(lhs_field) : NULL;
        const double* rhs_r =
            (!rhs_complex) ? (const double*) sim_field_data_const(rhs_field) : NULL;
        if (!out || (!lhs_complex && !lhs_r) || (!rhs_complex && !rhs_r)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (state->config.mode == SIM_MIXER_MODE_FEEDBACK &&
            state->config.feedback_split != SIM_MIXER_FEEDBACK_SPLIT_NONE) {
            double decay_dt =
                (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) ? 0.5 * dt : dt;
            mixer_feedback_decay_complex(&state->config, decay_dt, out, out_count);
            mixer_feedback_inject_complex(
                &state->config, lhs_c, lhs_r, rhs_c, rhs_r, out, out_count);
            if (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) {
                mixer_feedback_decay_complex(&state->config, decay_dt, out, out_count);
            }
            return SIM_RESULT_OK;
        }

        mixer_apply_complex_mode(&state->config, dt, lhs_c, lhs_r, rhs_c, rhs_r, out, out_count);
    } else {
        /* Real-only fallback path */
        double* lhs = sim_field_real_data(lhs_field);
        double* rhs = sim_field_real_data(rhs_field);
        double* out = sim_field_real_data(out_field);
        if (!lhs || !rhs || !out)
            return SIM_RESULT_INVALID_ARGUMENT;

        size_t count = sim_field_bytes(lhs_field) / sizeof(double);
        if (state->config.mode == SIM_MIXER_MODE_FEEDBACK &&
            state->config.feedback_split != SIM_MIXER_FEEDBACK_SPLIT_NONE) {
            double decay_dt =
                (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) ? 0.5 * dt : dt;
            mixer_feedback_decay_real(&state->config, decay_dt, out, count);
            mixer_feedback_inject_real(&state->config, lhs, rhs, out, count);
            if (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) {
                mixer_feedback_decay_real(&state->config, decay_dt, out, count);
            }
            return SIM_RESULT_OK;
        }

        if (mixer_try_apply_real_fm_pm_accel(state, &state->config, dt, lhs, rhs, out, count)) {
            return SIM_RESULT_OK;
        }
        mixer_apply_real_mode(&state->config, dt, lhs, rhs, out, count);
    }

    return SIM_RESULT_OK;
}

static void mixer_feedback_decay_complex(const SimMixerOperatorConfig* cfg,
                                         double                        dt,
                                         SimComplexDouble*             out,
                                         size_t                        count) {
    if (cfg == NULL || out == NULL || count == 0U) {
        return;
    }

    double epsilon = cfg->feedback_epsilon;
    double decay_scale =
        exp(-cfg->lhs_gain * fmax(dt, 0.0) *
            ((cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_FEEDBACK) ? epsilon : 1.0));

    for (size_t i = 0; i < count; ++i) {
        double complex P = out[i].re + I * out[i].im;
        double complex M = decay_scale * P;
        if (!mixer_sample_finite(M))
            continue;
        out[i].re = creal(M);
        out[i].im = cimag(M);
    }
}

static void
mixer_feedback_decay_real(const SimMixerOperatorConfig* cfg, double dt, double* out, size_t count) {
    if (cfg == NULL || out == NULL || count == 0U) {
        return;
    }

    double epsilon = cfg->feedback_epsilon;
    double decay_scale =
        exp(-cfg->lhs_gain * fmax(dt, 0.0) *
            ((cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_FEEDBACK) ? epsilon : 1.0));

    for (size_t i = 0; i < count; ++i) {
        double M = decay_scale * out[i];
        if (!isfinite(M))
            continue;
        out[i] = M;
    }
}

static void mixer_feedback_inject_complex(const SimMixerOperatorConfig* cfg,
                                          const SimComplexDouble*       lhs_c,
                                          const double*                 lhs_r,
                                          const SimComplexDouble*       rhs_c,
                                          const double*                 rhs_r,
                                          SimComplexDouble*             out,
                                          size_t                        count) {
    if (cfg == NULL || out == NULL || count == 0U) {
        return;
    }

    double epsilon = cfg->feedback_epsilon;
    double input_scale =
        (cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_INPUT) ? epsilon : 1.0;
    double lhs_gain  = cfg->lhs_gain;
    double rhs_gain  = cfg->rhs_gain;
    double rhs_scale = rhs_gain * input_scale;
    double bias      = cfg->bias;

    for (size_t i = 0; i < count; ++i) {
        double complex L =
            lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
        double complex R =
            rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
        double complex M = rhs_scale * (L + R) + bias;
        if (!mixer_sample_finite(M))
            continue;
        out[i].re += creal(M);
        out[i].im += cimag(M);
    }
}

static void mixer_feedback_inject_real(const SimMixerOperatorConfig* cfg,
                                       const double*                 lhs,
                                       const double*                 rhs,
                                       double*                       out,
                                       size_t                        count) {
    if (cfg == NULL || lhs == NULL || rhs == NULL || out == NULL || count == 0U) {
        return;
    }

    double epsilon = cfg->feedback_epsilon;
    double input_scale =
        (cfg->feedback_epsilon_mode == SIM_MIXER_FEEDBACK_EPS_INPUT) ? epsilon : 1.0;
    double lhs_gain  = cfg->lhs_gain;
    double rhs_gain  = cfg->rhs_gain;
    double rhs_scale = rhs_gain * input_scale;
    double bias      = cfg->bias;

    for (size_t i = 0; i < count; ++i) {
        double M = rhs_scale * (lhs_gain * lhs[i] + rhs_gain * rhs[i]) + bias;
        if (!isfinite(M))
            continue;
        out[i] += M;
    }
}

static bool mixer_feedback_decay_dt(const SimMixerOperatorState* state,
                                    size_t                       substep_index,
                                    double                       dt_sub,
                                    double*                      out_dt) {
    if (!state || !out_dt) {
        return false;
    }

    if (state->config.mode != SIM_MIXER_MODE_FEEDBACK) {
        return false;
    }

    SimMixerFeedbackSplitMode desired = state->config.feedback_split;
    if (desired == SIM_MIXER_FEEDBACK_SPLIT_NONE) {
        return false;
    }

    SimMixerFeedbackSplitMode layout = state->split_layout;

    if (desired == SIM_MIXER_FEEDBACK_SPLIT_LIE) {
        if (layout == SIM_MIXER_FEEDBACK_SPLIT_STRANG && substep_index != 0U) {
            return false;
        }
        *out_dt = (layout == SIM_MIXER_FEEDBACK_SPLIT_STRANG) ? dt_sub * 2.0 : dt_sub;
        return true;
    }

    if (desired == SIM_MIXER_FEEDBACK_SPLIT_STRANG) {
        *out_dt = (layout == SIM_MIXER_FEEDBACK_SPLIT_LIE) ? dt_sub * 0.5 : dt_sub;
        return true;
    }

    return false;
}

static SimResult mixer_feedback_decay_step(void*               state_ptr,
                                           struct SimContext*  context,
                                           struct SimOperator* self,
                                           size_t              substep_index,
                                           double              dt_sub,
                                           void*               scratch,
                                           size_t              scratch_size) {
    (void) self;
    (void) scratch;
    (void) scratch_size;

    SimMixerOperatorState* state = (SimMixerOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double decay_dt = 0.0;
    if (!mixer_feedback_decay_dt(state, substep_index, dt_sub, &decay_dt)) {
        return SIM_RESULT_OK;
    }

    SimField* out_field = sim_context_field(context, state->config.output_field);
    if (!out_field) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_is_complex(out_field)) {
        SimComplexDouble* out   = sim_field_complex_data(out_field);
        size_t            count = sim_field_element_count(&out_field->layout);
        if (!out || count == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        mixer_feedback_decay_complex(&state->config, decay_dt, out, count);
    } else {
        double* out   = sim_field_real_data(out_field);
        size_t  count = sim_field_bytes(out_field) / sizeof(double);
        if (!out || count == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        mixer_feedback_decay_real(&state->config, decay_dt, out, count);
    }

    return SIM_RESULT_OK;
}

static SimResult mixer_feedback_inject_step(void*               state_ptr,
                                            struct SimContext*  context,
                                            struct SimOperator* self,
                                            size_t              substep_index,
                                            double              dt_sub,
                                            void*               scratch,
                                            size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    SimMixerOperatorState* state = (SimMixerOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->config.mode != SIM_MIXER_MODE_FEEDBACK ||
        state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_NONE) {
        return mixer_apply(state_ptr, context, self, dt_sub);
    }

    SimMixerFeedbackSplitMode layout = state->split_layout;
    bool   post_decay    = (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG &&
                            layout == SIM_MIXER_FEEDBACK_SPLIT_LIE);
    double post_decay_dt = dt_sub * 0.5;

    SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField* out_field = sim_context_field(context, state->config.output_field);
    {
        SimResult validation = mixer_validate_fields(lhs_field, rhs_field, out_field);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    bool lhs_complex = sim_field_is_complex(lhs_field);
    bool rhs_complex = sim_field_is_complex(rhs_field);
    bool out_complex = sim_field_is_complex(out_field);
    bool any_complex = lhs_complex || rhs_complex || out_complex;

    size_t lhs_count = sim_field_element_count(&lhs_field->layout);
    size_t rhs_count = sim_field_element_count(&rhs_field->layout);
    size_t out_count = sim_field_element_count(&out_field->layout);

    if (any_complex) {
        SimComplexDouble*       out = sim_field_complex_data(out_field);
        const SimComplexDouble* lhs_c =
            lhs_complex ? sim_field_complex_data_const(lhs_field) : NULL;
        const SimComplexDouble* rhs_c =
            rhs_complex ? sim_field_complex_data_const(rhs_field) : NULL;
        const double* lhs_r =
            (!lhs_complex) ? (const double*) sim_field_data_const(lhs_field) : NULL;
        const double* rhs_r =
            (!rhs_complex) ? (const double*) sim_field_data_const(rhs_field) : NULL;
        if (!out || (!lhs_complex && !lhs_r) || (!rhs_complex && !rhs_r)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        mixer_feedback_inject_complex(&state->config, lhs_c, lhs_r, rhs_c, rhs_r, out, out_count);
        if (post_decay) {
            mixer_feedback_decay_complex(&state->config, post_decay_dt, out, out_count);
        }
    } else {
        double* lhs = sim_field_real_data(lhs_field);
        double* rhs = sim_field_real_data(rhs_field);
        double* out = sim_field_real_data(out_field);
        if (!lhs || !rhs || !out) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        size_t count = sim_field_bytes(lhs_field) / sizeof(double);
        mixer_feedback_inject_real(&state->config, lhs, rhs, out, count);
        if (post_decay) {
            mixer_feedback_decay_real(&state->config, post_decay_dt, out, count);
        }
    }

    return SIM_RESULT_OK;
}

static SimResult mixer_step(void*               state_ptr,
                            struct SimContext*  context,
                            struct SimOperator* self,
                            size_t              substep_index,
                            double              dt_sub,
                            void*               scratch,
                            size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return mixer_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_mixer_operator(struct SimContext*            context,
                                 const SimMixerOperatorConfig* config,
                                 size_t*                       out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimMixerOperatorState* state =
        (SimMixerOperatorState*) calloc(1U, sizeof(SimMixerOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimMixerOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.lhs_field             = 0U;
        local.rhs_field             = 0U;
        local.output_field          = 0U;
        local.lhs_gain              = 1.0;
        local.rhs_gain              = 1.0;
        local.mix                   = 0.5;
        local.bias                  = 0.0;
        local.feedback_epsilon      = 1.0;
        local.mode                  = SIM_MIXER_MODE_LINEAR;
        local.feedback_epsilon_mode = SIM_MIXER_FEEDBACK_EPS_INPUT;
        local.feedback_split        = SIM_MIXER_FEEDBACK_SPLIT_NONE;
        local.accumulate            = false;
        local.scale_by_dt           = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "mixer", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    mixer_normalize_config(&local);
    state->config = local;
    mixer_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "mixer");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_COUPLING;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "mixer";
    sim_operator_info_set_schema_identity(&info, "mixer");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;

    SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField* out_field = sim_context_field(context, state->config.output_field);
    {
        SimResult validation = mixer_validate_fields(lhs_field, rhs_field, out_field);
        if (validation != SIM_RESULT_OK) {
            mixer_destroy(state);
            return validation;
        }
    }
    bool needs_complex =
        (lhs_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(lhs_field))) ||
        (rhs_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(rhs_field))) ||
        (out_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(out_field)));
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    bool use_feedback_split = (state->config.mode == SIM_MIXER_MODE_FEEDBACK &&
                               state->config.feedback_split != SIM_MIXER_FEEDBACK_SPLIT_NONE);
    state->split_layout =
        use_feedback_split ? state->config.feedback_split : SIM_MIXER_FEEDBACK_SPLIT_NONE;
    if (use_feedback_split) {
        info.approximation.temporal_order =
            (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) ? 2.0 : 1.0;
    }
    SimResult result            = SIM_RESULT_OK;
    bool      registered_kernel = false;
    bool      allow_kernel = !(state->config.accumulate && state->config.scale_by_dt) &&
                             state->config.mode != SIM_MIXER_MODE_FEEDBACK && !use_feedback_split;

    /* Attempt kernel-backed registration when IR builder present, fields have matching components, and a backend is configured */
    if (allow_kernel && sim_operator_should_register_kernel_for_schema(
                            context, NULL, 0ULL, SIM_DET_NONE, "mixer")) {
        SimIRBuilder* builder = sim_context_ir_builder(context);
        if (builder != NULL) {
            SimOperatorKernelBindingDescriptor bindings[3];
            SimOperatorKernelOutputDescriptor  outputs[1];
            SimOperatorKernelDescriptor        kernel_desc = { 0 };

            if (lhs_field != NULL && rhs_field != NULL && out_field != NULL) {
                size_t lhs_c = sim_field_components(lhs_field);
                size_t rhs_c = sim_field_components(rhs_field);
                size_t dst_c = sim_field_components(out_field);
                if (lhs_c == rhs_c && lhs_c == dst_c) {
                    size_t    comps = lhs_c;
                    SimIRType vec_type =
                        (comps <= 1U) ? sim_ir_type_scalar() : sim_ir_type_vector(comps);
                    vec_type.scalar_domain =
                        needs_complex ? sim_scalar_domain_c64() : sim_scalar_domain_f64();
                    bool mode_supported = true;
                    bool allows_complex = true;
                    switch (state->config.mode) {
                        case SIM_MIXER_MODE_FM:
                        case SIM_MIXER_MODE_PM:
                        case SIM_MIXER_MODE_MAX:
                        case SIM_MIXER_MODE_MIN:
                        case SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE:
                            allows_complex = false;
                            break;
                        case SIM_MIXER_MODE_RING_MOD:
                            allows_complex = false;
                            break;
                        default:
                            break;
                    }

                    if (needs_complex && !allows_complex) {
                        mode_supported = false;
                    }

                    if (mode_supported) {
                        SimIRNodeId lhs_node =
                            sim_ir_builder_field_ref_typed(builder, 0U, vec_type);
                        SimIRNodeId rhs_node =
                            sim_ir_builder_field_ref_typed(builder, 1U, vec_type);
                        SimIRNodeId out_node =
                            sim_ir_builder_field_ref_typed(builder, 2U, vec_type);

                        if (lhs_node != SIM_IR_INVALID_NODE && rhs_node != SIM_IR_INVALID_NODE &&
                            out_node != SIM_IR_INVALID_NODE) {
                            SimIRNodeId lhs_gain = sim_ir_builder_constant_typed(
                                builder, state->config.lhs_gain, vec_type);
                            SimIRNodeId rhs_gain = sim_ir_builder_constant_typed(
                                builder, state->config.rhs_gain, vec_type);
                            SimIRNodeId bias = sim_ir_builder_constant_typed(
                                builder, state->config.bias, vec_type);

                            SimIRNodeId lhs_scaled =
                                sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, lhs_node, lhs_gain);
                            SimIRNodeId rhs_scaled =
                                sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, rhs_node, rhs_gain);

                            if (lhs_scaled != SIM_IR_INVALID_NODE &&
                                rhs_scaled != SIM_IR_INVALID_NODE) {
                                SimIRNodeId mixed = SIM_IR_INVALID_NODE;
                                switch (state->config.mode) {
                                    case SIM_MIXER_MODE_MULTIPLY:
                                        mixed = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_MUL, lhs_scaled, rhs_scaled);
                                        break;
                                    case SIM_MIXER_MODE_CROSSFADE: {
                                        SimIRNodeId one_minus = sim_ir_builder_constant_typed(
                                            builder, 1.0 - state->config.mix, vec_type);
                                        SimIRNodeId mix_node = sim_ir_builder_constant_typed(
                                            builder, state->config.mix, vec_type);
                                        SimIRNodeId lhs_p = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_MUL, one_minus, lhs_scaled);
                                        SimIRNodeId rhs_p = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_MUL, mix_node, rhs_scaled);
                                        if (lhs_p != SIM_IR_INVALID_NODE &&
                                            rhs_p != SIM_IR_INVALID_NODE) {
                                            mixed = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_ADD, lhs_p, rhs_p);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_SUM:
                                    case SIM_MIXER_MODE_LINEAR:
                                        mixed = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_ADD, lhs_scaled, rhs_scaled);
                                        break;
                                    case SIM_MIXER_MODE_POWER: {
                                        SimIRNodeId base = lhs_scaled;
                                        if (!needs_complex) {
                                            /* Match real-mode semantics: pow(fabs(L), R). */
                                            base = sim_ir_builder_call(
                                                builder, SIM_IR_CALL_ABS, lhs_scaled);
                                        }
                                        if (base != SIM_IR_INVALID_NODE) {
                                            mixed = sim_ir_builder_pow(builder, base, rhs_scaled);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_AM: {
                                        SimIRNodeId one =
                                            sim_ir_builder_constant_typed(builder, 1.0, vec_type);
                                        SimIRNodeId one_plus = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_ADD, one, rhs_scaled);
                                        if (one_plus != SIM_IR_INVALID_NODE) {
                                            mixed = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_MUL, lhs_scaled, one_plus);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_FM:
                                    case SIM_MIXER_MODE_PM: {
                                        SimIRNodeId cos_rhs = sim_ir_builder_call(
                                            builder, SIM_IR_CALL_COS, rhs_scaled);
                                        if (cos_rhs != SIM_IR_INVALID_NODE) {
                                            mixed = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_MUL, lhs_scaled, cos_rhs);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_RING_MOD: {
                                        SimIRNodeId prod = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_MUL, lhs_scaled, rhs_scaled);
                                        SimIRNodeId half =
                                            sim_ir_builder_constant_typed(builder, 0.5, vec_type);
                                        if (prod != SIM_IR_INVALID_NODE &&
                                            half != SIM_IR_INVALID_NODE) {
                                            mixed = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_MUL, prod, half);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_MAX: {
                                        SimIRNodeId sum = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_ADD, lhs_scaled, rhs_scaled);
                                        SimIRNodeId diff = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_SUB, lhs_scaled, rhs_scaled);
                                        SimIRNodeId abs_diff =
                                            sim_ir_builder_call(builder, SIM_IR_CALL_ABS, diff);
                                        SimIRNodeId half =
                                            sim_ir_builder_constant_typed(builder, 0.5, vec_type);
                                        if (sum != SIM_IR_INVALID_NODE &&
                                            abs_diff != SIM_IR_INVALID_NODE &&
                                            half != SIM_IR_INVALID_NODE) {
                                            SimIRNodeId sum_plus = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_ADD, sum, abs_diff);
                                            if (sum_plus != SIM_IR_INVALID_NODE) {
                                                mixed = sim_ir_builder_binary(
                                                    builder, SIM_IR_NODE_MUL, sum_plus, half);
                                            }
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_MIN: {
                                        SimIRNodeId sum = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_ADD, lhs_scaled, rhs_scaled);
                                        SimIRNodeId diff = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_SUB, lhs_scaled, rhs_scaled);
                                        SimIRNodeId abs_diff =
                                            sim_ir_builder_call(builder, SIM_IR_CALL_ABS, diff);
                                        SimIRNodeId half =
                                            sim_ir_builder_constant_typed(builder, 0.5, vec_type);
                                        if (sum != SIM_IR_INVALID_NODE &&
                                            abs_diff != SIM_IR_INVALID_NODE &&
                                            half != SIM_IR_INVALID_NODE) {
                                            SimIRNodeId sum_minus = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_SUB, sum, abs_diff);
                                            if (sum_minus != SIM_IR_INVALID_NODE) {
                                                mixed = sim_ir_builder_binary(
                                                    builder, SIM_IR_NODE_MUL, sum_minus, half);
                                            }
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_AVERAGE: {
                                        SimIRNodeId sum = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_ADD, lhs_scaled, rhs_scaled);
                                        SimIRNodeId half =
                                            sim_ir_builder_constant_typed(builder, 0.5, vec_type);
                                        if (sum != SIM_IR_INVALID_NODE &&
                                            half != SIM_IR_INVALID_NODE) {
                                            mixed = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_MUL, sum, half);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_DIFFERENCE:
                                        mixed = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_SUB, lhs_scaled, rhs_scaled);
                                        break;
                                    case SIM_MIXER_MODE_ABSOLUTE_DIFFERENCE: {
                                        SimIRNodeId diff = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_SUB, lhs_scaled, rhs_scaled);
                                        if (diff != SIM_IR_INVALID_NODE) {
                                            mixed =
                                                sim_ir_builder_call(builder, SIM_IR_CALL_ABS, diff);
                                        }
                                        break;
                                    }
                                    case SIM_MIXER_MODE_FEEDBACK:
                                    default:
                                        break;
                                }

                                if (mixed != SIM_IR_INVALID_NODE) {
                                    SimIRNodeId biased = sim_ir_builder_binary(
                                        builder, SIM_IR_NODE_ADD, mixed, bias);
                                    SimIRNodeId root = biased;
                                    if (state->config.accumulate) {
                                        root = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_ADD, out_node, biased);
                                    }

                                    if (root != SIM_IR_INVALID_NODE) {
                                        bindings[0].ir_field_index      = 0U;
                                        bindings[0].context_field_index = state->config.lhs_field;
                                        bindings[1].ir_field_index      = 1U;
                                        bindings[1].context_field_index = state->config.rhs_field;
                                        bindings[2].ir_field_index      = 2U;
                                        bindings[2].context_field_index =
                                            state->config.output_field;

                                        outputs[0].ir_field_index = 2U;
                                        outputs[0].expression     = root;

                                        kernel_desc.builder           = builder;
                                        kernel_desc.bindings          = bindings;
                                        kernel_desc.binding_count     = 3U;
                                        kernel_desc.outputs           = outputs;
                                        kernel_desc.output_count      = 1U;
                                        kernel_desc.required_features = 0ULL;

                                        SimOperatorDescriptor kdesc = { 0 };
                                        kdesc.name                  = name;
                                        kdesc.evaluate              = NULL;
                                        kdesc.destroy               = mixer_destroy;
                                        kdesc.userdata              = state;
                                        kdesc.kernel                = &kernel_desc;
                                        kdesc.info                  = info;
                                        /* Populate hazard masks for scheduler */
                                        kdesc.read_mask  = 0ULL;
                                        kdesc.write_mask = 0ULL;
                                        if (state->config.lhs_field < 64U)
                                            kdesc.read_mask |= (1ULL << state->config.lhs_field);
                                        if (state->config.rhs_field < 64U)
                                            kdesc.read_mask |= (1ULL << state->config.rhs_field);
                                        if (state->config.output_field < 64U)
                                            kdesc.write_mask |=
                                                (1ULL << state->config.output_field);

                                        result = sim_context_register_operator(
                                            context, &kdesc, out_index);
                                        if (result == SIM_RESULT_OK) {
                                            registered_kernel = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (!registered_kernel) {
        SimSplitPort ports[3] = {
            { .context_field_index = state->config.lhs_field, .require_complex = needs_complex },
            { .context_field_index = state->config.rhs_field, .require_complex = needs_complex },
            { .context_field_index = state->config.output_field, .require_complex = needs_complex }
        };

        if (use_feedback_split) {
            SimSplitAccess  decay_access       = { .port = 2, .mode = SIM_ACCESS_RW };
            SimSplitAccess  inject_accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                                   { .port = 1, .mode = SIM_ACCESS_READ },
                                                   { .port = 2, .mode = SIM_ACCESS_RW } };
            SimSplitSubstep substeps[3];
            size_t          substep_count = 0U;

            if (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) {
                substeps[substep_count++] = (SimSplitSubstep){ .name = "decay",
                                                               .fn   = mixer_feedback_decay_step,
                                                               .accesses          = &decay_access,
                                                               .access_count      = 1U,
                                                               .dt_scale          = 0.5,
                                                               .barrier_after     = false,
                                                               .error_measure     = NULL,
                                                               .required_features = 0U };
            } else {
                substeps[substep_count++] = (SimSplitSubstep){ .name = "decay",
                                                               .fn   = mixer_feedback_decay_step,
                                                               .accesses          = &decay_access,
                                                               .access_count      = 1U,
                                                               .dt_scale          = 1.0,
                                                               .barrier_after     = false,
                                                               .error_measure     = NULL,
                                                               .required_features = 0U };
            }

            substeps[substep_count++] = (SimSplitSubstep){ .name     = "inject",
                                                           .fn       = mixer_feedback_inject_step,
                                                           .accesses = inject_accesses,
                                                           .access_count      = 3U,
                                                           .dt_scale          = 1.0,
                                                           .barrier_after     = false,
                                                           .error_measure     = NULL,
                                                           .required_features = 0U };

            if (state->config.feedback_split == SIM_MIXER_FEEDBACK_SPLIT_STRANG) {
                substeps[substep_count++] = (SimSplitSubstep){ .name = "decay",
                                                               .fn   = mixer_feedback_decay_step,
                                                               .accesses          = &decay_access,
                                                               .access_count      = 1U,
                                                               .dt_scale          = 0.5,
                                                               .barrier_after     = false,
                                                               .error_measure     = NULL,
                                                               .required_features = 0U };
            }

            SimSplitDescriptor desc = { .name          = name,
                                        .ports         = ports,
                                        .port_count    = 3U,
                                        .substeps      = substeps,
                                        .substep_count = substep_count,
                                        .state         = state,
                                        .symbolic      = mixer_symbolic,
                                        .destroy       = mixer_destroy,
                                        .info          = info,
                                        .scratch       = { 0U, 0U } };

            result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
        } else {
            SimSplitAccess accesses[3] = {
                { .port = 0, .mode = SIM_ACCESS_READ },
                { .port = 1, .mode = SIM_ACCESS_READ },
                { .port = 2, .mode = SIM_ACCESS_RW }
            }; /* output may read-then-write when accumulate */

            SimSplitSubstep substep = { .name              = NULL,
                                        .fn                = mixer_step,
                                        .accesses          = accesses,
                                        .access_count      = 3U,
                                        .dt_scale          = 1.0,
                                        .barrier_after     = false,
                                        .error_measure     = NULL,
                                        .required_features = 0U };

            SimSplitDescriptor desc = { .name          = name,
                                        .ports         = ports,
                                        .port_count    = 3U,
                                        .substeps      = &substep,
                                        .substep_count = 1U,
                                        .state         = state,
                                        .symbolic      = mixer_symbolic,
                                        .destroy       = mixer_destroy,
                                        .info          = info,
                                        .scratch       = { 0U, 0U } };

            result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
        }
        if (result != SIM_RESULT_OK) {
            mixer_destroy(state);
        }
    }

    return result;
}

SimResult sim_mixer_config(struct SimContext*      context,
                           size_t                  operator_index,
                           SimMixerOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMixerOperatorState* state = (SimMixerOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_mixer_update(struct SimContext*            context,
                           size_t                        operator_index,
                           const SimMixerOperatorConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMixerOperatorState* state = (SimMixerOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimMixerOperatorConfig local = state->config;
    if (config) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "mixer", true, config->scale_by_dt);
    }

    mixer_normalize_config(&local);
    {
        SimField* lhs_field  = sim_context_field(context, local.lhs_field);
        SimField* rhs_field  = sim_context_field(context, local.rhs_field);
        SimField* out_field  = sim_context_field(context, local.output_field);
        SimResult validation = mixer_validate_fields(lhs_field, rhs_field, out_field);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }
    state->config = local;
    mixer_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
