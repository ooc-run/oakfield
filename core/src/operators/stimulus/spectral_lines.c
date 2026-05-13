#include "oakfield/operators/stimulus/spectral_lines.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_SPECTRAL_LINES_EPS 1.0e-9
#define STIM_SPECTRAL_LINES_MAX_HARMONICS 64U
#define STIM_SPECTRAL_LINES_MAX_TWIST_Q 64U
#define STIM_SPECTRAL_LINES_RENORM_INTERVAL 32U
#define STIM_SPECTRAL_LINES_VDSP_MIN_LEN 64U
#define STIM_SPECTRAL_LINES_COEF_IM_EPS 1.0e-12

#if defined(__APPLE__)
static inline void stim_spectral_lines_sincos(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void stim_spectral_lines_sincos(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void stim_spectral_lines_sincos(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

typedef struct SimStimulusSpectralLinesState {
    SimStimulusSpectralLinesConfig      config;
    SimClockMode                        clock_mode;
    double                              locked_time;
    size_t                              last_step_index;
    bool                                clock_initialized;
    double                              snapshot_locked_time;
    size_t                              snapshot_last_step_index;
    bool                                snapshot_clock_initialized;
    char                                symbolic[160];
    double                              harmonic_coef_re[STIM_SPECTRAL_LINES_MAX_HARMONICS];
    double                              harmonic_coef_im[STIM_SPECTRAL_LINES_MAX_HARMONICS];
    bool                                harmonic_coeffs_complex;
    unsigned int                        harmonic_coeffs_count;
    double                              harmonic_coeffs_power;
    double                              harmonic_coeffs_amplitude;
    SimStimulusSpectralLinesTwistKind   harmonic_coeffs_twist_kind;
    unsigned int                        harmonic_coeffs_twist_q;
    unsigned int                        harmonic_coeffs_twist_k;
    SimStimulusSpectralLinesTwistPreset harmonic_coeffs_twist_preset;
    bool                                harmonic_coeffs_twist_zero_non_units;
    bool                                harmonic_coeffs_twist_table_is_complex;
    uint64_t                            harmonic_coeffs_twist_table_version;
    double                              twist_table_re[STIM_SPECTRAL_LINES_MAX_TWIST_Q];
    double                              twist_table_im[STIM_SPECTRAL_LINES_MAX_TWIST_Q];
    unsigned int                        twist_table_q;
    bool                                twist_table_zero_non_units;
    bool                                twist_table_is_complex;
    uint64_t                            twist_table_version;
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta;
    double* vdsp_cos;
    double* vdsp_sin;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
} SimStimulusSpectralLinesState;

static void spectral_lines_normalize(SimStimulusSpectralLinesConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->wavenumber)) {
        config->wavenumber = 0.0;
    }
    if (!isfinite(config->kx)) {
        config->kx = 0.0;
    }
    if (!isfinite(config->ky)) {
        config->ky = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);

    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0) {
        config->nominal_dt = 0.0;
    }
    if (!isfinite(config->harmonic_power) || config->harmonic_power < 0.0) {
        config->harmonic_power = 0.0;
    }
    if (config->harmonic_count == 0U) {
        config->harmonic_count = 1U;
    }
    if (config->harmonic_count > STIM_SPECTRAL_LINES_MAX_HARMONICS) {
        config->harmonic_count = STIM_SPECTRAL_LINES_MAX_HARMONICS;
    }
    if (!config->use_wavevector && (fabs(config->kx) > STIM_SPECTRAL_LINES_EPS ||
                                    fabs(config->ky) > STIM_SPECTRAL_LINES_EPS)) {
        config->use_wavevector = true;
    }

    if ((int) config->twist_kind < (int) SIM_SPECTRAL_LINES_TWIST_NONE ||
        (int) config->twist_kind > (int) SIM_SPECTRAL_LINES_TWIST_DIRICHLET) {
        config->twist_kind = SIM_SPECTRAL_LINES_TWIST_NONE;
    }

    if ((int) config->twist_preset < (int) SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL ||
        (int) config->twist_preset > (int) SIM_SPECTRAL_LINES_TWIST_PRESET_TABLE) {
        config->twist_preset = SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL;
    }

    config->twist_zero_non_units   = config->twist_zero_non_units ? true : false;
    config->twist_table_is_complex = config->twist_table_is_complex ? true : false;

    if (config->twist_kind == SIM_SPECTRAL_LINES_TWIST_ALTERNATING) {
        config->twist_q                = 1U;
        config->twist_k                = 0U;
        config->twist_preset           = SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL;
        config->twist_zero_non_units   = false;
        config->twist_table_is_complex = false;
        return;
    }

    if (config->twist_kind != SIM_SPECTRAL_LINES_TWIST_DIRICHLET) {
        config->twist_q      = 1U;
        config->twist_k      = 0U;
        config->twist_preset = SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL;
        return;
    }

    if (config->twist_q == 0U) {
        config->twist_q = 1U;
    }
    if (config->twist_q > STIM_SPECTRAL_LINES_MAX_TWIST_Q) {
        config->twist_q = STIM_SPECTRAL_LINES_MAX_TWIST_Q;
    }

    if (config->twist_k > 3U) {
        config->twist_k = 3U;
    }
}

static SimClockMode
spectral_lines_resolve_clock_mode(const SimContext*                     context,
                                  const char*                           op_name,
                                  const SimStimulusSpectralLinesConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool forced = false;

    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;

    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_SPECTRAL_LINES_EPS, &forced);

    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_spectral_lines");
    }
    return resolved;
}

static void spectral_lines_refresh_symbolic(SimStimulusSpectralLinesState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusSpectralLinesConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "spectral_lines A=%.3g k0=%.3g H=%u",
                    cfg->amplitude,
                    cfg->wavenumber,
                    cfg->harmonic_count);
#else
    (void) state;
#endif
}

static const char* spectral_lines_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusSpectralLinesState* state = (const SimStimulusSpectralLinesState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static unsigned int stim_spectral_lines_gcd_u32(unsigned int a, unsigned int b) {
    while (b != 0U) {
        unsigned int t = a % b;
        a              = b;
        b              = t;
    }
    return a;
}

static int stim_spectral_lines_jacobi_symbol(unsigned int a, unsigned int n) {
    if ((n == 0U) || ((n & 1U) == 0U)) {
        return 0;
    }

    a %= n;
    int symbol = 1;

    while (a != 0U) {
        while ((a & 1U) == 0U) {
            a >>= 1U;
            unsigned int n_mod8 = n & 7U;
            if (n_mod8 == 3U || n_mod8 == 5U) {
                symbol = -symbol;
            }
        }

        unsigned int t = a;
        a              = n;
        n              = t;

        if (((a & 3U) == 3U) && ((n & 3U) == 3U)) {
            symbol = -symbol;
        }

        a %= n;
    }

    return (n == 1U) ? symbol : 0;
}

static SimStimulusSpectralLinesTwistPreset
spectral_lines_resolve_dirichlet_preset(const SimStimulusSpectralLinesConfig* cfg) {
    if (cfg == NULL) {
        return SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL;
    }
    if (cfg->twist_k == 0U) {
        return cfg->twist_preset;
    }
    switch (cfg->twist_k) {
        case 1U:
            return SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL;
        case 2U:
            return SIM_SPECTRAL_LINES_TWIST_PRESET_CHI4;
        case 3U:
            return SIM_SPECTRAL_LINES_TWIST_PRESET_QUADRATIC;
        default:
            return cfg->twist_preset;
    }
}

static void spectral_lines_dirichlet_preset(unsigned int                        n,
                                            unsigned int                        q,
                                            SimStimulusSpectralLinesTwistPreset preset,
                                            double*                             out_re,
                                            double*                             out_im) {
    if (out_re != NULL) {
        *out_re = 0.0;
    }
    if (out_im != NULL) {
        *out_im = 0.0;
    }
    if (q == 0U || out_re == NULL || out_im == NULL) {
        return;
    }

    unsigned int residue = n % q;
    unsigned int g       = stim_spectral_lines_gcd_u32(residue, q);
    if (g != 1U) {
        return;
    }

    switch (preset) {
        case SIM_SPECTRAL_LINES_TWIST_PRESET_CHI4: {
            unsigned int r4 = residue & 3U;
            if (r4 == 1U) {
                *out_re = 1.0;
            } else if (r4 == 3U) {
                *out_re = -1.0;
            }
            return;
        }
        case SIM_SPECTRAL_LINES_TWIST_PRESET_QUADRATIC:
            if ((q & 1U) == 1U) {
                *out_re = (double) stim_spectral_lines_jacobi_symbol(residue, q);
            } else {
                *out_re = 1.0;
            }
            return;
        case SIM_SPECTRAL_LINES_TWIST_PRESET_TABLE:
            /* Table mode is handled from state storage. */
            return;
        case SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL:
        default:
            *out_re = 1.0;
            return;
    }
}

static void spectral_lines_dirichlet_table(const SimStimulusSpectralLinesState* state,
                                           unsigned int                         n,
                                           bool                                 table_is_complex,
                                           bool                                 zero_non_units,
                                           unsigned int                         q,
                                           double*                              out_re,
                                           double*                              out_im) {
    if (out_re != NULL) {
        *out_re = 0.0;
    }
    if (out_im != NULL) {
        *out_im = 0.0;
    }
    if (state == NULL || out_re == NULL || out_im == NULL || q == 0U || state->twist_table_q != q ||
        q > STIM_SPECTRAL_LINES_MAX_TWIST_Q) {
        return;
    }

    unsigned int residue = n % q;
    if (zero_non_units && stim_spectral_lines_gcd_u32(residue, q) != 1U) {
        return;
    }

    *out_re = state->twist_table_re[residue];
    *out_im = table_is_complex ? state->twist_table_im[residue] : 0.0;
}

static void spectral_lines_refresh_coeffs(SimStimulusSpectralLinesState* state) {
    if (state == NULL) {
        return;
    }

    const SimStimulusSpectralLinesConfig* cfg = &state->config;

    unsigned int harmonics = cfg->harmonic_count;
    if (harmonics > STIM_SPECTRAL_LINES_MAX_HARMONICS) {
        harmonics = STIM_SPECTRAL_LINES_MAX_HARMONICS;
    }

    state->harmonic_coeffs_count                  = harmonics;
    state->harmonic_coeffs_power                  = cfg->harmonic_power;
    state->harmonic_coeffs_amplitude              = cfg->amplitude;
    state->harmonic_coeffs_twist_kind             = cfg->twist_kind;
    state->harmonic_coeffs_twist_q                = cfg->twist_q;
    state->harmonic_coeffs_twist_k                = cfg->twist_k;
    state->harmonic_coeffs_twist_preset           = cfg->twist_preset;
    state->harmonic_coeffs_twist_zero_non_units   = cfg->twist_zero_non_units;
    state->harmonic_coeffs_twist_table_is_complex = cfg->twist_table_is_complex;
    state->harmonic_coeffs_twist_table_version    = state->twist_table_version;

    state->harmonic_coeffs_complex = false;

    if (harmonics == 0U || cfg->amplitude == 0.0) {
        for (unsigned int i = 0U; i < STIM_SPECTRAL_LINES_MAX_HARMONICS; ++i) {
            state->harmonic_coef_re[i] = 0.0;
            state->harmonic_coef_im[i] = 0.0;
        }
        return;
    }

    SimStimulusSpectralLinesTwistPreset preset   = spectral_lines_resolve_dirichlet_preset(cfg);
    bool                                any_imag = false;

    for (unsigned int n = 1U; n <= harmonics; ++n) {
        double base = 1.0;
        if (cfg->harmonic_power > 0.0) {
            base = 1.0 / pow((double) n, cfg->harmonic_power);
        }

        double chi_re = 1.0;
        double chi_im = 0.0;

        switch (cfg->twist_kind) {
            case SIM_SPECTRAL_LINES_TWIST_ALTERNATING:
                chi_re = ((n & 1U) != 0U) ? -1.0 : 1.0;
                break;
            case SIM_SPECTRAL_LINES_TWIST_DIRICHLET: {
                unsigned int q = cfg->twist_q;
                if (q == 0U) {
                    q = 1U;
                } else if (q > STIM_SPECTRAL_LINES_MAX_TWIST_Q) {
                    q = STIM_SPECTRAL_LINES_MAX_TWIST_Q;
                }

                if (preset == SIM_SPECTRAL_LINES_TWIST_PRESET_TABLE) {
                    spectral_lines_dirichlet_table(state,
                                                   n,
                                                   cfg->twist_table_is_complex,
                                                   cfg->twist_zero_non_units,
                                                   q,
                                                   &chi_re,
                                                   &chi_im);
                } else {
                    spectral_lines_dirichlet_preset(n, q, preset, &chi_re, &chi_im);
                }
                break;
            }
            case SIM_SPECTRAL_LINES_TWIST_NONE:
            default:
                break;
        }

        double coef_re = cfg->amplitude * base * chi_re;
        double coef_im = cfg->amplitude * base * chi_im;
        if (!isfinite(coef_re)) {
            coef_re = 0.0;
        }
        if (!isfinite(coef_im)) {
            coef_im = 0.0;
        }

        state->harmonic_coef_re[n - 1U] = coef_re;
        state->harmonic_coef_im[n - 1U] = coef_im;

        if (fabs(coef_im) > STIM_SPECTRAL_LINES_COEF_IM_EPS) {
            any_imag = true;
        }
    }

    for (unsigned int n = harmonics; n < STIM_SPECTRAL_LINES_MAX_HARMONICS; ++n) {
        state->harmonic_coef_re[n] = 0.0;
        state->harmonic_coef_im[n] = 0.0;
    }

    state->harmonic_coeffs_complex = any_imag;
}

static bool spectral_lines_coeffs_stale(const SimStimulusSpectralLinesState* state) {
    if (state == NULL) {
        return false;
    }
    const SimStimulusSpectralLinesConfig* cfg = &state->config;
    return state->harmonic_coeffs_count != cfg->harmonic_count ||
           state->harmonic_coeffs_power != cfg->harmonic_power ||
           state->harmonic_coeffs_amplitude != cfg->amplitude ||
           state->harmonic_coeffs_twist_kind != cfg->twist_kind ||
           state->harmonic_coeffs_twist_q != cfg->twist_q ||
           state->harmonic_coeffs_twist_k != cfg->twist_k ||
           state->harmonic_coeffs_twist_preset != cfg->twist_preset ||
           state->harmonic_coeffs_twist_zero_non_units != cfg->twist_zero_non_units ||
           state->harmonic_coeffs_twist_table_is_complex != cfg->twist_table_is_complex ||
           state->harmonic_coeffs_twist_table_version != state->twist_table_version;
}

static void spectral_lines_eval_base(const double* coef_re,
                                     const double* coef_im,
                                     unsigned int  harmonics,
                                     double        theta_base,
                                     bool          need_sin,
                                     bool          want_im,
                                     double*       out_re,
                                     double*       out_im) {
    double re_sum = 0.0;
    double im_sum = 0.0;

    if (coef_re == NULL || coef_im == NULL || harmonics == 0U) {
        if (out_re != NULL) {
            *out_re = 0.0;
        }
        if (out_im != NULL) {
            *out_im = 0.0;
        }
        return;
    }

    double sin1 = 0.0;
    double cos1 = 0.0;

    stim_spectral_lines_sincos(theta_base, &sin1, &cos1);

    double sin_n = sin1;
    double cos_n = cos1;

    re_sum = coef_re[0] * cos_n;
    if (need_sin) {
        re_sum -= coef_im[0] * sin_n;
    }

    if (want_im && need_sin) {
        im_sum = coef_re[0] * sin_n + coef_im[0] * cos_n;
    }

    unsigned int mask = STIM_SPECTRAL_LINES_RENORM_INTERVAL - 1U;

    for (unsigned int n = 2U; n <= harmonics; ++n) {
        double next_sin = sin_n * cos1 + cos_n * sin1;
        double next_cos = cos_n * cos1 - sin_n * sin1;
        sin_n           = next_sin;
        cos_n           = next_cos;

        double cre = coef_re[n - 1U];
        double cim = coef_im[n - 1U];

        re_sum += cre * cos_n;
        if (need_sin) {
            re_sum -= cim * sin_n;
        }

        if (want_im && need_sin) {
            im_sum += cre * sin_n + cim * cos_n;
        }

        if ((n & mask) == 0U) {
            double theta = (double) n * theta_base;
            stim_spectral_lines_sincos(theta, &sin_n, &cos_n);
        }
    }

    if (out_re != NULL) {
        *out_re = re_sum;
    }
    if (out_im != NULL) {
        *out_im = (want_im && need_sin) ? im_sum : 0.0;
    }
}

static void spectral_lines_accumulate_line(double*       dst_re,
                                           double*       dst_im,
                                           size_t        stride,
                                           size_t        count,
                                           double        theta0,
                                           double        delta,
                                           const double* coef_re,
                                           const double* coef_im,
                                           unsigned int  harmonics,
                                           double        scale,
                                           bool          need_sin,
                                           bool          want_im) {
    if (dst_re == NULL || coef_re == NULL || coef_im == NULL || count == 0U || harmonics == 0U ||
        scale == 0.0) {
        return;
    }

    if (fabs(delta) <= STIM_SPECTRAL_LINES_EPS) {
        double base_re = 0.0;
        double base_im = 0.0;
        spectral_lines_eval_base(
            coef_re, coef_im, harmonics, theta0, need_sin, want_im, &base_re, &base_im);
        double value_re = scale * base_re;
        double value_im = scale * base_im;
        for (size_t i = 0U; i < count; ++i) {
            dst_re[i * stride] += value_re;
            if (want_im && dst_im != NULL) {
                dst_im[i * stride] += value_im;
            }
        }
        return;
    }

    size_t mask = STIM_SPECTRAL_LINES_RENORM_INTERVAL - 1U;

    for (unsigned int n = 1U; n <= harmonics; ++n) {
        double cre = scale * coef_re[n - 1U];
        double cim = scale * coef_im[n - 1U];
        if (cre == 0.0 && cim == 0.0) {
            continue;
        }

        double theta_n0  = theta0 * (double) n;
        double delta_n   = delta * (double) n;
        double sin_n     = 0.0;
        double cos_n     = 0.0;
        double sin_delta = 0.0;
        double cos_delta = 0.0;

        stim_spectral_lines_sincos(theta_n0, &sin_n, &cos_n);
        stim_spectral_lines_sincos(delta_n, &sin_delta, &cos_delta);

        for (size_t i = 0U; i < count; ++i) {
            dst_re[i * stride] += cre * cos_n;
            if (need_sin) {
                dst_re[i * stride] -= cim * sin_n;
            }

            if (want_im && need_sin && dst_im != NULL) {
                dst_im[i * stride] += cre * sin_n + cim * cos_n;
            }

            double next_sin = sin_n * cos_delta + cos_n * sin_delta;
            double next_cos = cos_n * cos_delta - sin_n * sin_delta;
            sin_n           = next_sin;
            cos_n           = next_cos;

            if (((i + 1U) & mask) == 0U) {
                double theta = theta_n0 + delta_n * (double) (i + 1U);
                stim_spectral_lines_sincos(theta, &sin_n, &cos_n);
            }
        }
    }
}

static bool spectral_lines_try_linear_rows(const SimStimulusSpectralLinesConfig* cfg,
                                           const SimField*                       field,
                                           bool                                  is_complex,
                                           double*                               dst_real,
                                           SimComplexDouble*                     dst_complex,
                                           size_t                                count,
                                           double                                scale,
                                           double                                k0,
                                           double                                theta_time,
                                           const double*                         coef_re,
                                           const double*                         coef_im,
                                           unsigned int                          harmonics,
                                           bool harmonic_coeffs_complex,
                                           bool use_wavevector) {
    if (cfg == NULL || field == NULL || coef_re == NULL || coef_im == NULL) {
        return false;
    }
    if (is_complex) {
        if (dst_complex == NULL) {
            return false;
        }
    } else {
        if (dst_real == NULL) {
            return false;
        }
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }
    if (count == 0U || harmonics == 0U) {
        return false;
    }

    double ax = 0.0;
    double by = 0.0;

    if (use_wavevector) {
        ax = cfg->kx;
        by = cfg->ky;
    } else {
        switch (cfg->coord.mode) {
            case SIM_STIMULUS_COORD_AXIS:
                if (cfg->coord.axis == SIM_STIMULUS_AXIS_X) {
                    ax = k0;
                } else {
                    by = k0;
                }
                break;
            case SIM_STIMULUS_COORD_ANGLE: {
                double angle_s = 0.0;
                double angle_c = 0.0;
                stim_spectral_lines_sincos(cfg->coord.angle, &angle_s, &angle_c);
                ax = k0 * angle_c;
                by = k0 * angle_s;
                break;
            }
            default:
                return false;
        }
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);

    if (width == 0U || height == 0U) {
        return false;
    }

    if (width > SIZE_MAX / height) {
        return false;
    }

    size_t plane = width * height;

    if (plane == 0U || count % plane != 0U) {
        return false;
    }

    size_t planes = count / plane;

    if (scale == 0.0) {
        return true;
    }

    double origin_x  = cfg->coord.origin_x;
    double spacing_x = cfg->coord.spacing_x;
    double origin_y  = cfg->coord.origin_y;
    double spacing_y = cfg->coord.spacing_y;

    double base_x = ax * origin_x;
    double delta  = ax * spacing_x;

    for (size_t plane_idx = 0U; plane_idx < planes; ++plane_idx) {
        double*           plane_real    = dst_real ? dst_real + plane_idx * plane : NULL;
        SimComplexDouble* plane_complex = dst_complex ? dst_complex + plane_idx * plane : NULL;

        for (size_t iy = 0U; iy < height; ++iy) {
            double y          = origin_y + (double) iy * spacing_y;
            double theta0     = theta_time + base_x + by * y;
            size_t row_offset = iy * width;

            if (!is_complex) {
                double* row_ptr = plane_real + row_offset;
                spectral_lines_accumulate_line(row_ptr,
                                               NULL,
                                               1U,
                                               width,
                                               theta0,
                                               delta,
                                               coef_re,
                                               coef_im,
                                               harmonics,
                                               scale,
                                               harmonic_coeffs_complex,
                                               false);
            } else {
                SimComplexDouble* row_complex = plane_complex + row_offset;
                double*           row_re      = &row_complex[0].re;
                double*           row_im      = &row_complex[0].im;
                spectral_lines_accumulate_line(row_re,
                                               row_im,
                                               2U,
                                               width,
                                               theta0,
                                               delta,
                                               coef_re,
                                               coef_im,
                                               harmonics,
                                               scale,
                                               true,
                                               true);
            }
        }
    }

    return true;
}

#if defined(SIM_HAVE_VDSP)
static bool spectral_lines_vdsp_ensure_buffers(SimStimulusSpectralLinesState* state,
                                               size_t                         length) {
    if (state == NULL || length == 0U) {
        return false;
    }
    if (state->vdsp_block != NULL && state->vdsp_capacity >= length) {
        return true;
    }

    if (length > SIZE_MAX / (5U * sizeof(double))) {
        return false;
    }

    double* block = (double*) realloc(state->vdsp_block, length * 5U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = length;
    state->vdsp_theta    = block;
    state->vdsp_cos      = block + length;
    state->vdsp_sin      = block + length * 2U;
    state->vdsp_accum_re = block + length * 3U;
    state->vdsp_accum_im = block + length * 4U;
    return true;
}

static bool spectral_lines_try_vdsp_axis_x(SimStimulusSpectralLinesState* state,
                                           const SimField*                field,
                                           bool                           is_complex,
                                           double*                        dst_real,
                                           SimComplexDouble*              dst_complex,
                                           size_t                         count,
                                           double                         scale,
                                           double                         k0,
                                           double                         theta_time,
                                           const double*                  coef_re,
                                           unsigned int                   harmonics) {
    if (state == NULL || field == NULL || coef_re == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width = sim_field_width(field);
    if (width == 0U || width > count || width < STIM_SPECTRAL_LINES_VDSP_MIN_LEN) {
        return false;
    }

    size_t row_count = count / width;
    if (row_count == 0U || row_count * width != count) {
        return false;
    }
    if (width > (size_t) INT_MAX) {
        return false;
    }

    if (!spectral_lines_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    if (!isfinite(theta_time) || !isfinite(k0) || !isfinite(scale) ||
        !isfinite(state->config.coord.origin_x) || !isfinite(state->config.coord.spacing_x)) {
        return false;
    }

    vDSP_Length len        = (vDSP_Length) width;
    int         vforce_len = (int) width;
    double*     theta      = state->vdsp_theta;
    double*     cos_buf    = state->vdsp_cos;
    double*     sin_buf    = state->vdsp_sin;
    double*     accum_re   = state->vdsp_accum_re;
    double*     accum_im   = state->vdsp_accum_im;

    vDSP_vclrD(accum_re, 1, len);
    if (is_complex) {
        vDSP_vclrD(accum_im, 1, len);
    }

    double theta0 = k0 * state->config.coord.origin_x + theta_time;
    double delta  = k0 * state->config.coord.spacing_x;
    if (scale == 0.0) {
        return true;
    }

    for (unsigned int n = 1U; n <= harmonics; ++n) {
        double factor = (double) n;
        double start  = theta0 * factor;
        double step   = delta * factor;
        vDSP_vrampD(&start, &step, theta, 1, len);
        if (is_complex) {
            vvsincos(sin_buf, cos_buf, theta, &vforce_len);
        } else {
            vvcos(cos_buf, theta, &vforce_len);
        }

        double weight = scale * coef_re[n - 1U];
        vDSP_vsmaD(cos_buf, 1, &weight, accum_re, 1, accum_re, 1, len);
        if (is_complex) {
            vDSP_vsmaD(sin_buf, 1, &weight, accum_im, 1, accum_im, 1, len);
        }
    }

    if (!is_complex) {
        if (dst_real == NULL) {
            return false;
        }
        for (size_t row = 0U; row < row_count; ++row) {
            double* row_ptr = dst_real + row * width;
            vDSP_vaddD(row_ptr, 1, accum_re, 1, row_ptr, 1, len);
        }
        return true;
    }

    if (dst_complex == NULL) {
        return false;
    }

    double* dst_re = &dst_complex[0].re;
    double* dst_im = &dst_complex[0].im;
    for (size_t row = 0U; row < row_count; ++row) {
        size_t  offset = row * width * 2U;
        double* row_re = dst_re + offset;
        double* row_im = dst_im + offset;
        vDSP_vaddD(row_re, 2, accum_re, 1, row_re, 2, len);
        vDSP_vaddD(row_im, 2, accum_im, 1, row_im, 2, len);
    }

    return true;
}
#endif

typedef struct StimulusSpectralLinesIRParams {
    bool needs_dt;
    bool needs_step_index;
    bool needs_time;
} StimulusSpectralLinesIRParams;

static SimIRNodeId spectral_lines_ir_binary(SimIRBuilder* builder,
                                            SimIRNodeType type,
                                            SimIRNodeId   lhs,
                                            SimIRNodeId   rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId
spectral_lines_ir_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    if (operand == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_call(builder, kind, operand);
}

static SimIRNodeId spectral_lines_build_ir(SimIRBuilder*                         builder,
                                           const SimStimulusSpectralLinesConfig* config,
                                           const double*                         coef_re,
                                           unsigned int                          harmonics,
                                           bool                                  complex_output,
                                           StimulusSpectralLinesIRParams*        params) {
    if (builder == NULL || config == NULL || coef_re == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt         = false;
    params->needs_step_index = false;
    params->needs_time       = false;

    if (harmonics == 0U) {
        if (complex_output) {
            return sim_ir_builder_constant_complex(builder, 0.0, 0.0);
        }
        return sim_ir_builder_constant(builder, 0.0);
    }

    SimIRNodeId index    = sim_ir_builder_index(builder);
    SimIRNodeId spacing  = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin   = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId x_offset = spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, spacing, index);
    SimIRNodeId x        = spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, origin, x_offset);

    if (x == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    bool        needs_dt  = config->scale_by_dt ||
                            (config->fixed_clock && config->nominal_dt <= STIM_SPECTRAL_LINES_EPS);
    SimIRNodeId dt_scaled = SIM_IR_INVALID_NODE;
    if (needs_dt) {
        params->needs_dt    = true;
        SimIRNodeId dt_node = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
        dt_scaled           = dt_node;
    }

    SimIRNodeId t = SIM_IR_INVALID_NODE;
    if (config->fixed_clock) {
        params->needs_step_index = true;
        SimIRNodeId step_index   = sim_ir_builder_param(builder, SIM_IR_PARAM_STEP_INDEX);
        SimIRNodeId increment    = SIM_IR_INVALID_NODE;
        if (config->nominal_dt > STIM_SPECTRAL_LINES_EPS) {
            increment = sim_ir_builder_constant(builder, config->nominal_dt);
        } else {
            increment = dt_scaled;
        }
        SimIRNodeId scaled_step =
            spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, step_index, increment);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, scaled_step, time_offset);
    } else {
        params->needs_time      = true;
        SimIRNodeId time_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_TIME);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, time_node, time_offset);
    }

    if (t == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId wavenumber = sim_ir_builder_constant(builder, config->wavenumber);
    SimIRNodeId omega      = sim_ir_builder_constant(builder, config->omega);
    SimIRNodeId phase      = sim_ir_builder_constant(builder, config->phase);
    SimIRNodeId kx         = spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, wavenumber, x);
    SimIRNodeId omega_t    = spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, omega, t);
    SimIRNodeId phase_t    = spectral_lines_ir_binary(builder, SIM_IR_NODE_SUB, kx, omega_t);
    SimIRNodeId theta_base = spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, phase_t, phase);

    SimIRNodeId sum_re = SIM_IR_INVALID_NODE;
    SimIRNodeId sum_im = SIM_IR_INVALID_NODE;
    for (unsigned int n = 1U; n <= harmonics; ++n) {
        double      weight      = coef_re[n - 1U];
        SimIRNodeId weight_node = sim_ir_builder_constant(builder, weight);
        SimIRNodeId n_node      = sim_ir_builder_constant(builder, (double) n);
        SimIRNodeId theta = spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, n_node, theta_base);
        SimIRNodeId cos_term = spectral_lines_ir_call(builder, SIM_IR_CALL_COS, theta);
        SimIRNodeId term_re =
            spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, weight_node, cos_term);
        if (sum_re == SIM_IR_INVALID_NODE) {
            sum_re = term_re;
        } else {
            sum_re = spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, sum_re, term_re);
        }
        if (complex_output) {
            SimIRNodeId sin_term = spectral_lines_ir_call(builder, SIM_IR_CALL_SIN, theta);
            SimIRNodeId term_im =
                spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, weight_node, sin_term);
            if (sum_im == SIM_IR_INVALID_NODE) {
                sum_im = term_im;
            } else {
                sum_im = spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, sum_im, term_im);
            }
        }
    }

    if (sum_re == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId scale = SIM_IR_INVALID_NODE;
    if (config->scale_by_dt) {
        scale = dt_scaled;
    } else {
        scale = sim_ir_builder_constant(builder, 1.0);
    }

    if (complex_output) {
        if (sum_im == SIM_IR_INVALID_NODE) {
            sum_im = sim_ir_builder_constant(builder, 0.0);
        }
        SimIRNodeId scaled_re = spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, sum_re, scale);
        SimIRNodeId scaled_im = spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, sum_im, scale);
        return sim_ir_builder_complex_pack(builder, scaled_re, scaled_im);
    }

    return spectral_lines_ir_binary(builder, SIM_IR_NODE_MUL, sum_re, scale);
}

static void spectral_lines_destroy(void* state_ptr) {
    SimStimulusSpectralLinesState* state = (SimStimulusSpectralLinesState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block = NULL;
    }
#endif
    free(state);
}

static SimResult
spectral_lines_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusSpectralLinesState* state = (SimStimulusSpectralLinesState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->snapshot_locked_time       = state->locked_time;
    state->snapshot_last_step_index   = state->last_step_index;
    state->snapshot_clock_initialized = state->clock_initialized;
    return SIM_RESULT_OK;
}

static SimResult
spectral_lines_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;

    SimStimulusSpectralLinesState* state = (SimStimulusSpectralLinesState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->locked_time       = state->snapshot_locked_time;
    state->last_step_index   = state->snapshot_last_step_index;
    state->clock_initialized = state->snapshot_clock_initialized;
    return SIM_RESULT_OK;
}

static double spectral_lines_drive_time(SimStimulusSpectralLinesState* state,
                                        double                         base_time,
                                        double                         dt,
                                        size_t                         step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_SPECTRAL_LINES_EPS) {
                return ((double) step_index) * state->config.nominal_dt + state->config.time_offset;
            }
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_ACCUMULATED_STATEFUL:
        default:
            break;
    }

    if (!state->config.fixed_clock) {
        state->clock_initialized = false;
        return current_time;
    }

    double increment =
        (state->config.nominal_dt > STIM_SPECTRAL_LINES_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

static SimResult spectral_lines_step(void*               state_ptr,
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

    SimStimulusSpectralLinesState* state = (SimStimulusSpectralLinesState*) state_ptr;
    if (state == NULL || context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusSpectralLinesConfig* cfg   = &state->config;
    SimField*                       field = sim_context_field(context, cfg->field_index);
    if (field == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    void* raw_data = sim_field_data(field);
    if (raw_data == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (!is_complex) {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        if (field->element_size != sizeof(SimComplexDouble))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double t          = spectral_lines_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U || cfg->amplitude == 0.0 || cfg->harmonic_count == 0U) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    double       scale          = cfg->scale_by_dt ? dt_sub : 1.0;
    double       k0             = cfg->wavenumber;
    double       w0             = cfg->omega;
    double       phase0         = cfg->phase;
    unsigned int H              = cfg->harmonic_count;
    bool         separable      = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool         use_wavevector = cfg->use_wavevector;
    double       theta_time     = -w0 * t + phase0;
    if (spectral_lines_coeffs_stale(state)) {
        spectral_lines_refresh_coeffs(state);
    }
    H                     = state->harmonic_coeffs_count;
    const double* coef_re = state->harmonic_coef_re;
    const double* coef_im = state->harmonic_coef_im;
    bool          separable_multiply =
        (!use_wavevector && separable && cfg->coord.combine == SIM_STIMULUS_SEPARABLE_MULTIPLY);
    bool need_sin = is_complex || state->harmonic_coeffs_complex || separable_multiply;
    bool want_im  = is_complex || separable_multiply;
    bool stationary_coord_velocity = fabs(cfg->coord.velocity_x) <= STIM_SPECTRAL_LINES_EPS &&
                                     fabs(cfg->coord.velocity_y) <= STIM_SPECTRAL_LINES_EPS;

    bool use_axis  = (!use_wavevector && !separable && cfg->coord.mode == SIM_STIMULUS_COORD_AXIS);
    bool axis_is_y = (cfg->coord.axis == SIM_STIMULUS_AXIS_Y);
    bool use_angle = (!use_wavevector && !separable && cfg->coord.mode == SIM_STIMULUS_COORD_ANGLE);
    double angle_s = 0.0;
    double angle_c = 1.0;
    if (use_angle) {
        stim_spectral_lines_sincos(cfg->coord.angle, &angle_s, &angle_c);
    }
    bool use_radial =
        (!use_wavevector && !separable && cfg->coord.mode == SIM_STIMULUS_COORD_RADIAL);
    double radial_cx = 0.0;
    double radial_cy = 0.0;
    if (use_radial) {
        radial_cx = cfg->coord.center_x + cfg->coord.velocity_x * t;
        radial_cy = cfg->coord.center_y + cfg->coord.velocity_y * t;
    }

    double*           dst_real    = NULL;
    SimComplexDouble* dst_complex = NULL;
    if (!is_complex) {
        dst_real = (double*) raw_data;
    } else {
        dst_complex = sim_field_complex_data(field);
    }

    if ((!separable || use_wavevector) && stationary_coord_velocity) {
        if (spectral_lines_try_linear_rows(cfg,
                                           field,
                                           is_complex,
                                           dst_real,
                                           dst_complex,
                                           count,
                                           scale,
                                           k0,
                                           theta_time,
                                           coef_re,
                                           coef_im,
                                           H,
                                           state->harmonic_coeffs_complex,
                                           use_wavevector)) {
            state->last_step_index = step_index;
            return SIM_RESULT_OK;
        }
    }

#if defined(SIM_HAVE_VDSP)
    if (!separable && !use_wavevector && stationary_coord_velocity &&
        cfg->coord.mode == SIM_STIMULUS_COORD_AXIS && cfg->coord.axis == SIM_STIMULUS_AXIS_X &&
        !state->harmonic_coeffs_complex) {
        if (spectral_lines_try_vdsp_axis_x(state,
                                           field,
                                           is_complex,
                                           dst_real,
                                           dst_complex,
                                           count,
                                           scale,
                                           k0,
                                           theta_time,
                                           coef_re,
                                           H)) {
            state->last_step_index = step_index;
            return SIM_RESULT_OK;
        }
    }
#endif

    for (size_t i = 0U; i < count; ++i) {
        double x        = 0.0;
        double y        = 0.0;
        double sample_x = 0.0;
        double sample_y = 0.0;
        if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

        double base_re = 0.0;
        double base_im = 0.0;
        if (use_wavevector) {
            double spatial    = cfg->kx * sample_x + cfg->ky * sample_y;
            double theta_base = spatial + theta_time;
            spectral_lines_eval_base(
                coef_re, coef_im, H, theta_base, need_sin, want_im, &base_re, &base_im);
        } else if (separable) {
            double theta_x = k0 * sample_x + theta_time;
            double theta_y = k0 * sample_y + theta_time;
            double fx_re   = 0.0;
            double fx_im   = 0.0;
            double fy_re   = 0.0;
            double fy_im   = 0.0;
            spectral_lines_eval_base(
                coef_re, coef_im, H, theta_x, need_sin, want_im, &fx_re, &fx_im);
            spectral_lines_eval_base(
                coef_re, coef_im, H, theta_y, need_sin, want_im, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = want_im ? (fx_im + fy_im) : 0.0;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double spatial = 0.0;
            double u       = 0.0;
            if (use_axis) {
                u = axis_is_y ? sample_y : sample_x;
            } else if (use_angle) {
                u = sample_x * angle_c + sample_y * angle_s;
            } else if (use_radial) {
                double dx = x - radial_cx;
                double dy = y - radial_cy;
                u         = sqrt(dx * dx + dy * dy);
            } else {
                u = sim_stimulus_coord_u(&cfg->coord, x, y, t);
            }
            spatial           = k0 * u;
            double theta_base = spatial + theta_time;
            spectral_lines_eval_base(
                coef_re, coef_im, H, theta_base, need_sin, want_im, &base_re, &base_im);
        }

        double value_re = base_re;
        double value_im = base_im;
        if (!is_complex) {
            if (isfinite(value_re)) {
                dst_real[i] += scale * value_re;
            }
        } else {
            if (isfinite(value_re) && isfinite(value_im)) {
                dst_complex[i].re += scale * value_re;
                dst_complex[i].im += scale * value_im;
            }
        }
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_spectral_lines_operator(struct SimContext*                    context,
                                                   const SimStimulusSpectralLinesConfig* config,
                                                   size_t* out_index) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusSpectralLinesConfig local = { 0 };
    if (config != NULL)
        local = *config;

    spectral_lines_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_spectral_lines",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusSpectralLinesState* state =
        (SimStimulusSpectralLinesState*) calloc(1U, sizeof(SimStimulusSpectralLinesState));
    if (state == NULL)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config = local;
    state->clock_mode =
        spectral_lines_resolve_clock_mode(context, "stimulus_spectral_lines", &state->config);
    state->locked_time         = 0.0;
    state->last_step_index     = 0U;
    state->clock_initialized   = false;
    state->twist_table_q       = 1U;
    state->twist_table_version = 1U;
    state->twist_table_re[0]   = 1.0;
    state->twist_table_im[0]   = 0.0;
    spectral_lines_refresh_symbolic(state);
    spectral_lines_refresh_coeffs(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_spectral_lines");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stimulus_spectral_lines";
    sim_operator_info_set_schema_identity(&info, "stimulus_spectral_lines");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    bool needs_complex =
        sim_field_is_complex(sim_context_field(context, state->config.field_index));

    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    bool ir_table_mode =
        (local.twist_kind == SIM_SPECTRAL_LINES_TWIST_DIRICHLET &&
         spectral_lines_resolve_dirichlet_preset(&local) == SIM_SPECTRAL_LINES_TWIST_PRESET_TABLE);
    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_spectral_lines") &&
        !state->harmonic_coeffs_complex && !ir_table_mode) {
        SimField* field = sim_context_field(context, local.field_index);
        if (field != NULL) {
            bool is_complex = sim_field_is_complex(field);
            if ((!is_complex && field->element_size != sizeof(double)) ||
                (is_complex && field->element_size != sizeof(SimComplexDouble))) {
                field = NULL;
            }
        }
        if (field != NULL) {
            size_t rank = sim_field_rank(field);
            if (rank != 1U || local.coord.mode != SIM_STIMULUS_COORD_AXIS ||
                local.coord.axis != SIM_STIMULUS_AXIS_X || local.use_wavevector) {
                field = NULL;
            }
        }
        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };
                StimulusSpectralLinesIRParams      ir_params   = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId delta      = spectral_lines_build_ir(builder,
                                                                 &local,
                                                                 state->harmonic_coef_re,
                                                                 state->harmonic_coeffs_count,
                                                                 is_complex,
                                                                 &ir_params);
                SimIRNodeId sum =
                    spectral_lines_ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

                if (field_node != SIM_IR_INVALID_NODE && delta != SIM_IR_INVALID_NODE &&
                    sum != SIM_IR_INVALID_NODE) {
                    int max_param = -1;
                    if (ir_params.needs_dt) {
                        max_param = (int) SIM_IR_PARAM_DT;
                    }
                    if (ir_params.needs_step_index && (int) SIM_IR_PARAM_STEP_INDEX > max_param) {
                        max_param = (int) SIM_IR_PARAM_STEP_INDEX;
                    }
                    if (ir_params.needs_time && (int) SIM_IR_PARAM_TIME > max_param) {
                        max_param = (int) SIM_IR_PARAM_TIME;
                    }
                    size_t param_count = (max_param >= 0) ? (size_t) max_param + 1U : 0U;

                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = sum;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = param_count;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.save_state            = spectral_lines_save;
                    kdesc.restore_state         = spectral_lines_restore;
                    kdesc.destroy               = spectral_lines_destroy;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    if (local.field_index < 64U) {
                        kdesc.read_mask |= (1ULL << local.field_index);
                        kdesc.write_mask |= (1ULL << local.field_index);
                    }

                    result = sim_context_register_operator(context, &kdesc, out_index);
                    if (result == SIM_RESULT_OK) {
                        registered_kernel = true;
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = spectral_lines_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = spectral_lines_symbolic,
                                .save_state    = spectral_lines_save,
                                .restore_state = spectral_lines_restore,
                                .destroy       = spectral_lines_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        spectral_lines_destroy(state);
    }
    return result;
}

static SimStimulusSpectralLinesState* spectral_lines_state_from_operator(SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    if (op->kernel != NULL) {
        return (SimStimulusSpectralLinesState*) sim_operator_payload(op);
    }
    return (SimStimulusSpectralLinesState*) sim_split_state(op);
}

SimResult sim_stimulus_spectral_lines_config(struct SimContext*              context,
                                             size_t                          operator_index,
                                             SimStimulusSpectralLinesConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusSpectralLinesState* state = spectral_lines_state_from_operator(op);
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_spectral_lines_update(struct SimContext*                    context,
                                             size_t                                operator_index,
                                             const SimStimulusSpectralLinesConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusSpectralLinesState* state = spectral_lines_state_from_operator(op);
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    SimStimulusSpectralLinesConfig local = state->config;
    if (config != NULL)
        local = *config;

    spectral_lines_normalize(&local);
    state->config     = local;
    state->clock_mode = spectral_lines_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_spectral_lines"), &state->config);
    spectral_lines_refresh_symbolic(state);
    spectral_lines_refresh_coeffs(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_spectral_lines_set_twist_table(struct SimContext* context,
                                                      size_t             operator_index,
                                                      unsigned int       q,
                                                      const double*      chi_re,
                                                      const double*      chi_im,
                                                      bool               zero_non_units) {
    if (context == NULL || chi_re == NULL || q == 0U || q > STIM_SPECTRAL_LINES_MAX_TWIST_Q) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    if (op->kernel != NULL) {
        /* IR kernels embed constant coefficients and cannot consume mutable χ tables. */
        return SIM_RESULT_NOT_SUPPORTED;
    }

    SimStimulusSpectralLinesState* state = spectral_lines_state_from_operator(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    bool any_imag = false;
    for (unsigned int i = 0U; i < q; ++i) {
        double re = chi_re[i];
        double im = (chi_im != NULL) ? chi_im[i] : 0.0;
        if (!isfinite(re)) {
            re = 0.0;
        }
        if (!isfinite(im)) {
            im = 0.0;
        }
        state->twist_table_re[i] = re;
        state->twist_table_im[i] = im;
        if (fabs(im) > STIM_SPECTRAL_LINES_COEF_IM_EPS) {
            any_imag = true;
        }
    }
    for (unsigned int i = q; i < STIM_SPECTRAL_LINES_MAX_TWIST_Q; ++i) {
        state->twist_table_re[i] = 0.0;
        state->twist_table_im[i] = 0.0;
    }

    state->twist_table_q              = q;
    state->twist_table_zero_non_units = zero_non_units ? true : false;
    state->twist_table_is_complex     = any_imag;
    state->twist_table_version += 1U;

    state->config.twist_kind             = SIM_SPECTRAL_LINES_TWIST_DIRICHLET;
    state->config.twist_preset           = SIM_SPECTRAL_LINES_TWIST_PRESET_TABLE;
    state->config.twist_q                = q;
    state->config.twist_zero_non_units   = state->twist_table_zero_non_units;
    state->config.twist_table_is_complex = any_imag;

    spectral_lines_refresh_coeffs(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
