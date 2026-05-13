#include "oakfield/operators/stimulus/digamma_square.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "sim_accel.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"
#include "oakfield/math/special_functions.h"

#include <limits.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_INV_PI
#define M_INV_PI 0.31830988618379067153776752674502872
#endif

#define STIM_DIGAMMA_SQUARE_RENORM_INTERVAL 256U
#define STIM_DIGAMMA_SQUARE_EPS 1.0e-12
#define STIM_DIGAMMA_SQUARE_A_EPS 1.0e-6
#define STIM_DIGAMMA_SQUARE_VDSP_MIN_LEN 64U

#if defined(__APPLE__)
static inline void sincos_pair(double angle, double* s_out, double* c_out) {
    __sincos(angle, s_out, c_out);
}
#elif defined(__clang__) || defined(__GNUC__)
static inline void sincos_pair(double angle, double* s_out, double* c_out) {
    sincos(angle, s_out, c_out);
}
#else
static inline void sincos_pair(double angle, double* s_out, double* c_out) {
    *s_out = sin(angle);
    *c_out = cos(angle);
}
#endif

static inline SimComplexDouble simcomplex_from_c99(double complex z) {
    SimComplexDouble out = { creal(z), cimag(z) };
    return out;
}

static inline double complex simcomplex_to_c99(SimComplexDouble z) {
    return CMPLX(z.re, z.im);
}

static inline double digamma_square_normalize_a_value(double a) {
    if (!isfinite(a)) {
        return SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT;
    }

    double normalized = fmod(a, 1.0);
    if (normalized < 0.0) {
        normalized += 1.0;
    }
    if (normalized > 0.5) {
        normalized = 1.0 - normalized;
    }
    if (normalized < STIM_DIGAMMA_SQUARE_A_EPS) {
        normalized = STIM_DIGAMMA_SQUARE_A_EPS;
    } else if (normalized > 0.5 - STIM_DIGAMMA_SQUARE_A_EPS) {
        normalized = 0.5 - STIM_DIGAMMA_SQUARE_A_EPS;
    }

    return normalized;
}

static void digamma_square_normalize(SimStimulusDigammaSquareConfig* config) {
    if (!config) {
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
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }
    if (!isfinite(config->harmonics)) {
        config->harmonics = 4.0;
    }
    if (!isfinite(config->a) || fabs(config->a) <= STIM_DIGAMMA_SQUARE_EPS) {
        config->a = SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT;
    } else {
        config->a = digamma_square_normalize_a_value(config->a);
    }
    if (!isfinite(config->warp_mix)) {
        config->warp_mix = 1.0;
    }
    if (!isfinite(config->warp_bias)) {
        config->warp_bias = 0.0;
    }
    switch (config->warp_mode) {
        case SIM_MIXER_MODE_SUM:
        case SIM_MIXER_MODE_MULTIPLY:
        case SIM_MIXER_MODE_CROSSFADE:
            break;
        default:
            config->warp_mode = SIM_MIXER_MODE_SUM;
            break;
    }
    if (!isfinite(config->tolerance) || config->tolerance <= 0.0) {
        config->tolerance = STIM_DIGAMMA_SQUARE_EPS;
    }
    if (!config->use_wavevector && (fabs(config->kx) > STIM_DIGAMMA_SQUARE_EPS ||
                                    fabs(config->ky) > STIM_DIGAMMA_SQUARE_EPS)) {
        config->use_wavevector = true;
    }

    if (!config->use_warp) {
        config->warp_field_index = SIZE_MAX;
    }

    if (config->backend < SIM_DIGAMMA_BACKEND_12_TAIL ||
        config->backend > SIM_DIGAMMA_BACKEND_MORTICI) {
        config->backend = SIM_DIGAMMA_BACKEND_12_TAIL;
    }

    if (config->shape < SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT ||
        config->shape > SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH) {
        config->shape = SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT;
    }
}

static SimClockMode
digamma_square_resolve_clock_mode(const SimContext*                     context,
                                  const char*                           op_name,
                                  const SimStimulusDigammaSquareConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_DIGAMMA_SQUARE_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_digamma_square");
    }
    return resolved;
}

static SimResult digamma_square_build_info(const SimContext*                     context,
                                           const SimStimulusDigammaSquareConfig* config,
                                           SimOperatorInfo*                      out_info) {
    if (context == NULL || config == NULL || out_info == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field((SimContext*) context, config->field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* warp_field = NULL;
    if (config->use_warp) {
        warp_field = sim_context_field((SimContext*) context, config->warp_field_index);
        if (warp_field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    bool field_complex       = sim_field_is_complex(field);
    bool warp_complex        = (warp_field != NULL) && sim_field_is_complex(warp_field);
    bool needs_complex_input = field_complex || warp_complex;

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
    info.abstract_id       = "stimulus_digamma_square";
    sim_operator_info_set_schema_identity(&info, "stimulus_digamma_square");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex_input ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex_input;
    info.representation.requires_complex_representation = needs_complex_input;
    info.representation.preserves_real_subspace         = info.preserves_real;

    *out_info = info;
    return SIM_RESULT_OK;
}

static void digamma_square_refresh_symbolic(SimDigammaSquareState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimStimulusDigammaSquareConfig* cfg = &state->config;

    char        backend_buf[48];
    char        warp_buf[96];
    const char* backend = NULL;
    const char* warp    = "";
    switch (cfg->backend) {
        case SIM_DIGAMMA_BACKEND_ADAPTIVE:
            (void) snprintf(backend_buf, sizeof(backend_buf), "adaptive tol=%.2g", cfg->tolerance);
            backend = backend_buf;
            break;
        case SIM_DIGAMMA_BACKEND_7_TAIL:
            backend = "tail7";
            break;
        case SIM_DIGAMMA_BACKEND_5_TAIL:
            backend = "tail5";
            break;
        case SIM_DIGAMMA_BACKEND_MORTICI:
            backend = "mortici";
            break;
        case SIM_DIGAMMA_BACKEND_12_TAIL:
        default:
            backend = "tail12";
            break;
    }

    if (cfg->use_warp) {
        const char* warp_mode = mixer_mode_name(cfg->warp_mode);
        if (warp_mode == NULL) {
            warp_mode = "sum";
        }
        (void) snprintf(warp_buf,
                        sizeof(warp_buf),
                        " warp %s mix=%.3g bias=%.3g",
                        warp_mode,
                        cfg->warp_mix,
                        cfg->warp_bias);
        warp = warp_buf;
    }

    switch (cfg->shape) {
        case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "default A=%.3g k=%.3g a=%.3g ω=%.3g %s%s",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->a,
                            cfg->omega,
                            backend,
                            warp);
            break;
        case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "triangle A=%.3g k=%.3g a=%.3g ω=%.3g %s%s",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->a,
                            cfg->omega,
                            backend,
                            warp);
            break;
        case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "sawtooth A=%.3g k=%.3g a=%.3g ω=%.3g %s%s",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->a,
                            cfg->omega,
                            backend,
                            warp);
            break;
        default:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "default A=%.3g k=%.3g a=%.3g ω=%.3g %s%s",
                            cfg->amplitude,
                            cfg->wavenumber,
                            cfg->a,
                            cfg->omega,
                            backend,
                            warp);
            break;
    }
#else
    (void) state;
#endif
}

static const char* digamma_square_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimDigammaSquareState* state = (const SimDigammaSquareState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

typedef struct StimulusDigammaIRParams {
    bool needs_dt;
    bool needs_step_index;
    bool needs_time;
} StimulusDigammaIRParams;

static SimIRNodeId
digamma_ir_binary(SimIRBuilder* builder, SimIRNodeType type, SimIRNodeId lhs, SimIRNodeId rhs) {
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_binary(builder, type, lhs, rhs);
}

static SimIRNodeId digamma_ir_call(SimIRBuilder* builder, SimIRCallKind kind, SimIRNodeId operand) {
    if (operand == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }
    return sim_ir_builder_call(builder, kind, operand);
}

static SimIRWarpProfile digamma_ir_profile(SimDigammaBackend backend) {
    switch (backend) {
        case SIM_DIGAMMA_BACKEND_7_TAIL:
            return SIM_IR_WARP_PROFILE_DIGAMMA_7_TAIL;
        case SIM_DIGAMMA_BACKEND_5_TAIL:
            return SIM_IR_WARP_PROFILE_DIGAMMA_5_TAIL;
        case SIM_DIGAMMA_BACKEND_ADAPTIVE:
            return SIM_IR_WARP_PROFILE_DIGAMMA_ADAPTIVE;
        case SIM_DIGAMMA_BACKEND_MORTICI:
            return SIM_IR_WARP_PROFILE_DIGAMMA_MORTICI;
        case SIM_DIGAMMA_BACKEND_12_TAIL:
        default:
            return SIM_IR_WARP_PROFILE_DIGAMMA;
    }
}

static SimIRNodeId digamma_square_build_ir(SimIRBuilder*                         builder,
                                           const SimStimulusDigammaSquareConfig* config,
                                           bool                                  use_warp,
                                           StimulusDigammaIRParams*              params) {
    if (builder == NULL || config == NULL || params == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    params->needs_dt         = false;
    params->needs_step_index = false;
    params->needs_time       = false;

    SimIRNodeId index    = sim_ir_builder_index(builder);
    SimIRNodeId spacing  = sim_ir_builder_constant(builder, config->coord.spacing_x);
    SimIRNodeId origin   = sim_ir_builder_constant(builder, config->coord.origin_x);
    SimIRNodeId x_offset = digamma_ir_binary(builder, SIM_IR_NODE_MUL, spacing, index);
    SimIRNodeId x        = digamma_ir_binary(builder, SIM_IR_NODE_ADD, origin, x_offset);
    if (x == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    bool        needs_dt  = config->scale_by_dt ||
                            (config->fixed_clock && config->nominal_dt <= STIM_DIGAMMA_SQUARE_EPS);
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
        if (config->nominal_dt > STIM_DIGAMMA_SQUARE_EPS) {
            increment = sim_ir_builder_constant(builder, config->nominal_dt);
        } else {
            increment = dt_scaled;
        }
        SimIRNodeId scaled_step =
            digamma_ir_binary(builder, SIM_IR_NODE_MUL, step_index, increment);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = digamma_ir_binary(builder, SIM_IR_NODE_ADD, scaled_step, time_offset);
    } else {
        params->needs_time      = true;
        SimIRNodeId time_node   = sim_ir_builder_param(builder, SIM_IR_PARAM_TIME);
        SimIRNodeId time_offset = sim_ir_builder_constant(builder, config->time_offset);
        t = digamma_ir_binary(builder, SIM_IR_NODE_ADD, time_node, time_offset);
    }

    if (t == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId velocity   = sim_ir_builder_constant(builder, config->coord.velocity_x);
    SimIRNodeId drift      = digamma_ir_binary(builder, SIM_IR_NODE_MUL, velocity, t);
    SimIRNodeId sample_x   = digamma_ir_binary(builder, SIM_IR_NODE_SUB, x, drift);
    SimIRNodeId wavenumber = sim_ir_builder_constant(builder, config->wavenumber);
    SimIRNodeId omega      = sim_ir_builder_constant(builder, config->omega);
    SimIRNodeId phase      = sim_ir_builder_constant(builder, config->phase);
    SimIRNodeId kx         = digamma_ir_binary(builder, SIM_IR_NODE_MUL, wavenumber, sample_x);
    SimIRNodeId omega_t    = digamma_ir_binary(builder, SIM_IR_NODE_MUL, omega, t);
    SimIRNodeId phase_t    = digamma_ir_binary(builder, SIM_IR_NODE_SUB, kx, omega_t);
    SimIRNodeId phase_full = digamma_ir_binary(builder, SIM_IR_NODE_ADD, phase_t, phase);

    SimIRNodeId sin_phase = digamma_ir_call(builder, SIM_IR_CALL_SIN, phase_full);
    SimIRNodeId cos_phase = digamma_ir_call(builder, SIM_IR_CALL_COS, phase_full);
    SimIRNodeId basis     = cos_phase;

    if (basis == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId warp_scale = sim_ir_builder_constant(builder, 1.0);
    if (use_warp) {
        warp_scale = sim_ir_builder_field_ref_typed(builder, 1U, sim_ir_type_scalar());
    }

    SimIRNodeId warp_bias   = sim_ir_builder_constant(builder, config->warp_bias);
    SimIRNodeId warp        = digamma_ir_binary(builder, SIM_IR_NODE_ADD, warp_scale, warp_bias);
    SimIRNodeId warp_mix    = sim_ir_builder_constant(builder, config->warp_mix);
    SimIRNodeId scaled_warp = digamma_ir_binary(builder, SIM_IR_NODE_MUL, warp_mix, warp);

    double crossfade = config->warp_mix;
    if (crossfade < 0.0) {
        crossfade = 0.0;
    } else if (crossfade > 1.0) {
        crossfade = 1.0;
    }

    SimIRNodeId harmonics    = sim_ir_builder_constant(builder, config->harmonics);
    SimIRNodeId harmonic_mix = SIM_IR_INVALID_NODE;

    switch (config->warp_mode) {
        case SIM_MIXER_MODE_MULTIPLY:
            harmonic_mix = digamma_ir_binary(builder, SIM_IR_NODE_MUL, harmonics, scaled_warp);
            break;
        case SIM_MIXER_MODE_CROSSFADE: {
            SimIRNodeId crossfade_node = sim_ir_builder_constant(builder, crossfade);
            SimIRNodeId one_minus      = sim_ir_builder_constant(builder, 1.0 - crossfade);
            SimIRNodeId term_a = digamma_ir_binary(builder, SIM_IR_NODE_MUL, one_minus, harmonics);
            SimIRNodeId term_b = digamma_ir_binary(builder, SIM_IR_NODE_MUL, crossfade_node, warp);
            harmonic_mix       = digamma_ir_binary(builder, SIM_IR_NODE_ADD, term_a, term_b);
            break;
        }
        case SIM_MIXER_MODE_SUM:
        default:
            harmonic_mix = digamma_ir_binary(builder, SIM_IR_NODE_ADD, harmonics, scaled_warp);
            break;
    }

    if (harmonic_mix == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId u     = digamma_ir_binary(builder, SIM_IR_NODE_MUL, harmonic_mix, basis);
    SimIRNodeId inner = digamma_ir_binary(
        builder, SIM_IR_NODE_MUL, u, sim_ir_builder_constant(builder, 2.0 * M_PI));
    SimIRNodeId sample =
        digamma_ir_binary(builder, SIM_IR_NODE_ADD, u, sim_ir_builder_constant(builder, 0.5));

    SimIRWarpSpec warp_spec = { 0 };
    warp_spec.operand       = sample;
    warp_spec.bias          = 0.0;
    warp_spec.delta         = 0.5 - config->a;
    warp_spec.lambda        = -1.0;
    warp_spec.tolerance =
        (config->backend == SIM_DIGAMMA_BACKEND_ADAPTIVE) ? config->tolerance : 0.0;
    warp_spec.profile         = digamma_ir_profile(config->backend);
    warp_spec.warp_class      = SIM_WARP_LEVEL_NONE;
    warp_spec.guard.mode      = (int) SIM_CONTINUITY_NONE;
    warp_spec.guard.clamp_min = 0.0;
    warp_spec.guard.clamp_max = 0.0;
    warp_spec.guard.tolerance = 0.0;
    warp_spec.result_type     = sim_ir_type_scalar();

    SimIRNodeId psi_diff     = sim_ir_builder_warp_spec(builder, &warp_spec);
    SimIRNodeId dpsi_over_pi = digamma_ir_binary(
        builder, SIM_IR_NODE_MUL, psi_diff, sim_ir_builder_constant(builder, M_INV_PI));

    SimIRNodeId cos_inner  = digamma_ir_call(builder, SIM_IR_CALL_COS, inner);
    SimIRNodeId cos_shift  = sim_ir_builder_constant(builder, cos(2.0 * M_PI * config->a));
    SimIRNodeId mod_factor = digamma_ir_binary(builder, SIM_IR_NODE_SUB, cos_inner, cos_shift);
    SimIRNodeId modulation = digamma_ir_binary(builder, SIM_IR_NODE_MUL, mod_factor, dpsi_over_pi);
    SimIRNodeId inside     = digamma_ir_binary(
        builder, SIM_IR_NODE_ADD, sim_ir_builder_constant(builder, 1.0), modulation);
    SimIRNodeId amplitude = sim_ir_builder_constant(builder, config->amplitude);
    SimIRNodeId base      = digamma_ir_binary(builder, SIM_IR_NODE_MUL, amplitude, inside);
    if (base == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId value = SIM_IR_INVALID_NODE;
    switch (config->shape) {
        case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE: {
            SimIRNodeId base_sq = digamma_ir_binary(builder, SIM_IR_NODE_MUL, base, base);
            value               = digamma_ir_binary(builder, SIM_IR_NODE_MUL, base_sq, sin_phase);
            break;
        }
        case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
            value = digamma_ir_binary(builder, SIM_IR_NODE_MUL, base, sin_phase);
            break;
        case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
        default: {
            double delta = config->wavenumber * config->coord.spacing_x;
            double scale = (fabs(delta) < STIM_DIGAMMA_SQUARE_EPS) ? 1.0 : 0.5;
            value        = digamma_ir_binary(
                builder, SIM_IR_NODE_MUL, base, sim_ir_builder_constant(builder, scale));
            break;
        }
    }

    if (value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId scale = SIM_IR_INVALID_NODE;
    if (config->scale_by_dt) {
        scale = dt_scaled;
    } else {
        scale = sim_ir_builder_constant(builder, 1.0);
    }
    return digamma_ir_binary(builder, SIM_IR_NODE_MUL, value, scale);
}

static void digamma_square_destroy(void* state_ptr) {
    SimDigammaSquareState* state = (SimDigammaSquareState*) state_ptr;
    if (!state) {
        return;
    }
    free(state->buffer);
    free(state->buffer_complex);
#if defined(SIM_HAVE_VDSP)
    free(state->vdsp_block);
    state->vdsp_block = NULL;
#endif
    free(state);
}

static SimResult digamma_square_ensure_buffer(SimDigammaSquareState* state, size_t count) {
    double* resized;

    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (state->buffer_capacity >= count) {
        return SIM_RESULT_OK;
    }

    resized = (double*) realloc(state->buffer, count * sizeof(double));
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->buffer          = resized;
    state->buffer_capacity = count;
    return SIM_RESULT_OK;
}

static SimResult digamma_square_ensure_buffer_complex(SimDigammaSquareState* state, size_t count) {
    SimComplexDouble* resized;

    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (state->buffer_complex_capacity >= count) {
        return SIM_RESULT_OK;
    }

    resized = (SimComplexDouble*) realloc(state->buffer_complex, count * sizeof(SimComplexDouble));
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->buffer_complex          = resized;
    state->buffer_complex_capacity = count;
    return SIM_RESULT_OK;
}

#if defined(SIM_HAVE_VDSP)
static bool digamma_square_vdsp_ensure_buffers(SimDigammaSquareState* state, size_t width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_block != NULL && state->vdsp_capacity >= width) {
        return true;
    }
    if (width > SIZE_MAX / (3U * sizeof(double))) {
        return false;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 3U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = width;
    state->vdsp_phase    = block;
    state->vdsp_sin      = block + width;
    state->vdsp_cos      = block + width * 2U;
    return true;
}
#endif

static double digamma_square_drive_time(SimDigammaSquareState* state,
                                        double                 base_time,
                                        double                 dt,
                                        size_t                 step_index) {
    double current_time = base_time + state->config.time_offset;

    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            state->clock_initialized = false;
            return current_time;
        case SIM_CLOCK_FROM_STEP_PURE:
            if (state->config.nominal_dt > STIM_DIGAMMA_SQUARE_EPS) {
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
        (state->config.nominal_dt > STIM_DIGAMMA_SQUARE_EPS) ? state->config.nominal_dt : dt;

    if (!state->clock_initialized || step_index <= state->last_step_index) {
        state->locked_time       = current_time;
        state->clock_initialized = true;
    }

    double drive_time = state->locked_time;
    state->locked_time += increment;
    return drive_time;
}

typedef double (*StimulusDigammaSquareRealTransform)(double base,
                                                     double sin_phase,
                                                     bool   phase_constant);
typedef double complex (*StimulusDigammaSquareComplexTransform)(double complex base,
                                                                double complex sin_phase,
                                                                bool           phase_constant);

static inline double digamma_square_mixed_harmonics(const SimStimulusDigammaSquareConfig* cfg,
                                                    double warp_scale) {
    double warp        = (warp_scale + cfg->warp_bias);
    double scaled_warp = cfg->warp_mix * warp;
    double crossfade   = cfg->warp_mix;

    if (crossfade < 0.0) {
        crossfade = 0.0;
    } else if (crossfade > 1.0) {
        crossfade = 1.0;
    }

    switch (cfg->warp_mode) {
        case SIM_MIXER_MODE_MULTIPLY:
            return cfg->harmonics * scaled_warp;
        case SIM_MIXER_MODE_CROSSFADE:
            return (1.0 - crossfade) * cfg->harmonics + crossfade * warp;
        case SIM_MIXER_MODE_SUM:
        default:
            return cfg->harmonics + scaled_warp;
    }
}

static inline double digamma_square_shape_basis_real(SimDigammaSquareWaveformShape shape,
                                                     double                        phase,
                                                     double                        cos_phase) {
    (void) shape;
    (void) phase;
    return cos_phase;
}

static inline double digamma_square_base_real(const SimStimulusDigammaSquareConfig* cfg,
                                              double                                inner) {
    if (fabs(cfg->a - SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT) < STIM_DIGAMMA_SQUARE_A_EPS) {
        return sim_digamma_square_base_real(cfg->amplitude, inner, cfg->backend, cfg->tolerance);
    }

    return sim_digamma_square_base_deformed_real(
        cfg->amplitude, cfg->a, inner, cfg->backend, cfg->tolerance);
}

static inline double complex digamma_square_base_complex(const SimStimulusDigammaSquareConfig* cfg,
                                                         double inner) {
    SimComplexDouble result = { 0.0, 0.0 };

    if (fabs(cfg->a - SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT) < STIM_DIGAMMA_SQUARE_A_EPS) {
        result = sim_digamma_square_base_complex(
            cfg->amplitude, (SimComplexDouble){ inner, 0.0 }, cfg->backend, cfg->tolerance);
    } else {
        result = sim_digamma_square_base_deformed_complex(
            cfg->amplitude, cfg->a, (SimComplexDouble){ inner, 0.0 }, cfg->backend, cfg->tolerance);
    }

    return simcomplex_to_c99(result);
}

static inline double transform_default_real(double base, double sin_phase, bool phase_constant) {
    (void) sin_phase;
    return phase_constant ? base : 0.5 * base;
}

static inline double transform_triangle_real(double base, double sin_phase, bool phase_constant) {
    (void) phase_constant;
    return base * base * sin_phase;
}

static inline double transform_sawtooth_real(double base, double sin_phase, bool phase_constant) {
    (void) phase_constant;
    return base * sin_phase;
}

static inline double complex transform_default_complex(double complex base,
                                                       double complex sin_phase,
                                                       bool           phase_constant) {
    (void) sin_phase;
    (void) phase_constant;
    return base;
}

static inline double complex transform_triangle_complex(double complex base,
                                                        double complex sin_phase,
                                                        bool           phase_constant) {
    (void) phase_constant;
    return base * base * sin_phase;
}

static inline double complex transform_sawtooth_complex(double complex base,
                                                        double complex sin_phase,
                                                        bool           phase_constant) {
    (void) phase_constant;
    return base * sin_phase;
}

static void fill_real(double* out, size_t count, double value) {
    for (size_t i = 0U; i < count; ++i) {
        out[i] = value;
    }
}

static void fill_complex(SimComplexDouble* out, size_t count, double complex value) {
    SimComplexDouble packed = simcomplex_from_c99(value);
    for (size_t i = 0U; i < count; ++i) {
        out[i] = packed;
    }
}

#if defined(SIM_HAVE_VDSP)
static bool digamma_square_linear_map(const SimStimulusDigammaSquareConfig* cfg,
                                      double*                               out_u_x,
                                      double*                               out_u_y) {
    if (cfg == NULL || out_u_x == NULL || out_u_y == NULL) {
        return false;
    }

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                *out_u_x = 0.0;
                *out_u_y = 1.0;
            } else {
                *out_u_x = 1.0;
                *out_u_y = 0.0;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE:
            *out_u_x = cos(cfg->coord.angle);
            *out_u_y = sin(cfg->coord.angle);
            return true;
        default:
            break;
    }

    return false;
}

static bool digamma_square_prepare_phase(SimDigammaSquareState* state,
                                         size_t                 width,
                                         double                 phase_start,
                                         double                 phase_step,
                                         bool                   need_sin) {
    if (state == NULL || width == 0U || !isfinite(phase_start) || !isfinite(phase_step)) {
        return false;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;

    vDSP_vrampD(&phase_start, &phase_step, state->vdsp_phase, 1, len);
    if (need_sin) {
        vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_phase, &vforce_len);
    } else {
        vvcos(state->vdsp_cos, state->vdsp_phase, &vforce_len);
    }
    return true;
}

static bool digamma_square_try_vdsp_rows(SimDigammaSquareState*  state,
                                         const SimField*         field,
                                         const double*           warp_real,
                                         const SimComplexDouble* warp_complex,
                                         bool                    output_complex,
                                         double*                 out_real,
                                         SimComplexDouble*       out_complex,
                                         size_t                  count,
                                         double                  t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }
    if (sim_field_rank(field) == 0U || sim_field_rank(field) > 2U) {
        return false;
    }
    if (sim_field_rank(field) == 1U && !state->config.use_wavevector &&
        state->config.coord.mode == SIM_STIMULUS_COORD_AXIS &&
        state->config.coord.axis == SIM_STIMULUS_AXIS_X) {
        return false;
    }

    const SimStimulusDigammaSquareConfig* cfg    = &state->config;
    size_t                                width  = sim_field_width(field);
    size_t                                height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_DIGAMMA_SQUARE_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!digamma_square_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    bool   use_wavevector = cfg->use_wavevector;
    bool   separable      = (!use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double u_x            = 0.0;
    double u_y            = 0.0;
    if (!use_wavevector && !separable && !digamma_square_linear_map(cfg, &u_x, &u_y)) {
        return false;
    }

    double x0         = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0         = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx         = cfg->coord.spacing_x;
    double dy         = cfg->coord.spacing_y;
    double phase_base = cfg->phase - cfg->omega * t;
    bool   need_sin   = (cfg->shape != SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT);
    bool   phase_constant;

    if (use_wavevector) {
        phase_constant =
            (fabs(cfg->kx) < STIM_DIGAMMA_SQUARE_EPS && fabs(cfg->ky) < STIM_DIGAMMA_SQUARE_EPS);
    } else {
        double delta   = cfg->wavenumber * cfg->coord.spacing_x;
        phase_constant = fabs(delta) < STIM_DIGAMMA_SQUARE_EPS;
    }

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(phase_base)) {
        return false;
    }

    double phase_x_start = 0.0;
    double phase_x_step  = 0.0;
    if (separable) {
        phase_x_start = cfg->wavenumber * x0 + phase_base;
        phase_x_step  = cfg->wavenumber * dx;
        if (!digamma_square_prepare_phase(state, width, phase_x_start, phase_x_step, need_sin)) {
            return false;
        }
    }

    for (size_t row = 0U; row < height; ++row) {
        size_t offset = row * width;

        if (!separable) {
            double sample_y = y0 + (double) row * dy;
            double phase_start;
            double phase_step;

            if (use_wavevector) {
                phase_start = cfg->kx * x0 + cfg->ky * sample_y + phase_base;
                phase_step  = cfg->kx * dx;
            } else {
                phase_start = cfg->wavenumber * (u_x * x0 + u_y * sample_y) + phase_base;
                phase_step  = cfg->wavenumber * u_x * dx;
            }

            if (!digamma_square_prepare_phase(state, width, phase_start, phase_step, need_sin)) {
                return false;
            }
        }

        double       y_sin    = 0.0;
        double       y_cos    = 1.0;
        double       y_basis  = 0.0;
        const double sample_y = y0 + (double) row * dy;
        if (separable) {
            double phase_y = cfg->wavenumber * sample_y + phase_base;
            sincos_pair(phase_y, &y_sin, &y_cos);
            y_basis = digamma_square_shape_basis_real(cfg->shape, phase_y, y_cos);
        }

        const double*           warp_real_row = (warp_real != NULL) ? (warp_real + offset) : NULL;
        const SimComplexDouble* warp_complex_row =
            (warp_complex != NULL) ? (warp_complex + offset) : NULL;

        if (!output_complex) {
            for (size_t col = 0U; col < width; ++col) {
                double warp_scale = 1.0;
                if (warp_real_row != NULL) {
                    warp_scale = warp_real_row[col];
                } else if (warp_complex_row != NULL) {
                    warp_scale = warp_complex_row[col].re;
                }

                double harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
                double sin_value    = need_sin ? state->vdsp_sin[col] : 0.0;
                double base         = 0.0;
                double value        = 0.0;

                if (separable) {
                    double basis_x = digamma_square_shape_basis_real(
                        cfg->shape, state->vdsp_phase[col], state->vdsp_cos[col]);
                    double base_x = digamma_square_base_real(cfg, harmonic_mix * basis_x);
                    double base_y = digamma_square_base_real(cfg, harmonic_mix * y_basis);
                    double value_x;
                    double value_y;

                    switch (cfg->shape) {
                        case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
                            value_x = transform_triangle_real(base_x, sin_value, phase_constant);
                            value_y = transform_triangle_real(base_y, y_sin, phase_constant);
                            break;
                        case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
                            value_x = transform_sawtooth_real(base_x, sin_value, phase_constant);
                            value_y = transform_sawtooth_real(base_y, y_sin, phase_constant);
                            break;
                        case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
                        default:
                            value_x = transform_default_real(base_x, 0.0, phase_constant);
                            value_y = transform_default_real(base_y, 0.0, phase_constant);
                            break;
                    }

                    value = (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD)
                                ? (value_x + value_y)
                                : (value_x * value_y);
                } else {
                    double basis = digamma_square_shape_basis_real(
                        cfg->shape, state->vdsp_phase[col], state->vdsp_cos[col]);
                    base = digamma_square_base_real(cfg, harmonic_mix * basis);

                    switch (cfg->shape) {
                        case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
                            value = transform_triangle_real(base, sin_value, phase_constant);
                            break;
                        case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
                            value = transform_sawtooth_real(base, sin_value, phase_constant);
                            break;
                        case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
                        default:
                            value = transform_default_real(base, 0.0, phase_constant);
                            break;
                    }
                }

                out_real[offset + col] = value;
            }
            continue;
        }

        for (size_t col = 0U; col < width; ++col) {
            double warp_scale = 1.0;
            if (warp_real_row != NULL) {
                warp_scale = warp_real_row[col];
            } else if (warp_complex_row != NULL) {
                warp_scale = warp_complex_row[col].re;
            }

            double         harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double         sin_value    = need_sin ? state->vdsp_sin[col] : 0.0;
            double complex value;

            if (separable) {
                double basis_x = digamma_square_shape_basis_real(
                    cfg->shape, state->vdsp_phase[col], state->vdsp_cos[col]);
                double complex base_x = digamma_square_base_complex(cfg, harmonic_mix * basis_x);
                double complex base_y = digamma_square_base_complex(cfg, harmonic_mix * y_basis);
                double complex value_x;
                double complex value_y;

                switch (cfg->shape) {
                    case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
                        value_x = transform_triangle_complex(base_x, sin_value, phase_constant);
                        value_y = transform_triangle_complex(base_y, y_sin, phase_constant);
                        break;
                    case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
                        value_x = transform_sawtooth_complex(base_x, sin_value, phase_constant);
                        value_y = transform_sawtooth_complex(base_y, y_sin, phase_constant);
                        break;
                    case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
                    default:
                        value_x = transform_default_complex(base_x, 0.0, phase_constant);
                        value_y = transform_default_complex(base_y, 0.0, phase_constant);
                        break;
                }

                value = (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) ? (value_x + value_y)
                                                                           : (value_x * value_y);
            } else {
                double basis = digamma_square_shape_basis_real(
                    cfg->shape, state->vdsp_phase[col], state->vdsp_cos[col]);
                double complex base = digamma_square_base_complex(cfg, harmonic_mix * basis);

                switch (cfg->shape) {
                    case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
                        value = transform_triangle_complex(base, sin_value, phase_constant);
                        break;
                    case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
                        value = transform_sawtooth_complex(base, sin_value, phase_constant);
                        break;
                    case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
                    default:
                        value = transform_default_complex(base, 0.0, phase_constant);
                        break;
                }
            }

            out_complex[offset + col] = simcomplex_from_c99(value);
        }
    }

    return true;
}
#endif

static void eval_digamma_square_real(const SimStimulusDigammaSquareConfig* cfg,
                                     const double*                         warp_real,
                                     const SimComplexDouble*               warp_complex,
                                     const SimField*                       field,
                                     double*                               out,
                                     size_t                                count,
                                     double                                t,
                                     StimulusDigammaSquareRealTransform    transform) {
    if (out == NULL || count == 0U) {
        return;
    }

    if (cfg->amplitude == 0.0) {
        fill_real(out, count, 0.0);
        return;
    }

    bool use_linear = true;
    if (field != NULL) {
        use_linear = (sim_field_rank(field) == 1U && cfg->coord.mode == SIM_STIMULUS_COORD_AXIS &&
                      cfg->coord.axis == SIM_STIMULUS_AXIS_X && !cfg->use_wavevector);
    }

    if (use_linear) {
        double delta       = cfg->wavenumber * cfg->coord.spacing_x;
        double phase0      = cfg->wavenumber * (cfg->coord.origin_x - cfg->coord.velocity_x * t) +
                             cfg->phase - cfg->omega * t;
        double phase_value = phase0;
        double sin_phase, cos_phase;
        sincos_pair(phase0, &sin_phase, &cos_phase);

        bool phase_constant = fabs(delta) < STIM_DIGAMMA_SQUARE_EPS;

        if (phase_constant && warp_real == NULL && warp_complex == NULL) {
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, 1.0);
            double basis        = digamma_square_shape_basis_real(cfg->shape, phase0, cos_phase);
            double base         = digamma_square_base_real(cfg, harmonic_mix * basis);
            double value        = transform(base, sin_phase, true);

            fill_real(out, count, value);
            return;
        }

        double sin_delta, cos_delta;
        sincos_pair(delta, &sin_delta, &cos_delta);

        unsigned int mask = STIM_DIGAMMA_SQUARE_RENORM_INTERVAL - 1U;

        for (size_t i = 0U; i < count; ++i) {
            double warp_scale = 1.0;
            if (warp_real != NULL) {
                warp_scale = warp_real[i];
            } else if (warp_complex != NULL) {
                warp_scale = warp_complex[i].re;
            }
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double basis = digamma_square_shape_basis_real(cfg->shape, phase_value, cos_phase);
            double base  = digamma_square_base_real(cfg, harmonic_mix * basis);
            out[i]       = transform(base, sin_phase, phase_constant);

            double next_cos = cos_phase * cos_delta - sin_phase * sin_delta;
            double next_sin = sin_phase * cos_delta + cos_phase * sin_delta;

            cos_phase = next_cos;
            sin_phase = next_sin;
            phase_value += delta;

            if (((unsigned int) (i + 1U) & mask) == 0U) {
                double phase_i = phase0 + delta * (double) (i + 1U);
                sincos_pair(phase_i, &sin_phase, &cos_phase);
                phase_value = phase_i;
            }
        }
        return;
    }

    bool separable      = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool use_wavevector = cfg->use_wavevector;
    bool phase_constant = false;
    if (use_wavevector) {
        phase_constant =
            (fabs(cfg->kx) < STIM_DIGAMMA_SQUARE_EPS && fabs(cfg->ky) < STIM_DIGAMMA_SQUARE_EPS);
    } else {
        double delta   = cfg->wavenumber * cfg->coord.spacing_x;
        phase_constant = fabs(delta) < STIM_DIGAMMA_SQUARE_EPS;
    }
    double phase_base = cfg->phase - cfg->omega * t;

    for (size_t i = 0U; i < count; ++i) {
        double x        = 0.0;
        double y        = 0.0;
        double sample_x = 0.0;
        double sample_y = 0.0;
        if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            out[i] = 0.0;
            continue;
        }
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

        double warp_scale = 1.0;
        if (warp_real != NULL) {
            warp_scale = warp_real[i];
        } else if (warp_complex != NULL) {
            warp_scale = warp_complex[i].re;
        }

        if (use_wavevector) {
            double spatial     = cfg->kx * sample_x + cfg->ky * sample_y;
            double phase_value = spatial + phase_base;
            double sin_phase   = 0.0;
            double cos_phase   = 1.0;
            sincos_pair(phase_value, &sin_phase, &cos_phase);
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double basis = digamma_square_shape_basis_real(cfg->shape, phase_value, cos_phase);
            double base  = digamma_square_base_real(cfg, harmonic_mix * basis);
            out[i]       = transform(base, sin_phase, phase_constant);
        } else if (separable) {
            double sin_x, cos_x;
            double sin_y, cos_y;
            double phase_x      = cfg->wavenumber * sample_x + phase_base;
            double phase_y      = cfg->wavenumber * sample_y + phase_base;
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            sincos_pair(phase_x, &sin_x, &cos_x);
            sincos_pair(phase_y, &sin_y, &cos_y);

            double basis_x = digamma_square_shape_basis_real(cfg->shape, phase_x, cos_x);
            double basis_y = digamma_square_shape_basis_real(cfg->shape, phase_y, cos_y);
            double base_x  = digamma_square_base_real(cfg, harmonic_mix * basis_x);
            double base_y  = digamma_square_base_real(cfg, harmonic_mix * basis_y);
            double value_x = transform(base_x, sin_x, phase_constant);
            double value_y = transform(base_y, sin_y, phase_constant);

            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                out[i] = value_x + value_y;
            } else {
                out[i] = value_x * value_y;
            }
        } else {
            double u           = sim_stimulus_coord_u(&cfg->coord, x, y, t);
            double phase_value = cfg->wavenumber * u + phase_base;
            double sin_phase   = 0.0;
            double cos_phase   = 1.0;
            sincos_pair(phase_value, &sin_phase, &cos_phase);
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double basis = digamma_square_shape_basis_real(cfg->shape, phase_value, cos_phase);
            double base  = digamma_square_base_real(cfg, harmonic_mix * basis);
            out[i]       = transform(base, sin_phase, phase_constant);
        }
    }
}

static void eval_digamma_square_complex(const SimStimulusDigammaSquareConfig* cfg,
                                        const double*                         warp_real,
                                        const SimComplexDouble*               warp_complex,
                                        const SimField*                       field,
                                        SimComplexDouble*                     out,
                                        size_t                                count,
                                        double                                t,
                                        StimulusDigammaSquareComplexTransform transform) {
    if (out == NULL || count == 0U) {
        return;
    }

    if (cfg->amplitude == 0.0) {
        fill_complex(out, count, 0.0);
        return;
    }

    bool use_linear = true;
    if (field != NULL) {
        use_linear = (sim_field_rank(field) == 1U && cfg->coord.mode == SIM_STIMULUS_COORD_AXIS &&
                      cfg->coord.axis == SIM_STIMULUS_AXIS_X && !cfg->use_wavevector);
    }

    if (use_linear) {
        double complex delta = cfg->wavenumber * cfg->coord.spacing_x;
        double complex phase0 =
            cfg->wavenumber * (cfg->coord.origin_x - cfg->coord.velocity_x * t) + cfg->phase -
            cfg->omega * t;
        double phase_value = creal(phase0);

        double complex sin_phase = csin(phase0);
        double complex cos_phase = ccos(phase0);

        bool phase_constant = cabs(delta) < STIM_DIGAMMA_SQUARE_EPS;

        if (phase_constant && warp_real == NULL && warp_complex == NULL) {
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, 1.0);
            double basis =
                digamma_square_shape_basis_real(cfg->shape, phase_value, creal(cos_phase));
            double complex base  = digamma_square_base_complex(cfg, harmonic_mix * basis);
            double complex value = transform(base, sin_phase, true);

            fill_complex(out, count, value);
            return;
        }

        double complex sin_delta = csin(delta);
        double complex cos_delta = ccos(delta);
        unsigned int   mask      = STIM_DIGAMMA_SQUARE_RENORM_INTERVAL - 1U;

        for (size_t i = 0U; i < count; ++i) {
            double warp_scale = 1.0;
            if (warp_real != NULL) {
                warp_scale = warp_real[i];
            } else if (warp_complex != NULL) {
                warp_scale = warp_complex[i].re;
            }
            double harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double basis =
                digamma_square_shape_basis_real(cfg->shape, phase_value, creal(cos_phase));
            double complex base = digamma_square_base_complex(cfg, harmonic_mix * basis);
            out[i]              = simcomplex_from_c99(transform(base, sin_phase, phase_constant));

            double complex next_cos = cos_phase * cos_delta - sin_phase * sin_delta;
            double complex next_sin = sin_phase * cos_delta + cos_phase * sin_delta;

            cos_phase = next_cos;
            sin_phase = next_sin;
            phase_value += creal(delta);

            if (((unsigned int) (i + 1U) & mask) == 0U) {
                double complex phase_i = phase0 + delta * (double) (i + 1U);
                cos_phase              = ccos(phase_i);
                sin_phase              = csin(phase_i);
                phase_value            = creal(phase_i);
            }
        }
        return;
    }

    bool separable      = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool use_wavevector = cfg->use_wavevector;
    bool phase_constant = false;
    if (use_wavevector) {
        phase_constant =
            (fabs(cfg->kx) < STIM_DIGAMMA_SQUARE_EPS && fabs(cfg->ky) < STIM_DIGAMMA_SQUARE_EPS);
    } else {
        double delta   = cfg->wavenumber * cfg->coord.spacing_x;
        phase_constant = fabs(delta) < STIM_DIGAMMA_SQUARE_EPS;
    }
    double phase_base = cfg->phase - cfg->omega * t;

    for (size_t i = 0U; i < count; ++i) {
        double x        = 0.0;
        double y        = 0.0;
        double sample_x = 0.0;
        double sample_y = 0.0;
        if (sim_stimulus_coord_xy(&cfg->coord, field, i, &x, &y) != SIM_RESULT_OK) {
            out[i] = simcomplex_from_c99(0.0);
            continue;
        }
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);

        double warp_scale = 1.0;
        if (warp_real != NULL) {
            warp_scale = warp_real[i];
        } else if (warp_complex != NULL) {
            warp_scale = warp_complex[i].re;
        }

        if (use_wavevector) {
            double         spatial      = cfg->kx * sample_x + cfg->ky * sample_y;
            double         phase_value  = spatial + phase_base;
            double complex sin_phase    = csin(phase_value);
            double complex cos_phase    = ccos(phase_value);
            double         harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double         basis =
                digamma_square_shape_basis_real(cfg->shape, phase_value, creal(cos_phase));
            double complex base = digamma_square_base_complex(cfg, harmonic_mix * basis);
            out[i]              = simcomplex_from_c99(transform(base, sin_phase, phase_constant));
        } else if (separable) {
            double         phase_x      = cfg->wavenumber * sample_x + phase_base;
            double         phase_y      = cfg->wavenumber * sample_y + phase_base;
            double complex sin_x        = csin(phase_x);
            double complex cos_x        = ccos(phase_x);
            double complex sin_y        = csin(phase_y);
            double complex cos_y        = ccos(phase_y);
            double         harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double basis_x = digamma_square_shape_basis_real(cfg->shape, phase_x, creal(cos_x));
            double basis_y = digamma_square_shape_basis_real(cfg->shape, phase_y, creal(cos_y));

            double complex base_x   = digamma_square_base_complex(cfg, harmonic_mix * basis_x);
            double complex base_y   = digamma_square_base_complex(cfg, harmonic_mix * basis_y);
            double complex value_x  = transform(base_x, sin_x, phase_constant);
            double complex value_y  = transform(base_y, sin_y, phase_constant);
            double complex combined = (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD)
                                          ? (value_x + value_y)
                                          : (value_x * value_y);

            out[i] = simcomplex_from_c99(combined);
        } else {
            double         u            = sim_stimulus_coord_u(&cfg->coord, x, y, t);
            double         phase_value  = cfg->wavenumber * u + phase_base;
            double complex sin_phase    = csin(phase_value);
            double complex cos_phase    = ccos(phase_value);
            double         harmonic_mix = digamma_square_mixed_harmonics(cfg, warp_scale);
            double         basis =
                digamma_square_shape_basis_real(cfg->shape, phase_value, creal(cos_phase));
            double complex base = digamma_square_base_complex(cfg, harmonic_mix * basis);
            out[i]              = simcomplex_from_c99(transform(base, sin_phase, phase_constant));
        }
    }
}

static void eval_digamma_square_default(const SimStimulusDigammaSquareConfig* cfg,
                                        const double*                         warp_real,
                                        const SimComplexDouble*               warp_complex,
                                        const SimField*                       field,
                                        double*                               out,
                                        size_t                                count,
                                        double                                t) {
    eval_digamma_square_real(
        cfg, warp_real, warp_complex, field, out, count, t, transform_default_real);
}

static void eval_digamma_square_triangle(const SimStimulusDigammaSquareConfig* cfg,
                                         const double*                         warp_real,
                                         const SimComplexDouble*               warp_complex,
                                         const SimField*                       field,
                                         double*                               out,
                                         size_t                                count,
                                         double                                t) {
    eval_digamma_square_real(
        cfg, warp_real, warp_complex, field, out, count, t, transform_triangle_real);
}

static void eval_digamma_square_sawtooth(const SimStimulusDigammaSquareConfig* cfg,
                                         const double*                         warp_real,
                                         const SimComplexDouble*               warp_complex,
                                         const SimField*                       field,
                                         double*                               out,
                                         size_t                                count,
                                         double                                t) {
    eval_digamma_square_real(
        cfg, warp_real, warp_complex, field, out, count, t, transform_sawtooth_real);
}

static void eval_digamma_square_default_complex(const SimStimulusDigammaSquareConfig* cfg,
                                                const double*                         warp_real,
                                                const SimComplexDouble*               warp_complex,
                                                const SimField*                       field,
                                                SimComplexDouble*                     out,
                                                size_t                                count,
                                                double                                t) {
    eval_digamma_square_complex(
        cfg, warp_real, warp_complex, field, out, count, t, transform_default_complex);
}

static void eval_digamma_square_triangle_complex(const SimStimulusDigammaSquareConfig* cfg,
                                                 const double*                         warp_real,
                                                 const SimComplexDouble*               warp_complex,
                                                 const SimField*                       field,
                                                 SimComplexDouble*                     out,
                                                 size_t                                count,
                                                 double                                t) {
    eval_digamma_square_complex(
        cfg, warp_real, warp_complex, field, out, count, t, transform_triangle_complex);
}

static void eval_digamma_square_sawtooth_complex(const SimStimulusDigammaSquareConfig* cfg,
                                                 const double*                         warp_real,
                                                 const SimComplexDouble*               warp_complex,
                                                 const SimField*                       field,
                                                 SimComplexDouble*                     out,
                                                 size_t                                count,
                                                 double                                t) {
    eval_digamma_square_complex(
        cfg, warp_real, warp_complex, field, out, count, t, transform_sawtooth_complex);
}

static SimResult digamma_square_step(void*               state_ptr,
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

    SimDigammaSquareState* state = (SimDigammaSquareState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    void*  raw_data   = sim_field_data(field);
    size_t count      = 0U;
    bool   is_complex = false;

    if (raw_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    is_complex = sim_field_is_complex(field);
    if (!is_complex) {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    size_t step_index = sim_context_step_index(context);
    double base_time  = sim_context_time(context);
    double drive_time = digamma_square_drive_time(state, base_time, dt_sub, step_index);

    if (count == 0U) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    if (state->config.amplitude == 0.0) {
        state->last_step_index = step_index;
        return SIM_RESULT_OK;
    }

    const double*           warp_real    = NULL;
    const SimComplexDouble* warp_complex = NULL;
    if (state->config.use_warp) {
        SimField* warp_field = sim_context_field(context, state->config.warp_field_index);
        if (warp_field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        bool   warp_is_complex = sim_field_is_complex(warp_field);
        size_t warp_count      = 0U;
        if (warp_is_complex) {
            if (warp_field->element_size != sizeof(SimComplexDouble)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            warp_count = sim_field_bytes(warp_field) / sizeof(SimComplexDouble);
        } else {
            if (warp_field->element_size != sizeof(double)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            warp_count = sim_field_bytes(warp_field) / sizeof(double);
        }
        if (warp_count != count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        const void* warp_raw = sim_field_data(warp_field);
        if (warp_raw == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (warp_is_complex) {
            warp_complex = (const SimComplexDouble*) warp_raw;
        } else {
            warp_real = (const double*) warp_raw;
        }
    }

    double scale    = state->config.scale_by_dt ? dt_sub : 1.0;
    double rotation = state->config.rotation;
    if (!is_complex) {
        SimResult prep = digamma_square_ensure_buffer(state, count);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }

#if defined(SIM_HAVE_VDSP)
        if (!digamma_square_try_vdsp_rows(state,
                                          field,
                                          warp_real,
                                          warp_complex,
                                          false,
                                          state->buffer,
                                          NULL,
                                          count,
                                          drive_time))
#endif
        {

            switch (state->config.shape) {
                case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
                    eval_digamma_square_triangle(&state->config,
                                                 warp_real,
                                                 warp_complex,
                                                 field,
                                                 state->buffer,
                                                 count,
                                                 drive_time);
                    break;
                case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
                    eval_digamma_square_sawtooth(&state->config,
                                                 warp_real,
                                                 warp_complex,
                                                 field,
                                                 state->buffer,
                                                 count,
                                                 drive_time);
                    break;
                case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
                default:
                    eval_digamma_square_default(&state->config,
                                                warp_real,
                                                warp_complex,
                                                field,
                                                state->buffer,
                                                count,
                                                drive_time);
                    break;
            }
        }

        double* dst = (double*) raw_data;
        sim_accel_copy_scale_real(state->buffer, dst, count, scale, true);
    } else {
        SimResult prep = digamma_square_ensure_buffer_complex(state, count);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }

#if defined(SIM_HAVE_VDSP)
        if (!digamma_square_try_vdsp_rows(state,
                                          field,
                                          warp_real,
                                          warp_complex,
                                          true,
                                          NULL,
                                          state->buffer_complex,
                                          count,
                                          drive_time))
#endif
        {
            switch (state->config.shape) {
                case SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE:
                    eval_digamma_square_triangle_complex(&state->config,
                                                         warp_real,
                                                         warp_complex,
                                                         field,
                                                         state->buffer_complex,
                                                         count,
                                                         drive_time);
                    break;
                case SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH:
                    eval_digamma_square_sawtooth_complex(&state->config,
                                                         warp_real,
                                                         warp_complex,
                                                         field,
                                                         state->buffer_complex,
                                                         count,
                                                         drive_time);
                    break;
                case SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT:
                default:
                    eval_digamma_square_default_complex(&state->config,
                                                        warp_real,
                                                        warp_complex,
                                                        field,
                                                        state->buffer_complex,
                                                        count,
                                                        drive_time);
                    break;
            }
        }

        double sin_r = 0.0;
        double cos_r = 1.0;
        if (rotation != 0.0) {
            sincos_pair(rotation, &sin_r, &cos_r);
        }
        double complex rot = cos_r + I * sin_r;

        SimComplexDouble* dst = sim_field_complex_data(field);
        for (size_t i = 0U; i < count; ++i) {
            double complex value = simcomplex_to_c99(state->buffer_complex[i]) * rot;
            value *= scale;
            dst[i].re += creal(value);
            dst[i].im += cimag(value);
        }
    }

    state->last_step_index = step_index;
    return SIM_RESULT_OK;
}

static SimResult digamma_square_register(struct SimContext*                    context,
                                         const SimStimulusDigammaSquareConfig* config,
                                         SimDigammaSquareWaveformShape         shape,
                                         const char*                           prefix,
                                         size_t*                               out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    (void) shape;

    SimStimulusDigammaSquareConfig local = { 0 };
    if (config) {
        local = *config;
    }

    digamma_square_normalize(&local);

    if (local.use_warp && local.warp_field_index == SIZE_MAX) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         prefix ? prefix : "stimulus_digamma_square",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimDigammaSquareState* state =
        (SimDigammaSquareState*) calloc(1U, sizeof(SimDigammaSquareState));

    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    state->clock_mode =
        digamma_square_resolve_clock_mode(context, "stimulus_digamma_square", &state->config);
    state->locked_time             = 0.0;
    state->last_step_index         = 0U;
    state->clock_initialized       = false;
    state->buffer                  = NULL;
    state->buffer_capacity         = 0U;
    state->buffer_complex          = NULL;
    state->buffer_complex_capacity = 0U;
#if defined(SIM_HAVE_VDSP)
    state->vdsp_block    = NULL;
    state->vdsp_phase    = NULL;
    state->vdsp_sin      = NULL;
    state->vdsp_cos      = NULL;
    state->vdsp_capacity = 0U;
#endif
    digamma_square_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), prefix ? prefix : "stimulus_digamma_square");
    const char*     schema_key = "stimulus_digamma_square";
    SimOperatorInfo info       = sim_operator_info_defaults();
    SimResult       info_rc    = digamma_square_build_info(context, &state->config, &info);
    if (info_rc != SIM_RESULT_OK) {
        digamma_square_destroy(state);
        return info_rc;
    }

    SimField* field         = sim_context_field(context, local.field_index);
    bool      field_complex = (field != NULL) && sim_field_is_complex(field);

    SimSplitPort ports[1] = { { .context_field_index = state->config.field_index,
                                .require_complex     = field_complex } };

    SimSplitAccess accesses[1] = { { .port = 0, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = digamma_square_step,
                                .accesses          = accesses,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimOperatorConfig op_config                 = sim_operator_config_defaults();
    SimResult         result                    = SIM_RESULT_OK;
    bool              registered_kernel         = false;
    bool              allow_kernel_registration = false;

    if (allow_kernel_registration &&
        sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_digamma_square")) {
        SimField* warp_field =
            local.use_warp ? sim_context_field(context, local.warp_field_index) : NULL;
        if (field != NULL && !sim_field_is_complex(field) &&
            field->element_size == sizeof(double)) {
            if (sim_field_rank(field) != 1U || local.coord.mode != SIM_STIMULUS_COORD_AXIS ||
                local.coord.axis != SIM_STIMULUS_AXIS_X || local.use_wavevector) {
                field = NULL;
            }
        }
        if (field != NULL) {
            bool warp_ok = true;
            if (local.use_warp) {
                warp_ok = (warp_field != NULL && !sim_field_is_complex(warp_field) &&
                           warp_field->element_size == sizeof(double));
            }

            if (warp_ok) {
                SimIRBuilder* builder = sim_context_ir_builder(context);
                if (builder != NULL) {
                    SimOperatorKernelBindingDescriptor bindings[2];
                    SimOperatorKernelOutputDescriptor  outputs[1];
                    SimOperatorKernelDescriptor        kernel_desc = { 0 };
                    StimulusDigammaIRParams            ir_params   = { 0 };

                    SimIRNodeId field_node =
                        sim_ir_builder_field_ref_typed(builder, 0U, sim_ir_type_scalar());
                    SimIRNodeId delta =
                        digamma_square_build_ir(builder, &local, local.use_warp, &ir_params);
                    SimIRNodeId sum =
                        digamma_ir_binary(builder, SIM_IR_NODE_ADD, field_node, delta);

                    if (field_node != SIM_IR_INVALID_NODE && delta != SIM_IR_INVALID_NODE &&
                        sum != SIM_IR_INVALID_NODE) {
                        int max_param = -1;
                        if (ir_params.needs_dt) {
                            max_param = (int) SIM_IR_PARAM_DT;
                        }
                        if (ir_params.needs_step_index &&
                            (int) SIM_IR_PARAM_STEP_INDEX > max_param) {
                            max_param = (int) SIM_IR_PARAM_STEP_INDEX;
                        }
                        if (ir_params.needs_time && (int) SIM_IR_PARAM_TIME > max_param) {
                            max_param = (int) SIM_IR_PARAM_TIME;
                        }
                        size_t param_count = (max_param >= 0) ? (size_t) max_param + 1U : 0U;

                        bindings[0].ir_field_index      = 0U;
                        bindings[0].context_field_index = local.field_index;
                        if (local.use_warp) {
                            bindings[1].ir_field_index      = 1U;
                            bindings[1].context_field_index = local.warp_field_index;
                        }

                        outputs[0].ir_field_index = 0U;
                        outputs[0].expression     = sum;

                        kernel_desc.builder           = builder;
                        kernel_desc.bindings          = bindings;
                        kernel_desc.binding_count     = local.use_warp ? 2U : 1U;
                        kernel_desc.outputs           = outputs;
                        kernel_desc.output_count      = 1U;
                        kernel_desc.param_count       = param_count;
                        kernel_desc.required_features = 0ULL;

                        SimOperatorDescriptor kdesc = { 0 };
                        kdesc.name                  = name;
                        kdesc.evaluate              = NULL;
                        kdesc.destroy               = digamma_square_destroy;
                        kdesc.userdata              = state;
                        kdesc.kernel                = &kernel_desc;
                        kdesc.info                  = info;
                        kdesc.config                = op_config;
                        if (local.field_index < 64U) {
                            kdesc.read_mask |= (1ULL << local.field_index);
                            kdesc.write_mask |= (1ULL << local.field_index);
                        }
                        if (local.use_warp && local.warp_field_index < 64U) {
                            kdesc.read_mask |= (1ULL << local.warp_field_index);
                        }

                        result = sim_context_register_operator(context, &kdesc, out_index);
                        if (result == SIM_RESULT_OK) {
                            registered_kernel = true;
                        }
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = digamma_square_symbolic,
                                .destroy       = digamma_square_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        digamma_square_destroy(state);
    }

    return result;
}

SimResult sim_add_stimulus_digamma_square_operator(struct SimContext*                    context,
                                                   const SimStimulusDigammaSquareConfig* config,
                                                   size_t* out_index) {
    return digamma_square_register(
        context, config, SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT, "stimulus_digamma_square", out_index);
}

SimResult sim_add_digamma_square_operator(struct SimContext*                    context,
                                          size_t                                field_index,
                                          const SimStimulusDigammaSquareConfig* config,
                                          size_t*                               out_index) {
    SimStimulusDigammaSquareConfig local = { 0 };
    if (config) {
        local = *config;
    }
    local.field_index = field_index;

    return digamma_square_register(
        context, &local, SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT, "digamma_square", out_index);
}

SimResult sim_stimulus_digamma_square_config(struct SimContext*              context,
                                             size_t                          operator_index,
                                             SimStimulusDigammaSquareConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimDigammaSquareState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimDigammaSquareState*) sim_operator_payload(op);
    } else {
        state = (SimDigammaSquareState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_digamma_square_update(struct SimContext*                    context,
                                             size_t                                operator_index,
                                             const SimStimulusDigammaSquareConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimDigammaSquareState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimDigammaSquareState*) sim_operator_payload(op);
    } else {
        state = (SimDigammaSquareState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusDigammaSquareConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }
    local.use_warp = (local.warp_field_index != SIZE_MAX);

    digamma_square_normalize(&local);

    SimOperatorInfo info = sim_operator_info_defaults();
    SimResult       rc   = digamma_square_build_info(context, &local, &info);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }

    state->config     = local;
    state->clock_mode = digamma_square_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_digamma_square"), &state->config);
    op->info = info;

    digamma_square_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);

    return SIM_RESULT_OK;
}
