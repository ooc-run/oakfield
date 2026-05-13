#include "oakfield/operators/nonlinear/complex_math.h"
#include "operators/common/operator_utils.h"

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

#define COMPLEX_MATH_SYMBOLIC_CAPACITY 192
#define COMPLEX_MATH_EPS_MIN 1.0e-12

typedef struct SimComplexMathOperatorState {
    SimComplexMathOperatorConfig config;
    char                         symbolic[COMPLEX_MATH_SYMBOLIC_CAPACITY];
} SimComplexMathOperatorState;

static const char* complex_math_mode_name(SimComplexMathMode mode) {
    switch (mode) {
        case SIM_COMPLEX_MATH_ADD:
            return "add";
        case SIM_COMPLEX_MATH_SUB:
            return "sub";
        case SIM_COMPLEX_MATH_MUL:
            return "mul";
        case SIM_COMPLEX_MATH_DIV:
            return "div";
        case SIM_COMPLEX_MATH_POW:
            return "pow";
        case SIM_COMPLEX_MATH_EXP:
            return "exp";
        case SIM_COMPLEX_MATH_LOG:
            return "log";
        case SIM_COMPLEX_MATH_SIN:
            return "sin";
        case SIM_COMPLEX_MATH_COS:
            return "cos";
        case SIM_COMPLEX_MATH_TAN:
            return "tan";
        case SIM_COMPLEX_MATH_SINH:
            return "sinh";
        case SIM_COMPLEX_MATH_COSH:
            return "cosh";
        case SIM_COMPLEX_MATH_TANH:
            return "tanh";
        case SIM_COMPLEX_MATH_SQRT:
            return "sqrt";
        case SIM_COMPLEX_MATH_CONJ:
            return "conj";
        case SIM_COMPLEX_MATH_ABS:
            return "abs";
        case SIM_COMPLEX_MATH_ARG:
            return "arg";
        case SIM_COMPLEX_MATH_REAL:
            return "real";
        case SIM_COMPLEX_MATH_IMAG:
            return "imag";
        case SIM_COMPLEX_MATH_NEG:
            return "neg";
        default:
            return "add";
    }
}

static const char* complex_math_rhs_name(SimComplexMathRhsSource source) {
    switch (source) {
        case SIM_COMPLEX_MATH_RHS_CONSTANT:
            return "constant";
        case SIM_COMPLEX_MATH_RHS_FIELD:
        default:
            return "field";
    }
}

static const char* complex_math_output_name(SimComplexMathOutputComponent component) {
    switch (component) {
        case SIM_COMPLEX_MATH_OUTPUT_IMAG:
            return "imag";
        case SIM_COMPLEX_MATH_OUTPUT_MAGNITUDE:
            return "magnitude";
        case SIM_COMPLEX_MATH_OUTPUT_PHASE:
            return "phase";
        case SIM_COMPLEX_MATH_OUTPUT_REAL:
        default:
            return "real";
    }
}

static const char* complex_math_phase_wrap_name(SimComplexMathPhaseWrap mode) {
    switch (mode) {
        case SIM_COMPLEX_MATH_PHASE_SIGNED:
            return "signed";
        case SIM_COMPLEX_MATH_PHASE_UNSIGNED:
            return "unsigned";
        case SIM_COMPLEX_MATH_PHASE_UNIT:
            return "unit";
        case SIM_COMPLEX_MATH_PHASE_NONE:
        default:
            return "none";
    }
}

static void complex_math_normalize_config(SimComplexMathOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (config->mode < SIM_COMPLEX_MATH_ADD || config->mode > SIM_COMPLEX_MATH_NEG) {
        config->mode = SIM_COMPLEX_MATH_ADD;
    }

    if (config->rhs_source != SIM_COMPLEX_MATH_RHS_FIELD &&
        config->rhs_source != SIM_COMPLEX_MATH_RHS_CONSTANT) {
        config->rhs_source = SIM_COMPLEX_MATH_RHS_FIELD;
    }

    if (config->output_component < SIM_COMPLEX_MATH_OUTPUT_REAL ||
        config->output_component > SIM_COMPLEX_MATH_OUTPUT_PHASE) {
        config->output_component = SIM_COMPLEX_MATH_OUTPUT_REAL;
    }

    if (config->phase_wrap < SIM_COMPLEX_MATH_PHASE_NONE ||
        config->phase_wrap > SIM_COMPLEX_MATH_PHASE_UNIT) {
        config->phase_wrap = SIM_COMPLEX_MATH_PHASE_NONE;
    }

    if (!isfinite(config->rhs_constant_re)) {
        config->rhs_constant_re = 0.0;
    }
    if (!isfinite(config->rhs_constant_im)) {
        config->rhs_constant_im = 0.0;
    }
    if (!isfinite(config->lhs_scale_re)) {
        config->lhs_scale_re = 1.0;
    }
    if (!isfinite(config->lhs_scale_im)) {
        config->lhs_scale_im = 0.0;
    }
    if (!isfinite(config->rhs_scale_re)) {
        config->rhs_scale_re = 1.0;
    }
    if (!isfinite(config->rhs_scale_im)) {
        config->rhs_scale_im = 0.0;
    }
    if (!isfinite(config->bias_re)) {
        config->bias_re = 0.0;
    }
    if (!isfinite(config->bias_im)) {
        config->bias_im = 0.0;
    }
    if (!isfinite(config->epsilon) || config->epsilon < COMPLEX_MATH_EPS_MIN) {
        config->epsilon = 1.0e-9;
    }

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static void complex_math_refresh_symbolic(SimComplexMathOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimComplexMathOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "complex_math mode=%s rhs=%s out=%s wrap=%s",
                    complex_math_mode_name(cfg->mode),
                    complex_math_rhs_name(cfg->rhs_source),
                    complex_math_output_name(cfg->output_component),
                    complex_math_phase_wrap_name(cfg->phase_wrap));
#else
    (void) state;
#endif
}

static const char* complex_math_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimComplexMathOperatorState* state = (const SimComplexMathOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void complex_math_destroy(void* state_ptr) {
    SimComplexMathOperatorState* state = (SimComplexMathOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state);
}

static double complex complex_from_parts(double re, double im) {
    return re + I * im;
}

static SimComplexDouble complex_to_struct(double complex value) {
    SimComplexDouble out = { creal(value), cimag(value) };
    return out;
}

static double complex complex_scale(double complex value, double scale_re, double scale_im) {
    return value * (scale_re + I * scale_im);
}

static double complex complex_log_safe(double complex value, double epsilon) {
    double radius = cabs(value);
    double phase  = carg(value);
    if (!isfinite(radius) || radius < epsilon) {
        radius = epsilon;
    }
    return log(radius) + I * phase;
}

static double complex_math_wrap_phase(double phase, SimComplexMathPhaseWrap mode) {
    if (!isfinite(phase)) {
        return phase;
    }

    const double two_pi = 2.0 * M_PI;
    switch (mode) {
        case SIM_COMPLEX_MATH_PHASE_SIGNED: {
            double t = fmod(phase + M_PI, two_pi);
            if (t < 0.0) {
                t += two_pi;
            }
            return t - M_PI;
        }
        case SIM_COMPLEX_MATH_PHASE_UNSIGNED: {
            double t = fmod(phase, two_pi);
            if (t < 0.0) {
                t += two_pi;
            }
            return t;
        }
        case SIM_COMPLEX_MATH_PHASE_UNIT: {
            double t = fmod(phase + M_PI, two_pi);
            if (t < 0.0) {
                t += two_pi;
            }
            return t / two_pi;
        }
        case SIM_COMPLEX_MATH_PHASE_NONE:
        default:
            return phase;
    }
}

static double complex_math_output_value(double complex                value,
                                        SimComplexMathOutputComponent component,
                                        SimComplexMathPhaseWrap       phase_wrap) {
    switch (component) {
        case SIM_COMPLEX_MATH_OUTPUT_IMAG:
            return cimag(value);
        case SIM_COMPLEX_MATH_OUTPUT_MAGNITUDE:
            return cabs(value);
        case SIM_COMPLEX_MATH_OUTPUT_PHASE:
            return complex_math_wrap_phase(carg(value), phase_wrap);
        case SIM_COMPLEX_MATH_OUTPUT_REAL:
        default:
            return creal(value);
    }
}

static SimResult complex_math_apply(void*               state_ptr,
                                    struct SimContext*  context,
                                    struct SimOperator* self,
                                    double              dt) {
    (void) self;

    SimComplexMathOperatorState* state = (SimComplexMathOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimComplexMathOperatorConfig* cfg = &state->config;

    SimField* lhs_field = sim_context_field(context, cfg->lhs_field);
    SimField* rhs_field = sim_context_field(context, cfg->rhs_field);
    SimField* out_field = sim_context_field(context, cfg->output_field);
    if (lhs_field == NULL || out_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&lhs_field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (sim_field_element_count(&out_field->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const bool lhs_complex = sim_field_is_complex(lhs_field);
    const bool rhs_complex = (rhs_field != NULL) ? sim_field_is_complex(rhs_field) : false;
    const bool out_complex = sim_field_is_complex(out_field);

    if (!lhs_complex && lhs_field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (!out_complex && out_field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (cfg->rhs_source == SIM_COMPLEX_MATH_RHS_FIELD && rhs_field != NULL && !rhs_complex &&
        rhs_field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    const double*           lhs_real = lhs_complex ? NULL : sim_field_real_data_const(lhs_field);
    const SimComplexDouble* lhs_c    = lhs_complex ? sim_field_complex_data_const(lhs_field) : NULL;
    if (lhs_real == NULL && lhs_c == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const double*           rhs_real = NULL;
    const SimComplexDouble* rhs_c    = NULL;
    if (cfg->rhs_source == SIM_COMPLEX_MATH_RHS_FIELD) {
        if (rhs_field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (sim_field_element_count(&rhs_field->layout) != count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (rhs_complex) {
            rhs_c = sim_field_complex_data_const(rhs_field);
        } else {
            rhs_real = sim_field_real_data_const(rhs_field);
        }
        if (rhs_real == NULL && rhs_c == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    double*           out_real = out_complex ? NULL : sim_field_real_data(out_field);
    SimComplexDouble* out_c    = out_complex ? sim_field_complex_data(out_field) : NULL;
    if (out_real == NULL && out_c == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const double         scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;
    const double complex rhs_constant =
        complex_from_parts(cfg->rhs_constant_re, cfg->rhs_constant_im);

    for (size_t i = 0U; i < count; ++i) {
        double complex lhs = lhs_complex ? complex_from_parts(lhs_c[i].re, lhs_c[i].im)
                                         : complex_from_parts(lhs_real[i], 0.0);
        double complex rhs = rhs_constant;
        if (cfg->rhs_source == SIM_COMPLEX_MATH_RHS_FIELD) {
            rhs = rhs_complex ? complex_from_parts(rhs_c[i].re, rhs_c[i].im)
                              : complex_from_parts(rhs_real[i], 0.0);
        }

        lhs = complex_scale(lhs, cfg->lhs_scale_re, cfg->lhs_scale_im);
        rhs = complex_scale(rhs, cfg->rhs_scale_re, cfg->rhs_scale_im);

        double complex value = 0.0;
        switch (cfg->mode) {
            case SIM_COMPLEX_MATH_ADD:
                value = lhs + rhs;
                break;
            case SIM_COMPLEX_MATH_SUB:
                value = lhs - rhs;
                break;
            case SIM_COMPLEX_MATH_MUL:
                value = lhs * rhs;
                break;
            case SIM_COMPLEX_MATH_DIV:
                if (cabs(rhs) <= cfg->epsilon) {
                    value = 0.0;
                } else {
                    value = lhs / rhs;
                }
                break;
            case SIM_COMPLEX_MATH_POW:
                value = cpow(lhs, rhs);
                break;
            case SIM_COMPLEX_MATH_EXP:
                value = cexp(lhs);
                break;
            case SIM_COMPLEX_MATH_LOG:
                value = complex_log_safe(lhs, cfg->epsilon);
                break;
            case SIM_COMPLEX_MATH_SIN:
                value = csin(lhs);
                break;
            case SIM_COMPLEX_MATH_COS:
                value = ccos(lhs);
                break;
            case SIM_COMPLEX_MATH_TAN:
                value = ctan(lhs);
                break;
            case SIM_COMPLEX_MATH_SINH:
                value = csinh(lhs);
                break;
            case SIM_COMPLEX_MATH_COSH:
                value = ccosh(lhs);
                break;
            case SIM_COMPLEX_MATH_TANH:
                value = ctanh(lhs);
                break;
            case SIM_COMPLEX_MATH_SQRT:
                value = csqrt(lhs);
                break;
            case SIM_COMPLEX_MATH_CONJ:
                value = conj(lhs);
                break;
            case SIM_COMPLEX_MATH_ABS: {
                double mag = cabs(lhs);
                value      = complex_from_parts(mag, 0.0);
                break;
            }
            case SIM_COMPLEX_MATH_ARG: {
                double phase = complex_math_wrap_phase(carg(lhs), cfg->phase_wrap);
                value        = complex_from_parts(phase, 0.0);
                break;
            }
            case SIM_COMPLEX_MATH_REAL:
                value = complex_from_parts(creal(lhs), 0.0);
                break;
            case SIM_COMPLEX_MATH_IMAG:
                value = complex_from_parts(cimag(lhs), 0.0);
                break;
            case SIM_COMPLEX_MATH_NEG:
                value = -lhs;
                break;
            default:
                value = lhs;
                break;
        }

        value += complex_from_parts(cfg->bias_re, cfg->bias_im);

        if (out_complex) {
            SimComplexDouble out_value = complex_to_struct(value);
            if (!isfinite(out_value.re) || !isfinite(out_value.im)) {
                continue;
            }
            if (cfg->accumulate) {
                out_c[i].re += scale * out_value.re;
                out_c[i].im += scale * out_value.im;
            } else {
                out_c[i] = out_value;
            }
        } else {
            double out_value;
            if (cfg->mode == SIM_COMPLEX_MATH_ARG) {
                out_value = creal(value);
            } else {
                out_value =
                    complex_math_output_value(value, cfg->output_component, cfg->phase_wrap);
            }
            if (!isfinite(out_value)) {
                continue;
            }
            if (cfg->accumulate) {
                out_real[i] += scale * out_value;
            } else {
                out_real[i] = out_value;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult complex_math_step(void*               state_ptr,
                                   struct SimContext*  context,
                                   struct SimOperator* self,
                                   size_t              substep_index,
                                   double              dt_sub,
                                   void*               scratch,
                                   size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return complex_math_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_complex_math_operator(struct SimContext*                  context,
                                        const SimComplexMathOperatorConfig* config,
                                        size_t*                             out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimComplexMathOperatorState* state =
        (SimComplexMathOperatorState*) calloc(1U, sizeof(SimComplexMathOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimComplexMathOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.lhs_field        = 0U;
        local.rhs_field        = 0U;
        local.output_field     = 0U;
        local.mode             = SIM_COMPLEX_MATH_ADD;
        local.rhs_source       = SIM_COMPLEX_MATH_RHS_CONSTANT;
        local.rhs_constant_re  = 0.0;
        local.rhs_constant_im  = 0.0;
        local.lhs_scale_re     = 1.0;
        local.lhs_scale_im     = 0.0;
        local.rhs_scale_re     = 1.0;
        local.rhs_scale_im     = 0.0;
        local.bias_re          = 0.0;
        local.bias_im          = 0.0;
        local.epsilon          = 1.0e-9;
        local.output_component = SIM_COMPLEX_MATH_OUTPUT_REAL;
        local.phase_wrap       = SIM_COMPLEX_MATH_PHASE_NONE;
        local.accumulate       = false;
        local.scale_by_dt      = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "complex_math", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    complex_math_normalize_config(&local);
    state->config = local;
    complex_math_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "complex_math");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_NONLINEAR;
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
    info.abstract_id       = "complex_math";
    sim_operator_info_set_schema_identity(&info, "complex_math");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;

    SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField* out_field = sim_context_field(context, state->config.output_field);
    if (lhs_field == NULL || rhs_field == NULL || out_field == NULL) {
        complex_math_destroy(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const bool lhs_complex = sim_field_is_complex(lhs_field);
    const bool rhs_complex = sim_field_is_complex(rhs_field);
    const bool out_complex = sim_field_is_complex(out_field);

    info.representation.value_kind =
        out_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = out_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimSplitPort ports[3] = {
        { .context_field_index = state->config.lhs_field, .require_complex = lhs_complex },
        { .context_field_index = state->config.rhs_field, .require_complex = rhs_complex },
        { .context_field_index = state->config.output_field, .require_complex = out_complex }
    };

    SimSplitAccess accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_READ },
                                   { .port = 2, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = complex_math_step,
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
                                .symbolic      = complex_math_symbolic,
                                .destroy       = complex_math_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        complex_math_destroy(state);
    }

    return result;
}

SimResult sim_complex_math_config(struct SimContext*            context,
                                  size_t                        operator_index,
                                  SimComplexMathOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimComplexMathOperatorState* state = (SimComplexMathOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_complex_math_update(struct SimContext*                  context,
                                  size_t                              operator_index,
                                  const SimComplexMathOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimComplexMathOperatorState* state = (SimComplexMathOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimComplexMathOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "complex_math", true, config->scale_by_dt);
    }

    complex_math_normalize_config(&local);
    state->config = local;
    complex_math_refresh_symbolic(state);

    return SIM_RESULT_OK;
}
