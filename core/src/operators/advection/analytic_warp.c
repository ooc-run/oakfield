#include "oakfield/operators/advection/analytic_warp.h"
#include "operators/common/operator_utils.h"
#include "operators/common/warp_safety.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/math/special_functions.h"
#include "oakfield/operator_split.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANALYTIC_WARP_DELTA_MIN 1.0e-6
#define ANALYTIC_WARP_DEFAULT_HYPEREXP_EPSILON 1.0
#define ANALYTIC_WARP_DEFAULT_HYPEREXP_DEPTH 8
#define ANALYTIC_WARP_MAX_HYPEREXP_DEPTH 8192
#define ANALYTIC_WARP_DEFAULT_CLAMP_MIN (-1.0e6)
#define ANALYTIC_WARP_DEFAULT_CLAMP_MAX (1.0e6)
#define ANALYTIC_WARP_DEFAULT_CONTINUITY_TOL 1.0e-6

#define ANALYTIC_WARP_STATE_MAGIC 0x41575250u

typedef struct AnalyticWarpOperatorState {
    uint32_t                   magic;
    AnalyticWarpOperatorConfig config;
    char                       symbolic[128];
    SimWarpGuard               kernel_guard;
    SimSpecialFallbackFn       kernel_fallback;
    void*                      kernel_fallback_userdata;
    size_t                     kernel_cached_step_index;
    double                     kernel_cached_dt;
    const SimField*            kernel_cached_field;
    size_t                     kernel_cached_count;
    size_t                     kernel_cached_element;
    double                     kernel_cached_value[2];
    bool                       kernel_cache_valid;
    bool                       kernel_cached_element_valid;
} AnalyticWarpOperatorState;

static const char* analytic_warp_profile_name(AnalyticWarpProfile profile) {
    switch (profile) {
        case ANALYTIC_WARP_PROFILE_QHYPEREXP:
            return "qhyperexp";
        case ANALYTIC_WARP_PROFILE_HYPEREXP:
            return "hyperexp";
        case ANALYTIC_WARP_PROFILE_TRIGAMMA:
            return "trigamma";
        case ANALYTIC_WARP_PROFILE_POWER:
            return "power";
        case ANALYTIC_WARP_PROFILE_TANH:
            return "tanh";
        case ANALYTIC_WARP_PROFILE_DIGAMMA:
        default:
            return "digamma";
    }
}

SimWarpLevel sim_analytic_warp_effective_level(const AnalyticWarpOperatorConfig* config) {
    if (config == NULL) {
        return SIM_WARP_LEVEL_NONE;
    }
    switch (config->profile) {
        case ANALYTIC_WARP_PROFILE_TANH:
            return SIM_WARP_LEVEL_LEVEL0;
        case ANALYTIC_WARP_PROFILE_POWER:
            /* Monotone and well-behaved for exponent >= 1.0; otherwise treat as risky. */
            return (isfinite(config->exponent) && config->exponent >= 1.0) ? SIM_WARP_LEVEL_LEVEL1
                                                                           : SIM_WARP_LEVEL_LEVEL2;
        case ANALYTIC_WARP_PROFILE_DIGAMMA:
        case ANALYTIC_WARP_PROFILE_TRIGAMMA:
        case ANALYTIC_WARP_PROFILE_HYPEREXP:
        case ANALYTIC_WARP_PROFILE_QHYPEREXP:
            return SIM_WARP_LEVEL_LEVEL2;
        default:
            return SIM_WARP_LEVEL_LEVEL2;
    }
}

static double analytic_warp_power_gradient(double sample, double exponent) {
    double magnitude = fabs(sample);

    if (!isfinite(exponent) || exponent <= 0.0) {
        return 0.0;
    }

    if (magnitude < ANALYTIC_WARP_DELTA_MIN) {
        magnitude = ANALYTIC_WARP_DELTA_MIN;
    }

    return exponent * pow(magnitude, exponent - 1.0);
}

static double analytic_warp_tanh_gradient(double sample) {
    double t = tanh(sample);
    return 1.0 - t * t;
}

static SimResult analytic_warp_profile_gradient(const AnalyticWarpOperatorState* state,
                                                double                           sample,
                                                SimSpecialFallbackFn             fallback,
                                                void*                            fallback_userdata,
                                                double*                          out_gradient) {
    if (state == NULL || out_gradient == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    switch (state->config.profile) {
        case ANALYTIC_WARP_PROFILE_DIGAMMA: {
            double    value = 0.0;
            SimResult rc    = sim_trigamma_safe(sample, fallback, fallback_userdata, NULL, &value);
            if (rc != SIM_RESULT_OK)
                return rc;
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
        case ANALYTIC_WARP_PROFILE_TRIGAMMA: {
            double    value = 0.0;
            SimResult rc = sim_tetragamma_safe(sample, fallback, fallback_userdata, NULL, &value);
            if (rc != SIM_RESULT_OK)
                return rc;
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
        case ANALYTIC_WARP_PROFILE_POWER: {
            double value = analytic_warp_power_gradient(sample, state->config.exponent);
            if (!isfinite(value)) {
                SimSpecialEvalReport report = { .fault           = SIM_SPECIAL_FAULT_NUMERIC,
                                                .function        = "analytic_warp_power_gradient",
                                                .input           = { sample, 0.0 },
                                                .q_param         = NAN,
                                                .aux_param       = NAN,
                                                .exponent_param  = state->config.exponent,
                                                .iteration_count = 0,
                                                .residual        = NAN,
                                                .tolerance       = 0.0 };
                if (fallback != NULL) {
                    SimComplexDouble fb_value = { NAN, NAN };
                    SimResult        fb       = fallback(fallback_userdata, &report, &fb_value);
                    if (fb == SIM_RESULT_OK) {
                        *out_gradient = fb_value.re;
                        return SIM_RESULT_OK;
                    }
                    return fb;
                }
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
        case ANALYTIC_WARP_PROFILE_TANH: {
            double value  = analytic_warp_tanh_gradient(sample);
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
        case ANALYTIC_WARP_PROFILE_HYPEREXP: {
            double epsilon = state->config.hyperexp_epsilon;
            int    depth   = state->config.hyperexp_depth;
            if (!isfinite(epsilon) || epsilon <= 0.0 || depth <= 0 ||
                depth > ANALYTIC_WARP_MAX_HYPEREXP_DEPTH) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            double    value = 0.0;
            SimResult rc    = sim_hyperexp_phi_deriv_safe(
                sample, epsilon, depth, fallback, fallback_userdata, NULL, &value);
            if (rc != SIM_RESULT_OK)
                return rc;
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
        case ANALYTIC_WARP_PROFILE_QHYPEREXP: {
            double epsilon = state->config.hyperexp_epsilon;
            int    depth   = state->config.hyperexp_depth;
            double q       = state->config.hyperexp_q;
            if (!isfinite(epsilon) || epsilon <= 0.0 || depth <= 0 ||
                depth > ANALYTIC_WARP_MAX_HYPEREXP_DEPTH) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (!isfinite(q) || q <= 0.0 || q >= 1.0) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            double value = sim_qhyperexp_phi_deriv(sample, epsilon, depth, q);
            if (!isfinite(value)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
        default: {
            double    value = 0.0;
            SimResult rc    = sim_trigamma_safe(sample, fallback, fallback_userdata, NULL, &value);
            if (rc != SIM_RESULT_OK)
                return rc;
            *out_gradient = value;
            return SIM_RESULT_OK;
        }
    }
}

static SimResult analytic_warp_gradient_probe(void*                userdata,
                                              double               sample,
                                              SimSpecialFallbackFn fallback,
                                              void*                fallback_userdata,
                                              double*              out_gradient) {
    const AnalyticWarpOperatorState* state = (const AnalyticWarpOperatorState*) userdata;
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    return analytic_warp_profile_gradient(state, sample, fallback, fallback_userdata, out_gradient);
}

static void analytic_warp_normalize_config(AnalyticWarpOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    switch (config->profile) {
        case ANALYTIC_WARP_PROFILE_DIGAMMA:
        case ANALYTIC_WARP_PROFILE_TRIGAMMA:
        case ANALYTIC_WARP_PROFILE_POWER:
        case ANALYTIC_WARP_PROFILE_TANH:
        case ANALYTIC_WARP_PROFILE_HYPEREXP:
            break;
        case ANALYTIC_WARP_PROFILE_QHYPEREXP:
            break;
        default:
            config->profile = ANALYTIC_WARP_PROFILE_DIGAMMA;
            break;
    }

    if (!isfinite(config->delta) || fabs(config->delta) < ANALYTIC_WARP_DELTA_MIN) {
        config->delta = ANALYTIC_WARP_DELTA_MIN;
    }
    if (config->delta < 0.0) {
        config->delta = -config->delta;
    }

    if (!isfinite(config->lambda)) {
        config->lambda = 1.0;
    }

    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }

    if (!isfinite(config->exponent)) {
        config->exponent = 2.0;
    }
    if (config->exponent == 0.0) {
        config->exponent = 1.0;
    }
    if (config->exponent < 0.0) {
        config->exponent = fabs(config->exponent);
    }

    /* Normalize complex mode enum. */
    switch (config->complex_mode) {
        case ANALYTIC_WARP_COMPLEX_MODE_COMPONENT:
        case ANALYTIC_WARP_COMPLEX_MODE_POLAR:
            break;
        default:
            config->complex_mode = ANALYTIC_WARP_COMPLEX_MODE_COMPONENT;
            break;
    }

    if (!isfinite(config->hyperexp_epsilon) || config->hyperexp_epsilon <= 0.0) {
        config->hyperexp_epsilon = ANALYTIC_WARP_DEFAULT_HYPEREXP_EPSILON;
    }
    if (config->hyperexp_depth <= 0) {
        config->hyperexp_depth = ANALYTIC_WARP_DEFAULT_HYPEREXP_DEPTH;
    } else if (config->hyperexp_depth > ANALYTIC_WARP_MAX_HYPEREXP_DEPTH) {
        config->hyperexp_depth = ANALYTIC_WARP_MAX_HYPEREXP_DEPTH;
    }

    switch (config->continuity) {
        case SIM_CONTINUITY_NONE:
        case SIM_CONTINUITY_STRICT:
        case SIM_CONTINUITY_CLAMPED:
        case SIM_CONTINUITY_LIMITED:
            break;
        default:
            config->continuity = SIM_CONTINUITY_NONE;
            break;
    }

    if (!isfinite(config->continuity_clamp_min)) {
        config->continuity_clamp_min = ANALYTIC_WARP_DEFAULT_CLAMP_MIN;
    }
    if (!isfinite(config->continuity_clamp_max)) {
        config->continuity_clamp_max = ANALYTIC_WARP_DEFAULT_CLAMP_MAX;
    }
    if (config->continuity_clamp_min == 0.0 && config->continuity_clamp_max == 0.0) {
        config->continuity_clamp_min = ANALYTIC_WARP_DEFAULT_CLAMP_MIN;
        config->continuity_clamp_max = ANALYTIC_WARP_DEFAULT_CLAMP_MAX;
    }
    if (config->continuity_clamp_min > config->continuity_clamp_max) {
        double tmp                   = config->continuity_clamp_min;
        config->continuity_clamp_min = config->continuity_clamp_max;
        config->continuity_clamp_max = tmp;
    }

    if (!isfinite(config->continuity_tolerance) || config->continuity_tolerance <= 0.0) {
        config->continuity_tolerance = ANALYTIC_WARP_DEFAULT_CONTINUITY_TOL;
    }

    if (!isfinite(config->hyperexp_q) || config->hyperexp_q <= 0.0 || config->hyperexp_q >= 1.0) {
        config->hyperexp_q = 0.9;
    }
}

static bool analytic_warp_continuity_valid(SimContinuityMode mode) {
    switch (mode) {
        case SIM_CONTINUITY_NONE:
        case SIM_CONTINUITY_STRICT:
        case SIM_CONTINUITY_CLAMPED:
        case SIM_CONTINUITY_LIMITED:
            return true;
        default:
            return false;
    }
}

static double analytic_warp_positive_or_default(double requested, double fallback) {
    if (isfinite(requested) && requested > 0.0) {
        return requested;
    }
    return fallback;
}

static SimWarpGuard analytic_warp_guard_from_config(const AnalyticWarpOperatorConfig* config) {
    const double kFallbackTolerance = 1.0e-6;
    SimWarpGuard guard = { .mode      = SIM_CONTINUITY_NONE,
                           .clamp_min = ANALYTIC_WARP_DEFAULT_CLAMP_MIN,
                           .clamp_max = ANALYTIC_WARP_DEFAULT_CLAMP_MAX,
                           .tolerance = analytic_warp_positive_or_default(
                               ANALYTIC_WARP_DEFAULT_CONTINUITY_TOL, kFallbackTolerance) };

    if (config != NULL) {
        if (analytic_warp_continuity_valid(config->continuity)) {
            guard.mode = config->continuity;
        }

        if (isfinite(config->continuity_clamp_min) && isfinite(config->continuity_clamp_max) &&
            config->continuity_clamp_min < config->continuity_clamp_max) {
            guard.clamp_min = config->continuity_clamp_min;
            guard.clamp_max = config->continuity_clamp_max;
        }

        guard.tolerance =
            analytic_warp_positive_or_default(config->continuity_tolerance, guard.tolerance);
    }

    guard.clamp_min = isfinite(guard.clamp_min) ? guard.clamp_min : ANALYTIC_WARP_DEFAULT_CLAMP_MIN;
    guard.clamp_max = isfinite(guard.clamp_max) ? guard.clamp_max : ANALYTIC_WARP_DEFAULT_CLAMP_MAX;
    if (!(guard.clamp_min < guard.clamp_max)) {
        guard.clamp_min = ANALYTIC_WARP_DEFAULT_CLAMP_MIN;
        guard.clamp_max = ANALYTIC_WARP_DEFAULT_CLAMP_MAX;
    }

    guard.tolerance = analytic_warp_positive_or_default(guard.tolerance, kFallbackTolerance);
    return guard;
}

static void analytic_warp_refresh_symbolic(AnalyticWarpOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const char* profile_name = analytic_warp_profile_name(state->config.profile);
    double      lambda       = state->config.lambda;
    double      delta        = fabs(state->config.delta);
    double      bias         = state->config.bias;

    switch (state->config.profile) {
        case ANALYTIC_WARP_PROFILE_POWER:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "%s(lambda=%.3g, delta=%.3g, bias=%.3g, p=%.3g)",
                            profile_name,
                            lambda,
                            delta,
                            bias,
                            state->config.exponent);
            break;
        case ANALYTIC_WARP_PROFILE_TANH:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "%s(lambda=%.3g, delta=%.3g, bias=%.3g)",
                            profile_name,
                            lambda,
                            delta,
                            bias);
            break;
        case ANALYTIC_WARP_PROFILE_HYPEREXP:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "%s(lambda=%.3g, delta=%.3g, bias=%.3g, eps=%.3g, K=%d)",
                            profile_name,
                            lambda,
                            delta,
                            bias,
                            state->config.hyperexp_epsilon,
                            (state->config.hyperexp_depth > 0)
                                ? state->config.hyperexp_depth
                                : ANALYTIC_WARP_DEFAULT_HYPEREXP_DEPTH);
            break;
        case ANALYTIC_WARP_PROFILE_QHYPEREXP:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "%s(lambda=%.3g, delta=%.3g, bias=%.3g, eps=%.3g, K=%d, q=%.3g)",
                            profile_name,
                            lambda,
                            delta,
                            bias,
                            state->config.hyperexp_epsilon,
                            (state->config.hyperexp_depth > 0)
                                ? state->config.hyperexp_depth
                                : ANALYTIC_WARP_DEFAULT_HYPEREXP_DEPTH,
                            state->config.hyperexp_q);
            break;
        case ANALYTIC_WARP_PROFILE_TRIGAMMA:
        case ANALYTIC_WARP_PROFILE_DIGAMMA:
        default:
            (void) snprintf(state->symbolic,
                            sizeof(state->symbolic),
                            "%s(lambda=%.3g, delta=%.3g, bias=%.3g)",
                            profile_name,
                            lambda,
                            delta,
                            bias);
            break;
    }
#else
    (void) state;
#endif
}

static const char* analytic_warp_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const AnalyticWarpOperatorState* state = (const AnalyticWarpOperatorState*) state_ptr;
    if (state == NULL) {
        return NULL;
    }
    return state->symbolic;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult analytic_warp_apply(void*               state_ptr,
                                     struct SimContext*  context,
                                     struct SimOperator* self,
                                     double              dt) {
    AnalyticWarpOperatorState* state = (AnalyticWarpOperatorState*) state_ptr;
    SimField*                  field;
    void*                      raw_data;
    bool                       is_complex = false;
    double                     delta;
    size_t                     i;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    raw_data = sim_field_data(field);
    if (raw_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = 0U;
    if (field->element_size == sizeof(double)) {
        count = sim_field_bytes(field) / sizeof(double);
    } else if (field->element_size == sizeof(double complex)) {
        count      = sim_field_bytes(field) / sizeof(double complex);
        is_complex = true;
    } else {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    delta = fabs(state->config.delta);
    if (delta < ANALYTIC_WARP_DELTA_MIN) {
        delta = ANALYTIC_WARP_DELTA_MIN;
    }

    SimSpecialFallbackFn fallback          = NULL;
    void*                fallback_userdata = NULL;
    sim_context_special_fallback_hook(context, &fallback, &fallback_userdata);

    SimWarpGuard guard;
    sim_warp_guard_from_operator(self,
                                 ANALYTIC_WARP_DEFAULT_CLAMP_MIN,
                                 ANALYTIC_WARP_DEFAULT_CLAMP_MAX,
                                 ANALYTIC_WARP_DEFAULT_CONTINUITY_TOL,
                                 &guard);

    SimWarpSampleSpec sample_spec = { .sample = 0.0,
                                      .bias   = state->config.bias,
                                      .delta  = delta,
                                      .lambda = state->config.lambda,
                                      .guard  = guard };

    if (!is_complex) {
        double* data = (double*) raw_data;
        for (i = 0U; i < count; ++i) {
            double response    = 0.0;
            sample_spec.sample = data[i];
            SimResult rc       = sim_warp_sample_response(&sample_spec,
                                                          analytic_warp_gradient_probe,
                                                          state,
                                                          fallback,
                                                          fallback_userdata,
                                                          &response);
            if (rc != SIM_RESULT_OK) {
                if (rc != SIM_RESULT_INVALID_ARGUMENT)
                    return rc;
                continue;
            }
            if (!isfinite(response)) {
                continue;
            }
            data[i] += dt * response;
        }
    } else {
        double complex* data = (double complex*) raw_data;
        if (state->config.complex_mode == ANALYTIC_WARP_COMPLEX_MODE_POLAR) {
            /* Polar mode: compute gradient at magnitude, apply along current phase. */
            for (i = 0U; i < count; ++i) {
                double complex z        = data[i];
                double         r        = cabs(z);
                double         response = 0.0;
                sample_spec.sample      = r;
                SimResult rc            = sim_warp_sample_response(&sample_spec,
                                                                   analytic_warp_gradient_probe,
                                                                   state,
                                                                   fallback,
                                                                   fallback_userdata,
                                                                   &response);
                if (rc != SIM_RESULT_OK) {
                    if (rc != SIM_RESULT_INVALID_ARGUMENT)
                        return rc;
                    continue;
                }
                if (!isfinite(response)) {
                    continue;
                }
                double complex unit = (r > 0.0) ? (z / r) : 1.0;
                data[i]             = z + dt * response * unit;
            }
        } else {
            /* Component-wise mode: operate on Re/Im separately. */
            for (i = 0U; i < count; ++i) {
                double complex value         = data[i];
                double         real_part     = creal(value);
                double         imag_part     = cimag(value);
                double         real_response = 0.0;
                double         imag_response = 0.0;
                sample_spec.sample           = real_part;
                SimResult rc_real  = sim_warp_sample_response(&sample_spec,
                                                              analytic_warp_gradient_probe,
                                                              state,
                                                              fallback,
                                                              fallback_userdata,
                                                              &real_response);
                sample_spec.sample = imag_part;
                SimResult rc_imag  = sim_warp_sample_response(&sample_spec,
                                                              analytic_warp_gradient_probe,
                                                              state,
                                                              fallback,
                                                              fallback_userdata,
                                                              &imag_response);

                if (rc_real != SIM_RESULT_OK) {
                    if (rc_real != SIM_RESULT_INVALID_ARGUMENT)
                        return rc_real;
                } else if (isfinite(real_response)) {
                    real_part += dt * real_response;
                }
                if (rc_imag != SIM_RESULT_OK) {
                    if (rc_imag != SIM_RESULT_INVALID_ARGUMENT)
                        return rc_imag;
                } else if (isfinite(imag_response)) {
                    imag_part += dt * imag_response;
                }

                data[i] = real_part + I * imag_part;
            }
        }
    }

    return SIM_RESULT_OK;
}

static double analytic_warp_kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult analytic_warp_ir_eval(void*           userdata,
                                       const KernelIR* kernel,
                                       size_t          element_index,
                                       size_t          component,
                                       double*         out_value) {
    if (out_value == NULL || userdata == NULL || kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    AnalyticWarpOperatorState* state = (AnalyticWarpOperatorState*) userdata;
    if (kernel->bindings == NULL || kernel->binding_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimKernelIRBinding* binding = &kernel->bindings[0];
    SimField*                 field   = binding->field;
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex      = sim_field_is_complex(field);
    size_t component_count = is_complex ? 2U : 1U;
    if (component >= component_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t element_count = sim_field_element_count(&field->layout);
    if (element_count == 0U || element_index >= element_count) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    double dt = analytic_warp_kernel_param_value(kernel, SIM_IR_PARAM_DT);

    size_t step_index = 0U;

    bool have_step =
        (kernel->params != NULL && kernel->param_count > (size_t) SIM_IR_PARAM_STEP_INDEX);

    if (have_step) {
        double step_value = kernel->params[SIM_IR_PARAM_STEP_INDEX];
        if (isfinite(step_value) && step_value >= 0.0) {
            step_index = (size_t) step_value;
        } else {
            have_step = false;
        }
    }

    bool need_update = !state->kernel_cache_valid || state->kernel_cached_field != field ||
                       state->kernel_cached_count != element_count ||
                       state->kernel_cached_dt != dt || !have_step ||
                       state->kernel_cached_step_index != step_index;

    if (need_update) {
        state->kernel_cached_field         = field;
        state->kernel_cached_count         = element_count;
        state->kernel_cached_dt            = dt;
        state->kernel_cached_step_index    = step_index;
        state->kernel_cache_valid          = have_step;
        state->kernel_cached_element_valid = false;
    }

    if (!state->kernel_cached_element_valid || state->kernel_cached_element != element_index) {
        double delta = fabs(state->config.delta);
        if (delta < ANALYTIC_WARP_DELTA_MIN) {
            delta = ANALYTIC_WARP_DELTA_MIN;
        }

        SimWarpSampleSpec sample_spec = { .sample = 0.0,
                                          .bias   = state->config.bias,
                                          .delta  = delta,
                                          .lambda = state->config.lambda,
                                          .guard  = state->kernel_guard };

        if (!is_complex) {
            const double* data = sim_field_real_data_const(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            double value       = data[element_index];
            double response    = 0.0;
            sample_spec.sample = value;
            SimResult rc       = sim_warp_sample_response(&sample_spec,
                                                          analytic_warp_gradient_probe,
                                                          state,
                                                          state->kernel_fallback,
                                                          state->kernel_fallback_userdata,
                                                          &response);
            if (rc != SIM_RESULT_OK) {
                if (rc != SIM_RESULT_INVALID_ARGUMENT) {
                    return rc;
                }
            } else if (isfinite(response)) {
                value += dt * response;
            }
            state->kernel_cached_value[0] = value;
        } else {
            const SimComplexDouble* data = sim_field_complex_data_const(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            SimComplexDouble value = data[element_index];
            if (state->config.complex_mode == ANALYTIC_WARP_COMPLEX_MODE_POLAR) {
                double r           = sqrt(value.re * value.re + value.im * value.im);
                double response    = 0.0;
                sample_spec.sample = r;
                SimResult rc       = sim_warp_sample_response(&sample_spec,
                                                              analytic_warp_gradient_probe,
                                                              state,
                                                              state->kernel_fallback,
                                                              state->kernel_fallback_userdata,
                                                              &response);
                if (rc != SIM_RESULT_OK) {
                    if (rc != SIM_RESULT_INVALID_ARGUMENT) {
                        return rc;
                    }
                } else if (isfinite(response)) {
                    double nx = (r > 0.0) ? (value.re / r) : 1.0;
                    double ny = (r > 0.0) ? (value.im / r) : 0.0;
                    value.re += dt * response * nx;
                    value.im += dt * response * ny;
                }
            } else {
                double real_part     = value.re;
                double imag_part     = value.im;
                double real_response = 0.0;
                double imag_response = 0.0;

                sample_spec.sample = real_part;
                SimResult rc_real  = sim_warp_sample_response(&sample_spec,
                                                              analytic_warp_gradient_probe,
                                                              state,
                                                              state->kernel_fallback,
                                                              state->kernel_fallback_userdata,
                                                              &real_response);
                sample_spec.sample = imag_part;
                SimResult rc_imag  = sim_warp_sample_response(&sample_spec,
                                                              analytic_warp_gradient_probe,
                                                              state,
                                                              state->kernel_fallback,
                                                              state->kernel_fallback_userdata,
                                                              &imag_response);

                if (rc_real != SIM_RESULT_OK) {
                    if (rc_real != SIM_RESULT_INVALID_ARGUMENT) {
                        return rc_real;
                    }
                } else if (isfinite(real_response)) {
                    real_part += dt * real_response;
                }

                if (rc_imag != SIM_RESULT_OK) {
                    if (rc_imag != SIM_RESULT_INVALID_ARGUMENT) {
                        return rc_imag;
                    }
                } else if (isfinite(imag_response)) {
                    imag_part += dt * imag_response;
                }

                value.re = real_part;
                value.im = imag_part;
            }

            state->kernel_cached_value[0] = value.re;
            state->kernel_cached_value[1] = value.im;
        }

        state->kernel_cached_element       = element_index;
        state->kernel_cached_element_valid = true;
    }

    *out_value = state->kernel_cached_value[component];
    return SIM_RESULT_OK;
}

static SimResult analytic_warp_step(void*               state_ptr,
                                    struct SimContext*  context,
                                    struct SimOperator* self,
                                    size_t              substep_index,
                                    double              dt_sub,
                                    void*               scratch,
                                    size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return analytic_warp_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_analytic_warp_operator(struct SimContext*                context,
                                         const AnalyticWarpOperatorConfig* config,
                                         size_t*                           out_index) {
    AnalyticWarpOperatorState* state;
    AnalyticWarpOperatorConfig local  = { 0 };
    SimResult                  result = SIM_RESULT_OK;
    char                       name[SIM_OPERATOR_NAME_MAX + 1U];

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool hyperexp_epsilon_explicit = false;
    bool hyperexp_depth_explicit   = false;
    if (config != NULL) {
        local = *config;
        hyperexp_epsilon_explicit =
            isfinite(config->hyperexp_epsilon) && config->hyperexp_epsilon > 0.0;
        hyperexp_depth_explicit = config->hyperexp_depth > 0;
    } else {
        local.continuity           = SIM_CONTINUITY_NONE;
        local.continuity_clamp_min = ANALYTIC_WARP_DEFAULT_CLAMP_MIN;
        local.continuity_clamp_max = ANALYTIC_WARP_DEFAULT_CLAMP_MAX;
        local.continuity_tolerance = ANALYTIC_WARP_DEFAULT_CONTINUITY_TOL;
    }

    analytic_warp_normalize_config(&local);

    if (!hyperexp_epsilon_explicit) {
        double u_epsilon = (double) sim_context_epsilon(context);
        if (isfinite(u_epsilon) && u_epsilon > 0.0) {
            local.hyperexp_epsilon = u_epsilon;
        }
    }
    if (!hyperexp_depth_explicit && sim_context_truncation_level(context) > 0U) {
        int depth = (int) sim_context_truncation_level(context);
        if (depth > ANALYTIC_WARP_MAX_HYPEREXP_DEPTH) {
            depth = ANALYTIC_WARP_MAX_HYPEREXP_DEPTH;
        }
        local.hyperexp_depth = (depth > 0) ? depth : local.hyperexp_depth;
    }

    state = (AnalyticWarpOperatorState*) calloc(1U, sizeof(AnalyticWarpOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->magic  = ANALYTIC_WARP_STATE_MAGIC;
    state->config = local;
    analytic_warp_refresh_symbolic(state);
    state->kernel_guard             = analytic_warp_guard_from_config(&state->config);
    state->kernel_fallback          = NULL;
    state->kernel_fallback_userdata = NULL;
    sim_context_special_fallback_hook(
        context, &state->kernel_fallback, &state->kernel_fallback_userdata);
    state->kernel_cached_step_index    = 0U;
    state->kernel_cached_dt            = 0.0;
    state->kernel_cached_field         = NULL;
    state->kernel_cached_count         = 0U;
    state->kernel_cached_element       = 0U;
    state->kernel_cached_value[0]      = 0.0;
    state->kernel_cached_value[1]      = 0.0;
    state->kernel_cache_valid          = false;
    state->kernel_cached_element_valid = false;

    sim_operator_make_unique_name(name, sizeof(name), "analytic_warp");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_ADVECTION;
    info.warp_level        = sim_analytic_warp_effective_level(&state->config);
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = true;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "analytic_warp";
    sim_operator_info_set_schema_identity(&info, "analytic_warp");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    SimField* field            = sim_context_field(context, local.field_index);
    bool      needs_complex    = (field != NULL) ? sim_field_is_complex(field) : false;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimOperatorConfig op_config = sim_operator_config_defaults();
    op_config.continuity        = local.continuity;
    op_config.clamp_min         = local.continuity_clamp_min;
    op_config.clamp_max         = local.continuity_clamp_max;
    op_config.continuity_tol    = local.continuity_tolerance;

    bool registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "analytic_warp")) {
        bool   is_complex    = needs_complex;
        size_t expected_size = is_complex ? sizeof(SimComplexDouble) : sizeof(double);
        if (field != NULL && field->element_size == expected_size) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRStatefulSpec spec = { 0 };
                spec.eval              = analytic_warp_ir_eval;
                spec.userdata          = state;
                spec.label             = "analytic_warp";
                spec.value_type        = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId warp_node  = sim_ir_builder_stateful_spec(builder, &spec);

                if (warp_node != SIM_IR_INVALID_NODE) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = warp_node;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_STEP_INDEX + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = free;
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
                                .fn                = analytic_warp_step,
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
                                .symbolic      = analytic_warp_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_analytic_warp_config(struct SimContext*          context,
                                   size_t                      operator_index,
                                   AnalyticWarpOperatorConfig* out_config) {
    SimOperator*               op;
    AnalyticWarpOperatorState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (AnalyticWarpOperatorState*) sim_operator_state(op);
    if (state == NULL || state->magic != ANALYTIC_WARP_STATE_MAGIC) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_analytic_warp_update(struct SimContext*                context,
                                   size_t                            operator_index,
                                   const AnalyticWarpOperatorConfig* config) {
    SimOperator*               op;
    AnalyticWarpOperatorState* state;
    AnalyticWarpOperatorConfig local;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (AnalyticWarpOperatorState*) sim_operator_state(op);
    if (state == NULL || state->magic != ANALYTIC_WARP_STATE_MAGIC) {
        return SIM_RESULT_INVALID_STATE;
    }

    local                          = state->config;
    bool hyperexp_epsilon_explicit = false;
    bool hyperexp_depth_explicit   = false;
    if (config != NULL) {
        local = *config;
        hyperexp_epsilon_explicit =
            isfinite(config->hyperexp_epsilon) && config->hyperexp_epsilon > 0.0;
        hyperexp_depth_explicit = config->hyperexp_depth > 0;
    }

    analytic_warp_normalize_config(&local);

    if (!hyperexp_epsilon_explicit) {
        double u_epsilon = (double) sim_context_epsilon(context);
        if (isfinite(u_epsilon) && u_epsilon > 0.0) {
            local.hyperexp_epsilon = u_epsilon;
        }
    }
    if (!hyperexp_depth_explicit && sim_context_truncation_level(context) > 0U) {
        int depth = (int) sim_context_truncation_level(context);
        if (depth > ANALYTIC_WARP_MAX_HYPEREXP_DEPTH) {
            depth = ANALYTIC_WARP_MAX_HYPEREXP_DEPTH;
        }
        local.hyperexp_depth = (depth > 0) ? depth : local.hyperexp_depth;
    }

    state->config = local;
    analytic_warp_refresh_symbolic(state);
    state->kernel_guard             = analytic_warp_guard_from_config(&state->config);
    state->kernel_fallback          = NULL;
    state->kernel_fallback_userdata = NULL;
    sim_context_special_fallback_hook(
        context, &state->kernel_fallback, &state->kernel_fallback_userdata);
    state->kernel_cached_field         = NULL;
    state->kernel_cached_count         = 0U;
    state->kernel_cached_dt            = 0.0;
    state->kernel_cached_step_index    = 0U;
    state->kernel_cache_valid          = false;
    state->kernel_cached_element_valid = false;
    SimOperatorConfig op_config        = sim_operator_config_defaults();
    op_config.continuity               = local.continuity;
    op_config.clamp_min                = local.continuity_clamp_min;
    op_config.clamp_max                = local.continuity_clamp_max;
    op_config.continuity_tol           = local.continuity_tolerance;
    sim_operator_config_set(op, &op_config);

    op->info.warp_level = sim_analytic_warp_effective_level(&state->config);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
