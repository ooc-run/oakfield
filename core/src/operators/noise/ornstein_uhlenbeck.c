#include "oakfield/operators/noise/ornstein_uhlenbeck.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_noise_source.h"
#include "oakfield/sim_seed.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ORNSTEIN_UHLENBECK_EPS 1.0e-12

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const char* ornstein_uhlenbeck_complex_mode_name(SimOrnsteinUhlenbeckComplexMode mode) {
    switch (mode) {
        case SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT:
            return "component";
        case SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_POLAR:
            return "polar";
        default:
            return "component";
    }
}

static inline void ou_sincos(double angle, double* s, double* c) {
#if defined(__APPLE__)
    __sincos(angle, s, c);
#elif defined(__clang__) || defined(__GNUC__)
    sincos(angle, s, c);
#else
    if (s != NULL) {
        *s = sin(angle);
    }
    if (c != NULL) {
        *c = cos(angle);
    }
#endif
}

typedef struct SimOrnsteinUhlenbeckState {
    SimOrnsteinUhlenbeckOperatorConfig config;
    SimNoiseSourceRng                  rng;
    char                               symbolic[160];
} SimOrnsteinUhlenbeckState;

static void ornstein_uhlenbeck_normalize(SimOrnsteinUhlenbeckOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    switch (config->complex_mode) {
        case SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT:
        case SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_POLAR:
            break;
        default:
            config->complex_mode = SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT;
            break;
    }

    if (!isfinite(config->mean)) {
        config->mean = 0.0;
    }
    if (!isfinite(config->sigma) || config->sigma < 0.0) {
        config->sigma = 0.0;
    }
    if (!isfinite(config->tau) || config->tau <= ORNSTEIN_UHLENBECK_EPS) {
        config->tau = 1.0;
    }
    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }
}

static void ornstein_uhlenbeck_refresh_symbolic(SimOrnsteinUhlenbeckState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "ornstein_uhlenbeck mean=%.3g sigma=%.3g tau=%.3g complex=%s",
                    state->config.mean,
                    state->config.sigma,
                    state->config.tau,
                    ornstein_uhlenbeck_complex_mode_name(state->config.complex_mode));
#else
    (void) state;
#endif
}

static const char* ornstein_uhlenbeck_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimOrnsteinUhlenbeckState* state = (const SimOrnsteinUhlenbeckState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static bool ornstein_uhlenbeck_validate_field(const SimField* field) {
    if (field == NULL) {
        return false;
    }
    if (field->element_size != sizeof(double) && field->element_size != sizeof(SimComplexDouble)) {
        return false;
    }
    return sim_field_element_count(&field->layout) > 0U;
}

static double ornstein_uhlenbeck_wrap_angle(double angle) {
    double wrapped = remainder(angle, 2.0 * M_PI);
    if (wrapped <= -M_PI) {
        wrapped += 2.0 * M_PI;
    } else if (wrapped > M_PI) {
        wrapped -= 2.0 * M_PI;
    }
    return wrapped;
}

static void ornstein_uhlenbeck_fill_info(SimOperatorInfo* info, bool needs_complex) {
    if (info == NULL) {
        return;
    }

    *info                   = sim_operator_info_defaults();
    info->category          = SIM_OPERATOR_CATEGORY_DIFFUSION;
    info->warp_level        = SIM_WARP_LEVEL_NONE;
    info->is_noise          = true;
    info->is_spectral       = false;
    info->is_local          = true;
    info->is_nonlocal       = false;
    info->is_linear         = false;
    info->is_warp           = false;
    info->is_differentiable = false;
    info->preserves_real    = !needs_complex;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "ornstein_uhlenbeck";
    sim_operator_info_set_schema_identity(info, "ornstein_uhlenbeck");
    info->algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info->representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info->representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = info->preserves_real;
}

static SimResult ornstein_uhlenbeck_apply(void*               state_ptr,
                                          struct SimContext*  context,
                                          struct SimOperator* self,
                                          double              dt) {
    (void) self;

    SimOrnsteinUhlenbeckState* state = (SimOrnsteinUhlenbeckState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (dt < 0.0) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (dt == 0.0) {
        return SIM_RESULT_OK;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!ornstein_uhlenbeck_validate_field(field)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const double mean       = state->config.mean;
    const double tau        = fmax(state->config.tau, ORNSTEIN_UHLENBECK_EPS);
    double       decay      = 1.0;
    double       variance   = 0.0;
    const bool   is_complex = (field->element_size == sizeof(SimComplexDouble));

    sim_noise_source_temporal_params(state->config.sigma, tau, 1.0, dt, &decay, &variance);

    if (!is_complex) {
        double* data = (double*) sim_field_data(field);
        size_t  count;

        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        count = sim_field_bytes(field) / sizeof(double);
        if (count == 0U) {
            return SIM_RESULT_OK;
        }

        sim_noise_source_apply_mean_reverting_real(data, count, mean, decay, variance, &state->rng);
        return SIM_RESULT_OK;
    }

    if (state->config.complex_mode == SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT) {
        SimComplexDouble* data  = sim_field_complex_data(field);
        size_t            count = sim_field_element_count(&field->layout);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (count == 0U) {
            return SIM_RESULT_OK;
        }

        sim_noise_source_apply_mean_reverting_complex(
            data, count, mean, decay, variance, &state->rng);
        return SIM_RESULT_OK;
    }

    SimComplexDouble* data  = sim_field_complex_data(field);
    size_t            count = sim_field_element_count(&field->layout);
    if (data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < count; ++i) {
        double radius = hypot(data[i].re, data[i].im);
        double phase  = atan2(data[i].im, data[i].re);
        double nr     = 0.0;
        double np     = 0.0;
        double s      = 0.0;
        double c      = 0.0;

        if (variance != 0.0) {
            nr = sim_noise_source_normal(&state->rng);
            np = sim_noise_source_normal(&state->rng);
        }

        radius = mean + decay * (radius - mean) + variance * nr;
        phase  = mean + decay * (phase - mean) + variance * np;
        if (radius < 0.0) {
            radius = -radius;
            phase += M_PI;
        }
        phase = ornstein_uhlenbeck_wrap_angle(phase);
        ou_sincos(phase, &s, &c);
        data[i].re = radius * c;
        data[i].im = radius * s;
    }

    return SIM_RESULT_OK;
}

static SimResult ornstein_uhlenbeck_step(void*               state_ptr,
                                         struct SimContext*  context,
                                         struct SimOperator* self,
                                         size_t              substep_index,
                                         double              dt_sub,
                                         void*               scratch,
                                         size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return ornstein_uhlenbeck_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_ornstein_uhlenbeck_operator(struct SimContext*                        context,
                                              const SimOrnsteinUhlenbeckOperatorConfig* config,
                                              size_t*                                   out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOrnsteinUhlenbeckOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.mean  = 0.0;
        local.sigma = 1.0;
        local.tau   = 1.0;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("ornstein_uhlenbeck"),
                                     sim_context_operator_count(context));
    }
    ornstein_uhlenbeck_normalize(&local);

    SimField* field = sim_context_field(context, local.field_index);
    if (!ornstein_uhlenbeck_validate_field(field)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOrnsteinUhlenbeckState* state =
        (SimOrnsteinUhlenbeckState*) calloc(1U, sizeof(SimOrnsteinUhlenbeckState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    sim_noise_source_seed(&state->rng, local.seed, local.seed ^ SIM_NOISE_SOURCE_STREAM_ORNSTEIN);
    ornstein_uhlenbeck_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "ornstein_uhlenbeck");

    SimOperatorInfo info = sim_operator_info_defaults();
    ornstein_uhlenbeck_fill_info(&info, field->element_size == sizeof(SimComplexDouble));

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = false };
    SimSplitAccess  access  = { .port = 0U, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = ornstein_uhlenbeck_step,
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
                                .symbolic      = ornstein_uhlenbeck_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_ornstein_uhlenbeck_config(struct SimContext*                  context,
                                        size_t                              operator_index,
                                        SimOrnsteinUhlenbeckOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimOrnsteinUhlenbeckState* state = (SimOrnsteinUhlenbeckState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_ornstein_uhlenbeck_update(struct SimContext*                        context,
                                        size_t                                    operator_index,
                                        const SimOrnsteinUhlenbeckOperatorConfig* config) {
    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimOrnsteinUhlenbeckState* state = (SimOrnsteinUhlenbeckState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimOrnsteinUhlenbeckOperatorConfig local = *config;
    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("ornstein_uhlenbeck"), operator_index);
    }
    ornstein_uhlenbeck_normalize(&local);

    SimField* field = sim_context_field(context, local.field_index);
    if (!ornstein_uhlenbeck_validate_field(field)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config = local;
    sim_noise_source_seed(&state->rng, local.seed, local.seed ^ SIM_NOISE_SOURCE_STREAM_ORNSTEIN);
    ornstein_uhlenbeck_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
