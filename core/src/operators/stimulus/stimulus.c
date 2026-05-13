#include "oakfield/operators/stimulus/stimulus.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIMULUS_EPS 1.0e-12

typedef struct StimulusOperatorState {
    StimulusOperatorConfig config;
    char                   symbolic[128];
} StimulusOperatorState;

static void normalize(StimulusOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->wavenumber)) {
        config->wavenumber = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void refresh_symbolic(StimulusOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const StimulusOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "%.3g*sin(%.3g x - %.3g t + %.3g)",
                    cfg->amplitude,
                    cfg->wavenumber,
                    cfg->omega,
                    cfg->phase);
#else
    (void) state;
#endif
}

static const char* symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const StimulusOperatorState* state = (const StimulusOperatorState*) state_ptr;
    if (state == NULL) {
        return NULL;
    }
    return state->symbolic;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult
apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    StimulusOperatorState* state = (StimulusOperatorState*) state_ptr;
    SimField*              field;
    double*                data;
    size_t                 count;
    double                 base_time;
    double                 eval_time;
    size_t                 i;

    (void) self;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const bool is_complex = sim_field_is_complex(field);
    if (!is_complex && field->element_size != sizeof(double)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = is_complex ? sim_field_element_count(&field->layout)
                       : sim_field_bytes(field) / sizeof(double);
    if (count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    data                    = is_complex ? NULL : (double*) sim_field_data(field);
    SimComplexDouble* cdata = is_complex ? sim_field_complex_data(field) : NULL;
    if ((is_complex && cdata == NULL) || (!is_complex && data == NULL)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    base_time = sim_context_time(context);
    eval_time = base_time + state->config.time_offset;

    /* Optional dt scaling for signal vs force semantics */
    double scale      = state->config.scale_by_dt ? dt : 1.0;
    double phase_time = state->config.phase - state->config.omega * eval_time;

    const SimStimulusCoordConfig* coord = &state->config.coord;

    for (i = 0U; i < count; ++i) {
        double x = 0.0;
        double y = 0.0;
        if (sim_stimulus_coord_xy(coord, field, i, &x, &y) != SIM_RESULT_OK) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        double forcing = 0.0;
        if (coord->mode == SIM_STIMULUS_COORD_SEPARABLE) {
            double fx = sin(state->config.wavenumber * x + phase_time);
            double fy = sin(state->config.wavenumber * y + phase_time);
            if (coord->combine == SIM_STIMULUS_SEPARABLE_ADD) {
                forcing = state->config.amplitude * (fx + fy);
            } else {
                forcing = state->config.amplitude * fx * fy;
            }
        } else {
            double spatial = 0.0;
            double u       = sim_stimulus_coord_u(coord, x, y, eval_time);
            spatial        = state->config.wavenumber * u;
            forcing        = state->config.amplitude * sin(spatial + phase_time);
        }
        if (is_complex) {
            cdata[i].re += scale * forcing;
        } else {
            data[i] += scale * forcing;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult step(void*               state_ptr,
                      struct SimContext*  context,
                      struct SimOperator* self,
                      size_t              substep_index,
                      double              dt_sub,
                      void*               scratch,
                      size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_stimulus_operator(struct SimContext*            context,
                                    const StimulusOperatorConfig* config,
                                    size_t*                       out_index) {
    StimulusOperatorState* state;
    StimulusOperatorConfig local;
    char                   name[SIM_OPERATOR_NAME_MAX + 1U];
    SimResult              result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void) memset(&local, 0, sizeof(local));
    if (config != NULL) {
        local = *config;
    }

    normalize(&local);

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "stimulus_sine", (config != NULL), (config != NULL) ? config->scale_by_dt : true);

    state = (StimulusOperatorState*) calloc(1U, sizeof(StimulusOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;

    refresh_symbolic(state);

    sim_operator_make_unique_name(name, sizeof(name), "stimulus_sine");

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
    info.abstract_id       = "stimulus_sine";
    sim_operator_info_set_schema_identity(&info, "stimulus_sine");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = false };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = step,
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
                                .symbolic      = symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }
    return result;
}

SimResult sim_stimulus_config(struct SimContext*      context,
                              size_t                  operator_index,
                              StimulusOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    StimulusOperatorState* state = (StimulusOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_update(struct SimContext*            context,
                              size_t                        operator_index,
                              const StimulusOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    StimulusOperatorState* state = (StimulusOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    StimulusOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    normalize(&local);

    state->config = local;
    refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
