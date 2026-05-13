#include "oakfield/operators/utility/zero_field.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ZeroFieldOperatorState {
    ZeroFieldOperatorConfig config;
    char                    symbolic[64];
} ZeroFieldOperatorState;

static void zero_field_refresh_symbolic(ZeroFieldOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state)
        return;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "zero_field field=%zu",
                    state->config.field_index);
#else
    (void) state;
#endif
}

static const char* zero_field_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const ZeroFieldOperatorState* state = (const ZeroFieldOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult
zero_field_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;
    (void) dt;

    ZeroFieldOperatorState* state = (ZeroFieldOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (sim_field_is_complex(field)) {
        SimComplexDouble* data = sim_field_complex_data(field);
        if (!data) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        memset(data, 0, count * sizeof(*data));
    } else {
        double* data = sim_field_real_data(field);
        if (!data) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        memset(data, 0, count * sizeof(*data));
    }

    return SIM_RESULT_OK;
}

static SimResult zero_field_step(void*               state_ptr,
                                 struct SimContext*  context,
                                 struct SimOperator* self,
                                 size_t              substep_index,
                                 double              dt_sub,
                                 void*               scratch,
                                 size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return zero_field_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_zero_field_operator(struct SimContext*             context,
                                      const ZeroFieldOperatorConfig* config,
                                      size_t*                        out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ZeroFieldOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    SimField* field = sim_context_field(context, local.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ZeroFieldOperatorState* state =
        (ZeroFieldOperatorState*) calloc(1U, sizeof(ZeroFieldOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    zero_field_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "zero_field");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_UTILITY;
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
    info.abstract_id       = "zero_field";
    sim_operator_info_set_schema_identity(&info, "zero_field");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = true;

    bool needs_complex = field != NULL && sim_field_is_complex(field);
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimSplitPort ports[1] = { { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex } };

    SimSplitAccess accesses[1] = { { .port = 0, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = zero_field_step,
                                .accesses          = accesses,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = zero_field_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_zero_field_config(struct SimContext*       context,
                                size_t                   operator_index,
                                ZeroFieldOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    ZeroFieldOperatorState* state = (ZeroFieldOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_zero_field_update(struct SimContext*             context,
                                size_t                         operator_index,
                                const ZeroFieldOperatorConfig* config) {
    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    ZeroFieldOperatorState* state = (ZeroFieldOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    if (config->field_index != state->config.field_index) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config = *config;
    zero_field_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
