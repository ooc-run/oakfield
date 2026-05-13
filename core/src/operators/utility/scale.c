#include "oakfield/operators/utility/scale.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct SimScaleOperatorState {
    SimScaleOperatorConfig config;
    char                   symbolic[96];
} SimScaleOperatorState;

static void scale_refresh_symbolic(SimScaleOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "scale in=%zu out=%zu scale=%.4g accum=%s scale_by_dt=%s",
                    state->config.input_field,
                    state->config.output_field,
                    state->config.scale,
                    state->config.accumulate ? "true" : "false",
                    state->config.scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* scale_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimScaleOperatorState* state = (const SimScaleOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void scale_normalize_config(SimScaleOperatorConfig* config) {
    if (!config) {
        return;
    }
    if (!isfinite(config->scale)) {
        config->scale = 0.0;
    }
    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static SimResult scale_validate_fields(const SimField* input, const SimField* output) {
    if (!input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U || sim_field_element_count(&output->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input->element_size != output->element_size) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (!(sim_operator_field_domain_is_f64(input) && sim_operator_field_domain_is_f64(output)) &&
        !(sim_field_is_complex(input) && sim_field_is_complex(output))) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    return SIM_RESULT_OK;
}

static SimResult
scale_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimScaleOperatorState* state = (SimScaleOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* input  = sim_context_field(context, state->config.input_field);
    SimField* output = sim_context_field(context, state->config.output_field);
    {
        SimResult validation = scale_validate_fields(input, output);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    const double dt_scale = state->config.scale_by_dt ? fmax(dt, 0.0) : 1.0;
    const double scale    = state->config.scale * dt_scale;

    if (sim_field_is_complex(input)) {
        const SimComplexDouble* src = sim_field_complex_data_const(input);
        SimComplexDouble*       dst = sim_field_complex_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        sim_accel_copy_scale_complex(src, dst, count, scale, state->config.accumulate);
    } else {
        const double* src = sim_field_real_data_const(input);
        double*       dst = sim_field_real_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        sim_accel_copy_scale_real(src, dst, count, scale, state->config.accumulate);
    }

    return SIM_RESULT_OK;
}

static SimResult scale_step(void*               state_ptr,
                            struct SimContext*  context,
                            struct SimOperator* self,
                            size_t              substep_index,
                            double              dt_sub,
                            void*               scratch,
                            size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return scale_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_scale_operator(struct SimContext*            context,
                                 const SimScaleOperatorConfig* config,
                                 size_t*                       out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimScaleOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "scale", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    scale_normalize_config(&local);

    SimField* input  = sim_context_field(context, local.input_field);
    SimField* output = sim_context_field(context, local.output_field);
    {
        SimResult validation = scale_validate_fields(input, output);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    SimScaleOperatorState* state =
        (SimScaleOperatorState*) calloc(1U, sizeof(SimScaleOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    scale_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "scale");

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
    info.abstract_id       = "scale";
    sim_operator_info_set_schema_identity(&info, "scale");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;

    bool needs_complex = sim_field_is_complex(input) || sim_field_is_complex(output);
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimSplitPort ports[2] = {
        { .context_field_index = state->config.input_field, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = scale_step,
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
                                .symbolic      = scale_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_scale_config(struct SimContext*      context,
                           size_t                  operator_index,
                           SimScaleOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimScaleOperatorState* state = (SimScaleOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_scale_update(struct SimContext*            context,
                           size_t                        operator_index,
                           const SimScaleOperatorConfig* config) {
    if (!context || !config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimScaleOperatorState* state = (SimScaleOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimScaleOperatorConfig local = *config;
    local.scale_by_dt            = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "scale"), true, local.scale_by_dt);
    scale_normalize_config(&local);

    SimField* input  = sim_context_field(context, local.input_field);
    SimField* output = sim_context_field(context, local.output_field);
    {
        SimResult validation = scale_validate_fields(input, output);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    state->config = local;
    scale_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
