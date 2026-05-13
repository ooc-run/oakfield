#include "oakfield/operators/utility/copy.h"
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

typedef struct SimCopyOperatorState {
    SimCopyOperatorConfig config;
    char                  symbolic[96];
} SimCopyOperatorState;

static void copy_refresh_symbolic(SimCopyOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "copy in=%zu out=%zu accum=%s scale_by_dt=%s",
                    state->config.input_field,
                    state->config.output_field,
                    state->config.accumulate ? "true" : "false",
                    state->config.scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* copy_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimCopyOperatorState* state = (const SimCopyOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void copy_normalize_config(SimCopyOperatorConfig* config) {
    if (!config) {
        return;
    }
    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static SimResult copy_validate_fields(const SimField*              input,
                                      const SimField*              output,
                                      const SimCopyOperatorConfig* config) {
    SimScalarDomain input_domain  = sim_scalar_domain_unknown();
    SimScalarDomain output_domain = sim_scalar_domain_unknown();

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

    input_domain  = sim_scalar_domain_from_field(input);
    output_domain = sim_scalar_domain_from_field(output);
    if (!sim_scalar_domain_equal(input_domain, output_domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (sim_operator_field_domain_is_f64(input) || sim_field_is_complex(input)) {
        return SIM_RESULT_OK;
    }

    if (sim_operator_domain_is_exact_integer(input_domain)) {
        if (config != NULL && (config->accumulate || config->scale_by_dt)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
        return SIM_RESULT_OK;
    }

    return SIM_RESULT_TYPE_MISMATCH;
}

static SimResult
copy_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimCopyOperatorState* state = (SimCopyOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* input  = sim_context_field(context, state->config.input_field);
    SimField* output = sim_context_field(context, state->config.output_field);
    {
        SimResult validation = copy_validate_fields(input, output, &state->config);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    const double scale = state->config.scale_by_dt ? fmax(dt, 0.0) : 1.0;

    if (sim_field_is_complex(input)) {
        const SimComplexDouble* src = sim_field_complex_data_const(input);
        SimComplexDouble*       dst = sim_field_complex_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        sim_accel_copy_scale_complex(src, dst, count, scale, state->config.accumulate);
    } else {
        SimScalarDomain domain = sim_scalar_domain_from_field(input);
        if (sim_operator_domain_is_exact_integer(domain)) {
            const void* src   = sim_field_data_const(input);
            void*       dst   = sim_field_data(output);
            size_t      bytes = sim_field_bytes(input);
            if (!src || !dst) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            (void) memcpy(dst, src, bytes);
        } else {
            const double* src = sim_field_real_data_const(input);
            double*       dst = sim_field_real_data(output);
            if (!src || !dst) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_accel_copy_scale_real(src, dst, count, scale, state->config.accumulate);
        }
    }

    return SIM_RESULT_OK;
}

static SimResult copy_step(void*               state_ptr,
                           struct SimContext*  context,
                           struct SimOperator* self,
                           size_t              substep_index,
                           double              dt_sub,
                           void*               scratch,
                           size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return copy_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_copy_operator(struct SimContext*           context,
                                const SimCopyOperatorConfig* config,
                                size_t*                      out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimCopyOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "copy", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    copy_normalize_config(&local);

    SimField* input  = sim_context_field(context, local.input_field);
    SimField* output = sim_context_field(context, local.output_field);
    {
        SimResult validation = copy_validate_fields(input, output, &local);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    SimCopyOperatorState* state = (SimCopyOperatorState*) calloc(1U, sizeof(SimCopyOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    copy_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "copy");

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
    info.abstract_id       = "copy";
    sim_operator_info_set_schema_identity(&info, "copy");
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
                                .fn                = copy_step,
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
                                .symbolic      = copy_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_copy_config(struct SimContext*     context,
                          size_t                 operator_index,
                          SimCopyOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimCopyOperatorState* state = (SimCopyOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_copy_update(struct SimContext*           context,
                          size_t                       operator_index,
                          const SimCopyOperatorConfig* config) {
    if (!context || !config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimCopyOperatorState* state = (SimCopyOperatorState*) sim_split_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimCopyOperatorConfig local = *config;
    local.scale_by_dt           = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "copy"), true, local.scale_by_dt);
    copy_normalize_config(&local);

    SimField* input  = sim_context_field(context, local.input_field);
    SimField* output = sim_context_field(context, local.output_field);
    {
        SimResult validation = copy_validate_fields(input, output, &local);
        if (validation != SIM_RESULT_OK) {
            return validation;
        }
    }

    state->config = local;
    copy_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
