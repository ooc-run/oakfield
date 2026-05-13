#include "oakfield/operators/utility/phase_rotate.h"
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

typedef struct PhaseRotateState {
    PhaseRotateOperatorConfig   config;
    SimIRNodeId                 phase_rate_node;
    char                        symbolic[128];
    SimAccelSplitComplexScratch split_scratch;
} PhaseRotateState;

static void phase_rotate_destroy(void* state_ptr) {
    PhaseRotateState* state = (PhaseRotateState*) state_ptr;
    if (state == NULL) {
        return;
    }
    sim_accel_split_release(&state->split_scratch);
    free(state);
}

static SimResult phase_rotate_describe_field(const SimField*         field,
                                             SimFieldRepresentation* out_repr,
                                             bool*                   out_needs_complex,
                                             bool*                   out_project_real) {
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFieldRepresentation repr          = sim_field_representation(field);
    const bool             needs_complex = sim_field_is_complex(field);
    const bool             project_real =
        !needs_complex || sim_field_representation_has_spectral_real_constraint(repr);

    if (needs_complex) {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else if (field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (out_repr != NULL) {
        *out_repr = repr;
    }
    if (out_needs_complex != NULL) {
        *out_needs_complex = needs_complex;
    }
    if (out_project_real != NULL) {
        *out_project_real = project_real;
    }

    return SIM_RESULT_OK;
}

static bool phase_rotate_has_phase_term(const PhaseRotateOperatorConfig* cfg) {
    return cfg != NULL && cfg->phase_rate != 0.0;
}

static SimFieldValueKind phase_rotate_output_value_kind(SimFieldRepresentation  repr,
                                                        bool                    needs_complex,
                                                        const PhaseRotateState* state) {
    if (repr.value_kind == SIM_FIELD_VALUE_UNKNOWN) {
        return needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    }
    if (sim_field_representation_has_imag_zero_constraint(repr) &&
        phase_rotate_has_phase_term(&state->config)) {
        return SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }
    return repr.value_kind;
}

static bool
phase_rotate_set_builder_scalar(SimIRBuilder* builder, SimIRNodeId node_id, double value) {
    if (builder == NULL || node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return false;
    }
    SimIRNode* node = &builder->nodes[node_id];
    if (node->type != SIM_IR_NODE_CONSTANT ||
        node->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX) {
        return false;
    }
    node->data.constant.scalar = value;
    return true;
}

static void phase_rotate_fill_info(SimOperatorInfo*        info,
                                   SimFieldRepresentation  repr,
                                   bool                    needs_complex,
                                   bool                    project_real,
                                   const PhaseRotateState* state) {
    if (info == NULL || state == NULL) {
        return;
    }

    const SimFieldValueKind value_kind = phase_rotate_output_value_kind(repr, needs_complex, state);

    *info                   = sim_operator_info_defaults();
    info->category          = SIM_OPERATOR_CATEGORY_UTILITY;
    info->warp_level        = SIM_WARP_LEVEL_NONE;
    info->is_noise          = false;
    info->is_spectral       = (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL);
    info->is_local          = true;
    info->is_nonlocal       = false;
    info->is_linear         = true;
    info->is_warp           = false;
    info->is_differentiable = true;
    info->preserves_real    = project_real || state->config.phase_rate == 0.0;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "phase_rotate";
    sim_operator_info_set_schema_identity(info, "phase_rotate");
    info->algebraic_flags = SIM_OPERATOR_ALG_LINEAR;
    info->representation.domain =
        (repr.domain == SIM_FIELD_DOMAIN_UNKNOWN) ? SIM_FIELD_DOMAIN_PHYSICAL : repr.domain;
    info->representation.value_kind                      = value_kind;
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = info->preserves_real;
}

static void phase_rotate_refresh_symbolic(PhaseRotateState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state)
        return;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "phase_rotate rate=%.3g",
                    state->config.phase_rate);
#else
    (void) state;
#endif
}

static const char* phase_rotate_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const PhaseRotateState* state = (const PhaseRotateState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult phase_rotate_apply(void*               state_ptr,
                                    struct SimContext*  context,
                                    struct SimOperator* self,
                                    double              dt) {
    (void) self;
    PhaseRotateState* state = (PhaseRotateState*) state_ptr;
    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    if (state->config.phase_rate == 0.0)
        return SIM_RESULT_OK;

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult field_rc = phase_rotate_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }

    size_t count = sim_field_element_count(&field->layout);
    if (count == 0U)
        return SIM_RESULT_OK;

    const bool imag_zero = sim_field_representation_has_imag_zero_constraint(repr);
    if (imag_zero) {
        SimFieldRepresentation updated_repr = repr;
        updated_repr.value_kind = phase_rotate_output_value_kind(repr, needs_complex, state);
        if (updated_repr.value_kind != repr.value_kind) {
            SimResult repr_rc = sim_field_set_representation(field, updated_repr);
            if (repr_rc != SIM_RESULT_OK) {
                return repr_rc;
            }
            repr = updated_repr;
        }
    }

    const double theta = (isfinite(dt) ? dt : 0.0) * state->config.phase_rate;
    const double c     = cos(theta);

    if (project_real) {
        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_accel_scale_inplace_complex_real(cdata, count, c);
        } else {
            double* data = sim_field_real_data(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_accel_scale_inplace_real(data, count, c);
        }
        return SIM_RESULT_OK;
    }

    const double      s     = sin(theta);
    SimComplexDouble* cdata = sim_field_complex_data(field);
    if (cdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!imag_zero && sim_accel_rotate_complex(&state->split_scratch, cdata, cdata, count, c, s)) {
        return SIM_RESULT_OK;
    }

    for (size_t i = 0U; i < count; ++i) {
        double re   = cdata[i].re;
        double im   = imag_zero ? 0.0 : cdata[i].im;
        cdata[i].re = re * c - im * s;
        cdata[i].im = re * s + im * c;
    }

    return SIM_RESULT_OK;
}

static SimResult phase_rotate_step(void*               state_ptr,
                                   struct SimContext*  context,
                                   struct SimOperator* self,
                                   size_t              substep_index,
                                   double              dt_sub,
                                   void*               scratch,
                                   size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return phase_rotate_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_phase_rotate_operator(struct SimContext*               context,
                                        const PhaseRotateOperatorConfig* config,
                                        size_t*                          out_index) {
    if (!context)
        return SIM_RESULT_INVALID_ARGUMENT;

    PhaseRotateOperatorConfig local = { 0 };
    if (config)
        local = *config;

    if (!isfinite(local.phase_rate))
        local.phase_rate = 0.0;

    PhaseRotateState* state = (PhaseRotateState*) calloc(1U, sizeof(PhaseRotateState));
    if (!state)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config          = local;
    state->phase_rate_node = SIM_IR_INVALID_NODE;
    phase_rotate_refresh_symbolic(state);

    SimField* field = sim_context_field(context, local.field_index);
    if (field == NULL) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult field_rc = phase_rotate_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        free(state);
        return field_rc;
    }

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "phase_rotate");

    SimOperatorInfo info = sim_operator_info_defaults();
    phase_rotate_fill_info(&info, repr, needs_complex, project_real, state);

    SimResult result            = SIM_RESULT_OK;
    bool      registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, NULL, 0ULL, SIM_DET_NONE, "phase_rotate")) {
        if (needs_complex && !project_real &&
            !sim_field_representation_has_imag_zero_constraint(repr)) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRType complex_type = sim_ir_type_complex();

                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, complex_type);
                SimIRNodeId dt_node    = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);

                SimIRNodeId rate_node = sim_ir_builder_constant(builder, local.phase_rate);
                SimIRNodeId angle =
                    sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, dt_node, rate_node);
                SimIRNodeId rotated = sim_ir_builder_complex_rotate(builder, field_node, angle);

                if (field_node != SIM_IR_INVALID_NODE && dt_node != SIM_IR_INVALID_NODE &&
                    rate_node != SIM_IR_INVALID_NODE && angle != SIM_IR_INVALID_NODE &&
                    rotated != SIM_IR_INVALID_NODE) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = rotated;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = phase_rotate_destroy;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.read_mask             = 0ULL;
                    kdesc.write_mask            = 0ULL;
                    if (local.field_index < 64U) {
                        kdesc.read_mask |= (1ULL << local.field_index);
                        kdesc.write_mask |= (1ULL << local.field_index);
                    }

                    result = sim_context_register_operator(context, &kdesc, out_index);
                    if (result == SIM_RESULT_OK) {
                        state->phase_rate_node = rate_node;
                        registered_kernel      = true;
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitPort    port    = { .context_field_index = local.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = phase_rotate_step,
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
                                .symbolic      = phase_rotate_symbolic,
                                .destroy       = phase_rotate_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult rc = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (rc != SIM_RESULT_OK) {
        phase_rotate_destroy(state);
    }
    return rc;
}

SimResult sim_phase_rotate_config(struct SimContext*         context,
                                  size_t                     operator_index,
                                  PhaseRotateOperatorConfig* out_config) {
    if (!context || !out_config)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op)
        return SIM_RESULT_NOT_FOUND;

    PhaseRotateState* state = (PhaseRotateState*) sim_operator_state(op);
    if (!state)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_phase_rotate_update(struct SimContext*               context,
                                  size_t                           operator_index,
                                  const PhaseRotateOperatorConfig* config) {
    if (!context || !config)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op)
        return SIM_RESULT_NOT_FOUND;

    PhaseRotateState* state = (PhaseRotateState*) sim_operator_state(op);
    if (!state)
        return SIM_RESULT_INVALID_STATE;

    PhaseRotateOperatorConfig local = *config;
    if (!isfinite(local.phase_rate))
        local.phase_rate = state->config.phase_rate;

    state->config = local;
    phase_rotate_refresh_symbolic(state);

    {
        SimFieldRepresentation repr          = { 0 };
        bool                   needs_complex = false;
        bool                   project_real  = false;
        SimResult              field_rc =
            phase_rotate_describe_field(sim_context_field(context, state->config.field_index),
                                        &repr,
                                        &needs_complex,
                                        &project_real);
        if (field_rc != SIM_RESULT_OK) {
            return field_rc;
        }
        phase_rotate_fill_info(&op->info, repr, needs_complex, project_real, state);
    }

    if (op->kernel != NULL && state->phase_rate_node != SIM_IR_INVALID_NODE) {
        SimIRBuilder* builder = (SimIRBuilder*) op->kernel->kernel.builder;
        (void) phase_rotate_set_builder_scalar(
            builder, state->phase_rate_node, state->config.phase_rate);
    }

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
