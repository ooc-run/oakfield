#include "oakfield/operators/diffusion/fractional_memory.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "sim_accel.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct FractionalMemoryOperatorState {
    FractionalMemoryOperatorConfig config;
    double*                        coefficients;
    double*                        history;
    double*                        scratch;
    double*                        history_snapshot;
    size_t                         history_snapshot_bytes;
    size_t                         snapshot_head;
    size_t                         snapshot_count;
    size_t                         capacity;
    size_t                         head;
    size_t                         count;
    char                           symbolic[160];
    size_t                         kernel_cached_count;
    size_t                         kernel_cached_step_index;
    double                         kernel_cached_dt;
    double                         response_scale_cache_dt;
    double                         response_scale_cache_value;
    const SimField*                kernel_cached_field;
    bool                           response_scale_cache_valid;
    bool                           kernel_cache_valid;
} FractionalMemoryOperatorState;

static void*  fractional_memory_alloc_zeroed(size_t bytes);
static double fractional_memory_response_scale(FractionalMemoryOperatorState* state, double dt);

static void fractional_memory_release(void* state_ptr) {
    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state->coefficients);
    free(state->history);
    free(state->scratch);
    free(state->history_snapshot);
    free(state);
}

static const char* fractional_memory_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const FractionalMemoryOperatorState* state = (const FractionalMemoryOperatorState*) state_ptr;
    if (state == NULL) {
        return NULL;
    }
    return state->symbolic;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult fractional_memory_prepare(FractionalMemoryOperatorState* state, size_t count) {
    size_t needed = state->config.memory_steps;

    if (needed == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->capacity != count) {
        size_t history_bytes = needed * count * sizeof(double);
        size_t scratch_bytes = count * sizeof(double);

        free(state->history);
        free(state->scratch);
        free(state->history_snapshot);

        state->history                = (double*) fractional_memory_alloc_zeroed(history_bytes);
        state->scratch                = (double*) fractional_memory_alloc_zeroed(scratch_bytes);
        state->history_snapshot       = NULL;
        state->history_snapshot_bytes = 0U;
        state->snapshot_head          = 0U;
        state->snapshot_count         = 0U;
        if ((history_bytes > 0U && state->history == NULL) ||
            (scratch_bytes > 0U && state->scratch == NULL)) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->capacity = count;
        state->head     = 0U;
        state->count    = 0U;
    }

    if (state->coefficients == NULL) {
        size_t j;
        state->coefficients = (double*) calloc(needed, sizeof(double));
        if (state->coefficients == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        state->coefficients[0] = 1.0;
        for (j = 1U; j < needed; ++j) {
            double numer           = ((double) j - 1.0) - state->config.order;
            double denom           = (double) j;
            state->coefficients[j] = state->coefficients[j - 1U] * (numer / denom);
        }
    }

    return SIM_RESULT_OK;
}

static void* fractional_memory_alloc_zeroed(size_t bytes) {
    if (bytes == 0U) {
        return NULL;
    }

#if defined(__APPLE__) || defined(_POSIX_VERSION)
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64U, bytes) != 0) {
        return NULL;
    }
    (void) memset(ptr, 0, bytes);
    return ptr;
#else
    return calloc(1U, bytes);
#endif
}

static double fractional_memory_response_scale(FractionalMemoryOperatorState* state, double dt) {
    if (state == NULL) {
        return 0.0;
    }

    if (state->response_scale_cache_valid && state->response_scale_cache_dt == dt) {
        return state->response_scale_cache_value;
    }

    double scale                      = dt * state->config.gain * pow(dt, -state->config.order);
    state->response_scale_cache_dt    = dt;
    state->response_scale_cache_value = scale;
    state->response_scale_cache_valid = true;
    return scale;
}

static void fractional_memory_apply_history_response(const FractionalMemoryOperatorState* state,
                                                     const double*                        current,
                                                     size_t                               count,
                                                     double  response_scale,
                                                     double* out) {
    if (state == NULL || current == NULL || out == NULL || count == 0U) {
        return;
    }

    size_t steps = state->config.memory_steps;
    if (steps == 0U || state->count == 0U) {
        if (out != current) {
            (void) memmove(out, current, count * sizeof(double));
        }
        return;
    }

    sim_accel_copy_scale_real(
        current, out, count, 1.0 + response_scale * state->coefficients[0], false);

    size_t idx = state->head;
    for (size_t j = 1U; j < state->count; ++j) {
        idx = (idx + 1U == steps) ? 0U : (idx + 1U);
        sim_accel_copy_scale_real(&state->history[idx * count],
                                  out,
                                  count,
                                  response_scale * state->coefficients[j],
                                  true);
    }
}

static SimResult
fractional_memory_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;
    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) userdata;
    if (state == NULL || state->history == NULL || state->capacity == 0U) {
        return SIM_RESULT_OK;
    }
    size_t bytes = state->config.memory_steps * state->capacity * sizeof(double);
    if (state->history_snapshot == NULL || state->history_snapshot_bytes < bytes) {
        double* next = (double*) realloc(state->history_snapshot, bytes);
        if (next == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->history_snapshot       = next;
        state->history_snapshot_bytes = bytes;
    }
    memcpy(state->history_snapshot, state->history, bytes);
    state->snapshot_head  = state->head;
    state->snapshot_count = state->count;
    return SIM_RESULT_OK;
}

static SimResult
fractional_memory_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;
    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) userdata;
    if (state == NULL || state->history_snapshot == NULL || state->history == NULL) {
        return SIM_RESULT_OK;
    }
    size_t bytes = state->config.memory_steps * state->capacity * sizeof(double);
    if (state->history_snapshot_bytes < bytes) {
        return SIM_RESULT_INVALID_STATE;
    }
    memcpy(state->history, state->history_snapshot, bytes);
    state->head  = state->snapshot_head;
    state->count = state->snapshot_count;
    return SIM_RESULT_OK;
}

static SimResult fractional_memory_apply(void*               state_ptr,
                                         struct SimContext*  context,
                                         struct SimOperator* self,
                                         double              dt) {
    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) state_ptr;
    SimField*                      field;
    double*                        data;
    size_t                         count;
    size_t                         steps;

    (void) self;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->config.memory_steps == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (dt <= 0.0) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* Treat the buffer as a flat array of doubles. For complex fields, this will
        naturally process both real and imaginary components componentwise. */
    data  = (double*) sim_field_data(field);
    count = sim_field_bytes(field) / sizeof(double);
    if (data == NULL || count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (fractional_memory_prepare(state, count) != SIM_RESULT_OK) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    steps = state->config.memory_steps;

    if (state->count < steps) {
        state->count += 1U;
    }

    if (steps > 0U) {
        state->head = (state->head == 0U) ? (steps - 1U) : (state->head - 1U);
        (void) memcpy(&state->history[state->head * count], data, count * sizeof(double));
    } else {
        state->count = 0U;
        return SIM_RESULT_OK;
    }

    fractional_memory_apply_history_response(
        state, data, count, fractional_memory_response_scale(state, dt), data);

    return SIM_RESULT_OK;
}

static SimResult fractional_memory_step(void*               state_ptr,
                                        struct SimContext*  context,
                                        struct SimOperator* self,
                                        size_t              substep_index,
                                        double              dt_sub,
                                        void*               scratch,
                                        size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return fractional_memory_apply(state_ptr, context, self, dt_sub);
}

static double fractional_memory_kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult fractional_memory_ir_eval(void*           userdata,
                                           const KernelIR* kernel,
                                           size_t          element_index,
                                           size_t          component,
                                           double*         out_value) {
    if (out_value == NULL || userdata == NULL || kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) userdata;
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
    if (field->element_size != (is_complex ? sizeof(SimComplexDouble) : sizeof(double))) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (component >= component_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t element_count = sim_field_element_count(&field->layout);
    if (element_count == 0U || element_index >= element_count) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    size_t count = sim_field_bytes(field) / sizeof(double);
    if (count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double dt = fractional_memory_kernel_param_value(kernel, SIM_IR_PARAM_DT);

    size_t step_index = 0U;
    bool   have_step =
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
                       state->kernel_cached_count != count || state->kernel_cached_dt != dt ||
                       !have_step || state->kernel_cached_step_index != step_index;

    if (need_update) {
        SimResult rc = fractional_memory_prepare(state, count);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        double* data = (double*) sim_field_data(field);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        size_t steps = state->config.memory_steps;
        if (steps == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (state->count < steps) {
            state->count += 1U;
        }

        state->head = (state->head == 0U) ? (steps - 1U) : (state->head - 1U);
        (void) memcpy(&state->history[state->head * count], data, count * sizeof(double));

        double response_scale = fractional_memory_response_scale(state, dt);
        fractional_memory_apply_history_response(
            state, data, count, response_scale, state->scratch);

        state->kernel_cached_field      = field;
        state->kernel_cached_count      = count;
        state->kernel_cached_dt         = dt;
        state->kernel_cached_step_index = step_index;
        state->kernel_cache_valid       = have_step;
    }

    size_t flat_index = element_index * component_count + component;
    if (flat_index >= count) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }
    *out_value = state->scratch[flat_index];
    return SIM_RESULT_OK;
}

SimResult sim_add_fractional_memory_operator(struct SimContext*                    context,
                                             const FractionalMemoryOperatorConfig* config,
                                             size_t*                               out_index) {
    FractionalMemoryOperatorState* state;
    FractionalMemoryOperatorConfig local;
    char                           name[SIM_OPERATOR_NAME_MAX + 1U];
    SimResult                      result = SIM_RESULT_OK;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void) memset(&local, 0, sizeof(local));
    if (config != NULL) {
        local = *config;
    }

    if (local.memory_steps == 0U) {
        local.memory_steps = 32U;
    }
    if (local.order <= 0.0) {
        local.order = 0.5;
    }

    state = (FractionalMemoryOperatorState*) calloc(1U, sizeof(FractionalMemoryOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config                     = local;
    state->capacity                   = 0U;
    state->head                       = 0U;
    state->count                      = 0U;
    state->kernel_cached_count        = 0U;
    state->kernel_cached_step_index   = 0U;
    state->kernel_cached_dt           = 0.0;
    state->response_scale_cache_dt    = 0.0;
    state->response_scale_cache_value = 0.0;
    state->kernel_cached_field        = NULL;
    state->response_scale_cache_valid = false;
    state->kernel_cache_valid         = false;

#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    (void) snprintf(
        state->symbolic, sizeof(state->symbolic), "%.3g * D_t^{%.3g} u", local.gain, local.order);
#endif

    sim_operator_make_unique_name(name, sizeof(name), "fractional_memory");

    SimField*       field         = sim_context_field(context, local.field_index);
    bool            needs_complex = field != NULL && sim_field_is_complex(field);
    SimOperatorInfo info          = sim_operator_info_defaults();
    info.category                 = SIM_OPERATOR_CATEGORY_DIFFUSION;
    info.warp_level               = SIM_WARP_LEVEL_NONE;
    info.is_noise                 = false;
    info.is_spectral              = false;
    info.is_local                 = true;
    info.is_nonlocal              = false;
    info.is_linear                = true;
    info.is_warp                  = false;
    info.is_differentiable        = true;
    info.preserves_real           = true;
    info.preferred_dt             = 0.0;
    info.abstract_id              = "fractional_memory";
    sim_operator_info_set_schema_identity(&info, "fractional_memory");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "fractional_memory")) {
        bool   is_complex    = needs_complex;
        size_t expected_size = is_complex ? sizeof(SimComplexDouble) : sizeof(double);
        if (field != NULL && field->element_size == expected_size) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRStatefulSpec spec = { 0 };
                spec.eval              = fractional_memory_ir_eval;
                spec.userdata          = state;
                spec.label             = "fractional_memory";
                spec.value_type        = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId mem_node   = sim_ir_builder_stateful_spec(builder, &spec);

                if (mem_node != SIM_IR_INVALID_NODE) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = mem_node;

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
                    kdesc.destroy               = fractional_memory_release;
                    kdesc.userdata              = state;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    kdesc.save_state            = fractional_memory_save;
                    kdesc.restore_state         = fractional_memory_restore;
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
                                .fn                = fractional_memory_step,
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
                                .symbolic      = fractional_memory_symbolic,
                                .save_state    = fractional_memory_save,
                                .restore_state = fractional_memory_restore,
                                .destroy       = fractional_memory_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        fractional_memory_release(state);
    }
    return result;
}

SimResult sim_fractional_memory_config(struct SimContext*              context,
                                       size_t                          operator_index,
                                       FractionalMemoryOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) sim_operator_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_fractional_memory_update(struct SimContext*                    context,
                                       size_t                                operator_index,
                                       const FractionalMemoryOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    FractionalMemoryOperatorState* state = (FractionalMemoryOperatorState*) sim_operator_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    FractionalMemoryOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    /* Normalize defaults similar to add path */
    if (local.memory_steps == 0U) {
        local.memory_steps = 32U;
    }
    if (local.order <= 0.0) {
        local.order = 0.5;
    }

    /* Rebuild triggers: order or memory_steps change */
    bool steps_changed = (local.memory_steps != state->config.memory_steps);
    bool order_changed = (local.order != state->config.order);

    state->config = local;
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    (void) snprintf(
        state->symbolic, sizeof(state->symbolic), "%.3g * D_t^{%.3g} u", local.gain, local.order);
#endif

    if (steps_changed) {
        free(state->history);
        free(state->scratch);
        free(state->history_snapshot);
        state->history                = NULL;
        state->scratch                = NULL;
        state->history_snapshot       = NULL;
        state->history_snapshot_bytes = 0U;
        state->snapshot_head          = 0U;
        state->snapshot_count         = 0U;
        state->capacity               = 0U;
        state->head                   = 0U;
        state->count                  = 0U;
    }
    if (steps_changed || order_changed) {
        free(state->coefficients);
        state->coefficients = NULL;
    }
    state->kernel_cached_field        = NULL;
    state->kernel_cached_count        = 0U;
    state->kernel_cached_step_index   = 0U;
    state->kernel_cached_dt           = 0.0;
    state->response_scale_cache_dt    = 0.0;
    state->response_scale_cache_value = 0.0;
    state->response_scale_cache_valid = false;
    state->kernel_cache_valid         = false;

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
