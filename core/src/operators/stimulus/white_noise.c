#include "oakfield/operators/stimulus/white_noise.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_noise_source.h"
#include "oakfield/sim_seed.h"
#include "oakfield/backend.h"
#include "oakfield/operator_identity.h"

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_WHITE_NOISE_EPS 1.0e-12

typedef struct SimStimulusWhiteNoiseState {
    SimStimulusWhiteNoiseConfig config;
    SimNoiseSourceRng           rng;
    SimClockMode                clock_mode;
    double                      locked_dt;
    bool                        clock_initialized;
    bool                        ir_gain_cache_valid;
    double                      ir_gain_cache_dt;
    double                      ir_gain_cache_value;
    char                        symbolic[128];
} SimStimulusWhiteNoiseState;

static void white_noise_normalize(SimStimulusWhiteNoiseConfig* config) {
    if (config == NULL) {
        return;
    }
    if (!isfinite(config->sigma) || config->sigma < 0.0) {
        config->sigma = 0.0;
    }
    if (!isfinite(config->mean)) {
        config->mean = 0.0;
    }
    if (!isfinite(config->nominal_dt) || config->nominal_dt < 0.0) {
        config->nominal_dt = 0.0;
    }
    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }
}

static SimClockMode white_noise_resolve_clock_mode(const SimContext*                  context,
                                                   const char*                        op_name,
                                                   const SimStimulusWhiteNoiseConfig* config) {
    if (config == NULL) {
        return SIM_CLOCK_FROM_TIME_PARAM;
    }

    bool         forced = false;
    SimClockMode requested =
        config->fixed_clock ? SIM_CLOCK_ACCUMULATED_STATEFUL : SIM_CLOCK_FROM_TIME_PARAM;
    SimClockMode resolved = sim_operator_choose_time_mode(
        context, NULL, requested, config->nominal_dt, STIM_WHITE_NOISE_EPS, &forced);
    if (forced) {
        sim_context_log_warning(context,
                                "%s: forcing pure time mode under strict representation.",
                                op_name ? op_name : "stimulus_white_noise");
    }
    return resolved;
}

static void white_noise_refresh_symbolic(SimStimulusWhiteNoiseState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    const SimStimulusWhiteNoiseConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "white_noise sigma=%.3g mean=%.3g",
                    cfg->sigma,
                    cfg->mean);
#else
    (void) state;
#endif
}

static const char* white_noise_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusWhiteNoiseState* state = (const SimStimulusWhiteNoiseState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void white_noise_destroy(void* state_ptr) {
    SimStimulusWhiteNoiseState* state = (SimStimulusWhiteNoiseState*) state_ptr;
    free(state);
}

static double white_noise_effective_dt(SimStimulusWhiteNoiseState* state, double dt) {
    switch (state->clock_mode) {
        case SIM_CLOCK_FROM_TIME_PARAM:
            return dt;
        case SIM_CLOCK_FROM_STEP_PURE: {
            double nominal = state->config.nominal_dt;
            return (nominal > STIM_WHITE_NOISE_EPS) ? nominal : dt;
        }
        case SIM_CLOCK_ACCUMULATED_STATEFUL:
        default:
            break;
    }

    if (!state->config.fixed_clock) {
        return dt;
    }

    double nominal = state->config.nominal_dt;
    if (nominal <= STIM_WHITE_NOISE_EPS) {
        nominal = dt;
    }

    if (!state->clock_initialized) {
        state->locked_dt         = nominal;
        state->clock_initialized = true;
    }
    return state->locked_dt;
}

static double kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static double white_noise_ir_gain(SimStimulusWhiteNoiseState* state, double dt) {
    if (state->ir_gain_cache_valid && state->ir_gain_cache_dt == dt) {
        return state->ir_gain_cache_value;
    }

    SimStimulusWhiteNoiseConfig* cfg          = &state->config;
    double                       effective_dt = white_noise_effective_dt(state, dt);
    if (effective_dt < STIM_WHITE_NOISE_EPS) {
        effective_dt = dt;
    }

    double gain = sim_noise_source_gaussian_gain(cfg->sigma, effective_dt, cfg->scale_by_dt);

    state->ir_gain_cache_dt    = dt;
    state->ir_gain_cache_value = gain;
    state->ir_gain_cache_valid = true;
    return gain;
}

static SimResult white_noise_ir_eval(void*           userdata,
                                     const KernelIR* kernel,
                                     size_t          element_index,
                                     size_t          component,
                                     double*         out_value) {
    (void) element_index;
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusWhiteNoiseState*  state = (SimStimulusWhiteNoiseState*) userdata;
    SimStimulusWhiteNoiseConfig* cfg   = &state->config;
    if (cfg->sigma == 0.0 && cfg->mean == 0.0) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    double dt    = kernel_param_value(kernel, SIM_IR_PARAM_DT);
    double gain  = white_noise_ir_gain(state, dt);
    double n     = sim_noise_source_normal(&state->rng);
    double value = cfg->mean + gain * n;
    *out_value   = value;
    return SIM_RESULT_OK;
}

static SimResult white_noise_step(void*               state_ptr,
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

    SimStimulusWhiteNoiseState* state = (SimStimulusWhiteNoiseState*) state_ptr;
    if (state == NULL || context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusWhiteNoiseConfig* cfg = &state->config;
    if (cfg->sigma == 0.0 && cfg->mean == 0.0) {
        return SIM_RESULT_OK;
    }

    SimField* field = sim_context_field(context, cfg->field_index);
    if (field == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    bool is_complex = sim_field_is_complex(field);

    size_t count = 0U;
    if (is_complex) {
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    } else {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    }
    if (count == 0U)
        return SIM_RESULT_OK;

    double effective_dt = white_noise_effective_dt(state, dt_sub);
    if (effective_dt < STIM_WHITE_NOISE_EPS) {
        effective_dt = dt_sub;
    }
    double gain = sim_noise_source_gaussian_gain(cfg->sigma, effective_dt, cfg->scale_by_dt);

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL)
            return SIM_RESULT_INVALID_ARGUMENT;
        sim_noise_source_apply_white_real(dst, count, cfg->mean, gain, &state->rng);
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL)
            return SIM_RESULT_INVALID_ARGUMENT;
        sim_noise_source_apply_white_complex(dst, count, cfg->mean, 0.0, gain, gain, &state->rng);
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_white_noise_operator(struct SimContext*                 context,
                                                const SimStimulusWhiteNoiseConfig* config,
                                                size_t*                            out_index) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusWhiteNoiseConfig local = { 0 };
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_white_noise"),
                                     sim_context_operator_count(context));
    }

    white_noise_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_white_noise",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusWhiteNoiseState* state =
        (SimStimulusWhiteNoiseState*) calloc(1U, sizeof(SimStimulusWhiteNoiseState));
    if (state == NULL)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config = local;
    state->clock_mode =
        white_noise_resolve_clock_mode(context, "stimulus_white_noise", &state->config);
    sim_noise_source_seed(
        &state->rng, local.seed, local.seed ^ SIM_NOISE_SOURCE_STREAM_STIMULUS_WHITE);
    state->clock_initialized   = false;
    state->locked_dt           = 0.0;
    state->ir_gain_cache_valid = false;
    state->ir_gain_cache_dt    = 0.0;
    state->ir_gain_cache_value = 0.0;
    white_noise_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_white_noise");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_NOISE;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = true;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stimulus_white_noise";
    sim_operator_info_set_schema_identity(&info, "stimulus_white_noise");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR | SIM_OPERATOR_ALG_COMMUTES_WITH_NOISE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    bool needs_complex =
        sim_field_is_complex(sim_context_field(context, state->config.field_index));

    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = white_noise_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_white_noise")) {
        SimField* field = sim_context_field(context, local.field_index);
        if (field != NULL) {
            bool is_complex = sim_field_is_complex(field);
            if ((!is_complex && field->element_size != sizeof(double)) ||
                (is_complex && field->element_size != sizeof(SimComplexDouble))) {
                field = NULL;
            }
        }
        if (field != NULL) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                bool        is_complex = sim_field_is_complex(field);
                SimIRType   field_type = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId noise_node = sim_ir_builder_stateful(
                    builder, white_noise_ir_eval, state, "stimulus_white_noise");
                if (is_complex && noise_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId noise_im = sim_ir_builder_stateful(
                        builder, white_noise_ir_eval, state, "stimulus_white_noise_im");
                    if (noise_im != SIM_IR_INVALID_NODE) {
                        noise_node = sim_ir_builder_complex_pack(builder, noise_node, noise_im);
                    }
                }
                SimIRNodeId sum =
                    sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, field_node, noise_node);

                if (field_node != SIM_IR_INVALID_NODE && noise_node != SIM_IR_INVALID_NODE &&
                    sum != SIM_IR_INVALID_NODE) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = sum;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 1U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_DT + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = white_noise_destroy;
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

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = white_noise_symbolic,
                                .destroy       = white_noise_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        white_noise_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_white_noise_config(struct SimContext*           context,
                                          size_t                       operator_index,
                                          SimStimulusWhiteNoiseConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusWhiteNoiseState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusWhiteNoiseState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusWhiteNoiseState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_white_noise_update(struct SimContext*                 context,
                                          size_t                             operator_index,
                                          const SimStimulusWhiteNoiseConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusWhiteNoiseState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusWhiteNoiseState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusWhiteNoiseState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    SimStimulusWhiteNoiseConfig local = state->config;
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_white_noise"), operator_index);
    }

    white_noise_normalize(&local);
    state->config     = local;
    state->clock_mode = white_noise_resolve_clock_mode(
        context, sim_operator_schema_key_or(op, "stimulus_white_noise"), &state->config);
    sim_noise_source_seed(
        &state->rng, local.seed, local.seed ^ SIM_NOISE_SOURCE_STREAM_STIMULUS_WHITE);
    state->clock_initialized   = false;
    state->locked_dt           = 0.0;
    state->ir_gain_cache_valid = false;
    state->ir_gain_cache_dt    = 0.0;
    state->ir_gain_cache_value = 0.0;
    white_noise_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
