#include "oakfield/operators/noise/stochastic_noise.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_noise_source.h"
#include "oakfield/sim_seed.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/backend.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct StochasticNoiseOperatorState {
    StochasticNoiseOperatorConfig config;
    double*                       noise_state;
    SimComplexDouble*             noise_state_c;
    size_t                        capacity;
    SimNoiseSourceRng             rng;
    char                          symbolic[128];
} StochasticNoiseOperatorState;

static void stochastic_noise_release(void* state_ptr) {
    StochasticNoiseOperatorState* state = (StochasticNoiseOperatorState*) state_ptr;
    if (!state)
        return;
    free(state->noise_state);
    free(state->noise_state_c);
    free(state);
}

static const char* stochastic_noise_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const StochasticNoiseOperatorState* state = (const StochasticNoiseOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult stochastic_noise_ensure_capacity(StochasticNoiseOperatorState* state,
                                                  size_t                        count,
                                                  bool                          is_complex) {
    return sim_noise_source_ensure_capacity(
        &state->noise_state, &state->noise_state_c, &state->capacity, count, is_complex);
}

/* Core apply: fractional Ornstein–Uhlenbeck process */
static SimResult stochastic_noise_apply(void*               state_ptr,
                                        struct SimContext*  context,
                                        struct SimOperator* self,
                                        double              dt) {
    (void) self;
    StochasticNoiseOperatorState* state = (StochasticNoiseOperatorState*) state_ptr;

    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;
    if (state->config.sigma == 0.0)
        return SIM_RESULT_OK;
    if (dt < 0.0)
        return SIM_RESULT_INVALID_ARGUMENT;
    if (dt == 0.0)
        return SIM_RESULT_OK;

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    /* If field is complex (or later becomes complex), handle complex noise */
    bool is_complex = sim_field_is_complex(field);

    size_t count;
    if (is_complex) {
        count = sim_field_element_count(&field->layout);
    } else {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    }
    if (count == 0U)
        return SIM_RESULT_OK;

    if (stochastic_noise_ensure_capacity(state, count, is_complex) != SIM_RESULT_OK)
        return SIM_RESULT_OUT_OF_MEMORY;
    if ((is_complex && state->noise_state_c == NULL) ||
        (!is_complex && state->noise_state == NULL)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    const double sigma    = state->config.sigma;
    const double sqrt_dt  = sqrt(dt);
    double       decay    = 1.0;
    double       variance = 0.0;

    /*
     * Itô interpretation: dX = σ dW(t) → ΔX = σ √dt ξ
     * Stratonovich interpretation: dX ∘ σ dW(t) → requires correction term
     *
     * For additive noise, both interpretations are equivalent, but we scale differently:
     * - Itô: noise term scales as σ √dt
     * - Stratonovich: noise term includes the state evolution (already in noise_state)
     */
    const bool use_ito = (state->config.law == SIM_IR_NOISE_LAW_ITO);
    sim_noise_source_temporal_params(
        sigma, state->config.tau, state->config.alpha, dt, &decay, &variance);

    if (is_complex) {
        SimComplexDouble* cdata = sim_field_complex_data(field);
        if (!cdata)
            return SIM_RESULT_INVALID_ARGUMENT;
        if (use_ito) {
            double gain = sim_noise_source_gaussian_gain(sigma, dt, true);
            sim_noise_source_apply_white_complex(cdata, count, 0.0, 0.0, gain, gain, &state->rng);
        } else {
            sim_noise_source_apply_temporal_complex(
                cdata, state->noise_state_c, count, sqrt_dt, decay, variance, &state->rng);
        }
    } else {
        double* data = (double*) sim_field_data(field);
        if (!data)
            return SIM_RESULT_INVALID_ARGUMENT;
        if (use_ito) {
            double gain = sim_noise_source_gaussian_gain(sigma, dt, true);
            sim_noise_source_apply_white_real(data, count, 0.0, gain, &state->rng);
        } else {
            sim_noise_source_apply_temporal_real(
                data, state->noise_state, count, sqrt_dt, decay, variance, &state->rng);
        }
    }

    return SIM_RESULT_OK;
}

/* --------------------------------------------------------- */
/*               Registration & Initialization               */
/* --------------------------------------------------------- */
static SimResult stochastic_noise_step(void*               state_ptr,
                                       struct SimContext*  context,
                                       struct SimOperator* self,
                                       size_t              substep_index,
                                       double              dt_sub,
                                       void*               scratch,
                                       size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    StochasticNoiseOperatorState* state = (StochasticNoiseOperatorState*) state_ptr;
    return stochastic_noise_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_stochastic_noise_operator(struct SimContext*                   context,
                                            const StochasticNoiseOperatorConfig* config,
                                            size_t*                              out_index) {
    if (!context)
        return SIM_RESULT_INVALID_ARGUMENT;

    StochasticNoiseOperatorState* state =
        (StochasticNoiseOperatorState*) calloc(1U, sizeof(StochasticNoiseOperatorState));
    if (!state)
        return SIM_RESULT_OUT_OF_MEMORY;

    StochasticNoiseOperatorConfig local = { 0 };
    if (config)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stochastic_noise"),
                                     sim_context_operator_count(context));
    }

    if (local.alpha <= 0.0)
        local.alpha = 1.0; /* default to OU process */

    /* Default to Itô interpretation if not specified */
    if (local.law != SIM_IR_NOISE_LAW_ITO && local.law != SIM_IR_NOISE_LAW_STRATONOVICH)
        local.law = SIM_IR_NOISE_LAW_ITO;

    state->config = local;
    sim_noise_source_seed(&state->rng, local.seed, local.seed ^ SIM_NOISE_SOURCE_STREAM_STOCHASTIC);

#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const char* law_str = (local.law == SIM_IR_NOISE_LAW_ITO) ? "Itô" : "Strat";
    snprintf(state->symbolic,
             sizeof(state->symbolic),
             "σ=%.3g τ=%.3g α=%.2f %s",
             local.sigma,
             local.tau,
             local.alpha,
             law_str);
#endif

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stochastic_noise");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_DIFFUSION;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = true;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stochastic_noise";
    sim_operator_info_set_schema_identity(&info, "stochastic_noise");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    SimField* field            = sim_context_field(context, local.field_index);
    bool      needs_complex    = (field != NULL) ? sim_field_is_complex(field) : false;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimResult result            = SIM_RESULT_OK;
    bool      registered_kernel = false;

    SimBackend* backend = sim_context_backend(context);
    if (backend != NULL && backend->type == SIM_BACKEND_TYPE_CPU &&
        local.law == SIM_IR_NOISE_LAW_ITO &&
        sim_operator_should_register_kernel_for_schema(
            context, NULL, 0ULL, SIM_DET_NONE, "stochastic_noise")) {
        if (field != NULL && !sim_field_is_complex(field) &&
            field->element_size == sizeof(double)) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRType   scalar_type = sim_ir_type_scalar();
                SimIRNodeId field_node  = sim_ir_builder_field_ref_typed(builder, 0U, scalar_type);

                double sigma = local.sigma;

                if (!isfinite(sigma)) {
                    sigma = 0.0;
                }

                double amplitude = fabs(sigma);

                SimIRNoiseSpec noise_spec;
                noise_spec.seed         = (uint32_t) sim_seed_mix64(local.seed);
                noise_spec.amplitude    = amplitude;
                noise_spec.variance     = amplitude * amplitude;
                noise_spec.law          = SIM_IR_NOISE_LAW_ITO;
                noise_spec.distribution = SIM_IR_NOISE_DISTRIBUTION_GAUSSIAN;
                noise_spec.value_type   = scalar_type;

                SimIRNodeId noise_node = sim_ir_builder_noise_spec(builder, &noise_spec);
                SimIRNodeId sqrt_dt    = sim_ir_builder_param(builder, SIM_IR_PARAM_SQRT_DT);
                SimIRNodeId scaled_noise =
                    sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, noise_node, sqrt_dt);
                SimIRNodeId sum =
                    sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, field_node, scaled_noise);

                if (field_node != SIM_IR_INVALID_NODE && noise_node != SIM_IR_INVALID_NODE &&
                    sqrt_dt != SIM_IR_INVALID_NODE && scaled_noise != SIM_IR_INVALID_NODE &&
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
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_SQRT_DT + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = stochastic_noise_release;
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
                                .fn                = stochastic_noise_step,
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
                                .symbolic      = stochastic_noise_symbolic,
                                .destroy       = stochastic_noise_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK)
        stochastic_noise_release(state);

    return result;
}

SimResult sim_stochastic_noise_config(struct SimContext*             context,
                                      size_t                         operator_index,
                                      StochasticNoiseOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    StochasticNoiseOperatorState* state = NULL;
    if (op->kernel != NULL) {
        state = (StochasticNoiseOperatorState*) op->userdata;
    } else {
        state = (StochasticNoiseOperatorState*) sim_operator_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stochastic_noise_update(struct SimContext*                   context,
                                      size_t                               operator_index,
                                      const StochasticNoiseOperatorConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    StochasticNoiseOperatorState* state = NULL;
    if (op->kernel != NULL) {
        state = (StochasticNoiseOperatorState*) op->userdata;
    } else {
        state = (StochasticNoiseOperatorState*) sim_operator_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    StochasticNoiseOperatorConfig local = state->config;
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stochastic_noise"), operator_index);
    }
    if (local.alpha <= 0.0) {
        local.alpha = 1.0;
    }
    if (local.law != SIM_IR_NOISE_LAW_ITO && local.law != SIM_IR_NOISE_LAW_STRATONOVICH) {
        local.law = SIM_IR_NOISE_LAW_ITO;
    }

    state->config = local;

    sim_noise_source_seed(&state->rng, local.seed, local.seed ^ SIM_NOISE_SOURCE_STREAM_STOCHASTIC);

    if (state->noise_state != NULL && state->capacity > 0U) {
        memset(state->noise_state, 0, state->capacity * sizeof(double));
    }
    if (state->noise_state_c != NULL && state->capacity > 0U) {
        memset(state->noise_state_c, 0, state->capacity * sizeof(SimComplexDouble));
    }

    const char* law_str = (local.law == SIM_IR_NOISE_LAW_ITO) ? "Itô" : "Strat";
    snprintf(state->symbolic,
             sizeof(state->symbolic),
             "σ=%.3g τ=%.3g α=%.2f %s",
             local.sigma,
             local.tau,
             local.alpha,
             law_str);

    return SIM_RESULT_OK;
}
