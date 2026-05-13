#include "oakfield/operators/measurement/phase_feature.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define PHASE_FEATURE_SYMBOLIC_CAPACITY 128

typedef struct SimPhaseFeatureOperatorState {
    SimPhaseFeatureOperatorConfig config;
    char                          symbolic[PHASE_FEATURE_SYMBOLIC_CAPACITY];
} SimPhaseFeatureOperatorState;

static void phase_feature_normalize_config(SimPhaseFeatureOperatorConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->threshold) || config->threshold < 0.0) {
        config->threshold = 0.0;
    }
    if (!isfinite(config->exponent) || config->exponent < 0.0) {
        config->exponent = 0.0;
    }
    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static double phase_feature_weight(double base, double exponent) {
    if (base <= 0.0) {
        return 0.0;
    }

    if (exponent == 0.0) {
        return 1.0;
    }

    double w = pow(base, exponent);
    if (!isfinite(w)) {
        return 0.0;
    }
    return w;
}

static void phase_feature_refresh_symbolic(SimPhaseFeatureOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimPhaseFeatureOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "phase_feature thresh=%.3g exponent=%.3g accum=%s scale_by_dt=%s",
                    cfg->threshold,
                    cfg->exponent,
                    cfg->accumulate ? "true" : "false",
                    cfg->scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* phase_feature_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimPhaseFeatureOperatorState* state = (const SimPhaseFeatureOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult phase_feature_apply(void*               state_ptr,
                                     struct SimContext*  context,
                                     struct SimOperator* self,
                                     double              dt) {
    (void) self;

    SimPhaseFeatureOperatorState* state = (SimPhaseFeatureOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimPhaseFeatureOperatorConfig* cfg   = &state->config;
    const double                         scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    SimField* input  = sim_context_field(context, state->config.input_field);
    SimField* output = sim_context_field(context, state->config.output_field);

    if (!input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool input_complex  = sim_field_is_complex(input);
    bool output_complex = sim_field_is_complex(output);
    if (input_complex != output_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    size_t input_bytes  = sim_field_bytes(input);
    size_t output_bytes = sim_field_bytes(output);
    if (input_bytes == 0U || output_bytes == 0U) {
        return SIM_RESULT_OK;
    }

    size_t input_stride  = input->element_size;
    size_t output_stride = output->element_size;

    if (input_stride != output_stride) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (!input_complex) {
        if (input_stride != sizeof(double)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }

        size_t count = input_bytes / sizeof(double);
        if (count != output_bytes / sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        double* src = (double*) sim_field_data(input);
        double* dst = (double*) sim_field_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double sample    = src[i];
            double magnitude = fabs(sample);
            if (magnitude <= cfg->threshold) {
                if (!cfg->accumulate) {
                    dst[i] = 0.0;
                }
                continue;
            }

            double base   = magnitude - cfg->threshold;
            double weight = phase_feature_weight(base, cfg->exponent);
            if (weight == 0.0) {
                if (!cfg->accumulate) {
                    dst[i] = 0.0;
                }
                continue;
            }

            double phase = (sample >= 0.0) ? 1.0 : -1.0;
            double value = phase * weight;
            if (!isfinite(value)) {
                if (!state->config.accumulate) {
                    dst[i] = 0.0;
                }
                continue;
            }

            if (cfg->accumulate) {
                dst[i] += scale * value;
            } else {
                dst[i] = value;
            }
        }

        return SIM_RESULT_OK;
    }

    if (input_stride != sizeof(double) * 2U) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    size_t count = input_bytes / (sizeof(double) * 2U);
    if (count != output_bytes / (sizeof(double) * 2U)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimComplexDouble* src = sim_field_complex_data(input);
    SimComplexDouble* dst = sim_field_complex_data(output);
    if (!src || !dst) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < count; ++i) {
        double re        = src[i].re;
        double im        = src[i].im;
        double magnitude = hypot(re, im);
        if (magnitude <= cfg->threshold || magnitude <= 0.0) {
            if (!cfg->accumulate) {
                dst[i].re = 0.0;
                dst[i].im = 0.0;
            }
            continue;
        }

        double base   = magnitude - cfg->threshold;
        double weight = phase_feature_weight(base, cfg->exponent);
        if (weight == 0.0) {
            if (!cfg->accumulate) {
                dst[i].re = 0.0;
                dst[i].im = 0.0;
            }
            continue;
        }

        double unit_re    = re / magnitude;
        double unit_im    = im / magnitude;
        double feature_re = unit_re * weight;
        double feature_im = unit_im * weight;

        if (!isfinite(feature_re) || !isfinite(feature_im)) {
            if (!state->config.accumulate) {
                dst[i].re = 0.0;
                dst[i].im = 0.0;
            }
            continue;
        }

        if (cfg->accumulate) {
            dst[i].re += scale * feature_re;
            dst[i].im += scale * feature_im;
        } else {
            dst[i].re = feature_re;
            dst[i].im = feature_im;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult phase_feature_step(void*               state_ptr,
                                    struct SimContext*  context,
                                    struct SimOperator* self,
                                    size_t              substep_index,
                                    double              dt_sub,
                                    void*               scratch,
                                    size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return phase_feature_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_phase_feature_operator(struct SimContext*                   context,
                                         const SimPhaseFeatureOperatorConfig* config,
                                         size_t*                              out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimPhaseFeatureOperatorState* state =
        (SimPhaseFeatureOperatorState*) calloc(1U, sizeof(SimPhaseFeatureOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimPhaseFeatureOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.input_field  = 0U;
        local.output_field = 0U;
        local.threshold    = 0.0;
        local.exponent     = 0.0;
        local.accumulate   = false;
        local.scale_by_dt  = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "phase_feature", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    phase_feature_normalize_config(&local);
    state->config = local;
    phase_feature_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "phase_feature");

    SimField*       input_field   = sim_context_field(context, state->config.input_field);
    SimField*       output_field  = sim_context_field(context, state->config.output_field);
    bool            needs_complex = (input_field != NULL && sim_field_is_complex(input_field)) ||
                                    (output_field != NULL && sim_field_is_complex(output_field));
    SimOperatorInfo info          = sim_operator_info_defaults();
    info.category                 = SIM_OPERATOR_CATEGORY_MEASUREMENT;
    info.warp_level               = SIM_WARP_LEVEL_NONE;
    info.is_noise                 = false;
    info.is_spectral              = false;
    info.is_local                 = true;
    info.is_nonlocal              = false;
    info.is_linear                = false;
    info.is_warp                  = false;
    info.is_differentiable        = false;
    info.preserves_real           = true;
    info.preferred_dt             = 0.0;
    info.abstract_id              = "phase_feature";
    sim_operator_info_set_schema_identity(&info, "phase_feature");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimResult result            = SIM_RESULT_OK;
    bool      registered_kernel = false;
    if (sim_operator_should_register_kernel_for_schema(
            context, NULL, 0ULL, SIM_DET_NONE, "phase_feature")) {
        SimIRBuilder* builder = sim_context_ir_builder(context);
        if (builder != NULL && state->config.exponent > 0.0) {
            if (input_field != NULL && output_field != NULL && !sim_field_is_complex(input_field) &&
                !sim_field_is_complex(output_field)) {
                SimOperatorKernelBindingDescriptor bindings[2];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRType   scalar_type = sim_ir_type_scalar();
                SimIRNodeId input_node  = sim_ir_builder_field_ref_typed(builder, 0U, scalar_type);
                SimIRNodeId output_node = sim_ir_builder_field_ref_typed(builder, 1U, scalar_type);

                if (input_node != SIM_IR_INVALID_NODE && output_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId abs_input =
                        sim_ir_builder_call(builder, SIM_IR_CALL_ABS, input_node);
                    SimIRNodeId threshold = sim_ir_builder_constant_typed(
                        builder, state->config.threshold, scalar_type);
                    SimIRNodeId base =
                        sim_ir_builder_binary(builder, SIM_IR_NODE_SUB, abs_input, threshold);
                    SimIRNodeId abs_base = sim_ir_builder_call(builder, SIM_IR_CALL_ABS, base);
                    SimIRNodeId base_sum =
                        sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, base, abs_base);
                    SimIRNodeId half = sim_ir_builder_constant_typed(builder, 0.5, scalar_type);
                    SimIRNodeId base_pos =
                        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, base_sum, half);
                    SimIRNodeId exponent =
                        sim_ir_builder_constant_typed(builder, state->config.exponent, scalar_type);
                    SimIRNodeId weight = sim_ir_builder_pow(builder, base_pos, exponent);
                    SimIRNodeId sign   = sim_ir_builder_call(builder, SIM_IR_CALL_SIGN, input_node);
                    SimIRNodeId value =
                        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, weight, sign);

                    if (value != SIM_IR_INVALID_NODE) {
                        SimIRNodeId root = value;
                        if (state->config.accumulate) {
                            SimIRNodeId scaled = value;
                            if (state->config.scale_by_dt) {
                                SimIRNodeId dt_node =
                                    sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
                                scaled =
                                    sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, value, dt_node);
                            }
                            root = sim_ir_builder_binary(
                                builder, SIM_IR_NODE_ADD, output_node, scaled);
                        }

                        if (root != SIM_IR_INVALID_NODE) {
                            bindings[0].ir_field_index      = 0U;
                            bindings[0].context_field_index = state->config.input_field;
                            bindings[1].ir_field_index      = 1U;
                            bindings[1].context_field_index = state->config.output_field;

                            outputs[0].ir_field_index = 1U;
                            outputs[0].expression     = root;

                            kernel_desc.builder           = builder;
                            kernel_desc.bindings          = bindings;
                            kernel_desc.binding_count     = 2U;
                            kernel_desc.outputs           = outputs;
                            kernel_desc.output_count      = 1U;
                            kernel_desc.required_features = 0ULL;

                            SimOperatorDescriptor kdesc = { 0 };
                            kdesc.name                  = name;
                            kdesc.evaluate              = NULL;
                            kdesc.destroy               = free;
                            kdesc.userdata              = state;
                            kdesc.kernel                = &kernel_desc;
                            kdesc.info                  = info;
                            /* Populate hazard masks for scheduler */
                            kdesc.read_mask  = 0ULL;
                            kdesc.write_mask = 0ULL;
                            if (state->config.input_field < 64U)
                                kdesc.read_mask |= (1ULL << state->config.input_field);
                            if (state->config.output_field < 64U)
                                kdesc.write_mask |= (1ULL << state->config.output_field);

                            result = sim_context_register_operator(context, &kdesc, out_index);
                            if (result == SIM_RESULT_OK) {
                                registered_kernel = true;
                            }
                        }
                    }
                }
            }
        }
    }

    if (!registered_kernel) {
        SimSplitPort ports[2] = {
            { .context_field_index = state->config.input_field, .require_complex = needs_complex },
            { .context_field_index = state->config.output_field, .require_complex = needs_complex }
        };

        SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                       { .port = 1, .mode = SIM_ACCESS_RW } };

        SimSplitSubstep substep = { .name              = NULL,
                                    .fn                = phase_feature_step,
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
                                    .symbolic      = phase_feature_symbolic,
                                    .destroy       = free,
                                    .info          = info,
                                    .scratch       = { 0U, 0U } };

        result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    }
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_phase_feature_config(struct SimContext*             context,
                                   size_t                         operator_index,
                                   SimPhaseFeatureOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimPhaseFeatureOperatorState* state = (SimPhaseFeatureOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_phase_feature_update(struct SimContext*                   context,
                                   size_t                               operator_index,
                                   const SimPhaseFeatureOperatorConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimPhaseFeatureOperatorState* state = (SimPhaseFeatureOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimPhaseFeatureOperatorConfig local = state->config;
    if (config) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "phase_feature", true, config->scale_by_dt);
    }

    phase_feature_normalize_config(&local);
    state->config = local;
    phase_feature_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
