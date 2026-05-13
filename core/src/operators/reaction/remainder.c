#include "oakfield/operators/reaction/remainder.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define REMAINDER_SYMBOLIC_CAPACITY 160
#define REMAINDER_EPSILON_MIN 1.0e-9
#define REMAINDER_EXPONENT_MIN 1.0e-3

typedef struct SimRemainderOperatorState {
    SimRemainderOperatorConfig config;
    char                       symbolic[REMAINDER_SYMBOLIC_CAPACITY];
} SimRemainderOperatorState;

static const char* remainder_mode_name(SimRemainderNonlinearity mode) {
    switch (mode) {
        case SIM_REMAINDER_NONLINEARITY_ABS:
            return "abs";
        case SIM_REMAINDER_NONLINEARITY_LOG_ABS:
            return "log_abs";
        case SIM_REMAINDER_NONLINEARITY_POWER:
            return "power";
        case SIM_REMAINDER_NONLINEARITY_TANH:
            return "tanh";
        case SIM_REMAINDER_NONLINEARITY_IDENTITY:
        default:
            return "identity";
    }
}

static void remainder_normalize_config(SimRemainderOperatorConfig* config) {
    if (!config) {
        return;
    }

    if (!isfinite(config->weight)) {
        config->weight = 1.0;
    }
    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }
    if (!isfinite(config->exponent)) {
        config->exponent = 1.0;
    }
    if (config->exponent < REMAINDER_EXPONENT_MIN) {
        config->exponent = 1.0;
    }
    if (!isfinite(config->epsilon) || config->epsilon < REMAINDER_EPSILON_MIN) {
        config->epsilon = REMAINDER_EPSILON_MIN;
    }

    switch (config->nonlinearity) {
        case SIM_REMAINDER_NONLINEARITY_IDENTITY:
        case SIM_REMAINDER_NONLINEARITY_ABS:
        case SIM_REMAINDER_NONLINEARITY_LOG_ABS:
        case SIM_REMAINDER_NONLINEARITY_POWER:
        case SIM_REMAINDER_NONLINEARITY_TANH:
            break;
        default:
            config->nonlinearity = SIM_REMAINDER_NONLINEARITY_IDENTITY;
            break;
    }

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
    switch (config->complex_mode) {
        case SIM_REMAINDER_COMPLEX_MODE_COMPONENT:
        case SIM_REMAINDER_COMPLEX_MODE_POLAR:
            break;
        default:
            config->complex_mode = SIM_REMAINDER_COMPLEX_MODE_COMPONENT;
            break;
    }
}

static void remainder_refresh_symbolic(SimRemainderOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    const SimRemainderOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "remainder mode=%s weight=%.3g bias=%.3g accum=%s scale_by_dt=%s",
                    remainder_mode_name(cfg->nonlinearity),
                    cfg->weight,
                    cfg->bias,
                    cfg->accumulate ? "true" : "false",
                    cfg->scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static double
remainder_eval(SimRemainderNonlinearity mode, double value, double exponent, double epsilon) {
    double magnitude;

    switch (mode) {
        case SIM_REMAINDER_NONLINEARITY_ABS:
            return fabs(value);
        case SIM_REMAINDER_NONLINEARITY_LOG_ABS:
            magnitude = fabs(value);
            return log1p(magnitude / epsilon);
        case SIM_REMAINDER_NONLINEARITY_POWER:
            magnitude = pow(fabs(value) + epsilon, exponent);
            return copysign(magnitude, value);
        case SIM_REMAINDER_NONLINEARITY_TANH:
            return tanh(value);
        case SIM_REMAINDER_NONLINEARITY_IDENTITY:
        default:
            return value;
    }
}

static SimIRNodeId remainder_ir_constant(SimIRBuilder* builder, double value, SimIRType type) {
    if (builder == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    if (sim_scalar_domain_is_complex(sim_ir_type_scalar_domain(type))) {
        return sim_ir_builder_constant_complex(builder, value, 0.0);
    }

    if (sim_ir_type_is_scalar(type)) {
        return sim_ir_builder_constant_typed(builder, value, type);
    }

    size_t components = (type.components == 0U) ? 1U : type.components;
    if (components > SIM_IR_SMALL_CONSTANT_CAPACITY) {
        return SIM_IR_INVALID_NODE;
    }

    double values[SIM_IR_SMALL_CONSTANT_CAPACITY] = { 0.0 };
    for (size_t i = 0U; i < components; ++i) {
        values[i] = value;
    }
    return sim_ir_builder_constant_vector_typed(builder, values, components, type);
}

static SimIRNodeId remainder_ir_eval(SimIRBuilder*            builder,
                                     SimIRNodeId              value,
                                     SimRemainderNonlinearity mode,
                                     double                   exponent,
                                     double                   epsilon,
                                     SimIRType                type) {
    if (builder == NULL || value == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId abs_value = SIM_IR_INVALID_NODE;
    SimIRNodeId result    = SIM_IR_INVALID_NODE;

    switch (mode) {
        case SIM_REMAINDER_NONLINEARITY_IDENTITY:
            result = value;
            break;
        case SIM_REMAINDER_NONLINEARITY_ABS:
            result = sim_ir_builder_call(builder, SIM_IR_CALL_ABS, value);
            break;
        case SIM_REMAINDER_NONLINEARITY_LOG_ABS: {
            abs_value            = sim_ir_builder_call(builder, SIM_IR_CALL_ABS, value);
            SimIRNodeId eps_node = remainder_ir_constant(builder, epsilon, type);
            SimIRNodeId one_node = remainder_ir_constant(builder, 1.0, type);
            SimIRNodeId ratio =
                sim_ir_builder_binary(builder, SIM_IR_NODE_DIV, abs_value, eps_node);
            SimIRNodeId sum = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, one_node, ratio);
            if (sum != SIM_IR_INVALID_NODE) {
                result = sim_ir_builder_call(builder, SIM_IR_CALL_LOG, sum);
            }
            break;
        }
        case SIM_REMAINDER_NONLINEARITY_POWER: {
            abs_value            = sim_ir_builder_call(builder, SIM_IR_CALL_ABS, value);
            SimIRNodeId eps_node = remainder_ir_constant(builder, epsilon, type);
            SimIRNodeId base = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, abs_value, eps_node);
            SimIRNodeId exp_node = remainder_ir_constant(builder, exponent, type);
            SimIRNodeId mag      = sim_ir_builder_pow(builder, base, exp_node);
            SimIRNodeId sign     = sim_ir_builder_call(builder, SIM_IR_CALL_SIGN, value);
            result               = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, mag, sign);
            break;
        }
        case SIM_REMAINDER_NONLINEARITY_TANH:
            result = sim_ir_builder_call(builder, SIM_IR_CALL_TANH, value);
            break;
        default:
            result = SIM_IR_INVALID_NODE;
            break;
    }

    return result;
}

static const char* remainder_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimRemainderOperatorState* state = (const SimRemainderOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult
remainder_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimRemainderOperatorState* state = (SimRemainderOperatorState*) state_ptr;
    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimRemainderOperatorConfig* cfg   = &state->config;
    const double                      scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    SimField* warped    = sim_context_field(context, state->config.warped_field);
    SimField* reference = sim_context_field(context, state->config.reference_field);
    SimField* output    = sim_context_field(context, state->config.output_field);

    if (!warped || !reference || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_operator_field_domain_is_f64_or_c64(warped) ||
        !sim_operator_field_domain_is_f64_or_c64(reference) ||
        !sim_operator_field_domain_is_f64_or_c64(output)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    bool warped_complex    = (warped->element_size == sizeof(SimComplexDouble));
    bool reference_complex = (reference->element_size == sizeof(SimComplexDouble));
    bool output_complex    = (output->element_size == sizeof(SimComplexDouble));

    if (!(warped->element_size == sizeof(double) || warped_complex)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    /* Reference must match warped representation so we can compare samples. */
    if (reference_complex != warped_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    /* Allow writing into complex outputs even when inputs are real; otherwise enforce matching types. */
    if (warped_complex) {
        if (!output_complex) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else {
        if (!(output->element_size == sizeof(double) || output_complex)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    }

    void* warped_data_raw    = sim_field_data(warped);
    void* reference_data_raw = sim_field_data(reference);
    void* output_data_raw    = sim_field_data(output);

    if (!warped_data_raw || !reference_data_raw || !output_data_raw) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex      = warped_complex;
    size_t warped_count    = sim_field_bytes(warped) / warped->element_size;
    size_t reference_count = sim_field_bytes(reference) / reference->element_size;
    size_t output_count    = sim_field_bytes(output) / output->element_size;

    if (warped_count == 0U || reference_count == 0U || output_count == 0U) {
        return SIM_RESULT_OK;
    }

    if (warped_count != reference_count || warped_count != output_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!is_complex && !output_complex) {
        double* warped_data    = (double*) warped_data_raw;
        double* reference_data = (double*) reference_data_raw;
        double* output_data    = (double*) output_data_raw;

        for (size_t i = 0U; i < warped_count; ++i) {
            double warped_sample    = warped_data[i];
            double reference_sample = reference_data[i];

            double warped_eval =
                remainder_eval(cfg->nonlinearity, warped_sample, cfg->exponent, cfg->epsilon);
            double reference_eval =
                remainder_eval(cfg->nonlinearity, reference_sample, cfg->exponent, cfg->epsilon);

            double residue = (warped_eval - reference_eval) * cfg->weight + cfg->bias;
            if (!isfinite(residue)) {
                continue;
            }

            if (cfg->accumulate) {
                output_data[i] += scale * residue;
            } else {
                output_data[i] = residue;
            }
        }
    } else if (!is_complex && output_complex) {
        double*           warped_data    = (double*) warped_data_raw;
        double*           reference_data = (double*) reference_data_raw;
        SimComplexDouble* output_data    = sim_field_complex_data(output);
        if (!output_data) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < warped_count; ++i) {
            double warped_sample    = warped_data[i];
            double reference_sample = reference_data[i];

            double warped_eval =
                remainder_eval(cfg->nonlinearity, warped_sample, cfg->exponent, cfg->epsilon);
            double reference_eval =
                remainder_eval(cfg->nonlinearity, reference_sample, cfg->exponent, cfg->epsilon);

            double residue = (warped_eval - reference_eval) * cfg->weight + cfg->bias;
            if (!isfinite(residue)) {
                continue;
            }

            if (cfg->accumulate) {
                output_data[i].re += scale * residue;
            } else {
                output_data[i].re = residue;
                output_data[i].im = 0.0;
            }
        }
    } else {
        double complex* warped_z    = (double complex*) warped_data_raw;
        double complex* reference_z = (double complex*) reference_data_raw;
        double complex* output_z    = (double complex*) output_data_raw;

        if (cfg->complex_mode == SIM_REMAINDER_COMPLEX_MODE_POLAR) {
            /* Polar mode: compute scalar residue from magnitudes; write along warped phase. */
            for (size_t i = 0U; i < warped_count; ++i) {
                double rw      = cabs(warped_z[i]);
                double rr      = cabs(reference_z[i]);
                double ew      = remainder_eval(cfg->nonlinearity, rw, cfg->exponent, cfg->epsilon);
                double er      = remainder_eval(cfg->nonlinearity, rr, cfg->exponent, cfg->epsilon);
                double residue = (ew - er) * cfg->weight + cfg->bias;
                if (!isfinite(residue)) {
                    continue;
                }
                double complex unit  = (rw > 0.0) ? (warped_z[i] / rw) : 1.0;
                double complex value = residue * unit;
                if (cfg->accumulate) {
                    output_z[i] += scale * value;
                } else {
                    output_z[i] = value;
                }
            }
        } else {
            for (size_t i = 0U; i < warped_count; ++i) {
                double wr = creal(warped_z[i]);
                double wi = cimag(warped_z[i]);
                double rr = creal(reference_z[i]);
                double ri = cimag(reference_z[i]);

                double wr_eval = remainder_eval(cfg->nonlinearity, wr, cfg->exponent, cfg->epsilon);
                double wi_eval = remainder_eval(cfg->nonlinearity, wi, cfg->exponent, cfg->epsilon);
                double rr_eval = remainder_eval(cfg->nonlinearity, rr, cfg->exponent, cfg->epsilon);
                double ri_eval = remainder_eval(cfg->nonlinearity, ri, cfg->exponent, cfg->epsilon);

                double res_r = (wr_eval - rr_eval) * cfg->weight + cfg->bias;
                double res_i = (wi_eval - ri_eval) * cfg->weight + cfg->bias;

                double new_r = creal(output_z[i]);
                double new_i = cimag(output_z[i]);

                if (isfinite(res_r)) {
                    if (cfg->accumulate) {
                        new_r += scale * res_r;
                    } else {
                        new_r = res_r;
                    }
                }

                if (isfinite(res_i)) {
                    if (cfg->accumulate) {
                        new_i += scale * res_i;
                    } else {
                        new_i = res_i;
                    }
                }

                output_z[i] = new_r + I * new_i;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult remainder_step(void*               state_ptr,
                                struct SimContext*  context,
                                struct SimOperator* self,
                                size_t              substep_index,
                                double              dt_sub,
                                void*               scratch,
                                size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return remainder_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_remainder_operator(struct SimContext*                context,
                                     const SimRemainderOperatorConfig* config,
                                     size_t*                           out_index) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimRemainderOperatorState* state =
        (SimRemainderOperatorState*) calloc(1U, sizeof(SimRemainderOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimRemainderOperatorConfig local = { 0 };
    if (config) {
        local = *config;
    } else {
        local.warped_field    = 0U;
        local.reference_field = 0U;
        local.output_field    = 0U;
        local.weight          = 1.0;
        local.bias            = 0.0;
        local.exponent        = 1.0;
        local.epsilon         = REMAINDER_EPSILON_MIN;
        local.nonlinearity    = SIM_REMAINDER_NONLINEARITY_IDENTITY;
        local.accumulate      = false;
        local.scale_by_dt     = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "remainder", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    remainder_normalize_config(&local);
    state->config = local;
    remainder_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "remainder");

    SimField* warped_field    = sim_context_field(context, state->config.warped_field);
    SimField* reference_field = sim_context_field(context, state->config.reference_field);
    SimField* output_field    = sim_context_field(context, state->config.output_field);
    if (warped_field == NULL || reference_field == NULL || output_field == NULL) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_operator_field_domain_is_f64_or_c64(warped_field) ||
        !sim_operator_field_domain_is_f64_or_c64(reference_field) ||
        !sim_operator_field_domain_is_f64_or_c64(output_field)) {
        free(state);
        return SIM_RESULT_TYPE_MISMATCH;
    }
    bool warped_complex      = warped_field != NULL && sim_field_is_complex(warped_field);
    bool reference_complex   = reference_field != NULL && sim_field_is_complex(reference_field);
    bool output_complex      = output_field != NULL && sim_field_is_complex(output_field);
    bool needs_complex_input = warped_complex || reference_complex;
    bool needs_complex_representation = needs_complex_input || output_complex;
    SimOperatorInfo info              = sim_operator_info_defaults();
    info.category                     = SIM_OPERATOR_CATEGORY_REACTION;
    info.warp_level                   = SIM_WARP_LEVEL_NONE;
    info.is_noise                     = false;
    info.is_spectral                  = false;
    info.is_local                     = true;
    info.is_nonlocal                  = false;
    info.is_linear                    = false;
    info.is_warp                      = false;
    info.is_differentiable            = false;
    info.preserves_real               = true;
    info.preferred_dt                 = 0.0;
    info.abstract_id                  = "remainder";
    sim_operator_info_set_schema_identity(&info, "remainder");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex_representation ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex_input;
    info.representation.requires_complex_representation = needs_complex_representation;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimResult result            = SIM_RESULT_OK;
    bool      registered_kernel = false;
    if (sim_operator_should_register_kernel_for_schema(
            context, NULL, 0ULL, SIM_DET_NONE, "remainder")) {
        SimIRBuilder* builder = sim_context_ir_builder(context);
        if (builder != NULL) {
            if (warped_field != NULL && reference_field != NULL && output_field != NULL) {
                bool mode_supported = true;

                if (warped_complex != reference_complex) {
                    mode_supported = false;
                }
                if (warped_complex && !output_complex) {
                    mode_supported = false;
                }
                if (warped_complex &&
                    state->config.complex_mode == SIM_REMAINDER_COMPLEX_MODE_POLAR) {
                    mode_supported = false;
                }

                SimIRType eval_type = sim_ir_type_scalar();
                SimIRType out_type;
                bool      vector_mode = false;
                if (mode_supported && warped_complex) {
                    eval_type   = sim_ir_type_vector(2U);
                    vector_mode = true;
                }
                if (mode_supported && !warped_complex && output_complex) {
                    out_type = sim_ir_type_complex();
                } else {
                    out_type = eval_type;
                }

                if (vector_mode && state->config.accumulate && state->config.scale_by_dt) {
                    mode_supported = false;
                }

                if (mode_supported) {
                    SimOperatorKernelBindingDescriptor bindings[3];
                    SimOperatorKernelOutputDescriptor  outputs[1];
                    SimOperatorKernelDescriptor        kernel_desc = { 0 };

                    SimIRNodeId warped_node =
                        sim_ir_builder_field_ref_typed(builder, 0U, eval_type);
                    SimIRNodeId reference_node =
                        sim_ir_builder_field_ref_typed(builder, 1U, eval_type);
                    SimIRNodeId output_node = sim_ir_builder_field_ref_typed(builder, 2U, out_type);

                    if (warped_node != SIM_IR_INVALID_NODE &&
                        reference_node != SIM_IR_INVALID_NODE &&
                        output_node != SIM_IR_INVALID_NODE) {
                        SimIRNodeId warped_eval    = remainder_ir_eval(builder,
                                                                       warped_node,
                                                                       state->config.nonlinearity,
                                                                       state->config.exponent,
                                                                       state->config.epsilon,
                                                                       eval_type);
                        SimIRNodeId reference_eval = remainder_ir_eval(builder,
                                                                       reference_node,
                                                                       state->config.nonlinearity,
                                                                       state->config.exponent,
                                                                       state->config.epsilon,
                                                                       eval_type);

                        if (warped_eval != SIM_IR_INVALID_NODE &&
                            reference_eval != SIM_IR_INVALID_NODE) {
                            SimIRNodeId diff = sim_ir_builder_binary(
                                builder, SIM_IR_NODE_SUB, warped_eval, reference_eval);
                            SimIRNodeId weight =
                                remainder_ir_constant(builder, state->config.weight, eval_type);
                            SimIRNodeId bias =
                                remainder_ir_constant(builder, state->config.bias, eval_type);
                            SimIRNodeId scaled =
                                sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, diff, weight);
                            SimIRNodeId residue =
                                sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, scaled, bias);

                            if (residue != SIM_IR_INVALID_NODE) {
                                SimIRNodeId root = residue;

                                if (!warped_complex && output_complex) {
                                    SimIRNodeId imag_zero = sim_ir_builder_constant_typed(
                                        builder, 0.0, sim_ir_type_scalar());
                                    SimIRNodeId packed         = SIM_IR_INVALID_NODE;
                                    SimIRNodeId scaled_residue = residue;
                                    root                       = SIM_IR_INVALID_NODE;
                                    if (state->config.accumulate && state->config.scale_by_dt) {
                                        SimIRNodeId dt_node =
                                            sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
                                        scaled_residue = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_MUL, residue, dt_node);
                                    }
                                    packed = sim_ir_builder_complex_pack(
                                        builder, scaled_residue, imag_zero);
                                    if (packed != SIM_IR_INVALID_NODE) {
                                        root = packed;
                                        if (state->config.accumulate) {
                                            root = sim_ir_builder_binary(
                                                builder, SIM_IR_NODE_ADD, output_node, packed);
                                        }
                                    }
                                } else if (state->config.accumulate) {
                                    SimIRNodeId scaled_residue = residue;
                                    if (state->config.scale_by_dt) {
                                        SimIRNodeId dt_node =
                                            sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
                                        scaled_residue = sim_ir_builder_binary(
                                            builder, SIM_IR_NODE_MUL, residue, dt_node);
                                    }
                                    root = sim_ir_builder_binary(
                                        builder, SIM_IR_NODE_ADD, output_node, scaled_residue);
                                }

                                if (root != SIM_IR_INVALID_NODE) {
                                    bindings[0].ir_field_index      = 0U;
                                    bindings[0].context_field_index = state->config.warped_field;
                                    bindings[1].ir_field_index      = 1U;
                                    bindings[1].context_field_index = state->config.reference_field;
                                    bindings[2].ir_field_index      = 2U;
                                    bindings[2].context_field_index = state->config.output_field;

                                    outputs[0].ir_field_index = 2U;
                                    outputs[0].expression     = root;

                                    kernel_desc.builder           = builder;
                                    kernel_desc.bindings          = bindings;
                                    kernel_desc.binding_count     = 3U;
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
                                    if (state->config.warped_field < 64U)
                                        kdesc.read_mask |= (1ULL << state->config.warped_field);
                                    if (state->config.reference_field < 64U)
                                        kdesc.read_mask |= (1ULL << state->config.reference_field);
                                    if (state->config.output_field < 64U)
                                        kdesc.write_mask |= (1ULL << state->config.output_field);

                                    result =
                                        sim_context_register_operator(context, &kdesc, out_index);
                                    if (result == SIM_RESULT_OK) {
                                        registered_kernel = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (!registered_kernel) {
        SimSplitPort ports[3] = { { .context_field_index = state->config.warped_field,
                                    .require_complex     = warped_complex },
                                  { .context_field_index = state->config.reference_field,
                                    .require_complex     = reference_complex },
                                  { .context_field_index = state->config.output_field,
                                    .require_complex     = output_complex } };

        SimSplitAccess accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                       { .port = 1, .mode = SIM_ACCESS_READ },
                                       { .port = 2, .mode = SIM_ACCESS_RW } };

        SimSplitSubstep substep = { .name              = NULL,
                                    .fn                = remainder_step,
                                    .accesses          = accesses,
                                    .access_count      = 3U,
                                    .dt_scale          = 1.0,
                                    .barrier_after     = false,
                                    .error_measure     = NULL,
                                    .required_features = 0U };

        SimSplitDescriptor desc = { .name          = name,
                                    .ports         = ports,
                                    .port_count    = 3U,
                                    .substeps      = &substep,
                                    .substep_count = 1U,
                                    .state         = state,
                                    .symbolic      = remainder_symbolic,
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

SimResult sim_remainder_config(struct SimContext*          context,
                               size_t                      operator_index,
                               SimRemainderOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimRemainderOperatorState* state = (SimRemainderOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_remainder_update(struct SimContext*                context,
                               size_t                            operator_index,
                               const SimRemainderOperatorConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimRemainderOperatorState* state = (SimRemainderOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimRemainderOperatorConfig local = state->config;
    if (config) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "remainder", true, config->scale_by_dt);
    }

    remainder_normalize_config(&local);
    state->config = local;
    remainder_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
