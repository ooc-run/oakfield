#include "oakfield/operators/coupling/metal_mix.h"
#include "operators/common/operator_utils.h"
#include "sim_accel.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define METAL_MIX_SYMBOLIC_CAPACITY 160
#define METAL_MIX_ACCEL_MIN_LEN 64U

typedef struct SimMetalMixKernelNodes {
    SimIRNodeId lhs_gain;
    SimIRNodeId rhs_gain;
    SimIRNodeId mix;
    SimIRNodeId one_minus_mix;
    SimIRNodeId bias;
    SimIRNodeId crossfade_flag;
    SimIRNodeId accumulate_flag;
    SimIRNodeId dt_scale_flag;
    bool        valid;
} SimMetalMixKernelNodes;

typedef struct SimMetalMixOperatorState {
    SimMetalMixOperatorConfig config;
    SimMetalMixKernelNodes    kernel_nodes;
    char                      symbolic[METAL_MIX_SYMBOLIC_CAPACITY];
} SimMetalMixOperatorState;

static inline bool metal_mix_mode_supported(SimMixerMode mode) {
    return mode == SIM_MIXER_MODE_LINEAR || mode == SIM_MIXER_MODE_CROSSFADE;
}

const char* metal_mix_mode_name(SimMixerMode mode) {
    if (!metal_mix_mode_supported(mode)) {
        mode = SIM_MIXER_MODE_LINEAR;
    }
    return mixer_mode_name(mode);
}

bool metal_mix_mode_from_name(const char* name, SimMixerMode* out_mode) {
    SimMixerMode parsed;
    if (!mixer_mode_from_name(name, &parsed)) {
        return false;
    }
    if (!metal_mix_mode_supported(parsed)) {
        return false;
    }
    if (out_mode != NULL) {
        *out_mode = parsed;
    }
    return true;
}

static void metal_mix_normalize_config(SimMetalMixOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->lhs_gain)) {
        config->lhs_gain = 1.0;
    }
    if (!isfinite(config->rhs_gain)) {
        config->rhs_gain = 1.0;
    }
    if (!isfinite(config->mix)) {
        config->mix = 0.5;
    }
    if (config->mix < 0.0) {
        config->mix = 0.0;
    } else if (config->mix > 1.0) {
        config->mix = 1.0;
    }
    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }
    if (!metal_mix_mode_supported(config->mode)) {
        config->mode = SIM_MIXER_MODE_LINEAR;
    }
    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static void metal_mix_refresh_symbolic(SimMetalMixOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "metal_mix mode=%s lhs=%.3g rhs=%.3g mix=%.3g bias=%.3g acc=%s dt_scale=%s",
                    metal_mix_mode_name(state->config.mode),
                    state->config.lhs_gain,
                    state->config.rhs_gain,
                    state->config.mix,
                    state->config.bias,
                    state->config.accumulate ? "true" : "false",
                    state->config.scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* metal_mix_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimMetalMixOperatorState* state = (const SimMetalMixOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static inline bool metal_mix_sample_finite(double complex value) {
    return isfinite(creal(value)) && isfinite(cimag(value));
}

static void metal_mix_apply_complex(const SimMetalMixOperatorConfig* cfg,
                                    double                           dt,
                                    const SimComplexDouble*          lhs_c,
                                    const double*                    lhs_r,
                                    const SimComplexDouble*          rhs_c,
                                    const double*                    rhs_r,
                                    SimComplexDouble*                out,
                                    size_t                           count) {
    const double add_scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;
    const double lhs_coef  = (cfg->mode == SIM_MIXER_MODE_CROSSFADE)
                                 ? ((1.0 - cfg->mix) * cfg->lhs_gain)
                                 : cfg->lhs_gain;
    const double rhs_coef =
        (cfg->mode == SIM_MIXER_MODE_CROSSFADE) ? (cfg->mix * cfg->rhs_gain) : cfg->rhs_gain;

    if (count >= METAL_MIX_ACCEL_MIN_LEN) {
        SimAccelSplitComplexScratch scratch = { 0 };
        if (sim_accel_affine_mix_complex(&scratch,
                                         lhs_c,
                                         lhs_r,
                                         rhs_c,
                                         rhs_r,
                                         out,
                                         count,
                                         lhs_coef,
                                         rhs_coef,
                                         cfg->bias,
                                         cfg->accumulate,
                                         add_scale)) {
            sim_accel_split_release(&scratch);
            return;
        }
        sim_accel_split_release(&scratch);
    }

    for (size_t i = 0U; i < count; ++i) {
        double complex lhs =
            cfg->lhs_gain * (lhs_c ? (lhs_c[i].re + I * lhs_c[i].im) : (lhs_r ? lhs_r[i] : 0.0));
        double complex rhs =
            cfg->rhs_gain * (rhs_c ? (rhs_c[i].re + I * rhs_c[i].im) : (rhs_r ? rhs_r[i] : 0.0));
        double complex mixed = lhs + rhs;
        if (cfg->mode == SIM_MIXER_MODE_CROSSFADE) {
            mixed = (1.0 - cfg->mix) * lhs + cfg->mix * rhs;
        }
        mixed += cfg->bias;
        if (!metal_mix_sample_finite(mixed)) {
            continue;
        }
        if (cfg->accumulate) {
            out[i].re += add_scale * creal(mixed);
            out[i].im += add_scale * cimag(mixed);
        } else {
            out[i].re = creal(mixed);
            out[i].im = cimag(mixed);
        }
    }
}

static void metal_mix_apply_real(const SimMetalMixOperatorConfig* cfg,
                                 double                           dt,
                                 const double*                    lhs,
                                 const double*                    rhs,
                                 double*                          out,
                                 size_t                           count) {
    const double add_scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;
    const double lhs_coef  = (cfg->mode == SIM_MIXER_MODE_CROSSFADE)
                                 ? ((1.0 - cfg->mix) * cfg->lhs_gain)
                                 : cfg->lhs_gain;
    const double rhs_coef =
        (cfg->mode == SIM_MIXER_MODE_CROSSFADE) ? (cfg->mix * cfg->rhs_gain) : cfg->rhs_gain;

    if (count >= METAL_MIX_ACCEL_MIN_LEN &&
        sim_accel_affine_mix_real(
            lhs, rhs, out, count, lhs_coef, rhs_coef, cfg->bias, cfg->accumulate, add_scale)) {
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double left  = cfg->lhs_gain * lhs[i];
        double right = cfg->rhs_gain * rhs[i];
        double mixed = left + right;
        if (cfg->mode == SIM_MIXER_MODE_CROSSFADE) {
            mixed = (1.0 - cfg->mix) * left + cfg->mix * right;
        }
        mixed += cfg->bias;
        if (!isfinite(mixed)) {
            continue;
        }
        if (cfg->accumulate) {
            out[i] += add_scale * mixed;
        } else {
            out[i] = mixed;
        }
    }
}

static SimResult
metal_mix_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimMetalMixOperatorState* state = (SimMetalMixOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField* out_field = sim_context_field(context, state->config.output_field);
    if (lhs_field == NULL || rhs_field == NULL || out_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const size_t lhs_count = sim_field_element_count(&lhs_field->layout);
    const size_t rhs_count = sim_field_element_count(&rhs_field->layout);
    const size_t out_count = sim_field_element_count(&out_field->layout);
    if (lhs_count != rhs_count || lhs_count != out_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const bool lhs_complex = sim_field_is_complex(lhs_field);
    const bool rhs_complex = sim_field_is_complex(rhs_field);
    const bool out_complex = sim_field_is_complex(out_field);
    const bool any_complex = lhs_complex || rhs_complex || out_complex;

    if (any_complex) {
        if (!out_complex) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
        SimComplexDouble*       out = sim_field_complex_data(out_field);
        const SimComplexDouble* lhs_c =
            lhs_complex ? sim_field_complex_data_const(lhs_field) : NULL;
        const SimComplexDouble* rhs_c =
            rhs_complex ? sim_field_complex_data_const(rhs_field) : NULL;
        const double* lhs_r =
            (!lhs_complex) ? (const double*) sim_field_data_const(lhs_field) : NULL;
        const double* rhs_r =
            (!rhs_complex) ? (const double*) sim_field_data_const(rhs_field) : NULL;
        if (out == NULL || (!lhs_complex && lhs_r == NULL) || (!rhs_complex && rhs_r == NULL)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        metal_mix_apply_complex(&state->config, dt, lhs_c, lhs_r, rhs_c, rhs_r, out, out_count);
        return SIM_RESULT_OK;
    }

    const size_t lhs_scalar_count = sim_field_bytes(lhs_field) / sizeof(double);
    const size_t rhs_scalar_count = sim_field_bytes(rhs_field) / sizeof(double);
    const size_t out_scalar_count = sim_field_bytes(out_field) / sizeof(double);
    if (lhs_scalar_count != rhs_scalar_count || lhs_scalar_count != out_scalar_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double* lhs = sim_field_data(lhs_field);
    double* rhs = sim_field_data(rhs_field);
    double* out = sim_field_data(out_field);
    if (lhs == NULL || rhs == NULL || out == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    metal_mix_apply_real(&state->config, dt, lhs, rhs, out, lhs_scalar_count);
    return SIM_RESULT_OK;
}

static SimResult metal_mix_step(void*               state_ptr,
                                struct SimContext*  context,
                                struct SimOperator* self,
                                size_t              substep_index,
                                double              dt_sub,
                                void*               scratch,
                                size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return metal_mix_apply(state_ptr, context, self, dt_sub);
}

static SimIRNodeId
metal_mix_broadcast_constant(SimIRBuilder* builder, SimIRType type, double value) {
    size_t components = type.components;
    if (components <= 1U) {
        return sim_ir_builder_constant_typed(builder, value, type);
    }

    double values[SIM_IR_SMALL_CONSTANT_CAPACITY] = { 0.0 };
    if (components > SIM_IR_SMALL_CONSTANT_CAPACITY) {
        return SIM_IR_INVALID_NODE;
    }
    for (size_t i = 0U; i < components; ++i) {
        values[i] = value;
    }
    return sim_ir_builder_constant_vector_typed(builder, values, components, type);
}

static SimIRNodeId metal_mix_bias_constant(SimIRBuilder* builder, SimIRType type, double value) {
    size_t components = type.components;
    if (components <= 1U) {
        return sim_ir_builder_constant_typed(builder, value, type);
    }

    if (components > SIM_IR_SMALL_CONSTANT_CAPACITY) {
        return SIM_IR_INVALID_NODE;
    }

    double values[SIM_IR_SMALL_CONSTANT_CAPACITY] = { 0.0 };
    values[0]                                     = value;
    return sim_ir_builder_constant_vector_typed(builder, values, components, type);
}

static void metal_mix_reset_kernel_nodes(SimMetalMixKernelNodes* nodes) {
    if (nodes == NULL) {
        return;
    }
    nodes->lhs_gain        = SIM_IR_INVALID_NODE;
    nodes->rhs_gain        = SIM_IR_INVALID_NODE;
    nodes->mix             = SIM_IR_INVALID_NODE;
    nodes->one_minus_mix   = SIM_IR_INVALID_NODE;
    nodes->bias            = SIM_IR_INVALID_NODE;
    nodes->crossfade_flag  = SIM_IR_INVALID_NODE;
    nodes->accumulate_flag = SIM_IR_INVALID_NODE;
    nodes->dt_scale_flag   = SIM_IR_INVALID_NODE;
    nodes->valid           = false;
}

static bool
metal_mix_set_builder_constant(SimIRBuilder* builder, SimIRNodeId node_id, double value) {
    if (builder == NULL || node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return false;
    }

    SimIRNode* node = &builder->nodes[node_id];
    if (node->type != SIM_IR_NODE_CONSTANT) {
        return false;
    }

    const SimIRType value_type = node->value_type;
    if (node->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX) {
        if (value_type.components > 1U && value_type.components <= SIM_IR_SMALL_CONSTANT_CAPACITY) {
            for (size_t i = 0U; i < value_type.components; ++i) {
                node->data.constant.small[i] = value;
            }
            return true;
        }
        node->data.constant.scalar = value;
        return true;
    }

    size_t pool_index = node->data.constant.constant_index;
    if (builder->constants_data == NULL || pool_index >= builder->constants_count ||
        builder->constants_components == NULL || builder->constants_offsets == NULL) {
        return false;
    }
    size_t components = builder->constants_components[pool_index];
    size_t offset     = builder->constants_offsets[pool_index];
    for (size_t i = 0U; i < components; ++i) {
        builder->constants_data[offset + i] = value;
    }
    return true;
}

static bool
metal_mix_set_builder_bias_constant(SimIRBuilder* builder, SimIRNodeId node_id, double value) {
    if (builder == NULL || node_id == SIM_IR_INVALID_NODE || node_id >= builder->count) {
        return false;
    }

    SimIRNode* node = &builder->nodes[node_id];
    if (node->type != SIM_IR_NODE_CONSTANT) {
        return false;
    }

    const SimIRType value_type = node->value_type;
    if (value_type.components <= 1U) {
        if (node->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX) {
            node->data.constant.scalar = value;
            return true;
        }

        size_t pool_index = node->data.constant.constant_index;
        if (builder->constants_data == NULL || pool_index >= builder->constants_count ||
            builder->constants_components == NULL || builder->constants_offsets == NULL) {
            return false;
        }
        size_t components = builder->constants_components[pool_index];
        size_t offset     = builder->constants_offsets[pool_index];
        if (components == 0U) {
            return false;
        }
        builder->constants_data[offset] = value;
        for (size_t i = 1U; i < components; ++i) {
            builder->constants_data[offset + i] = 0.0;
        }
        return true;
    }

    if (node->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX) {
        if (value_type.components > SIM_IR_SMALL_CONSTANT_CAPACITY) {
            return false;
        }
        node->data.constant.small[0] = value;
        for (size_t i = 1U; i < value_type.components; ++i) {
            node->data.constant.small[i] = 0.0;
        }
        return true;
    }

    size_t pool_index = node->data.constant.constant_index;
    if (builder->constants_data == NULL || pool_index >= builder->constants_count ||
        builder->constants_components == NULL || builder->constants_offsets == NULL) {
        return false;
    }

    size_t components = builder->constants_components[pool_index];
    size_t offset     = builder->constants_offsets[pool_index];
    if (components == 0U) {
        return false;
    }
    builder->constants_data[offset] = value;
    for (size_t i = 1U; i < components; ++i) {
        builder->constants_data[offset + i] = 0.0;
    }
    return true;
}

static void metal_mix_refresh_kernel_constants(SimMetalMixOperatorState* state,
                                               SimIRBuilder*             builder) {
    if (state == NULL || builder == NULL || !state->kernel_nodes.valid) {
        return;
    }

    const SimMetalMixOperatorConfig* cfg = &state->config;
    const double mode_crossfade          = (cfg->mode == SIM_MIXER_MODE_CROSSFADE) ? 1.0 : 0.0;
    const double accumulate              = cfg->accumulate ? 1.0 : 0.0;
    const double dt_scale                = (cfg->accumulate && cfg->scale_by_dt) ? 1.0 : 0.0;

    (void) metal_mix_set_builder_constant(builder, state->kernel_nodes.lhs_gain, cfg->lhs_gain);
    (void) metal_mix_set_builder_constant(builder, state->kernel_nodes.rhs_gain, cfg->rhs_gain);
    (void) metal_mix_set_builder_constant(builder, state->kernel_nodes.mix, cfg->mix);
    (void) metal_mix_set_builder_constant(
        builder, state->kernel_nodes.one_minus_mix, 1.0 - cfg->mix);
    (void) metal_mix_set_builder_bias_constant(builder, state->kernel_nodes.bias, cfg->bias);
    (void) metal_mix_set_builder_constant(
        builder, state->kernel_nodes.crossfade_flag, mode_crossfade);
    (void) metal_mix_set_builder_constant(builder, state->kernel_nodes.accumulate_flag, accumulate);
    (void) metal_mix_set_builder_constant(builder, state->kernel_nodes.dt_scale_flag, dt_scale);
}

static SimIRNodeId metal_mix_build_kernel_expr(SimIRBuilder*                    builder,
                                               const SimMetalMixOperatorConfig* cfg,
                                               SimIRType                        value_type,
                                               SimMetalMixKernelNodes*          out_nodes) {
    if (builder == NULL || cfg == NULL || out_nodes == NULL) {
        return SIM_IR_INVALID_NODE;
    }

    metal_mix_reset_kernel_nodes(out_nodes);

    SimIRNodeId lhs = sim_ir_builder_field_ref_typed(builder, 0U, value_type);
    SimIRNodeId rhs = sim_ir_builder_field_ref_typed(builder, 1U, value_type);
    SimIRNodeId out = sim_ir_builder_field_ref_typed(builder, 2U, value_type);
    SimIRNodeId dt  = sim_ir_builder_param(builder, SIM_IR_PARAM_DT);
    if (lhs == SIM_IR_INVALID_NODE || rhs == SIM_IR_INVALID_NODE || out == SIM_IR_INVALID_NODE ||
        dt == SIM_IR_INVALID_NODE) {
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId dt_vec = dt;
    if (!sim_ir_type_is_scalar(value_type)) {
        dt_vec = sim_ir_builder_complex_pack(builder, dt, dt);
        if (dt_vec == SIM_IR_INVALID_NODE) {
            return SIM_IR_INVALID_NODE;
        }
    }

    out_nodes->lhs_gain       = metal_mix_broadcast_constant(builder, value_type, cfg->lhs_gain);
    out_nodes->rhs_gain       = metal_mix_broadcast_constant(builder, value_type, cfg->rhs_gain);
    out_nodes->mix            = metal_mix_broadcast_constant(builder, value_type, cfg->mix);
    out_nodes->one_minus_mix  = metal_mix_broadcast_constant(builder, value_type, 1.0 - cfg->mix);
    out_nodes->bias           = metal_mix_bias_constant(builder, value_type, cfg->bias);
    out_nodes->crossfade_flag = metal_mix_broadcast_constant(
        builder, value_type, (cfg->mode == SIM_MIXER_MODE_CROSSFADE) ? 1.0 : 0.0);
    out_nodes->accumulate_flag =
        metal_mix_broadcast_constant(builder, value_type, cfg->accumulate ? 1.0 : 0.0);
    out_nodes->dt_scale_flag = metal_mix_broadcast_constant(
        builder, value_type, (cfg->accumulate && cfg->scale_by_dt) ? 1.0 : 0.0);
    SimIRNodeId one = metal_mix_broadcast_constant(builder, value_type, 1.0);

    if (out_nodes->lhs_gain == SIM_IR_INVALID_NODE || out_nodes->rhs_gain == SIM_IR_INVALID_NODE ||
        out_nodes->mix == SIM_IR_INVALID_NODE || out_nodes->one_minus_mix == SIM_IR_INVALID_NODE ||
        out_nodes->bias == SIM_IR_INVALID_NODE ||
        out_nodes->crossfade_flag == SIM_IR_INVALID_NODE ||
        out_nodes->accumulate_flag == SIM_IR_INVALID_NODE ||
        out_nodes->dt_scale_flag == SIM_IR_INVALID_NODE || one == SIM_IR_INVALID_NODE) {
        metal_mix_reset_kernel_nodes(out_nodes);
        return SIM_IR_INVALID_NODE;
    }

    SimIRNodeId lhs_scaled =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, lhs, out_nodes->lhs_gain);
    SimIRNodeId rhs_scaled =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, rhs, out_nodes->rhs_gain);
    SimIRNodeId linear = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, lhs_scaled, rhs_scaled);

    SimIRNodeId lhs_cross =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, out_nodes->one_minus_mix, lhs_scaled);
    SimIRNodeId rhs_cross =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, out_nodes->mix, rhs_scaled);
    SimIRNodeId cross = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, lhs_cross, rhs_cross);

    SimIRNodeId delta = sim_ir_builder_binary(builder, SIM_IR_NODE_SUB, cross, linear);
    SimIRNodeId delta_flag =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, delta, out_nodes->crossfade_flag);
    SimIRNodeId mixed  = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, linear, delta_flag);
    SimIRNodeId biased = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, mixed, out_nodes->bias);

    SimIRNodeId one_minus_dt_flag =
        sim_ir_builder_binary(builder, SIM_IR_NODE_SUB, one, out_nodes->dt_scale_flag);
    SimIRNodeId scaled_no_dt =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, biased, one_minus_dt_flag);
    SimIRNodeId scaled_dt_pre = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, biased, dt_vec);
    SimIRNodeId scaled_dt =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, scaled_dt_pre, out_nodes->dt_scale_flag);
    SimIRNodeId scaled = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, scaled_no_dt, scaled_dt);

    SimIRNodeId out_term =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, out, out_nodes->accumulate_flag);
    SimIRNodeId root = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, out_term, scaled);
    if (root == SIM_IR_INVALID_NODE) {
        metal_mix_reset_kernel_nodes(out_nodes);
        return SIM_IR_INVALID_NODE;
    }

    out_nodes->valid = true;
    return root;
}

SimResult sim_add_metal_mix_operator(struct SimContext*               context,
                                     const SimMetalMixOperatorConfig* config,
                                     size_t*                          out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimMetalMixOperatorState* state =
        (SimMetalMixOperatorState*) calloc(1U, sizeof(SimMetalMixOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    metal_mix_reset_kernel_nodes(&state->kernel_nodes);

    SimMetalMixOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.lhs_field    = 0U;
        local.rhs_field    = 0U;
        local.output_field = 0U;
        local.lhs_gain     = 1.0;
        local.rhs_gain     = 1.0;
        local.mix          = 0.5;
        local.bias         = 0.0;
        local.mode         = SIM_MIXER_MODE_LINEAR;
        local.accumulate   = false;
        local.scale_by_dt  = true;
    }
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "metal_mix", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    metal_mix_normalize_config(&local);
    state->config = local;
    metal_mix_refresh_symbolic(state);

    SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField* out_field = sim_context_field(context, state->config.output_field);
    if (lhs_field == NULL || rhs_field == NULL || out_field == NULL) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "metal_mix");

    const bool needs_complex = sim_field_is_complex(lhs_field) || sim_field_is_complex(rhs_field) ||
                               sim_field_is_complex(out_field);

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_COUPLING;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "metal_mix";
    sim_operator_info_set_schema_identity(&info, "metal_mix");
    info.algebraic_flags       = SIM_OPERATOR_ALG_AFFINE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimResult result            = SIM_RESULT_OK;
    bool      registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, NULL, 0ULL, SIM_DET_NONE, "metal_mix")) {
        SimIRBuilder* builder = sim_context_ir_builder(context);
        if (builder != NULL) {
            const size_t lhs_components = sim_field_components(lhs_field);
            const size_t rhs_components = sim_field_components(rhs_field);
            const size_t out_components = sim_field_components(out_field);
            const bool   compatible_components =
                (lhs_components == rhs_components) && (lhs_components == out_components);
            const bool scalar_or_complex =
                (lhs_components == 1U) || (needs_complex && lhs_components == 2U);

            if (compatible_components && scalar_or_complex) {
                SimIRType value_type = needs_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId root     = metal_mix_build_kernel_expr(
                    builder, &state->config, value_type, &state->kernel_nodes);
                if (root != SIM_IR_INVALID_NODE) {
                    SimOperatorKernelBindingDescriptor bindings[3];
                    SimOperatorKernelOutputDescriptor  outputs[1];
                    SimOperatorKernelDescriptor        kernel_desc = { 0 };

                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = state->config.lhs_field;
                    bindings[1].ir_field_index      = 1U;
                    bindings[1].context_field_index = state->config.rhs_field;
                    bindings[2].ir_field_index      = 2U;
                    bindings[2].context_field_index = state->config.output_field;

                    outputs[0].ir_field_index = 2U;
                    outputs[0].expression     = root;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = 3U;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_DT + 1U;
                    kernel_desc.required_features = 0ULL;
                    kernel_desc.complex_semantics = needs_complex
                                                        ? SIM_KERNEL_COMPLEX_SEMANTICS_COMPONENTWISE
                                                        : SIM_KERNEL_COMPLEX_SEMANTICS_TRUE_COMPLEX;

                    SimOperatorDescriptor descriptor = { 0 };
                    descriptor.name                  = name;
                    descriptor.destroy               = free;
                    descriptor.userdata              = state;
                    descriptor.kernel                = &kernel_desc;
                    descriptor.info                  = info;
                    if (state->config.lhs_field < 64U) {
                        descriptor.read_mask |= (1ULL << state->config.lhs_field);
                    }
                    if (state->config.rhs_field < 64U) {
                        descriptor.read_mask |= (1ULL << state->config.rhs_field);
                    }
                    if (state->config.output_field < 64U) {
                        descriptor.write_mask |= (1ULL << state->config.output_field);
                    }

                    result = sim_context_register_operator(context, &descriptor, out_index);
                    if (result == SIM_RESULT_OK) {
                        metal_mix_refresh_kernel_constants(state, builder);
                        registered_kernel = true;
                    } else {
                        metal_mix_reset_kernel_nodes(&state->kernel_nodes);
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    SimSplitPort ports[3] = {
        { .context_field_index = state->config.lhs_field, .require_complex = needs_complex },
        { .context_field_index = state->config.rhs_field, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };
    SimSplitAccess     accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                       { .port = 1, .mode = SIM_ACCESS_READ },
                                       { .port = 2, .mode = SIM_ACCESS_RW } };
    SimSplitSubstep    substep     = { .name              = NULL,
                                       .fn                = metal_mix_step,
                                       .accesses          = accesses,
                                       .access_count      = 3U,
                                       .dt_scale          = 1.0,
                                       .barrier_after     = false,
                                       .error_measure     = NULL,
                                       .required_features = 0U };
    SimSplitDescriptor split_desc  = { .name          = name,
                                       .ports         = ports,
                                       .port_count    = 3U,
                                       .substeps      = &substep,
                                       .substep_count = 1U,
                                       .state         = state,
                                       .symbolic      = metal_mix_symbolic,
                                       .destroy       = free,
                                       .info          = info,
                                       .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &split_desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }
    return result;
}

SimResult sim_metal_mix_config(struct SimContext*         context,
                               size_t                     operator_index,
                               SimMetalMixOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMetalMixOperatorState* state = (SimMetalMixOperatorState*) sim_operator_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_metal_mix_update(struct SimContext*               context,
                               size_t                           operator_index,
                               const SimMetalMixOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMetalMixOperatorState* state = (SimMetalMixOperatorState*) sim_operator_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimMetalMixOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "metal_mix", true, config->scale_by_dt);
    }

    /* Field bindings are fixed at registration time. */
    local.lhs_field    = state->config.lhs_field;
    local.rhs_field    = state->config.rhs_field;
    local.output_field = state->config.output_field;

    metal_mix_normalize_config(&local);
    state->config = local;
    metal_mix_refresh_symbolic(state);

    if (op->kernel != NULL && state->kernel_nodes.valid) {
        SimIRBuilder* builder = (SimIRBuilder*) op->kernel->kernel.builder;
        metal_mix_refresh_kernel_constants(state, builder);
    }

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
