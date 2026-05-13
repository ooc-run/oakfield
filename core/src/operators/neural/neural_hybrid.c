#include "oakfield/operators/neural/neural_hybrid.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/neural_models.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct SimNeuralHybridOperatorState {
    SimNeuralHybridOperatorConfig config;
    char                          symbolic[240];
} SimNeuralHybridOperatorState;

static void neural_hybrid_shape_constraints_normalize(SimNeuralShapeConstraints* constraints) {
    if (constraints == NULL) {
        return;
    }
    if (constraints->min_rank == 0U) {
        constraints->min_rank = 1U;
    }
    if (constraints->max_rank > 0U && constraints->max_rank < constraints->min_rank) {
        constraints->max_rank = constraints->min_rank;
    }
    if (constraints->channel_axis != SIM_NEURAL_CHANNEL_AXIS_AUTO &&
        constraints->channel_axis > 15U) {
        constraints->channel_axis = SIM_NEURAL_CHANNEL_AXIS_AUTO;
    }
    if (constraints->max_channels > 0U && constraints->max_channels < constraints->min_channels) {
        constraints->max_channels = constraints->min_channels;
    }
}

static void neural_hybrid_base_normalize(SimNeuralBaseConfig* base) {
    if (base == NULL) {
        return;
    }
    if (!isfinite(base->input_scale) || base->input_scale == 0.0) {
        base->input_scale = 1.0;
    }
    if (!isfinite(base->input_bias)) {
        base->input_bias = 0.0;
    }
    if (!isfinite(base->output_scale) || base->output_scale == 0.0) {
        base->output_scale = 1.0;
    }
    if (!isfinite(base->output_bias)) {
        base->output_bias = 0.0;
    }
    base->accumulate      = base->accumulate ? true : false;
    base->scale_by_dt     = base->scale_by_dt ? true : false;
    base->normalize_input = base->normalize_input ? true : false;
    if (base->model_id[0] == '\0') {
        (void) strncpy(base->model_id, "model", sizeof(base->model_id));
        base->model_id[sizeof(base->model_id) - 1U] = '\0';
    }

    switch (base->determinism_policy) {
        case SIM_NEURAL_DETERMINISM_INHERIT:
        case SIM_NEURAL_DETERMINISM_STRICT:
        case SIM_NEURAL_DETERMINISM_BEST_EFFORT:
        case SIM_NEURAL_DETERMINISM_OFF:
            break;
        default:
            base->determinism_policy = SIM_NEURAL_DETERMINISM_INHERIT;
            break;
    }
    switch (base->device_requirement) {
        case SIM_NEURAL_DEVICE_ANY:
        case SIM_NEURAL_DEVICE_CPU_ONLY:
        case SIM_NEURAL_DEVICE_ACCELERATOR_PREFERRED:
        case SIM_NEURAL_DEVICE_ACCELERATOR_REQUIRED:
            break;
        default:
            base->device_requirement = SIM_NEURAL_DEVICE_ANY;
            break;
    }
    switch (base->precision_mode) {
        case SIM_NEURAL_PRECISION_DEFAULT:
        case SIM_NEURAL_PRECISION_FP32:
        case SIM_NEURAL_PRECISION_FP64:
        case SIM_NEURAL_PRECISION_MIXED:
        case SIM_NEURAL_PRECISION_FP16:
        case SIM_NEURAL_PRECISION_BF16:
            break;
        default:
            base->precision_mode = SIM_NEURAL_PRECISION_DEFAULT;
            break;
    }
    neural_hybrid_shape_constraints_normalize(&base->shape_constraints);
}

static void neural_hybrid_normalize_config(SimNeuralHybridOperatorConfig* config) {
    if (config == NULL) {
        return;
    }
    neural_hybrid_base_normalize(&config->base);
    if (!isfinite(config->analytic_gain)) {
        config->analytic_gain = 1.0;
    }
    if (!isfinite(config->residual_gain)) {
        config->residual_gain = 1.0;
    }
}

static bool neural_hybrid_fields_compatible(const SimField* input, const SimField* output) {
    if (input == NULL || output == NULL) {
        return false;
    }
    if (input->layout.rank != output->layout.rank) {
        return false;
    }
    if (input->layout.shape == NULL || output->layout.shape == NULL) {
        return false;
    }
    for (size_t i = 0U; i < input->layout.rank; ++i) {
        if (input->layout.shape[i] != output->layout.shape[i]) {
            return false;
        }
    }
    if (input->element_size != output->element_size) {
        return false;
    }
    if (!(input->element_size == sizeof(double) ||
          input->element_size == sizeof(SimComplexDouble))) {
        return false;
    }
    return true;
}

static bool
neural_hybrid_shape_constraints_allow_field(const SimNeuralShapeConstraints* constraints,
                                            const SimField*                  field) {
    size_t rank;
    size_t channel_axis;
    size_t channels;

    if (constraints == NULL || field == NULL || field->layout.shape == NULL) {
        return false;
    }
    rank = field->layout.rank;
    if (rank == 0U) {
        return false;
    }
    if (constraints->min_rank > 0U && rank < (size_t) constraints->min_rank) {
        return false;
    }
    if (constraints->max_rank > 0U && rank > (size_t) constraints->max_rank) {
        return false;
    }
    if (!constraints->allow_complex_input && sim_field_is_complex(field)) {
        return false;
    }

    if (constraints->channel_axis == SIM_NEURAL_CHANNEL_AXIS_AUTO) {
        channel_axis = (rank <= 1U) ? 0U : (constraints->channels_last ? (rank - 1U) : 0U);
    } else {
        channel_axis = constraints->channel_axis;
    }
    if (channel_axis >= rank) {
        return false;
    }
    channels = field->layout.shape[channel_axis];
    if (constraints->min_channels > 0U && channels < (size_t) constraints->min_channels) {
        return false;
    }
    if (constraints->max_channels > 0U && channels > (size_t) constraints->max_channels) {
        return false;
    }
    return true;
}

static SimDeterminismFlags neural_hybrid_policy_to_flags(SimNeuralDeterminismPolicy policy) {
    switch (policy) {
        case SIM_NEURAL_DETERMINISM_STRICT:
            return SIM_DET_PURE_TIME | SIM_DET_REWIND_SAFE | SIM_DET_NO_STATEFUL_NODES;
        case SIM_NEURAL_DETERMINISM_BEST_EFFORT:
            return SIM_DET_PURE_TIME;
        case SIM_NEURAL_DETERMINISM_OFF:
        case SIM_NEURAL_DETERMINISM_INHERIT:
        default:
            return SIM_DET_NONE;
    }
}

static void neural_hybrid_refresh_symbolic(SimNeuralHybridOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "neural_hybrid model=%s in=%zu out=%zu analytic=%.3g residual=%.3g",
                    state->config.base.model_id,
                    state->config.base.input_field,
                    state->config.base.output_field,
                    state->config.analytic_gain,
                    state->config.residual_gain);
#else
    (void) state;
#endif
}

static const char* neural_hybrid_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimNeuralHybridOperatorState* state = (const SimNeuralHybridOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult neural_hybrid_fill_prediction(const SimNeuralHybridOperatorState* state,
                                               const SimField*                     input,
                                               SimField*                           prediction,
                                               const SimContext*                   context,
                                               double                              dt) {
    const SimNeuralBaseConfig* base;
    SimNeuralInferenceRequest  request;
    if (state == NULL || input == NULL || prediction == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    base = &state->config.base;

    request = (SimNeuralInferenceRequest){
        .model_id           = base->model_id,
        .determinism_policy = base->determinism_policy,
        .device_requirement = base->device_requirement,
        .precision_mode     = base->precision_mode,
        .shape_constraints  = base->shape_constraints,
        .normalize_input    = base->normalize_input,
        .input_scale        = base->input_scale,
        .input_bias         = base->input_bias,
        .output_scale       = base->output_scale,
        .output_bias        = base->output_bias,
        .step_index         = sim_context_step_index(context),
        .dt                 = dt,
        .sim_time           = sim_context_time(context),
    };

    if (base->inference_fn != NULL) {
        return base->inference_fn(base->inference_userdata, input, prediction, &request);
    }

    if (context != NULL && base->model_id[0] != '\0' && sim_neural_model_count(context) > 0U) {
        SimResult model_rc = sim_neural_model_infer(
            (SimContext*) context, base->model_id, input, prediction, &request);
        if (model_rc != SIM_RESULT_NOT_FOUND) {
            return model_rc;
        }
    }

    {
        size_t count = sim_field_element_count(&input->layout);
        if (sim_field_is_complex(input)) {
            const SimComplexDouble* src = sim_field_complex_data_const(input);
            SimComplexDouble*       dst = sim_field_complex_data(prediction);
            if (src == NULL || dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            for (size_t i = 0U; i < count; ++i) {
                double re = src[i].re;
                double im = src[i].im;
                if (base->normalize_input) {
                    re = re * base->input_scale + base->input_bias;
                    im = im * base->input_scale + base->input_bias;
                }
                re        = re * base->output_scale + base->output_bias;
                im        = im * base->output_scale + base->output_bias;
                dst[i].re = re;
                dst[i].im = im;
            }
        } else {
            const double* src = sim_field_real_data_const(input);
            double*       dst = sim_field_real_data(prediction);
            if (src == NULL || dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            for (size_t i = 0U; i < count; ++i) {
                double x = src[i];
                if (base->normalize_input) {
                    x = x * base->input_scale + base->input_bias;
                }
                x      = x * base->output_scale + base->output_bias;
                dst[i] = x;
            }
        }
    }
    return SIM_RESULT_OK;
}

static SimResult neural_hybrid_step(void*               state_ptr,
                                    struct SimContext*  context,
                                    struct SimOperator* self,
                                    size_t              substep_index,
                                    double              dt_sub,
                                    void*               scratch,
                                    size_t              scratch_size) {
    SimNeuralHybridOperatorState* state = (SimNeuralHybridOperatorState*) state_ptr;
    SimField*                     input;
    SimField*                     output;
    SimField                      prediction      = { 0 };
    void*                         prediction_data = NULL;
    size_t                        count;
    double                        write_scale;
    SimResult                     rc;
    const SimNeuralBaseConfig*    base;

    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    base = &state->config.base;

    input  = sim_context_field(context, base->input_field);
    output = sim_context_field(context, base->output_field);
    if (!neural_hybrid_fields_compatible(input, output) ||
        !neural_hybrid_shape_constraints_allow_field(&base->shape_constraints, input)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = sim_field_element_count(&output->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }
    prediction_data = calloc(count, output->element_size);
    if (prediction_data == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    rc = sim_field_wrap(
        &prediction, &output->layout, output->element_size, output->storage, prediction_data);
    if (rc != SIM_RESULT_OK) {
        free(prediction_data);
        return rc;
    }

    rc = neural_hybrid_fill_prediction(state, input, &prediction, context, dt_sub);
    if (rc != SIM_RESULT_OK) {
        sim_field_destroy(&prediction);
        free(prediction_data);
        return rc;
    }

    write_scale = base->scale_by_dt ? fmax(dt_sub, 0.0) : 1.0;
    if (sim_field_is_complex(output)) {
        const SimComplexDouble* model = sim_field_complex_data_const(&prediction);
        const SimComplexDouble* src   = sim_field_complex_data_const(input);
        SimComplexDouble*       dst   = sim_field_complex_data(output);
        if (model == NULL || src == NULL || dst == NULL) {
            sim_field_destroy(&prediction);
            free(prediction_data);
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            double re =
                state->config.analytic_gain * src[i].re + state->config.residual_gain * model[i].re;
            double im =
                state->config.analytic_gain * src[i].im + state->config.residual_gain * model[i].im;
            if (base->accumulate) {
                dst[i].re += write_scale * re;
                dst[i].im += write_scale * im;
            } else {
                dst[i].re = write_scale * re;
                dst[i].im = write_scale * im;
            }
        }
    } else {
        const double* model = sim_field_real_data_const(&prediction);
        const double* src   = sim_field_real_data_const(input);
        double*       dst   = sim_field_real_data(output);
        if (model == NULL || src == NULL || dst == NULL) {
            sim_field_destroy(&prediction);
            free(prediction_data);
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            double value =
                state->config.analytic_gain * src[i] + state->config.residual_gain * model[i];
            if (base->accumulate) {
                dst[i] += write_scale * value;
            } else {
                dst[i] = write_scale * value;
            }
        }
    }

    sim_field_destroy(&prediction);
    free(prediction_data);
    return SIM_RESULT_OK;
}

SimResult sim_add_neural_hybrid_operator(struct SimContext*                   context,
                                         const SimNeuralHybridOperatorConfig* config,
                                         size_t*                              out_index) {
    SimNeuralHybridOperatorConfig local = { 0 };
    SimNeuralHybridOperatorState* state;
    SimField*                     input;
    SimField*                     output;
    bool                          needs_complex;
    char                          name[SIM_OPERATOR_NAME_MAX + 1U];
    SimOperatorInfo               info = sim_operator_info_defaults();
    SimSplitPort                  ports[2];
    SimSplitAccess                accesses[2];
    SimSplitSubstep               substep;
    SimSplitDescriptor            desc = { 0 };
    SimResult                     result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (config != NULL) {
        local = *config;
    }
    local.base.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "neural_hybrid",
                                         (config != NULL),
                                         (config != NULL) ? config->base.scale_by_dt : true);
    neural_hybrid_normalize_config(&local);

    input  = sim_context_field(context, local.base.input_field);
    output = sim_context_field(context, local.base.output_field);
    if (!neural_hybrid_fields_compatible(input, output) ||
        !neural_hybrid_shape_constraints_allow_field(&local.base.shape_constraints, input)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimNeuralHybridOperatorState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    state->config = local;
    neural_hybrid_refresh_symbolic(state);

    sim_operator_make_unique_name(name, sizeof(name), "neural_hybrid");
    needs_complex = sim_field_is_complex(input) || sim_field_is_complex(output);

    info.category          = SIM_OPERATOR_CATEGORY_NONLINEAR;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = false;
    info.is_nonlocal       = true;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = !needs_complex;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "neural_hybrid";
    sim_operator_info_set_schema_identity(&info, "neural_hybrid");
    info.algebraic_flags           = SIM_OPERATOR_ALG_NONE;
    info.determinism_flags         = neural_hybrid_policy_to_flags(local.base.determinism_policy);
    info.neural.enabled            = true;
    info.neural.determinism_policy = local.base.determinism_policy;
    info.neural.device_requirement = local.base.device_requirement;
    info.neural.precision_mode     = local.base.precision_mode;
    info.neural.shape              = local.base.shape_constraints;
    info.representation.domain     = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = sim_field_is_complex(input);
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    ports[0]    = (SimSplitPort){ .context_field_index = state->config.base.input_field,
                                  .require_complex     = sim_field_is_complex(input) };
    ports[1]    = (SimSplitPort){ .context_field_index = state->config.base.output_field,
                                  .require_complex     = needs_complex };
    accesses[0] = (SimSplitAccess){ .port = 0U, .mode = SIM_ACCESS_READ };
    accesses[1] = (SimSplitAccess){ .port = 1U, .mode = SIM_ACCESS_RW };
    substep     = (SimSplitSubstep){
        .name              = NULL,
        .fn                = neural_hybrid_step,
        .accesses          = accesses,
        .access_count      = 2U,
        .dt_scale          = 1.0,
        .barrier_after     = false,
        .error_measure     = NULL,
        .required_features = 0U,
    };
    desc = (SimSplitDescriptor){
        .name          = name,
        .ports         = ports,
        .port_count    = 2U,
        .substeps      = &substep,
        .substep_count = 1U,
        .state         = state,
        .symbolic      = neural_hybrid_symbolic,
        .save_state    = NULL,
        .restore_state = NULL,
        .destroy       = free,
        .info          = info,
        .config        = sim_operator_config_defaults(),
        .scratch       = { 0U, 0U },
    };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }
    return result;
}

SimResult sim_neural_hybrid_config(struct SimContext*             context,
                                   size_t                         operator_index,
                                   SimNeuralHybridOperatorConfig* out_config) {
    SimOperator*                  op;
    SimNeuralHybridOperatorState* state;
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }
    state = (SimNeuralHybridOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }
    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_neural_hybrid_update(struct SimContext*                   context,
                                   size_t                               operator_index,
                                   const SimNeuralHybridOperatorConfig* config) {
    SimOperator*                  op;
    SimNeuralHybridOperatorState* state;
    SimNeuralHybridOperatorConfig local;
    SimField*                     input;
    SimField*                     output;

    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }
    state = (SimNeuralHybridOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local                  = *config;
    local.base.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "neural_hybrid"), true, local.base.scale_by_dt);
    neural_hybrid_normalize_config(&local);

    input  = sim_context_field(context, local.base.input_field);
    output = sim_context_field(context, local.base.output_field);
    if (!neural_hybrid_fields_compatible(input, output) ||
        !neural_hybrid_shape_constraints_allow_field(&local.base.shape_constraints, input)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config = local;
    neural_hybrid_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
