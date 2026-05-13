#include "oakfield/operators/stimulus/fourier.h"
#include "operators/common/operator_utils.h"

#include "oakfield/math/fourier.h"
#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "sim_accel.h"
#include "oakfield/backend.h"
#include "oakfield/operator_identity.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_FOURIER_EPS 1.0e-12
#define STIM_FOURIER_TWO_PI (2.0 * M_PI)
#define STIM_FOURIER_MAX_HARMONICS 16384

typedef struct SimFourierWaveformState {
    SimFourierWaveformConfig config;
    double                   phase;      /* normalized cycles in [0,1) */
    double                   blit_state; /* single integrator for saw/square BLIT */
    double                   tri_vel;    /* first integrator for triangle (poly/mini) */
    double                   tri_pos;    /* second integrator for triangle (poly/mini) */
    double                   kernel_cached_sample;
    size_t                   kernel_cached_step_index;
    bool                     kernel_cached_valid;
    char                     symbolic[192];
} SimFourierWaveformState;

static double stim_fourier_wrap_phase(double p) {
    if (!isfinite(p)) {
        return 0.0;
    }
    p = fmod(p, 1.0);
    if (p < 0.0) {
        p += 1.0;
    }
    return p;
}

static void fourier_normalize(SimFourierWaveformConfig* cfg) {
    if (cfg == NULL) {
        return;
    }

    if (!isfinite(cfg->amplitude)) {
        cfg->amplitude = 0.0;
    }
    if (!isfinite(cfg->frequency) || cfg->frequency < 0.0) {
        cfg->frequency = 0.0;
    }
    if (!isfinite(cfg->phase)) {
        cfg->phase = 0.0;
    }
    cfg->phase = stim_fourier_wrap_phase(cfg->phase);

    if (!isfinite(cfg->duty)) {
        cfg->duty = 0.5;
    }
    if (cfg->duty < STIM_FOURIER_EPS) {
        cfg->duty = STIM_FOURIER_EPS;
    } else if (cfg->duty > 1.0 - STIM_FOURIER_EPS) {
        cfg->duty = 1.0 - STIM_FOURIER_EPS;
    }

    if (!isfinite(cfg->rotation)) {
        cfg->rotation = 0.0;
    }
    if (!isfinite(cfg->nominal_dt) || cfg->nominal_dt < 0.0) {
        cfg->nominal_dt = 0.0;
    }

    if (cfg->shape < SIM_FOURIER_WAVEFORM_SAW || cfg->shape > SIM_FOURIER_WAVEFORM_TRIANGLE) {
        cfg->shape = SIM_FOURIER_WAVEFORM_SAW;
    }
    if (cfg->method < SIM_FOURIER_METHOD_BLIT || cfg->method > SIM_FOURIER_METHOD_MINIBLEP) {
        cfg->method = SIM_FOURIER_METHOD_POLYBLEP;
    }
}

static const char* fourier_shape_name(SimFourierWaveformShape shape) {
    switch (shape) {
        case SIM_FOURIER_WAVEFORM_SQUARE:
            return "square";
        case SIM_FOURIER_WAVEFORM_TRIANGLE:
            return "triangle";
        case SIM_FOURIER_WAVEFORM_SAW:
        default:
            return "saw";
    }
}

static const char* fourier_method_name(SimFourierWaveformMethod method) {
    switch (method) {
        case SIM_FOURIER_METHOD_BLIT:
            return "blit";
        case SIM_FOURIER_METHOD_MINIBLEP:
            return "miniblep";
        case SIM_FOURIER_METHOD_POLYBLEP:
        default:
            return "polyblep";
    }
}

static void fourier_refresh_symbolic(SimFourierWaveformState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimFourierWaveformConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "%s/%s %.3g Hz (duty=%.2g, phase=%.2g)",
                    fourier_shape_name(cfg->shape),
                    fourier_method_name(cfg->method),
                    cfg->frequency,
                    cfg->duty,
                    cfg->phase);
#else
    (void) state;
#endif
}

static const char* fourier_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimFourierWaveformState* state = (const SimFourierWaveformState*) state_ptr;
    if (state == NULL) {
        return NULL;
    }
    return state->symbolic;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static inline double fourier_phase_increment(const SimFourierWaveformConfig* cfg, double dt_sub) {
    if (cfg == NULL) {
        return 0.0;
    }
    double dt = cfg->fixed_clock ? ((cfg->nominal_dt > STIM_FOURIER_EPS) ? cfg->nominal_dt : dt_sub)
                                 : dt_sub;
    if (dt <= 0.0) {
        return 0.0;
    }
    return cfg->frequency * dt;
}

static double kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static inline double fourier_saw_correction(double phase, double dphase, bool mini) {
    double corr = mini ? sim_fourier_miniblep(phase, dphase) : sim_fourier_polyblep(phase, dphase);
    double next = phase + dphase;
    if (next >= 1.0) {
        corr -= mini ? sim_fourier_miniblep(next - 1.0, dphase)
                     : sim_fourier_polyblep(next - 1.0, dphase);
    }
    return corr;
}

static inline double
fourier_square_correction(double phase, double dphase, double duty, bool mini) {
    double rising =
        mini ? sim_fourier_miniblep(phase, dphase) : sim_fourier_polyblep(phase, dphase);
    double edge_phase = phase - duty;
    if (edge_phase < 0.0) {
        edge_phase += 1.0;
    }
    double falling =
        mini ? sim_fourier_miniblep(edge_phase, dphase) : sim_fourier_polyblep(edge_phase, dphase);
    return rising - falling;
}

static inline double fourier_center_bipolar(double x) {
    double wrapped = fmod(x + 1.0, 2.0);
    if (wrapped < 0.0) {
        wrapped += 2.0;
    }
    return wrapped - 1.0;
}
static inline int fourier_harmonics(double dphase) {
    if (dphase <= 0.0) {
        return 0;
    }
    double partials = floor(0.5 / dphase);
    if (partials < 1.0) {
        partials = 1.0;
    }
    if (partials > (double) STIM_FOURIER_MAX_HARMONICS) {
        partials = (double) STIM_FOURIER_MAX_HARMONICS;
    }
    return (int) partials;
}
static double fourier_sample(SimFourierWaveformState* state, double dphase) {
    const SimFourierWaveformConfig* cfg = &state->config;

    double sample = 0.0;

    switch (cfg->method) {
        case SIM_FOURIER_METHOD_BLIT: {
            int harmonics = fourier_harmonics(dphase);
            if (harmonics <= 0) {
                break;
            }

            double phase_radians  = STIM_FOURIER_TWO_PI * state->phase;
            double dphase_radians = STIM_FOURIER_TWO_PI * dphase;

            if (cfg->shape == SIM_FOURIER_WAVEFORM_SAW) {
                sample = sim_fourier_saw_blit(
                    phase_radians, dphase_radians, harmonics, &state->blit_state);
            } else if (cfg->shape == SIM_FOURIER_WAVEFORM_SQUARE) {
                sample = sim_fourier_square_blit(
                    phase_radians, dphase_radians, harmonics, cfg->duty, &state->blit_state);
            } else /* triangle */
            {
                sample = sim_fourier_triangle_blit(
                    phase_radians, dphase_radians, harmonics, &state->tri_vel, &state->tri_pos);
            }
            break;
        }
        case SIM_FOURIER_METHOD_MINIBLEP:
        case SIM_FOURIER_METHOD_POLYBLEP:
        default: {
            bool mini = (cfg->method == SIM_FOURIER_METHOD_MINIBLEP);
            if (cfg->shape == SIM_FOURIER_WAVEFORM_SAW) {
                double saw = 2.0 * state->phase - 1.0;
                saw -= fourier_saw_correction(state->phase, dphase, mini);
                sample = saw;
            } else if (cfg->shape == SIM_FOURIER_WAVEFORM_SQUARE) {
                double square = (state->phase < cfg->duty) ? 1.0 : -1.0;
                square += fourier_square_correction(state->phase, dphase, cfg->duty, mini);
                sample = square;
            } else /* triangle */
            {
                double square = (state->phase < cfg->duty) ? 1.0 : -1.0;
                square += fourier_square_correction(state->phase, dphase, cfg->duty, mini);
                double slope   = 4.0 * dphase;
                state->tri_vel = square;
                state->tri_pos += square * slope;
                state->tri_pos = fourier_center_bipolar(state->tri_pos);
                sample         = state->tri_pos;
            }
            break;
        }
    }

    state->phase = stim_fourier_wrap_phase(state->phase + dphase);
    return sample;
}

static SimResult fourier_ir_eval(void*           userdata,
                                 const KernelIR* kernel,
                                 size_t          element_index,
                                 size_t          component,
                                 double*         out_value) {
    (void) element_index;
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFourierWaveformState*  state = (SimFourierWaveformState*) userdata;
    SimFourierWaveformConfig* cfg   = &state->config;
    if (cfg->amplitude == 0.0 || cfg->frequency <= 0.0) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    size_t step_index = (size_t) kernel_param_value(kernel, SIM_IR_PARAM_STEP_INDEX);
    if (state->kernel_cached_valid && state->kernel_cached_step_index == step_index) {
        *out_value = state->kernel_cached_sample;
        return SIM_RESULT_OK;
    }

    double dt     = kernel_param_value(kernel, SIM_IR_PARAM_DT);
    double dphase = fourier_phase_increment(cfg, dt);
    if (dphase <= 0.0) {
        state->kernel_cached_sample     = 0.0;
        state->kernel_cached_step_index = step_index;
        state->kernel_cached_valid      = true;
        *out_value                      = 0.0;
        return SIM_RESULT_OK;
    }

    double scale  = cfg->scale_by_dt ? dt : 1.0;
    double sample = cfg->amplitude * fourier_sample(state, dphase);
    if (!isfinite(sample)) {
        sample = 0.0;
    }
    sample *= scale;

    state->kernel_cached_sample     = sample;
    state->kernel_cached_step_index = step_index;
    state->kernel_cached_valid      = true;
    *out_value                      = sample;
    return SIM_RESULT_OK;
}

static SimResult fourier_step(void*               state_ptr,
                              struct SimContext*  context,
                              struct SimOperator* self,
                              size_t              substep_index,
                              double              dt,
                              void*               scratch,
                              size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    SimFourierWaveformState* state = (SimFourierWaveformState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const bool is_complex = sim_field_is_complex(field);
    if (!is_complex && field->element_size != sizeof(double)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = is_complex ? sim_field_element_count(&field->layout)
                              : sim_field_bytes(field) / sizeof(double);
    if (count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double*           dst_real    = is_complex ? NULL : (double*) sim_field_data(field);
    SimComplexDouble* dst_complex = is_complex ? sim_field_complex_data(field) : NULL;
    if ((is_complex && dst_complex == NULL) || (!is_complex && dst_real == NULL)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double dphase = fourier_phase_increment(&state->config, dt);
    if (dphase <= 0.0 || state->config.frequency <= 0.0) {
        return SIM_RESULT_OK;
    }

    double scale     = state->config.scale_by_dt ? dt : 1.0;
    double amplitude = state->config.amplitude;
    double c_rot = 1.0, s_rot = 0.0;
    if (is_complex) {
        c_rot = cos(state->config.rotation);
        s_rot = sin(state->config.rotation);
    }

    double sample = amplitude * fourier_sample(state, dphase);
    if (!isfinite(sample)) {
        return SIM_RESULT_OK;
    }
    double delta = scale * sample;
    if (!isfinite(delta) || delta == 0.0) {
        return SIM_RESULT_OK;
    }

    if (is_complex) {
        sim_accel_add_scalar_complex(dst_complex, count, delta * c_rot, delta * s_rot);
    } else {
        sim_accel_add_scalar_real(dst_real, count, delta);
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_fourier_operator(struct SimContext*              context,
                                            const SimFourierWaveformConfig* config,
                                            size_t*                         out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFourierWaveformConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    fourier_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_fourier",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimFourierWaveformState* state =
        (SimFourierWaveformState*) calloc(1U, sizeof(SimFourierWaveformState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config                   = local;
    state->phase                    = local.phase;
    state->blit_state               = 0.0;
    state->tri_vel                  = 0.0;
    state->tri_pos                  = 0.0;
    state->kernel_cached_sample     = 0.0;
    state->kernel_cached_step_index = SIZE_MAX;
    state->kernel_cached_valid      = false;
    fourier_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_fourier");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
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
    info.abstract_id       = "stimulus_fourier";
    sim_operator_info_set_schema_identity(&info, "stimulus_fourier");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
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

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_fourier")) {
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
                SimIRNodeId sample_node =
                    sim_ir_builder_stateful(builder, fourier_ir_eval, state, "stimulus_fourier");
                if (is_complex && sample_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId zero   = sim_ir_builder_constant(builder, 0.0);
                    SimIRNodeId packed = sim_ir_builder_complex_pack(builder, sample_node, zero);
                    if (packed != SIM_IR_INVALID_NODE && local.rotation != 0.0) {
                        SimIRNodeId angle = sim_ir_builder_constant(builder, local.rotation);
                        packed            = sim_ir_builder_complex_rotate(builder, packed, angle);
                    }
                    sample_node = packed;
                }
                SimIRNodeId sum =
                    sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, field_node, sample_node);

                if (field_node != SIM_IR_INVALID_NODE && sample_node != SIM_IR_INVALID_NODE &&
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
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_STEP_INDEX + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = free;
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

    SimSplitPort   port   = { .context_field_index = state->config.field_index,
                              .require_complex     = needs_complex };
    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = fourier_step,
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
                                .symbolic      = fourier_symbolic,
                                .destroy       = free,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }
    return result;
}

SimResult sim_stimulus_fourier_config(struct SimContext*        context,
                                      size_t                    operator_index,
                                      SimFourierWaveformConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimFourierWaveformState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimFourierWaveformState*) sim_operator_payload(op);
    } else {
        state = (SimFourierWaveformState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_fourier_update(struct SimContext*              context,
                                      size_t                          operator_index,
                                      const SimFourierWaveformConfig* config) {
    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimFourierWaveformState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimFourierWaveformState*) sim_operator_payload(op);
    } else {
        state = (SimFourierWaveformState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimFourierWaveformConfig local = *config;
    fourier_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context, "stimulus_fourier", true, config->scale_by_dt);

    bool reset_states = (state->config.method != local.method) ||
                        (state->config.shape != local.shape) ||
                        (fabs(state->config.duty - local.duty) > 1e-9);

    state->config = local;
    state->phase  = local.phase;
    if (reset_states) {
        state->blit_state = 0.0;
        state->tri_vel    = 0.0;
        state->tri_pos    = 0.0;
    }
    state->kernel_cached_valid = false;
    fourier_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
