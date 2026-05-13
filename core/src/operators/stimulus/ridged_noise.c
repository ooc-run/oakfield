#include "oakfield/operators/stimulus/ridged_noise.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/stimulus/coords.h"
#include "octave_noise_common.h"
#include "static_cache.h"

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_seed.h"
#include "sim_accel.h"
#include "oakfield/backend.h"
#include "oakfield/operator_identity.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_RIDGED_MAX_OCTAVES 16U
typedef struct SimStimulusRidgedNoiseState {
    SimStimulusRidgedNoiseConfig      config;
    double*                           phases;
    unsigned int                      allocated_octaves;
    SimStimulusStaticCache            cache;
    SimStimulusOctaveNoiseVdspBuffers vdsp;
    char                              symbolic[160];
} SimStimulusRidgedNoiseState;

static void ridged_noise_normalize(SimStimulusRidgedNoiseConfig* config) {
    if (config == NULL) {
        return;
    }
    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->hurst) || config->hurst <= 0.0 || config->hurst >= 1.0) {
        config->hurst = 0.5;
    }
    if (!isfinite(config->lacunarity) || config->lacunarity < 1.0) {
        config->lacunarity = 2.0;
    }

    sim_stimulus_coord_normalize(&config->coord);

    if (config->octaves == 0U) {
        config->octaves = 4U;
    }
    if (config->octaves > STIM_RIDGED_MAX_OCTAVES) {
        config->octaves = STIM_RIDGED_MAX_OCTAVES;
    }
    if (config->seed == 0ULL) {
        config->seed = 1ULL;
    }
}

static void ridged_noise_refresh_symbolic(SimStimulusRidgedNoiseState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    const SimStimulusRidgedNoiseConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "ridged A=%.3g H=%.3g O=%u",
                    cfg->amplitude,
                    cfg->hurst,
                    cfg->octaves);
#else
    (void) state;
#endif
}

static const char* ridged_noise_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusRidgedNoiseState* state = (const SimStimulusRidgedNoiseState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void ridged_noise_eval_base(const SimStimulusRidgedNoiseState* state,
                                   double                             u,
                                   double                             hurst,
                                   double                             lacunarity,
                                   unsigned int                       octaves,
                                   double*                            out_re,
                                   double*                            out_im) {
    sim_stimulus_octave_noise_eval_base(SIM_STIMULUS_OCTAVE_NOISE_RIDGED,
                                        state != NULL ? state->phases : NULL,
                                        u,
                                        hurst,
                                        lacunarity,
                                        octaves,
                                        out_re,
                                        out_im);
}

static void ridged_noise_destroy(void* state_ptr) {
    SimStimulusRidgedNoiseState* state = (SimStimulusRidgedNoiseState*) state_ptr;
    if (state == NULL) {
        return;
    }
    sim_stimulus_static_cache_destroy(&state->cache);
    sim_stimulus_octave_noise_vdsp_release(&state->vdsp);
    free(state->phases);
    free(state);
}

static SimResult ridged_noise_ensure_octaves(SimStimulusRidgedNoiseState* state) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    unsigned int desired = state->config.octaves;
    if (desired == 0U) {
        return SIM_RESULT_OK;
    }

    return sim_stimulus_octave_noise_ensure_phases(
        &state->phases, &state->allocated_octaves, desired, state->config.seed);
}

static bool ridged_noise_can_use_static_cache(const SimStimulusRidgedNoiseState* state) {
    return state != NULL && sim_stimulus_coord_is_time_invariant(&state->config.coord);
}

static SimResult ridged_noise_fill_static_cache(void*                               userdata,
                                                const SimStimulusStaticCacheLayout* layout,
                                                bool                                need_imag,
                                                double*                             out_real,
                                                double*                             out_imag) {
    SimStimulusRidgedNoiseState* state = (SimStimulusRidgedNoiseState*) userdata;
    if (state == NULL || layout == NULL || out_real == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimStimulusRidgedNoiseConfig* cfg        = &state->config;
    double                              hurst      = cfg->hurst;
    double                              lacunarity = cfg->lacunarity;
    unsigned int                        octaves    = cfg->octaves;
    bool separable = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

    for (size_t idx = 0U; idx < layout->count; ++idx) {
        size_t ix = 0U;
        size_t iy = 0U;
        sim_stimulus_static_cache_index_to_xy(layout, idx, &ix, &iy);

        double x        = cfg->coord.origin_x + (double) ix * cfg->coord.spacing_x;
        double y        = cfg->coord.origin_y + (double) iy * cfg->coord.spacing_y;
        double sample_x = 0.0;
        double sample_y = 0.0;
        sim_stimulus_coord_sample_xy(&cfg->coord, x, y, 0.0, &sample_x, &sample_y);

        double base_re = 0.0;
        double base_im = 0.0;
        if (separable) {
            double fx_re = 0.0;
            double fx_im = 0.0;
            double fy_re = 0.0;
            double fy_im = 0.0;
            ridged_noise_eval_base(state, sample_x, hurst, lacunarity, octaves, &fx_re, &fx_im);
            ridged_noise_eval_base(state, sample_y, hurst, lacunarity, octaves, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double u = sim_stimulus_coord_u(&cfg->coord, sample_x, sample_y, 0.0);
            ridged_noise_eval_base(state, u, hurst, lacunarity, octaves, &base_re, &base_im);
        }

        double value_re = cfg->amplitude * base_re;
        double value_im = cfg->amplitude * base_im;
        out_real[idx]   = isfinite(value_re) ? value_re : 0.0;
        if (need_imag && out_imag != NULL) {
            out_imag[idx] = isfinite(value_im) ? value_im : 0.0;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult ridged_noise_ensure_static_cache(SimStimulusRidgedNoiseState*        state,
                                                  const SimStimulusStaticCacheLayout* layout,
                                                  bool                                need_imag) {
    if (state == NULL || layout == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimResult prep = ridged_noise_ensure_octaves(state);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    return sim_stimulus_static_cache_ensure(
        &state->cache, layout, need_imag, ridged_noise_fill_static_cache, state);
}

static double kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult ridged_noise_kernel_value(SimStimulusRidgedNoiseState* state,
                                           const KernelIR*              kernel,
                                           size_t                       element_index,
                                           bool                         use_imag,
                                           double*                      out_value) {
    if (out_value == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRidgedNoiseConfig* cfg = &state->config;
    if (cfg->amplitude == 0.0) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    double dt    = kernel_param_value(kernel, SIM_IR_PARAM_DT);
    double t     = kernel_param_value(kernel, SIM_IR_PARAM_TIME);
    double scale = cfg->scale_by_dt ? dt : 1.0;

    if (kernel->bindings == NULL || kernel->binding_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimKernelIRBinding* binding = &kernel->bindings[0];
    const size_t*             shape   = binding->shape;
    const size_t*             strides = binding->strides;
    size_t                    rank    = binding->rank;
    if ((shape == NULL || strides == NULL || rank == 0U) && binding->field != NULL) {
        shape   = binding->field->layout.shape;
        strides = binding->field->layout.strides;
        rank    = binding->field->layout.rank;
    }
    if (shape == NULL || strides == NULL || rank == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    assert(rank == 1U || rank == 2U);

    SimStimulusStaticCacheLayout layout = { 0 };
    SimResult                    layout_rc =
        sim_stimulus_static_cache_layout_from_arrays(shape, strides, rank, &layout);
    if (layout_rc != SIM_RESULT_OK) {
        return layout_rc;
    }
    size_t count = layout.count;
    assert(element_index < count);

    if (ridged_noise_can_use_static_cache(state)) {
        SimResult prep = ridged_noise_ensure_static_cache(state, &layout, use_imag);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }
        double value =
            use_imag ? state->cache.imag[element_index] : state->cache.real[element_index];
        value *= scale;
        *out_value = isfinite(value) ? value : 0.0;
        return SIM_RESULT_OK;
    }

    SimResult prep = ridged_noise_ensure_octaves(state);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    size_t    ix       = 0U;
    size_t    iy       = 0U;
    SimResult coord_rc = sim_kernel_binding_index_to_xy(binding, element_index, &ix, &iy);
    assert(coord_rc == SIM_RESULT_OK);
    size_t extent_x = shape[rank - 1U];
    size_t extent_y = (rank == 1U) ? 1U : shape[rank - 2U];
    assert(ix < extent_x);
    assert(iy < extent_y);
    (void) iy;

    double       x          = cfg->coord.origin_x + (double) ix * cfg->coord.spacing_x;
    double       y          = cfg->coord.origin_y + (double) iy * cfg->coord.spacing_y;
    double       sample_x   = x - cfg->coord.velocity_x * t;
    double       sample_y   = y - cfg->coord.velocity_y * t;
    double       hurst      = cfg->hurst;
    double       lacunarity = cfg->lacunarity;
    unsigned int octaves    = cfg->octaves;
    bool         separable  = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

    double base_re = 0.0;
    double base_im = 0.0;
    if (separable) {
        double fx_re = 0.0;
        double fx_im = 0.0;
        double fy_re = 0.0;
        double fy_im = 0.0;
        ridged_noise_eval_base(state, sample_x, hurst, lacunarity, octaves, &fx_re, &fx_im);
        ridged_noise_eval_base(state, sample_y, hurst, lacunarity, octaves, &fy_re, &fy_im);
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            base_re = fx_re + fy_re;
            base_im = fx_im + fy_im;
        } else {
            base_re = fx_re * fy_re - fx_im * fy_im;
            base_im = fx_re * fy_im + fx_im * fy_re;
        }
    } else {
        double u = sim_stimulus_coord_u(&cfg->coord, sample_x, sample_y, 0.0);
        ridged_noise_eval_base(state, u, hurst, lacunarity, octaves, &base_re, &base_im);
    }

    double value = cfg->amplitude * (use_imag ? base_im : base_re);
    value *= scale;
    *out_value = isfinite(value) ? value : 0.0;
    return SIM_RESULT_OK;
}

static SimResult ridged_noise_ir_eval(void*           userdata,
                                      const KernelIR* kernel,
                                      size_t          element_index,
                                      size_t          component,
                                      double*         out_value) {
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRidgedNoiseState* state = (SimStimulusRidgedNoiseState*) userdata;
    return ridged_noise_kernel_value(state, kernel, element_index, false, out_value);
}

static SimResult ridged_noise_ir_eval_imag(void*           userdata,
                                           const KernelIR* kernel,
                                           size_t          element_index,
                                           size_t          component,
                                           double*         out_value) {
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRidgedNoiseState* state = (SimStimulusRidgedNoiseState*) userdata;
    return ridged_noise_kernel_value(state, kernel, element_index, true, out_value);
}

static SimResult ridged_noise_step(void*               state_ptr,
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

    SimStimulusRidgedNoiseState* state = (SimStimulusRidgedNoiseState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRidgedNoiseConfig* cfg = &state->config;
    if (cfg->amplitude == 0.0) {
        return SIM_RESULT_OK;
    }

    SimField* field = sim_context_field(context, cfg->field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (is_complex) {
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    } else {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(double);
    }
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    SimStimulusStaticCacheLayout layout    = { 0 };
    SimResult                    layout_rc = sim_stimulus_static_cache_layout_from_arrays(
        field->layout.shape, field->layout.strides, field->layout.rank, &layout);
    if (layout_rc != SIM_RESULT_OK) {
        return layout_rc;
    }

    size_t width  = layout.extent_x;
    size_t height = layout.extent_y;
    assert(field->layout.rank == 1U || field->layout.rank == 2U);
    assert(width > 0U);
    assert(height > 0U);
    assert(layout.count == count);

    double       hurst      = cfg->hurst;
    double       lacunarity = cfg->lacunarity;
    unsigned int octaves    = cfg->octaves;
    double       scale      = cfg->scale_by_dt ? dt_sub : 1.0;
    double       t          = sim_context_time(context);
    bool         separable  = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

    double*           dst_real    = NULL;
    SimComplexDouble* dst_complex = NULL;
    if (!is_complex) {
        dst_real = (double*) sim_field_data(field);
        if (dst_real == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else {
        dst_complex = sim_field_complex_data(field);
        if (dst_complex == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    if (ridged_noise_can_use_static_cache(state)) {
        SimResult prep = ridged_noise_ensure_static_cache(state, &layout, is_complex);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }

        if (!is_complex) {
            sim_accel_copy_scale_real(state->cache.real, dst_real, count, scale, true);
        } else {
            sim_accel_accumulate_real_to_complex(state->cache.real, dst_complex, count, scale, 0.0);
            sim_accel_accumulate_real_to_complex(state->cache.imag, dst_complex, count, 0.0, scale);
        }
        return SIM_RESULT_OK;
    }

    SimResult prep = ridged_noise_ensure_octaves(state);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    if (sim_stimulus_octave_noise_try_vdsp_rows(SIM_STIMULUS_OCTAVE_NOISE_RIDGED,
                                                &state->vdsp,
                                                &cfg->coord,
                                                state->phases,
                                                cfg->amplitude,
                                                hurst,
                                                lacunarity,
                                                octaves,
                                                scale,
                                                t,
                                                field,
                                                dst_real,
                                                dst_complex,
                                                count)) {
        return SIM_RESULT_OK;
    }

    for (size_t idx = 0U; idx < count; ++idx) {
        size_t    ix       = 0U;
        size_t    iy       = 0U;
        SimResult coord_rc = sim_field_index_to_xy(field, idx, &ix, &iy);
        assert(coord_rc == SIM_RESULT_OK);
        assert(ix < width);
        assert(iy < height);

        double x        = cfg->coord.origin_x + (double) ix * cfg->coord.spacing_x;
        double y        = cfg->coord.origin_y + (double) iy * cfg->coord.spacing_y;
        double sample_x = x - cfg->coord.velocity_x * t;
        double sample_y = y - cfg->coord.velocity_y * t;
        double base_re  = 0.0;
        double base_im  = 0.0;

        if (separable) {
            double fx_re = 0.0;
            double fx_im = 0.0;
            double fy_re = 0.0;
            double fy_im = 0.0;
            ridged_noise_eval_base(state, sample_x, hurst, lacunarity, octaves, &fx_re, &fx_im);
            ridged_noise_eval_base(state, sample_y, hurst, lacunarity, octaves, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double u = sim_stimulus_coord_u(&cfg->coord, sample_x, sample_y, 0.0);
            ridged_noise_eval_base(state, u, hurst, lacunarity, octaves, &base_re, &base_im);
        }

        double value_re = cfg->amplitude * base_re;
        double value_im = cfg->amplitude * base_im;
        if (!is_complex) {
            if (isfinite(value_re)) {
                dst_real[idx] += scale * value_re;
            }
        } else {
            if (isfinite(value_re) && isfinite(value_im)) {
                dst_complex[idx].re += scale * value_re;
                dst_complex[idx].im += scale * value_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_ridged_noise_operator(struct SimContext*                  context,
                                                 const SimStimulusRidgedNoiseConfig* config,
                                                 size_t*                             out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusRidgedNoiseConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_ridged_noise"),
                                     sim_context_operator_count(context));
    }

    ridged_noise_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_ridged_noise",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusRidgedNoiseState* state =
        (SimStimulusRidgedNoiseState*) calloc(1U, sizeof(SimStimulusRidgedNoiseState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config            = local;
    state->phases            = NULL;
    state->allocated_octaves = 0U;
    ridged_noise_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_ridged_noise");

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
    info.abstract_id       = "stimulus_ridged_noise";
    sim_operator_info_set_schema_identity(&info, "stimulus_ridged_noise");
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

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = ridged_noise_step,
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
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_ridged_noise")) {
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

                bool        is_complex  = sim_field_is_complex(field);
                SimIRType   field_type  = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId field_node  = sim_ir_builder_field_ref_typed(builder, 0U, field_type);
                SimIRNodeId sample_node = sim_ir_builder_stateful(
                    builder, ridged_noise_ir_eval, state, "stimulus_ridged_noise");
                if (is_complex && sample_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId sample_im = sim_ir_builder_stateful(
                        builder, ridged_noise_ir_eval_imag, state, "stimulus_ridged_noise_im");
                    if (sample_im != SIM_IR_INVALID_NODE) {
                        sample_node = sim_ir_builder_complex_pack(builder, sample_node, sample_im);
                    }
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
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_TIME + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = ridged_noise_destroy;
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
                                .symbolic      = ridged_noise_symbolic,
                                .destroy       = ridged_noise_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        ridged_noise_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_ridged_noise_config(struct SimContext*            context,
                                           size_t                        operator_index,
                                           SimStimulusRidgedNoiseConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusRidgedNoiseState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusRidgedNoiseState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusRidgedNoiseState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_ridged_noise_update(struct SimContext*                  context,
                                           size_t                              operator_index,
                                           const SimStimulusRidgedNoiseConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusRidgedNoiseState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusRidgedNoiseState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusRidgedNoiseState*) sim_split_state(op);
    }
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusRidgedNoiseConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_ridged_noise"), operator_index);
    }

    ridged_noise_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "stimulus_ridged_noise"), true, local.scale_by_dt);
    state->config            = local;
    state->allocated_octaves = 0U;
    sim_stimulus_static_cache_invalidate(&state->cache);
    ridged_noise_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
