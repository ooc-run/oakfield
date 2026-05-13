#include "oakfield/operators/stimulus/fbm.h"
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

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_FBM_EPS 1.0e-9
#define STIM_FBM_MAX_OCTAVES 16U

typedef struct SimStimulusFbmState {
    SimStimulusFbmConfig              config;
    double*                           phases;
    unsigned int                      allocated_octaves;
    SimStimulusStaticCache            cache;
    SimStimulusOctaveNoiseVdspBuffers vdsp;
    char                              symbolic[160];
} SimStimulusFbmState;

static void fbm_normalize(SimStimulusFbmConfig* config) {
    if (config == NULL) {
        return;
    }
    if (!isfinite(config->amplitude))
        config->amplitude = 0.0;
    if (!isfinite(config->hurst) || config->hurst <= 0.0 || config->hurst >= 1.0)
        config->hurst = 0.5;
    if (!isfinite(config->lacunarity) || config->lacunarity < 1.0)
        config->lacunarity = 2.0;

    sim_stimulus_coord_normalize(&config->coord);

    if (config->octaves == 0U)
        config->octaves = 4U;
    if (config->octaves > STIM_FBM_MAX_OCTAVES)
        config->octaves = STIM_FBM_MAX_OCTAVES;
    if (config->seed == 0ULL)
        config->seed = 1ULL;
}

static void fbm_refresh_symbolic(SimStimulusFbmState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    const SimStimulusFbmConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "fbm A=%.3g H=%.3g O=%u",
                    cfg->amplitude,
                    cfg->hurst,
                    cfg->octaves);
#else
    (void) state;
#endif
}

static const char* fbm_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusFbmState* state = (const SimStimulusFbmState*) state_ptr;
    return state != NULL ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void fbm_eval_base(const SimStimulusFbmState* state,
                          double                     u,
                          double                     hurst,
                          double                     lacunarity,
                          unsigned int               octaves,
                          double*                    out_re,
                          double*                    out_im) {
    sim_stimulus_octave_noise_eval_base(SIM_STIMULUS_OCTAVE_NOISE_FBM,
                                        state != NULL ? state->phases : NULL,
                                        u,
                                        hurst,
                                        lacunarity,
                                        octaves,
                                        out_re,
                                        out_im);
}

static void fbm_destroy(void* state_ptr) {
    SimStimulusFbmState* state = (SimStimulusFbmState*) state_ptr;
    if (state == NULL) {
        return;
    }
    sim_stimulus_static_cache_destroy(&state->cache);
    sim_stimulus_octave_noise_vdsp_release(&state->vdsp);
    free(state->phases);
    free(state);
}

static SimResult fbm_ensure_octaves(SimStimulusFbmState* state) {
    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    unsigned int desired = state->config.octaves;
    if (desired == 0U)
        return SIM_RESULT_OK;

    return sim_stimulus_octave_noise_ensure_phases(
        &state->phases, &state->allocated_octaves, desired, state->config.seed);
}

static bool fbm_can_use_static_cache(const SimStimulusFbmState* state) {
    return state != NULL && sim_stimulus_coord_is_time_invariant(&state->config.coord);
}

static SimResult fbm_fill_static_cache(void*                               userdata,
                                       const SimStimulusStaticCacheLayout* layout,
                                       bool                                need_imag,
                                       double*                             out_real,
                                       double*                             out_imag) {
    SimStimulusFbmState* state = (SimStimulusFbmState*) userdata;
    if (state == NULL || layout == NULL || out_real == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimStimulusFbmConfig* cfg        = &state->config;
    double                      hurst      = cfg->hurst;
    double                      lacunarity = cfg->lacunarity;
    unsigned int                octaves    = cfg->octaves;
    bool                        separable  = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

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
            fbm_eval_base(state, sample_x, hurst, lacunarity, octaves, &fx_re, &fx_im);
            fbm_eval_base(state, sample_y, hurst, lacunarity, octaves, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double u = sim_stimulus_coord_u(&cfg->coord, sample_x, sample_y, 0.0);
            fbm_eval_base(state, u, hurst, lacunarity, octaves, &base_re, &base_im);
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

static SimResult fbm_ensure_static_cache(SimStimulusFbmState*                state,
                                         const SimStimulusStaticCacheLayout* layout,
                                         bool                                need_imag) {
    if (state == NULL || layout == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimResult prep = fbm_ensure_octaves(state);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    return sim_stimulus_static_cache_ensure(
        &state->cache, layout, need_imag, fbm_fill_static_cache, state);
}

static double kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult fbm_kernel_value(SimStimulusFbmState* state,
                                  const KernelIR*      kernel,
                                  size_t               element_index,
                                  bool                 use_sin,
                                  double*              out_value) {
    if (out_value == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusFbmConfig* cfg = &state->config;
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

    if (fbm_can_use_static_cache(state)) {
        SimResult prep = fbm_ensure_static_cache(state, &layout, use_sin);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }
        double value =
            use_sin ? state->cache.imag[element_index] : state->cache.real[element_index];
        value *= scale;
        *out_value = isfinite(value) ? value : 0.0;
        return SIM_RESULT_OK;
    }

    SimResult prep = fbm_ensure_octaves(state);
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
    double       H          = cfg->hurst;
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
        fbm_eval_base(state, sample_x, H, lacunarity, octaves, &fx_re, &fx_im);
        fbm_eval_base(state, sample_y, H, lacunarity, octaves, &fy_re, &fy_im);
        if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
            base_re = fx_re + fy_re;
            base_im = fx_im + fy_im;
        } else {
            base_re = fx_re * fy_re - fx_im * fy_im;
            base_im = fx_re * fy_im + fx_im * fy_re;
        }
    } else {
        double u = sim_stimulus_coord_u(&cfg->coord, sample_x, sample_y, 0.0);
        fbm_eval_base(state, u, H, lacunarity, octaves, &base_re, &base_im);
    }

    double value = cfg->amplitude * (use_sin ? base_im : base_re);
    value *= scale;
    *out_value = isfinite(value) ? value : 0.0;
    return SIM_RESULT_OK;
}

static SimResult fbm_ir_eval(void*           userdata,
                             const KernelIR* kernel,
                             size_t          element_index,
                             size_t          component,
                             double*         out_value) {
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusFbmState* state = (SimStimulusFbmState*) userdata;
    return fbm_kernel_value(state, kernel, element_index, false, out_value);
}

static SimResult fbm_ir_eval_imag(void*           userdata,
                                  const KernelIR* kernel,
                                  size_t          element_index,
                                  size_t          component,
                                  double*         out_value) {
    (void) component;

    if (out_value == NULL || userdata == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusFbmState* state = (SimStimulusFbmState*) userdata;
    return fbm_kernel_value(state, kernel, element_index, true, out_value);
}

static SimResult fbm_step(void*               state_ptr,
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

    SimStimulusFbmState* state = (SimStimulusFbmState*) state_ptr;
    if (state == NULL || context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusFbmConfig* cfg = &state->config;
    if (cfg->amplitude == 0.0)
        return SIM_RESULT_OK;

    SimField* field = sim_context_field(context, cfg->field_index);
    if (field == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (is_complex) {
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    } else {
        if (field->element_size != sizeof(double))
            return SIM_RESULT_INVALID_ARGUMENT;
        count = sim_field_bytes(field) / sizeof(double);
    }
    if (count == 0U)
        return SIM_RESULT_OK;

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

    double       H          = cfg->hurst;
    double       lacunarity = cfg->lacunarity;
    unsigned int octaves    = cfg->octaves;
    double       scale      = cfg->scale_by_dt ? dt_sub : 1.0;
    double       t          = sim_context_time(context);
    bool         separable  = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

    double*           dst_real    = NULL;
    SimComplexDouble* dst_complex = NULL;
    if (!is_complex) {
        dst_real = (double*) sim_field_data(field);
        if (dst_real == NULL)
            return SIM_RESULT_INVALID_ARGUMENT;
    } else {
        dst_complex = sim_field_complex_data(field);
        if (dst_complex == NULL)
            return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (fbm_can_use_static_cache(state)) {
        SimResult prep = fbm_ensure_static_cache(state, &layout, is_complex);
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

    SimResult prep = fbm_ensure_octaves(state);
    if (prep != SIM_RESULT_OK)
        return prep;

    if (sim_stimulus_octave_noise_try_vdsp_rows(SIM_STIMULUS_OCTAVE_NOISE_FBM,
                                                &state->vdsp,
                                                &cfg->coord,
                                                state->phases,
                                                cfg->amplitude,
                                                H,
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
            fbm_eval_base(state, sample_x, H, lacunarity, octaves, &fx_re, &fx_im);
            fbm_eval_base(state, sample_y, H, lacunarity, octaves, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double u = sim_stimulus_coord_u(&cfg->coord, sample_x, sample_y, 0.0);
            fbm_eval_base(state, u, H, lacunarity, octaves, &base_re, &base_im);
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

SimResult sim_add_stimulus_fbm_operator(struct SimContext*          context,
                                        const SimStimulusFbmConfig* config,
                                        size_t*                     out_index) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimStimulusFbmConfig local = { 0 };
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(sim_context_seed(context),
                                     sim_seed_tag("stimulus_fbm"),
                                     sim_context_operator_count(context));
    }

    fbm_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "stimulus_fbm", (config != NULL), (config != NULL) ? config->scale_by_dt : true);

    SimStimulusFbmState* state = (SimStimulusFbmState*) calloc(1U, sizeof(SimStimulusFbmState));
    if (state == NULL)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config            = local;
    state->phases            = NULL;
    state->allocated_octaves = 0U;
    fbm_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_fbm");

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
    info.abstract_id       = "stimulus_fbm";
    sim_operator_info_set_schema_identity(&info, "stimulus_fbm");
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
                                .fn                = fbm_step,
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
            context, &op_config, 0ULL, SIM_DET_NONE, "stimulus_fbm")) {
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
                    sim_ir_builder_stateful(builder, fbm_ir_eval, state, "stimulus_fbm");
                if (is_complex && sample_node != SIM_IR_INVALID_NODE) {
                    SimIRNodeId sample_im = sim_ir_builder_stateful(
                        builder, fbm_ir_eval_imag, state, "stimulus_fbm_im");
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
                    kdesc.destroy               = fbm_destroy;
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
                                .symbolic      = fbm_symbolic,
                                .destroy       = fbm_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        fbm_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_fbm_config(struct SimContext*    context,
                                  size_t                operator_index,
                                  SimStimulusFbmConfig* out_config) {
    if (context == NULL || out_config == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusFbmState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusFbmState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusFbmState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_fbm_update(struct SimContext*          context,
                                  size_t                      operator_index,
                                  const SimStimulusFbmConfig* config) {
    if (context == NULL)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL)
        return SIM_RESULT_NOT_FOUND;

    SimStimulusFbmState* state = NULL;
    if (op->kernel != NULL) {
        state = (SimStimulusFbmState*) sim_operator_payload(op);
    } else {
        state = (SimStimulusFbmState*) sim_split_state(op);
    }
    if (state == NULL)
        return SIM_RESULT_INVALID_STATE;

    SimStimulusFbmConfig local = state->config;
    if (config != NULL)
        local = *config;

    if (local.seed == 0ULL) {
        local.seed = sim_seed_derive(
            sim_context_seed(context), sim_seed_tag("stimulus_fbm"), operator_index);
    }

    fbm_normalize(&local);
    state->config            = local;
    state->allocated_octaves = 0U;
    sim_stimulus_static_cache_invalidate(&state->cache);
    fbm_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
