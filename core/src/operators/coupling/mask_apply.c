#include "oakfield/operators/coupling/mask_apply.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define MASK_SYMBOLIC_CAPACITY 152

typedef struct SimMaskOperatorState {
    SimMaskOperatorConfig config;
    char                  symbolic[MASK_SYMBOLIC_CAPACITY];
} SimMaskOperatorState;

static const char* mask_mode_name(SimMaskMode mode) {
    switch (mode) {
        case SIM_MASK_MODE_INVERT:
            return "invert";
        case SIM_MASK_MODE_APPLY:
        default:
            return "apply";
    }
}

static void mask_normalize_config(SimMaskOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (config->mode != SIM_MASK_MODE_APPLY && config->mode != SIM_MASK_MODE_INVERT) {
        config->mode = SIM_MASK_MODE_APPLY;
    }

    if (!isfinite(config->threshold)) {
        config->threshold = 0.5;
    }
    if (!isfinite(config->feather) || config->feather < 0.0) {
        config->feather = 0.0;
    }
    if (!isfinite(config->fill_value)) {
        config->fill_value = 0.0;
    }
    if (!isfinite(config->fill_value_im)) {
        config->fill_value_im = 0.0;
    }

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static void mask_refresh_symbolic(SimMaskOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimMaskOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "mask mode=%s thr=%.3g feather=%.3g fill=%.3g scale_by_dt=%s",
                    mask_mode_name(cfg->mode),
                    cfg->threshold,
                    cfg->feather,
                    cfg->fill_value,
                    cfg->scale_by_dt ? "true" : "false");
#else
    (void) state;
#endif
}

static const char* mask_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimMaskOperatorState* state = (const SimMaskOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void mask_destroy(void* state_ptr) {
    SimMaskOperatorState* state = (SimMaskOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state);
}

static double mask_smoothstep(double edge0, double edge1, double x) {
    if (edge1 <= edge0) {
        return (x >= edge0) ? 1.0 : 0.0;
    }
    double t = (x - edge0) / (edge1 - edge0);
    if (t <= 0.0) {
        return 0.0;
    }
    if (t >= 1.0) {
        return 1.0;
    }
    return t * t * (3.0 - 2.0 * t);
}

static bool mask_integer_fill_raw(const SimMaskOperatorConfig* cfg,
                                  SimScalarDomain              domain,
                                  uint64_t*                    out_fill_raw) {
    if (cfg == NULL || out_fill_raw == NULL || !sim_operator_domain_is_exact_integer(domain)) {
        return false;
    }
    if (!isfinite(cfg->fill_value_im) || cfg->fill_value_im != 0.0) {
        return false;
    }
    return sim_operator_integer_raw_from_double(cfg->fill_value, domain, out_fill_raw);
}

static bool mask_supports_integer_path(const SimField*              input_field,
                                       const SimField*              output_field,
                                       const SimMaskOperatorConfig* cfg,
                                       uint64_t*                    out_fill_raw) {
    SimScalarDomain input_domain;
    SimScalarDomain output_domain;
    uint64_t        fill_raw = 0U;

    if (input_field == NULL || output_field == NULL || cfg == NULL ||
        sim_field_is_complex(input_field) || sim_field_is_complex(output_field) ||
        cfg->accumulate || cfg->feather != 0.0) {
        return false;
    }

    input_domain  = sim_field_scalar_domain(input_field);
    output_domain = sim_field_scalar_domain(output_field);
    if (!sim_operator_domain_is_exact_integer(input_domain) ||
        !sim_operator_domain_is_exact_integer(output_domain) ||
        !sim_scalar_domain_equal(input_domain, output_domain) ||
        !mask_integer_fill_raw(cfg, output_domain, &fill_raw)) {
        return false;
    }

    if (out_fill_raw != NULL) {
        *out_fill_raw = fill_raw;
    }
    return true;
}

static SimResult mask_validate_fields(const SimField*              input_field,
                                      const SimField*              mask_field,
                                      const SimField*              output_field,
                                      const SimMaskOperatorConfig* cfg) {
    SimScalarDomain input_domain;
    SimScalarDomain output_domain;

    if (input_field == NULL || mask_field == NULL || output_field == NULL || cfg == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_is_complex(input_field) != sim_field_is_complex(output_field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    input_domain  = sim_field_scalar_domain(input_field);
    output_domain = sim_field_scalar_domain(output_field);
    if (sim_operator_domain_is_exact_integer(input_domain) ||
        sim_operator_domain_is_exact_integer(output_domain)) {
        return mask_supports_integer_path(input_field, output_field, cfg, NULL)
                   ? SIM_RESULT_OK
                   : SIM_RESULT_TYPE_MISMATCH;
    }

    return SIM_RESULT_OK;
}

static SimResult
mask_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimMaskOperatorState* state = (SimMaskOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimMaskOperatorConfig* cfg = &state->config;

    SimField* input_field = sim_context_field(context, cfg->input_field);
    SimField* mask_field  = sim_context_field(context, cfg->mask_field);
    SimField* out_field   = sim_context_field(context, cfg->output_field);
    if (input_field == NULL || mask_field == NULL || out_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool input_complex  = sim_field_is_complex(input_field);
    bool output_complex = sim_field_is_complex(out_field);
    if (input_complex != output_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    size_t count = sim_field_element_count(&input_field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }
    if (sim_field_element_count(&out_field->layout) != count ||
        sim_field_element_count(&mask_field->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const double scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    bool                    mask_complex      = sim_field_is_complex(mask_field);
    const SimComplexDouble* mask_complex_data = NULL;
    const double*           mask_real_data    = NULL;
    if (mask_complex) {
        mask_complex_data = sim_field_complex_data_const(mask_field);
        if (mask_complex_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else {
        mask_real_data = sim_field_real_data_const(mask_field);
        if (mask_real_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    {
        uint64_t fill_raw = 0U;
        if (!input_complex && mask_supports_integer_path(input_field, out_field, cfg, &fill_raw)) {
            SimScalarDomain input_domain  = sim_field_scalar_domain(input_field);
            SimScalarDomain output_domain = sim_field_scalar_domain(out_field);
            const void*     input_raw     = sim_field_data_const(input_field);
            void*           out_raw       = sim_field_data(out_field);

            if (input_raw == NULL || out_raw == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            for (size_t i = 0U; i < count; ++i) {
                double   mask_value = 0.0;
                bool     keep       = false;
                uint64_t value_raw  = fill_raw;

                if (mask_complex) {
                    double re  = mask_complex_data[i].re;
                    double im  = mask_complex_data[i].im;
                    mask_value = hypot(re, im);
                } else {
                    mask_value = mask_real_data[i];
                }
                if (!isfinite(mask_value)) {
                    continue;
                }

                keep = (mask_value >= cfg->threshold);
                if (cfg->mode == SIM_MASK_MODE_INVERT) {
                    keep = !keep;
                }
                if (keep && !sim_operator_integer_read(input_raw, input_domain, i, &value_raw)) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                if (!sim_operator_integer_write(out_raw, output_domain, i, value_raw)) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
            }
            return SIM_RESULT_OK;
        }
        if (!input_complex &&
            (sim_operator_domain_is_exact_integer(sim_field_scalar_domain(input_field)) ||
             sim_operator_domain_is_exact_integer(sim_field_scalar_domain(out_field)))) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    }

    if (input_complex) {
        const SimComplexDouble* input = sim_field_complex_data_const(input_field);
        SimComplexDouble*       out   = sim_field_complex_data(out_field);
        if (input == NULL || out == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double mask_value = 0.0;
            if (mask_complex) {
                double re  = mask_complex_data[i].re;
                double im  = mask_complex_data[i].im;
                mask_value = hypot(re, im);
            } else {
                mask_value = mask_real_data[i];
            }
            if (!isfinite(mask_value)) {
                continue;
            }

            double weight = 0.0;
            if (cfg->feather > 0.0) {
                weight = mask_smoothstep(
                    cfg->threshold - cfg->feather, cfg->threshold + cfg->feather, mask_value);
            } else {
                weight = (mask_value >= cfg->threshold) ? 1.0 : 0.0;
            }
            if (cfg->mode == SIM_MASK_MODE_INVERT) {
                weight = 1.0 - weight;
            }

            double re = weight * input[i].re + (1.0 - weight) * cfg->fill_value;
            double im = weight * input[i].im + (1.0 - weight) * cfg->fill_value_im;
            if (!isfinite(re) || !isfinite(im)) {
                continue;
            }

            if (cfg->accumulate) {
                out[i].re += scale * re;
                out[i].im += scale * im;
            } else {
                out[i].re = re;
                out[i].im = im;
            }
        }
    } else {
        const double* input = sim_field_real_data_const(input_field);
        double*       out   = sim_field_real_data(out_field);
        if (input == NULL || out == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double mask_value = 0.0;
            if (mask_complex) {
                double re  = mask_complex_data[i].re;
                double im  = mask_complex_data[i].im;
                mask_value = hypot(re, im);
            } else {
                mask_value = mask_real_data[i];
            }
            if (!isfinite(mask_value)) {
                continue;
            }

            double weight = 0.0;
            if (cfg->feather > 0.0) {
                weight = mask_smoothstep(
                    cfg->threshold - cfg->feather, cfg->threshold + cfg->feather, mask_value);
            } else {
                weight = (mask_value >= cfg->threshold) ? 1.0 : 0.0;
            }
            if (cfg->mode == SIM_MASK_MODE_INVERT) {
                weight = 1.0 - weight;
            }

            double value = weight * input[i] + (1.0 - weight) * cfg->fill_value;
            if (!isfinite(value)) {
                continue;
            }

            if (cfg->accumulate) {
                out[i] += scale * value;
            } else {
                out[i] = value;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult mask_step(void*               state_ptr,
                           struct SimContext*  context,
                           struct SimOperator* self,
                           size_t              substep_index,
                           double              dt_sub,
                           void*               scratch,
                           size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return mask_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_mask_operator(struct SimContext*           context,
                                const SimMaskOperatorConfig* config,
                                size_t*                      out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimMaskOperatorState* state = (SimMaskOperatorState*) calloc(1U, sizeof(SimMaskOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimMaskOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.input_field   = 0U;
        local.mask_field    = 0U;
        local.output_field  = 0U;
        local.mode          = SIM_MASK_MODE_APPLY;
        local.threshold     = 0.5;
        local.feather       = 0.0;
        local.fill_value    = 0.0;
        local.fill_value_im = 0.0;
        local.accumulate    = false;
        local.scale_by_dt   = true;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "mask", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    mask_normalize_config(&local);
    state->config = local;
    mask_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "mask");

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
    info.abstract_id       = "mask";
    sim_operator_info_set_schema_identity(&info, "mask");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;

    SimField* input_field  = sim_context_field(context, state->config.input_field);
    SimField* mask_field   = sim_context_field(context, state->config.mask_field);
    SimField* output_field = sim_context_field(context, state->config.output_field);
    bool      needs_complex =
        (input_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(input_field))) ||
        (output_field && sim_scalar_domain_is_complex(sim_scalar_domain_from_field(output_field)));
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    if (input_field == NULL || mask_field == NULL || output_field == NULL) {
        mask_destroy(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    {
        SimResult validate =
            mask_validate_fields(input_field, mask_field, output_field, &state->config);
        if (validate != SIM_RESULT_OK) {
            mask_destroy(state);
            return validate;
        }
    }

    SimSplitPort ports[3] = {
        { .context_field_index = state->config.input_field, .require_complex = needs_complex },
        { .context_field_index = state->config.mask_field, .require_complex = false },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_READ },
                                   { .port = 2, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = mask_step,
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
                                .symbolic      = mask_symbolic,
                                .destroy       = mask_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        mask_destroy(state);
    }

    return result;
}

SimResult sim_mask_config(struct SimContext*     context,
                          size_t                 operator_index,
                          SimMaskOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMaskOperatorState* state = (SimMaskOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_mask_update(struct SimContext*           context,
                          size_t                       operator_index,
                          const SimMaskOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMaskOperatorState* state = (SimMaskOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimMaskOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context, "mask", true, config->scale_by_dt);
    }

    mask_normalize_config(&local);

    {
        SimField* input_field  = sim_context_field(context, local.input_field);
        SimField* mask_field   = sim_context_field(context, local.mask_field);
        SimField* output_field = sim_context_field(context, local.output_field);
        SimResult validate = mask_validate_fields(input_field, mask_field, output_field, &local);
        if (validate != SIM_RESULT_OK) {
            return validate;
        }
    }
    state->config = local;
    mask_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
