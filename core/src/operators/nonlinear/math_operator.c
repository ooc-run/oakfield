#include "oakfield/operators/nonlinear/math_operator.h"
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

#define MATH_SYMBOLIC_CAPACITY 168
#define MATH_EPS_MIN 1.0e-12

typedef struct SimElementwiseMathOperatorState {
    SimElementwiseMathOperatorConfig config;
    char                             symbolic[MATH_SYMBOLIC_CAPACITY];
} SimElementwiseMathOperatorState;

static const char* math_mode_name(SimElementwiseMathMode mode) {
    switch (mode) {
        case SIM_ELEMENTWISE_MATH_FLOOR:
            return "floor";
        case SIM_ELEMENTWISE_MATH_FRACT:
            return "fract";
        case SIM_ELEMENTWISE_MATH_MOD:
            return "mod";
        case SIM_ELEMENTWISE_MATH_STEP:
            return "step";
        case SIM_ELEMENTWISE_MATH_EQ:
            return "eq";
        case SIM_ELEMENTWISE_MATH_LT:
            return "lt";
        case SIM_ELEMENTWISE_MATH_GT:
            return "gt";
        case SIM_ELEMENTWISE_MATH_SELECT:
            return "select";
        default:
            return "floor";
    }
}

static const char* math_rhs_source_name(SimElementwiseMathRhsSource source) {
    switch (source) {
        case SIM_ELEMENTWISE_MATH_RHS_CONSTANT:
            return "constant";
        case SIM_ELEMENTWISE_MATH_RHS_FIELD:
        default:
            return "field";
    }
}

static void math_normalize_config(SimElementwiseMathOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (config->mode < SIM_ELEMENTWISE_MATH_FLOOR || config->mode > SIM_ELEMENTWISE_MATH_SELECT) {
        config->mode = SIM_ELEMENTWISE_MATH_FLOOR;
    }

    if (config->rhs_source != SIM_ELEMENTWISE_MATH_RHS_FIELD &&
        config->rhs_source != SIM_ELEMENTWISE_MATH_RHS_CONSTANT) {
        config->rhs_source = SIM_ELEMENTWISE_MATH_RHS_FIELD;
    }

    if (!isfinite(config->rhs_constant)) {
        config->rhs_constant = 1.0;
    }
    if (!isfinite(config->threshold)) {
        config->threshold = 0.0;
    }
    if (!isfinite(config->epsilon) || config->epsilon < MATH_EPS_MIN) {
        config->epsilon = 1.0e-6;
    }
    if (!isfinite(config->true_value)) {
        config->true_value = 1.0;
    }
    if (!isfinite(config->false_value)) {
        config->false_value = 0.0;
    }

    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;
}

static void math_refresh_symbolic(SimElementwiseMathOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimElementwiseMathOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "math mode=%s rhs=%s c=%.3g thr=%.3g eps=%.3g t=%.3g f=%.3g",
                    math_mode_name(cfg->mode),
                    math_rhs_source_name(cfg->rhs_source),
                    cfg->rhs_constant,
                    cfg->threshold,
                    cfg->epsilon,
                    cfg->true_value,
                    cfg->false_value);
#else
    (void) state;
#endif
}

static const char* math_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimElementwiseMathOperatorState* state =
        (const SimElementwiseMathOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void math_destroy(void* state_ptr) {
    SimElementwiseMathOperatorState* state = (SimElementwiseMathOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state);
}

static bool math_mode_uses_rhs_operand(SimElementwiseMathMode mode) {
    return mode == SIM_ELEMENTWISE_MATH_MOD || mode == SIM_ELEMENTWISE_MATH_EQ ||
           mode == SIM_ELEMENTWISE_MATH_LT || mode == SIM_ELEMENTWISE_MATH_GT ||
           mode == SIM_ELEMENTWISE_MATH_SELECT;
}

static bool math_integer_mode_supported(SimElementwiseMathMode mode) {
    return mode == SIM_ELEMENTWISE_MATH_MOD || mode == SIM_ELEMENTWISE_MATH_STEP ||
           mode == SIM_ELEMENTWISE_MATH_EQ || mode == SIM_ELEMENTWISE_MATH_LT ||
           mode == SIM_ELEMENTWISE_MATH_GT;
}

static bool math_integer_config_supported(const SimElementwiseMathOperatorConfig* cfg,
                                          const SimField*                         lhs_field,
                                          const SimField*                         rhs_field,
                                          const SimField*                         out_field) {
    SimScalarDomain lhs_domain;
    SimScalarDomain out_domain;
    uint64_t        raw = 0U;

    if (cfg == NULL || lhs_field == NULL || out_field == NULL || sim_field_is_complex(lhs_field)) {
        return false;
    }

    lhs_domain = sim_field_scalar_domain(lhs_field);
    out_domain = sim_field_scalar_domain(out_field);
    if (!sim_operator_domain_is_exact_integer(lhs_domain) ||
        !math_integer_mode_supported(cfg->mode)) {
        return false;
    }

    if (math_mode_uses_rhs_operand(cfg->mode)) {
        if (cfg->rhs_source == SIM_ELEMENTWISE_MATH_RHS_FIELD) {
            if (rhs_field == NULL || sim_field_is_complex(rhs_field) ||
                !sim_scalar_domain_equal(sim_field_scalar_domain(rhs_field), lhs_domain)) {
                return false;
            }
        } else if (!sim_operator_integer_raw_from_double(cfg->rhs_constant, lhs_domain, &raw)) {
            return false;
        }
    }

    if (cfg->mode == SIM_ELEMENTWISE_MATH_STEP &&
        !sim_operator_integer_raw_from_double(cfg->threshold, lhs_domain, &raw)) {
        return false;
    }

    if (cfg->mode == SIM_ELEMENTWISE_MATH_MOD) {
        if (!sim_operator_domain_is_exact_integer(out_domain) ||
            !sim_scalar_domain_equal(out_domain, lhs_domain)) {
            return false;
        }
        return !(cfg->accumulate && cfg->scale_by_dt);
    }

    if (sim_operator_domain_is_exact_integer(out_domain)) {
        uint64_t true_raw  = 0U;
        uint64_t false_raw = 0U;
        if (!sim_operator_integer_raw_from_double(cfg->true_value, out_domain, &true_raw) ||
            !sim_operator_integer_raw_from_double(cfg->false_value, out_domain, &false_raw)) {
            return false;
        }
        return !(cfg->accumulate && cfg->scale_by_dt);
    }

    return sim_operator_field_domain_is_f64_or_c64(out_field);
}

static SimResult
math_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;

    SimElementwiseMathOperatorState* state = (SimElementwiseMathOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimElementwiseMathOperatorConfig* cfg = &state->config;

    SimField*       lhs_field = sim_context_field(context, cfg->lhs_field);
    SimField*       rhs_field = sim_context_field(context, cfg->rhs_field);
    SimField*       out_field = sim_context_field(context, cfg->output_field);
    SimScalarDomain lhs_domain;
    SimScalarDomain out_domain;
    if (lhs_field == NULL || out_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    lhs_domain = sim_field_scalar_domain(lhs_field);
    out_domain = sim_field_scalar_domain(out_field);

    bool lhs_complex = sim_field_is_complex(lhs_field);
    bool out_complex = sim_field_is_complex(out_field);
    if (lhs_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    size_t count = sim_field_element_count(&lhs_field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (sim_field_element_count(&out_field->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_operator_domain_is_exact_integer(lhs_domain)) {
        const void*       lhs_data         = sim_field_data_const(lhs_field);
        const void*       rhs_data         = NULL;
        void*             out_data         = NULL;
        double*           out_real         = NULL;
        SimComplexDouble* out_complex_data = NULL;
        uint64_t          rhs_constant_raw = 0U;
        uint64_t          threshold_raw    = 0U;
        uint64_t          true_raw         = 0U;
        uint64_t          false_raw        = 0U;
        const double      scale            = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

        if (!math_integer_config_supported(cfg, lhs_field, rhs_field, out_field)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
        if (lhs_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (math_mode_uses_rhs_operand(cfg->mode) &&
            cfg->rhs_source == SIM_ELEMENTWISE_MATH_RHS_FIELD) {
            if (sim_field_element_count(&rhs_field->layout) != count) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            rhs_data = sim_field_data_const(rhs_field);
            if (rhs_data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
        } else if (math_mode_uses_rhs_operand(cfg->mode)) {
            (void) sim_operator_integer_raw_from_double(
                cfg->rhs_constant, lhs_domain, &rhs_constant_raw);
        }
        if (cfg->mode == SIM_ELEMENTWISE_MATH_STEP) {
            (void) sim_operator_integer_raw_from_double(cfg->threshold, lhs_domain, &threshold_raw);
        }

        if (sim_operator_domain_is_exact_integer(out_domain)) {
            out_data = sim_field_data(out_field);
            if (out_data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (cfg->mode != SIM_ELEMENTWISE_MATH_MOD) {
                (void) sim_operator_integer_raw_from_double(cfg->true_value, out_domain, &true_raw);
                (void) sim_operator_integer_raw_from_double(
                    cfg->false_value, out_domain, &false_raw);
            }
        } else if (out_complex) {
            out_complex_data = sim_field_complex_data(out_field);
            if (out_complex_data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
        } else {
            out_real = sim_field_real_data(out_field);
            if (out_real == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
        }

        for (size_t i = 0U; i < count; ++i) {
            uint64_t lhs_raw = 0U;
            uint64_t rhs_raw = rhs_constant_raw;
            if (!sim_operator_integer_read(lhs_data, lhs_domain, i, &lhs_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rhs_data != NULL && !sim_operator_integer_read(rhs_data, lhs_domain, i, &rhs_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            if (cfg->mode == SIM_ELEMENTWISE_MATH_MOD) {
                uint64_t value_raw = 0U;
                if (sim_operator_integer_truncate(rhs_raw, lhs_domain) != 0U) {
                    if (lhs_domain.is_signed) {
                        int64_t lhs_value = sim_operator_integer_as_i64(lhs_raw, lhs_domain);
                        int64_t rhs_value = sim_operator_integer_as_i64(rhs_raw, lhs_domain);
                        value_raw         = sim_operator_integer_truncate(
                            (uint64_t) ((rhs_value == -1) ? 0 : (lhs_value % rhs_value)),
                            lhs_domain);
                    } else {
                        uint64_t lhs_value = sim_operator_integer_truncate(lhs_raw, lhs_domain);
                        uint64_t rhs_value = sim_operator_integer_truncate(rhs_raw, lhs_domain);
                        value_raw =
                            sim_operator_integer_truncate(lhs_value % rhs_value, lhs_domain);
                    }
                }
                if (cfg->accumulate) {
                    uint64_t existing_raw = 0U;
                    if (!sim_operator_integer_read(out_data, out_domain, i, &existing_raw)) {
                        return SIM_RESULT_INVALID_ARGUMENT;
                    }
                    value_raw = sim_operator_integer_truncate(existing_raw + value_raw, out_domain);
                }
                if (!sim_operator_integer_write(out_data, out_domain, i, value_raw)) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                continue;
            }

            {
                int  cmp  = 0;
                bool cond = false;
                switch (cfg->mode) {
                    case SIM_ELEMENTWISE_MATH_STEP:
                        cmp  = sim_operator_integer_compare(lhs_raw, threshold_raw, lhs_domain);
                        cond = (cmp >= 0);
                        break;
                    case SIM_ELEMENTWISE_MATH_EQ:
                        cond = sim_operator_integer_compare(lhs_raw, rhs_raw, lhs_domain) == 0;
                        break;
                    case SIM_ELEMENTWISE_MATH_LT:
                        cond = sim_operator_integer_compare(lhs_raw, rhs_raw, lhs_domain) < 0;
                        break;
                    case SIM_ELEMENTWISE_MATH_GT:
                        cond = sim_operator_integer_compare(lhs_raw, rhs_raw, lhs_domain) > 0;
                        break;
                    default:
                        return SIM_RESULT_TYPE_MISMATCH;
                }

                if (out_data != NULL) {
                    uint64_t value_raw = cond ? true_raw : false_raw;
                    if (cfg->accumulate) {
                        uint64_t existing_raw = 0U;
                        if (!sim_operator_integer_read(out_data, out_domain, i, &existing_raw)) {
                            return SIM_RESULT_INVALID_ARGUMENT;
                        }
                        value_raw =
                            sim_operator_integer_truncate(existing_raw + value_raw, out_domain);
                    }
                    if (!sim_operator_integer_write(out_data, out_domain, i, value_raw)) {
                        return SIM_RESULT_INVALID_ARGUMENT;
                    }
                } else {
                    double value = cond ? cfg->true_value : cfg->false_value;
                    if (out_complex_data != NULL) {
                        if (cfg->accumulate) {
                            out_complex_data[i].re += scale * value;
                            out_complex_data[i].im = 0.0;
                        } else {
                            out_complex_data[i].re = value;
                            out_complex_data[i].im = 0.0;
                        }
                    } else {
                        if (cfg->accumulate) {
                            out_real[i] += scale * value;
                        } else {
                            out_real[i] = value;
                        }
                    }
                }
            }
        }

        return SIM_RESULT_OK;
    }

    if (!sim_operator_field_domain_is_f64(lhs_field) ||
        !sim_operator_field_domain_is_f64_or_c64(out_field) ||
        (cfg->rhs_source == SIM_ELEMENTWISE_MATH_RHS_FIELD &&
         !sim_operator_field_domain_is_f64(rhs_field))) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    const double*     lhs              = sim_field_real_data_const(lhs_field);
    double*           out              = NULL;
    SimComplexDouble* out_complex_data = NULL;
    if (out_complex) {
        out_complex_data = sim_field_complex_data(out_field);
    } else {
        out = sim_field_real_data(out_field);
    }
    if (lhs == NULL || (!out && !out_complex_data)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const double* rhs = NULL;
    if (cfg->rhs_source == SIM_ELEMENTWISE_MATH_RHS_FIELD) {
        if (rhs_field == NULL || sim_field_is_complex(rhs_field)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (sim_field_element_count(&rhs_field->layout) != count) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        rhs = sim_field_real_data_const(rhs_field);
        if (rhs == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    const double scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    for (size_t i = 0U; i < count; ++i) {
        double lhs_value = lhs[i];
        double rhs_value =
            (cfg->rhs_source == SIM_ELEMENTWISE_MATH_RHS_FIELD) ? rhs[i] : cfg->rhs_constant;
        double value = 0.0;
        switch (cfg->mode) {
            case SIM_ELEMENTWISE_MATH_FLOOR:
                value = floor(lhs_value);
                break;
            case SIM_ELEMENTWISE_MATH_FRACT:
                value = lhs_value - floor(lhs_value);
                break;
            case SIM_ELEMENTWISE_MATH_MOD:
                if (fabs(rhs_value) <= cfg->epsilon) {
                    value = 0.0;
                } else {
                    value = fmod(lhs_value, rhs_value);
                }
                break;
            case SIM_ELEMENTWISE_MATH_STEP:
                value = (lhs_value >= cfg->threshold) ? cfg->true_value : cfg->false_value;
                break;
            case SIM_ELEMENTWISE_MATH_EQ:
                value = (fabs(lhs_value - rhs_value) <= cfg->epsilon) ? cfg->true_value
                                                                      : cfg->false_value;
                break;
            case SIM_ELEMENTWISE_MATH_LT:
                value = (lhs_value < rhs_value) ? cfg->true_value : cfg->false_value;
                break;
            case SIM_ELEMENTWISE_MATH_GT:
                value = (lhs_value > rhs_value) ? cfg->true_value : cfg->false_value;
                break;
            case SIM_ELEMENTWISE_MATH_SELECT:
                value = (lhs_value >= cfg->threshold) ? rhs_value : cfg->false_value;
                break;
            default:
                value = lhs_value;
                break;
        }

        if (!isfinite(value)) {
            continue;
        }

        if (out_complex) {
            if (cfg->accumulate) {
                out_complex_data[i].re += scale * value;
                out_complex_data[i].im = 0.0;
            } else {
                out_complex_data[i].re = value;
                out_complex_data[i].im = 0.0;
            }
        } else {
            if (cfg->accumulate) {
                out[i] += scale * value;
            } else {
                out[i] = value;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult math_step(void*               state_ptr,
                           struct SimContext*  context,
                           struct SimOperator* self,
                           size_t              substep_index,
                           double              dt_sub,
                           void*               scratch,
                           size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return math_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_elementwise_math_operator(struct SimContext*                      context,
                                            const SimElementwiseMathOperatorConfig* config,
                                            size_t*                                 out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimElementwiseMathOperatorState* state =
        (SimElementwiseMathOperatorState*) calloc(1U, sizeof(SimElementwiseMathOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimElementwiseMathOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.lhs_field    = 0U;
        local.rhs_field    = 0U;
        local.output_field = 0U;
        local.mode         = SIM_ELEMENTWISE_MATH_FLOOR;
        local.rhs_source   = SIM_ELEMENTWISE_MATH_RHS_CONSTANT;
        local.rhs_constant = 1.0;
        local.threshold    = 0.0;
        local.epsilon      = 1.0e-6;
        local.true_value   = 1.0;
        local.false_value  = 0.0;
        local.accumulate   = false;
        local.scale_by_dt  = true;
    }

    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "elementwise_math",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);
    math_normalize_config(&local);
    state->config = local;
    math_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "elementwise_math");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_NONLINEAR;
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
    info.abstract_id       = "elementwise_math";
    sim_operator_info_set_schema_identity(&info, "elementwise_math");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_NONE;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimField*       lhs_field = sim_context_field(context, state->config.lhs_field);
    SimField*       rhs_field = sim_context_field(context, state->config.rhs_field);
    SimField*       out_field = sim_context_field(context, state->config.output_field);
    SimScalarDomain lhs_domain;
    if (lhs_field == NULL || rhs_field == NULL || out_field == NULL) {
        math_destroy(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    lhs_domain = sim_field_scalar_domain(lhs_field);
    if (sim_operator_domain_is_exact_integer(lhs_domain)) {
        if (!math_integer_config_supported(&state->config, lhs_field, rhs_field, out_field)) {
            math_destroy(state);
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else if (!sim_operator_field_domain_is_f64(lhs_field) ||
               !sim_operator_field_domain_is_f64(rhs_field) ||
               !sim_operator_field_domain_is_f64_or_c64(out_field)) {
        math_destroy(state);
        return SIM_RESULT_TYPE_MISMATCH;
    }

    SimSplitPort ports[3] = {
        { .context_field_index = state->config.lhs_field, .require_complex = false },
        { .context_field_index = state->config.rhs_field, .require_complex = false },
        { .context_field_index = state->config.output_field, .require_complex = false }
    };

    SimSplitAccess accesses[3] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_READ },
                                   { .port = 2, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = math_step,
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
                                .symbolic      = math_symbolic,
                                .destroy       = math_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        math_destroy(state);
    }

    return result;
}

SimResult sim_elementwise_math_config(struct SimContext*                context,
                                      size_t                            operator_index,
                                      SimElementwiseMathOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimElementwiseMathOperatorState* state = (SimElementwiseMathOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_elementwise_math_update(struct SimContext*                      context,
                                      size_t                                  operator_index,
                                      const SimElementwiseMathOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimElementwiseMathOperatorState* state = (SimElementwiseMathOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimElementwiseMathOperatorConfig local = state->config;
    if (config != NULL) {
        local             = *config;
        local.scale_by_dt = sim_operator_resolve_scale_by_dt(
            context, "elementwise_math", true, config->scale_by_dt);
    }

    math_normalize_config(&local);
    state->config = local;
    {
        SimField* lhs_field = sim_context_field(context, state->config.lhs_field);
        SimField* rhs_field = sim_context_field(context, state->config.rhs_field);
        SimField* out_field = sim_context_field(context, state->config.output_field);
        if (lhs_field == NULL || rhs_field == NULL || out_field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (sim_operator_domain_is_exact_integer(sim_field_scalar_domain(lhs_field))) {
            if (!math_integer_config_supported(&state->config, lhs_field, rhs_field, out_field)) {
                return SIM_RESULT_TYPE_MISMATCH;
            }
        } else if (!sim_operator_field_domain_is_f64(lhs_field) ||
                   !sim_operator_field_domain_is_f64(rhs_field) ||
                   !sim_operator_field_domain_is_f64_or_c64(out_field)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    }
    math_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
