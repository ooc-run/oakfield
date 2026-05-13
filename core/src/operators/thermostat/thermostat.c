#include "oakfield/operators/thermostat/thermostat.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* ----------------------------- State Structure ----------------------------- */
typedef struct ThermostatOperatorState {
    ThermostatOperatorConfig cfg;
    double                   lambda_eff;  /* effective regulated value */
    double                   lambda_prev; /* previous smoothed value */
    double                   energy_last; /* last measured energy */
    double                   snapshot_lambda_eff;
    double                   snapshot_lambda_prev;
    double                   snapshot_energy_last;
    char                     symbolic[128]; /* symbolic summary */
    size_t                   kernel_cached_step_index;
    double                   kernel_cached_dt;
    const SimField*          kernel_cached_field;
    size_t                   kernel_cached_count;
    bool                     kernel_cache_valid;
    size_t                   kernel_cached_element;
    double                   kernel_cached_value[2];
    bool                     kernel_cached_element_valid;
} ThermostatOperatorState;

static void thermostat_normalize_config(ThermostatOperatorConfig* cfg) {
    if (cfg == NULL) {
        return;
    }

    if (cfg->lambda_rebuild_thresh <= 0.0) {
        cfg->lambda_rebuild_thresh = 1e-3;
    }

    if (!isfinite(cfg->lambda_min)) {
        cfg->lambda_min = -DBL_MAX;
    }

    if (!isfinite(cfg->lambda_max)) {
        cfg->lambda_max = DBL_MAX;
    }

    if (!cfg->use_memory_field) {
        cfg->memory_field = (size_t) -1;
    } else if (cfg->memory_field == (size_t) -1) {
        cfg->use_memory_field = false;
    }
}

static void thermostat_refresh_symbolic(ThermostatOperatorState* st) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (st == NULL) {
        return;
    }

    snprintf(st->symbolic,
             sizeof(st->symbolic),
             "λ_base=%.3g target=%.3g gain=%.3g",
             st->cfg.lambda_base,
             st->cfg.E_target,
             st->cfg.lambda_soft_gain);
#else
    (void) st;
#endif
}

static void thermostat_release(void* state_ptr) {
    ThermostatOperatorState* st = (ThermostatOperatorState*) state_ptr;
    free(st);
}

static const char* thermostat_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const ThermostatOperatorState* st = (const ThermostatOperatorState*) state_ptr;
    return st ? st->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

/* Compute mean energy: E = (1/N) * sum |u_i|^2 */
static double thermostat_compute_energy(const SimField* field) {
    if (!field)
        return 0.0;
    if (sim_field_is_complex(field)) {
        const SimComplexDouble* cdata = sim_field_complex_data_const(field);
        size_t                  count = sim_field_element_count(&field->layout);
        if (!cdata || count == 0U)
            return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double re = cdata[i].re;
            double im = cdata[i].im;
            sum += re * re + im * im;
        }
        return sum / (double) count;
    } else {
        if (field->element_size != sizeof(double))
            return 0.0;
        const double* data  = (const double*) sim_field_data((SimField*) field);
        size_t        count = sim_field_bytes((SimField*) field) / sizeof(double);
        if (!data || count == 0U)
            return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double v = data[i];
            sum += v * v;
        }
        return sum / (double) count;
    }
}

static inline double softplus(double x, double k) {
    if (k <= 0.0)
        return x; /* linear when softness disabled */
    /* numerically stable softplus approximation */
    if (x > 50.0 / k)
        return x; /* large x, softplus ~ x */
    return (1.0 / k) * log1p(exp(k * x));
}

static double thermostat_apply_soft_lambda(ThermostatOperatorState* st, double energy) {
    /* lambda_eff = lambda_base + gain * (E - E_target) */
    double raw = st->cfg.lambda_base + st->cfg.lambda_soft_gain * (energy - st->cfg.E_target);
    /* optional soft clamp via softplus to enforce min boundary smoothly */
    if (isfinite(st->cfg.lambda_min)) {
        double shift = raw - st->cfg.lambda_min;
        raw          = st->cfg.lambda_min + softplus(shift, st->cfg.softplus_k);
    }
    if (isfinite(st->cfg.lambda_max)) {
        double shift_max = st->cfg.lambda_max - raw;
        raw              = st->cfg.lambda_max - softplus(shift_max, st->cfg.softplus_k);
    }
    /* smoothing */
    double s        = fmax(0.0, fmin(1.0, st->cfg.lambda_smooth));
    double smoothed = (1.0 - s) * raw + s * st->lambda_prev;
    st->lambda_prev = smoothed;
    return smoothed;
}

static SimResult
thermostat_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;
    ThermostatOperatorState* st = (ThermostatOperatorState*) userdata;
    if (!st)
        return SIM_RESULT_INVALID_ARGUMENT;
    st->snapshot_lambda_eff  = st->lambda_eff;
    st->snapshot_lambda_prev = st->lambda_prev;
    st->snapshot_energy_last = st->energy_last;
    return SIM_RESULT_OK;
}

static SimResult
thermostat_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;
    ThermostatOperatorState* st = (ThermostatOperatorState*) userdata;
    if (!st)
        return SIM_RESULT_INVALID_ARGUMENT;
    st->lambda_eff  = st->snapshot_lambda_eff;
    st->lambda_prev = st->snapshot_lambda_prev;
    st->energy_last = st->snapshot_energy_last;
    return SIM_RESULT_OK;
}

static SimResult
thermostat_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;
    (void) dt;
    ThermostatOperatorState* st = (ThermostatOperatorState*) state_ptr;
    if (!st || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimField* field = sim_context_field(context, st->cfg.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    double energy   = thermostat_compute_energy(field);
    st->energy_last = energy;

    switch (st->cfg.mode) {
        case THERMOSTAT_SOFT_LAMBDA:
            st->lambda_eff = thermostat_apply_soft_lambda(st, energy);
            break;
        case THERMOSTAT_ADD:
        case THERMOSTAT_MULT: {
            /* Apply relaxation directly to the field values.
         * ADD:  u += dt * mu * (M - u^2)
         * MULT: u += dt * (-mu * (u^2 - M) * u)
         * M: per-point memory field if provided, else global E_target.
         */
            if (dt <= 0.0) {
                st->lambda_eff = st->cfg.lambda_base;
                break;
            }

            const double mu = fmax(0.0, st->cfg.mu);
            if (mu <= 0.0) {
                st->lambda_eff = st->cfg.lambda_base;
                break;
            }

            bool is_complex = sim_field_is_complex(field);

            double*           u  = NULL;
            SimComplexDouble* uc = NULL;
            size_t            count;
            if (is_complex) {
                uc    = sim_field_complex_data(field);
                count = sim_field_element_count(&field->layout);
            } else {
                u     = (double*) sim_field_data(field);
                count = sim_field_bytes(field) / sizeof(double);
            }
            const double  global_M = st->cfg.E_target;
            const bool    use_M    = st->cfg.use_memory_field;
            const size_t  midx     = st->cfg.memory_field;
            const double* M_ptr    = NULL;
            if (use_M) {
                SimField* Mfield = sim_context_field(context, midx);
                if (Mfield && Mfield->element_size == sizeof(double) &&
                    sim_field_bytes(Mfield) / sizeof(double) >= count) {
                    M_ptr = (const double*) sim_field_data(Mfield);
                }
            }

            if (!is_complex) {
                if (st->cfg.mode == THERMOSTAT_ADD) {
                    for (size_t i = 0; i < count; ++i) {
                        double ui  = u[i];
                        double Mi  = M_ptr ? M_ptr[i] : global_M;
                        double rhs = mu * (Mi - ui * ui);
                        u[i]       = ui + dt * rhs;
                    }
                } else /* THERMOSTAT_MULT */
                {
                    for (size_t i = 0; i < count; ++i) {
                        double ui  = u[i];
                        double Mi  = M_ptr ? M_ptr[i] : global_M;
                        double rhs = -mu * (ui * ui - Mi) * ui;
                        u[i]       = ui + dt * rhs;
                    }
                }
            } else {
                /* Complex-valued regulation uses |u|^2 and acts radially */
                const double eps = 1e-12;
                if (st->cfg.mode == THERMOSTAT_ADD) {
                    for (size_t i = 0; i < count; ++i) {
                        double re = uc[i].re, im = uc[i].im;
                        double mag2 = re * re + im * im;
                        double Mi   = M_ptr ? M_ptr[i] : global_M;
                        double rhs  = mu * (Mi - mag2);
                        double mag  = sqrt(mag2);
                        double nx   = (mag > eps) ? (re / mag) : 1.0;
                        double ny   = (mag > eps) ? (im / mag) : 0.0;
                        /* Move along current phase direction */
                        uc[i].re = re + dt * rhs * nx;
                        uc[i].im = im + dt * rhs * ny;
                    }
                } else /* THERMOSTAT_MULT */
                {
                    for (size_t i = 0; i < count; ++i) {
                        double re = uc[i].re, im = uc[i].im;
                        double mag2 = re * re + im * im;
                        double Mi   = M_ptr ? M_ptr[i] : global_M;
                        double s    = -mu * (mag2 - Mi);
                        /* u += dt * s * u */
                        uc[i].re = re + dt * s * re;
                        uc[i].im = im + dt * s * im;
                    }
                }
            }

            st->lambda_eff = st->cfg.lambda_base;
            break;
        }
        case THERMOSTAT_NONE:
        default:
            st->lambda_eff = st->cfg.lambda_base;
            break;
    }

#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    snprintf(st->symbolic,
             sizeof(st->symbolic),
             "E=%.3g lambda=%.3g (mode=%d)",
             st->energy_last,
             st->lambda_eff,
             (int) st->cfg.mode);
#endif

    return SIM_RESULT_OK;
}

static SimResult thermostat_step(void*               state_ptr,
                                 struct SimContext*  context,
                                 struct SimOperator* self,
                                 size_t              substep_index,
                                 double              dt,
                                 void*               scratch,
                                 size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return thermostat_apply(state_ptr, context, self, dt);
}

static double thermostat_kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult thermostat_ir_eval(void*           userdata,
                                    const KernelIR* kernel,
                                    size_t          element_index,
                                    size_t          component,
                                    double*         out_value) {
    if (out_value == NULL || userdata == NULL || kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ThermostatOperatorState* st = (ThermostatOperatorState*) userdata;
    if (kernel->bindings == NULL || kernel->binding_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimKernelIRBinding* binding = &kernel->bindings[0];
    SimField*                 field   = binding->field;
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex      = sim_field_is_complex(field);
    size_t component_count = is_complex ? 2U : 1U;
    size_t element_count   = sim_field_element_count(&field->layout);
    if (component >= component_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (element_count == 0U || element_index >= element_count) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    double dt = thermostat_kernel_param_value(kernel, SIM_IR_PARAM_DT);

    size_t step_index = 0U;
    bool   have_step =
        (kernel->params != NULL && kernel->param_count > (size_t) SIM_IR_PARAM_STEP_INDEX);
    if (have_step) {
        double step_value = kernel->params[SIM_IR_PARAM_STEP_INDEX];
        if (isfinite(step_value) && step_value >= 0.0) {
            step_index = (size_t) step_value;
        } else {
            have_step = false;
        }
    }

    bool need_update = !st->kernel_cache_valid || st->kernel_cached_field != field ||
                       st->kernel_cached_count != element_count || st->kernel_cached_dt != dt ||
                       !have_step || st->kernel_cached_step_index != step_index;

    if (need_update) {
        double energy   = thermostat_compute_energy(field);
        st->energy_last = energy;

        switch (st->cfg.mode) {
            case THERMOSTAT_SOFT_LAMBDA:
                st->lambda_eff = thermostat_apply_soft_lambda(st, energy);
                break;
            case THERMOSTAT_ADD:
            case THERMOSTAT_MULT:
            case THERMOSTAT_NONE:
            default:
                st->lambda_eff = st->cfg.lambda_base;
                break;
        }

#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
        (void) snprintf(st->symbolic,
                        sizeof(st->symbolic),
                        "E=%.3g lambda=%.3g (mode=%d)",
                        st->energy_last,
                        st->lambda_eff,
                        (int) st->cfg.mode);
#endif

        st->kernel_cached_field         = field;
        st->kernel_cached_count         = element_count;
        st->kernel_cached_dt            = dt;
        st->kernel_cached_step_index    = step_index;
        st->kernel_cache_valid          = have_step;
        st->kernel_cached_element_valid = false;
    }

    if (!st->kernel_cached_element_valid || st->kernel_cached_element != element_index) {
        const double global_M = st->cfg.E_target;
        const double mu       = fmax(0.0, st->cfg.mu);
        const bool   do_update =
            (st->cfg.mode == THERMOSTAT_ADD || st->cfg.mode == THERMOSTAT_MULT) && (dt > 0.0) &&
            (mu > 0.0);

        const double* memory_data = NULL;
        if (st->cfg.use_memory_field && kernel->binding_count > 1U) {
            const SimKernelIRBinding* memory_binding = &kernel->bindings[1];
            const SimField*           memory_field   = memory_binding->field;
            if (memory_field != NULL && memory_field->element_size == sizeof(double)) {
                size_t mem_count = sim_field_bytes((SimField*) memory_field) / sizeof(double);
                if (mem_count > element_index) {
                    memory_data = sim_field_real_data_const(memory_field);
                }
            }
        }

        if (!is_complex) {
            const double* u = sim_field_real_data_const(field);
            if (u == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            double value = u[element_index];
            if (do_update) {
                double Mi = memory_data ? memory_data[element_index] : global_M;
                if (st->cfg.mode == THERMOSTAT_ADD) {
                    double rhs = mu * (Mi - value * value);
                    value      = value + dt * rhs;
                } else {
                    double rhs = -mu * (value * value - Mi) * value;
                    value      = value + dt * rhs;
                }
            }
            st->kernel_cached_value[0] = value;
        } else {
            const SimComplexDouble* u = sim_field_complex_data_const(field);
            if (u == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            SimComplexDouble value = u[element_index];
            if (do_update) {
                double Mi = memory_data ? memory_data[element_index] : global_M;
                if (st->cfg.mode == THERMOSTAT_ADD) {
                    const double eps  = 1e-12;
                    double       re   = value.re;
                    double       im   = value.im;
                    double       mag2 = re * re + im * im;
                    double       rhs  = mu * (Mi - mag2);
                    double       mag  = sqrt(mag2);
                    double       nx   = (mag > eps) ? (re / mag) : 1.0;
                    double       ny   = (mag > eps) ? (im / mag) : 0.0;
                    value.re          = re + dt * rhs * nx;
                    value.im          = im + dt * rhs * ny;
                } else {
                    double re   = value.re;
                    double im   = value.im;
                    double mag2 = re * re + im * im;
                    double s    = -mu * (mag2 - Mi);
                    value.re    = re + dt * s * re;
                    value.im    = im + dt * s * im;
                }
            }
            st->kernel_cached_value[0] = value.re;
            st->kernel_cached_value[1] = value.im;
        }

        st->kernel_cached_element       = element_index;
        st->kernel_cached_element_valid = true;
    }

    *out_value = st->kernel_cached_value[component];
    return SIM_RESULT_OK;
}

SimResult sim_add_thermostat_operator(struct SimContext*              context,
                                      const ThermostatOperatorConfig* config,
                                      size_t*                         out_index) {
    if (!context)
        return SIM_RESULT_INVALID_ARGUMENT;

    ThermostatOperatorState* st =
        (ThermostatOperatorState*) calloc(1U, sizeof(ThermostatOperatorState));
    if (!st)
        return SIM_RESULT_OUT_OF_MEMORY;

    ThermostatOperatorConfig local = { 0 };
    if (config)
        local = *config;

    thermostat_normalize_config(&local);

    st->cfg                         = local;
    st->lambda_prev                 = local.lambda_base;
    st->lambda_eff                  = local.lambda_base;
    st->kernel_cached_step_index    = 0U;
    st->kernel_cached_dt            = 0.0;
    st->kernel_cached_field         = NULL;
    st->kernel_cached_count         = 0U;
    st->kernel_cache_valid          = false;
    st->kernel_cached_element       = 0U;
    st->kernel_cached_element_valid = false;
    st->kernel_cached_value[0]      = 0.0;
    st->kernel_cached_value[1]      = 0.0;

    thermostat_refresh_symbolic(st);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "thermostat");

    SimField*       field         = sim_context_field(context, local.field_index);
    bool            needs_complex = field != NULL && sim_field_is_complex(field);
    SimOperatorInfo info          = sim_operator_info_defaults();
    info.category                 = SIM_OPERATOR_CATEGORY_THERMOSTAT;
    info.warp_level               = SIM_WARP_LEVEL_NONE;
    info.is_noise                 = false;
    info.is_spectral              = false;
    info.is_local                 = true;
    info.is_nonlocal              = true;
    info.is_linear                = false;
    info.is_warp                  = false;
    info.is_differentiable        = false;
    info.preserves_real           = true;
    info.preferred_dt             = 0.0;
    info.abstract_id              = "thermostat";
    sim_operator_info_set_schema_identity(&info, "thermostat");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "thermostat")) {
        bool   is_complex    = needs_complex;
        size_t expected_size = is_complex ? sizeof(SimComplexDouble) : sizeof(double);
        if (field != NULL && field->element_size == expected_size) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[2];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc   = { 0 };
                size_t                             binding_count = 1U;

                bindings[0].ir_field_index      = 0U;
                bindings[0].context_field_index = local.field_index;

                if (local.use_memory_field) {
                    SimField* memory_field = sim_context_field(context, local.memory_field);
                    if (memory_field != NULL && memory_field->element_size == sizeof(double)) {
                        bindings[1].ir_field_index      = 1U;
                        bindings[1].context_field_index = local.memory_field;
                        binding_count                   = 2U;
                    }
                }

                SimIRStatefulSpec spec = { 0 };
                spec.eval              = thermostat_ir_eval;
                spec.userdata          = st;
                spec.label             = "thermostat";
                spec.value_type        = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                SimIRNodeId node       = sim_ir_builder_stateful_spec(builder, &spec);

                if (node != SIM_IR_INVALID_NODE) {
                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = node;

                    kernel_desc.builder           = builder;
                    kernel_desc.bindings          = bindings;
                    kernel_desc.binding_count     = binding_count;
                    kernel_desc.outputs           = outputs;
                    kernel_desc.output_count      = 1U;
                    kernel_desc.param_count       = (size_t) SIM_IR_PARAM_STEP_INDEX + 1U;
                    kernel_desc.required_features = 0ULL;

                    SimOperatorDescriptor kdesc = { 0 };
                    kdesc.name                  = name;
                    kdesc.evaluate              = NULL;
                    kdesc.destroy               = thermostat_release;
                    kdesc.userdata              = st;
                    kdesc.kernel                = &kernel_desc;
                    kdesc.info                  = info;
                    kdesc.config                = op_config;
                    kdesc.save_state            = thermostat_save;
                    kdesc.restore_state         = thermostat_restore;
                    if (local.field_index < 64U) {
                        kdesc.read_mask |= (1ULL << local.field_index);
                        kdesc.write_mask |= (1ULL << local.field_index);
                    }
                    if (binding_count > 1U && local.memory_field < 64U) {
                        kdesc.read_mask |= (1ULL << local.memory_field);
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

    SimSplitPort port = { .context_field_index = st->cfg.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = thermostat_step,
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
                                .state         = st,
                                .symbolic      = thermostat_symbolic,
                                .save_state    = thermostat_save,
                                .restore_state = thermostat_restore,
                                .destroy       = thermostat_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK)
        thermostat_release(st);
    return result;
}

SimResult sim_thermostat_config(struct SimContext*        context,
                                size_t                    operator_index,
                                ThermostatOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    ThermostatOperatorState* st = (ThermostatOperatorState*) sim_operator_state(op);
    if (st == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = st->cfg;
    return SIM_RESULT_OK;
}

SimResult sim_thermostat_update(struct SimContext*              context,
                                size_t                          operator_index,
                                const ThermostatOperatorConfig* config) {
    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    ThermostatOperatorState* st = (ThermostatOperatorState*) sim_operator_state(op);
    if (st == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    ThermostatOperatorConfig local = *config;
    thermostat_normalize_config(&local);

    st->cfg         = local;
    st->lambda_prev = local.lambda_base;
    st->lambda_eff  = local.lambda_base;
    thermostat_refresh_symbolic(st);
    st->kernel_cached_field         = NULL;
    st->kernel_cached_count         = 0U;
    st->kernel_cached_dt            = 0.0;
    st->kernel_cached_step_index    = 0U;
    st->kernel_cache_valid          = false;
    st->kernel_cached_element_valid = false;

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
