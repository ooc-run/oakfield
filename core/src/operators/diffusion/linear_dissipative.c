#include "oakfield/operators/diffusion/linear_dissipative.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/common/fft_plan.h"

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

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static double linear_dissipative_axis_index(size_t index, size_t count) {
    return (index <= count / 2U) ? (double) index : -((double) (count - index));
}

static double linear_dissipative_resolve_spacing(double spacing) {
    return (spacing > 0.0 && isfinite(spacing)) ? spacing : 1.0;
}

static SimResult linear_dissipative_describe_field(const SimField*         field,
                                                   SimFieldRepresentation* out_repr,
                                                   bool*                   out_needs_complex,
                                                   bool*                   out_imag_zero) {
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFieldRepresentation repr          = sim_field_representation(field);
    const bool             needs_complex = sim_field_is_complex(field);
    const bool             imag_zero     = sim_field_representation_has_imag_zero_constraint(repr);

    if (needs_complex) {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else if (field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (out_repr != NULL) {
        *out_repr = repr;
    }
    if (out_needs_complex != NULL) {
        *out_needs_complex = needs_complex;
    }
    if (out_imag_zero != NULL) {
        *out_imag_zero = imag_zero;
    }

    return SIM_RESULT_OK;
}

static void linear_dissipative_fill_info(SimOperatorInfo*       info,
                                         SimFieldRepresentation repr,
                                         bool                   needs_complex) {
    if (info == NULL) {
        return;
    }

    *info                   = sim_operator_info_defaults();
    info->category          = SIM_OPERATOR_CATEGORY_DIFFUSION;
    info->warp_level        = SIM_WARP_LEVEL_NONE;
    info->is_noise          = false;
    info->is_spectral       = true;
    info->is_local          = false;
    info->is_nonlocal       = true;
    info->is_linear         = true;
    info->is_warp           = false;
    info->is_differentiable = true;
    info->preserves_real    = true;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "linear_dissipative";
    sim_operator_info_set_schema_identity(info, "linear_dissipative");
    info->algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info->representation.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    info->representation.value_kind =
        (repr.value_kind != SIM_FIELD_VALUE_UNKNOWN)
            ? repr.value_kind
            : (needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR);
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = true;
}

typedef struct LinearDissipativeOperatorState {
    LinearDissipativeOperatorConfig config;
    double*                         lambda;
    double complex*                 time_buffer;
    double complex*                 freq_buffer;
    FFTPlan                         plan;
    FFTPlan2D                       plan2d;
    size_t                          plan_rank;
    size_t                          plan_shape[2];
    size_t                          capacity;
    bool                            lambda_dirty;
    size_t                          cached_count;
    size_t                          cached_rank;
    size_t                          cached_shape[2];
    size_t                          cached_step_index;
    double                          cached_dt;
    SimFieldDomain                  cached_domain;
    const SimField*                 cached_field;
    bool                            cache_valid;
    char                            symbolic[160];
} LinearDissipativeOperatorState;

static SimResult linear_dissipative_step(void*               state_ptr,
                                         struct SimContext*  context,
                                         struct SimOperator* self,
                                         size_t              substep_index,
                                         double              dt_sub,
                                         void*               scratch,
                                         size_t              scratch_size);

static void linear_dissipative_release(void* state_ptr) {
    LinearDissipativeOperatorState* state = (LinearDissipativeOperatorState*) state_ptr;
    if (!state) {
        return;
    }

    free(state->lambda);
    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->time_buffer);
    free(state->freq_buffer);
    free(state);
}

static const char* linear_dissipative_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const LinearDissipativeOperatorState* state = (const LinearDissipativeOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult linear_dissipative_ensure_capacity(LinearDissipativeOperatorState* state,
                                                    const SimFieldLayout*           layout) {
    double* lambda;

    if (!state) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!layout || layout->shape == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (layout->rank == 0U || layout->rank > 2U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    size_t count = sim_field_element_count(layout);
    size_t rank  = layout->rank;
    size_t rows  = layout->shape[0];
    size_t cols  = (rank == 2U) ? layout->shape[1] : 0U;

    const bool shape_match =
        (state->plan_rank == rank) &&
        (rank == 1U ? state->plan_shape[0] == rows
                    : (state->plan_shape[0] == rows && state->plan_shape[1] == cols));

    if (count == 0U) {
        free(state->lambda);
        fft_plan_destroy(&state->plan);
        fft_plan2d_destroy(&state->plan2d);
        free(state->time_buffer);
        free(state->freq_buffer);
        state->time_buffer   = NULL;
        state->freq_buffer   = NULL;
        state->lambda        = NULL;
        state->capacity      = 0U;
        state->plan_rank     = 0U;
        state->plan_shape[0] = 0U;
        state->plan_shape[1] = 0U;
        state->lambda_dirty  = true;
        return SIM_RESULT_OK;
    }

    if (state->capacity == count && shape_match) {
        return SIM_RESULT_OK;
    }

    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->time_buffer);
    free(state->freq_buffer);
    state->time_buffer = NULL;
    state->freq_buffer = NULL;

    (void) posix_memalign((void**) &lambda, 64U, count * sizeof(double));

    if (!lambda)
        lambda = (double*) calloc(count, sizeof(double));
    else
        (void) memset(lambda, 0, count * sizeof(double));

    if (!lambda) {
        free(lambda);
        state->capacity = 0U;
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    double complex* time_buffer = (double complex*) calloc(count, sizeof(double complex));
    double complex* freq_buffer = (double complex*) calloc(count, sizeof(double complex));
    if (time_buffer == NULL || freq_buffer == NULL) {
        free(lambda);
        free(time_buffer);
        free(freq_buffer);
        state->capacity = 0U;
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimResult plan_rc = SIM_RESULT_OK;
    if (rank == 1U) {
        plan_rc = fft_plan_init(&state->plan, count);
    } else {
        plan_rc = fft_plan2d_init(&state->plan2d, rows, cols, cols, 1U);
    }
    if (plan_rc != SIM_RESULT_OK) {
        free(lambda);
        free(time_buffer);
        free(freq_buffer);
        state->capacity = 0U;
        return plan_rc;
    }

    free(state->lambda);

    state->lambda        = lambda;
    state->time_buffer   = time_buffer;
    state->freq_buffer   = freq_buffer;
    state->capacity      = count;
    state->plan_rank     = rank;
    state->plan_shape[0] = rows;
    state->plan_shape[1] = cols;
    state->lambda_dirty  = true;
    return SIM_RESULT_OK;
}

static void linear_dissipative_update_lambda(LinearDissipativeOperatorState* state,
                                             const SimFieldLayout*           layout) {
    if (!state || !layout || layout->shape == NULL) {
        return;
    }
    if (layout->rank == 0U || layout->rank > 2U) {
        return;
    }
    if (state->lambda == NULL) {
        return;
    }

    size_t rows = layout->shape[0];
    size_t cols = (layout->rank == 2U) ? layout->shape[1] : 0U;

    double spacing_x = linear_dissipative_resolve_spacing(state->config.spacing);
    double spacing_y = spacing_x;
    double base_k    = 0.0;
    double base_kx   = 0.0;
    double base_ky   = 0.0;

    if (layout->rank == 1U) {
        double length = spacing_x * (double) rows;
        base_k        = (rows > 0U && length > 0.0) ? (2.0 * M_PI / length) : 0.0;
        for (size_t i = 0U; i < rows; ++i) {
            double freq_index = linear_dissipative_axis_index(i, rows);
            double k_abs      = fabs(freq_index * base_k);
            double value      = 0.0;

            if (k_abs > 0.0 && state->config.alpha > 0.0) {
                value = -pow(k_abs, state->config.alpha);
            }

            state->lambda[i] = state->config.viscosity * value;
        }
    } else {
        double length_x = spacing_x * (double) cols;
        double length_y = spacing_y * (double) rows;
        base_kx         = (cols > 0U && length_x > 0.0) ? (2.0 * M_PI / length_x) : 0.0;
        base_ky         = (rows > 0U && length_y > 0.0) ? (2.0 * M_PI / length_y) : 0.0;
        for (size_t y = 0U; y < rows; ++y) {
            double ky_index = linear_dissipative_axis_index(y, rows);
            double ky       = ky_index * base_ky;
            size_t row_base = y * cols;
            for (size_t x = 0U; x < cols; ++x) {
                double kx_index = linear_dissipative_axis_index(x, cols);
                double kx       = kx_index * base_kx;
                double k_abs    = sqrt(kx * kx + ky * ky);
                double value    = 0.0;
                if (k_abs > 0.0 && state->config.alpha > 0.0) {
                    value = -pow(k_abs, state->config.alpha);
                }
                state->lambda[row_base + x] = state->config.viscosity * value;
            }
        }
    }

    state->lambda_dirty = false;
}

static double linear_dissipative_kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult linear_dissipative_ir_eval(void*           userdata,
                                            const KernelIR* kernel,
                                            size_t          element_index,
                                            size_t          component,
                                            double*         out_value) {
    if (out_value == NULL || userdata == NULL || kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    LinearDissipativeOperatorState* state = (LinearDissipativeOperatorState*) userdata;
    if (kernel->bindings == NULL || kernel->binding_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimKernelIRBinding* binding = &kernel->bindings[0];
    SimField*                 field   = binding->field;
    if (field == NULL || !sim_field_is_complex(field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (component > 1U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimFieldLayout* layout = &field->layout;
    size_t                count  = sim_field_element_count(layout);
    if (count == 0U || element_index >= count) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }
    if (layout->rank == 0U || layout->rank > 2U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }
    size_t rank    = layout->rank;
    size_t rows    = layout->shape[0];
    size_t cols    = (rank == 2U) ? layout->shape[1] : 0U;
    size_t stride0 = layout->strides[0];
    size_t stride1 = (rank == 2U) ? layout->strides[1] : 0U;

    SimFieldRepresentation repr      = sim_field_representation(field);
    const bool             imag_zero = sim_field_representation_has_imag_zero_constraint(repr);
    double                 dt = linear_dissipative_kernel_param_value(kernel, SIM_IR_PARAM_DT);
    if (!isfinite(dt) || dt < 0.0) {
        dt = 0.0;
    }

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

    bool need_update = !state->cache_valid || state->cached_field != field ||
                       state->cached_count != count || state->cached_domain != repr.domain ||
                       state->cached_dt != dt || !have_step ||
                       state->cached_step_index != step_index || state->lambda_dirty ||
                       state->cached_rank != rank || state->cached_shape[0] != rows ||
                       (rank == 2U && state->cached_shape[1] != cols);

    if (need_update) {
        SimResult rc = linear_dissipative_ensure_capacity(state, layout);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        SimComplexDouble* cdata = sim_field_complex_data(field);
        if (cdata == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (state->lambda_dirty) {
            linear_dissipative_update_lambda(state, layout);
        }

        const bool apply_decay = (state->config.viscosity != 0.0 && dt != 0.0);

        if (!apply_decay) {
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx            = i * stride0;
                    state->time_buffer[i] = CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t out_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx = row_base + x * stride1;
                        state->time_buffer[out_base + x] =
                            CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                    }
                }
            }
        } else if (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL) {
            const double* restrict lambda = state->lambda;
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    double factor = exp(dt * lambda[i]);
                    state->time_buffer[i] =
                        CMPLX(cdata[idx].re * factor, imag_zero ? 0.0 : (cdata[idx].im * factor));
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t out_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx                       = row_base + x * stride1;
                        double factor                    = exp(dt * lambda[out_base + x]);
                        state->time_buffer[out_base + x] = CMPLX(
                            cdata[idx].re * factor, imag_zero ? 0.0 : (cdata[idx].im * factor));
                    }
                }
            }
        } else {
            for (size_t y = 0U; y < rows; ++y) {
                size_t row_base = y * stride0;
                size_t out_base = y * cols;
                for (size_t x = 0U; x < cols; ++x) {
                    size_t idx = row_base + x * stride1;
                    state->time_buffer[out_base + x] =
                        CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                }
            }
            if (rank == 1U) {
                rc = fft_plan_forward(&state->plan, state->time_buffer, state->freq_buffer);
            } else {
                rc = fft_plan2d_forward(&state->plan2d, state->time_buffer, state->freq_buffer);
            }
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
            const double* restrict lambda = state->lambda;
            for (size_t i = 0U; i < count; ++i) {
                double factor = exp(dt * lambda[i]);
                state->freq_buffer[i] *= factor;
            }
            if (rank == 1U) {
                rc = fft_plan_inverse(&state->plan, state->freq_buffer, state->time_buffer);
            } else {
                rc = fft_plan2d_inverse(&state->plan2d, state->freq_buffer, state->time_buffer);
            }
            if (rc != SIM_RESULT_OK) {
                return rc;
            }
        }

        state->cached_field      = field;
        state->cached_count      = count;
        state->cached_rank       = rank;
        state->cached_shape[0]   = rows;
        state->cached_shape[1]   = cols;
        state->cached_domain     = repr.domain;
        state->cached_dt         = dt;
        state->cached_step_index = step_index;
        state->cache_valid       = have_step;
    }

    if (!state->time_buffer) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double complex value = state->time_buffer[element_index];
    if (imag_zero && component == 1U) {
        *out_value = 0.0;
    } else {
        *out_value = (component == 0U) ? creal(value) : cimag(value);
    }
    return SIM_RESULT_OK;
}

SimResult sim_add_linear_dissipative_operator(struct SimContext*                     context,
                                              const LinearDissipativeOperatorConfig* config,
                                              size_t*                                out_index) {
    LinearDissipativeOperatorState* state;
    LinearDissipativeOperatorConfig local;
    char                            name[SIM_OPERATOR_NAME_MAX + 1U];

    SimResult result = SIM_RESULT_OK;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void) memset(&local, 0, sizeof(local));
    if (config != NULL) {
        local = *config;
    }

    if (local.alpha <= 0.0) {
        local.alpha = 2.0;
    }
    if (local.spacing <= 0.0) {
        local.spacing = 1.0;
    }

    state = (LinearDissipativeOperatorState*) calloc(1U, sizeof(LinearDissipativeOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config            = local;
    state->lambda            = NULL;
    state->capacity          = 0U;
    state->plan_rank         = 0U;
    state->plan_shape[0]     = 0U;
    state->plan_shape[1]     = 0U;
    state->lambda_dirty      = true;
    state->cached_count      = 0U;
    state->cached_rank       = 0U;
    state->cached_shape[0]   = 0U;
    state->cached_shape[1]   = 0U;
    state->cached_step_index = 0U;
    state->cached_dt         = 0.0;
    state->cached_domain     = SIM_FIELD_DOMAIN_PHYSICAL;
    state->cached_field      = NULL;
    state->cache_valid       = false;

#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "-%.3g * (-Laplacian)^{%.3g/2}",
                    local.viscosity,
                    local.alpha);
#endif

    sim_operator_make_unique_name(name, sizeof(name), "linear_dissipative");

    SimField*              field         = sim_context_field(context, local.field_index);
    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    SimResult field_rc = linear_dissipative_describe_field(field, &repr, &needs_complex, NULL);
    if (field_rc != SIM_RESULT_OK) {
        linear_dissipative_release(state);
        return field_rc;
    }

    SimOperatorInfo info = sim_operator_info_defaults();
    linear_dissipative_fill_info(&info, repr, needs_complex);

    SimOperatorConfig op_config = sim_operator_config_defaults();

    bool registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "linear_dissipative")) {
        if (field != NULL && needs_complex && field->element_size == sizeof(SimComplexDouble)) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRStatefulSpec spec = { 0 };
                spec.eval              = linear_dissipative_ir_eval;
                spec.userdata          = state;
                spec.label             = "linear_dissipative";
                spec.value_type        = sim_ir_type_complex();
                SimIRNodeId diss_node  = sim_ir_builder_stateful_spec(builder, &spec);

                if (diss_node != SIM_IR_INVALID_NODE) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = diss_node;

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
                    kdesc.destroy               = linear_dissipative_release;
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

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = linear_dissipative_step,
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
                                .symbolic      = linear_dissipative_symbolic,
                                .destroy       = linear_dissipative_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        linear_dissipative_release(state);
    }

    return result;
}

SimResult sim_linear_dissipative_config(struct SimContext*               context,
                                        size_t                           operator_index,
                                        LinearDissipativeOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    LinearDissipativeOperatorState* state =
        (LinearDissipativeOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_linear_dissipative_update(struct SimContext*                     context,
                                        size_t                                 operator_index,
                                        const LinearDissipativeOperatorConfig* config) {
    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    LinearDissipativeOperatorState* state =
        (LinearDissipativeOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    LinearDissipativeOperatorConfig local = state->config;
    if (config) {
        local = *config;
    }

    if (local.alpha <= 0.0) {
        local.alpha = 2.0;
    }
    if (local.spacing <= 0.0) {
        local.spacing = 1.0;
    }

    state->config       = local;
    state->lambda_dirty = true;
    state->cache_valid  = false;

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}

SimResult linear_dissipative_apply(void*               state_ptr,
                                   struct SimContext*  context,
                                   struct SimOperator* self,
                                   double              dt) {
    LinearDissipativeOperatorState* state = (LinearDissipativeOperatorState*) state_ptr;
    SimField*                       field;
    size_t                          count;
    (void) self;
    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    if (state->config.viscosity == 0.0)
        return SIM_RESULT_OK;

    /* --- Fetch the field --- */
    field = sim_context_field(context, state->config.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   imag_zero     = false;
    SimResult              field_rc =
        linear_dissipative_describe_field(field, &repr, &needs_complex, &imag_zero);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }

    const SimFieldLayout* layout = &field->layout;
    count                        = sim_field_element_count(layout);
    if (count == 0U)
        return SIM_RESULT_OK;
    if (layout->rank == 0U || layout->rank > 2U)
        return SIM_RESULT_NOT_SUPPORTED;

    size_t rank    = layout->rank;
    size_t rows    = layout->shape[0];
    size_t cols    = (rank == 2U) ? layout->shape[1] : 0U;
    size_t stride0 = layout->strides[0];
    size_t stride1 = (rank == 2U) ? layout->strides[1] : 0U;

    /* --- Ensure operator buffers --- */
    SimResult result = linear_dissipative_ensure_capacity(state, layout);
    if (result != SIM_RESULT_OK)
        return result;
    if (state->lambda == NULL || state->time_buffer == NULL || state->freq_buffer == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (state->lambda_dirty)
        linear_dissipative_update_lambda(state, layout);

    if (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL) {
        /* Spectral domain: apply damping directly to Fourier coefficients. */
        const double* restrict lambda = state->lambda;
        double dt_local               = dt;
        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    double factor = exp(dt_local * lambda[i]);
                    cdata[idx].re *= factor;
                    cdata[idx].im = imag_zero ? 0.0 : (cdata[idx].im * factor);
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base    = y * stride0;
                    size_t lambda_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx    = row_base + x * stride1;
                        double factor = exp(dt_local * lambda[lambda_base + x]);
                        cdata[idx].re *= factor;
                        cdata[idx].im = imag_zero ? 0.0 : (cdata[idx].im * factor);
                    }
                }
            }
        } else {
            double* data = sim_field_real_data(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    double factor = exp(dt_local * lambda[i]);
                    data[idx] *= factor;
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base    = y * stride0;
                    size_t lambda_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx    = row_base + x * stride1;
                        double factor = exp(dt_local * lambda[lambda_base + x]);
                        data[idx] *= factor;
                    }
                }
            }
        }
    } else {
        /* Physical domain fallback: transform to spectral, apply damping, and transform back. */
        double complex* restrict time_buffer = state->time_buffer;
        double complex* restrict freq_buffer = state->freq_buffer;
        const double* restrict lambda        = state->lambda;

        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx     = i * stride0;
                    time_buffer[i] = CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                }
                result = fft_plan_forward(&state->plan, time_buffer, freq_buffer);
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t out_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx = row_base + x * stride1;
                        time_buffer[out_base + x] =
                            CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                    }
                }
                result = fft_plan2d_forward(&state->plan2d, time_buffer, freq_buffer);
            }
        } else {
            double* data = sim_field_real_data(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx     = i * stride0;
                    time_buffer[i] = CMPLX(data[idx], 0.0);
                }
                result = fft_plan_forward(&state->plan, time_buffer, freq_buffer);
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t out_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx                = row_base + x * stride1;
                        time_buffer[out_base + x] = CMPLX(data[idx], 0.0);
                    }
                }
                result = fft_plan2d_forward(&state->plan2d, time_buffer, freq_buffer);
            }
        }
        if (result != SIM_RESULT_OK) {
            return result;
        }
        for (size_t i = 0U; i < count; ++i) {
            double factor = exp(dt * lambda[i]); /* rate-based decay */
            freq_buffer[i] *= factor;
        }
        if (rank == 1U) {
            result = fft_plan_inverse(&state->plan, freq_buffer, time_buffer);
        } else {
            result = fft_plan2d_inverse(&state->plan2d, freq_buffer, time_buffer);
        }
        if (result != SIM_RESULT_OK) {
            return result;
        }
        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    cdata[idx].re = creal(time_buffer[i]);
                    cdata[idx].im = imag_zero ? 0.0 : cimag(time_buffer[i]);
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t in_base  = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx    = row_base + x * stride1;
                        cdata[idx].re = creal(time_buffer[in_base + x]);
                        cdata[idx].im = imag_zero ? 0.0 : cimag(time_buffer[in_base + x]);
                    }
                }
            }
        } else {
            double* data = sim_field_real_data(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx = i * stride0;
                    data[idx]  = creal(time_buffer[i]);
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t in_base  = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx = row_base + x * stride1;
                        data[idx]  = creal(time_buffer[in_base + x]);
                    }
                }
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult linear_dissipative_step(void*               state_ptr,
                                         struct SimContext*  context,
                                         struct SimOperator* self,
                                         size_t              substep_index,
                                         double              dt_sub,
                                         void*               scratch,
                                         size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return linear_dissipative_apply(state_ptr, context, self, dt_sub);
}
