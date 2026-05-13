#include "oakfield/operators/diffusion/dispersion.h"
#include "operators/common/operator_utils.h"
#include "oakfield/operators/common/fft_plan.h"
#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct DispersionState {
    DispersionOperatorConfig config;
    size_t                   capacity;
    FFTPlan                  plan;
    FFTPlan2D                plan2d;
    size_t                   plan_rank;
    size_t                   plan_shape[2];
    double complex*          time_buffer;
    double complex*          freq_buffer;
    size_t                   cached_count;
    size_t                   cached_rank;
    size_t                   cached_shape[2];
    size_t                   cached_step_index;
    double                   cached_dt;
    SimFieldDomain           cached_domain;
    const SimField*          cached_field;
    bool                     cache_valid;
    char                     symbolic[128];
} DispersionState;

static SimResult dispersion_describe_field(const SimField*         field,
                                           SimFieldRepresentation* out_repr,
                                           bool*                   out_needs_complex,
                                           bool*                   out_project_real) {
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimFieldRepresentation repr          = sim_field_representation(field);
    const bool             needs_complex = sim_field_is_complex(field);
    const bool             project_real =
        !needs_complex || sim_field_representation_has_spectral_real_constraint(repr);

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
    if (out_project_real != NULL) {
        *out_project_real = project_real;
    }

    return SIM_RESULT_OK;
}

static bool dispersion_has_phase_term(const DispersionOperatorConfig* cfg) {
    return cfg != NULL && cfg->coefficient != 0.0;
}

static SimFieldValueKind dispersion_output_value_kind(SimFieldRepresentation repr,
                                                      bool                   needs_complex,
                                                      const DispersionState* state) {
    if (repr.value_kind == SIM_FIELD_VALUE_UNKNOWN) {
        return needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    }
    if (sim_field_representation_has_imag_zero_constraint(repr) &&
        dispersion_has_phase_term(&state->config)) {
        return SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }
    return repr.value_kind;
}

static void dispersion_fill_info(SimOperatorInfo*       info,
                                 SimFieldRepresentation repr,
                                 bool                   needs_complex,
                                 bool                   project_real,
                                 const DispersionState* state) {
    if (info == NULL || state == NULL) {
        return;
    }

    const SimFieldValueKind value_kind = dispersion_output_value_kind(repr, needs_complex, state);

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
    info->preserves_real    = project_real || state->config.coefficient == 0.0;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "dispersion";
    sim_operator_info_set_schema_identity(info, "dispersion");
    info->algebraic_flags                                = SIM_OPERATOR_ALG_LINEAR;
    info->representation.domain                          = SIM_FIELD_DOMAIN_SPECTRAL;
    info->representation.value_kind                      = value_kind;
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = info->preserves_real;
}

/* Dataflow (1-D flatten):
 *   field (SimComplexDouble) -> copy to time_buffer -> FFT -> freq_buffer -> phase multiply -> IFFT -> copy back
 * Ownership: this state owns FFTPlan + time/freq buffers; rebuilt when element count changes.
 */
static void dispersion_release(void* state_ptr) {
    DispersionState* state = (DispersionState*) state_ptr;
    if (!state) {
        return;
    }
    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->time_buffer);
    free(state->freq_buffer);
    free(state);
}

static const char* dispersion_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const DispersionState* state = (const DispersionState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult dispersion_ensure_capacity(DispersionState* state, const SimFieldLayout* layout) {
    if (!state || !layout || layout->shape == NULL) {
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

    if (state->capacity == count && shape_match) {
        return SIM_RESULT_OK;
    }

    fft_plan_destroy(&state->plan);
    fft_plan2d_destroy(&state->plan2d);
    free(state->time_buffer);
    free(state->freq_buffer);
    state->time_buffer   = NULL;
    state->freq_buffer   = NULL;
    state->capacity      = 0U;
    state->plan_rank     = 0U;
    state->plan_shape[0] = 0U;
    state->plan_shape[1] = 0U;

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    SimResult plan_rc = SIM_RESULT_OK;
    if (rank == 1U) {
        plan_rc = fft_plan_init(&state->plan, count);
    } else {
        plan_rc = fft_plan2d_init(&state->plan2d, rows, cols, cols, 1U);
    }
    if (plan_rc != SIM_RESULT_OK) {
        return plan_rc;
    }

    state->time_buffer = (double complex*) calloc(count, sizeof(double complex));
    state->freq_buffer = (double complex*) calloc(count, sizeof(double complex));
    if (state->time_buffer == NULL || state->freq_buffer == NULL) {
        free(state->time_buffer);
        free(state->freq_buffer);
        state->time_buffer = NULL;
        state->freq_buffer = NULL;
        fft_plan_destroy(&state->plan);
        fft_plan2d_destroy(&state->plan2d);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->capacity      = count;
    state->plan_rank     = rank;
    state->plan_shape[0] = rows;
    state->plan_shape[1] = cols;
    return SIM_RESULT_OK;
}

static void dispersion_refresh_symbolic(DispersionState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state)
        return;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "dispersion coeff=%.3g order=%.3g k0=%.3g",
                    state->config.coefficient,
                    state->config.order,
                    state->config.reference_k);
#else
    (void) state;
#endif
}

static double dispersion_axis_index(size_t index, size_t count) {
    return (index <= count / 2U) ? (double) index : -((double) (count - index));
}

static double dispersion_resolve_spacing(double spacing) {
    return (spacing > 0.0 && isfinite(spacing)) ? spacing : 1.0;
}

static double dispersion_kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult dispersion_ir_eval(void*           userdata,
                                    const KernelIR* kernel,
                                    size_t          element_index,
                                    size_t          component,
                                    double*         out_value) {
    if (out_value == NULL || userdata == NULL || kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    DispersionState* state = (DispersionState*) userdata;
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

    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult field_rc = dispersion_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }
    const bool imag_zero = sim_field_representation_has_imag_zero_constraint(repr);
    double     dt        = dispersion_kernel_param_value(kernel, SIM_IR_PARAM_DT);
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

    bool need_update =
        !state->cache_valid || state->cached_field != field || state->cached_count != count ||
        state->cached_domain != repr.domain || state->cached_dt != dt || !have_step ||
        state->cached_step_index != step_index || state->cached_rank != rank ||
        state->cached_shape[0] != rows || (rank == 2U && state->cached_shape[1] != cols);

    if (need_update) {
        SimResult rc = dispersion_ensure_capacity(state, layout);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
        if (state->time_buffer == NULL || state->freq_buffer == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        SimComplexDouble* cdata = sim_field_complex_data(field);
        if (cdata == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        double spacing_x = dispersion_resolve_spacing(state->config.spacing);
        double spacing_y = spacing_x;
        double base_k    = 0.0;
        double base_kx   = 0.0;
        double base_ky   = 0.0;

        if (rank == 1U) {
            double length = spacing_x * (double) rows;
            base_k        = (rows > 0U && length > 0.0) ? (2.0 * M_PI / length) : 0.0;
        } else {
            double length_x = spacing_x * (double) cols;
            double length_y = spacing_y * (double) rows;
            base_kx         = (cols > 0U && length_x > 0.0) ? (2.0 * M_PI / length_x) : 0.0;
            base_ky         = (rows > 0U && length_y > 0.0) ? (2.0 * M_PI / length_y) : 0.0;
        }

        if (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL) {
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx        = i * stride0;
                    double freq_index = dispersion_axis_index(i, rows);
                    double k_abs      = fabs(freq_index * base_k);
                    double k_shift    = fabs(k_abs - state->config.reference_k);
                    double phase =
                        dt * state->config.coefficient * pow(k_shift, state->config.order);
                    double re = cdata[idx].re;
                    double im = imag_zero ? 0.0 : cdata[idx].im;
                    if (project_real) {
                        double scale          = cos(phase);
                        state->time_buffer[i] = CMPLX(re * scale, im * scale);
                    } else {
                        double s              = sin(phase);
                        double c              = cos(phase);
                        state->time_buffer[i] = CMPLX(re * c - im * s, re * s + im * c);
                    }
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    double ky_index = dispersion_axis_index(y, rows);
                    double ky       = ky_index * base_ky;
                    size_t row_base = y * stride0;
                    size_t out_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        double kx_index = dispersion_axis_index(x, cols);
                        double kx       = kx_index * base_kx;
                        double k_abs    = sqrt(kx * kx + ky * ky);
                        double k_shift  = fabs(k_abs - state->config.reference_k);
                        double phase =
                            dt * state->config.coefficient * pow(k_shift, state->config.order);
                        size_t idx = row_base + x * stride1;
                        double re  = cdata[idx].re;
                        double im  = imag_zero ? 0.0 : cdata[idx].im;
                        if (project_real) {
                            double scale                     = cos(phase);
                            state->time_buffer[out_base + x] = CMPLX(re * scale, im * scale);
                        } else {
                            double s = sin(phase);
                            double c = cos(phase);
                            state->time_buffer[out_base + x] =
                                CMPLX(re * c - im * s, re * s + im * c);
                        }
                    }
                }
            }
        } else {
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx            = i * stride0;
                    state->time_buffer[i] = CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                }
                rc = fft_plan_forward(&state->plan, state->time_buffer, state->freq_buffer);
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
                rc = fft_plan2d_forward(&state->plan2d, state->time_buffer, state->freq_buffer);
            }
            if (rc != SIM_RESULT_OK) {
                return rc;
            }

            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    double freq_index = dispersion_axis_index(i, rows);
                    double k_abs      = fabs(freq_index * base_k);
                    double k_shift    = fabs(k_abs - state->config.reference_k);
                    double phase =
                        dt * state->config.coefficient * pow(k_shift, state->config.order);
                    if (project_real) {
                        state->freq_buffer[i] *= cos(phase);
                    } else {
                        double         s            = sin(phase);
                        double         c            = cos(phase);
                        double complex phase_factor = c + I * s;
                        state->freq_buffer[i] *= phase_factor;
                    }
                }
                rc = fft_plan_inverse(&state->plan, state->freq_buffer, state->time_buffer);
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    double ky_index = dispersion_axis_index(y, rows);
                    double ky       = ky_index * base_ky;
                    size_t row_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        double kx_index = dispersion_axis_index(x, cols);
                        double kx       = kx_index * base_kx;
                        double k_abs    = sqrt(kx * kx + ky * ky);
                        double k_shift  = fabs(k_abs - state->config.reference_k);
                        double phase =
                            dt * state->config.coefficient * pow(k_shift, state->config.order);
                        if (project_real) {
                            state->freq_buffer[row_base + x] *= cos(phase);
                        } else {
                            double complex phase_factor = cexp(I * phase);
                            state->freq_buffer[row_base + x] *= phase_factor;
                        }
                    }
                }
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
    *out_value           = (component == 0U) ? creal(value) : cimag(value);
    return SIM_RESULT_OK;
}

static SimResult
dispersion_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;
    DispersionState* state = (DispersionState*) state_ptr;
    if (!state || !context)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimField* field = sim_context_field(context, state->config.field_index);
    if (!field)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult field_rc = dispersion_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }
    const bool imag_zero = sim_field_representation_has_imag_zero_constraint(repr);

    double                dt_safe = isfinite(dt) ? dt : 0.0;
    const SimFieldLayout* layout  = &field->layout;
    size_t                count   = sim_field_element_count(layout);
    if (count == 0U)
        return SIM_RESULT_OK;
    if (state->config.coefficient == 0.0)
        return SIM_RESULT_OK;
    if (layout->rank == 0U || layout->rank > 2U)
        return SIM_RESULT_NOT_SUPPORTED;

    if (imag_zero) {
        SimFieldRepresentation updated_repr = repr;
        updated_repr.value_kind = dispersion_output_value_kind(repr, needs_complex, state);
        if (updated_repr.value_kind != repr.value_kind) {
            SimResult repr_rc = sim_field_set_representation(field, updated_repr);
            if (repr_rc != SIM_RESULT_OK) {
                return repr_rc;
            }
            repr = updated_repr;
        }
    }

    size_t rank    = layout->rank;
    size_t rows    = layout->shape[0];
    size_t cols    = (rank == 2U) ? layout->shape[1] : 0U;
    size_t stride0 = layout->strides[0];
    size_t stride1 = (rank == 2U) ? layout->strides[1] : 0U;

    double spacing_x = dispersion_resolve_spacing(state->config.spacing);
    double spacing_y = spacing_x;
    double base_k    = 0.0;
    double base_kx   = 0.0;
    double base_ky   = 0.0;
    if (rank == 1U) {
        double length = spacing_x * (double) rows;
        base_k        = (rows > 0U && length > 0.0) ? (2.0 * M_PI / length) : 0.0;
    } else {
        double length_x = spacing_x * (double) cols;
        double length_y = spacing_y * (double) rows;
        base_kx         = (cols > 0U && length_x > 0.0) ? (2.0 * M_PI / length_x) : 0.0;
        base_ky         = (rows > 0U && length_y > 0.0) ? (2.0 * M_PI / length_y) : 0.0;
    }

    if (repr.domain == SIM_FIELD_DOMAIN_SPECTRAL) {
        /* Spectral domain: apply phase rotation directly to Fourier coefficients. */
        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx        = i * stride0;
                    double freq_index = dispersion_axis_index(i, rows);
                    double k_abs      = fabs(freq_index * base_k);
                    double k_shift    = fabs(k_abs - state->config.reference_k);
                    double phase =
                        dt_safe * state->config.coefficient * pow(k_shift, state->config.order);
                    if (project_real) {
                        double scale = cos(phase);
                        cdata[idx].re *= scale;
                        cdata[idx].im = (imag_zero ? 0.0 : cdata[idx].im) * scale;
                    } else {
                        double complex phase_factor = cexp(I * phase);
                        double complex coeff =
                            CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im) * phase_factor;
                        cdata[idx].re = creal(coeff);
                        cdata[idx].im = cimag(coeff);
                    }
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    double ky_index = dispersion_axis_index(y, rows);
                    double ky       = ky_index * base_ky;
                    size_t row_base = y * stride0;
                    for (size_t x = 0U; x < cols; ++x) {
                        double kx_index = dispersion_axis_index(x, cols);
                        double kx       = kx_index * base_kx;
                        double k_abs    = sqrt(kx * kx + ky * ky);
                        double k_shift  = fabs(k_abs - state->config.reference_k);
                        double phase =
                            dt_safe * state->config.coefficient * pow(k_shift, state->config.order);
                        size_t idx = row_base + x * stride1;
                        if (project_real) {
                            double scale = cos(phase);
                            cdata[idx].re *= scale;
                            cdata[idx].im = (imag_zero ? 0.0 : cdata[idx].im) * scale;
                        } else {
                            double complex phase_factor = cexp(I * phase);
                            double complex coeff =
                                CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im) *
                                phase_factor;
                            cdata[idx].re = creal(coeff);
                            cdata[idx].im = cimag(coeff);
                        }
                    }
                }
            }
        }
    } else {
        /* Physical domain fallback: FFT -> phase multiply -> IFFT. */
        SimResult rc = dispersion_ensure_capacity(state, layout);
        if (rc != SIM_RESULT_OK)
            return rc;
        if (state->time_buffer == NULL || state->freq_buffer == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx            = i * stride0;
                    state->time_buffer[i] = CMPLX(cdata[idx].re, imag_zero ? 0.0 : cdata[idx].im);
                }
                rc = fft_plan_forward(&state->plan, state->time_buffer, state->freq_buffer);
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
                rc = fft_plan2d_forward(&state->plan2d, state->time_buffer, state->freq_buffer);
            }
        } else {
            double* data = sim_field_real_data(field);
            if (data == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx            = i * stride0;
                    state->time_buffer[i] = CMPLX(data[idx], 0.0);
                }
                rc = fft_plan_forward(&state->plan, state->time_buffer, state->freq_buffer);
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t out_base = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx                       = row_base + x * stride1;
                        state->time_buffer[out_base + x] = CMPLX(data[idx], 0.0);
                    }
                }
                rc = fft_plan2d_forward(&state->plan2d, state->time_buffer, state->freq_buffer);
            }
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        if (rank == 1U) {
            for (size_t i = 0U; i < rows; ++i) {
                double freq_index = dispersion_axis_index(i, rows);
                double k_abs      = fabs(freq_index * base_k);
                double k_shift    = fabs(k_abs - state->config.reference_k);
                double phase =
                    dt_safe * state->config.coefficient * pow(k_shift, state->config.order);
                if (project_real) {
                    state->freq_buffer[i] *= cos(phase);
                } else {
                    double complex phase_factor = cexp(I * phase);
                    state->freq_buffer[i] *= phase_factor;
                }
            }
            rc = fft_plan_inverse(&state->plan, state->freq_buffer, state->time_buffer);
        } else {
            for (size_t y = 0U; y < rows; ++y) {
                double ky_index = dispersion_axis_index(y, rows);
                double ky       = ky_index * base_ky;
                size_t row_base = y * cols;
                for (size_t x = 0U; x < cols; ++x) {
                    double kx_index = dispersion_axis_index(x, cols);
                    double kx       = kx_index * base_kx;
                    double k_abs    = sqrt(kx * kx + ky * ky);
                    double k_shift  = fabs(k_abs - state->config.reference_k);
                    double phase =
                        dt_safe * state->config.coefficient * pow(k_shift, state->config.order);
                    if (project_real) {
                        state->freq_buffer[row_base + x] *= cos(phase);
                    } else {
                        double complex phase_factor = cexp(I * phase);
                        state->freq_buffer[row_base + x] *= phase_factor;
                    }
                }
            }
            rc = fft_plan2d_inverse(&state->plan2d, state->freq_buffer, state->time_buffer);
        }
        if (rc != SIM_RESULT_OK) {
            return rc;
        }

        if (needs_complex) {
            SimComplexDouble* cdata = sim_field_complex_data(field);
            if (cdata == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (rank == 1U) {
                for (size_t i = 0U; i < rows; ++i) {
                    size_t idx    = i * stride0;
                    cdata[idx].re = creal(state->time_buffer[i]);
                    cdata[idx].im = cimag(state->time_buffer[i]);
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t in_base  = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx    = row_base + x * stride1;
                        cdata[idx].re = creal(state->time_buffer[in_base + x]);
                        cdata[idx].im = cimag(state->time_buffer[in_base + x]);
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
                    data[idx]  = creal(state->time_buffer[i]);
                }
            } else {
                for (size_t y = 0U; y < rows; ++y) {
                    size_t row_base = y * stride0;
                    size_t in_base  = y * cols;
                    for (size_t x = 0U; x < cols; ++x) {
                        size_t idx = row_base + x * stride1;
                        data[idx]  = creal(state->time_buffer[in_base + x]);
                    }
                }
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult dispersion_step(void*               state_ptr,
                                 struct SimContext*  context,
                                 struct SimOperator* self,
                                 size_t              substep_index,
                                 double              dt_sub,
                                 void*               scratch,
                                 size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return dispersion_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_dispersion_operator(struct SimContext*              context,
                                      const DispersionOperatorConfig* config,
                                      size_t*                         out_index) {
    if (!context)
        return SIM_RESULT_INVALID_ARGUMENT;

    DispersionOperatorConfig local = { 0 };
    if (config)
        local = *config;

    if (!isfinite(local.coefficient))
        local.coefficient = 0.0;
    if (!isfinite(local.order) || local.order < 0.0)
        local.order = 1.0;
    if (local.spacing <= 0.0)
        local.spacing = 1.0;
    if (!isfinite(local.reference_k) || local.reference_k < 0.0)
        local.reference_k = 0.0;

    DispersionState* state = (DispersionState*) calloc(1U, sizeof(DispersionState));
    if (!state)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->config            = local;
    state->capacity          = 0U;
    state->plan_rank         = 0U;
    state->plan_shape[0]     = 0U;
    state->plan_shape[1]     = 0U;
    state->cached_count      = 0U;
    state->cached_rank       = 0U;
    state->cached_shape[0]   = 0U;
    state->cached_shape[1]   = 0U;
    state->cached_step_index = 0U;
    state->cached_dt         = 0.0;
    state->cached_domain     = SIM_FIELD_DOMAIN_PHYSICAL;
    state->cached_field      = NULL;
    state->cache_valid       = false;
    dispersion_refresh_symbolic(state);

    SimField*              field         = sim_context_field(context, local.field_index);
    SimFieldRepresentation repr          = { 0 };
    bool                   needs_complex = false;
    bool                   project_real  = false;
    SimResult field_rc = dispersion_describe_field(field, &repr, &needs_complex, &project_real);
    if (field_rc != SIM_RESULT_OK) {
        dispersion_release(state);
        return field_rc;
    }

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "dispersion");

    SimOperatorInfo info = sim_operator_info_defaults();
    dispersion_fill_info(&info, repr, needs_complex, project_real, state);

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    SimResult         result            = SIM_RESULT_OK;
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "dispersion")) {
        if (field != NULL && needs_complex && field->element_size == sizeof(SimComplexDouble) &&
            !sim_field_representation_has_imag_zero_constraint(repr)) {
            SimIRBuilder* builder = sim_context_ir_builder(context);
            if (builder != NULL) {
                SimOperatorKernelBindingDescriptor bindings[1];
                SimOperatorKernelOutputDescriptor  outputs[1];
                SimOperatorKernelDescriptor        kernel_desc = { 0 };

                SimIRStatefulSpec spec = { 0 };
                spec.eval              = dispersion_ir_eval;
                spec.userdata          = state;
                spec.label             = "dispersion";
                spec.value_type        = sim_ir_type_complex();
                SimIRNodeId disp_node  = sim_ir_builder_stateful_spec(builder, &spec);

                if (disp_node != SIM_IR_INVALID_NODE) {
                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.field_index;

                    outputs[0].ir_field_index = 0U;
                    outputs[0].expression     = disp_node;

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
                    kdesc.destroy               = dispersion_release;
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

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = dispersion_step,
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
                                .symbolic      = dispersion_symbolic,
                                .destroy       = dispersion_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult rc = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (rc != SIM_RESULT_OK) {
        dispersion_release(state);
    }

    return rc;
}

SimResult sim_dispersion_config(struct SimContext*        context,
                                size_t                    operator_index,
                                DispersionOperatorConfig* out_config) {
    if (!context || !out_config)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op)
        return SIM_RESULT_NOT_FOUND;

    DispersionState* state = (DispersionState*) sim_operator_state(op);
    if (!state)
        return SIM_RESULT_INVALID_STATE;

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_dispersion_update(struct SimContext*              context,
                                size_t                          operator_index,
                                const DispersionOperatorConfig* config) {
    if (!context || !config)
        return SIM_RESULT_INVALID_ARGUMENT;

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op)
        return SIM_RESULT_NOT_FOUND;

    DispersionState* state = (DispersionState*) sim_operator_state(op);
    if (!state)
        return SIM_RESULT_INVALID_STATE;

    DispersionOperatorConfig local = *config;
    if (!isfinite(local.coefficient))
        local.coefficient = state->config.coefficient;
    if (!isfinite(local.order) || local.order < 0.0)
        local.order = state->config.order;
    if (local.spacing <= 0.0)
        local.spacing = state->config.spacing;
    if (!isfinite(local.reference_k) || local.reference_k < 0.0)
        local.reference_k = state->config.reference_k;

    state->config = local;
    dispersion_refresh_symbolic(state);
    state->cache_valid = false;

    {
        SimFieldRepresentation repr          = { 0 };
        bool                   needs_complex = false;
        bool                   project_real  = false;
        SimResult              field_rc =
            dispersion_describe_field(sim_context_field(context, state->config.field_index),
                                      &repr,
                                      &needs_complex,
                                      &project_real);
        if (field_rc != SIM_RESULT_OK) {
            return field_rc;
        }
        dispersion_fill_info(&op->info, repr, needs_complex, project_real, state);
    }

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
