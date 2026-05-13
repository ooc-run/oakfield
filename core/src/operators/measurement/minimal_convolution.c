#include "oakfield/operators/measurement/minimal_convolution.h"
#include "operators/common/operator_utils.h"

#include "oakfield/operator.h"
#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "sim_accel.h"
#include "oakfield/operator_split.h"
#include "oakfield/operator_identity.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN_CONV_SYMBOLIC_CAPACITY 192

static const size_t g_supported_kernel_lengths[] = { 3U, 5U, 7U, 9U };

typedef struct SimMinimalConvolutionOperatorState {
    SimMinimalConvolutionOperatorConfig config;
    size_t                              kernel_length;
    double                              kernel[SIM_MINIMAL_CONVOLUTION_MAX_TAPS];
    size_t                              kernel_rows;
    size_t                              kernel_cols;
    double                              kernel_2d[SIM_MINIMAL_CONVOLUTION_MAX_TAPS_2D];
    char                                symbolic[MIN_CONV_SYMBOLIC_CAPACITY];
    size_t                              kernel_cached_step_index;
    double                              kernel_cached_dt;
    const SimField*                     kernel_cached_input;
    const SimField*                     kernel_cached_output;
    size_t                              kernel_cached_count;
    bool                                kernel_cache_valid;
    void*                               scratch;
    size_t                              scratch_bytes;
    bool                                scratch_complex;
    void*                               kernel_cache;
    size_t                              kernel_cache_bytes;
    bool                                kernel_cache_complex;
} SimMinimalConvolutionOperatorState;

static bool minimal_convolution_is_supported_length(size_t value) {
    for (size_t i = 0U;
         i < sizeof(g_supported_kernel_lengths) / sizeof(g_supported_kernel_lengths[0]);
         ++i) {
        if (g_supported_kernel_lengths[i] == value) {
            return true;
        }
    }
    return false;
}

static size_t minimal_convolution_default_length(void) {
    return g_supported_kernel_lengths[0];
}

static void minimal_convolution_default_kernel(double* kernel, size_t length) {
    if (!kernel || length == 0U) {
        return;
    }

    double weight = 1.0 / (double) length;
    for (size_t i = 0U; i < length; ++i) {
        kernel[i] = weight;
    }
    for (size_t i = length; i < SIM_MINIMAL_CONVOLUTION_MAX_TAPS; ++i) {
        kernel[i] = 0.0;
    }
}

static const char* minimal_convolution_mode_name(SimMinimalConvolutionMode mode) {
    switch (mode) {
        case SIM_MINIMAL_CONVOLUTION_MODE_SEPARABLE:
            return "separable";
        case SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D:
            return "kernel_2d";
        case SIM_MINIMAL_CONVOLUTION_MODE_AXIS:
        default:
            return "axis";
    }
}

static const char* minimal_convolution_axis_name(SimMinimalConvolutionAxis axis) {
    switch (axis) {
        case SIM_MINIMAL_CONVOLUTION_AXIS_Y:
            return "y";
        case SIM_MINIMAL_CONVOLUTION_AXIS_X:
        default:
            return "x";
    }
}

static void minimal_convolution_default_kernel_2d(double* kernel, size_t rows, size_t cols) {
    if (!kernel || rows == 0U || cols == 0U) {
        return;
    }

    size_t length = rows * cols;
    double weight = 1.0 / (double) length;
    for (size_t i = 0U; i < length; ++i) {
        kernel[i] = weight;
    }
    for (size_t i = length; i < SIM_MINIMAL_CONVOLUTION_MAX_TAPS_2D; ++i) {
        kernel[i] = 0.0;
    }
}

static bool minimal_convolution_validate_fields(struct SimContext*                         context,
                                                const SimMinimalConvolutionOperatorConfig* cfg) {
    SimField* input;
    SimField* output;
    size_t    input_count;
    size_t    output_count;
    size_t    rank;

    if (!context || !cfg) {
        return false;
    }

    input  = sim_context_field(context, cfg->input_field);
    output = sim_context_field(context, cfg->output_field);
    if (!input || !output) {
        return false;
    }

    if (input == output) {
        return false;
    }

    rank = sim_field_rank(input);
    if (rank == 0U || rank > 2U || sim_field_rank(output) != rank) {
        return false;
    }

    input_count  = sim_field_element_count(&input->layout);
    output_count = sim_field_element_count(&output->layout);
    if (input_count != output_count || input_count == 0U) {
        return false;
    }

    if (rank == 2U) {
        const size_t* in_shape  = sim_field_shape(input);
        const size_t* out_shape = sim_field_shape(output);
        if (in_shape == NULL || out_shape == NULL) {
            return false;
        }
        if (in_shape[0] != out_shape[0] || in_shape[1] != out_shape[1]) {
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

static void minimal_convolution_refresh_symbolic(SimMinimalConvolutionOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (!state) {
        return;
    }

    char   kernel_repr[96];
    size_t offset  = 0U;
    int    written = 0;
    kernel_repr[0] = '\0';

    for (size_t i = 0U; i < state->kernel_length; ++i) {
        written = snprintf(kernel_repr + offset,
                           sizeof(kernel_repr) - offset,
                           (i + 1U < state->kernel_length) ? "%.3g," : "%.3g",
                           state->kernel[i]);
        if (written <= 0) {
            break;
        }
        offset += (size_t) written;
        if (offset >= sizeof(kernel_repr)) {
            break;
        }
    }

    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "minimal_convolution mode=%s axis=%s taps=%zu kernel2d=%zux%zu stride=%zu "
                    "boundary=%s wrap=%s accumulate=%s scale_by_dt=%s kernel=[%s]",
                    minimal_convolution_mode_name(state->config.mode),
                    minimal_convolution_axis_name(state->config.axis),
                    state->kernel_length,
                    state->kernel_rows,
                    state->kernel_cols,
                    state->config.stride,
                    sim_boundary_policy_name(state->config.boundary),
                    state->config.wrap ? "true" : "false",
                    state->config.accumulate ? "true" : "false",
                    state->config.scale_by_dt ? "true" : "false",
                    kernel_repr);
#else
    (void) state;
#endif
}

static bool minimal_convolution_validate_kernel(const SimMinimalConvolutionOperatorConfig* config) {
    if (!config) {
        return false;
    }

    if (!minimal_convolution_is_supported_length(config->kernel_length)) {
        return false;
    }

    /* Reject extra coefficients beyond the declared tap count. */
    for (size_t i = config->kernel_length; i < SIM_MINIMAL_CONVOLUTION_MAX_TAPS; ++i) {
        double coeff = config->kernel[i];
        if (!isfinite(coeff)) {
            return false;
        }
        if (fabs(coeff) > 0.0) {
            return false;
        }
    }

    return true;
}

static bool
minimal_convolution_validate_kernel_2d(const SimMinimalConvolutionOperatorConfig* config) {
    if (!config) {
        return false;
    }

    if (!minimal_convolution_is_supported_length(config->kernel_rows) ||
        !minimal_convolution_is_supported_length(config->kernel_cols)) {
        return false;
    }

    size_t length = config->kernel_rows * config->kernel_cols;
    if (length == 0U || length > SIM_MINIMAL_CONVOLUTION_MAX_TAPS_2D) {
        return false;
    }

    return true;
}

static bool minimal_convolution_normalize_config(SimMinimalConvolutionOperatorConfig* config) {
    if (!config) {
        return false;
    }

    if (config->mode < SIM_MINIMAL_CONVOLUTION_MODE_AXIS ||
        config->mode > SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D) {
        config->mode = SIM_MINIMAL_CONVOLUTION_MODE_AXIS;
    }

    if (config->axis < SIM_MINIMAL_CONVOLUTION_AXIS_X ||
        config->axis > SIM_MINIMAL_CONVOLUTION_AXIS_Y) {
        config->axis = SIM_MINIMAL_CONVOLUTION_AXIS_X;
    }

    if (!minimal_convolution_is_supported_length(config->kernel_length)) {
        config->kernel_length = minimal_convolution_default_length();
    }

    if (!minimal_convolution_validate_kernel(config)) {
        return false;
    }

    if (config->stride == 0U) {
        config->stride = 1U;
    }
    if (config->stride > 16U) {
        config->stride = 16U;
    }

    bool kernel_has_values = false;
    for (size_t i = 0U; i < config->kernel_length; ++i) {
        if (isfinite(config->kernel[i]) && config->kernel[i] != 0.0) {
            kernel_has_values = true;
            break;
        }
    }

    if (!kernel_has_values) {
        minimal_convolution_default_kernel(config->kernel, config->kernel_length);
    } else {
        for (size_t i = 0U; i < config->kernel_length; ++i) {
            if (!isfinite(config->kernel[i])) {
                config->kernel[i] = 0.0;
            }
        }
    }

    for (size_t i = config->kernel_length; i < SIM_MINIMAL_CONVOLUTION_MAX_TAPS; ++i) {
        config->kernel[i] = 0.0;
    }

    if (!minimal_convolution_is_supported_length(config->kernel_rows)) {
        config->kernel_rows = config->kernel_length;
    }
    if (!minimal_convolution_is_supported_length(config->kernel_cols)) {
        config->kernel_cols = config->kernel_length;
    }

    if (!minimal_convolution_validate_kernel_2d(config)) {
        return false;
    }

    size_t kernel2d_length     = config->kernel_rows * config->kernel_cols;
    bool   kernel2d_has_values = false;
    for (size_t i = 0U; i < kernel2d_length; ++i) {
        if (isfinite(config->kernel_2d[i]) && config->kernel_2d[i] != 0.0) {
            kernel2d_has_values = true;
            break;
        }
    }

    if (!kernel2d_has_values) {
        minimal_convolution_default_kernel_2d(
            config->kernel_2d, config->kernel_rows, config->kernel_cols);
    } else {
        for (size_t i = 0U; i < kernel2d_length; ++i) {
            if (!isfinite(config->kernel_2d[i])) {
                config->kernel_2d[i] = 0.0;
            }
        }
        for (size_t i = kernel2d_length; i < SIM_MINIMAL_CONVOLUTION_MAX_TAPS_2D; ++i) {
            config->kernel_2d[i] = 0.0;
        }
    }

    /* Map boundary to wrap flag; wrap remains for backward compatibility. */
    switch (config->boundary) {
        case SIM_IR_BOUNDARY_PERIODIC:
            config->wrap = true;
            break;
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            config->wrap = false;
            break;
        default:
            config->boundary = SIM_IR_BOUNDARY_PERIODIC;
            config->wrap     = true;
            break;
    }

    config->wrap = config->wrap ? true : false;
    if (config->wrap) {
        config->boundary = SIM_IR_BOUNDARY_PERIODIC;
    }
    config->accumulate  = config->accumulate ? true : false;
    config->scale_by_dt = config->scale_by_dt ? true : false;

    return true;
}

static size_t minimal_convolution_index(const SimMinimalConvolutionOperatorState* state,
                                        ptrdiff_t                                 index,
                                        size_t                                    count,
                                        bool*                                     out_valid) {
    if (out_valid != NULL) {
        *out_valid = true;
    }

    if (count == 0U) {
        if (out_valid != NULL) {
            *out_valid = false;
        }
        return 0U;
    }

    switch (state->config.boundary) {
        case SIM_IR_BOUNDARY_PERIODIC: {
            ptrdiff_t mod = index % (ptrdiff_t) count;
            if (mod < 0) {
                mod += (ptrdiff_t) count;
            }
            return (size_t) mod;
        }
        case SIM_IR_BOUNDARY_REFLECTIVE: {
            if (index < 0) {
                index = -index;
            }
            if ((size_t) index >= count) {
                index = (ptrdiff_t) (count - 1U);
            }
            return (size_t) index;
        }
        case SIM_IR_BOUNDARY_DIRICHLET:
            if (index < 0 || (size_t) index >= count) {
                if (out_valid != NULL) {
                    *out_valid = false;
                }
                return 0U;
            }
            return (size_t) index;
        case SIM_IR_BOUNDARY_NEUMANN:
        default:
            break;
    }

    if (index < 0) {
        return 0U;
    }

    if ((size_t) index >= count) {
        return count - 1U;
    }

    return (size_t) index;
}

static bool minimal_convolution_layout_2d(const SimField* field,
                                          size_t*         out_width,
                                          size_t*         out_height,
                                          size_t*         out_stride_x,
                                          size_t*         out_stride_y) {
    if (field == NULL || out_width == NULL || out_height == NULL || out_stride_x == NULL ||
        out_stride_y == NULL) {
        return false;
    }

    if (field->layout.rank != 2U || field->layout.shape == NULL || field->layout.strides == NULL) {
        return false;
    }

    size_t axis_x = field->layout.rank - 1U;
    size_t axis_y = field->layout.rank - 2U;

    *out_width    = field->layout.shape[axis_x];
    *out_height   = field->layout.shape[axis_y];
    *out_stride_x = field->layout.strides[axis_x];
    *out_stride_y = field->layout.strides[axis_y];

    return (*out_width > 0U && *out_height > 0U);
}

static void* minimal_convolution_alloc_buffer(size_t bytes) {
    if (bytes == 0U) {
        return NULL;
    }

#if defined(__APPLE__) || defined(_POSIX_VERSION)
    void* buffer = NULL;
    if (posix_memalign(&buffer, 64U, bytes) != 0) {
        return NULL;
    }
    return buffer;
#else
    return malloc(bytes);
#endif
}

static bool minimal_convolution_require_buffer(void**  buffer_ptr,
                                               size_t* bytes_ptr,
                                               bool*   complex_ptr,
                                               size_t  bytes,
                                               bool    is_complex) {
    if (buffer_ptr == NULL || bytes_ptr == NULL || complex_ptr == NULL) {
        return false;
    }

    if (bytes == 0U) {
        return true;
    }

    if (*buffer_ptr != NULL && *bytes_ptr >= bytes && *complex_ptr == is_complex) {
        return true;
    }

    void* buffer = minimal_convolution_alloc_buffer(bytes);
    if (buffer == NULL) {
        return false;
    }

    if (*buffer_ptr != NULL) {
        free(*buffer_ptr);
    }

    *buffer_ptr  = buffer;
    *bytes_ptr   = bytes;
    *complex_ptr = is_complex;
    return true;
}

static bool minimal_convolution_require_scratch(SimMinimalConvolutionOperatorState* state,
                                                size_t                              bytes,
                                                bool                                is_complex) {
    if (state == NULL) {
        return false;
    }
    return minimal_convolution_require_buffer(
        &state->scratch, &state->scratch_bytes, &state->scratch_complex, bytes, is_complex);
}

static bool minimal_convolution_require_kernel_cache(SimMinimalConvolutionOperatorState* state,
                                                     size_t                              bytes,
                                                     bool is_complex) {
    if (state == NULL) {
        return false;
    }
    return minimal_convolution_require_buffer(&state->kernel_cache,
                                              &state->kernel_cache_bytes,
                                              &state->kernel_cache_complex,
                                              bytes,
                                              is_complex);
}

static bool
minimal_convolution_is_row_major_2d(size_t width, size_t height, size_t stride_x, size_t stride_y) {
    return width > 0U && height > 0U && stride_x == 1U && stride_y == width;
}

static void
minimal_convolution_apply_real_periodic_row(const SimMinimalConvolutionOperatorState* state,
                                            const double*                             src,
                                            double*                                   dst,
                                            size_t                                    count,
                                            bool                                      accumulate,
                                            double accumulate_scale) {
    if (state == NULL || src == NULL || dst == NULL || count == 0U) {
        return;
    }

    if (!accumulate) {
        (void) memset(dst, 0, count * sizeof(double));
    }

    size_t radius = state->kernel_length / 2U;
    double scale  = accumulate ? accumulate_scale : 1.0;

    for (size_t k = 0U; k < state->kernel_length; ++k) {
        double coeff = scale * state->kernel[k];
        if (coeff == 0.0) {
            continue;
        }

        ptrdiff_t shift = (ptrdiff_t) k - (ptrdiff_t) radius;
        if (shift == 0) {
            sim_accel_copy_scale_real(src, dst, count, coeff, true);
            continue;
        }

        size_t offset = (size_t) ((shift < 0) ? -shift : shift) % count;
        if (offset == 0U) {
            sim_accel_copy_scale_real(src, dst, count, coeff, true);
            continue;
        }

        size_t main_count = count - offset;
        if (shift > 0) {
            sim_accel_copy_scale_real(src + offset, dst, main_count, coeff, true);
            sim_accel_copy_scale_real(src, dst + main_count, offset, coeff, true);
        } else {
            sim_accel_copy_scale_real(src, dst + offset, main_count, coeff, true);
            sim_accel_copy_scale_real(src + main_count, dst, offset, coeff, true);
        }
    }
}

static void
minimal_convolution_apply_complex_periodic_row(const SimMinimalConvolutionOperatorState* state,
                                               const SimComplexDouble*                   src,
                                               SimComplexDouble*                         dst,
                                               size_t                                    count,
                                               bool                                      accumulate,
                                               double accumulate_scale) {
    if (state == NULL || src == NULL || dst == NULL || count == 0U) {
        return;
    }

    if (!accumulate) {
        (void) memset(dst, 0, count * sizeof(SimComplexDouble));
    }

    size_t radius = state->kernel_length / 2U;
    double scale  = accumulate ? accumulate_scale : 1.0;

    for (size_t k = 0U; k < state->kernel_length; ++k) {
        double coeff = scale * state->kernel[k];
        if (coeff == 0.0) {
            continue;
        }

        ptrdiff_t shift = (ptrdiff_t) k - (ptrdiff_t) radius;
        if (shift == 0) {
            sim_accel_copy_scale_complex(src, dst, count, coeff, true);
            continue;
        }

        size_t offset = (size_t) ((shift < 0) ? -shift : shift) % count;
        if (offset == 0U) {
            sim_accel_copy_scale_complex(src, dst, count, coeff, true);
            continue;
        }

        size_t main_count = count - offset;
        if (shift > 0) {
            sim_accel_copy_scale_complex(src + offset, dst, main_count, coeff, true);
            sim_accel_copy_scale_complex(src, dst + main_count, offset, coeff, true);
        } else {
            sim_accel_copy_scale_complex(src, dst + offset, main_count, coeff, true);
            sim_accel_copy_scale_complex(src + main_count, dst, offset, coeff, true);
        }
    }
}

static void minimal_convolution_apply_real(const SimMinimalConvolutionOperatorState* state,
                                           const double*                             src,
                                           double*                                   dst,
                                           size_t                                    count,
                                           double accumulate_scale) {
    if (!state || !src || !dst || count == 0U) {
        return;
    }

    size_t radius = state->kernel_length / 2U;
    size_t stride = (state->config.stride > 0U) ? state->config.stride : 1U;
    if (state->config.boundary == SIM_IR_BOUNDARY_PERIODIC && stride == 1U) {
        minimal_convolution_apply_real_periodic_row(
            state, src, dst, count, state->config.accumulate, accumulate_scale);
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double accum = 0.0;
        for (size_t k = 0U; k < state->kernel_length; ++k) {
            ptrdiff_t offset       = (ptrdiff_t) (i * stride) + (ptrdiff_t) k - (ptrdiff_t) radius;
            bool      valid        = true;
            size_t    sample_index = minimal_convolution_index(state, offset, count, &valid);
            if (valid) {
                accum += state->kernel[k] * src[sample_index];
            }
        }

        if (state->config.accumulate) {
            dst[i] += accumulate_scale * accum;
        } else {
            dst[i] = accum;
        }
    }
}

static void minimal_convolution_apply_real_axis_2d(const SimMinimalConvolutionOperatorState* state,
                                                   const double*                             src,
                                                   double*                                   dst,
                                                   size_t                                    width,
                                                   size_t                                    height,
                                                   size_t                    stride_x,
                                                   size_t                    stride_y,
                                                   SimMinimalConvolutionAxis axis,
                                                   bool                      accumulate,
                                                   double                    accumulate_scale) {
    if (!state || !src || !dst || width == 0U || height == 0U) {
        return;
    }

    size_t radius = state->kernel_length / 2U;
    size_t stride = (state->config.stride > 0U) ? state->config.stride : 1U;
    if (axis == SIM_MINIMAL_CONVOLUTION_AXIS_X &&
        state->config.boundary == SIM_IR_BOUNDARY_PERIODIC && stride == 1U &&
        minimal_convolution_is_row_major_2d(width, height, stride_x, stride_y)) {
        for (size_t y = 0U; y < height; ++y) {
            size_t row_offset = y * stride_y;
            minimal_convolution_apply_real_periodic_row(
                state, src + row_offset, dst + row_offset, width, accumulate, accumulate_scale);
        }
        return;
    }

    for (size_t y = 0U; y < height; ++y) {
        size_t row_offset = y * stride_y;
        for (size_t x = 0U; x < width; ++x) {
            double accum = 0.0;
            if (axis == SIM_MINIMAL_CONVOLUTION_AXIS_Y) {
                for (size_t k = 0U; k < state->kernel_length; ++k) {
                    ptrdiff_t offset =
                        (ptrdiff_t) (y * stride) + (ptrdiff_t) k - (ptrdiff_t) radius;
                    bool   valid    = true;
                    size_t sample_y = minimal_convolution_index(state, offset, height, &valid);
                    if (valid) {
                        size_t index = sample_y * stride_y + x * stride_x;
                        accum += state->kernel[k] * src[index];
                    }
                }
            } else {
                for (size_t k = 0U; k < state->kernel_length; ++k) {
                    ptrdiff_t offset =
                        (ptrdiff_t) (x * stride) + (ptrdiff_t) k - (ptrdiff_t) radius;
                    bool   valid    = true;
                    size_t sample_x = minimal_convolution_index(state, offset, width, &valid);
                    if (valid) {
                        size_t index = row_offset + sample_x * stride_x;
                        accum += state->kernel[k] * src[index];
                    }
                }
            }

            size_t out_index = row_offset + x * stride_x;
            if (accumulate) {
                dst[out_index] += accumulate_scale * accum;
            } else {
                dst[out_index] = accum;
            }
        }
    }
}

static void
minimal_convolution_apply_real_kernel_2d(const SimMinimalConvolutionOperatorState* state,
                                         const double*                             src,
                                         double*                                   dst,
                                         size_t                                    width,
                                         size_t                                    height,
                                         size_t                                    stride_x,
                                         size_t                                    stride_y,
                                         bool                                      accumulate,
                                         double accumulate_scale) {
    if (!state || !src || !dst || width == 0U || height == 0U) {
        return;
    }

    size_t radius_x = state->kernel_cols / 2U;
    size_t radius_y = state->kernel_rows / 2U;
    size_t stride   = (state->config.stride > 0U) ? state->config.stride : 1U;

    for (size_t y = 0U; y < height; ++y) {
        size_t row_offset = y * stride_y;
        for (size_t x = 0U; x < width; ++x) {
            double accum = 0.0;
            for (size_t ky = 0U; ky < state->kernel_rows; ++ky) {
                ptrdiff_t offset_y =
                    (ptrdiff_t) (y * stride) + (ptrdiff_t) ky - (ptrdiff_t) radius_y;
                bool   valid_y  = true;
                size_t sample_y = minimal_convolution_index(state, offset_y, height, &valid_y);
                if (!valid_y) {
                    continue;
                }
                for (size_t kx = 0U; kx < state->kernel_cols; ++kx) {
                    ptrdiff_t offset_x =
                        (ptrdiff_t) (x * stride) + (ptrdiff_t) kx - (ptrdiff_t) radius_x;
                    bool   valid_x  = true;
                    size_t sample_x = minimal_convolution_index(state, offset_x, width, &valid_x);
                    if (!valid_x) {
                        continue;
                    }

                    size_t sample_index = sample_y * stride_y + sample_x * stride_x;
                    size_t kernel_index = ky * state->kernel_cols + kx;
                    accum += state->kernel_2d[kernel_index] * src[sample_index];
                }
            }

            size_t out_index = row_offset + x * stride_x;
            if (accumulate) {
                dst[out_index] += accumulate_scale * accum;
            } else {
                dst[out_index] = accum;
            }
        }
    }
}

static void minimal_convolution_apply_complex(const SimMinimalConvolutionOperatorState* state,
                                              const SimComplexDouble*                   src,
                                              SimComplexDouble*                         dst,
                                              size_t                                    count,
                                              double accumulate_scale) {
    if (!state || !src || !dst || count == 0U) {
        return;
    }

    size_t radius = state->kernel_length / 2U;
    size_t stride = (state->config.stride > 0U) ? state->config.stride : 1U;
    if (state->config.boundary == SIM_IR_BOUNDARY_PERIODIC && stride == 1U) {
        minimal_convolution_apply_complex_periodic_row(
            state, src, dst, count, state->config.accumulate, accumulate_scale);
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double accum_re = 0.0;
        double accum_im = 0.0;
        for (size_t k = 0U; k < state->kernel_length; ++k) {
            ptrdiff_t offset       = (ptrdiff_t) (i * stride) + (ptrdiff_t) k - (ptrdiff_t) radius;
            bool      valid        = true;
            size_t    sample_index = minimal_convolution_index(state, offset, count, &valid);
            if (valid) {
                accum_re += state->kernel[k] * src[sample_index].re;
                accum_im += state->kernel[k] * src[sample_index].im;
            }
        }

        if (state->config.accumulate) {
            dst[i].re += accumulate_scale * accum_re;
            dst[i].im += accumulate_scale * accum_im;
        } else {
            dst[i].re = accum_re;
            dst[i].im = accum_im;
        }
    }
}

static void
minimal_convolution_apply_complex_axis_2d(const SimMinimalConvolutionOperatorState* state,
                                          const SimComplexDouble*                   src,
                                          SimComplexDouble*                         dst,
                                          size_t                                    width,
                                          size_t                                    height,
                                          size_t                                    stride_x,
                                          size_t                                    stride_y,
                                          SimMinimalConvolutionAxis                 axis,
                                          bool                                      accumulate,
                                          double accumulate_scale) {
    if (!state || !src || !dst || width == 0U || height == 0U) {
        return;
    }

    size_t radius = state->kernel_length / 2U;
    size_t stride = (state->config.stride > 0U) ? state->config.stride : 1U;
    if (axis == SIM_MINIMAL_CONVOLUTION_AXIS_X &&
        state->config.boundary == SIM_IR_BOUNDARY_PERIODIC && stride == 1U &&
        minimal_convolution_is_row_major_2d(width, height, stride_x, stride_y)) {
        for (size_t y = 0U; y < height; ++y) {
            size_t row_offset = y * stride_y;
            minimal_convolution_apply_complex_periodic_row(
                state, src + row_offset, dst + row_offset, width, accumulate, accumulate_scale);
        }
        return;
    }

    for (size_t y = 0U; y < height; ++y) {
        size_t row_offset = y * stride_y;
        for (size_t x = 0U; x < width; ++x) {
            double accum_re = 0.0;
            double accum_im = 0.0;
            if (axis == SIM_MINIMAL_CONVOLUTION_AXIS_Y) {
                for (size_t k = 0U; k < state->kernel_length; ++k) {
                    ptrdiff_t offset =
                        (ptrdiff_t) (y * stride) + (ptrdiff_t) k - (ptrdiff_t) radius;
                    bool   valid    = true;
                    size_t sample_y = minimal_convolution_index(state, offset, height, &valid);
                    if (valid) {
                        size_t index = sample_y * stride_y + x * stride_x;
                        accum_re += state->kernel[k] * src[index].re;
                        accum_im += state->kernel[k] * src[index].im;
                    }
                }
            } else {
                for (size_t k = 0U; k < state->kernel_length; ++k) {
                    ptrdiff_t offset =
                        (ptrdiff_t) (x * stride) + (ptrdiff_t) k - (ptrdiff_t) radius;
                    bool   valid    = true;
                    size_t sample_x = minimal_convolution_index(state, offset, width, &valid);
                    if (valid) {
                        size_t index = row_offset + sample_x * stride_x;
                        accum_re += state->kernel[k] * src[index].re;
                        accum_im += state->kernel[k] * src[index].im;
                    }
                }
            }

            size_t out_index = row_offset + x * stride_x;
            if (accumulate) {
                dst[out_index].re += accumulate_scale * accum_re;
                dst[out_index].im += accumulate_scale * accum_im;
            } else {
                dst[out_index].re = accum_re;
                dst[out_index].im = accum_im;
            }
        }
    }
}

static void
minimal_convolution_apply_complex_kernel_2d(const SimMinimalConvolutionOperatorState* state,
                                            const SimComplexDouble*                   src,
                                            SimComplexDouble*                         dst,
                                            size_t                                    width,
                                            size_t                                    height,
                                            size_t                                    stride_x,
                                            size_t                                    stride_y,
                                            bool                                      accumulate,
                                            double accumulate_scale) {
    if (!state || !src || !dst || width == 0U || height == 0U) {
        return;
    }

    size_t radius_x = state->kernel_cols / 2U;
    size_t radius_y = state->kernel_rows / 2U;
    size_t stride   = (state->config.stride > 0U) ? state->config.stride : 1U;

    for (size_t y = 0U; y < height; ++y) {
        size_t row_offset = y * stride_y;
        for (size_t x = 0U; x < width; ++x) {
            double accum_re = 0.0;
            double accum_im = 0.0;
            for (size_t ky = 0U; ky < state->kernel_rows; ++ky) {
                ptrdiff_t offset_y =
                    (ptrdiff_t) (y * stride) + (ptrdiff_t) ky - (ptrdiff_t) radius_y;
                bool   valid_y  = true;
                size_t sample_y = minimal_convolution_index(state, offset_y, height, &valid_y);
                if (!valid_y) {
                    continue;
                }
                for (size_t kx = 0U; kx < state->kernel_cols; ++kx) {
                    ptrdiff_t offset_x =
                        (ptrdiff_t) (x * stride) + (ptrdiff_t) kx - (ptrdiff_t) radius_x;
                    bool   valid_x  = true;
                    size_t sample_x = minimal_convolution_index(state, offset_x, width, &valid_x);
                    if (!valid_x) {
                        continue;
                    }
                    size_t sample_index = sample_y * stride_y + sample_x * stride_x;
                    size_t kernel_index = ky * state->kernel_cols + kx;
                    double coeff        = state->kernel_2d[kernel_index];
                    accum_re += coeff * src[sample_index].re;
                    accum_im += coeff * src[sample_index].im;
                }
            }

            size_t out_index = row_offset + x * stride_x;
            if (accumulate) {
                dst[out_index].re += accumulate_scale * accum_re;
                dst[out_index].im += accumulate_scale * accum_im;
            } else {
                dst[out_index].re = accum_re;
                dst[out_index].im = accum_im;
            }
        }
    }
}

static SimResult minimal_convolution_build_kernel_cache(SimMinimalConvolutionOperatorState* state,
                                                        const SimField*                     input,
                                                        const SimField*                     output,
                                                        double accumulate_scale) {
    if (state == NULL || input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (input->element_size != output->element_size) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t rank = sim_field_rank(input);
    if (rank == 0U || rank > 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex = (input->element_size == sizeof(SimComplexDouble));
    size_t bytes      = count * (is_complex ? sizeof(SimComplexDouble) : sizeof(double));
    if (!minimal_convolution_require_kernel_cache(state, bytes, is_complex)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (!is_complex) {
        const double* src   = sim_field_real_data_const(input);
        const double* base  = state->config.accumulate ? sim_field_real_data_const(output) : NULL;
        double*       cache = (double*) state->kernel_cache;

        if (src == NULL || cache == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (state->config.accumulate) {
            if (base == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            (void) memcpy(cache, base, bytes);
        }

        if (rank == 1U) {
            minimal_convolution_apply_real(state, src, cache, count, accumulate_scale);
            return SIM_RESULT_OK;
        }

        size_t width    = 0U;
        size_t height   = 0U;
        size_t stride_x = 0U;
        size_t stride_y = 0U;
        if (!minimal_convolution_layout_2d(input, &width, &height, &stride_x, &stride_y)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_SEPARABLE) {
            if (!minimal_convolution_require_scratch(state, bytes, false)) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }

            double*                   scratch     = (double*) state->scratch;
            SimMinimalConvolutionAxis first_axis  = state->config.axis;
            SimMinimalConvolutionAxis second_axis = (first_axis == SIM_MINIMAL_CONVOLUTION_AXIS_Y)
                                                        ? SIM_MINIMAL_CONVOLUTION_AXIS_X
                                                        : SIM_MINIMAL_CONVOLUTION_AXIS_Y;

            minimal_convolution_apply_real_axis_2d(
                state, src, scratch, width, height, stride_x, stride_y, first_axis, false, 1.0);
            minimal_convolution_apply_real_axis_2d(state,
                                                   scratch,
                                                   cache,
                                                   width,
                                                   height,
                                                   stride_x,
                                                   stride_y,
                                                   second_axis,
                                                   state->config.accumulate,
                                                   accumulate_scale);
        } else if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D) {
            minimal_convolution_apply_real_kernel_2d(state,
                                                     src,
                                                     cache,
                                                     width,
                                                     height,
                                                     stride_x,
                                                     stride_y,
                                                     state->config.accumulate,
                                                     accumulate_scale);
        } else {
            minimal_convolution_apply_real_axis_2d(state,
                                                   src,
                                                   cache,
                                                   width,
                                                   height,
                                                   stride_x,
                                                   stride_y,
                                                   state->config.axis,
                                                   state->config.accumulate,
                                                   accumulate_scale);
        }

        return SIM_RESULT_OK;
    }

    const SimComplexDouble* src = sim_field_complex_data_const(input);
    const SimComplexDouble* base =
        state->config.accumulate ? sim_field_complex_data_const(output) : NULL;
    SimComplexDouble* cache = (SimComplexDouble*) state->kernel_cache;

    if (src == NULL || cache == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (state->config.accumulate) {
        if (base == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        (void) memcpy(cache, base, bytes);
    }

    if (rank == 1U) {
        minimal_convolution_apply_complex(state, src, cache, count, accumulate_scale);
        return SIM_RESULT_OK;
    }

    size_t width    = 0U;
    size_t height   = 0U;
    size_t stride_x = 0U;
    size_t stride_y = 0U;
    if (!minimal_convolution_layout_2d(input, &width, &height, &stride_x, &stride_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_SEPARABLE) {
        if (!minimal_convolution_require_scratch(state, bytes, true)) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        SimComplexDouble*         scratch     = (SimComplexDouble*) state->scratch;
        SimMinimalConvolutionAxis first_axis  = state->config.axis;
        SimMinimalConvolutionAxis second_axis = (first_axis == SIM_MINIMAL_CONVOLUTION_AXIS_Y)
                                                    ? SIM_MINIMAL_CONVOLUTION_AXIS_X
                                                    : SIM_MINIMAL_CONVOLUTION_AXIS_Y;

        minimal_convolution_apply_complex_axis_2d(
            state, src, scratch, width, height, stride_x, stride_y, first_axis, false, 1.0);
        minimal_convolution_apply_complex_axis_2d(state,
                                                  scratch,
                                                  cache,
                                                  width,
                                                  height,
                                                  stride_x,
                                                  stride_y,
                                                  second_axis,
                                                  state->config.accumulate,
                                                  accumulate_scale);
    } else if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D) {
        minimal_convolution_apply_complex_kernel_2d(state,
                                                    src,
                                                    cache,
                                                    width,
                                                    height,
                                                    stride_x,
                                                    stride_y,
                                                    state->config.accumulate,
                                                    accumulate_scale);
    } else {
        minimal_convolution_apply_complex_axis_2d(state,
                                                  src,
                                                  cache,
                                                  width,
                                                  height,
                                                  stride_x,
                                                  stride_y,
                                                  state->config.axis,
                                                  state->config.accumulate,
                                                  accumulate_scale);
    }

    return SIM_RESULT_OK;
}

static SimResult minimal_convolution_evaluate(SimMinimalConvolutionOperatorState* state,
                                              struct SimContext*                  context,
                                              double accumulate_scale) {
    SimField* input;
    SimField* output;
    size_t    count;
    size_t    rank;

    if (!state || !context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    input  = sim_context_field(context, state->config.input_field);
    output = sim_context_field(context, state->config.output_field);
    if (!input || !output) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = sim_field_element_count(&input->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    if (count != sim_field_element_count(&output->layout)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input->element_size != output->element_size) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rank = sim_field_rank(input);

    if (input->element_size == sizeof(double)) {
        const double* src = sim_field_real_data_const(input);
        double*       dst = sim_field_real_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (rank == 1U) {
            minimal_convolution_apply_real(state, src, dst, count, accumulate_scale);
        } else if (rank == 2U) {
            size_t width    = 0U;
            size_t height   = 0U;
            size_t stride_x = 0U;
            size_t stride_y = 0U;
            if (!minimal_convolution_layout_2d(input, &width, &height, &stride_x, &stride_y)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_SEPARABLE) {
                if (!minimal_convolution_require_scratch(state, count * sizeof(double), false)) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                double*                   scratch    = (double*) state->scratch;
                SimMinimalConvolutionAxis first_axis = state->config.axis;
                SimMinimalConvolutionAxis second_axis =
                    (first_axis == SIM_MINIMAL_CONVOLUTION_AXIS_Y) ? SIM_MINIMAL_CONVOLUTION_AXIS_X
                                                                   : SIM_MINIMAL_CONVOLUTION_AXIS_Y;

                minimal_convolution_apply_real_axis_2d(
                    state, src, scratch, width, height, stride_x, stride_y, first_axis, false, 1.0);
                minimal_convolution_apply_real_axis_2d(state,
                                                       scratch,
                                                       dst,
                                                       width,
                                                       height,
                                                       stride_x,
                                                       stride_y,
                                                       second_axis,
                                                       state->config.accumulate,
                                                       accumulate_scale);
            } else if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D) {
                minimal_convolution_apply_real_kernel_2d(state,
                                                         src,
                                                         dst,
                                                         width,
                                                         height,
                                                         stride_x,
                                                         stride_y,
                                                         state->config.accumulate,
                                                         accumulate_scale);
            } else {
                minimal_convolution_apply_real_axis_2d(state,
                                                       src,
                                                       dst,
                                                       width,
                                                       height,
                                                       stride_x,
                                                       stride_y,
                                                       state->config.axis,
                                                       state->config.accumulate,
                                                       accumulate_scale);
            }
        } else {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else if (input->element_size == sizeof(SimComplexDouble)) {
        const SimComplexDouble* src = sim_field_complex_data_const(input);
        SimComplexDouble*       dst = sim_field_complex_data(output);
        if (!src || !dst) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (rank == 1U) {
            minimal_convolution_apply_complex(state, src, dst, count, accumulate_scale);
        } else if (rank == 2U) {
            size_t width    = 0U;
            size_t height   = 0U;
            size_t stride_x = 0U;
            size_t stride_y = 0U;
            if (!minimal_convolution_layout_2d(input, &width, &height, &stride_x, &stride_y)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_SEPARABLE) {
                if (!minimal_convolution_require_scratch(
                        state, count * sizeof(SimComplexDouble), true)) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                SimComplexDouble*         scratch    = (SimComplexDouble*) state->scratch;
                SimMinimalConvolutionAxis first_axis = state->config.axis;
                SimMinimalConvolutionAxis second_axis =
                    (first_axis == SIM_MINIMAL_CONVOLUTION_AXIS_Y) ? SIM_MINIMAL_CONVOLUTION_AXIS_X
                                                                   : SIM_MINIMAL_CONVOLUTION_AXIS_Y;

                minimal_convolution_apply_complex_axis_2d(
                    state, src, scratch, width, height, stride_x, stride_y, first_axis, false, 1.0);
                minimal_convolution_apply_complex_axis_2d(state,
                                                          scratch,
                                                          dst,
                                                          width,
                                                          height,
                                                          stride_x,
                                                          stride_y,
                                                          second_axis,
                                                          state->config.accumulate,
                                                          accumulate_scale);
            } else if (state->config.mode == SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D) {
                minimal_convolution_apply_complex_kernel_2d(state,
                                                            src,
                                                            dst,
                                                            width,
                                                            height,
                                                            stride_x,
                                                            stride_y,
                                                            state->config.accumulate,
                                                            accumulate_scale);
            } else {
                minimal_convolution_apply_complex_axis_2d(state,
                                                          src,
                                                          dst,
                                                          width,
                                                          height,
                                                          stride_x,
                                                          stride_y,
                                                          state->config.axis,
                                                          state->config.accumulate,
                                                          accumulate_scale);
            }
        } else {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    return SIM_RESULT_OK;
}

static SimResult minimal_convolution_step(void*               state_ptr,
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
    SimMinimalConvolutionOperatorState* state = (SimMinimalConvolutionOperatorState*) state_ptr;
    double                              accumulate_scale = 1.0;
    if (state != NULL && state->config.scale_by_dt) {
        accumulate_scale = fmax(dt_sub, 0.0);
    }
    return minimal_convolution_evaluate(state, context, accumulate_scale);
}

static const char* minimal_convolution_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimMinimalConvolutionOperatorState* state =
        (const SimMinimalConvolutionOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void minimal_convolution_destroy(void* state_ptr) {
    SimMinimalConvolutionOperatorState* state = (SimMinimalConvolutionOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    if (state->scratch != NULL) {
        free(state->scratch);
        state->scratch = NULL;
    }
    if (state->kernel_cache != NULL) {
        free(state->kernel_cache);
        state->kernel_cache = NULL;
    }
    free(state);
}

static double minimal_convolution_kernel_param_value(const KernelIR* kernel, SimIRParamKind param) {
    if (kernel == NULL || kernel->params == NULL || kernel->param_count <= (size_t) param) {
        return 0.0;
    }
    double value = kernel->params[param];
    return isfinite(value) ? value : 0.0;
}

static SimResult minimal_convolution_ir_eval(void*           userdata,
                                             const KernelIR* kernel,
                                             size_t          element_index,
                                             size_t          component,
                                             double*         out_value) {
    if (out_value == NULL || userdata == NULL || kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimMinimalConvolutionOperatorState* state = (SimMinimalConvolutionOperatorState*) userdata;
    if (kernel->bindings == NULL || kernel->binding_count < 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimKernelIRBinding* input_binding  = &kernel->bindings[0];
    const SimKernelIRBinding* output_binding = &kernel->bindings[1];
    SimField*                 input          = input_binding->field;
    SimField*                 output         = output_binding->field;
    if (input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t element_count = sim_field_element_count(&input->layout);
    if (element_count == 0U || element_index >= element_count) {
        *out_value = 0.0;
        return SIM_RESULT_OK;
    }

    if (element_count != sim_field_element_count(&output->layout)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (input->element_size != output->element_size) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t rank = sim_field_rank(input);
    if (rank == 0U || rank > 2U || sim_field_rank(output) != rank) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex      = sim_field_is_complex(input);
    size_t component_count = is_complex ? 2U : 1U;
    if (component >= component_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double dt               = minimal_convolution_kernel_param_value(kernel, SIM_IR_PARAM_DT);
    double accumulate_scale = state->config.scale_by_dt ? fmax(dt, 0.0) : 1.0;

    size_t step_index = 0U;

    bool have_step =
        (kernel->params != NULL && kernel->param_count > (size_t) SIM_IR_PARAM_STEP_INDEX);
    if (have_step) {
        double step_value = kernel->params[SIM_IR_PARAM_STEP_INDEX];
        if (isfinite(step_value) && step_value >= 0.0) {
            step_index = (size_t) step_value;
        } else {
            have_step = false;
        }
    }

    bool dt_changed = state->config.scale_by_dt && state->kernel_cached_dt != dt;
    bool need_reset = !state->kernel_cache_valid || state->kernel_cached_input != input ||
                      state->kernel_cached_output != output ||
                      state->kernel_cached_count != element_count || dt_changed || !have_step ||
                      state->kernel_cached_step_index != step_index;

    if (need_reset) {
        SimResult rc =
            minimal_convolution_build_kernel_cache(state, input, output, accumulate_scale);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
        state->kernel_cached_input      = input;
        state->kernel_cached_output     = output;
        state->kernel_cached_count      = element_count;
        state->kernel_cached_dt         = dt;
        state->kernel_cached_step_index = step_index;
        state->kernel_cache_valid       = have_step;
    }

    const double* cache = (const double*) state->kernel_cache;
    if (cache == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_value = cache[element_index * component_count + component];
    return SIM_RESULT_OK;
}

SimResult sim_add_minimal_convolution_operator(struct SimContext*                         context,
                                               const SimMinimalConvolutionOperatorConfig* config,
                                               size_t* out_index) {
    SimMinimalConvolutionOperatorConfig local = { 0 };
    SimMinimalConvolutionOperatorState* state = NULL;
    SimOperatorInfo                     info;
    SimSplitPort                        ports[2];
    SimSplitAccess                      accesses[2];
    SimSplitSubstep                     substep = { 0 };
    SimSplitDescriptor                  desc    = { 0 };
    char                                name[SIM_OPERATOR_NAME_MAX + 1U];
    SimResult                           result = SIM_RESULT_OK;

    if (!context) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state = (SimMinimalConvolutionOperatorState*) calloc(
        1U, sizeof(SimMinimalConvolutionOperatorState));
    if (!state) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (config) {
        local = *config;
    } else {
        local.input_field   = 0U;
        local.output_field  = 0U;
        local.kernel_length = minimal_convolution_default_length();
        minimal_convolution_default_kernel(local.kernel, local.kernel_length);
        local.mode        = SIM_MINIMAL_CONVOLUTION_MODE_AXIS;
        local.axis        = SIM_MINIMAL_CONVOLUTION_AXIS_X;
        local.kernel_rows = local.kernel_length;
        local.kernel_cols = local.kernel_length;
        minimal_convolution_default_kernel_2d(
            local.kernel_2d, local.kernel_rows, local.kernel_cols);
        local.wrap        = true;
        local.boundary    = SIM_IR_BOUNDARY_PERIODIC;
        local.accumulate  = false;
        local.scale_by_dt = true;
        local.stride      = 1U;
    }

    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "minimal_convolution",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);
    if (!minimal_convolution_normalize_config(&local)) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!minimal_convolution_validate_fields(context, &local)) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config        = local;
    state->kernel_length = local.kernel_length;
    memcpy(state->kernel, local.kernel, sizeof(state->kernel));
    state->kernel_rows = local.kernel_rows;
    state->kernel_cols = local.kernel_cols;
    memcpy(state->kernel_2d, local.kernel_2d, sizeof(state->kernel_2d));
    minimal_convolution_refresh_symbolic(state);
    state->kernel_cached_step_index = 0U;
    state->kernel_cached_dt         = 0.0;
    state->kernel_cached_input      = NULL;
    state->kernel_cached_output     = NULL;
    state->kernel_cached_count      = 0U;
    state->kernel_cache_valid       = false;
    state->scratch                  = NULL;
    state->scratch_bytes            = 0U;
    state->scratch_complex          = false;
    state->kernel_cache             = NULL;
    state->kernel_cache_bytes       = 0U;
    state->kernel_cache_complex     = false;

    info                   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_MEASUREMENT;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = false;
    info.is_nonlocal       = true;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "minimal_convolution";
    sim_operator_info_set_schema_identity(&info, "minimal_convolution");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    SimField* input_field      = sim_context_field(context, local.input_field);
    bool      needs_complex    = (input_field != NULL) ? sim_field_is_complex(input_field) : false;
    info.representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = needs_complex;
    info.representation.requires_complex_representation = needs_complex;
    info.representation.preserves_real_subspace         = info.preserves_real;
    info.representation.boundary                        = local.boundary;

    SimOperatorConfig op_config         = sim_operator_config_defaults();
    bool              registered_kernel = false;

    if (sim_operator_should_register_kernel_for_schema(
            context, &op_config, 0ULL, SIM_DET_NONE, "minimal_convolution")) {
        SimField* input  = input_field;
        SimField* output = sim_context_field(context, local.output_field);
        if (input != NULL && output != NULL && input->element_size == output->element_size) {
            bool   is_complex    = sim_field_is_complex(input);
            size_t expected_size = is_complex ? sizeof(SimComplexDouble) : sizeof(double);
            if (input->element_size == expected_size) {
                SimIRBuilder* builder = sim_context_ir_builder(context);
                if (builder != NULL) {
                    SimOperatorKernelBindingDescriptor bindings[2];
                    SimOperatorKernelOutputDescriptor  outputs[1];
                    SimOperatorKernelDescriptor        kernel_desc = { 0 };

                    bindings[0].ir_field_index      = 0U;
                    bindings[0].context_field_index = local.input_field;
                    bindings[1].ir_field_index      = 1U;
                    bindings[1].context_field_index = local.output_field;

                    SimIRStatefulSpec spec = { 0 };
                    spec.eval              = minimal_convolution_ir_eval;
                    spec.userdata          = state;
                    spec.label             = "minimal_convolution";
                    spec.value_type  = is_complex ? sim_ir_type_complex() : sim_ir_type_scalar();
                    SimIRNodeId node = sim_ir_builder_stateful_spec(builder, &spec);

                    if (node != SIM_IR_INVALID_NODE) {
                        outputs[0].ir_field_index = 1U;
                        outputs[0].expression     = node;

                        kernel_desc.builder           = builder;
                        kernel_desc.bindings          = bindings;
                        kernel_desc.binding_count     = 2U;
                        kernel_desc.outputs           = outputs;
                        kernel_desc.output_count      = 1U;
                        kernel_desc.param_count       = (size_t) SIM_IR_PARAM_STEP_INDEX + 1U;
                        kernel_desc.required_features = 0ULL;

                        SimOperatorDescriptor kdesc = { 0 };
                        kdesc.name                  = name;
                        kdesc.evaluate              = NULL;
                        kdesc.destroy               = minimal_convolution_destroy;
                        kdesc.userdata              = state;
                        kdesc.kernel                = &kernel_desc;
                        kdesc.info                  = info;
                        kdesc.config                = op_config;
                        if (local.input_field < 64U) {
                            kdesc.read_mask |= (1ULL << local.input_field);
                        }
                        if (local.output_field < 64U) {
                            kdesc.read_mask |= (1ULL << local.output_field);
                            kdesc.write_mask |= (1ULL << local.output_field);
                        }

                        result = sim_context_register_operator(context, &kdesc, out_index);
                        if (result == SIM_RESULT_OK) {
                            registered_kernel = true;
                        }
                    }
                }
            }
        }
    }

    if (registered_kernel) {
        return result;
    }

    ports[0].context_field_index = local.input_field;
    ports[0].require_complex     = needs_complex;
    ports[1].context_field_index = local.output_field;
    ports[1].require_complex     = needs_complex;

    accesses[0].port = 0U;
    accesses[0].mode = SIM_ACCESS_READ;
    accesses[1].port = 1U;
    accesses[1].mode = SIM_ACCESS_WRITE;

    substep.fn                = minimal_convolution_step;
    substep.accesses          = accesses;
    substep.access_count      = 2U;
    substep.barrier_after     = false;
    substep.error_measure     = NULL;
    substep.required_features = 0U;

    sim_operator_make_unique_name(name, sizeof(name), "minimal_convolution");

    desc.name                     = name;
    desc.ports                    = ports;
    desc.port_count               = 2U;
    desc.substeps                 = &substep;
    desc.substep_count            = 1U;
    desc.state                    = state;
    desc.symbolic                 = minimal_convolution_symbolic;
    desc.destroy                  = minimal_convolution_destroy;
    desc.info                     = info;
    desc.config                   = sim_operator_config_defaults();
    desc.scratch.bytes_per_worker = 0U;
    desc.scratch.alignment        = 0U;

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_minimal_convolution_config(struct SimContext*                   context,
                                         size_t                               operator_index,
                                         SimMinimalConvolutionOperatorConfig* out_config) {
    if (!context || !out_config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMinimalConvolutionOperatorState* state =
        (SimMinimalConvolutionOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_minimal_convolution_update(struct SimContext*                         context,
                                         size_t                                     operator_index,
                                         const SimMinimalConvolutionOperatorConfig* config) {
    if (!context || !config) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (!op) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimMinimalConvolutionOperatorState* state =
        (SimMinimalConvolutionOperatorState*) sim_operator_state(op);
    if (!state) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimMinimalConvolutionOperatorConfig local = *config;
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context, "minimal_convolution", true, config->scale_by_dt);
    if (!minimal_convolution_normalize_config(&local)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!minimal_convolution_validate_fields(context, &local)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->config        = local;
    state->kernel_length = local.kernel_length;
    memcpy(state->kernel, local.kernel, sizeof(state->kernel));
    state->kernel_rows = local.kernel_rows;
    state->kernel_cols = local.kernel_cols;
    memcpy(state->kernel_2d, local.kernel_2d, sizeof(state->kernel_2d));
    minimal_convolution_refresh_symbolic(state);
    state->kernel_cached_input      = NULL;
    state->kernel_cached_output     = NULL;
    state->kernel_cached_count      = 0U;
    state->kernel_cached_dt         = 0.0;
    state->kernel_cached_step_index = 0U;
    state->kernel_cache_valid       = false;

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
