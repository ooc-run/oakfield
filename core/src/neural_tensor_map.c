/**
 * @file neural_tensor_map.c
 * @brief SimField to canonical tensor view mapping for neural operators.
 *
 * The mapper converts field rank, channel placement, and scalar storage into a
 * tensor contract suitable for inference backends. It may expose a zero-copy
 * view or allocate staging buffers for layout/channel/complex conversions, then
 * commits output mappings back into the caller-owned field on unmap.
 */
#include "oakfield/neural_tensor_map.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct SimNeuralTensorPlan {
    SimNeuralTensorMapConfig  config;
    SimNeuralTensorValueType  value_type;
    size_t                    value_size;
    size_t                    tensor_rank;
    size_t                    tensor_shape[SIM_NEURAL_TENSOR_MAX_RANK];
    size_t                    tensor_strides[SIM_NEURAL_TENSOR_MAX_RANK];
    size_t                    tensor_values;
    size_t                    tensor_channel_axis;
    size_t                    logical_channels;
    size_t                    tensor_channels;
    bool                      has_field_channel_axis;
    size_t                    field_channel_axis;
    size_t                    field_spatial_rank;
    size_t                    field_spatial_axes[3];
} SimNeuralTensorPlan;

static void neural_tensor_mapping_reset(SimNeuralTensorMapping* mapping) {
    if (mapping == NULL) {
        return;
    }
    (void) memset(mapping, 0, sizeof(*mapping));
}

SimNeuralTensorMapConfig sim_neural_tensor_map_config_defaults(void) {
    SimNeuralTensorMapConfig config = { 0 };
    config.channel_axis              = SIM_NEURAL_TENSOR_CHANNEL_AXIS_AUTO;
    config.channels_last             = true;
    config.batch_size                = 1U;
    config.complex_mode              = SIM_NEURAL_TENSOR_COMPLEX_INTERLEAVED;
    config.affine.enabled            = false;
    config.affine.scale              = 1.0;
    config.affine.bias               = 0.0;
    return config;
}

static void neural_tensor_config_resolve(const SimNeuralTensorMapConfig* src,
                                         SimNeuralTensorMapConfig*       out_config) {
    SimNeuralTensorMapConfig config = sim_neural_tensor_map_config_defaults();
    if (out_config == NULL) {
        return;
    }
    if (src != NULL) {
        config = *src;
    }
    if (config.batch_size == 0U) {
        config.batch_size = 1U;
    }
    switch (config.complex_mode) {
        case SIM_NEURAL_TENSOR_COMPLEX_INTERLEAVED:
        case SIM_NEURAL_TENSOR_COMPLEX_SPLIT_CHANNELS:
            break;
        default:
            config.complex_mode = SIM_NEURAL_TENSOR_COMPLEX_INTERLEAVED;
            break;
    }
    if (!isfinite(config.affine.scale)) {
        config.affine.scale = 1.0;
    }
    if (!isfinite(config.affine.bias)) {
        config.affine.bias = 0.0;
    }
    *out_config = config;
}

static size_t neural_tensor_product(size_t rank, const size_t* shape) {
    size_t product = 1U;
    if (shape == NULL) {
        return 0U;
    }
    for (size_t i = 0U; i < rank; ++i) {
        if (shape[i] == 0U) {
            return 0U;
        }
        if (shape[i] > (SIZE_MAX / product)) {
            return 0U;
        }
        product *= shape[i];
    }
    return product;
}

static void neural_tensor_compute_contiguous_strides(size_t rank,
                                                     const size_t* shape,
                                                     size_t*       strides) {
    if (shape == NULL || strides == NULL || rank == 0U) {
        return;
    }
    strides[rank - 1U] = 1U;
    for (size_t axis = rank - 1U; axis > 0U; --axis) {
        strides[axis - 1U] = strides[axis] * shape[axis];
    }
}

static SimResult neural_tensor_plan_build(const SimField*                field,
                                          const SimNeuralTensorMapConfig* requested,
                                          SimNeuralTensorPlan*           out_plan) {
    SimNeuralTensorPlan      plan = { 0 };
    SimNeuralTensorMapConfig config;
    size_t                   rank;
    size_t                   spatial_cursor = 0U;
    size_t                   channels;
    bool                     field_complex;

    if (field == NULL || out_plan == NULL || field->layout.shape == NULL || field->layout.strides == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!(field->element_size == sizeof(double) || field->element_size == sizeof(SimComplexDouble))) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    rank = field->layout.rank;
    if (rank == 0U || rank > 3U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    neural_tensor_config_resolve(requested, &config);
    if (config.batch_size != 1U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    plan.config = config;

    if (config.channel_axis == SIM_NEURAL_TENSOR_CHANNEL_AXIS_AUTO) {
        if (rank == 1U) {
            plan.has_field_channel_axis = false;
            plan.field_channel_axis     = 0U;
        } else {
            plan.has_field_channel_axis = true;
            plan.field_channel_axis     = config.channels_last ? (rank - 1U) : 0U;
        }
    } else {
        if ((size_t) config.channel_axis >= rank) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        plan.has_field_channel_axis = true;
        plan.field_channel_axis     = config.channel_axis;
    }

    if (plan.has_field_channel_axis) {
        channels = field->layout.shape[plan.field_channel_axis];
    } else {
        channels = 1U;
    }
    if (channels == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    plan.logical_channels = channels;

    for (size_t axis = 0U; axis < rank; ++axis) {
        if (plan.has_field_channel_axis && axis == plan.field_channel_axis) {
            continue;
        }
        if (spatial_cursor >= 3U) {
            return SIM_RESULT_NOT_SUPPORTED;
        }
        plan.field_spatial_axes[spatial_cursor++] = axis;
    }
    plan.field_spatial_rank = spatial_cursor;

    field_complex = sim_field_is_complex(field);
    if (field_complex) {
        if (config.complex_mode == SIM_NEURAL_TENSOR_COMPLEX_SPLIT_CHANNELS) {
            if (channels > (SIZE_MAX / 2U)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            plan.tensor_channels = channels * 2U;
            plan.value_type      = SIM_NEURAL_TENSOR_VALUE_REAL_F64;
            plan.value_size      = sizeof(double);
        } else {
            plan.tensor_channels = channels;
            plan.value_type      = SIM_NEURAL_TENSOR_VALUE_COMPLEX_F64_INTERLEAVED;
            plan.value_size      = sizeof(SimComplexDouble);
        }
    } else {
        plan.tensor_channels = channels;
        plan.value_type      = SIM_NEURAL_TENSOR_VALUE_REAL_F64;
        plan.value_size      = sizeof(double);
    }

    plan.tensor_rank = 2U + plan.field_spatial_rank;
    if (plan.tensor_rank == 0U || plan.tensor_rank > SIM_NEURAL_TENSOR_MAX_RANK) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    if (config.channels_last) {
        plan.tensor_shape[0] = config.batch_size;
        for (size_t i = 0U; i < plan.field_spatial_rank; ++i) {
            plan.tensor_shape[1U + i] = field->layout.shape[plan.field_spatial_axes[i]];
        }
        plan.tensor_channel_axis                   = 1U + plan.field_spatial_rank;
        plan.tensor_shape[plan.tensor_channel_axis] = plan.tensor_channels;
    } else {
        plan.tensor_shape[0] = config.batch_size;
        plan.tensor_shape[1] = plan.tensor_channels;
        plan.tensor_channel_axis = 1U;
        for (size_t i = 0U; i < plan.field_spatial_rank; ++i) {
            plan.tensor_shape[2U + i] = field->layout.shape[plan.field_spatial_axes[i]];
        }
    }

    plan.tensor_values = neural_tensor_product(plan.tensor_rank, plan.tensor_shape);
    if (plan.tensor_values == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    neural_tensor_compute_contiguous_strides(plan.tensor_rank, plan.tensor_shape, plan.tensor_strides);

    *out_plan = plan;
    return SIM_RESULT_OK;
}

static bool neural_tensor_can_zero_copy(const SimField*          field,
                                        const SimNeuralTensorPlan* plan) {
    if (field == NULL || plan == NULL) {
        return false;
    }
    if (plan->config.affine.enabled) {
        return false;
    }
    if (field->storage != SIM_FIELD_STORAGE_ROW_MAJOR || !field->layout.contiguous) {
        return false;
    }
    if (plan->config.batch_size != 1U) {
        return false;
    }
    if (sim_field_is_complex(field) &&
        plan->config.complex_mode != SIM_NEURAL_TENSOR_COMPLEX_INTERLEAVED) {
        return false;
    }
    if (plan->has_field_channel_axis) {
        if (plan->config.channels_last) {
            if (plan->field_channel_axis != (size_t) (field->layout.rank - 1U)) {
                return false;
            }
        } else {
            if (plan->field_channel_axis != 0U) {
                return false;
            }
        }
    }
    return true;
}

static void neural_tensor_decode_index(size_t rank,
                                       const size_t* shape,
                                       size_t        linear_index,
                                       size_t*       out_indices) {
    if (shape == NULL || out_indices == NULL) {
        return;
    }
    for (size_t axis = rank; axis-- > 0U;) {
        size_t dim = shape[axis];
        if (dim == 0U) {
            out_indices[axis] = 0U;
            continue;
        }
        out_indices[axis] = linear_index % dim;
        linear_index /= dim;
    }
}

static size_t neural_field_data_offset(const SimFieldLayout* layout, const size_t* logical_indices) {
    size_t offset = 0U;
    if (layout == NULL || logical_indices == NULL || layout->strides == NULL) {
        return 0U;
    }
    for (size_t axis = 0U; axis < layout->rank; ++axis) {
        offset += logical_indices[axis] * layout->strides[axis];
    }
    return offset;
}

static size_t neural_tensor_value_offset(const SimNeuralTensorPlan* plan,
                                         size_t                     channel_index,
                                         const size_t*              spatial_indices) {
    size_t offset         = 0U;
    size_t spatial_cursor = 0U;
    if (plan == NULL) {
        return 0U;
    }
    for (size_t axis = 0U; axis < plan->tensor_rank; ++axis) {
        size_t index = 0U;
        if (axis == 0U) {
            index = 0U; /* batch index */
        } else if (axis == plan->tensor_channel_axis) {
            index = channel_index;
        } else if (spatial_indices != NULL && spatial_cursor < plan->field_spatial_rank) {
            index = spatial_indices[spatial_cursor++];
        }
        offset += index * plan->tensor_strides[axis];
    }
    return offset;
}

static double neural_tensor_affine_apply(double value, const SimNeuralTensorAffineHook* affine) {
    if (affine == NULL || !affine->enabled) {
        return value;
    }
    return value * affine->scale + affine->bias;
}

static SimResult neural_tensor_copy_from_field(const SimField*            field,
                                               const SimNeuralTensorPlan* plan,
                                               SimNeuralTensorMapping*    mapping) {
    size_t logical_count;

    if (field == NULL || plan == NULL || mapping == NULL || mapping->view.data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    logical_count = sim_field_element_count(&field->layout);
    if (logical_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_is_complex(field)) {
        const SimComplexDouble* src = sim_field_complex_data_const(field);
        if (src == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t linear = 0U; linear < logical_count; ++linear) {
            size_t field_indices[3] = { 0U, 0U, 0U };
            size_t spatial_indices[3] = { 0U, 0U, 0U };
            size_t field_offset;
            size_t channel = 0U;
            SimComplexDouble value;

            neural_tensor_decode_index(field->layout.rank, field->layout.shape, linear, field_indices);
            field_offset = neural_field_data_offset(&field->layout, field_indices);
            value        = src[field_offset];

            if (plan->has_field_channel_axis) {
                channel = field_indices[plan->field_channel_axis];
            }
            for (size_t i = 0U; i < plan->field_spatial_rank; ++i) {
                spatial_indices[i] = field_indices[plan->field_spatial_axes[i]];
            }

            if (plan->value_type == SIM_NEURAL_TENSOR_VALUE_COMPLEX_F64_INTERLEAVED) {
                SimComplexDouble* dst = (SimComplexDouble*) mapping->view.data;
                size_t            offset =
                    neural_tensor_value_offset(plan, channel, spatial_indices);
                value.re = neural_tensor_affine_apply(value.re, &plan->config.affine);
                value.im = neural_tensor_affine_apply(value.im, &plan->config.affine);
                if (plan->config.affine.enabled) {
                    mapping->metrics.affine_element_ops += 2U;
                }
                dst[offset] = value;
            } else {
                double* dst      = (double*) mapping->view.data;
                size_t  ch_re    = channel * 2U;
                size_t  ch_im    = ch_re + 1U;
                size_t  off_re   = neural_tensor_value_offset(plan, ch_re, spatial_indices);
                size_t  off_im   = neural_tensor_value_offset(plan, ch_im, spatial_indices);
                double  re_value = neural_tensor_affine_apply(value.re, &plan->config.affine);
                double  im_value = neural_tensor_affine_apply(value.im, &plan->config.affine);
                if (plan->config.affine.enabled) {
                    mapping->metrics.affine_element_ops += 2U;
                }
                dst[off_re] = re_value;
                dst[off_im] = im_value;
            }
        }
    } else {
        const double* src = sim_field_real_data_const(field);
        if (src == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t linear = 0U; linear < logical_count; ++linear) {
            size_t field_indices[3] = { 0U, 0U, 0U };
            size_t spatial_indices[3] = { 0U, 0U, 0U };
            size_t field_offset;
            size_t channel = 0U;
            size_t tensor_offset;
            double value;

            neural_tensor_decode_index(field->layout.rank, field->layout.shape, linear, field_indices);
            field_offset = neural_field_data_offset(&field->layout, field_indices);
            value        = src[field_offset];

            if (plan->has_field_channel_axis) {
                channel = field_indices[plan->field_channel_axis];
            }
            for (size_t i = 0U; i < plan->field_spatial_rank; ++i) {
                spatial_indices[i] = field_indices[plan->field_spatial_axes[i]];
            }

            value         = neural_tensor_affine_apply(value, &plan->config.affine);
            tensor_offset = neural_tensor_value_offset(plan, channel, spatial_indices);
            if (plan->config.affine.enabled) {
                mapping->metrics.affine_element_ops += 1U;
            }
            ((double*) mapping->view.data)[tensor_offset] = value;
        }
    }

    mapping->metrics.bytes_copied_from_field = logical_count * field->element_size;
    return SIM_RESULT_OK;
}

static SimResult neural_tensor_copy_to_field(SimField*                  field,
                                             const SimNeuralTensorPlan* plan,
                                             SimNeuralTensorMapping*    mapping) {
    size_t logical_count;

    if (field == NULL || plan == NULL || mapping == NULL || mapping->view.data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    logical_count = sim_field_element_count(&field->layout);
    if (logical_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_is_complex(field)) {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t linear = 0U; linear < logical_count; ++linear) {
            size_t field_indices[3] = { 0U, 0U, 0U };
            size_t spatial_indices[3] = { 0U, 0U, 0U };
            size_t field_offset;
            size_t channel = 0U;
            SimComplexDouble value = { 0.0, 0.0 };

            neural_tensor_decode_index(field->layout.rank, field->layout.shape, linear, field_indices);
            field_offset = neural_field_data_offset(&field->layout, field_indices);

            if (plan->has_field_channel_axis) {
                channel = field_indices[plan->field_channel_axis];
            }
            for (size_t i = 0U; i < plan->field_spatial_rank; ++i) {
                spatial_indices[i] = field_indices[plan->field_spatial_axes[i]];
            }

            if (plan->value_type == SIM_NEURAL_TENSOR_VALUE_COMPLEX_F64_INTERLEAVED) {
                const SimComplexDouble* src = (const SimComplexDouble*) mapping->view.data;
                size_t                  offset =
                    neural_tensor_value_offset(plan, channel, spatial_indices);
                value = src[offset];
            } else {
                const double* src   = (const double*) mapping->view.data;
                size_t        ch_re = channel * 2U;
                size_t        ch_im = ch_re + 1U;
                size_t        off_re =
                    neural_tensor_value_offset(plan, ch_re, spatial_indices);
                size_t off_im = neural_tensor_value_offset(plan, ch_im, spatial_indices);
                value.re      = src[off_re];
                value.im      = src[off_im];
            }

            value.re = neural_tensor_affine_apply(value.re, &plan->config.affine);
            value.im = neural_tensor_affine_apply(value.im, &plan->config.affine);
            if (plan->config.affine.enabled) {
                mapping->metrics.affine_element_ops += 2U;
            }
            dst[field_offset] = value;
        }
    } else {
        SimComplexDouble const* src_complex = NULL;
        double const*           src_real    = NULL;
        double*                 dst         = sim_field_real_data(field);

        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (plan->value_type == SIM_NEURAL_TENSOR_VALUE_COMPLEX_F64_INTERLEAVED) {
            src_complex = (const SimComplexDouble*) mapping->view.data;
        } else {
            src_real = (const double*) mapping->view.data;
        }

        for (size_t linear = 0U; linear < logical_count; ++linear) {
            size_t field_indices[3] = { 0U, 0U, 0U };
            size_t spatial_indices[3] = { 0U, 0U, 0U };
            size_t field_offset;
            size_t channel = 0U;
            size_t tensor_offset;
            double value;

            neural_tensor_decode_index(field->layout.rank, field->layout.shape, linear, field_indices);
            field_offset = neural_field_data_offset(&field->layout, field_indices);

            if (plan->has_field_channel_axis) {
                channel = field_indices[plan->field_channel_axis];
            }
            for (size_t i = 0U; i < plan->field_spatial_rank; ++i) {
                spatial_indices[i] = field_indices[plan->field_spatial_axes[i]];
            }

            tensor_offset = neural_tensor_value_offset(plan, channel, spatial_indices);
            if (src_complex != NULL) {
                value = src_complex[tensor_offset].re;
            } else {
                value = src_real[tensor_offset];
            }

            value = neural_tensor_affine_apply(value, &plan->config.affine);
            if (plan->config.affine.enabled) {
                mapping->metrics.affine_element_ops += 1U;
            }
            dst[field_offset] = value;
        }
    }

    mapping->metrics.bytes_copied_to_field = logical_count * field->element_size;
    return SIM_RESULT_OK;
}

static void neural_tensor_mapping_set_view(SimNeuralTensorMapping*   mapping,
                                           const SimNeuralTensorPlan* plan,
                                           void*                      data_ptr) {
    if (mapping == NULL || plan == NULL) {
        return;
    }
    mapping->view.data         = data_ptr;
    mapping->view.value_type   = plan->value_type;
    mapping->view.value_size   = plan->value_size;
    mapping->view.rank         = plan->tensor_rank;
    mapping->view.batch_axis   = 0U;
    mapping->view.channel_axis = (uint8_t) plan->tensor_channel_axis;
    mapping->view.spatial_rank = (uint8_t) plan->field_spatial_rank;
    for (size_t i = 0U; i < SIM_NEURAL_TENSOR_MAX_RANK; ++i) {
        if (i < plan->tensor_rank) {
            mapping->view.shape[i]   = plan->tensor_shape[i];
            mapping->view.strides[i] = plan->tensor_strides[i];
        } else {
            mapping->view.shape[i]   = 0U;
            mapping->view.strides[i] = 0U;
        }
    }
}

static SimResult neural_tensor_map_common(const SimField*                field_const,
                                          SimField*                      field_mut,
                                          const SimNeuralTensorMapConfig* requested,
                                          bool                           output_mode,
                                          SimNeuralTensorMapping*        out_mapping) {
    SimNeuralTensorPlan plan;
    SimResult           result;
    bool                zero_copy;
    size_t              bytes;
    void*               data_ptr;

    if (out_mapping == NULL || field_const == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (output_mode && field_mut == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    result = neural_tensor_plan_build(field_const, requested, &plan);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    neural_tensor_mapping_reset(out_mapping);
    out_mapping->config                 = plan.config;
    out_mapping->field_const            = field_const;
    out_mapping->field_mut              = field_mut;
    out_mapping->output_mode            = output_mode;
    out_mapping->active                 = true;
    out_mapping->has_field_channel_axis = plan.has_field_channel_axis;
    out_mapping->field_channel_axis     = (uint8_t) plan.field_channel_axis;
    out_mapping->field_rank             = (uint8_t) field_const->layout.rank;
    out_mapping->field_spatial_rank     = (uint8_t) plan.field_spatial_rank;
    for (size_t i = 0U; i < 3U; ++i) {
        out_mapping->field_spatial_axes[i] = (uint8_t) plan.field_spatial_axes[i];
    }

    zero_copy = neural_tensor_can_zero_copy(field_const, &plan);
    out_mapping->metrics.zero_copy          = zero_copy;
    out_mapping->metrics.used_copy_fallback = !zero_copy;

    if (zero_copy) {
        data_ptr = output_mode ? sim_field_data(field_mut) : (void*) sim_field_data_const(field_const);
        if (data_ptr == NULL) {
            neural_tensor_mapping_reset(out_mapping);
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        out_mapping->scratch      = NULL;
        out_mapping->scratch_bytes = 0U;
        neural_tensor_mapping_set_view(out_mapping, &plan, data_ptr);
        return SIM_RESULT_OK;
    }

    if (plan.value_size > 0U && plan.tensor_values > (SIZE_MAX / plan.value_size)) {
        neural_tensor_mapping_reset(out_mapping);
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    bytes = plan.tensor_values * plan.value_size;
    if (bytes == 0U) {
        neural_tensor_mapping_reset(out_mapping);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    out_mapping->scratch = calloc(1U, bytes);
    if (out_mapping->scratch == NULL) {
        neural_tensor_mapping_reset(out_mapping);
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    out_mapping->scratch_bytes = bytes;
    neural_tensor_mapping_set_view(out_mapping, &plan, out_mapping->scratch);

    if (!output_mode) {
        result = neural_tensor_copy_from_field(field_const, &plan, out_mapping);
        if (result != SIM_RESULT_OK) {
            sim_neural_tensor_mapping_release(out_mapping);
            return result;
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_neural_tensor_map_input(const SimField*                 field,
                                      const SimNeuralTensorMapConfig* config,
                                      SimNeuralTensorMapping*         out_mapping) {
    return neural_tensor_map_common(field, NULL, config, false, out_mapping);
}

SimResult sim_neural_tensor_map_output(SimField*                      field,
                                       const SimNeuralTensorMapConfig* config,
                                       SimNeuralTensorMapping*        out_mapping) {
    return neural_tensor_map_common((const SimField*) field, field, config, true, out_mapping);
}

SimResult sim_neural_tensor_unmap_output(SimNeuralTensorMapping* mapping) {
    SimNeuralTensorPlan plan;
    SimResult           result;

    if (mapping == NULL || !mapping->active || !mapping->output_mode || mapping->field_mut == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (mapping->metrics.zero_copy) {
        return SIM_RESULT_OK;
    }

    result = neural_tensor_plan_build(mapping->field_mut, &mapping->config, &plan);
    if (result != SIM_RESULT_OK) {
        return result;
    }
    return neural_tensor_copy_to_field(mapping->field_mut, &plan, mapping);
}

void sim_neural_tensor_mapping_release(SimNeuralTensorMapping* mapping) {
    if (mapping == NULL) {
        return;
    }
    free(mapping->scratch);
    neural_tensor_mapping_reset(mapping);
}
