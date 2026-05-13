/**
 * @file sim_buffer.c
 * @brief Standalone typed buffer allocation and generic view indexing.
 *
 * SimBuffer owns contiguous typed storage and shape/stride metadata, while
 * SimBufferView exposes caller-owned storage through linear or strided indexing.
 * Helpers convert scalar domains to storage types, validate offsets, and clone
 * non-contiguous views into compact row-major buffers.
 */
#include "oakfield/sim_buffer.h"

#include <stdlib.h>
#include <string.h>

static bool sim_buffer_shape_count(size_t rank, const size_t* shape, size_t* out_count) {
    size_t count = 1U;

    if (out_count != NULL) {
        *out_count = 0U;
    }
    if (rank == 0U || shape == NULL) {
        return false;
    }

    for (size_t i = 0U; i < rank; ++i) {
        if (shape[i] != 0U && count > ((size_t) -1) / shape[i]) {
            return false;
        }
        count *= shape[i];
    }

    if (out_count != NULL) {
        *out_count = count;
    }
    return true;
}

static bool sim_buffer_layout_count(const SimFieldLayout* layout, size_t* out_count) {
    if (out_count != NULL) {
        *out_count = 0U;
    }
    if (layout == NULL || layout->rank == 0U || layout->shape == NULL) {
        return false;
    }
    return sim_buffer_shape_count(layout->rank, layout->shape, out_count);
}

static bool sim_buffer_shape_linear_index(size_t        rank,
                                          const size_t* shape,
                                          const size_t* indices,
                                          size_t*       out_linear_index) {
    size_t linear_index = 0U;
    size_t stride       = 1U;

    if (out_linear_index != NULL) {
        *out_linear_index = 0U;
    }
    if (rank == 0U || shape == NULL || indices == NULL) {
        return false;
    }

    for (size_t axis = rank; axis-- > 0U;) {
        size_t term = 0U;
        if (indices[axis] >= shape[axis]) {
            return false;
        }
        if (indices[axis] > 0U && stride > ((size_t) -1) / indices[axis]) {
            return false;
        }
        term = indices[axis] * stride;
        if (linear_index > ((size_t) -1) - term) {
            return false;
        }
        linear_index += term;
        if (shape[axis] > 0U && stride > ((size_t) -1) / shape[axis]) {
            return false;
        }
        stride *= (shape[axis] > 0U) ? shape[axis] : 1U;
    }

    if (out_linear_index != NULL) {
        *out_linear_index = linear_index;
    }
    return true;
}

static void sim_buffer_layout_reset(SimBuffer* buffer) {
    if (buffer == NULL) {
        return;
    }

    free(buffer->shape_storage);
    free(buffer->stride_storage);
    buffer->shape_storage    = NULL;
    buffer->stride_storage   = NULL;
    buffer->layout.rank      = 0U;
    buffer->layout.shape     = NULL;
    buffer->layout.strides   = NULL;
    buffer->layout.contiguous = false;
}

static bool sim_buffer_layout_assign(SimBuffer* buffer, size_t rank, const size_t* shape) {
    size_t* local_shape   = NULL;
    size_t* local_strides = NULL;
    size_t  stride        = 1U;

    if (buffer == NULL || rank == 0U || shape == NULL) {
        return false;
    }

    local_shape = (size_t*) calloc(rank, sizeof(size_t));
    local_strides = (size_t*) calloc(rank, sizeof(size_t));
    if (local_shape == NULL || local_strides == NULL) {
        free(local_shape);
        free(local_strides);
        return false;
    }

    for (size_t i = 0U; i < rank; ++i) {
        local_shape[i] = shape[i];
    }
    for (size_t i = rank; i-- > 0U;) {
        local_strides[i] = stride;
        if (local_shape[i] > 0U && stride > ((size_t) -1) / local_shape[i]) {
            free(local_shape);
            free(local_strides);
            return false;
        }
        stride *= (local_shape[i] > 0U) ? local_shape[i] : 1U;
    }

    sim_buffer_layout_reset(buffer);
    buffer->shape_storage     = local_shape;
    buffer->stride_storage    = local_strides;
    buffer->layout.rank       = rank;
    buffer->layout.shape      = buffer->shape_storage;
    buffer->layout.strides    = buffer->stride_storage;
    buffer->layout.contiguous = true;
    return true;
}

size_t sim_buffer_element_size(SimBufferDataType type) {
    switch (type) {
        case SIM_BUFFER_DOUBLE:
            return sizeof(double);
        case SIM_BUFFER_COMPLEX_DOUBLE:
            return sizeof(SimComplexDouble);
        case SIM_BUFFER_I8:
            return sizeof(int8_t);
        case SIM_BUFFER_U8:
            return sizeof(uint8_t);
        case SIM_BUFFER_I32:
            return sizeof(int32_t);
        case SIM_BUFFER_I64:
            return sizeof(int64_t);
        case SIM_BUFFER_U32:
            return sizeof(uint32_t);
        case SIM_BUFFER_U64:
            return sizeof(uint64_t);
        case SIM_BUFFER_UNKNOWN:
        default:
            return 0U;
    }
}

SimBufferDataType sim_buffer_data_type_from_element_size(size_t element_size) {
    if (element_size == sizeof(double)) {
        return SIM_BUFFER_DOUBLE;
    }
    if (element_size == sizeof(SimComplexDouble)) {
        return SIM_BUFFER_COMPLEX_DOUBLE;
    }
    return SIM_BUFFER_UNKNOWN;
}

SimBufferDataType sim_buffer_data_type_from_scalar_domain(SimScalarDomain domain) {
    if (!sim_scalar_domain_validate(domain)) {
        return SIM_BUFFER_UNKNOWN;
    }

    switch (domain.kind) {
        case SIM_SCALAR_DOMAIN_REAL:
            return (domain.bit_width == 64U) ? SIM_BUFFER_DOUBLE : SIM_BUFFER_UNKNOWN;
        case SIM_SCALAR_DOMAIN_COMPLEX:
            return (domain.bit_width == 64U) ? SIM_BUFFER_COMPLEX_DOUBLE : SIM_BUFFER_UNKNOWN;
        case SIM_SCALAR_DOMAIN_INTEGER:
            if (domain.bit_width == 8U) {
                return domain.is_signed ? SIM_BUFFER_I8 : SIM_BUFFER_U8;
            }
            if (domain.bit_width == 32U) {
                return domain.is_signed ? SIM_BUFFER_I32 : SIM_BUFFER_U32;
            }
            if (domain.bit_width == 64U) {
                return domain.is_signed ? SIM_BUFFER_I64 : SIM_BUFFER_U64;
            }
            return SIM_BUFFER_UNKNOWN;
        case SIM_SCALAR_DOMAIN_UNKNOWN:
        case SIM_SCALAR_DOMAIN_MODULAR:
        default:
            return SIM_BUFFER_UNKNOWN;
    }
}

SimScalarDomain sim_buffer_data_type_to_scalar_domain(SimBufferDataType type) {
    switch (type) {
        case SIM_BUFFER_DOUBLE:
            return sim_scalar_domain_f64();
        case SIM_BUFFER_COMPLEX_DOUBLE:
            return sim_scalar_domain_c64();
        case SIM_BUFFER_I8:
            return sim_scalar_domain_i8();
        case SIM_BUFFER_U8:
            return sim_scalar_domain_u8();
        case SIM_BUFFER_I32:
            return sim_scalar_domain_i32();
        case SIM_BUFFER_I64:
            return sim_scalar_domain_i64();
        case SIM_BUFFER_U32:
            return sim_scalar_domain_u32();
        case SIM_BUFFER_U64:
            return sim_scalar_domain_u64();
        case SIM_BUFFER_UNKNOWN:
        default:
            return sim_scalar_domain_unknown();
    }
}

const char* sim_buffer_data_type_name(SimBufferDataType type) {
    switch (type) {
        case SIM_BUFFER_DOUBLE:
            return "double";
        case SIM_BUFFER_COMPLEX_DOUBLE:
            return "complex_double";
        case SIM_BUFFER_I8:
            return "i8";
        case SIM_BUFFER_U8:
            return "u8";
        case SIM_BUFFER_I32:
            return "i32";
        case SIM_BUFFER_I64:
            return "i64";
        case SIM_BUFFER_U32:
            return "u32";
        case SIM_BUFFER_U64:
            return "u64";
        case SIM_BUFFER_UNKNOWN:
        default:
            return "unknown";
    }
}

bool sim_buffer_view_is_valid(const SimBufferView* view) {
    return view != NULL && view->data != NULL && view->count > 0U &&
           sim_buffer_element_size(view->type) > 0U;
}

bool sim_buffer_view_offset_for_indices(const SimBufferView* view,
                                        const size_t*        indices,
                                        size_t               index_count,
                                        size_t*              out_offset) {
    size_t offset = 0U;
    size_t logical_index = 0U;

    if (out_offset != NULL) {
        *out_offset = 0U;
    }
    if (view == NULL || indices == NULL) {
        return false;
    }

    if (view->layout.rank == 0U || view->layout.shape == NULL || view->layout.strides == NULL) {
        if (index_count != 1U || indices[0] >= view->count) {
            return false;
        }
        if (out_offset != NULL) {
            *out_offset = indices[0];
        }
        return true;
    }

    if (index_count != view->layout.rank) {
        return false;
    }

    if (view->logical_to_physical != NULL) {
        if (!sim_buffer_shape_linear_index(
                view->layout.rank, view->layout.shape, indices, &logical_index)) {
            return false;
        }
        return sim_buffer_view_offset_for_linear_index(view, logical_index, out_offset);
    }

    for (size_t axis = 0U; axis < view->layout.rank; ++axis) {
        if (indices[axis] >= view->layout.shape[axis]) {
            return false;
        }
        offset += indices[axis] * view->layout.strides[axis];
    }

    if (out_offset != NULL) {
        *out_offset = offset;
    }
    return true;
}

bool sim_buffer_view_offset_for_linear_index(const SimBufferView* view,
                                             size_t               logical_index,
                                             size_t*              out_offset) {
    size_t offset = 0U;
    size_t count  = 0U;

    if (out_offset != NULL) {
        *out_offset = 0U;
    }
    if (view == NULL || logical_index >= view->count) {
        return false;
    }

    if (view->logical_to_physical != NULL) {
        if (out_offset != NULL) {
            *out_offset = view->logical_to_physical[logical_index];
        }
        return true;
    }

    if (view->layout.rank == 0U || view->layout.shape == NULL || view->layout.strides == NULL) {
        if (out_offset != NULL) {
            *out_offset = logical_index;
        }
        return true;
    }

    if (!sim_buffer_layout_count(&view->layout, &count) || count != view->count) {
        return false;
    }

    for (size_t axis = view->layout.rank; axis-- > 0U;) {
        size_t dim = view->layout.shape[axis];
        size_t idx = 0U;
        if (dim == 0U) {
            return false;
        }
        idx = logical_index % dim;
        logical_index /= dim;
        offset += idx * view->layout.strides[axis];
    }

    if (logical_index != 0U) {
        return false;
    }
    if (out_offset != NULL) {
        *out_offset = offset;
    }
    return true;
}

bool sim_buffer_view_get_complex(const SimBufferView* view,
                                 size_t               logical_index,
                                 SimComplexDouble*    out_value) {
    size_t offset = 0U;

    if (out_value != NULL) {
        out_value->re = 0.0;
        out_value->im = 0.0;
    }
    if (view == NULL || out_value == NULL || view->data == NULL ||
        !sim_buffer_view_offset_for_linear_index(view, logical_index, &offset)) {
        return false;
    }

    switch (view->type) {
        case SIM_BUFFER_DOUBLE: {
            const double* data = (const double*) view->data;
            out_value->re      = data[offset];
            out_value->im      = 0.0;
            return true;
        }
        case SIM_BUFFER_COMPLEX_DOUBLE: {
            const SimComplexDouble* data = (const SimComplexDouble*) view->data;
            *out_value                  = data[offset];
            return true;
        }
        case SIM_BUFFER_I8: {
            const int8_t* data = (const int8_t*) view->data;
            out_value->re      = (double) data[offset];
            out_value->im      = 0.0;
            return true;
        }
        case SIM_BUFFER_U8: {
            const uint8_t* data = (const uint8_t*) view->data;
            out_value->re       = (double) data[offset];
            out_value->im       = 0.0;
            return true;
        }
        case SIM_BUFFER_I32: {
            const int32_t* data = (const int32_t*) view->data;
            out_value->re       = (double) data[offset];
            out_value->im       = 0.0;
            return true;
        }
        case SIM_BUFFER_U32: {
            const uint32_t* data = (const uint32_t*) view->data;
            out_value->re        = (double) data[offset];
            out_value->im        = 0.0;
            return true;
        }
        case SIM_BUFFER_I64: {
            const int64_t* data = (const int64_t*) view->data;
            out_value->re       = (double) data[offset];
            out_value->im       = 0.0;
            return true;
        }
        case SIM_BUFFER_U64: {
            const uint64_t* data = (const uint64_t*) view->data;
            out_value->re        = (double) data[offset];
            out_value->im        = 0.0;
            return true;
        }
        case SIM_BUFFER_UNKNOWN:
        default:
            return false;
    }
}

bool sim_buffer_view_set_complex(SimBufferView*   view,
                                 size_t           logical_index,
                                 SimComplexDouble value) {
    size_t offset = 0U;

    if (view == NULL || view->data == NULL ||
        !sim_buffer_view_offset_for_linear_index(view, logical_index, &offset)) {
        return false;
    }

    switch (view->type) {
        case SIM_BUFFER_DOUBLE: {
            double* data = (double*) view->data;
            if (value.im != 0.0) {
                return false;
            }
            data[offset] = value.re;
            return true;
        }
        case SIM_BUFFER_COMPLEX_DOUBLE: {
            SimComplexDouble* data = (SimComplexDouble*) view->data;
            data[offset]           = value;
            return true;
        }
        case SIM_BUFFER_UNKNOWN:
        default:
            return false;
    }
}

SimBuffer* sim_buffer_create(size_t rank, const size_t* shape, SimBufferDataType type) {
    SimBuffer* buffer       = NULL;
    size_t     element_size = sim_buffer_element_size(type);
    size_t     count        = 0U;
    size_t     bytes        = 0U;

    if (element_size == 0U || !sim_buffer_shape_count(rank, shape, &count)) {
        return NULL;
    }
    if (count > ((size_t) -1) / element_size) {
        return NULL;
    }

    buffer = (SimBuffer*) calloc(1U, sizeof(*buffer));
    if (buffer == NULL) {
        return NULL;
    }

    if (!sim_buffer_layout_assign(buffer, rank, shape)) {
        free(buffer);
        return NULL;
    }

    bytes = count * element_size;
    if (bytes > 0U) {
        buffer->data = calloc(count, element_size);
        if (buffer->data == NULL) {
            sim_buffer_layout_reset(buffer);
            free(buffer);
            return NULL;
        }
    }

    buffer->count = count;
    buffer->bytes = bytes;
    buffer->type  = type;
    return buffer;
}

SimBuffer* sim_buffer_create_1d(size_t count, SimBufferDataType type) {
    size_t shape[1] = { count };
    return sim_buffer_create(1U, shape, type);
}

SimBuffer* sim_buffer_clone_view(const SimBufferView* view) {
    SimBuffer* buffer = NULL;
    size_t     shape_1d[1]   = { 0U };
    size_t     element_size  = 0U;
    unsigned char* dst_bytes = NULL;
    const unsigned char* src_bytes = NULL;

    if (view == NULL || sim_buffer_element_size(view->type) == 0U) {
        return NULL;
    }
    element_size = sim_buffer_element_size(view->type);

    if (view->layout.rank > 0U && view->layout.shape != NULL) {
        size_t count = 0U;
        if (!sim_buffer_layout_count(&view->layout, &count) || count != view->count) {
            return NULL;
        }
        buffer = sim_buffer_create(view->layout.rank, view->layout.shape, view->type);
    } else {
        shape_1d[0] = view->count;
        buffer      = sim_buffer_create(1U, shape_1d, view->type);
    }
    if (buffer == NULL) {
        return NULL;
    }

    if (view->count == 0U) {
        return buffer;
    }

    dst_bytes = (unsigned char*) sim_buffer_data(buffer);
    src_bytes = (const unsigned char*) view->data;
    if (dst_bytes == NULL || src_bytes == NULL) {
        sim_buffer_destroy(buffer);
        return NULL;
    }

    for (size_t i = 0U; i < view->count; ++i) {
        size_t offset = 0U;
        if (!sim_buffer_view_offset_for_linear_index(view, i, &offset)) {
            sim_buffer_destroy(buffer);
            return NULL;
        }
        memcpy(dst_bytes + i * element_size, src_bytes + offset * element_size, element_size);
    }
    return buffer;
}

SimResult sim_buffer_reshape(SimBuffer* buffer, size_t rank, const size_t* shape) {
    size_t new_count = 0U;

    if (buffer == NULL || !sim_buffer_shape_count(rank, shape, &new_count)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (new_count != buffer->count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_buffer_layout_assign(buffer, rank, shape)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    return SIM_RESULT_OK;
}

void sim_buffer_destroy(SimBuffer* buffer) {
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    buffer->data = NULL;
    sim_buffer_layout_reset(buffer);
    free(buffer);
}

void* sim_buffer_data(SimBuffer* buffer) {
    return (buffer != NULL) ? buffer->data : NULL;
}

const void* sim_buffer_const_data(const SimBuffer* buffer) {
    return (buffer != NULL) ? buffer->data : NULL;
}

size_t sim_buffer_count(const SimBuffer* buffer) {
    return (buffer != NULL) ? buffer->count : 0U;
}

size_t sim_buffer_bytes(const SimBuffer* buffer) {
    return (buffer != NULL) ? buffer->bytes : 0U;
}

SimBufferDataType sim_buffer_type(const SimBuffer* buffer) {
    return (buffer != NULL) ? buffer->type : SIM_BUFFER_UNKNOWN;
}

const SimFieldLayout* sim_buffer_layout(const SimBuffer* buffer) {
    return (buffer != NULL) ? &buffer->layout : NULL;
}

SimBufferView sim_buffer_view(const SimBuffer* buffer) {
    SimBufferView view = { 0 };

    if (buffer == NULL) {
        return view;
    }

    view.data   = buffer->data;
    view.count  = buffer->count;
    view.type   = buffer->type;
    view.layout = buffer->layout;
    view.logical_to_physical = NULL;
    return view;
}
