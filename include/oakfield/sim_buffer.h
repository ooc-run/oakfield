/**
 * @file sim_buffer.h
 * @brief Shared standalone buffer and buffer-view types.
 * @ingroup oakfield_buffers
 *
 * Buffers provide small owning allocations for typed scalar data outside the
 * full SimField lifecycle. Views describe caller-owned storage with optional
 * shape/stride metadata and logical-to-physical remapping, which lets patch and
 * neural helpers read non-contiguous data through one indexing contract.
 */
#ifndef OAKFIELD_SIM_BUFFER_H
#define OAKFIELD_SIM_BUFFER_H

#include <stdbool.h>
#include <stddef.h>

#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scalar storage types supported by standalone buffers and generic views.
 */
typedef enum SimBufferDataType {
    SIM_BUFFER_DOUBLE = 0,     /**< Double-precision real scalar storage. */
    SIM_BUFFER_COMPLEX_DOUBLE, /**< Interleaved SimComplexDouble storage. */
    SIM_BUFFER_UNKNOWN,        /**< Unknown or unsupported storage type. */
    SIM_BUFFER_I32,            /**< Signed 32-bit integer storage. */
    SIM_BUFFER_I64,            /**< Signed 64-bit integer storage. */
    SIM_BUFFER_U32,            /**< Unsigned 32-bit integer storage. */
    SIM_BUFFER_U64,            /**< Unsigned 64-bit integer storage. */
    SIM_BUFFER_I8,             /**< Signed 8-bit integer storage. */
    SIM_BUFFER_U8              /**< Unsigned 8-bit integer storage. */
} SimBufferDataType;

/**
 * @brief Non-owning typed view over contiguous, strided, or indexed scalar storage.
 */
typedef struct SimBufferView {
    void *data;                        /**< Caller-owned element storage. */
    size_t count;                      /**< Logical element count. */
    SimBufferDataType type;            /**< Scalar storage type for each element. */
    SimFieldLayout layout;             /**< Optional shape/stride metadata. */
    const size_t *logical_to_physical; /**< Optional linear logical-to-physical index map. */
} SimBufferView;

/**
 * @brief Owning contiguous typed buffer for research-facing bulk APIs.
 */
typedef struct SimBuffer {
    void *data;             /**< Owned element storage. */
    size_t count;           /**< Number of logical elements. */
    size_t bytes;           /**< Allocation size in bytes. */
    SimBufferDataType type; /**< Scalar storage type for each element. */
    size_t *shape_storage;  /**< Owned shape array backing layout.shape. */
    size_t *stride_storage; /**< Owned row-major stride array backing layout.strides. */
    SimFieldLayout layout;  /**< Public layout view over owned shape/stride arrays. */
} SimBuffer;

/**
 * @brief Return the storage size of one buffer element.
 *
 * @param type Buffer storage type.
 * @return Element size in bytes, or 0 for `SIM_BUFFER_UNKNOWN`.
 */
size_t sim_buffer_element_size(SimBufferDataType type);

/**
 * @brief Infer a legacy buffer type from an element byte size.
 *
 * Only double and SimComplexDouble are recognized by size alone.
 *
 * @param element_size Size in bytes.
 * @return Matching buffer type, or `SIM_BUFFER_UNKNOWN`.
 */
SimBufferDataType sim_buffer_data_type_from_element_size(size_t element_size);

/**
 * @brief Map a scalar-domain descriptor to a supported buffer storage type.
 *
 * @param domain Valid scalar domain descriptor.
 * @return Matching buffer type, or `SIM_BUFFER_UNKNOWN` for unsupported domains.
 */
SimBufferDataType sim_buffer_data_type_from_scalar_domain(SimScalarDomain domain);

/**
 * @brief Map a buffer storage type to its scalar-domain descriptor.
 *
 * @param type Buffer storage type.
 * @return Scalar-domain descriptor, or unknown for unsupported types.
 */
SimScalarDomain sim_buffer_data_type_to_scalar_domain(SimBufferDataType type);

/**
 * @brief Return a stable lowercase name for a buffer storage type.
 *
 * @param type Buffer storage type.
 * @return Static string such as "double", "complex_double", or "unknown".
 */
const char *sim_buffer_data_type_name(SimBufferDataType type);

/**
 * @brief Test whether a buffer view has data, count, and a supported type.
 *
 * Layout metadata is optional; this predicate validates the minimal readable
 * view contract.
 *
 * @param view View to inspect.
 * @return true when the view can be indexed.
 */
bool sim_buffer_view_is_valid(const SimBufferView *view);

/**
 * @brief Resolve multidimensional logical indices to a physical element offset.
 *
 * When the view has no layout, a single index is interpreted as the linear
 * element offset. When logical_to_physical is present, multidimensional indices
 * are first converted to row-major logical order before remapping.
 *
 * @param view View to index.
 * @param indices Array of logical indices.
 * @param index_count Number of entries in @p indices.
 * @param[out] out_offset Receives the physical element offset when non-NULL.
 * @return true when the indices are in range and the offset is representable.
 */
bool sim_buffer_view_offset_for_indices(const SimBufferView *view, const size_t *indices,
                                        size_t index_count, size_t *out_offset);

/**
 * @brief Resolve a linear logical index to a physical element offset.
 *
 * @param view View to index.
 * @param logical_index Linear logical index in [0, view->count).
 * @param[out] out_offset Receives the physical element offset when non-NULL.
 * @return true when the index is valid for the view.
 */
bool sim_buffer_view_offset_for_linear_index(const SimBufferView *view, size_t logical_index,
                                             size_t *out_offset);

/**
 * @brief Read a logical value as a SimComplexDouble.
 *
 * Real and integer storage types are promoted to a complex value with zero
 * imaginary part. Unsupported types or invalid indices write zero and return
 * false.
 *
 * @param view View to read.
 * @param logical_index Linear logical index.
 * @param[out] out_value Receives the promoted complex value.
 * @return true when a value was read.
 */
bool sim_buffer_view_get_complex(const SimBufferView *view, size_t logical_index,
                                 SimComplexDouble *out_value);

/**
 * @brief Write a SimComplexDouble value into a logical view element.
 *
 * Complex storage receives both channels. Real double storage accepts only
 * values with an exact zero imaginary component. Integer view writes are not
 * currently supported.
 *
 * @param view Mutable view to write.
 * @param logical_index Linear logical index.
 * @param value Value to store.
 * @return true when the value was written.
 */
bool sim_buffer_view_set_complex(SimBufferView *view, size_t logical_index, SimComplexDouble value);

/**
 * @brief Allocate a zero-initialized owning buffer with row-major layout.
 *
 * @param rank Number of dimensions; must be nonzero.
 * @param shape Array of @p rank extents.
 * @param type Scalar storage type.
 * @return Newly allocated buffer, or NULL on invalid input or allocation failure.
 */
SimBuffer *sim_buffer_create(size_t rank, const size_t *shape, SimBufferDataType type);

/**
 * @brief Allocate a one-dimensional zero-initialized owning buffer.
 *
 * @param count Number of elements.
 * @param type Scalar storage type.
 * @return Newly allocated buffer, or NULL on invalid input or allocation failure.
 */
SimBuffer *sim_buffer_create_1d(size_t count, SimBufferDataType type);

/**
 * @brief Copy a buffer view into a compact owning row-major buffer.
 *
 * The clone preserves the view's shape when valid layout metadata is present;
 * otherwise it creates a one-dimensional buffer of view->count elements.
 *
 * @param view Source view to copy.
 * @return New owning buffer, or NULL if the view cannot be indexed or allocated.
 */
SimBuffer *sim_buffer_clone_view(const SimBufferView *view);

/**
 * @brief Replace a buffer's layout without changing its element count.
 *
 * @param buffer Owning buffer to reshape.
 * @param rank New rank; must be nonzero.
 * @param shape New shape with product equal to sim_buffer_count(buffer).
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_buffer_reshape(SimBuffer *buffer, size_t rank, const size_t *shape);

/**
 * @brief Release an owning buffer and its storage.
 *
 * @param buffer Buffer returned by sim_buffer_create(); NULL is ignored.
 */
void sim_buffer_destroy(SimBuffer *buffer);

/**
 * @brief Return a mutable pointer to owned element storage.
 *
 * @param buffer Buffer to inspect.
 * @return Mutable data pointer, or NULL when @p buffer is NULL.
 */
void *sim_buffer_data(SimBuffer *buffer);

/**
 * @brief Return a read-only pointer to owned element storage.
 *
 * @param buffer Buffer to inspect.
 * @return Read-only data pointer, or NULL when @p buffer is NULL.
 */
const void *sim_buffer_const_data(const SimBuffer *buffer);

/**
 * @brief Return the logical element count.
 *
 * @param buffer Buffer to inspect.
 * @return Element count, or 0 when @p buffer is NULL.
 */
size_t sim_buffer_count(const SimBuffer *buffer);

/**
 * @brief Return the storage size in bytes.
 *
 * @param buffer Buffer to inspect.
 * @return Byte count, or 0 when @p buffer is NULL.
 */
size_t sim_buffer_bytes(const SimBuffer *buffer);

/**
 * @brief Return the buffer storage type.
 *
 * @param buffer Buffer to inspect.
 * @return Storage type, or `SIM_BUFFER_UNKNOWN` when @p buffer is NULL.
 */
SimBufferDataType sim_buffer_type(const SimBuffer *buffer);

/**
 * @brief Return the immutable layout descriptor for an owning buffer.
 *
 * @param buffer Buffer to inspect.
 * @return Pointer to internal layout metadata, or NULL when @p buffer is NULL.
 */
const SimFieldLayout *sim_buffer_layout(const SimBuffer *buffer);

/**
 * @brief Return a non-owning view over an owning buffer.
 *
 * The view points at the buffer's internal storage and layout arrays, so it is
 * invalidated when the buffer is reshaped or destroyed.
 *
 * @param buffer Buffer to view.
 * @return View descriptor, or a zeroed invalid view when @p buffer is NULL.
 */
SimBufferView sim_buffer_view(const SimBuffer *buffer);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_BUFFER_H */
