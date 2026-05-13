/**
 * @file field_patch.h
 * @brief Shared index-space field patch descriptors and zero-copy patch views.
 *
 * A patch is a non-empty rectangle in the fastest two axes of a SimField, using
 * x/y coordinates and half-open extents. Views expose patch storage through a
 * SimBufferView and are zero-copy when the source field has a supported
 * row-major layout.
 */
#ifndef OAKFIELD_FIELD_PATCH_H
#define OAKFIELD_FIELD_PATCH_H

#include "sim_buffer.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    SIM_FIELD_PATCH_FLAG_NONE = 0u,                  /**< No derived patch property. */
    SIM_FIELD_PATCH_FLAG_COVERS_FULL_ROWS = 1u << 0, /**< Patch spans x across full rows. */
    SIM_FIELD_PATCH_FLAG_IS_FULL_FIELD = 1u << 1,    /**< Patch covers the complete field. */
    SIM_FIELD_PATCH_FLAG_TOUCHES_BOUNDARY = 1u << 2  /**< Patch touches at least one field edge. */
};

/**
 * @brief Rectangular non-empty region of a 2D field index space.
 */
typedef struct SimFieldPatch {
    size_t x0;           /**< Left coordinate in field index space. */
    size_t y0;           /**< Top coordinate in field index space. */
    size_t width;        /**< Patch width in elements. */
    size_t height;       /**< Patch height in elements. */
    size_t field_width;  /**< Width of the source field. */
    size_t field_height; /**< Height of the source field. */
    unsigned int flags;  /**< Derived SIM_FIELD_PATCH_FLAG_* values. */
} SimFieldPatch;

/**
 * @brief Zero-copy view over a field patch.
 *
 * The buffer view points into the source field's storage. The shape/stride
 * storage arrays are owned by the view struct and back buffer_view.layout.
 */
typedef struct SimFieldPatchView {
    SimBufferView buffer_view; /**< Typed view over patch elements. */
    SimFieldPatch patch;       /**< Patch coordinates represented by the view. */
    size_t row_stride;         /**< Physical element stride between rows. */
    bool readonly;             /**< Advisory flag for callers that must not mutate. */
    bool zero_copy;            /**< True when data points into source field storage. */
    size_t shape_storage[2];   /**< Local shape backing for buffer_view.layout. */
    size_t stride_storage[2];  /**< Local stride backing for buffer_view.layout. */
} SimFieldPatchView;

/**
 * @brief Row callback used by sim_field_patch_iter_rows().
 *
 * @param row_index Zero-based row number within the patch view.
 * @param row_data Pointer to the first element of the row.
 * @param row_length Number of elements in the row.
 * @param userdata Caller-provided context pointer.
 * @return #SIM_RESULT_OK to continue, or any other result to stop iteration.
 */
typedef SimResult (*SimFieldPatchRowIterFn)(size_t row_index, void *row_data, size_t row_length,
                                            void *userdata);

/**
 * @brief Validate patch bounds, dimensions, and source-field extents.
 *
 * @param patch Patch descriptor to inspect.
 * @return true when the patch is non-empty and lies within its field.
 */
bool sim_field_patch_is_valid(const SimFieldPatch *patch);

/**
 * @brief Construct a patch covering a complete field.
 *
 * @param field_width Source field width; must be nonzero.
 * @param field_height Source field height; must be nonzero.
 * @param[out] out_patch Receives the patch descriptor.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_field_patch_full(size_t field_width, size_t field_height, SimFieldPatch *out_patch);

/**
 * @brief Construct a patch from x/y origin and width/height extents.
 *
 * @param field_width Source field width; must be nonzero.
 * @param field_height Source field height; must be nonzero.
 * @param x0 Left coordinate, in [0, field_width).
 * @param y0 Top coordinate, in [0, field_height).
 * @param width Patch width; must be nonzero and fit inside the field.
 * @param height Patch height; must be nonzero and fit inside the field.
 * @param[out] out_patch Receives the normalized patch descriptor.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_field_patch_from_xywh(size_t field_width, size_t field_height, size_t x0, size_t y0,
                                    size_t width, size_t height, SimFieldPatch *out_patch);

/**
 * @brief Intersect two patches from the same field.
 *
 * @param lhs First patch.
 * @param rhs Second patch.
 * @param[out] out_patch Optional receiver for the intersection patch.
 * @return true when the patches are valid, share extents, and overlap.
 */
bool sim_field_patch_intersect(const SimFieldPatch *lhs, const SimFieldPatch *rhs,
                               SimFieldPatch *out_patch);

/**
 * @brief Construct a patch from normalized [0, 1] coordinate bounds.
 *
 * Bounds are clamped to [0, 1], min values use floor(), and max values use
 * ceil() so non-empty continuous regions cover all touched samples.
 *
 * @param field_width Source field width; must be nonzero.
 * @param field_height Source field height; must be nonzero.
 * @param min_x Normalized left bound.
 * @param min_y Normalized top bound.
 * @param max_x Normalized right bound; must be greater than @p min_x after clamping.
 * @param max_y Normalized bottom bound; must be greater than @p min_y after clamping.
 * @param[out] out_patch Receives the patch descriptor.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_field_patch_from_normalized_region(size_t field_width, size_t field_height,
                                                 double min_x, double min_y, double max_x,
                                                 double max_y, SimFieldPatch *out_patch);

/**
 * @brief Validate a patch view and its backing buffer-view layout.
 *
 * @param view View descriptor to inspect.
 * @return true when the view references a valid patch and matching buffer view.
 */
bool sim_field_patch_view_is_valid(const SimFieldPatchView *view);

/**
 * @brief Create a zero-copy view over a patch of a row-major field.
 *
 * Supports one- and two-dimensional row-major fields whose fastest axis has
 * unit stride. The resulting view points into @p field storage and is invalid
 * after the field is destroyed or reallocated.
 *
 * @param field Source field to view.
 * @param patch Valid patch matching the field width and height.
 * @param readonly Advisory flag stored in the view for callers.
 * @param[out] out_view Receives the view descriptor.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_SUPPORTED,
 *         or #SIM_RESULT_INVALID_STATE.
 */
SimResult sim_field_patch_view_from_field(SimField *field, const SimFieldPatch *patch,
                                          bool readonly, SimFieldPatchView *out_view);

/**
 * @brief Iterate physical rows in a patch view.
 *
 * @param view Valid patch view.
 * @param row_fn Callback invoked once per row.
 * @param userdata Caller data passed to @p row_fn.
 * @return #SIM_RESULT_OK when all rows complete, or the first callback error.
 */
SimResult sim_field_patch_iter_rows(const SimFieldPatchView *view, SimFieldPatchRowIterFn row_fn,
                                    void *userdata);

/**
 * @brief Split a patch into row-major tiles.
 *
 * Tile dimensions of zero, or larger than the patch, collapse to the full patch
 * dimension. Passing out_tiles NULL lets callers query the required tile count.
 *
 * @param patch Patch to split.
 * @param tile_width Requested tile width.
 * @param tile_height Requested tile height.
 * @param[out] out_tiles Optional array receiving up to @p out_capacity tiles.
 * @param out_capacity Number of entries available in @p out_tiles.
 * @return Total number of tiles required, or 0 when @p patch is invalid.
 */
size_t sim_field_patch_split_tiles(const SimFieldPatch *patch, size_t tile_width,
                                   size_t tile_height, SimFieldPatch *out_tiles,
                                   size_t out_capacity);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_FIELD_PATCH_H */
