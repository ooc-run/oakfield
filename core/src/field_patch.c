/**
 * @file field_patch.c
 * @brief Index-space patch construction and zero-copy row iteration for fields.
 *
 * Patches describe rectangular regions in a field's fastest two axes. The
 * implementation validates inclusive/exclusive bounds, derives patch flags for
 * full-row/full-field/boundary cases, and exposes row-major SimBufferView
 * windows without copying when the source field layout supports it.
 */
#include "oakfield/field_patch.h"

#include <math.h>
#include <string.h>

static double sim_field_patch_clampd(double value, double min_value, double max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static unsigned int sim_field_patch_compute_flags(size_t field_width,
                                                  size_t field_height,
                                                  size_t x0,
                                                  size_t y0,
                                                  size_t width,
                                                  size_t height) {
    unsigned int flags = SIM_FIELD_PATCH_FLAG_NONE;

    if (x0 == 0U && width == field_width) {
        flags |= SIM_FIELD_PATCH_FLAG_COVERS_FULL_ROWS;
    }
    if ((flags & SIM_FIELD_PATCH_FLAG_COVERS_FULL_ROWS) != 0U && y0 == 0U &&
        height == field_height) {
        flags |= SIM_FIELD_PATCH_FLAG_IS_FULL_FIELD;
    }
    if (x0 == 0U || y0 == 0U || x0 + width == field_width || y0 + height == field_height) {
        flags |= SIM_FIELD_PATCH_FLAG_TOUCHES_BOUNDARY;
    }
    return flags;
}

bool sim_field_patch_is_valid(const SimFieldPatch* patch) {
    if (patch == NULL || patch->field_width == 0U || patch->field_height == 0U ||
        patch->width == 0U || patch->height == 0U) {
        return false;
    }

    if (patch->x0 >= patch->field_width || patch->y0 >= patch->field_height) {
        return false;
    }
    if (patch->width > patch->field_width - patch->x0 ||
        patch->height > patch->field_height - patch->y0) {
        return false;
    }

    return true;
}

SimResult sim_field_patch_full(size_t field_width, size_t field_height, SimFieldPatch* out_patch) {
    return sim_field_patch_from_xywh(
        field_width, field_height, 0U, 0U, field_width, field_height, out_patch);
}

SimResult sim_field_patch_from_xywh(size_t         field_width,
                                    size_t         field_height,
                                    size_t         x0,
                                    size_t         y0,
                                    size_t         width,
                                    size_t         height,
                                    SimFieldPatch* out_patch) {
    SimFieldPatch patch = { 0 };

    if (out_patch == NULL || field_width == 0U || field_height == 0U || width == 0U ||
        height == 0U || x0 >= field_width || y0 >= field_height ||
        width > field_width - x0 || height > field_height - y0) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    patch.x0           = x0;
    patch.y0           = y0;
    patch.width        = width;
    patch.height       = height;
    patch.field_width  = field_width;
    patch.field_height = field_height;
    patch.flags =
        sim_field_patch_compute_flags(field_width, field_height, x0, y0, width, height);
    *out_patch = patch;
    return SIM_RESULT_OK;
}

bool sim_field_patch_intersect(const SimFieldPatch* lhs,
                               const SimFieldPatch* rhs,
                               SimFieldPatch*       out_patch) {
    size_t x0 = 0U;
    size_t y0 = 0U;
    size_t x1 = 0U;
    size_t y1 = 0U;

    if (out_patch != NULL) {
        (void) memset(out_patch, 0, sizeof(*out_patch));
    }

    if (!sim_field_patch_is_valid(lhs) || !sim_field_patch_is_valid(rhs) ||
        lhs->field_width != rhs->field_width || lhs->field_height != rhs->field_height) {
        return false;
    }

    x0 = (lhs->x0 > rhs->x0) ? lhs->x0 : rhs->x0;
    y0 = (lhs->y0 > rhs->y0) ? lhs->y0 : rhs->y0;
    x1 = (lhs->x0 + lhs->width < rhs->x0 + rhs->width) ? (lhs->x0 + lhs->width)
                                                       : (rhs->x0 + rhs->width);
    y1 = (lhs->y0 + lhs->height < rhs->y0 + rhs->height) ? (lhs->y0 + lhs->height)
                                                          : (rhs->y0 + rhs->height);
    if (x1 <= x0 || y1 <= y0) {
        return false;
    }

    if (out_patch != NULL) {
        return sim_field_patch_from_xywh(
                   lhs->field_width, lhs->field_height, x0, y0, x1 - x0, y1 - y0, out_patch) ==
               SIM_RESULT_OK;
    }
    return true;
}

SimResult sim_field_patch_from_normalized_region(size_t         field_width,
                                                 size_t         field_height,
                                                 double         min_x,
                                                 double         min_y,
                                                 double         max_x,
                                                 double         max_y,
                                                 SimFieldPatch* out_patch) {
    size_t x0 = 0U;
    size_t y0 = 0U;
    size_t x1 = 0U;
    size_t y1 = 0U;

    if (out_patch == NULL || field_width == 0U || field_height == 0U || !isfinite(min_x) ||
        !isfinite(min_y) || !isfinite(max_x) || !isfinite(max_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    min_x = sim_field_patch_clampd(min_x, 0.0, 1.0);
    min_y = sim_field_patch_clampd(min_y, 0.0, 1.0);
    max_x = sim_field_patch_clampd(max_x, 0.0, 1.0);
    max_y = sim_field_patch_clampd(max_y, 0.0, 1.0);
    if (!(max_x > min_x) || !(max_y > min_y)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    x0 = (size_t) floor(min_x * (double) field_width);
    y0 = (size_t) floor(min_y * (double) field_height);
    x1 = (size_t) ceil(max_x * (double) field_width);
    y1 = (size_t) ceil(max_y * (double) field_height);

    if (x0 >= field_width) {
        x0 = field_width - 1U;
    }
    if (y0 >= field_height) {
        y0 = field_height - 1U;
    }
    if (x1 == 0U) {
        x1 = 1U;
    }
    if (y1 == 0U) {
        y1 = 1U;
    }
    if (x1 > field_width) {
        x1 = field_width;
    }
    if (y1 > field_height) {
        y1 = field_height;
    }
    if (x1 <= x0) {
        x1 = x0 + 1U;
    }
    if (y1 <= y0) {
        y1 = y0 + 1U;
    }
    if (x1 > field_width || y1 > field_height) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    return sim_field_patch_from_xywh(
        field_width, field_height, x0, y0, x1 - x0, y1 - y0, out_patch);
}

bool sim_field_patch_view_is_valid(const SimFieldPatchView* view) {
    size_t expected_count = 0U;

    if (view == NULL || !sim_field_patch_is_valid(&view->patch) ||
        !sim_buffer_view_is_valid(&view->buffer_view)) {
        return false;
    }

    expected_count = view->patch.width * view->patch.height;
    if (view->buffer_view.count != expected_count || view->row_stride < view->patch.width) {
        return false;
    }

    if (view->buffer_view.layout.rank == 1U) {
        return view->buffer_view.layout.shape == view->shape_storage &&
               view->buffer_view.layout.strides == view->stride_storage &&
               view->shape_storage[0] == view->patch.width &&
               view->stride_storage[0] == 1U &&
               view->buffer_view.layout.contiguous;
    }

    if (view->buffer_view.layout.rank != 2U || view->buffer_view.layout.shape != view->shape_storage ||
        view->buffer_view.layout.strides != view->stride_storage) {
        return false;
    }

    return view->shape_storage[0] == view->patch.height &&
           view->shape_storage[1] == view->patch.width && view->stride_storage[1] == 1U &&
           view->stride_storage[0] == view->row_stride;
}

SimResult sim_field_patch_view_from_field(SimField*            field,
                                          const SimFieldPatch* patch,
                                          bool                 readonly,
                                          SimFieldPatchView*   out_view) {
    size_t          rank = 0U;
    size_t          axis_x = 0U;
    size_t          axis_y = 0U;
    size_t          base_index = 0U;
    SimBufferDataType type = SIM_BUFFER_UNKNOWN;
    SimFieldPatchView view = { 0 };

    if (out_view == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    (void) memset(out_view, 0, sizeof(*out_view));

    if (field == NULL || !sim_field_patch_is_valid(patch)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    rank = sim_field_rank(field);
    if (rank == 0U || sim_field_data(field) == NULL || sim_field_width(field) != patch->field_width ||
        sim_field_height(field) != patch->field_height) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (rank > 2U || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR || field->layout.shape == NULL ||
        field->layout.strides == NULL) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    axis_x = rank - 1U;
    if (field->layout.strides[axis_x] != 1U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    if (sim_field_xy_to_index(field, patch->x0, patch->y0, &base_index) != SIM_RESULT_OK) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    type = sim_buffer_data_type_from_scalar_domain(sim_field_scalar_domain(field));
    if (sim_buffer_element_size(type) == 0U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    view.patch                = *patch;
    view.row_stride           = patch->width;
    view.readonly             = readonly;
    view.zero_copy            = true;
    view.buffer_view.data =
        (unsigned char*) sim_field_data(field) + base_index * sim_buffer_element_size(type);
    view.buffer_view.count               = patch->width * patch->height;
    view.buffer_view.type                = type;
    view.buffer_view.logical_to_physical = NULL;

    if (rank == 1U) {
        view.shape_storage[0]             = patch->width;
        view.stride_storage[0]            = 1U;
        view.row_stride                   = patch->width;
        view.buffer_view.layout.rank      = 1U;
        view.buffer_view.layout.shape     = view.shape_storage;
        view.buffer_view.layout.strides   = view.stride_storage;
        view.buffer_view.layout.contiguous = true;
    } else {
        axis_y = rank - 2U;
        view.shape_storage[0]             = patch->height;
        view.shape_storage[1]             = patch->width;
        view.stride_storage[0]            = field->layout.strides[axis_y];
        view.stride_storage[1]            = field->layout.strides[axis_x];
        view.row_stride                   = view.stride_storage[0];
        view.buffer_view.layout.rank      = 2U;
        view.buffer_view.layout.shape     = view.shape_storage;
        view.buffer_view.layout.strides   = view.stride_storage;
        view.buffer_view.layout.contiguous =
            (patch->flags & SIM_FIELD_PATCH_FLAG_COVERS_FULL_ROWS) != 0U;
    }

    if (!sim_field_patch_view_is_valid(&view)) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_view = view;
    out_view->buffer_view.layout.shape   = out_view->shape_storage;
    out_view->buffer_view.layout.strides = out_view->stride_storage;
    return SIM_RESULT_OK;
}

SimResult sim_field_patch_iter_rows(const SimFieldPatchView* view,
                                    SimFieldPatchRowIterFn   row_fn,
                                    void*                    userdata) {
    size_t row_count = 0U;
    size_t elem_size = 0U;

    if (!sim_field_patch_view_is_valid(view) || row_fn == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    row_count = (view->buffer_view.layout.rank == 1U) ? 1U : view->patch.height;
    elem_size = sim_buffer_element_size(view->buffer_view.type);

    for (size_t row = 0U; row < row_count; ++row) {
        void* row_data =
            (unsigned char*) view->buffer_view.data + row * view->row_stride * elem_size;
        SimResult rc = row_fn(row, row_data, view->patch.width, userdata);
        if (rc != SIM_RESULT_OK) {
            return rc;
        }
    }

    return SIM_RESULT_OK;
}

size_t sim_field_patch_split_tiles(const SimFieldPatch* patch,
                                   size_t               tile_width,
                                   size_t               tile_height,
                                   SimFieldPatch*       out_tiles,
                                   size_t               out_capacity) {
    size_t count = 0U;

    if (!sim_field_patch_is_valid(patch)) {
        return 0U;
    }

    if (tile_width == 0U || tile_width > patch->width) {
        tile_width = patch->width;
    }
    if (tile_height == 0U || tile_height > patch->height) {
        tile_height = patch->height;
    }

    for (size_t y = 0U; y < patch->height; y += tile_height) {
        size_t local_height = patch->height - y;
        if (local_height > tile_height) {
            local_height = tile_height;
        }

        for (size_t x = 0U; x < patch->width; x += tile_width) {
            size_t local_width = patch->width - x;
            if (local_width > tile_width) {
                local_width = tile_width;
            }

            if (out_tiles != NULL && count < out_capacity) {
                (void) sim_field_patch_from_xywh(patch->field_width,
                                                 patch->field_height,
                                                 patch->x0 + x,
                                                 patch->y0 + y,
                                                 local_width,
                                                 local_height,
                                                 &out_tiles[count]);
            }
            count += 1U;
        }
    }

    return count;
}
