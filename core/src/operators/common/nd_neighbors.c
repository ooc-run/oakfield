#include "operators/common/nd_neighbors.h"

#include <limits.h>
#include <stdint.h>

static bool sim_nd_map_coord(ptrdiff_t          coord,
                             size_t             extent,
                             SimIRBoundaryPolicy boundary,
                             ptrdiff_t*         out_coord,
                             bool*              out_wrapped,
                             bool*              out_valid) {
    if (out_coord == NULL || out_wrapped == NULL || out_valid == NULL) {
        return false;
    }

    *out_coord   = 0;
    *out_wrapped = false;
    *out_valid   = false;

    if (extent == 0U) {
        return false;
    }

    if (coord >= 0 && coord < (ptrdiff_t) extent) {
        *out_coord = coord;
        *out_valid = true;
        return true;
    }

    switch (boundary) {
        case SIM_IR_BOUNDARY_PERIODIC: {
            ptrdiff_t mod = coord % (ptrdiff_t) extent;
            if (mod < 0) {
                mod += (ptrdiff_t) extent;
            }
            *out_coord   = mod;
            *out_wrapped = true;
            *out_valid   = true;
            return true;
        }
        case SIM_IR_BOUNDARY_REFLECTIVE: {
            if (extent == 1U) {
                *out_coord   = 0;
                *out_wrapped = true;
                *out_valid   = true;
                return true;
            }
            ptrdiff_t period = 2 * (ptrdiff_t) (extent - 1U);
            ptrdiff_t mod    = coord % period;
            if (mod < 0) {
                mod += period;
            }
            if (mod >= (ptrdiff_t) extent) {
                mod = period - mod;
            }
            *out_coord   = mod;
            *out_wrapped = true;
            *out_valid   = true;
            return true;
        }
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_NEUMANN:
        default:
            *out_valid = false;
            return true;
    }
}

SimResult sim_nd_offset_neighbor(const SimField*       field,
                                 size_t                element_index,
                                 const ptrdiff_t*      offsets,
                                 size_t                offset_count,
                                 SimIRBoundaryPolicy   boundary,
                                 SimNdNeighbor*        out_neighbor) {
    if (field == NULL || offsets == NULL || out_neighbor == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const size_t* shape   = sim_field_shape(field);
    const size_t* strides = sim_field_strides(field);
    size_t        rank    = sim_field_rank(field);
    if (shape == NULL || strides == NULL || rank == 0U || offset_count != rank) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (element_index > (size_t) PTRDIFF_MAX) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ptrdiff_t index_delta = 0;
    bool      wrapped     = false;

    for (size_t axis = 0U; axis < rank; ++axis) {
        size_t extent = shape[axis];
        size_t stride = strides[axis];
        if (extent == 0U || stride == 0U) {
            out_neighbor->valid   = false;
            out_neighbor->wrapped = false;
            out_neighbor->index   = element_index;
            return SIM_RESULT_OK;
        }

        ptrdiff_t coord = (ptrdiff_t) ((element_index / stride) % extent);
        ptrdiff_t target;
        bool      axis_wrapped = false;
        bool      axis_valid   = false;

        if ((offsets[axis] > 0 && coord > PTRDIFF_MAX - offsets[axis]) ||
            (offsets[axis] < 0 && coord < PTRDIFF_MIN - offsets[axis])) {
            out_neighbor->valid   = false;
            out_neighbor->wrapped = false;
            out_neighbor->index   = element_index;
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        target = coord + offsets[axis];

        if (!sim_nd_map_coord(target, extent, boundary, &target, &axis_wrapped, &axis_valid)) {
            out_neighbor->valid   = false;
            out_neighbor->wrapped = false;
            out_neighbor->index   = element_index;
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        if (!axis_valid) {
            out_neighbor->valid   = false;
            out_neighbor->wrapped = false;
            out_neighbor->index   = element_index;
            return SIM_RESULT_OK;
        }

        if (target != coord) {
            ptrdiff_t delta = (target - coord) * (ptrdiff_t) stride;
            if ((delta > 0 && index_delta > PTRDIFF_MAX - delta) ||
                (delta < 0 && index_delta < PTRDIFF_MIN - delta)) {
                out_neighbor->valid   = false;
                out_neighbor->wrapped = false;
                out_neighbor->index   = element_index;
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            index_delta += delta;
        }
        wrapped = wrapped || axis_wrapped;
    }

    ptrdiff_t base = (ptrdiff_t) element_index;
    if ((index_delta > 0 && base > PTRDIFF_MAX - index_delta) ||
        (index_delta < 0 && base < PTRDIFF_MIN - index_delta)) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ptrdiff_t target_index = base + index_delta;
    if (target_index < 0 || (uintmax_t) target_index > SIZE_MAX) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    out_neighbor->valid   = true;
    out_neighbor->wrapped = wrapped;
    out_neighbor->index   = (size_t) target_index;
    return SIM_RESULT_OK;
}

SimResult sim_nd_axis_offset_neighbor(const SimField*     field,
                                      size_t              element_index,
                                      size_t              axis,
                                      ptrdiff_t           offset,
                                      SimIRBoundaryPolicy boundary,
                                      SimNdNeighbor*      out_neighbor) {
    if (field == NULL || out_neighbor == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const size_t* shape   = sim_field_shape(field);
    const size_t* strides = sim_field_strides(field);
    size_t        rank    = sim_field_rank(field);
    if (shape == NULL || strides == NULL || rank == 0U || axis >= rank) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (element_index > (size_t) PTRDIFF_MAX) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t extent = shape[axis];
    size_t stride = strides[axis];
    if (extent == 0U || stride == 0U) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_OK;
    }

    ptrdiff_t coord = (ptrdiff_t) ((element_index / stride) % extent);
    ptrdiff_t target;
    bool      axis_wrapped = false;
    bool      axis_valid   = false;

    if ((offset > 0 && coord > PTRDIFF_MAX - offset) ||
        (offset < 0 && coord < PTRDIFF_MIN - offset)) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    target = coord + offset;

    if (!sim_nd_map_coord(target, extent, boundary, &target, &axis_wrapped, &axis_valid)) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!axis_valid) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_OK;
    }

    ptrdiff_t delta = (target - coord) * (ptrdiff_t) stride;
    ptrdiff_t base  = (ptrdiff_t) element_index;
    if ((delta > 0 && base > PTRDIFF_MAX - delta) ||
        (delta < 0 && base < PTRDIFF_MIN - delta)) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ptrdiff_t target_index = base + delta;
    if (target_index < 0 || (uintmax_t) target_index > SIZE_MAX) {
        out_neighbor->valid   = false;
        out_neighbor->wrapped = false;
        out_neighbor->index   = element_index;
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    out_neighbor->valid   = true;
    out_neighbor->wrapped = axis_wrapped;
    out_neighbor->index   = (size_t) target_index;
    return SIM_RESULT_OK;
}

SimResult sim_nd_axis_neighbors(const SimField*       field,
                                size_t                element_index,
                                size_t                axis,
                                SimIRBoundaryPolicy   boundary,
                                SimNdAxisNeighbors*   out_neighbors) {
    if (field == NULL || out_neighbors == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimResult result =
        sim_nd_axis_offset_neighbor(field, element_index, axis, 1, boundary, &out_neighbors->forward);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_nd_axis_offset_neighbor(
        field, element_index, axis, -1, boundary, &out_neighbors->backward);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    return SIM_RESULT_OK;
}
