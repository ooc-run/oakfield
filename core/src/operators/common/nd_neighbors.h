/**
 * @file nd_neighbors.h
 * @brief N-dimensional neighbor lookup helpers with shared boundary policies.
 */
#ifndef OAKFIELD_ND_NEIGHBORS_H
#define OAKFIELD_ND_NEIGHBORS_H

#include "oakfield/field.h"
#include "oakfield/kernel_ir.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SimNdNeighbor {
    bool   valid;   /**< False when the boundary policy drops the neighbor. */
    bool   wrapped; /**< True when boundary handling changed an axis coordinate. */
    size_t index;   /**< Linear field index for the mapped neighbor. */
} SimNdNeighbor;

typedef struct SimNdAxisNeighbors {
    SimNdNeighbor forward;  /**< Neighbor at +1 along the requested axis. */
    SimNdNeighbor backward; /**< Neighbor at -1 along the requested axis. */
} SimNdAxisNeighbors;

/**
 * @brief Resolve an arbitrary N-dimensional offset from a linear field index.
 *
 * @param field Field whose shape and strides define the coordinate system.
 * @param element_index Linear index of the source element.
 * @param offsets Per-axis signed offsets; must contain @p offset_count entries.
 * @param offset_count Number of offsets; must equal field rank.
 * @param boundary Boundary policy used when offsets leave the field extent.
 * @param[out] out_neighbor Receives validity, wrap flag, and resulting index.
 * @return #SIM_RESULT_OK on success, including invalid dropped neighbors, or
 *         #SIM_RESULT_INVALID_ARGUMENT for malformed inputs or index overflow.
 */
SimResult sim_nd_offset_neighbor(const SimField*     field,
                                 size_t              element_index,
                                 const ptrdiff_t*    offsets,
                                 size_t              offset_count,
                                 SimIRBoundaryPolicy boundary,
                                 SimNdNeighbor*      out_neighbor);

/**
 * @brief Resolve a signed offset along one axis from a linear field index.
 *
 * @param field Field whose shape and strides define the coordinate system.
 * @param element_index Linear index of the source element.
 * @param axis Axis to offset.
 * @param offset Signed offset along @p axis.
 * @param boundary Boundary policy used when the offset leaves the field extent.
 * @param[out] out_neighbor Receives validity, wrap flag, and resulting index.
 * @return #SIM_RESULT_OK on success, including invalid dropped neighbors, or
 *         #SIM_RESULT_INVALID_ARGUMENT for malformed inputs or index overflow.
 */
SimResult sim_nd_axis_offset_neighbor(const SimField*     field,
                                      size_t              element_index,
                                      size_t              axis,
                                      ptrdiff_t           offset,
                                      SimIRBoundaryPolicy boundary,
                                      SimNdNeighbor*      out_neighbor);

/**
 * @brief Resolve immediate forward and backward neighbors on one axis.
 *
 * @param field Field whose shape and strides define the coordinate system.
 * @param element_index Linear index of the source element.
 * @param axis Axis to inspect.
 * @param boundary Boundary policy used at the axis ends.
 * @param[out] out_neighbors Receives +1 and -1 neighbors.
 * @return #SIM_RESULT_OK on success, or an error code from either axis lookup.
 */
SimResult sim_nd_axis_neighbors(const SimField*     field,
                                size_t              element_index,
                                size_t              axis,
                                SimIRBoundaryPolicy boundary,
                                SimNdAxisNeighbors* out_neighbors);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_ND_NEIGHBORS_H */
