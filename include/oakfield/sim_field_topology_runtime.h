/**
 * @file sim_field_topology_runtime.h
 * @brief Runtime cache and texture-packing state for field-topology extraction.
 *
 * @details SimFieldTopologyRuntimeState keeps reusable cell storage, scratch
 * arrays, cadence counters, and a packed-summary cache for context-owned fields.
 * Callers mark the state dirty when the source field changes and request
 * recomputation on demand.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "field.h"
#include "sim_field_topology.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scratch arrays reserved for topology cache and future graph-style passes.
 */
typedef struct SimFieldTopologyWorkspace {
    uint32_t *labels;    /**< Per-cell labels for connected-component style analysis. */
    uint8_t *visited;    /**< Per-cell visitation markers. */
    uint8_t *seam_chain; /**< Per-cell seam-chain markers. */
    uint8_t *ambiguity;  /**< Per-cell ambiguity map mirrored from extracted cells. */
    size_t capacity;     /**< Number of cell entries allocated in each scratch array. */
} SimFieldTopologyWorkspace;

/**
 * @brief Reusable runtime state for one field's topology cache.
 */
typedef struct SimFieldTopologyRuntimeState {
    bool enabled;         /**< Enables recomputation; disabled states report invalid topology. */
    bool valid;           /**< True when cached cells and summary describe current topology. */
    bool dirty;           /**< Forces recomputation on the next request. */
    size_t width;         /**< Cached topology width. */
    size_t height;        /**< Cached topology height. */
    size_t cell_capacity; /**< Number of SimFieldTopologyCell entries allocated. */
    size_t cadence_steps; /**< Minimum step interval between clean recomputations. */
    size_t last_computed_step;       /**< Step index used for the most recent recompute. */
    uint64_t generation;             /**< Monotonic counter bumped on resize and recompute. */
    uint64_t request_count;          /**< Total recompute requests observed while enabled. */
    uint64_t recompute_count;        /**< Requests that performed extraction work. */
    uint64_t skip_count;             /**< Requests skipped because cadence allowed cache reuse. */
    uint64_t pack_count;             /**< Successful RGBA8 pack operations. */
    SimFieldTopologySummary summary; /**< Cached aggregate topology summary. */
    SimFieldTopologyConfig config;   /**< Extraction thresholds used for recompute. */
    SimFieldTopologyCell *cells;     /**< Cached topology cells, width * height entries. */
    SimFieldTopologyWorkspace workspace; /**< Reusable auxiliary storage. */
} SimFieldTopologyRuntimeState;

/**
 * @brief Initialize runtime topology state with default configuration.
 *
 * @param[out] state State object to initialize; ignored when NULL.
 */
void sim_field_topology_runtime_init(SimFieldTopologyRuntimeState *state);

/**
 * @brief Invalidate cached topology while retaining allocated buffers.
 *
 * @param state State object to reset; ignored when NULL.
 */
void sim_field_topology_runtime_reset(SimFieldTopologyRuntimeState *state);

/**
 * @brief Mark cached topology stale.
 *
 * @param state State object to mark; ignored when NULL.
 */
void sim_field_topology_runtime_mark_dirty(SimFieldTopologyRuntimeState *state);

/**
 * @brief Ensure cached cell and workspace storage can hold a topology grid.
 *
 * @param state State object to resize.
 * @param width Required topology width.
 * @param height Required topology height.
 * @return SIM_RESULT_OK on success, SIM_RESULT_INVALID_ARGUMENT for NULL state, or
 * SIM_RESULT_OUT_OF_MEMORY when allocation fails.
 */
SimResult sim_field_topology_runtime_resize(SimFieldTopologyRuntimeState *state, size_t width,
                                            size_t height);

/**
 * @brief Free cached topology buffers and reinitialize the state.
 *
 * @param state State object to free; ignored when NULL.
 */
void sim_field_topology_runtime_free(SimFieldTopologyRuntimeState *state);

/**
 * @brief Recompute cached topology when enabled, dirty, or past the cadence interval.
 *
 * @param state Runtime cache state.
 * @param field Source field to inspect.
 * @param step_index Current simulation step index.
 * @return true when the cache is valid after the request.
 */
bool sim_field_topology_runtime_recompute(SimFieldTopologyRuntimeState *state,
                                          const struct SimField *field, size_t step_index);

/**
 * @brief Pack cached topology cells into an RGBA8 image.
 *
 * @param state Runtime cache state with valid topology.
 * @param[out] dest Destination byte buffer.
 * @param capacity Destination capacity in bytes.
 * @param[out] out_width Optional packed image width.
 * @param[out] out_height Optional packed image height.
 * @param[out] out_bytes Optional number of bytes written.
 * @return true when the cache was valid and @p dest had enough capacity.
 */
bool sim_field_topology_runtime_pack_rgba8(SimFieldTopologyRuntimeState *state, uint8_t *dest,
                                           size_t capacity, size_t *out_width, size_t *out_height,
                                           size_t *out_bytes);

#ifdef __cplusplus
}
#endif
