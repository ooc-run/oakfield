/**
 * @file static_cache.h
 * @brief Shared cache helpers for static stimulus fields.
 *
 * The cache stores reusable real and optional imaginary stimulus samples for
 * layouts whose coordinates do not change between steps. Callers provide a fill
 * callback that rebuilds the cache whenever layout, rank, or imaginary-output
 * requirements change.
 */
#ifndef OAKFIELD_STIMULUS_STATIC_CACHE_H
#define OAKFIELD_STIMULUS_STATIC_CACHE_H

#include "oakfield/field.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SimStimulusStaticCacheLayout {
    size_t rank;
    size_t count;
    size_t extent_x;
    size_t extent_y;
    size_t stride_x;
    size_t stride_y;
} SimStimulusStaticCacheLayout;

typedef struct SimStimulusStaticCache {
    double* real;
    double* imag;
    size_t  capacity;
    size_t  count;
    size_t  rank;
    size_t  extent_x;
    size_t  extent_y;
    size_t  stride_x;
    size_t  stride_y;
    bool    valid;
} SimStimulusStaticCache;

/**
 * @brief Callback used to populate static stimulus cache storage.
 *
 * The callback receives already allocated output buffers with layout->count
 * entries. @p out_imag may be NULL when @p need_imag is false.
 *
 * @param userdata Caller-owned state passed through from sim_stimulus_static_cache_ensure().
 * @param layout Layout being cached.
 * @param need_imag Whether the caller needs imaginary samples.
 * @param[out] out_real Real output buffer with layout->count entries.
 * @param[out] out_imag Optional imaginary output buffer with layout->count entries.
 * @return #SIM_RESULT_OK on success, or an error code to leave the cache invalid.
 */
typedef SimResult (*SimStimulusStaticCacheFillFn)(void*                               userdata,
                                                  const SimStimulusStaticCacheLayout* layout,
                                                  bool                                need_imag,
                                                  double*                             out_real,
                                                  double*                             out_imag);

/**
 * @brief Build a 1D or 2D cache layout descriptor from field shape and strides.
 *
 * Rank 1 maps the sole axis to X with extent_y = 1. Rank 2 uses the last axis as
 * X and the preceding axis as Y. Zero extents or strides are invalid.
 *
 * @param shape Field shape array containing at least @p rank entries.
 * @param strides Field stride array containing at least @p rank entries.
 * @param rank Number of field axes; only 1 and 2 are supported.
 * @param[out] out_layout Receives the normalized cache layout.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for invalid
 *         inputs, or #SIM_RESULT_OUT_OF_MEMORY on extent multiplication overflow.
 */
SimResult
sim_stimulus_static_cache_layout_from_arrays(const size_t*                   shape,
                                             const size_t*                   strides,
                                             size_t                          rank,
                                             SimStimulusStaticCacheLayout*   out_layout);

/**
 * @brief Convert a cached linear element index to X/Y lattice indices.
 *
 * NULL output pointers are ignored. Outputs are initialized to zero before
 * layout-dependent conversion, so a NULL layout leaves both coordinates at zero.
 *
 * @param layout Cache layout describing rank, extents, and strides.
 * @param index Linear element index in the cached field.
 * @param[out] out_x Optional destination for the X index.
 * @param[out] out_y Optional destination for the Y index.
 */
void sim_stimulus_static_cache_index_to_xy(const SimStimulusStaticCacheLayout* layout,
                                           size_t                              index,
                                           size_t*                             out_x,
                                           size_t*                             out_y);

/**
 * @brief Free all storage owned by a static stimulus cache and reset it to zero.
 *
 * @param cache Cache to destroy; NULL is ignored.
 */
void sim_stimulus_static_cache_destroy(SimStimulusStaticCache* cache);

/**
 * @brief Mark a static stimulus cache invalid while retaining allocated storage.
 *
 * @param cache Cache to invalidate; NULL is ignored.
 */
void sim_stimulus_static_cache_invalidate(SimStimulusStaticCache* cache);

/**
 * @brief Ensure a static stimulus cache is populated for a layout and value kind.
 *
 * Existing valid storage is reused when layout and imaginary-output requirements
 * match. Otherwise the cache grows buffers as needed, calls @p fill, and marks
 * the cache valid only after a successful fill.
 *
 * @param cache Cache object to reuse or populate.
 * @param layout Target layout and element count.
 * @param need_imag Whether an imaginary cache plane is required.
 * @param fill Callback that writes real and optional imaginary samples.
 * @param userdata Caller-owned pointer passed to @p fill.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         required inputs, #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or
 *         the callback's error code.
 */
SimResult sim_stimulus_static_cache_ensure(SimStimulusStaticCache*           cache,
                                           const SimStimulusStaticCacheLayout* layout,
                                           bool                               need_imag,
                                           SimStimulusStaticCacheFillFn       fill,
                                           void*                              userdata);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_STATIC_CACHE_H */
