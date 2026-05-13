#include "static_cache.h"

#include <stdlib.h>
#include <string.h>

SimResult
sim_stimulus_static_cache_layout_from_arrays(const size_t*                 shape,
                                             const size_t*                 strides,
                                             size_t                        rank,
                                             SimStimulusStaticCacheLayout* out_layout) {
    if (shape == NULL || strides == NULL || out_layout == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (rank == 0U || rank > 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusStaticCacheLayout layout = { 0 };
    layout.rank                         = rank;
    if (rank == 1U) {
        if (shape[0] == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        layout.count    = shape[0];
        layout.extent_x = shape[0];
        layout.extent_y = 1U;
        layout.stride_x = 1U;
        layout.stride_y = 1U;
    } else {
        size_t axis_x = rank - 1U;
        size_t axis_y = rank - 2U;
        if (shape[axis_x] == 0U || shape[axis_y] == 0U || strides[axis_x] == 0U ||
            strides[axis_y] == 0U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (shape[axis_x] > SIZE_MAX / shape[axis_y]) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        layout.count    = shape[axis_x] * shape[axis_y];
        layout.extent_x = shape[axis_x];
        layout.extent_y = shape[axis_y];
        layout.stride_x = strides[axis_x];
        layout.stride_y = strides[axis_y];
    }

    *out_layout = layout;
    return SIM_RESULT_OK;
}

void sim_stimulus_static_cache_index_to_xy(const SimStimulusStaticCacheLayout* layout,
                                           size_t                              index,
                                           size_t*                             out_x,
                                           size_t*                             out_y) {
    if (out_x != NULL) {
        *out_x = 0U;
    }
    if (out_y != NULL) {
        *out_y = 0U;
    }
    if (layout == NULL) {
        return;
    }
    if (layout->rank <= 1U) {
        if (out_x != NULL) {
            *out_x = index;
        }
        return;
    }

    if (out_x != NULL) {
        *out_x = (index / layout->stride_x) % layout->extent_x;
    }
    if (out_y != NULL) {
        *out_y = (index / layout->stride_y) % layout->extent_y;
    }
}

void sim_stimulus_static_cache_destroy(SimStimulusStaticCache* cache) {
    if (cache == NULL) {
        return;
    }
    free(cache->real);
    free(cache->imag);
    (void) memset(cache, 0, sizeof(*cache));
}

void sim_stimulus_static_cache_invalidate(SimStimulusStaticCache* cache) {
    if (cache == NULL) {
        return;
    }
    cache->valid = false;
    cache->count = 0U;
    cache->rank  = 0U;
    cache->extent_x = 0U;
    cache->extent_y = 0U;
    cache->stride_x = 0U;
    cache->stride_y = 0U;
}

static bool sim_stimulus_static_cache_matches(const SimStimulusStaticCache*            cache,
                                              const SimStimulusStaticCacheLayout*      layout,
                                              bool                                     need_imag) {
    if (cache == NULL || layout == NULL || !cache->valid) {
        return false;
    }
    if (cache->count != layout->count || cache->rank != layout->rank ||
        cache->extent_x != layout->extent_x || cache->extent_y != layout->extent_y ||
        cache->stride_x != layout->stride_x || cache->stride_y != layout->stride_y) {
        return false;
    }
    if (need_imag && cache->imag == NULL) {
        return false;
    }
    return cache->real != NULL;
}

SimResult sim_stimulus_static_cache_ensure(SimStimulusStaticCache*            cache,
                                           const SimStimulusStaticCacheLayout* layout,
                                           bool                                need_imag,
                                           SimStimulusStaticCacheFillFn        fill,
                                           void*                               userdata) {
    if (cache == NULL || layout == NULL || fill == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (layout->count == 0U) {
        return SIM_RESULT_OK;
    }

    if (sim_stimulus_static_cache_matches(cache, layout, need_imag)) {
        return SIM_RESULT_OK;
    }

    double* resized_real = cache->real;
    if (cache->capacity < layout->count) {
        resized_real = (double*) realloc(cache->real, layout->count * sizeof(double));
        if (resized_real == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    double* resized_imag = cache->imag;
    if (need_imag && cache->imag == NULL) {
        resized_imag = (double*) malloc(layout->count * sizeof(double));
        if (resized_imag == NULL) {
            if (resized_real != cache->real) {
                cache->real = resized_real;
                cache->capacity = layout->count;
            }
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    cache->real = resized_real;
    cache->imag = resized_imag;
    if (cache->capacity < layout->count) {
        cache->capacity = layout->count;
    }

    SimResult result = fill(userdata, layout, need_imag, cache->real, cache->imag);
    if (result != SIM_RESULT_OK) {
        cache->valid = false;
        return result;
    }

    cache->count    = layout->count;
    cache->rank     = layout->rank;
    cache->extent_x = layout->extent_x;
    cache->extent_y = layout->extent_y;
    cache->stride_x = layout->stride_x;
    cache->stride_y = layout->stride_y;
    cache->valid    = true;
    return SIM_RESULT_OK;
}
