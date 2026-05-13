/**
 * @file backend_metal_internal.h
 * @brief Shared private Metal backend state for runtime and debug helpers.
 *
 * This header is intentionally private to the Metal backend implementation and
 * test/debug helpers. It exposes Objective-C Metal objects and C++ cache maps,
 * so it should not be included by portable C code.
 */
#ifndef BACKEND_METAL_INTERNAL_H
#define BACKEND_METAL_INTERNAL_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "oakfield/backend.h"
#include <unordered_map>

/**
 * @brief Per-KernelIR Metal pipeline and buffer cache entry.
 *
 * Entries keep converted float buffers for field data and constants plus the
 * compiled compute pipeline associated with a generated MSL source hash.
 */
typedef struct SimBackendMetalPipeline {
    const KernelIR*             kernel;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer>               constants_buffer;
    size_t                      constants_bytes;
    id<MTLBuffer>*              field_buffers;
    size_t                      field_buffer_count;
    id<MTLBuffer>               out_buffer;
    size_t                      out_buffer_bytes;
    size_t                      src_hash;
} SimBackendMetalPipeline;

/**
 * @brief Metal backend private state.
 *
 * The pipeline cache maps generated MSL source hashes to compiled pipeline
 * states with explicit reference counts so multiple KernelIR entries can share
 * equivalent generated shader code.
 */
typedef struct SimBackendMetalState {
    id<MTLDevice>            device;
    id<MTLCommandQueue>      queue;
    SimBackendMetalPipeline* pipelines;
    size_t                   pipeline_count;
    size_t                   pipeline_capacity;
    struct PipelineCacheEntry {
        id<MTLComputePipelineState> pipeline;
        size_t                      refcount;
    };
    std::unordered_map<size_t, PipelineCacheEntry> pipeline_cache_by_hash;
} SimBackendMetalState;

#endif /* BACKEND_METAL_INTERNAL_H */
