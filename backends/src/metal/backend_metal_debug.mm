/**
 * @file backend_metal_debug.mm
 * @brief Test/debug inspection helpers for Metal backend internals.
 *
 * These helpers expose device and pipeline-cache state to tests when
 * SIM_TESTING declarations are enabled in the public backend header. They do
 * not mutate pipeline cache contents.
 */

#include "internal/backend_metal_internal.h"

#include <stdio.h>

extern "C" {

/**
 * @brief Copy the active Metal device name into a caller-owned buffer.
 */
SimResult sim_backend_metal_debug_copy_device_name(SimBackend *backend,
                                                   char       *out_name,
                                                   size_t      capacity)
{
    if (backend == NULL || out_name == NULL || capacity == 0U) return SIM_RESULT_INVALID_ARGUMENT;
    SimBackendMetalState *state = (SimBackendMetalState *)backend->impl;
    if (state == NULL || state->device == nil) return SIM_RESULT_INVALID_ARGUMENT;
    NSString *name = [state->device name];
    if (name == nil) return SIM_RESULT_INVALID_ARGUMENT;
    const char *utf8 = [name UTF8String];
    if (utf8 == NULL) return SIM_RESULT_INVALID_ARGUMENT;
    (void) snprintf(out_name, capacity, "%s", utf8);
    return SIM_RESULT_OK;
}

/**
 * @brief Return the number of cached Metal pipeline source entries.
 */
SimResult sim_backend_metal_debug_get_pipeline_cache_count(SimBackend *backend, size_t *out_count)
{
    if (backend == NULL || out_count == NULL) return SIM_RESULT_INVALID_ARGUMENT;
    SimBackendMetalState *state = (SimBackendMetalState *)backend->impl;
    if (state == NULL) return SIM_RESULT_INVALID_ARGUMENT;
    *out_count = state->pipeline_cache_by_hash.size();
    return SIM_RESULT_OK;
}

/**
 * @brief Sum reference counts across the Metal pipeline cache.
 */
SimResult sim_backend_metal_debug_get_total_pipeline_refcount(SimBackend *backend, size_t *out_total)
{
    if (backend == NULL || out_total == NULL) return SIM_RESULT_INVALID_ARGUMENT;
    SimBackendMetalState *state = (SimBackendMetalState *)backend->impl;
    if (state == NULL) return SIM_RESULT_INVALID_ARGUMENT;
    size_t total = 0;
    for (auto &kv : state->pipeline_cache_by_hash)
        total += kv.second.refcount;
    *out_total = total;
    return SIM_RESULT_OK;
}

} /* extern "C" */
