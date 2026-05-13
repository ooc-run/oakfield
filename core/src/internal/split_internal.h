/**
 * @file split_internal.h
 * @brief Internal utilities for split operators (scratch arenas & notifications).
 */
#ifndef OAKFIELD_SPLIT_INTERNAL_H
#define OAKFIELD_SPLIT_INTERNAL_H

#include "oakfield/sim_context.h"

typedef struct SimSplitWorkerScratch {
    uint8_t* arena;     /**< Worker-local scratch arena. */
    size_t   size;      /**< Arena size in bytes. */
    size_t   alignment; /**< Arena alignment in bytes. */
} SimSplitWorkerScratch;

/**
 * @brief Allocate scratch arenas for split-operator workers.
 *
 * @param ctx Context used for memory accounting.
 * @param bytes_per_worker Scratch bytes requested for each worker.
 * @param alignment Requested alignment; implementation may apply a safer minimum.
 * @param[out] out_scratch Receives owned scratch descriptors.
 * @param[out] out_count Receives number of descriptors.
 * @return #SIM_RESULT_OK or an allocation/accounting error.
 */
SimResult sim_split_configure_scratch(struct SimContext*      ctx,
                                      size_t                  bytes_per_worker,
                                      size_t                  alignment,
                                      SimSplitWorkerScratch** out_scratch,
                                      size_t*                 out_count);

/**
 * @brief Release scratch arenas allocated by sim_split_configure_scratch().
 *
 * @param ctx Context used for memory accounting.
 * @param scratch Scratch descriptor array to release; NULL is ignored.
 * @param count Number of entries in @p scratch.
 */
void sim_split_release_scratch(struct SimContext*     ctx,
                               SimSplitWorkerScratch* scratch,
                               size_t                 count);

/**
 * @brief Notify the active integrator about a split substep result.
 *
 * @param ctx Context containing the active integrator state.
 * @param dt_sub Substep timestep.
 * @param err_estimate Optional local error estimate.
 */
void sim_split_notify_integrator(struct SimContext* ctx, double dt_sub, double err_estimate);

#endif  // OAKFIELD_SPLIT_INTERNAL_H
