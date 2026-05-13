/**
 * @file sim_scheduler_state.h
 * @brief Execution plan and backend coordination state.
 */
#ifndef OAKFIELD_SIM_SCHEDULER_STATE_H
#define OAKFIELD_SIM_SCHEDULER_STATE_H

#include <stdbool.h>

#include "kernel_ir.h"
#include "operator.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimBackend;

/**
 * @brief Scheduler-local state for plan resolution and backend dispatch.
 */
typedef struct SimSchedulerPlan {
    SimIRBuilder ir_builder;    /**< Shared builder for fused kernels. */
    SimOperatorPlan plan;       /**< Cached execution plan. */
    bool plan_valid;            /**< Indicates whether plan reflects registry state. */
    struct SimBackend *backend; /**< Active backend used for kernel launches (optional). */
} SimSchedulerPlan;

/**
 * @brief Initialize scheduler state and its shared IR builder.
 *
 * @param state Scheduler state to initialize.
 * @return #SIM_RESULT_OK or a builder initialization error.
 */
SimResult sim_scheduler_plan_init(SimSchedulerPlan *state);

/**
 * @brief Destroy scheduler state and release owned resources.
 *
 * @param state Scheduler state to destroy; NULL is ignored.
 */
void sim_scheduler_plan_destroy(SimSchedulerPlan *state);

/**
 * @brief Mark the cached plan as invalid.
 *
 * @param state Scheduler state to update; NULL is ignored.
 */
void sim_scheduler_plan_invalidate(SimSchedulerPlan *state);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_SCHEDULER_STATE_H */
