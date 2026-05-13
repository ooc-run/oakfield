/**
 * @file sim_scheduler_state.c
 * @brief Scheduler state helpers.
 */
#include "oakfield/sim_scheduler_state.h"

#include <string.h>

SimResult sim_scheduler_plan_init(SimSchedulerPlan *state)
{
    if (state == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    memset(&state->plan, 0, sizeof(state->plan));
    state->plan_valid = false;
    state->backend = NULL;
    return sim_ir_builder_init(&state->ir_builder);
}

void sim_scheduler_plan_destroy(SimSchedulerPlan *state)
{
    if (state == NULL)
    {
        return;
    }

    sim_operator_plan_destroy(&state->plan);
    sim_ir_builder_destroy(&state->ir_builder);
    state->plan_valid = false;
    state->backend = NULL;
}

void sim_scheduler_plan_invalidate(SimSchedulerPlan *state)
{
    if (state == NULL)
    {
        return;
    }

    state->plan_valid = false;
}
