/**
 * @file sim_integrator_state.c
 * @brief Integrator state helpers.
 */
#include "oakfield/sim_integrator_state.h"
#include <stdlib.h>

SimResult sim_integrator_state_init(SimIntegratorState *state)
{
    SimResult result;

    if (state == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    state->active = NULL;
    state->stepping = NULL;
    state->sequence = NULL;
    state->sequence_count = 0U;
    state->sequence_capacity = 0U;
    state->registry_ready = false;
    result = integrator_registry_init(&state->registry);
    if (result != SIM_RESULT_OK)
    {
        return result;
    }

    state->registry_ready = true;
    return SIM_RESULT_OK;
}

void sim_integrator_state_destroy(SimIntegratorState *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->registry_ready)
    {
        integrator_registry_destroy(&state->registry);
    }
    free(state->sequence);
    state->sequence = NULL;
    state->sequence_count = 0U;
    state->sequence_capacity = 0U;
    state->registry_ready = false;
    state->active = NULL;
    state->stepping = NULL;
}

void sim_integrator_state_set_active(SimIntegratorState *state, struct Integrator *integrator)
{
    if (state == NULL)
    {
        return;
    }

    state->active = integrator;
    state->stepping = NULL;
    state->sequence_count = 0U;
}

SimResult sim_integrator_state_set_sequence(SimIntegratorState *state,
                                            struct Integrator *const *integrators,
                                            size_t count)
{
    if (state == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (count == 0U)
    {
        state->sequence_count = 0U;
        state->active = NULL;
        state->stepping = NULL;
        return SIM_RESULT_OK;
    }

    if (integrators == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < count; ++i)
    {
        if (integrators[i] == NULL)
        {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    if (state->sequence_capacity < count)
    {
        struct Integrator **next =
            (struct Integrator **)realloc(state->sequence, count * sizeof(struct Integrator *));
        if (next == NULL)
        {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->sequence = next;
        state->sequence_capacity = count;
    }

    for (size_t i = 0U; i < count; ++i)
    {
        state->sequence[i] = integrators[i];
    }
    state->sequence_count = count;
    state->active = state->sequence[0];
    state->stepping = NULL;
    return SIM_RESULT_OK;
}

struct Integrator *sim_integrator_state_active(const SimIntegratorState *state)
{
    if (state == NULL)
    {
        return NULL;
    }

    return state->active;
}

void sim_integrator_state_set_stepping(SimIntegratorState *state, struct Integrator *integrator)
{
    if (state == NULL)
    {
        return;
    }

    state->stepping = integrator;
}

struct Integrator *sim_integrator_state_stepping(const SimIntegratorState *state)
{
    if (state == NULL)
    {
        return NULL;
    }

    return state->stepping;
}

const struct Integrator *const *sim_integrator_state_sequence(const SimIntegratorState *state,
                                                              size_t *out_count)
{
    if (out_count != NULL)
    {
        *out_count = (state != NULL) ? state->sequence_count : 0U;
    }
    if (state == NULL || state->sequence_count == 0U)
    {
        return NULL;
    }
    return (const struct Integrator *const *)state->sequence;
}

IntegratorRegistry *sim_integrator_state_registry(SimIntegratorState *state)
{
    if (state == NULL)
    {
        return NULL;
    }

    return state->registry_ready ? &state->registry : NULL;
}

const IntegratorRegistry *sim_integrator_state_registry_const(const SimIntegratorState *state)
{
    if (state == NULL)
    {
        return NULL;
    }

    return state->registry_ready ? &state->registry : NULL;
}
