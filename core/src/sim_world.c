/**
 * @file sim_world.c
 * @brief Static world container implementation.
 */
#include "oakfield/sim_world.h"

#include <stdlib.h>
#include <string.h>

#include "oakfield/field.h"

static void sim_world_copy_universe_defaults(SimUniverseSpec *dst)
{
    if (dst == NULL)
    {
        return;
    }

    dst->poles = NULL;
    dst->pole_count = 0U;
    dst->q = SIM_WORLD_DEFAULT_Q;
    dst->K = SIM_WORLD_DEFAULT_K;
    dst->epsilon = SIM_WORLD_DEFAULT_EPSILON;
    dst->sieve_sigma = SIM_WORLD_DEFAULT_SIEVE_SIGMA;
}

static SimResult sim_world_copy_universe(SimUniverseSpec *dst, const SimUniverseSpec *src)
{
    if (dst == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (src == NULL)
    {
        sim_world_copy_universe_defaults(dst);
        return SIM_RESULT_OK;
    }

    dst->q = src->q;
    dst->K = src->K;
    dst->epsilon = src->epsilon;
    dst->sieve_sigma = src->sieve_sigma;
    dst->pole_count = src->pole_count;
    dst->poles = NULL;

    if (src->pole_count == 0U || src->poles == NULL)
    {
        return SIM_RESULT_OK;
    }

    size_t bytes = src->pole_count * sizeof(SimPole);
    dst->poles = (SimPole *)malloc(bytes);
    if (dst->poles == NULL)
    {
        dst->pole_count = 0U;
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    memcpy(dst->poles, src->poles, bytes);
    return SIM_RESULT_OK;
}

static void sim_world_release_universe(SimUniverseSpec *spec)
{
    if (spec == NULL)
    {
        return;
    }

    if (spec->poles != NULL)
    {
        free(spec->poles);
        spec->poles = NULL;
    }
    spec->pole_count = 0U;
    spec->q = SIM_WORLD_DEFAULT_Q;
    spec->K = SIM_WORLD_DEFAULT_K;
    spec->epsilon = SIM_WORLD_DEFAULT_EPSILON;
    spec->sieve_sigma = SIM_WORLD_DEFAULT_SIEVE_SIGMA;
}

SimResult sim_world_init(SimWorld *world, const SimUniverseSpec *universe_spec)
{
    SimResult result;

    if (world == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    world->fields = NULL;
    world->field_count = 0U;
    world->field_capacity = 0U;
    world->field_continuity = NULL;
    world->operators.records = NULL;
    world->operators.count = 0U;
    world->operators.capacity = 0U;
    sim_world_copy_universe_defaults(&world->universe);

    result = sim_world_copy_universe(&world->universe, universe_spec);
    if (result != SIM_RESULT_OK)
    {
        sim_world_release_universe(&world->universe);
        return result;
    }

    result = sim_operator_registry_init(&world->operators);
    if (result != SIM_RESULT_OK)
    {
        sim_world_release_universe(&world->universe);
        world->operators.records = NULL;
        world->operators.count = 0U;
        world->operators.capacity = 0U;
    }

    return result;
}

SimResult sim_world_reserve_fields(SimWorld *world, size_t additional)
{
    SimField *new_fields;
    size_t required;
    size_t new_capacity;
    SimFieldContinuityOverride *new_continuity;
    size_t previous_capacity;

    if (world == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (additional == 0U)
    {
        return SIM_RESULT_OK;
    }

    required = world->field_count + additional;
    if (required <= world->field_capacity)
    {
        return SIM_RESULT_OK;
    }

    new_capacity = (world->field_capacity == 0U) ? 4U : world->field_capacity;
    while (new_capacity < required)
    {
        new_capacity *= 2U;
    }

    new_fields = (SimField *)realloc(world->fields, new_capacity * sizeof(SimField));
    if (new_fields == NULL)
    {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    world->fields = new_fields;

    new_continuity = (SimFieldContinuityOverride *)realloc(world->field_continuity,
                                                           new_capacity * sizeof(SimFieldContinuityOverride));
    if (new_continuity == NULL)
    {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    previous_capacity = world->field_capacity;
    world->field_continuity = new_continuity;
    for (size_t i = previous_capacity; i < new_capacity; ++i)
    {
        world->field_continuity[i].enabled = false;
        world->field_continuity[i].config = sim_operator_config_defaults();
    }
    world->field_capacity = new_capacity;
    return SIM_RESULT_OK;
}

void sim_world_reset_fields(SimWorld *world)
{
    if (world == NULL)
    {
        return;
    }

    if (world->fields != NULL)
    {
        for (size_t i = 0U; i < world->field_count; ++i)
        {
            sim_field_destroy(&world->fields[i]);
        }
        free(world->fields);
        world->fields = NULL;
    }

    world->field_count = 0U;
    world->field_capacity = 0U;
    if (world->field_continuity != NULL)
    {
        free(world->field_continuity);
        world->field_continuity = NULL;
    }
}

void sim_world_destroy(SimWorld *world)
{
    if (world == NULL)
    {
        return;
    }

    sim_world_reset_fields(world);
    sim_operator_registry_destroy(&world->operators);
    world->operators.records = NULL;
    world->operators.count = 0U;
    world->operators.capacity = 0U;

    sim_world_release_universe(&world->universe);
}

SimResult sim_world_set_field_continuity_override(SimWorld *world,
                                                  size_t field_index,
                                                  bool enabled,
                                                  const SimOperatorConfig *config)
{
    SimOperatorConfig normalized = sim_operator_config_defaults();

    if (world == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field_index >= world->field_count || world->field_continuity == NULL)
    {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (config != NULL)
    {
        normalized = *config;
    }
    sim_operator_config_normalize(&normalized);

    if (!enabled)
    {
        world->field_continuity[field_index].enabled = false;
        world->field_continuity[field_index].config = sim_operator_config_defaults();
        return SIM_RESULT_OK;
    }

    world->field_continuity[field_index].enabled = true;
    world->field_continuity[field_index].config = normalized;
    return SIM_RESULT_OK;
}

bool sim_world_field_continuity_override(const SimWorld *world,
                                         size_t field_index,
                                         SimOperatorConfig *out_config)
{
    if (world == NULL || field_index >= world->field_count || world->field_continuity == NULL)
    {
        return false;
    }

    const SimFieldContinuityOverride *slot = &world->field_continuity[field_index];
    if (!slot->enabled)
    {
        return false;
    }

    if (out_config != NULL)
    {
        *out_config = slot->config;
    }
    return true;
}
