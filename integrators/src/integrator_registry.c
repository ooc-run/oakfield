/**
 * @file integrator_registry.c
 * @brief Runtime registry for built-in and custom integrator factories.
 *
 * The registry owns only its table of name-to-factory entries. Concrete
 * Integrator instances remain caller-owned and are initialized by the selected
 * factory. Registering a duplicate name replaces the previous factory to allow
 * tests or downstream modules to override built-ins deliberately.
 */

#include "oakfield/integrator_registry.h"

#include <stdlib.h>
#include <string.h>

/* Built-in factories implemented by the individual integrator modules. */
extern SimResult integrator_euler_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_heun_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_rk4_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_rkf45_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_crank_nicolson_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_backward_euler_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_etdrk4_create(const IntegratorConfig* config, Integrator* out);
extern SimResult integrator_subordination_create(const IntegratorConfig* config, Integrator* out);

/**
 * @brief Expand registry entry storage when the table is full.
 *
 * @param registry Registry whose entry array may be reallocated.
 * @return #SIM_RESULT_OK or #SIM_RESULT_OUT_OF_MEMORY.
 */
static SimResult integrator_registry_grow(IntegratorRegistry* registry) {
    size_t                   new_capacity;
    IntegratorRegistryEntry* entries;

    if (registry->count < registry->capacity) {
        return SIM_RESULT_OK;
    }

    new_capacity = (registry->capacity == 0U) ? 4U : (registry->capacity * 2U);
    entries      = (IntegratorRegistryEntry*) realloc(registry->entries,
                                                 new_capacity * sizeof(IntegratorRegistryEntry));
    if (entries == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    registry->entries  = entries;
    registry->capacity = new_capacity;
    return SIM_RESULT_OK;
}

SimResult integrator_registry_init(IntegratorRegistry* registry) {
    SimResult result;

    if (registry == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    registry->entries  = NULL;
    registry->count    = 0U;
    registry->capacity = 0U;

    result = integrator_registry_register_builtin(registry);
    if (result != SIM_RESULT_OK) {
        integrator_registry_destroy(registry);
        return result;
    }

    return SIM_RESULT_OK;
}

void integrator_registry_destroy(IntegratorRegistry* registry) {
    if (registry == NULL) {
        return;
    }

    free(registry->entries);
    registry->entries  = NULL;
    registry->count    = 0U;
    registry->capacity = 0U;
}

SimResult integrator_registry_register(IntegratorRegistry* registry,
                                       const char*         name,
                                       IntegratorCreateFn  create_fn) {
    size_t    i;
    size_t    length;
    SimResult result;

    if (registry == NULL || name == NULL || create_fn == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (i = 0U; i < registry->count; ++i) {
        if (strncmp(registry->entries[i].name, name, sizeof(registry->entries[i].name)) == 0) {
            registry->entries[i].create = create_fn;
            return SIM_RESULT_OK;
        }
    }

    result = integrator_registry_grow(registry);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    length = strlen(name);
    if (length >= sizeof(registry->entries[registry->count].name)) {
        length = sizeof(registry->entries[registry->count].name) - 1U;
    }

    (void) memcpy(registry->entries[registry->count].name, name, length);
    registry->entries[registry->count].name[length] = '\0';
    registry->entries[registry->count].create       = create_fn;
    registry->count += 1U;

    return SIM_RESULT_OK;
}

IntegratorCreateFn integrator_registry_lookup(const IntegratorRegistry* registry,
                                              const char*               name) {
    size_t i;

    if (registry == NULL || name == NULL) {
        return NULL;
    }

    for (i = 0U; i < registry->count; ++i) {
        if (strncmp(registry->entries[i].name, name, sizeof(registry->entries[i].name)) == 0) {
            return registry->entries[i].create;
        }
    }

    return NULL;
}

SimResult integrator_registry_create(const IntegratorRegistry* registry,
                                     const char*               name,
                                     const IntegratorConfig*   config,
                                     Integrator*               out) {
    IntegratorCreateFn factory;

    if (registry == NULL || name == NULL || out == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    factory = integrator_registry_lookup(registry, name);
    if (factory == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    return factory(config, out);
}

SimResult integrator_registry_register_builtin(IntegratorRegistry* registry) {
    SimResult result;

    result = integrator_registry_register(registry, "euler", integrator_euler_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = integrator_registry_register(registry, "heun", integrator_heun_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = integrator_registry_register(registry, "rk4", integrator_rk4_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = integrator_registry_register(registry, "rkf45", integrator_rkf45_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result =
        integrator_registry_register(registry, "backward_euler", integrator_backward_euler_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result =
        integrator_registry_register(registry, "crank_nicolson", integrator_crank_nicolson_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = integrator_registry_register(registry, "etdrk4", integrator_etdrk4_create);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result =
        integrator_registry_register(registry, "subordination", integrator_subordination_create);

    if (result != SIM_RESULT_OK) {
        return result;
    }

    return SIM_RESULT_OK;
}
