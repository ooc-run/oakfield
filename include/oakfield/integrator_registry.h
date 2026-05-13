/**
 * @file integrator_registry.h
 * @brief Runtime selection utilities for libsimintegrators.
 *
 * The registry maps stable textual integrator names such as `"rk4"` and
 * `"etdrk4"` to factory callbacks. Registry storage is owned by the registry;
 * created Integrator instances are written into caller-provided memory and must
 * later be passed to integrator_destroy().
 */
#ifndef LIBSIMINTEGRATORS_INTEGRATOR_REGISTRY_H
#define LIBSIMINTEGRATORS_INTEGRATOR_REGISTRY_H

#include "integrator.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Factory function creating an integrator instance.
 *
 * @param config Optional configuration forwarded to the concrete factory.
 * @param[out] out Caller-owned integrator storage populated on success.
 * @return #SIM_RESULT_OK on success, or an error from validation,
 * allocation, or concrete integrator setup.
 */
typedef SimResult (*IntegratorCreateFn)(const IntegratorConfig *config, Integrator *out);

/**
 * @brief Entry stored by the registry.
 *
 * Names are copied into fixed-size storage and compared by their stored
 * byte sequence. Registering an existing name replaces its factory.
 */
typedef struct IntegratorRegistryEntry {
    char name[32];             /**< Integrator identifier. */
    IntegratorCreateFn create; /**< Factory callback. */
} IntegratorRegistryEntry;

/**
 * @brief Registry container for integrator factories.
 *
 * The entry array is heap-owned after initialization and must be released
 * with integrator_registry_destroy().
 */
typedef struct IntegratorRegistry {
    IntegratorRegistryEntry *entries; /**< Dynamic entry array. */
    size_t count;                     /**< Number of registered entries. */
    size_t capacity;                  /**< Allocated capacity. */
} IntegratorRegistry;

/**
 * @brief Initialize a registry and register built-in integrator factories.
 *
 * @param[out] registry Registry storage to initialize.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or
 * #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult integrator_registry_init(IntegratorRegistry *registry);

/**
 * @brief Release registry storage.
 *
 * @param registry Registry to destroy; NULL is ignored.
 */
void integrator_registry_destroy(IntegratorRegistry *registry);

/**
 * @brief Add or replace a factory under a registry name.
 *
 * @param registry Registry to mutate.
 * @param name Null-terminated integrator name copied into the registry.
 * @param create_fn Factory callback for the name.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or
 * #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult integrator_registry_register(IntegratorRegistry *registry, const char *name,
                                       IntegratorCreateFn create_fn);

/**
 * @brief Look up a factory by registry name.
 *
 * @param registry Registry to inspect.
 * @param name Null-terminated integrator name.
 * @return Factory callback, or NULL when the name is absent or inputs are invalid.
 */
IntegratorCreateFn integrator_registry_lookup(const IntegratorRegistry *registry, const char *name);

/**
 * @brief Create an integrator from a registered factory.
 *
 * @param registry Registry to inspect.
 * @param name Null-terminated integrator name.
 * @param config Optional configuration forwarded to the factory.
 * @param[out] out Caller-owned integrator storage populated on success.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_FOUND,
 * or an error returned by the selected factory.
 */
SimResult integrator_registry_create(const IntegratorRegistry *registry, const char *name,
                                     const IntegratorConfig *config, Integrator *out);

/**
 * @brief Register the built-in integrator factory set.
 *
 * Built-ins currently include `euler`, `heun`, `rk4`, `rkf45`,
 * `backward_euler`, `crank_nicolson`, `etdrk4`, and `subordination`.
 *
 * @param registry Registry to mutate.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or
 * #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult integrator_registry_register_builtin(IntegratorRegistry *registry);

#ifdef __cplusplus
}
#endif

#endif /* LIBSIMINTEGRATORS_INTEGRATOR_REGISTRY_H */
