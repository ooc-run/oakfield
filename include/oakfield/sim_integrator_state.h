/**
 * @file sim_integrator_state.h
 * @brief Integrator registry and active integrator tracking.
 */
#ifndef OAKFIELD_SIM_INTEGRATOR_STATE_H
#define OAKFIELD_SIM_INTEGRATOR_STATE_H

#include <stdbool.h>
#include <stddef.h>

#include "oakfield/integrator_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Integrator registry container paired with active execution pointers.
 */
typedef struct SimIntegratorState {
    IntegratorRegistry registry;  /**< Registry of available integrator factories. */
    struct Integrator *active;    /**< Currently bound integrator (not owned). */
    struct Integrator *stepping;  /**< Integrator currently executing a step (not owned). */
    struct Integrator **sequence; /**< Optional integrator sequence (not owned). */
    size_t sequence_count;        /**< Number of integrators in @ref sequence. */
    size_t sequence_capacity;     /**< Capacity of @ref sequence storage. */
    bool registry_ready;          /**< True when registry initialized successfully. */
} SimIntegratorState;

/**
 * @brief Initialize integrator state and register built-in factories.
 *
 * @param state State object to initialize.
 * @return #SIM_RESULT_OK or an error from registry initialization.
 */
SimResult sim_integrator_state_init(SimIntegratorState *state);

/**
 * @brief Destroy integrator state and release registry storage.
 *
 * @param state State object to destroy; NULL is ignored.
 */
void sim_integrator_state_destroy(SimIntegratorState *state);

/**
 * @brief Assign the active integrator pointer.
 *
 * @param state State object to update.
 * @param integrator Active integrator pointer; not owned by the state.
 */
void sim_integrator_state_set_active(SimIntegratorState *state, struct Integrator *integrator);

/**
 * @brief Assign an optional ordered integrator sequence.
 *
 * When @p count is zero, the sequence and active pointer are both cleared. The
 * sequence array is copied, but the Integrator instances remain caller-owned.
 *
 * @param state State object to update.
 * @param integrators Array of integrator pointers when @p count is nonzero.
 * @param count Number of entries in @p integrators.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_integrator_state_set_sequence(SimIntegratorState *state,
                                            struct Integrator *const *integrators, size_t count);

/**
 * @brief Retrieve the active integrator pointer.
 *
 * @param state State object to inspect.
 * @return Active integrator pointer, or NULL when none is set.
 */
struct Integrator *sim_integrator_state_active(const SimIntegratorState *state);

/**
 * @brief Mark an integrator as the one currently stepping.
 *
 * @param state State object to update.
 * @param integrator Integrator currently executing a step; not owned.
 */
void sim_integrator_state_set_stepping(SimIntegratorState *state, struct Integrator *integrator);

/**
 * @brief Retrieve the integrator currently executing a step, if any.
 *
 * @param state State object to inspect.
 * @return Stepping integrator pointer, or NULL.
 */
struct Integrator *sim_integrator_state_stepping(const SimIntegratorState *state);

/**
 * @brief Retrieve the optional integrator sequence.
 *
 * @param state State object to inspect.
 * @param[out] out_count Optional receiver for sequence length.
 * @return Internal sequence pointer, or NULL when no sequence is configured.
 */
const struct Integrator *const *sim_integrator_state_sequence(const SimIntegratorState *state,
                                                              size_t *out_count);

/**
 * @brief Access the underlying registry.
 *
 * @param state State object to inspect.
 * @return Mutable registry pointer, or NULL if unavailable.
 */
IntegratorRegistry *sim_integrator_state_registry(SimIntegratorState *state);

/**
 * @brief Access the underlying registry (const).
 *
 * @param state State object to inspect.
 * @return Const registry pointer, or NULL if unavailable.
 */
const IntegratorRegistry *sim_integrator_state_registry_const(const SimIntegratorState *state);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_INTEGRATOR_STATE_H */
