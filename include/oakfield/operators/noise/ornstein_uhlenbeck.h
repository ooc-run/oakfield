/**
 * @file ornstein_uhlenbeck.h
 * @brief Ornstein-Uhlenbeck stochastic process operator.
 */
#ifndef OAKFIELD_ORNSTEIN_UHLENBECK_H
#define OAKFIELD_ORNSTEIN_UHLENBECK_H

#include "oakfield/operator_split.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Coordinate system used when applying OU noise to complex fields.
 */
typedef enum SimOrnsteinUhlenbeckComplexMode {
    SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT =
        0, /**< Evolve real/imaginary scalar components independently. */
    SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_POLAR =
        1 /**< Evolve radius/phase coordinates independently, then reconstruct. */
} SimOrnsteinUhlenbeckComplexMode;

/**
 * @brief Configuration for the Ornstein-Uhlenbeck operator.
 */
typedef struct SimOrnsteinUhlenbeckOperatorConfig {
    size_t field_index; /**< Target field index. */
    double mean;        /**< Mean reversion target for each evolved scalar coordinate. */
    double sigma;       /**< Stationary standard deviation of the process. */
    double tau;         /**< Relaxation time constant in seconds. */
    uint64_t seed;      /**< Seed for deterministic random streams (0 => auto). */
    SimOrnsteinUhlenbeckComplexMode complex_mode; /**< Complex-field update mode. */
} SimOrnsteinUhlenbeckOperatorConfig;

/**
 * @brief Register an Ornstein-Uhlenbeck process operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional OU configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, or split registration.
 */
SimResult sim_add_ornstein_uhlenbeck_operator(struct SimContext *context,
                                              const SimOrnsteinUhlenbeckOperatorConfig *config,
                                              size_t *out_index);

/**
 * @brief Copy the current OU configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_ornstein_uhlenbeck_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no OU state.
 */
SimResult sim_ornstein_uhlenbeck_config(struct SimContext *context, size_t operator_index,
                                        SimOrnsteinUhlenbeckOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered OU operator.
 *
 * @p config is required. Updating reseeds the process RNG, refreshes symbolic
 * metadata, and validates the target field before storing the replacement.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the OU operator to update.
 * @param config Replacement OU configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field validation, or state validation.
 */
SimResult sim_ornstein_uhlenbeck_update(struct SimContext *context, size_t operator_index,
                                        const SimOrnsteinUhlenbeckOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_ORNSTEIN_UHLENBECK_H */
