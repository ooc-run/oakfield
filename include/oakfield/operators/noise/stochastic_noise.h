/**
 * @file stochastic_noise.h
 * @brief Additive stochastic noise operator (OU/pink/blue via spectral shaping and calculus law).
 *
 * Noise can be interpreted under Ito/Stratonovich (see SimIRNoiseLaw). Real and complex fields
 * are both supported; complex fields receive independent real/imaginary draws.
 */
#ifndef OAKFIELD_STOCHASTIC_NOISE_H
#define OAKFIELD_STOCHASTIC_NOISE_H

#include "oakfield/kernel_ir.h"
#include "oakfield/operator_split.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration parameters for the additive stochastic noise operator.
 */
typedef struct StochasticNoiseOperatorConfig {
    size_t field_index;      /**< Field index receiving the noise term. */
    double sigma;            /**< Noise strength (standard deviation). */
    double tau;              /**< Autocorrelation decay time (seconds). */
    double alpha;            /**< Spectral exponent (1.0 = OU, <1 pinkish, >1 bluish). */
    unsigned long long seed; /**< Seed for deterministic random streams. */
    SimIRNoiseLaw law;       /**< Stochastic calculus interpretation (Ito or Stratonovich). */
} StochasticNoiseOperatorConfig;

/**
 * @brief Register an additive stochastic noise operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional noise generation parameters; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         allocation, kernel registration, or split registration.
 */
SimResult sim_add_stochastic_noise_operator(struct SimContext *context,
                                            const StochasticNoiseOperatorConfig *config,
                                            size_t *out_index);

/**
 * @brief Copy the current stochastic noise configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stochastic_noise_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no noise state.
 */
SimResult sim_stochastic_noise_config(struct SimContext *context, size_t operator_index,
                                      StochasticNoiseOperatorConfig *out_config);

/**
 * @brief Replace or renormalize a registered stochastic noise configuration.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update reseeds the RNG and clears cached noise
 * state buffers.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the stochastic noise operator to update.
 * @param config Optional replacement noise configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, or state validation.
 */
SimResult sim_stochastic_noise_update(struct SimContext *context, size_t operator_index,
                                      const StochasticNoiseOperatorConfig *config);

#ifdef __cplusplus
}
#endif
#endif /* OAKFIELD_STOCHASTIC_NOISE_H */
