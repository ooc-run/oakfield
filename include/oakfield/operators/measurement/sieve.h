/**
 * @file sieve.h
 * @brief Discrete convolution-based filtering on real or complex fields.
 *
 * Supports Gaussian, difference-of-Gaussians, Savitzky-Golay, and windowed responses.
 * Complex support is component-wise (Re/Im filtered independently).
 */
#ifndef OAKFIELD_SIEVE_H
#define OAKFIELD_SIEVE_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Available response shapes for the sieve operator.
 */
typedef enum SimSieveMode {
    SIM_SIEVE_MODE_LOW_PASS = 0,       /**< Gaussian-like low-pass response. */
    SIM_SIEVE_MODE_HIGH_PASS,          /**< High-pass response. */
    SIM_SIEVE_MODE_BAND_PASS_DOG,      /**< Difference-of-Gaussians band-pass response. */
    SIM_SIEVE_MODE_BAND_STOP_DOG,      /**< Difference-of-Gaussians band-stop response. */
    SIM_SIEVE_MODE_SAVGOL_SMOOTH,      /**< Savitzky-Golay smoothing response. */
    SIM_SIEVE_MODE_SAVGOL_DERIVATIVE,  /**< Savitzky-Golay derivative response. */
    SIM_SIEVE_MODE_HANN_LOW_PASS,      /**< Hann-windowed low-pass response. */
    SIM_SIEVE_MODE_HANN_HIGH_PASS,     /**< Hann-windowed high-pass response. */
    SIM_SIEVE_MODE_BLACKMAN_LOW_PASS,  /**< Blackman-windowed low-pass response. */
    SIM_SIEVE_MODE_BLACKMAN_HIGH_PASS, /**< Blackman-windowed high-pass response. */
    SIM_SIEVE_MODE_TUKEY_LOW_PASS,     /**< Tukey-windowed low-pass response. */
    SIM_SIEVE_MODE_TUKEY_HIGH_PASS     /**< Tukey-windowed high-pass response. */
} SimSieveMode;

/**
 * @brief Configuration parameters for the sieve operator family.
 */
typedef struct SimSieveOperatorConfig {
    size_t input_field;            /**< Field to be filtered. */
    size_t output_field;           /**< Field receiving the filtered result. */
    unsigned int taps;             /**< Kernel length (odd, >=3). */
    double sigma;                  /**< Primary Gaussian/window scale. */
    double sigma2;                 /**< Secondary scale used by band-pass/stop (DoG). */
    unsigned int poly_order;       /**< Polynomial order for Savitzky-Golay modes. */
    unsigned int derivative_order; /**< Derivative order for Savitzky-Golay derivative mode. */
    double sample_spacing;         /**< Sample spacing used to scale derivatives. */
    double window_alpha;           /**< Tukey alpha override; <=0 derives from sigma. */
    double gain;                   /**< Output gain applied after filtering. */
    SimSieveMode mode;             /**< Selected sieve response family. */
    SimIRBoundaryPolicy boundary;  /**< Boundary handling policy for out-of-range samples. */
    bool accumulate;               /**< Add into output when true. */
    bool scale_by_dt;              /**< When true, scale accumulated writes by substep dt. */
} SimSieveOperatorConfig;

/**
 * @brief Register a sieve filtering operator with the provided configuration.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional sieve configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         kernel construction, field compatibility checks, allocation, or registration.
 */
SimResult sim_add_sieve_operator(struct SimContext *context, const SimSieveOperatorConfig *config,
                                 size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a sieve operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_sieve_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no sieve state.
 */
SimResult sim_sieve_config(struct SimContext *context, size_t operator_index,
                           SimSieveOperatorConfig *out_config);

/**
 * @brief Update an existing sieve operator in-place.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update rebuilds the filter kernel and invalidates
 * the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the sieve operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code from lookup, kernel
 *         construction, field compatibility checks, or state validation.
 */
SimResult sim_sieve_update(struct SimContext *context, size_t operator_index,
                           const SimSieveOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIEVE_H */
