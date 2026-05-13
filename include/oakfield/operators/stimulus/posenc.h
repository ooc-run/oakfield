/**
 * @file posenc.h
 * @brief NeRF-style positional encoding stimulus.
 *
 * Encodes coordinates with dyadic Fourier bands and injects the summed embedding:
 *   e(u, t) = [optional u] + sum_l cos(k_l u - omega t + phase) + i sin(...)
 * where k_l = base_wavenumber * band_growth^l.
 */
#ifndef OAKFIELD_STIMULUS_POSENC_H
#define OAKFIELD_STIMULUS_POSENC_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for NeRF-style positional encoding stimulus bands.
 */
typedef struct SimStimulusPosEncConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double base_wavenumber;       /**< Base spatial wavenumber (rad / unit). */
    double band_growth;           /**< Geometric growth factor between bands. */
    unsigned int band_count;      /**< Number of positional-encoding bands. */
    double kx;                    /**< Optional wavevector X component. */
    double ky;                    /**< Optional wavevector Y component. */
    double omega;                 /**< Temporal angular frequency (rad / s). */
    double phase;                 /**< Global phase offset (radians). */
    SimStimulusCoordConfig coord; /**< Coordinate mapping (when not wavevector mode). */
    double time_offset;           /**< Additional time offset applied before evaluation. */
    double rotation;              /**< Complex-output rotation (radians, complex only). */
    bool include_identity;        /**< Add identity term u before band sum. */
    bool use_wavevector;          /**< Use (kx,ky) projection instead of coord mapping. */
    bool scale_by_dt;             /**< Scale writes by dt when true; else dt-independent. */
} SimStimulusPosEncConfig;

/**
 * @brief Register a positional-encoding stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that sums the configured Fourier encoding bands on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional positional-encoding configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         buffer setup, or split-operator registration.
 */
SimResult sim_add_stimulus_posenc_operator(struct SimContext *context,
                                           const SimStimulusPosEncConfig *config,
                                           size_t *out_index);

/**
 * @brief Copy the current positional-encoding configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_posenc_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_posenc_config(struct SimContext *context, size_t operator_index,
                                     SimStimulusPosEncConfig *out_config);

/**
 * @brief Replace or renormalize a registered positional-encoding configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the positional-encoding operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, buffer setup,
 *         or state validation fails.
 */
SimResult sim_stimulus_posenc_update(struct SimContext *context, size_t operator_index,
                                     const SimStimulusPosEncConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_POSENC_H */
