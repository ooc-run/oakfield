/**
 * @file traveling_wave_packet.h
 * @brief Gaussian-envelope traveling wave packet stimulus.
 *
 * Evaluates a drifting local packet frame (u, v) and injects
 *   A * exp(-0.5 * ((u / sigma_u)^2 + (v / sigma_v)^2))
 *     * exp(i * (k_u * u + k_v * v - omega * t + phi)),
 * where the packet center and orientation can evolve over time.
 *
 * Real fields receive the real component. Complex fields receive the full complex
 * packet with an additional global rotation.
 */
#ifndef OAKFIELD_STIMULUS_TRAVELING_WAVE_PACKET_H
#define OAKFIELD_STIMULUS_TRAVELING_WAVE_PACKET_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Configuration for a drifting Gaussian-envelope traveling wave packet.
 */
typedef struct SimStimulusTravelingWavePacketConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Output amplitude scale. */
    double sigma_u;               /**< Gaussian packet width along local u. */
    double sigma_v;               /**< Gaussian packet width along local v. */
    double center_u;              /**< Packet center in local u. */
    double center_v;              /**< Packet center in local v. */
    double velocity_u;            /**< Packet-center drift in local u. */
    double velocity_v;            /**< Packet-center drift in local v. */
    double orientation;           /**< Local packet orientation angle. */
    double orientation_rate;      /**< Orientation drift rate. */
    double carrier_u;             /**< Carrier wavenumber along local u. */
    double carrier_v;             /**< Carrier wavenumber along local v. */
    double omega;                 /**< Temporal angular frequency. */
    double phase;                 /**< Phase offset. */
    SimStimulusCoordConfig coord; /**< Coordinate mapping into the local packet frame. */
    double time_offset;           /**< Additional time offset before evaluation. */
    double rotation;              /**< Global complex-output rotation. */
    bool scale_by_dt;             /**< Scale writes by dt when true. */
} SimStimulusTravelingWavePacketConfig;

/**
 * @brief Register a Gaussian-envelope traveling wave packet stimulus operator.
 *
 * The implementation copies and normalizes @p config, then registers a split
 * operator that writes the drifting packet into the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional wave-packet configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         or split-operator registration.
 */
SimResult
sim_add_stimulus_traveling_wave_packet_operator(struct SimContext *context,
                                                const SimStimulusTravelingWavePacketConfig *config,
                                                size_t *out_index);

/**
 * @brief Copy the current traveling wave-packet configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by
 *        sim_add_stimulus_traveling_wave_packet_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult
sim_stimulus_traveling_wave_packet_config(struct SimContext *context, size_t operator_index,
                                          SimStimulusTravelingWavePacketConfig *out_config);

/**
 * @brief Replace or renormalize a registered traveling wave-packet configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes symbolic state and invalidates the
 * scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the traveling wave-packet operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup or state
 *         validation fails.
 */
SimResult
sim_stimulus_traveling_wave_packet_update(struct SimContext *context, size_t operator_index,
                                          const SimStimulusTravelingWavePacketConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_TRAVELING_WAVE_PACKET_H */
