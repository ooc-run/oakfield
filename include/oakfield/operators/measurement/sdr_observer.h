/**
 * @file sdr_observer.h
 * @brief Experimental SDR observation operator with RTL-SDR and synthetic fallback status.
 *
 * @warning Experimental API. Backend ownership, blocking/timing behavior,
 * demodulation placement, status fields, and fallback semantics may change
 * while the SDR stack moves toward production-grade capture.
 */
#ifndef OAKFIELD_SDR_OBSERVER_H
#define OAKFIELD_SDR_OBSERVER_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Demodulation mode applied to observed SDR IQ samples.
 */
typedef enum SimSdrObserverDemod {
    SIM_SDR_OBSERVER_DEMOD_RAW = 0, /**< Raw IQ passthrough into complex field. */
    SIM_SDR_OBSERVER_DEMOD_AM,      /**< Amplitude envelope demodulation. */
    SIM_SDR_OBSERVER_DEMOD_FM,      /**< Instantaneous frequency deviation. */
    SIM_SDR_OBSERVER_DEMOD_PM,      /**< Instantaneous phase. */
} SimSdrObserverDemod;

/**
 * @brief Active SDR sample backend.
 */
typedef enum SimSdrObserverBackendMode {
    SIM_SDR_OBSERVER_BACKEND_UNKNOWN = 0, /**< Backend has not been selected. */
    SIM_SDR_OBSERVER_BACKEND_SYNTHETIC,   /**< Deterministic synthetic fallback source. */
    SIM_SDR_OBSERVER_BACKEND_RTL_SDR,     /**< RTL-SDR hardware source. */
} SimSdrObserverBackendMode;

/**
 * @brief Reason the observer most recently fell back from hardware input.
 */
typedef enum SimSdrObserverFallbackReason {
    SIM_SDR_OBSERVER_FALLBACK_NONE = 0,        /**< Hardware fallback has not occurred. */
    SIM_SDR_OBSERVER_FALLBACK_RTLSDR_DISABLED, /**< Build or runtime configuration disabled RTL-SDR.
                                                */
    SIM_SDR_OBSERVER_FALLBACK_DEVICE_OPEN_FAILED,       /**< Device open failed. */
    SIM_SDR_OBSERVER_FALLBACK_DEVICE_CONFIG_FAILED,     /**< Device configuration failed. */
    SIM_SDR_OBSERVER_FALLBACK_BUFFER_ALLOCATION_FAILED, /**< IQ read buffer allocation failed. */
    SIM_SDR_OBSERVER_FALLBACK_READ_FAILED,              /**< Hardware read failed. */
} SimSdrObserverFallbackReason;

#define SIM_SDR_OBSERVER_STATUS_MESSAGE_MAX 159U

/**
 * @brief Configuration for an SDR observer operator.
 */
typedef struct SimSdrObserverConfig {
    size_t field_index;        /**< Target complex-double field index. */
    double center_freq;        /**< Center frequency in Hz. */
    double sample_rate;        /**< Sample rate in Hz. */
    double gain;               /**< Tuner gain in dB; 0 = auto. */
    SimSdrObserverDemod demod; /**< Demodulation mode. */
    double freq_offset;        /**< Fine frequency offset in Hz. */
    double bandwidth;          /**< IF bandwidth in Hz; 0 = use sample_rate. */
    double amplitude;          /**< Output amplitude scale. */
    bool normalize;            /**< Normalize output to unit amplitude per step. */
    bool accumulate;           /**< Accumulate into field instead of overwriting. */
    bool scale_by_dt;          /**< Multiply output by dt. */
    int device_index;          /**< RTL-SDR device index. */
} SimSdrObserverConfig;

/**
 * @brief Runtime status snapshot for an SDR observer operator.
 */
typedef struct SimSdrObserverStatus {
    SimSdrObserverBackendMode active_backend;          /**< Current data source. */
    SimSdrObserverFallbackReason last_fallback_reason; /**< Sticky last fallback cause. */
    bool rtl_sdr_enabled;                              /**< Build includes RTL-SDR support. */
    bool device_open;                                  /**< Device handle currently open. */
    bool using_synthetic;                              /**< Current step source is synthetic. */
    bool has_successful_read;                          /**< At least one hardware read succeeded. */
    bool has_last_error;                               /**< Sticky last-error flag. */
    double effective_tuned_freq;  /**< Derived tuned/display frequency in Hz. */
    size_t successful_read_count; /**< Successful hardware-read count. */
    size_t fallback_count;        /**< Number of fallback events. */
    size_t last_read_iq_bytes;    /**< Most recent successful IQ byte count. */
    SimResult last_error_code;    /**< Sticky libsimcore-facing error code. */
    int last_backend_error;       /**< Sticky raw backend error code. */
    char last_error_message[SIM_SDR_OBSERVER_STATUS_MESSAGE_MAX +
                            1U]; /**< Sticky backend/error detail. */
} SimSdrObserverStatus;

/**
 * @brief Return the schema name for an SDR backend mode.
 *
 * @param mode Backend mode enum value.
 * @return Stable lowercase backend mode name.
 */
const char *sim_sdr_observer_backend_mode_name(SimSdrObserverBackendMode mode);

/**
 * @brief Return the schema name for an SDR fallback reason.
 *
 * @param reason Fallback reason enum value.
 * @return Stable lowercase reason name.
 */
const char *sim_sdr_observer_fallback_reason_name(SimSdrObserverFallbackReason reason);

/**
 * @brief Register an experimental SDR observer operator.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional SDR observer configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field validation, allocation, backend setup, or split registration.
 */
SimResult sim_add_sdr_observer_operator(struct SimContext *context,
                                        const SimSdrObserverConfig *config, size_t *out_index);

/**
 * @brief Copy the current SDR observer configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_sdr_observer_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no observer state.
 */
SimResult sim_sdr_observer_config(struct SimContext *context, size_t operator_index,
                                  SimSdrObserverConfig *out_config);

/**
 * @brief Copy runtime status from a registered experimental SDR observer.
 *
 * Status includes active backend, fallback reason, read counters, and sticky
 * backend error details.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_sdr_observer_operator().
 * @param[out] out_status Receives the current observer status snapshot.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no observer state.
 */
SimResult sim_sdr_observer_status(struct SimContext *context, size_t operator_index,
                                  SimSdrObserverStatus *out_status);

/**
 * @brief Replace the configuration of a registered experimental SDR observer.
 *
 * @p config is required. A successful update refreshes backend state and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the SDR observer to update.
 * @param config Replacement SDR observer configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, field validation, backend setup, or state validation.
 */
SimResult sim_sdr_observer_update(struct SimContext *context, size_t operator_index,
                                  const SimSdrObserverConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SDR_OBSERVER_H */
