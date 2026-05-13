/**
 * @file sim_field_stats_runtime.h
 * @brief Runtime field-statistics configuration, accumulation, and profiling helpers.
 * @ingroup oakfield_fields
 *
 * @details This header extends the public SimFieldStats snapshot with optional
 * feature masks, phase-coherence controls, generic field views, and runtime
 * profiling counters used by contexts, visual caches, and drift evaluation paths.
 */
#ifndef SIM_FIELD_STATS_RUNTIME_H
#define SIM_FIELD_STATS_RUNTIME_H

#include "field.h"
#include "sim_buffer.h"
#include "sim_field_stats.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Controls thresholded phase-coherence metrics and phase-lock detection.
 */
typedef struct SimPhaseCoherenceConfig {
    double abs_threshold;      /**< Absolute magnitude threshold for phase samples. */
    double rel_threshold;      /**< Relative threshold scaled by mean magnitude. */
    bool weighted;             /**< Use magnitude-weighted coherence for lock decisions. */
    double lock_on;            /**< EMA threshold that enters the locked state. */
    double lock_off;           /**< EMA threshold that exits the locked state. */
    double smoothing_constant; /**< EMA smoothing time constant in seconds. */
    bool deramp_enabled;       /**< Remove the dominant spectral bin before k0 coherence. */
} SimPhaseCoherenceConfig;

/**
 * @brief Optional field-statistics feature families.
 */
typedef enum SimFieldStatsFeature {
    SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY = 1u << 0, /**< Basic coherence and circularity. */
    SIM_FIELD_STATS_FEATURE_SPECTRAL = 1u << 1,      /**< FFT-derived entropy and bandwidth. */
    SIM_FIELD_STATS_FEATURE_PHASE_ADVANCED = 1u << 2 /**< Weighted, EMA, k0, and regime metrics. */
} SimFieldStatsFeature;

/** Empty optional field-statistics feature mask. */
#define SIM_FIELD_STATS_FEATURE_MASK_NONE 0u

/** Default feature mask used when callers omit an explicit compute config. */
#define SIM_FIELD_STATS_FEATURE_MASK_DEFAULT                                                       \
    (SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY | SIM_FIELD_STATS_FEATURE_SPECTRAL |                    \
     SIM_FIELD_STATS_FEATURE_PHASE_ADVANCED)

/**
 * @brief Feature selection for one field-statistics computation.
 */
typedef struct SimFieldStatsComputeConfig {
    uint32_t feature_mask; /**< Bitwise OR of SimFieldStatsFeature values. */
} SimFieldStatsComputeConfig;

/**
 * @brief Wall-clock timings captured for one statistics computation.
 */
typedef struct SimFieldStatsComputeTimings {
    uint64_t total_wall_ns;     /**< Total elapsed compute time in nanoseconds. */
    uint64_t reduction_wall_ns; /**< Reduction/statistical moment stage duration. */
    uint64_t spectral_wall_ns;  /**< Spectral feature stage duration. */
    uint64_t phase_wall_ns;     /**< Advanced phase feature stage duration. */
} SimFieldStatsComputeTimings;

/**
 * @brief Source path recorded for the most recent stats request or computation.
 */
typedef enum SimFieldStatsProfileSource {
    SIM_FIELD_STATS_PROFILE_SOURCE_NONE = 0,            /**< No source has been recorded. */
    SIM_FIELD_STATS_PROFILE_SOURCE_VIS_CACHE = 1,       /**< Visualization stats cache hit. */
    SIM_FIELD_STATS_PROFILE_SOURCE_VIS_DRIFT_CACHE = 2, /**< Visualization drift-cache hit. */
    SIM_FIELD_STATS_PROFILE_SOURCE_VIS_COMPUTE = 3,     /**< Visualization cache miss computed. */
    SIM_FIELD_STATS_PROFILE_SOURCE_DRIFT_COMPUTE = 4,   /**< Drift-evaluation stats computation. */
    SIM_FIELD_STATS_PROFILE_SOURCE_DIRECT_COMPUTE = 5   /**< Direct caller-requested computation. */
} SimFieldStatsProfileSource;

/**
 * @brief Rolling counters for runtime field-statistics requests and costs.
 */
typedef struct SimFieldStatsRuntimeProfile {
    SimFieldStatsComputeConfig config; /**< Stats configuration being profiled. */
    uint64_t request_count;            /**< Total stats requests observed. */
    uint64_t cache_hit_count;          /**< Visualization cache hits. */
    uint64_t drift_cache_hit_count;    /**< Visualization drift-cache hits. */
    uint64_t cache_miss_count;         /**< Visualization requests that required compute. */
    uint64_t compute_count;            /**< Total computations recorded. */
    uint64_t drift_compute_count;      /**< Computations performed during drift evaluation. */
    uint64_t direct_compute_count;     /**< Direct caller-requested computations. */
    uint64_t phase_lock_count;         /**< Number of phase-lock updates recorded. */
    uint64_t phase_lock_total_ns;      /**< Accumulated phase-lock update time. */
    uint64_t phase_lock_last_ns;       /**< Most recent phase-lock update time. */
    SimFieldStatsComputeTimings total_compute; /**< Accumulated compute-stage timings. */
    SimFieldStatsComputeTimings last_compute;  /**< Most recent compute-stage timings. */
    size_t last_field_index;                /**< Field index from the most recent profile event. */
    size_t last_step_index;                 /**< Step index from the most recent profile event. */
    SimFieldStatsProfileSource last_source; /**< Source from the most recent profile event. */
} SimFieldStatsRuntimeProfile;

/**
 * @brief Legacy field-data type alias backed by SimBufferDataType.
 */
typedef SimBufferDataType SimFieldDataType;

/**
 * @brief Legacy field-data type constants mapped to buffer storage constants.
 */
enum {
    SIM_FIELD_DOUBLE = SIM_BUFFER_DOUBLE,                 /**< Double-precision real values. */
    SIM_FIELD_COMPLEX_DOUBLE = SIM_BUFFER_COMPLEX_DOUBLE, /**< Interleaved complex doubles. */
    SIM_FIELD_UNKNOWN = SIM_BUFFER_UNKNOWN,               /**< Unknown or unsupported storage. */
    SIM_FIELD_I32 = SIM_BUFFER_I32,                       /**< Signed 32-bit integer values. */
    SIM_FIELD_I64 = SIM_BUFFER_I64,                       /**< Signed 64-bit integer values. */
    SIM_FIELD_U32 = SIM_BUFFER_U32,                       /**< Unsigned 32-bit integer values. */
    SIM_FIELD_U64 = SIM_BUFFER_U64,                       /**< Unsigned 64-bit integer values. */
    SIM_FIELD_I8 = SIM_BUFFER_I8,                         /**< Signed 8-bit integer values. */
    SIM_FIELD_U8 = SIM_BUFFER_U8                          /**< Unsigned 8-bit integer values. */
};

/**
 * @brief Legacy field view alias backed by SimBufferView.
 */
typedef SimBufferView SimFieldView;

/**
 * @brief Streaming accumulator for incrementally building SimFieldStats.
 */
typedef struct SimFieldStatsAccumulator {
    double mean_re, M2_re;   /**< Welford mean and M2 accumulator for Re(D). */
    double mean_im, M2_im;   /**< Welford mean and M2 accumulator for Im(D). */
    double mean_abs, M2_abs; /**< Welford mean and M2 accumulator for |D|. */

    double sum_mag_sq;                     /**< Sum of |D|^2 for RMS magnitude. */
    double max_abs;                        /**< Maximum observed magnitude. */
    double mean_square_re, mean_square_im; /**< Running E[D^2] components. */
    double phase_mean_re, phase_mean_im;   /**< Running E[D/|D|] components. */
    size_t phase_sample_count;             /**< Samples with nonzero magnitude for phase stats. */
    size_t sample_count;                   /**< Number of accumulated samples. */
    uint32_t feature_mask;                 /**< Normalized feature mask used by this accumulator. */
    SimFieldStats *stats;                  /**< Destination stats snapshot finalized by finish(). */
} SimFieldStatsAccumulator;

/**
 * @brief Clamp and expand a stats feature mask to supported dependencies.
 *
 * @param feature_mask Caller-supplied feature mask.
 * @return Supported mask; phase-advanced implies phase-summary and spectral.
 */
uint32_t sim_field_stats_normalize_feature_mask(uint32_t feature_mask);

/**
 * @brief Return the default stats compute configuration.
 *
 * @return Config with SIM_FIELD_STATS_FEATURE_MASK_DEFAULT enabled.
 */
SimFieldStatsComputeConfig sim_field_stats_default_compute_config(void);

/**
 * @brief Test whether @p feature is enabled in a compute config.
 *
 * @param config Optional config; NULL selects the default config.
 * @param feature Feature bit to test.
 * @return true when the normalized mask contains @p feature.
 */
bool sim_field_stats_feature_enabled(const SimFieldStatsComputeConfig *config, uint32_t feature);

/**
 * @brief Build a generic field view over a SimField.
 *
 * @param field Field to view.
 * @return Buffer view over the field storage, or a zeroed invalid view for NULL.
 */
SimFieldView sim_field_view_from_field(SimField *field);

/**
 * @brief Initialize a streaming accumulator with an explicit compute config.
 *
 * @param[out] acc Accumulator to initialize; NULL is ignored.
 * @param[out] stats Destination stats snapshot finalized by finish(); may be NULL.
 * @param config Optional feature config; NULL selects the default config.
 */
void sim_field_stats_accumulator_begin_with_config(SimFieldStatsAccumulator *acc,
                                                   SimFieldStats *stats,
                                                   const SimFieldStatsComputeConfig *config);

/**
 * @brief Initialize a streaming accumulator with the default compute config.
 *
 * @param[out] acc Accumulator to initialize; NULL is ignored.
 * @param[out] stats Destination stats snapshot finalized by finish(); may be NULL.
 */
void sim_field_stats_accumulator_begin(SimFieldStatsAccumulator *acc, SimFieldStats *stats);

/**
 * @brief Accumulate one real sample.
 *
 * @param acc Accumulator to update.
 * @param value Real sample value.
 */
void sim_field_stats_accumulate_real(SimFieldStatsAccumulator *acc, double value);

/**
 * @brief Accumulate one complex sample.
 *
 * @param acc Accumulator to update.
 * @param real Real component.
 * @param imag Imaginary component.
 */
void sim_field_stats_accumulate_complex(SimFieldStatsAccumulator *acc, double real, double imag);

/**
 * @brief Finalize accumulated statistics into the destination snapshot.
 *
 * @param acc Accumulator created by sim_field_stats_accumulator_begin().
 */
void sim_field_stats_accumulator_finish(SimFieldStatsAccumulator *acc);

/**
 * @brief Compute field statistics with optional feature selection and timings.
 *
 * @param field Field to inspect.
 * @param[out] out Destination stats snapshot.
 * @param config Optional feature config; NULL selects the default config.
 * @param[out] timings Optional receiver for compute-stage timings.
 */
void sim_field_stats_compute_with_config(const struct SimField *field, SimFieldStats *out,
                                         const SimFieldStatsComputeConfig *config,
                                         SimFieldStatsComputeTimings *timings);

/**
 * @brief Compute FFT-derived spectral metrics for a generic field view.
 *
 * @param view View to inspect.
 * @param[in,out] stats Stats snapshot receiving spectral fields.
 * @param[out] out_dominant_k Optional dominant spectral bin receiver.
 * @return true when spectral metrics were computed.
 */
bool sim_field_stats_compute_spectral_view(const SimFieldView *view, SimFieldStats *stats,
                                           size_t *out_dominant_k);

/**
 * @brief Compute thresholded, weighted, and deramped phase metrics for a view.
 *
 * @param view View to inspect.
 * @param[in,out] stats Stats snapshot receiving phase fields.
 * @param config Optional phase config; NULL uses the global phase config.
 * @param dominant_k Dominant spectral bin used for deramping.
 */
void sim_field_stats_compute_phase_metrics(const SimFieldView *view, SimFieldStats *stats,
                                           const SimPhaseCoherenceConfig *config,
                                           size_t dominant_k);

/**
 * @brief Update per-field EMA phase lock state and write it into stats.
 *
 * @param context Simulation context containing phase-lock runtime arrays.
 * @param field_index Field index to update.
 * @param[in,out] stats Stats snapshot receiving EMA and lock state.
 */
void sim_field_stats_update_phase_lock(struct SimContext *context, size_t field_index,
                                       SimFieldStats *stats);

/**
 * @brief Replace the global phase-coherence configuration.
 *
 * Invalid negative thresholds are clamped and lock thresholds are ordered.
 *
 * @param config Configuration to copy; NULL is ignored.
 */
void sim_field_stats_set_phase_config(const SimPhaseCoherenceConfig *config);

/**
 * @brief Return the global phase-coherence configuration.
 *
 * @return Pointer to internal read-only configuration storage.
 */
const SimPhaseCoherenceConfig *sim_field_stats_get_phase_config(void);

/**
 * @brief Update global absolute and relative phase sample thresholds.
 *
 * @param abs_threshold Non-negative absolute threshold; negative values leave it unchanged.
 * @param rel_threshold Non-negative relative threshold; negative values leave it unchanged.
 */
void sim_field_stats_set_phase_thresholds(double abs_threshold, double rel_threshold);

/**
 * @brief Select whether phase-lock decisions use weighted coherence.
 *
 * @param weighted true to use magnitude-weighted coherence.
 */
void sim_field_stats_set_phase_weighted(bool weighted);

/**
 * @brief Update global phase-lock hysteresis thresholds.
 *
 * @param on_threshold Threshold that enters the locked state.
 * @param off_threshold Threshold that exits the locked state.
 */
void sim_field_stats_set_phase_lock_thresholds(double on_threshold, double off_threshold);

/**
 * @brief Update the global phase-lock EMA smoothing time constant.
 *
 * @param smoothing_seconds Non-negative smoothing constant in seconds.
 */
void sim_field_stats_set_phase_smoothing(double smoothing_seconds);

/**
 * @brief Enable or disable dominant-bin deramping for advanced phase metrics.
 *
 * @param enabled true to compute k0 coherence after deramping.
 */
void sim_field_stats_set_phase_deramp(bool enabled);

/**
 * @brief Clear topology-related fields in a stats snapshot.
 *
 * @param stats Stats snapshot to update; NULL is ignored.
 */
void sim_field_stats_reset_topology_fields(SimFieldStats *stats);

/**
 * @brief Reset runtime stats profiling counters.
 *
 * @param[out] profile Profile to reset; NULL is ignored.
 * @param config Optional compute config to store in the profile.
 */
void sim_field_stats_profile_reset(SimFieldStatsRuntimeProfile *profile,
                                   const SimFieldStatsComputeConfig *config);

/**
 * @brief Record a field-statistics request and its cache/source path.
 *
 * @param profile Profile to update; NULL is ignored.
 * @param field_index Field index associated with the event.
 * @param step_index Step index associated with the event.
 * @param source Request source to record.
 */
void sim_field_stats_profile_record_request(SimFieldStatsRuntimeProfile *profile,
                                            size_t field_index, size_t step_index,
                                            SimFieldStatsProfileSource source);

/**
 * @brief Record a field-statistics computation and optional timings.
 *
 * @param profile Profile to update; NULL is ignored.
 * @param field_index Field index associated with the event.
 * @param step_index Step index associated with the event.
 * @param source Compute source to record.
 * @param timings Optional timings to accumulate.
 */
void sim_field_stats_profile_record_compute(SimFieldStatsRuntimeProfile *profile,
                                            size_t field_index, size_t step_index,
                                            SimFieldStatsProfileSource source,
                                            const SimFieldStatsComputeTimings *timings);

/**
 * @brief Record phase-lock update duration.
 *
 * @param profile Profile to update; NULL is ignored.
 * @param phase_lock_ns Duration in nanoseconds.
 */
void sim_field_stats_profile_record_phase_lock(SimFieldStatsRuntimeProfile *profile,
                                               uint64_t phase_lock_ns);

#ifdef __cplusplus
}
#endif
#endif
