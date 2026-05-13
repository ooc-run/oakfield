#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimField;

/**
 * @brief Summary statistics for a scalar or complex field slice.
 */
typedef struct SimFieldStats {
    double mean_re;                         /**< Mean real component. */
    double mean_im;                         /**< Mean imaginary component. */
    double mean;                            /**< Legacy alias for @ref mean_re. */
    double mean_abs;                        /**< Mean magnitude. */
    double rms;                             /**< Root-mean-square magnitude. */
    double var_re;                          /**< Variance of the real component. */
    double var_im;                          /**< Variance of the imaginary component. */
    double var_abs;                         /**< Variance of magnitudes. */
    double max_abs;                         /**< Maximum magnitude. */
    double phase_coherence;                 /**< Unweighted phase coherence in [0, 1]. */
    double circularity;                     /**< Circularity metric for the complex sample cloud. */
    double spectral_entropy;                /**< Normalized spectral entropy. */
    double spectral_bandwidth;              /**< Spectral bandwidth estimate. */
    double phase_coherence_weighted;        /**< Magnitude-weighted phase coherence. */
    double phase_coherence_ema;             /**< Exponential moving average of phase coherence. */
    double phase_coherence_k0;              /**< Phase coherence after dominant-bin deramping. */
    size_t phase_sample_count;              /**< Number of samples used for phase metrics. */
    uint8_t phase_lock_state;               /**< Hysteresis lock state derived from coherence. */
    uint8_t phase_regime;                   /**< Coarse phase-regime classification. */
    uint8_t topology_valid;                 /**< True when topology fields contain valid counts. */
    size_t count;                           /**< Number of field samples included. */
    size_t topology_positive_singularities; /**< Count of positive phase singularities. */
    size_t topology_negative_singularities; /**< Count of negative phase singularities. */
    size_t topology_seam_edge_count;        /**< Count of detected topology seam edges. */
    size_t topology_ambiguous_cell_count;   /**< Count of ambiguous topology cells. */
    uint64_t
        continuity_dirty_ops; /**< Operators that wrote without continuity guards (per step). */
    uint64_t continuity_stable_ops; /**< Operators that wrote with clamped/limited continuity. */
} SimFieldStats;

void sim_field_stats_compute(const struct SimField *field, SimFieldStats *out);

#ifdef __cplusplus
}
#endif
