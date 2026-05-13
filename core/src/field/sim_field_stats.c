#include "oakfield/sim_field_stats.h"
#include "oakfield/field.h"
#include <math.h>
#include <stddef.h>
#include <time.h>
#include "oakfield/sim_field_stats_runtime.h"

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

static double sim_field_stats_safe_var(double mean_square, double mean) {
    double var = mean_square - mean * mean;
    return (var < 0.0 && var > -1e-15) ? 0.0 : var;
}

static uint64_t sim_field_stats_now_ns(void) {
#if defined(__APPLE__)
    static mach_timebase_info_data_t info  = { 0, 0 };
    uint64_t                         ticks = mach_absolute_time();
    if (info.denom == 0U) {
        mach_timebase_info(&info);
    }
    return ticks * (uint64_t) info.numer / (uint64_t) info.denom;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0U;
    }
    return ((uint64_t) ts.tv_sec * 1000000000ULL) + (uint64_t) ts.tv_nsec;
#endif
}

static void sim_field_stats_reduce_real(const double*                    src,
                                        size_t                           count,
                                        SimFieldStats*                   stats,
                                        const SimFieldStatsComputeConfig* config) {
    bool   phase_summary_enabled = sim_field_stats_feature_enabled(
        config, SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY);
    double sum_re        = 0.0;
    double sum_abs       = 0.0;
    double sum_mag_sq    = 0.0;
    double phase_sum_re  = 0.0;
    size_t phase_samples = 0U;
    double max_abs       = 0.0;

    size_t i     = 0U;
    size_t limit = count & ~((size_t) 3U);
    for (; i < limit; i += 4U) {
        double v0 = src[i];
        double v1 = src[i + 1U];
        double v2 = src[i + 2U];
        double v3 = src[i + 3U];

        double a0 = fabs(v0);
        double a1 = fabs(v1);
        double a2 = fabs(v2);
        double a3 = fabs(v3);

        sum_re += v0 + v1 + v2 + v3;
        sum_abs += a0 + a1 + a2 + a3;
        sum_mag_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;

        if (phase_summary_enabled) {
            if (a0 > 0.0) {
                phase_sum_re += copysign(1.0, v0);
                phase_samples += 1U;
            }
            if (a1 > 0.0) {
                phase_sum_re += copysign(1.0, v1);
                phase_samples += 1U;
            }
            if (a2 > 0.0) {
                phase_sum_re += copysign(1.0, v2);
                phase_samples += 1U;
            }
            if (a3 > 0.0) {
                phase_sum_re += copysign(1.0, v3);
                phase_samples += 1U;
            }
        }

        double block_max = fmax(fmax(a0, a1), fmax(a2, a3));
        if (block_max > max_abs)
            max_abs = block_max;
    }

    for (; i < count; ++i) {
        double v = src[i];
        double a = fabs(v);
        sum_re += v;
        sum_abs += a;
        sum_mag_sq += v * v;
        if (phase_summary_enabled && a > 0.0) {
            phase_sum_re += copysign(1.0, v);
            phase_samples += 1U;
        }
        if (a > max_abs)
            max_abs = a;
    }

    double inv_n           = 1.0 / (double) count;
    double mean_re         = sum_re * inv_n;
    double mean_abs        = sum_abs * inv_n;
    double mean_mag_sq     = sum_mag_sq * inv_n;
    double mean_square_mag = mean_mag_sq;

    stats->count    = count;
    stats->mean_re  = mean_re;
    stats->mean_im  = 0.0;
    stats->mean     = mean_re;
    stats->mean_abs = mean_abs;
    stats->rms      = sqrt(mean_mag_sq);
    stats->var_re   = sim_field_stats_safe_var(mean_mag_sq, mean_re);
    stats->var_im   = 0.0;
    stats->var_abs  = sim_field_stats_safe_var(mean_mag_sq, mean_abs);
    stats->max_abs  = max_abs;
    if (phase_summary_enabled) {
        stats->phase_coherence =
            (phase_samples > 0U) ? fabs(phase_sum_re) / (double) phase_samples : 0.0;
        stats->circularity = (mean_mag_sq > 0.0) ? (mean_square_mag / mean_mag_sq) : 0.0;
        if (stats->circularity > 1.0 && stats->circularity < 1.0 + 1e-12)
            stats->circularity = 1.0;
        stats->phase_sample_count = phase_samples;
    } else {
        stats->phase_coherence  = 0.0;
        stats->circularity      = 0.0;
        stats->phase_sample_count = 0U;
    }
    stats->phase_coherence_weighted = 0.0;
    stats->phase_coherence_ema      = 0.0;
    stats->phase_coherence_k0       = 0.0;
    stats->phase_lock_state         = 0U;
    stats->phase_regime             = 0U;
    stats->continuity_dirty_ops     = 0ULL;
    stats->continuity_stable_ops    = 0ULL;
    sim_field_stats_reset_topology_fields(stats);
}

static void sim_field_stats_reduce_complex(const double*                    src,
                                           size_t                           count,
                                           SimFieldStats*                   stats,
                                           const SimFieldStatsComputeConfig* config) {
    bool   phase_summary_enabled = sim_field_stats_feature_enabled(
        config, SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY);
    double sum_re        = 0.0;
    double sum_im        = 0.0;
    double sum_abs       = 0.0;
    double sum_mag_sq    = 0.0;
    double sum_re_sq     = 0.0;
    double sum_im_sq     = 0.0;
    double sum_square_re = 0.0;
    double sum_square_im = 0.0;
    double phase_sum_re  = 0.0;
    double phase_sum_im  = 0.0;
    size_t phase_samples = 0U;
    double max_mag_sq    = 0.0;

    size_t i     = 0U;
    size_t limit = count & ~((size_t) 1U);
    for (; i < limit; i += 2U) {
        const double re0 = src[(i * 2U)];
        const double im0 = src[(i * 2U) + 1U];
        const double re1 = src[(i * 2U) + 2U];
        const double im1 = src[(i * 2U) + 3U];

        const double re0_sq = re0 * re0;
        const double im0_sq = im0 * im0;
        const double re1_sq = re1 * re1;
        const double im1_sq = im1 * im1;
        const double mag2_0 = re0_sq + im0_sq;
        const double mag2_1 = re1_sq + im1_sq;
        const double mag_0  = sqrt(mag2_0);
        const double mag_1  = sqrt(mag2_1);

        sum_re += re0 + re1;
        sum_im += im0 + im1;
        sum_abs += mag_0 + mag_1;
        sum_mag_sq += mag2_0 + mag2_1;
        sum_re_sq += re0_sq + re1_sq;
        sum_im_sq += im0_sq + im1_sq;

        if (phase_summary_enabled) {
            sum_square_re += (re0_sq - im0_sq) + (re1_sq - im1_sq);
            sum_square_im += (2.0 * re0 * im0) + (2.0 * re1 * im1);

            if (mag_0 > 0.0) {
                double inv0 = 1.0 / mag_0;
                phase_sum_re += re0 * inv0;
                phase_sum_im += im0 * inv0;
                phase_samples += 1U;
            }
            if (mag_1 > 0.0) {
                double inv1 = 1.0 / mag_1;
                phase_sum_re += re1 * inv1;
                phase_sum_im += im1 * inv1;
                phase_samples += 1U;
            }
        }

        if (mag2_0 > max_mag_sq)
            max_mag_sq = mag2_0;
        if (mag2_1 > max_mag_sq)
            max_mag_sq = mag2_1;
    }

    for (; i < count; ++i) {
        double re    = src[i * 2U];
        double im    = src[i * 2U + 1U];
        double re_sq = re * re;
        double im_sq = im * im;
        double mag2  = re_sq + im_sq;
        double mag   = sqrt(mag2);

        sum_re += re;
        sum_im += im;
        sum_abs += mag;
        sum_mag_sq += mag2;
        sum_re_sq += re_sq;
        sum_im_sq += im_sq;
        if (phase_summary_enabled) {
            sum_square_re += re_sq - im_sq;
            sum_square_im += 2.0 * re * im;

            if (mag > 0.0) {
                double inv = 1.0 / mag;
                phase_sum_re += re * inv;
                phase_sum_im += im * inv;
                phase_samples += 1U;
            }
        }

        if (mag2 > max_mag_sq)
            max_mag_sq = mag2;
    }

    double inv_n          = 1.0 / (double) count;
    double mean_re        = sum_re * inv_n;
    double mean_im        = sum_im * inv_n;
    double mean_abs       = sum_abs * inv_n;
    double mean_mag_sq    = sum_mag_sq * inv_n;
    double mean_square_re = sum_square_re * inv_n;
    double mean_square_im = sum_square_im * inv_n;
    double mean_square_mag =
        sqrt(mean_square_re * mean_square_re + mean_square_im * mean_square_im);

    stats->count    = count;
    stats->mean_re  = mean_re;
    stats->mean_im  = mean_im;
    stats->mean     = mean_re;
    stats->mean_abs = mean_abs;
    stats->rms      = sqrt(mean_mag_sq);
    stats->var_re   = sim_field_stats_safe_var(sum_re_sq * inv_n, mean_re);
    stats->var_im   = sim_field_stats_safe_var(sum_im_sq * inv_n, mean_im);
    stats->var_abs  = sim_field_stats_safe_var(mean_mag_sq, mean_abs);
    stats->max_abs  = sqrt(max_mag_sq);

    if (phase_summary_enabled) {
        double coh_mag = sqrt(phase_sum_re * phase_sum_re + phase_sum_im * phase_sum_im);
        stats->phase_coherence = (phase_samples > 0U) ? (coh_mag / (double) phase_samples) : 0.0;
        if (stats->phase_coherence > 1.0 && stats->phase_coherence < 1.0 + 1e-12)
            stats->phase_coherence = 1.0;

        stats->circularity = (mean_mag_sq > 0.0) ? (mean_square_mag / mean_mag_sq) : 0.0;
        if (stats->circularity > 1.0 && stats->circularity < 1.0 + 1e-12)
            stats->circularity = 1.0;
        stats->phase_sample_count = phase_samples;
    } else {
        stats->phase_coherence  = 0.0;
        stats->circularity      = 0.0;
        stats->phase_sample_count = 0U;
    }
    stats->phase_coherence_weighted = 0.0;
    stats->phase_coherence_ema      = 0.0;
    stats->phase_coherence_k0       = 0.0;
    stats->phase_lock_state         = 0U;
    stats->phase_regime             = 0U;
    stats->continuity_dirty_ops     = 0ULL;
    stats->continuity_stable_ops    = 0ULL;
    sim_field_stats_reset_topology_fields(stats);
}

static bool sim_field_stats_reduce_generic_view(const SimFieldView*            view,
                                                SimFieldStats*                 stats,
                                                const SimFieldStatsComputeConfig* config,
                                                bool                           is_complex) {
    SimFieldStatsAccumulator acc;

    if (view == NULL || stats == NULL || view->data == NULL || view->count == 0U) {
        return false;
    }

    sim_field_stats_accumulator_begin_with_config(&acc, stats, config);
    for (size_t i = 0U; i < view->count; ++i) {
        SimComplexDouble value = { 0 };
        if (!sim_buffer_view_get_complex(view, i, &value)) {
            return false;
        }
        if (is_complex) {
            sim_field_stats_accumulate_complex(&acc, value.re, value.im);
        } else {
            sim_field_stats_accumulate_real(&acc, value.re);
        }
    }
    sim_field_stats_accumulator_finish(&acc);
    return true;
}

void sim_field_stats_compute(const SimField* field, SimFieldStats* out) {
    sim_field_stats_compute_with_config(field, out, NULL, NULL);
}

void sim_field_stats_compute_with_config(const SimField*                 field,
                                         SimFieldStats*                  out,
                                         const SimFieldStatsComputeConfig* config,
                                         SimFieldStatsComputeTimings*    timings) {
    SimFieldStats stats = { 0 };
    SimFieldStatsComputeConfig effective =
        (config != NULL) ? *config : sim_field_stats_default_compute_config();
    uint64_t total_start_ns     = 0U;
    uint64_t stage_start_ns     = 0U;

    if (!field || !out)
        return;

    effective.feature_mask = sim_field_stats_normalize_feature_mask(effective.feature_mask);
    if (timings != NULL) {
        *timings = (SimFieldStatsComputeTimings) { 0 };
        total_start_ns = sim_field_stats_now_ns();
        stage_start_ns = total_start_ns;
    }

    SimFieldView view = sim_field_view_from_field((SimField*) field);
    if (!view.data || view.count == 0U) {
        *out = stats;
        return;
    }

    switch (view.type) {
        case SIM_FIELD_DOUBLE:
            sim_field_stats_reduce_real((const double*) view.data, view.count, &stats, &effective);
            break;
        case SIM_FIELD_COMPLEX_DOUBLE:
            sim_field_stats_reduce_complex(
                (const double*) view.data, view.count, &stats, &effective);
            break;
        case SIM_FIELD_I8:
        case SIM_FIELD_U8:
        case SIM_FIELD_I32:
        case SIM_FIELD_U32:
        case SIM_FIELD_I64:
        case SIM_FIELD_U64:
            if (!sim_field_stats_reduce_generic_view(&view, &stats, &effective, false)) {
                *out = stats;
                return;
            }
            break;
        default:
            *out = stats;
            return;
    }
    if (timings != NULL) {
        uint64_t now_ns = sim_field_stats_now_ns();
        if (now_ns >= stage_start_ns) {
            timings->reduction_wall_ns = now_ns - stage_start_ns;
        }
        stage_start_ns = now_ns;
    }

    if (sim_field_stats_feature_enabled(&effective, SIM_FIELD_STATS_FEATURE_SPECTRAL)) {
        size_t dominant_k = 0U;
        sim_field_stats_compute_spectral_view(&view, &stats, &dominant_k);
        if (timings != NULL) {
            uint64_t now_ns = sim_field_stats_now_ns();
            if (now_ns >= stage_start_ns) {
                timings->spectral_wall_ns = now_ns - stage_start_ns;
            }
            stage_start_ns = now_ns;
        }
        if (sim_field_stats_feature_enabled(&effective, SIM_FIELD_STATS_FEATURE_PHASE_ADVANCED)) {
            sim_field_stats_compute_phase_metrics(&view, &stats, NULL, dominant_k);
        }
    }
    if (!sim_field_stats_feature_enabled(&effective, SIM_FIELD_STATS_FEATURE_SPECTRAL)) {
        stats.spectral_entropy   = 0.0;
        stats.spectral_bandwidth = 0.0;
    }
    if (!sim_field_stats_feature_enabled(&effective, SIM_FIELD_STATS_FEATURE_PHASE_ADVANCED)) {
        stats.phase_coherence_weighted = 0.0;
        stats.phase_coherence_ema      = 0.0;
        stats.phase_coherence_k0       = 0.0;
        stats.phase_lock_state         = 0U;
        stats.phase_regime             = 0U;
    }
    if (timings != NULL) {
        uint64_t end_ns = sim_field_stats_now_ns();
        if (end_ns >= stage_start_ns) {
            timings->phase_wall_ns = end_ns - stage_start_ns;
        }
        if (end_ns >= total_start_ns) {
            timings->total_wall_ns = end_ns - total_start_ns;
        }
    }

    *out = stats;
}
