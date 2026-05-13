#include "oakfield/sim_field_stats_runtime.h"
#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operators/common/fft_plan.h"
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SimFieldView sim_field_view_from_field(SimField* field) {
    SimFieldView view = { 0 };

    if (field == NULL)
        return view;

    view.data   = sim_field_data(field);
    view.count  = sim_field_element_count(&field->layout);
    view.type   = sim_buffer_data_type_from_scalar_domain(sim_field_scalar_domain(field));
    view.layout = field->layout;

    return view;
}

typedef struct SimFieldSpectralScratch {
    FFTPlan         plan;
    double complex* time;
    double complex* freq;
    size_t          capacity;
    size_t          length;
    bool            plan_ready;
} SimFieldSpectralScratch;

static SimFieldSpectralScratch g_spectral_scratch = { 0 };
static SimPhaseCoherenceConfig g_phase_config     = { .abs_threshold      = 1.0e-12,
                                                      .rel_threshold      = 0.02,
                                                      .weighted           = true,
                                                      .lock_on            = 0.8,
                                                      .lock_off           = 0.6,
                                                      .smoothing_constant = 0.5,
                                                      .deramp_enabled     = true };

uint32_t sim_field_stats_normalize_feature_mask(uint32_t feature_mask) {
    feature_mask &= SIM_FIELD_STATS_FEATURE_MASK_DEFAULT;
    if ((feature_mask & SIM_FIELD_STATS_FEATURE_PHASE_ADVANCED) != 0U) {
        feature_mask |= SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY;
        feature_mask |= SIM_FIELD_STATS_FEATURE_SPECTRAL;
    }
    return feature_mask;
}

SimFieldStatsComputeConfig sim_field_stats_default_compute_config(void) {
    SimFieldStatsComputeConfig config = { 0 };
    config.feature_mask               = SIM_FIELD_STATS_FEATURE_MASK_DEFAULT;
    return config;
}

bool sim_field_stats_feature_enabled(const SimFieldStatsComputeConfig* config, uint32_t feature) {
    uint32_t feature_mask = SIM_FIELD_STATS_FEATURE_MASK_DEFAULT;

    if (config != NULL) {
        feature_mask = sim_field_stats_normalize_feature_mask(config->feature_mask);
    }
    return (feature_mask & feature) != 0U;
}

void sim_field_stats_profile_reset(SimFieldStatsRuntimeProfile*       profile,
                                   const SimFieldStatsComputeConfig* config) {
    if (profile == NULL) {
        return;
    }

    (void) memset(profile, 0, sizeof(*profile));
    profile->config = (config != NULL) ? *config : sim_field_stats_default_compute_config();
    profile->config.feature_mask = sim_field_stats_normalize_feature_mask(
        profile->config.feature_mask);
}

void sim_field_stats_profile_record_request(SimFieldStatsRuntimeProfile* profile,
                                            size_t                       field_index,
                                            size_t                       step_index,
                                            SimFieldStatsProfileSource   source) {
    if (profile == NULL) {
        return;
    }

    profile->request_count += 1U;
    profile->last_field_index = field_index;
    profile->last_step_index  = step_index;
    profile->last_source      = source;

    switch (source) {
        case SIM_FIELD_STATS_PROFILE_SOURCE_VIS_CACHE:
            profile->cache_hit_count += 1U;
            break;
        case SIM_FIELD_STATS_PROFILE_SOURCE_VIS_DRIFT_CACHE:
            profile->drift_cache_hit_count += 1U;
            break;
        case SIM_FIELD_STATS_PROFILE_SOURCE_VIS_COMPUTE:
            profile->cache_miss_count += 1U;
            break;
        default:
            break;
    }
}

void sim_field_stats_profile_record_compute(SimFieldStatsRuntimeProfile*        profile,
                                            size_t                              field_index,
                                            size_t                              step_index,
                                            SimFieldStatsProfileSource          source,
                                            const SimFieldStatsComputeTimings* timings) {
    if (profile == NULL) {
        return;
    }

    profile->compute_count += 1U;
    profile->last_field_index = field_index;
    profile->last_step_index  = step_index;
    profile->last_source      = source;

    if (timings != NULL) {
        profile->last_compute = *timings;
        profile->total_compute.total_wall_ns += timings->total_wall_ns;
        profile->total_compute.reduction_wall_ns += timings->reduction_wall_ns;
        profile->total_compute.spectral_wall_ns += timings->spectral_wall_ns;
        profile->total_compute.phase_wall_ns += timings->phase_wall_ns;
    } else {
        profile->last_compute = (SimFieldStatsComputeTimings) { 0 };
    }

    switch (source) {
        case SIM_FIELD_STATS_PROFILE_SOURCE_DRIFT_COMPUTE:
            profile->drift_compute_count += 1U;
            break;
        case SIM_FIELD_STATS_PROFILE_SOURCE_DIRECT_COMPUTE:
            profile->direct_compute_count += 1U;
            break;
        default:
            break;
    }
}

void sim_field_stats_profile_record_phase_lock(SimFieldStatsRuntimeProfile* profile,
                                               uint64_t                     phase_lock_ns) {
    if (profile == NULL) {
        return;
    }

    profile->phase_lock_count += 1U;
    profile->phase_lock_last_ns  = phase_lock_ns;
    profile->phase_lock_total_ns += phase_lock_ns;
}

void sim_field_stats_set_phase_config(const SimPhaseCoherenceConfig* config) {
    if (!config)
        return;
    g_phase_config = *config;
    if (g_phase_config.lock_on < g_phase_config.lock_off) {
        double tmp              = g_phase_config.lock_on;
        g_phase_config.lock_on  = g_phase_config.lock_off;
        g_phase_config.lock_off = tmp;
    }
    if (g_phase_config.abs_threshold < 0.0)
        g_phase_config.abs_threshold = 0.0;
    if (g_phase_config.rel_threshold < 0.0)
        g_phase_config.rel_threshold = 0.0;
    if (g_phase_config.smoothing_constant < 0.0)
        g_phase_config.smoothing_constant = 0.0;
}

const SimPhaseCoherenceConfig* sim_field_stats_get_phase_config(void) {
    return &g_phase_config;
}

void sim_field_stats_set_phase_thresholds(double abs_threshold, double rel_threshold) {
    SimPhaseCoherenceConfig cfg = g_phase_config;
    if (abs_threshold >= 0.0) {
        cfg.abs_threshold = abs_threshold;
    }
    if (rel_threshold >= 0.0) {
        cfg.rel_threshold = rel_threshold;
    }
    sim_field_stats_set_phase_config(&cfg);
}

void sim_field_stats_set_phase_weighted(bool weighted) {
    SimPhaseCoherenceConfig cfg = g_phase_config;
    cfg.weighted = weighted;
    sim_field_stats_set_phase_config(&cfg);
}

void sim_field_stats_set_phase_lock_thresholds(double on_threshold, double off_threshold) {
    SimPhaseCoherenceConfig cfg = g_phase_config;
    cfg.lock_on  = on_threshold;
    cfg.lock_off = off_threshold;
    sim_field_stats_set_phase_config(&cfg);
}

void sim_field_stats_set_phase_smoothing(double smoothing_seconds) {
    SimPhaseCoherenceConfig cfg = g_phase_config;
    cfg.smoothing_constant = smoothing_seconds;
    sim_field_stats_set_phase_config(&cfg);
}

void sim_field_stats_set_phase_deramp(bool enabled) {
    SimPhaseCoherenceConfig cfg = g_phase_config;
    cfg.deramp_enabled = enabled;
    sim_field_stats_set_phase_config(&cfg);
}

void sim_field_stats_reset_topology_fields(SimFieldStats* stats) {
    if (stats == NULL) {
        return;
    }

    stats->topology_valid                  = 0U;
    stats->topology_positive_singularities = 0U;
    stats->topology_negative_singularities = 0U;
    stats->topology_seam_edge_count        = 0U;
    stats->topology_ambiguous_cell_count   = 0U;
}

static void sim_field_stats_spectral_reset(SimFieldSpectralScratch* scratch) {
    if (!scratch)
        return;
    if (scratch->plan_ready) {
        fft_plan_destroy(&scratch->plan);
        scratch->plan_ready = false;
    }
    free(scratch->time);
    free(scratch->freq);
    scratch->time     = NULL;
    scratch->freq     = NULL;
    scratch->capacity = 0U;
    scratch->length   = 0U;
}

static bool sim_field_stats_spectral_ensure(SimFieldSpectralScratch* scratch, size_t length) {
    if (!scratch || length == 0U)
        return false;

    if (!scratch->plan_ready || scratch->length != length) {
        sim_field_stats_spectral_reset(scratch);
        if (fft_plan_init(&scratch->plan, length) != SIM_RESULT_OK) {
            return false;
        }
        scratch->plan_ready = true;
        scratch->length     = length;
    }

    if (scratch->capacity < length) {
        double complex* time =
            (double complex*) realloc(scratch->time, length * sizeof(double complex));
        double complex* freq =
            (double complex*) realloc(scratch->freq, length * sizeof(double complex));
        if (!time || !freq) {
            free(time);
            free(freq);
            sim_field_stats_spectral_reset(scratch);
            return false;
        }
        scratch->time     = time;
        scratch->freq     = freq;
        scratch->capacity = length;
    }

    return true;
}

static double sim_field_stats_frequency_bin(size_t index, size_t length) {
    ptrdiff_t signed_k =
        (index <= length / 2U) ? (ptrdiff_t) index : (ptrdiff_t) index - (ptrdiff_t) length;
    return (double) signed_k;
}

void sim_field_stats_accumulator_begin_with_config(SimFieldStatsAccumulator*         acc,
                                                   SimFieldStats*                    stats,
                                                   const SimFieldStatsComputeConfig* config) {
    SimFieldStatsComputeConfig effective = (config != NULL)
                                               ? *config
                                               : sim_field_stats_default_compute_config();

    if (!acc)
        return;

    effective.feature_mask = sim_field_stats_normalize_feature_mask(effective.feature_mask);

    /* Initialize Welford accumulators and sums. */
    acc->mean_re = acc->M2_re = 0.0;
    acc->mean_im = acc->M2_im = 0.0;
    acc->mean_abs = acc->M2_abs = 0.0;
    acc->sum_mag_sq             = 0.0;
    acc->max_abs                = 0.0;
    acc->mean_square_re = acc->mean_square_im = 0.0;
    acc->phase_mean_re = acc->phase_mean_im = 0.0;
    acc->phase_sample_count                 = 0U;
    acc->sample_count                       = 0;
    acc->feature_mask                       = effective.feature_mask;
    acc->stats                              = stats;

    if (stats) {
        stats->count   = 0;
        stats->mean_re = stats->mean_im = 0.0;
        stats->mean                     = 0.0; /* legacy alias */
        stats->mean_abs = stats->rms = 0.0;
        stats->var_re = stats->var_im = stats->var_abs = 0.0;
        stats->max_abs                                 = 0.0;
        stats->phase_coherence                         = 0.0;
        stats->circularity                             = 0.0;
        stats->spectral_entropy                        = 0.0;
        stats->spectral_bandwidth                      = 0.0;
        stats->continuity_dirty_ops                    = 0ULL;
        stats->continuity_stable_ops                   = 0ULL;
        sim_field_stats_reset_topology_fields(stats);
    }
}

void sim_field_stats_accumulator_begin(SimFieldStatsAccumulator* acc, SimFieldStats* stats) {
    sim_field_stats_accumulator_begin_with_config(acc, stats, NULL);
}

static void sim_field_stats_accumulate_sample(SimFieldStatsAccumulator* acc, double re, double im) {
    bool   phase_summary_enabled;
    if (!acc)
        return;

    phase_summary_enabled =
        (acc->feature_mask & SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY) != 0U;
    double mag2 = re * re + im * im;
    double mag  = (im == 0.0) ? fabs(re) : sqrt(mag2);

    size_t n1 = acc->sample_count; /* previous count */
    size_t n  = n1 + 1;            /* new count */

    /* Welford update for Re */
    double delta_re = re - acc->mean_re;
    acc->mean_re += delta_re / (double) n;
    double delta2_re = re - acc->mean_re;
    acc->M2_re += delta_re * delta2_re;

    /* Welford update for Im */
    double delta_im = im - acc->mean_im;
    acc->mean_im += delta_im / (double) n;
    double delta2_im = im - acc->mean_im;
    acc->M2_im += delta_im * delta2_im;

    /* Welford update for |D| */
    double delta_abs = mag - acc->mean_abs;
    acc->mean_abs += delta_abs / (double) n;
    double delta2_abs = mag - acc->mean_abs;
    acc->M2_abs += delta_abs * delta2_abs;

    acc->sum_mag_sq += mag2;
    if (mag > acc->max_abs)
        acc->max_abs = mag;

    if (phase_summary_enabled) {
        /* Circularity / anisotropy: E[D^2] components */
        double sq_re = re * re - im * im;
        double sq_im = 2.0 * re * im;
        acc->mean_square_re += (sq_re - acc->mean_square_re) / (double) n;
        acc->mean_square_im += (sq_im - acc->mean_square_im) / (double) n;

        /* Phase coherence: mean of unit-magnitude samples (skip zeros) */
        if (mag > 0.0) {
            double inv_mag = 1.0 / mag;
            double u_re    = re * inv_mag;
            double u_im    = im * inv_mag;
            size_t phase_n = acc->phase_sample_count + 1U;
            acc->phase_mean_re += (u_re - acc->phase_mean_re) / (double) phase_n;
            acc->phase_mean_im += (u_im - acc->phase_mean_im) / (double) phase_n;
            acc->phase_sample_count = phase_n;
        }
    }

    acc->sample_count = n;
}

void sim_field_stats_accumulate_real(SimFieldStatsAccumulator* acc, double value) {
    sim_field_stats_accumulate_sample(acc, value, 0.0);
}

void sim_field_stats_accumulate_complex(SimFieldStatsAccumulator* acc, double real, double imag) {
    sim_field_stats_accumulate_sample(acc, real, imag);
}

void sim_field_stats_accumulator_finish(SimFieldStatsAccumulator* acc) {
    if (!acc || !acc->stats)
        return;

    size_t n = acc->sample_count;

    if (n == 0) {
        acc->stats->count   = 0;
        acc->stats->mean_re = acc->stats->mean_im = 0.0;
        acc->stats->mean                          = 0.0;
        acc->stats->mean_abs = acc->stats->rms = 0.0;
        acc->stats->var_re = acc->stats->var_im = acc->stats->var_abs = 0.0;
        acc->stats->max_abs                                           = 0.0;
        acc->stats->phase_coherence                                   = 0.0;
        acc->stats->circularity                                       = 0.0;
        acc->stats->spectral_entropy                                  = 0.0;
        acc->stats->spectral_bandwidth                                = 0.0;
        acc->stats->phase_coherence_weighted                          = 0.0;
        acc->stats->phase_coherence_ema                               = 0.0;
        acc->stats->phase_coherence_k0                                = 0.0;
        acc->stats->phase_sample_count                                = 0U;
        acc->stats->phase_lock_state                                  = 0U;
        acc->stats->phase_regime                                      = 0U;
        acc->stats->continuity_dirty_ops                              = 0ULL;
        acc->stats->continuity_stable_ops                             = 0ULL;
        sim_field_stats_reset_topology_fields(acc->stats);
        return;
    }

    acc->stats->count    = n;
    acc->stats->mean_re  = acc->mean_re;
    acc->stats->mean_im  = acc->mean_im;
    acc->stats->mean     = acc->mean_re; /* legacy alias */
    acc->stats->mean_abs = acc->mean_abs;
    acc->stats->rms      = sqrt(acc->sum_mag_sq / (double) n);
    acc->stats->var_re   = acc->M2_re / (double) n; /* population variance */
    acc->stats->var_im   = acc->M2_im / (double) n;
    acc->stats->var_abs  = acc->M2_abs / (double) n;
    acc->stats->max_abs  = acc->max_abs;

    if ((acc->feature_mask & SIM_FIELD_STATS_FEATURE_PHASE_SUMMARY) != 0U) {
        double phase_mean_mag =
            sqrt(acc->phase_mean_re * acc->phase_mean_re + acc->phase_mean_im * acc->phase_mean_im);
        double coherence = (acc->phase_sample_count > 0U) ? phase_mean_mag : 0.0;
        if (coherence > 1.0 && coherence < 1.0 + 1e-12)
            coherence = 1.0;
        acc->stats->phase_coherence = coherence;

        double mean_mag_sq = acc->sum_mag_sq / (double) n;
        double mean_square_mag =
            sqrt(acc->mean_square_re * acc->mean_square_re +
                 acc->mean_square_im * acc->mean_square_im);
        double rho = (mean_mag_sq > 0.0) ? (mean_square_mag / mean_mag_sq) : 0.0;
        if (rho > 1.0 && rho < 1.0 + 1e-12)
            rho = 1.0; /* clamp small numerical drift */
        acc->stats->circularity        = rho;
        acc->stats->phase_sample_count = acc->phase_sample_count;
    } else {
        acc->stats->phase_coherence = 0.0;
        acc->stats->circularity     = 0.0;
        acc->stats->phase_sample_count = 0U;
    }
    acc->stats->phase_coherence_weighted = 0.0;
    acc->stats->phase_coherence_ema      = 0.0;
    acc->stats->phase_coherence_k0       = 0.0;
    acc->stats->phase_lock_state         = 0U;
    acc->stats->phase_regime             = 0U;
    sim_field_stats_reset_topology_fields(acc->stats);
}

bool sim_field_stats_compute_spectral_view(const SimFieldView* view,
                                           SimFieldStats*      stats,
                                           size_t*             out_dominant_k) {
    double sum_power  = 0.0;
    double entropy    = 0.0;
    double mean_k     = 0.0;
    double variance_k = 0.0;
    size_t n;

    if (!view || !stats || !view->data || view->count == 0U)
        return false;

    n = view->count;
    if (!sim_field_stats_spectral_ensure(&g_spectral_scratch, n)) {
        stats->spectral_entropy   = 0.0;
        stats->spectral_bandwidth = 0.0;
        return false;
    }

    /* Copy into scratch buffer as complex */
    switch (view->type) {
        case SIM_FIELD_DOUBLE:
        case SIM_FIELD_I8:
        case SIM_FIELD_U8:
        case SIM_FIELD_I32:
        case SIM_FIELD_U32:
        case SIM_FIELD_I64:
        case SIM_FIELD_U64: {
            for (size_t i = 0U; i < n; ++i) {
                SimComplexDouble value = { 0 };
                if (!sim_buffer_view_get_complex(view, i, &value)) {
                    stats->spectral_entropy   = 0.0;
                    stats->spectral_bandwidth = 0.0;
                    return false;
                }
                g_spectral_scratch.time[i] = CMPLX(value.re, 0.0);
            }
            break;
        }
        case SIM_FIELD_COMPLEX_DOUBLE: {
            for (size_t i = 0U; i < n; ++i) {
                SimComplexDouble value = { 0 };
                if (!sim_buffer_view_get_complex(view, i, &value)) {
                    stats->spectral_entropy   = 0.0;
                    stats->spectral_bandwidth = 0.0;
                    return false;
                }
                g_spectral_scratch.time[i] = CMPLX(value.re, value.im);
            }
            break;
        }
        default:
            stats->spectral_entropy   = 0.0;
            stats->spectral_bandwidth = 0.0;
            return false;
    }

    if (fft_plan_forward(&g_spectral_scratch.plan,
                         g_spectral_scratch.time,
                         g_spectral_scratch.freq) != SIM_RESULT_OK) {
        stats->spectral_entropy   = 0.0;
        stats->spectral_bandwidth = 0.0;
        return false;
    }

    size_t dominant_k     = 0U;
    double dominant_power = 0.0;

    for (size_t i = 0U; i < n; ++i) {
        double re    = creal(g_spectral_scratch.freq[i]);
        double im    = cimag(g_spectral_scratch.freq[i]);
        double power = re * re + im * im;
        sum_power += power;
        if (power > dominant_power) {
            dominant_power = power;
            dominant_k     = i;
        }
    }

    if (!(sum_power > 0.0)) {
        stats->spectral_entropy   = 0.0;
        stats->spectral_bandwidth = 0.0;
        return false;
    }

    double inv_sum = 1.0 / sum_power;
    for (size_t i = 0U; i < n; ++i) {
        double re    = creal(g_spectral_scratch.freq[i]);
        double im    = cimag(g_spectral_scratch.freq[i]);
        double power = (re * re + im * im);
        double p     = power * inv_sum;
        double k     = sim_field_stats_frequency_bin(i, n);
        if (p > 0.0)
            entropy -= p * log(p);
        mean_k += p * k;
    }

    for (size_t i = 0U; i < n; ++i) {
        double re    = creal(g_spectral_scratch.freq[i]);
        double im    = cimag(g_spectral_scratch.freq[i]);
        double power = (re * re + im * im);
        double p     = power * inv_sum;
        double k     = sim_field_stats_frequency_bin(i, n);
        double delta = k - mean_k;
        variance_k += p * (delta * delta);
    }

    double log_norm           = log((double) n);
    stats->spectral_entropy   = (log_norm > 0.0) ? (entropy / log_norm) : 0.0;
    stats->spectral_bandwidth = sqrt(variance_k);
    if (out_dominant_k)
        *out_dominant_k = dominant_k;
    return true;
}

void sim_field_stats_compute_phase_metrics(const SimFieldView*            view,
                                           SimFieldStats*                 stats,
                                           const SimPhaseCoherenceConfig* config,
                                           size_t                         dominant_k) {
    double                         sum_re = 0.0, sum_im = 0.0;
    double                         wsum = 0.0, wre = 0.0, wim = 0.0;
    double                         deramp_re = 0.0, deramp_im = 0.0;
    double                         deramp_wre = 0.0, deramp_wim = 0.0;
    size_t                         sample_count = 0U;
    size_t                         n;
    const SimPhaseCoherenceConfig* cfg    = config ? config : &g_phase_config;
    double                         rel    = (cfg->rel_threshold > 0.0) ? cfg->rel_threshold : 0.0;
    double                         tau    = fmax(cfg->abs_threshold, rel * stats->mean_abs);
    bool                           deramp = cfg->deramp_enabled;
    double                         two_pi_over_n = 0.0;

    if (!view || !stats || !view->data || view->count == 0U)
        return;

    n = view->count;
    if (deramp && n > 0U)
        two_pi_over_n = -2.0 * M_PI / (double) n;

    switch (view->type) {
        case SIM_FIELD_DOUBLE:
        case SIM_FIELD_I8:
        case SIM_FIELD_U8:
        case SIM_FIELD_I32:
        case SIM_FIELD_U32:
        case SIM_FIELD_I64:
        case SIM_FIELD_U64: {
            for (size_t i = 0U; i < n; ++i) {
                SimComplexDouble value = { 0 };
                if (!sim_buffer_view_get_complex(view, i, &value)) {
                    return;
                }
                double re  = value.re;
                double mag = fabs(re);
                if (mag < tau)
                    continue;
                double inv  = (mag > 0.0) ? (1.0 / mag) : 0.0;
                double u_re = re * inv;
                double u_im = 0.0;
                sample_count++;
                sum_re += u_re;
                sum_im += u_im;
                wsum += mag;
                wre += u_re * mag;
                wim += u_im * mag;

                if (deramp) {
                    double phase = two_pi_over_n * (double) (dominant_k * i);
                    double c     = cos(phase);
                    double s     = sin(phase);
                    double d_re  = u_re * c - u_im * s;
                    double d_im  = u_re * s + u_im * c;
                    deramp_re += d_re;
                    deramp_im += d_im;
                    deramp_wre += d_re * mag;
                    deramp_wim += d_im * mag;
                }
            }
            break;
        }
        case SIM_FIELD_COMPLEX_DOUBLE: {
            for (size_t i = 0U; i < n; ++i) {
                SimComplexDouble value = { 0 };
                if (!sim_buffer_view_get_complex(view, i, &value)) {
                    return;
                }
                double re   = value.re;
                double im   = value.im;
                double mag2 = re * re + im * im;
                double mag  = sqrt(mag2);
                if (mag < tau)
                    continue;
                double inv  = (mag > 0.0) ? (1.0 / mag) : 0.0;
                double u_re = re * inv;
                double u_im = im * inv;
                sample_count++;
                sum_re += u_re;
                sum_im += u_im;
                wsum += mag;
                wre += u_re * mag;
                wim += u_im * mag;

                if (deramp) {
                    double phase = two_pi_over_n * (double) (dominant_k * i);
                    double c     = cos(phase);
                    double s     = sin(phase);
                    double d_re  = u_re * c - u_im * s;
                    double d_im  = u_re * s + u_im * c;
                    deramp_re += d_re;
                    deramp_im += d_im;
                    deramp_wre += d_re * mag;
                    deramp_wim += d_im * mag;
                }
            }
            break;
        }
        default:
            break;
    }

    stats->phase_sample_count = sample_count;
    if (sample_count > 0U) {
        double coh = sqrt(sum_re * sum_re + sum_im * sum_im) / (double) sample_count;
        if (coh > 1.0 && coh < 1.0 + 1e-12)
            coh = 1.0;
        stats->phase_coherence = coh;
    } else {
        stats->phase_coherence = 0.0;
    }

    if (wsum > 0.0) {
        double coh_w = sqrt(wre * wre + wim * wim) / wsum;
        if (coh_w > 1.0 && coh_w < 1.0 + 1e-12)
            coh_w = 1.0;
        stats->phase_coherence_weighted = coh_w;
    } else {
        stats->phase_coherence_weighted = 0.0;
    }

    if (deramp && sample_count > 0U) {
        double coh_deramp;
        if (cfg->weighted && wsum > 0.0) {
            coh_deramp = sqrt(deramp_wre * deramp_wre + deramp_wim * deramp_wim) / wsum;
        } else {
            coh_deramp =
                sqrt(deramp_re * deramp_re + deramp_im * deramp_im) / (double) sample_count;
        }
        if (coh_deramp > 1.0 && coh_deramp < 1.0 + 1e-12)
            coh_deramp = 1.0;
        stats->phase_coherence_k0 = coh_deramp;
    } else {
        stats->phase_coherence_k0 = 0.0;
    }

    {
        double R_use = cfg->weighted ? stats->phase_coherence_weighted : stats->phase_coherence;
        double rho   = stats->circularity;
        const double R_low    = 0.3;
        const double R_high   = 0.7;
        const double rho_high = 0.6;
        uint8_t      regime   = 0U;
        if (R_use >= R_high && rho < rho_high * 0.7)
            regime = 1U; /* locked/phase-aligned */
        else if (R_use >= R_low && rho >= rho_high)
            regime = 2U; /* elliptical/vortex */
        else
            regime = 3U; /* dispersed */
        stats->phase_regime = regime;
    }
}

void sim_field_stats_update_phase_lock(SimContext*    context,
                                       size_t         field_index,
                                       SimFieldStats* stats) {
    double  now;
    double  current;
    double  prev;
    double  ema;
    double  tau;
    double  alpha;
    uint8_t locked;

    if (!context || !stats || field_index >= context->runtime.continuity_capacity)
        return;

    if (!context->runtime.field_phase_ema || !context->runtime.field_phase_last_time ||
        !context->runtime.field_phase_lock_state || !context->runtime.field_phase_initialized) {
        return;
    }

    now     = context->runtime.time_accumulated;
    locked  = context->runtime.field_phase_lock_state[field_index];
    prev    = context->runtime.field_phase_ema[field_index];
    tau     = g_phase_config.smoothing_constant;
    current = g_phase_config.weighted ? stats->phase_coherence_weighted : stats->phase_coherence;

    if (!context->runtime.field_phase_initialized[field_index]) {
        ema                                                   = current;
        context->runtime.field_phase_initialized[field_index] = 1U;
    } else {
        double last_time = context->runtime.field_phase_last_time[field_index];
        double dt        = now - last_time;
        if (tau > 0.0 && dt > 0.0) {
            alpha = exp(-dt / tau);
        } else {
            alpha = 0.0;
        }
        ema = alpha * prev + (1.0 - alpha) * current;
    }

    context->runtime.field_phase_last_time[field_index] = now;
    context->runtime.field_phase_ema[field_index]       = ema;

    if (locked) {
        if (ema <= g_phase_config.lock_off)
            locked = 0U;
    } else {
        if (ema >= g_phase_config.lock_on)
            locked = 1U;
    }
    context->runtime.field_phase_lock_state[field_index] = locked;

    stats->phase_coherence_ema = ema;
    stats->phase_lock_state    = locked;
}
