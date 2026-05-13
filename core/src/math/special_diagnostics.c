/**
 * @file special_diagnostics.c
 * @brief Thread-local diagnostics counters for special-function evaluations.
 *
 * These counters are compiled only when `SIM_DIAGNOSTICS` is enabled. They are
 * intentionally approximate instrumentation for numerical health checks, not a
 * synchronization mechanism or a global profiler.
 */

#include "internal/polygamma_internal.h"

#if SIM_DIAGNOSTICS
/**
 * @brief Per-thread counters tracking reflection, recurrence, and tail behavior.
 *
 * One instance exists per thread through `_Thread_local`, so callers can sample
 * diagnostics without taking locks around normal special-function evaluation.
 */
typedef struct SimSpecialDiagnosticsTLS {
    uint64_t reflection_count;
    uint64_t recurrence_shift_samples;
    double   max_recurrence_shift;
    uint64_t stirling_tail_samples;
    double   max_stirling_tail;
    uint64_t pole_proximity_samples;
    double   min_pole_distance;
} SimSpecialDiagnosticsTLS;

static _Thread_local SimSpecialDiagnosticsTLS g_sim_special_diag_tls = { 0,   0, 0.0,     0,
                                                                         0.0, 0, INFINITY };

/**
 * @brief Copy the current thread-local diagnostics snapshot, optionally resetting it.
 *
 * @param out Destination snapshot; ignored when NULL.
 * @param reset When true, zero counters after copying the snapshot.
 */
void sim_special_diagnostics_snapshot(SimSpecialDiagnosticsSnapshot* out, bool reset) {
    if (out == NULL) {
        return;
    }

    out->reflection_count         = g_sim_special_diag_tls.reflection_count;
    out->recurrence_shift_samples = g_sim_special_diag_tls.recurrence_shift_samples;
    out->max_recurrence_shift     = g_sim_special_diag_tls.max_recurrence_shift;
    out->stirling_tail_samples    = g_sim_special_diag_tls.stirling_tail_samples;
    out->max_stirling_tail        = g_sim_special_diag_tls.max_stirling_tail;
    out->pole_proximity_samples   = g_sim_special_diag_tls.pole_proximity_samples;
    out->min_pole_distance        = g_sim_special_diag_tls.min_pole_distance;

    if (reset) {
        g_sim_special_diag_tls.reflection_count         = 0ULL;
        g_sim_special_diag_tls.recurrence_shift_samples = 0ULL;
        g_sim_special_diag_tls.max_recurrence_shift     = 0.0;
        g_sim_special_diag_tls.stirling_tail_samples    = 0ULL;
        g_sim_special_diag_tls.max_stirling_tail        = 0.0;
        g_sim_special_diag_tls.pole_proximity_samples   = 0ULL;
        g_sim_special_diag_tls.min_pole_distance        = INFINITY;
    }
}

/**
 * @brief Track the distance to the nearest pole encountered during evaluation.
 *
 * Non-finite and negative distances are ignored.
 *
 * @param distance Nonnegative distance to a monitored pole.
 */
void sim_special_diag_track_pole_distance(double distance) {
    if (!(distance >= 0.0) || !isfinite(distance)) {
        return;
    }
    g_sim_special_diag_tls.pole_proximity_samples += 1ULL;
    if (distance < g_sim_special_diag_tls.min_pole_distance) {
        g_sim_special_diag_tls.min_pole_distance = distance;
    }
}

/**
 * @brief Track recurrence shifts applied before asymptotic evaluation.
 *
 * @param shift_magnitude Positive number of recurrence steps or equivalent shift.
 */
void sim_special_diag_track_recurrence(double shift_magnitude) {
    if (!(shift_magnitude > 0.0) || !isfinite(shift_magnitude)) {
        return;
    }
    g_sim_special_diag_tls.recurrence_shift_samples += 1ULL;
    if (shift_magnitude > g_sim_special_diag_tls.max_recurrence_shift) {
        g_sim_special_diag_tls.max_recurrence_shift = shift_magnitude;
    }
}

/**
 * @brief Track the magnitude of a Stirling tail contribution.
 *
 * @param tail Tail contribution; its absolute value is sampled.
 */
void sim_special_diag_track_stirling_tail(double tail) {
    double mag = fabs(tail);
    if (!isfinite(mag)) {
        return;
    }
    g_sim_special_diag_tls.stirling_tail_samples += 1ULL;
    if (mag > g_sim_special_diag_tls.max_stirling_tail) {
        g_sim_special_diag_tls.max_stirling_tail = mag;
    }
}

/**
 * @brief Increment the reflection-path evaluation counter.
 *
 * Called whenever a reflection identity is selected by an instrumented path.
 */
void sim_special_diag_track_reflection(void) {
    g_sim_special_diag_tls.reflection_count += 1ULL;
}
#endif
