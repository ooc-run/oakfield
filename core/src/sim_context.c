/**
 * @file sim_context.c
 * @brief Runtime context implementation for fields, operators, planning, and diagnostics.
 *
 * SimContext owns the simulation world state, scheduler state, runtime clocks,
 * diagnostics, memory accounting, backend selection, and integrator bindings.
 * Execution prepares hazard-aware plans, validates field representations,
 * dispatches symbolic kernels or CPU callbacks, and records continuity,
 * profiler, special-function, and step-metric state.
 */
#include "oakfield/sim_context.h"

#include <stdlib.h>
#include <stdarg.h>
#include "oakfield/backend.h"
#include "oakfield/integrator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include "oakfield/sim_field_stats_runtime.h"
#include "oakfield/sim_flux_lens.h"
#include "sim_accel.h"

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

#define CONTINUITY_MASK_WIDTH 64U

static void        sim_context_record_continuity_hit(SimContext* context, const SimOperator* op);
static void        sim_context_apply_override_to_operator(SimContext* context, SimOperator* op);
static bool        sim_kernel_requires_boundary_feature(const KernelIR* kernel);
static bool        sim_kernel_requires_warp_profile_fallback(const KernelIR*   kernel,
                                                             const SimBackend* backend);
static bool        sim_kernel_contains_node_type(const KernelIR* kernel, SimIRNodeType type);
static bool        sim_kernel_uses_integer_domain(const KernelIR* kernel);
static void        sim_context_reset_kernel_dispatch_diag(SimDiagnostics* diag);
static void        sim_context_record_kernel_dispatch(SimContext*              context,
                                                      const SimOperator*       op,
                                                      const SimBackend*        requested_backend,
                                                      SimDiagnosticBackendKind executed_backend,
                                                      bool                     fallback_used,
                                                      const char*              reason);
static const char* sim_context_known_backend_fallback_reason(const SimBackend* backend,
                                                             const KernelIR*   kernel);
static SimResult   sim_context_prepare_complex_fields(SimContext* context);
static bool sim_context_check_invariants_enabled(const SimContext* context, const SimOperator* op);
static SimResult sim_context_resolve_representations(SimContext* context);
static SimResult sim_context_validate_field_representations(SimContext* context);
static void      sim_context_apply_norm_budget(SimContext* context, const SimOperator* op);
static SimRepresentationMode sim_context_representation_mode_normalize(SimRepresentationMode mode);
static bool                  sim_context_allows_determinism_mode(SimRepresentationMode mode,
                                                                 SimDeterminismFlags   flags);

#if defined(SIM_HAVE_VDSP)
static bool sim_context_measure_invariant_real_vdsp(const double*                 data,
                                                    size_t                        total,
                                                    SimOperatorInvariantKind      kind,
                                                    double*                       out_value) {
    if (data == NULL || out_value == NULL || total == 0U) {
        return false;
    }

    const vDSP_Length length = (vDSP_Length) total;
    double            sum    = 0.0;
    double            sum_sq = 0.0;

    switch (kind) {
        case SIM_OPERATOR_INVARIANT_L2_NORM:
            vDSP_dotprD(data, 1, data, 1, &sum_sq, length);
            *out_value = sqrt(sum_sq);
            return true;
        case SIM_OPERATOR_INVARIANT_ENERGY:
            vDSP_dotprD(data, 1, data, 1, &sum_sq, length);
            *out_value = sum_sq;
            return true;
        case SIM_OPERATOR_INVARIANT_MASS:
            vDSP_sveD(data, 1, &sum, length);
            *out_value = sum;
            return true;
        case SIM_OPERATOR_INVARIANT_MEAN:
            vDSP_meanvD(data, 1, out_value, length);
            return true;
        case SIM_OPERATOR_INVARIANT_L1_NORM:
            vDSP_svemgD(data, 1, out_value, length);
            return true;
        case SIM_OPERATOR_INVARIANT_VARIANCE: {
            double mean_sq = 0.0;
            double mean    = 0.0;
            vDSP_meanvD(data, 1, &mean, length);
            vDSP_measqvD(data, 1, &mean_sq, length);
            *out_value = fmax(0.0, mean_sq - mean * mean);
            return true;
        }
        default:
            return false;
    }
}

static bool sim_context_measure_invariant_complex_vdsp(const SimComplexDouble*        data,
                                                       size_t                         total,
                                                       SimOperatorInvariantKind       kind,
                                                       double*                        out_value) {
    SimAccelSplitComplexScratch scratch = { 0 };
    bool                        ok      = false;
    double                      sum     = 0.0;
    double                      mean    = 0.0;
    double                      mean_sq = 0.0;

    if (data == NULL || out_value == NULL || total == 0U) {
        return false;
    }
    if (!sim_accel_split_load_interleaved(&scratch, data, total)) {
        return false;
    }

    DSPDoubleSplitComplex split = { .realp = scratch.a.realp, .imagp = scratch.a.imagp };
    const vDSP_Length     length = (vDSP_Length) total;

    switch (kind) {
        case SIM_OPERATOR_INVARIANT_L2_NORM:
            vDSP_zvmagsD(&split, 1, scratch.b.realp, 1, length);
            vDSP_sveD(scratch.b.realp, 1, &sum, length);
            *out_value = sqrt(sum);
            ok         = true;
            break;
        case SIM_OPERATOR_INVARIANT_ENERGY:
            vDSP_zvmagsD(&split, 1, scratch.b.realp, 1, length);
            vDSP_sveD(scratch.b.realp, 1, out_value, length);
            ok = true;
            break;
        case SIM_OPERATOR_INVARIANT_L1_NORM:
        case SIM_OPERATOR_INVARIANT_MASS:
            vDSP_zvabsD(&split, 1, scratch.b.realp, 1, length);
            vDSP_sveD(scratch.b.realp, 1, out_value, length);
            ok = true;
            break;
        case SIM_OPERATOR_INVARIANT_MEAN:
            vDSP_zvabsD(&split, 1, scratch.b.realp, 1, length);
            vDSP_meanvD(scratch.b.realp, 1, out_value, length);
            ok = true;
            break;
        case SIM_OPERATOR_INVARIANT_VARIANCE:
            vDSP_zvabsD(&split, 1, scratch.b.realp, 1, length);
            vDSP_meanvD(scratch.b.realp, 1, &mean, length);
            vDSP_measqvD(scratch.b.realp, 1, &mean_sq, length);
            *out_value = fmax(0.0, mean_sq - mean * mean);
            ok         = true;
            break;
        default:
            break;
    }

    sim_accel_split_release(&scratch);
    return ok;
}
#endif

static bool sim_context_add_size(size_t a, size_t b, size_t* out) {
    if (out == NULL) {
        return false;
    }
    if (a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static SimContextMemoryLimits sim_context_memory_limits_default(void) {
    SimContextMemoryLimits limits;
    limits.max_field_elements             = SIM_CONTEXT_MAX_FIELD_ELEMENTS_DEFAULT;
    limits.max_field_bytes                = SIM_CONTEXT_MAX_FIELD_BYTES_DEFAULT;
    limits.max_fields                     = SIM_CONTEXT_MAX_FIELDS_DEFAULT;
    limits.max_total_field_bytes          = SIM_CONTEXT_MAX_TOTAL_FIELD_BYTES_DEFAULT;
    limits.max_scratch_bytes_per_operator = SIM_CONTEXT_MAX_SCRATCH_BYTES_PER_OPERATOR_DEFAULT;
    return limits;
}

static SimRepresentationMode sim_context_representation_mode_normalize(SimRepresentationMode mode) {
    switch (mode) {
        case SIM_REPRESENTATION_MODE_STRICT:
        case SIM_REPRESENTATION_MODE_RELAXED:
        case SIM_REPRESENTATION_MODE_EXPLORATION:
            return mode;
        default:
            return SIM_REPRESENTATION_MODE_RELAXED;
    }
}

static bool sim_context_allows_determinism_mode(SimRepresentationMode mode,
                                                SimDeterminismFlags   flags) {
    if (flags == SIM_DET_NONE) {
        return mode != SIM_REPRESENTATION_MODE_STRICT;
    }

    uint32_t required = 0U;
    switch (mode) {
        case SIM_REPRESENTATION_MODE_STRICT:
            required = SIM_DET_PURE_TIME | SIM_DET_REWIND_SAFE | SIM_DET_NO_STATEFUL_NODES |
                       SIM_DET_DETERMINISTIC_RNG_ONLY;
            break;
        case SIM_REPRESENTATION_MODE_RELAXED:
            required = SIM_DET_NO_STATEFUL_NODES | SIM_DET_DETERMINISTIC_RNG_ONLY;
            break;
        case SIM_REPRESENTATION_MODE_EXPLORATION:
            return true;
        default:
            required = SIM_DET_NO_STATEFUL_NODES | SIM_DET_DETERMINISTIC_RNG_ONLY;
            break;
    }

    return (flags & required) == required;
}

static SimDiagnosticBackendKind sim_context_backend_diag_kind(const SimBackend* backend) {
    if (backend == NULL) {
        return SIM_DIAGNOSTIC_BACKEND_UNKNOWN;
    }

    switch (backend->type) {
        case SIM_BACKEND_TYPE_CPU:
            return SIM_DIAGNOSTIC_BACKEND_CPU;
        case SIM_BACKEND_TYPE_CUDA:
            return SIM_DIAGNOSTIC_BACKEND_CUDA;
        case SIM_BACKEND_TYPE_METAL:
            return SIM_DIAGNOSTIC_BACKEND_METAL;
        default:
            break;
    }
    return SIM_DIAGNOSTIC_BACKEND_UNKNOWN;
}

static void sim_context_reset_kernel_dispatch_diag(SimDiagnostics* diag) {
    if (diag == NULL) {
        return;
    }

    diag->kernel_dispatch_count          = 0ULL;
    diag->kernel_fallback_count          = 0ULL;
    diag->kernel_last_valid              = false;
    diag->kernel_last_fallback_used      = false;
    diag->kernel_last_requested_backend  = SIM_DIAGNOSTIC_BACKEND_UNKNOWN;
    diag->kernel_last_executed_backend   = SIM_DIAGNOSTIC_BACKEND_UNKNOWN;
    diag->kernel_last_operator[0]        = '\0';
    diag->kernel_last_fallback_reason[0] = '\0';
}

static void sim_context_record_kernel_dispatch(SimContext*              context,
                                               const SimOperator*       op,
                                               const SimBackend*        requested_backend,
                                               SimDiagnosticBackendKind executed_backend,
                                               bool                     fallback_used,
                                               const char*              reason) {
    SimDiagnostics* diag;
    const char*     operator_name;
    const char*     fallback_reason;

    if (context == NULL) {
        return;
    }

    diag            = &context->diag;
    operator_name   = (op != NULL && op->name[0] != '\0') ? op->name : "<unnamed>";
    fallback_reason = (fallback_used && reason != NULL) ? reason : "";

    diag->kernel_dispatch_count += 1ULL;
    if (fallback_used) {
        diag->kernel_fallback_count += 1ULL;
    }
    diag->kernel_last_valid             = true;
    diag->kernel_last_fallback_used     = fallback_used;
    diag->kernel_last_requested_backend = sim_context_backend_diag_kind(requested_backend);
    diag->kernel_last_executed_backend  = executed_backend;
    (void) snprintf(
        diag->kernel_last_operator, sizeof(diag->kernel_last_operator), "%s", operator_name);
    (void) snprintf(diag->kernel_last_fallback_reason,
                    sizeof(diag->kernel_last_fallback_reason),
                    "%s",
                    fallback_reason);
}

typedef struct SimInvariantSample {
    size_t                   field_index;
    SimOperatorInvariantKind kind;
    double                   before;
    double                   tolerance;
} SimInvariantSample;
static bool sim_context_capture_invariants(const SimContext*    context,
                                           const SimOperator*   op,
                                           SimInvariantSample** out_samples,
                                           size_t*              out_count);

static uint64_t sim_context_now_ns(void) {
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

static void sim_context_invalidate_field_stats_caches(SimContext* context) {
    size_t field_count;

    if (context == NULL) {
        return;
    }

    field_count = context->world.field_count;
    if (context->runtime.field_stats_cache_step_index != NULL && field_count > 0U) {
        (void) memset(context->runtime.field_stats_cache_step_index,
                      0xFF,
                      field_count * sizeof(size_t));
    }
    size_t limit = context->runtime.continuity_capacity;
    if (limit > field_count) {
        limit = field_count;
    }
    if (context->runtime.drift_field_stats_valid != NULL && limit > 0U) {
        (void) memset(context->runtime.drift_field_stats_valid, 0, limit * sizeof(bool));
    }
    context->runtime.drift_field_stats_step_index = SIZE_MAX;
}

static void sim_context_reset_field_stats_phase_state(SimContext* context) {
    size_t field_count;

    if (context == NULL) {
        return;
    }

    field_count = context->world.field_count;
    if (context->runtime.field_phase_ema != NULL && field_count > 0U) {
        (void) memset(context->runtime.field_phase_ema, 0, field_count * sizeof(double));
    }
    if (context->runtime.field_phase_last_time != NULL && field_count > 0U) {
        for (size_t i = 0U; i < field_count; ++i) {
            context->runtime.field_phase_last_time[i] = context->runtime.time_accumulated;
        }
    }
    if (context->runtime.field_phase_lock_state != NULL && field_count > 0U) {
        (void) memset(context->runtime.field_phase_lock_state, 0, field_count * sizeof(uint8_t));
    }
    if (context->runtime.field_phase_initialized != NULL && field_count > 0U) {
        (void) memset(context->runtime.field_phase_initialized, 0, field_count * sizeof(uint8_t));
    }
}

static size_t sim_context_collect_fields(uint64_t mask, size_t* out_indices, size_t capacity) {
    size_t count = 0U;
    while (mask != 0U && count < capacity) {
        unsigned int bit = __builtin_ctzll(mask);
        mask &= ~(1ULL << bit);
        out_indices[count++] = (size_t) bit;
    }
    return count;
}

static const char* sim_context_invariant_name(SimOperatorInvariantKind kind) {
    switch (kind) {
        case SIM_OPERATOR_INVARIANT_L2_NORM:
            return "l2";
        case SIM_OPERATOR_INVARIANT_L1_NORM:
            return "l1";
        case SIM_OPERATOR_INVARIANT_ENERGY:
            return "energy";
        case SIM_OPERATOR_INVARIANT_MASS:
            return "mass";
        case SIM_OPERATOR_INVARIANT_MEAN:
            return "mean";
        case SIM_OPERATOR_INVARIANT_VARIANCE:
            return "variance";
        default:
            break;
    }
    return "unknown";
}

static bool sim_context_measure_invariant(const SimField*          field,
                                          SimOperatorInvariantKind kind,
                                          double*                  out_value) {
    if (field == NULL || out_value == NULL) {
        return false;
    }

    const double*           data         = sim_field_real_data_const(field);
    const SimComplexDouble* complex_data = sim_field_complex_data_const(field);
    const size_t            total        = sim_field_element_count(&field->layout);
    const size_t            comps        = sim_field_components(field);
    const bool              is_complex   = sim_field_storage_is_complex(field);
    if (((!is_complex) && data == NULL) || (is_complex && complex_data == NULL) || total == 0U ||
        comps == 0U) {
        return false;
    }

#if defined(SIM_HAVE_VDSP)
    if (!is_complex && sim_context_measure_invariant_real_vdsp(data, total, kind, out_value)) {
        return true;
    }
    if (is_complex && sim_context_measure_invariant_complex_vdsp(
                          complex_data, total, kind, out_value)) {
        return true;
    }
#endif

    double sum    = 0.0;
    double sum_sq = 0.0;
    double mean   = 0.0;

    for (size_t i = 0U; i < total; ++i) {
        size_t base   = i * comps;
        double re     = data[base];
        double im     = (comps > 1U) ? data[base + 1U] : 0.0;
        double sample = is_complex ? hypot(re, im) : re;

        switch (kind) {
            case SIM_OPERATOR_INVARIANT_L2_NORM:
            case SIM_OPERATOR_INVARIANT_ENERGY:
                sum_sq += sample * sample;
                break;
            case SIM_OPERATOR_INVARIANT_L1_NORM:
                sum += fabs(sample);
                break;
            case SIM_OPERATOR_INVARIANT_MASS:
            case SIM_OPERATOR_INVARIANT_MEAN:
            case SIM_OPERATOR_INVARIANT_VARIANCE:
                sum += sample;
                break;
            default:
                return false;
        }
    }

    switch (kind) {
        case SIM_OPERATOR_INVARIANT_L2_NORM:
            *out_value = sqrt(sum_sq);
            return true;
        case SIM_OPERATOR_INVARIANT_ENERGY:
            *out_value = sum_sq;
            return true;
        case SIM_OPERATOR_INVARIANT_L1_NORM:
            *out_value = sum;
            return true;
        case SIM_OPERATOR_INVARIANT_MASS:
            *out_value = sum;
            return true;
        case SIM_OPERATOR_INVARIANT_MEAN:
            *out_value = (total > 0U) ? (sum / (double) total) : 0.0;
            return true;
        case SIM_OPERATOR_INVARIANT_VARIANCE:
            if (total == 0U) {
                *out_value = 0.0;
                return true;
            }
            mean   = sum / (double) total;
            sum_sq = 0.0;
            for (size_t i = 0U; i < total; ++i) {
                size_t base   = i * comps;
                double re     = data[base];
                double im     = (comps > 1U) ? data[base + 1U] : 0.0;
                double sample = is_complex ? hypot(re, im) : re;
                double diff   = sample - mean;
                sum_sq += diff * diff;
            }
            *out_value = sum_sq / (double) total;
            return true;
        default:
            break;
    }
    return false;
}

static bool sim_context_check_invariants_enabled(const SimContext* context, const SimOperator* op) {
    if (context == NULL || op == NULL) {
        return false;
    }
    if (!context->diag.enable_invariant_checks) {
        return false;
    }
    return op->info.invariant_count > 0U;
}

static bool sim_context_capture_invariants(const SimContext*    context,
                                           const SimOperator*   op,
                                           SimInvariantSample** out_samples,
                                           size_t*              out_count) {
    if (!sim_context_check_invariants_enabled(context, op) || out_samples == NULL ||
        out_count == NULL) {
        return false;
    }

    uint64_t mask = (op->write_mask != 0ULL) ? op->write_mask : op->read_mask;
    if (mask == 0ULL) {
        return false;
    }

    size_t field_indices[64];
    size_t field_count = sim_context_collect_fields(mask, field_indices, 64U);
    if (field_count == 0U) {
        return false;
    }

    size_t              capacity = field_count * (size_t) op->info.invariant_count;
    SimInvariantSample* samples =
        (SimInvariantSample*) calloc(capacity, sizeof(SimInvariantSample));
    if (samples == NULL) {
        return false;
    }

    size_t cursor = 0U;
    for (size_t f = 0U; f < field_count; ++f) {
        SimField* field = sim_context_field((SimContext*) context, field_indices[f]);
        if (field == NULL) {
            free(samples);
            return false;
        }
        for (uint8_t k = 0U; k < op->info.invariant_count; ++k) {
            const SimOperatorInvariant* inv = &op->info.invariants[k];
            if (inv->kind == SIM_OPERATOR_INVARIANT_NONE) {
                continue;
            }
            double value = 0.0;
            if (!sim_context_measure_invariant(field, inv->kind, &value)) {
                continue;
            }
            if (cursor < capacity) {
                samples[cursor].field_index = field_indices[f];
                samples[cursor].kind        = inv->kind;
                samples[cursor].before      = value;
                samples[cursor].tolerance   = inv->tolerance;
                cursor += 1U;
            }
        }
    }

    if (cursor == 0U) {
        free(samples);
        return false;
    }

    *out_samples = samples;
    *out_count   = cursor;
    return true;
}

static void sim_context_fault_lock(SimContext* context) {
    if (context == NULL) {
        return;
    }

    while (__sync_lock_test_and_set(&context->diag.fault_lock, 1)) {
    }
}

static void sim_context_fault_unlock(SimContext* context) {
    if (context == NULL) {
        return;
    }

    __sync_lock_release(&context->diag.fault_lock);
}

static bool sim_context_configure_profiler(SimContext* context, size_t operators) {
    if (context == NULL) {
        return false;
    }

    if (operators == 0U) {
        if (context->profiler_ready) {
            sim_profiler_destroy(&context->profiler);
            context->profiler_ready = false;
        }
        return false;
    }

    if (!context->profiler_ready || context->profiler.counter_count != operators) {
        if (context->profiler_ready) {
            sim_profiler_destroy(&context->profiler);
            context->profiler_ready = false;
        }
        if (sim_profiler_init(&context->profiler, operators) == SIM_RESULT_OK) {
            context->profiler_ready = true;
        }
    }

    return context->profiler_ready;
}

static SimResult sim_context_special_fallback_dispatch(void*                       userdata,
                                                       const SimSpecialEvalReport* report,
                                                       SimComplexDouble*           value_out) {
    SimContext* context = (SimContext*) userdata;

    if (context != NULL && report != NULL) {
        sim_context_fault_lock(context);
        const char* fn = (report->function != NULL && report->function[0] != '\0')
                             ? report->function
                             : "<unknown>";
        (void) snprintf(
            context->diag.fault_function, sizeof(context->diag.fault_function), "%s", fn);
        context->diag.fault_last          = *report;
        context->diag.fault_last.function = context->diag.fault_function;
        context->diag.fault_count += 1ULL;
        sim_context_fault_unlock(context);
    }

    if (context != NULL && context->diag.fallback_user != NULL) {
        return context->diag.fallback_user(context->diag.fallback_userdata, report, value_out);
    }

    if (value_out != NULL) {
        value_out->re = 0.0;
        value_out->im = 0.0;
    }

    return SIM_RESULT_OK;
}

SimDiagnostics* sim_context_diagnostics(SimContext* context) {
    if (context == NULL) {
        return NULL;
    }
    return &context->diag;
}

const SimDiagnostics* sim_context_diagnostics_const(const SimContext* context) {
    if (context == NULL) {
        return NULL;
    }
    return &context->diag;
}

#if SIM_DIAGNOSTICS
void sim_context_flush_special_diagnostics(SimContext* context) {
    if (context == NULL) {
        return;
    }

    SimSpecialDiagnosticsSnapshot snapshot;
    sim_special_diagnostics_snapshot(&snapshot, true);

    SimDiagnostics* diag = &context->diag;
    diag->reflection_count += snapshot.reflection_count;
    diag->recurrence_shift_samples += snapshot.recurrence_shift_samples;
    if (snapshot.max_recurrence_shift > diag->max_recurrence_shift) {
        diag->max_recurrence_shift = snapshot.max_recurrence_shift;
    }
    diag->stirling_tail_samples += snapshot.stirling_tail_samples;
    if (snapshot.max_stirling_tail > diag->max_stirling_tail) {
        diag->max_stirling_tail = snapshot.max_stirling_tail;
    }
    diag->pole_proximity_samples += snapshot.pole_proximity_samples;
    if (snapshot.pole_proximity_samples > 0U &&
        snapshot.min_pole_distance < diag->min_pole_distance) {
        diag->min_pole_distance = snapshot.min_pole_distance;
    }
}
#endif

size_t sim_context_operator_count(const SimContext* context) {
    if (context == NULL) {
        return 0U;
    }
    return context->world.operators.count;
}

size_t sim_context_plan_operator_count(const SimContext* context) {
    if (context == NULL) {
        return 0U;
    }
    return context->scheduler.plan.count;
}

bool sim_context_plan_is_valid(const SimContext* context) {
    if (context == NULL) {
        return false;
    }
    return context->scheduler.plan_valid;
}

void sim_context_reset_continuity_counters(SimContext* context) {
    size_t count;

    if (context == NULL) {
        return;
    }

    count = context->world.field_count;
    if (context->runtime.field_dirty_counts != NULL && count > 0U) {
        (void) memset(context->runtime.field_dirty_counts, 0, count * sizeof(uint64_t));
    }
    if (context->runtime.field_stable_counts != NULL && count > 0U) {
        (void) memset(context->runtime.field_stable_counts, 0, count * sizeof(uint64_t));
    }

    context->runtime.current_step_warp_mask = 0U;
}

static void sim_context_capture_drift_field_stats(SimContext* context) {
    if (context == NULL || !sim_context_in_drift(context)) {
        return;
    }
    size_t field_count = context->world.field_count;
    if (field_count < 2U) {
        return;
    }

    bool snapshot_requested = false;
    bool stats_requested    = false;
    size_t limit            = context->runtime.continuity_capacity;
    if (limit > field_count) {
        limit = field_count;
    }
    if (context->runtime.drift_field_stats_requested != NULL) {
        for (size_t i = 0U; i < limit; ++i) {
            if (context->runtime.drift_field_stats_requested[i]) {
                stats_requested = true;
                break;
            }
        }
    }
    if (context->runtime.drift_field_snapshot_requested != NULL) {
        for (size_t i = 0U; i < limit; ++i) {
            if (context->runtime.drift_field_snapshot_requested[i]) {
                snapshot_requested = true;
                break;
            }
        }
    }

    if (!stats_requested && !snapshot_requested) {
        return;
    }

    if (context->runtime.drift_field_stats == NULL ||
        context->runtime.drift_field_stats_valid == NULL ||
        context->runtime.drift_field_stats_requested == NULL ||
        context->runtime.continuity_capacity < field_count) {
        if (sim_runtime_state_ensure_continuity_capacity(&context->runtime, field_count) !=
            SIM_RESULT_OK) {
            if (context->runtime.drift_field_stats_valid != NULL && limit > 0U) {
                (void) memset(context->runtime.drift_field_stats_valid,
                              0,
                              limit * sizeof(bool));
            }
            context->runtime.drift_field_stats_step_index = SIZE_MAX;
            return;
        }
    }

    if (stats_requested) {
        (void) memset(context->runtime.drift_field_stats_valid, 0, field_count * sizeof(bool));
        for (size_t i = 0U; i < field_count; ++i) {
            if (!context->runtime.drift_field_stats_requested[i]) {
                continue;
            }

            SimFieldStats               stats   = { 0 };
            SimFieldStatsComputeTimings timings = { 0 };
            SimField*                   field   = sim_context_field(context, i);
            if (field != NULL) {
                sim_field_stats_compute_with_config(field,
                                                    &stats,
                                                    &context->runtime.field_stats_config,
                                                    &timings);
                sim_field_stats_profile_record_compute(
                    &context->runtime.field_stats_profile,
                    i,
                    context->runtime.step_index + 1U,
                    SIM_FIELD_STATS_PROFILE_SOURCE_DRIFT_COMPUTE,
                    &timings);
                context->runtime.drift_field_stats_valid[i] = true;
            }
            context->runtime.drift_field_stats[i] = stats;
        }

        context->runtime.drift_field_stats_step_index = context->runtime.step_index + 1U;
    }

    if (snapshot_requested && context->runtime.drift_field_snapshots != NULL &&
        context->runtime.drift_field_snapshot_requested != NULL &&
        context->runtime.drift_field_snapshot_valid != NULL &&
        context->runtime.drift_field_snapshot_count != NULL &&
        context->runtime.drift_field_snapshot_capacity != NULL &&
        context->runtime.drift_field_snapshot_step_index != NULL) {
        for (size_t snapshot_index = 0U; snapshot_index < field_count; ++snapshot_index) {
            if (!context->runtime.drift_field_snapshot_requested[snapshot_index]) {
                continue;
            }

            SimField* field = sim_context_field(context, snapshot_index);
            if (field == NULL) {
                context->runtime.drift_field_snapshot_valid[snapshot_index] = false;
                context->runtime.drift_field_snapshot_count[snapshot_index] = 0U;
                continue;
            }

            SimFieldView view  = sim_field_view_from_field(field);
            size_t       count = view.count;
            if (count == 0U) {
                context->runtime.drift_field_snapshot_valid[snapshot_index] = false;
                context->runtime.drift_field_snapshot_count[snapshot_index] = 0U;
                continue;
            }

            if (count > context->runtime.drift_field_snapshot_capacity[snapshot_index] ||
                context->runtime.drift_field_snapshots[snapshot_index] == NULL) {
                SimComplexDouble* new_snapshot = (SimComplexDouble*) realloc(
                    context->runtime.drift_field_snapshots[snapshot_index],
                    count * sizeof(SimComplexDouble));
                if (new_snapshot == NULL) {
                    context->runtime.drift_field_snapshot_valid[snapshot_index] = false;
                    context->runtime.drift_field_snapshot_count[snapshot_index] = 0U;
                    continue;
                }
                context->runtime.drift_field_snapshots[snapshot_index]         = new_snapshot;
                context->runtime.drift_field_snapshot_capacity[snapshot_index] = count;
            }

            SimComplexDouble* dest = context->runtime.drift_field_snapshots[snapshot_index];
            if (view.type == SIM_FIELD_COMPLEX_DOUBLE) {
                const SimComplexDouble* src = (const SimComplexDouble*) view.data;
                memcpy(dest, src, count * sizeof(SimComplexDouble));
            } else if (view.type == SIM_FIELD_DOUBLE) {
                const double* src = (const double*) view.data;
                for (size_t i = 0U; i < count; ++i) {
                    dest[i].re = src[i];
                    dest[i].im = 0.0;
                }
            } else {
                context->runtime.drift_field_snapshot_valid[snapshot_index] = false;
                context->runtime.drift_field_snapshot_count[snapshot_index] = 0U;
                continue;
            }

            context->runtime.drift_field_snapshot_count[snapshot_index] = count;
            context->runtime.drift_field_snapshot_step_index[snapshot_index] =
                context->runtime.step_index + 1U;
            context->runtime.drift_field_snapshot_valid[snapshot_index] = true;
        }
    }
}

void sim_context_field_continuity_counts(const SimContext* context,
                                         size_t            field_index,
                                         uint64_t*         out_dirty,
                                         uint64_t*         out_stable) {
    if (out_dirty != NULL) {
        *out_dirty = 0ULL;
    }
    if (out_stable != NULL) {
        *out_stable = 0ULL;
    }

    if (context == NULL || field_index >= context->world.field_count) {
        return;
    }

    if (out_dirty != NULL && context->runtime.field_dirty_counts != NULL) {
        *out_dirty = context->runtime.field_dirty_counts[field_index];
    }
    if (out_stable != NULL && context->runtime.field_stable_counts != NULL) {
        *out_stable = context->runtime.field_stable_counts[field_index];
    }
}

static void sim_context_record_continuity_hit(SimContext* context, const SimOperator* op) {
    uint64_t mask;
    size_t   bit = 0U;
    bool     stabilized;
    bool     dirty;

    if (context == NULL || op == NULL || context->world.field_count == 0U) {
        return;
    }

    if (context->runtime.field_dirty_counts == NULL ||
        context->runtime.field_stable_counts == NULL) {
        return;
    }

    /* write_mask currently tracks the first 64 fields */
    mask = op->write_mask;
    if (mask == 0ULL) {
        return;
    }

    stabilized = (op->config.continuity == SIM_CONTINUITY_CLAMPED ||
                  op->config.continuity == SIM_CONTINUITY_LIMITED);
    dirty      = (op->config.continuity == SIM_CONTINUITY_NONE ||
             op->config.continuity == SIM_CONTINUITY_STRICT);

    if (!stabilized && !dirty) {
        return;
    }

    while (mask != 0ULL && bit < CONTINUITY_MASK_WIDTH) {
        if (mask & 1ULL) {
            if (bit < context->world.field_count) {
                if (stabilized) {
                    __sync_fetch_and_add(&context->runtime.field_stable_counts[bit], 1ULL);
                } else if (dirty) {
                    __sync_fetch_and_add(&context->runtime.field_dirty_counts[bit], 1ULL);
                }
            }
        }
        mask >>= 1U;
        bit += 1U;
    }
}

static void sim_context_apply_override_to_operator(SimContext* context, SimOperator* op) {
    if (context == NULL || op == NULL) {
        return;
    }

    sim_operator_config_set(op, &context->continuity_override);
}

static uint32_t sim_context_warp_level_bit(SimWarpLevel level) {
    if (level <= SIM_WARP_LEVEL_NONE || level > SIM_WARP_LEVEL_LEVEL2) {
        return 0U;
    }
    return (1U << (unsigned) level);
}

static void sim_context_accumulate_continuity_totals(const SimContext* context,
                                                     uint64_t*         out_dirty,
                                                     uint64_t*         out_stable) {
    uint64_t dirty  = 0ULL;
    uint64_t stable = 0ULL;
    size_t   count  = 0U;
    size_t   limit  = 0U;

    if (context != NULL) {
        count = context->world.field_count;
        limit = (context->runtime.continuity_capacity < count)
                    ? context->runtime.continuity_capacity
                    : count;
    }

    if (context != NULL && context->runtime.field_dirty_counts != NULL) {
        for (size_t i = 0U; i < limit; ++i) {
            dirty += context->runtime.field_dirty_counts[i];
        }
    }

    if (context != NULL && context->runtime.field_stable_counts != NULL) {
        for (size_t i = 0U; i < limit; ++i) {
            stable += context->runtime.field_stable_counts[i];
        }
    }

    if (out_dirty != NULL) {
        *out_dirty = dirty;
    }
    if (out_stable != NULL) {
        *out_stable = stable;
    }
}

static SimResult sim_context_resolve_representations(SimContext* context) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperatorRegistry* registry = &context->world.operators;
    for (size_t i = 0U; i < registry->count; ++i) {
        SimOperator*          op                  = &registry->records[i];
        const SimOperatorInfo info                = op->info;
        const bool            need_complex_inputs = info.representation.requires_complex_input;
        const bool            output_is_complex =
            (info.representation.value_kind != SIM_FIELD_VALUE_REAL_SCALAR);
        const bool need_complex_outputs =
            info.representation.requires_complex_representation || output_is_complex;

        if (need_complex_inputs) {
            uint64_t mask = op->read_mask;
            while (mask != 0ULL) {
                unsigned int bit = __builtin_ctzll(mask);
                mask &= ~(1ULL << bit);
                SimField*       field = sim_context_field(context, (size_t) bit);
                SimScalarDomain field_domain;
                if (field == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                field_domain = sim_scalar_domain_from_field(field);
                if (!sim_scalar_domain_is_complex(field_domain)) {
                    fprintf(stderr,
                            "[ERROR] representation resolver: op '%s' requires complex inputs but "
                            "field %u is real\n",
                            op->name,
                            bit);
                    return SIM_RESULT_TYPE_MISMATCH;
                }
            }
            for (size_t ri = 0U; ri < op->read_index_count; ++ri) {
                const size_t    idx   = op->read_indices[ri];
                SimField*       field = sim_context_field(context, idx);
                SimScalarDomain field_domain;
                if (field == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                field_domain = sim_scalar_domain_from_field(field);
                if (!sim_scalar_domain_is_complex(field_domain)) {
                    fprintf(stderr,
                            "[ERROR] representation resolver: op '%s' requires complex inputs but "
                            "field %zu is real\n",
                            op->name,
                            idx);
                    return SIM_RESULT_TYPE_MISMATCH;
                }
            }
        }

        if (need_complex_outputs) {
            uint64_t mask = op->write_mask;
            while (mask != 0ULL) {
                unsigned int bit = __builtin_ctzll(mask);
                mask &= ~(1ULL << bit);
                SimField*       field = sim_context_field(context, (size_t) bit);
                SimScalarDomain field_domain;
                if (field == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                field_domain = sim_scalar_domain_from_field(field);
                if (!sim_scalar_domain_is_complex(field_domain)) {
                    return SIM_RESULT_TYPE_MISMATCH;
                }
            }
            for (size_t wi = 0U; wi < op->write_index_count; ++wi) {
                const size_t    idx   = op->write_indices[wi];
                SimField*       field = sim_context_field(context, idx);
                SimScalarDomain field_domain;
                if (field == NULL) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                field_domain = sim_scalar_domain_from_field(field);
                if (!sim_scalar_domain_is_complex(field_domain)) {
                    return SIM_RESULT_TYPE_MISMATCH;
                }
            }
        }

        /* Domain ownership:
         * - Field domains are owned by field creation / explicit conversion operators.
         * - The runtime does not implicitly adjust/flip domains during planning.
         */
    }

    return SIM_RESULT_OK;
}

static SimResult sim_context_validate_field_representations(SimContext* context) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < context->world.field_count; ++i) {
        SimField*              field = &context->world.fields[i];
        SimFieldRepresentation repr  = field->repr;
        SimResult              rc    = sim_field_validate_representation(field, repr);
        if (rc != SIM_RESULT_OK) {
            fprintf(stderr,
                    "[ERROR] representation validator: field %zu has invalid representation "
                    "(domain=%s, value_kind=%s, code=%d)\n",
                    i,
                    sim_field_domain_name(repr.domain),
                    sim_field_value_kind_name(repr.value_kind),
                    (int) rc);
            return rc;
        }
    }

    return SIM_RESULT_OK;
}

static uint32_t sim_context_active_warp_mask(const SimContext* context) {
    if (context == NULL) {
        return 0U;
    }
    return context->runtime.current_step_warp_mask;
}

SimResult sim_context_init(SimContext* context) {
    return sim_context_init_with_universe(context, NULL);
}

// New function that takes universe spec
SimResult sim_context_init_with_universe(SimContext*            context,
                                         const SimUniverseSpec* universe_spec) {
    SimResult result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    memset(context, 0, sizeof(*context));

    result = sim_world_init(&context->world, universe_spec);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_runtime_state_init(&context->runtime);
    if (result != SIM_RESULT_OK) {
        sim_world_destroy(&context->world);
        return result;
    }

    result = sim_integrator_state_init(&context->integrators);
    if (result != SIM_RESULT_OK) {
        sim_runtime_state_destroy(&context->runtime);
        sim_world_destroy(&context->world);
        return result;
    }

    result = sim_scheduler_plan_init(&context->scheduler);
    if (result != SIM_RESULT_OK) {
        sim_integrator_state_destroy(&context->integrators);
        sim_runtime_state_destroy(&context->runtime);
        sim_world_destroy(&context->world);
        return result;
    }

    result = sim_neural_model_registry_init(&context->neural_models);
    if (result != SIM_RESULT_OK) {
        sim_scheduler_plan_destroy(&context->scheduler);
        sim_integrator_state_destroy(&context->integrators);
        sim_runtime_state_destroy(&context->runtime);
        sim_world_destroy(&context->world);
        return result;
    }

    context->continuity_override_enabled      = false;
    context->continuity_override              = sim_operator_config_defaults();
    context->preferred_gui_visual_mode        = -1;
    context->preferred_gui_phase_mode         = -1;
    context->preferred_gui_visual_auto_scale  = -1;
    context->preferred_gui_visual_scale       = -1.0;
    context->preferred_gui_visual_field_index = -1;
    memset(context->preferred_gui_visual_field_selected,
           0,
           sizeof(context->preferred_gui_visual_field_selected));
    if (SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY > 0U) {
        context->preferred_gui_visual_field_selected[0] = true;
    }
    context->diag.fallback_user     = NULL;
    context->diag.fallback_userdata = NULL;
    context->diag.fault_lock        = 0;
    context->diag.fault_count       = 0ULL;
    memset(&context->diag.fault_last, 0, sizeof(context->diag.fault_last));
    context->diag.fault_function[0] = '\0';
    sim_context_reset_kernel_dispatch_diag(&context->diag);
#if SIM_DIAGNOSTICS
    context->diag.reflection_count         = 0ULL;
    context->diag.recurrence_shift_samples = 0ULL;
    context->diag.max_recurrence_shift     = 0.0;
    context->diag.stirling_tail_samples    = 0ULL;
    context->diag.max_stirling_tail        = 0.0;
    context->diag.pole_proximity_samples   = 0ULL;
    context->diag.min_pole_distance        = INFINITY;
#endif
    context->profiler_ready = false;
    memset(&context->profiler, 0, sizeof(context->profiler));
    context->base_seed            = SIM_SEED_DEFAULT;
    context->representation_mode  = SIM_REPRESENTATION_MODE_RELAXED;
    context->memory_limits        = sim_context_memory_limits_default();
    context->bytes_fields_in_use  = 0U;
    context->bytes_scratch_in_use = 0U;
    context->bytes_total_in_use   = 0U;
    context->log_fn               = NULL;
    context->log_userdata         = NULL;
    flux_lens_init(&context->runtime.flux_lens);
    flux_lens_workspace_init(&context->runtime.flux_workspace);

    return SIM_RESULT_OK;
}

void sim_context_set_seed(SimContext* context, uint64_t seed) {
    if (context == NULL) {
        return;
    }
    context->base_seed = sim_seed_normalize(seed);
}

uint64_t sim_context_seed(const SimContext* context) {
    if (context == NULL) {
        return SIM_SEED_DEFAULT;
    }
    return sim_seed_normalize(context->base_seed);
}

void sim_context_set_representation_mode(SimContext* context, SimRepresentationMode mode) {
    if (context == NULL) {
        return;
    }
    context->representation_mode = sim_context_representation_mode_normalize(mode);
}

SimRepresentationMode sim_context_representation_mode(const SimContext* context) {
    if (context == NULL) {
        return SIM_REPRESENTATION_MODE_RELAXED;
    }
    return sim_context_representation_mode_normalize(context->representation_mode);
}

void sim_context_set_logger(SimContext* context,
                            void (*log_fn)(SimLogLevel level, const char* message, void* userdata),
                            void* userdata) {
    if (context == NULL) {
        return;
    }
    context->log_fn       = log_fn;
    context->log_userdata = userdata;
}

void sim_context_log_warning(const SimContext* context, const char* fmt, ...) {
    if (context == NULL || fmt == NULL) {
        return;
    }

    char    buffer[SIM_ASYNC_LOGGER_MESSAGE_MAX + 1U];
    va_list args;
    va_start(args, fmt);
    (void) vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    buffer[SIM_ASYNC_LOGGER_MESSAGE_MAX] = '\0';

    if (context->log_fn != NULL) {
        context->log_fn(SIM_LOG_LEVEL_WARN, buffer, context->log_userdata);
    } else {
        fprintf(stderr, "[WARN] %s\n", buffer);
    }
}

bool sim_context_allows_determinism(const SimContext* context, SimDeterminismFlags flags) {
    if (context == NULL) {
        return false;
    }
    return sim_context_allows_determinism_mode(sim_context_representation_mode(context), flags);
}

bool sim_context_kernel_allowed(const SimContext*   context,
                                uint64_t            required_features,
                                SimDeterminismFlags determinism_flags) {
    if (context == NULL) {
        return false;
    }
    return sim_context_kernel_allowed_mode(
        context, sim_context_representation_mode(context), required_features, determinism_flags);
}

bool sim_context_kernel_allowed_mode(const SimContext*     context,
                                     SimRepresentationMode mode,
                                     uint64_t              required_features,
                                     SimDeterminismFlags   determinism_flags) {
    if (context == NULL) {
        return false;
    }

    mode = sim_context_representation_mode_normalize(mode);

    SimBackend* backend = sim_context_backend((SimContext*) context);
    if (backend == NULL) {
        return false;
    }

    if ((required_features != 0ULL) &&
        ((backend->features & required_features) != required_features)) {
        return false;
    }

    return sim_context_allows_determinism_mode(mode, determinism_flags);
}

void sim_context_set_memory_limits(SimContext* context, const SimContextMemoryLimits* limits) {
    if (context == NULL || limits == NULL) {
        return;
    }
    context->memory_limits = *limits;
}

void sim_context_get_memory_limits(const SimContext* context, SimContextMemoryLimits* out_limits) {
    if (context == NULL || out_limits == NULL) {
        return;
    }
    *out_limits = context->memory_limits;
}

void sim_context_memory_usage(const SimContext* context,
                              size_t*           out_fields,
                              size_t*           out_scratch,
                              size_t*           out_total) {
    if (context == NULL) {
        return;
    }
    if (out_fields != NULL) {
        *out_fields = context->bytes_fields_in_use;
    }
    if (out_scratch != NULL) {
        *out_scratch = context->bytes_scratch_in_use;
    }
    if (out_total != NULL) {
        *out_total = context->bytes_total_in_use;
    }
}

SimResult sim_context_check_field_limits(const SimContext* context,
                                         size_t            element_count,
                                         size_t            field_bytes) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (element_count == 0U || field_bytes == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimContextMemoryLimits* limits = &context->memory_limits;
    if (limits->max_field_elements > 0U && element_count > limits->max_field_elements) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (limits->max_field_bytes > 0U && field_bytes > limits->max_field_bytes) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (limits->max_fields > 0U && context->world.field_count >= limits->max_fields) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (limits->max_total_field_bytes > 0U) {
        if (field_bytes > limits->max_total_field_bytes) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        if (context->bytes_fields_in_use > limits->max_total_field_bytes - field_bytes) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    } else {
        if (context->bytes_fields_in_use > SIZE_MAX - field_bytes) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_context_reserve_scratch(SimContext* context, size_t bytes) {
    size_t new_scratch = 0U;
    size_t new_total   = 0U;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (bytes == 0U) {
        return SIM_RESULT_OK;
    }
    if (!sim_context_add_size(context->bytes_scratch_in_use, bytes, &new_scratch)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (!sim_context_add_size(context->bytes_fields_in_use, new_scratch, &new_total)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    context->bytes_scratch_in_use = new_scratch;
    context->bytes_total_in_use   = new_total;
    return SIM_RESULT_OK;
}

void sim_context_release_scratch(SimContext* context, size_t bytes) {
    size_t new_total = 0U;

    if (context == NULL || bytes == 0U) {
        return;
    }

    if (bytes >= context->bytes_scratch_in_use) {
        context->bytes_scratch_in_use = 0U;
    } else {
        context->bytes_scratch_in_use -= bytes;
    }

    if (!sim_context_add_size(
            context->bytes_fields_in_use, context->bytes_scratch_in_use, &new_total)) {
        context->bytes_total_in_use = SIZE_MAX;
        return;
    }
    context->bytes_total_in_use = new_total;
}

void sim_context_destroy(SimContext* context) {
    if (context == NULL) {
        return;
    }

    sim_scheduler_plan_destroy(&context->scheduler);
    sim_integrator_state_destroy(&context->integrators);

    if (context->profiler_ready) {
        sim_profiler_destroy(&context->profiler);
        context->profiler_ready = false;
    }

    flux_lens_release(&context->runtime.flux_lens, &context->runtime.flux_workspace, context);
    sim_runtime_state_destroy(&context->runtime);
    sim_world_destroy(&context->world);
    sim_neural_model_registry_destroy(&context->neural_models);

    context->diag.fallback_user     = NULL;
    context->diag.fallback_userdata = NULL;
    context->diag.fault_lock        = 0;
    context->diag.fault_count       = 0ULL;
    memset(&context->diag.fault_last, 0, sizeof(context->diag.fault_last));
    context->diag.fault_function[0] = '\0';
    sim_context_reset_kernel_dispatch_diag(&context->diag);

    context->scheduler.plan_valid = false;
    context->scheduler.backend    = NULL;
    context->bytes_fields_in_use  = 0U;
    context->bytes_scratch_in_use = 0U;
    context->bytes_total_in_use   = 0U;
}

void sim_context_set_preferred_visual_mode(SimContext* context, int mode) {
    if (context == NULL) {
        return;
    }
    context->preferred_gui_visual_mode = mode;
}

void sim_context_set_preferred_phase_mode(SimContext* context, int mode) {
    if (context == NULL) {
        return;
    }
    context->preferred_gui_phase_mode = mode;
}

int sim_context_preferred_visual_mode(const SimContext* context) {
    if (context == NULL) {
        return -1;
    }
    return context->preferred_gui_visual_mode;
}

int sim_context_preferred_phase_mode(const SimContext* context) {
    if (context == NULL) {
        return -1;
    }
    return context->preferred_gui_phase_mode;
}

void sim_context_set_preferred_visual_auto_scale(SimContext* context, int enabled) {
    if (context == NULL) {
        return;
    }
    if (enabled < 0) {
        context->preferred_gui_visual_auto_scale = -1;
    } else {
        context->preferred_gui_visual_auto_scale = enabled ? 1 : 0;
    }
}

int sim_context_preferred_visual_auto_scale(const SimContext* context) {
    if (context == NULL) {
        return -1;
    }
    return context->preferred_gui_visual_auto_scale;
}

void sim_context_set_preferred_visual_scale(SimContext* context, double scale) {
    if (context == NULL) {
        return;
    }
    if (!isfinite(scale) || scale <= 0.0) {
        context->preferred_gui_visual_scale = -1.0;
    } else {
        context->preferred_gui_visual_scale = scale;
    }
}

double sim_context_preferred_visual_scale(const SimContext* context) {
    if (context == NULL) {
        return -1.0;
    }
    return context->preferred_gui_visual_scale;
}

void sim_context_set_preferred_visual_field_enabled(SimContext* context,
                                                    size_t      field_index,
                                                    bool        enabled) {
    bool any_selected = false;
    if (context == NULL || field_index >= SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY) {
        return;
    }

    context->preferred_gui_visual_field_selected[field_index] = enabled;
    for (size_t i = 0U; i < SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY; ++i) {
        if (context->preferred_gui_visual_field_selected[i]) {
            any_selected = true;
            break;
        }
    }
    if (!any_selected && SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY > 0U) {
        context->preferred_gui_visual_field_selected[0] = true;
    }
}

bool sim_context_preferred_visual_field_enabled(const SimContext* context, size_t field_index) {
    if (context == NULL || field_index >= SIM_CONTEXT_PREFERRED_VISUAL_FIELD_CAPACITY) {
        return false;
    }
    return context->preferred_gui_visual_field_selected[field_index];
}

SimResult sim_context_add_field(SimContext* context, SimField* field, size_t* out_index) {
    SimResult result;
    size_t    index;
    size_t    element_count;
    size_t    field_bytes;
    size_t    next_fields = 0U;
    size_t    next_total  = 0U;

    if (context == NULL || field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field->data == NULL || field->element_size == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    element_count = sim_field_element_count(&field->layout);
    field_bytes   = sim_field_bytes(field);
    if (element_count == 0U || field_bytes == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    result = sim_context_check_field_limits(context, element_count, field_bytes);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    if (!sim_context_add_size(context->bytes_fields_in_use, field_bytes, &next_fields)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (!sim_context_add_size(next_fields, context->bytes_scratch_in_use, &next_total)) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    result = sim_world_reserve_fields(&context->world, 1U);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_runtime_state_ensure_continuity_capacity(&context->runtime,
                                                          context->world.field_count + 1U);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    index = context->world.field_count;

    context->world.fields[index] = *field;
    context->world.field_count += 1U;

    if (context->runtime.field_dirty_counts != NULL) {
        context->runtime.field_dirty_counts[index] = 0ULL;
    }
    if (context->runtime.field_stable_counts != NULL) {
        context->runtime.field_stable_counts[index] = 0ULL;
    }
    if (context->runtime.field_phase_ema != NULL) {
        context->runtime.field_phase_ema[index] = 0.0;
    }
    if (context->runtime.field_phase_last_time != NULL) {
        context->runtime.field_phase_last_time[index] = context->runtime.time_accumulated;
    }
    if (context->runtime.field_phase_lock_state != NULL) {
        context->runtime.field_phase_lock_state[index] = 0U;
    }
    if (context->runtime.field_phase_initialized != NULL) {
        context->runtime.field_phase_initialized[index] = 0U;
    }
    if (context->runtime.field_stats_cache != NULL) {
        context->runtime.field_stats_cache[index] = (SimFieldStats) { 0 };
    }
    if (context->runtime.field_stats_cache_step_index != NULL) {
        context->runtime.field_stats_cache_step_index[index] = SIZE_MAX;
    }
    if (context->runtime.field_topology_cache != NULL &&
        index < context->runtime.continuity_capacity) {
        sim_field_topology_runtime_init(&context->runtime.field_topology_cache[index]);
    }
    if (context->runtime.field_health_nan_counts != NULL) {
        context->runtime.field_health_nan_counts[index] = 0U;
    }
    if (context->runtime.field_health_inf_counts != NULL) {
        context->runtime.field_health_inf_counts[index] = 0U;
    }
    if (context->runtime.field_health_cache_step_index != NULL) {
        context->runtime.field_health_cache_step_index[index] = SIZE_MAX;
    }
    if (context->runtime.drift_field_stats != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_stats[index] = (SimFieldStats) { 0 };
    }
    if (context->runtime.drift_field_stats_valid != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_stats_valid[index] = false;
    }
    if (context->runtime.drift_field_stats_requested != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_stats_requested[index] = false;
    }
    context->runtime.drift_field_stats_step_index = SIZE_MAX;
    if (context->runtime.drift_field_snapshots != NULL &&
        index < context->runtime.continuity_capacity) {
        free(context->runtime.drift_field_snapshots[index]);
        context->runtime.drift_field_snapshots[index] = NULL;
    }
    if (context->runtime.drift_field_snapshot_capacity != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_snapshot_capacity[index] = 0U;
    }
    if (context->runtime.drift_field_snapshot_count != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_snapshot_count[index] = 0U;
    }
    if (context->runtime.drift_field_snapshot_step_index != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_snapshot_step_index[index] = 0U;
    }
    if (context->runtime.drift_field_snapshot_valid != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_snapshot_valid[index] = false;
    }
    if (context->runtime.drift_field_snapshot_requested != NULL &&
        index < context->runtime.continuity_capacity) {
        context->runtime.drift_field_snapshot_requested[index] = false;
    }

    field->data               = NULL;
    field->layout.shape       = NULL;
    field->layout.strides     = NULL;
    field->layout.rank        = 0U;
    field->element_size       = 0U;
    field->owns_data          = false;
    field->allocator.allocate = NULL;
    field->allocator.release  = NULL;
    field->allocator.userdata = NULL;
    field->storage            = SIM_FIELD_STORAGE_ROW_MAJOR;
    field->magic              = 0U;

    sim_scheduler_plan_invalidate(&context->scheduler);

    context->bytes_fields_in_use = next_fields;
    context->bytes_total_in_use  = next_total;

    if (out_index != NULL) {
        *out_index = index;
    }

    return SIM_RESULT_OK;
}

SimField* sim_context_field(SimContext* context, size_t index) {
    if (context == NULL || index >= context->world.field_count) {
        return NULL;
    }
    return &context->world.fields[index];
}

size_t sim_context_field_count(const SimContext* context) {
    if (context == NULL) {
        return 0U;
    }
    return context->world.field_count;
}

SimResult sim_context_register_operator(SimContext*                  context,
                                        const SimOperatorDescriptor* descriptor,
                                        size_t*                      out_index) {
    SimResult result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    result = sim_operator_registry_register(&context->world.operators, descriptor, out_index);
    if (result == SIM_RESULT_OK) {
        sim_scheduler_plan_invalidate(&context->scheduler);
        size_t assigned_index =
            (out_index != NULL) ? *out_index : (context->world.operators.count - 1U);
        SimOperator* slot = sim_operator_registry_get(&context->world.operators, assigned_index);

        if (slot != NULL) {
            if (context->continuity_override_enabled) {
                sim_context_apply_override_to_operator(context, slot);
            }
        }
    }
    return result;
}

SimResult sim_context_add_operator_dependency(SimContext* context,
                                              size_t      operator_index,
                                              size_t      dependency_index) {
    SimOperator* op;
    size_t*      deps;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (operator_index >= context->world.operators.count ||
        dependency_index >= context->world.operators.count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (operator_index == dependency_index) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    for (size_t i = 0U; i < op->dependency_count; ++i) {
        if (op->dependencies[i] == dependency_index) {
            return SIM_RESULT_OK;
        }
    }

    deps = (size_t*) realloc(op->dependencies, (op->dependency_count + 1U) * sizeof(size_t));
    if (deps == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    op->dependencies                       = deps;
    op->dependencies[op->dependency_count] = dependency_index;
    op->dependency_count += 1U;
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}

SimResult sim_context_prepare_plan(SimContext* context) {
    SimResult result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    result = sim_context_prepare_complex_fields(context);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_context_resolve_representations(context);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = sim_context_validate_field_representations(context);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    if (context->scheduler.plan_valid) {
        return SIM_RESULT_OK;
    }

    result = sim_operator_resolve_plan(&context->world.operators, &context->scheduler.plan);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    context->scheduler.plan_valid = true;
    return SIM_RESULT_OK;
}

static SimResult sim_context_launch_cpu_fallback(KernelIR* kernel) {
    SimBackend fallback = { 0 };
    SimResult  result;

    if (kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    fallback.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&fallback);
    result = fallback.last_error;
    if (result != SIM_RESULT_OK) {
        backend_destroy(&fallback);
        return result;
    }

    backend_launch(&fallback, kernel);
    result = fallback.last_error;
    backend_destroy(&fallback);
    return result;
}

static SimResult sim_context_execute_kernel(SimContext* context, SimOperator* op) {
    size_t             i;
    SimOperatorKernel* kernel;
    SimBackend*        requested_backend;
    SimResult          result;
    const char*        fallback_reason = NULL;

    if (context == NULL || op == NULL || op->kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (context->scheduler.backend == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    kernel            = op->kernel;
    requested_backend = context->scheduler.backend;

    if (kernel->kernel.params != NULL && kernel->kernel.param_count > SIM_IR_PARAM_DT) {
        double dt = sim_context_timestep(context);
        if (!isfinite(dt)) {
            dt = 0.0;
        }
        kernel->kernel.params[SIM_IR_PARAM_DT] = dt;
        if (kernel->kernel.param_count > SIM_IR_PARAM_STEP_INDEX) {
            kernel->kernel.params[SIM_IR_PARAM_STEP_INDEX] =
                (double) sim_context_step_index(context);
        }
        if (kernel->kernel.param_count > SIM_IR_PARAM_SQRT_DT) {
            kernel->kernel.params[SIM_IR_PARAM_SQRT_DT] = (dt > 0.0) ? sqrt(dt) : 0.0;
        }
        if (kernel->kernel.param_count > SIM_IR_PARAM_TIME) {
            kernel->kernel.params[SIM_IR_PARAM_TIME] = sim_context_time(context);
        }
    }

    for (i = 0U; i < kernel->binding_count; ++i) {
        size_t    field_index = kernel->binding_map[i].context_field_index;
        SimField* field       = sim_context_field(context, field_index);
        if (field == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        kernel->bindings[i].field_index = kernel->binding_map[i].ir_field_index;
        kernel->bindings[i].field       = field;
        kernel->bindings[i].shape       = sim_field_shape(field);
        kernel->bindings[i].strides     = sim_field_strides(field);
        kernel->bindings[i].rank        = sim_field_rank(field);
    }

    if (sim_kernel_requires_boundary_feature(&kernel->kernel) &&
        !backend_supports_feature(requested_backend, SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS)) {
        result = sim_context_launch_cpu_fallback(&kernel->kernel);
        sim_context_record_kernel_dispatch(context,
                                           op,
                                           requested_backend,
                                           SIM_DIAGNOSTIC_BACKEND_CPU,
                                           sim_context_backend_diag_kind(requested_backend) !=
                                               SIM_DIAGNOSTIC_BACKEND_CPU,
                                           "boundary_diff_not_supported");
        return result;
    }

    if (sim_kernel_requires_warp_profile_fallback(&kernel->kernel, requested_backend)) {
        result = sim_context_launch_cpu_fallback(&kernel->kernel);
        sim_context_record_kernel_dispatch(context,
                                           op,
                                           requested_backend,
                                           SIM_DIAGNOSTIC_BACKEND_CPU,
                                           sim_context_backend_diag_kind(requested_backend) !=
                                               SIM_DIAGNOSTIC_BACKEND_CPU,
                                           "warp_profile_not_supported");
        return result;
    }

    if (!backend_supports_features(requested_backend, kernel->kernel.required_features)) {
        result = sim_context_launch_cpu_fallback(&kernel->kernel);
        sim_context_record_kernel_dispatch(context,
                                           op,
                                           requested_backend,
                                           SIM_DIAGNOSTIC_BACKEND_CPU,
                                           sim_context_backend_diag_kind(requested_backend) !=
                                               SIM_DIAGNOSTIC_BACKEND_CPU,
                                           "missing_backend_features");
        return result;
    }

    fallback_reason = sim_context_known_backend_fallback_reason(requested_backend, &kernel->kernel);
    if (fallback_reason != NULL) {
        result = sim_context_launch_cpu_fallback(&kernel->kernel);
        sim_context_record_kernel_dispatch(context,
                                           op,
                                           requested_backend,
                                           SIM_DIAGNOSTIC_BACKEND_CPU,
                                           sim_context_backend_diag_kind(requested_backend) !=
                                               SIM_DIAGNOSTIC_BACKEND_CPU,
                                           fallback_reason);
        return result;
    }

    backend_launch(requested_backend, &kernel->kernel);
    result = requested_backend->last_error;

    if (result == SIM_RESULT_NOT_SUPPORTED) {
        SimResult fallback_result = sim_context_launch_cpu_fallback(&kernel->kernel);
        sim_context_record_kernel_dispatch(context,
                                           op,
                                           requested_backend,
                                           SIM_DIAGNOSTIC_BACKEND_CPU,
                                           sim_context_backend_diag_kind(requested_backend) !=
                                               SIM_DIAGNOSTIC_BACKEND_CPU,
                                           "unsupported_ir_subset");
        return fallback_result;
    }

    sim_context_record_kernel_dispatch(context,
                                       op,
                                       requested_backend,
                                       sim_context_backend_diag_kind(requested_backend),
                                       false,
                                       NULL);
    return result;
}

SimResult sim_context_apply_operator(SimContext* context, SimOperator* op) {
    SimResult           result;
    bool                did_save           = false;
    bool                in_drift           = sim_context_in_drift(context);
    SimInvariantSample* invariant_baseline = NULL;
    size_t              invariant_count    = 0U;

    if (context == NULL || op == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (in_drift && op->save_state != NULL) {
        result = op->save_state(context, op, op->userdata);
        if (result != SIM_RESULT_OK) {
            return result;
        }
        did_save = true;
    }

    (void) sim_context_capture_invariants(context, op, &invariant_baseline, &invariant_count);

    if (op->kernel != NULL) {
        result = sim_context_execute_kernel(context, op);
    } else if (op->evaluate != NULL) {
        result = op->evaluate(context, op, op->userdata);
    } else {
        result = SIM_RESULT_INVALID_ARGUMENT;
    }

    if (in_drift && did_save && op->restore_state != NULL) {
        SimResult restore_result = op->restore_state(context, op, op->userdata);
        if (result == SIM_RESULT_OK) {
            result = restore_result;
        }
    }

    if (result == SIM_RESULT_OK && !sim_context_in_drift(context)) {
        sim_context_apply_norm_budget(context, op);
        sim_context_record_continuity_hit(context, op);
        uint32_t warp_bit = sim_context_warp_level_bit(op->info.warp_level);
        if (warp_bit != 0U) {
            (void) __sync_fetch_and_or(&context->runtime.current_step_warp_mask, warp_bit);
        }
    }

    free(invariant_baseline);

    return result;
}

SimResult sim_context_execute_prepared(SimContext* context) {
    SimResult result;
    size_t    i;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!context->scheduler.plan_valid) {
        return SIM_RESULT_INVALID_STATE;
    }

    sim_context_reset_continuity_counters(context);

    bool profiler_ready = sim_context_configure_profiler(context, context->scheduler.plan.count);

    if (profiler_ready) {
        sim_profiler_begin_frame(&context->profiler);
    }

    for (i = 0U; i < context->scheduler.plan.count; ++i) {
        size_t       op_index = context->scheduler.plan.order[i];
        SimOperator* op       = sim_operator_registry_get(&context->world.operators, op_index);
        if (op == NULL) {
            result = SIM_RESULT_INVALID_ARGUMENT;
            return result;
        }

        uint64_t start_ns = profiler_ready ? sim_context_now_ns() : 0ULL;
        result            = sim_context_apply_operator(context, op);
        uint64_t end_ns   = profiler_ready ? sim_context_now_ns() : 0ULL;

        if (profiler_ready) {
            sim_profiler_record_operator(
                &context->profiler, op_index, (end_ns >= start_ns) ? (end_ns - start_ns) : 0ULL);
        }

        if (result != SIM_RESULT_OK) {
            const char* op_name = (op->name[0] != '\0') ? op->name : "<unnamed>";
            fprintf(stderr,
                    "[ERROR] sim_context_execute: operator '%s' failed with %d\n",
                    op_name,
                    (int) result);
            return result;
        }
    }

    if (profiler_ready) {
        sim_profiler_end_frame(&context->profiler);
    }

    if (sim_context_in_drift(context)) {
        sim_context_capture_drift_field_stats(context);
    }

    result = SIM_RESULT_OK;
    return result;
}

SimResult sim_context_execute(SimContext* context) {
    SimResult result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    result = sim_context_prepare_plan(context);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    return sim_context_execute_prepared(context);
}

SimIRBuilder* sim_context_ir_builder(SimContext* context) {
    if (context == NULL) {
        return NULL;
    }
    sim_ir_builder_reset(&context->scheduler.ir_builder);
    /* Sync builder default boundary with the current context override so helpers inherit it. */
    sim_ir_builder_set_default_boundary(&context->scheduler.ir_builder,
                                        context->continuity_override.boundary);
    return &context->scheduler.ir_builder;
}

static bool sim_kernel_requires_boundary_feature(const KernelIR* kernel) {
    if (kernel == NULL || kernel->builder == NULL) {
        return false;
    }

    for (size_t i = 0; i < kernel->builder->count; ++i) {
        const SimIRNode* node = &kernel->builder->nodes[i];
        if (node != NULL && node->type == SIM_IR_NODE_DIFF) {
            if (node->data.diff.method != SIM_IR_DIFF_METHOD_AUTO) {
                return true;
            }
            switch (node->data.diff.boundary) {
                case SIM_IR_BOUNDARY_NEUMANN:
                    break;
                case SIM_IR_BOUNDARY_DIRICHLET:
                case SIM_IR_BOUNDARY_PERIODIC:
                case SIM_IR_BOUNDARY_REFLECTIVE:
                default:
                    return true;
            }
        }
    }
    return false;
}

static bool sim_kernel_contains_node_type(const KernelIR* kernel, SimIRNodeType type) {
    if (kernel == NULL || kernel->builder == NULL) {
        return false;
    }

    for (size_t i = 0; i < kernel->builder->count; ++i) {
        const SimIRNode* node = &kernel->builder->nodes[i];
        if (node != NULL && node->type == type) {
            return true;
        }
    }

    return false;
}

static bool sim_kernel_uses_integer_domain(const KernelIR* kernel) {
    if (kernel == NULL) {
        return false;
    }

    if (kernel->builder != NULL && kernel->outputs != NULL) {
        for (size_t i = 0; i < kernel->output_count; ++i) {
            SimIRNodeId output_id = kernel->outputs[i].expression;
            if (output_id != SIM_IR_INVALID_NODE && output_id < kernel->builder->count) {
                const SimIRNode* node = &kernel->builder->nodes[output_id];
                if (node != NULL &&
                    sim_scalar_domain_is_integer(sim_ir_type_scalar_domain(node->value_type))) {
                    return true;
                }
            }
        }
    }

    if (kernel->bindings != NULL) {
        for (size_t i = 0; i < kernel->binding_count; ++i) {
            const SimKernelIRBinding* binding = &kernel->bindings[i];
            if (binding != NULL && binding->field != NULL &&
                sim_scalar_domain_is_integer(sim_scalar_domain_from_field(binding->field))) {
                return true;
            }
        }
    }

    return false;
}

static bool sim_kernel_requires_warp_profile_fallback(const KernelIR*   kernel,
                                                      const SimBackend* backend) {
    if (kernel == NULL || kernel->builder == NULL || backend == NULL) {
        return false;
    }

    if (backend->type == SIM_BACKEND_TYPE_CPU) {
        return false;
    }

    for (size_t i = 0; i < kernel->builder->count; ++i) {
        const SimIRNode* node = &kernel->builder->nodes[i];
        if (node != NULL && node->type == SIM_IR_NODE_WARP) {
            switch (node->data.warp.profile) {
                case SIM_IR_WARP_PROFILE_DIGAMMA:
                case SIM_IR_WARP_PROFILE_TRIGAMMA:
                    break;
                default:
                    return true;
            }
        }
    }

    return false;
}

static const char* sim_context_known_backend_fallback_reason(const SimBackend* backend,
                                                             const KernelIR*   kernel) {
    if (backend == NULL || kernel == NULL) {
        return NULL;
    }

    if (backend->type == SIM_BACKEND_TYPE_CPU) {
        return NULL;
    }

    if ((backend->type == SIM_BACKEND_TYPE_METAL || backend->type == SIM_BACKEND_TYPE_CUDA) &&
        sim_kernel_uses_integer_domain(kernel)) {
        return "integer_domain_not_supported";
    }

    if ((backend->type == SIM_BACKEND_TYPE_METAL || backend->type == SIM_BACKEND_TYPE_CUDA) &&
        sim_kernel_contains_node_type(kernel, SIM_IR_NODE_STATEFUL)) {
        return "stateful_node_not_supported";
    }

    if ((backend->type == SIM_BACKEND_TYPE_METAL || backend->type == SIM_BACKEND_TYPE_CUDA) &&
        sim_ir_kernel_has_unsupported_complex_semantics(kernel)) {
        return "complex_semantics_not_supported";
    }

    if ((backend->type == SIM_BACKEND_TYPE_METAL || backend->type == SIM_BACKEND_TYPE_CUDA) &&
        kernel->output_count != 1U) {
        return "pointwise_backend_requires_single_output";
    }

    return NULL;
}

static SimResult sim_context_prepare_complex_fields(SimContext* context) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < context->world.field_count; ++i) {
        SimField* field                 = &context->world.fields[i];
        bool      needs_complex_storage = sim_field_complex_mode(field);
        if (needs_complex_storage) {
            if (!sim_field_storage_is_complex(field)) {
                size_t element_count   = sim_field_element_count(&field->layout);
                size_t old_bytes       = sim_field_bytes(field);
                size_t new_bytes       = 0U;
                size_t baseline_fields = 0U;
                size_t next_fields     = 0U;
                size_t next_total      = 0U;

                if (element_count == 0U) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }
                if (element_count > (SIZE_MAX / sizeof(SimComplexDouble))) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                new_bytes = element_count * sizeof(SimComplexDouble);

                if (context->memory_limits.max_field_elements > 0U &&
                    element_count > context->memory_limits.max_field_elements) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                if (context->memory_limits.max_field_bytes > 0U &&
                    new_bytes > context->memory_limits.max_field_bytes) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }

                if (context->bytes_fields_in_use < old_bytes) {
                    return SIM_RESULT_INVALID_STATE;
                }
                baseline_fields = context->bytes_fields_in_use - old_bytes;
                if (context->memory_limits.max_total_field_bytes > 0U) {
                    if (new_bytes > context->memory_limits.max_total_field_bytes) {
                        return SIM_RESULT_OUT_OF_MEMORY;
                    }
                    if (baseline_fields >
                        context->memory_limits.max_total_field_bytes - new_bytes) {
                        return SIM_RESULT_OUT_OF_MEMORY;
                    }
                }
                if (!sim_context_add_size(baseline_fields, new_bytes, &next_fields)) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                if (!sim_context_add_size(
                        next_fields, context->bytes_scratch_in_use, &next_total)) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }

                SimResult promote = sim_field_require_complex(field);
                if (promote != SIM_RESULT_OK) {
                    return promote;
                }
                context->bytes_fields_in_use = next_fields;
                context->bytes_total_in_use  = next_total;
            }
        }
    }

    return SIM_RESULT_OK;
}

void sim_context_set_integrator(SimContext* context, struct Integrator* integrator) {
    if (context == NULL) {
        return;
    }
    sim_integrator_state_set_active(&context->integrators, integrator);
}

struct Integrator* sim_context_integrator(SimContext* context) {
    if (context == NULL) {
        return NULL;
    }
    return sim_integrator_state_active(&context->integrators);
}

SimResult sim_context_set_integrator_sequence(SimContext*               context,
                                              struct Integrator* const* integrators,
                                              size_t                    count) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    return sim_integrator_state_set_sequence(&context->integrators, integrators, count);
}

size_t sim_context_integrator_sequence_count(const SimContext* context) {
    size_t count = 0U;
    if (context == NULL) {
        return 0U;
    }
    (void) sim_integrator_state_sequence(&context->integrators, &count);
    return count;
}

struct Integrator* sim_context_integrator_sequence_at(const SimContext* context, size_t index) {
    size_t                          count = 0U;
    const struct Integrator* const* seq;

    if (context == NULL) {
        return NULL;
    }
    seq = sim_integrator_state_sequence(&context->integrators, &count);
    if (seq == NULL || index >= count) {
        return NULL;
    }
    return (struct Integrator*) seq[index];
}

void sim_context_set_backend(SimContext* context, struct SimBackend* backend) {
    if (context == NULL) {
        return;
    }
    context->scheduler.backend = backend;
}

struct SimBackend* sim_context_backend(SimContext* context) {
    if (context == NULL) {
        return NULL;
    }
    return context->scheduler.backend;
}

void sim_context_set_timestep(SimContext* context, double dt) {
    if (context == NULL) {
        return;
    }
    context->runtime.dt = (dt > 0.0) ? dt : SIM_RUNTIME_DEFAULT_DT;
}

double sim_context_timestep(const SimContext* context) {
    if (context == NULL || context->runtime.dt <= 0.0) {
        return SIM_RUNTIME_DEFAULT_DT;
    }
    return context->runtime.dt;
}

void sim_context_set_field_stats_features(SimContext* context, uint32_t feature_mask) {
    SimFieldStatsComputeConfig config;

    if (context == NULL) {
        return;
    }

    config = context->runtime.field_stats_config;
    config.feature_mask = sim_field_stats_normalize_feature_mask(feature_mask);
    if (config.feature_mask == context->runtime.field_stats_config.feature_mask) {
        return;
    }

    context->runtime.field_stats_config = config;
    sim_field_stats_profile_reset(&context->runtime.field_stats_profile, &config);
    sim_context_invalidate_field_stats_caches(context);
    sim_context_reset_field_stats_phase_state(context);
}

uint32_t sim_context_field_stats_features(const SimContext* context) {
    if (context == NULL) {
        return SIM_FIELD_STATS_FEATURE_MASK_DEFAULT;
    }
    return context->runtime.field_stats_config.feature_mask;
}

void sim_context_reset_field_stats_profile(SimContext* context) {
    if (context == NULL) {
        return;
    }
    sim_field_stats_profile_reset(&context->runtime.field_stats_profile,
                                  &context->runtime.field_stats_config);
}

bool sim_context_field_stats_profile(const SimContext*         context,
                                     SimFieldStatsRuntimeProfile* out_profile) {
    if (context == NULL || out_profile == NULL) {
        return false;
    }

    *out_profile = context->runtime.field_stats_profile;
    return true;
}

void sim_context_set_time_model(SimContext* context, SimTimeModel model) {
    if (context == NULL) {
        return;
    }
    if (context->runtime.time_model != model) {
        context->runtime.time_model = model;
    }
}

void sim_context_begin_drift(SimContext* context) {
    if (context == NULL) {
        return;
    }
    if (context->runtime.drift_mode_depth < UINT32_MAX) {
        context->runtime.drift_mode_depth += 1U;
    }
}

void sim_context_end_drift(SimContext* context) {
    if (context == NULL) {
        return;
    }
    if (context->runtime.drift_mode_depth > 0U) {
        context->runtime.drift_mode_depth -= 1U;
        if (context->runtime.drift_mode_depth == 0U) {
            context->runtime.drift_time_override_active = false;
            context->runtime.drift_time_override        = 0.0;
        }
    }
}

bool sim_context_in_drift(const SimContext* context) {
    if (context == NULL) {
        return false;
    }
    return context->runtime.drift_mode_depth > 0U;
}

void sim_context_set_drift_time_override(SimContext* context, double time_value) {
    if (context == NULL || context->runtime.drift_mode_depth == 0U) {
        return;
    }
    context->runtime.drift_time_override        = time_value;
    context->runtime.drift_time_override_active = true;
}

void sim_context_clear_drift_time_override(SimContext* context) {
    if (context == NULL) {
        return;
    }
    context->runtime.drift_time_override_active = false;
    context->runtime.drift_time_override        = 0.0;
}

size_t sim_context_step_index(const SimContext* context) {
    if (context == NULL) {
        return 0U;
    }
    return context->runtime.step_index;
}

double sim_context_time(const SimContext* context) {
    if (context == NULL) {
        return 0.0;
    }
    if (context->runtime.drift_time_override_active) {
        return context->runtime.drift_time_override;
    }
    return context->runtime.time_accumulated;
}

void sim_context_record_step_metrics(SimContext* context,
                                     double      requested_dt,
                                     double      accepted_dt,
                                     double      rms_error) {
    sim_context_record_step_metrics_with_timing(
        context, requested_dt, accepted_dt, rms_error, 0ULL, 0ULL, 0ULL);
}

void sim_context_record_step_metrics_with_timing(SimContext* context,
                                                 double      requested_dt,
                                                 double      accepted_dt,
                                                 double      rms_error,
                                                 uint64_t    step_wall_ns,
                                                 uint64_t    integrator_wall_ns,
                                                 uint64_t    operator_wall_ns) {
    SimStepMetrics metrics;
    Integrator*     integrator;

    if (context == NULL || sim_context_in_drift(context)) {
        return;
    }
    integrator = sim_context_integrator(context);
    metrics.step_index =
        (context->runtime.step_index > 0U) ? (context->runtime.step_index - 1U) : 0U;
    metrics.requested_dt = (requested_dt > 0.0) ? requested_dt : sim_context_timestep(context);
    metrics.accepted_dt  = (accepted_dt > 0.0) ? accepted_dt : metrics.requested_dt;
    metrics.next_dt      = sim_context_timestep(context);
    metrics.rms_error    = (rms_error >= 0.0) ? rms_error : 0.0;
    metrics.step_wall_ns = step_wall_ns;
    metrics.integrator_wall_ns = integrator_wall_ns;
    metrics.operator_wall_ns   = operator_wall_ns;
    metrics.integrator_workspace_bytes = integrator_workspace_bytes(integrator);
    metrics.integrator_drift_scratch_bytes = integrator_drift_scratch_bytes(integrator);
    metrics.integrator_attempt_count = (integrator != NULL) ? integrator->last_attempt_count : 0U;
    metrics.integrator_rejection_count =
        (integrator != NULL) ? integrator->last_rejection_count : 0U;
    sim_context_accumulate_continuity_totals(
        context, &metrics.dirty_write_count, &metrics.stable_write_count);
    metrics.active_warp_mask = sim_context_active_warp_mask(context);

    sim_runtime_state_record_step_metrics(&context->runtime, &metrics);
}

void sim_context_accept_step(SimContext* context, double accepted_dt) {
    if (context == NULL) {
        return;
    }

    double dt = accepted_dt;

    if (!(dt > 0.0) || !isfinite(dt)) {
        dt = sim_context_timestep(context);
    }
    if (!(dt > 0.0) || !isfinite(dt)) {
        dt = SIM_RUNTIME_DEFAULT_DT;
    }

    context->runtime.step_index += 1U;
    context->runtime.time_accumulated += (double) dt;

    flux_lens_update(context);
}

bool sim_context_latest_step_metrics(const SimContext* context, SimStepMetrics* out_metrics) {
    if (context == NULL) {
        return false;
    }
    return sim_runtime_state_latest_step_metrics(&context->runtime, out_metrics);
}

size_t
sim_context_step_metrics_history(const SimContext* context, SimStepMetrics* dest, size_t capacity) {
    if (context == NULL) {
        return 0U;
    }
    return sim_runtime_state_copy_step_metrics(&context->runtime, dest, capacity);
}

bool sim_context_profiler_counters(SimContext*         context,
                                   SimProfilerCounter* out_counters,
                                   size_t              capacity,
                                   size_t*             out_count) {
    if (out_count)
        *out_count = 0U;
    if (context == NULL || out_counters == NULL || capacity == 0U || !context->profiler_ready) {
        return false;
    }

    size_t emit =
        (context->profiler.counter_count < capacity) ? context->profiler.counter_count : capacity;
    for (size_t i = 0U; i < emit; ++i) {
        out_counters[i] = context->profiler.counters[i];
    }
    if (out_count)
        *out_count = emit;
    return emit > 0U;
}

bool sim_context_profiler_snapshot(SimContext* context, SimProfilerSnapshot* out_snapshot) {
    if (out_snapshot) {
        memset(out_snapshot, 0, sizeof(*out_snapshot));
    }

    if (context == NULL || out_snapshot == NULL || !context->profiler_ready) {
        return false;
    }

    return sim_profiler_snapshot(&context->profiler, out_snapshot) == SIM_RESULT_OK;
}

void sim_context_set_special_fallback(SimContext*          context,
                                      SimSpecialFallbackFn fallback,
                                      void*                userdata) {
    if (context == NULL) {
        return;
    }

    sim_context_fault_lock(context);
    context->diag.fallback_user     = fallback;
    context->diag.fallback_userdata = userdata;
    sim_context_fault_unlock(context);
}

void sim_context_special_fallback_hook(const SimContext*     context,
                                       SimSpecialFallbackFn* out_fallback,
                                       void**                out_userdata) {
    if (out_fallback != NULL) {
        *out_fallback = (context != NULL) ? sim_context_special_fallback_dispatch : NULL;
    }

    if (out_userdata != NULL) {
        *out_userdata = (void*) context;
    }
}

uint64_t sim_context_special_fault_count(const SimContext* context) {
    if (context == NULL) {
        return 0ULL;
    }

    SimContext* mutable_context = (SimContext*) context;
    uint64_t    count           = 0ULL;

    sim_context_fault_lock(mutable_context);
    count = mutable_context->diag.fault_count;
    sim_context_fault_unlock(mutable_context);

    return count;
}

bool sim_context_last_special_fault(const SimContext* context, SimSpecialEvalReport* out_report) {
    if (context == NULL || out_report == NULL) {
        return false;
    }

    SimContext* mutable_context = (SimContext*) context;
    bool        have_fault      = false;

    sim_context_fault_lock(mutable_context);
    if (mutable_context->diag.fault_count > 0ULL) {
        *out_report = mutable_context->diag.fault_last;
        have_fault  = true;
    }
    sim_context_fault_unlock(mutable_context);

    return have_fault;
}

void sim_context_clear_special_faults(SimContext* context) {
    if (context == NULL) {
        return;
    }

    sim_context_fault_lock(context);
    context->diag.fault_count = 0ULL;
    memset(&context->diag.fault_last, 0, sizeof(context->diag.fault_last));
    context->diag.fault_function[0] = '\0';
    sim_context_fault_unlock(context);
}

void sim_context_set_continuity_override(SimContext*              context,
                                         bool                     enabled,
                                         const SimOperatorConfig* config) {
    size_t            i;
    SimOperatorConfig normalized = sim_operator_config_defaults();

    if (context == NULL) {
        return;
    }

    if (config != NULL) {
        normalized = *config;
    }
    sim_operator_config_normalize(&normalized);

    bool was_enabled             = context->continuity_override_enabled;
    context->continuity_override = normalized;

    if (!enabled) {
        context->continuity_override_enabled = false;
        return;
    }

    context->continuity_override_enabled = true;

    for (i = 0U; i < context->world.operators.count; ++i) {
        SimOperator* op = &context->world.operators.records[i];
        sim_context_apply_override_to_operator(context, op);
    }
}

void sim_split_notify_integrator(SimContext* context, double dt_sub, double error_estimate) {
    Integrator* integrator;

    if (context == NULL) {
        return;
    }

    integrator = sim_integrator_state_stepping(&context->integrators);
    if (integrator == NULL) {
        integrator = sim_integrator_state_active(&context->integrators);
    }
    if (integrator == NULL) {
        return;
    }
    if (!isfinite(dt_sub) || !(dt_sub > 0.0)) {
        dt_sub = sim_context_timestep(context);
    }

    if (dt_sub > 0.0) {
        integrator->split_feedback_dt += dt_sub;
    }

    if (error_estimate >= 0.0) {
        if (error_estimate > integrator->split_feedback_max_error) {
            integrator->split_feedback_max_error = error_estimate;
        }
    }

    if (integrator->split_feedback_substeps < UINT32_MAX) {
        integrator->split_feedback_substeps += 1U;
    }
}

size_t sim_context_truncation_level(const SimContext* context) {
    if (context == NULL) {
        return SIM_WORLD_DEFAULT_K;
    }
    return context->world.universe.K;
}

double sim_context_epsilon(const SimContext* context) {
    if (context == NULL) {
        return SIM_WORLD_DEFAULT_EPSILON;
    }
    return context->world.universe.epsilon;
}

const SimPole* sim_context_poles(const SimContext* context, size_t* out_count) {
    if (context == NULL) {
        if (out_count != NULL) {
            *out_count = 0U;
        }
        return NULL;
    }

    if (out_count != NULL) {
        *out_count = context->world.universe.pole_count;
    }

    return context->world.universe.poles;
}

SimPoleFieldOptions sim_pole_field_options_default(void) {
    SimPoleFieldOptions opts = { .origin_x  = 0.0,
                                 .origin_y  = 0.0,
                                 .spacing_x = 1.0,
                                 .spacing_y = 1.0,
                                 .plane_z   = 0.0,
                                 .softening = 1.0e-3 };
    return opts;
}

static double sim_context_sample_from_poles(const SimPole* poles,
                                            size_t         pole_count,
                                            double         x,
                                            double         y,
                                            double         plane_z,
                                            double         softening) {
    if (poles == NULL || pole_count == 0U) {
        return 0.0;
    }

    double soft    = (softening > 0.0) ? softening : 1.0e-6;
    double soft_sq = soft * soft;
    double value   = 0.0;

    for (size_t i = 0U; i < pole_count; ++i) {
        double dx    = x - (double) poles[i].x;
        double dy    = y - (double) poles[i].y;
        double dz    = plane_z - (double) poles[i].z;
        double r2    = dx * dx + dy * dy + dz * dz;
        double denom = sqrt(r2 + soft_sq);
        if (!(denom > 0.0)) {
            denom = soft;
        }
        value += (double) poles[i].residue / denom;
    }

    return value;
}

SimResult sim_context_synthesize_pole_field(SimContext*                context,
                                            size_t                     field_index,
                                            const SimPoleFieldOptions* options) {
    SimField*           field;
    SimPoleFieldOptions local_opts = sim_pole_field_options_default();
    const SimPole*      poles;
    size_t              pole_count = 0U;
    size_t              total_elements;
    size_t              width;
    size_t              height;
    double              spacing_x;
    double              spacing_y;
    double              plane_z;
    double              softening;
    void*               data;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, field_index);
    if (field == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    if (field->layout.rank == 0U || field->data == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    if (!field->layout.contiguous) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    if (options != NULL) {
        local_opts = *options;
    }

    spacing_x = (fabs(local_opts.spacing_x) > 0.0) ? local_opts.spacing_x : 1.0;
    spacing_y = (fabs(local_opts.spacing_y) > 0.0) ? local_opts.spacing_y : 1.0;
    plane_z   = local_opts.plane_z;
    softening = (local_opts.softening >= 0.0) ? local_opts.softening : 1.0e-3;

    total_elements = sim_field_element_count(&field->layout);
    if (total_elements == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (field->layout.rank > 2U) {
        return SIM_RESULT_NOT_SUPPORTED;
    }

    width = sim_field_width(field);
    if (width == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    height = sim_field_height(field);
    if (height == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (total_elements != width * height) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    data  = field->data;
    poles = sim_context_poles(context, &pole_count);

    for (size_t row = 0U; row < height; ++row) {
        double y = local_opts.origin_y + (double) row * spacing_y;
        for (size_t col = 0U; col < width; ++col) {
            size_t    idx    = 0U;
            SimResult idx_rc = sim_field_xy_to_index(field, col, row, &idx);
            if (idx_rc != SIM_RESULT_OK) {
                return idx_rc;
            }
            double x = local_opts.origin_x + (double) col * spacing_x;
            double sample =
                sim_context_sample_from_poles(poles, pole_count, x, y, plane_z, softening);

            if (field->element_size == sizeof(double)) {
                double* dst = (double*) data;
                dst[idx]    = sample;
            } else if (field->element_size == sizeof(SimComplexDouble)) {
                SimComplexDouble* dst = (SimComplexDouble*) data;
                dst[idx].re           = sample;
                dst[idx].im           = 0.0;
            } else {
                return SIM_RESULT_NOT_SUPPORTED;
            }
        }
    }

    return SIM_RESULT_OK;
}
static bool
sim_context_collect_unique_fields(const SimOperator* op, size_t** out_indices, size_t* out_count) {
    size_t* indices  = NULL;
    size_t  count    = 0U;
    size_t  capacity = 0U;

    if (op == NULL || out_indices == NULL || out_count == NULL) {
        return false;
    }

#define APPEND_FIELD(val)                                                                          \
    do {                                                                                           \
        size_t v_    = (val);                                                                      \
        bool   seen_ = false;                                                                      \
        for (size_t i_ = 0U; i_ < count; ++i_) {                                                   \
            if (indices[i_] == v_) {                                                               \
                seen_ = true;                                                                      \
                break;                                                                             \
            }                                                                                      \
        }                                                                                          \
        if (!seen_) {                                                                              \
            if (count == capacity) {                                                               \
                size_t  new_cap = (capacity == 0U) ? 8U : (capacity * 2U);                         \
                size_t* new_buf = (size_t*) realloc(indices, new_cap * sizeof(size_t));            \
                if (new_buf == NULL) {                                                             \
                    free(indices);                                                                 \
                    return false;                                                                  \
                }                                                                                  \
                indices  = new_buf;                                                                \
                capacity = new_cap;                                                                \
            }                                                                                      \
            indices[count++] = v_;                                                                 \
        }                                                                                          \
    } while (0)

    uint64_t mask = (op->write_mask != 0ULL) ? op->write_mask : op->read_mask;
    while (mask != 0ULL) {
        unsigned int bit = __builtin_ctzll(mask);
        mask &= ~(1ULL << bit);
        APPEND_FIELD((size_t) bit);
    }
    for (size_t i = 0U; i < op->read_index_count; ++i) {
        APPEND_FIELD(op->read_indices[i]);
    }
    for (size_t i = 0U; i < op->write_index_count; ++i) {
        APPEND_FIELD(op->write_indices[i]);
    }

#undef APPEND_FIELD

    *out_indices = indices;
    *out_count   = count;
    return true;
}

static void sim_context_apply_norm_budget(SimContext* context, const SimOperator* op) {
    if (context == NULL || op == NULL) {
        return;
    }
    const double budget = op->config.norm_budget;
    if (!(budget > 0.0)) {
        return;
    }

    size_t* fields      = NULL;
    size_t  field_count = 0U;
    if (!sim_context_collect_unique_fields(op, &fields, &field_count)) {
        return;
    }

    /* Apply budget only to written fields if available. */
    if (op->write_index_count == 0U && op->write_mask == 0ULL) {
        /* fall back to collected fields */
    }

    for (size_t i = 0U; i < field_count; ++i) {
        SimField* field = sim_context_field(context, fields[i]);
        if (field == NULL) {
            continue;
        }
        /* Only enforce on double/complex-double fields. */
        size_t comps = sim_field_components(field);
        if (comps == 0U || (field->element_size != sizeof(double) &&
                            field->element_size != sizeof(SimComplexDouble))) {
            continue;
        }
        size_t elements = sim_field_element_count(&field->layout);
        if (elements == 0U) {
            continue;
        }
        const double* data = sim_field_real_data_const(field);
        if (data == NULL) {
            continue;
        }
        double sum_sq        = 0.0;
        size_t total_scalars = elements * comps;
        for (size_t j = 0U; j < total_scalars; ++j) {
            double v = data[j];
            sum_sq += v * v;
        }
        double norm = sqrt(sum_sq);
        if (norm > budget && norm > 0.0) {
            double  scale = budget / norm;
            double* wdata = sim_field_real_data(field);
            if (wdata == NULL) {
                continue;
            }
            for (size_t j = 0U; j < total_scalars; ++j) {
                wdata[j] *= scale;
            }
        }
    }

    free(fields);
}
