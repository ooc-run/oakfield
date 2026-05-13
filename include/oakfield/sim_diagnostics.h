/**
 * @file sim_diagnostics.h
 * @brief Simulation diagnostics state (faults, fallback hooks, counters).
 */
#ifndef OAKFIELD_SIM_DIAGNOSTICS_H
#define OAKFIELD_SIM_DIAGNOSTICS_H

#include "math/special_functions.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Backend families recorded in diagnostic fallback events.
 */
typedef enum SimDiagnosticBackendKind {
    SIM_DIAGNOSTIC_BACKEND_CPU = 0,      /**< CPU execution backend. */
    SIM_DIAGNOSTIC_BACKEND_CUDA = 1,     /**< CUDA execution backend. */
    SIM_DIAGNOSTIC_BACKEND_METAL = 2,    /**< Metal execution backend. */
    SIM_DIAGNOSTIC_BACKEND_UNKNOWN = 255 /**< Backend was unavailable or not recorded. */
} SimDiagnosticBackendKind;

/**
 * @brief Diagnostics container for special-function faults and fallback.
 */
typedef struct SimDiagnostics {
    SimSpecialFallbackFn fallback_user; /**< Optional user-provided fallback. */
    void *fallback_userdata;            /**< Userdata forwarded to fallback. */
    volatile int fault_lock;            /**< Spin-lock guarding fault state. */
    uint64_t fault_count;               /**< Total observed special-function faults. */
    SimSpecialEvalReport fault_last;    /**< Last recorded fault report. */
    char fault_function[64];            /**< Storage for last fault function label. */
    uint64_t kernel_dispatch_count;     /**< Total kernel dispatch attempts. */
    uint64_t kernel_fallback_count;     /**< Kernel dispatches that fell back to CPU. */
    bool kernel_last_valid;             /**< True when a kernel dispatch has been recorded. */
    bool kernel_last_fallback_used;     /**< True when the last kernel ran via CPU fallback. */
    SimDiagnosticBackendKind
        kernel_last_requested_backend; /**< Requested backend for last kernel. */
    SimDiagnosticBackendKind
        kernel_last_executed_backend;     /**< Backend that executed the last kernel. */
    char kernel_last_operator[64];        /**< Last kernel operator name. */
    char kernel_last_fallback_reason[96]; /**< Last CPU fallback reason. */
    bool enable_invariant_checks;         /**< Enable lightweight invariant checks post-operator. */
    uint64_t representation_complex_promotions; /**< Count of complex promotions performed by
                                                   resolver. */
    uint64_t representation_domain_adjustments; /**< Count of domain adjustments performed by
                                                   resolver. */
#if SIM_DIAGNOSTICS
    /* Aggregated special-function health metrics (lifetime counters). */
    uint64_t reflection_count;         /**< Total reflection-path evaluations. */
    uint64_t recurrence_shift_samples; /**< Steps that performed recurrence shifts. */
    double max_recurrence_shift;       /**< Maximum recurrence depth observed. */
    uint64_t stirling_tail_samples;    /**< Steps that evaluated a Stirling tail. */
    double max_stirling_tail;          /**< Maximum |tail| observed. */
    uint64_t pole_proximity_samples;   /**< Samples near special-function poles. */
    double min_pole_distance;          /**< Smallest distance to a pole. */
#endif
} SimDiagnostics;

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_DIAGNOSTICS_H */
