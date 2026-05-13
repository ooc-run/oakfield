/**
 * @file integrator.c
 * @brief Shared workspace, timestep, drift, and stochastic utilities for integrators.
 *
 * This module owns the common Integrator lifecycle, aligned scratch buffers,
 * adaptive timestep controller, error norms, RNG-backed noise sources, and the
 * context-drift adapter used by context-backed steppers. It preserves caller
 * field state during drift evaluation by snapshotting context fields and
 * restoring them after prepared-plan execution.
 */

#include "oakfield/integrator.h"

#include "sim_accel.h"
#include "oakfield/sim_context.h"
#include "oakfield/sim_seed.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef INTEGRATOR_MAX_GROWTH
#define INTEGRATOR_MAX_GROWTH 2.0
#endif

#ifndef INTEGRATOR_MIN_SHRINK
#define INTEGRATOR_MIN_SHRINK 0.25
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef INTEGRATOR_WORKSPACE_ALIGNMENT
#define INTEGRATOR_WORKSPACE_ALIGNMENT 64U
#endif

/**
 * @brief Round a byte count up to the requested alignment.
 *
 * @return Aligned value, or 0 when the rounded value would overflow.
 */
static size_t integrator_align_up(size_t value, size_t alignment) {
    size_t remainder;

    if (alignment <= 1U) {
        return value;
    }

    remainder = value % alignment;
    if (remainder == 0U) {
        return value;
    }
    if (value > SIZE_MAX - (alignment - remainder)) {
        return 0U;
    }
    return value + (alignment - remainder);
}

/**
 * @brief Allocate an aligned scratch buffer for SIMD-friendly field kernels.
 *
 * Zero-byte requests allocate one byte so callers receive a valid allocation
 * when the platform allocator permits it.
 */
static void* integrator_aligned_alloc_bytes(size_t bytes) {
    void*  ptr       = NULL;
    size_t alignment = INTEGRATOR_WORKSPACE_ALIGNMENT;
    size_t alloc_sz;

    if (bytes == 0U) {
        bytes = 1U;
    }
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

    alloc_sz = integrator_align_up(bytes, alignment);
    if (alloc_sz == 0U) {
        return NULL;
    }

#if defined(__APPLE__) || defined(_POSIX_VERSION)
    if (posix_memalign(&ptr, alignment, alloc_sz) != 0) {
        return NULL;
    }
    return ptr;
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    return aligned_alloc(alignment, alloc_sz);
#else
    (void) alignment;
    return malloc(alloc_sz);
#endif
}

/**
 * @brief Reallocate aligned storage by allocate-copy-free.
 *
 * The returned allocation may have a different alignment-compatible address
 * even when the requested size shrinks. Existing bytes are preserved up to the
 * smaller of old and new sizes.
 */
static void* integrator_aligned_realloc_copy(void*  existing,
                                             size_t existing_bytes,
                                             size_t new_bytes,
                                             bool   zero_new_allocation) {
    void*  replacement;
    size_t copy_bytes;

    if (new_bytes == 0U) {
        free(existing);
        return NULL;
    }

    replacement = integrator_aligned_alloc_bytes(new_bytes);
    if (replacement == NULL) {
        return NULL;
    }

    if (zero_new_allocation) {
        memset(replacement, 0, new_bytes);
    }

    copy_bytes = (existing_bytes < new_bytes) ? existing_bytes : new_bytes;
    if (existing != NULL && copy_bytes > 0U) {
        memcpy(replacement, existing, copy_bytes);
    }

    free(existing);
    return replacement;
}

/**
 * @brief Advance the integrator's xorshift RNG state.
 */
static uint32_t integrator_rng_next(Integrator* integrator) {
    uint32_t x = integrator->rng_state;
    x ^= x << 13U;
    x ^= x >> 17U;
    x ^= x << 5U;
    integrator->rng_state = x;
    return x;
}

/**
 * @brief Return a deterministic uniform sample in the closed interval [0, 1].
 */
static double integrator_rng_uniform(Integrator* integrator) {
    return (double) integrator_rng_next(integrator) / (double) UINT32_MAX;
}

/**
 * @brief Fill a buffer with Gaussian samples via the Box-Muller transform.
 *
 * The first uniform variate is clamped away from zero to avoid `log(0)`.
 */
static void integrator_fill_normal_samples(Integrator* integrator, double* out, size_t count) {
    if (integrator == NULL || out == NULL) {
        return;
    }

    for (size_t i = 0U; i < count; i += 2U) {
        double u1    = integrator_rng_uniform(integrator);
        double u2    = integrator_rng_uniform(integrator);
        double mag;
        double angle;
        double s     = 0.0;
        double c     = 0.0;

        if (u1 <= 1e-12) {
            u1 = 1e-12;
        }

        mag   = sqrt(-2.0 * log(u1));
        angle = 2.0 * M_PI * u2;
#if defined(__APPLE__)
        __sincos(angle, &s, &c);
#elif defined(__clang__) || defined(__GNUC__)
        sincos(angle, &s, &c);
#else
        s = sin(angle);
        c = cos(angle);
#endif

        out[i] = mag * c;
        if (i + 1U < count) {
            out[i + 1U] = mag * s;
        }
    }
}

/**
 * @brief Fill a buffer with unit-variance uniform samples.
 */
static void integrator_fill_uniform_samples(Integrator* integrator, double* out, size_t count) {
    const double scale = sqrt(3.0);

    if (integrator == NULL || out == NULL) {
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double u = integrator_rng_uniform(integrator);
        out[i]   = (2.0 * u - 1.0) * scale;
    }
}

/**
 * @brief Fill a buffer with unit-variance Laplace samples.
 *
 * Uniform input is clamped away from 0 and 1 so the inverse-CDF transform stays
 * finite.
 */
static void integrator_fill_laplace_samples(Integrator* integrator, double* out, size_t count) {
    const double b = sqrt(0.5);

    if (integrator == NULL || out == NULL) {
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        double u = integrator_rng_uniform(integrator);
        if (u < 1e-12) {
            u = 1e-12;
        } else if (u > 1.0 - 1e-12) {
            u = 1.0 - 1e-12;
        }

        out[i] = ((u < 0.5) ? -1.0 : 1.0) * (-log1p(-2.0 * fabs(u - 0.5)) * b);
    }
}

/**
 * @brief Return a diagnostic label for the configured built-in noise law.
 */
static const char* integrator_noise_label(const Integrator* integrator) {
    if (integrator == NULL) {
        return "gaussian";
    }
    if (integrator->noise == integrator_noise_uniform) {
        return "uniform";
    }
    if (integrator->noise == integrator_noise_laplace) {
        return "laplace";
    }
    return "gaussian";
}

uint64_t integrator_workspace_bytes(const Integrator* integrator) {
    if (integrator == NULL) {
        return 0ULL;
    }
    return (uint64_t) integrator->buffer_count * (uint64_t) integrator->buffer_elements *
           (uint64_t) integrator->buffer_element_size;
}

uint64_t integrator_drift_scratch_bytes(const Integrator* integrator) {
    uint64_t total = 0ULL;

    if (integrator == NULL) {
        return 0ULL;
    }

    total += (uint64_t) integrator->drift_state_scratch_capacity;
    if (integrator->drift_snapshot_capacities != NULL) {
        for (size_t i = 0U; i < integrator->drift_snapshot_count; ++i) {
            total += (uint64_t) integrator->drift_snapshot_capacities[i];
        }
    }

    return total;
}

/**
 * @brief Ensure per-context-field snapshot slots exist for drift evaluation.
 *
 * Existing snapshot pointers and capacities are preserved. Newly allocated slot
 * metadata is zero-initialized.
 */
static SimResult integrator_ensure_drift_snapshot_slots(Integrator* integrator, size_t count) {
    void**  new_snapshots;
    size_t* new_sizes;
    size_t* new_capacities;

    if (integrator == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (count <= integrator->drift_snapshot_count) {
        return SIM_RESULT_OK;
    }

    new_snapshots = (void**) calloc(count, sizeof(void*));
    new_sizes = (size_t*) calloc(count, sizeof(size_t));
    new_capacities = (size_t*) calloc(count, sizeof(size_t));
    if (new_snapshots == NULL || new_sizes == NULL || new_capacities == NULL) {
        free(new_snapshots);
        free(new_sizes);
        free(new_capacities);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (integrator->drift_snapshot_count > 0U) {
        size_t copy_bytes = integrator->drift_snapshot_count * sizeof(void*);
        memcpy(new_snapshots, integrator->drift_snapshots, copy_bytes);
        copy_bytes = integrator->drift_snapshot_count * sizeof(size_t);
        memcpy(new_sizes, integrator->drift_snapshot_sizes, copy_bytes);
        memcpy(new_capacities, integrator->drift_snapshot_capacities, copy_bytes);
    }

    free(integrator->drift_snapshots);
    free(integrator->drift_snapshot_sizes);
    free(integrator->drift_snapshot_capacities);

    integrator->drift_snapshots = new_snapshots;
    integrator->drift_snapshot_sizes = new_sizes;
    integrator->drift_snapshot_capacities = new_capacities;
    integrator->drift_snapshot_count = count;
    return SIM_RESULT_OK;
}

/**
 * @brief Ensure one drift snapshot slot can hold a field byte image.
 *
 * @param integrator Integrator that owns snapshot storage.
 * @param index Snapshot slot index.
 * @param bytes Required capacity for this field snapshot.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
static SimResult integrator_ensure_drift_snapshot_buffer(Integrator* integrator,
                                                         size_t      index,
                                                         size_t      bytes) {
    void*  resized;
    size_t old_bytes;

    if (integrator == NULL || index >= integrator->drift_snapshot_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (bytes == 0U) {
        integrator->drift_snapshot_sizes[index] = 0U;
        return SIM_RESULT_OK;
    }
    if (integrator->drift_snapshots[index] != NULL &&
        integrator->drift_snapshot_capacities[index] >= bytes) {
        integrator->drift_snapshot_sizes[index] = bytes;
        return SIM_RESULT_OK;
    }

    old_bytes = integrator->drift_snapshot_capacities[index];
    resized   = integrator_aligned_realloc_copy(
          integrator->drift_snapshots[index], old_bytes, bytes, false);
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    integrator->drift_snapshots[index] = resized;
    integrator->drift_snapshot_capacities[index] = bytes;
    integrator->drift_snapshot_sizes[index] = bytes;
    return SIM_RESULT_OK;
}

/**
 * @brief Ensure reusable scratch storage for the target state during drift.
 */
static SimResult integrator_ensure_drift_state_scratch(Integrator* integrator, size_t bytes) {
    void*  resized;
    size_t old_bytes;

    if (integrator == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (bytes == 0U) {
        return SIM_RESULT_OK;
    }
    if (integrator->drift_state_scratch != NULL &&
        integrator->drift_state_scratch_capacity >= bytes) {
        return SIM_RESULT_OK;
    }

    old_bytes = integrator->drift_state_scratch_capacity;
    resized   =
        integrator_aligned_realloc_copy(integrator->drift_state_scratch, old_bytes, bytes, false);
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    integrator->drift_state_scratch = resized;
    integrator->drift_state_scratch_capacity = bytes;
    return SIM_RESULT_OK;
}

SimResult integrator_noise_gaussian(Integrator*  integrator,
                                    const Field* field,
                                    double*      out_noise,
                                    size_t       count) {
    (void) field;
    if (integrator == NULL || out_noise == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    integrator_fill_normal_samples(integrator, out_noise, count);
    return SIM_RESULT_OK;
}

SimResult integrator_noise_uniform(Integrator*  integrator,
                                   const Field* field,
                                   double*      out_noise,
                                   size_t       count) {
    (void) field;
    if (integrator == NULL || out_noise == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    integrator_fill_uniform_samples(integrator, out_noise, count);
    return SIM_RESULT_OK;
}

SimResult integrator_noise_laplace(Integrator*  integrator,
                                   const Field* field,
                                   double*      out_noise,
                                   size_t       count) {
    (void) field;
    if (integrator == NULL || out_noise == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    integrator_fill_laplace_samples(integrator, out_noise, count);
    return SIM_RESULT_OK;
}

SimResult integrator_context_drift(Integrator*   integrator,
                                   const Field*  field,
                                   const double* state,
                                   double*       out_derivative,
                                   size_t        count) {
    SimContext* ctx;
    SimResult   result = SIM_RESULT_OK;
    double      dt;
    size_t      field_count    = 0U;

    if (integrator == NULL || field == NULL || state == NULL || out_derivative == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    ctx = (SimContext*) integrator->userdata;
    if (ctx == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    dt = sim_context_timestep(ctx);
    if (!(dt > 0.0)) {
        dt = 1.0;
    }

    integrator->is_complex = sim_field_domain_is_complex(field);

    field_count = sim_context_field_count(ctx);
    if (field_count > 0U) {
        result = integrator_ensure_drift_snapshot_slots(integrator, field_count);
        if (result != SIM_RESULT_OK) {
            return result;
        }
        for (size_t fi = 0U; fi < field_count; ++fi) {
            SimField*   snapshot_field = sim_context_field(ctx, fi);
            size_t      bytes          = sim_field_bytes(snapshot_field);
            const void* src            = sim_field_data_const(snapshot_field);
            integrator->drift_snapshot_sizes[fi] = 0U;
            if (bytes > 0U && src != NULL) {
                result = integrator_ensure_drift_snapshot_buffer(integrator, fi, bytes);
                if (result != SIM_RESULT_OK) {
                    return result;
                }
                memcpy(integrator->drift_snapshots[fi], src, bytes);
            }
        }
    }

    sim_context_begin_drift(ctx);

    if (integrator->is_complex) {
        const SimComplexDouble* cstate         = (const SimComplexDouble*) state;
        SimComplexDouble*       scratch        = NULL;
        SimComplexDouble*       field_ptr      = NULL;
        size_t                  scratch_bytes  = count * sizeof(SimComplexDouble);

        result = integrator_ensure_drift_state_scratch(integrator, scratch_bytes);
        if (result != SIM_RESULT_OK) {
            goto cleanup_snapshot;
        }
        scratch = (SimComplexDouble*) integrator->drift_state_scratch;

        memcpy(scratch, cstate, scratch_bytes);

        field_ptr = sim_field_complex_data((SimField*) field);
        if (field_ptr == NULL) {
            result = SIM_RESULT_INVALID_ARGUMENT;
            goto cleanup_complex;
        }
        memcpy(field_ptr, cstate, count * sizeof(SimComplexDouble));

        if (!sim_context_plan_is_valid(ctx)) {
            result = sim_context_prepare_plan(ctx);
            if (result != SIM_RESULT_OK) {
                goto cleanup_complex;
            }
        }

        result = sim_context_execute_prepared(ctx);
        if (result != SIM_RESULT_OK) {
            field_ptr = sim_field_complex_data((SimField*) field);
            if (field_ptr != NULL) {
                memcpy(field_ptr, scratch, scratch_bytes);
            }
            goto cleanup_complex;
        }

        field_ptr = sim_field_complex_data((SimField*) field);
        if (field_ptr == NULL) {
            result = SIM_RESULT_DEPENDENCY_ERROR;
            goto cleanup_complex;
        }

        SimComplexDouble* deriv = (SimComplexDouble*) out_derivative;
        for (size_t i = 0U; i < count; ++i) {
            double dre = field_ptr[i].re - scratch[i].re;
            double dim = field_ptr[i].im - scratch[i].im;
            deriv[i].re = dre / dt;
            deriv[i].im = dim / dt;
            field_ptr[i] = scratch[i];
        }

    cleanup_complex:
        goto cleanup_snapshot;
    }

    double* scratch = NULL;
    size_t  scratch_bytes = count * sizeof(double);
    result = integrator_ensure_drift_state_scratch(integrator, scratch_bytes);
    if (result != SIM_RESULT_OK) {
        goto cleanup_snapshot;
    }
    scratch = (double*) integrator->drift_state_scratch;

    memcpy(scratch, state, scratch_bytes);

    double* field_ptr = sim_field_real_data((SimField*) field);
    if (field_ptr == NULL) {
        result = SIM_RESULT_INVALID_ARGUMENT;
        goto cleanup_real;
    }
    memcpy(field_ptr, state, count * sizeof(double));

    if (!sim_context_plan_is_valid(ctx)) {
        result = sim_context_prepare_plan(ctx);
        if (result != SIM_RESULT_OK) {
            goto cleanup_real;
        }
    }

    result = sim_context_execute_prepared(ctx);
    if (result != SIM_RESULT_OK) {
        field_ptr = sim_field_real_data((SimField*) field);
        if (field_ptr != NULL) {
            memcpy(field_ptr, scratch, scratch_bytes);
        }
        goto cleanup_real;
    }

    if (sim_field_storage_is_complex(field)) {
        SimComplexDouble* complex_field = sim_field_complex_data((SimField*) field);
        if (complex_field == NULL) {
            result = SIM_RESULT_DEPENDENCY_ERROR;
            goto cleanup_real;
        }
        for (size_t i = 0U; i < count; ++i) {
            double updated    = complex_field[i].re;
            out_derivative[i] = (updated - scratch[i]) / dt;
            complex_field[i].re = scratch[i];
            complex_field[i].im = 0.0;
        }
    } else {
        field_ptr = sim_field_real_data((SimField*) field);
        if (field_ptr == NULL) {
            result = SIM_RESULT_DEPENDENCY_ERROR;
            goto cleanup_real;
        }
        for (size_t i = 0U; i < count; ++i) {
            double updated    = field_ptr[i];
            out_derivative[i] = (updated - scratch[i]) / dt;
            field_ptr[i]      = scratch[i];
        }
    }

cleanup_real:
cleanup_snapshot:
    if (integrator->drift_snapshots != NULL && integrator->drift_snapshot_sizes != NULL) {
        for (size_t fi = 0U; fi < field_count; ++fi) {
            if (integrator->drift_snapshots[fi] != NULL &&
                integrator->drift_snapshot_sizes[fi] > 0U) {
                SimField* restore_field = sim_context_field(ctx, fi);
                void*     dst           = sim_field_data(restore_field);
                if (dst != NULL) {
                    memcpy(dst,
                           integrator->drift_snapshots[fi],
                           integrator->drift_snapshot_sizes[fi]);
                }
            }
        }
    }
    sim_context_end_drift(ctx);
    return result;
}

SimResult integrator_configure(Integrator*             integrator,
                               const char*             name,
                               IntegratorStepFn        step_fn,
                               const IntegratorConfig* config) {
    IntegratorConfig defaults;
    size_t           name_len;

    if (integrator == NULL || name == NULL || step_fn == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (config == NULL) {
        defaults.drift               = NULL;
        defaults.noise               = NULL;
        defaults.destroy             = NULL;
        defaults.userdata            = NULL;
        defaults.target_field_index  = 0U;
        defaults.initial_dt          = 0.01;
        defaults.min_dt              = 1.0e-6;
        defaults.max_dt              = 0.5;
        defaults.tolerance           = 1.0e-4;
        defaults.safety              = 0.9;
        defaults.adaptive            = true;
        defaults.enable_stochastic   = false;
        defaults.stochastic_strength = 0.01;
        defaults.random_seed         = 0U;
        defaults.workspace_hint      = 0U;
        defaults.subordination_alpha = 0.0;
        defaults.subordination_quadrature_n = 0U;
        config                       = &defaults;
    }

    if (config->drift == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void) memset(integrator, 0, sizeof(*integrator));

    name_len = strlen(name);
    if (name_len >= sizeof(integrator->name)) {
        name_len = sizeof(integrator->name) - 1U;
    }
    (void) memcpy(integrator->name, name, name_len);
    integrator->name[name_len] = '\0';

    integrator->step              = step_fn;
    integrator->drift             = config->drift;
    integrator->noise             = config->noise;
    integrator->destroy           = config->destroy;
    integrator->userdata          = config->userdata;
    integrator->target_field_index = config->target_field_index;
    integrator->adaptive          = config->adaptive;
    integrator->enable_stochastic = config->enable_stochastic;
    integrator->min_dt            = (config->min_dt > 0.0) ? config->min_dt : 1.0e-6;
    integrator->max_dt =
        (config->max_dt > integrator->min_dt) ? config->max_dt : (integrator->min_dt * 100.0);
    integrator->tolerance  = (config->tolerance > 0.0) ? config->tolerance : 1.0e-4;
    integrator->safety     = (config->safety > 0.0) ? config->safety : 0.9;
    integrator->current_dt = (config->initial_dt > 0.0) ? config->initial_dt : integrator->min_dt;
    integrator->current_dt = integrator_clamp_dt(integrator, integrator->current_dt);
    integrator->last_step  = integrator->current_dt;
    integrator->last_error = 0.0;
    integrator->last_attempt_count   = 1U;
    integrator->last_rejection_count = 0U;
    integrator->stochastic_strength =
        (config->stochastic_strength >= 0.0) ? config->stochastic_strength : 0.0;
    integrator->subordination_alpha =
        (config->subordination_alpha > 0.0) ? config->subordination_alpha : 0.0;
    integrator->subordination_quadrature_n = config->subordination_quadrature_n;
    {
        uint64_t seed64       = sim_seed_normalize((uint64_t) config->random_seed);
        integrator->rng_state = (uint32_t) (seed64 & 0xFFFFFFFFULL);
        if (integrator->rng_state == 0U) {
            integrator->rng_state = 1U;
        }
    }

    integrator->buffers             = NULL;
    integrator->buffer_count        = 0U;
    integrator->buffer_elements     = 0U;
    integrator->buffer_element_size = sizeof(double);

    if (config->workspace_hint > 0U) {
        SimResult prep = integrator_ensure_workspace(integrator, 1U, config->workspace_hint);
        if (prep != SIM_RESULT_OK) {
            return prep;
        }
    }

    return SIM_RESULT_OK;
}

void integrator_destroy(Integrator* integrator) {
    size_t i;

    if (integrator == NULL) {
        return;
    }

    if (integrator->destroy != NULL) {
        IntegratorDestroyFn destroy_fn = integrator->destroy;
        integrator->destroy            = NULL;
        destroy_fn(integrator);
    }

    if (integrator->buffers != NULL) {
        for (i = 0U; i < integrator->buffer_count; ++i) {
            free(integrator->buffers[i]);
        }
        free(integrator->buffers);
    }
    if (integrator->drift_snapshots != NULL) {
        for (i = 0U; i < integrator->drift_snapshot_count; ++i) {
            free(integrator->drift_snapshots[i]);
        }
    }
    free(integrator->drift_snapshots);
    free(integrator->drift_snapshot_sizes);
    free(integrator->drift_snapshot_capacities);
    free(integrator->drift_state_scratch);

    (void) memset(integrator, 0, sizeof(*integrator));
}

SimResult integrator_ensure_workspace(Integrator* integrator, size_t buffers, size_t elements) {
    size_t   i;
    size_t   old_bytes;
    size_t   new_bytes;
    double** new_buffers;

    if (integrator == NULL || buffers == 0U || elements == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t elem_size = integrator->is_complex ? sizeof(SimComplexDouble) : sizeof(double);

    if (integrator->buffer_count < buffers) {
        new_buffers = (double**) realloc(integrator->buffers, buffers * sizeof(double*));
        if (new_buffers == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        integrator->buffers = new_buffers;
        for (i = integrator->buffer_count; i < buffers; ++i) {
            integrator->buffers[i] = NULL;
        }
        integrator->buffer_count = buffers;
    }

    bool needs_resize =
        (integrator->buffer_elements < elements) || (integrator->buffer_element_size != elem_size);

    if (needs_resize) {
        old_bytes = integrator->buffer_elements * integrator->buffer_element_size;
        new_bytes = elements * elem_size;
        for (i = 0U; i < integrator->buffer_count; ++i) {
            void* resized = integrator_aligned_realloc_copy(
                integrator->buffers[i], old_bytes, new_bytes, false);

            if (resized == NULL) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            integrator->buffers[i] = resized;
        }
        integrator->buffer_elements     = elements;
        integrator->buffer_element_size = elem_size;
    } else {
        for (i = 0U; i < integrator->buffer_count; ++i) {
            if (integrator->buffers[i] == NULL) {
                integrator->buffers[i] =
                    (double*) integrator_aligned_alloc_bytes(elements * elem_size);
                if (integrator->buffers[i] == NULL) {
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
                memset(integrator->buffers[i], 0, elements * elem_size);
            }
        }
    }

    return SIM_RESULT_OK;
}

double* integrator_buffer(Integrator* integrator, size_t index) {
    if (integrator == NULL || index >= integrator->buffer_count) {
        return NULL;
    }
    return integrator->buffers[index];
}

size_t integrator_state_length(const Field* field) {
    size_t bytes;

    if (field == NULL) {
        return 0U;
    }

    bytes = sim_field_bytes(field);
    if (bytes == 0U) {
        return 0U;
    }

    if (field->element_size == sizeof(double)) {
        return (bytes / sizeof(double));
    }

    if (field->element_size == sizeof(SimComplexDouble)) {
        return (bytes / sizeof(SimComplexDouble));
    }

    return 0U;
}

double integrator_clamp_dt(const Integrator* integrator, double dt) {
    double clamped;

    if (integrator == NULL) {
        return dt;
    }

    clamped = dt;
    if (clamped < integrator->min_dt) {
        clamped = integrator->min_dt;
    }
    if (clamped > integrator->max_dt) {
        clamped = integrator->max_dt;
    }
    return clamped;
}

double integrator_suggest_dt(const Integrator* integrator,
                             double            dt,
                             double            error_norm,
                             double            method_order) {
    double factor;
    double safeguarded_error;

    if (integrator == NULL) {
        return dt;
    }

    if (!integrator->adaptive) {
        return integrator_clamp_dt(integrator, dt);
    }

    safeguarded_error = (error_norm <= 1.0e-12) ? 1.0e-12 : error_norm;

    if (method_order < 1.0) {
        method_order = 1.0;
    }

    factor = integrator->safety *
             pow(integrator->tolerance / safeguarded_error, 1.0 / (method_order + 1.0));

    if (factor > INTEGRATOR_MAX_GROWTH) {
        factor = INTEGRATOR_MAX_GROWTH;
    } else if (factor < INTEGRATOR_MIN_SHRINK) {
        factor = INTEGRATOR_MIN_SHRINK;
    }

    return integrator_clamp_dt(integrator, dt * factor);
}

double integrator_reject_dt(const Integrator* integrator,
                            double            dt,
                            double            error_norm,
                            double            method_order) {
    double suggested;
    double fallback;

    if (integrator == NULL) {
        return dt;
    }
    if (!(dt > integrator->min_dt) || !isfinite(dt)) {
        return integrator_clamp_dt(integrator, integrator->min_dt);
    }

    suggested = integrator_suggest_dt(integrator, dt, error_norm, method_order);
    if (!isfinite(suggested) || !(suggested > 0.0) || suggested >= dt) {
        fallback = integrator->safety * 0.5 * dt;
        if (!isfinite(fallback) || !(fallback > 0.0) || fallback >= dt) {
            fallback = 0.5 * dt;
        }
        suggested = fallback;
    }

    if (suggested < integrator->min_dt) {
        suggested = integrator->min_dt;
    }
    if (suggested >= dt) {
        suggested = fmax(integrator->min_dt, 0.5 * dt);
    }

    return integrator_clamp_dt(integrator, suggested);
}

double
integrator_measure_error_complex(const SimComplexDouble* a, const SimComplexDouble* b, size_t n) {
    double sum0 = 0.0;
    double sum1 = 0.0;
    size_t i    = 0U;

    if (a == NULL || b == NULL || n == 0U) {
        return 0.0;
    }

    for (; i + 1U < n; i += 2U) {
        double dr0     = a[i].re - b[i].re;
        double di0     = a[i].im - b[i].im;
        double diff20  = dr0 * dr0 + di0 * di0;
        double an20    = a[i].re * a[i].re + a[i].im * a[i].im;
        double bn20    = b[i].re * b[i].re + b[i].im * b[i].im;
        double scale20 = fmax(fmax(an20, bn20), 1.0);

        double dr1     = a[i + 1U].re - b[i + 1U].re;
        double di1     = a[i + 1U].im - b[i + 1U].im;
        double diff21  = dr1 * dr1 + di1 * di1;
        double an21    = a[i + 1U].re * a[i + 1U].re + a[i + 1U].im * a[i + 1U].im;
        double bn21    = b[i + 1U].re * b[i + 1U].re + b[i + 1U].im * b[i + 1U].im;
        double scale21 = fmax(fmax(an21, bn21), 1.0);

        sum0 += diff20 / scale20;
        sum1 += diff21 / scale21;
    }

    for (; i < n; ++i) {
        double dr     = a[i].re - b[i].re;
        double di     = a[i].im - b[i].im;
        double diff2  = dr * dr + di * di;
        double an2    = a[i].re * a[i].re + a[i].im * a[i].im;
        double bn2    = b[i].re * b[i].re + b[i].im * b[i].im;
        double scale2 = fmax(fmax(an2, bn2), 1.0);
        sum0 += diff2 / scale2;
    }

    return sqrt((sum0 + sum1) / (double) n);
}

SimComplexDouble* integrator_buffer_complex(Integrator* integrator, unsigned int index) {
    return (SimComplexDouble*) integrator_buffer(integrator, index);
}

double integrator_measure_error(const double* a, const double* b, size_t count) {
    size_t i;
    double max0 = 0.0;
    double max1 = 0.0;
    double max2 = 0.0;
    double max3 = 0.0;

    if (a == NULL || b == NULL || count == 0U) {
        return 0.0;
    }

    for (i = 0U; i + 3U < count; i += 4U) {
        double diff0  = fabs(a[i] - b[i]);
        double scale0 = fmax(fabs(a[i]), fabs(b[i]));
        double diff1  = fabs(a[i + 1U] - b[i + 1U]);
        double scale1 = fmax(fabs(a[i + 1U]), fabs(b[i + 1U]));
        double diff2  = fabs(a[i + 2U] - b[i + 2U]);
        double scale2 = fmax(fabs(a[i + 2U]), fabs(b[i + 2U]));
        double diff3  = fabs(a[i + 3U] - b[i + 3U]);
        double scale3 = fmax(fabs(a[i + 3U]), fabs(b[i + 3U]));

        if (scale0 < 1.0) {
            scale0 = 1.0;
        }
        if (scale1 < 1.0) {
            scale1 = 1.0;
        }
        if (scale2 < 1.0) {
            scale2 = 1.0;
        }
        if (scale3 < 1.0) {
            scale3 = 1.0;
        }

        max0 = fmax(max0, diff0 / scale0);
        max1 = fmax(max1, diff1 / scale1);
        max2 = fmax(max2, diff2 / scale2);
        max3 = fmax(max3, diff3 / scale3);
    }

    for (; i < count; ++i) {
        double diff  = fabs(a[i] - b[i]);
        double scale = fmax(fabs(a[i]), fabs(b[i]));
        if (scale < 1.0) {
            scale = 1.0;
        }
        max0 = fmax(max0, diff / scale);
    }

    return fmax(fmax(max0, max1), fmax(max2, max3));
}

double integrator_rng_normal(Integrator* integrator) {
    double u1;
    double u2;

    if (integrator == NULL) {
        return 0.0;
    }

    u1 = integrator_rng_uniform(integrator);
    if (u1 <= 1e-12) {
        u1 = 1e-12;
    }
    u2 = integrator_rng_uniform(integrator);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

void integrator_apply_stochastic(Integrator* integrator,
                                 Field*      field,
                                 double*     scratch,
                                 size_t      count,
                                 double      dt) {
    double  scale;
    double* state;

    if (integrator == NULL || field == NULL || scratch == NULL) {
        return;
    }

    if (!integrator->enable_stochastic || integrator->stochastic_strength <= 0.0) {
        return;
    }

    state = (double*) sim_field_data(field);

    if (state == NULL) {
        return;
    }

    if (sim_field_storage_is_complex(field)) {
        SimComplexDouble* state = sim_field_complex_data(field);
        if (!state)
            return;

        /* Preserve current complex stochastic behavior: built-in complex noise remains Gaussian. */
        integrator_fill_normal_samples(integrator, scratch, count * 2U);

        scale = integrator->stochastic_strength * sqrt((double) dt);
        sim_accel_copy_scale_real(scratch, (double*) state, count * 2U, scale, true);

        return;
    }

    if (integrator->noise == NULL || integrator->noise == integrator_noise_gaussian) {
        integrator_fill_normal_samples(integrator, scratch, count);
    } else if (integrator->noise == integrator_noise_uniform) {
        integrator_fill_uniform_samples(integrator, scratch, count);
    } else if (integrator->noise == integrator_noise_laplace) {
        integrator_fill_laplace_samples(integrator, scratch, count);
    } else if (integrator->noise != NULL) {
        if (integrator->noise(integrator, field, scratch, count) != SIM_RESULT_OK) {
            return;
        }
    }

    scale = integrator->stochastic_strength * sqrt((double) dt);
    sim_accel_copy_scale_real(scratch, state, count, scale, true);
}

double integrator_last_step(const Integrator* integrator) {
    if (integrator == NULL) {
        return 0.0;
    }
    return integrator->last_step;
}

double integrator_next_step(const Integrator* integrator) {
    if (integrator == NULL) {
        return 0.0;
    }
    return integrator->current_dt;
}

double integrator_last_error(const Integrator* integrator) {
    if (integrator == NULL) {
        return 0.0;
    }
    return integrator->last_error;
}

SimResult integrator_step_context(Integrator*        integrator,
                                  struct SimContext* context,
                                  struct SimBackend* backend,
                                  double             dt) {
    SimField* primary_field;
    SimResult result;
    Integrator* previous_stepping = NULL;
#if !defined(NDEBUG)
    uint64_t* field_sigs = NULL;
    size_t    sig_count  = 0U;
#endif

    if (integrator == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    previous_stepping = sim_integrator_state_stepping(&context->integrators);
    sim_integrator_state_set_stepping(&context->integrators, integrator);

    /* Ensure operators see the same dt as the integrator step. */
    sim_context_set_timestep(context, dt);

    result = sim_context_prepare_plan(context);

    if (result != SIM_RESULT_OK) {
        sim_integrator_state_set_stepping(&context->integrators, previous_stepping);
        return result;
    }

    primary_field = sim_context_field(context, integrator->target_field_index);

    if (primary_field == NULL) {
        sim_integrator_state_set_stepping(&context->integrators, previous_stepping);
        return SIM_RESULT_NOT_FOUND;
    }

#if !defined(NDEBUG)
    sig_count = sim_context_field_count(context);
    if (sig_count > 0U) {
        field_sigs = (uint64_t*) calloc(sig_count, sizeof(*field_sigs));
        if (field_sigs == NULL) {
            sim_integrator_state_set_stepping(&context->integrators, previous_stepping);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        for (size_t fi = 0U; fi < sig_count; ++fi) {
            const SimField* field = sim_context_field(context, fi);
            uint64_t        h     = 0xcbf29ce484222325ULL;
            if (field != NULL) {
                SimFieldRepresentation repr = sim_field_representation(field);
                h ^= (uint64_t) (uintptr_t) sim_field_data_const(field);
                h = (h * 0x100000001b3ULL) ^ (uint64_t) sim_field_bytes(field);
                h = (h * 0x100000001b3ULL) ^ (uint64_t) field->element_size;
                h = (h * 0x100000001b3ULL) ^ (uint64_t) repr.domain;
                h = (h * 0x100000001b3ULL) ^ (uint64_t) repr.value_kind;
                h = (h * 0x100000001b3ULL) ^ (uint64_t) field->layout.rank;
                for (size_t d = 0U; d < field->layout.rank; ++d) {
                    h = (h * 0x100000001b3ULL) ^ (uint64_t) field->layout.shape[d];
                }
                for (size_t d = 0U; d < field->layout.rank; ++d) {
                    h = (h * 0x100000001b3ULL) ^ (uint64_t) field->layout.strides[d];
                }
            }
            field_sigs[fi] = h;
        }
    }
#endif

    integrator->is_complex               = sim_field_domain_is_complex(primary_field);
    integrator->split_feedback_dt        = 0.0;
    integrator->split_feedback_max_error = 0.0;
    integrator->split_feedback_substeps  = 0U;
    integrator->step(integrator, primary_field, dt);

#if !defined(NDEBUG)
    if (sig_count != sim_context_field_count(context)) {
        fprintf(stderr, "[ERROR] integrator_step_context: field count changed during step\n");
        assert(false && "illegal mid-step field table change");
    }
    for (size_t fi = 0U; fi < sig_count; ++fi) {
        const SimField* field = sim_context_field(context, fi);
        uint64_t        h     = 0xcbf29ce484222325ULL;
        if (field != NULL) {
            SimFieldRepresentation repr = sim_field_representation(field);
            h ^= (uint64_t) (uintptr_t) sim_field_data_const(field);
            h = (h * 0x100000001b3ULL) ^ (uint64_t) sim_field_bytes(field);
            h = (h * 0x100000001b3ULL) ^ (uint64_t) field->element_size;
            h = (h * 0x100000001b3ULL) ^ (uint64_t) repr.domain;
            h = (h * 0x100000001b3ULL) ^ (uint64_t) repr.value_kind;
            h = (h * 0x100000001b3ULL) ^ (uint64_t) field->layout.rank;
            for (size_t d = 0U; d < field->layout.rank; ++d) {
                h = (h * 0x100000001b3ULL) ^ (uint64_t) field->layout.shape[d];
            }
            for (size_t d = 0U; d < field->layout.rank; ++d) {
                h = (h * 0x100000001b3ULL) ^ (uint64_t) field->layout.strides[d];
            }
        }
        if (field_sigs != NULL && field_sigs[fi] != h) {
            fprintf(
                stderr,
                "[ERROR] integrator_step_context: field %zu representation changed during step\n",
                fi);
            assert(false && "illegal mid-step representation change");
        }
    }
    free(field_sigs);
#endif

    if (integrator->split_feedback_substeps > 0U) {
        if (integrator->split_feedback_dt > 0.0) {
            integrator->last_step = integrator->split_feedback_dt;
        }
        if (integrator->split_feedback_max_error > integrator->last_error) {
            integrator->last_error = integrator->split_feedback_max_error;
        }
    }

    result = SIM_RESULT_OK;
    (void) backend; /* Kept for signature compatibility. */
    sim_integrator_state_set_stepping(&context->integrators, previous_stepping);
    return result;
}
