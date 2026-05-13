/**
 * @file integrator.h
 * @brief Base definitions for time-integration schemes in libsimintegrators.
 * @ingroup oakfield_integrators
 *
 * Integrators advance one target field by repeatedly evaluating a drift
 * callback, optionally adding stochastic increments, and recording timestep
 * diagnostics for adaptive controllers and runtime reporting. The public
 * contract uses scalar counts: real fields have one scalar per element, while
 * complex fields have one `SimComplexDouble` per element and are passed through
 * callback buffers as `double*` storage with real/imag components preserved by
 * the field representation.
 */
#ifndef LIBSIMINTEGRATORS_INTEGRATOR_H
#define LIBSIMINTEGRATORS_INTEGRATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oakfield/field.h"

struct SimContext;
struct SimBackend;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Alias matching the simulation field type used by integrator APIs.
 */
typedef SimField Field;

struct Integrator;

/**
 * @brief Drift callback computing deterministic derivatives.
 *
 * The callback must evaluate the derivative at @p state without taking
 * ownership of any buffer. Real states contain @p count doubles. Complex states
 * are passed as storage compatible with `SimComplexDouble[count]`, cast through
 * `double*` for the shared callback signature.
 *
 * @param integrator Calling integrator instance.
 * @param field Field metadata associated with the state.
 * @param state Read-only state vector owned by the caller.
 * @param[out] out_derivative Output buffer receiving d(state)/dt.
 * @param count Number of real entries or complex entries in the state vector.
 * @return #SIM_RESULT_OK on success, or an error code propagated by the stepper.
 */
typedef SimResult (*IntegratorDriftFn)(struct Integrator *integrator, const Field *field,
                                       const double *state, double *out_derivative, size_t count);

/**
 * @brief Hook producing stochastic source samples.
 *
 * The hook fills caller-owned storage with unit-variance, zero-mean samples
 * before `stochastic_strength * sqrt(dt)` scaling is applied.
 *
 * @param integrator Calling integrator instance.
 * @param field Field metadata associated with the state.
 * @param[out] out_noise Output buffer receiving @p count samples.
 * @param count Number of scalar elements to populate.
 * @return #SIM_RESULT_OK on success, or an error code when sampling fails.
 */
typedef SimResult (*IntegratorNoiseFn)(struct Integrator *integrator, const Field *field,
                                       double *out_noise, size_t count);

/**
 * @brief Optional teardown hook for integrator-owned state.
 *
 * @param integrator Integrator being destroyed; the hook may clear userdata.
 */
typedef void (*IntegratorDestroyFn)(struct Integrator *integrator);

/**
 * @brief Integrator step function signature.
 *
 * The implementation updates @p field in place and records `last_step`,
 * `last_error`, attempt counters, and the next `current_dt` suggestion.
 *
 * @param integrator Configured integrator instance.
 * @param field Target field advanced in place.
 * @param dt Requested timestep; nonpositive values let the integrator use its
 * current timestep suggestion.
 */
typedef void (*IntegratorStepFn)(struct Integrator *integrator, Field *field, double dt);

/**
 * @brief Configuration parameters for integrator construction.
 *
 * Zero-initialized optional fields select the defaults in
 * integrator_configure(). The drift callback is required for all concrete
 * integrators.
 */
typedef struct IntegratorConfig {
    IntegratorDriftFn drift;     /**< Deterministic drift evaluator (required). */
    IntegratorNoiseFn noise;     /**< Optional stochastic sample generator. */
    IntegratorDestroyFn destroy; /**< Optional teardown hook for integrator-owned state. */
    void *userdata;              /**< User payload forwarded to callbacks. */
    size_t target_field_index;   /**< Context field index advanced by this integrator. */
    double initial_dt;           /**< Initial timestep suggestion. */
    double min_dt;               /**< Lower bound on adaptive timesteps. */
    double max_dt;               /**< Upper bound on adaptive timesteps. */
    double tolerance;            /**< Error tolerance for adaptivity. */
    double safety;               /**< Safety factor for step adjustment. */
    bool adaptive;               /**< Enable adaptive timestep control. */
    bool enable_stochastic;      /**< Enable stochastic source injection. */
    double stochastic_strength;  /**< Scaling applied to stochastic sources. */
    uint32_t random_seed;        /**< Seed for the internal RNG. */
    size_t workspace_hint;       /**< Optional preallocation hint (elements). */
    double subordination_alpha;  /**< Optional alpha override for the subordination integrator. */
    size_t subordination_quadrature_n; /**< Optional quadrature sample count override for
                                          subordination. */
} IntegratorConfig;

/**
 * @brief Integrator instance shared across schemes.
 *
 * The struct is public so host runtimes can inspect diagnostics and attach
 * context userdata. Buffers and drift snapshots are owned by the integrator
 * after successful configuration and are released by integrator_destroy().
 */
typedef struct Integrator {
    char name[32];                     /**< Identifier for diagnostics. */
    IntegratorStepFn step;             /**< Active stepper implementation. */
    IntegratorDriftFn drift;           /**< Deterministic drift evaluator. */
    IntegratorNoiseFn noise;           /**< Stochastic hook (optional). */
    IntegratorDestroyFn destroy;       /**< Optional teardown hook for owned state. */
    void *userdata;                    /**< Opaque payload. */
    bool adaptive;                     /**< Whether adaptive stepping is enabled. */
    bool enable_stochastic;            /**< Toggle for stochastic contributions. */
    double min_dt;                     /**< Minimum admissible timestep. */
    double max_dt;                     /**< Maximum admissible timestep. */
    double tolerance;                  /**< Error tolerance target. */
    double safety;                     /**< Safety multiplier for adaptation. */
    double current_dt;                 /**< Suggested timestep for the next call. */
    double last_step;                  /**< Timestep realized by the last update. */
    double last_error;                 /**< Error norm recorded during last update. */
    double stochastic_strength;        /**< Amplitude of stochastic increments. */
    uint32_t rng_state;                /**< RNG state for stochastic hooks. */
    uint32_t last_attempt_count;       /**< Attempts spent producing the last realized step. */
    uint32_t last_rejection_count;     /**< Rejected attempts before the last realized step. */
    double **buffers;                  /**< Scratch buffers for intermediate storage. */
    size_t buffer_count;               /**< Number of allocated buffers. */
    size_t buffer_elements;            /**< Length (in scalars) of each buffer. */
    size_t buffer_element_size;        /**< Size of each scalar in buffers (bytes). */
    void **drift_snapshots;            /**< Reusable per-field snapshots for drift evaluation. */
    size_t *drift_snapshot_sizes;      /**< Bytes captured into each drift snapshot slot. */
    size_t *drift_snapshot_capacities; /**< Allocated bytes per drift snapshot slot. */
    size_t drift_snapshot_count;       /**< Number of drift snapshot slots currently tracked. */
    void *drift_state_scratch; /**< Reusable scratch copy of the target state during drift. */
    size_t drift_state_scratch_capacity; /**< Allocated bytes in @ref drift_state_scratch. */
    size_t target_field_index;           /**< Context field index advanced by this integrator. */
    double subordination_alpha;        /**< Active alpha when using the subordination integrator. */
    size_t subordination_quadrature_n; /**< Active quadrature sample count for subordination. */
    bool is_complex;                   /**< Whether the integrator is handling complex data. */
    double split_feedback_dt; /**< Sum of substep durations reported during the current step. */
    double split_feedback_max_error;  /**< Maximum error estimate reported via split feedback. */
    uint32_t split_feedback_substeps; /**< Count of substeps contributing feedback this step. */
} Integrator;

/**
 * @brief Initialize an integrator object.
 *
 * @param[out] integrator Target integrator; overwritten on success.
 * @param name Human-readable identifier copied into `Integrator::name`.
 * @param step_fn Step function implementation; must be non-NULL.
 * @param config Configuration parameters, or NULL to use defaults.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT when required
 * pointers or drift callbacks are missing, or #SIM_RESULT_OUT_OF_MEMORY when
 * preallocated workspace cannot be prepared.
 */
SimResult integrator_configure(Integrator *integrator, const char *name, IntegratorStepFn step_fn,
                               const IntegratorConfig *config);

/**
 * @brief Release resources held by an integrator.
 *
 * Calls the optional destroy hook, frees scratch buffers and drift snapshots,
 * and zeroes the struct. Passing NULL is a no-op.
 *
 * @param integrator Integrator to destroy; may be NULL.
 */
void integrator_destroy(Integrator *integrator);

/**
 * @brief Ensure scratch buffers are available.
 *
 * Buffers are aligned for SIMD-friendly field kernels and sized for either
 * real doubles or `SimComplexDouble` entries depending on `integrator->is_complex`.
 *
 * @param integrator Integrator instance.
 * @param buffers Number of buffers required; must be greater than zero.
 * @param elements Number of real or complex entries per buffer.
 * @return #SIM_RESULT_OK when the workspace is ready,
 * #SIM_RESULT_INVALID_ARGUMENT for invalid input, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult integrator_ensure_workspace(Integrator *integrator, size_t buffers, size_t elements);

/**
 * @brief Access a real workspace buffer.
 *
 * @param integrator Integrator instance.
 * @param index Zero-based buffer index.
 * @return Pointer to the buffer, or NULL when @p index is unavailable.
 */
double *integrator_buffer(Integrator *integrator, size_t index);

/**
 * @brief Compute the scalar length of a field.
 *
 * @param field Field instance.
 * @return Number of real entries for real fields or complex entries for complex
 * fields; returns 0 for NULL, empty, or unsupported element sizes.
 */
size_t integrator_state_length(const Field *field);

/**
 * @brief Clamp a timestep to configured bounds.
 *
 * @param integrator Integrator instance; NULL leaves @p dt unchanged.
 * @param dt Proposed timestep.
 * @return Timestep constrained to `[min_dt, max_dt]` when possible.
 */
double integrator_clamp_dt(const Integrator *integrator, double dt);

/**
 * @brief Suggest a new timestep based on an error estimate.
 *
 * The controller applies the configured tolerance, safety factor, and growth
 * clamps. Non-adaptive integrators simply return the clamped current timestep.
 *
 * @param integrator Integrator instance.
 * @param dt Current timestep.
 * @param error_norm Dimensionless normalized error estimate.
 * @param method_order Order of the embedded error estimator; values below 1
 * are treated as first order.
 * @return Suggested timestep for the next iteration, clamped to integrator bounds.
 */
double integrator_suggest_dt(const Integrator *integrator, double dt, double error_norm,
                             double method_order);

/**
 * @brief Shrink a rejected timestep using the observed error when available.
 *
 * Falls back to a conservative halving strategy if the controller would not
 * reduce the step or the estimate is not finite.
 *
 * @param integrator Integrator instance.
 * @param dt Rejected timestep.
 * @param error_norm Dimensionless normalized error estimate.
 * @param method_order Order of the embedded error estimator.
 * @return Smaller timestep clamped to the configured timestep bounds.
 */
double integrator_reject_dt(const Integrator *integrator, double dt, double error_norm,
                            double method_order);

/**
 * @brief Measure scaled infinity-norm error between two real state vectors.
 *
 * Each component is normalized by `max(abs(a[i]), abs(b[i]), 1)`.
 *
 * @param a First real state vector.
 * @param b Second real state vector.
 * @param count Number of double entries.
 * @return Maximum normalized component difference, or 0 for invalid/empty input.
 */
double integrator_measure_error(const double *a, const double *b, size_t count);

/**
 * @brief Generate a standard normal variate using the integrator RNG.
 *
 * @param integrator Integrator containing RNG state.
 * @return One zero-mean, unit-variance Gaussian sample, or 0 for NULL input.
 */
double integrator_rng_normal(Integrator *integrator);

/**
 * @brief Report allocated workspace bytes owned by the integrator buffers.
 *
 * @param integrator Integrator to inspect.
 * @return Buffer footprint in bytes, or 0 for NULL.
 */
uint64_t integrator_workspace_bytes(const Integrator *integrator);

/**
 * @brief Report reusable drift scratch bytes owned by the integrator.
 *
 * @param integrator Integrator to inspect.
 * @return Drift scratch and snapshot footprint in bytes, or 0 for NULL.
 */
uint64_t integrator_drift_scratch_bytes(const Integrator *integrator);

/**
 * @brief Apply stochastic increments to the field when enabled.
 *
 * Built-in real noise laws are Gaussian, uniform, and Laplace, all normalized
 * to unit variance before scaling by `stochastic_strength * sqrt(dt)`. Complex
 * storage currently receives Gaussian real/imag samples.
 *
 * @param integrator Integrator instance.
 * @param field Field to modify in place.
 * @param scratch Scratch buffer to reuse for noise samples.
 * @param count Number of real entries or complex entries in the state.
 * @param dt Duration of the realized timestep.
 */
void integrator_apply_stochastic(Integrator *integrator, Field *field, double *scratch,
                                 size_t count, double dt);

/**
 * @brief Retrieve the timestep used by the previous update.
 *
 * @param integrator Integrator to inspect.
 * @return Last realized timestep, or 0 for NULL.
 */
double integrator_last_step(const Integrator *integrator);

/**
 * @brief Retrieve the suggested timestep for the next call.
 *
 * @param integrator Integrator to inspect.
 * @return Current timestep suggestion, or 0 for NULL.
 */
double integrator_next_step(const Integrator *integrator);

/**
 * @brief Measure scaled RMS error between two complex state vectors.
 *
 * Component differences use squared complex magnitude and are normalized by
 * `max(|a[i]|^2, |b[i]|^2, 1)` before the mean square root is taken.
 *
 * @param a First complex state vector.
 * @param b Second complex state vector.
 * @param n Number of complex entries.
 * @return Normalized RMS complex difference, or 0 for invalid/empty input.
 */
double integrator_measure_error_complex(const SimComplexDouble *a, const SimComplexDouble *b,
                                        size_t n);

/**
 * @brief Access a complex workspace buffer.
 *
 * @param integrator Integrator instance.
 * @param index Zero-based buffer index.
 * @return Pointer to complex scratch storage, or NULL when unavailable.
 */
SimComplexDouble *integrator_buffer_complex(Integrator *integrator, unsigned int index);

/**
 * @brief Fill a buffer with zero-mean, unit-variance Gaussian noise samples.
 *
 * @param integrator Integrator containing RNG state.
 * @param field Field associated with the samples; currently informational.
 * @param[out] out_noise Buffer receiving @p count samples.
 * @param count Number of samples to generate.
 * @return #SIM_RESULT_OK, or #SIM_RESULT_INVALID_ARGUMENT for NULL inputs.
 */
SimResult integrator_noise_gaussian(struct Integrator *integrator, const Field *field,
                                    double *out_noise, size_t count);

/**
 * @brief Fill a buffer with zero-mean, unit-variance uniform noise samples.
 *
 * Samples are drawn from `[-sqrt(3), sqrt(3)]`.
 *
 * @param integrator Integrator containing RNG state.
 * @param field Field associated with the samples; currently informational.
 * @param[out] out_noise Buffer receiving @p count samples.
 * @param count Number of samples to generate.
 * @return #SIM_RESULT_OK, or #SIM_RESULT_INVALID_ARGUMENT for NULL inputs.
 */
SimResult integrator_noise_uniform(struct Integrator *integrator, const Field *field,
                                   double *out_noise, size_t count);

/**
 * @brief Fill a buffer with zero-mean, unit-variance Laplace noise samples.
 *
 * @param integrator Integrator containing RNG state.
 * @param field Field associated with the samples; currently informational.
 * @param[out] out_noise Buffer receiving @p count samples.
 * @param count Number of samples to generate.
 * @return #SIM_RESULT_OK, or #SIM_RESULT_INVALID_ARGUMENT for NULL inputs.
 */
SimResult integrator_noise_laplace(struct Integrator *integrator, const Field *field,
                                   double *out_noise, size_t count);

/**
 * @brief Retrieve the error norm reported by the last step.
 *
 * @param integrator Integrator to inspect.
 * @return Last normalized error estimate, or 0 for NULL.
 */
double integrator_last_error(const Integrator *integrator);

/**
 * @brief Drift helper evaluating the current context plan side-effect free.
 *
 * The integrator userdata must point at a @ref SimContext. The helper snapshots
 * context fields, writes @p state into @p field, executes the prepared plan, and
 * restores the original field data before returning the finite-difference drift.
 *
 * @param integrator Integrator whose userdata is a `SimContext*`.
 * @param field Target field metadata.
 * @param state Candidate state vector owned by the caller.
 * @param[out] out_derivative Buffer receiving the context-derived derivative.
 * @param count Number of real entries or complex entries in @p state.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or an error propagated
 * from context plan preparation/execution.
 */
SimResult integrator_context_drift(Integrator *integrator, const Field *field, const double *state,
                                   double *out_derivative, size_t count);

/**
 * @brief Step a context-backed integrator without advancing context time.
 *
 * This updates field state and integrator metadata only. Call
 * @ref sim_context_accept_step after a successful step if you want the context
 * clock and step index to advance. The backend parameter is retained for API
 * compatibility and is not used by the current implementation.
 *
 * @param integrator Integrator to execute.
 * @param context Simulation context containing the target field.
 * @param backend Optional backend pointer reserved for future dispatch.
 * @param dt Requested timestep passed to the integrator step function.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_FOUND,
 * #SIM_RESULT_OUT_OF_MEMORY in debug invariant tracking, or a plan preparation
 * error from the context.
 */
SimResult integrator_step_context(Integrator *integrator, struct SimContext *context,
                                  struct SimBackend *backend, double dt);

#ifdef __cplusplus
}
#endif

#endif /* LIBSIMINTEGRATORS_INTEGRATOR_H */
