/**
 * @file subordination.c
 * @brief Midpoint-quadrature subordination integrator for context-backed flows.
 *
 * The subordination step approximates a subordinated semigroup by sampling
 * drift evaluations over an auxiliary time variable `s` with weight
 * `exp(-pow(s, alpha))`. The implementation is complex-capable, temporarily
 * changes the owning context timestep for each quadrature node, and restores
 * the original timestep before each return path.
 */

#include "oakfield/integrator.h"
#include "oakfield/sim_context.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Runtime state for the subordination quadrature rule.
 */
typedef struct {
    double             alpha;        /* Subordinator exponent, clamped to (0, 1]. */
    size_t             quadrature_n; /* number of quadrature samples */
    double             truncation;   /* finite quadrature upper limit */
    struct SimContext* context;      /* owning context for drift evaluation */
} SubordinationState;

/**
 * @brief Release subordination state owned through Integrator::userdata.
 *
 * @param integrator Integrator being destroyed; NULL is ignored.
 */
static void integrator_subordination_destroy_state(Integrator* integrator) {
    if (integrator == NULL || integrator->userdata == NULL) {
        return;
    }
    free(integrator->userdata);
    integrator->userdata = NULL;
}

/**
 * @brief Evaluate context drift through a proxy integrator.
 *
 * The configured subordination integrator stores its private state in
 * `userdata`, so this helper builds a temporary proxy whose userdata points to
 * the owning SimContext expected by integrator_context_drift().
 *
 * @param integrator Subordination integrator containing private state.
 * @param field Target field metadata.
 * @param state Candidate state vector.
 * @param[out] out_derivative Buffer receiving the context-derived derivative.
 * @param count Number of real or complex entries in @p state.
 * @return #SIM_RESULT_OK or an error from validation/allocation/context drift.
 */
static SimResult subordination_drift(struct Integrator* integrator,
                                     const Field*       field,
                                     const double*      state,
                                     double*            out_derivative,
                                     size_t             count) {
    SubordinationState* sstate;
    Integrator          proxy;
    SimResult           result;

    if (!integrator) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    sstate = (SubordinationState*) integrator->userdata;
    if (sstate == NULL || sstate->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    proxy          = *integrator;
    proxy.destroy  = NULL;
    proxy.userdata = sstate->context;
    proxy.buffers = NULL;
    proxy.buffer_count = 0U;
    proxy.buffer_elements = 0U;
    proxy.buffer_element_size = sizeof(double);

    result = integrator_context_drift(&proxy, field, state, out_derivative, count);
    integrator_destroy(&proxy);
    return result;
}

/**
 * @brief Advance one field by midpoint quadrature over subordinated drift.
 *
 * Each node evaluates the drift at `eval_dt = dt * s`, falling back to the
 * context base timestep and then 1.0 if no positive timestep is available. The
 * accumulated drift is multiplied by the requested outer @p dt before being
 * applied to the target field.
 *
 * @param integrator Configured subordination integrator.
 * @param field Field updated in place.
 * @param dt Outer timestep for the subordinated update.
 */
static void subordination_step(Integrator* integrator, Field* field, double dt) {
    if (!integrator || !field || !integrator->drift)
        return;

    SubordinationState* sstate = (SubordinationState*) integrator->userdata;
    if (sstate == NULL || sstate->context == NULL)
        return;

    SimContext*  ctx     = sstate->context;
    const double base_dt = sim_context_timestep(ctx);

    bool is_complex        = sim_field_domain_is_complex(field);
    integrator->is_complex = is_complex;

    double alpha  = sstate->alpha;
    size_t N      = sstate->quadrature_n;
    double Lambda = sstate->truncation;

    if (N == 0U || !(Lambda > 0.0))
        return;

    double h = Lambda / (double) N;

    if (!is_complex) {
        /* ------------------- REAL BRANCH ------------------- */
        double* state = (double*) sim_field_data(field);
        if (!state)
            return;

        size_t count = integrator_state_length(field);
        if (count == 0U)
            return;

        if (integrator_ensure_workspace(integrator, 3U, count) != SIM_RESULT_OK)
            return;

        double* drift = integrator_buffer(integrator, 0U);
        double* accum = integrator_buffer(integrator, 1U);
        memset(accum, 0, count * sizeof(double));

        for (size_t i = 0; i < N; ++i) {
            double s      = (i + 0.5) * h;
            double weight = exp(-pow(s, alpha)) * h;

            double eval_dt = dt * s;
            if (!(eval_dt > 0.0))
                eval_dt = dt;
            if (!(eval_dt > 0.0))
                eval_dt = base_dt;
            if (!(eval_dt > 0.0))
                eval_dt = 1.0;
            sim_context_set_timestep(ctx, eval_dt);

            SimResult drift_result = integrator->drift(integrator, field, state, drift, count);
            sim_context_set_timestep(ctx, base_dt);
            if (drift_result != SIM_RESULT_OK)
                return;

            for (size_t j = 0; j < count; ++j)
                accum[j] += weight * drift[j];
        }

        for (size_t j = 0; j < count; ++j)
            state[j] += dt * accum[j];

        integrator_apply_stochastic(integrator, field, drift, count, dt);

        integrator->last_step  = dt;
        integrator->last_error = 0.0;
        integrator->last_attempt_count   = 1U;
        integrator->last_rejection_count = 0U;
        integrator->current_dt = integrator_clamp_dt(integrator, dt);
        return;
    }

    /* ------------------- COMPLEX BRANCH ------------------- */
    SimComplexDouble* state = sim_field_complex_data(field);
    if (!state)
        return;

    size_t count = integrator_state_length(field);
    if (count == 0U)
        return;

    if (integrator_ensure_workspace(integrator, 3U, count) != SIM_RESULT_OK)
        return;

    SimComplexDouble* drift = integrator_buffer_complex(integrator, 0U);
    SimComplexDouble* accum = integrator_buffer_complex(integrator, 1U);
    memset(accum, 0, count * sizeof(SimComplexDouble));

    for (size_t i = 0; i < N; ++i) {
        double s      = (i + 0.5) * h;
        double weight = exp(-pow(s, alpha)) * h;

        double eval_dt = dt * s;
        if (!(eval_dt > 0.0))
            eval_dt = dt;
        if (!(eval_dt > 0.0))
            eval_dt = base_dt;
        if (!(eval_dt > 0.0))
            eval_dt = 1.0;
        sim_context_set_timestep(ctx, eval_dt);

        SimResult drift_result =
            integrator->drift(integrator, field, (double*) state, (double*) drift, count);
        sim_context_set_timestep(ctx, base_dt);
        if (drift_result != SIM_RESULT_OK)
            return;

        for (size_t j = 0; j < count; ++j) {
            accum[j].re += weight * drift[j].re;
            accum[j].im += weight * drift[j].im;
        }
    }

    for (size_t j = 0; j < count; ++j) {
        state[j].re += dt * accum[j].re;
        state[j].im += dt * accum[j].im;
    }

    integrator_apply_stochastic(integrator, field, (double*) drift, count, dt);

    integrator->last_step  = dt;
    integrator->last_error = 0.0;
    integrator->last_attempt_count   = 1U;
    integrator->last_rejection_count = 0U;
    integrator->current_dt = integrator_clamp_dt(integrator, dt);
}

/**
 * @brief Create a subordination integrator.
 *
 * A useful instance needs a `SimContext*` in `config->userdata`; without one,
 * construction can succeed but the stepper has no context drift to evaluate.
 * Optional `subordination_alpha` values greater than zero override the default
 * 0.75 and are clamped to 1.0 from above. Optional positive
 * `subordination_quadrature_n` values override the default midpoint sample
 * count.
 *
 * @param config Integrator configuration containing context userdata.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_OUT_OF_MEMORY,
 * or an error from integrator_configure().
 */
SimResult integrator_subordination_create(const IntegratorConfig* config, Integrator* out) {
    if (!out)
        return SIM_RESULT_INVALID_ARGUMENT;

    struct SimContext*  ctx   = (config != NULL) ? (struct SimContext*) config->userdata : NULL;
    SubordinationState* state = (SubordinationState*) calloc(1, sizeof(SubordinationState));
    if (!state)
        return SIM_RESULT_OUT_OF_MEMORY;

    state->alpha        = 0.75;
    state->quadrature_n = 128;
    state->truncation   = 10.0;
    state->context      = ctx;

    if (config != NULL) {
        if (config->subordination_alpha > 0.0) {
            state->alpha = config->subordination_alpha;
            if (state->alpha > 1.0) {
                state->alpha = 1.0;
            }
        }
        if (config->subordination_quadrature_n > 0U) {
            state->quadrature_n = config->subordination_quadrature_n;
        }
    }

    IntegratorConfig local = (config != NULL) ? *config : (IntegratorConfig) { 0 };
    local.userdata         = state;
    local.drift            = subordination_drift;
    local.destroy          = integrator_subordination_destroy_state;

    SimResult result = integrator_configure(out, "subordination", subordination_step, &local);
    if (result != SIM_RESULT_OK) {
        free(state);
        return result;
    }

    out->subordination_alpha = state->alpha;
    out->subordination_quadrature_n = state->quadrature_n;

    return SIM_RESULT_OK;
}
