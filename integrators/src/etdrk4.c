/**
 * @file etdrk4.c
 * @brief Exponential time-differencing RK4 integrator for semilinear complex plans.
 *
 * ETDRK4 is specialized to context-backed complex fields whose scheduler plan
 * can be split into exact linear-flow operators and same-field nonlinear or
 * dt-scaled source operators. The implementation snapshots context fields
 * around each stage, applies exact linear flows through the existing operator
 * surface, and approximates phi-function weights with fixed four-point
 * Gauss-Legendre quadrature.
 */

#include "oakfield/integrator.h"

#include "oakfield/operator.h"
#include "operator_semilinear.h"
#include "oakfield/sim_context.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ETDRK4_WORKSPACE_BUFFERS 14U
#define ETDRK4_GAUSS_POINTS 4U

/**
 * @brief Private ETDRK4 state attached to Integrator::userdata.
 */
typedef struct ETDRK4State {
    SimContext* context;
    bool        warned_unsupported;
} ETDRK4State;

/**
 * @brief Full-context field snapshots used to make stage evaluations reversible.
 */
typedef struct ETDRK4Snapshots {
    void**  copies;
    size_t* sizes;
    size_t  count;
} ETDRK4Snapshots;

static void etdrk4_destroy_snapshots(ETDRK4Snapshots* snapshots);

static const double k_etdrk4_nodes[ETDRK4_GAUSS_POINTS] = {
    0.06943184420297371,
    0.33000947820757187,
    0.6699905217924281,
    0.9305681557970262,
};

static const double k_etdrk4_weights[ETDRK4_GAUSS_POINTS] = {
    0.17392742256872692,
    0.32607257743127305,
    0.32607257743127305,
    0.17392742256872692,
};

/**
 * @brief Return true when an explicit field-index list refers only to one field.
 *
 * Empty lists are treated as unconstrained by the caller-side mask checks.
 */
static bool etdrk4_indices_only_field(const size_t* indices, size_t count, size_t field_index) {
    if (indices == NULL) {
        return count == 0U;
    }
    for (size_t i = 0U; i < count; ++i) {
        if (indices[i] != field_index) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Return true when a dependency mask is empty or contains exactly one field.
 */
static bool etdrk4_mask_only_field(uint64_t mask, size_t field_index) {
    if (mask == 0ULL) {
        return true;
    }
    if (field_index >= 64U) {
        return false;
    }
    return mask == (1ULL << field_index);
}

/**
 * @brief Return true when an operator reads only the ETDRK4 target field.
 */
static bool etdrk4_operator_reads_only_field(const SimOperator* op, size_t field_index) {
    if (op == NULL) {
        return false;
    }
    return etdrk4_mask_only_field(op->read_mask, field_index) &&
           etdrk4_indices_only_field(op->read_indices, op->read_index_count, field_index);
}

/**
 * @brief Return true when an operator writes the ETDRK4 target field and nothing else.
 */
static bool etdrk4_operator_writes_target(const SimOperator* op, size_t field_index) {
    if (op == NULL) {
        return false;
    }
    if (!etdrk4_mask_only_field(op->write_mask, field_index)) {
        return false;
    }
    if (!etdrk4_indices_only_field(op->write_indices, op->write_index_count, field_index)) {
        return false;
    }
    if (op->write_index_count > 0U) {
        return true;
    }
    if (op->write_mask != 0ULL) {
        return true;
    }
    return false;
}

/**
 * @brief Capture byte-for-byte copies of every field in the context.
 *
 * Stage evaluation mutates context fields through ordinary operator dispatch.
 * These snapshots allow each stage to restore the base state before applying a
 * candidate stage vector.
 *
 * @param ctx Context whose fields are copied.
 * @param[out] snapshots Snapshot container initialized by this function.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
static SimResult etdrk4_capture_snapshots(SimContext* ctx, ETDRK4Snapshots* snapshots) {
    if (ctx == NULL || snapshots == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    memset(snapshots, 0, sizeof(*snapshots));
    snapshots->count = sim_context_field_count(ctx);
    if (snapshots->count == 0U) {
        return SIM_RESULT_OK;
    }

    snapshots->copies = (void**) calloc(snapshots->count, sizeof(void*));
    snapshots->sizes  = (size_t*) calloc(snapshots->count, sizeof(size_t));
    if (snapshots->copies == NULL || snapshots->sizes == NULL) {
        free(snapshots->copies);
        free(snapshots->sizes);
        memset(snapshots, 0, sizeof(*snapshots));
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    for (size_t fi = 0U; fi < snapshots->count; ++fi) {
        SimField*   field = sim_context_field(ctx, fi);
        size_t      bytes = sim_field_bytes(field);
        const void* src   = sim_field_data_const(field);

        snapshots->sizes[fi] = bytes;
        if (bytes == 0U || src == NULL) {
            continue;
        }

        snapshots->copies[fi] = malloc(bytes);
        if (snapshots->copies[fi] == NULL) {
            etdrk4_destroy_snapshots(snapshots);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        memcpy(snapshots->copies[fi], src, bytes);
    }

    return SIM_RESULT_OK;
}

/**
 * @brief Release snapshot buffers and reset the snapshot container.
 */
static void etdrk4_destroy_snapshots(ETDRK4Snapshots* snapshots) {
    if (snapshots == NULL) {
        return;
    }
    if (snapshots->copies != NULL) {
        for (size_t fi = 0U; fi < snapshots->count; ++fi) {
            free(snapshots->copies[fi]);
        }
    }
    free(snapshots->copies);
    free(snapshots->sizes);
    memset(snapshots, 0, sizeof(*snapshots));
}

/**
 * @brief Restore all captured context fields from a snapshot set.
 *
 * @param ctx Context whose fields are restored.
 * @param snapshots Previously captured snapshot set.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_NOT_FOUND.
 */
static SimResult etdrk4_restore_snapshots(SimContext* ctx, const ETDRK4Snapshots* snapshots) {
    if (ctx == NULL || snapshots == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t fi = 0U; fi < snapshots->count; ++fi) {
        SimField* field = sim_context_field(ctx, fi);
        void*     dst   = sim_field_data(field);
        if (field == NULL) {
            return SIM_RESULT_NOT_FOUND;
        }
        if (snapshots->sizes[fi] == 0U || snapshots->copies[fi] == NULL || dst == NULL) {
            continue;
        }
        memcpy(dst, snapshots->copies[fi], snapshots->sizes[fi]);
    }

    return SIM_RESULT_OK;
}

/**
 * @brief Copy the target complex field into caller-owned state storage.
 */
static SimResult etdrk4_copy_field(const Field* field,
                                   SimComplexDouble* out,
                                   size_t            count) {
    const SimComplexDouble* data = sim_field_complex_data_const(field);
    if (data == NULL || out == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    memcpy(out, data, count * sizeof(SimComplexDouble));
    return SIM_RESULT_OK;
}

/**
 * @brief Replace the target complex field with caller-provided state storage.
 */
static SimResult etdrk4_write_field(Field* field, const SimComplexDouble* state, size_t count) {
    SimComplexDouble* data = sim_field_complex_data(field);
    if (data == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    memcpy(data, state, count * sizeof(SimComplexDouble));
    return SIM_RESULT_OK;
}

static void etdrk4_zero(SimComplexDouble* values, size_t count) {
    memset(values, 0, count * sizeof(SimComplexDouble));
}

static void etdrk4_axpy(SimComplexDouble* dst,
                        double            scale,
                        const SimComplexDouble* src,
                        size_t            count) {
    for (size_t i = 0U; i < count; ++i) {
        dst[i].re += scale * src[i].re;
        dst[i].im += scale * src[i].im;
    }
}

static void etdrk4_linear_combine(SimComplexDouble*       dst,
                                  const SimComplexDouble* lhs,
                                  double                  lhs_scale,
                                  const SimComplexDouble* rhs,
                                  double                  rhs_scale,
                                  size_t                  count) {
    for (size_t i = 0U; i < count; ++i) {
        dst[i].re = lhs_scale * lhs[i].re + rhs_scale * rhs[i].re;
        dst[i].im = lhs_scale * lhs[i].im + rhs_scale * rhs[i].im;
    }
}

/**
 * @brief Classify a prepared scheduler plan into linear-flow and nonlinear stages.
 *
 * ETDRK4 accepts only operators that read and write the target field. Exact
 * linear-flow operators are applied through the semigroup path; dt-scaled
 * increments and general nonlinear operators are evaluated as nonlinear stage
 * derivatives. Noise and cross-field operators are rejected.
 *
 * @param state ETDRK4 private state containing the context.
 * @param field Target complex field.
 * @param target_field_index Field index advanced by the integrator.
 * @param[out] out_linear_indices Heap array of exact-linear operator indices.
 * @param[out] out_linear_count Number of linear operator indices.
 * @param[out] out_nonlinear_indices Heap array of nonlinear/source operator indices.
 * @param[out] out_nonlinear_count Number of nonlinear operator indices.
 * @return #SIM_RESULT_OK, #SIM_RESULT_TYPE_MISMATCH, #SIM_RESULT_NOT_SUPPORTED,
 * #SIM_RESULT_NOT_FOUND, #SIM_RESULT_OUT_OF_MEMORY, or a classifier error.
 */
static SimResult etdrk4_classify_plan(ETDRK4State* state,
                                      Field*       field,
                                      size_t       target_field_index,
                                      size_t**     out_linear_indices,
                                      size_t*      out_linear_count,
                                      size_t**     out_nonlinear_indices,
                                      size_t*      out_nonlinear_count) {
    SimContext* ctx;
    size_t      plan_count;
    size_t*     linear_indices     = NULL;
    size_t*     nonlinear_indices  = NULL;
    size_t      linear_count       = 0U;
    size_t      nonlinear_count    = 0U;

    if (out_linear_indices != NULL) {
        *out_linear_indices = NULL;
    }
    if (out_linear_count != NULL) {
        *out_linear_count = 0U;
    }
    if (out_nonlinear_indices != NULL) {
        *out_nonlinear_indices = NULL;
    }
    if (out_nonlinear_count != NULL) {
        *out_nonlinear_count = 0U;
    }

    if (state == NULL || field == NULL || out_linear_indices == NULL || out_linear_count == NULL ||
        out_nonlinear_indices == NULL || out_nonlinear_count == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    ctx = state->context;
    if (ctx == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!sim_field_is_complex(field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    plan_count = ctx->scheduler.plan.count;
    linear_indices = (size_t*) calloc((plan_count > 0U) ? plan_count : 1U, sizeof(size_t));
    nonlinear_indices =
        (size_t*) calloc((plan_count > 0U) ? plan_count : 1U, sizeof(size_t));
    if (linear_indices == NULL || nonlinear_indices == NULL) {
        free(linear_indices);
        free(nonlinear_indices);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    for (size_t order_index = 0U; order_index < plan_count; ++order_index) {
        size_t              op_index = ctx->scheduler.plan.order[order_index];
        SimOperator*        op       = sim_operator_registry_get(&ctx->world.operators, op_index);
        SimResult           class_result;
        SimSemilinearTraits traits = sim_operator_semilinear_traits_defaults();

        if (op == NULL) {
            free(linear_indices);
            free(nonlinear_indices);
            return SIM_RESULT_NOT_FOUND;
        }

        if (!etdrk4_operator_writes_target(op, target_field_index) ||
            !etdrk4_operator_reads_only_field(op, target_field_index)) {
            free(linear_indices);
            free(nonlinear_indices);
            return SIM_RESULT_NOT_SUPPORTED;
        }

        if (op->info.is_noise) {
            free(linear_indices);
            free(nonlinear_indices);
            return SIM_RESULT_NOT_SUPPORTED;
        }

        class_result = sim_operator_classify_semilinear(ctx, op_index, target_field_index, &traits);
        if (class_result != SIM_RESULT_OK) {
            free(linear_indices);
            free(nonlinear_indices);
            return class_result;
        }

        if (traits.role == SIM_SEMILINEAR_ROLE_EXACT_LINEAR_FLOW) {
            linear_indices[linear_count++] = op_index;
            continue;
        }

        if (traits.role == SIM_SEMILINEAR_ROLE_DT_SCALED_INCREMENT ||
            traits.role == SIM_SEMILINEAR_ROLE_GENERAL_NONLINEAR) {
            nonlinear_indices[nonlinear_count++] = op_index;
            continue;
        }

        free(linear_indices);
        free(nonlinear_indices);
        return SIM_RESULT_NOT_SUPPORTED;
    }

    *out_linear_indices    = linear_indices;
    *out_linear_count      = linear_count;
    *out_nonlinear_indices = nonlinear_indices;
    *out_nonlinear_count   = nonlinear_count;
    return SIM_RESULT_OK;
}

/**
 * @brief Apply the exact linear subflow to a complex stage vector.
 *
 * The helper restores the base context snapshot, writes @p input to the target
 * field, sets the context timestep to @p dt_flow, applies all exact-linear
 * operators, restores the original timestep, and copies the resulting field to
 * @p output. With no linear operators or zero flow duration, it copies input to
 * output exactly.
 *
 * @return #SIM_RESULT_OK or an error from snapshot restoration, field writes,
 * operator lookup, or operator application.
 */
static SimResult etdrk4_apply_linear_flow(SimContext*             ctx,
                                          Field*                  field,
                                          const ETDRK4Snapshots*  snapshots,
                                          const size_t*           linear_indices,
                                          size_t                  linear_count,
                                          const SimComplexDouble* input,
                                          SimComplexDouble*       output,
                                          size_t                  count,
                                          double                  dt_flow,
                                          double                  restore_dt) {
    SimResult result;

    if (ctx == NULL || field == NULL || snapshots == NULL || input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (linear_count == 0U || dt_flow == 0.0) {
        memcpy(output, input, count * sizeof(SimComplexDouble));
        return SIM_RESULT_OK;
    }

    result = etdrk4_restore_snapshots(ctx, snapshots);
    if (result != SIM_RESULT_OK) {
        return result;
    }
    result = etdrk4_write_field(field, input, count);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    sim_context_set_timestep(ctx, dt_flow);
    for (size_t i = 0U; i < linear_count; ++i) {
        SimOperator* op = sim_operator_registry_get(&ctx->world.operators, linear_indices[i]);
        if (op == NULL) {
            sim_context_set_timestep(ctx, restore_dt);
            return SIM_RESULT_NOT_FOUND;
        }
        result = sim_context_apply_operator(ctx, op);
        if (result != SIM_RESULT_OK) {
            sim_context_set_timestep(ctx, restore_dt);
            return result;
        }
    }
    sim_context_set_timestep(ctx, restore_dt);

    return etdrk4_copy_field(field, output, count);
}

/**
 * @brief Evaluate nonlinear/source stage derivative for a candidate state.
 *
 * Operators are applied at @p eval_time using context timestep @p step_dt.
 * The derivative is recovered from `(updated_state - stage_state) / step_dt`
 * and the context time/timestep overrides are cleared before returning.
 *
 * @return #SIM_RESULT_OK or an error from snapshot restoration, field writes,
 * operator lookup, or operator application.
 */
static SimResult etdrk4_eval_nonlinear(SimContext*             ctx,
                                       Field*                  field,
                                       const ETDRK4Snapshots*  snapshots,
                                       const size_t*           nonlinear_indices,
                                       size_t                  nonlinear_count,
                                       const SimComplexDouble* stage_state,
                                       SimComplexDouble*       out_derivative,
                                       size_t                  count,
                                       double                  eval_time,
                                       double                  step_dt,
                                       double                  restore_dt) {
    SimResult         result;
    SimComplexDouble* data;

    if (ctx == NULL || field == NULL || snapshots == NULL || stage_state == NULL ||
        out_derivative == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (nonlinear_count == 0U) {
        etdrk4_zero(out_derivative, count);
        return SIM_RESULT_OK;
    }

    result = etdrk4_restore_snapshots(ctx, snapshots);
    if (result != SIM_RESULT_OK) {
        return result;
    }
    result = etdrk4_write_field(field, stage_state, count);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    sim_context_set_drift_time_override(ctx, eval_time);
    sim_context_set_timestep(ctx, step_dt);
    for (size_t i = 0U; i < nonlinear_count; ++i) {
        SimOperator* op = sim_operator_registry_get(&ctx->world.operators, nonlinear_indices[i]);
        if (op == NULL) {
            sim_context_set_timestep(ctx, restore_dt);
            sim_context_clear_drift_time_override(ctx);
            return SIM_RESULT_NOT_FOUND;
        }
        result = sim_context_apply_operator(ctx, op);
        if (result != SIM_RESULT_OK) {
            sim_context_set_timestep(ctx, restore_dt);
            sim_context_clear_drift_time_override(ctx);
            return result;
        }
    }
    sim_context_set_timestep(ctx, restore_dt);
    sim_context_clear_drift_time_override(ctx);

    data = sim_field_complex_data(field);
    if (data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < count; ++i) {
        out_derivative[i].re = (data[i].re - stage_state[i].re) / step_dt;
        out_derivative[i].im = (data[i].im - stage_state[i].im) / step_dt;
    }

    return SIM_RESULT_OK;
}

/**
 * @brief Quadrature weight for the half-step ETDRK4 `q` coefficient.
 */
static double etdrk4_weight_q(double theta, double step) {
    (void) theta;
    return 0.5 * step;
}

/**
 * @brief Quadrature polynomial for the first final-stage ETDRK4 coefficient.
 */
static double etdrk4_weight_b1(double theta, double step) {
    return step * (1.0 - 3.0 * theta + 2.0 * theta * theta);
}

/**
 * @brief Quadrature polynomial for the middle final-stage ETDRK4 coefficient.
 */
static double etdrk4_weight_b2(double theta, double step) {
    return 2.0 * step * (theta - theta * theta);
}

/**
 * @brief Quadrature polynomial for the last final-stage ETDRK4 coefficient.
 */
static double etdrk4_weight_b3(double theta, double step) {
    return step * (-theta + 2.0 * theta * theta);
}

/**
 * @brief Approximate an ETDRK4 phi-weighted linear-flow integral.
 *
 * Fixed four-point Gauss-Legendre nodes sample the linear flow at
 * `(1 - theta) * step` scaled by @p flow_scale. The caller supplies the
 * polynomial weight function for q, b1, b2, or b3. If @p scratch aliases input
 * or output, a temporary heap buffer is used to preserve the contract.
 *
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_OUT_OF_MEMORY,
 * or an error from applying the linear flow.
 */
static SimResult etdrk4_apply_quadrature(SimContext*                  ctx,
                                         Field*                       field,
                                         const ETDRK4Snapshots*       snapshots,
                                         const size_t*                linear_indices,
                                         size_t                       linear_count,
                                         const SimComplexDouble*      input,
                                         SimComplexDouble*            output,
                                         SimComplexDouble*            scratch,
                                         size_t                       count,
                                         double                       step,
                                         double                       flow_scale,
                                         double (*weight_fn)(double theta, double step),
                                         double                       restore_dt) {
    SimResult result;
    SimComplexDouble* node_scratch = scratch;
    bool              heap_scratch = false;

    if (ctx == NULL || field == NULL || snapshots == NULL || input == NULL || output == NULL ||
        scratch == NULL || weight_fn == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (scratch == input || scratch == output) {
        node_scratch = (SimComplexDouble*) malloc(count * sizeof(SimComplexDouble));
        if (node_scratch == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        heap_scratch = true;
    }

    etdrk4_zero(output, count);
    for (size_t i = 0U; i < ETDRK4_GAUSS_POINTS; ++i) {
        double theta     = k_etdrk4_nodes[i];
        double quad_w    = k_etdrk4_weights[i] * weight_fn(theta, step);
        double flow_dt   = flow_scale * (1.0 - theta) * step;

        if (quad_w == 0.0) {
            continue;
        }

        result = etdrk4_apply_linear_flow(
            ctx,
            field,
            snapshots,
            linear_indices,
            linear_count,
            input,
            node_scratch,
            count,
            flow_dt,
            restore_dt);
        if (result != SIM_RESULT_OK) {
            if (heap_scratch) {
                free(node_scratch);
            }
            return result;
        }
        etdrk4_axpy(output, quad_w, node_scratch, count);
    }

    if (heap_scratch) {
        free(node_scratch);
    }

    return SIM_RESULT_OK;
}

/**
 * @brief Release private ETDRK4 state owned by the integrator.
 */
static void etdrk4_destroy_state(Integrator* integrator) {
    if (integrator == NULL) {
        return;
    }
    free(integrator->userdata);
    integrator->userdata = NULL;
}

/**
 * @brief Adapter that exposes context drift through ETDRK4 private state.
 */
static SimResult etdrk4_context_drift(Integrator*   integrator,
                                      const Field*  field,
                                      const double* state,
                                      double*       out_derivative,
                                      size_t        count) {
    ETDRK4State* state_ptr;
    Integrator   proxy;

    if (integrator == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    state_ptr = (ETDRK4State*) integrator->userdata;
    if (state_ptr == NULL || state_ptr->context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    proxy          = *integrator;
    proxy.userdata = state_ptr->context;
    return integrator_context_drift(&proxy, field, state, out_derivative, count);
}

/**
 * @brief Advance one complex field with ETDRK4.
 *
 * The step is fixed-size: `adaptive` is disabled by the factory and
 * `last_error` is reported as zero. Unsupported real fields or unsupported
 * scheduler plans are reported once to stderr and leave the field unchanged.
 *
 * @param integrator Configured ETDRK4 integrator.
 * @param field Complex target field updated in place.
 * @param dt Requested timestep, or nonpositive to use the stored suggestion.
 */
static void etdrk4_step(Integrator* integrator, Field* field, double dt) {
    ETDRK4State*    state;
    SimContext*     ctx;
    ETDRK4Snapshots snapshots = { 0 };
    size_t*         linear_indices    = NULL;
    size_t*         nonlinear_indices = NULL;
    size_t          linear_count      = 0U;
    size_t          nonlinear_count   = 0U;
    size_t          count;
    double          step;
    double          restore_dt;
    double          base_time;
    SimResult       result = SIM_RESULT_OK;

    if (integrator == NULL || field == NULL) {
        return;
    }

    state = (ETDRK4State*) integrator->userdata;
    ctx   = (state != NULL) ? state->context : NULL;
    if (ctx == NULL || !sim_field_is_complex(field)) {
        if (state != NULL && !state->warned_unsupported) {
            fprintf(stderr, "[WARN] etdrk4: requires a complex context-backed target field\n");
            state->warned_unsupported = true;
        }
        return;
    }

    integrator->is_complex = true;
    count                  = integrator_state_length(field);
    if (count == 0U) {
        return;
    }

    if (integrator_ensure_workspace(integrator, ETDRK4_WORKSPACE_BUFFERS, count) != SIM_RESULT_OK) {
        return;
    }

    step = (dt > 0.0) ? dt : integrator_next_step(integrator);
    step = integrator_clamp_dt(integrator, step);
    if (!(step > 0.0)) {
        return;
    }

    result = etdrk4_classify_plan(state,
                                  field,
                                  integrator->target_field_index,
                                  &linear_indices,
                                  &linear_count,
                                  &nonlinear_indices,
                                  &nonlinear_count);
    if (result != SIM_RESULT_OK) {
        if (!state->warned_unsupported) {
            fprintf(stderr, "[WARN] etdrk4: plan is not a supported single-field semilinear form\n");
            state->warned_unsupported = true;
        }
        goto cleanup;
    }

    result = etdrk4_capture_snapshots(ctx, &snapshots);
    if (result != SIM_RESULT_OK) {
        goto cleanup;
    }

    restore_dt = sim_context_timestep(ctx);
    base_time  = sim_context_time(ctx);
    sim_context_begin_drift(ctx);

    SimComplexDouble* u0     = integrator_buffer_complex(integrator, 0U);
    SimComplexDouble* e2u    = integrator_buffer_complex(integrator, 1U);
    SimComplexDouble* n0     = integrator_buffer_complex(integrator, 2U);
    SimComplexDouble* a      = integrator_buffer_complex(integrator, 3U);
    SimComplexDouble* na     = integrator_buffer_complex(integrator, 4U);
    SimComplexDouble* b      = integrator_buffer_complex(integrator, 5U);
    SimComplexDouble* nb     = integrator_buffer_complex(integrator, 6U);
    SimComplexDouble* c      = integrator_buffer_complex(integrator, 7U);
    SimComplexDouble* nc     = integrator_buffer_complex(integrator, 8U);
    SimComplexDouble* temp   = integrator_buffer_complex(integrator, 9U);
    SimComplexDouble* accum  = integrator_buffer_complex(integrator, 10U);
    SimComplexDouble* work   = integrator_buffer_complex(integrator, 11U);
    SimComplexDouble* final  = integrator_buffer_complex(integrator, 12U);
    SimComplexDouble* noise  = integrator_buffer_complex(integrator, 13U);

    if (u0 == NULL || e2u == NULL || n0 == NULL || a == NULL || na == NULL || b == NULL ||
        nb == NULL || c == NULL || nc == NULL || temp == NULL || accum == NULL ||
        work == NULL || final == NULL || noise == NULL) {
        result = SIM_RESULT_OUT_OF_MEMORY;
        goto cleanup_drift;
    }

    result = etdrk4_copy_field(field, u0, count);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    result = etdrk4_apply_linear_flow(
        ctx, field, &snapshots, linear_indices, linear_count, u0, e2u, count, 0.5 * step, restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    result = etdrk4_eval_nonlinear(
        ctx,
        field,
        &snapshots,
        nonlinear_indices,
        nonlinear_count,
        u0,
        n0,
        count,
        base_time,
        step,
        restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    result = etdrk4_apply_quadrature(ctx,
                                     field,
                                     &snapshots,
                                     linear_indices,
                                     linear_count,
                                     n0,
                                     temp,
                                     accum,
                                     count,
                                     step,
                                     0.5,
                                     etdrk4_weight_q,
                                     restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    etdrk4_linear_combine(a, e2u, 1.0, temp, 1.0, count);

    result = etdrk4_eval_nonlinear(
        ctx,
        field,
        &snapshots,
        nonlinear_indices,
        nonlinear_count,
        a,
        na,
        count,
        base_time + 0.5 * step,
        step,
        restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    result = etdrk4_apply_quadrature(ctx,
                                     field,
                                     &snapshots,
                                     linear_indices,
                                     linear_count,
                                     na,
                                     temp,
                                     accum,
                                     count,
                                     step,
                                     0.5,
                                     etdrk4_weight_q,
                                     restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    etdrk4_linear_combine(b, e2u, 1.0, temp, 1.0, count);

    result = etdrk4_eval_nonlinear(
        ctx,
        field,
        &snapshots,
        nonlinear_indices,
        nonlinear_count,
        b,
        nb,
        count,
        base_time + 0.5 * step,
        step,
        restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    etdrk4_linear_combine(temp, nb, 2.0, n0, -1.0, count);
    result = etdrk4_apply_linear_flow(
        ctx, field, &snapshots, linear_indices, linear_count, a, work, count, 0.5 * step, restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    result = etdrk4_apply_quadrature(ctx,
                                     field,
                                     &snapshots,
                                     linear_indices,
                                     linear_count,
                                     temp,
                                     accum,
                                     temp,
                                     count,
                                     step,
                                     0.5,
                                     etdrk4_weight_q,
                                     restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    etdrk4_linear_combine(c, work, 1.0, accum, 1.0, count);

    result = etdrk4_eval_nonlinear(
        ctx,
        field,
        &snapshots,
        nonlinear_indices,
        nonlinear_count,
        c,
        nc,
        count,
        base_time + step,
        step,
        restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    result = etdrk4_apply_linear_flow(
        ctx, field, &snapshots, linear_indices, linear_count, u0, final, count, step, restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }

    result = etdrk4_apply_quadrature(ctx,
                                     field,
                                     &snapshots,
                                     linear_indices,
                                     linear_count,
                                     n0,
                                     temp,
                                     accum,
                                     count,
                                     step,
                                     1.0,
                                     etdrk4_weight_b1,
                                     restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    etdrk4_axpy(final, 1.0, temp, count);

    etdrk4_linear_combine(work, na, 1.0, nb, 1.0, count);
    result = etdrk4_apply_quadrature(ctx,
                                     field,
                                     &snapshots,
                                     linear_indices,
                                     linear_count,
                                     work,
                                     temp,
                                     accum,
                                     count,
                                     step,
                                     1.0,
                                     etdrk4_weight_b2,
                                     restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    etdrk4_axpy(final, 1.0, temp, count);

    result = etdrk4_apply_quadrature(ctx,
                                     field,
                                     &snapshots,
                                     linear_indices,
                                     linear_count,
                                     nc,
                                     temp,
                                     accum,
                                     count,
                                     step,
                                     1.0,
                                     etdrk4_weight_b3,
                                     restore_dt);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    etdrk4_axpy(final, 1.0, temp, count);

    result = etdrk4_restore_snapshots(ctx, &snapshots);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    result = etdrk4_write_field(field, final, count);
    if (result != SIM_RESULT_OK) {
        goto cleanup_drift;
    }
    integrator_apply_stochastic(integrator, field, (double*) noise, count, step);

cleanup_drift:
    sim_context_set_timestep(ctx, restore_dt);
    sim_context_clear_drift_time_override(ctx);
    sim_context_end_drift(ctx);
    etdrk4_destroy_snapshots(&snapshots);
    if (result != SIM_RESULT_OK) {
        goto cleanup;
    }

    integrator->last_step  = step;
    integrator->last_error = 0.0;
    integrator->last_attempt_count   = 1U;
    integrator->last_rejection_count = 0U;
    integrator->current_dt = integrator_clamp_dt(integrator, step);

cleanup:
    free(linear_indices);
    free(nonlinear_indices);
}

/**
 * @brief Create an ETDRK4 integrator for a context-backed complex field.
 *
 * The configuration userdata must be a `SimContext*`. The factory installs a
 * context-drift adapter, owns a private state object through `userdata`, and
 * forces fixed-step execution by setting `adaptive = false`.
 *
 * @param config Integrator configuration containing context userdata.
 * @param[out] out Integrator storage populated on success.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_OUT_OF_MEMORY,
 * or an error from integrator_configure().
 */
SimResult integrator_etdrk4_create(const IntegratorConfig* config, Integrator* out) {
    ETDRK4State*    state;
    IntegratorConfig local;

    if (out == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    memset(&local, 0, sizeof(local));
    if (config != NULL) {
        local = *config;
    }

    state = (ETDRK4State*) calloc(1U, sizeof(ETDRK4State));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->context = (SimContext*) local.userdata;
    if (state->context == NULL) {
        free(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    local.drift    = etdrk4_context_drift;
    local.destroy  = etdrk4_destroy_state;
    local.userdata = state;
    local.adaptive = false;

    return integrator_configure(out, "etdrk4", etdrk4_step, &local);
}
