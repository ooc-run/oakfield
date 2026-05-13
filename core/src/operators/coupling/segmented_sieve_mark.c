#include "oakfield/operators/coupling/segmented_sieve_mark.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_context.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEGMENTED_SIEVE_MARK_SYMBOLIC_CAPACITY 128

typedef struct SimSegmentedSieveMarkOperatorState {
    SimSegmentedSieveMarkOperatorConfig config;
    char                                symbolic[SEGMENTED_SIEVE_MARK_SYMBOLIC_CAPACITY];
} SimSegmentedSieveMarkOperatorState;

static void segmented_sieve_mark_refresh_symbolic(SimSegmentedSieveMarkOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "segmented_sieve_mark prime=%" PRIu64,
                    state->config.prime);
#else
    (void) state;
#endif
}

static const char* segmented_sieve_mark_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimSegmentedSieveMarkOperatorState* state =
        (const SimSegmentedSieveMarkOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static bool segmented_sieve_mark_prime_to_domain_raw(uint64_t        prime_value,
                                                     SimScalarDomain domain,
                                                     uint64_t*       out_raw) {
    if (!sim_operator_domain_is_exact_integer(domain) || prime_value < 2U || out_raw == NULL) {
        return false;
    }

    if (domain.is_signed) {
        if ((domain.bit_width == 32U && prime_value > (uint64_t) INT32_MAX) ||
            (domain.bit_width == 64U && prime_value > (uint64_t) INT64_MAX)) {
            return false;
        }
        return sim_operator_integer_raw_from_signed((int64_t) prime_value, domain, out_raw);
    }

    return sim_operator_integer_raw_from_unsigned(prime_value, domain, out_raw);
}

static bool segmented_sieve_mark_flag_domain_supported(SimScalarDomain domain) {
    return sim_operator_domain_is_exact_integer(domain) ||
           sim_scalar_domain_equal(domain, sim_scalar_domain_f64());
}

static SimResult
segmented_sieve_mark_validate_fields(const SimField*                            candidate_field,
                                     const SimField*                            flags_field,
                                     const SimSegmentedSieveMarkOperatorConfig* config) {
    SimScalarDomain candidate_domain;
    SimScalarDomain flags_domain;
    uint64_t        prime_raw = 0U;

    if (candidate_field == NULL || flags_field == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (sim_field_element_count(&candidate_field->layout) !=
        sim_field_element_count(&flags_field->layout)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (sim_field_is_complex(candidate_field) || sim_field_is_complex(flags_field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    candidate_domain = sim_field_scalar_domain(candidate_field);
    flags_domain     = sim_field_scalar_domain(flags_field);
    if (!sim_operator_domain_is_exact_integer(candidate_domain) ||
        !segmented_sieve_mark_flag_domain_supported(flags_domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (!segmented_sieve_mark_prime_to_domain_raw(config->prime, candidate_domain, &prime_raw)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    return SIM_RESULT_OK;
}

static bool segmented_sieve_mark_candidate_hits(uint64_t        candidate_raw,
                                                uint64_t        prime_raw,
                                                SimScalarDomain domain) {
    if (!sim_operator_domain_is_exact_integer(domain)) {
        return false;
    }

    if (sim_operator_integer_compare(candidate_raw, prime_raw, domain) <= 0) {
        return false;
    }

    if (domain.is_signed) {
        int64_t candidate = sim_operator_integer_as_i64(candidate_raw, domain);
        int64_t prime     = sim_operator_integer_as_i64(prime_raw, domain);
        if (prime <= 1) {
            return false;
        }
        return (candidate % prime) == 0;
    }

    candidate_raw = sim_operator_integer_truncate(candidate_raw, domain);
    prime_raw     = sim_operator_integer_truncate(prime_raw, domain);
    if (prime_raw <= 1U) {
        return false;
    }
    return (candidate_raw % prime_raw) == 0U;
}

static SimResult segmented_sieve_mark_apply(void*               state_ptr,
                                            struct SimContext*  context,
                                            struct SimOperator* self,
                                            double              dt) {
    SimSegmentedSieveMarkOperatorState* state = (SimSegmentedSieveMarkOperatorState*) state_ptr;
    SimField*                           candidate_field;
    SimField*                           flags_field;
    SimScalarDomain                     candidate_domain;
    SimScalarDomain                     flags_domain;
    const void*                         candidate_data;
    uint64_t                            prime_raw = 0U;
    size_t                              count;

    (void) self;
    (void) dt;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    candidate_field = sim_context_field(context, state->config.candidate_field);
    flags_field     = sim_context_field(context, state->config.flags_field);
    {
        SimResult validate =
            segmented_sieve_mark_validate_fields(candidate_field, flags_field, &state->config);
        if (validate != SIM_RESULT_OK) {
            return validate;
        }
    }

    candidate_domain = sim_field_scalar_domain(candidate_field);
    flags_domain     = sim_field_scalar_domain(flags_field);
    if (!segmented_sieve_mark_prime_to_domain_raw(
            state->config.prime, candidate_domain, &prime_raw)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    candidate_data = sim_field_data_const(candidate_field);
    if (candidate_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    count = sim_field_element_count(&candidate_field->layout);
    if (sim_operator_domain_is_exact_integer(flags_domain)) {
        void* flags_data = sim_field_data(flags_field);
        if (flags_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            uint64_t flag_raw      = 0U;
            uint64_t candidate_raw = 0U;

            if (!sim_operator_integer_read(flags_data, flags_domain, i, &flag_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (flag_raw == 0U) {
                continue;
            }
            if (!sim_operator_integer_read(candidate_data, candidate_domain, i, &candidate_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (!segmented_sieve_mark_candidate_hits(candidate_raw, prime_raw, candidate_domain)) {
                continue;
            }
            if (!sim_operator_integer_write(flags_data, flags_domain, i, 0U)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
        }
    } else {
        double* flags_data = sim_field_real_data(flags_field);
        if (flags_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            uint64_t candidate_raw = 0U;

            if (flags_data[i] == 0.0) {
                continue;
            }
            if (!sim_operator_integer_read(candidate_data, candidate_domain, i, &candidate_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (segmented_sieve_mark_candidate_hits(candidate_raw, prime_raw, candidate_domain)) {
                flags_data[i] = 0.0;
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult segmented_sieve_mark_step(void*               state_ptr,
                                           struct SimContext*  context,
                                           struct SimOperator* self,
                                           size_t              substep_index,
                                           double              dt_sub,
                                           void*               scratch,
                                           size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return segmented_sieve_mark_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_segmented_sieve_mark_operator(struct SimContext*                         context,
                                                const SimSegmentedSieveMarkOperatorConfig* config,
                                                size_t* out_index) {
    SimSegmentedSieveMarkOperatorConfig local = { 0 };
    SimSegmentedSieveMarkOperatorState* state = NULL;
    SimField*                           candidate_field;
    SimField*                           flags_field;
    char                                name[SIM_OPERATOR_NAME_MAX + 1U];
    SimOperatorInfo                     info;
    SimSplitPort                        ports[2];
    SimSplitAccess                      accesses[2];
    SimSplitSubstep                     substep;
    SimSplitDescriptor                  desc = { 0 };
    SimResult                           result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    local.prime = 2U;
    if (config != NULL) {
        local = *config;
    }

    candidate_field = sim_context_field(context, local.candidate_field);
    flags_field     = sim_context_field(context, local.flags_field);
    result          = segmented_sieve_mark_validate_fields(candidate_field, flags_field, &local);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    state = (SimSegmentedSieveMarkOperatorState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    state->config = local;
    segmented_sieve_mark_refresh_symbolic(state);

    sim_operator_make_unique_name(name, sizeof(name), "segmented_sieve_mark");

    info                   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_COUPLING;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = false;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "segmented_sieve_mark";
    sim_operator_info_set_schema_identity(&info, "segmented_sieve_mark");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_NONE;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = true;

    ports[0].context_field_index = state->config.candidate_field;
    ports[0].require_complex     = false;
    ports[1].context_field_index = state->config.flags_field;
    ports[1].require_complex     = false;

    accesses[0].port = 0;
    accesses[0].mode = SIM_ACCESS_READ;
    accesses[1].port = 1;
    accesses[1].mode = SIM_ACCESS_RW;

    substep.name              = NULL;
    substep.fn                = segmented_sieve_mark_step;
    substep.accesses          = accesses;
    substep.access_count      = 2U;
    substep.dt_scale          = 1.0;
    substep.barrier_after     = false;
    substep.error_measure     = NULL;
    substep.required_features = 0U;

    desc.name                     = name;
    desc.ports                    = ports;
    desc.port_count               = 2U;
    desc.substeps                 = &substep;
    desc.substep_count            = 1U;
    desc.state                    = state;
    desc.symbolic                 = segmented_sieve_mark_symbolic;
    desc.destroy                  = free;
    desc.info                     = info;
    desc.scratch.bytes_per_worker = 0U;
    desc.scratch.alignment        = 0U;

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        free(state);
    }

    return result;
}

SimResult sim_segmented_sieve_mark_config(struct SimContext*                   context,
                                          size_t                               operator_index,
                                          SimSegmentedSieveMarkOperatorConfig* out_config) {
    SimOperator*                        op;
    SimSegmentedSieveMarkOperatorState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimSegmentedSieveMarkOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_segmented_sieve_mark_update(struct SimContext*                         context,
                                          size_t                                     operator_index,
                                          const SimSegmentedSieveMarkOperatorConfig* config) {
    SimOperator*                        op;
    SimSegmentedSieveMarkOperatorState* state;
    SimSegmentedSieveMarkOperatorConfig local;
    SimField*                           candidate_field;
    SimField*                           flags_field;
    SimResult                           result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimSegmentedSieveMarkOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local = state->config;
    if (config != NULL) {
        local = *config;
    }

    candidate_field = sim_context_field(context, local.candidate_field);
    flags_field     = sim_context_field(context, local.flags_field);
    result          = segmented_sieve_mark_validate_fields(candidate_field, flags_field, &local);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    state->config = local;
    segmented_sieve_mark_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
