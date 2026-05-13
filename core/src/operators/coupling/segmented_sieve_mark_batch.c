#include "oakfield/operators/coupling/segmented_sieve_mark_batch.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_context.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEGMENTED_SIEVE_MARK_BATCH_SYMBOLIC_CAPACITY 160

typedef struct SimSegmentedSieveMarkBatchOperatorState {
    SimSegmentedSieveMarkBatchOperatorConfig config;
    uint64_t*                                prime_cache;
    uint64_t*                                prime_values;
    size_t                                   prime_cache_capacity;
    char                                     symbolic[SEGMENTED_SIEVE_MARK_BATCH_SYMBOLIC_CAPACITY];
} SimSegmentedSieveMarkBatchOperatorState;

static void
segmented_sieve_mark_batch_refresh_symbolic(SimSegmentedSieveMarkBatchOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "segmented_sieve_mark_batch candidate=%zu primes=%zu flags=%zu",
                    state->config.candidate_field,
                    state->config.primes_field,
                    state->config.flags_field);
#else
    (void) state;
#endif
}

static const char* segmented_sieve_mark_batch_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimSegmentedSieveMarkBatchOperatorState* state =
        (const SimSegmentedSieveMarkBatchOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void segmented_sieve_mark_batch_destroy(void* state_ptr) {
    SimSegmentedSieveMarkBatchOperatorState* state =
        (SimSegmentedSieveMarkBatchOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state->prime_cache);
    free(state->prime_values);
    free(state);
}

static bool segmented_sieve_mark_batch_flag_domain_supported(SimScalarDomain domain) {
    return sim_operator_domain_is_exact_integer(domain);
}

static bool segmented_sieve_mark_batch_read_positive_prime(const void*     data,
                                                           SimScalarDomain domain,
                                                           size_t          index,
                                                           uint64_t*       out_prime) {
    uint64_t raw = 0U;

    if (data == NULL || out_prime == NULL || !sim_operator_domain_is_exact_integer(domain) ||
        !sim_operator_integer_read(data, domain, index, &raw)) {
        return false;
    }

    if (domain.is_signed) {
        int64_t value = sim_operator_integer_as_i64(raw, domain);
        if (value < 2) {
            return false;
        }
        *out_prime = (uint64_t) value;
        return true;
    }

    raw = sim_operator_integer_truncate(raw, domain);
    if (raw < 2U) {
        return false;
    }
    *out_prime = raw;
    return true;
}

static bool segmented_sieve_mark_batch_prime_to_candidate_raw(uint64_t        prime_value,
                                                              SimScalarDomain candidate_domain,
                                                              uint64_t*       out_raw) {
    if (out_raw == NULL || !sim_operator_domain_is_exact_integer(candidate_domain) ||
        prime_value < 2U) {
        return false;
    }

    if (candidate_domain.is_signed) {
        if ((candidate_domain.bit_width == 32U && prime_value > (uint64_t) INT32_MAX) ||
            (candidate_domain.bit_width == 64U && prime_value > (uint64_t) INT64_MAX)) {
            return false;
        }
        return sim_operator_integer_raw_from_signed(
            (int64_t) prime_value, candidate_domain, out_raw);
    }

    return sim_operator_integer_raw_from_unsigned(prime_value, candidate_domain, out_raw);
}

static bool segmented_sieve_mark_batch_candidate_hits(uint64_t        candidate_raw,
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

static bool segmented_sieve_mark_batch_nonnegative_value(uint64_t        raw,
                                                         SimScalarDomain domain,
                                                         uint64_t*       out_value) {
    if (out_value == NULL || !sim_operator_domain_is_exact_integer(domain)) {
        return false;
    }

    if (domain.is_signed) {
        int64_t value = sim_operator_integer_as_i64(raw, domain);
        if (value < 0) {
            return false;
        }
        *out_value = (uint64_t) value;
        return true;
    }

    *out_value = sim_operator_integer_truncate(raw, domain);
    return true;
}

static bool segmented_sieve_mark_batch_detect_unit_range(const void*     candidate_data,
                                                         SimScalarDomain candidate_domain,
                                                         size_t          candidate_count,
                                                         uint64_t*       out_start,
                                                         uint64_t*       out_end) {
    uint64_t start = 0U;
    uint64_t prev  = 0U;

    if (candidate_data == NULL || out_start == NULL || out_end == NULL || candidate_count == 0U) {
        return false;
    }

    for (size_t i = 0U; i < candidate_count; ++i) {
        uint64_t raw   = 0U;
        uint64_t value = 0U;

        if (!sim_operator_integer_read(candidate_data, candidate_domain, i, &raw) ||
            !segmented_sieve_mark_batch_nonnegative_value(raw, candidate_domain, &value)) {
            return false;
        }

        if (i == 0U) {
            start = value;
            prev  = value;
            continue;
        }

        if (prev == UINT64_MAX || value != (prev + 1U)) {
            return false;
        }
        prev = value;
    }

    *out_start = start;
    *out_end   = prev;
    return true;
}

static bool segmented_sieve_mark_batch_first_hit(uint64_t  range_start,
                                                 uint64_t  range_end,
                                                 uint64_t  prime_value,
                                                 uint64_t* out_index) {
    uint64_t first_value = range_start;
    uint64_t remainder   = 0U;

    if (out_index == NULL || prime_value < 2U || range_start > range_end) {
        return false;
    }

    if (range_end <= prime_value) {
        return false;
    }

    if (first_value <= prime_value) {
        if (prime_value == UINT64_MAX) {
            return false;
        }
        first_value = prime_value + 1U;
    }

    remainder = first_value % prime_value;
    if (remainder != 0U) {
        uint64_t delta = prime_value - remainder;
        if (UINT64_MAX - first_value < delta) {
            return false;
        }
        first_value += delta;
    }

    if (first_value > range_end || first_value < range_start) {
        return false;
    }

    *out_index = first_value - range_start;
    return true;
}

static SimResult
segmented_sieve_mark_batch_apply_unit_range_integer_flags(void*           flags_data,
                                                          SimScalarDomain flags_domain,
                                                          uint64_t        range_start,
                                                          uint64_t        range_end,
                                                          size_t          candidate_count,
                                                          const uint64_t* prime_values,
                                                          size_t          prime_count) {
    uint64_t count_u64 = (uint64_t) candidate_count;

    if (flags_data == NULL || prime_values == NULL ||
        !sim_operator_domain_is_exact_integer(flags_domain)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t prime_index = 0U; prime_index < prime_count; ++prime_index) {
        uint64_t start_index = 0U;
        uint64_t prime_value = prime_values[prime_index];

        if (!segmented_sieve_mark_batch_first_hit(
                range_start, range_end, prime_value, &start_index)) {
            continue;
        }

        if (sim_scalar_domain_equal(flags_domain, sim_scalar_domain_i8())) {
            int8_t* typed = (int8_t*) flags_data;
            for (uint64_t index = start_index; index < count_u64; index += prime_value) {
                typed[index] = 0;
            }
            continue;
        }
        if (sim_scalar_domain_equal(flags_domain, sim_scalar_domain_u8())) {
            uint8_t* typed = (uint8_t*) flags_data;
            for (uint64_t index = start_index; index < count_u64; index += prime_value) {
                typed[index] = 0U;
            }
            continue;
        }
        if (sim_scalar_domain_equal(flags_domain, sim_scalar_domain_i32())) {
            int32_t* typed = (int32_t*) flags_data;
            for (uint64_t index = start_index; index < count_u64; index += prime_value) {
                typed[index] = 0;
            }
            continue;
        }
        if (sim_scalar_domain_equal(flags_domain, sim_scalar_domain_u32())) {
            uint32_t* typed = (uint32_t*) flags_data;
            for (uint64_t index = start_index; index < count_u64; index += prime_value) {
                typed[index] = 0U;
            }
            continue;
        }
        if (sim_scalar_domain_equal(flags_domain, sim_scalar_domain_i64())) {
            int64_t* typed = (int64_t*) flags_data;
            for (uint64_t index = start_index; index < count_u64; index += prime_value) {
                typed[index] = 0;
            }
            continue;
        }
        if (sim_scalar_domain_equal(flags_domain, sim_scalar_domain_u64())) {
            uint64_t* typed = (uint64_t*) flags_data;
            for (uint64_t index = start_index; index < count_u64; index += prime_value) {
                typed[index] = 0U;
            }
            continue;
        }
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    return SIM_RESULT_OK;
}

static SimResult
segmented_sieve_mark_batch_validate_fields(const SimField* candidate_field,
                                           const SimField* primes_field,
                                           const SimField* flags_field,
                                           const SimSegmentedSieveMarkBatchOperatorConfig* config) {
    SimScalarDomain candidate_domain;
    SimScalarDomain primes_domain;
    SimScalarDomain flags_domain;

    if (candidate_field == NULL || primes_field == NULL || flags_field == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (sim_field_is_complex(candidate_field) || sim_field_is_complex(primes_field) ||
        sim_field_is_complex(flags_field)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }
    if (sim_field_element_count(&candidate_field->layout) !=
        sim_field_element_count(&flags_field->layout)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    candidate_domain = sim_field_scalar_domain(candidate_field);
    primes_domain    = sim_field_scalar_domain(primes_field);
    flags_domain     = sim_field_scalar_domain(flags_field);
    if (!sim_operator_domain_is_exact_integer(candidate_domain) ||
        !sim_operator_domain_is_exact_integer(primes_domain) ||
        !segmented_sieve_mark_batch_flag_domain_supported(flags_domain)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    return SIM_RESULT_OK;
}

static SimResult
segmented_sieve_mark_batch_ensure_prime_cache(SimSegmentedSieveMarkBatchOperatorState* state,
                                              const SimField* candidate_field,
                                              const SimField* primes_field,
                                              size_t*         out_prime_count) {
    SimScalarDomain candidate_domain;
    SimScalarDomain primes_domain;
    const void*     primes_data;
    size_t          prime_count;

    if (state == NULL || candidate_field == NULL || primes_field == NULL ||
        out_prime_count == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    candidate_domain = sim_field_scalar_domain(candidate_field);
    primes_domain    = sim_field_scalar_domain(primes_field);
    primes_data      = sim_field_data_const(primes_field);
    prime_count      = sim_field_element_count(&primes_field->layout);

    *out_prime_count = 0U;
    if (prime_count == 0U) {
        return SIM_RESULT_OK;
    }
    if (primes_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (state->prime_cache_capacity < prime_count) {
        uint64_t* new_cache =
            (uint64_t*) realloc(state->prime_cache, prime_count * sizeof(uint64_t));
        uint64_t* new_values = NULL;
        if (new_cache == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        new_values = (uint64_t*) realloc(state->prime_values, prime_count * sizeof(uint64_t));
        if (new_values == NULL) {
            state->prime_cache = new_cache;
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->prime_cache          = new_cache;
        state->prime_values         = new_values;
        state->prime_cache_capacity = prime_count;
    }

    for (size_t i = 0U; i < prime_count; ++i) {
        uint64_t prime_value = 0U;
        if (!segmented_sieve_mark_batch_read_positive_prime(
                primes_data, primes_domain, i, &prime_value) ||
            !segmented_sieve_mark_batch_prime_to_candidate_raw(
                prime_value, candidate_domain, &state->prime_cache[i])) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        state->prime_values[i] = prime_value;
    }

    *out_prime_count = prime_count;
    return SIM_RESULT_OK;
}

static SimResult segmented_sieve_mark_batch_apply(void*               state_ptr,
                                                  struct SimContext*  context,
                                                  struct SimOperator* self,
                                                  double              dt) {
    SimSegmentedSieveMarkBatchOperatorState* state =
        (SimSegmentedSieveMarkBatchOperatorState*) state_ptr;
    SimField*       candidate_field;
    SimField*       primes_field;
    SimField*       flags_field;
    SimScalarDomain candidate_domain;
    SimScalarDomain flags_domain;
    const void*     candidate_data;
    size_t          candidate_count;
    size_t          prime_count = 0U;
    uint64_t        range_start = 0U;
    uint64_t        range_end   = 0U;
    SimResult       result;

    (void) self;
    (void) dt;

    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    candidate_field = sim_context_field(context, state->config.candidate_field);
    primes_field    = sim_context_field(context, state->config.primes_field);
    flags_field     = sim_context_field(context, state->config.flags_field);
    result          = segmented_sieve_mark_batch_validate_fields(
        candidate_field, primes_field, flags_field, &state->config);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    result = segmented_sieve_mark_batch_ensure_prime_cache(
        state, candidate_field, primes_field, &prime_count);
    if (result != SIM_RESULT_OK || prime_count == 0U) {
        return result;
    }

    candidate_domain = sim_field_scalar_domain(candidate_field);
    flags_domain     = sim_field_scalar_domain(flags_field);
    candidate_data   = sim_field_data_const(candidate_field);
    candidate_count  = sim_field_element_count(&candidate_field->layout);
    if (candidate_data == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (candidate_count == 0U) {
        return SIM_RESULT_OK;
    }

    // Fast path for segmented ranges generated by exact integer coordinate ramps.
    if (segmented_sieve_mark_batch_detect_unit_range(
            candidate_data, candidate_domain, candidate_count, &range_start, &range_end)) {
        if (sim_operator_domain_is_exact_integer(flags_domain)) {
            return segmented_sieve_mark_batch_apply_unit_range_integer_flags(
                sim_field_data(flags_field),
                flags_domain,
                range_start,
                range_end,
                candidate_count,
                state->prime_values,
                prime_count);
        }
    }

    if (sim_operator_domain_is_exact_integer(flags_domain)) {
        void* flags_data = sim_field_data(flags_field);
        if (flags_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < candidate_count; ++i) {
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

            for (size_t prime_index = 0U; prime_index < prime_count; ++prime_index) {
                if (segmented_sieve_mark_batch_candidate_hits(
                        candidate_raw, state->prime_cache[prime_index], candidate_domain)) {
                    if (!sim_operator_integer_write(flags_data, flags_domain, i, 0U)) {
                        return SIM_RESULT_INVALID_ARGUMENT;
                    }
                    break;
                }
            }
        }
    } else {
        double* flags_data = sim_field_real_data(flags_field);
        if (flags_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < candidate_count; ++i) {
            uint64_t candidate_raw = 0U;

            if (flags_data[i] == 0.0) {
                continue;
            }
            if (!sim_operator_integer_read(candidate_data, candidate_domain, i, &candidate_raw)) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            for (size_t prime_index = 0U; prime_index < prime_count; ++prime_index) {
                if (segmented_sieve_mark_batch_candidate_hits(
                        candidate_raw, state->prime_cache[prime_index], candidate_domain)) {
                    flags_data[i] = 0.0;
                    break;
                }
            }
        }
    }

    return SIM_RESULT_OK;
}

static SimResult segmented_sieve_mark_batch_step(void*               state_ptr,
                                                 struct SimContext*  context,
                                                 struct SimOperator* self,
                                                 size_t              substep_index,
                                                 double              dt_sub,
                                                 void*               scratch,
                                                 size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return segmented_sieve_mark_batch_apply(state_ptr, context, self, dt_sub);
}

SimResult
sim_add_segmented_sieve_mark_batch_operator(struct SimContext*                              context,
                                            const SimSegmentedSieveMarkBatchOperatorConfig* config,
                                            size_t* out_index) {
    SimSegmentedSieveMarkBatchOperatorConfig local = { 0 };
    SimSegmentedSieveMarkBatchOperatorState* state = NULL;
    SimField*                                candidate_field;
    SimField*                                primes_field;
    SimField*                                flags_field;
    char                                     name[SIM_OPERATOR_NAME_MAX + 1U];
    SimOperatorInfo                          info;
    SimSplitPort                             ports[3];
    SimSplitAccess                           accesses[3];
    SimSplitSubstep                          substep;
    SimSplitDescriptor                       desc = { 0 };
    SimResult                                result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (config != NULL) {
        local = *config;
    }

    candidate_field = sim_context_field(context, local.candidate_field);
    primes_field    = sim_context_field(context, local.primes_field);
    flags_field     = sim_context_field(context, local.flags_field);
    result          = segmented_sieve_mark_batch_validate_fields(
        candidate_field, primes_field, flags_field, &local);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    state = (SimSegmentedSieveMarkBatchOperatorState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    state->config = local;
    segmented_sieve_mark_batch_refresh_symbolic(state);

    sim_operator_make_unique_name(name, sizeof(name), "segmented_sieve_mark_batch");

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
    info.abstract_id       = "segmented_sieve_mark_batch";
    sim_operator_info_set_schema_identity(&info, "segmented_sieve_mark_batch");
    info.algebraic_flags                                = SIM_OPERATOR_ALG_NONE;
    info.representation.domain                          = SIM_FIELD_DOMAIN_PHYSICAL;
    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = true;

    ports[0].context_field_index = state->config.candidate_field;
    ports[0].require_complex     = false;
    ports[1].context_field_index = state->config.primes_field;
    ports[1].require_complex     = false;
    ports[2].context_field_index = state->config.flags_field;
    ports[2].require_complex     = false;

    accesses[0].port = 0U;
    accesses[0].mode = SIM_ACCESS_READ;
    accesses[1].port = 1U;
    accesses[1].mode = SIM_ACCESS_READ;
    accesses[2].port = 2U;
    accesses[2].mode = SIM_ACCESS_RW;

    substep.name              = NULL;
    substep.fn                = segmented_sieve_mark_batch_step;
    substep.accesses          = accesses;
    substep.access_count      = 3U;
    substep.dt_scale          = 1.0;
    substep.barrier_after     = false;
    substep.error_measure     = NULL;
    substep.required_features = 0U;

    desc.name                     = name;
    desc.ports                    = ports;
    desc.port_count               = 3U;
    desc.substeps                 = &substep;
    desc.substep_count            = 1U;
    desc.state                    = state;
    desc.symbolic                 = segmented_sieve_mark_batch_symbolic;
    desc.destroy                  = segmented_sieve_mark_batch_destroy;
    desc.info                     = info;
    desc.scratch.bytes_per_worker = 0U;
    desc.scratch.alignment        = 0U;

    result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        segmented_sieve_mark_batch_destroy(state);
    }

    return result;
}

SimResult
sim_segmented_sieve_mark_batch_config(struct SimContext*                        context,
                                      size_t                                    operator_index,
                                      SimSegmentedSieveMarkBatchOperatorConfig* out_config) {
    SimOperator*                             op;
    SimSegmentedSieveMarkBatchOperatorState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimSegmentedSieveMarkBatchOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult
sim_segmented_sieve_mark_batch_update(struct SimContext* context,
                                      size_t             operator_index,
                                      const SimSegmentedSieveMarkBatchOperatorConfig* config) {
    SimOperator*                             op;
    SimSegmentedSieveMarkBatchOperatorState* state;
    SimSegmentedSieveMarkBatchOperatorConfig local;
    SimField*                                candidate_field;
    SimField*                                primes_field;
    SimField*                                flags_field;
    SimResult                                result;

    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimSegmentedSieveMarkBatchOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local = state->config;
    if (config != NULL) {
        local = *config;
    }

    candidate_field = sim_context_field(context, local.candidate_field);
    primes_field    = sim_context_field(context, local.primes_field);
    flags_field     = sim_context_field(context, local.flags_field);
    result          = segmented_sieve_mark_batch_validate_fields(
        candidate_field, primes_field, flags_field, &local);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    state->config = local;
    segmented_sieve_mark_batch_refresh_symbolic(state);
    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
