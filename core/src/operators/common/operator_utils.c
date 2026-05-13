#include "operators/common/operator_utils.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static uint64_t sim_operator_integer_mask(SimScalarDomain domain) {
    if (domain.bit_width == 0U) {
        return 0U;
    }
    if (domain.bit_width >= 64U) {
        return UINT64_MAX;
    }
    return (UINT64_C(1) << domain.bit_width) - UINT64_C(1);
}

void sim_operator_make_unique_name(char* buffer, size_t capacity, const char* prefix) {
    static unsigned long long counter = 0ULL;
    unsigned long long        id;

    if (buffer == NULL || capacity == 0U || prefix == NULL) {
        return;
    }

    id = ++counter;
    (void) snprintf(buffer, capacity, "%s_%llu", prefix, (unsigned long long) id);
}

bool sim_operator_resolve_scale_by_dt(const struct SimContext* context,
                                      const char*              op_name,
                                      bool                     has_user_value,
                                      bool                     requested) {
    bool resolved = has_user_value ? requested : true; /* default to dt-scaled */
    return resolved;
}

SimClockMode sim_operator_choose_time_mode(const struct SimContext* context,
                                           const SimOperatorConfig* op_config,
                                           SimClockMode             requested,
                                           double                   nominal_dt,
                                           double                   epsilon,
                                           bool*                    out_forced_pure) {
    bool                  forced = false;
    SimRepresentationMode mode   = SIM_REPRESENTATION_MODE_RELAXED;

    if (context != NULL) {
        mode = sim_context_representation_mode(context);
    }
    if (op_config != NULL && op_config->representation_mode_override_enabled) {
        mode = op_config->representation_mode_override;
    }

    SimClockMode resolved = requested;

    if (mode == SIM_REPRESENTATION_MODE_STRICT) {
        if (requested == SIM_CLOCK_ACCUMULATED_STATEFUL) {
            if (isfinite(nominal_dt) && nominal_dt > epsilon) {
                resolved = SIM_CLOCK_FROM_STEP_PURE;
            } else {
                resolved = SIM_CLOCK_FROM_TIME_PARAM;
            }
            forced = true;
        } else if (requested == SIM_CLOCK_FROM_STEP_PURE) {
            if (!(nominal_dt > epsilon) || !isfinite(nominal_dt)) {
                resolved = SIM_CLOCK_FROM_TIME_PARAM;
                forced   = true;
            }
        }
    }

    if (out_forced_pure != NULL) {
        *out_forced_pure = forced;
    }

    return resolved;
}

bool sim_operator_should_register_kernel(const struct SimContext* context,
                                         const SimOperatorConfig* op_config,
                                         uint64_t                 required_features,
                                         SimDeterminismFlags      determinism_flags) {
    if (context == NULL) {
        return false;
    }

    SimRepresentationMode mode = sim_context_representation_mode(context);
    if (op_config != NULL && op_config->representation_mode_override_enabled) {
        mode = op_config->representation_mode_override;
    }

    return sim_context_kernel_allowed_mode(context, mode, required_features, determinism_flags);
}

static bool sim_operator_bool_env_enabled(const char* value) {
    if (value == NULL || value[0] == '\0') {
        return false;
    }
    return strcasecmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0;
}

static bool sim_operator_schema_split_only_default(const char* schema_key) {
    static const char* const split_only_schemas[] = { "mixer",
                                                      "analytic_warp",
                                                      "linear_spectral_fusion",
                                                      "linear_dissipative",
                                                      "dispersion",
                                                      "phase_rotate",
                                                      "phase_feature",
                                                      "remainder",
                                                      "thermostat",
                                                      "spatial_derivative",
                                                      "fractional_memory",
                                                      "minimal_convolution",
                                                      "stochastic_noise",
                                                      "stimulus_sine",
                                                      "stimulus_standing",
                                                      "stimulus_chirp",
                                                      "stimulus_spectral_lines",
                                                      "stimulus_checkerboard",
                                                      "stimulus_digamma_square",
                                                      "stimulus_fbm",
                                                      "stimulus_fourier",
                                                      "stimulus_gabor",
                                                      "stimulus_gaussian",
                                                      "stimulus_random_fourier",
                                                      "stimulus_white_noise" };

    if (schema_key == NULL || schema_key[0] == '\0') {
        return false;
    }

    for (size_t i = 0U; i < sizeof(split_only_schemas) / sizeof(split_only_schemas[0]); ++i) {
        if (strcmp(schema_key, split_only_schemas[i]) == 0) {
            return true;
        }
    }
    return false;
}

bool sim_operator_should_register_kernel_for_schema(const struct SimContext* context,
                                                    const SimOperatorConfig* op_config,
                                                    uint64_t                 required_features,
                                                    SimDeterminismFlags      determinism_flags,
                                                    const char*              schema_key) {
    const char* override_env;

    if (!sim_operator_should_register_kernel(context, op_config, required_features, determinism_flags)) {
        return false;
    }

    if (!sim_operator_schema_split_only_default(schema_key)) {
        return true;
    }

    override_env = getenv("OAKFIELD_ENABLE_EXPERIMENTAL_LEGACY_KERNELS");
    if (sim_operator_bool_env_enabled(override_env)) {
        return true;
    }

    return false;
}

bool sim_operator_field_domain_is_f64(const SimField* field) {
    if (field == NULL) {
        return false;
    }

    return sim_scalar_domain_equal(sim_field_scalar_domain(field), sim_scalar_domain_f64());
}

bool sim_operator_field_domain_is_f64_or_c64(const SimField* field) {
    SimScalarDomain domain;

    if (field == NULL) {
        return false;
    }

    domain = sim_field_scalar_domain(field);
    return sim_scalar_domain_equal(domain, sim_scalar_domain_f64()) ||
           sim_scalar_domain_equal(domain, sim_scalar_domain_c64());
}

bool sim_operator_domain_is_exact_integer(SimScalarDomain domain) {
    return sim_scalar_domain_is_integer(domain) &&
           (domain.bit_width == 8U || domain.bit_width == 32U || domain.bit_width == 64U);
}

uint64_t sim_operator_integer_truncate(uint64_t raw, SimScalarDomain domain) {
    if (!sim_operator_domain_is_exact_integer(domain)) {
        return 0U;
    }
    if (domain.bit_width >= 64U) {
        return raw;
    }
    return raw & sim_operator_integer_mask(domain);
}

int64_t sim_operator_integer_as_i64(uint64_t raw, SimScalarDomain domain) {
    raw = sim_operator_integer_truncate(raw, domain);
    if (!sim_operator_domain_is_exact_integer(domain)) {
        return 0;
    }
    if (!domain.is_signed || domain.bit_width >= 64U) {
        return (int64_t) raw;
    }

    {
        uint64_t mask     = sim_operator_integer_mask(domain);
        uint64_t sign_bit = UINT64_C(1) << (domain.bit_width - 1U);
        if ((raw & sign_bit) != 0U) {
            raw |= ~mask;
        }
    }
    return (int64_t) raw;
}

bool sim_operator_integer_raw_from_signed(int64_t value, SimScalarDomain domain, uint64_t* out_raw) {
    if (!sim_operator_domain_is_exact_integer(domain) || !domain.is_signed || out_raw == NULL) {
        return false;
    }
    if (domain.bit_width == 8U && (value < INT8_MIN || value > INT8_MAX)) {
        return false;
    }
    if (domain.bit_width == 32U && (value < INT32_MIN || value > INT32_MAX)) {
        return false;
    }
    *out_raw = sim_operator_integer_truncate((uint64_t) value, domain);
    return true;
}

bool sim_operator_integer_raw_from_unsigned(uint64_t value,
                                            SimScalarDomain domain,
                                            uint64_t*       out_raw) {
    if (!sim_operator_domain_is_exact_integer(domain) || domain.is_signed || out_raw == NULL) {
        return false;
    }
    if (domain.bit_width == 8U && value > UINT8_MAX) {
        return false;
    }
    if (domain.bit_width == 32U && value > UINT32_MAX) {
        return false;
    }
    *out_raw = sim_operator_integer_truncate(value, domain);
    return true;
}

bool sim_operator_integer_raw_from_double(double value, SimScalarDomain domain, uint64_t* out_raw) {
    static const double kSafeExactIntegerDouble = 9007199254740992.0; /* 2^53 */

    if (!sim_operator_domain_is_exact_integer(domain) || out_raw == NULL || !isfinite(value) ||
        trunc(value) != value || fabs(value) > kSafeExactIntegerDouble) {
        return false;
    }

    if (domain.is_signed) {
        if (domain.bit_width == 8U) {
            if (value < (double) INT8_MIN || value > (double) INT8_MAX) {
                return false;
            }
        } else if (domain.bit_width == 32U) {
            if (value < (double) INT32_MIN || value > (double) INT32_MAX) {
                return false;
            }
        } else if (value < (double) INT64_MIN || value > (double) INT64_MAX) {
            return false;
        }
        return sim_operator_integer_raw_from_signed((int64_t) value, domain, out_raw);
    }

    if (value < 0.0) {
        return false;
    }
    if (domain.bit_width == 8U && value > (double) UINT8_MAX) {
        return false;
    }
    if (domain.bit_width == 32U && value > (double) UINT32_MAX) {
        return false;
    }
    return sim_operator_integer_raw_from_unsigned((uint64_t) value, domain, out_raw);
}

int sim_operator_integer_compare(uint64_t lhs_raw, uint64_t rhs_raw, SimScalarDomain domain) {
    if (!sim_operator_domain_is_exact_integer(domain)) {
        return 0;
    }
    lhs_raw = sim_operator_integer_truncate(lhs_raw, domain);
    rhs_raw = sim_operator_integer_truncate(rhs_raw, domain);
    if (domain.is_signed) {
        int64_t lhs = sim_operator_integer_as_i64(lhs_raw, domain);
        int64_t rhs = sim_operator_integer_as_i64(rhs_raw, domain);
        if (lhs < rhs) {
            return -1;
        }
        if (lhs > rhs) {
            return 1;
        }
        return 0;
    }
    if (lhs_raw < rhs_raw) {
        return -1;
    }
    if (lhs_raw > rhs_raw) {
        return 1;
    }
    return 0;
}

bool sim_operator_integer_read(const void* data,
                               SimScalarDomain domain,
                               size_t          index,
                               uint64_t*       out_raw) {
    if (data == NULL || out_raw == NULL || !sim_operator_domain_is_exact_integer(domain)) {
        return false;
    }

    if (sim_scalar_domain_equal(domain, sim_scalar_domain_i8())) {
        *out_raw = sim_operator_integer_truncate((uint64_t) ((const int8_t*) data)[index], domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_i32())) {
        *out_raw = sim_operator_integer_truncate((uint64_t) ((const int32_t*) data)[index], domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_i64())) {
        *out_raw = sim_operator_integer_truncate((uint64_t) ((const int64_t*) data)[index], domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_u8())) {
        *out_raw = sim_operator_integer_truncate((uint64_t) ((const uint8_t*) data)[index], domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_u32())) {
        *out_raw = sim_operator_integer_truncate((uint64_t) ((const uint32_t*) data)[index], domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_u64())) {
        *out_raw = sim_operator_integer_truncate(((const uint64_t*) data)[index], domain);
        return true;
    }
    return false;
}

bool sim_operator_integer_write(void* data, SimScalarDomain domain, size_t index, uint64_t raw) {
    raw = sim_operator_integer_truncate(raw, domain);
    if (data == NULL || !sim_operator_domain_is_exact_integer(domain)) {
        return false;
    }

    if (sim_scalar_domain_equal(domain, sim_scalar_domain_i8())) {
        ((int8_t*) data)[index] = (int8_t) sim_operator_integer_as_i64(raw, domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_i32())) {
        ((int32_t*) data)[index] = (int32_t) sim_operator_integer_as_i64(raw, domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_i64())) {
        ((int64_t*) data)[index] = sim_operator_integer_as_i64(raw, domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_u8())) {
        ((uint8_t*) data)[index] = (uint8_t) sim_operator_integer_truncate(raw, domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_u32())) {
        ((uint32_t*) data)[index] = (uint32_t) sim_operator_integer_truncate(raw, domain);
        return true;
    }
    if (sim_scalar_domain_equal(domain, sim_scalar_domain_u64())) {
        ((uint64_t*) data)[index] = sim_operator_integer_truncate(raw, domain);
        return true;
    }
    return false;
}
