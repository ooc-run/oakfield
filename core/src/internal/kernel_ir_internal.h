/**
 * @file kernel_ir_internal.h
 * @brief Internal KernelIR helpers shared by builders and evaluators.
 *
 * The helpers centralize scalar-domain normalization, exact integer packing,
 * warp guard conversion, and evaluator cache state. They are private to the
 * core KernelIR implementation and intentionally expose only the invariants
 * needed by adjacent translation units.
 */
#ifndef KERNEL_IR_INTERNAL_H
#define KERNEL_IR_INTERNAL_H

#include "oakfield/kernel_ir.h"

#include "oakfield/operator.h"
#include "operators/common/warp_safety.h"
#include "oakfield/math/special_functions.h"

#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline double sim_ir_positive_or_default(double value, double fallback) {
    if (!isfinite(value) || value <= 0.0) {
        return fallback;
    }
    return value;
}

static inline double sim_ir_canonicalize_constant(double value) {
    if (value == 0.0) {
        return 0.0;
    }
    return value;
}

static inline bool sim_ir_domain_is_exact_integer(SimScalarDomain domain) {
    return sim_scalar_domain_is_integer(domain);
}

static inline uint64_t sim_ir_integer_mask(SimScalarDomain domain) {
    if (domain.bit_width >= 64U) {
        return UINT64_MAX;
    }
    return (UINT64_C(1) << domain.bit_width) - UINT64_C(1);
}

static inline uint64_t sim_ir_integer_truncate(uint64_t raw, SimScalarDomain domain) {
    if (domain.bit_width >= 64U) {
        return raw;
    }
    return raw & sim_ir_integer_mask(domain);
}

static inline int64_t sim_ir_integer_as_i64(uint64_t raw, SimScalarDomain domain) {
    raw = sim_ir_integer_truncate(raw, domain);
    if (!domain.is_signed) {
        return (int64_t) raw;
    }
    if (domain.bit_width >= 64U) {
        return (int64_t) raw;
    }

    uint64_t mask     = sim_ir_integer_mask(domain);
    uint64_t sign_bit = UINT64_C(1) << (domain.bit_width - 1U);
    if ((raw & sign_bit) != 0U) {
        raw |= ~mask;
    }
    return (int64_t) raw;
}

static inline uint64_t sim_ir_integer_from_i64(int64_t value, SimScalarDomain domain) {
    return sim_ir_integer_truncate((uint64_t) value, domain);
}

static inline bool sim_ir_integer_raw_from_signed(int64_t value, SimScalarDomain domain, uint64_t* out_raw) {
    if (!sim_ir_domain_is_exact_integer(domain) || !domain.is_signed || out_raw == NULL) {
        return false;
    }
    if (domain.bit_width == 32U && (value < INT32_MIN || value > INT32_MAX)) {
        return false;
    }
    if (domain.bit_width != 32U && domain.bit_width != 64U) {
        return false;
    }
    *out_raw = sim_ir_integer_from_i64(value, domain);
    return true;
}

static inline bool
sim_ir_integer_raw_from_unsigned(uint64_t value, SimScalarDomain domain, uint64_t* out_raw) {
    if (!sim_ir_domain_is_exact_integer(domain) || domain.is_signed || out_raw == NULL) {
        return false;
    }
    if (domain.bit_width == 32U && value > UINT32_MAX) {
        return false;
    }
    if (domain.bit_width != 32U && domain.bit_width != 64U) {
        return false;
    }
    *out_raw = sim_ir_integer_truncate(value, domain);
    return true;
}

static inline bool sim_ir_integer_raw_from_double(double value, SimScalarDomain domain, uint64_t* out_raw) {
    if (!sim_ir_domain_is_exact_integer(domain) || out_raw == NULL || !isfinite(value) ||
        trunc(value) != value) {
        return false;
    }

    if (domain.is_signed) {
        if (domain.bit_width == 32U) {
            if (value < (double) INT32_MIN || value > (double) INT32_MAX) {
                return false;
            }
            return sim_ir_integer_raw_from_signed((int64_t) value, domain, out_raw);
        }
        if (value < (double) INT64_MIN || value > (double) INT64_MAX) {
            return false;
        }
        return sim_ir_integer_raw_from_signed((int64_t) value, domain, out_raw);
    }

    if (value < 0.0) {
        return false;
    }
    if (domain.bit_width == 32U) {
        if (value > (double) UINT32_MAX) {
            return false;
        }
        return sim_ir_integer_raw_from_unsigned((uint64_t) value, domain, out_raw);
    }
    if (value > (double) UINT64_MAX) {
        return false;
    }
    return sim_ir_integer_raw_from_unsigned((uint64_t) value, domain, out_raw);
}

static inline double sim_ir_integer_raw_to_double(uint64_t raw, SimScalarDomain domain) {
    if (domain.is_signed) {
        return (double) sim_ir_integer_as_i64(raw, domain);
    }
    return (double) sim_ir_integer_truncate(raw, domain);
}


static inline SimIRType sim_ir_type_normalize(SimIRType type) {
    if (type.components == 0U) {
        type.components = 1U;
    }

    if (!sim_scalar_domain_validate(type.scalar_domain) ||
        type.scalar_domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        type.scalar_domain = sim_scalar_domain_f64();
    }

    if (sim_scalar_domain_is_complex(type.scalar_domain) && type.components < 2U) {
        type.components = 2U;
    }

    if (type.components <= 1U) {
        type.kind       = SIM_IR_VALUE_SCALAR;
        type.components = 1U;
    } else {
        type.kind = SIM_IR_VALUE_VECTOR;
    }
    return type;
}

static inline SimWarpGuard sim_ir_guard_to_runtime(const SimIRWarpGuard* guard) {
    SimWarpGuard runtime = { .mode = guard ? (SimContinuityMode) guard->mode : SIM_CONTINUITY_NONE,
                             .clamp_min = guard ? guard->clamp_min : 0.0,
                             .clamp_max = guard ? guard->clamp_max : 0.0,
                             .tolerance = guard ? guard->tolerance : 0.0 };
    return runtime;
}


typedef struct SimIREvalState {
    const SimIRBuilder*          builder;
    const SimIREvaluator*        evaluator;
    const SimIREvaluatorComplex* evaluator_complex;
    double*                      cache;
    SimComplexDouble*            cache_complex;
    unsigned char*               flags;
} SimIREvalState;

bool sim_ir_constant_component(const SimIRBuilder* builder,
                               const SimIRNode*    node,
                               size_t              component,
                               double*             out_value);

static inline SimIRDomainValue sim_ir_domain_value_zero(SimScalarDomain domain) {
    SimIRDomainValue value;
    value.domain           = domain;
    value.value.as_u64     = 0U;
    value.value.as_i64     = 0;
    value.value.as_f64     = 0.0;
    value.value.as_complex = (SimComplexDouble) { 0.0, 0.0 };
    return value;
}

static inline SimIRDomainValue sim_ir_domain_value_from_integer_raw(SimScalarDomain domain, uint64_t raw) {
    SimIRDomainValue value = sim_ir_domain_value_zero(domain);
    raw                    = sim_ir_integer_truncate(raw, domain);
    if (domain.is_signed) {
        value.value.as_i64 = sim_ir_integer_as_i64(raw, domain);
    } else {
        value.value.as_u64 = raw;
    }
    return value;
}

static inline bool sim_ir_domain_value_to_integer_raw(SimIRDomainValue value,
                                               SimScalarDomain  expected_domain,
                                               uint64_t*        out_raw) {
    if (out_raw == NULL || !sim_ir_domain_is_exact_integer(expected_domain)) {
        return false;
    }

    if (value.domain.kind == SIM_SCALAR_DOMAIN_UNKNOWN) {
        value.domain = expected_domain;
    }
    if (!sim_scalar_domain_equal(value.domain, expected_domain)) {
        return false;
    }

    if (expected_domain.is_signed) {
        return sim_ir_integer_raw_from_signed(value.value.as_i64, expected_domain, out_raw);
    }
    return sim_ir_integer_raw_from_unsigned(value.value.as_u64, expected_domain, out_raw);
}

#endif /* KERNEL_IR_INTERNAL_H */
