/**
 * @file kernel_ir.c
 * @brief KernelIR type helpers, warp sampling, and complex GPU legality checks.
 *
 * This file contains cross-cutting KernelIR utilities that are shared by the
 * builder, evaluators, and operator dispatch. It preserves scalar-domain
 * conventions, assigns warp classifications from analytic profiles, and reports
 * when complex-valued nodes require CPU fallback because backend semantics are
 * not yet supported.
 */
#include "internal/kernel_ir_internal.h"

typedef struct SimIRWarpGradientContext {
    SimIRWarpProfile profile;
    double           delta;
    double           tolerance;
} SimIRWarpGradientContext;

static bool sim_ir_node_has_supported_complex_gpu_semantics(const SimIRNode* node) {
    SimScalarDomain domain;

    if (node == NULL) {
        return false;
    }

    domain = sim_ir_type_scalar_domain(node->value_type);
    if (!sim_scalar_domain_is_complex(domain)) {
        return true;
    }

    switch (node->type) {
        case SIM_IR_NODE_CONSTANT:
        case SIM_IR_NODE_FIELD_REF:
        case SIM_IR_NODE_ADD:
        case SIM_IR_NODE_SUB:
        case SIM_IR_NODE_DIFF:
        case SIM_IR_NODE_WARP:
        case SIM_IR_NODE_COMPLEX_PACK:
        case SIM_IR_NODE_COMPLEX_ROTATE:
            return true;
        default:
            return false;
    }
}

bool sim_ir_kernel_has_unsupported_complex_semantics(const struct KernelIR* kernel) {
    if (kernel == NULL || kernel->builder == NULL) {
        return false;
    }

    if (kernel->complex_semantics == SIM_KERNEL_COMPLEX_SEMANTICS_COMPONENTWISE) {
        return false;
    }

    for (size_t i = 0U; i < kernel->builder->count; ++i) {
        const SimIRNode* node = &kernel->builder->nodes[i];
        if (!sim_ir_node_has_supported_complex_gpu_semantics(node)) {
            return true;
        }
    }

    return false;
}

static SimResult sim_ir_warp_gradient_probe(void*                userdata,
                                            double               sample,
                                            SimSpecialFallbackFn fallback,
                                            void*                fallback_userdata,
                                            double*              out_gradient) {
    (void) fallback;
    (void) fallback_userdata;

    if (userdata == NULL || out_gradient == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimIRWarpGradientContext* ctx   = (const SimIRWarpGradientContext*) userdata;
    double                          delta = sim_ir_positive_or_default(ctx->delta, 1.0e-6);

    double f_plus  = sim_ir_warp_profile_eval(ctx->profile, sample + delta, ctx->tolerance);
    double f_minus = sim_ir_warp_profile_eval(ctx->profile, sample - delta, ctx->tolerance);

    if (!isfinite(f_plus) || !isfinite(f_minus)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    double gradient = (f_plus - f_minus) / (2.0 * delta);
    if (!isfinite(gradient)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_gradient = gradient;
    return SIM_RESULT_OK;
}

SimResult sim_ir_warp_sample_response(const SimWarpSampleSpec* spec,
                                      SimIRWarpProfile         profile,
                                      double                   tolerance,
                                      SimSpecialFallbackFn     fallback,
                                      void*                    fallback_userdata,
                                      double*                  out_response) {
    if (spec == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimIRWarpGradientContext ctx = { .profile   = profile,
                                     .delta     = sim_ir_positive_or_default(spec->delta, 1.0e-6),
                                     .tolerance = sim_ir_positive_or_default(tolerance, 1.0e-6) };

    return sim_warp_sample_response(
        spec, sim_ir_warp_gradient_probe, &ctx, fallback, fallback_userdata, out_response);
}

double sim_ir_warp_profile_eval(SimIRWarpProfile profile, double x, double tolerance) {
    switch (profile) {
        case SIM_IR_WARP_PROFILE_TRIGAMMA:
            return sim_special_trigamma(x);
        case SIM_IR_WARP_PROFILE_DIGAMMA_7_TAIL:
            return sim_digamma_f64_7(x);
        case SIM_IR_WARP_PROFILE_DIGAMMA_5_TAIL:
            return sim_digamma_f64_5(x);
        case SIM_IR_WARP_PROFILE_DIGAMMA_ADAPTIVE:
            return sim_digamma_f64_tail(x, sim_ir_positive_or_default(tolerance, 1.0e-6));
        case SIM_IR_WARP_PROFILE_DIGAMMA_MORTICI:
            return sim_digamma_f64_mortici(x);
        case SIM_IR_WARP_PROFILE_DIGAMMA:
        default:
            return sim_special_digamma(x);
    }
}

double
sim_ir_warp_difference(SimIRWarpProfile profile, double sample, double delta, double tolerance) {
    SimWarpGuard guard = {
        .mode = SIM_CONTINUITY_NONE, .clamp_min = 0.0, .clamp_max = 0.0, .tolerance = 0.0
    };

    SimWarpSampleSpec spec = {
        .sample = sample, .bias = 0.0, .delta = delta, .lambda = 1.0, .guard = guard
    };

    double    response = 0.0;
    SimResult rc = sim_ir_warp_sample_response(&spec, profile, tolerance, NULL, NULL, &response);
    if (rc != SIM_RESULT_OK) {
        return 0.0;
    }

    return response;
}

SimIRType sim_ir_type_scalar(void) {
    SimIRType t = { SIM_IR_VALUE_SCALAR, 1U, sim_scalar_domain_f64() };
    return t;
}

SimIRType sim_ir_type_scalar_domain_typed(SimScalarDomain domain) {
    SimIRType t = { SIM_IR_VALUE_SCALAR, 1U, domain };
    return sim_ir_type_normalize(t);
}

SimIRType sim_ir_type_vector(size_t n) {
    SimIRType t = { (n <= 1U) ? SIM_IR_VALUE_SCALAR : SIM_IR_VALUE_VECTOR,
                    (n == 0U ? 1U : n),
                    sim_scalar_domain_f64() };
    return t;
}

SimIRType sim_ir_type_complex(void) {
    SimIRType t = { SIM_IR_VALUE_VECTOR, 2U, sim_scalar_domain_c64() };
    return t;
}

bool sim_ir_type_is_scalar(SimIRType type) {
    type = sim_ir_type_normalize(type);
    return type.kind == SIM_IR_VALUE_SCALAR;
}

bool sim_ir_type_equal(SimIRType lhs, SimIRType rhs) {
    lhs = sim_ir_type_normalize(lhs);
    rhs = sim_ir_type_normalize(rhs);
    return (lhs.kind == rhs.kind) && (lhs.components == rhs.components) &&
           sim_scalar_domain_equal(lhs.scalar_domain, rhs.scalar_domain);
}

SimScalarDomain sim_ir_type_scalar_domain(SimIRType type) {
    type = sim_ir_type_normalize(type);
    return type.scalar_domain;
}
