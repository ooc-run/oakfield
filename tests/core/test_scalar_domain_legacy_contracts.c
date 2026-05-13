#include <oakfield/sim.h>

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef struct DomainEvalProbe {
    int field_calls;
    int param_calls;
    SimScalarDomain last_field_domain;
    SimScalarDomain last_param_domain;
} DomainEvalProbe;

static SimResult run_scalar_domain_field_probe(void *userdata, size_t field_id, SimIRType type,
                                               SimScalarDomain domain,
                                               SimIRDomainValue *out_value) {
    DomainEvalProbe *probe = (DomainEvalProbe *)userdata;
    (void)type;

    if (probe == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    probe->field_calls += 1;
    probe->last_field_domain = domain;
    if (field_id != 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_scalar_domain_is_complex(domain)) {
        out_value->domain = domain;
        out_value->value.as_complex.re = 2.0;
        out_value->value.as_complex.im = -0.5;
    } else {
        out_value->domain = domain;
        out_value->value.as_f64 = 2.0;
    }
    return SIM_RESULT_OK;
}

static SimResult run_scalar_domain_param_probe(void *userdata, SimIRParamKind param,
                                               SimScalarDomain domain,
                                               SimIRDomainValue *out_value) {
    DomainEvalProbe *probe = (DomainEvalProbe *)userdata;
    if (probe == NULL || out_value == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    probe->param_calls += 1;
    probe->last_param_domain = domain;
    if (param != SIM_IR_PARAM_DT) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    out_value->domain = domain;
    out_value->value.as_f64 = 3.5;
    return SIM_RESULT_OK;
}

static bool run_scalar_domain_validation_case(void) {
    SimScalarDomain unknown = sim_scalar_domain_unknown();
    SimScalarDomain f64 = sim_scalar_domain_f64();
    SimScalarDomain c64 = sim_scalar_domain_c64();
    SimScalarDomain i8 = sim_scalar_domain_i8();
    SimScalarDomain i32 = sim_scalar_domain_i32();
    SimScalarDomain i64 = sim_scalar_domain_i64();
    SimScalarDomain u8 = sim_scalar_domain_u8();
    SimScalarDomain u32 = sim_scalar_domain_u32();
    SimScalarDomain u64 = sim_scalar_domain_u64();
    SimScalarDomain invalid = {
        .kind = SIM_SCALAR_DOMAIN_REAL, .bit_width = 32U, .is_signed = true, .modulus = 0U};

    if (!sim_scalar_domain_validate(unknown)) {
        fprintf(stderr, "[FAIL] unknown scalar domain should be valid\n");
        return false;
    }
    if (!sim_scalar_domain_validate(f64) || !sim_scalar_domain_validate(c64) ||
        !sim_scalar_domain_validate(i8) || !sim_scalar_domain_validate(i32) ||
        !sim_scalar_domain_validate(i64) || !sim_scalar_domain_validate(u8) ||
        !sim_scalar_domain_validate(u32) || !sim_scalar_domain_validate(u64)) {
        fprintf(stderr, "[FAIL] expected canonical scalar domains to validate\n");
        return false;
    }
    if (sim_scalar_domain_validate(invalid)) {
        fprintf(stderr, "[FAIL] invalid real(32-bit) domain should be rejected\n");
        return false;
    }
    return true;
}

static bool run_scalar_domain_name_case(void) {
    SimScalarDomain parsed = sim_scalar_domain_unknown();

    if (strcmp(sim_scalar_domain_name(sim_scalar_domain_f64()), "f64") != 0) {
        fprintf(stderr, "[FAIL] f64 name mismatch\n");
        return false;
    }
    if (strcmp(sim_scalar_domain_name(sim_scalar_domain_c64()), "c64") != 0) {
        fprintf(stderr, "[FAIL] c64 name mismatch\n");
        return false;
    }
    if (strcmp(sim_scalar_domain_name(sim_scalar_domain_i8()), "i8") != 0) {
        fprintf(stderr, "[FAIL] i8 name mismatch\n");
        return false;
    }
    if (strcmp(sim_scalar_domain_name(sim_scalar_domain_i32()), "i32") != 0) {
        fprintf(stderr, "[FAIL] i32 name mismatch\n");
        return false;
    }
    if (!sim_scalar_domain_from_name("u8", &parsed)) {
        fprintf(stderr, "[FAIL] expected u8 parse success\n");
        return false;
    }
    if (!sim_scalar_domain_equal(parsed, sim_scalar_domain_u8())) {
        fprintf(stderr, "[FAIL] parsed u8 mismatch\n");
        return false;
    }
    if (!sim_scalar_domain_from_name("u64", &parsed)) {
        fprintf(stderr, "[FAIL] expected u64 parse success\n");
        return false;
    }
    if (!sim_scalar_domain_equal(parsed, sim_scalar_domain_u64())) {
        fprintf(stderr, "[FAIL] parsed u64 mismatch\n");
        return false;
    }
    if (sim_scalar_domain_from_name("f16", &parsed)) {
        fprintf(stderr, "[FAIL] unexpected parse success for unsupported f16\n");
        return false;
    }
    return true;
}

static bool run_scalar_domain_capability_case(void) {
    SimScalarDomain f64 = sim_scalar_domain_f64();
    SimScalarDomain c64 = sim_scalar_domain_c64();
    SimScalarDomain i32 = sim_scalar_domain_i32();

    if (!sim_scalar_domain_supports(f64, SIM_SCALAR_CAP_ANALYTIC_CALL | SIM_SCALAR_CAP_FLOOR |
                                             SIM_SCALAR_CAP_MODULO)) {
        fprintf(stderr, "[FAIL] f64 should support analytic/floor/mod\n");
        return false;
    }
    if (!sim_scalar_domain_supports(c64, SIM_SCALAR_CAP_COMPLEX_ROTATION)) {
        fprintf(stderr, "[FAIL] c64 should support complex rotation\n");
        return false;
    }
    if (sim_scalar_domain_supports(c64, SIM_SCALAR_CAP_ANALYTIC_CALL | SIM_SCALAR_CAP_FLOOR)) {
        fprintf(stderr, "[FAIL] c64 should not expose real analytic/floor capability by default\n");
        return false;
    }
    if (!sim_scalar_domain_supports(i32, SIM_SCALAR_CAP_ORDERING | SIM_SCALAR_CAP_MODULO |
                                             SIM_SCALAR_CAP_DIVISION | SIM_SCALAR_CAP_POWER)) {
        fprintf(stderr, "[FAIL] i32 should support ordering/modulo/division/power\n");
        return false;
    }
    return true;
}

static bool run_scalar_domain_field_bridge_case(void) {
    SimField real_field = {0};
    SimField complex_field = {0};
    size_t shape[1] = {4U};
    SimResult rc;

    rc = sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real field init failed\n");
        return false;
    }
    rc = sim_field_init(&complex_field, 1U, shape, sizeof(SimComplexDouble),
                        SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex field init failed\n");
        sim_field_destroy(&real_field);
        return false;
    }
    complex_field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
    complex_field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT;

    SimScalarDomain real_domain = sim_scalar_domain_from_field(&real_field);
    SimScalarDomain complex_domain = sim_scalar_domain_from_field(&complex_field);

    if (!sim_scalar_domain_equal(real_domain, sim_scalar_domain_f64())) {
        fprintf(stderr, "[FAIL] real field should map to f64 scalar domain\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&complex_field);
        return false;
    }
    if (!sim_scalar_domain_equal(complex_domain, sim_scalar_domain_c64())) {
        fprintf(stderr, "[FAIL] constrained complex field should map to c64 scalar domain\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&complex_field);
        return false;
    }

    sim_field_destroy(&real_field);
    sim_field_destroy(&complex_field);
    return true;
}

static bool run_scalar_domain_representation_helper_case(void) {
    SimFieldRepresentation real_repr = {.domain = SIM_FIELD_DOMAIN_PHYSICAL,
                                        .value_kind = SIM_FIELD_VALUE_REAL_SCALAR};
    SimFieldRepresentation complex_repr = {.domain = SIM_FIELD_DOMAIN_PHYSICAL,
                                           .value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR};
    SimFieldRepresentation imag_zero_repr = {.domain = SIM_FIELD_DOMAIN_PHYSICAL,
                                             .value_kind =
                                                 SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT};
    SimFieldRepresentation constrained_repr = {
        .domain = SIM_FIELD_DOMAIN_SPECTRAL, .value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT};
    SimFieldRepresentation invalid_constraint_repr = {
        .domain = SIM_FIELD_DOMAIN_PHYSICAL, .value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT};
    SimField real_field = {0};
    SimField imag_zero_field = {0};
    SimField constrained_field = {0};
    size_t shape[1] = {2U};
    SimResult rc;

    if (sim_field_value_kind_is_complex_valued(SIM_FIELD_VALUE_REAL_SCALAR)) {
        fprintf(stderr, "[FAIL] representation helper: real value-kind should not be complex\n");
        return false;
    }
    if (!sim_field_value_kind_is_complex_valued(SIM_FIELD_VALUE_COMPLEX_SCALAR)) {
        fprintf(stderr, "[FAIL] representation helper: complex value-kind should be complex\n");
        return false;
    }
    if (!sim_field_value_kind_is_complex_valued(SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT)) {
        fprintf(
            stderr,
            "[FAIL] representation helper: imag-zero constrained value-kind should be complex\n");
        return false;
    }
    if (!sim_field_value_kind_is_complex_valued(SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT)) {
        fprintf(stderr,
                "[FAIL] representation helper: constrained complex value-kind should be complex\n");
        return false;
    }
    if (sim_field_value_kind_has_imag_zero_constraint(SIM_FIELD_VALUE_COMPLEX_SCALAR)) {
        fprintf(
            stderr,
            "[FAIL] representation helper: unconstrained complex should not report imag-zero\n");
        return false;
    }
    if (!sim_field_value_kind_has_imag_zero_constraint(
            SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT)) {
        fprintf(stderr, "[FAIL] representation helper: imag-zero constraint should be reported\n");
        return false;
    }
    if (sim_field_value_kind_has_spectral_real_constraint(SIM_FIELD_VALUE_COMPLEX_SCALAR)) {
        fprintf(
            stderr,
            "[FAIL] representation helper: unconstrained complex should not report constraint\n");
        return false;
    }
    if (!sim_field_value_kind_has_spectral_real_constraint(
            SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT)) {
        fprintf(stderr,
                "[FAIL] representation helper: constrained complex should report constraint\n");
        return false;
    }
    if (sim_field_representation_requires_complex_storage(real_repr)) {
        fprintf(stderr,
                "[FAIL] representation helper: real repr should not require complex storage\n");
        return false;
    }
    if (!sim_field_representation_requires_complex_storage(complex_repr) ||
        !sim_field_representation_requires_complex_storage(imag_zero_repr) ||
        !sim_field_representation_requires_complex_storage(constrained_repr)) {
        fprintf(stderr,
                "[FAIL] representation helper: complex repr should require complex storage\n");
        return false;
    }
    if (!sim_field_representation_has_imag_zero_constraint(imag_zero_repr)) {
        fprintf(
            stderr,
            "[FAIL] representation helper: imag-zero repr should report imag-zero constraint\n");
        return false;
    }
    if (sim_field_representation_has_imag_zero_constraint(constrained_repr)) {
        fprintf(
            stderr,
            "[FAIL] representation helper: spectral real constraint must not imply imag-zero\n");
        return false;
    }
    if (!sim_field_representation_has_spectral_real_constraint(constrained_repr)) {
        fprintf(
            stderr,
            "[FAIL] representation helper: constrained spectral repr should report constraint\n");
        return false;
    }
    if (sim_field_representation_has_spectral_real_constraint(invalid_constraint_repr)) {
        fprintf(stderr, "[FAIL] representation helper: non-spectral repr must not report spectral "
                        "constraint\n");
        return false;
    }

    rc = sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation helper: real field init failed\n");
        return false;
    }
    if (sim_field_storage_is_complex(&real_field) || sim_field_domain_is_complex(&real_field)) {
        fprintf(stderr,
                "[FAIL] representation helper: real field should be real in storage+domain\n");
        sim_field_destroy(&real_field);
        return false;
    }
    if (sim_field_complex_mode(&real_field)) {
        fprintf(stderr,
                "[FAIL] representation helper: real field should not require complex mode\n");
        sim_field_destroy(&real_field);
        return false;
    }

    /* Compatibility bridge: legacy hint still requests complex promotion. */
    real_field.complex_mode = true;
    if (!sim_field_complex_mode(&real_field)) {
        fprintf(stderr, "[FAIL] representation helper: legacy complex_mode hint ignored\n");
        sim_field_destroy(&real_field);
        return false;
    }
    {
        bool storage_complex = sim_field_storage_is_complex(&real_field);
        bool domain_complex = sim_field_domain_is_complex(&real_field);
        if (storage_complex || !domain_complex) {
            fprintf(stderr,
                    "[FAIL] representation helper: complex_mode bridge should affect domain, not "
                    "storage (storage=%d domain=%d complex_mode=%d)\n",
                    storage_complex ? 1 : 0, domain_complex ? 1 : 0,
                    sim_field_complex_mode(&real_field) ? 1 : 0);
            sim_field_destroy(&real_field);
            return false;
        }
    }

    rc = sim_field_init(&constrained_field, 1U, shape, sizeof(SimComplexDouble),
                        SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation helper: constrained field init failed\n");
        sim_field_destroy(&real_field);
        return false;
    }

    rc = sim_field_init(&imag_zero_field, 1U, shape, sizeof(SimComplexDouble),
                        SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation helper: imag-zero field init failed\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&constrained_field);
        return false;
    }

    rc = sim_field_set_representation(&imag_zero_field, imag_zero_repr);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation helper: set imag-zero repr failed (%d)\n", (int)rc);
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }

    rc = sim_field_set_representation(&constrained_field, constrained_repr);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation helper: set constrained repr failed (%d)\n",
                (int)rc);
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }

    rc = sim_field_require_complex(&constrained_field);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation helper: require_complex failed (%d)\n", (int)rc);
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }

    imag_zero_repr = sim_field_representation(&imag_zero_field);
    if (imag_zero_repr.domain != SIM_FIELD_DOMAIN_PHYSICAL ||
        imag_zero_repr.value_kind != SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT) {
        fprintf(stderr, "[FAIL] representation helper: imag-zero repr should be preserved\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }
    constrained_repr = sim_field_representation(&constrained_field);
    if (constrained_repr.domain != SIM_FIELD_DOMAIN_SPECTRAL ||
        constrained_repr.value_kind != SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT) {
        fprintf(stderr,
                "[FAIL] representation helper: require_complex should preserve constrained repr\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }
    if (!sim_field_storage_is_complex(&imag_zero_field) ||
        !sim_field_domain_is_complex(&imag_zero_field)) {
        fprintf(stderr, "[FAIL] representation helper: imag-zero complex should be complex in "
                        "storage+domain\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }
    if (!sim_field_storage_is_complex(&constrained_field) ||
        !sim_field_domain_is_complex(&constrained_field)) {
        fprintf(stderr, "[FAIL] representation helper: constrained complex should be complex in "
                        "storage+domain\n");
        sim_field_destroy(&real_field);
        sim_field_destroy(&imag_zero_field);
        sim_field_destroy(&constrained_field);
        return false;
    }

    sim_field_destroy(&real_field);
    sim_field_destroy(&imag_zero_field);
    sim_field_destroy(&constrained_field);
    return true;
}

static bool run_scalar_domain_ir_bridge_case(void) {
    SimIRType scalar_type = sim_ir_type_scalar();
    SimIRType complex_type = sim_ir_type_complex();
    SimIRType vector_real_type = sim_ir_type_vector(4U);

    if (!sim_scalar_domain_equal(sim_ir_type_scalar_domain(scalar_type), sim_scalar_domain_f64())) {
        fprintf(stderr, "[FAIL] scalar IR type should map to f64\n");
        return false;
    }
    if (!sim_scalar_domain_equal(sim_ir_type_scalar_domain(complex_type),
                                 sim_scalar_domain_c64())) {
        fprintf(stderr, "[FAIL] complex IR type should map to c64\n");
        return false;
    }
    if (!sim_scalar_domain_equal(sim_ir_type_scalar_domain(vector_real_type),
                                 sim_scalar_domain_f64())) {
        fprintf(stderr, "[FAIL] real vector IR type should map to f64 domain\n");
        return false;
    }
    return true;
}

static bool run_scalar_domain_ir_legality_case(void) {
    SimIRBuilder builder = {0};
    SimIRType int_type;
    SimIRNodeId int_a;
    SimIRNodeId int_b;
    SimIRNodeId int_add;
    SimIRNodeId int_div;
    SimIRNodeId int_pow;
    SimIRNodeId real_a;
    SimIRNodeId real_b;
    SimIRNodeId real_div;
    SimIRNodeId real_pow;
    SimIRNodeId pack_real;
    SimIRNodeId complex_node;
    SimIRNodeId pack_invalid;

    if (sim_ir_builder_init(&builder) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] ir legality: builder init failed\n");
        return false;
    }

    int_type = sim_ir_type_scalar();
    int_type.scalar_domain = sim_scalar_domain_i32();

    int_a = sim_ir_builder_constant_typed(&builder, 6.0, int_type);
    int_b = sim_ir_builder_constant_typed(&builder, 2.0, int_type);
    if (int_a == SIM_IR_INVALID_NODE || int_b == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: integer constants failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    int_add = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, int_a, int_b);
    if (int_add == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: integer add should be legal\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    int_div = sim_ir_builder_binary(&builder, SIM_IR_NODE_DIV, int_a, int_b);
    if (int_div == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: integer div should be legal\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    int_pow = sim_ir_builder_binary(&builder, SIM_IR_NODE_POW, int_a, int_b);
    if (int_pow == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: integer pow should be legal\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    real_a = sim_ir_builder_constant(&builder, 6.0);
    real_b = sim_ir_builder_constant(&builder, 2.0);
    if (real_a == SIM_IR_INVALID_NODE || real_b == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: real constants failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    real_div = sim_ir_builder_binary(&builder, SIM_IR_NODE_DIV, real_a, real_b);
    if (real_div == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: real div should be legal\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    real_pow = sim_ir_builder_binary(&builder, SIM_IR_NODE_POW, real_a, real_b);
    if (real_pow == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: real pow should be legal\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    pack_real = sim_ir_builder_complex_pack(&builder, real_a, real_b);
    if (pack_real == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: complex pack(real,real) should be legal\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    complex_node = sim_ir_builder_constant_complex(&builder, 1.0, -1.0);
    if (complex_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: complex constant failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    pack_invalid = sim_ir_builder_complex_pack(&builder, complex_node, real_a);
    if (pack_invalid != SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] ir legality: complex pack should reject complex operands\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool run_scalar_domain_unified_evaluator_case(void) {
    SimIRBuilder builder = {0};
    SimIRNodeId param_node;
    SimIRNodeId complex_field_node;
    SimIRNodeId int_constant_node;
    SimIRType int_type;
    SimIRDomainValue value = {0};
    DomainEvalProbe probe = {0};
    SimIREvaluatorDomain evaluator = {0};

    if (sim_ir_builder_init(&builder) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] unified evaluator: builder init failed\n");
        return false;
    }

    evaluator.field_value = run_scalar_domain_field_probe;
    evaluator.param_value = run_scalar_domain_param_probe;
    evaluator.userdata = &probe;

    param_node = sim_ir_builder_param(&builder, SIM_IR_PARAM_DT);
    if (param_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] unified evaluator: param node failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    if (sim_ir_evaluate_domain(&builder, param_node, &evaluator, &value) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] unified evaluator: real-domain evaluate failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }
    if (!sim_scalar_domain_equal(value.domain, sim_scalar_domain_f64()) ||
        value.value.as_f64 != 3.5) {
        fprintf(stderr, "[FAIL] unified evaluator: real-domain value mismatch\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    complex_field_node = sim_ir_builder_field_ref_typed(&builder, 0U, sim_ir_type_complex());
    if (complex_field_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] unified evaluator: complex field node failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    if (sim_ir_evaluate_domain(&builder, complex_field_node, &evaluator, &value) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] unified evaluator: complex-domain evaluate failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }
    if (!sim_scalar_domain_equal(value.domain, sim_scalar_domain_c64()) ||
        value.value.as_complex.re != 2.0 || value.value.as_complex.im != -0.5) {
        fprintf(stderr, "[FAIL] unified evaluator: complex-domain value mismatch\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    int_type = sim_ir_type_scalar();
    int_type.scalar_domain = sim_scalar_domain_i32();
    int_constant_node = sim_ir_builder_constant_typed(&builder, 7.0, int_type);
    if (int_constant_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] unified evaluator: integer constant failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }
    if (sim_ir_evaluate_domain(&builder, int_constant_node, NULL, &value) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] unified evaluator: integer evaluate failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }
    if (!sim_scalar_domain_equal(value.domain, sim_scalar_domain_i32()) ||
        value.value.as_i64 != 7) {
        fprintf(stderr, "[FAIL] unified evaluator: integer-domain value mismatch\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    if (probe.param_calls != 1 || probe.field_calls != 1 ||
        !sim_scalar_domain_equal(probe.last_param_domain, sim_scalar_domain_f64()) ||
        !sim_scalar_domain_equal(probe.last_field_domain, sim_scalar_domain_c64())) {
        fprintf(stderr, "[FAIL] unified evaluator: callback probe mismatch\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool run_scalar_domain_real_kernel_case(void) {
    SimContext context = {0};
    SimBackend backend = {0};
    SimField src = {0};
    SimField dst = {0};
    SimIRBuilder *builder = NULL;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor output;
    SimOperatorKernelDescriptor kernel_desc = {0};
    SimOperatorDescriptor descriptor = {0};
    size_t shape[1] = {4U};
    size_t src_index = 0U;
    size_t dst_index = 0U;
    bool context_ready = false;
    bool backend_ready = false;
    bool src_owned = false;
    bool dst_owned = false;
    bool ok = false;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: src init failed\n");
        goto cleanup;
    }
    src_owned = true;
    if (sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: dst init failed\n");
        goto cleanup;
    }
    dst_owned = true;

    {
        double *src_data = sim_field_real_data(&src);
        if (src_data == NULL) {
            fprintf(stderr, "[FAIL] real kernel: src data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            src_data[i] = (double)(i + 1U);
        }
    }

    if (sim_context_add_field(&context, &src, &src_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: add src failed\n");
        goto cleanup;
    }
    src_owned = false;
    if (sim_context_add_field(&context, &dst, &dst_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: add dst failed\n");
        goto cleanup;
    }
    dst_owned = false;

    builder = sim_context_ir_builder(&context);
    if (builder == NULL) {
        fprintf(stderr, "[FAIL] real kernel: builder missing\n");
        goto cleanup;
    }

    SimIRNodeId src_node = sim_ir_builder_field_ref(builder, 0U);
    SimIRNodeId one_node = sim_ir_builder_constant(builder, 1.0);
    SimIRNodeId sum_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, src_node, one_node);
    if (src_node == SIM_IR_INVALID_NODE || one_node == SIM_IR_INVALID_NODE ||
        sum_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] real kernel: ir node build failed\n");
        goto cleanup;
    }

    bindings[0].ir_field_index = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index = 1U;
    bindings[1].context_field_index = dst_index;
    output.ir_field_index = 1U;
    output.expression = sum_node;

    kernel_desc.builder = builder;
    kernel_desc.bindings = bindings;
    kernel_desc.binding_count = 2U;
    kernel_desc.outputs = &output;
    kernel_desc.output_count = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name = "scalar_domain_real_kernel";
    descriptor.kernel = &kernel_desc;
    if (sim_context_register_operator(&context, &descriptor, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: operator registration failed\n");
        goto cleanup;
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: backend init failed\n");
        goto cleanup;
    }
    backend_ready = true;
    sim_context_set_backend(&context, &backend);

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] real kernel: execute failed\n");
        goto cleanup;
    }

    {
        const SimField *dst_field = sim_context_field(&context, dst_index);
        const double *dst_data = (dst_field != NULL) ? sim_field_real_data_const(dst_field) : NULL;
        if (dst_data == NULL) {
            fprintf(stderr, "[FAIL] real kernel: dst data missing\n");
            goto cleanup;
        }
        if (!sim_scalar_domain_equal(sim_scalar_domain_from_field(dst_field),
                                     sim_scalar_domain_f64())) {
            fprintf(stderr, "[FAIL] real kernel: dst scalar domain should be f64\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            double expected = (double)(i + 2U);
            if (fabs(dst_data[i] - expected) > 1.0e-12) {
                fprintf(stderr, "[FAIL] real kernel: dst[%zu]=%g expected %g\n", i, dst_data[i],
                        expected);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    if (src_owned) {
        sim_field_destroy(&src);
    }
    if (dst_owned) {
        sim_field_destroy(&dst);
    }
    if (backend_ready) {
        backend_destroy(&backend);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_complex_kernel_case(void) {
    SimContext context = {0};
    SimBackend backend = {0};
    SimField src = {0};
    SimField dst = {0};
    SimIRBuilder *builder = NULL;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor output;
    SimOperatorKernelDescriptor kernel_desc = {0};
    SimOperatorDescriptor descriptor = {0};
    size_t shape[1] = {4U};
    size_t src_index = 0U;
    size_t dst_index = 0U;
    bool context_ready = false;
    bool backend_ready = false;
    bool src_owned = false;
    bool dst_owned = false;
    bool ok = false;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&src, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: src init failed\n");
        goto cleanup;
    }
    src_owned = true;
    if (sim_field_init(&dst, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: dst init failed\n");
        goto cleanup;
    }
    dst_owned = true;

    {
        SimComplexDouble *src_data = sim_field_complex_data(&src);
        if (src_data == NULL) {
            fprintf(stderr, "[FAIL] complex kernel: src data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            src_data[i].re = (double)i;
            src_data[i].im = -(double)(i + 1U);
        }
    }

    if (sim_context_add_field(&context, &src, &src_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: add src failed\n");
        goto cleanup;
    }
    src_owned = false;
    if (sim_context_add_field(&context, &dst, &dst_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: add dst failed\n");
        goto cleanup;
    }
    dst_owned = false;

    builder = sim_context_ir_builder(&context);
    if (builder == NULL) {
        fprintf(stderr, "[FAIL] complex kernel: builder missing\n");
        goto cleanup;
    }

    SimIRNodeId src_node = sim_ir_builder_field_ref_typed(builder, 0U, sim_ir_type_complex());
    SimIRNodeId sum_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, src_node, src_node);
    if (src_node == SIM_IR_INVALID_NODE || sum_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] complex kernel: ir node build failed\n");
        goto cleanup;
    }

    bindings[0].ir_field_index = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index = 1U;
    bindings[1].context_field_index = dst_index;
    output.ir_field_index = 1U;
    output.expression = sum_node;

    kernel_desc.builder = builder;
    kernel_desc.bindings = bindings;
    kernel_desc.binding_count = 2U;
    kernel_desc.outputs = &output;
    kernel_desc.output_count = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name = "scalar_domain_complex_kernel";
    descriptor.kernel = &kernel_desc;
    if (sim_context_register_operator(&context, &descriptor, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: operator registration failed\n");
        goto cleanup;
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: backend init failed\n");
        goto cleanup;
    }
    backend_ready = true;
    sim_context_set_backend(&context, &backend);

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex kernel: execute failed\n");
        goto cleanup;
    }

    {
        const SimField *dst_field = sim_context_field(&context, dst_index);
        const SimComplexDouble *dst_data =
            (dst_field != NULL) ? sim_field_complex_data_const(dst_field) : NULL;
        if (dst_data == NULL) {
            fprintf(stderr, "[FAIL] complex kernel: dst data missing\n");
            goto cleanup;
        }
        if (!sim_scalar_domain_equal(sim_scalar_domain_from_field(dst_field),
                                     sim_scalar_domain_c64())) {
            fprintf(stderr, "[FAIL] complex kernel: dst scalar domain should be c64\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            double expected_re = (double)(2U * i);
            double expected_im = -2.0 * (double)(i + 1U);
            if (fabs(dst_data[i].re - expected_re) > 1.0e-12 ||
                fabs(dst_data[i].im - expected_im) > 1.0e-12) {
                fprintf(stderr, "[FAIL] complex kernel: dst[%zu]=(%g,%g) expected (%g,%g)\n", i,
                        dst_data[i].re, dst_data[i].im, expected_re, expected_im);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    if (src_owned) {
        sim_field_destroy(&src);
    }
    if (dst_owned) {
        sim_field_destroy(&dst);
    }
    if (backend_ready) {
        backend_destroy(&backend);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_mixed_promotion_kernel_case(void) {
    SimContext context = {0};
    SimBackend backend = {0};
    SimField src = {0};
    SimField dst = {0};
    SimIRBuilder *builder = NULL;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor output;
    SimOperatorKernelDescriptor kernel_desc = {0};
    SimOperatorDescriptor descriptor = {0};
    size_t shape[1] = {4U};
    size_t src_index = 0U;
    size_t dst_index = 0U;
    bool context_ready = false;
    bool backend_ready = false;
    bool src_owned = false;
    bool dst_owned = false;
    bool ok = false;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: src init failed\n");
        goto cleanup;
    }
    src_owned = true;
    if (sim_field_init(&dst, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: dst init failed\n");
        goto cleanup;
    }
    dst_owned = true;

    {
        double *src_data = sim_field_real_data(&src);
        if (src_data == NULL) {
            fprintf(stderr, "[FAIL] mixed kernel: src data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            src_data[i] = 10.0 + (double)i;
        }
    }

    if (sim_context_add_field(&context, &src, &src_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: add src failed\n");
        goto cleanup;
    }
    src_owned = false;
    if (sim_context_add_field(&context, &dst, &dst_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: add dst failed\n");
        goto cleanup;
    }
    dst_owned = false;

    builder = sim_context_ir_builder(&context);
    if (builder == NULL) {
        fprintf(stderr, "[FAIL] mixed kernel: builder missing\n");
        goto cleanup;
    }

    SimIRNodeId src_node = sim_ir_builder_field_ref(builder, 0U);
    SimIRNodeId zero_node = sim_ir_builder_constant(builder, 0.0);
    SimIRNodeId packed = sim_ir_builder_complex_pack(builder, src_node, zero_node);
    SimIRNodeId bias = sim_ir_builder_constant_complex(builder, 1.0, 2.0);
    SimIRNodeId out_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, packed, bias);
    if (src_node == SIM_IR_INVALID_NODE || zero_node == SIM_IR_INVALID_NODE ||
        packed == SIM_IR_INVALID_NODE || bias == SIM_IR_INVALID_NODE ||
        out_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] mixed kernel: ir node build failed\n");
        goto cleanup;
    }

    bindings[0].ir_field_index = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index = 1U;
    bindings[1].context_field_index = dst_index;
    output.ir_field_index = 1U;
    output.expression = out_node;

    kernel_desc.builder = builder;
    kernel_desc.bindings = bindings;
    kernel_desc.binding_count = 2U;
    kernel_desc.outputs = &output;
    kernel_desc.output_count = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name = "scalar_domain_mixed_kernel";
    descriptor.kernel = &kernel_desc;
    if (sim_context_register_operator(&context, &descriptor, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: operator registration failed\n");
        goto cleanup;
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: backend init failed\n");
        goto cleanup;
    }
    backend_ready = true;
    sim_context_set_backend(&context, &backend);

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mixed kernel: execute failed\n");
        goto cleanup;
    }

    {
        const SimField *dst_field = sim_context_field(&context, dst_index);
        const SimComplexDouble *dst_data =
            (dst_field != NULL) ? sim_field_complex_data_const(dst_field) : NULL;
        if (dst_data == NULL) {
            fprintf(stderr, "[FAIL] mixed kernel: dst data missing\n");
            goto cleanup;
        }
        if (!sim_scalar_domain_equal(sim_scalar_domain_from_field(dst_field),
                                     sim_scalar_domain_c64())) {
            fprintf(stderr, "[FAIL] mixed kernel: dst scalar domain should be c64\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            double expected_re = 11.0 + (double)i;
            double expected_im = 2.0;
            if (fabs(dst_data[i].re - expected_re) > 1.0e-12 ||
                fabs(dst_data[i].im - expected_im) > 1.0e-12) {
                fprintf(stderr, "[FAIL] mixed kernel: dst[%zu]=(%g,%g) expected (%g,%g)\n", i,
                        dst_data[i].re, dst_data[i].im, expected_re, expected_im);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    if (src_owned) {
        sim_field_destroy(&src);
    }
    if (dst_owned) {
        sim_field_destroy(&dst);
    }
    if (backend_ready) {
        backend_destroy(&backend);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_representation_invariant_case(void) {
    SimField real_field = {0};
    SimField complex_field = {0};
    size_t shape[1] = {2U};
    SimResult rc;
    bool ok = false;

    rc = sim_field_init(&real_field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation invariants: real field init failed\n");
        return false;
    }

    rc = sim_field_init(&complex_field, 1U, shape, sizeof(SimComplexDouble),
                        SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] representation invariants: complex field init failed\n");
        sim_field_destroy(&real_field);
        return false;
    }

    if (sim_field_validate_representation(
            &real_field, (SimFieldRepresentation){.domain = SIM_FIELD_DOMAIN_SPECTRAL,
                                                  .value_kind = SIM_FIELD_VALUE_REAL_SCALAR}) !=
        SIM_RESULT_TYPE_MISMATCH) {
        fprintf(stderr, "[FAIL] representation invariants: spectral+real should be rejected\n");
        goto cleanup;
    }

    if (sim_field_validate_representation(
            &real_field, (SimFieldRepresentation){.domain = SIM_FIELD_DOMAIN_PHYSICAL,
                                                  .value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR}) !=
        SIM_RESULT_TYPE_MISMATCH) {
        fprintf(stderr,
                "[FAIL] representation invariants: real storage cannot accept complex repr\n");
        goto cleanup;
    }

    if (sim_field_validate_representation(
            &complex_field,
            (SimFieldRepresentation){.domain = SIM_FIELD_DOMAIN_PHYSICAL,
                                     .value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT}) !=
        SIM_RESULT_OK) {
        fprintf(stderr,
                "[FAIL] representation invariants: physical imag-zero complex should validate\n");
        goto cleanup;
    }

    if (sim_field_validate_representation(
            &complex_field,
            (SimFieldRepresentation){.domain = SIM_FIELD_DOMAIN_PHYSICAL,
                                     .value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT}) !=
        SIM_RESULT_INVALID_ARGUMENT) {
        fprintf(stderr,
                "[FAIL] representation invariants: physical real-constraint should be rejected\n");
        goto cleanup;
    }

    if (sim_field_validate_representation(
            &complex_field,
            (SimFieldRepresentation){.domain = SIM_FIELD_DOMAIN_SPECTRAL,
                                     .value_kind = SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT}) !=
        SIM_RESULT_OK) {
        fprintf(stderr,
                "[FAIL] representation invariants: spectral constrained complex should validate\n");
        goto cleanup;
    }

    if (sim_field_validate_representation(
            &complex_field, (SimFieldRepresentation){.domain = SIM_FIELD_DOMAIN_SPECTRAL,
                                                     .value_kind = SIM_FIELD_VALUE_UNKNOWN}) !=
        SIM_RESULT_INVALID_ARGUMENT) {
        fprintf(stderr,
                "[FAIL] representation invariants: unknown value kind should be rejected\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_field_destroy(&real_field);
    sim_field_destroy(&complex_field);
    return ok;
}

static bool run_scalar_domain_mixer_promotion_case(void) {
    SimContext context = {0};
    SimField lhs = {0};
    SimField rhs = {0};
    SimField out = {0};
    size_t shape[1] = {3U};
    size_t lhs_index = 0U;
    size_t rhs_index = 0U;
    size_t out_index = 0U;
    bool context_ready = false;
    bool lhs_owned = false;
    bool rhs_owned = false;
    bool out_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mixer: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&lhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&rhs, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&out, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mixer: field init failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = true;

    if (sim_context_add_field(&context, &lhs, &lhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &rhs, &rhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &out, &out_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mixer: add field failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = false;

    SimMixerOperatorConfig mix_cfg = {0};
    mix_cfg.lhs_field = lhs_index;
    mix_cfg.rhs_field = rhs_index;
    mix_cfg.output_field = out_index;
    mix_cfg.lhs_gain = 1.0;
    mix_cfg.rhs_gain = 1.0;
    mix_cfg.mode = SIM_MIXER_MODE_SUM;
    mix_cfg.accumulate = false;
    mix_cfg.scale_by_dt = false;

    rc = sim_add_mixer_operator(&context, &mix_cfg, NULL);
    if (rc != SIM_RESULT_TYPE_MISMATCH) {
        fprintf(stderr,
                "[FAIL] promotion matrix/mixer: expected TYPE_MISMATCH before explicit promotion "
                "(got %d)\n",
                (int)rc);
        goto cleanup;
    }

    {
        SimField *lhs_field = sim_context_field(&context, lhs_index);
        SimField *out_field = sim_context_field(&context, out_index);
        if (lhs_field == NULL || out_field == NULL) {
            fprintf(stderr, "[FAIL] promotion matrix/mixer: field lookup failed\n");
            goto cleanup;
        }
        if (sim_field_require_complex(lhs_field) != SIM_RESULT_OK ||
            sim_field_require_complex(out_field) != SIM_RESULT_OK) {
            fprintf(stderr,
                    "[FAIL] promotion matrix/mixer: explicit promotion to complex failed\n");
            goto cleanup;
        }
    }

    rc = sim_add_mixer_operator(&context, &mix_cfg, NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mixer: add after explicit promotion failed (%d)\n",
                (int)rc);
        goto cleanup;
    }

    ok = true;

cleanup:
    if (lhs_owned) {
        sim_field_destroy(&lhs);
    }
    if (rhs_owned) {
        sim_field_destroy(&rhs);
    }
    if (out_owned) {
        sim_field_destroy(&out);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_sieve_promotion_case(void) {
    SimContext context = {0};
    SimField input = {0};
    SimField output = {0};
    size_t shape[1] = {5U};
    size_t input_index = 0U;
    size_t output_index = 0U;
    bool context_ready = false;
    bool input_owned = false;
    bool output_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/sieve: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/sieve: field init failed\n");
        goto cleanup;
    }
    input_owned = output_owned = true;

    {
        double *input_data = sim_field_real_data(&input);
        if (input_data == NULL) {
            fprintf(stderr, "[FAIL] promotion matrix/sieve: input data missing\n");
            goto cleanup;
        }
        input_data[0] = 0.0;
        input_data[1] = 1.0;
        input_data[2] = 2.0;
        input_data[3] = 1.0;
        input_data[4] = 0.0;
    }

    if (sim_context_add_field(&context, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/sieve: add field failed\n");
        goto cleanup;
    }
    input_owned = output_owned = false;

    SimSieveOperatorConfig cfg = {0};
    cfg.input_field = input_index;
    cfg.output_field = output_index;
    cfg.taps = 3U;
    cfg.sigma = 1.0;
    cfg.sigma2 = 2.0;
    cfg.poly_order = 2U;
    cfg.gain = 1.0;
    cfg.mode = SIM_SIEVE_MODE_LOW_PASS;
    cfg.accumulate = false;
    cfg.scale_by_dt = false;
    rc = sim_add_sieve_operator(&context, &cfg, NULL);
    if (rc != SIM_RESULT_TYPE_MISMATCH) {
        fprintf(
            stderr,
            "[FAIL] promotion matrix/sieve: expected TYPE_MISMATCH for mixed storage (got %d)\n",
            (int)rc);
        goto cleanup;
    }

    {
        const SimField *in_field = sim_context_field(&context, input_index);
        const SimField *out_field = sim_context_field(&context, output_index);
        if (in_field == NULL || out_field == NULL) {
            fprintf(stderr, "[FAIL] promotion matrix/sieve: field lookup failed\n");
            goto cleanup;
        }
        if (sim_field_storage_is_complex(in_field) || !sim_field_storage_is_complex(out_field)) {
            fprintf(
                stderr,
                "[FAIL] promotion matrix/sieve: mixed storage precondition changed unexpectedly\n");
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (input_owned) {
        sim_field_destroy(&input);
    }
    if (output_owned) {
        sim_field_destroy(&output);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_mask_promotion_case(void) {
    SimContext context = {0};
    SimField input = {0};
    SimField mask = {0};
    SimField output = {0};
    size_t shape[1] = {4U};
    size_t input_index = 0U;
    size_t mask_index = 0U;
    size_t output_index = 0U;
    bool context_ready = false;
    bool input_owned = false;
    bool mask_owned = false;
    bool output_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mask: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&mask, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mask: field init failed\n");
        goto cleanup;
    }
    input_owned = mask_owned = output_owned = true;

    {
        double *input_data = sim_field_real_data(&input);
        double *mask_data = sim_field_real_data(&mask);
        if (input_data == NULL || mask_data == NULL) {
            fprintf(stderr, "[FAIL] promotion matrix/mask: field data missing\n");
            goto cleanup;
        }
        input_data[0] = 1.0;
        input_data[1] = 2.0;
        input_data[2] = 3.0;
        input_data[3] = 4.0;
        mask_data[0] = 1.0;
        mask_data[1] = 0.0;
        mask_data[2] = 1.0;
        mask_data[3] = 0.0;
    }

    if (sim_context_add_field(&context, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &mask, &mask_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] promotion matrix/mask: add field failed\n");
        goto cleanup;
    }
    input_owned = mask_owned = output_owned = false;

    SimMaskOperatorConfig cfg = {0};
    cfg.input_field = input_index;
    cfg.mask_field = mask_index;
    cfg.output_field = output_index;
    cfg.mode = SIM_MASK_MODE_APPLY;
    cfg.threshold = 0.5;
    cfg.feather = 0.0;
    cfg.fill_value = -5.0;
    cfg.fill_value_im = 2.0;
    cfg.accumulate = false;
    cfg.scale_by_dt = false;
    rc = sim_add_mask_operator(&context, &cfg, NULL);
    if (rc != SIM_RESULT_TYPE_MISMATCH) {
        fprintf(stderr,
                "[FAIL] promotion matrix/mask: expected TYPE_MISMATCH for mixed storage (got %d)\n",
                (int)rc);
        goto cleanup;
    }

    {
        const SimField *in_field = sim_context_field(&context, input_index);
        const SimField *out_field = sim_context_field(&context, output_index);
        if (in_field == NULL || out_field == NULL) {
            fprintf(stderr, "[FAIL] promotion matrix/mask: field lookup failed\n");
            goto cleanup;
        }
        if (sim_field_storage_is_complex(in_field) || !sim_field_storage_is_complex(out_field)) {
            fprintf(
                stderr,
                "[FAIL] promotion matrix/mask: mixed storage precondition changed unexpectedly\n");
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (input_owned) {
        sim_field_destroy(&input);
    }
    if (mask_owned) {
        sim_field_destroy(&mask);
    }
    if (output_owned) {
        sim_field_destroy(&output);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_mask_integer_constraint_case(void) {
    SimContext context = {0};
    SimField input = {0};
    SimField mask = {0};
    SimField output = {0};
    size_t shape[1] = {4U};
    size_t input_index = 0U;
    size_t mask_index = 0U;
    size_t output_index = 0U;
    bool context_ready = false;
    bool input_owned = false;
    bool mask_owned = false;
    bool output_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask constraints: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init_typed(&input, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init(&mask, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init_typed(&output, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask constraints: field init failed\n");
        goto cleanup;
    }
    input_owned = mask_owned = output_owned = true;

    if (sim_context_add_field(&context, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &mask, &mask_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask constraints: add field failed\n");
        goto cleanup;
    }
    input_owned = mask_owned = output_owned = false;

    SimMaskOperatorConfig cfg = {0};
    cfg.input_field = input_index;
    cfg.mask_field = mask_index;
    cfg.output_field = output_index;
    cfg.mode = SIM_MASK_MODE_APPLY;
    cfg.threshold = 0.5;
    cfg.feather = 0.25;
    cfg.fill_value = 0.0;
    cfg.fill_value_im = 0.0;
    cfg.accumulate = false;
    cfg.scale_by_dt = false;
    rc = sim_add_mask_operator(&context, &cfg, NULL);
    if (rc != SIM_RESULT_TYPE_MISMATCH) {
        fprintf(stderr,
                "[FAIL] integer mask constraints: expected TYPE_MISMATCH for feathered integer "
                "mask (got %d)\n",
                (int)rc);
        goto cleanup;
    }

    ok = true;

cleanup:
    if (input_owned) {
        sim_field_destroy(&input);
    }
    if (mask_owned) {
        sim_field_destroy(&mask);
    }
    if (output_owned) {
        sim_field_destroy(&output);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_segmented_sieve_mark_constraint_case(void) {
    SimContext context = {0};
    SimField candidate = {0};
    SimField flags = {0};
    size_t shape[1] = {4U};
    size_t candidate_index = 0U;
    size_t flags_index = 0U;
    bool context_ready = false;
    bool candidate_owned = false;
    bool flags_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark constraints: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&candidate, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init_typed(&flags, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark constraints: field init failed\n");
        goto cleanup;
    }
    candidate_owned = true;
    flags_owned = true;

    if (sim_context_add_field(&context, &candidate, &candidate_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &flags, &flags_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark constraints: add field failed\n");
        goto cleanup;
    }
    candidate_owned = false;
    flags_owned = false;

    {
        SimSegmentedSieveMarkOperatorConfig cfg = {0};
        cfg.candidate_field = candidate_index;
        cfg.flags_field = flags_index;
        cfg.prime = 2U;
        rc = sim_add_segmented_sieve_mark_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] segmented sieve mark constraints: expected TYPE_MISMATCH for "
                    "non-integer candidate (got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (candidate_owned) {
        sim_field_destroy(&candidate);
    }
    if (flags_owned) {
        sim_field_destroy(&flags);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_segmented_sieve_mark_batch_constraint_case(void) {
    SimContext context = {0};
    SimField candidate = {0};
    SimField primes = {0};
    SimField flags = {0};
    size_t candidate_shape[1] = {4U};
    size_t primes_shape[1] = {2U};
    size_t candidate_index = 0U;
    size_t primes_index = 0U;
    size_t flags_index = 0U;
    bool context_ready = false;
    bool candidate_owned = false;
    bool primes_owned = false;
    bool flags_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark batch constraints: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&candidate, 1U, candidate_shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&primes, 1U, primes_shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&flags, 1U, candidate_shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark batch constraints: field init failed\n");
        goto cleanup;
    }
    candidate_owned = true;
    primes_owned = true;
    flags_owned = true;

    {
        uint64_t *prime_data = sim_field_u64_data(&primes);
        if (prime_data == NULL) {
            fprintf(stderr, "[FAIL] segmented sieve mark batch constraints: prime data missing\n");
            goto cleanup;
        }
        prime_data[0] = 2U;
        prime_data[1] = 3U;
    }

    if (sim_context_add_field(&context, &candidate, &candidate_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &primes, &primes_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &flags, &flags_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark batch constraints: add field failed\n");
        goto cleanup;
    }
    candidate_owned = false;
    primes_owned = false;
    flags_owned = false;

    {
        SimSegmentedSieveMarkBatchOperatorConfig cfg = {0};
        cfg.candidate_field = candidate_index;
        cfg.primes_field = primes_index;
        cfg.flags_field = flags_index;
        rc = sim_add_segmented_sieve_mark_batch_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] segmented sieve mark batch constraints: expected TYPE_MISMATCH for "
                    "non-integer candidate (got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (candidate_owned) {
        sim_field_destroy(&candidate);
    }
    if (primes_owned) {
        sim_field_destroy(&primes);
    }
    if (flags_owned) {
        sim_field_destroy(&flags);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_operator_promotion_matrix_case(void) {
    if (!run_scalar_domain_mixer_promotion_case()) {
        return false;
    }
    if (!run_scalar_domain_sieve_promotion_case()) {
        return false;
    }
    if (!run_scalar_domain_mask_promotion_case()) {
        return false;
    }
    if (!run_scalar_domain_mask_integer_constraint_case()) {
        return false;
    }
    if (!run_scalar_domain_segmented_sieve_mark_constraint_case()) {
        return false;
    }
    if (!run_scalar_domain_segmented_sieve_mark_batch_constraint_case()) {
        return false;
    }
    return true;
}

static bool run_scalar_domain_unsupported_discrete_operator_case(void) {
    SimContext context = {0};
    SimField lhs = {0};
    SimField rhs = {0};
    SimField out = {0};
    size_t shape[1] = {5U};
    size_t lhs_index = 0U;
    size_t rhs_index = 0U;
    size_t out_index = 0U;
    bool context_ready = false;
    bool lhs_owned = false;
    bool rhs_owned = false;
    bool out_owned = false;
    bool ok = false;
    SimResult rc;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer discrete operators: context init failed\n");
        return false;
    }
    context_ready = true;

    rc = sim_field_init_typed(&lhs, 1U, shape, sim_scalar_domain_u64(), SIM_FIELD_STORAGE_ROW_MAJOR,
                              NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer discrete operators: lhs init failed\n");
        goto cleanup;
    }
    lhs_owned = true;

    rc = sim_field_init_typed(&rhs, 1U, shape, sim_scalar_domain_u64(), SIM_FIELD_STORAGE_ROW_MAJOR,
                              NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer discrete operators: rhs init failed\n");
        goto cleanup;
    }
    rhs_owned = true;

    rc = sim_field_init_typed(&out, 1U, shape, sim_scalar_domain_u64(), SIM_FIELD_STORAGE_ROW_MAJOR,
                              NULL);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer discrete operators: out init failed\n");
        goto cleanup;
    }
    out_owned = true;

    if (sim_context_add_field(&context, &lhs, &lhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &rhs, &rhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &out, &out_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer discrete operators: add field failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = false;

    {
        SimCopyOperatorConfig cfg = {0};
        cfg.input_field = lhs_index;
        cfg.output_field = out_index;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_copy_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] integer discrete operators: copy expected OK (got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    {
        SimScaleOperatorConfig cfg = {0};
        cfg.input_field = lhs_index;
        cfg.output_field = out_index;
        cfg.scale = 2.0;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_scale_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] integer discrete operators: scale expected TYPE_MISMATCH (got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    {
        SimCoordinateOperatorConfig cfg = {0};
        cfg.output_field = out_index;
        cfg.mode = SIM_COORD_MODE_INDEX;
        cfg.normalize = SIM_COORD_NORMALIZE_UNIT;
        cfg.gain = 1.0;
        cfg.bias = 0.0;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_coordinate_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(
                stderr,
                "[FAIL] integer discrete operators: coordinate expected TYPE_MISMATCH (got %d)\n",
                (int)rc);
            goto cleanup;
        }
    }

    {
        SimMixerOperatorConfig cfg = {0};
        cfg.lhs_field = lhs_index;
        cfg.rhs_field = rhs_index;
        cfg.output_field = out_index;
        cfg.mode = SIM_MIXER_MODE_LINEAR;
        cfg.lhs_gain = 1.0;
        cfg.rhs_gain = 1.0;
        cfg.mix = 0.5;
        cfg.bias = 0.0;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_mixer_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] integer discrete operators: mixer expected TYPE_MISMATCH (got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    {
        SimElementwiseMathOperatorConfig cfg = {0};
        cfg.lhs_field = lhs_index;
        cfg.rhs_field = rhs_index;
        cfg.output_field = out_index;
        cfg.mode = SIM_ELEMENTWISE_MATH_FRACT;
        cfg.rhs_source = SIM_ELEMENTWISE_MATH_RHS_FIELD;
        cfg.rhs_constant = 2.0;
        cfg.epsilon = 1.0e-6;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_elementwise_math_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] integer discrete operators: elementwise_math expected TYPE_MISMATCH "
                    "(got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    {
        SimRemainderOperatorConfig cfg = {0};
        cfg.warped_field = lhs_index;
        cfg.reference_field = rhs_index;
        cfg.output_field = out_index;
        cfg.weight = 1.0;
        cfg.bias = 0.0;
        cfg.exponent = 1.0;
        cfg.epsilon = 1.0e-6;
        cfg.nonlinearity = SIM_REMAINDER_NONLINEARITY_IDENTITY;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_remainder_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(
                stderr,
                "[FAIL] integer discrete operators: remainder expected TYPE_MISMATCH (got %d)\n",
                (int)rc);
            goto cleanup;
        }
    }

    {
        SimSieveOperatorConfig cfg = {0};
        cfg.input_field = lhs_index;
        cfg.output_field = out_index;
        cfg.taps = 3U;
        cfg.sigma = 1.0;
        cfg.sigma2 = 2.0;
        cfg.poly_order = 2U;
        cfg.gain = 1.0;
        cfg.mode = SIM_SIEVE_MODE_LOW_PASS;
        cfg.accumulate = false;
        cfg.scale_by_dt = false;
        rc = sim_add_sieve_operator(&context, &cfg, NULL);
        if (rc != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] integer discrete operators: sieve expected TYPE_MISMATCH (got %d)\n",
                    (int)rc);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    if (lhs_owned) {
        sim_field_destroy(&lhs);
    }
    if (rhs_owned) {
        sim_field_destroy(&rhs);
    }
    if (out_owned) {
        sim_field_destroy(&out);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_exact_integer_constant_case(void) {
    SimIRBuilder builder = {0};
    SimIRType int_type = sim_ir_type_scalar_domain_typed(sim_scalar_domain_i64());
    SimIRNodeId constant_node;
    SimIRDomainValue value = {0};
    const int64_t exact_value = ((int64_t)1 << 53) + 3;

    if (sim_ir_builder_init(&builder) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer constant: builder init failed\n");
        return false;
    }

    constant_node = sim_ir_builder_constant_i64_typed(&builder, exact_value, int_type);
    if (constant_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] integer constant: exact i64 literal failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }

    if (sim_ir_evaluate_domain(&builder, constant_node, NULL, &value) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer constant: domain evaluate failed\n");
        sim_ir_builder_destroy(&builder);
        return false;
    }
    if (!sim_scalar_domain_equal(value.domain, sim_scalar_domain_i64()) ||
        value.value.as_i64 != exact_value) {
        fprintf(stderr, "[FAIL] integer constant: expected %" PRId64 " got %" PRId64 "\n",
                exact_value, value.value.as_i64);
        sim_ir_builder_destroy(&builder);
        return false;
    }

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool run_scalar_domain_integer_kernel_case(void) {
    SimContext context = {0};
    SimBackend backend = {0};
    SimField lhs = {0};
    SimField rhs = {0};
    SimField out = {0};
    SimIRBuilder *builder = NULL;
    SimOperatorKernelBindingDescriptor bindings[3];
    SimOperatorKernelOutputDescriptor output;
    SimOperatorKernelDescriptor kernel_desc = {0};
    SimOperatorDescriptor descriptor = {0};
    SimIRType int_type = sim_ir_type_scalar_domain_typed(sim_scalar_domain_i32());
    size_t shape[1] = {4U};
    size_t lhs_index = 0U;
    size_t rhs_index = 0U;
    size_t out_index = 0U;
    bool context_ready = false;
    bool backend_ready = false;
    bool lhs_owned = false;
    bool rhs_owned = false;
    bool out_owned = false;
    bool ok = false;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&lhs, 1U, shape, sizeof(int32_t), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&rhs, 1U, shape, sizeof(int32_t), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&out, 1U, shape, sizeof(int32_t), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: field init failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = true;

    if (sim_field_set_scalar_domain(&lhs, sim_scalar_domain_i32()) != SIM_RESULT_OK ||
        sim_field_set_scalar_domain(&rhs, sim_scalar_domain_i32()) != SIM_RESULT_OK ||
        sim_field_set_scalar_domain(&out, sim_scalar_domain_i32()) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: field domain setup failed\n");
        goto cleanup;
    }

    {
        int32_t *lhs_data = sim_field_i32_data(&lhs);
        int32_t *rhs_data = sim_field_i32_data(&rhs);
        if (lhs_data == NULL || rhs_data == NULL) {
            fprintf(stderr, "[FAIL] integer kernel: typed field access failed\n");
            goto cleanup;
        }
        lhs_data[0] = 2;
        lhs_data[1] = -3;
        lhs_data[2] = 5;
        lhs_data[3] = -7;
        rhs_data[0] = 4;
        rhs_data[1] = 6;
        rhs_data[2] = -2;
        rhs_data[3] = 3;
    }

    if (sim_context_add_field(&context, &lhs, &lhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &rhs, &rhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &out, &out_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: add field failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = false;

    builder = sim_context_ir_builder(&context);
    if (builder == NULL) {
        fprintf(stderr, "[FAIL] integer kernel: builder missing\n");
        goto cleanup;
    }

    SimIRNodeId lhs_node = sim_ir_builder_field_ref_typed(builder, 0U, int_type);
    SimIRNodeId rhs_node = sim_ir_builder_field_ref_typed(builder, 1U, int_type);
    SimIRNodeId mul_node = sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, lhs_node, rhs_node);
    SimIRNodeId sum_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, mul_node, rhs_node);
    if (lhs_node == SIM_IR_INVALID_NODE || rhs_node == SIM_IR_INVALID_NODE ||
        mul_node == SIM_IR_INVALID_NODE || sum_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] integer kernel: ir node build failed\n");
        goto cleanup;
    }

    bindings[0].ir_field_index = 0U;
    bindings[0].context_field_index = lhs_index;
    bindings[1].ir_field_index = 1U;
    bindings[1].context_field_index = rhs_index;
    bindings[2].ir_field_index = 2U;
    bindings[2].context_field_index = out_index;
    output.ir_field_index = 2U;
    output.expression = sum_node;

    kernel_desc.builder = builder;
    kernel_desc.bindings = bindings;
    kernel_desc.binding_count = 3U;
    kernel_desc.outputs = &output;
    kernel_desc.output_count = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name = "scalar_domain_integer_kernel";
    descriptor.kernel = &kernel_desc;
    if (sim_context_register_operator(&context, &descriptor, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: operator registration failed\n");
        goto cleanup;
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: backend init failed\n");
        goto cleanup;
    }
    backend_ready = true;
    sim_context_set_backend(&context, &backend);

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer kernel: execute failed\n");
        goto cleanup;
    }

    {
        const SimField *out_field = sim_context_field(&context, out_index);
        const int32_t *out_data = (out_field != NULL) ? sim_field_i32_data_const(out_field) : NULL;
        const int32_t expected[] = {12, -12, -12, -18};
        if (out_data == NULL) {
            fprintf(stderr, "[FAIL] integer kernel: output data missing\n");
            goto cleanup;
        }
        if (!sim_scalar_domain_equal(sim_scalar_domain_from_field(out_field),
                                     sim_scalar_domain_i32())) {
            fprintf(stderr, "[FAIL] integer kernel: output scalar domain mismatch\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (out_data[i] != expected[i]) {
                fprintf(stderr, "[FAIL] integer kernel: out[%zu]=%d expected %d\n", i, out_data[i],
                        expected[i]);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    if (lhs_owned) {
        sim_field_destroy(&lhs);
    }
    if (rhs_owned) {
        sim_field_destroy(&rhs);
    }
    if (out_owned) {
        sim_field_destroy(&out);
    }
    if (backend_ready) {
        backend_destroy(&backend);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_integer_mod_kernel_case(void) {
    SimContext context = {0};
    SimBackend backend = {0};
    SimField lhs = {0};
    SimField rhs = {0};
    SimField out = {0};
    SimIRBuilder *builder = NULL;
    SimOperatorKernelBindingDescriptor bindings[3];
    SimOperatorKernelOutputDescriptor output;
    SimOperatorKernelDescriptor kernel_desc = {0};
    SimOperatorDescriptor descriptor = {0};
    SimIRType int_type = sim_ir_type_scalar_domain_typed(sim_scalar_domain_i32());
    size_t shape[1] = {4U};
    size_t lhs_index = 0U;
    size_t rhs_index = 0U;
    size_t out_index = 0U;
    bool context_ready = false;
    bool backend_ready = false;
    bool lhs_owned = false;
    bool rhs_owned = false;
    bool out_owned = false;
    bool ok = false;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: context init failed\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&lhs, 1U, shape, sizeof(int32_t), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&rhs, 1U, shape, sizeof(int32_t), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&out, 1U, shape, sizeof(int32_t), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: field init failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = true;

    if (sim_field_set_scalar_domain(&lhs, sim_scalar_domain_i32()) != SIM_RESULT_OK ||
        sim_field_set_scalar_domain(&rhs, sim_scalar_domain_i32()) != SIM_RESULT_OK ||
        sim_field_set_scalar_domain(&out, sim_scalar_domain_i32()) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: field domain setup failed\n");
        goto cleanup;
    }

    {
        int32_t *lhs_data = sim_field_i32_data(&lhs);
        int32_t *rhs_data = sim_field_i32_data(&rhs);
        if (lhs_data == NULL || rhs_data == NULL) {
            fprintf(stderr, "[FAIL] integer mod kernel: typed field access failed\n");
            goto cleanup;
        }
        lhs_data[0] = -7;
        lhs_data[1] = -5;
        lhs_data[2] = 7;
        lhs_data[3] = 5;
        rhs_data[0] = 3;
        rhs_data[1] = 3;
        rhs_data[2] = 3;
        rhs_data[3] = 3;
    }

    if (sim_context_add_field(&context, &lhs, &lhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &rhs, &rhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &out, &out_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: add field failed\n");
        goto cleanup;
    }
    lhs_owned = rhs_owned = out_owned = false;

    builder = sim_context_ir_builder(&context);
    if (builder == NULL) {
        fprintf(stderr, "[FAIL] integer mod kernel: builder missing\n");
        goto cleanup;
    }

    SimIRNodeId lhs_node = sim_ir_builder_field_ref_typed(builder, 0U, int_type);
    SimIRNodeId rhs_node = sim_ir_builder_field_ref_typed(builder, 1U, int_type);
    SimIRNodeId mod_node = sim_ir_builder_mod(builder, lhs_node, rhs_node);
    if (lhs_node == SIM_IR_INVALID_NODE || rhs_node == SIM_IR_INVALID_NODE ||
        mod_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "[FAIL] integer mod kernel: ir node build failed\n");
        goto cleanup;
    }

    bindings[0].ir_field_index = 0U;
    bindings[0].context_field_index = lhs_index;
    bindings[1].ir_field_index = 1U;
    bindings[1].context_field_index = rhs_index;
    bindings[2].ir_field_index = 2U;
    bindings[2].context_field_index = out_index;
    output.ir_field_index = 2U;
    output.expression = mod_node;

    kernel_desc.builder = builder;
    kernel_desc.bindings = bindings;
    kernel_desc.binding_count = 3U;
    kernel_desc.outputs = &output;
    kernel_desc.output_count = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name = "scalar_domain_integer_mod_kernel";
    descriptor.kernel = &kernel_desc;
    if (sim_context_register_operator(&context, &descriptor, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: operator registration failed\n");
        goto cleanup;
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: backend init failed\n");
        goto cleanup;
    }
    backend_ready = true;
    sim_context_set_backend(&context, &backend);

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mod kernel: execute failed\n");
        goto cleanup;
    }

    {
        const SimField *out_field = sim_context_field(&context, out_index);
        const int32_t *out_data = (out_field != NULL) ? sim_field_i32_data_const(out_field) : NULL;
        const int32_t expected[] = {-1, -2, 1, 2};
        if (out_data == NULL) {
            fprintf(stderr, "[FAIL] integer mod kernel: output data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (out_data[i] != expected[i]) {
                fprintf(stderr, "[FAIL] integer mod kernel: out[%zu]=%d expected %d\n", i,
                        out_data[i], expected[i]);
                goto cleanup;
            }
        }
    }

    ok = true;

cleanup:
    if (lhs_owned) {
        sim_field_destroy(&lhs);
    }
    if (rhs_owned) {
        sim_field_destroy(&rhs);
    }
    if (out_owned) {
        sim_field_destroy(&out);
    }
    if (backend_ready) {
        backend_destroy(&backend);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    return ok;
}

static bool run_scalar_domain_typed_field_construction_case(void) {
    static const struct {
        const char *name;
        SimScalarDomain domain;
        size_t element_size;
        SimFieldDataType view_type;
    } cases[] = {
        {"i8",
         {.kind = SIM_SCALAR_DOMAIN_INTEGER, .bit_width = 8U, .is_signed = true, .modulus = 0U},
         sizeof(int8_t),
         SIM_FIELD_I8},
        {"i32",
         {.kind = SIM_SCALAR_DOMAIN_INTEGER, .bit_width = 32U, .is_signed = true, .modulus = 0U},
         sizeof(int32_t),
         SIM_FIELD_I32},
        {"i64",
         {.kind = SIM_SCALAR_DOMAIN_INTEGER, .bit_width = 64U, .is_signed = true, .modulus = 0U},
         sizeof(int64_t),
         SIM_FIELD_I64},
        {"u8",
         {.kind = SIM_SCALAR_DOMAIN_INTEGER, .bit_width = 8U, .is_signed = false, .modulus = 0U},
         sizeof(uint8_t),
         SIM_FIELD_U8},
        {"u32",
         {.kind = SIM_SCALAR_DOMAIN_INTEGER, .bit_width = 32U, .is_signed = false, .modulus = 0U},
         sizeof(uint32_t),
         SIM_FIELD_U32},
        {"u64",
         {.kind = SIM_SCALAR_DOMAIN_INTEGER, .bit_width = 64U, .is_signed = false, .modulus = 0U},
         sizeof(uint64_t),
         SIM_FIELD_U64},
    };
    size_t shape[1] = {4U};

    for (size_t i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        SimField field = {0};
        SimResult result;
        SimFieldView view;

        result = sim_field_init_typed(&field, 1U, shape, cases[i].domain,
                                      SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] typed field init failed for %s\n", cases[i].name);
            return false;
        }
        if (!sim_scalar_domain_equal(sim_field_scalar_domain(&field), cases[i].domain)) {
            fprintf(stderr, "[FAIL] typed field domain mismatch for %s\n", cases[i].name);
            sim_field_destroy(&field);
            return false;
        }
        if (field.element_size != cases[i].element_size) {
            fprintf(stderr,
                    "[FAIL] typed field element size mismatch for %s: got %zu expected %zu\n",
                    cases[i].name, field.element_size, cases[i].element_size);
            sim_field_destroy(&field);
            return false;
        }
        view = sim_field_view_from_field(&field);
        if (view.type != cases[i].view_type) {
            fprintf(stderr, "[FAIL] typed field view mismatch for %s: got %d expected %d\n",
                    cases[i].name, (int)view.type, (int)cases[i].view_type);
            sim_field_destroy(&field);
            return false;
        }
        sim_field_destroy(&field);
    }

    {
        uint64_t backing[3] = {5U, 7U, 11U};
        size_t shape_wrap[1] = {3U};
        size_t strides[1] = {1U};
        SimFieldLayout layout = {
            .rank = 1U,
            .shape = shape_wrap,
            .strides = strides,
            .contiguous = true,
        };
        SimField wrapped = {0};
        SimResult result;
        SimFieldView view;

        result = sim_field_wrap_typed(&wrapped, &layout, sim_scalar_domain_u64(),
                                      SIM_FIELD_STORAGE_ROW_MAJOR, backing);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] typed field wrap failed for u64\n");
            return false;
        }
        if (sim_field_u64_data(&wrapped) != backing) {
            fprintf(stderr, "[FAIL] typed field wrap did not preserve u64 backing pointer\n");
            sim_field_destroy(&wrapped);
            return false;
        }
        view = sim_field_view_from_field(&wrapped);
        if (view.type != SIM_FIELD_U64) {
            fprintf(stderr, "[FAIL] typed wrapped field view type mismatch\n");
            sim_field_destroy(&wrapped);
            return false;
        }
        sim_field_destroy(&wrapped);
    }

    return true;
}

int main(void) {
    if (!run_scalar_domain_validation_case()) {
        return 1;
    }
    if (!run_scalar_domain_name_case()) {
        return 1;
    }
    if (!run_scalar_domain_capability_case()) {
        return 1;
    }
    if (!run_scalar_domain_field_bridge_case()) {
        return 1;
    }
    if (!run_scalar_domain_representation_helper_case()) {
        return 1;
    }
    if (!run_scalar_domain_ir_bridge_case()) {
        return 1;
    }
    if (!run_scalar_domain_ir_legality_case()) {
        return 1;
    }
    if (!run_scalar_domain_unified_evaluator_case()) {
        return 1;
    }
    if (!run_scalar_domain_real_kernel_case()) {
        return 1;
    }
    if (!run_scalar_domain_complex_kernel_case()) {
        return 1;
    }
    if (!run_scalar_domain_mixed_promotion_kernel_case()) {
        return 1;
    }
    if (!run_scalar_domain_representation_invariant_case()) {
        return 1;
    }
    if (!run_scalar_domain_operator_promotion_matrix_case()) {
        return 1;
    }
    if (!run_scalar_domain_unsupported_discrete_operator_case()) {
        return 1;
    }
    if (!run_scalar_domain_exact_integer_constant_case()) {
        return 1;
    }
    if (!run_scalar_domain_integer_kernel_case()) {
        return 1;
    }
    if (!run_scalar_domain_integer_mod_kernel_case()) {
        return 1;
    }
    if (!run_scalar_domain_typed_field_construction_case()) {
        return 1;
    }

    printf("test_scalar_domain: ok\n");
    return 0;
}
