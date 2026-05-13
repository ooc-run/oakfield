#define SIM_TESTING 1
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "operators/common/warp_safety.h"
#include <oakfield/backend.h>
#include <oakfield/kernel_ir.h>

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "[FAIL] %s\n", msg);                                                   \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

static void test_set_env_text(const char* key, const char* value) {
    if (key == NULL || key[0] == '\0') {
        return;
    }
#if defined(_WIN32)
    (void) _putenv_s(key, (value != NULL) ? value : "");
#else
    if (value != NULL) {
        (void) setenv(key, value, 1);
    } else {
        (void) unsetenv(key);
    }
#endif
}

static void compute_central_difference(const double* src, size_t length, double dx, double* dst) {
    size_t i;

    for (i = 0; i < length; ++i) {
        double center   = src[i];
        double forward  = (i + 1U < length) ? src[i + 1U] : center;
        double backward = (i > 0U) ? src[i - 1U] : center;

        if (i > 0U && (i + 1U) < length) {
            dst[i] = (forward - backward) / (2.0 * dx);
        } else if ((i + 1U) < length) {
            dst[i] = (forward - center) / dx;
        } else if (i > 0U) {
            dst[i] = (center - backward) / dx;
        } else {
            dst[i] = 0.0;
        }
    }
}

static bool test_ir_semantics_metadata(void) {
    SimIRBuilder   builder;
    SimResult      result;
    SimIRNodeId    scalar_a;
    SimIRNodeId    scalar_b;
    SimIRNodeId    scalar_sum;
    SimIRNodeId    vec_a;
    SimIRNodeId    vec_b;
    SimIRNodeId    vec_sum;
    SimIRNodeId    diff_node;
    SimIRNodeId    noise_node;
    SimIRType      type;
    SimIRDiffSpec  diff_spec;
    SimIRNoiseSpec noise_spec;
    SimIREvaluator evaluator = { 0 };
    double         value     = 0.0;

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "IR builder init failed");

    scalar_a = sim_ir_builder_constant_typed(&builder, 2.0, sim_ir_type_scalar());
    CHECK(scalar_a != SIM_IR_INVALID_NODE, "scalar constant A failed");

    scalar_b = sim_ir_builder_constant_typed(&builder, 3.0, sim_ir_type_scalar());
    CHECK(scalar_b != SIM_IR_INVALID_NODE, "scalar constant B failed");

    scalar_sum = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, scalar_a, scalar_b);
    CHECK(scalar_sum != SIM_IR_INVALID_NODE, "scalar addition failed");

    type = sim_ir_builder_node_type(&builder, scalar_sum);
    CHECK(sim_ir_type_is_scalar(type), "scalar sum type mismatch");

    result = sim_ir_evaluate(&builder, scalar_sum, &evaluator, &value);
    CHECK(result == SIM_RESULT_OK, "scalar evaluator failed");
    CHECK(fabs(value - 5.0) < 1.0e-12, "scalar evaluator value mismatch");

    vec_a = sim_ir_builder_constant_typed(&builder, 1.0, sim_ir_type_vector(3U));
    CHECK(vec_a != SIM_IR_INVALID_NODE, "vector constant A failed");

    vec_b = sim_ir_builder_constant_typed(&builder, 2.0, sim_ir_type_vector(3U));
    CHECK(vec_b != SIM_IR_INVALID_NODE, "vector constant B failed");

    vec_sum = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, vec_a, vec_b);
    CHECK(vec_sum != SIM_IR_INVALID_NODE, "vector addition failed");

    type = sim_ir_builder_node_type(&builder, vec_sum);
    CHECK(type.kind == SIM_IR_VALUE_VECTOR && type.components == 3U,
          "vector type metadata incorrect");

    (void) sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, vec_a, scalar_a);
    CHECK(builder.count == 6U, "type mismatch should not grow builder");

    sim_ir_diff_spec_init(&diff_spec, &builder);
    diff_spec.operand       = scalar_a;
    diff_spec.dx            = 0.5;
    diff_spec.scale         = 2.0;
    diff_spec.stencil_order = 2U;
    diff_spec.boundary      = SIM_IR_BOUNDARY_DIRICHLET;
    diff_node               = sim_ir_builder_diff_spec(&builder, &diff_spec);
    CHECK(diff_node != SIM_IR_INVALID_NODE, "diff spec construction failed");

    noise_spec.seed         = 42U;
    noise_spec.amplitude    = 0.25;
    noise_spec.variance     = noise_spec.amplitude * noise_spec.amplitude;
    noise_spec.law          = SIM_IR_NOISE_LAW_STRATONOVICH;
    noise_spec.distribution = SIM_IR_NOISE_DISTRIBUTION_GAUSSIAN;
    noise_spec.value_type   = sim_ir_type_scalar();
    noise_node              = sim_ir_builder_noise_spec(&builder, &noise_spec);
    CHECK(noise_node != SIM_IR_INVALID_NODE, "noise spec construction failed");

    const SimIRNode* diff_ptr = sim_ir_builder_get(&builder, diff_node);
    CHECK(diff_ptr != NULL && diff_ptr->data.diff.boundary == SIM_IR_BOUNDARY_DIRICHLET,
          "diff boundary metadata incorrect");

    const SimIRNode* noise_ptr = sim_ir_builder_get(&builder, noise_node);
    CHECK(noise_ptr != NULL && noise_ptr->data.noise.law == SIM_IR_NOISE_LAW_STRATONOVICH,
          "noise law metadata incorrect");

    sim_ir_builder_set_default_boundary(&builder, SIM_IR_BOUNDARY_PERIODIC);
    SimIRNodeId periodic_diff = sim_ir_builder_diff(&builder, scalar_a, 0U, 1.0, 1.0);
    CHECK(periodic_diff != SIM_IR_INVALID_NODE, "default-boundary diff construction failed");
    const SimIRNode* periodic_ptr = sim_ir_builder_get(&builder, periodic_diff);
    CHECK(periodic_ptr != NULL && periodic_ptr->data.diff.boundary == SIM_IR_BOUNDARY_PERIODIC,
          "default boundary not applied to diff helper");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_mathview_render(void) {
    SimIRBuilder   builder;
    SimResult      result;
    SimIRNodeId    field_node;
    SimIRNodeId    constant_node;
    SimIRNodeId    integer_constant_node;
    SimIRNodeId    sum_node;
    SimIRNodeId    diff_node;
    SimIRNodeId    noise_node;
    SimIRNodeId    combined;
    SimIRNodeId    root;
    SimIRDiffSpec  diff_spec;
    SimIRNoiseSpec noise_spec;
    size_t         length         = 0U;
    size_t         integer_length = 0U;
    char*          canonical      = NULL;
    char*          json           = NULL;
    char*          integer_render = NULL;
    char*          latex          = NULL;
    unsigned char  hash_bytes[32];
    bool           any_nonzero = false;

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "IR builder init failed (mathview)");

    field_node = sim_ir_builder_field_ref_typed(&builder, 0U, sim_ir_type_scalar());
    CHECK(field_node != SIM_IR_INVALID_NODE, "field ref failed (mathview)");

    constant_node = sim_ir_builder_constant(&builder, 2.0);
    CHECK(constant_node != SIM_IR_INVALID_NODE, "constant failed (mathview)");

    sum_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, field_node, constant_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "sum node failed (mathview)");

    sim_ir_diff_spec_init(&diff_spec, &builder);
    diff_spec.operand       = sum_node;
    diff_spec.axis          = 0U;
    diff_spec.dx            = 0.25;
    diff_spec.scale         = 1.0;
    diff_spec.order         = 1U;
    diff_spec.stencil_order = 2U;
    diff_spec.method        = SIM_IR_DIFF_METHOD_CENTRAL;
    diff_spec.boundary      = SIM_IR_BOUNDARY_PERIODIC;
    diff_node               = sim_ir_builder_diff_spec(&builder, &diff_spec);
    CHECK(diff_node != SIM_IR_INVALID_NODE, "diff node failed (mathview)");

    memset(&noise_spec, 0, sizeof(noise_spec));
    noise_spec.seed         = 7U;
    noise_spec.amplitude    = 0.2;
    noise_spec.variance     = 0.04;
    noise_spec.law          = SIM_IR_NOISE_LAW_ITO;
    noise_spec.distribution = SIM_IR_NOISE_DISTRIBUTION_UNIFORM;
    noise_spec.value_type   = sim_ir_type_scalar();
    noise_node              = sim_ir_builder_noise_spec(&builder, &noise_spec);
    CHECK(noise_node != SIM_IR_INVALID_NODE, "noise node failed (mathview)");

    combined = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, diff_node, noise_node);
    CHECK(combined != SIM_IR_INVALID_NODE, "combined node failed (mathview)");

    root = sim_ir_builder_call(&builder, SIM_IR_CALL_SIN, combined);
    CHECK(root != SIM_IR_INVALID_NODE, "call node failed (mathview)");

    result = sim_ir_mathview_render(&builder, root, NULL, 0U, &length);
    CHECK(result == SIM_RESULT_OK && length > 0U, "mathview canonical length failed");
    canonical = (char*) calloc(length + 1U, 1U);
    CHECK(canonical != NULL, "mathview canonical allocation failed");
    result = sim_ir_mathview_render(&builder, root, canonical, length + 1U, NULL);
    CHECK(result == SIM_RESULT_OK, "mathview canonical render failed");
    CHECK(strstr(canonical, "math:v1.0") != NULL, "mathview header missing");
    CHECK(strstr(canonical, "\"diff\"") != NULL, "mathview diff missing");
    CHECK(strstr(canonical, "\"noise\"") != NULL, "mathview noise missing");

    length = 0U;
    result = sim_ir_mathview_render_json(&builder, root, NULL, 0U, &length);
    CHECK(result == SIM_RESULT_OK && length > 0U, "mathview json length failed");
    json = (char*) calloc(length + 1U, 1U);
    CHECK(json != NULL, "mathview json allocation failed");
    result = sim_ir_mathview_render_json(&builder, root, json, length + 1U, NULL);
    CHECK(result == SIM_RESULT_OK, "mathview json render failed");
    CHECK(strstr(json, "\"node\":\"Call\"") != NULL, "mathview json root missing");
    CHECK(strstr(json, "\"name\":\"diff\"") != NULL, "mathview json diff missing");
    CHECK(strstr(json, "\"name\":\"noise\"") != NULL, "mathview json noise missing");

    length = 0U;
    result = sim_ir_mathview_render_latex(&builder, root, NULL, 0U, &length);
    CHECK(result == SIM_RESULT_OK && length > 0U, "mathview latex length failed");
    latex = (char*) calloc(length + 1U, 1U);
    CHECK(latex != NULL, "mathview latex allocation failed");
    result = sim_ir_mathview_render_latex(&builder, root, latex, length + 1U, NULL);
    CHECK(result == SIM_RESULT_OK, "mathview latex render failed");
    CHECK(strstr(latex, "\\sin") != NULL, "mathview latex sin missing");
    CHECK(strstr(latex, "\\operatorname{diff}") != NULL, "mathview latex diff missing");

    integer_constant_node = sim_ir_builder_constant_i64_typed(
        &builder, 9007199254740993LL, sim_ir_type_scalar_domain_typed(sim_scalar_domain_i64()));
    CHECK(integer_constant_node != SIM_IR_INVALID_NODE, "integer constant failed (mathview)");

    result = sim_ir_mathview_render(&builder, integer_constant_node, NULL, 0U, &integer_length);
    CHECK(result == SIM_RESULT_OK && integer_length > 0U,
          "mathview integer canonical length failed");
    integer_render = (char*) realloc(integer_render, integer_length + 1U);
    CHECK(integer_render != NULL, "mathview integer canonical allocation failed");
    result = sim_ir_mathview_render(
        &builder, integer_constant_node, integer_render, integer_length + 1U, NULL);
    CHECK(result == SIM_RESULT_OK, "mathview integer canonical render failed");
    CHECK(strstr(integer_render, "9007199254740993") != NULL,
          "mathview canonical integer lost exact value");

    result =
        sim_ir_mathview_render_json(&builder, integer_constant_node, NULL, 0U, &integer_length);
    CHECK(result == SIM_RESULT_OK && integer_length > 0U, "mathview integer json length failed");
    integer_render = (char*) realloc(integer_render, integer_length + 1U);
    CHECK(integer_render != NULL, "mathview integer json allocation failed");
    result = sim_ir_mathview_render_json(
        &builder, integer_constant_node, integer_render, integer_length + 1U, NULL);
    CHECK(result == SIM_RESULT_OK, "mathview integer json render failed");
    CHECK(strstr(integer_render, "9007199254740993") != NULL,
          "mathview json integer lost exact value");

    result =
        sim_ir_mathview_render_latex(&builder, integer_constant_node, NULL, 0U, &integer_length);
    CHECK(result == SIM_RESULT_OK && integer_length > 0U, "mathview integer latex length failed");
    integer_render = (char*) realloc(integer_render, integer_length + 1U);
    CHECK(integer_render != NULL, "mathview integer latex allocation failed");
    result = sim_ir_mathview_render_latex(
        &builder, integer_constant_node, integer_render, integer_length + 1U, NULL);
    CHECK(result == SIM_RESULT_OK, "mathview integer latex render failed");
    CHECK(strstr(integer_render, "9007199254740993") != NULL,
          "mathview latex integer lost exact value");

    result = sim_ir_mathview_hash_sha256(&builder, root, hash_bytes);
    CHECK(result == SIM_RESULT_OK, "mathview sha256 failed");
    for (size_t i = 0U; i < sizeof(hash_bytes); ++i) {
        if (hash_bytes[i] != 0U) {
            any_nonzero = true;
            break;
        }
    }
    CHECK(any_nonzero, "mathview sha256 returned all zeros");

    free(canonical);
    free(integer_render);
    free(json);
    free(latex);
    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_field_allocation(void) {
    SimField  field    = { 0 };
    size_t    shape[2] = { 4U, 4U };
    SimResult result;
    size_t    offset     = 0U;
    size_t    indices[2] = { 1U, 2U };

    result = sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "field allocation failed");
    CHECK(sim_field_data(&field) != NULL, "field data pointer NULL");

    result = sim_field_index_offset(&field, indices, &offset);
    CHECK(result == SIM_RESULT_OK, "index offset failed");
    CHECK(offset == ((indices[0] * shape[1] + indices[1]) * sizeof(double)), "unexpected offset");

    sim_field_destroy(&field);
    return true;
}

static bool test_field_index_mapping_2d(void) {
    SimField  field    = { 0 };
    size_t    shape[2] = { 3U, 4U };
    SimResult result =
        sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "field init failed (2d mapping)");

    CHECK(sim_field_width(&field) == 4U, "field width mismatch");
    CHECK(sim_field_height(&field) == 3U, "field height mismatch");

    size_t x = 0U;
    size_t y = 0U;
    result   = sim_field_index_to_xy(&field, 0U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 0U && y == 0U, "idx 0 mapping incorrect");
    result = sim_field_index_to_xy(&field, 3U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 3U && y == 0U, "idx 3 mapping incorrect");
    result = sim_field_index_to_xy(&field, 4U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 0U && y == 1U, "idx 4 mapping incorrect");
    result = sim_field_index_to_xy(&field, 7U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 3U && y == 1U, "idx 7 mapping incorrect");
    result = sim_field_index_to_xy(&field, 11U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 3U && y == 2U, "idx 11 mapping incorrect");

    size_t idx = 0U;
    result     = sim_field_xy_to_index(&field, 0U, 0U, &idx);
    CHECK(result == SIM_RESULT_OK && idx == 0U, "xy (0,0) mapping incorrect");
    result = sim_field_xy_to_index(&field, 3U, 0U, &idx);
    CHECK(result == SIM_RESULT_OK && idx == 3U, "xy (3,0) mapping incorrect");
    result = sim_field_xy_to_index(&field, 0U, 1U, &idx);
    CHECK(result == SIM_RESULT_OK && idx == 4U, "xy (0,1) mapping incorrect");
    result = sim_field_xy_to_index(&field, 3U, 2U, &idx);
    CHECK(result == SIM_RESULT_OK && idx == 11U, "xy (3,2) mapping incorrect");

    sim_field_destroy(&field);
    return true;
}

static bool test_kernel_binding_index_mapping_2d(void) {
    SimField  field    = { 0 };
    size_t    shape[2] = { 2U, 5U };
    SimResult result =
        sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "field init failed (binding mapping)");

    SimKernelIRBinding binding = { .field_index = 0U,
                                   .field       = &field,
                                   .shape       = field.layout.shape,
                                   .strides     = field.layout.strides,
                                   .rank        = field.layout.rank };

    size_t x = 0U;
    size_t y = 0U;
    result   = sim_kernel_binding_index_to_xy(&binding, 0U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 0U && y == 0U, "binding idx 0 mapping incorrect");
    result = sim_kernel_binding_index_to_xy(&binding, 4U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 4U && y == 0U, "binding idx 4 mapping incorrect");
    result = sim_kernel_binding_index_to_xy(&binding, 5U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 0U && y == 1U, "binding idx 5 mapping incorrect");

    binding.shape   = NULL;
    binding.strides = NULL;
    binding.rank    = 0U;
    result          = sim_kernel_binding_index_to_xy(&binding, 9U, &x, &y);
    CHECK(result == SIM_RESULT_OK && x == 4U && y == 1U,
          "binding field fallback mapping incorrect");

    sim_field_destroy(&field);
    return true;
}

static bool test_gradient_operator(void) {
    SimField           src      = { 0 };
    SimField           grad     = { 0 };
    SimField           lap      = { 0 };
    size_t             shape[1] = { 8U };
    SimResult          result;
    SimIRBuilder       builder;
    SimIRNodeId        ref_node;
    SimIRNodeId        grad_node;
    SimIRNodeId        lap_node;
    SimKernelIRBinding bindings[3];
    SimKernelIROutput  outputs[2];
    KernelIR           kernel;
    SimBackend         backend = { 0 };
    double*            src_data;
    double*            grad_data;
    double*            lap_data;
    double             expected_grad[8] = { 0 };
    double             expected_lap[8]  = { 0 };
    size_t             i;
    const double       dx = 1.0;

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src field init failed");

    result = sim_field_init(&grad, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "grad field init failed");

    result = sim_field_init(&lap, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "lap field init failed");

    src_data  = (double*) sim_field_data(&src);
    grad_data = (double*) sim_field_data(&grad);
    lap_data  = (double*) sim_field_data(&lap);
    CHECK(src_data != NULL && grad_data != NULL && lap_data != NULL, "field data access failed");

    for (i = 0; i < shape[0]; ++i) {
        src_data[i]  = (double) i * (double) i * 0.5;
        grad_data[i] = 0.0;
        lap_data[i]  = 0.0;
    }

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "IR builder init failed");

    ref_node = sim_ir_builder_field_ref(&builder, 0U);
    CHECK(ref_node != SIM_IR_INVALID_NODE, "field ref node failed");

    grad_node = sim_ir_builder_diff(&builder, ref_node, 0U, dx, 1.0);
    CHECK(grad_node != SIM_IR_INVALID_NODE, "first diff node failed");

    lap_node = sim_ir_builder_diff(&builder, grad_node, 0U, dx, 1.0);
    CHECK(lap_node != SIM_IR_INVALID_NODE, "second diff node failed");

    bindings[0].field_index = 0U;
    bindings[0].field       = &src;
    bindings[1].field_index = 1U;
    bindings[1].field       = &grad;
    bindings[2].field_index = 2U;
    bindings[2].field       = &lap;

    outputs[0].field_index = 1U;
    outputs[0].expression  = grad_node;
    outputs[1].field_index = 2U;
    outputs[1].expression  = lap_node;

    kernel.builder           = &builder;
    kernel.bindings          = bindings;
    kernel.binding_count     = 3U;
    kernel.outputs           = outputs;
    kernel.output_count      = 2U;
    kernel.required_features = 0U;

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");

    backend_launch(&backend, &kernel);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend launch failed");

    compute_central_difference(src_data, shape[0], dx, expected_grad);
    compute_central_difference(expected_grad, shape[0], dx, expected_lap);

    for (i = 0; i < shape[0]; ++i) {
        if (fabs(grad_data[i] - expected_grad[i]) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] gradient mismatch at %zu: got %.6f expected %.6f\n",
                    i,
                    grad_data[i],
                    expected_grad[i]);
            sim_ir_builder_destroy(&builder);
            backend_destroy(&backend);
            sim_field_destroy(&lap);
            sim_field_destroy(&grad);
            sim_field_destroy(&src);
            return false;
        }

        if (fabs(lap_data[i] - expected_lap[i]) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] laplacian mismatch at %zu: got %.6f expected %.6f\n",
                    i,
                    lap_data[i],
                    expected_lap[i]);
            sim_ir_builder_destroy(&builder);
            backend_destroy(&backend);
            sim_field_destroy(&lap);
            sim_field_destroy(&grad);
            sim_field_destroy(&src);
            return false;
        }
    }

    sim_ir_builder_destroy(&builder);
    backend_destroy(&backend);
    sim_field_destroy(&lap);
    sim_field_destroy(&grad);
    sim_field_destroy(&src);
    return true;
}

static bool test_context_kernel_operator(void) {
    SimContext                         context  = { 0 };
    SimField                           src      = { 0 };
    SimField                           dst      = { 0 };
    SimBackend                         backend  = { 0 };
    size_t                             shape[1] = { 4U };
    SimResult                          result;
    double*                            src_data;
    size_t                             src_index = 0U;
    size_t                             dst_index = 0U;
    SimIRBuilder*                      builder;
    SimIRNodeId                        src_node;
    SimIRNodeId                        constant_node;
    SimIRNodeId                        sum_node;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor  output;
    SimOperatorKernelDescriptor        kernel_desc = { 0 };
    SimOperatorDescriptor              descriptor  = { 0 };
    double*                            dst_data;
    size_t                             i;

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src field init failed");

    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst field init failed");

    src_data = (double*) sim_field_data(&src);
    CHECK(src_data != NULL, "src data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        src_data[i] = (double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    CHECK(result == SIM_RESULT_OK, "add src field failed");

    result = sim_context_add_field(&context, &dst, &dst_index);
    CHECK(result == SIM_RESULT_OK, "add dst field failed");

    builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "context builder NULL");

    src_node = sim_ir_builder_field_ref(builder, 0U);
    CHECK(src_node != SIM_IR_INVALID_NODE, "builder field ref failed");

    constant_node = sim_ir_builder_constant(builder, 1.5);
    CHECK(constant_node != SIM_IR_INVALID_NODE, "constant node failed");

    sum_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, src_node, constant_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "addition node failed");

    bindings[0].ir_field_index      = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index      = 1U;
    bindings[1].context_field_index = dst_index;

    output.ir_field_index = 1U;
    output.expression     = sum_node;

    kernel_desc.builder           = builder;
    kernel_desc.bindings          = bindings;
    kernel_desc.binding_count     = 2U;
    kernel_desc.outputs           = &output;
    kernel_desc.output_count      = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name             = "kernel_add";
    descriptor.evaluate         = NULL;
    descriptor.dependencies     = NULL;
    descriptor.dependency_count = 0U;
    descriptor.kernel           = &kernel_desc;
    size_t op_index             = SIZE_MAX;
    result                      = sim_context_register_operator(&context, &descriptor, &op_index);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");
    SimOperator* op = sim_operator_registry_get(&context.world.operators, op_index);
    CHECK(op != NULL, "operator not found");
    if (op->kernel != NULL && op->kernel->kernel.builder != NULL) {
        const SimIRBuilder* kb = op->kernel->kernel.builder;
        fprintf(stderr,
                "[DEBUG] kernel builder node count %zu constants_count %zu\n",
                kb->count,
                kb->constants_count);
    }
    descriptor.dependencies     = NULL;
    descriptor.dependency_count = 0U;
    descriptor.kernel           = &kernel_desc;

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[DEBUG] kernel inline sim_context_execute returned %d\n", (int) result);
    }
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    dst_data = (double*) sim_field_data(sim_context_field(&context, dst_index));
    CHECK(dst_data != NULL, "dst data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        double expected = (double) i + 1.5;
        if (fabs(dst_data[i] - expected) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] kernel operator mismatch at %zu: got %.6f expected %.6f\n",
                    i,
                    dst_data[i],
                    expected);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_complex(void) {
    SimContext                         context  = { 0 };
    SimField                           src      = { 0 };
    SimField                           dst      = { 0 };
    SimBackend                         backend  = { 0 };
    size_t                             shape[1] = { 4U };
    SimResult                          result;
    SimComplexDouble*                  src_data;
    SimComplexDouble*                  dst_data;
    size_t                             src_index = 0U;
    size_t                             dst_index = 0U;
    SimIRBuilder*                      builder;
    SimIRNodeId                        src_node;
    SimIRNodeId                        sum_node;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor  output;
    SimOperatorKernelDescriptor        kernel_desc = { 0 };
    SimOperatorDescriptor              descriptor  = { 0 };
    size_t                             i;

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result =
        sim_field_init(&src, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src complex field init failed");

    result =
        sim_field_init(&dst, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst complex field init failed");

    src_data = sim_field_complex_data(&src);
    CHECK(src_data != NULL, "src complex data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        src_data[i].re = (double) i;
        src_data[i].im = -(double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    CHECK(result == SIM_RESULT_OK, "add src field failed");
    result = sim_context_add_field(&context, &dst, &dst_index);
    CHECK(result == SIM_RESULT_OK, "add dst field failed");

    builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "context builder NULL");

    src_node = sim_ir_builder_field_ref_typed(builder, 0U, sim_ir_type_vector(2U));
    CHECK(src_node != SIM_IR_INVALID_NODE, "builder complex field ref failed");

    sum_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, src_node, src_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "complex addition node failed");

    bindings[0].ir_field_index      = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index      = 1U;
    bindings[1].context_field_index = dst_index;

    output.ir_field_index = 1U;
    output.expression     = sum_node;

    kernel_desc.builder           = builder;
    kernel_desc.bindings          = bindings;
    kernel_desc.binding_count     = 2U;
    kernel_desc.outputs           = &output;
    kernel_desc.output_count      = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name             = "kernel_add_complex";
    descriptor.evaluate         = NULL;
    descriptor.destroy          = NULL;
    descriptor.userdata         = NULL;
    descriptor.dependencies     = NULL;
    descriptor.dependency_count = 0U;
    descriptor.kernel           = &kernel_desc;

    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[DEBUG] kernel pool sim_context_execute returned %d\n", (int) result);
    }
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    dst_data = sim_field_complex_data(sim_context_field(&context, dst_index));
    CHECK(dst_data != NULL, "dst complex data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        double expected_re = src_data[i].re + src_data[i].re;
        double expected_im = src_data[i].im + src_data[i].im;
        if (fabs(dst_data[i].re - expected_re) > 1e-9 ||
            fabs(dst_data[i].im - expected_im) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] kernel operator complex mismatch at %zu: got (%.6f,%.6f) expected "
                    "(%.6f,%.6f)\n",
                    i,
                    dst_data[i].re,
                    dst_data[i].im,
                    expected_re,
                    expected_im);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_vector_constant(void) {
    SimContext                         context  = { 0 };
    SimField                           src      = { 0 };
    SimField                           dst      = { 0 };
    SimBackend                         backend  = { 0 };
    size_t                             shape[1] = { 4U };
    SimResult                          result;
    SimComplexDouble*                  src_data;
    SimComplexDouble*                  dst_data;
    size_t                             src_index = 0U;
    size_t                             dst_index = 0U;
    SimIRBuilder*                      builder;
    SimIRNodeId                        src_node;
    SimIRNodeId                        const_node;
    SimIRNodeId                        sum_node;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor  output;
    SimOperatorKernelDescriptor        kernel_desc = { 0 };
    SimOperatorDescriptor              descriptor  = { 0 };
    size_t                             i;

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result =
        sim_field_init(&src, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src complex field init failed");

    result =
        sim_field_init(&dst, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst complex field init failed");

    src_data = sim_field_complex_data(&src);
    CHECK(src_data != NULL, "src complex data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        src_data[i].re = (double) i;
        src_data[i].im = -(double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    CHECK(result == SIM_RESULT_OK, "add src field failed");
    result = sim_context_add_field(&context, &dst, &dst_index);
    CHECK(result == SIM_RESULT_OK, "add dst field failed");

    builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "context builder NULL");

    src_node = sim_ir_builder_field_ref_typed(builder, 0U, sim_ir_type_vector(2U));
    CHECK(src_node != SIM_IR_INVALID_NODE, "builder complex field ref failed");

    double konst[2] = { 1.5, -0.5 };
    const_node      = sim_ir_builder_constant_vector(builder, konst, 2U);
    CHECK(const_node != SIM_IR_INVALID_NODE, "vector constant failed");

    sum_node = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, src_node, const_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "vector addition node failed");

    bindings[0].ir_field_index      = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index      = 1U;
    bindings[1].context_field_index = dst_index;

    output.ir_field_index = 1U;
    output.expression     = sum_node;

    kernel_desc.builder           = builder;
    kernel_desc.bindings          = bindings;
    kernel_desc.binding_count     = 2U;
    kernel_desc.outputs           = &output;
    kernel_desc.output_count      = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name             = "kernel_constant_vector";
    descriptor.evaluate         = NULL;
    descriptor.destroy          = NULL;
    descriptor.userdata         = NULL;
    descriptor.dependencies     = NULL;
    descriptor.dependency_count = 0U;
    descriptor.kernel           = &kernel_desc;

    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[DEBUG] pool sim_context_execute returned %d\n", (int) result);
    }
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    dst_data = sim_field_complex_data(sim_context_field(&context, dst_index));
    CHECK(dst_data != NULL, "dst complex data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        double expected_re = src_data[i].re + konst[0];
        double expected_im = src_data[i].im + konst[1];
        if (fabs(dst_data[i].re - expected_re) > 1e-9 ||
            fabs(dst_data[i].im - expected_im) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] kernel operator vector const mismatch at %zu: got (%.6f,%.6f) expected "
                    "(%.6f,%.6f)\n",
                    i,
                    dst_data[i].re,
                    dst_data[i].im,
                    expected_re,
                    expected_im);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_cuda(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 4U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");

    for (size_t i = 0; i < shape[0]; ++i) {
        ((double*) sim_field_data(&src))[i] = (double) i;
    }

    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");

    SimIRBuilder* builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "builder null");

    SimIRNodeId a   = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId b   = sim_ir_builder_constant(builder, 2.0);
    SimIRNodeId out = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, a, b);

    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = out;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    kernel_desc.required_features           = 0;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "cuda_test_add";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    backend.type = SIM_BACKEND_TYPE_CUDA;
    backend_init(&backend);
    /* Skip the test if CUDA not available */
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i) {
        if (fabs(out_data[i] - ((double) i + 2.0)) > 1e-9) {
            fprintf(stderr, "[FAIL] cuda kernel mismatch at %zu\n", i);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_cuda_nested(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 8U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i)
        ((double*) sim_field_data(&src))[i] = (double) i;
    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder* builder = sim_context_ir_builder(&context);
    SimIRNodeId   a       = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId   add     = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, a, a);
    SimIRNodeId   mul =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, add, sim_ir_builder_constant(builder, 2.5));
    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = mul;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    kernel_desc.required_features           = 0;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "cuda_test_nested";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");
    backend.type = SIM_BACKEND_TYPE_CUDA;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i) {
        double expected = ((double) i + (double) i) * 2.5;
        if (fabs(out_data[i] - expected) > 1e-9) {
            fprintf(stderr, "[FAIL] cuda nested mismatch at %zu\n", i);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_metal(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 4U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");

    for (size_t i = 0; i < shape[0]; ++i) {
        ((double*) sim_field_data(&src))[i] = (double) i;
    }

    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");

    SimIRBuilder* builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "builder null");

    SimIRNodeId a   = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId b   = sim_ir_builder_constant(builder, 2.0);
    SimIRNodeId out = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, a, b);

    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = out;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    kernel_desc.required_features           = 0;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_add";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i) {
        if (fabs(out_data[i] - ((double) i + 2.0)) > 1e-6) {
            fprintf(stderr,
                    "[FAIL] metal kernel mismatch at %zu: got %g exp %g\n",
                    i,
                    out_data[i],
                    (double) i + 2.0);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}
// static bool test_context_metal_pipeline_refcount(void) { return true; }

static bool test_context_kernel_operator_metal_nested(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 8U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i)
        ((double*) sim_field_data(&src))[i] = (double) i;

    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");

    SimIRBuilder* builder = sim_context_ir_builder(&context);
    SimIRNodeId   a       = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId   add     = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, a, a);
    SimIRNodeId   mul =
        sim_ir_builder_binary(builder, SIM_IR_NODE_MUL, add, sim_ir_builder_constant(builder, 2.5));

    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = mul;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    kernel_desc.required_features           = 0;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_nested";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i) {
        double expected = ((double) i + (double) i) * 2.5;
        if (fabs(out_data[i] - expected) > 1e-6) {
            fprintf(stderr, "[FAIL] metal nested mismatch at %zu\n", i);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_metal_diff(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 8U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };
    result              = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i)
        ((double*) sim_field_data(&src))[i] = (double) i * (double) i + 1.0;
    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder*                      builder = sim_context_ir_builder(&context);
    SimIRNodeId                        a       = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId                        diff    = sim_ir_builder_diff(builder, a, 0, 1.0, 1.0);
    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = diff;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_diff";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");
    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double        expected[8];
    const double* src_data = (const double*) sim_field_data(sim_context_field(&context, src_idx));
    CHECK(src_data != NULL, "src data NULL");
    compute_central_difference(src_data, shape[0], 1.0, expected);
    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i)
        if (fabs(out_data[i] - expected[i]) > 1e-6) {
            fprintf(stderr,
                    "[FAIL] metal diff mismatch at %zu: got %g exp %g\n",
                    i,
                    out_data[i],
                    expected[i]);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_metal_noise(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 8U };
    SimField   dst      = { 0 };
    result              = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    size_t dst_idx;
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder*                      builder = sim_context_ir_builder(&context);
    SimIRNodeId                        noise   = sim_ir_builder_noise(builder, 42U, 0.5);
    SimOperatorKernelBindingDescriptor bindings[1];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 0;
    output.expression                       = noise;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 1;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_noise";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");
    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i) {
        uint32_t h = (uint32_t) i * 1664525u + 42u;
        h ^= (h >> 16);
        uint32_t v   = h & 0xFFFFFFu;
        float    rn  = (float) v / 16777216.0f;
        float    val = (rn * 2.0f - 1.0f) * 0.5f;
        if (fabs(out_data[i] - (double) val) > 1e-6) {
            fprintf(stderr,
                    "[FAIL] metal noise mismatch at %zu: got %g exp %g\n",
                    i,
                    out_data[i],
                    (double) val);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_metal_warp(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 8U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };
    result              = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i)
        ((double*) sim_field_data(&src))[i] = (double) (i + 1);
    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder* builder = sim_context_ir_builder(&context);
    SimIRNodeId   node    = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId   warp =
        sim_ir_builder_warp(builder, node, SIM_IR_WARP_PROFILE_DIGAMMA, 0.5, 1.0, 1.0);
    CHECK(sim_ir_builder_node_warp_class(builder, warp) == SIM_WARP_LEVEL_LEVEL2,
          "warp node classification mismatch");
    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = warp;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_warp";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "registration failed");
    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double* out      = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    double* src_data = (double*) sim_field_data(sim_context_field(&context, src_idx));
    CHECK(src_data != NULL, "src data NULL");
    for (size_t i = 0; i < shape[0]; ++i) {
        SimWarpGuard guard = {
            .mode = SIM_CONTINUITY_NONE, .clamp_min = 0.0, .clamp_max = 0.0, .tolerance = 0.0
        };
        SimWarpSampleSpec spec = {
            .sample = src_data[i], .bias = 0.5, .delta = 1.0, .lambda = 1.0, .guard = guard
        };
        double expected = 0.0;
        CHECK(sim_ir_warp_sample_response(
                  &spec, SIM_IR_WARP_PROFILE_DIGAMMA, 0.0, NULL, NULL, &expected) == SIM_RESULT_OK,
              "warp helper evaluation failed");
        if (fabs(out[i] - expected) > 1e-4) {
            fprintf(
                stderr, "[FAIL] metal warp mismatch at %zu: got %g exp %g\n", i, out[i], expected);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_warp_guarded(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 1U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };

    CHECK(sim_context_init(&context) == SIM_RESULT_OK, "context init failed");
    CHECK(sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) ==
              SIM_RESULT_OK,
          "src init failed");
    CHECK(sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) ==
              SIM_RESULT_OK,
          "dst init failed");

    double* src_data = (double*) sim_field_data(&src);
    double* dst_data = (double*) sim_field_data(&dst);
    CHECK(src_data != NULL && dst_data != NULL, "field data access failed");
    src_data[0] = 0.0; /* digamma singularity */
    dst_data[0] = 0.0;

    size_t src_idx = 0U;
    size_t dst_idx = 0U;
    CHECK(sim_context_add_field(&context, &src, &src_idx) == SIM_RESULT_OK, "add src failed");
    CHECK(sim_context_add_field(&context, &dst, &dst_idx) == SIM_RESULT_OK, "add dst failed");

    SimIRBuilder* builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "builder lookup failed");
    SimIRNodeId ref = sim_ir_builder_field_ref(builder, 0);
    CHECK(ref != SIM_IR_INVALID_NODE, "field ref build failed");

    SimIRWarpSpec spec = { .operand     = ref,
                           .bias        = 0.0,
                           .delta       = 1.0,
                           .lambda      = 1.0,
                           .tolerance   = 0.0,
                           .profile     = SIM_IR_WARP_PROFILE_DIGAMMA,
                           .warp_class  = SIM_WARP_LEVEL_NONE,
                           .guard       = { .mode      = (int) SIM_CONTINUITY_CLAMPED,
                                            .clamp_min = -5.0,
                                            .clamp_max = 5.0,
                                            .tolerance = 1.0e-3 },
                           .result_type = sim_ir_type_scalar() };

    SimIRNodeId warp = sim_ir_builder_warp_spec(builder, &spec);
    CHECK(warp != SIM_IR_INVALID_NODE, "warp node build failed");

    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;

    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index = 1;
    output.expression     = warp;

    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    kernel_desc.required_features           = SIM_BACKEND_FEATURE_ANALYTIC_WARP;

    SimOperatorDescriptor descriptor = { 0 };
    descriptor.name                  = "cpu_warp_guard";
    descriptor.kernel                = &kernel_desc;

    CHECK(sim_context_register_operator(&context, &descriptor, NULL) == SIM_RESULT_OK,
          "registration failed");

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    double* out = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    CHECK(out != NULL, "output field fetch failed");

    SimWarpGuard guard = {
        .mode = SIM_CONTINUITY_CLAMPED, .clamp_min = -5.0, .clamp_max = 5.0, .tolerance = 1.0e-3
    };
    SimWarpSampleSpec sample_spec = {
        .sample = src_data[0], .bias = 0.0, .delta = 1.0, .lambda = 1.0, .guard = guard
    };
    double expected = 0.0;
    CHECK(sim_ir_warp_sample_response(
              &sample_spec, SIM_IR_WARP_PROFILE_DIGAMMA, 0.0, NULL, NULL, &expected) ==
              SIM_RESULT_OK,
          "guarded sample response failed");

    if (!isfinite(out[0])) {
        fprintf(stderr, "[FAIL] guarded warp produced non-finite value\n");
        backend_destroy(&backend);
        sim_context_destroy(&context);
        return false;
    }

    double clamp_abs = fmax(fabs(guard.clamp_min), fabs(guard.clamp_max));
    double response_limit =
        fabs(sample_spec.lambda) * (2.0 * fabs(sample_spec.delta)) * clamp_abs + 1.0e-6;
    if (fabs(out[0]) > response_limit) {
        fprintf(stderr, "[FAIL] guarded warp exceeded clamp bounds: %g\n", out[0]);
        backend_destroy(&backend);
        sim_context_destroy(&context);
        return false;
    }

    if (fabs(out[0] - expected) > 1.0e-6) {
        fprintf(stderr, "[FAIL] guarded warp mismatch: got %g expected %g\n", out[0], expected);
        backend_destroy(&backend);
        sim_context_destroy(&context);
        return false;
    }

    backend_destroy(&backend);
    sim_context_destroy(&context);
    return true;
}

static bool test_context_kernel_operator_metal_vector_constant(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 4U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };
    result              = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result =
        sim_field_init(&src, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result =
        sim_field_init(&dst, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i) {
        ((double*) sim_field_data(&src))[i * 2]     = (double) i;
        ((double*) sim_field_data(&src))[i * 2 + 1] = (double) i * 2.0;
    }
    size_t sidx, didx;
    result = sim_context_add_field(&context, &src, &sidx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &didx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder* builder = sim_context_ir_builder(&context);
    SimIRNodeId   node    = sim_ir_builder_field_ref_typed(builder, 0, sim_ir_type_vector(2U));
    double        vals[2] = { 1.0, -0.5 };
    SimIRNodeId   cnode   = sim_ir_builder_constant_vector(builder, vals, 2U);
    SimIRNodeId   add     = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, node, cnode);
    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = sidx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = didx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = add;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_vec_const";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "registration failed");
    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double* out = (double*) sim_field_data(sim_context_field(&context, didx));
    for (size_t i = 0; i < shape[0]; ++i) {
        double e0 = (double) i + 1.0;
        double e1 = (double) i * 2.0 - 0.5;
        if (fabs(out[i * 2] - e0) > 1e-9 || fabs(out[i * 2 + 1] - e1) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] metal vec const mismatch at %zu: got (%g,%g) exp (%g,%g)\n",
                    i,
                    out[i * 2],
                    out[i * 2 + 1],
                    e0,
                    e1);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_metal_pipeline_refcount(void) {
#if OAKFIELD_ENABLE_METAL
    SimResult  result;
    SimBackend backend = { 0 };
    backend.type       = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK)
        return true; /* skip if metal not available */

    SimContext context = { 0 };
    result             = sim_context_init(&context);
    if (result != SIM_RESULT_OK)
        return false;

    /* Create two kernels that should produce identical MSL (field + constant add) */
    SimField f0       = { 0 };
    SimField f1       = { 0 };
    size_t   shape[1] = { 64 };
    result = sim_field_init(&f0, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "f0 init");
    result = sim_field_init(&f1, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "f1 init");
    double* d0 = (double*) sim_field_data(&f0);
    double* d1 = (double*) sim_field_data(&f1);
    for (size_t i = 0; i < shape[0]; ++i) {
        d0[i] = (double) i;
        d1[i] = 0.0;
    }
    size_t f0_idx = 0, f1_idx = 1;
    result = sim_context_add_field(&context, &f0, &f0_idx);
    CHECK(result == SIM_RESULT_OK, "add f0");
    result = sim_context_add_field(&context, &f1, &f1_idx);
    CHECK(result == SIM_RESULT_OK, "add f1");

    SimIRBuilder*                      b1   = sim_context_ir_builder(&context);
    SimIRNodeId                        a1   = sim_ir_builder_field_ref(b1, 0);
    SimIRNodeId                        c1   = sim_ir_builder_constant(b1, 2.0);
    SimIRNodeId                        out1 = sim_ir_builder_binary(b1, SIM_IR_NODE_ADD, a1, c1);
    SimOperatorKernelBindingDescriptor bindings1[2];
    bindings1[0].ir_field_index      = 0;
    bindings1[0].context_field_index = 0;
    bindings1[1].ir_field_index      = 1;
    bindings1[1].context_field_index = 1;
    SimOperatorKernelOutputDescriptor outdesc1;
    outdesc1.ir_field_index         = 1;
    outdesc1.expression             = out1;
    SimOperatorKernelDescriptor kd1 = { 0 };
    kd1.builder                     = b1;
    kd1.bindings                    = bindings1;
    kd1.binding_count               = 2;
    kd1.outputs                     = &outdesc1;
    kd1.output_count                = 1;
    SimOperatorDescriptor desc1     = { 0 };
    desc1.name                      = "metal_refcount_test_1";
    desc1.kernel                    = &kd1;
    result                          = sim_context_register_operator(&context, &desc1, NULL);
    CHECK(result == SIM_RESULT_OK, "register op 1");

    /* second kernel - same emitted MSL */
    SimIRBuilder*                      b2   = sim_context_ir_builder(&context);
    SimIRNodeId                        a2   = sim_ir_builder_field_ref(b2, 0);
    SimIRNodeId                        c2   = sim_ir_builder_constant(b2, 2.0);
    SimIRNodeId                        out2 = sim_ir_builder_binary(b2, SIM_IR_NODE_ADD, a2, c2);
    SimOperatorKernelBindingDescriptor bindings2[2];
    bindings2[0].ir_field_index      = 0;
    bindings2[0].context_field_index = 0;
    bindings2[1].ir_field_index      = 1;
    bindings2[1].context_field_index = 1;
    SimOperatorKernelOutputDescriptor outdesc2;
    outdesc2.ir_field_index         = 1;
    outdesc2.expression             = out2;
    SimOperatorKernelDescriptor kd2 = { 0 };
    kd2.builder                     = b2;
    kd2.bindings                    = bindings2;
    kd2.binding_count               = 2;
    kd2.outputs                     = &outdesc2;
    kd2.output_count                = 1;
    SimOperatorDescriptor desc2     = { 0 };
    desc2.name                      = "metal_refcount_test_2";
    desc2.kernel                    = &kd2;
    result                          = sim_context_register_operator(&context, &desc2, NULL);
    CHECK(result == SIM_RESULT_OK, "register op 2");

    sim_context_set_backend(&context, &backend);

    /* Run the context twice - register both operators and execute to compile
     * pipelines. This should cause both descriptors to use the same pipeline in cache. */
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "execute op 1");
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "execute op 2");

    /* Check cache counts via debug API */
    size_t count = 0, total_ref = 0;
    sim_backend_metal_debug_get_pipeline_cache_count(&backend, &count);
    sim_backend_metal_debug_get_total_pipeline_refcount(&backend, &total_ref);
    if (count != 1 || total_ref != 2) {
        fprintf(stderr,
                "[FAIL] expected 1 pipeline with total refcount 2, got count=%zu ref=%zu\n",
                count,
                total_ref);
        sim_context_destroy(&context);
        backend_destroy(&backend);
        return false;
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);

    /* Re-init a fresh backend and assert the cache is empty (no pipelines) */
    SimBackend backend2 = { 0 };
    backend2.type       = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend2);
    if (backend2.last_error == SIM_RESULT_OK) {
        size_t count2 = 0;
        sim_backend_metal_debug_get_pipeline_cache_count(&backend2, &count2);
        if (count2 != 0) {
            fprintf(stderr,
                    "[FAIL] expected new metal backend to have empty pipeline cache got %zu\n",
                    count2);
            backend_destroy(&backend2);
            return false;
        }
        backend_destroy(&backend2);
    }
    return true;
#else
    return true;
#endif
}

static bool test_context_kernel_operator_cuda_warp(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 8U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };
    result              = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result = sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i)
        ((double*) sim_field_data(&src))[i] = (double) (i + 1);
    size_t src_idx, dst_idx;
    result = sim_context_add_field(&context, &src, &src_idx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder* builder = sim_context_ir_builder(&context);
    SimIRNodeId   node    = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId   warp =
        sim_ir_builder_warp(builder, node, SIM_IR_WARP_PROFILE_DIGAMMA, 0.5, 1.0, 1.0);
    CHECK(sim_ir_builder_node_warp_class(builder, warp) == SIM_WARP_LEVEL_LEVEL2,
          "warp node classification mismatch");
    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = src_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = warp;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "cuda_test_warp";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "registration failed");
    backend.type = SIM_BACKEND_TYPE_CUDA;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double* out      = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    double* src_data = (double*) sim_field_data(sim_context_field(&context, src_idx));
    CHECK(src_data != NULL, "src data NULL");
    for (size_t i = 0; i < shape[0]; ++i) {
        SimWarpGuard guard = {
            .mode = SIM_CONTINUITY_NONE, .clamp_min = 0.0, .clamp_max = 0.0, .tolerance = 0.0
        };
        SimWarpSampleSpec spec = {
            .sample = src_data[i], .bias = 0.5, .delta = 1.0, .lambda = 1.0, .guard = guard
        };
        double expected = 0.0;
        CHECK(sim_ir_warp_sample_response(
                  &spec, SIM_IR_WARP_PROFILE_DIGAMMA, 0.0, NULL, NULL, &expected) == SIM_RESULT_OK,
              "warp helper evaluation failed");
        if (fabs(out[i] - expected) > 1e-6) {
            fprintf(
                stderr, "[FAIL] cuda warp mismatch at %zu: got %g exp %g\n", i, out[i], expected);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_cuda_vector_constant(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 4U };
    SimField   src      = { 0 };
    SimField   dst      = { 0 };
    result              = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result =
        sim_field_init(&src, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src init failed");
    result =
        sim_field_init(&dst, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");
    for (size_t i = 0; i < shape[0]; ++i) {
        ((double*) sim_field_data(&src))[i * 2]     = (double) i;
        ((double*) sim_field_data(&src))[i * 2 + 1] = (double) i * 2.0;
    }
    size_t sidx, didx;
    result = sim_context_add_field(&context, &src, &sidx);
    CHECK(result == SIM_RESULT_OK, "add src failed");
    result = sim_context_add_field(&context, &dst, &didx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");
    SimIRBuilder* builder = sim_context_ir_builder(&context);
    SimIRNodeId   node    = sim_ir_builder_field_ref_typed(builder, 0, sim_ir_type_vector(2U));
    double        vals[2] = { 1.0, -0.5 };
    SimIRNodeId   cnode   = sim_ir_builder_constant_vector(builder, vals, 2U);
    SimIRNodeId   add     = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, node, cnode);
    SimOperatorKernelBindingDescriptor bindings[2];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = sidx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = didx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 1;
    output.expression                       = add;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 2;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "cuda_test_vec_const";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "registration failed");
    backend.type = SIM_BACKEND_TYPE_CUDA;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");
    double* out = (double*) sim_field_data(sim_context_field(&context, didx));
    for (size_t i = 0; i < shape[0]; ++i) {
        double e0 = (double) i + 1.0;
        double e1 = (double) i * 2.0 - 0.5;
        if (fabs(out[i * 2] - e0) > 1e-9 || fabs(out[i * 2 + 1] - e1) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] cuda vec const mismatch at %zu: got (%g,%g) exp (%g,%g)\n",
                    i,
                    out[i * 2],
                    out[i * 2 + 1],
                    e0,
                    e1);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }
    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_context_kernel_operator_metal_multi_fields(void) {
    SimResult  result;
    SimContext context  = { 0 };
    SimBackend backend  = { 0 };
    size_t     shape[1] = { 4U };
    SimField   a        = { 0 };
    SimField   b        = { 0 };
    SimField   dst      = { 0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result = sim_field_init(&a, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "a init failed");
    result = sim_field_init(&b, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "b init failed");
    result = sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst init failed");

    for (size_t i = 0; i < shape[0]; ++i) {
        ((double*) sim_field_data(&a))[i] = (double) i;
        ((double*) sim_field_data(&b))[i] = (double) i * 2.0;
    }

    size_t a_idx, b_idx, dst_idx;
    result = sim_context_add_field(&context, &a, &a_idx);
    CHECK(result == SIM_RESULT_OK, "add a failed");
    result = sim_context_add_field(&context, &b, &b_idx);
    CHECK(result == SIM_RESULT_OK, "add b failed");
    result = sim_context_add_field(&context, &dst, &dst_idx);
    CHECK(result == SIM_RESULT_OK, "add dst failed");

    SimIRBuilder* builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "builder null");

    SimIRNodeId a_node = sim_ir_builder_field_ref(builder, 0);
    SimIRNodeId b_node = sim_ir_builder_field_ref(builder, 1);
    SimIRNodeId out    = sim_ir_builder_binary(builder, SIM_IR_NODE_ADD, a_node, b_node);

    SimOperatorKernelBindingDescriptor bindings[3];
    bindings[0].ir_field_index      = 0;
    bindings[0].context_field_index = a_idx;
    bindings[1].ir_field_index      = 1;
    bindings[1].context_field_index = b_idx;
    bindings[2].ir_field_index      = 2;
    bindings[2].context_field_index = dst_idx;
    SimOperatorKernelOutputDescriptor output;
    output.ir_field_index                   = 2;
    output.expression                       = out;
    SimOperatorKernelDescriptor kernel_desc = { 0 };
    kernel_desc.builder                     = builder;
    kernel_desc.bindings                    = bindings;
    kernel_desc.binding_count               = 3;
    kernel_desc.outputs                     = &output;
    kernel_desc.output_count                = 1;
    kernel_desc.required_features           = 0;
    SimOperatorDescriptor descriptor        = { 0 };
    descriptor.name                         = "metal_test_multi_fields";
    descriptor.kernel                       = &kernel_desc;
    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        sim_context_destroy(&context);
        return true;
    }
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    double* out_data = (double*) sim_field_data(sim_context_field(&context, dst_idx));
    for (size_t i = 0; i < shape[0]; ++i) {
        double expected = ((double) i) + ((double) i * 2.0);
        if (fabs(out_data[i] - expected) > 1e-6) {
            fprintf(stderr,
                    "[FAIL] metal multi-fields mismatch at %zu: got %g exp %g\n",
                    i,
                    out_data[i],
                    expected);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_kernel_builder_copy(void) {
    SimContext                         context  = { 0 };
    SimField                           src      = { 0 };
    SimField                           dst      = { 0 };
    SimBackend                         backend  = { 0 };
    size_t                             shape[1] = { 3U };
    SimResult                          result;
    SimComplexDouble*                  src_data;
    SimComplexDouble*                  dst_data;
    size_t                             src_index = 0U;
    size_t                             dst_index = 0U;
    SimIRBuilder                       builder;
    SimIRNodeId                        src_node;
    SimIRNodeId                        const_node;
    SimIRNodeId                        sum_node;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor  output;
    SimOperatorKernelDescriptor        kernel_desc = { 0 };
    SimOperatorDescriptor              descriptor  = { 0 };
    size_t                             i;
    double                             konst[2] = { 2.5, 1.0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result =
        sim_field_init(&src, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src complex field init failed");

    result =
        sim_field_init(&dst, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst complex field init failed");

    src_data = sim_field_complex_data(&src);
    for (i = 0U; i < shape[0]; ++i) {
        src_data[i].re = (double) i;
        src_data[i].im = -(double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    CHECK(result == SIM_RESULT_OK, "add src field failed");
    result = sim_context_add_field(&context, &dst, &dst_index);
    CHECK(result == SIM_RESULT_OK, "add dst field failed");

    /* Build IR with local builder (not context builder) */
    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "local builder init failed");
    src_node = sim_ir_builder_field_ref_typed(&builder, 0U, sim_ir_type_vector(2U));
    CHECK(src_node != SIM_IR_INVALID_NODE, "builder complex field ref failed");
    const_node = sim_ir_builder_constant_vector(&builder, konst, 2U);
    CHECK(const_node != SIM_IR_INVALID_NODE, "vector constant failed");
    sum_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, src_node, const_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "vector addition node failed");

    bindings[0].ir_field_index      = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index      = 1U;
    bindings[1].context_field_index = dst_index;
    output.ir_field_index           = 1U;
    output.expression               = sum_node;

    kernel_desc.builder           = &builder;
    kernel_desc.bindings          = bindings;
    kernel_desc.binding_count     = 2U;
    kernel_desc.outputs           = &output;
    kernel_desc.output_count      = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name             = "kernel_copy_test";
    descriptor.evaluate         = NULL;
    descriptor.destroy          = NULL;
    descriptor.userdata         = NULL;
    descriptor.dependencies     = NULL;
    descriptor.dependency_count = 0U;
    descriptor.kernel           = &kernel_desc;

    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    /* Destroy the local builder - kernel should have its own copy */
    sim_ir_builder_destroy(&builder);

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    dst_data = sim_field_complex_data(sim_context_field(&context, dst_index));
    CHECK(dst_data != NULL, "dst complex data NULL");
    for (i = 0U; i < shape[0]; ++i) {
        double expected_re = src_data[i].re + konst[0];
        double expected_im = src_data[i].im + konst[1];
        if (fabs(dst_data[i].re - expected_re) > 1e-9 ||
            fabs(dst_data[i].im - expected_im) > 1e-9) {
            fprintf(stderr,
                    "[FAIL] kernel builder copy mismatch at %zu: got (%.6f,%.6f) expected "
                    "(%.6f,%.6f)\n",
                    i,
                    dst_data[i].re,
                    dst_data[i].im,
                    expected_re,
                    expected_im);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    return true;
}

static bool test_kernel_builder_copy_mutation_inline(void) {
    fprintf(stderr, "[DEBUG] starting inline builder mutation test\n");
    SimContext                         context  = { 0 };
    SimField                           src      = { 0 };
    SimField                           dst      = { 0 };
    SimBackend                         backend  = { 0 };
    size_t                             shape[1] = { 2U };
    SimResult                          result;
    SimComplexDouble*                  src_data;
    SimComplexDouble*                  dst_data;
    size_t                             src_index = 0U;
    size_t                             dst_index = 0U;
    SimIRBuilder                       builder;
    SimIRNodeId                        src_node;
    SimIRNodeId                        const_node;
    SimIRNodeId                        sum_node;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor  output;
    SimOperatorKernelDescriptor        kernel_desc   = { 0 };
    SimOperatorDescriptor              descriptor    = { 0 };
    const double                       konst_orig[2] = { 3.0, -1.0 };
    const double                       konst_mut[2]  = { 10.0, 20.0 };
    size_t                             i;

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result =
        sim_field_init(&src, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src complex field init failed");
    result =
        sim_field_init(&dst, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst complex field init failed");

    src_data = sim_field_complex_data(&src);
    for (i = 0U; i < shape[0]; ++i) {
        src_data[i].re = (double) i;
        src_data[i].im = -(double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    CHECK(result == SIM_RESULT_OK, "add src field failed");
    result = sim_context_add_field(&context, &dst, &dst_index);
    CHECK(result == SIM_RESULT_OK, "add dst field failed");

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "local builder init failed");
    src_node = sim_ir_builder_field_ref_typed(&builder, 0U, sim_ir_type_vector(2U));
    CHECK(src_node != SIM_IR_INVALID_NODE, "builder field ref failed");
    const_node = sim_ir_builder_constant_vector(&builder, konst_orig, 2U);
    CHECK(const_node != SIM_IR_INVALID_NODE, "vector constant failed");
    sum_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, src_node, const_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "vector addition node failed");

    bindings[0].ir_field_index      = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index      = 1U;
    bindings[1].context_field_index = dst_index;
    output.ir_field_index           = 1U;
    output.expression               = sum_node;

    kernel_desc.builder           = &builder;
    kernel_desc.bindings          = bindings;
    kernel_desc.binding_count     = 2U;
    kernel_desc.outputs           = &output;
    kernel_desc.output_count      = 1U;
    kernel_desc.required_features = 0U;

    descriptor.name             = "kernel_copy_inline_mutation";
    descriptor.evaluate         = NULL;
    descriptor.destroy          = NULL;
    descriptor.userdata         = NULL;
    descriptor.dependencies     = NULL;
    descriptor.dependency_count = 0U;
    descriptor.kernel           = &kernel_desc;

    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    /* mutate original builder's constant values - kernel should NOT see these */
    const SimIRNode* nptr = sim_ir_builder_get(&builder, const_node);
    CHECK(nptr != NULL, "get const node failed");
    /* The builder node small[] is mutable directly */
    SimIRNode* mutable_node              = (SimIRNode*) &builder.nodes[const_node];
    mutable_node->data.constant.small[0] = konst_mut[0];
    mutable_node->data.constant.small[1] = konst_mut[1];

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    dst_data = sim_field_complex_data(sim_context_field(&context, dst_index));
    for (i = 0U; i < shape[0]; ++i) {
        double expected_re = src_data[i].re + konst_orig[0];
        double expected_im = src_data[i].im + konst_orig[1];
        if (fabs(dst_data[i].re - expected_re) > 1e-9 ||
            fabs(dst_data[i].im - expected_im) > 1e-9) {
            fprintf(stderr, "[FAIL] kernel inline mutation used mutated value at %zu\n", i);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            sim_ir_builder_destroy(&builder);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_kernel_builder_copy_mutation_pool(void) {
    fprintf(stderr, "[DEBUG] starting pool builder mutation test\n");
    SimContext                         context    = { 0 };
    SimField                           src        = { 0 };
    SimField                           dst        = { 0 };
    SimBackend                         backend    = { 0 };
    size_t                             components = SIM_IR_SMALL_CONSTANT_CAPACITY + 1U;
    size_t                             shape[1]   = { 2U };
    SimResult                          result;
    double*                            src_data;
    double*                            dst_data;
    size_t                             src_index = 0U;
    size_t                             dst_index = 0U;
    SimIRBuilder                       builder;
    SimIRNodeId                        src_node;
    SimIRNodeId                        const_node;
    SimIRNodeId                        sum_node;
    SimOperatorKernelBindingDescriptor bindings[2];
    SimOperatorKernelOutputDescriptor  output;
    SimOperatorKernelDescriptor        kernel_desc = { 0 };
    SimOperatorDescriptor              descriptor  = { 0 };
    double                             vals[SIM_IR_SMALL_CONSTANT_CAPACITY + 1U];
    double                             mutated[SIM_IR_SMALL_CONSTANT_CAPACITY + 1U];
    size_t                             i;

    for (i = 0U; i < components; ++i) {
        vals[i]    = (double) (i + 2);
        mutated[i] = (double) (i + 100);
    }

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");
    result = sim_field_init(
        &src, 1U, shape, sizeof(double) * components, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "src alloc failed");
    result = sim_field_init(
        &dst, 1U, shape, sizeof(double) * components, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "dst alloc failed");

    src_data = (double*) sim_field_real_data(&src);
    for (i = 0U; i < components * shape[0]; ++i) {
        src_data[i] = (double) i;
    }

    result = sim_context_add_field(&context, &src, &src_index);
    CHECK(result == SIM_RESULT_OK, "add src field failed");
    result = sim_context_add_field(&context, &dst, &dst_index);
    CHECK(result == SIM_RESULT_OK, "add dst field failed");

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "builder init failed");
    src_node = sim_ir_builder_field_ref_typed(&builder, 0U, sim_ir_type_vector(components));
    CHECK(src_node != SIM_IR_INVALID_NODE, "field ref failed");
    const_node = sim_ir_builder_constant_vector(&builder, vals, components);
    CHECK(const_node != SIM_IR_INVALID_NODE, "vector const failed");
    sum_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, src_node, const_node);
    CHECK(sum_node != SIM_IR_INVALID_NODE, "sum node failed");

    bindings[0].ir_field_index      = 0U;
    bindings[0].context_field_index = src_index;
    bindings[1].ir_field_index      = 1U;
    bindings[1].context_field_index = dst_index;
    output.ir_field_index           = 1U;
    output.expression               = sum_node;

    kernel_desc.builder           = &builder;
    kernel_desc.bindings          = bindings;
    kernel_desc.binding_count     = 2U;
    kernel_desc.outputs           = &output;
    kernel_desc.output_count      = 1U;
    kernel_desc.required_features = 0U;
    descriptor.name               = "kernel_copy_pool_mutation";
    descriptor.evaluate           = NULL;
    descriptor.destroy            = NULL;
    descriptor.userdata           = NULL;
    descriptor.dependencies       = NULL;
    descriptor.dependency_count   = 0U;
    descriptor.kernel             = &kernel_desc;

    result = sim_context_register_operator(&context, &descriptor, NULL);
    CHECK(result == SIM_RESULT_OK, "operator registration failed");

    /* mutate pool data */
    const SimIRNode* cn = sim_ir_builder_get(&builder, const_node);
    CHECK(cn != NULL, "pool const get failed");
    size_t idx = cn->data.constant.constant_index;
    CHECK(idx != SIM_IR_INVALID_CONSTANT_INDEX, "pool const shouldn't be inline");
    size_t off = builder.constants_offsets[idx];
    /* Debug: print builder constant pool before mutation */
    fprintf(stderr,
            "[DEBUG] builder.constants_count=%zu components=%zu offset=%zu\n",
            builder.constants_count,
            builder.constants_components[idx],
            off);
    for (i = 0U; i < components; ++i) {
        fprintf(stderr,
                "[DEBUG] builder.consts[%zu] = %g\n",
                (size_t) (off + i),
                builder.constants_data[off + i]);
    }
    for (i = 0U; i < components; ++i) {
        builder.constants_data[off + i] = mutated[i];
    }
    /* Debug: print builder constant pool after mutation */
    for (i = 0U; i < components; ++i) {
        fprintf(stderr,
                "[DEBUG] mutated builder.consts[%zu] = %g\n",
                (size_t) (off + i),
                builder.constants_data[off + i]);
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);
    result = sim_context_execute(&context);
    CHECK(result == SIM_RESULT_OK, "context execute failed");

    dst_data = (double*) sim_field_real_data(sim_context_field(&context, dst_index));
    for (i = 0U; i < components * shape[0]; ++i) {
        double expected = ((double) i) + vals[i % components];
        if (fabs(dst_data[i] - expected) > 1e-9) {
            fprintf(
                stderr,
                "[FAIL] kernel pool mutation used mutated value at %zu: got %.12g expected %.12g\n",
                i,
                dst_data[i],
                expected);
            sim_context_destroy(&context);
            backend_destroy(&backend);
            sim_ir_builder_destroy(&builder);
            return false;
        }
    }

    sim_context_destroy(&context);
    backend_destroy(&backend);
    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_field_complex_allocation(void) {
    SimField          field    = { 0 };
    size_t            shape[1] = { 4U };
    SimResult         result;
    SimComplexDouble* cdata;
    SimComplexDouble  expected[4] = { { 0.0, 1.0 }, { 2.0, 3.0 }, { 4.0, 5.0 }, { 6.0, 7.0 } };
    size_t            i;

    result =
        sim_field_init(&field, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "complex field init failed");
    CHECK(sim_field_is_complex(&field) == true, "complex detection failed");
    CHECK(sim_field_components(&field) == 2U, "components count mismatch");

    cdata = sim_field_complex_data(&field);
    CHECK(cdata != NULL, "complex data pointer NULL");
    for (i = 0; i < shape[0]; ++i) {
        cdata[i] = expected[i];
    }

    for (i = 0; i < shape[0]; ++i) {
        if (cdata[i].re != expected[i].re || cdata[i].im != expected[i].im) {
            fprintf(stderr, "[FAIL] complex value mismatch at %zu\n", i);
            sim_field_destroy(&field);
            return false;
        }
    }

    sim_field_destroy(&field);
    return true;
}

static bool test_field_require_complex_promotion(void) {
    SimField          field    = { 0 };
    size_t            shape[1] = { 3U };
    SimResult         result;
    double*           rdata;
    SimComplexDouble* cdata;

    result = sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "real field init failed");

    rdata = sim_field_real_data(&field);
    CHECK(rdata != NULL, "real data pointer NULL");
    for (size_t i = 0; i < shape[0]; ++i) {
        rdata[i] = (double) (i + 1U);
    }

    result = sim_field_require_complex(&field);
    CHECK(result == SIM_RESULT_OK, "require_complex promotion failed");
    CHECK(sim_field_is_complex(&field), "field not complex after promotion");
    CHECK(sim_field_complex_mode(&field), "complex_mode not set after promotion");

    cdata = sim_field_complex_data(&field);
    CHECK(cdata != NULL, "complex data pointer NULL after promotion");
    for (size_t i = 0; i < shape[0]; ++i) {
        if (cdata[i].re != (double) (i + 1U) || cdata[i].im != 0.0) {
            fprintf(stderr,
                    "[FAIL] promoted complex value mismatch at %zu: got (%.3f,%.3f)\n",
                    i,
                    cdata[i].re,
                    cdata[i].im);
            sim_field_destroy(&field);
            return false;
        }
    }

    SimFieldRepresentation repr = sim_field_representation(&field);
    CHECK(repr.value_kind == SIM_FIELD_VALUE_COMPLEX_SCALAR,
          "representation not complex after promotion");

    sim_field_destroy(&field);
    return true;
}

static bool test_field_require_complex_rejects_incompatible(void) {
    SimField       noncontig   = { 0 };
    SimField       float_field = { 0 };
    size_t         shape2d[2]  = { 2U, 2U };
    size_t         strides[2]  = { 3U, 1U };
    double         buffer[6]   = { 0 };
    SimFieldLayout layout      = {
        .rank = 2U, .shape = shape2d, .strides = strides, .contiguous = false
    };
    size_t    shape1d[1] = { 3U };
    SimResult result;

    result =
        sim_field_wrap(&noncontig, &layout, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, buffer);
    CHECK(result == SIM_RESULT_OK, "non-contiguous wrap init failed");
    result = sim_field_require_complex(&noncontig);
    CHECK(result == SIM_RESULT_INVALID_STATE, "non-contiguous promotion should fail");
    sim_field_destroy(&noncontig);

    result =
        sim_field_init(&float_field, 1U, shape1d, sizeof(float), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "float field init failed");
    result = sim_field_require_complex(&float_field);
    CHECK(result == SIM_RESULT_DEPENDENCY_ERROR, "non-double promotion should fail");
    sim_field_destroy(&float_field);

    return true;
}

static bool test_field_representation_unknown_rejected(void) {
    SimField               field    = { 0 };
    size_t                 shape[1] = { 4U };
    SimResult              result;
    SimFieldRepresentation repr;

    result = sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "field init failed");

    repr            = sim_field_representation(&field);
    repr.value_kind = SIM_FIELD_VALUE_UNKNOWN;
    result          = sim_field_set_representation(&field, repr);
    CHECK(result == SIM_RESULT_INVALID_ARGUMENT, "unknown value kind should be rejected");

    repr        = sim_field_representation(&field);
    repr.domain = SIM_FIELD_DOMAIN_UNKNOWN;
    result      = sim_field_set_representation(&field, repr);
    CHECK(result == SIM_RESULT_INVALID_ARGUMENT, "unknown domain should be rejected");

    sim_field_destroy(&field);
    return true;
}

static bool test_ir_field_ref_from_field(void) {
    SimContext    context  = { 0 };
    SimField      field    = { 0 };
    size_t        shape[1] = { 4U };
    SimResult     result;
    SimIRBuilder* builder;
    SimIRNodeId   node;
    SimIRType     type;

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    result =
        sim_field_init(&field, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "complex field init failed");

    result = sim_context_add_field(&context, &field, NULL);
    CHECK(result == SIM_RESULT_OK, "add field failed");

    builder = sim_context_ir_builder(&context);
    CHECK(builder != NULL, "context builder NULL");

    node = sim_ir_builder_field_ref_typed(builder, 0U, sim_ir_type_vector(2U));
    CHECK(node != SIM_IR_INVALID_NODE, "typed field ref failed");

    type = sim_ir_builder_node_type(builder, node);
    CHECK(type.kind == SIM_IR_VALUE_VECTOR && type.components == 2U, "typed node type mismatch");

    sim_context_destroy(&context);
    return true;
}

static bool test_ir_constant_vector_inline(void) {
    printf("[TEST START] test_ir_constant_vector_inline\n");
    fflush(stdout);
    SimIRBuilder     builder;
    SimResult        result;
    SimIRNodeId      node;
    const SimIRNode* nptr;
    const double     vals[2] = { 1.25, -0.75 };

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    node = sim_ir_builder_constant_vector(&builder, vals, 2U);
    CHECK(node != SIM_IR_INVALID_NODE, "inline vector constant node creation failed");

    SimIRType t = sim_ir_builder_node_type(&builder, node);
    CHECK(t.kind == SIM_IR_VALUE_VECTOR && t.components == 2U,
          "vector typed constant node type mismatch");

    nptr = sim_ir_builder_get(&builder, node);
    CHECK(nptr != NULL, "typed constant node pointer NULL");

    /* small inline storage should be used for 2 lanes */
    CHECK(nptr->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX,
          "inline constant unexpectedly has pool index");
    CHECK(fabs(nptr->data.constant.small[0] - vals[0]) < 1e-12, "inline constant value 0 mismatch");
    CHECK(fabs(nptr->data.constant.small[1] - vals[1]) < 1e-12, "inline constant value 1 mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_constant_vector_pool(void) {
    printf("[TEST START] test_ir_constant_vector_pool\n");
    fflush(stdout);
    SimIRBuilder     builder;
    SimResult        result;
    SimIRNodeId      node;
    const SimIRNode* nptr;
    size_t           comps = SIM_IR_SMALL_CONSTANT_CAPACITY + 1U;
    double           vals[SIM_IR_SMALL_CONSTANT_CAPACITY + 1U];
    size_t           i;

    for (i = 0U; i < comps; ++i) {
        vals[i] = (double) i * 1.5;
    }

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    node = sim_ir_builder_constant_vector(&builder, vals, comps);
    CHECK(node != SIM_IR_INVALID_NODE, "pool vector constant node creation failed");

    nptr = sim_ir_builder_get(&builder, node);
    CHECK(nptr != NULL, "pool constant node pointer NULL");

    CHECK(nptr->data.constant.constant_index != SIM_IR_INVALID_CONSTANT_INDEX,
          "pool constant missing index");
    size_t idx = nptr->data.constant.constant_index;
    CHECK(idx < builder.constants_count, "constant index out of range");
    CHECK(builder.constants_components[idx] == comps, "constant pool components mismatch");
    size_t offset = builder.constants_offsets[idx];
    for (i = 0U; i < comps; ++i) {
        CHECK(fabs(builder.constants_data[offset + i] - vals[i]) < 1e-12,
              "pool constant data mismatch");
    }

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_constant_vector_dedup(void) {
    printf("[TEST START] test_ir_constant_vector_dedup\n");
    fflush(stdout);
    SimIRBuilder builder;
    SimResult    result;
    SimIRNodeId  n1, n2;
    const size_t comps = SIM_IR_SMALL_CONSTANT_CAPACITY + 1U;
    double       vals[SIM_IR_SMALL_CONSTANT_CAPACITY + 1U];
    size_t       i;

    for (i = 0; i < comps; ++i) {
        vals[i] = (double) (i + 1);
    }

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "builder init failed");

    n1 = sim_ir_builder_constant_vector(&builder, vals, comps);
    CHECK(n1 != SIM_IR_INVALID_NODE, "first pool vector creation failed");
    n2 = sim_ir_builder_constant_vector(&builder, vals, comps);
    CHECK(n2 != SIM_IR_INVALID_NODE, "second pool vector creation failed");

    const SimIRNode* ptr1 = sim_ir_builder_get(&builder, n1);
    const SimIRNode* ptr2 = sim_ir_builder_get(&builder, n2);
    CHECK(ptr1 != NULL && ptr2 != NULL, "get node failed");
    CHECK(ptr1->data.constant.constant_index == ptr2->data.constant.constant_index,
          "dedup did not reuse pool entry");
    CHECK(builder.constants_count == 1U, "expected constants_count == 1 after dedup");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_evaluate_complex_constant_inline(void) {
    printf("[TEST START] test_ir_evaluate_complex_constant_inline\n");
    fflush(stdout);
    SimIRBuilder          builder;
    SimResult             result;
    SimIRNodeId           node;
    SimComplexDouble      value     = { 0.0, 0.0 };
    SimIREvaluatorComplex evaluator = { 0 };

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    node = sim_ir_builder_constant_complex(&builder, 1.25, -0.5);
    CHECK(node != SIM_IR_INVALID_NODE, "complex constant node creation failed");

    result = sim_ir_evaluate_complex(&builder, node, &evaluator, &value);
    CHECK(result == SIM_RESULT_OK, "complex evaluator failed");
    CHECK(fabs(value.re - 1.25) < 1.0e-12, "complex constant real mismatch");
    CHECK(fabs(value.im + 0.5) < 1.0e-12, "complex constant imag mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_evaluate_complex_constant_pool(void) {
    printf("[TEST START] test_ir_evaluate_complex_constant_pool\n");
    fflush(stdout);
    SimIRBuilder          builder;
    SimResult             result;
    SimIRNodeId           node;
    SimComplexDouble      value     = { 0.0, 0.0 };
    SimIREvaluatorComplex evaluator = { 0 };
    size_t                comps     = SIM_IR_SMALL_CONSTANT_CAPACITY + 2U;
    double                vals[SIM_IR_SMALL_CONSTANT_CAPACITY + 2U];

    for (size_t i = 0U; i < comps; ++i) {
        vals[i] = (double) i * 0.25 + 1.0;
    }

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    node = sim_ir_builder_constant_vector_typed(&builder, vals, comps, sim_ir_type_vector(comps));
    CHECK(node != SIM_IR_INVALID_NODE, "pool vector constant node creation failed");

    result = sim_ir_evaluate_complex(&builder, node, &evaluator, &value);
    CHECK(result == SIM_RESULT_OK, "complex evaluator failed");
    CHECK(fabs(value.re - vals[0]) < 1.0e-12, "complex pool constant real mismatch");
    CHECK(fabs(value.im - vals[1]) < 1.0e-12, "complex pool constant imag mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_evaluate_pow_real(void) {
    printf("[TEST START] test_ir_evaluate_pow_real\n");
    fflush(stdout);
    SimIRBuilder   builder;
    SimResult      result;
    SimIRNodeId    base_node;
    SimIRNodeId    exponent_node;
    SimIRNodeId    pow_node;
    double         value     = 0.0;
    SimIREvaluator evaluator = { 0 };

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    base_node     = sim_ir_builder_constant(&builder, 9.0);
    exponent_node = sim_ir_builder_constant(&builder, 0.5);
    pow_node      = sim_ir_builder_binary(&builder, SIM_IR_NODE_POW, base_node, exponent_node);
    CHECK(pow_node != SIM_IR_INVALID_NODE, "real pow node creation failed");

    result = sim_ir_evaluate(&builder, pow_node, &evaluator, &value);
    CHECK(result == SIM_RESULT_OK, "real pow evaluator failed");
    CHECK(fabs(value - 3.0) < 1.0e-12, "real pow value mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_evaluate_pow_complex(void) {
    printf("[TEST START] test_ir_evaluate_pow_complex\n");
    fflush(stdout);
    SimIRBuilder          builder;
    SimResult             result;
    SimIRNodeId           base_node;
    SimIRNodeId           exponent_node;
    SimIRNodeId           pow_node;
    SimComplexDouble      value     = { 0.0, 0.0 };
    SimIREvaluatorComplex evaluator = { 0 };

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    base_node     = sim_ir_builder_constant_complex(&builder, 1.0, 1.0);
    exponent_node = sim_ir_builder_constant_complex(&builder, 2.0, 0.0);
    pow_node      = sim_ir_builder_binary(&builder, SIM_IR_NODE_POW, base_node, exponent_node);
    CHECK(pow_node != SIM_IR_INVALID_NODE, "complex pow node creation failed");

    result = sim_ir_evaluate_complex(&builder, pow_node, &evaluator, &value);
    CHECK(result == SIM_RESULT_OK, "complex pow evaluator failed");
    CHECK(fabs(value.re) < 1.0e-12, "complex pow real mismatch");
    CHECK(fabs(value.im - 2.0) < 1.0e-12, "complex pow imag mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_evaluate_complex_pack(void) {
    printf("[TEST START] test_ir_evaluate_complex_pack\n");
    fflush(stdout);
    SimIRBuilder          builder;
    SimResult             result;
    SimIRNodeId           re_node;
    SimIRNodeId           im_node;
    SimIRNodeId           pack_node;
    SimComplexDouble      value     = { 0.0, 0.0 };
    SimIREvaluatorComplex evaluator = { 0 };

    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    re_node   = sim_ir_builder_constant(&builder, -2.0);
    im_node   = sim_ir_builder_constant(&builder, 0.75);
    pack_node = sim_ir_builder_complex_pack(&builder, re_node, im_node);
    CHECK(pack_node != SIM_IR_INVALID_NODE, "complex pack node creation failed");

    result = sim_ir_evaluate_complex(&builder, pack_node, &evaluator, &value);
    CHECK(result == SIM_RESULT_OK, "complex pack evaluator failed");
    CHECK(fabs(value.re + 2.0) < 1.0e-12, "complex pack real mismatch");
    CHECK(fabs(value.im - 0.75) < 1.0e-12, "complex pack imag mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_ir_constant_scalar_broadcast(void) {
    printf("[TEST START] test_ir_constant_scalar_broadcast\n");
    fflush(stdout);
    SimIRBuilder     builder;
    SimResult        result;
    SimIRNodeId      node;
    const SimIRNode* nptr;
    const double     scalar = 2.5;

    printf("[TEST START] test_kernel_builder_copy\n");
    fflush(stdout);
    result = sim_ir_builder_init(&builder);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    printf("[TEST START] test_kernel_builder_copy_mutation_inline\n");
    fflush(stdout);
    node = sim_ir_builder_constant_typed(&builder, scalar, sim_ir_type_vector(3U));
    CHECK(node != SIM_IR_INVALID_NODE, "typed scalar vector node creation failed");

    printf("[TEST START] test_kernel_builder_copy_mutation_pool\n");
    fflush(stdout);
    nptr = sim_ir_builder_get(&builder, node);
    CHECK(nptr != NULL, "typed scalar node pointer NULL");

    CHECK(nptr->data.constant.constant_index == SIM_IR_INVALID_CONSTANT_INDEX,
          "scalar broadcast unexpectedly has pool index");
    CHECK(fabs(nptr->data.constant.scalar - scalar) < 1e-12, "scalar broadcast value mismatch");

    sim_ir_builder_destroy(&builder);
    return true;
}

static bool test_mixer_kernel_registration(void) {
    SimContext             context  = { 0 };
    SimField               lhs      = { 0 };
    SimField               rhs      = { 0 };
    SimField               out      = { 0 };
    size_t                 shape[1] = { 4U };
    SimResult              result;
    size_t                 lhs_idx = 0U, rhs_idx = 0U, out_idx = 0U;
    size_t                 op_index = 0U;
    SimMixerOperatorConfig cfg      = { 0 };

    result = sim_context_init(&context);
    CHECK(result == SIM_RESULT_OK, "context init failed");

    /* Create complex fields all with 2 components and expect kernel registration */
    result =
        sim_field_init(&lhs, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "lhs complex field init failed");
    result =
        sim_field_init(&rhs, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "rhs complex field init failed");
    result =
        sim_field_init(&out, 1U, shape, sizeof(double) * 2U, SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "out complex field init failed");

    result = sim_context_add_field(&context, &lhs, &lhs_idx);
    CHECK(result == SIM_RESULT_OK, "add lhs failed");
    result = sim_context_add_field(&context, &rhs, &rhs_idx);
    CHECK(result == SIM_RESULT_OK, "add rhs failed");
    result = sim_context_add_field(&context, &out, &out_idx);
    CHECK(result == SIM_RESULT_OK, "add out failed");

    CHECK(sim_field_components(sim_context_field(&context, lhs_idx)) == 2U,
          "lhs components mismatch");
    CHECK(sim_field_components(sim_context_field(&context, rhs_idx)) == 2U,
          "rhs components mismatch");
    CHECK(sim_field_components(sim_context_field(&context, out_idx)) == 2U,
          "out components mismatch");

    cfg.lhs_field    = lhs_idx;
    cfg.rhs_field    = rhs_idx;
    cfg.output_field = out_idx;
    cfg.lhs_gain     = 1.0;
    cfg.rhs_gain     = 1.0;
    cfg.mix          = 0.5;
    cfg.bias         = 0.0;
    cfg.mode         = SIM_MIXER_MODE_LINEAR;
    cfg.accumulate   = false;

    /* Configure a CPU backend so kernel-backed registration is allowed */
    SimBackend backend = { 0 };
    backend.type       = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    CHECK(backend.last_error == SIM_RESULT_OK, "backend init failed");
    sim_context_set_backend(&context, &backend);

    /* Exercise the guarded legacy kernel path rather than the default split-only fallback. */
    test_set_env_text("OAKFIELD_ENABLE_EXPERIMENTAL_LEGACY_KERNELS", "1");

    result = sim_add_mixer_operator(&context, &cfg, &op_index);
    CHECK(result == SIM_RESULT_OK, "mixer registration failed");
    SimOperator* op = sim_operator_registry_get(&context.world.operators, op_index);
    CHECK(op != NULL, "operator not found");
    CHECK(op->kernel != NULL, "mixer operator should have kernel for matching component fields");

    /* Now test fallback: mix scalar and complex fields -> should not register kernel */
    SimField scalar = { 0 };
    result = sim_field_init(&scalar, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    CHECK(result == SIM_RESULT_OK, "scalar init failed");

    size_t scalar_idx = 0U;
    result            = sim_context_add_field(&context, &scalar, &scalar_idx);
    CHECK(result == SIM_RESULT_OK, "add scalar failed");

    cfg.lhs_field    = scalar_idx; /* scalar */
    cfg.rhs_field    = rhs_idx;    /* complex */
    cfg.output_field = out_idx;    /* complex */

    size_t op_count_before = sim_context_operator_count(&context);
    result                 = sim_add_mixer_operator(&context, &cfg, &op_index);
    CHECK(result == SIM_RESULT_TYPE_MISMATCH, "mixer should reject complex requirement mismatch");
    CHECK(sim_context_operator_count(&context) == op_count_before,
          "mixer fallback should not register operator");

    test_set_env_text("OAKFIELD_ENABLE_EXPERIMENTAL_LEGACY_KERNELS", NULL);
    backend_destroy(&backend);
    sim_context_destroy(&context);
    return true;
}

int main(void) {
    int failures = 0;

    if (!test_field_allocation()) {
        failures += 1;
    }

    if (!test_field_index_mapping_2d()) {
        failures += 1;
    }

    if (!test_kernel_binding_index_mapping_2d()) {
        failures += 1;
    }

    if (!test_ir_semantics_metadata()) {
        failures += 1;
    }

    if (!test_gradient_operator()) {
        failures += 1;
    }

    if (!test_context_kernel_operator()) {
        failures += 1;
    }

    if (!test_context_kernel_operator_complex()) {
        failures += 1;
    }

    if (!test_context_kernel_operator_cuda()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_cuda_nested()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_cuda_warp()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_cuda_vector_constant()) {
        failures += 1;
    }

    if (!test_context_kernel_operator_metal()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_metal_nested()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_metal_diff()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_metal_noise()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_metal_warp()) {
        failures += 1;
    }

    if (!test_context_kernel_operator_warp_guarded()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_metal_vector_constant()) {
        failures += 1;
    }
    if (!test_context_kernel_operator_metal_multi_fields()) {
        failures += 1;
    }
    if (!test_context_metal_pipeline_refcount()) {
        failures += 1;
    }

    if (!test_field_complex_allocation()) {
        failures += 1;
    }

    if (!test_field_require_complex_promotion()) {
        failures += 1;
    }

    if (!test_field_require_complex_rejects_incompatible()) {
        failures += 1;
    }

    if (!test_field_representation_unknown_rejected()) {
        failures += 1;
    }

    if (!test_ir_field_ref_from_field()) {
        failures += 1;
    }

    if (!test_mixer_kernel_registration()) {
        failures += 1;
    }

    if (failures == 0) {
        printf("All field tests passed.\n");
        fflush(stdout);
        fprintf(stderr, "[TESTGROUP] Starting IR-level tests\n");
    }

    /* IR constant tests */
    fprintf(stderr, "[TEST] test_ir_constant_vector_inline\n");
    if (!test_ir_constant_vector_inline()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_constant_vector_pool\n");
    if (!test_ir_constant_vector_pool()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_constant_vector_dedup\n");
    if (!test_ir_constant_vector_dedup()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_constant_scalar_broadcast\n");
    if (!test_ir_constant_scalar_broadcast()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_mathview_render\n");
    if (!test_ir_mathview_render()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_evaluate_complex_constant_inline\n");
    if (!test_ir_evaluate_complex_constant_inline()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_evaluate_complex_constant_pool\n");
    if (!test_ir_evaluate_complex_constant_pool()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_evaluate_pow_real\n");
    if (!test_ir_evaluate_pow_real()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_evaluate_pow_complex\n");
    if (!test_ir_evaluate_pow_complex()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_ir_evaluate_complex_pack\n");
    if (!test_ir_evaluate_complex_pack()) {
        failures += 1;
    }

    fprintf(stderr, "[TEST] test_kernel_builder_copy\n");
    if (!test_kernel_builder_copy()) {
        failures += 1;
    }

    fprintf(stderr, "[TEST] test_kernel_builder_copy_mutation_inline\n");
    if (!test_kernel_builder_copy_mutation_inline()) {
        failures += 1;
    }
    fprintf(stderr, "[TEST] test_kernel_builder_copy_mutation_pool\n");
    if (!test_kernel_builder_copy_mutation_pool()) {
        failures += 1;
    }

    return (failures == 0) ? 0 : 1;
}
