/*
 * Adapted from the base field/scalar-domain tests for the standalone 
 * package. This smoke test intentionally avoids Lua, app runtime, and optional
 * backend dependencies.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "[FAIL] %s\n", msg);                                                   \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

static bool near_double(double lhs, double rhs) {
    return fabs(lhs - rhs) <= 1.0e-12;
}

static bool test_field_layout_and_promotion(void) {
    SimField field         = { 0 };
    size_t   shape[2]      = { 2U, 3U };
    size_t   indices[2]    = { 1U, 2U };
    size_t   offset        = 0U;
    size_t   element_index = 0U;

    CHECK(sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) ==
              SIM_RESULT_OK,
          "sim_field_init failed");
    CHECK(sim_field_rank(&field) == 2U, "field rank mismatch");
    CHECK(sim_field_shape(&field)[0] == 2U && sim_field_shape(&field)[1] == 3U,
          "field shape mismatch");
    CHECK(sim_field_strides(&field)[0] == 3U && sim_field_strides(&field)[1] == 1U,
          "field strides mismatch");
    CHECK(sim_field_bytes(&field) == 6U * sizeof(double), "field byte size mismatch");
    CHECK(sim_scalar_domain_equal(sim_field_scalar_domain(&field), sim_scalar_domain_f64()),
          "real field scalar domain mismatch");

    double* real_data = sim_field_real_data(&field);
    CHECK(real_data != NULL, "real field data unavailable");
    for (size_t i = 0; i < 6U; ++i) {
        real_data[i] = (double) i + 0.25;
    }

    CHECK(sim_field_element_index(&field, indices, 2U, &element_index) == SIM_RESULT_OK,
          "field element index failed");
    CHECK(element_index == 5U, "field element index mismatch");
    CHECK(sim_field_index_offset(&field, indices, &offset) == SIM_RESULT_OK,
          "field byte offset failed");
    CHECK(offset == 5U * sizeof(double), "field byte offset mismatch");

    CHECK(sim_field_promote_inplace_to_complex(&field) == SIM_RESULT_OK,
          "field promotion to complex failed");
    CHECK(sim_field_storage_is_complex(&field), "promoted field storage is not complex");
    CHECK(sim_field_is_complex(&field), "promoted field semantic complex flag mismatch");
    CHECK(sim_scalar_domain_equal(sim_field_scalar_domain(&field), sim_scalar_domain_c64()),
          "promoted field scalar domain mismatch");

    const SimComplexDouble* complex_data = sim_field_complex_data_const(&field);
    CHECK(complex_data != NULL, "complex field data unavailable");
    for (size_t i = 0; i < 6U; ++i) {
        CHECK(near_double(complex_data[i].re, (double) i + 0.25),
              "promoted field real component mismatch");
        CHECK(near_double(complex_data[i].im, 0.0), "promoted field imaginary component mismatch");
    }

    sim_field_destroy(&field);
    return true;
}

static bool test_typed_field_and_representation_helpers(void) {
    SimField field    = { 0 };
    size_t   shape[1] = { 4U };

    CHECK(sim_field_init_typed(
              &field, 1U, shape, sim_scalar_domain_i32(), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) ==
              SIM_RESULT_OK,
          "typed i32 field init failed");
    CHECK(sim_field_i32_data(&field) != NULL, "typed i32 data unavailable");
    CHECK(sim_scalar_domain_equal(sim_field_scalar_domain(&field), sim_scalar_domain_i32()),
          "typed field scalar domain mismatch");
    CHECK(strcmp(sim_scalar_domain_name(sim_field_scalar_domain(&field)), "i32") == 0,
          "typed field scalar domain name mismatch");

    SimFieldRepresentation repr = {
        .domain     = SIM_FIELD_DOMAIN_PHYSICAL,
        .value_kind = SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT,
    };
    CHECK(sim_field_representation_requires_complex_storage(repr),
          "imag-zero representation should require complex storage");
    CHECK(sim_field_representation_has_imag_zero_constraint(repr),
          "imag-zero representation helper mismatch");

    sim_field_destroy(&field);
    return true;
}

static bool test_buffer_views(void) {
    size_t     shape[2] = { 2U, 2U };
    SimBuffer* buffer   = sim_buffer_create(2U, shape, SIM_BUFFER_DOUBLE);
    CHECK(buffer != NULL, "buffer create failed");
    CHECK(sim_buffer_count(buffer) == 4U, "buffer count mismatch");
    CHECK(sim_buffer_bytes(buffer) == 4U * sizeof(double), "buffer byte size mismatch");
    CHECK(sim_buffer_type(buffer) == SIM_BUFFER_DOUBLE, "buffer type mismatch");

    SimBufferView view = sim_buffer_view(buffer);
    CHECK(sim_buffer_view_is_valid(&view), "buffer view invalid");
    CHECK(sim_buffer_view_set_complex(&view, 2U, (SimComplexDouble){ 4.5, 0.0 }),
          "real buffer write failed");
    CHECK(!sim_buffer_view_set_complex(&view, 3U, (SimComplexDouble){ 1.0, 0.5 }),
          "real buffer accepted nonzero imaginary write");

    SimComplexDouble value = { 0.0, 0.0 };
    CHECK(sim_buffer_view_get_complex(&view, 2U, &value), "buffer read failed");
    CHECK(near_double(value.re, 4.5) && near_double(value.im, 0.0), "buffer read mismatch");

    size_t indices[2] = { 1U, 0U };
    size_t offset     = 0U;
    CHECK(sim_buffer_view_offset_for_indices(&view, indices, 2U, &offset),
          "buffer multidimensional offset failed");
    CHECK(offset == 2U, "buffer multidimensional offset mismatch");

    CHECK(sim_buffer_reshape(buffer, 1U, (size_t[]){ 4U }) == SIM_RESULT_OK,
          "buffer reshape failed");
    const SimFieldLayout* layout = sim_buffer_layout(buffer);
    CHECK(layout != NULL && layout->rank == 1U && layout->shape[0] == 4U,
          "buffer reshape layout mismatch");

    sim_buffer_destroy(buffer);
    return true;
}

int main(void) {
    bool ok = true;
    ok      = test_field_layout_and_promotion() && ok;
    ok      = test_typed_field_and_representation_helpers() && ok;
    ok      = test_buffer_views() && ok;
    return ok ? 0 : 1;
}
