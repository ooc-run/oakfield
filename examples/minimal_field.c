#include <oakfield/field.h>

#include <stdio.h>

static int fail_result(const char* label, SimResult result) {
    fprintf(stderr, "%s failed (%d)\n", label, result);
    return 1;
}

int main(void) {
    SimField field     = { 0 };
    size_t   shape[2]  = { 2U, 3U };
    size_t   index[2]  = { 1U, 2U };
    size_t   element   = 0U;
    SimResult result   = sim_field_init(
        &field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);

    if (result != SIM_RESULT_OK) {
        return fail_result("sim_field_init", result);
    }

    double* data = sim_field_real_data(&field);
    if (data == NULL) {
        sim_field_destroy(&field);
        return fail_result("sim_field_real_data", SIM_RESULT_INVALID_STATE);
    }

    for (size_t i = 0U; i < sim_field_element_count(&field.layout); ++i) {
        data[i] = (double) i + 0.25;
    }

    result = sim_field_element_index(&field, index, 2U, &element);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return fail_result("sim_field_element_index", result);
    }

    printf("field rank=%zu shape=%zux%zu strides=%zu,%zu bytes=%zu domain=%s\n",
           sim_field_rank(&field),
           sim_field_shape(&field)[0],
           sim_field_shape(&field)[1],
           sim_field_strides(&field)[0],
           sim_field_strides(&field)[1],
           sim_field_bytes(&field),
           sim_scalar_domain_name(sim_field_scalar_domain(&field)));
    printf("field[1,2] element=%zu value=%.2f\n", element, data[element]);

    result = sim_field_promote_inplace_to_complex(&field);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return fail_result("sim_field_promote_inplace_to_complex", result);
    }

    SimComplexDouble* complex_data = sim_field_complex_data(&field);
    if (complex_data == NULL) {
        sim_field_destroy(&field);
        return fail_result("sim_field_complex_data", SIM_RESULT_INVALID_STATE);
    }
    complex_data[element].im = -0.5;

    printf("promoted domain=%s field[1,2]=(%.2f, %.2f)\n",
           sim_scalar_domain_name(sim_field_scalar_domain(&field)),
           complex_data[element].re,
           complex_data[element].im);

    sim_field_destroy(&field);
    return 0;
}
