#include <oakfield/field.h>
#include <oakfield/integrator.h>
#include <oakfield/integrator_registry.h>

#include <math.h>
#include <stdio.h>

static int fail_result(const char* label, SimResult result) {
    fprintf(stderr, "%s failed (%d)\n", label, result);
    return 1;
}

static SimResult zero_drift(Integrator*   integrator,
                            const Field*  field,
                            const double* state,
                            double*       out_derivative,
                            size_t        count) {
    (void) integrator;
    (void) field;
    (void) state;
    for (size_t i = 0U; i < count; ++i) {
        out_derivative[i] = 0.0;
    }
    return SIM_RESULT_OK;
}

static SimResult init_real_field(SimField* field, size_t count) {
    size_t shape[1] = { count };
    SimResult result =
        sim_field_init(field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    double* data = sim_field_real_data(field);
    if (data == NULL) {
        sim_field_destroy(field);
        return SIM_RESULT_INVALID_STATE;
    }
    for (size_t i = 0U; i < count; ++i) {
        data[i] = 0.0;
    }
    return SIM_RESULT_OK;
}

int main(void) {
    enum { ELEMENTS = 8, STEPS = 3 };
    SimField first = { 0 };
    SimField second = { 0 };
    SimResult result = init_real_field(&first, ELEMENTS);
    if (result != SIM_RESULT_OK) {
        return fail_result("first field init", result);
    }
    result = init_real_field(&second, ELEMENTS);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&first);
        return fail_result("second field init", result);
    }

    IntegratorRegistry registry = { 0 };
    result = integrator_registry_init(&registry);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&second);
        sim_field_destroy(&first);
        return fail_result("integrator_registry_init", result);
    }

    IntegratorConfig config = {
        .drift = zero_drift,
        .noise = integrator_noise_gaussian,
        .initial_dt = 0.05,
        .min_dt = 0.05,
        .max_dt = 0.05,
        .adaptive = false,
        .enable_stochastic = true,
        .stochastic_strength = 0.25,
        .random_seed = 12345U,
        .workspace_hint = ELEMENTS,
    };
    Integrator a = { 0 };
    Integrator b = { 0 };
    result = integrator_registry_create(&registry, "euler", &config, &a);
    if (result != SIM_RESULT_OK) {
        integrator_registry_destroy(&registry);
        sim_field_destroy(&second);
        sim_field_destroy(&first);
        return fail_result("create first euler", result);
    }
    result = integrator_registry_create(&registry, "euler", &config, &b);
    if (result != SIM_RESULT_OK) {
        integrator_destroy(&a);
        integrator_registry_destroy(&registry);
        sim_field_destroy(&second);
        sim_field_destroy(&first);
        return fail_result("create second euler", result);
    }

    for (int step = 0; step < STEPS; ++step) {
        a.step(&a, &first, 0.05);
        b.step(&b, &second, 0.05);
    }

    const double* first_data = sim_field_real_data_const(&first);
    const double* second_data = sim_field_real_data_const(&second);
    double max_abs_diff = 0.0;
    for (size_t i = 0U; i < ELEMENTS; ++i) {
        double diff = fabs(first_data[i] - second_data[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    printf("seeded stochastic euler max_abs_diff=%.3g sample0=%.8f sample1=%.8f\n",
           max_abs_diff,
           first_data[0],
           first_data[1]);

    integrator_destroy(&b);
    integrator_destroy(&a);
    integrator_registry_destroy(&registry);
    sim_field_destroy(&second);
    sim_field_destroy(&first);
    return (max_abs_diff <= 1.0e-15) ? 0 : 1;
}
