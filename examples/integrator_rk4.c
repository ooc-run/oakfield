#include <oakfield/field.h>
#include <oakfield/integrator.h>
#include <oakfield/integrator_registry.h>

#include <math.h>
#include <stdio.h>

static int fail_result(const char* label, SimResult result) {
    fprintf(stderr, "%s failed (%d)\n", label, result);
    return 1;
}

static SimResult decay_drift(Integrator*   integrator,
                             const Field*  field,
                             const double* state,
                             double*       out_derivative,
                             size_t        count) {
    (void) field;
    double rate = (integrator != NULL && integrator->userdata != NULL)
                      ? *(const double*) integrator->userdata
                      : 1.0;
    for (size_t i = 0U; i < count; ++i) {
        out_derivative[i] = -rate * state[i];
    }
    return SIM_RESULT_OK;
}

int main(void) {
    SimField field = { 0 };
    size_t   shape[1] = { 4U };
    double   rate = 0.75;
    double   dt = 0.1;
    int      steps = 10;
    SimResult result = sim_field_init(
        &field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        return fail_result("field init", result);
    }

    double* data = sim_field_real_data(&field);
    if (data == NULL) {
        sim_field_destroy(&field);
        return fail_result("field data", SIM_RESULT_INVALID_STATE);
    }
    for (size_t i = 0U; i < shape[0]; ++i) {
        data[i] = (double) (i + 1U);
    }

    IntegratorRegistry registry = { 0 };
    result = integrator_registry_init(&registry);
    if (result != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return fail_result("integrator_registry_init", result);
    }

    IntegratorConfig config = {
        .drift = decay_drift,
        .userdata = &rate,
        .initial_dt = dt,
        .min_dt = dt,
        .max_dt = dt,
        .tolerance = 1.0e-8,
        .safety = 0.9,
        .adaptive = false,
        .workspace_hint = shape[0],
    };
    Integrator rk4 = { 0 };
    result = integrator_registry_create(&registry, "rk4", &config, &rk4);
    if (result != SIM_RESULT_OK) {
        integrator_registry_destroy(&registry);
        sim_field_destroy(&field);
        return fail_result("integrator_registry_create(rk4)", result);
    }

    for (int i = 0; i < steps; ++i) {
        rk4.step(&rk4, &field, dt);
    }

    double expected0 = exp(-rate * dt * (double) steps);
    printf("rk4 decay after %d steps: value0=%.8f expected0=%.8f last_dt=%.3f\n",
           steps,
           data[0],
           expected0,
           integrator_last_step(&rk4));

    integrator_destroy(&rk4);
    integrator_registry_destroy(&registry);
    sim_field_destroy(&field);
    return 0;
}
