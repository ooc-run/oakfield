/*
 * Exercises SimContext-owned special-function fallback hooks and fault
 * accounting for domain/error cases in the safe math entry points.
 */
#include <oakfield/sim_context.h>
#include <oakfield/math/special_functions.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TestFallbackData {
    int calls;
    SimComplexDouble value;
    SimResult result;
} TestFallbackData;

static SimResult test_custom_fallback(void *userdata, const SimSpecialEvalReport *report,
                                      SimComplexDouble *value_out) {
    (void)report;
    TestFallbackData *data = (TestFallbackData *)userdata;
    if (data != NULL) {
        data->calls += 1;
        if (value_out != NULL) {
            *value_out = data->value;
        }
        return (data->result == SIM_RESULT_OK) ? SIM_RESULT_OK : data->result;
    }

    if (value_out != NULL) {
        value_out->re = 0.0;
        value_out->im = 0.0;
    }
    return SIM_RESULT_OK;
}

int main(void) {
    SimContext context;
    int status = EXIT_SUCCESS;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_init failed\n");
        return EXIT_FAILURE;
    }

    SimSpecialFallbackFn fallback = NULL;
    void *userdata = NULL;
    sim_context_special_fallback_hook(&context, &fallback, &userdata);

    SimSpecialEvalReport report = {0};
    double value = -1.0;
    SimResult rc = sim_q_zeta_safe(0.95, 1.0, 0.8, fallback, userdata, &report, &value);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "default fallback returned rc=%d\n", rc);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (fabs(value) > 1e-12) {
        fprintf(stderr, "default fallback produced %.12f\n", value);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (sim_context_special_fault_count(&context) != 1ULL) {
        fprintf(stderr, "expected 1 fault, got %llu\n",
                (unsigned long long)sim_context_special_fault_count(&context));
        status = EXIT_FAILURE;
        goto cleanup;
    }
    {
        SimSpecialEvalReport stored;
        if (!sim_context_last_special_fault(&context, &stored)) {
            fprintf(stderr, "failed to retrieve last fault\n");
            status = EXIT_FAILURE;
            goto cleanup;
        }
        if (stored.fault != SIM_SPECIAL_FAULT_DOMAIN || stored.function == NULL ||
            strcmp(stored.function, "sim_q_zeta") != 0) {
            fprintf(stderr, "unexpected fault metadata\n");
            status = EXIT_FAILURE;
            goto cleanup;
        }
    }

    TestFallbackData data = {
        .calls = 0, .value = {.re = 42.0, .im = -3.0}, .result = SIM_RESULT_OK};

    sim_context_set_special_fallback(&context, test_custom_fallback, &data);
    sim_context_special_fallback_hook(&context, &fallback, &userdata);

    report = (SimSpecialEvalReport){0};
    value = 0.0;
    rc = sim_q_zeta_safe(0.95, 1.0, 0.8, fallback, userdata, &report, &value);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "custom fallback returned rc=%d\n", rc);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (fabs(value - data.value.re) > 1e-12) {
        fprintf(stderr, "custom fallback value mismatch: %.12f\n", value);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (data.calls != 1) {
        fprintf(stderr, "custom fallback expected 1 call, got %d\n", data.calls);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (sim_context_special_fault_count(&context) != 2ULL) {
        fprintf(stderr, "expected 2 faults, got %llu\n",
                (unsigned long long)sim_context_special_fault_count(&context));
        status = EXIT_FAILURE;
        goto cleanup;
    }

    report = (SimSpecialEvalReport){0};
    double digamma_value = 0.0;
    rc = sim_digamma_safe(0.0, fallback, userdata, &report, &digamma_value);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "sim_digamma_safe returned rc=%d\n", rc);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (fabs(digamma_value - data.value.re) > 1e-12) {
        fprintf(stderr, "sim_digamma_safe fallback mismatch: %.12f\n", digamma_value);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (data.calls != 2) {
        fprintf(stderr, "custom fallback expected 2 calls after digamma, got %d\n", data.calls);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (sim_context_special_fault_count(&context) != 3ULL) {
        fprintf(stderr, "expected 3 faults after digamma, got %llu\n",
                (unsigned long long)sim_context_special_fault_count(&context));
        status = EXIT_FAILURE;
        goto cleanup;
    }

    report = (SimSpecialEvalReport){0};
    double hyperexp_value = 0.0;
    rc = sim_hyperexp_phi_deriv_safe(-1.0, 1.0, 2, fallback, userdata, &report, &hyperexp_value);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "sim_hyperexp_phi_deriv_safe returned rc=%d\n", rc);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (fabs(hyperexp_value - data.value.re) > 1e-12) {
        fprintf(stderr, "sim_hyperexp_phi_deriv_safe fallback mismatch: %.12f\n", hyperexp_value);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (data.calls != 3) {
        fprintf(stderr, "custom fallback expected 3 calls after hyperexp, got %d\n", data.calls);
        status = EXIT_FAILURE;
        goto cleanup;
    }
    if (sim_context_special_fault_count(&context) != 4ULL) {
        fprintf(stderr, "expected 4 faults after hyperexp, got %llu\n",
                (unsigned long long)sim_context_special_fault_count(&context));
        status = EXIT_FAILURE;
        goto cleanup;
    }

cleanup:
    sim_context_destroy(&context);
    return status;
}
