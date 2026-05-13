#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static bool approx_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= tol * scale;
}

static bool execute_sdr_case(const SimSdrObserverConfig *config, size_t sample_count, double dt,
                             SimComplexDouble *out_values, SimOperatorInfo *out_info) {
    size_t shape[1] = {sample_count};
    SimContext context = {0};
    SimField field = {0};
    bool context_ready = false;
    bool field_ready = false;
    size_t field_index = SIZE_MAX;
    size_t op_index = SIZE_MAX;
    bool success = false;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        goto cleanup;
    }
    field_ready = true;
    memset(sim_field_complex_data(&field), 0, sample_count * sizeof(SimComplexDouble));

    if (sim_context_add_field(&context, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        goto cleanup;
    }
    field_ready = false;

    sim_context_set_timestep(&context, dt);

    SimSdrObserverConfig local = *config;
    local.field_index = field_index;
    if (sim_add_sdr_observer_operator(&context, &local, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_sdr_observer_operator\n");
        goto cleanup;
    }

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_execute\n");
        goto cleanup;
    }

    if (out_values != NULL) {
        const SimField *out_field = sim_context_field(&context, field_index);
        const SimComplexDouble *values = sim_field_complex_data_const(out_field);
        memcpy(out_values, values, sample_count * sizeof(SimComplexDouble));
    }

    if (out_info != NULL) {
        SimOperator *op = sim_operator_registry_get(&context.world.operators, op_index);
        if (op == NULL) {
            fprintf(stderr, "FAIL: sim_operator_registry_get\n");
            goto cleanup;
        }
        *out_info = op->info;
    }

    success = true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (field_ready) {
        sim_field_destroy(&field);
    }
    return success;
}

static bool expect_output(const char *label, const SimComplexDouble *actual,
                          const SimComplexDouble *expected, size_t count, double tol) {
    for (size_t i = 0U; i < count; ++i) {
        if (!approx_equal(actual[i].re, expected[i].re, tol) ||
            !approx_equal(actual[i].im, expected[i].im, tol)) {
            fprintf(stderr, "FAIL: %s index %zu got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                    label, i, actual[i].re, actual[i].im, expected[i].re, expected[i].im);
            return false;
        }
    }
    return true;
}

static bool run_raw_synth_case(void) {
    enum { sample_count = 4 };
    const double root_half = sqrt(0.5);
    SimComplexDouble values[sample_count];
    SimComplexDouble expected[sample_count] = {
        {.re = 2.0, .im = 0.0},
        {.re = 2.0 * root_half, .im = 2.0 * root_half},
        {.re = 0.0, .im = 2.0},
        {.re = -2.0 * root_half, .im = 2.0 * root_half},
    };
    SimOperatorInfo info = {0};
    SimSdrObserverConfig config;

    memset(&config, 0, sizeof(config));
    config.sample_rate = 8.0;
    config.freq_offset = 1.0;
    config.amplitude = 2.0;
    config.demod = SIM_SDR_OBSERVER_DEMOD_RAW;
    config.device_index = 9999;

    if (!execute_sdr_case(&config, sample_count, 1.0 / config.sample_rate, values, &info)) {
        return false;
    }

    if (info.category != SIM_OPERATOR_CATEGORY_MEASUREMENT || info.is_noise) {
        fprintf(stderr, "FAIL: operator metadata category/is_noise mismatch\n");
        return false;
    }

    return expect_output("raw_synth", values, expected, sample_count, 1.0e-9);
}

static bool run_am_synth_case(void) {
    enum { sample_count = 4 };
    SimComplexDouble values[sample_count];
    SimComplexDouble expected[sample_count] = {
        {.re = 1.5, .im = 0.0},
        {.re = 1.5, .im = 0.0},
        {.re = 1.5, .im = 0.0},
        {.re = 1.5, .im = 0.0},
    };
    SimSdrObserverConfig config;

    memset(&config, 0, sizeof(config));
    config.sample_rate = 8.0;
    config.freq_offset = 1.0;
    config.amplitude = 1.5;
    config.demod = SIM_SDR_OBSERVER_DEMOD_AM;
    config.device_index = 9999;

    if (!execute_sdr_case(&config, sample_count, 1.0 / config.sample_rate, values, NULL)) {
        return false;
    }

    return expect_output("am_synth", values, expected, sample_count, 1.0e-9);
}

static bool run_fm_synth_case(void) {
    enum { sample_count = 4 };
    SimComplexDouble values[sample_count];
    SimComplexDouble expected[sample_count] = {
        {.re = 2.0, .im = 0.0},
        {.re = 2.0, .im = 0.0},
        {.re = 2.0, .im = 0.0},
        {.re = 2.0, .im = 0.0},
    };
    SimSdrObserverConfig config;

    memset(&config, 0, sizeof(config));
    config.sample_rate = 16.0;
    config.freq_offset = 2.0;
    config.amplitude = 1.0;
    config.demod = SIM_SDR_OBSERVER_DEMOD_FM;
    config.device_index = 9999;

    if (!execute_sdr_case(&config, sample_count, 1.0 / config.sample_rate, values, NULL)) {
        return false;
    }

    return expect_output("fm_synth", values, expected, sample_count, 1.0e-9);
}

static bool run_pm_synth_case(void) {
    enum { sample_count = 4 };
    SimComplexDouble values[sample_count];
    SimComplexDouble expected[sample_count] = {
        {.re = 0.0, .im = 0.0},
        {.re = M_PI / 4.0, .im = 0.0},
        {.re = M_PI / 2.0, .im = 0.0},
        {.re = 3.0 * M_PI / 4.0, .im = 0.0},
    };
    SimSdrObserverConfig config;

    memset(&config, 0, sizeof(config));
    config.sample_rate = 8.0;
    config.freq_offset = 1.0;
    config.amplitude = 1.0;
    config.demod = SIM_SDR_OBSERVER_DEMOD_PM;
    config.device_index = 9999;

    if (!execute_sdr_case(&config, sample_count, 1.0 / config.sample_rate, values, NULL)) {
        return false;
    }

    return expect_output("pm_synth", values, expected, sample_count, 1.0e-9);
}

int main(void) {
    if (!run_raw_synth_case() || !run_am_synth_case() || !run_fm_synth_case() ||
        !run_pm_synth_case()) {
        return 1;
    }
    return 0;
}
