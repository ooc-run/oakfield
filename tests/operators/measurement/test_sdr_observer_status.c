#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool expect(bool condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

static bool expect_close(double actual, double expected, double tol, const char* message) {
    if (fabs(actual - expected) > tol) {
        fprintf(stderr, "FAIL: %s (actual=%.12f expected=%.12f)\n", message, actual, expected);
        return false;
    }
    return true;
}

int main(void) {
    size_t     shape[1]    = { 8U };
    SimContext context     = { 0 };
    SimField   field       = { 0 };
    size_t     field_index = SIZE_MAX;
    size_t     op_index    = SIZE_MAX;
    bool       ok          = true;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 1;
    }

    if (sim_field_init(
            &field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&context);
        return 1;
    }

    if (sim_context_add_field(&context, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&context);
        return 1;
    }

    SimSdrObserverConfig config;
    memset(&config, 0, sizeof(config));
    config.field_index  = field_index;
    config.center_freq  = 101.7e6;
    config.freq_offset  = 125.0e3;
    config.sample_rate  = 2.048e6;
    config.demod        = SIM_SDR_OBSERVER_DEMOD_RAW;
    config.device_index = 9999;

    if (sim_add_sdr_observer_operator(&context, &config, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_sdr_observer_operator\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimSdrObserverStatus status;
    memset(&status, 0, sizeof(status));
    if (sim_sdr_observer_status(&context, op_index, &status) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_sdr_observer_status initial\n");
        sim_context_destroy(&context);
        return 1;
    }

    ok = expect(status.active_backend == SIM_SDR_OBSERVER_BACKEND_SYNTHETIC,
                "initial backend should be synthetic") &&
         ok;
    ok = expect(status.using_synthetic, "initial using_synthetic should be true") && ok;
    ok = expect(!status.device_open, "device should not be open after invalid open attempt") && ok;
    ok = expect(status.has_last_error, "status should report sticky last error") && ok;
    ok = expect(!status.has_successful_read, "status should report no successful reads yet") && ok;
    ok = expect(status.fallback_count == 1U, "initial fallback count should be 1") && ok;
    ok = expect(status.last_read_iq_bytes == 0U, "initial read byte count should be 0") && ok;
    ok = expect_close(status.effective_tuned_freq,
                      config.center_freq + config.freq_offset,
                      1.0e-9,
                      "initial effective tuned frequency should reflect center + offset") &&
         ok;
    ok = expect(status.last_error_message[0] != '\0', "initial error message should be present") &&
         ok;
#if OAKFIELD_ENABLE_RTLSDR
    ok = expect(status.rtl_sdr_enabled, "rtl_sdr_enabled should be true when compiled in") && ok;
    ok = expect(status.last_fallback_reason == SIM_SDR_OBSERVER_FALLBACK_DEVICE_OPEN_FAILED,
                "fallback reason should report device open failure") &&
         ok;
    ok = expect(status.last_error_code == SIM_RESULT_NOT_FOUND,
                "last error code should report not_found for invalid device") &&
         ok;
#else
    ok = expect(!status.rtl_sdr_enabled, "rtl_sdr_enabled should be false without RTL-SDR") && ok;
    ok = expect(status.last_fallback_reason == SIM_SDR_OBSERVER_FALLBACK_RTLSDR_DISABLED,
                "fallback reason should report disabled RTL-SDR support") &&
         ok;
    ok = expect(status.last_error_code == SIM_RESULT_NOT_SUPPORTED,
                "last error code should report not_supported without RTL-SDR") &&
         ok;
#endif

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_execute\n");
        sim_context_destroy(&context);
        return 1;
    }

    memset(&status, 0, sizeof(status));
    if (sim_sdr_observer_status(&context, op_index, &status) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_sdr_observer_status after execute\n");
        sim_context_destroy(&context);
        return 1;
    }

    ok = expect(status.active_backend == SIM_SDR_OBSERVER_BACKEND_SYNTHETIC,
                "backend should remain synthetic after execute") &&
         ok;
    ok = expect(status.fallback_count == 1U,
                "plain synthetic execution should not increment fallback count") &&
         ok;

    config.device_index = 9998;
    config.center_freq  = 101.9e6;
    config.freq_offset  = -250.0e3;
    if (sim_sdr_observer_update(&context, op_index, &config) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_sdr_observer_update\n");
        sim_context_destroy(&context);
        return 1;
    }

    memset(&status, 0, sizeof(status));
    if (sim_sdr_observer_status(&context, op_index, &status) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_sdr_observer_status after reopen attempt\n");
        sim_context_destroy(&context);
        return 1;
    }

    ok = expect(status.fallback_count == 2U,
                "reopen with another invalid device should increment fallback count") &&
         ok;
    ok = expect(status.active_backend == SIM_SDR_OBSERVER_BACKEND_SYNTHETIC,
                "backend should remain synthetic after failed reopen") &&
         ok;
    ok = expect_close(status.effective_tuned_freq,
                      config.center_freq + config.freq_offset,
                      1.0e-9,
                      "effective tuned frequency should update after config changes") &&
         ok;
    ok = expect(status.last_error_message[0] != '\0',
                "error message should remain populated after failed reopen") &&
         ok;

    sim_context_destroy(&context);
    return ok ? 0 : 1;
}
