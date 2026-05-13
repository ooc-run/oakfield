#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

static double monotonic_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#elif defined(__APPLE__)
    static mach_timebase_info_data_t info = {0, 0};
    uint64_t now = mach_absolute_time();
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    return (double)now * (double)info.numer / (double)info.denom / 1.0e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
#endif
}

static void swap_fields(SimField **a, SimField **b) {
    SimField *tmp = *a;
    *a = *b;
    *b = tmp;
}

static bool parse_positive_int(const char *text, int *out_value) {
    char *endptr = NULL;
    long parsed;

    if (text == NULL || out_value == NULL || text[0] == '\0') {
        return false;
    }

    parsed = strtol(text, &endptr, 10);
    if (endptr == text || (endptr != NULL && *endptr != '\0') || parsed <= 0L ||
        parsed > INT32_MAX) {
        return false;
    }

    *out_value = (int)parsed;
    return true;
}

int main(int argc, char **argv) {
    size_t shape[2] = {256U, 256U};
    int iterations = 64;
    bool validate = false;
    double tolerance = 1.0e-9;
    SimField field_a = {0};
    SimField field_b = {0};
    SimField *src = &field_a;
    SimField *dst = &field_b;
    SimResult result;
    double *data_a;
    double *data_b;
    SimIRBuilder builder;
    SimIRNodeId src_ref;
    SimIRNodeId diff_x;
    SimIRNodeId lap_x;
    SimIRNodeId diff_y;
    SimIRNodeId lap_y;
    SimIRNodeId lap_sum;
    SimIRNodeId diffusion_scale;
    SimIRNodeId diff_node;
    SimIRNodeId update_node;
    SimKernelIRBinding bindings[2];
    SimKernelIROutput output;
    KernelIR kernel;
    SimBackend backend = {0};
    double start_time;
    double end_time;
    double checksum = 0.0;
    double max_error = 0.0;
    double amplification = 0.0;
    const unsigned int mode_x = 3U;
    const unsigned int mode_y = 5U;
    const double diffusivity = 0.05;
    int i;

    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        int parsed = 0;

        if (strncmp(arg, "--iterations=", 13) == 0) {
            if (parse_positive_int(arg + 13, &parsed)) {
                iterations = parsed;
            }
            continue;
        }
        if (strcmp(arg, "--validate") == 0) {
            validate = true;
            continue;
        }
        if (strncmp(arg, "--tolerance=", 12) == 0) {
            double parsed_tol = atof(arg + 12);
            if (parsed_tol > 0.0) {
                tolerance = parsed_tol;
            }
            continue;
        }
        if (parse_positive_int(arg, &parsed)) {
            iterations = parsed;
        }
    }

    result = sim_field_init(src, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "failed to initialize source field (%d)\n", result);
        return 1;
    }

    result = sim_field_init(dst, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "failed to initialize destination field (%d)\n", result);
        sim_field_destroy(&field_a);
        return 1;
    }

    data_a = (double *)sim_field_data(src);
    data_b = (double *)sim_field_data(dst);
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[0]; ++x) {
            size_t idx = y * shape[0] + x;
            double phase_x = 2.0 * M_PI * (double)mode_x * (double)x / (double)shape[0];
            double phase_y = 2.0 * M_PI * (double)mode_y * (double)y / (double)shape[1];
            data_a[idx] = sin(phase_x) * cos(phase_y);
            data_b[idx] = 0.0;
        }
    }

    result = sim_ir_builder_init(&builder);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "failed to initialize IR builder (%d)\n", result);
        sim_field_destroy(&field_b);
        sim_field_destroy(&field_a);
        return 1;
    }

    sim_ir_builder_set_default_boundary(&builder, SIM_IR_BOUNDARY_PERIODIC);
    src_ref = sim_ir_builder_field_ref(&builder, 0U);
    diff_x = sim_ir_builder_diff(&builder, src_ref, 0U, 1.0, 1.0);
    lap_x = sim_ir_builder_diff(&builder, diff_x, 0U, 1.0, 1.0);
    diff_y = sim_ir_builder_diff(&builder, src_ref, 1U, 1.0, 1.0);
    lap_y = sim_ir_builder_diff(&builder, diff_y, 1U, 1.0, 1.0);
    lap_sum = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, lap_x, lap_y);
    diffusion_scale = sim_ir_builder_constant(&builder, diffusivity);
    diff_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_MUL, lap_sum, diffusion_scale);

    if (src_ref == SIM_IR_INVALID_NODE || diff_x == SIM_IR_INVALID_NODE ||
        lap_x == SIM_IR_INVALID_NODE || diff_y == SIM_IR_INVALID_NODE ||
        lap_y == SIM_IR_INVALID_NODE || lap_sum == SIM_IR_INVALID_NODE ||
        diffusion_scale == SIM_IR_INVALID_NODE || diff_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "failed to build diffusion kernel nodes\n");
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&field_b);
        sim_field_destroy(&field_a);
        return 1;
    }

    update_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, src_ref, diff_node);
    if (update_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "failed to create diffusion update node\n");
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&field_b);
        sim_field_destroy(&field_a);
        return 1;
    }

    bindings[0].field_index = 0U;
    bindings[1].field_index = 1U;
    output.field_index = 1U;
    output.expression = update_node;

    kernel.builder = &builder;
    kernel.bindings = bindings;
    kernel.binding_count = 2U;
    kernel.outputs = &output;
    kernel.output_count = 1U;
    kernel.required_features = 0U;

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "backend initialization failed (%d)\n", backend.last_error);
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&field_b);
        sim_field_destroy(&field_a);
        return 1;
    }

    start_time = monotonic_seconds();

    for (i = 0; i < iterations; ++i) {
        bindings[0].field = src;
        bindings[1].field = dst;

        backend_launch(&backend, &kernel);
        if (backend.last_error != SIM_RESULT_OK) {
            fprintf(stderr, "backend launch failed at iteration %d (%d)\n", i, backend.last_error);
            backend_destroy(&backend);
            sim_ir_builder_destroy(&builder);
            sim_field_destroy(&field_b);
            sim_field_destroy(&field_a);
            return 1;
        }

        swap_fields(&src, &dst);
    }

    end_time = monotonic_seconds();

    data_a = (double *)sim_field_data(src);
    /* The benchmark composes first derivatives twice, so the periodic Fourier-mode
     * eigenvalue is -sin(theta)^2 per axis rather than the direct second-difference form. */
    amplification = 1.0 - diffusivity * (sin(2.0 * M_PI * (double)mode_x / (double)shape[0]) *
                                             sin(2.0 * M_PI * (double)mode_x / (double)shape[0]) +
                                         sin(2.0 * M_PI * (double)mode_y / (double)shape[1]) *
                                             sin(2.0 * M_PI * (double)mode_y / (double)shape[1]));
    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[0]; ++x) {
            size_t idx = y * shape[0] + x;
            double phase_x = 2.0 * M_PI * (double)mode_x * (double)x / (double)shape[0];
            double phase_y = 2.0 * M_PI * (double)mode_y * (double)y / (double)shape[1];
            double expected = pow(amplification, (double)iterations) * sin(phase_x) * cos(phase_y);
            double diff = fabs(data_a[idx] - expected);
            if (diff > max_error) {
                max_error = diff;
            }
            checksum += data_a[idx];
        }
    }

    printf("benchmark_diffusion iterations=%d size=%zux%zu time=%.3f ms checksum=%.6f "
           "max_error=%.12e amplification=%.12f\n",
           iterations, shape[0], shape[1], (end_time - start_time) * 1000.0, checksum, max_error,
           amplification);

    if (validate && max_error > tolerance) {
        fprintf(stderr, "[FAIL] diffusion benchmark max_error %.12e exceeds tolerance %.12e\n",
                max_error, tolerance);
        backend_destroy(&backend);
        sim_ir_builder_destroy(&builder);
        sim_field_destroy(&field_b);
        sim_field_destroy(&field_a);
        return 1;
    }

    backend_destroy(&backend);
    sim_ir_builder_destroy(&builder);
    sim_field_destroy(&field_b);
    sim_field_destroy(&field_a);

    return 0;
}
