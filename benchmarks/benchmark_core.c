#include <oakfield/sim.h>
#include <oakfield/math/airy.h>
#include <oakfield/math/bessel.h>

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

typedef struct BenchConfig {
    size_t elements;
    int    iterations;
} BenchConfig;

typedef struct BenchResult {
    const char* name;
    double      seconds;
    double      units;
    const char* unit_name;
    double      checksum;
} BenchResult;

static volatile double g_sink = 0.0;

static double monotonic_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency = { 0 };
    LARGE_INTEGER        counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return (double) counter.QuadPart / (double) frequency.QuadPart;
#elif defined(__APPLE__)
    static mach_timebase_info_data_t info = { 0, 0 };
    uint64_t                         now  = mach_absolute_time();
    if (info.denom == 0U) {
        mach_timebase_info(&info);
    }
    return (double) now * (double) info.numer / (double) info.denom / 1.0e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1.0e9;
#endif
}

static bool parse_size_arg(const char* arg, const char* prefix, size_t* out_value) {
    char*         endptr = NULL;
    unsigned long parsed;
    size_t        prefix_len;

    if (arg == NULL || prefix == NULL || out_value == NULL) {
        return false;
    }

    prefix_len = strlen(prefix);
    if (strncmp(arg, prefix, prefix_len) != 0) {
        return false;
    }

    parsed = strtoul(arg + prefix_len, &endptr, 10);
    if (endptr == arg + prefix_len || (endptr != NULL && *endptr != '\0') || parsed == 0UL) {
        return false;
    }

    *out_value = (size_t) parsed;
    return true;
}

static bool parse_int_arg(const char* arg, const char* prefix, int* out_value) {
    char*  endptr = NULL;
    long   parsed;
    size_t prefix_len;

    if (arg == NULL || prefix == NULL || out_value == NULL) {
        return false;
    }

    prefix_len = strlen(prefix);
    if (strncmp(arg, prefix, prefix_len) != 0) {
        return false;
    }

    parsed = strtol(arg + prefix_len, &endptr, 10);
    if (endptr == arg + prefix_len || (endptr != NULL && *endptr != '\0') || parsed <= 0L ||
        parsed > INT32_MAX) {
        return false;
    }

    *out_value = (int) parsed;
    return true;
}

static void print_environment(const BenchConfig* config) {
    printf("benchmark_core elements=%zu iterations=%d\n", config->elements, config->iterations);
    printf("compiler=%s\n",
#if defined(__clang__)
           "clang"
#elif defined(__GNUC__)
           "gcc"
#elif defined(_MSC_VER)
           "msvc"
#else
           "unknown"
#endif
    );
    printf("platform=%s\n",
#if defined(_WIN32)
           "windows"
#elif defined(__APPLE__)
           "apple"
#elif defined(__linux__)
           "linux"
#else
           "unknown"
#endif
    );
    printf("backend=cpu\n");
    printf("build_type=%s\n",
#if defined(NDEBUG)
           "release-like"
#else
           "debug-like"
#endif
    );
    printf("sim_fast_math=%s\n",
#if OAKFIELD_BENCH_OAKFIELD_ENABLE_FAST_MATH
           "enabled"
#else
           "disabled"
#endif
    );
    printf("benchmark_tu_fast_math=%s\n",
#if defined(__FAST_MATH__)
           "on"
#else
           "off"
#endif
    );
    printf("openmp=%s\n",
#if defined(SIM_HAVE_OPENMP)
           "available"
#else
           "unavailable"
#endif
    );
    printf("vdsp=%s\n",
#if defined(SIM_HAVE_VDSP)
           "available"
#else
           "unavailable"
#endif
    );
}

static void print_result(const BenchResult* result) {
    double per_second = 0.0;
    if (result->seconds > 0.0) {
        per_second = result->units / result->seconds;
    }
    printf("%s seconds=%.6f %s=%.0f rate=%.3f/%s checksum=%.12g\n",
           result->name,
           result->seconds,
           result->unit_name,
           result->units,
           per_second,
           result->unit_name,
           result->checksum);
}

static bool run_field_alloc_benchmark(const BenchConfig* config, BenchResult* out_result) {
    double start;
    double checksum = 0.0;
    size_t shape[1];

    shape[0] = config->elements;
    start    = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        SimField field = { 0 };
        if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
            fprintf(stderr, "field allocation benchmark failed at iteration %d\n", i);
            return false;
        }
        checksum += (double) sim_field_bytes(&field);
        sim_field_destroy(&field);
    }

    out_result->name      = "field_alloc_destroy";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations;
    out_result->unit_name = "allocations";
    out_result->checksum  = checksum;
    g_sink += checksum;
    return true;
}

static bool run_field_wrap_benchmark(const BenchConfig* config, BenchResult* out_result) {
    double*        storage = NULL;
    size_t         shape_storage[1];
    size_t         stride_storage[1];
    SimFieldLayout layout;
    double         start;
    double         checksum = 0.0;

    storage = (double*) malloc(config->elements * sizeof(double));
    if (storage == NULL) {
        fprintf(stderr, "field wrap benchmark allocation failed\n");
        return false;
    }

    for (size_t i = 0; i < config->elements; ++i) {
        storage[i] = (double) (i % 1024U) * 0.001;
    }

    shape_storage[0]  = config->elements;
    stride_storage[0] = 1U;
    layout.rank       = 1U;
    layout.shape      = shape_storage;
    layout.strides    = stride_storage;
    layout.contiguous = true;

    start = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        SimField field = { 0 };
        if (sim_field_wrap(&field, &layout, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, storage) !=
            SIM_RESULT_OK) {
            fprintf(stderr, "field wrap benchmark failed at iteration %d\n", i);
            free(storage);
            return false;
        }
        checksum += (double) sim_field_element_count(&field.layout);
        sim_field_destroy(&field);
    }

    out_result->name      = "field_wrap_view";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations;
    out_result->unit_name = "wraps";
    out_result->checksum  = checksum;
    g_sink += checksum;
    free(storage);
    return true;
}

static bool run_field_promotion_benchmark(const BenchConfig* config, BenchResult* out_result) {
    double start;
    double checksum = 0.0;
    size_t shape[1];

    shape[0] = config->elements;
    start    = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        SimField field = { 0 };
        double*  data;

        if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
            fprintf(stderr, "field promotion setup failed at iteration %d\n", i);
            return false;
        }

        data = sim_field_real_data(&field);
        if (data == NULL) {
            fprintf(stderr, "field promotion data unavailable at iteration %d\n", i);
            sim_field_destroy(&field);
            return false;
        }
        for (size_t j = 0; j < config->elements; ++j) {
            data[j] = (double) (j % 1024U) * 0.001;
        }

        if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
            fprintf(stderr, "field promotion failed at iteration %d\n", i);
            sim_field_destroy(&field);
            return false;
        }

        {
            const SimComplexDouble* complex_data = sim_field_complex_data_const(&field);
            if (complex_data != NULL) {
                size_t probe = config->elements / 2U;
                if (probe + 1U < config->elements) {
                    ++probe;
                }
                checksum += complex_data[probe].re;
            }
        }
        sim_field_destroy(&field);
    }

    out_result->name      = "field_promote_complex";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations * (double) config->elements;
    out_result->unit_name = "elements";
    out_result->checksum  = checksum;
    g_sink += checksum;
    return true;
}

static bool setup_two_field_context(const BenchConfig* config,
                                    SimContext*        context,
                                    size_t*            out_src_index,
                                    size_t*            out_dst_index) {
    SimField src      = { 0 };
    SimField dst      = { 0 };
    size_t   shape[1] = { 0U };
    double*  src_data = NULL;

    shape[0] = config->elements;
    if (sim_context_init(context) != SIM_RESULT_OK) {
        fprintf(stderr, "utility operator benchmark context init failed\n");
        return false;
    }
    if (sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "utility operator benchmark source field init failed\n");
        sim_context_destroy(context);
        return false;
    }
    if (sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "utility operator benchmark destination field init failed\n");
        sim_field_destroy(&src);
        sim_context_destroy(context);
        return false;
    }

    src_data = sim_field_real_data(&src);
    if (src_data == NULL) {
        fprintf(stderr, "utility operator benchmark source data unavailable\n");
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        sim_context_destroy(context);
        return false;
    }
    for (size_t i = 0; i < config->elements; ++i) {
        src_data[i] = (double) (i % 4096U) * 0.0005;
    }

    if (sim_context_add_field(context, &src, out_src_index) != SIM_RESULT_OK) {
        fprintf(stderr, "utility operator benchmark could not add source field\n");
        sim_field_destroy(&dst);
        sim_field_destroy(&src);
        sim_context_destroy(context);
        return false;
    }
    if (sim_context_add_field(context, &dst, out_dst_index) != SIM_RESULT_OK) {
        fprintf(stderr, "utility operator benchmark could not add destination field\n");
        sim_field_destroy(&dst);
        sim_context_destroy(context);
        return false;
    }

    return true;
}

static bool run_copy_operator_benchmark(const BenchConfig* config, BenchResult* out_result) {
    SimContext context   = { 0 };
    size_t     src_index = 0U;
    size_t     dst_index = 0U;
    double     start;
    double     checksum = 0.0;
    bool       ok       = false;

    if (!setup_two_field_context(config, &context, &src_index, &dst_index)) {
        return false;
    }

    {
        SimCopyOperatorConfig copy_config = {
            .input_field  = src_index,
            .output_field = dst_index,
            .accumulate   = false,
            .scale_by_dt  = false,
        };
        if (sim_add_copy_operator(&context, &copy_config, NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "copy operator benchmark registration failed\n");
            goto cleanup;
        }
    }

    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "copy operator benchmark plan preparation failed\n");
        goto cleanup;
    }

    start = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        if (sim_context_execute_prepared(&context) != SIM_RESULT_OK) {
            fprintf(stderr, "copy operator benchmark execution failed at iteration %d\n", i);
            goto cleanup;
        }
    }

    {
        const SimField* dst      = sim_context_field(&context, dst_index);
        const double*   dst_data = (dst != NULL) ? sim_field_real_data_const(dst) : NULL;
        if (dst_data == NULL) {
            fprintf(stderr, "copy operator benchmark destination data unavailable\n");
            goto cleanup;
        }
        for (size_t i = 0; i < config->elements; i += 113U) {
            checksum += dst_data[i];
        }
    }

    out_result->name      = "utility_copy_operator";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations * (double) config->elements;
    out_result->unit_name = "elements";
    out_result->checksum  = checksum;
    g_sink += checksum;
    ok = true;

cleanup:
    sim_context_destroy(&context);
    return ok;
}

static bool run_scale_operator_benchmark(const BenchConfig* config, BenchResult* out_result) {
    SimContext context   = { 0 };
    size_t     src_index = 0U;
    size_t     dst_index = 0U;
    double     start;
    double     checksum = 0.0;
    bool       ok       = false;

    if (!setup_two_field_context(config, &context, &src_index, &dst_index)) {
        return false;
    }

    {
        SimScaleOperatorConfig scale_config = {
            .input_field  = src_index,
            .output_field = dst_index,
            .scale        = 1.125,
            .accumulate   = false,
            .scale_by_dt  = false,
        };
        if (sim_add_scale_operator(&context, &scale_config, NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "scale operator benchmark registration failed\n");
            goto cleanup;
        }
    }

    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "scale operator benchmark plan preparation failed\n");
        goto cleanup;
    }

    start = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        if (sim_context_execute_prepared(&context) != SIM_RESULT_OK) {
            fprintf(stderr, "scale operator benchmark execution failed at iteration %d\n", i);
            goto cleanup;
        }
    }

    {
        const SimField* dst      = sim_context_field(&context, dst_index);
        const double*   dst_data = (dst != NULL) ? sim_field_real_data_const(dst) : NULL;
        if (dst_data == NULL) {
            fprintf(stderr, "scale operator benchmark destination data unavailable\n");
            goto cleanup;
        }
        for (size_t i = 0; i < config->elements; i += 113U) {
            checksum += dst_data[i];
        }
    }

    out_result->name      = "utility_scale_operator";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations * (double) config->elements;
    out_result->unit_name = "elements";
    out_result->checksum  = checksum;
    g_sink += checksum;
    ok = true;

cleanup:
    sim_context_destroy(&context);
    return ok;
}

static bool run_special_math_benchmark(const BenchConfig* config, BenchResult* out_result) {
    double start;
    double checksum = 0.0;
    size_t samples  = config->elements;

    start = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        for (size_t j = 0; j < samples; ++j) {
            double x = 0.25 + (double) ((j % 1024U) + 1U) / 1024.0;
            checksum += sim_special_digamma(x);
            checksum += sim_special_trigamma(x + 1.0);
            checksum += sim_airy_ai_f64(x * 0.125);
            checksum += sim_bessel_j0_f64(x * 0.5);
        }
    }

    out_result->name      = "stable_special_math";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations * (double) samples * 4.0;
    out_result->unit_name = "calls";
    out_result->checksum  = checksum;
    g_sink += checksum;
    return true;
}

static SimResult decay_drift(Integrator*   integrator,
                             const Field*  field,
                             const double* state,
                             double*       out_derivative,
                             size_t        count) {
    (void) integrator;
    (void) field;

    if (state == NULL || out_derivative == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < count; ++i) {
        out_derivative[i] = -0.5 * state[i];
    }
    return SIM_RESULT_OK;
}

static bool run_integrator_benchmark(const BenchConfig* config,
                                     const char*        integrator_name,
                                     BenchResult*       out_result) {
    IntegratorRegistry registry   = { 0 };
    Integrator         integrator = { 0 };
    SimField           field      = { 0 };
    IntegratorConfig   int_config = { 0 };
    double*            data       = NULL;
    double             start;
    double             checksum = 0.0;
    size_t             shape[1];
    bool               ok = false;

    shape[0] = config->elements;
    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "%s integrator benchmark field init failed\n", integrator_name);
        return false;
    }

    data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "%s integrator benchmark data unavailable\n", integrator_name);
        goto cleanup_field;
    }
    for (size_t i = 0; i < config->elements; ++i) {
        data[i] = 1.0 + (double) (i % 257U) * 0.00025;
    }

    int_config.drift      = decay_drift;
    int_config.initial_dt = 0.001;
    int_config.min_dt     = 1.0e-6;
    int_config.max_dt     = 0.01;
    int_config.tolerance  = 1.0e-6;
    int_config.safety     = 0.9;
    int_config.adaptive   = false;

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "%s integrator benchmark registry init failed\n", integrator_name);
        goto cleanup_field;
    }
    if (integrator_registry_create(&registry, integrator_name, &int_config, &integrator) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "%s integrator benchmark creation failed\n", integrator_name);
        goto cleanup_registry;
    }

    start = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        integrator.step(&integrator, &field, 0.001);
        if (integrator.last_error < 0.0) {
            fprintf(stderr,
                    "%s integrator benchmark step failed at iteration %d\n",
                    integrator_name,
                    i);
            goto cleanup_integrator;
        }
    }

    data = sim_field_real_data(&field);
    if (data == NULL) {
        fprintf(stderr, "%s integrator benchmark result data unavailable\n", integrator_name);
        goto cleanup_integrator;
    }
    for (size_t i = 0; i < config->elements; i += 127U) {
        checksum += data[i];
    }

    out_result->name =
        (strcmp(integrator_name, "rk4") == 0) ? "integrator_rk4_decay" : "integrator_euler_decay";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations * (double) config->elements;
    out_result->unit_name = "elements";
    out_result->checksum  = checksum;
    g_sink += checksum;
    ok = true;

cleanup_integrator:
    integrator_destroy(&integrator);
cleanup_registry:
    integrator_registry_destroy(&registry);
cleanup_field:
    sim_field_destroy(&field);
    return ok;
}

static bool run_kernel_ir_benchmark(const BenchConfig* config, BenchResult* out_result) {
    SimField           src     = { 0 };
    SimField           dst     = { 0 };
    SimIRBuilder       builder = { 0 };
    SimBackend         backend = { 0 };
    SimKernelIRBinding bindings[2];
    SimKernelIROutput  output;
    KernelIR           kernel;
    SimIRNodeId        src_node;
    SimIRNodeId        scale_node;
    SimIRNodeId        bias_node;
    SimIRNodeId        scaled_node;
    SimIRNodeId        output_node;
    double*            src_data;
    const double*      dst_data;
    double             start;
    double             checksum = 0.0;
    size_t             shape[1];
    bool               ok = false;

    shape[0] = config->elements;
    if (sim_field_init(&src, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "kernel benchmark source field init failed\n");
        return false;
    }
    if (sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "kernel benchmark destination field init failed\n");
        goto cleanup_fields;
    }

    src_data = sim_field_real_data(&src);
    if (src_data == NULL) {
        fprintf(stderr, "kernel benchmark source data unavailable\n");
        goto cleanup_fields;
    }
    for (size_t i = 0; i < config->elements; ++i) {
        src_data[i] = (double) (i % 8192U) * 0.00025;
    }

    if (sim_ir_builder_init(&builder) != SIM_RESULT_OK) {
        fprintf(stderr, "kernel benchmark IR builder init failed\n");
        goto cleanup_fields;
    }

    src_node    = sim_ir_builder_field_ref(&builder, 0U);
    scale_node  = sim_ir_builder_constant(&builder, 1.25);
    bias_node   = sim_ir_builder_constant(&builder, 0.5);
    scaled_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_MUL, src_node, scale_node);
    output_node = sim_ir_builder_binary(&builder, SIM_IR_NODE_ADD, scaled_node, bias_node);
    if (src_node == SIM_IR_INVALID_NODE || scale_node == SIM_IR_INVALID_NODE ||
        bias_node == SIM_IR_INVALID_NODE || scaled_node == SIM_IR_INVALID_NODE ||
        output_node == SIM_IR_INVALID_NODE) {
        fprintf(stderr, "kernel benchmark IR node creation failed\n");
        goto cleanup_builder;
    }

    bindings[0].field_index = 0U;
    bindings[0].field       = &src;
    bindings[0].shape       = sim_field_shape(&src);
    bindings[0].strides     = sim_field_strides(&src);
    bindings[0].rank        = sim_field_rank(&src);
    bindings[1].field_index = 1U;
    bindings[1].field       = &dst;
    bindings[1].shape       = sim_field_shape(&dst);
    bindings[1].strides     = sim_field_strides(&dst);
    bindings[1].rank        = sim_field_rank(&dst);

    output.field_index = 1U;
    output.expression  = output_node;

    kernel.builder           = &builder;
    kernel.bindings          = bindings;
    kernel.binding_count     = 2U;
    kernel.outputs           = &output;
    kernel.output_count      = 1U;
    kernel.params            = NULL;
    kernel.param_count       = 0U;
    kernel.required_features = 0U;
    kernel.complex_semantics = SIM_KERNEL_COMPLEX_SEMANTICS_TRUE_COMPLEX;

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "kernel benchmark backend init failed (%d)\n", backend.last_error);
        goto cleanup_builder;
    }

    start = monotonic_seconds();
    for (int i = 0; i < config->iterations; ++i) {
        backend_launch(&backend, &kernel);
        if (backend.last_error != SIM_RESULT_OK) {
            fprintf(stderr,
                    "kernel benchmark launch failed at iteration %d (%d)\n",
                    i,
                    backend.last_error);
            goto cleanup_backend;
        }
    }

    dst_data = sim_field_real_data_const(&dst);
    if (dst_data == NULL) {
        fprintf(stderr, "kernel benchmark destination data unavailable\n");
        goto cleanup_backend;
    }
    for (size_t i = 0; i < config->elements; i += 97U) {
        checksum += dst_data[i];
    }

    out_result->name      = "cpu_kernel_ir_affine";
    out_result->seconds   = monotonic_seconds() - start;
    out_result->units     = (double) config->iterations * (double) config->elements;
    out_result->unit_name = "elements";
    out_result->checksum  = checksum;
    g_sink += checksum;
    ok = true;

cleanup_backend:
    backend_destroy(&backend);
cleanup_builder:
    sim_ir_builder_destroy(&builder);
cleanup_fields:
    sim_field_destroy(&dst);
    sim_field_destroy(&src);
    return ok;
}

int main(int argc, char** argv) {
    BenchConfig config = {
        .elements   = 65536U,
        .iterations = 128,
    };
    BenchResult result = { 0 };

    for (int i = 1; i < argc; ++i) {
        if (parse_size_arg(argv[i], "--elements=", &config.elements)) {
            continue;
        }
        if (parse_int_arg(argv[i], "--iterations=", &config.iterations)) {
            continue;
        }
        if (strcmp(argv[i], "--help") == 0) {
            printf("usage: benchmark_core [--elements=N] [--iterations=N]\n");
            return 0;
        }
        fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
        return 1;
    }

    print_environment(&config);

    if (!run_field_alloc_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_field_wrap_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_field_promotion_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_copy_operator_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_scale_operator_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_special_math_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_kernel_ir_benchmark(&config, &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_integrator_benchmark(&config, "euler", &result)) {
        return 1;
    }
    print_result(&result);

    if (!run_integrator_benchmark(&config, "rk4", &result)) {
        return 1;
    }
    print_result(&result);

    if (g_sink == -1.0) {
        printf("sink=%.1f\n", g_sink);
    }

    return 0;
}
