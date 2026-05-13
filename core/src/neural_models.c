/**
 * @file neural_models.c
 * @brief Context-level neural model registry and inference dispatch implementation.
 *
 * Neural model entries describe callback, external-process, and in-process
 * backend routes. This module normalizes model metadata, selects supported
 * devices, serializes process requests when needed, records runtime statistics,
 * and reports model lookup or inference failures through SimResult values.
 */
#include "oakfield/neural_models.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#define SIM_NEURAL_PROCESS_SCHEMA_VERSION "oakfield.neural.process.v1"

static int sim_neural_text_iequals(const char* a, const char* b) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    while (*a != '\0' && *b != '\0') {
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) {
            return 0;
        }
        ++a;
        ++b;
    }
    return (*a == '\0' && *b == '\0') ? 1 : 0;
}

static void sim_neural_copy_text(char* dst, size_t capacity, const char* src) {
    if (dst == NULL || capacity == 0U) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    (void) strncpy(dst, src, capacity - 1U);
    dst[capacity - 1U] = '\0';
}

static uint64_t sim_neural_now_ns(void) {
#if defined(__APPLE__)
    static mach_timebase_info_data_t info  = { 0, 0 };
    uint64_t                         ticks = mach_absolute_time();
    if (info.denom == 0U) {
        mach_timebase_info(&info);
    }
    return ticks * (uint64_t) info.numer / (uint64_t) info.denom;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0U;
    }
    return ((uint64_t) ts.tv_sec * 1000000000ULL) + (uint64_t) ts.tv_nsec;
#endif
}

static uint32_t sim_neural_default_device_mask(uint32_t mask) {
    return (mask != 0U) ? mask : SIM_NEURAL_DEVICE_MASK_CPU;
}

static bool sim_neural_backend_kind_valid(SimNeuralBackendKind kind) {
    switch (kind) {
        case SIM_NEURAL_BACKEND_CALLBACK:
        case SIM_NEURAL_BACKEND_EXTERNAL_PROCESS:
        case SIM_NEURAL_BACKEND_INPROCESS_ONNX_RUNTIME:
        case SIM_NEURAL_BACKEND_INPROCESS_LIBTORCH:
            return true;
        default:
            break;
    }
    return false;
}

static bool sim_neural_precision_valid(SimNeuralPrecisionMode mode) {
    switch (mode) {
        case SIM_NEURAL_PRECISION_DEFAULT:
        case SIM_NEURAL_PRECISION_FP32:
        case SIM_NEURAL_PRECISION_FP64:
        case SIM_NEURAL_PRECISION_MIXED:
        case SIM_NEURAL_PRECISION_FP16:
        case SIM_NEURAL_PRECISION_BF16:
            return true;
        default:
            break;
    }
    return false;
}

static const char* sim_neural_precision_mode_name(SimNeuralPrecisionMode mode) {
    switch (mode) {
        case SIM_NEURAL_PRECISION_DEFAULT:
            return "default";
        case SIM_NEURAL_PRECISION_FP32:
            return "fp32";
        case SIM_NEURAL_PRECISION_FP64:
            return "fp64";
        case SIM_NEURAL_PRECISION_MIXED:
            return "mixed";
        case SIM_NEURAL_PRECISION_FP16:
            return "fp16";
        case SIM_NEURAL_PRECISION_BF16:
            return "bf16";
        default:
            break;
    }
    return "default";
}

static SimNeuralPrecisionMode
sim_neural_effective_precision(const SimNeuralModelConfig*     config,
                               const SimNeuralInferenceRequest* request) {
    SimNeuralPrecisionMode mode = (request != NULL) ? request->precision_mode : SIM_NEURAL_PRECISION_DEFAULT;
    if (mode == SIM_NEURAL_PRECISION_DEFAULT && config != NULL &&
        sim_neural_precision_valid(config->default_precision)) {
        mode = config->default_precision;
    }
    if (!sim_neural_precision_valid(mode)) {
        mode = SIM_NEURAL_PRECISION_DEFAULT;
    }
    return mode;
}

static SimResult sim_neural_route_device(const SimNeuralModelConfig*     config,
                                         const SimNeuralInferenceRequest* request,
                                         SimNeuralExecutionDevice*        out_device) {
    uint32_t supported;
    if (config == NULL || out_device == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    supported = sim_neural_default_device_mask(config->supported_device_mask);

    if (request != NULL && request->determinism_policy == SIM_NEURAL_DETERMINISM_STRICT &&
        (supported & SIM_NEURAL_DEVICE_MASK_CPU) != 0U) {
        *out_device = SIM_NEURAL_EXEC_DEVICE_CPU;
        return SIM_RESULT_OK;
    }

    switch ((request != NULL) ? request->device_requirement : SIM_NEURAL_DEVICE_ANY) {
        case SIM_NEURAL_DEVICE_CPU_ONLY:
            if ((supported & SIM_NEURAL_DEVICE_MASK_CPU) == 0U) {
                return SIM_RESULT_NOT_SUPPORTED;
            }
            *out_device = SIM_NEURAL_EXEC_DEVICE_CPU;
            return SIM_RESULT_OK;
        case SIM_NEURAL_DEVICE_ACCELERATOR_REQUIRED:
            if ((supported & SIM_NEURAL_DEVICE_MASK_CUDA) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_CUDA;
                return SIM_RESULT_OK;
            }
            if ((supported & SIM_NEURAL_DEVICE_MASK_MPS) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_MPS;
                return SIM_RESULT_OK;
            }
            return SIM_RESULT_NOT_SUPPORTED;
        case SIM_NEURAL_DEVICE_ACCELERATOR_PREFERRED:
            if ((supported & SIM_NEURAL_DEVICE_MASK_CUDA) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_CUDA;
                return SIM_RESULT_OK;
            }
            if ((supported & SIM_NEURAL_DEVICE_MASK_MPS) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_MPS;
                return SIM_RESULT_OK;
            }
            if ((supported & SIM_NEURAL_DEVICE_MASK_CPU) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_CPU;
                return SIM_RESULT_OK;
            }
            return SIM_RESULT_NOT_SUPPORTED;
        case SIM_NEURAL_DEVICE_ANY:
        default:
            if ((supported & SIM_NEURAL_DEVICE_MASK_CPU) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_CPU;
                return SIM_RESULT_OK;
            }
            if ((supported & SIM_NEURAL_DEVICE_MASK_CUDA) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_CUDA;
                return SIM_RESULT_OK;
            }
            if ((supported & SIM_NEURAL_DEVICE_MASK_MPS) != 0U) {
                *out_device = SIM_NEURAL_EXEC_DEVICE_MPS;
                return SIM_RESULT_OK;
            }
            return SIM_RESULT_NOT_SUPPORTED;
    }
}

static SimNeuralModelEntry*
sim_neural_model_lookup_mut(SimNeuralModelRegistry* registry, const char* model_id, size_t* out_index) {
    if (registry == NULL || model_id == NULL || model_id[0] == '\0') {
        return NULL;
    }
    for (size_t i = 0U; i < registry->count; ++i) {
        if (strcmp(registry->entries[i].config.model_id, model_id) == 0) {
            if (out_index != NULL) {
                *out_index = i;
            }
            return &registry->entries[i];
        }
    }
    return NULL;
}

static const SimNeuralModelEntry*
sim_neural_model_lookup(const SimNeuralModelRegistry* registry, const char* model_id, size_t* out_index) {
    return sim_neural_model_lookup_mut((SimNeuralModelRegistry*) registry, model_id, out_index);
}

static SimResult sim_neural_model_registry_reserve(SimNeuralModelRegistry* registry, size_t needed) {
    SimNeuralModelEntry* new_entries;
    size_t               new_capacity;
    if (registry == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (needed <= registry->capacity) {
        return SIM_RESULT_OK;
    }
    new_capacity = (registry->capacity == 0U) ? 4U : registry->capacity;
    while (new_capacity < needed) {
        if (new_capacity > (SIZE_MAX / 2U)) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        new_capacity *= 2U;
    }
    new_entries =
        (SimNeuralModelEntry*) realloc(registry->entries, new_capacity * sizeof(*new_entries));
    if (new_entries == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    registry->entries   = new_entries;
    registry->capacity  = new_capacity;
    return SIM_RESULT_OK;
}

static bool sim_neural_make_temp_path(char* buffer, size_t capacity, const char* prefix) {
    if (buffer == NULL || capacity < 32U) {
        return false;
    }
#if defined(_WIN32)
    {
        char temp_name[L_tmpnam];
        errno_t rc = tmpnam_s(temp_name, sizeof(temp_name));
        if (rc != 0) {
            return false;
        }
        if (snprintf(buffer, capacity, "%s_%s", temp_name, (prefix != NULL) ? prefix : "tmp") >=
            (int) capacity) {
            return false;
        }
        return true;
    }
#else
    {
        char pattern[64];
        int  fd;
        if (snprintf(pattern,
                     sizeof(pattern),
                     "/tmp/oakfield_%s_XXXXXX",
                     (prefix != NULL) ? prefix : "neural") >= (int) sizeof(pattern)) {
            return false;
        }
        fd = mkstemp(pattern);
        if (fd < 0) {
            return false;
        }
        (void) close(fd);
        sim_neural_copy_text(buffer, capacity, pattern);
        return true;
    }
#endif
}

static void sim_neural_remove_temp_path(const char* path) {
    if (path == NULL || path[0] == '\0') {
        return;
    }
    (void) remove(path);
}

static void sim_neural_set_env_text(const char* key, const char* value) {
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

static void sim_neural_set_env_u64(const char* key, uint64_t value) {
    char buffer[64];
    (void) snprintf(buffer, sizeof(buffer), "%llu", (unsigned long long) value);
    sim_neural_set_env_text(key, buffer);
}

static void sim_neural_set_env_double(const char* key, double value) {
    char buffer[64];
    (void) snprintf(buffer, sizeof(buffer), "%.17g", value);
    sim_neural_set_env_text(key, buffer);
}

static SimResult sim_neural_write_process_request(FILE*                            out,
                                                  const SimField*                  input,
                                                  const SimNeuralInferenceRequest* request,
                                                  SimNeuralExecutionDevice         device,
                                                  SimNeuralPrecisionMode           precision) {
    size_t count;
    if (out == NULL || input == NULL || request == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    count = sim_field_element_count(&input->layout);
    fprintf(out, "schema_version %s\n", SIM_NEURAL_PROCESS_SCHEMA_VERSION);
    fprintf(out, "model_id %s\n", (request->model_id != NULL) ? request->model_id : "");
    fprintf(out, "device %s\n", sim_neural_execution_device_name(device));
    fprintf(out, "precision %s\n", sim_neural_precision_mode_name(precision));
    fprintf(out, "field_kind %s\n", sim_field_is_complex(input) ? "complex" : "real");
    fprintf(out, "rank %zu\n", input->layout.rank);
    for (size_t i = 0U; i < input->layout.rank; ++i) {
        fprintf(out, "shape %zu %zu\n", i, input->layout.shape[i]);
    }
    fprintf(out, "count %zu\n", count);
    fprintf(out, "normalize_input %d\n", request->normalize_input ? 1 : 0);
    fprintf(out, "input_scale %.17g\n", request->input_scale);
    fprintf(out, "input_bias %.17g\n", request->input_bias);
    fprintf(out, "output_scale %.17g\n", request->output_scale);
    fprintf(out, "output_bias %.17g\n", request->output_bias);
    fprintf(out, "step_index %zu\n", request->step_index);
    fprintf(out, "dt %.17g\n", request->dt);
    fprintf(out, "sim_time %.17g\n", request->sim_time);
    if (sim_field_is_complex(input)) {
        const SimComplexDouble* values = sim_field_complex_data_const(input);
        if (values == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            fprintf(out, "value %zu %.17g %.17g\n", i, values[i].re, values[i].im);
        }
    } else {
        const double* values = sim_field_real_data_const(input);
        if (values == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        for (size_t i = 0U; i < count; ++i) {
            fprintf(out, "value %zu %.17g\n", i, values[i]);
        }
    }
    return ferror(out) ? SIM_RESULT_INVALID_STATE : SIM_RESULT_OK;
}

static SimResult sim_neural_read_process_response(FILE*      in,
                                                  SimField*  output,
                                                  char*      error_text,
                                                  size_t     error_capacity) {
    SimField temp = { 0 };
    void*    temp_data = NULL;
    char     line[512];
    bool     saw_status = false;
    bool     ok_status = false;
    bool     saw_count = false;
    size_t   expected_count;
    size_t   declared_count = 0U;
    size_t   seen_values = 0U;
    bool     output_complex;
    SimResult result = SIM_RESULT_INVALID_STATE;

    if (in == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    expected_count = sim_field_element_count(&output->layout);
    output_complex = sim_field_is_complex(output);
    temp_data      = calloc(expected_count, output->element_size);
    if (temp_data == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    result = sim_field_wrap(&temp, &output->layout, output->element_size, output->storage, temp_data);
    if (result != SIM_RESULT_OK) {
        free(temp_data);
        return result;
    }

    while (fgets(line, sizeof(line), in) != NULL) {
        char token[64];
        size_t index = 0U;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') {
            continue;
        }
        if (strncmp(line, "status ", 7) == 0) {
            saw_status = true;
            ok_status  = strcmp(line + 7, "ok") == 0;
            continue;
        }
        if (strncmp(line, "message ", 8) == 0) {
            sim_neural_copy_text(error_text, error_capacity, line + 8);
            continue;
        }
        if (sscanf(line, "%63s", token) != 1) {
            continue;
        }
        if (strcmp(token, "field_kind") == 0) {
            const char* kind = line + 11;
            if ((output_complex && strcmp(kind, "complex") != 0) ||
                (!output_complex && strcmp(kind, "real") != 0)) {
                result = SIM_RESULT_TYPE_MISMATCH;
                goto cleanup;
            }
            continue;
        }
        if (strcmp(token, "count") == 0) {
            if (sscanf(line, "count %zu", &declared_count) != 1) {
                result = SIM_RESULT_INVALID_STATE;
                goto cleanup;
            }
            saw_count = true;
            continue;
        }
        if (strcmp(token, "value") == 0) {
            if (output_complex) {
                double re = 0.0;
                double im = 0.0;
                if (sscanf(line, "value %zu %lf %lf", &index, &re, &im) != 3) {
                    result = SIM_RESULT_INVALID_STATE;
                    goto cleanup;
                }
                if (index >= expected_count) {
                    result = SIM_RESULT_INVALID_ARGUMENT;
                    goto cleanup;
                }
                sim_field_complex_data(&temp)[index].re = re;
                sim_field_complex_data(&temp)[index].im = im;
                seen_values += 1U;
            } else {
                double value = 0.0;
                if (sscanf(line, "value %zu %lf", &index, &value) != 2) {
                    result = SIM_RESULT_INVALID_STATE;
                    goto cleanup;
                }
                if (index >= expected_count) {
                    result = SIM_RESULT_INVALID_ARGUMENT;
                    goto cleanup;
                }
                sim_field_real_data(&temp)[index] = value;
                seen_values += 1U;
            }
            continue;
        }
    }

    if (!saw_status || !ok_status) {
        result = SIM_RESULT_DEPENDENCY_ERROR;
        goto cleanup;
    }
    if (!saw_count || declared_count != expected_count || seen_values != expected_count) {
        result = SIM_RESULT_INVALID_STATE;
        goto cleanup;
    }

    memcpy(sim_field_data(output), sim_field_data(&temp), sim_field_bytes(output));
    result = SIM_RESULT_OK;

cleanup:
    sim_field_destroy(&temp);
    free(temp_data);
    return result;
}

static SimResult sim_neural_run_external_process(const SimNeuralModelConfig*     config,
                                                 const SimField*                  input,
                                                 SimField*                        output,
                                                 const SimNeuralInferenceRequest* request,
                                                 SimNeuralExecutionDevice         device,
                                                 SimNeuralPrecisionMode           precision,
                                                 char*                            error_text,
                                                 size_t                           error_capacity) {
    char request_path[256];
    char response_path[256];
    FILE* request_file = NULL;
    FILE* response_file = NULL;
    int   system_result;
    SimResult result = SIM_RESULT_INVALID_STATE;

    if (config == NULL || input == NULL || output == NULL || request == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (config->external_command[0] == '\0') {
        sim_neural_copy_text(error_text, error_capacity, "external command is empty");
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (!sim_neural_make_temp_path(request_path, sizeof(request_path), "neural_req") ||
        !sim_neural_make_temp_path(response_path, sizeof(response_path), "neural_rsp")) {
        sim_neural_copy_text(error_text, error_capacity, "failed to allocate temp file");
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    request_file = fopen(request_path, "w");
    if (request_file == NULL) {
        sim_neural_copy_text(error_text, error_capacity, "failed to open request file");
        result = SIM_RESULT_INVALID_STATE;
        goto cleanup;
    }
    result = sim_neural_write_process_request(request_file, input, request, device, precision);
    fclose(request_file);
    request_file = NULL;
    if (result != SIM_RESULT_OK) {
        sim_neural_copy_text(error_text, error_capacity, "failed to serialize request");
        goto cleanup;
    }

    sim_neural_set_env_text("OAKFIELD_NEURAL_REQUEST", request_path);
    sim_neural_set_env_text("OAKFIELD_NEURAL_RESPONSE", response_path);
    sim_neural_set_env_text("OAKFIELD_NEURAL_MODEL_ID", request->model_id);
    sim_neural_set_env_text("OAKFIELD_NEURAL_DEVICE", sim_neural_execution_device_name(device));
    sim_neural_set_env_text("OAKFIELD_NEURAL_PRECISION", sim_neural_precision_mode_name(precision));
    sim_neural_set_env_u64("OAKFIELD_NEURAL_STEP_INDEX", (uint64_t) request->step_index);
    sim_neural_set_env_double("OAKFIELD_NEURAL_DT", request->dt);
    sim_neural_set_env_double("OAKFIELD_NEURAL_SIM_TIME", request->sim_time);

    system_result = system(config->external_command);
    if (system_result != 0) {
        char buffer[96];
        (void) snprintf(buffer, sizeof(buffer), "external command failed (%d)", system_result);
        sim_neural_copy_text(error_text, error_capacity, buffer);
        result = SIM_RESULT_DEPENDENCY_ERROR;
        goto cleanup;
    }

    response_file = fopen(response_path, "r");
    if (response_file == NULL) {
        sim_neural_copy_text(error_text, error_capacity, "external adapter did not write response");
        result = SIM_RESULT_INVALID_STATE;
        goto cleanup;
    }
    result = sim_neural_read_process_response(response_file, output, error_text, error_capacity);
    fclose(response_file);
    response_file = NULL;

cleanup:
    if (request_file != NULL) {
        fclose(request_file);
    }
    if (response_file != NULL) {
        fclose(response_file);
    }
    sim_neural_remove_temp_path(request_path);
    sim_neural_remove_temp_path(response_path);
    return result;
}

static void sim_neural_record_runtime(SimNeuralModelRuntimeStats*      stats,
                                      SimResult                        result,
                                      SimNeuralExecutionDevice         device,
                                      uint64_t                         elapsed_ns,
                                      size_t                           element_count,
                                      const SimNeuralInferenceRequest* request,
                                      const char*                      error_text) {
    if (stats == NULL) {
        return;
    }
    stats->invocation_count += 1ULL;
    if (result == SIM_RESULT_OK) {
        stats->success_count += 1ULL;
    } else {
        stats->failure_count += 1ULL;
    }
    stats->total_elements += (uint64_t) element_count;
    stats->last_wall_ms = (double) elapsed_ns / 1.0e6;
    stats->total_wall_ms += stats->last_wall_ms;
    stats->last_step_index = (request != NULL) ? request->step_index : 0U;
    stats->last_sim_time   = (request != NULL) ? request->sim_time : 0.0;
    stats->last_result     = result;
    stats->last_device     = device;
    if (error_text != NULL && error_text[0] != '\0') {
        sim_neural_copy_text(stats->last_error, sizeof(stats->last_error), error_text);
    } else {
        stats->last_error[0] = '\0';
    }
}

SimNeuralModelConfig sim_neural_model_config_defaults(void) {
    SimNeuralModelConfig config;
    (void) memset(&config, 0, sizeof(config));
    config.backend_kind          = SIM_NEURAL_BACKEND_CALLBACK;
    config.supported_device_mask = SIM_NEURAL_DEVICE_MASK_CPU;
    config.default_precision     = SIM_NEURAL_PRECISION_DEFAULT;
    config.deterministic         = true;
    return config;
}

void sim_neural_model_runtime_stats_reset(SimNeuralModelRuntimeStats* stats) {
    if (stats == NULL) {
        return;
    }
    (void) memset(stats, 0, sizeof(*stats));
    stats->last_result = SIM_RESULT_OK;
    stats->last_device = SIM_NEURAL_EXEC_DEVICE_CPU;
}

const char* sim_neural_backend_kind_name(SimNeuralBackendKind kind) {
    switch (kind) {
        case SIM_NEURAL_BACKEND_CALLBACK:
            return "callback";
        case SIM_NEURAL_BACKEND_EXTERNAL_PROCESS:
            return "external_process";
        case SIM_NEURAL_BACKEND_INPROCESS_ONNX_RUNTIME:
            return "onnx_runtime";
        case SIM_NEURAL_BACKEND_INPROCESS_LIBTORCH:
            return "libtorch";
        default:
            break;
    }
    return "callback";
}

bool sim_neural_backend_kind_from_string(const char* text, SimNeuralBackendKind* out_kind) {
    if (text == NULL || out_kind == NULL) {
        return false;
    }
    if (sim_neural_text_iequals(text, "callback")) {
        *out_kind = SIM_NEURAL_BACKEND_CALLBACK;
        return true;
    }
    if (sim_neural_text_iequals(text, "external_process") ||
        sim_neural_text_iequals(text, "external-process") ||
        sim_neural_text_iequals(text, "process")) {
        *out_kind = SIM_NEURAL_BACKEND_EXTERNAL_PROCESS;
        return true;
    }
    if (sim_neural_text_iequals(text, "onnx_runtime") ||
        sim_neural_text_iequals(text, "onnxruntime") ||
        sim_neural_text_iequals(text, "onnx")) {
        *out_kind = SIM_NEURAL_BACKEND_INPROCESS_ONNX_RUNTIME;
        return true;
    }
    if (sim_neural_text_iequals(text, "libtorch") ||
        sim_neural_text_iequals(text, "torch")) {
        *out_kind = SIM_NEURAL_BACKEND_INPROCESS_LIBTORCH;
        return true;
    }
    return false;
}

const char* sim_neural_execution_device_name(SimNeuralExecutionDevice device) {
    switch (device) {
        case SIM_NEURAL_EXEC_DEVICE_CPU:
            return "cpu";
        case SIM_NEURAL_EXEC_DEVICE_CUDA:
            return "cuda";
        case SIM_NEURAL_EXEC_DEVICE_MPS:
            return "mps";
        default:
            break;
    }
    return "cpu";
}

bool sim_neural_execution_device_from_string(const char* text, SimNeuralExecutionDevice* out_device) {
    if (text == NULL || out_device == NULL) {
        return false;
    }
    if (sim_neural_text_iequals(text, "cpu")) {
        *out_device = SIM_NEURAL_EXEC_DEVICE_CPU;
        return true;
    }
    if (sim_neural_text_iequals(text, "cuda") || sim_neural_text_iequals(text, "gpu")) {
        *out_device = SIM_NEURAL_EXEC_DEVICE_CUDA;
        return true;
    }
    if (sim_neural_text_iequals(text, "mps") || sim_neural_text_iequals(text, "metal")) {
        *out_device = SIM_NEURAL_EXEC_DEVICE_MPS;
        return true;
    }
    return false;
}

bool sim_neural_model_supports_device(const SimNeuralModelConfig* config,
                                      SimNeuralExecutionDevice    device) {
    uint32_t supported;
    if (config == NULL) {
        return false;
    }
    supported = sim_neural_default_device_mask(config->supported_device_mask);
    switch (device) {
        case SIM_NEURAL_EXEC_DEVICE_CPU:
            return (supported & SIM_NEURAL_DEVICE_MASK_CPU) != 0U;
        case SIM_NEURAL_EXEC_DEVICE_CUDA:
            return (supported & SIM_NEURAL_DEVICE_MASK_CUDA) != 0U;
        case SIM_NEURAL_EXEC_DEVICE_MPS:
            return (supported & SIM_NEURAL_DEVICE_MASK_MPS) != 0U;
        default:
            break;
    }
    return false;
}

SimResult sim_neural_model_registry_init(SimNeuralModelRegistry* registry) {
    if (registry == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    (void) memset(registry, 0, sizeof(*registry));
    return SIM_RESULT_OK;
}

void sim_neural_model_registry_destroy(SimNeuralModelRegistry* registry) {
    if (registry == NULL) {
        return;
    }
    free(registry->entries);
    registry->entries  = NULL;
    registry->count    = 0U;
    registry->capacity = 0U;
}

SimResult sim_neural_model_register(struct SimContext*          context,
                                    const SimNeuralModelConfig* config,
                                    size_t*                     out_index) {
    SimNeuralModelConfig local;
    SimNeuralModelEntry* entry;
    size_t               index = 0U;
    SimResult            rc;

    if (context == NULL || config == NULL || config->model_id[0] == '\0') {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    local = *config;
    if (!sim_neural_backend_kind_valid(local.backend_kind)) {
        local.backend_kind = SIM_NEURAL_BACKEND_CALLBACK;
    }
    local.supported_device_mask = sim_neural_default_device_mask(local.supported_device_mask);
    if (!sim_neural_precision_valid(local.default_precision)) {
        local.default_precision = SIM_NEURAL_PRECISION_DEFAULT;
    }

    if (local.backend_kind == SIM_NEURAL_BACKEND_CALLBACK && local.callback_fn == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (local.backend_kind == SIM_NEURAL_BACKEND_EXTERNAL_PROCESS &&
        local.external_command[0] == '\0') {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    entry = sim_neural_model_lookup_mut(&context->neural_models, local.model_id, &index);
    if (entry != NULL) {
        entry->config = local;
        sim_neural_model_runtime_stats_reset(&entry->stats);
        if (out_index != NULL) {
            *out_index = index;
        }
        return SIM_RESULT_OK;
    }

    rc = sim_neural_model_registry_reserve(&context->neural_models, context->neural_models.count + 1U);
    if (rc != SIM_RESULT_OK) {
        return rc;
    }
    index = context->neural_models.count;
    entry = &context->neural_models.entries[index];
    (void) memset(entry, 0, sizeof(*entry));
    entry->config = local;
    sim_neural_model_runtime_stats_reset(&entry->stats);
    context->neural_models.count += 1U;
    if (out_index != NULL) {
        *out_index = index;
    }
    return SIM_RESULT_OK;
}

size_t sim_neural_model_count(const struct SimContext* context) {
    if (context == NULL) {
        return 0U;
    }
    return context->neural_models.count;
}

SimResult sim_neural_model_config_at(const struct SimContext* context,
                                     size_t                   model_index,
                                     SimNeuralModelConfig*    out_config) {
    if (context == NULL || out_config == NULL || model_index >= context->neural_models.count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    *out_config = context->neural_models.entries[model_index].config;
    return SIM_RESULT_OK;
}

SimResult sim_neural_model_stats_at(const struct SimContext*    context,
                                    size_t                      model_index,
                                    SimNeuralModelRuntimeStats* out_stats) {
    if (context == NULL || out_stats == NULL || model_index >= context->neural_models.count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    *out_stats = context->neural_models.entries[model_index].stats;
    return SIM_RESULT_OK;
}

SimResult sim_neural_model_config(const struct SimContext* context,
                                  const char*              model_id,
                                  SimNeuralModelConfig*    out_config) {
    const SimNeuralModelEntry* entry = NULL;
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    entry = sim_neural_model_lookup(&context->neural_models, model_id, NULL);
    if (entry == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }
    *out_config = entry->config;
    return SIM_RESULT_OK;
}

SimResult sim_neural_model_stats(const struct SimContext*    context,
                                 const char*                 model_id,
                                 SimNeuralModelRuntimeStats* out_stats) {
    const SimNeuralModelEntry* entry = NULL;
    if (context == NULL || out_stats == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    entry = sim_neural_model_lookup(&context->neural_models, model_id, NULL);
    if (entry == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }
    *out_stats = entry->stats;
    return SIM_RESULT_OK;
}

SimResult sim_neural_model_infer(struct SimContext*               context,
                                 const char*                      model_id,
                                 const SimField*                  input,
                                 SimField*                        output,
                                 const SimNeuralInferenceRequest* request) {
    SimNeuralModelEntry*       entry;
    SimNeuralInferenceRequest  effective_request = { 0 };
    SimNeuralExecutionDevice   device = SIM_NEURAL_EXEC_DEVICE_CPU;
    SimNeuralPrecisionMode     precision;
    SimResult                  route_rc;
    SimResult                  result = SIM_RESULT_INVALID_ARGUMENT;
    uint64_t                   started_ns;
    uint64_t                   elapsed_ns;
    char                       error_text[SIM_NEURAL_MODEL_ERROR_MAX + 1U];

    if (context == NULL || model_id == NULL || model_id[0] == '\0' || input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    entry = sim_neural_model_lookup_mut(&context->neural_models, model_id, NULL);
    if (entry == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    if (request != NULL) {
        effective_request = *request;
    }
    effective_request.model_id = entry->config.model_id;
    precision = sim_neural_effective_precision(&entry->config, &effective_request);
    route_rc  = sim_neural_route_device(&entry->config, &effective_request, &device);
    if (route_rc != SIM_RESULT_OK) {
        sim_neural_record_runtime(&entry->stats,
                                  route_rc,
                                  SIM_NEURAL_EXEC_DEVICE_CPU,
                                  0U,
                                  sim_field_element_count(&input->layout),
                                  &effective_request,
                                  "requested device is not supported by the registered model");
        return route_rc;
    }
    if (effective_request.determinism_policy == SIM_NEURAL_DETERMINISM_STRICT &&
        !entry->config.deterministic) {
        sim_neural_record_runtime(&entry->stats,
                                  SIM_RESULT_NOT_SUPPORTED,
                                  device,
                                  0U,
                                  sim_field_element_count(&input->layout),
                                  &effective_request,
                                  "strict determinism requested for non-deterministic model");
        return SIM_RESULT_NOT_SUPPORTED;
    }

    error_text[0] = '\0';
    started_ns    = sim_neural_now_ns();
    switch (entry->config.backend_kind) {
        case SIM_NEURAL_BACKEND_CALLBACK:
            result = entry->config.callback_fn(
                entry->config.callback_userdata, input, output, &effective_request);
            break;
        case SIM_NEURAL_BACKEND_EXTERNAL_PROCESS:
            result = sim_neural_run_external_process(&entry->config,
                                                     input,
                                                     output,
                                                     &effective_request,
                                                     device,
                                                     precision,
                                                     error_text,
                                                     sizeof(error_text));
            break;
        case SIM_NEURAL_BACKEND_INPROCESS_ONNX_RUNTIME:
            sim_neural_copy_text(error_text, sizeof(error_text), "onnx runtime backend is not compiled in this build");
            result = SIM_RESULT_NOT_SUPPORTED;
            break;
        case SIM_NEURAL_BACKEND_INPROCESS_LIBTORCH:
            sim_neural_copy_text(error_text, sizeof(error_text), "libtorch backend is not compiled in this build");
            result = SIM_RESULT_NOT_SUPPORTED;
            break;
        default:
            sim_neural_copy_text(error_text, sizeof(error_text), "unknown neural backend");
            result = SIM_RESULT_INVALID_ARGUMENT;
            break;
    }
    elapsed_ns = sim_neural_now_ns() - started_ns;
    sim_neural_record_runtime(&entry->stats,
                              result,
                              device,
                              elapsed_ns,
                              sim_field_element_count(&input->layout),
                              &effective_request,
                              error_text);
    return result;
}
