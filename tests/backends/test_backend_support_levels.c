#include <oakfield/backend.h>

#include <stdio.h>

static int expect_true(const char* label, int ok) {
    if (!ok) {
        fprintf(stderr, "backend support-level smoke check failed: %s\n", label);
        return 1;
    }
    return 0;
}

static int check_cpu_backend(void) {
    SimBackend backend = { 0 };
    int        failures = 0;

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    failures += expect_true("cpu init", backend.last_error == SIM_RESULT_OK);
    failures += expect_true("cpu type", backend.type == SIM_BACKEND_TYPE_CPU);
    failures += expect_true("cpu analytic warp feature",
                            backend_supports_feature(&backend, SIM_BACKEND_FEATURE_ANALYTIC_WARP));
    failures += expect_true(
        "cpu boundary-aware feature",
        backend_supports_feature(&backend, SIM_BACKEND_FEATURE_BOUNDARY_AWARE_DIFFS));

    backend_destroy(&backend);
    failures += expect_true("cpu destroy clears features", backend.features == SIM_BACKEND_FEATURE_NONE);
    return failures;
}

static int check_unavailable_optional_backend(SimBackendType type, const char* label) {
    SimBackend backend = { 0 };
    int        failures = 0;

    backend.type = type;
    backend_init(&backend);
    failures += expect_true(label, backend.last_error == SIM_RESULT_NOT_FOUND);
    failures += expect_true("unavailable backend has no features",
                            backend.features == SIM_BACKEND_FEATURE_NONE);
    backend_destroy(&backend);
    return failures;
}

int main(void) {
    int failures = 0;

    failures += check_cpu_backend();

#if !defined(SIM_HAVE_CUDA)
    failures += check_unavailable_optional_backend(SIM_BACKEND_TYPE_CUDA, "cuda unavailable");
#endif

#if !defined(SIM_HAVE_METAL)
    failures += check_unavailable_optional_backend(SIM_BACKEND_TYPE_METAL, "metal unavailable");
#endif

    return failures == 0 ? 0 : 1;
}
