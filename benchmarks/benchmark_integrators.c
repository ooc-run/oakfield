#include <oakfield/sim.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#endif

/* Forward declarations for integrator factories (not exposed via headers). */
SimResult integrator_euler_create(const IntegratorConfig *config, Integrator *out);
SimResult integrator_heun_create(const IntegratorConfig *config, Integrator *out);
SimResult integrator_rk4_create(const IntegratorConfig *config, Integrator *out);
SimResult integrator_rkf45_create(const IntegratorConfig *config, Integrator *out);
SimResult integrator_backward_euler_create(const IntegratorConfig *config, Integrator *out);
SimResult integrator_crank_nicolson_create(const IntegratorConfig *config, Integrator *out);

#include <oakfield/integrator_registry.h>

typedef SimResult (*IntegratorFactory)(const IntegratorConfig *config, Integrator *out);

typedef struct IntegratorCase {
    const char *name;
    IntegratorFactory create;
} IntegratorCase;

typedef enum NoiseMode { NOISE_NONE = 0, NOISE_GAUSSIAN, NOISE_UNIFORM, NOISE_LAPLACE } NoiseMode;

typedef struct IntegratorRunResult {
    double ms_per_step;
    double elapsed;
    double max_error;
    bool success;
    bool has_error_metric;
    bool effective_adaptive;
} IntegratorRunResult;

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq = {0}, counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
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

static SimResult simple_drift(Integrator *integrator, const Field *field, const double *state,
                              double *out_derivative, size_t count) {
    (void)integrator;
    (void)field;
    for (size_t i = 0; i < count; ++i) {
        out_derivative[i] = -0.1 * state[i];
    }
    return SIM_RESULT_OK;
}

static NoiseMode noise_mode_from_string(const char *text) {
    if (text == NULL) {
        return NOISE_NONE;
    }
    if (strcasecmp(text, "gaussian") == 0)
        return NOISE_GAUSSIAN;
    if (strcasecmp(text, "uniform") == 0)
        return NOISE_UNIFORM;
    if (strcasecmp(text, "laplace") == 0)
        return NOISE_LAPLACE;
    return NOISE_NONE;
}

static const char *noise_mode_label(NoiseMode mode) {
    switch (mode) {
    case NOISE_GAUSSIAN:
        return "gaussian";
    case NOISE_UNIFORM:
        return "uniform";
    case NOISE_LAPLACE:
        return "laplace";
    default:
        return "none";
    }
}

static IntegratorNoiseFn noise_fn_for_mode(NoiseMode mode) {
    switch (mode) {
    case NOISE_UNIFORM:
        return integrator_noise_uniform;
    case NOISE_LAPLACE:
        return integrator_noise_laplace;
    case NOISE_GAUSSIAN:
    default:
        return integrator_noise_gaussian;
    }
}

static void fill_field(SimField *field, double seed) {
    double *data = (double *)sim_field_data(field);
    size_t count = sim_field_bytes(field) / sizeof(double);
    for (size_t i = 0; i < count; ++i) {
        data[i] = seed + 0.001 * (double)i;
    }
}

static double evaluate_linear_drift_error(const SimField *field, double seed, double elapsed) {
    const double *data = (const double *)sim_field_data_const(field);
    size_t count = sim_field_bytes(field) / sizeof(double);
    double decay = exp(-0.1 * elapsed);
    double max_error = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double initial = seed + 0.001 * (double)i;
        double expected = initial * decay;
        double error = fabs(data[i] - expected);
        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}

static double validation_tolerance_for_case(const char *name, bool adaptive) {
    (void)adaptive;

    if (name == NULL) {
        return 0.0;
    }
    if (strcmp(name, "euler") == 0) {
        return 1.0e-4;
    }
    if (strcmp(name, "heun") == 0) {
        return 1.0e-6;
    }
    if (strcmp(name, "rk4") == 0) {
        return 1.0e-8;
    }
    if (strcmp(name, "rkf45") == 0) {
        return 1.0e-7;
    }
    if (strcmp(name, "backward_euler") == 0) {
        return 1.0e-4;
    }
    if (strcmp(name, "crank_nicolson") == 0) {
        return 1.0e-6;
    }
    return 0.0;
}

static IntegratorRunResult run_integrator_case(const IntegratorCase *icase, const size_t *shape,
                                               size_t rank, int iterations, bool adaptive,
                                               NoiseMode noise_mode) {
    const double seed = 0.5;
    SimField field = {0};
    size_t seed_shape[3] = {shape[0], shape[1], shape[2]};
    IntegratorRunResult metrics = {0.0, 0.0, 0.0, false, false, false};
    if (sim_field_init(&field, rank, seed_shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[ERROR] %s: field init failed\n", icase->name);
        metrics.ms_per_step = -1.0;
        return metrics;
    }

    IntegratorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.drift = simple_drift;
    cfg.noise = (noise_mode == NOISE_NONE) ? NULL : noise_fn_for_mode(noise_mode);
    cfg.initial_dt = 0.01;
    cfg.min_dt = 1.0e-6;
    cfg.max_dt = 0.1;
    cfg.tolerance = 1.0e-4;
    cfg.safety = 0.9;
    cfg.adaptive = adaptive;
    cfg.enable_stochastic = (noise_mode != NOISE_NONE);
    cfg.stochastic_strength = (noise_mode != NOISE_NONE) ? 1.0 : 0.0;
    cfg.random_seed = 0x12345678U;
    cfg.workspace_hint = 0U;

    Integrator integrator;
    if (icase->create(&cfg, &integrator) != SIM_RESULT_OK) {
        fprintf(stderr, "[ERROR] %s: integrator create failed\n", icase->name);
        sim_field_destroy(&field);
        metrics.ms_per_step = -1.0;
        return metrics;
    }
    metrics.effective_adaptive = integrator.adaptive;

    fill_field(&field, seed);

    double t0 = monotonic_seconds();
    for (int i = 0; i < iterations; ++i) {
        double last_step = 0.0;
        integrator.step(&integrator, &field, cfg.initial_dt);
        last_step = integrator_last_step(&integrator);
        if (!(last_step > 0.0)) {
            last_step = cfg.initial_dt;
        }
        metrics.elapsed += last_step;
    }
    double t1 = monotonic_seconds();

    metrics.ms_per_step = ((t1 - t0) / (double)iterations) * 1.0e3; /* ms per step */
    metrics.has_error_metric = (noise_mode == NOISE_NONE);
    if (metrics.has_error_metric) {
        metrics.max_error = evaluate_linear_drift_error(&field, seed, metrics.elapsed);
    }
    metrics.success = true;

    integrator_destroy(&integrator);
    sim_field_destroy(&field);

    return metrics;
}

int main(int argc, char **argv) {
    int iterations = 1024;
    size_t rank = 1U;
    size_t shape[3] = {1024U, 1024U, 1U};
    const char *integrator_filter = NULL;
    bool filter_adaptive_set = false;
    bool filter_adaptive = false;
    NoiseMode filter_noise = NOISE_NONE;
    bool filter_noise_set = false;
    bool validate = false;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--validate") == 0) {
            validate = true;
            continue;
        }
        if (strncmp(arg, "--iterations=", 13) == 0) {
            int parsed = atoi(arg + 13);
            if (parsed > 0) {
                iterations = parsed;
            }
            continue;
        }
        if (strncmp(arg, "--shape=", 8) == 0) {
            size_t w = 0U, h = 0U;
            if (sscanf(arg + 8, "%zux%zu", &w, &h) == 2 && w > 0U && h > 0U) {
                shape[0] = w;
                shape[1] = h;
            }
            continue;
        }
        if (strncmp(arg, "--rank=", 7) == 0) {
            size_t r = (size_t)atoi(arg + 7);
            if (r >= 1U && r <= 3U) {
                rank = r;
            }
            continue;
        }
        if (strncmp(arg, "--integrator=", 13) == 0) {
            integrator_filter = arg + 13;
            continue;
        }
        if (strcmp(arg, "--adaptive=true") == 0) {
            filter_adaptive_set = true;
            filter_adaptive = true;
            continue;
        }
        if (strcmp(arg, "--adaptive=false") == 0) {
            filter_adaptive_set = true;
            filter_adaptive = false;
            continue;
        }
        if (strncmp(arg, "--noise=", 8) == 0) {
            filter_noise = noise_mode_from_string(arg + 8);
            filter_noise_set = true;
            continue;
        }
        if (arg[0] != '-') {
            int parsed = atoi(arg);
            if (parsed > 0) {
                iterations = parsed;
            }
        }
    }

    if (validate && !filter_noise_set) {
        filter_noise = NOISE_NONE;
        filter_noise_set = true;
    }

    const IntegratorCase integrators[] = {
        {"euler", integrator_euler_create},
        {"heun", integrator_heun_create},
        {"rk4", integrator_rk4_create},
        {"rkf45", integrator_rkf45_create},
        {"backward_euler", integrator_backward_euler_create},
        {"crank_nicolson", integrator_crank_nicolson_create},
    };
    size_t integrator_count = sizeof(integrators) / sizeof(integrators[0]);

    const NoiseMode noises[] = {NOISE_NONE, NOISE_GAUSSIAN, NOISE_UNIFORM, NOISE_LAPLACE};
    const bool adaptive_modes[] = {false, true};

    printf("Integrator benchmark (iterations=%d, rank=%zu, shape=%zux%zu%s)\n", iterations, rank,
           shape[0], shape[1], (rank > 2U) ? "x1" : "");

    bool ran_any = false;
    bool validation_ok = true;
    for (size_t i = 0; i < integrator_count; ++i) {
        if (integrator_filter != NULL && strcmp(integrator_filter, integrators[i].name) != 0) {
            continue;
        }
        for (size_t a = 0; a < sizeof(adaptive_modes) / sizeof(adaptive_modes[0]); ++a) {
            bool adaptive = adaptive_modes[a];
            if (filter_adaptive_set && adaptive != filter_adaptive) {
                continue;
            }
            for (size_t n = 0; n < sizeof(noises) / sizeof(noises[0]); ++n) {
                NoiseMode mode = noises[n];
                if (filter_noise_set && mode != filter_noise) {
                    continue;
                }
                IntegratorRunResult run =
                    run_integrator_case(&integrators[i], shape, rank, iterations, adaptive, mode);
                if (!run.success || run.ms_per_step < 0.0) {
                    printf("%-18s | adaptive=%-7s | noise=%-8s : ERROR\n", integrators[i].name,
                           (run.effective_adaptive == adaptive)
                               ? (adaptive ? "on" : "off")
                               : (adaptive ? "on->off" : "off->on"),
                           noise_mode_label(mode));
                    validation_ok = false;
                } else {
                    if (run.has_error_metric) {
                        printf("%-18s | adaptive=%-7s | noise=%-8s : %.4f ms | elapsed=%.6f | "
                               "max_error=%.12e\n",
                               integrators[i].name,
                               (run.effective_adaptive == adaptive)
                                   ? (adaptive ? "on" : "off")
                                   : (adaptive ? "on->off" : "off->on"),
                               noise_mode_label(mode), run.ms_per_step, run.elapsed, run.max_error);
                    } else {
                        printf("%-18s | adaptive=%-7s | noise=%-8s : %.4f ms\n",
                               integrators[i].name,
                               (run.effective_adaptive == adaptive)
                                   ? (adaptive ? "on" : "off")
                                   : (adaptive ? "on->off" : "off->on"),
                               noise_mode_label(mode), run.ms_per_step);
                    }

                    if (validate && run.has_error_metric) {
                        double tolerance = validation_tolerance_for_case(integrators[i].name,
                                                                         run.effective_adaptive);
                        if (!(tolerance > 0.0) || run.max_error > tolerance) {
                            fprintf(stderr, "[FAIL] %s adaptive=%s max_error %.12e exceeds %.12e\n",
                                    integrators[i].name, run.effective_adaptive ? "on" : "off",
                                    run.max_error, tolerance);
                            validation_ok = false;
                        }
                    }
                }
                ran_any = true;
            }
        }
    }

    if (!ran_any) {
        fprintf(stderr, "[ERROR] no integrators matched the requested filters\n");
        return 1;
    }

    return validation_ok ? 0 : 1;
}
