/*
 * Migrated noise operator coverage for complex Ornstein-Uhlenbeck contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct {
    uint64_t state;
    uint64_t inc;
} test_pcg32_t;

static uint32_t test_pcg32_random(test_pcg32_t *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void test_pcg32_seed(test_pcg32_t *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    (void)test_pcg32_random(rng);
    rng->state += initstate;
    (void)test_pcg32_random(rng);
}

static double test_uniform(test_pcg32_t *rng) { return ldexp(test_pcg32_random(rng), -32); }

static void test_sincos(double angle, double *s, double *c) {
#if defined(__APPLE__)
    __sincos(angle, s, c);
#elif defined(__clang__) || defined(__GNUC__)
    sincos(angle, s, c);
#else
    if (s != NULL) {
        *s = sin(angle);
    }
    if (c != NULL) {
        *c = cos(angle);
    }
#endif
}

static void test_normal_pair(test_pcg32_t *rng, double *n0, double *n1) {
    double u1 = test_uniform(rng);
    double u2 = test_uniform(rng);
    if (u1 <= 1.0e-12) {
        u1 = 1.0e-12;
    }

    double mag = sqrt(-2.0 * log(u1));
    double angle = 2.0 * M_PI * u2;
    double s = 0.0;
    double c = 0.0;

    test_sincos(angle, &s, &c);
    if (n0 != NULL) {
        *n0 = mag * c;
    }
    if (n1 != NULL) {
        *n1 = mag * s;
    }
}

static double test_normal(test_pcg32_t *rng, bool *has_spare, double *spare) {
    if (*has_spare) {
        *has_spare = false;
        return *spare;
    }

    double n0 = 0.0;
    double n1 = 0.0;

    test_normal_pair(rng, &n0, &n1);
    *spare = n1;
    *has_spare = true;
    return n0;
}

static bool approx_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= tol * scale;
}

static double wrap_angle(double angle) {
    double wrapped = remainder(angle, 2.0 * M_PI);
    if (wrapped <= -M_PI) {
        wrapped += 2.0 * M_PI;
    } else if (wrapped > M_PI) {
        wrapped -= 2.0 * M_PI;
    }
    return wrapped;
}

static void apply_expected_component(SimComplexDouble *values, size_t count, double mean,
                                     double sigma, double tau, double dt, uint64_t seed) {
    test_pcg32_t rng;
    bool has_spare = false;
    double spare = 0.0;
    const double decay = exp(-dt / tau);
    const double noise_scale = sigma * sqrt(fmax(0.0, 1.0 - decay * decay));

    test_pcg32_seed(&rng, seed, seed ^ 0xD8D12A6C9B4E41B7ULL);

    for (size_t i = 0U; i < count; ++i) {
        double nr = 0.0;
        double ni = 0.0;

        if (noise_scale != 0.0) {
            nr = test_normal(&rng, &has_spare, &spare);
            ni = test_normal(&rng, &has_spare, &spare);
        }

        values[i].re = mean + decay * (values[i].re - mean) + noise_scale * nr;
        values[i].im = mean + decay * (values[i].im - mean) + noise_scale * ni;
    }
}

static void apply_expected_polar(SimComplexDouble *values, size_t count, double mean, double sigma,
                                 double tau, double dt, uint64_t seed) {
    test_pcg32_t rng;
    bool has_spare = false;
    double spare = 0.0;
    const double decay = exp(-dt / tau);
    const double noise_scale = sigma * sqrt(fmax(0.0, 1.0 - decay * decay));

    test_pcg32_seed(&rng, seed, seed ^ 0xD8D12A6C9B4E41B7ULL);

    for (size_t i = 0U; i < count; ++i) {
        double radius = hypot(values[i].re, values[i].im);
        double phase = atan2(values[i].im, values[i].re);
        double nr = 0.0;
        double np = 0.0;
        double s = 0.0;
        double c = 0.0;

        if (noise_scale != 0.0) {
            nr = test_normal(&rng, &has_spare, &spare);
            np = test_normal(&rng, &has_spare, &spare);
        }

        radius = mean + decay * (radius - mean) + noise_scale * nr;
        phase = mean + decay * (phase - mean) + noise_scale * np;
        if (radius < 0.0) {
            radius = -radius;
            phase += M_PI;
        }
        phase = wrap_angle(phase);
        test_sincos(phase, &s, &c);
        values[i].re = radius * c;
        values[i].im = radius * s;
    }
}

static bool check_complex_values(const SimComplexDouble *got, const SimComplexDouble *expected,
                                 size_t count, const char *label) {
    for (size_t i = 0U; i < count; ++i) {
        if (!approx_equal(got[i].re, expected[i].re, 1.0e-12) ||
            !approx_equal(got[i].im, expected[i].im, 1.0e-12)) {
            fprintf(stderr, "FAIL: %s mismatch at %zu got=(%.17g, %.17g) expected=(%.17g, %.17g)\n",
                    label, i, got[i].re, got[i].im, expected[i].re, expected[i].im);
            return false;
        }
    }
    return true;
}

int main(void) {
    static const SimComplexDouble initial_values[] = {
        {1.0, -0.5}, {0.25, 0.75}, {-1.25, 0.5}, {0.0, -0.9}, {0.4, 0.2}, {-0.7, -0.3},
    };

    SimContext context;
    SimField field = {0};
    size_t shape[2] = {2U, 3U};
    size_t field_index = 0U;
    size_t op_index = 0U;
    SimComplexDouble expected_component[sizeof(initial_values) / sizeof(initial_values[0])];
    SimComplexDouble expected_polar[sizeof(initial_values) / sizeof(initial_values[0])];
    const size_t count = sizeof(initial_values) / sizeof(initial_values[0]);

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 1;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&context);
        return 1;
    }

    memcpy(sim_field_data(&field), initial_values, sizeof(initial_values));
    memcpy(expected_component, initial_values, sizeof(initial_values));
    memcpy(expected_polar, initial_values, sizeof(initial_values));

    if (sim_context_add_field(&context, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimOrnsteinUhlenbeckOperatorConfig config = {
        .field_index = field_index,
        .mean = 0.15,
        .sigma = 0.35,
        .tau = 0.8,
        .seed = 12345ULL,
        .complex_mode = SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT,
    };

    if (sim_add_ornstein_uhlenbeck_operator(&context, &config, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_ornstein_uhlenbeck_operator\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimOperator *op = sim_operator_registry_get(&context.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: operator lookup\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimOperatorInfo info = sim_operator_info(op);
    if (info.category != SIM_OPERATOR_CATEGORY_DIFFUSION || !info.is_noise ||
        info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
        !info.representation.requires_complex_input ||
        !info.representation.requires_complex_representation || info.preserves_real ||
        info.representation.preserves_real_subspace) {
        fprintf(stderr, "FAIL: complex operator metadata mismatch\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimOrnsteinUhlenbeckOperatorConfig fetched = {0};
    if (sim_ornstein_uhlenbeck_config(&context, op_index, &fetched) != SIM_RESULT_OK ||
        !approx_equal(fetched.mean, config.mean, 1.0e-12) ||
        !approx_equal(fetched.sigma, config.sigma, 1.0e-12) ||
        !approx_equal(fetched.tau, config.tau, 1.0e-12) || fetched.seed != config.seed ||
        fetched.complex_mode != SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_COMPONENT) {
        fprintf(stderr, "FAIL: sim_ornstein_uhlenbeck_config\n");
        sim_context_destroy(&context);
        return 1;
    }

    sim_context_set_timestep(&context, 0.2);
    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK ||
        sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: first execute\n");
        sim_context_destroy(&context);
        return 1;
    }

    apply_expected_component(expected_component, count, config.mean, config.sigma, config.tau, 0.2,
                             config.seed);

    if (!check_complex_values(
            sim_field_complex_data_const(sim_context_field(&context, field_index)),
            expected_component, count, "component")) {
        sim_context_destroy(&context);
        return 1;
    }

    config.mean = -0.2;
    config.sigma = 0.1;
    config.tau = 0.5;
    config.seed = 777ULL;
    config.complex_mode = SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_POLAR;

    if (sim_ornstein_uhlenbeck_update(&context, op_index, &config) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_ornstein_uhlenbeck_update\n");
        sim_context_destroy(&context);
        return 1;
    }

    if (sim_ornstein_uhlenbeck_config(&context, op_index, &fetched) != SIM_RESULT_OK ||
        !approx_equal(fetched.mean, config.mean, 1.0e-12) ||
        !approx_equal(fetched.sigma, config.sigma, 1.0e-12) ||
        !approx_equal(fetched.tau, config.tau, 1.0e-12) || fetched.seed != config.seed ||
        fetched.complex_mode != SIM_ORNSTEIN_UHLENBECK_COMPLEX_MODE_POLAR) {
        fprintf(stderr, "FAIL: updated config mismatch\n");
        sim_context_destroy(&context);
        return 1;
    }

    memcpy(expected_polar, expected_component, sizeof(expected_polar));
    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK ||
        sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: second execute\n");
        sim_context_destroy(&context);
        return 1;
    }

    apply_expected_polar(expected_polar, count, config.mean, config.sigma, config.tau, 0.2,
                         config.seed);

    if (!check_complex_values(
            sim_field_complex_data_const(sim_context_field(&context, field_index)), expected_polar,
            count, "polar")) {
        sim_context_destroy(&context);
        return 1;
    }

    sim_context_destroy(&context);
    return 0;
}
