/*
 * Migrated noise operator coverage for real Ornstein-Uhlenbeck contracts.
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
#if defined(__APPLE__)
    __sincos(angle, &s, &c);
#elif defined(__clang__) || defined(__GNUC__)
    sincos(angle, &s, &c);
#else
    s = sin(angle);
    c = cos(angle);
#endif
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

static void apply_expected(double *values, size_t count, double mean, double sigma, double tau,
                           double dt, uint64_t seed) {
    test_pcg32_t rng;
    bool has_spare = false;
    double spare = 0.0;

    test_pcg32_seed(&rng, seed, seed ^ 0xD8D12A6C9B4E41B7ULL);

    const double decay = exp(-dt / tau);
    const double noise_scale = sigma * sqrt(fmax(0.0, 1.0 - decay * decay));

    for (size_t i = 0U; i < count; ++i) {
        double n = (noise_scale == 0.0) ? 0.0 : test_normal(&rng, &has_spare, &spare);
        values[i] = mean + decay * (values[i] - mean) + noise_scale * n;
    }
}

static bool check_values(const double *got, const double *expected, size_t count,
                         const char *label) {
    for (size_t i = 0U; i < count; ++i) {
        if (!approx_equal(got[i], expected[i], 1.0e-12)) {
            fprintf(stderr, "FAIL: %s mismatch at %zu got=%.17g expected=%.17g\n", label, i, got[i],
                    expected[i]);
            return false;
        }
    }
    return true;
}

int main(void) {
    static const double initial_values[] = {1.0, -0.5, 0.25, -1.25, 0.0, 0.75, -0.9, 0.4};

    SimContext context;
    SimField field = {0};
    size_t shape[1] = {sizeof(initial_values) / sizeof(initial_values[0])};
    size_t field_index = 0U;
    size_t op_index = 0U;
    double expected[sizeof(initial_values) / sizeof(initial_values[0])];

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 1;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&context);
        return 1;
    }

    memcpy(sim_field_data(&field), initial_values, sizeof(initial_values));
    memcpy(expected, initial_values, sizeof(initial_values));

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
    };

    if (sim_add_ornstein_uhlenbeck_operator(&context, &config, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_ornstein_uhlenbeck_operator\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimOrnsteinUhlenbeckOperatorConfig fetched = {0};
    if (sim_ornstein_uhlenbeck_config(&context, op_index, &fetched) != SIM_RESULT_OK ||
        !approx_equal(fetched.mean, config.mean, 1.0e-12) ||
        !approx_equal(fetched.sigma, config.sigma, 1.0e-12) ||
        !approx_equal(fetched.tau, config.tau, 1.0e-12) || fetched.seed != config.seed) {
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

    apply_expected(expected, shape[0], config.mean, config.sigma, config.tau, 0.2, config.seed);

    if (!check_values((const double *)sim_field_data(sim_context_field(&context, field_index)),
                      expected, shape[0], "step1")) {
        sim_context_destroy(&context);
        return 1;
    }

    config.mean = -0.2;
    config.sigma = 0.1;
    config.tau = 0.5;
    config.seed = 777ULL;

    if (sim_ornstein_uhlenbeck_update(&context, op_index, &config) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_ornstein_uhlenbeck_update\n");
        sim_context_destroy(&context);
        return 1;
    }
    if (sim_ornstein_uhlenbeck_config(&context, op_index, &fetched) != SIM_RESULT_OK ||
        !approx_equal(fetched.mean, config.mean, 1.0e-12) ||
        !approx_equal(fetched.sigma, config.sigma, 1.0e-12) ||
        !approx_equal(fetched.tau, config.tau, 1.0e-12) || fetched.seed != config.seed) {
        fprintf(stderr, "FAIL: updated config mismatch\n");
        sim_context_destroy(&context);
        return 1;
    }

    if (sim_context_prepare_plan(&context) != SIM_RESULT_OK ||
        sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: second execute\n");
        sim_context_destroy(&context);
        return 1;
    }

    apply_expected(expected, shape[0], config.mean, config.sigma, config.tau, 0.2, config.seed);

    if (!check_values((const double *)sim_field_data(sim_context_field(&context, field_index)),
                      expected, shape[0], "step2")) {
        sim_context_destroy(&context);
        return 1;
    }

    sim_context_destroy(&context);
    return 0;
}
