#include <oakfield/math.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile double g_sink = 0.0;

static double seconds_now(void) {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

static int parse_iterations(int argc, char** argv, int* out_iterations) {
    for (int i = 1; i < argc; ++i) {
        const char* prefix     = "--iterations=";
        size_t      prefix_len = strlen(prefix);
        char*       endptr     = NULL;
        long        parsed;

        if (strcmp(argv[i], "--help") == 0) {
            printf("usage: benchmark_experimental_math [--iterations=N]\n");
            return 1;
        }
        if (strncmp(argv[i], prefix, prefix_len) != 0) {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            return -1;
        }

        parsed = strtol(argv[i] + prefix_len, &endptr, 10);
        if (endptr == argv[i] + prefix_len || *endptr != '\0' || parsed <= 0L ||
            parsed > 1000000L) {
            fprintf(stderr, "invalid iteration count: %s\n", argv[i]);
            return -1;
        }
        *out_iterations = (int) parsed;
    }
    return 0;
}

static void print_result(const char* name, double seconds, double calls, double checksum) {
    double rate = seconds > 0.0 ? calls / seconds : 0.0;
    printf("%s seconds=%.6f calls=%.0f calls_per_sec=%.3f checksum=%.12g\n",
           name,
           seconds,
           calls,
           rate,
           checksum);
}

static void run_q_methods(int iterations) {
    double start    = seconds_now();
    double checksum = 0.0;

    for (int i = 0; i < iterations; ++i) {
        double q = 0.72 + 0.2 * (double) (i % 17) / 16.0;
        checksum += sim_q_number(5.0, q);
        checksum += sim_qhyperexp_phi(0.62, 0.08, 7, q);
        checksum += sim_qhyperexp_phi_deriv(0.62, 0.08, 7, q);
        checksum += sim_q_digamma(1.5, q);
    }

    g_sink += checksum;
    print_result(
        "EXPERIMENTAL_q_methods", seconds_now() - start, (double) iterations * 4.0, checksum);
}

static void run_zeta_methods(int iterations) {
#if OAKFIELD_ENABLE_ZETA_CORE
    double         start        = seconds_now();
    double         checksum     = 0.0;
    SimZetaContext zeta_context = sim_zeta_context_interactive();
    SimXiContext   xi_context   = sim_xi_context_interactive();

    for (int i = 0; i < iterations; ++i) {
        double        t    = 1.0 + 0.125 * (double) (i % 32);
        SimZetaResult zeta = sim_zeta_eval((SimComplexDouble){ 0.5, t }, &zeta_context);
        SimXiResult   xi   = sim_xi_eval((SimComplexDouble){ 0.5, t }, &xi_context);
        checksum += zeta.value.re + zeta.value.im + xi.value.re + xi.value.im;
    }

    g_sink += checksum;
    print_result("zeta_xi", seconds_now() - start, (double) iterations * 2.0, checksum);
#else
    (void) iterations;
    printf("zeta_xi skipped: OAKFIELD_ENABLE_ZETA_CORE=0\n");
#endif
}

int main(int argc, char** argv) {
    int iterations   = 32;
    int parse_result = parse_iterations(argc, argv, &iterations);

    if (parse_result > 0) {
        return 0;
    }
    if (parse_result < 0) {
        return 1;
    }

    printf("benchmark_experimental_math iterations=%d\n", iterations);
    printf("These timings include supported Zeta/Xi paths and exploratory q-method APIs.\n");

    run_q_methods(iterations);
    run_zeta_methods(iterations);

    if (g_sink == -1.0) {
        printf("sink=%.1f\n", g_sink);
    }
    return 0;
}
