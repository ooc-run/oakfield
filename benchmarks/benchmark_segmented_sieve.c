#include <oakfield/sim.h>

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double benchmark_now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + ((double)ts.tv_nsec * 1.0e-9);
}

static bool build_base_primes(uint64_t limit, uint64_t **out_primes, size_t *out_count) {
    bool *marked = NULL;
    uint64_t *primes = NULL;
    size_t count = 0U;

    if (out_primes == NULL || out_count == NULL) {
        return false;
    }
    *out_primes = NULL;
    *out_count = 0U;
    if (limit < 2U) {
        return true;
    }

    marked = (bool *)calloc((size_t)(limit + 1U), sizeof(bool));
    primes = (uint64_t *)calloc((size_t)(limit + 1U), sizeof(uint64_t));
    if (marked == NULL || primes == NULL) {
        free(marked);
        free(primes);
        return false;
    }

    for (uint64_t p = 2U; p <= limit; ++p) {
        if (marked[p]) {
            continue;
        }
        primes[count++] = p;
        for (uint64_t multiple = p * p; multiple <= limit; multiple += p) {
            marked[multiple] = true;
        }
    }

    free(marked);
    *out_primes = primes;
    *out_count = count;
    return true;
}

int main(int argc, char **argv) {
    const uint64_t default_limit = 200000U;
    const size_t default_segment_size = 32768U;
    uint64_t limit = default_limit;
    size_t segment_size = default_segment_size;
    uint64_t *base_primes = NULL;
    size_t base_prime_count = 0U;
    SimContext context = {0};
    SimBackend backend = {0};
    SimField candidates = {0};
    SimField primes = {0};
    SimField flags = {0};
    size_t shape[1];
    size_t primes_shape[1];
    size_t candidates_index = 0U;
    size_t primes_index = 0U;
    size_t flags_index = 0U;
    bool context_ready = false;
    bool backend_ready = false;
    bool candidates_owned = false;
    bool primes_owned = false;
    bool flags_owned = false;
    uint64_t prime_count = 0U;
    double elapsed_seconds = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;

    if (argc >= 2) {
        limit = (uint64_t)strtoull(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        segment_size = (size_t)strtoull(argv[2], NULL, 10);
    }
    if (limit < 2U || segment_size == 0U) {
        fprintf(stderr, "usage: %s [limit>=2] [segment_size>=1]\n", argv[0]);
        return 1;
    }

    {
        uint64_t root = 1U;
        while ((root + 1U) * (root + 1U) <= limit) {
            root += 1U;
        }
        if (!build_base_primes(root, &base_primes, &base_prime_count)) {
            fprintf(stderr, "failed to build base primes\n");
            return 1;
        }
    }

    shape[0] = segment_size;
    primes_shape[0] = base_prime_count;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init context\n");
        free(base_primes);
        return 1;
    }
    context_ready = true;

    if (sim_field_init_typed(&candidates, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&primes, 1U, primes_shape, sim_scalar_domain_u32(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&flags, 1U, shape, sim_scalar_domain_u8(), SIM_FIELD_STORAGE_ROW_MAJOR,
                             NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init integer fields\n");
        goto cleanup;
    }
    candidates_owned = true;
    primes_owned = true;
    flags_owned = true;

    if (base_prime_count > 0U) {
        uint32_t *prime_data = sim_field_u32_data(&primes);
        if (prime_data == NULL) {
            fprintf(stderr, "failed to access base prime field\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < base_prime_count; ++i) {
            if (base_primes[i] > (uint64_t)UINT32_MAX) {
                fprintf(stderr, "base prime exceeds u32 storage range\n");
                goto cleanup;
            }
            prime_data[i] = (uint32_t)base_primes[i];
        }
    }

    if (sim_context_add_field(&context, &candidates, &candidates_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &primes, &primes_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &flags, &flags_index) != SIM_RESULT_OK) {
        fprintf(stderr, "failed to add integer fields to context\n");
        goto cleanup;
    }
    candidates_owned = false;
    primes_owned = false;
    flags_owned = false;

    {
        SimSegmentedSieveMarkBatchOperatorConfig config = {0};
        config.candidate_field = candidates_index;
        config.primes_field = primes_index;
        config.flags_field = flags_index;
        if (sim_add_segmented_sieve_mark_batch_operator(&context, &config, NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "failed to add segmented_sieve_mark_batch operator\n");
            goto cleanup;
        }
    }

    backend.type = SIM_BACKEND_TYPE_CPU;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        fprintf(stderr, "failed to init CPU backend\n");
        goto cleanup;
    }
    backend_ready = true;
    sim_context_set_backend(&context, &backend);

    start_time = benchmark_now_seconds();
    for (uint64_t low = 0U; low <= limit; low += (uint64_t)segment_size) {
        uint64_t high = low + (uint64_t)segment_size - 1U;
        size_t active_count = segment_size;
        uint64_t *candidate_data;
        uint8_t *flag_data;

        if (high > limit) {
            high = limit;
            active_count = (size_t)(high - low + 1U);
        }

        candidate_data = sim_field_u64_data(sim_context_field(&context, candidates_index));
        flag_data = sim_field_u8_data(sim_context_field(&context, flags_index));
        if (candidate_data == NULL || flag_data == NULL) {
            fprintf(stderr, "failed to access segmented sieve fields\n");
            goto cleanup;
        }

        for (size_t i = 0U; i < segment_size; ++i) {
            candidate_data[i] = low + (uint64_t)i;
            flag_data[i] = (candidate_data[i] >= 2U) ? 1U : 0U;
        }

        if (sim_context_execute(&context) != SIM_RESULT_OK) {
            fprintf(stderr, "segmented sieve mark execute failed for low=%" PRIu64 "\n", low);
            goto cleanup;
        }

        for (size_t i = 0U; i < active_count; ++i) {
            if (flag_data[i] != 0U) {
                prime_count += 1U;
            }
        }
    }
    end_time = benchmark_now_seconds();
    elapsed_seconds = end_time - start_time;

    printf("benchmark_segmented_sieve limit=%" PRIu64
           " segment_size=%zu base_primes=%zu primes=%" PRIu64 " time=%.3f ms\n",
           limit, segment_size, base_prime_count, prime_count, elapsed_seconds * 1000.0);

cleanup:
    free(base_primes);
    if (backend_ready) {
        backend_destroy(&backend);
    }
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (candidates_owned) {
        sim_field_destroy(&candidates);
    }
    if (primes_owned) {
        sim_field_destroy(&primes);
    }
    if (flags_owned) {
        sim_field_destroy(&flags);
    }

    return 0;
}
