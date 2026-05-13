/**
 * @file sim_seed.c
 * @brief Seed derivation helpers for deterministic RNG streams.
 */
#include "oakfield/sim_seed.h"

#include <stddef.h>

uint64_t sim_seed_normalize(uint64_t seed)
{
    return (seed == 0ULL) ? SIM_SEED_DEFAULT : seed;
}

uint64_t sim_seed_mix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

uint64_t sim_seed_tag(const char *tag)
{
    const unsigned char *ptr = (const unsigned char *)tag;
    uint64_t hash = 14695981039346656037ULL;
    if (ptr == NULL)
    {
        return sim_seed_mix64(hash);
    }
    while (*ptr != '\0')
    {
        hash ^= (uint64_t)(*ptr++);
        hash *= 1099511628211ULL;
    }
    return sim_seed_mix64(hash);
}

uint64_t sim_seed_derive(uint64_t base_seed, uint64_t tag_hash, uint64_t index)
{
    uint64_t base = sim_seed_normalize(base_seed);
    uint64_t mixed = base ^ tag_hash ^ (index * 0xD1342543DE82EF95ULL);
    uint64_t seed = sim_seed_mix64(mixed);
    return (seed == 0ULL) ? SIM_SEED_DEFAULT : seed;
}

void sim_seed_stream(uint64_t base_seed,
                     const char *tag,
                     uint64_t index,
                     uint64_t *out_state,
                     uint64_t *out_inc)
{
    uint64_t tag_hash = sim_seed_tag(tag);
    uint64_t state = sim_seed_derive(base_seed, tag_hash, index);
    uint64_t inc = sim_seed_derive(base_seed, tag_hash ^ 0x8C9E37F02E1A9C63ULL, index);
    if (out_state != NULL)
    {
        *out_state = state;
    }
    if (out_inc != NULL)
    {
        *out_inc = (inc << 1) | 1ULL;
    }
}
