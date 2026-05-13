/**
 * @file sim_seed.h
 * @brief Seed derivation helpers for deterministic RNG streams.
 */
#ifndef OAKFIELD_SIM_SEED_H
#define OAKFIELD_SIM_SEED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Default base seed used when a caller provides zero. */
#define SIM_SEED_DEFAULT 0xA5A5A5A5A5A5A5A5ULL

/**
 * @brief Normalize a seed (zero maps to SIM_SEED_DEFAULT).
 *
 * @param seed Caller-provided seed.
 * @return @p seed when nonzero, otherwise #SIM_SEED_DEFAULT.
 */
uint64_t sim_seed_normalize(uint64_t seed);

/**
 * @brief Mix a 64-bit value into a high-quality hash.
 *
 * @param value Input value.
 * @return Mixed 64-bit value suitable for stream derivation.
 */
uint64_t sim_seed_mix64(uint64_t value);

/**
 * @brief Hash a domain tag into a 64-bit value.
 *
 * @param tag Null-terminated tag; NULL hashes as an empty tag.
 * @return Deterministic 64-bit tag hash.
 */
uint64_t sim_seed_tag(const char *tag);

/**
 * @brief Derive a deterministic seed from a base seed, tag hash, and index.
 *
 * @param base_seed Base seed; zero is normalized to #SIM_SEED_DEFAULT.
 * @param tag_hash Domain tag hash, often from sim_seed_tag().
 * @param index Numeric substream index.
 * @return Derived child seed.
 */
uint64_t sim_seed_derive(uint64_t base_seed, uint64_t tag_hash, uint64_t index);

/**
 * @brief Derive a PCG32 stream (state + increment) from a base seed and domain tag.
 *
 * @param base_seed Base seed; zero is normalized to #SIM_SEED_DEFAULT.
 * @param tag Null-terminated domain tag for stream separation.
 * @param index Numeric substream index.
 * @param[out] out_state Receives the PCG state when non-NULL.
 * @param[out] out_inc Receives the PCG increment when non-NULL.
 */
void sim_seed_stream(uint64_t base_seed, const char *tag, uint64_t index, uint64_t *out_state,
                     uint64_t *out_inc);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_SEED_H */
