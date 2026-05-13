/**
 * @file sim_world.h
 * @brief Static world container coordinating fields, operators, and universe specification.
 */
#ifndef OAKFIELD_SIM_WORLD_H
#define OAKFIELD_SIM_WORLD_H

#include <stdbool.h>
#include <stddef.h>

#include "field.h"
#include "operator.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SIM_WORLD_DEFAULT_Q
#define SIM_WORLD_DEFAULT_Q 1.0f
#endif

#ifndef SIM_WORLD_DEFAULT_K
#define SIM_WORLD_DEFAULT_K 20U
#endif

#ifndef SIM_WORLD_DEFAULT_EPSILON
#define SIM_WORLD_DEFAULT_EPSILON 0.5f
#endif

#ifndef SIM_WORLD_DEFAULT_SIEVE_SIGMA
#define SIM_WORLD_DEFAULT_SIEVE_SIGMA 0.1f
#endif

struct SimOperator;

/**
 * @brief Pole singularity specification for universe geometry.
 */
typedef struct SimPole {
    double x, y, z;   /**< Position (z unused for 2D). */
    double residue;   /**< Charge/strength at pole. */
    const char *type; /**< "digamma", "trigamma", "tetragamma". */
} SimPole;

/**
 * @brief Options controlling synthesized pole fields.
 */
typedef struct SimPoleFieldOptions {
    double origin_x;  /**< World-space x origin for sample 0. */
    double origin_y;  /**< World-space y origin for row 0. */
    double spacing_x; /**< World-space x spacing between adjacent columns. */
    double spacing_y; /**< World-space y spacing between adjacent rows. */
    double plane_z;   /**< Fixed z coordinate used when sampling 2D fields. */
    double softening; /**< Positive denominator softening near pole singularities. */
} SimPoleFieldOptions;

/**
 * @brief Universe/experiment specification defining mathematical structure.
 *
 * This describes the "physics" of the toy universe - immutable after init.
 */
typedef struct SimUniverseSpec {
    SimPole *poles;    /**< Owned array of singularity specifications. */
    size_t pole_count; /**< Number of entries in poles. */

    double q;       /**< Quantum deformation parameter ∈ (0,1); 1.0 = classical. */
    size_t K;       /**< Hyperexponential truncation level. */
    double epsilon; /**< Base offset for hyperexponential φ_ε^[K]. */

    double sieve_sigma; /**< Default sieve scale for remainder analysis. */
} SimUniverseSpec;

/**
 * @brief Static world container holding fields, registry, and universe spec.
 */
typedef struct SimFieldContinuityOverride {
    bool enabled;             /**< True when an override is active for the field. */
    SimOperatorConfig config; /**< Continuity parameters applied when enabled. */
} SimFieldContinuityOverride;

/**
 * @brief Mutable world container that owns fields, operators, and universe data.
 */
typedef struct SimWorld {
    SimField *fields;                             /**< Owned field collection. */
    size_t field_count;                           /**< Number of active fields. */
    size_t field_capacity;                        /**< Allocated capacity for fields. */
    SimFieldContinuityOverride *field_continuity; /**< Per-field continuity overrides. */
    SimOperatorRegistry operators;                /**< Operator registry. */
    SimUniverseSpec universe; /**< Universe specification (immutable structure). */
} SimWorld;

/**
 * @brief Initialize world container and copy universe specification.
 *
 * @param world World container to initialize.
 * @param universe_spec Optional universe specification; NULL selects defaults.
 * @return #SIM_RESULT_OK or an allocation/registry initialization error.
 */
SimResult sim_world_init(SimWorld *world, const SimUniverseSpec *universe_spec);

/**
 * @brief Ensure field storage can accommodate @p additional entries.
 *
 * @param world World container to grow.
 * @param additional Number of additional fields required.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_world_reserve_fields(SimWorld *world, size_t additional);

/**
 * @brief Reset owned field storage and release buffers.
 *
 * @param world World container to reset; NULL is ignored.
 */
void sim_world_reset_fields(SimWorld *world);

/**
 * @brief Destroy world container resources.
 *
 * @param world World container to destroy; NULL is ignored.
 */
void sim_world_destroy(SimWorld *world);

/**
 * @brief Assign or clear a per-field continuity override.
 *
 * @param world World container to update.
 * @param field_index Field index to configure.
 * @param enabled Whether the override should be active.
 * @param config Continuity config to copy when enabled; required if enabled.
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_world_set_field_continuity_override(SimWorld *world, size_t field_index, bool enabled,
                                                  const SimOperatorConfig *config);

/**
 * @brief Query the active continuity override for a field, if present.
 *
 * @param world World container to inspect.
 * @param field_index Field index to query.
 * @param[out] out_config Optional receiver for the override config.
 * @return true when an enabled override exists for the field.
 */
bool sim_world_field_continuity_override(const SimWorld *world, size_t field_index,
                                         SimOperatorConfig *out_config);
#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_SIM_WORLD_H */
