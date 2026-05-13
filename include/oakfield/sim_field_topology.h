/**
 * @file sim_field_topology.h
 * @brief Phase-topology extraction helpers for complex simulation fields.
 *
 * @details The topology pass interprets the fastest-varying field dimension as
 * image width and folds all remaining elements into rows. For complex fields
 * with at least a 2x2 grid, each non-border cell records local phase winding,
 * phase-seam crossings, ambiguity flags, and an optional singularity core
 * estimate.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Bit mask describing which edges of a cell cross a phase seam.
 */
typedef enum SimFieldTopologySeamMask {
    SIM_FIELD_TOPOLOGY_SEAM_BOTTOM = 1u << 0, /**< Bottom edge exceeds the seam threshold. */
    SIM_FIELD_TOPOLOGY_SEAM_RIGHT = 1u << 1,  /**< Right edge exceeds the seam threshold. */
    SIM_FIELD_TOPOLOGY_SEAM_TOP = 1u << 2,    /**< Top edge exceeds the seam threshold. */
    SIM_FIELD_TOPOLOGY_SEAM_LEFT = 1u << 3    /**< Left edge exceeds the seam threshold. */
} SimFieldTopologySeamMask;

/**
 * @brief Flags describing confidence and validity for a topology cell.
 */
typedef enum SimFieldTopologyCellFlag {
    SIM_FIELD_TOPOLOGY_CELL_VALID = 1u << 0, /**< Cell was evaluated from input samples. */
    SIM_FIELD_TOPOLOGY_CELL_AMBIGUOUS =
        1u << 1, /**< Cell is near a threshold, low magnitude, or numerically uncertain. */
    SIM_FIELD_TOPOLOGY_CELL_CORE_OFFSET_VALID =
        1u << 2, /**< Core offset fields contain a localized singularity estimate. */
    SIM_FIELD_TOPOLOGY_CELL_NONFINITE = 1u << 3 /**< One or more source samples were non-finite. */
} SimFieldTopologyCellFlag;

/**
 * @brief Per-cell phase-topology measurement.
 */
typedef struct SimFieldTopologyCell {
    int8_t charge;       /**< Quantized winding charge: -1, 0, or +1. */
    uint8_t seam_mask;   /**< Bitwise OR of SimFieldTopologySeamMask values. */
    uint8_t flags;       /**< Bitwise OR of SimFieldTopologyCellFlag values. */
    float core_offset_x; /**< Estimated core X offset from cell center in [-0.5, 0.5]. */
    float core_offset_y; /**< Estimated core Y offset from cell center in [-0.5, 0.5]. */
    float confidence;    /**< Confidence score in [0, 1] after charge and contrast checks. */
} SimFieldTopologyCell;

/**
 * @brief Aggregate counts reported by a topology extraction pass.
 */
typedef struct SimFieldTopologySummary {
    bool valid;                    /**< True when complex topology was actually evaluated. */
    size_t positive_singularities; /**< Number of cells with charge +1. */
    size_t negative_singularities; /**< Number of cells with charge -1. */
    size_t seam_edge_count;        /**< Total number of seam-marked cell edges. */
    size_t ambiguous_cell_count;   /**< Number of cells carrying the ambiguous flag. */
} SimFieldTopologySummary;

/**
 * @brief Tunable thresholds used by phase-topology extraction.
 */
typedef struct SimFieldTopologyConfig {
    float seam_phase_threshold; /**< Raw phase jump magnitude that marks a seam edge. */
    float seam_ambiguity_band;  /**< Distance around the seam threshold treated as ambiguous. */
    float charge_ambiguity_threshold; /**< Maximum winding error before marking ambiguity. */
    float magnitude_epsilon;          /**< Magnitude floor for low-signal and divide guards. */
    bool enable_core_localization;    /**< Estimate singularity core offsets when possible. */
} SimFieldTopologyConfig;

/**
 * @brief Fill a topology configuration with default thresholds.
 *
 * @param[out] out_config Receives defaults when non-NULL.
 */
void sim_field_topology_config_default(SimFieldTopologyConfig *out_config);

/**
 * @brief Resolve the 2D topology dimensions for a field.
 *
 * @param field Field whose layout is inspected.
 * @param[out] out_width Receives the fastest-varying dimension when non-NULL.
 * @param[out] out_height Receives the flattened row count when non-NULL.
 * @return true when the field layout has a nonzero element count and width.
 */
bool sim_field_topology_dimensions(const struct SimField *field, size_t *out_width,
                                   size_t *out_height);

/**
 * @brief Extract phase winding and seam data from a field.
 *
 * @param field Field to inspect; only complex-double fields produce valid topology.
 * @param[out] out_cells Optional cell buffer with one entry per resolved field element.
 * @param cell_capacity Number of entries available in @p out_cells.
 * @param[out] out_summary Optional aggregate topology counts.
 * @param config Optional extraction thresholds; defaults are used when NULL.
 * @param[out] out_width Optional resolved width.
 * @param[out] out_height Optional resolved height.
 * @return true when the field could be inspected; check SimFieldTopologySummary::valid to
 * distinguish an unsupported/non-complex field from a topology-bearing result.
 *
 * @details The last row and last column do not form full 2x2 cells and are
 * written as zeroed cells when @p out_cells is provided.
 */
bool sim_field_topology_extract(const struct SimField *field, SimFieldTopologyCell *out_cells,
                                size_t cell_capacity, SimFieldTopologySummary *out_summary,
                                const SimFieldTopologyConfig *config, size_t *out_width,
                                size_t *out_height);

/**
 * @brief Pack one topology cell into an RGBA8 texel.
 *
 * @param cell Cell to pack; NULL is treated as a zero-valued cell.
 * @param[out] out_texel Receives charge, seam/flag bits, and core-offset channels.
 *
 * @details R encodes charge mapped from [-1, +1] into [0, 255]. G stores flags
 * in the high nibble and seam bits in the low nibble. B and A store core X/Y
 * offsets biased into [0, 255], or 128 when no core offset is valid.
 */
void sim_field_topology_cell_pack_rgba8(const SimFieldTopologyCell *cell, uint8_t out_texel[4]);

#ifdef __cplusplus
}
#endif
