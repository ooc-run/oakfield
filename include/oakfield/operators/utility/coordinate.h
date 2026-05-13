/**
 * @file coordinate.h
 * @brief Coordinate/index generator operator.
 *
 * Writes element index or coordinate-based values into a field, with optional normalization
 * and scaling. Coordinates reuse the stimulus coordinate configuration for axis/angle/radial
 * mappings.
 */
#ifndef OAKFIELD_COORDINATE_H
#define OAKFIELD_COORDINATE_H

#include "oakfield/operator_split.h"
#include "oakfield/operators/stimulus/coords.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Coordinate source mode.
 *
 * Index mode writes the linear element index. Coordinate mode evaluates the
 * shared stimulus coordinate mapping at each element position.
 */
typedef enum SimCoordinateMode {
    SIM_COORD_MODE_INDEX = 0, /**< Use linear element index (0..N-1). */
    SIM_COORD_MODE_COORD      /**< Use coordinate mapping configured in the coord field. */
} SimCoordinateMode;

/**
 * @brief Normalization mode for coordinate values.
 */
typedef enum SimCoordinateNormalizeMode {
    SIM_COORD_NORMALIZE_NONE = 0, /**< No normalization. */
    SIM_COORD_NORMALIZE_UNIT,     /**< Map to [0, 1]. */
    SIM_COORD_NORMALIZE_CENTERED, /**< Map to [-0.5, 0.5]. */
    SIM_COORD_NORMALIZE_SIGNED    /**< Map to [-1, 1]. */
} SimCoordinateNormalizeMode;

/**
 * @brief Configuration parameters for the coordinate generator operator.
 */
typedef struct SimCoordinateOperatorConfig {
    size_t output_field;                  /**< Field receiving the coordinate values. */
    SimCoordinateMode mode;               /**< Index vs coordinate mapping. */
    SimCoordinateNormalizeMode normalize; /**< Optional normalization mode. */
    SimStimulusCoordConfig coord;         /**< Coordinate mapping configuration. */
    double gain;                          /**< Gain applied after normalization. */
    double bias;                          /**< Bias added after gain. */
    double time_offset;                   /**< Time offset (used by radial velocity). */
    bool accumulate;                      /**< Add into output when true. */
    bool scale_by_dt;                     /**< Scale accumulated writes by substep dt. */
    bool exact_gain_enabled; /**< Interpret @ref exact_gain_raw in the output integer domain. */
    uint64_t exact_gain_raw; /**< Exact gain literal encoded in the output integer domain. */
    bool exact_bias_enabled; /**< Interpret @ref exact_bias_raw in the output integer domain. */
    uint64_t exact_bias_raw; /**< Exact bias literal encoded in the output integer domain. */
} SimCoordinateOperatorConfig;

/**
 * @brief Register a coordinate generator operator with the provided configuration.
 *
 * The implementation copies and normalizes @p config, resolves the default
 * scale-by-dt policy, and validates that the output field is real double,
 * complex double, or an exact-integer domain supported by the exact affine path.
 * Passing NULL selects index mode on field 0 with gain 1 and bias 0.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional coordinate-generator configuration.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context
 *         or missing output field, #SIM_RESULT_TYPE_MISMATCH for unsupported
 *         output domains, #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or a
 *         split-registration error.
 */
SimResult sim_add_coordinate_operator(struct SimContext *context,
                                      const SimCoordinateOperatorConfig *config, size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a coordinate operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_coordinate_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no coordinate state.
 */
SimResult sim_coordinate_config(struct SimContext *context, size_t operator_index,
                                SimCoordinateOperatorConfig *out_config);

/**
 * @brief Update an existing coordinate operator in-place.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes runtime state, exact-integer
 * affine metadata, symbolic state, and the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the coordinate operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context
 *         or missing output field, #SIM_RESULT_NOT_FOUND for a missing operator,
 *         #SIM_RESULT_INVALID_STATE for missing state, or #SIM_RESULT_TYPE_MISMATCH
 *         for unsupported output domains.
 */
SimResult sim_coordinate_update(struct SimContext *context, size_t operator_index,
                                const SimCoordinateOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_COORDINATE_H */
