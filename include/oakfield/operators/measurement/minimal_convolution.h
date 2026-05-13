/**
 * @file minimal_convolution.h
 * @brief Minimal convolution operator with small odd-length kernels (1D/2D).
 */
#ifndef OAKFIELD_MINIMAL_CONVOLUTION_H
#define OAKFIELD_MINIMAL_CONVOLUTION_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

enum {
    SIM_MINIMAL_CONVOLUTION_MAX_TAPS = 9,
    SIM_MINIMAL_CONVOLUTION_MAX_TAPS_2D =
        SIM_MINIMAL_CONVOLUTION_MAX_TAPS * SIM_MINIMAL_CONVOLUTION_MAX_TAPS,
};

/**
 * @brief Supported minimal-convolution kernel layouts.
 */
typedef enum SimMinimalConvolutionMode {
    SIM_MINIMAL_CONVOLUTION_MODE_AXIS = 0,  /**< Apply a one-dimensional kernel along one axis. */
    SIM_MINIMAL_CONVOLUTION_MODE_SEPARABLE, /**< Apply separable row/column kernels. */
    SIM_MINIMAL_CONVOLUTION_MODE_KERNEL_2D  /**< Apply a full two-dimensional kernel. */
} SimMinimalConvolutionMode;

/**
 * @brief Axis selection for one-dimensional kernels on two-dimensional fields.
 */
typedef enum SimMinimalConvolutionAxis {
    SIM_MINIMAL_CONVOLUTION_AXIS_X = 0, /**< Convolve along the X axis. */
    SIM_MINIMAL_CONVOLUTION_AXIS_Y      /**< Convolve along the Y axis. */
} SimMinimalConvolutionAxis;

/**
 * @brief Configuration for the minimal convolution operator.
 */
typedef struct SimMinimalConvolutionOperatorConfig {
    size_t input_field;                              /**< Source field index. */
    size_t output_field;                             /**< Destination field index. */
    size_t kernel_length;                            /**< Number of taps (3/5/7/9). */
    double kernel[SIM_MINIMAL_CONVOLUTION_MAX_TAPS]; /**< Convolution coefficients. */
    SimMinimalConvolutionMode mode;                  /**< Convolution mode (axis/separable/2D). */
    SimMinimalConvolutionAxis axis;                  /**< Axis for 2D axis mode. */
    size_t kernel_rows;                              /**< Kernel rows for 2D mode. */
    size_t kernel_cols;                              /**< Kernel cols for 2D mode. */
    double kernel_2d[SIM_MINIMAL_CONVOLUTION_MAX_TAPS_2D]; /**< 2D kernel coefficients. */
    size_t stride;                                         /**< Stride between samples (>=1). */
    bool wrap;                    /**< Use periodic boundary conditions when true, else clamp. */
    SimIRBoundaryPolicy boundary; /**< Boundary handling policy (wrap maps to periodic). */
    bool accumulate;              /**< Add into the destination instead of overwriting. */
    bool scale_by_dt;             /**< When true, scale accumulated writes by substep dt. */
} SimMinimalConvolutionOperatorConfig;

/**
 * @brief Register a minimal convolution operator.
 *
 * The implementation copies and normalizes @p config, validates kernel shape and
 * field compatibility, and registers a split operator over matching real or
 * complex fields.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional convolution configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         kernel validation, field compatibility checks, allocation, or registration.
 */
SimResult sim_add_minimal_convolution_operator(struct SimContext *context,
                                               const SimMinimalConvolutionOperatorConfig *config,
                                               size_t *out_index);

/**
 * @brief Copy the current minimal-convolution configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_minimal_convolution_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no convolution state.
 */
SimResult sim_minimal_convolution_config(struct SimContext *context, size_t operator_index,
                                         SimMinimalConvolutionOperatorConfig *out_config);

/**
 * @brief Replace the configuration of a registered minimal-convolution operator.
 *
 * @p config is required. The replacement is normalized and kernel/field
 * compatibility is checked before storing it.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the minimal-convolution operator to update.
 * @param config Replacement convolution configuration.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         lookup, kernel validation, field compatibility checks, or state validation.
 */
SimResult sim_minimal_convolution_update(struct SimContext *context, size_t operator_index,
                                         const SimMinimalConvolutionOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MINIMAL_CONVOLUTION_H */
