/**
 * @file neural_tensor_map.h
 * @brief Canonical SimField <-> tensor mapping helpers for neural operators.
 *
 * Mapping helpers expose fields to neural backends as batch/channel/spatial
 * tensor views. The mapping may be zero-copy or may allocate a scratch tensor
 * when layout, channel placement, complex encoding, or affine conversion cannot
 * be represented directly.
 */
#ifndef OAKFIELD_NEURAL_TENSOR_MAP_H
#define OAKFIELD_NEURAL_TENSOR_MAP_H

#include "field.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum tensor rank supported by the mapping helper (N + C + up to 3 spatial). */
#define SIM_NEURAL_TENSOR_MAX_RANK 5U

/** Automatic channel-axis selection value for SimNeuralTensorMapConfig::channel_axis. */
#define SIM_NEURAL_TENSOR_CHANNEL_AXIS_AUTO 255U

/**
 * @brief Canonical value type exposed to tensor backends.
 */
typedef enum SimNeuralTensorValueType {
    SIM_NEURAL_TENSOR_VALUE_REAL_F64 = 0,            /**< Real-valued double tensor. */
    SIM_NEURAL_TENSOR_VALUE_COMPLEX_F64_INTERLEAVED, /**< Interleaved complex<double>. */
} SimNeuralTensorValueType;

/**
 * @brief Complex-field encoding mode for tensor exposure.
 */
typedef enum SimNeuralTensorComplexMode {
    SIM_NEURAL_TENSOR_COMPLEX_INTERLEAVED = 0, /**< Keep complex values interleaved (complex128). */
    SIM_NEURAL_TENSOR_COMPLEX_SPLIT_CHANNELS,  /**< Expand complex values as doubled real channels.
                                                */
} SimNeuralTensorComplexMode;

/**
 * @brief Optional affine hook applied during mapping/unmapping.
 *
 * When enabled, values are transformed as: y = x * scale + bias.
 */
typedef struct SimNeuralTensorAffineHook {
    bool enabled; /**< True when affine conversion should be applied. */
    double scale; /**< Multiplicative scale applied to each value. */
    double bias;  /**< Additive bias applied after scaling. */
} SimNeuralTensorAffineHook;

/**
 * @brief Mapping policy from SimField layout to canonical tensor layout.
 *
 * Current implementation supports batch_size == 1.
 */
typedef struct SimNeuralTensorMapConfig {
    uint8_t channel_axis; /**< Field channel axis or @ref SIM_NEURAL_TENSOR_CHANNEL_AXIS_AUTO. */
    bool channels_last; /**< Tensor layout convention: NHWC-style when true, NCHW-style when false.
                         */
    size_t batch_size;  /**< Canonical batch dimension; currently only 1 is supported. */
    SimNeuralTensorComplexMode complex_mode; /**< Complex encoding policy for complex fields. */
    SimNeuralTensorAffineHook affine;        /**< Optional affine transform hook. */
} SimNeuralTensorMapConfig;

/**
 * @brief Canonical tensor view generated from a field.
 *
 * Strides are expressed in logical tensor values (not bytes).
 */
typedef struct SimNeuralTensorView {
    void *data;                                 /**< Tensor data pointer. */
    SimNeuralTensorValueType value_type;        /**< Value representation. */
    size_t value_size;                          /**< Size of one tensor value in bytes. */
    size_t rank;                                /**< Tensor rank. */
    size_t shape[SIM_NEURAL_TENSOR_MAX_RANK];   /**< Extent per tensor axis. */
    size_t strides[SIM_NEURAL_TENSOR_MAX_RANK]; /**< Stride per tensor axis in tensor values. */
    uint8_t batch_axis;                         /**< Always 0 in current canonical mapping. */
    uint8_t channel_axis;                       /**< Channel axis in tensor coordinates. */
    uint8_t spatial_rank;                       /**< Number of spatial dimensions in tensor view. */
} SimNeuralTensorView;

/**
 * @brief Explicit copy/transform cost metrics emitted by mapping helpers.
 */
typedef struct SimNeuralTensorMapMetrics {
    bool zero_copy;                 /**< True when tensor view aliases field memory. */
    bool used_copy_fallback;        /**< True when an intermediate tensor buffer was required. */
    size_t bytes_copied_from_field; /**< Bytes read+materialized from field into tensor buffer. */
    size_t bytes_copied_to_field; /**< Bytes written back from tensor buffer into field storage. */
    size_t affine_element_ops;    /**< Number of element-wise affine operations applied. */
} SimNeuralTensorMapMetrics;

/**
 * @brief Opaque-ish mapping handle used by input/output mapping paths.
 *
 * Call @ref sim_neural_tensor_mapping_release after use.
 * For output mappings created by @ref sim_neural_tensor_map_output, call
 * @ref sim_neural_tensor_unmap_output before release when fallback copy was used.
 */
typedef struct SimNeuralTensorMapping {
    SimNeuralTensorView view;          /**< Tensor view exposed to the backend. */
    SimNeuralTensorMapMetrics metrics; /**< Copy/transform costs observed during mapping. */
    SimNeuralTensorMapConfig config;   /**< Normalized mapping policy. */

    /* Internal bookkeeping fields (public for C ABI simplicity). */
    const SimField *field_const; /**< Borrowed source field for input mappings. */
    SimField *field_mut;         /**< Borrowed mutable field for output mappings. */
    void *scratch;              /**< Owned fallback tensor buffer, released by mapping_release(). */
    size_t scratch_bytes;       /**< Allocated byte size of @ref scratch. */
    uint8_t field_rank;         /**< Rank of the mapped field layout. */
    uint8_t field_channel_axis; /**< Resolved channel axis in field coordinates. */
    uint8_t field_spatial_rank; /**< Number of spatial axes in the mapped field. */
    uint8_t field_spatial_axes[3]; /**< Field axes corresponding to tensor spatial dimensions. */
    bool has_field_channel_axis;   /**< True when the field exposes an explicit channel axis. */
    bool output_mode;              /**< True when mapping is intended for write-back. */
    bool active;                   /**< True while the mapping owns/references live resources. */
} SimNeuralTensorMapping;

/**
 * @brief Return default mapping config (batch=1, channels_last=true, auto channel axis).
 *
 * @return Mapping config suitable for common NHWC-style inference inputs.
 */
SimNeuralTensorMapConfig sim_neural_tensor_map_config_defaults(void);

/**
 * @brief Map a field as an input tensor view.
 *
 * Uses zero-copy aliasing when policy/layout permits; otherwise allocates an
 * intermediate tensor buffer and copies data into canonical order.
 *
 * @param field Source field to expose; caller retains ownership.
 * @param config Optional mapping policy, or NULL for defaults.
 * @param[out] out_mapping Receives the tensor view and release bookkeeping.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_SUPPORTED,
 *         or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_neural_tensor_map_input(const SimField *field, const SimNeuralTensorMapConfig *config,
                                      SimNeuralTensorMapping *out_mapping);

/**
 * @brief Map a field as an output tensor view.
 *
 * For fallback-copy mappings, backends should write outputs into the returned
 * tensor view then call @ref sim_neural_tensor_unmap_output to commit.
 *
 * @param field Destination field to expose; caller retains ownership.
 * @param config Optional mapping policy, or NULL for defaults.
 * @param[out] out_mapping Receives the tensor view and release bookkeeping.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_NOT_SUPPORTED,
 *         or #SIM_RESULT_OUT_OF_MEMORY.
 */
SimResult sim_neural_tensor_map_output(SimField *field, const SimNeuralTensorMapConfig *config,
                                       SimNeuralTensorMapping *out_mapping);

/**
 * @brief Commit output tensor results back into the mapped field.
 *
 * Required only for fallback-copy output mappings. Safe to call on zero-copy
 * mappings (no-op).
 *
 * @param mapping Active mapping returned by sim_neural_tensor_map_output().
 * @return #SIM_RESULT_OK or #SIM_RESULT_INVALID_ARGUMENT.
 */
SimResult sim_neural_tensor_unmap_output(SimNeuralTensorMapping *mapping);

/**
 * @brief Release resources held by a tensor mapping handle.
 *
 * @param mapping Mapping handle to release; NULL is ignored.
 */
void sim_neural_tensor_mapping_release(SimNeuralTensorMapping *mapping);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_NEURAL_TENSOR_MAP_H */
