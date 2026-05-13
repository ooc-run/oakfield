/**
 * @file fft_convert.h
 * @brief Utility operator that converts fields between physical and spectral domains.
 *
 * FFT conversion is modeled as an explicit graph-IR domain transition. Each
 * conversion allocates a dedicated output field; in-place conversion is not
 * supported.
 */
#ifndef OAKFIELD_FFT_CONVERT_H
#define OAKFIELD_FFT_CONVERT_H

#include "oakfield/operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Direction of a physical/spectral FFT conversion.
 */
typedef enum SimFFTConvertDirection {
    SIM_FFT_CONVERT_FORWARD = 0, /**< Physical-domain input to spectral-domain output. */
    SIM_FFT_CONVERT_INVERSE = 1  /**< Spectral-domain input to physical-domain output. */
} SimFFTConvertDirection;

/**
 * @brief Register an FFT conversion operator and allocate its output field.
 *
 * Forward conversion writes a complex spectral field; real scalar inputs produce
 * a complex-real-constrained spectrum. Inverse conversion writes a physical field
 * and returns real scalar output when the input has the complex-real constraint.
 *
 * @param context Simulation context that will own the output field and operator.
 * @param input_field Index of the field to transform.
 * @param direction Forward physical-to-spectral or inverse spectral-to-physical.
 * @param in_place Must be false; true returns #SIM_RESULT_NOT_SUPPORTED.
 * @param[out] out_field_index Optional destination for the newly allocated output field index.
 * @param[out] out_operator_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL context,
 *         missing input, empty input, or allocation-size overflow,
 *         #SIM_RESULT_NOT_SUPPORTED for in-place conversion,
 *         #SIM_RESULT_TYPE_MISMATCH for inverse conversion from real scalar input,
 *         #SIM_RESULT_OUT_OF_MEMORY on allocation failure, or a field/graph/operator
 *         registration error.
 */
SimResult sim_add_fft_convert(struct SimContext *context, size_t input_field,
                              SimFFTConvertDirection direction, bool in_place,
                              size_t *out_field_index, size_t *out_operator_index);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_FFT_CONVERT_H */
