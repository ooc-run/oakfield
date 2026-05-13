/**
 * @file octave_noise_common.h
 * @brief Shared octave-noise basis helpers for fractal stimulus operators.
 *
 * The helpers centralize seeded phase generation, scalar basis evaluation, and
 * optional row-vectorized evaluation for fBm, ridged noise, hybrid fBm,
 * multifractal, and turbulence stimuli.
 */
#ifndef OAKFIELD_STIMULUS_OCTAVE_NOISE_COMMON_H
#define OAKFIELD_STIMULUS_OCTAVE_NOISE_COMMON_H

#include "oakfield/field.h"
#include "oakfield/operator_split.h"
#include "oakfield/operators/stimulus/coords.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SimStimulusOctaveNoiseKind {
    SIM_STIMULUS_OCTAVE_NOISE_FBM = 0,
    SIM_STIMULUS_OCTAVE_NOISE_RIDGED,
    SIM_STIMULUS_OCTAVE_NOISE_HYBRID_FBM,
    SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL,
    SIM_STIMULUS_OCTAVE_NOISE_TURBULENCE
} SimStimulusOctaveNoiseKind;

typedef struct SimStimulusOctaveNoiseVdspBuffers {
    double* theta;
    double* basis_re;
    double* basis_im;
    double* accum_re;
    double* accum_im;
    double* aux_re;
    double* aux_im;
    double* block;
    size_t  capacity;
} SimStimulusOctaveNoiseVdspBuffers;

/**
 * @brief Allocate and seed the per-octave random phase table.
 *
 * The table is resized to @p desired_octaves entries and filled with phases in
 * [0, 2*pi). If the existing table already has the requested length, it is reused.
 *
 * @param[in,out] phases Pointer to the owned phase array; may point to NULL.
 * @param[in,out] allocated_octaves Current and resulting number of allocated phases.
 * @param desired_octaves Number of phases required; zero is a no-op.
 * @param seed Deterministic seed used to regenerate all phases after resize.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         bookkeeping pointers, or #SIM_RESULT_OUT_OF_MEMORY on resize failure.
 */
SimResult sim_stimulus_octave_noise_ensure_phases(double**      phases,
                                                  unsigned int* allocated_octaves,
                                                  unsigned int  desired_octaves,
                                                  uint64_t      seed);

/**
 * @brief Evaluate one octave-noise basis at a scalar coordinate.
 *
 * The selected @p kind controls the accumulation law. A NULL @p out_im requests
 * real-only evaluation; otherwise the corresponding sine/imaginary basis is
 * computed as well. NULL phases or zero octaves yield zero outputs.
 *
 * @param kind Octave-noise family to evaluate.
 * @param phases Phase table with at least @p octaves entries.
 * @param u Scalar coordinate in stimulus units.
 * @param hurst Hurst exponent controlling amplitude decay.
 * @param lacunarity Frequency multiplier between octaves.
 * @param octaves Number of octaves to accumulate.
 * @param[out] out_re Optional destination for the real basis sum.
 * @param[out] out_im Optional destination for the imaginary basis sum.
 */
void sim_stimulus_octave_noise_eval_base(SimStimulusOctaveNoiseKind kind,
                                         const double*              phases,
                                         double                     u,
                                         double                     hurst,
                                         double                     lacunarity,
                                         unsigned int               octaves,
                                         double*                    out_re,
                                         double*                    out_im);

/**
 * @brief Release vectorized octave-noise scratch buffers.
 *
 * Frees the single backing block and clears all derived lane pointers. Safe to
 * call on a zeroed or NULL buffer struct.
 *
 * @param buffers Scratch buffer set to release.
 */
void sim_stimulus_octave_noise_vdsp_release(SimStimulusOctaveNoiseVdspBuffers* buffers);

/**
 * @brief Try to evaluate octave noise with the row-vectorized vDSP path.
 *
 * The fast path is used only for contiguous row-major fields and coordinate modes
 * that reduce to linear row projections or separable rows. On unsupported inputs,
 * allocation failure, non-finite intermediate values, or builds without vDSP, the
 * function returns false so the caller can run its scalar fallback.
 *
 * @param kind Octave-noise family to evaluate.
 * @param buffers Reusable vDSP scratch buffers.
 * @param coord Normalized coordinate mapping.
 * @param phases Phase table with at least @p octaves entries.
 * @param amplitude Stimulus amplitude multiplier.
 * @param hurst Hurst exponent controlling amplitude decay.
 * @param lacunarity Frequency multiplier between octaves.
 * @param octaves Number of octaves to accumulate.
 * @param scale Additional caller-provided scale, often dt-dependent.
 * @param t Evaluation time in seconds.
 * @param field Field whose layout and storage are being written.
 * @param[out] dst_real Real destination buffer when writing a real field.
 * @param[out] dst_complex Complex destination buffer when writing a complex field.
 * @param count Number of destination elements.
 * @return true if the vectorized path handled all writes; false if the caller
 *         should use the scalar fallback.
 */
bool sim_stimulus_octave_noise_try_vdsp_rows(SimStimulusOctaveNoiseKind         kind,
                                             SimStimulusOctaveNoiseVdspBuffers* buffers,
                                             const SimStimulusCoordConfig*      coord,
                                             const double*                      phases,
                                             double                             amplitude,
                                             double                             hurst,
                                             double                             lacunarity,
                                             unsigned int                       octaves,
                                             double                             scale,
                                             double                             t,
                                             const SimField*                    field,
                                             double*                            dst_real,
                                             SimComplexDouble*                  dst_complex,
                                             size_t                             count);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_OCTAVE_NOISE_COMMON_H */
