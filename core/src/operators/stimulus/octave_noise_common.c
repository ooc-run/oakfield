#include "octave_noise_common.h"

#include "sim_accel.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_OCTAVE_NOISE_RIDGED_PROFILE_MEAN (1.5 - 4.0 / M_PI)
#define STIM_OCTAVE_NOISE_TURBULENCE_ABS_MEAN (2.0 / M_PI)
#define STIM_OCTAVE_NOISE_VDSP_MIN_LEN 64U

typedef struct SimStimulusOctaveNoisePcg32 {
    uint64_t state;
    uint64_t inc;
} SimStimulusOctaveNoisePcg32;

static uint32_t sim_stimulus_octave_noise_pcg32_random(SimStimulusOctaveNoisePcg32* rng) {
    uint64_t old        = rng->state;
    rng->state          = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t) (old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void sim_stimulus_octave_noise_pcg32_seed(SimStimulusOctaveNoisePcg32* rng,
                                                 uint64_t                     initstate,
                                                 uint64_t                     initseq) {
    rng->state = 0U;
    rng->inc   = (initseq << 1u) | 1u;
    (void) sim_stimulus_octave_noise_pcg32_random(rng);
    rng->state += initstate;
    (void) sim_stimulus_octave_noise_pcg32_random(rng);
}

static double sim_stimulus_octave_noise_uniform(SimStimulusOctaveNoisePcg32* rng) {
    return ldexp(sim_stimulus_octave_noise_pcg32_random(rng), -32);
}

static double sim_stimulus_octave_noise_ridged_profile(double base_value) {
    double ridge = 1.0 - fabs(base_value);
    return ridge * ridge - STIM_OCTAVE_NOISE_RIDGED_PROFILE_MEAN;
}

static void sim_stimulus_octave_noise_eval_scalar(SimStimulusOctaveNoiseKind kind,
                                                  const double*              phases,
                                                  double                     u,
                                                  double                     hurst,
                                                  double                     lacunarity,
                                                  unsigned int               octaves,
                                                  bool                       need_imag,
                                                  double*                    out_re,
                                                  double*                    out_im) {
    double amp_decay = pow(lacunarity, -hurst);
    double amp       = 1.0;
    double freq      = 1.0;
    double sum_re    = 0.0;
    double sum_im    = 0.0;
    double aux_re    = 1.0;
    double aux_im    = 1.0;

    if (phases == NULL || octaves == 0U) {
        if (out_re != NULL) {
            *out_re = 0.0;
        }
        if (out_im != NULL) {
            *out_im = 0.0;
        }
        return;
    }

    if (kind == SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL) {
        sum_re = 1.0;
        if (need_imag) {
            sum_im = 1.0;
        }
    }

    for (unsigned int o = 0U; o < octaves; ++o) {
        double theta    = 2.0 * M_PI * freq * u + phases[o];
        double basis_re = cos(theta);
        double basis_im = need_imag ? sin(theta) : 0.0;

        switch (kind) {
            case SIM_STIMULUS_OCTAVE_NOISE_FBM:
                sum_re += amp * basis_re;
                if (need_imag) {
                    sum_im += amp * basis_im;
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_RIDGED:
                sum_re += amp * sim_stimulus_octave_noise_ridged_profile(basis_re);
                if (need_imag) {
                    sum_im += amp * sim_stimulus_octave_noise_ridged_profile(basis_im);
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_HYBRID_FBM:
                sum_re += amp * aux_re * basis_re;
                aux_re *= 0.5 * (1.0 + fabs(basis_re));
                if (need_imag) {
                    sum_im += amp * aux_im * basis_im;
                    aux_im *= 0.5 * (1.0 + fabs(basis_im));
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL:
                sum_re *= 1.0 + 0.5 * amp * basis_re;
                if (need_imag) {
                    sum_im *= 1.0 + 0.5 * amp * basis_im;
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_TURBULENCE:
                sum_re += amp * (fabs(basis_re) - STIM_OCTAVE_NOISE_TURBULENCE_ABS_MEAN);
                if (need_imag) {
                    sum_im += amp * (fabs(basis_im) - STIM_OCTAVE_NOISE_TURBULENCE_ABS_MEAN);
                }
                break;
        }

        freq *= lacunarity;
        amp *= amp_decay;
    }

    if (kind == SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL) {
        sum_re -= 1.0;
        if (need_imag) {
            sum_im -= 1.0;
        }
    }

    if (out_re != NULL) {
        *out_re = sum_re;
    }
    if (out_im != NULL) {
        *out_im = need_imag ? sum_im : 0.0;
    }
}

SimResult sim_stimulus_octave_noise_ensure_phases(double**      phases,
                                                  unsigned int* allocated_octaves,
                                                  unsigned int  desired_octaves,
                                                  uint64_t      seed) {
    if (phases == NULL || allocated_octaves == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (desired_octaves == 0U) {
        return SIM_RESULT_OK;
    }
    if (*allocated_octaves == desired_octaves && *phases != NULL) {
        return SIM_RESULT_OK;
    }

    double* resized = (double*) realloc(*phases, (size_t) desired_octaves * sizeof(double));
    if (resized == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    *phases            = resized;
    *allocated_octaves = desired_octaves;

    SimStimulusOctaveNoisePcg32 rng;
    sim_stimulus_octave_noise_pcg32_seed(&rng, seed, seed ^ 0x51ED2705B4C3A59DULL);
    for (unsigned int o = 0U; o < desired_octaves; ++o) {
        (*phases)[o] = 2.0 * M_PI * sim_stimulus_octave_noise_uniform(&rng);
    }
    return SIM_RESULT_OK;
}

void sim_stimulus_octave_noise_eval_base(SimStimulusOctaveNoiseKind kind,
                                         const double*              phases,
                                         double                     u,
                                         double                     hurst,
                                         double                     lacunarity,
                                         unsigned int               octaves,
                                         double*                    out_re,
                                         double*                    out_im) {
    sim_stimulus_octave_noise_eval_scalar(
        kind, phases, u, hurst, lacunarity, octaves, out_im != NULL, out_re, out_im);
}

void sim_stimulus_octave_noise_vdsp_release(SimStimulusOctaveNoiseVdspBuffers* buffers) {
    if (buffers == NULL) {
        return;
    }
    free(buffers->block);
    buffers->theta    = NULL;
    buffers->basis_re = NULL;
    buffers->basis_im = NULL;
    buffers->accum_re = NULL;
    buffers->accum_im = NULL;
    buffers->aux_re   = NULL;
    buffers->aux_im   = NULL;
    buffers->block    = NULL;
    buffers->capacity = 0U;
}

#if defined(SIM_HAVE_VDSP)
static bool sim_stimulus_octave_noise_vdsp_reserve(SimStimulusOctaveNoiseVdspBuffers* buffers,
                                                   size_t                             width) {
    if (buffers == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (buffers->block != NULL && buffers->capacity >= width) {
        return true;
    }

    double* block = (double*) realloc(buffers->block, width * 7U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    buffers->block    = block;
    buffers->capacity = width;
    buffers->theta    = block;
    buffers->basis_re = block + width;
    buffers->basis_im = block + width * 2U;
    buffers->accum_re = block + width * 3U;
    buffers->accum_im = block + width * 4U;
    buffers->aux_re   = block + width * 5U;
    buffers->aux_im   = block + width * 6U;
    return true;
}

static bool sim_stimulus_octave_noise_linear_projection(const SimStimulusCoordConfig* coord,
                                                        double*                       out_proj_x,
                                                        double*                       out_proj_y) {
    if (coord == NULL || out_proj_x == NULL || out_proj_y == NULL) {
        return false;
    }

    switch (coord->mode) {
        case SIM_STIMULUS_COORD_AXIS:
            *out_proj_x = (coord->axis == SIM_STIMULUS_AXIS_Y) ? 0.0 : 1.0;
            *out_proj_y = (coord->axis == SIM_STIMULUS_AXIS_Y) ? 1.0 : 0.0;
            return true;
        case SIM_STIMULUS_COORD_ANGLE:
            *out_proj_x = cos(coord->angle);
            *out_proj_y = sin(coord->angle);
            return true;
        default:
            return false;
    }
}

static bool sim_stimulus_octave_noise_accumulate_row(SimStimulusOctaveNoiseKind         kind,
                                                     SimStimulusOctaveNoiseVdspBuffers* buffers,
                                                     size_t                             width,
                                                     double                             u0,
                                                     double                             du,
                                                     const double*                      phases,
                                                     double                             hurst,
                                                     double                             lacunarity,
                                                     unsigned int                       octaves,
                                                     bool                               need_imag) {
    if (buffers == NULL || phases == NULL || width == 0U) {
        return false;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    double            amp_decay  = pow(lacunarity, -hurst);
    double            amp        = 1.0;
    double            freq       = 1.0;

    for (size_t i = 0U; i < width; ++i) {
        if (kind == SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL) {
            buffers->accum_re[i] = 1.0;
            if (need_imag) {
                buffers->accum_im[i] = 1.0;
            }
        } else {
            buffers->accum_re[i] = 0.0;
            if (need_imag) {
                buffers->accum_im[i] = 0.0;
            }
        }
        if (kind == SIM_STIMULUS_OCTAVE_NOISE_HYBRID_FBM) {
            buffers->aux_re[i] = 1.0;
            if (need_imag) {
                buffers->aux_im[i] = 1.0;
            }
        }
    }

    for (unsigned int o = 0U; o < octaves; ++o) {
        double start = 2.0 * M_PI * freq * u0 + phases[o];
        double step  = 2.0 * M_PI * freq * du;
        if (!isfinite(start) || !isfinite(step) || !isfinite(amp)) {
            return false;
        }

        vDSP_vrampD(&start, &step, buffers->theta, 1, len);
        if (need_imag) {
            vvsincos(buffers->basis_im, buffers->basis_re, buffers->theta, &vforce_len);
        } else {
            vvcos(buffers->basis_re, buffers->theta, &vforce_len);
        }

        switch (kind) {
            case SIM_STIMULUS_OCTAVE_NOISE_FBM:
                vDSP_vsmaD(buffers->basis_re,
                           1,
                           &amp,
                           buffers->accum_re,
                           1,
                           buffers->accum_re,
                           1,
                           len);
                if (need_imag) {
                    vDSP_vsmaD(buffers->basis_im,
                               1,
                               &amp,
                               buffers->accum_im,
                               1,
                               buffers->accum_im,
                               1,
                               len);
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_RIDGED:
                for (size_t i = 0U; i < width; ++i) {
                    buffers->basis_re[i] =
                        sim_stimulus_octave_noise_ridged_profile(buffers->basis_re[i]);
                    if (need_imag) {
                        buffers->basis_im[i] =
                            sim_stimulus_octave_noise_ridged_profile(buffers->basis_im[i]);
                    }
                }
                vDSP_vsmaD(buffers->basis_re,
                           1,
                           &amp,
                           buffers->accum_re,
                           1,
                           buffers->accum_re,
                           1,
                           len);
                if (need_imag) {
                    vDSP_vsmaD(buffers->basis_im,
                               1,
                               &amp,
                               buffers->accum_im,
                               1,
                               buffers->accum_im,
                               1,
                               len);
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_HYBRID_FBM:
                for (size_t i = 0U; i < width; ++i) {
                    double basis_re = buffers->basis_re[i];
                    buffers->accum_re[i] += amp * buffers->aux_re[i] * basis_re;
                    buffers->aux_re[i] *= 0.5 * (1.0 + fabs(basis_re));
                    if (need_imag) {
                        double basis_im = buffers->basis_im[i];
                        buffers->accum_im[i] += amp * buffers->aux_im[i] * basis_im;
                        buffers->aux_im[i] *= 0.5 * (1.0 + fabs(basis_im));
                    }
                }
                break;
            case SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL: {
                double half_amp = 0.5 * amp;
                for (size_t i = 0U; i < width; ++i) {
                    buffers->accum_re[i] *= 1.0 + half_amp * buffers->basis_re[i];
                    if (need_imag) {
                        buffers->accum_im[i] *= 1.0 + half_amp * buffers->basis_im[i];
                    }
                }
                break;
            }
            case SIM_STIMULUS_OCTAVE_NOISE_TURBULENCE:
                for (size_t i = 0U; i < width; ++i) {
                    buffers->basis_re[i] =
                        fabs(buffers->basis_re[i]) - STIM_OCTAVE_NOISE_TURBULENCE_ABS_MEAN;
                    if (need_imag) {
                        buffers->basis_im[i] =
                            fabs(buffers->basis_im[i]) - STIM_OCTAVE_NOISE_TURBULENCE_ABS_MEAN;
                    }
                }
                vDSP_vsmaD(buffers->basis_re,
                           1,
                           &amp,
                           buffers->accum_re,
                           1,
                           buffers->accum_re,
                           1,
                           len);
                if (need_imag) {
                    vDSP_vsmaD(buffers->basis_im,
                               1,
                               &amp,
                               buffers->accum_im,
                               1,
                               buffers->accum_im,
                               1,
                               len);
                }
                break;
        }

        freq *= lacunarity;
        amp *= amp_decay;
    }

    if (kind == SIM_STIMULUS_OCTAVE_NOISE_MULTIFRACTAL) {
        double negative_one = -1.0;
        vDSP_vsaddD(buffers->accum_re, 1, &negative_one, buffers->accum_re, 1, len);
        if (need_imag) {
            vDSP_vsaddD(buffers->accum_im, 1, &negative_one, buffers->accum_im, 1, len);
        }
    }

    return true;
}
#endif

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
                                             size_t                             count) {
#if defined(SIM_HAVE_VDSP)
    if (buffers == NULL || coord == NULL || phases == NULL || field == NULL || count == 0U) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_OCTAVE_NOISE_VDSP_MIN_LEN ||
        width > count || width > (size_t) INT_MAX) {
        return false;
    }
    if (width * height != count) {
        return false;
    }

    double row_scale = amplitude * scale;
    if (!isfinite(row_scale)) {
        return false;
    }
    if (row_scale == 0.0) {
        return true;
    }
    if (!sim_stimulus_octave_noise_vdsp_reserve(buffers, width)) {
        return false;
    }

    bool   is_complex       = (dst_complex != NULL);
    bool   separable        = (coord->mode == SIM_STIMULUS_COORD_SEPARABLE);
    bool   need_imag        = is_complex ||
                       (separable && coord->combine == SIM_STIMULUS_SEPARABLE_MULTIPLY);
    bool   linear_mode      = false;
    double proj_x           = 0.0;
    double proj_y           = 0.0;
    double x0               = coord->origin_x - coord->velocity_x * t;
    double y0               = coord->origin_y - coord->velocity_y * t;
    double dx               = coord->spacing_x;
    double dy               = coord->spacing_y;

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy)) {
        return false;
    }

    if (separable) {
        linear_mode = true;
    } else {
        linear_mode = sim_stimulus_octave_noise_linear_projection(coord, &proj_x, &proj_y);
    }
    if (!linear_mode) {
        return false;
    }

    if (separable) {
        if (!sim_stimulus_octave_noise_accumulate_row(
                kind, buffers, width, x0, dx, phases, hurst, lacunarity, octaves, need_imag)) {
            return false;
        }
        memcpy(buffers->basis_re, buffers->accum_re, width * sizeof(double));
        if (need_imag) {
            memcpy(buffers->basis_im, buffers->accum_im, width * sizeof(double));
        }
    }

    for (size_t row = 0U; row < height; ++row) {
        if (separable) {
            double fy_re    = 0.0;
            double fy_im    = 0.0;
            double sample_y = y0 + (double) row * dy;
            sim_stimulus_octave_noise_eval_scalar(
                kind, phases, sample_y, hurst, lacunarity, octaves, need_imag, &fy_re, &fy_im);
            if (!isfinite(fy_re) || (need_imag && !isfinite(fy_im))) {
                return false;
            }

            if (coord->combine == SIM_STIMULUS_SEPARABLE_ADD) {
                memcpy(buffers->accum_re, buffers->basis_re, width * sizeof(double));
                sim_accel_add_scalar_real(buffers->accum_re, width, fy_re);
                if (need_imag) {
                    memcpy(buffers->accum_im, buffers->basis_im, width * sizeof(double));
                    sim_accel_add_scalar_real(buffers->accum_im, width, fy_im);
                }
            } else {
                for (size_t i = 0U; i < width; ++i) {
                    double fx_re = buffers->basis_re[i];
                    double fx_im = need_imag ? buffers->basis_im[i] : 0.0;
                    buffers->accum_re[i] = fx_re * fy_re - fx_im * fy_im;
                    if (need_imag) {
                        buffers->accum_im[i] = fx_re * fy_im + fx_im * fy_re;
                    }
                }
            }
        } else {
            double u0 = proj_x * x0 + proj_y * (y0 + (double) row * dy);
            double du = proj_x * dx;
            if (!isfinite(u0) || !isfinite(du) ||
                !sim_stimulus_octave_noise_accumulate_row(
                    kind, buffers, width, u0, du, phases, hurst, lacunarity, octaves, need_imag)) {
                return false;
            }
        }

        size_t offset = row * width;
        if (!is_complex) {
            if (sim_accel_scan_real_finite_maxabs(buffers->accum_re, width, NULL)) {
                sim_accel_copy_scale_real(
                    buffers->accum_re, dst_real + offset, width, row_scale, true);
            } else {
                for (size_t i = 0U; i < width; ++i) {
                    double value = buffers->accum_re[i];
                    if (isfinite(value)) {
                        dst_real[offset + i] += row_scale * value;
                    }
                }
            }
            continue;
        }

        if (sim_accel_scan_real_finite_maxabs(buffers->accum_re, width, NULL) &&
            (!need_imag || sim_accel_scan_real_finite_maxabs(buffers->accum_im, width, NULL))) {
            sim_accel_accumulate_real_to_complex(
                buffers->accum_re, dst_complex + offset, width, row_scale, 0.0);
            if (need_imag) {
                sim_accel_accumulate_real_to_complex(
                    buffers->accum_im, dst_complex + offset, width, 0.0, row_scale);
            }
        } else {
            for (size_t i = 0U; i < width; ++i) {
                double value_re = buffers->accum_re[i];
                double value_im = need_imag ? buffers->accum_im[i] : 0.0;
                if (isfinite(value_re) && isfinite(value_im)) {
                    dst_complex[offset + i].re += row_scale * value_re;
                    dst_complex[offset + i].im += row_scale * value_im;
                }
            }
        }
    }

    return true;
#else
    (void) kind;
    (void) buffers;
    (void) coord;
    (void) phases;
    (void) amplitude;
    (void) hurst;
    (void) lacunarity;
    (void) octaves;
    (void) scale;
    (void) t;
    (void) field;
    (void) dst_real;
    (void) dst_complex;
    (void) count;
    return false;
#endif
}
