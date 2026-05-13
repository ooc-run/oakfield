/**
 * @file sim_accel.h
 * @brief Inline scalar and optional vDSP acceleration helpers for field kernels.
 *
 * These internal helpers provide vectorized loops, split-complex scratch storage,
 * finite-value scans, affine mixes, weighted sums, and simple in-place transforms
 * used by CPU operators. When SIM_HAVE_VDSP is enabled, selected paths use
 * Apple's vDSP with the same SimComplexDouble cartesian storage contract.
 */
#ifndef OAKFIELD_SIM_ACCEL_H
#define OAKFIELD_SIM_ACCEL_H

#include "oakfield/field.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#if defined(__clang__)
#define SIM_ACCEL_VECTORIZE_LOOP \
    _Pragma("clang loop vectorize(enable)") _Pragma("clang loop interleave(enable)")
#else
#define SIM_ACCEL_VECTORIZE_LOOP
#endif

typedef struct SimAccelSplitDouble {
    double* realp;
    double* imagp;
} SimAccelSplitDouble;

typedef struct SimAccelSplitComplexScratch {
    double*            block;
    size_t             capacity;
    SimAccelSplitDouble a;
    SimAccelSplitDouble b;
} SimAccelSplitComplexScratch;

static inline void sim_accel_split_release(SimAccelSplitComplexScratch* scratch) {
    if (scratch == NULL) {
        return;
    }
    free(scratch->block);
    scratch->block    = NULL;
    scratch->capacity = 0U;
    scratch->a.realp  = NULL;
    scratch->a.imagp  = NULL;
    scratch->b.realp  = NULL;
    scratch->b.imagp  = NULL;
}

static inline bool sim_accel_split_reserve(SimAccelSplitComplexScratch* scratch, size_t count) {
    if (scratch == NULL) {
        return false;
    }
    if (count == 0U) {
        return true;
    }
    if (scratch->capacity >= count && scratch->block != NULL) {
        return true;
    }

    double* block = (double*) realloc(scratch->block, count * 4U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    scratch->block    = block;
    scratch->capacity = count;
    scratch->a.realp  = block;
    scratch->a.imagp  = block + count;
    scratch->b.realp  = block + count * 2U;
    scratch->b.imagp  = block + count * 3U;
    return true;
}

static inline bool sim_accel_split_load_interleaved(SimAccelSplitComplexScratch* scratch,
                                                    const SimComplexDouble*      src,
                                                    size_t                       count) {
    if (scratch == NULL || src == NULL) {
        return false;
    }
    if (count == 0U) {
        return true;
    }
    if (!sim_accel_split_reserve(scratch, count)) {
        return false;
    }

#if defined(SIM_HAVE_VDSP)
    DSPDoubleSplitComplex split = { .realp = scratch->a.realp, .imagp = scratch->a.imagp };
    vDSP_ctozD((const DSPDoubleComplex*) src, 2, &split, 1, (vDSP_Length) count);
#else
    for (size_t i = 0U; i < count; ++i) {
        scratch->a.realp[i] = src[i].re;
        scratch->a.imagp[i] = src[i].im;
    }
#endif
    return true;
}

static inline void sim_accel_split_store_interleaved(const SimAccelSplitDouble* src,
                                                     SimComplexDouble*          dst,
                                                     size_t                     count) {
    if (src == NULL || dst == NULL || count == 0U) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    DSPDoubleSplitComplex split = { .realp = src->realp, .imagp = src->imagp };
    vDSP_ztocD(&split, 1, (DSPDoubleComplex*) dst, 2, (vDSP_Length) count);
#else
    for (size_t i = 0U; i < count; ++i) {
        dst[i].re = src->realp[i];
        dst[i].im = src->imagp[i];
    }
#endif
}

static inline bool sim_accel_rotate_complex(SimAccelSplitComplexScratch* scratch,
                                            const SimComplexDouble*      src,
                                            SimComplexDouble*            dst,
                                            size_t                       count,
                                            double                       c,
                                            double                       s) {
    if (src == NULL || dst == NULL) {
        return false;
    }
    if (count == 0U) {
        return true;
    }

#if defined(SIM_HAVE_VDSP)
    if (!sim_accel_split_load_interleaved(scratch, src, count)) {
        return false;
    }

    double scalar_re = c;
    double scalar_im = s;
    DSPDoubleSplitComplex input  = { .realp = scratch->a.realp, .imagp = scratch->a.imagp };
    DSPDoubleSplitComplex output = { .realp = scratch->b.realp, .imagp = scratch->b.imagp };
    DSPDoubleSplitComplex scalar = { .realp = &scalar_re, .imagp = &scalar_im };
    vDSP_zvzsmlD(&input, 1, &scalar, &output, 1, (vDSP_Length) count);
    sim_accel_split_store_interleaved(&scratch->b, dst, count);
    return true;
#else
    for (size_t i = 0U; i < count; ++i) {
        double re = src[i].re;
        double im = src[i].im;
        dst[i].re = re * c - im * s;
        dst[i].im = re * s + im * c;
    }
    return true;
#endif
}

static inline bool sim_accel_scan_real_finite_maxabs(const double* data,
                                                     size_t        count,
                                                     double*       out_max_abs) {
    if (out_max_abs != NULL) {
        *out_max_abs = 0.0;
    }
    if (data == NULL) {
        return false;
    }

    double max_abs = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            return false;
        }
        double abs_value = fabs(value);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }

    if (out_max_abs != NULL) {
        *out_max_abs = max_abs;
    }
    return true;
}

static inline bool sim_accel_scan_complex_finite_maxabs(const SimComplexDouble* data,
                                                        size_t                  count,
                                                        double*                 out_max_re,
                                                        double*                 out_max_im) {
    if (out_max_re != NULL) {
        *out_max_re = 0.0;
    }
    if (out_max_im != NULL) {
        *out_max_im = 0.0;
    }
    if (data == NULL) {
        return false;
    }

    double max_re = 0.0;
    double max_im = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double re = data[i].re;
        double im = data[i].im;
        if (!isfinite(re) || !isfinite(im)) {
            return false;
        }
        double abs_re = fabs(re);
        double abs_im = fabs(im);
        if (abs_re > max_re) {
            max_re = abs_re;
        }
        if (abs_im > max_im) {
            max_im = abs_im;
        }
    }

    if (out_max_re != NULL) {
        *out_max_re = max_re;
    }
    if (out_max_im != NULL) {
        *out_max_im = max_im;
    }
    return true;
}

static inline bool sim_accel_affine_mix_real(const double* lhs,
                                             const double* rhs,
                                             double*       out,
                                             size_t        count,
                                             double        lhs_coef,
                                             double        rhs_coef,
                                             double        bias,
                                             bool          accumulate,
                                             double        add_scale) {
    if (lhs == NULL || rhs == NULL || out == NULL) {
        return false;
    }
    if (count == 0U) {
        return true;
    }

#if defined(SIM_HAVE_VDSP)
    double lhs_max = 0.0;
    double rhs_max = 0.0;
    double max_term;

    if (!sim_accel_scan_real_finite_maxabs(lhs, count, &lhs_max) ||
        !sim_accel_scan_real_finite_maxabs(rhs, count, &rhs_max)) {
        return false;
    }

    max_term = fabs(lhs_coef) * lhs_max + fabs(rhs_coef) * rhs_max + fabs(bias);
    if (!isfinite(max_term) || max_term > DBL_MAX) {
        return false;
    }
    if (accumulate && add_scale == 0.0) {
        return true;
    }

    const vDSP_Length len       = (vDSP_Length) count;
    const double      lhs_term  = accumulate ? add_scale * lhs_coef : lhs_coef;
    const double      rhs_term  = accumulate ? add_scale * rhs_coef : rhs_coef;
    const double      bias_term = accumulate ? add_scale * bias : bias;

    if (accumulate) {
        if (lhs == out && rhs == out) {
            double self_term = 1.0 + lhs_term + rhs_term;
            vDSP_vsmulD(out, 1, &self_term, out, 1, len);
        } else if (lhs == out) {
            double self_term = 1.0 + lhs_term;
            vDSP_vsmulD(out, 1, &self_term, out, 1, len);
            if (rhs_term != 0.0) {
                vDSP_vsmaD(rhs, 1, &rhs_term, out, 1, out, 1, len);
            }
        } else if (rhs == out) {
            double self_term = 1.0 + rhs_term;
            vDSP_vsmulD(out, 1, &self_term, out, 1, len);
            if (lhs_term != 0.0) {
                vDSP_vsmaD(lhs, 1, &lhs_term, out, 1, out, 1, len);
            }
        } else {
            if (lhs_term != 0.0) {
                vDSP_vsmaD(lhs, 1, &lhs_term, out, 1, out, 1, len);
            }
            if (rhs_term != 0.0) {
                vDSP_vsmaD(rhs, 1, &rhs_term, out, 1, out, 1, len);
            }
        }
        if (bias_term != 0.0) {
            vDSP_vsaddD(out, 1, &bias_term, out, 1, len);
        }
        return true;
    }

    if (lhs == out && rhs == out) {
        double self_term = lhs_term + rhs_term;
        vDSP_vsmulD(out, 1, &self_term, out, 1, len);
    } else if (lhs == out) {
        vDSP_vsmulD(out, 1, &lhs_term, out, 1, len);
        if (rhs_term != 0.0) {
            vDSP_vsmaD(rhs, 1, &rhs_term, out, 1, out, 1, len);
        }
    } else if (rhs == out) {
        vDSP_vsmulD(out, 1, &rhs_term, out, 1, len);
        if (lhs_term != 0.0) {
            vDSP_vsmaD(lhs, 1, &lhs_term, out, 1, out, 1, len);
        }
    } else if (lhs_term != 0.0) {
        vDSP_vsmulD(lhs, 1, &lhs_term, out, 1, len);
        if (rhs_term != 0.0) {
            vDSP_vsmaD(rhs, 1, &rhs_term, out, 1, out, 1, len);
        }
    } else if (rhs_term != 0.0) {
        vDSP_vsmulD(rhs, 1, &rhs_term, out, 1, len);
    } else {
        vDSP_vclrD(out, 1, len);
    }

    if (bias_term != 0.0) {
        vDSP_vsaddD(out, 1, &bias_term, out, 1, len);
    }
    return true;
#else
    (void) lhs;
    (void) rhs;
    (void) out;
    (void) count;
    (void) lhs_coef;
    (void) rhs_coef;
    (void) bias;
    (void) accumulate;
    (void) add_scale;
    return false;
#endif
}

static inline bool sim_accel_affine_mix_complex(SimAccelSplitComplexScratch* scratch,
                                                const SimComplexDouble*      lhs_c,
                                                const double*                lhs_r,
                                                const SimComplexDouble*      rhs_c,
                                                const double*                rhs_r,
                                                SimComplexDouble*            out,
                                                size_t                       count,
                                                double                       lhs_coef,
                                                double                       rhs_coef,
                                                double                       bias,
                                                bool                         accumulate,
                                                double                       add_scale) {
    if (scratch == NULL || out == NULL) {
        return false;
    }
    if ((lhs_c == NULL && lhs_r == NULL) || (rhs_c == NULL && rhs_r == NULL)) {
        return false;
    }
    if (count == 0U) {
        return true;
    }

#if defined(SIM_HAVE_VDSP)
    double lhs_max_re = 0.0;
    double lhs_max_im = 0.0;
    double rhs_max_re = 0.0;
    double rhs_max_im = 0.0;
    double max_re;
    double max_im;

    if (lhs_c != NULL) {
        if (!sim_accel_scan_complex_finite_maxabs(lhs_c, count, &lhs_max_re, &lhs_max_im)) {
            return false;
        }
    } else if (!sim_accel_scan_real_finite_maxabs(lhs_r, count, &lhs_max_re)) {
        return false;
    }

    if (rhs_c != NULL) {
        if (!sim_accel_scan_complex_finite_maxabs(rhs_c, count, &rhs_max_re, &rhs_max_im)) {
            return false;
        }
    } else if (!sim_accel_scan_real_finite_maxabs(rhs_r, count, &rhs_max_re)) {
        return false;
    }

    max_re = fabs(lhs_coef) * lhs_max_re + fabs(rhs_coef) * rhs_max_re + fabs(bias);
    max_im = fabs(lhs_coef) * lhs_max_im + fabs(rhs_coef) * rhs_max_im;
    if (!isfinite(max_re) || !isfinite(max_im) || max_re > DBL_MAX || max_im > DBL_MAX) {
        return false;
    }
    if (accumulate && add_scale == 0.0) {
        return true;
    }
    if (!sim_accel_split_reserve(scratch, count)) {
        return false;
    }

    const vDSP_Length     len       = (vDSP_Length) count;
    const double          lhs_term  = accumulate ? add_scale * lhs_coef : lhs_coef;
    const double          rhs_term  = accumulate ? add_scale * rhs_coef : rhs_coef;
    const double          bias_term = accumulate ? add_scale * bias : bias;
    DSPDoubleSplitComplex accum     = { .realp = scratch->a.realp, .imagp = scratch->a.imagp };
    DSPDoubleSplitComplex temp      = { .realp = scratch->b.realp, .imagp = scratch->b.imagp };

    if (accumulate) {
        vDSP_ctozD((const DSPDoubleComplex*) out, 2, &accum, 1, len);
    } else {
        vDSP_vclrD(accum.realp, 1, len);
        vDSP_vclrD(accum.imagp, 1, len);
    }

    if (lhs_c != NULL && lhs_term != 0.0) {
        vDSP_ctozD((const DSPDoubleComplex*) lhs_c, 2, &temp, 1, len);
        vDSP_vsmaD(temp.realp, 1, &lhs_term, accum.realp, 1, accum.realp, 1, len);
        vDSP_vsmaD(temp.imagp, 1, &lhs_term, accum.imagp, 1, accum.imagp, 1, len);
    } else if (lhs_r != NULL && lhs_term != 0.0) {
        vDSP_vsmaD(lhs_r, 1, &lhs_term, accum.realp, 1, accum.realp, 1, len);
    }

    if (rhs_c != NULL && rhs_term != 0.0) {
        vDSP_ctozD((const DSPDoubleComplex*) rhs_c, 2, &temp, 1, len);
        vDSP_vsmaD(temp.realp, 1, &rhs_term, accum.realp, 1, accum.realp, 1, len);
        vDSP_vsmaD(temp.imagp, 1, &rhs_term, accum.imagp, 1, accum.imagp, 1, len);
    } else if (rhs_r != NULL && rhs_term != 0.0) {
        vDSP_vsmaD(rhs_r, 1, &rhs_term, accum.realp, 1, accum.realp, 1, len);
    }

    if (bias_term != 0.0) {
        vDSP_vsaddD(accum.realp, 1, &bias_term, accum.realp, 1, len);
    }

    sim_accel_split_store_interleaved(&scratch->a, out, count);
    return true;
#else
    (void) scratch;
    (void) lhs_c;
    (void) lhs_r;
    (void) rhs_c;
    (void) rhs_r;
    (void) out;
    (void) count;
    (void) lhs_coef;
    (void) rhs_coef;
    (void) bias;
    (void) accumulate;
    (void) add_scale;
    return false;
#endif
}

static inline void sim_accel_copy_scale_real(const double* src,
                                             double*       dst,
                                             size_t        count,
                                             double        scale,
                                             bool          accumulate) {
    if (src == NULL || dst == NULL || count == 0U) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    const vDSP_Length length = (vDSP_Length) count;
    if (accumulate) {
        vDSP_vsmaD(src, 1, &scale, dst, 1, dst, 1, length);
        return;
    }

    if (scale == 1.0) {
        if (dst != src) {
            memmove(dst, src, count * sizeof(double));
        }
        return;
    }

    vDSP_vsmulD(src, 1, &scale, dst, 1, length);
#else
    if (accumulate) {
        for (size_t i = 0U; i < count; ++i) {
            dst[i] += scale * src[i];
        }
        return;
    }

    if (scale == 1.0) {
        if (dst != src) {
            memmove(dst, src, count * sizeof(double));
        }
        return;
    }

    for (size_t i = 0U; i < count; ++i) {
        dst[i] = scale * src[i];
    }
#endif
}

static inline void sim_accel_weighted_sum_real(const double* const* srcs,
                                               const double*       coeffs,
                                               size_t              source_count,
                                               double*             out,
                                               size_t              count) {
    if (srcs == NULL || coeffs == NULL || out == NULL || count == 0U) {
        return;
    }

    switch (source_count) {
        case 1U: {
            const double* src0 = srcs[0];
            const double  c0   = coeffs[0];

            if (src0 == NULL || c0 == 0.0) {
                memset(out, 0, count * sizeof(double));
                return;
            }
            if (c0 == 1.0) {
                if (out != src0) {
                    memmove(out, src0, count * sizeof(double));
                }
                return;
            }

            SIM_ACCEL_VECTORIZE_LOOP
            for (size_t i = 0U; i < count; ++i) {
                out[i] = c0 * src0[i];
            }
            return;
        }
        case 2U: {
            const double* src0 = srcs[0];
            const double* src1 = srcs[1];
            const double  c0   = coeffs[0];
            const double  c1   = coeffs[1];

            SIM_ACCEL_VECTORIZE_LOOP
            for (size_t i = 0U; i < count; ++i) {
                double value = 0.0;
                if (src0 != NULL && c0 != 0.0) {
                    value += c0 * src0[i];
                }
                if (src1 != NULL && c1 != 0.0) {
                    value += c1 * src1[i];
                }
                out[i] = value;
            }
            return;
        }
        case 3U: {
            const double* src0 = srcs[0];
            const double* src1 = srcs[1];
            const double* src2 = srcs[2];
            const double  c0   = coeffs[0];
            const double  c1   = coeffs[1];
            const double  c2   = coeffs[2];

            SIM_ACCEL_VECTORIZE_LOOP
            for (size_t i = 0U; i < count; ++i) {
                double value = 0.0;
                if (src0 != NULL && c0 != 0.0) {
                    value += c0 * src0[i];
                }
                if (src1 != NULL && c1 != 0.0) {
                    value += c1 * src1[i];
                }
                if (src2 != NULL && c2 != 0.0) {
                    value += c2 * src2[i];
                }
                out[i] = value;
            }
            return;
        }
        case 4U: {
            const double* src0 = srcs[0];
            const double* src1 = srcs[1];
            const double* src2 = srcs[2];
            const double* src3 = srcs[3];
            const double  c0   = coeffs[0];
            const double  c1   = coeffs[1];
            const double  c2   = coeffs[2];
            const double  c3   = coeffs[3];

            SIM_ACCEL_VECTORIZE_LOOP
            for (size_t i = 0U; i < count; ++i) {
                double value = 0.0;
                if (src0 != NULL && c0 != 0.0) {
                    value += c0 * src0[i];
                }
                if (src1 != NULL && c1 != 0.0) {
                    value += c1 * src1[i];
                }
                if (src2 != NULL && c2 != 0.0) {
                    value += c2 * src2[i];
                }
                if (src3 != NULL && c3 != 0.0) {
                    value += c3 * src3[i];
                }
                out[i] = value;
            }
            return;
        }
        case 5U: {
            const double* src0 = srcs[0];
            const double* src1 = srcs[1];
            const double* src2 = srcs[2];
            const double* src3 = srcs[3];
            const double* src4 = srcs[4];
            const double  c0   = coeffs[0];
            const double  c1   = coeffs[1];
            const double  c2   = coeffs[2];
            const double  c3   = coeffs[3];
            const double  c4   = coeffs[4];

            SIM_ACCEL_VECTORIZE_LOOP
            for (size_t i = 0U; i < count; ++i) {
                double value = 0.0;
                if (src0 != NULL && c0 != 0.0) {
                    value += c0 * src0[i];
                }
                if (src1 != NULL && c1 != 0.0) {
                    value += c1 * src1[i];
                }
                if (src2 != NULL && c2 != 0.0) {
                    value += c2 * src2[i];
                }
                if (src3 != NULL && c3 != 0.0) {
                    value += c3 * src3[i];
                }
                if (src4 != NULL && c4 != 0.0) {
                    value += c4 * src4[i];
                }
                out[i] = value;
            }
            return;
        }
        case 6U: {
            const double* src0 = srcs[0];
            const double* src1 = srcs[1];
            const double* src2 = srcs[2];
            const double* src3 = srcs[3];
            const double* src4 = srcs[4];
            const double* src5 = srcs[5];
            const double  c0   = coeffs[0];
            const double  c1   = coeffs[1];
            const double  c2   = coeffs[2];
            const double  c3   = coeffs[3];
            const double  c4   = coeffs[4];
            const double  c5   = coeffs[5];

            SIM_ACCEL_VECTORIZE_LOOP
            for (size_t i = 0U; i < count; ++i) {
                double value = 0.0;
                if (src0 != NULL && c0 != 0.0) {
                    value += c0 * src0[i];
                }
                if (src1 != NULL && c1 != 0.0) {
                    value += c1 * src1[i];
                }
                if (src2 != NULL && c2 != 0.0) {
                    value += c2 * src2[i];
                }
                if (src3 != NULL && c3 != 0.0) {
                    value += c3 * src3[i];
                }
                if (src4 != NULL && c4 != 0.0) {
                    value += c4 * src4[i];
                }
                if (src5 != NULL && c5 != 0.0) {
                    value += c5 * src5[i];
                }
                out[i] = value;
            }
            return;
        }
        default:
            break;
    }

    SIM_ACCEL_VECTORIZE_LOOP
    for (size_t element = 0U; element < count; ++element) {
        double value = 0.0;
        for (size_t source = 0U; source < source_count; ++source) {
            const double* src = srcs[source];
            if (src == NULL || coeffs[source] == 0.0) {
                continue;
            }
            value += coeffs[source] * src[element];
        }
        out[element] = value;
    }
}

static inline void sim_accel_copy_scale_complex(const SimComplexDouble* src,
                                                SimComplexDouble*       dst,
                                                size_t                  count,
                                                double                  scale,
                                                bool                    accumulate) {
    if (src == NULL || dst == NULL || count == 0U) {
        return;
    }

    sim_accel_copy_scale_real((const double*) src, (double*) dst, count * 2U, scale, accumulate);
}

static inline void sim_accel_add_scalar_real(double* dst, size_t count, double value) {
    if (dst == NULL || count == 0U || value == 0.0) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    const vDSP_Length length = (vDSP_Length) count;
    vDSP_vsaddD(dst, 1, &value, dst, 1, length);
#else
    for (size_t i = 0U; i < count; ++i) {
        dst[i] += value;
    }
#endif
}

static inline void sim_accel_add_scalar_complex(SimComplexDouble* dst,
                                                size_t            count,
                                                double            real_add,
                                                double            imag_add) {
    if (dst == NULL || count == 0U || (real_add == 0.0 && imag_add == 0.0)) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    const vDSP_Length length = (vDSP_Length) count;
    double*           realp  = (double*) dst;
    double*           imagp  = realp + 1U;

    if (real_add != 0.0) {
        vDSP_vsaddD(realp, 2, &real_add, realp, 2, length);
    }
    if (imag_add != 0.0) {
        vDSP_vsaddD(imagp, 2, &imag_add, imagp, 2, length);
    }
#else
    for (size_t i = 0U; i < count; ++i) {
        dst[i].re += real_add;
        dst[i].im += imag_add;
    }
#endif
}

static inline void sim_accel_accumulate_real_to_complex(const double*      src,
                                                        SimComplexDouble* dst,
                                                        size_t            count,
                                                        double            real_scale,
                                                        double            imag_scale) {
    if (src == NULL || dst == NULL || count == 0U ||
        (real_scale == 0.0 && imag_scale == 0.0)) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    const vDSP_Length length = (vDSP_Length) count;
    double*           realp  = (double*) dst;
    double*           imagp  = realp + 1U;

    if (real_scale != 0.0) {
        vDSP_vsmaD(src, 1, &real_scale, realp, 2, realp, 2, length);
    }
    if (imag_scale != 0.0) {
        vDSP_vsmaD(src, 1, &imag_scale, imagp, 2, imagp, 2, length);
    }
#else
    for (size_t i = 0U; i < count; ++i) {
        dst[i].re += real_scale * src[i];
        dst[i].im += imag_scale * src[i];
    }
#endif
}

static inline void sim_accel_scale_inplace_real(double* data, size_t count, double scale) {
    if (data == NULL || count == 0U || scale == 1.0) {
        return;
    }

#if defined(SIM_HAVE_VDSP)
    const vDSP_Length length = (vDSP_Length) count;
    vDSP_vsmulD(data, 1, &scale, data, 1, length);
#else
    for (size_t i = 0U; i < count; ++i) {
        data[i] *= scale;
    }
#endif
}

static inline void sim_accel_scale_inplace_complex_real(SimComplexDouble* data,
                                                        size_t            count,
                                                        double            scale) {
    if (data == NULL || count == 0U || scale == 1.0) {
        return;
    }

    sim_accel_scale_inplace_real((double*) data, count * 2U, scale);
}

#endif /* OAKFIELD_SIM_ACCEL_H */
