#include "oakfield/operators/stimulus/posenc.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_POSENC_EPS 1.0e-12
#define STIM_POSENC_MAX_BANDS 64U
#define STIM_POSENC_VDSP_MIN_LEN 64U

typedef struct SimStimulusPosEncState {
    SimStimulusPosEncConfig config;
    double*                 vdsp_block;
    double*                 vdsp_theta;
    double*                 vdsp_value;
    double*                 vdsp_accum_re;
    double*                 vdsp_accum_im;
    size_t                  vdsp_capacity;
    char                    symbolic[208];
} SimStimulusPosEncState;

static void posenc_normalize(SimStimulusPosEncConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->base_wavenumber)) {
        config->base_wavenumber = 1.0;
    }
    if (!isfinite(config->band_growth) || fabs(config->band_growth) <= STIM_POSENC_EPS) {
        config->band_growth = 2.0;
    }
    if (config->band_count == 0U) {
        config->band_count = 6U;
    }
    if (config->band_count > STIM_POSENC_MAX_BANDS) {
        config->band_count = STIM_POSENC_MAX_BANDS;
    }
    if (!isfinite(config->kx)) {
        config->kx = 0.0;
    }
    if (!isfinite(config->ky)) {
        config->ky = 0.0;
    }
    if (!isfinite(config->omega)) {
        config->omega = 0.0;
    }
    if (!isfinite(config->phase)) {
        config->phase = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }

    if (!config->use_wavevector &&
        (fabs(config->kx) > STIM_POSENC_EPS || fabs(config->ky) > STIM_POSENC_EPS)) {
        config->use_wavevector = true;
    }

    if (config->use_wavevector && fabs(config->kx) <= STIM_POSENC_EPS &&
        fabs(config->ky) <= STIM_POSENC_EPS) {
        config->kx = 1.0;
        config->ky = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void posenc_refresh_symbolic(SimStimulusPosEncState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusPosEncConfig* cfg = &state->config;
    if (cfg->use_wavevector) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "posenc A=%.3g k0=%.3g g=%.3g B=%u k=(%.3g,%.3g)",
                        cfg->amplitude,
                        cfg->base_wavenumber,
                        cfg->band_growth,
                        cfg->band_count,
                        cfg->kx,
                        cfg->ky);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "posenc A=%.3g k0=%.3g g=%.3g B=%u",
                        cfg->amplitude,
                        cfg->base_wavenumber,
                        cfg->band_growth,
                        cfg->band_count);
    }
#else
    (void) state;
#endif
}

static const char* posenc_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusPosEncState* state = (const SimStimulusPosEncState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void posenc_eval_scalar(const SimStimulusPosEncConfig* cfg,
                               double                         u,
                               double                         t,
                               double*                        out_re,
                               double*                        out_im) {
    double re_sum = cfg->include_identity ? u : 0.0;
    double im_sum = 0.0;

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->band_count; ++i) {
        double k     = cfg->base_wavenumber * band_scale;
        double theta = k * u - cfg->omega * t + cfg->phase;
        re_sum += cos(theta);
        im_sum += sin(theta);
        band_scale *= cfg->band_growth;
    }

    double count = (double) cfg->band_count + (cfg->include_identity ? 1.0 : 0.0);
    if (count > STIM_POSENC_EPS) {
        double norm = 1.0 / sqrt(count);
        re_sum *= norm;
        im_sum *= norm;
    }

    *out_re = re_sum;
    *out_im = im_sum;
}

static void posenc_eval_wavevector(const SimStimulusPosEncConfig* cfg,
                                   double                         x,
                                   double                         y,
                                   double                         t,
                                   double*                        out_re,
                                   double*                        out_im) {
    double projection = cfg->kx * x + cfg->ky * y;
    double kv_norm    = hypot(cfg->kx, cfg->ky);

    double identity_u = (kv_norm > STIM_POSENC_EPS) ? (projection / kv_norm) : x;
    double re_sum     = cfg->include_identity ? identity_u : 0.0;
    double im_sum     = 0.0;

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->band_count; ++i) {
        double k     = cfg->base_wavenumber * band_scale;
        double theta = k * projection - cfg->omega * t + cfg->phase;
        re_sum += cos(theta);
        im_sum += sin(theta);
        band_scale *= cfg->band_growth;
    }

    double count = (double) cfg->band_count + (cfg->include_identity ? 1.0 : 0.0);
    if (count > STIM_POSENC_EPS) {
        double norm = 1.0 / sqrt(count);
        re_sum *= norm;
        im_sum *= norm;
    }

    *out_re = re_sum;
    *out_im = im_sum;
}

static void posenc_destroy(void* state_ptr) {
    SimStimulusPosEncState* state = (SimStimulusPosEncState*) state_ptr;
    if (state != NULL) {
        free(state->vdsp_block);
    }
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool posenc_vdsp_ensure_buffers(SimStimulusPosEncState* state, size_t width) {
    if (state == NULL) {
        return false;
    }
    if (width == 0U) {
        return true;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* resized = (double*) realloc(state->vdsp_block, width * 4U * sizeof(double));
    if (resized == NULL) {
        return false;
    }

    state->vdsp_block    = resized;
    state->vdsp_capacity = width;
    state->vdsp_theta    = resized;
    state->vdsp_value    = resized + width;
    state->vdsp_accum_re = resized + width * 2U;
    state->vdsp_accum_im = resized + width * 3U;
    return true;
}

static bool posenc_vdsp_eval_encoding(SimStimulusPosEncState* state,
                                      size_t                  width,
                                      double                  theta_start_base,
                                      double                  theta_step_base,
                                      double                  identity_start,
                                      double                  identity_step,
                                      double                  phase_bias) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (!posenc_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusPosEncConfig* cfg = &state->config;
    const vDSP_Length              len = (vDSP_Length) width;
    const int                      n   = (int) width;
    const double                   one = 1.0;

    if (!isfinite(theta_start_base) || !isfinite(theta_step_base) || !isfinite(identity_start) ||
        !isfinite(identity_step) || !isfinite(phase_bias)) {
        return false;
    }

    if (cfg->include_identity) {
        vDSP_vrampD(&identity_start, &identity_step, state->vdsp_accum_re, 1, len);
    } else {
        vDSP_vclrD(state->vdsp_accum_re, 1, len);
    }
    vDSP_vclrD(state->vdsp_accum_im, 1, len);

    double band_scale = 1.0;
    for (unsigned int i = 0U; i < cfg->band_count; ++i) {
        double k           = cfg->base_wavenumber * band_scale;
        double theta_start = k * theta_start_base + phase_bias;
        double theta_step  = k * theta_step_base;
        if (!isfinite(k) || !isfinite(theta_start) || !isfinite(theta_step)) {
            return false;
        }

        vDSP_vrampD(&theta_start, &theta_step, state->vdsp_theta, 1, len);
        vvcos(state->vdsp_value, state->vdsp_theta, &n);
        vDSP_vsmaD(
            state->vdsp_value, 1, &one, state->vdsp_accum_re, 1, state->vdsp_accum_re, 1, len);
        vvsin(state->vdsp_value, state->vdsp_theta, &n);
        vDSP_vsmaD(
            state->vdsp_value, 1, &one, state->vdsp_accum_im, 1, state->vdsp_accum_im, 1, len);

        band_scale *= cfg->band_growth;
        if (!isfinite(band_scale) && i + 1U < cfg->band_count) {
            return false;
        }
    }

    double norm_count = (double) cfg->band_count + (cfg->include_identity ? 1.0 : 0.0);
    if (!isfinite(norm_count) || norm_count <= STIM_POSENC_EPS) {
        return false;
    }

    double norm = 1.0 / sqrt(norm_count);
    if (!isfinite(norm)) {
        return false;
    }
    sim_accel_scale_inplace_real(state->vdsp_accum_re, width, norm);
    sim_accel_scale_inplace_real(state->vdsp_accum_im, width, norm);
    return true;
}

static bool posenc_try_vdsp_linear_rows(SimStimulusPosEncState* state,
                                        const SimField*         field,
                                        bool                    is_complex,
                                        double*                 dst_real,
                                        SimComplexDouble*       dst_complex,
                                        size_t                  count,
                                        double                  scale,
                                        double                  t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t rank = field->layout.rank;
    if (rank == 0U || rank > 2U) {
        return false;
    }

    const SimStimulusPosEncConfig* cfg            = &state->config;
    bool                           use_wavevector = cfg->use_wavevector;
    bool   separable = (!use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double u_x       = 0.0;
    double u_y       = 0.0;

    if (!use_wavevector) {
        switch (cfg->coord.mode) {
            case SIM_STIMULUS_COORD_AXIS:
                if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                    u_x = 0.0;
                    u_y = 1.0;
                } else {
                    u_x = 1.0;
                    u_y = 0.0;
                }
                break;
            case SIM_STIMULUS_COORD_ANGLE:
                u_x = cos(cfg->coord.angle);
                u_y = sin(cfg->coord.angle);
                break;
            case SIM_STIMULUS_COORD_SEPARABLE:
                u_x = 1.0;
                u_y = 0.0;
                break;
            default:
                return false;
        }
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_POSENC_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!posenc_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double x0         = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0         = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx         = cfg->coord.spacing_x;
    double dy         = cfg->coord.spacing_y;
    double phase_bias = cfg->phase - cfg->omega * t;
    double kv_norm    = 1.0;
    if (use_wavevector) {
        kv_norm = hypot(cfg->kx, cfg->ky);
        if (!isfinite(kv_norm) || kv_norm <= STIM_POSENC_EPS) {
            return false;
        }
    }

    double out_re_scale = scale * cfg->amplitude;
    double out_im_scale = 0.0;
    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            sin_r = sin(cfg->rotation);
            cos_r = cos(cfg->rotation);
        }
        out_re_scale = scale * cfg->amplitude * cos_r;
        out_im_scale = scale * cfg->amplitude * sin_r;
        if (!isfinite(out_re_scale) || !isfinite(out_im_scale)) {
            return false;
        }
    } else if (!isfinite(out_re_scale)) {
        return false;
    }

    const vDSP_Length len = (vDSP_Length) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        bool   ok       = false;

        if (use_wavevector) {
            double proj_start = cfg->kx * x0 + cfg->ky * sample_y;
            double proj_step  = cfg->kx * dx;
            ok                = posenc_vdsp_eval_encoding(state,
                                                          width,
                                                          proj_start,
                                                          proj_step,
                                                          proj_start / kv_norm,
                                                          proj_step / kv_norm,
                                                          phase_bias);
        } else if (separable) {
            ok = posenc_vdsp_eval_encoding(state, width, x0, dx, x0, dx, phase_bias);
            if (ok) {
                double y_re = 0.0;
                double y_im = 0.0;
                posenc_eval_scalar(cfg, sample_y, t, &y_re, &y_im);
                if (!isfinite(y_re) || !isfinite(y_im)) {
                    return false;
                }
                if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    sim_accel_add_scalar_real(state->vdsp_accum_re, width, y_re);
                    sim_accel_add_scalar_real(state->vdsp_accum_im, width, y_im);
                } else {
                    memmove(state->vdsp_value, state->vdsp_accum_re, width * sizeof(double));
                    sim_accel_scale_inplace_real(state->vdsp_accum_re, width, y_re);
                    {
                        double neg_y_im = -y_im;
                        vDSP_vsmaD(state->vdsp_accum_im,
                                   1,
                                   &neg_y_im,
                                   state->vdsp_accum_re,
                                   1,
                                   state->vdsp_accum_re,
                                   1,
                                   len);
                    }
                    sim_accel_scale_inplace_real(state->vdsp_accum_im, width, y_re);
                    vDSP_vsmaD(state->vdsp_value,
                               1,
                               &y_im,
                               state->vdsp_accum_im,
                               1,
                               state->vdsp_accum_im,
                               1,
                               len);
                }
            }
        } else {
            double u_start = u_x * x0 + u_y * sample_y;
            double u_step  = u_x * dx;
            ok             = posenc_vdsp_eval_encoding(
                state, width, u_start, u_step, u_start, u_step, phase_bias);
        }

        if (!ok) {
            return false;
        }

        if (!is_complex) {
            double* row_ptr = dst_real + row * width;
            vDSP_vsmaD(state->vdsp_accum_re, 1, &out_re_scale, row_ptr, 1, row_ptr, 1, len);
        } else {
            SimComplexDouble* row_ptr = dst_complex + row * width;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;
            vDSP_vsmaD(state->vdsp_accum_re, 1, &out_re_scale, row_re, 2, row_re, 2, len);
            if (out_im_scale != 0.0) {
                double neg_out_im = -out_im_scale;
                vDSP_vsmaD(state->vdsp_accum_im, 1, &neg_out_im, row_re, 2, row_re, 2, len);
            }
            if (out_im_scale != 0.0) {
                vDSP_vsmaD(state->vdsp_accum_re, 1, &out_im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_accum_im, 1, &out_re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static SimResult posenc_step(void*               state_ptr,
                             struct SimContext*  context,
                             struct SimOperator* self,
                             size_t              substep_index,
                             double              dt_sub,
                             void*               scratch,
                             size_t              scratch_size) {
    (void) self;
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;

    SimStimulusPosEncState* state = (SimStimulusPosEncState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;

    if (!is_complex) {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(double);
    } else {
        if (field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        count = sim_field_bytes(field) / sizeof(SimComplexDouble);
    }

    if (count == 0U || state->config.amplitude == 0.0 || state->config.band_count == 0U) {
        return SIM_RESULT_OK;
    }

    double scale          = state->config.scale_by_dt ? dt_sub : 1.0;
    double t              = sim_context_time(context) + state->config.time_offset;
    bool   use_wavevector = state->config.use_wavevector;
    bool separable = (!use_wavevector && state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

#if defined(SIM_HAVE_VDSP)
    if (posenc_try_vdsp_linear_rows(state,
                                    field,
                                    is_complex,
                                    is_complex ? NULL : (double*) sim_field_data(field),
                                    is_complex ? sim_field_complex_data(field) : NULL,
                                    count,
                                    scale,
                                    t)) {
        return SIM_RESULT_OK;
    }
#endif

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t i = 0U; i < count; ++i) {
            double x        = 0.0;
            double y        = 0.0;
            double sample_x = 0.0;
            double sample_y = 0.0;
            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_stimulus_coord_sample_xy(&state->config.coord, x, y, t, &sample_x, &sample_y);

            double re = 0.0;
            double im = 0.0;

            if (use_wavevector) {
                posenc_eval_wavevector(&state->config, sample_x, sample_y, t, &re, &im);
            } else if (separable) {
                double rx = 0.0;
                double ix = 0.0;
                double ry = 0.0;
                double iy = 0.0;
                posenc_eval_scalar(&state->config, sample_x, t, &rx, &ix);
                posenc_eval_scalar(&state->config, sample_y, t, &ry, &iy);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = rx + ry;
                    im = ix + iy;
                } else {
                    re = rx * ry - ix * iy;
                    im = rx * iy + ix * ry;
                }
            } else {
                double u = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                posenc_eval_scalar(&state->config, u, t, &re, &im);
            }

            (void) im;
            double value = state->config.amplitude * re;
            if (isfinite(value)) {
                dst[i] += scale * value;
            }
        }
    } else {
        SimComplexDouble* dst = sim_field_complex_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        double sin_r = 0.0;
        double cos_r = 1.0;
        if (state->config.rotation != 0.0) {
            sin_r = sin(state->config.rotation);
            cos_r = cos(state->config.rotation);
        }

        for (size_t i = 0U; i < count; ++i) {
            double x        = 0.0;
            double y        = 0.0;
            double sample_x = 0.0;
            double sample_y = 0.0;
            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sim_stimulus_coord_sample_xy(&state->config.coord, x, y, t, &sample_x, &sample_y);

            double re = 0.0;
            double im = 0.0;

            if (use_wavevector) {
                posenc_eval_wavevector(&state->config, sample_x, sample_y, t, &re, &im);
            } else if (separable) {
                double rx = 0.0;
                double ix = 0.0;
                double ry = 0.0;
                double iy = 0.0;
                posenc_eval_scalar(&state->config, sample_x, t, &rx, &ix);
                posenc_eval_scalar(&state->config, sample_y, t, &ry, &iy);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = rx + ry;
                    im = ix + iy;
                } else {
                    re = rx * ry - ix * iy;
                    im = rx * iy + ix * ry;
                }
            } else {
                double u = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                posenc_eval_scalar(&state->config, u, t, &re, &im);
            }

            re *= state->config.amplitude;
            im *= state->config.amplitude;
            double out_re = re * cos_r - im * sin_r;
            double out_im = re * sin_r + im * cos_r;
            if (isfinite(out_re) && isfinite(out_im)) {
                dst[i].re += scale * out_re;
                dst[i].im += scale * out_im;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_posenc_operator(struct SimContext*             context,
                                           const SimStimulusPosEncConfig* config,
                                           size_t*                        out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusPosEncConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    posenc_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_posenc",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusPosEncState* state = (SimStimulusPosEncState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    posenc_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_posenc");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = true;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stimulus_posenc";
    sim_operator_info_set_schema_identity(&info, "stimulus_posenc");
    info.algebraic_flags       = SIM_OPERATOR_ALG_LINEAR;
    info.representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    bool needs_complex =
        sim_field_is_complex(sim_context_field(context, state->config.field_index));

    info.representation.value_kind                      = SIM_FIELD_VALUE_REAL_SCALAR;
    info.representation.requires_complex_input          = false;
    info.representation.requires_complex_representation = false;
    info.representation.preserves_real_subspace         = info.preserves_real;
    if (needs_complex) {
        info.representation.value_kind                      = SIM_FIELD_VALUE_COMPLEX_SCALAR;
        info.representation.requires_complex_input          = true;
        info.representation.requires_complex_representation = true;
    }

    SimOperatorConfig op_config = sim_operator_config_defaults();

    SimSplitPort port = { .context_field_index = state->config.field_index,
                          .require_complex     = needs_complex };

    SimSplitAccess access = { .port = 0, .mode = SIM_ACCESS_RW };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = posenc_step,
                                .accesses          = &access,
                                .access_count      = 1U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = &port,
                                .port_count    = 1U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = posenc_symbolic,
                                .destroy       = posenc_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        posenc_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_posenc_config(struct SimContext*       context,
                                     size_t                   operator_index,
                                     SimStimulusPosEncConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusPosEncState* state = (SimStimulusPosEncState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_posenc_update(struct SimContext*             context,
                                     size_t                         operator_index,
                                     const SimStimulusPosEncConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusPosEncState* state = (SimStimulusPosEncState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusPosEncConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    posenc_normalize(&local);
    state->config = local;
    posenc_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
