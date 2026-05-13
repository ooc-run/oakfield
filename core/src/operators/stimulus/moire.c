#include "oakfield/operators/stimulus/moire.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "sim_accel.h"
#include "oakfield/sim_context.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_MOIRE_EPS 1.0e-12
#define STIM_MOIRE_VDSP_MIN_LEN 64U

typedef struct SimStimulusMoireState {
    SimStimulusMoireConfig config;
    char                   symbolic[192];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta;
    double* vdsp_cos;
    double* vdsp_sin;
    double* vdsp_accum_re;
    double* vdsp_accum_im;
    size_t  vdsp_capacity;
#endif
} SimStimulusMoireState;

static void moire_normalize(SimStimulusMoireConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->wavenumber_a)) {
        config->wavenumber_a = 1.0;
    }
    if (!isfinite(config->wavenumber_b)) {
        config->wavenumber_b = config->wavenumber_a;
    }
    if (!isfinite(config->omega_a)) {
        config->omega_a = 0.0;
    }
    if (!isfinite(config->omega_b)) {
        config->omega_b = 0.0;
    }
    if (!isfinite(config->phase_a)) {
        config->phase_a = 0.0;
    }
    if (!isfinite(config->phase_b)) {
        config->phase_b = 0.0;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }

    if (!isfinite(config->k1x)) {
        config->k1x = 0.0;
    }
    if (!isfinite(config->k1y)) {
        config->k1y = 0.0;
    }
    if (!isfinite(config->k2x)) {
        config->k2x = 0.0;
    }
    if (!isfinite(config->k2y)) {
        config->k2y = 0.0;
    }

    if (!config->use_wavevectors &&
        (fabs(config->k1x) > STIM_MOIRE_EPS || fabs(config->k1y) > STIM_MOIRE_EPS ||
         fabs(config->k2x) > STIM_MOIRE_EPS || fabs(config->k2y) > STIM_MOIRE_EPS)) {
        config->use_wavevectors = true;
    }

    if (config->use_wavevectors) {
        if (fabs(config->k1x) <= STIM_MOIRE_EPS && fabs(config->k1y) <= STIM_MOIRE_EPS) {
            config->k1x = config->wavenumber_a;
            config->k1y = 0.0;
        }
        if (fabs(config->k2x) <= STIM_MOIRE_EPS && fabs(config->k2y) <= STIM_MOIRE_EPS) {
            config->k2x = config->wavenumber_b;
            config->k2y = 0.0;
        }
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void moire_refresh_symbolic(SimStimulusMoireState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }
    const SimStimulusMoireConfig* cfg = &state->config;
    if (cfg->use_wavevectors) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "moire A=%.3g k1=(%.3g,%.3g) k2=(%.3g,%.3g)",
                        cfg->amplitude,
                        cfg->k1x,
                        cfg->k1y,
                        cfg->k2x,
                        cfg->k2y);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "moire A=%.3g k1=%.3g k2=%.3g",
                        cfg->amplitude,
                        cfg->wavenumber_a,
                        cfg->wavenumber_b);
    }
#else
    (void) state;
#endif
}

static const char* moire_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusMoireState* state = (const SimStimulusMoireState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void moire_eval_1d(const SimStimulusMoireConfig* cfg,
                          double                        u,
                          double                        t,
                          double*                       out_re,
                          double*                       out_im) {
    double theta_a = cfg->wavenumber_a * u - cfg->omega_a * t + cfg->phase_a;
    double theta_b = cfg->wavenumber_b * u - cfg->omega_b * t + cfg->phase_b;
    *out_re        = 0.5 * (cos(theta_a) + cos(theta_b));
    *out_im        = 0.5 * (sin(theta_a) + sin(theta_b));
}

static void moire_eval_wavevectors(const SimStimulusMoireConfig* cfg,
                                   double                        x,
                                   double                        y,
                                   double                        t,
                                   double*                       out_re,
                                   double*                       out_im) {
    double theta_a = cfg->k1x * x + cfg->k1y * y - cfg->omega_a * t + cfg->phase_a;
    double theta_b = cfg->k2x * x + cfg->k2y * y - cfg->omega_b * t + cfg->phase_b;
    *out_re        = 0.5 * (cos(theta_a) + cos(theta_b));
    *out_im        = 0.5 * (sin(theta_a) + sin(theta_b));
}

#if defined(SIM_HAVE_VDSP)
static bool moire_vdsp_ensure_buffers(SimStimulusMoireState* state, size_t width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 5U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = width;
    state->vdsp_theta    = block;
    state->vdsp_cos      = block + width;
    state->vdsp_sin      = block + width * 2U;
    state->vdsp_accum_re = block + width * 3U;
    state->vdsp_accum_im = block + width * 4U;
    return true;
}

static bool moire_linear_map(const SimStimulusMoireConfig* cfg, double* out_u_x, double* out_u_y) {
    if (cfg == NULL || out_u_x == NULL || out_u_y == NULL) {
        return false;
    }

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                *out_u_x = 0.0;
                *out_u_y = 1.0;
            } else {
                *out_u_x = 1.0;
                *out_u_y = 0.0;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE:
            *out_u_x = cos(cfg->coord.angle);
            *out_u_y = sin(cfg->coord.angle);
            return true;
        case SIM_STIMULUS_COORD_SEPARABLE:
            *out_u_x = 1.0;
            *out_u_y = 0.0;
            return true;
        default:
            break;
    }

    return false;
}

static bool moire_vdsp_eval_row(SimStimulusMoireState* state,
                                size_t                 width,
                                double                 theta_a_start,
                                double                 theta_a_step,
                                double                 theta_b_start,
                                double                 theta_b_step) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (!moire_vdsp_ensure_buffers(state, width)) {
        return false;
    }
    if (!isfinite(theta_a_start) || !isfinite(theta_a_step) || !isfinite(theta_b_start) ||
        !isfinite(theta_b_step)) {
        return false;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    const double      half       = 0.5;

    vDSP_vrampD(&theta_a_start, &theta_a_step, state->vdsp_theta, 1, len);
    vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
    vDSP_vsmulD(state->vdsp_cos, 1, &half, state->vdsp_accum_re, 1, len);
    vDSP_vsmulD(state->vdsp_sin, 1, &half, state->vdsp_accum_im, 1, len);

    vDSP_vrampD(&theta_b_start, &theta_b_step, state->vdsp_theta, 1, len);
    vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
    vDSP_vsmaD(state->vdsp_cos, 1, &half, state->vdsp_accum_re, 1, state->vdsp_accum_re, 1, len);
    vDSP_vsmaD(state->vdsp_sin, 1, &half, state->vdsp_accum_im, 1, state->vdsp_accum_im, 1, len);
    return true;
}

static bool moire_try_vdsp_rows(SimStimulusMoireState* state,
                                const SimField*        field,
                                bool                   is_complex,
                                double*                dst_real,
                                SimComplexDouble*      dst_complex,
                                size_t                 count,
                                double                 scale,
                                double                 t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_MOIRE_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!moire_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusMoireConfig* cfg            = &state->config;
    bool                          use_wavevector = cfg->use_wavevectors;
    bool   separable = (!use_wavevector && cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double u_x       = 0.0;
    double u_y       = 0.0;

    if (!use_wavevector && !separable && !moire_linear_map(cfg, &u_x, &u_y)) {
        return false;
    }

    double x0              = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0              = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx              = cfg->coord.spacing_x;
    double dy              = cfg->coord.spacing_y;
    double output_re_scale = scale * cfg->amplitude;
    double output_im_scale = 0.0;

    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            sin_r = sin(cfg->rotation);
            cos_r = cos(cfg->rotation);
        }
        output_re_scale = scale * cfg->amplitude * cos_r;
        output_im_scale = scale * cfg->amplitude * sin_r;
    }

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(output_re_scale) || !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len = (vDSP_Length) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        bool   ok       = false;

        if (use_wavevector) {
            double theta_a_start =
                cfg->k1x * x0 + cfg->k1y * sample_y - cfg->omega_a * t + cfg->phase_a;
            double theta_a_step = cfg->k1x * dx;
            double theta_b_start =
                cfg->k2x * x0 + cfg->k2y * sample_y - cfg->omega_b * t + cfg->phase_b;
            double theta_b_step = cfg->k2x * dx;
            ok                  = moire_vdsp_eval_row(
                state, width, theta_a_start, theta_a_step, theta_b_start, theta_b_step);
        } else if (separable) {
            double theta_a_start = cfg->wavenumber_a * x0 - cfg->omega_a * t + cfg->phase_a;
            double theta_a_step  = cfg->wavenumber_a * dx;
            double theta_b_start = cfg->wavenumber_b * x0 - cfg->omega_b * t + cfg->phase_b;
            double theta_b_step  = cfg->wavenumber_b * dx;
            ok                   = moire_vdsp_eval_row(
                state, width, theta_a_start, theta_a_step, theta_b_start, theta_b_step);
            if (ok) {
                double y_re = 0.0;
                double y_im = 0.0;
                moire_eval_1d(cfg, sample_y, t, &y_re, &y_im);
                if (!isfinite(y_re) || !isfinite(y_im)) {
                    return false;
                }
                if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    sim_accel_add_scalar_real(state->vdsp_accum_re, width, y_re);
                    sim_accel_add_scalar_real(state->vdsp_accum_im, width, y_im);
                } else {
                    memmove(state->vdsp_cos, state->vdsp_accum_re, width * sizeof(double));
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
                    vDSP_vsmaD(state->vdsp_cos,
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
            double u_start       = u_x * x0 + u_y * sample_y;
            double u_step        = u_x * dx;
            double theta_a_start = cfg->wavenumber_a * u_start - cfg->omega_a * t + cfg->phase_a;
            double theta_a_step  = cfg->wavenumber_a * u_step;
            double theta_b_start = cfg->wavenumber_b * u_start - cfg->omega_b * t + cfg->phase_b;
            double theta_b_step  = cfg->wavenumber_b * u_step;
            ok                   = moire_vdsp_eval_row(
                state, width, theta_a_start, theta_a_step, theta_b_start, theta_b_step);
        }

        if (!ok) {
            return false;
        }

        if (!is_complex) {
            sim_accel_copy_scale_real(
                state->vdsp_accum_re, dst_real + row * width, width, output_re_scale, true);
        } else {
            SimComplexDouble* row_ptr = dst_complex + row * width;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;
            vDSP_vsmaD(state->vdsp_accum_re, 1, &output_re_scale, row_re, 2, row_re, 2, len);
            if (output_im_scale != 0.0) {
                double neg_output_im_scale = -output_im_scale;
                vDSP_vsmaD(
                    state->vdsp_accum_im, 1, &neg_output_im_scale, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_accum_re, 1, &output_im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_accum_im, 1, &output_re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static void moire_destroy(void* state_ptr) {
    SimStimulusMoireState* state = (SimStimulusMoireState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block    = NULL;
        state->vdsp_capacity = 0U;
    }
#endif
    free(state);
}

static SimResult moire_step(void*               state_ptr,
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

    SimStimulusMoireState* state = (SimStimulusMoireState*) state_ptr;
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

    if (count == 0U || state->config.amplitude == 0.0) {
        return SIM_RESULT_OK;
    }

    double scale          = state->config.scale_by_dt ? dt_sub : 1.0;
    double t              = sim_context_time(context) + state->config.time_offset;
    bool   use_wavevector = state->config.use_wavevectors;
    bool separable = (!use_wavevector && state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

#if defined(SIM_HAVE_VDSP)
    if (moire_try_vdsp_rows(state,
                            field,
                            is_complex,
                            (double*) sim_field_data(field),
                            sim_field_complex_data(field),
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
                moire_eval_wavevectors(&state->config, sample_x, sample_y, t, &re, &im);
            } else if (separable) {
                double rx = 0.0;
                double ix = 0.0;
                double ry = 0.0;
                double iy = 0.0;
                moire_eval_1d(&state->config, sample_x, t, &rx, &ix);
                moire_eval_1d(&state->config, sample_y, t, &ry, &iy);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = rx + ry;
                    im = ix + iy;
                } else {
                    re = rx * ry - ix * iy;
                    im = rx * iy + ix * ry;
                }
            } else {
                double u = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                moire_eval_1d(&state->config, u, t, &re, &im);
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
                moire_eval_wavevectors(&state->config, sample_x, sample_y, t, &re, &im);
            } else if (separable) {
                double rx = 0.0;
                double ix = 0.0;
                double ry = 0.0;
                double iy = 0.0;
                moire_eval_1d(&state->config, sample_x, t, &rx, &ix);
                moire_eval_1d(&state->config, sample_y, t, &ry, &iy);
                if (state->config.coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                    re = rx + ry;
                    im = ix + iy;
                } else {
                    re = rx * ry - ix * iy;
                    im = rx * iy + ix * ry;
                }
            } else {
                double u = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                moire_eval_1d(&state->config, u, t, &re, &im);
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

SimResult sim_add_stimulus_moire_operator(struct SimContext*            context,
                                          const SimStimulusMoireConfig* config,
                                          size_t*                       out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusMoireConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    moire_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "stimulus_moire", (config != NULL), (config != NULL) ? config->scale_by_dt : true);

    SimStimulusMoireState* state = (SimStimulusMoireState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    moire_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_moire");

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
    info.abstract_id       = "stimulus_moire";
    sim_operator_info_set_schema_identity(&info, "stimulus_moire");
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
                                .fn                = moire_step,
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
                                .symbolic      = moire_symbolic,
                                .destroy       = moire_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        moire_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_moire_config(struct SimContext*      context,
                                    size_t                  operator_index,
                                    SimStimulusMoireConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusMoireState* state = (SimStimulusMoireState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_moire_update(struct SimContext*            context,
                                    size_t                        operator_index,
                                    const SimStimulusMoireConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusMoireState* state = (SimStimulusMoireState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusMoireConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    moire_normalize(&local);
    state->config = local;
    moire_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
