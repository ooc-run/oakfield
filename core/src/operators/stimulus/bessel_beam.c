#include "oakfield/operators/stimulus/bessel_beam.h"

#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "sim_accel.h"
#include "oakfield/math/bessel.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_BESSEL_MIN_SCALE 1.0e-6
#define STIM_BESSEL_DEFAULT_SCALE 1.0
#define STIM_BESSEL_DEFAULT_RADIAL_WAVENUMBER 8.0
#define STIM_BESSEL_MAX_ORDER 16
#define STIM_BESSEL_VDSP_MIN_LEN 64U

typedef struct SimStimulusBesselBeamState {
    SimStimulusBesselBeamConfig config;
    char                        symbolic[256];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusBesselBeamState;

static void bessel_beam_normalize(SimStimulusBesselBeamConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->radial_wavenumber)) {
        config->radial_wavenumber = STIM_BESSEL_DEFAULT_RADIAL_WAVENUMBER;
    }
    if (!isfinite(config->scale_u) || fabs(config->scale_u) < STIM_BESSEL_MIN_SCALE) {
        config->scale_u = STIM_BESSEL_DEFAULT_SCALE;
    }
    if (!isfinite(config->scale_v) || fabs(config->scale_v) < STIM_BESSEL_MIN_SCALE) {
        config->scale_v = config->scale_u;
    }
    config->scale_u = fabs(config->scale_u);
    config->scale_v = fabs(config->scale_v);

    if (config->order > STIM_BESSEL_MAX_ORDER) {
        config->order = STIM_BESSEL_MAX_ORDER;
    } else if (config->order < -STIM_BESSEL_MAX_ORDER) {
        config->order = -STIM_BESSEL_MAX_ORDER;
    }

    if (!isfinite(config->center_u)) {
        config->center_u = 0.0;
    }
    if (!isfinite(config->center_v)) {
        config->center_v = 0.0;
    }
    if (!isfinite(config->velocity_u)) {
        config->velocity_u = 0.0;
    }
    if (!isfinite(config->velocity_v)) {
        config->velocity_v = 0.0;
    }
    if (!isfinite(config->orientation)) {
        config->orientation = 0.0;
    }
    if (!isfinite(config->orientation_rate)) {
        config->orientation_rate = 0.0;
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

    sim_stimulus_coord_normalize(&config->coord);
}

static void bessel_beam_refresh_symbolic(SimStimulusBesselBeamState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    {
        const SimStimulusBesselBeamConfig* cfg = &state->config;
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "bessel_beam A=%.3g n=%d k_r=%.3g scale=(%.3g,%.3g)",
                        cfg->amplitude,
                        cfg->order,
                        cfg->radial_wavenumber,
                        cfg->scale_u,
                        cfg->scale_v);
    }
#else
    (void) state;
#endif
}

static const char* bessel_beam_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusBesselBeamState* state = (const SimStimulusBesselBeamState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void bessel_beam_map_coord(const SimStimulusBesselBeamConfig* cfg,
                                  double                             x,
                                  double                             y,
                                  double                             t,
                                  double*                            out_u,
                                  double*                            out_v) {
    const SimStimulusCoordConfig* coord    = &cfg->coord;
    double                        sample_x = x;
    double                        sample_y = y;
    sim_stimulus_coord_sample_xy(coord, x, y, t, &sample_x, &sample_y);

    switch (coord->mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (coord->axis == SIM_STIMULUS_AXIS_Y) {
                *out_u = sample_y;
                *out_v = sample_x;
            } else {
                *out_u = sample_x;
                *out_v = sample_y;
            }
            return;
        case SIM_STIMULUS_COORD_ANGLE: {
            double s = sin(coord->angle);
            double c = cos(coord->angle);
            *out_u   = sample_x * c + sample_y * s;
            *out_v   = -sample_x * s + sample_y * c;
            return;
        }
        case SIM_STIMULUS_COORD_RADIAL:
        case SIM_STIMULUS_COORD_POLAR:
        case SIM_STIMULUS_COORD_AZIMUTH:
        case SIM_STIMULUS_COORD_ELLIPTIC:
        case SIM_STIMULUS_COORD_SPIRAL: {
            double dx = 0.0;
            double dy = 0.0;
            sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);

            if (coord->mode == SIM_STIMULUS_COORD_SPIRAL) {
                double r  = hypot(dx, dy);
                double th = atan2(dy, dx);
                *out_u = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
                         coord->spiral_angular_velocity * t;
                *out_v = th;
            } else if (coord->mode == SIM_STIMULUS_COORD_POLAR) {
                *out_u = hypot(dx, dy);
                *out_v = atan2(dy, dx);
            } else if (coord->mode == SIM_STIMULUS_COORD_AZIMUTH) {
                *out_u = atan2(dy, dx);
                *out_v = hypot(dx, dy);
            } else if (coord->mode == SIM_STIMULUS_COORD_ELLIPTIC) {
                sim_stimulus_coord_elliptic_local(coord, dx, dy, out_u, out_v);
            } else {
                *out_u = dx;
                *out_v = dy;
            }
            return;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
        default:
            *out_u = sample_x;
            *out_v = sample_y;
            return;
    }
}

static void bessel_beam_eval_uv(const SimStimulusBesselBeamConfig* cfg,
                                double                             u,
                                double                             v,
                                double                             t,
                                double*                            out_re,
                                double*                            out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s     = sin(theta);
    double c     = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du       = u - center_u;
    double dv       = v - center_v;

    double ur       = du * c + dv * s;
    double vr       = -du * s + dv * c;
    double xu       = ur / cfg->scale_u;
    double yv       = vr / cfg->scale_v;
    double rho      = hypot(xu, yv);
    double azimuth  = atan2(yv, xu);
    double envelope = sim_bessel_jn_f64(cfg->order, cfg->radial_wavenumber * rho);
    double phase    = (double) cfg->order * azimuth - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(phase);
    *out_im = envelope * sin(phase);
}

#if defined(SIM_HAVE_VDSP)
static bool bessel_beam_vdsp_ensure_buffers(SimStimulusBesselBeamState* state, size_t width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_capacity >= width && state->vdsp_block != NULL) {
        return true;
    }

    if (width > SIZE_MAX / (5U * sizeof(double))) {
        return false;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 5U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = width;
    state->vdsp_u        = block;
    state->vdsp_v        = block + width;
    state->vdsp_phase    = block + width * 2U;
    state->vdsp_value    = block + width * 3U;
    state->vdsp_work     = block + width * 4U;
    return true;
}

static bool bessel_beam_linear_map(const SimStimulusBesselBeamConfig* cfg,
                                   double*                            out_u_x,
                                   double*                            out_u_y,
                                   double*                            out_v_x,
                                   double*                            out_v_y) {
    if (cfg == NULL || out_u_x == NULL || out_u_y == NULL || out_v_x == NULL || out_v_y == NULL) {
        return false;
    }

    switch (cfg->coord.mode) {
        case SIM_STIMULUS_COORD_AXIS:
            if (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) {
                *out_u_x = 0.0;
                *out_u_y = 1.0;
                *out_v_x = 1.0;
                *out_v_y = 0.0;
            } else {
                *out_u_x = 1.0;
                *out_u_y = 0.0;
                *out_v_x = 0.0;
                *out_v_y = 1.0;
            }
            return true;
        case SIM_STIMULUS_COORD_ANGLE: {
            double s = sin(cfg->coord.angle);
            double c = cos(cfg->coord.angle);
            *out_u_x = c;
            *out_u_y = s;
            *out_v_x = -s;
            *out_v_y = c;
            return true;
        }
        case SIM_STIMULUS_COORD_SEPARABLE:
            *out_u_x = 1.0;
            *out_u_y = 0.0;
            *out_v_x = 0.0;
            *out_v_y = 1.0;
            return true;
        default:
            break;
    }

    return false;
}

static bool bessel_beam_try_vdsp_linear_rows(SimStimulusBesselBeamState* state,
                                             const SimField*             field,
                                             bool                        is_complex,
                                             double*                     dst_real,
                                             SimComplexDouble*           dst_complex,
                                             size_t                      count,
                                             double                      scale,
                                             double                      t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL ||
        !bessel_beam_linear_map(&state->config, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_BESSEL_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!bessel_beam_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusBesselBeamConfig* cfg      = &state->config;
    double                             x0       = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double                             y0       = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double                             dx       = cfg->coord.spacing_x;
    double                             dy       = cfg->coord.spacing_y;
    double                             theta    = cfg->orientation + cfg->orientation_rate * t;
    double                             s        = sin(theta);
    double                             c        = cos(theta);
    double                             center_u = cfg->center_u + cfg->velocity_u * t;
    double                             center_v = cfg->center_v + cfg->velocity_v * t;
    double                             ur_bias  = -center_u * c - center_v * s;
    double                             vr_bias  = center_u * s - center_v * c;
    double                             ur_x     = c * u_x + s * v_x;
    double                             ur_y     = c * u_y + s * v_y;
    double                             vr_x     = -s * u_x + c * v_x;
    double                             vr_y     = -s * u_y + c * v_y;
    double                             ur_step  = ur_x * dx;
    double                             vr_step  = vr_x * dx;
    double                             inv_scale_u       = 1.0 / cfg->scale_u;
    double                             inv_scale_v       = 1.0 / cfg->scale_v;
    double                             phase_scale       = (double) cfg->order;
    double                             phase_bias        = -cfg->omega * t + cfg->phase;
    double                             radial_wavenumber = cfg->radial_wavenumber;
    double                             output_scale      = scale * cfg->amplitude;
    double                             output_re_scale   = output_scale;
    double                             output_im_scale   = 0.0;

    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            sin_r = sin(cfg->rotation);
            cos_r = cos(cfg->rotation);
        }
        output_re_scale = output_scale * cos_r;
        output_im_scale = output_scale * sin_r;
    }

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(theta) ||
        !isfinite(c) || !isfinite(s) || !isfinite(center_u) || !isfinite(center_v) ||
        !isfinite(ur_bias) || !isfinite(vr_bias) || !isfinite(ur_x) || !isfinite(ur_y) ||
        !isfinite(vr_x) || !isfinite(vr_y) || !isfinite(ur_step) || !isfinite(vr_step) ||
        !isfinite(inv_scale_u) || !isfinite(inv_scale_v) || !isfinite(phase_scale) ||
        !isfinite(phase_bias) || !isfinite(radial_wavenumber) || !isfinite(output_re_scale) ||
        !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double ur_start = ur_x * x0 + ur_y * sample_y + ur_bias;
        double vr_start = vr_x * x0 + vr_y * sample_y + vr_bias;

        if (!isfinite(sample_y) || !isfinite(ur_start) || !isfinite(vr_start)) {
            return false;
        }

        vDSP_vrampD(&ur_start, &ur_step, state->vdsp_u, 1, len);
        vDSP_vrampD(&vr_start, &vr_step, state->vdsp_v, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_u, width, inv_scale_u);
        sim_accel_scale_inplace_real(state->vdsp_v, width, inv_scale_v);

        vvatan2(state->vdsp_phase, state->vdsp_v, state->vdsp_u, &vforce_len);
        sim_accel_scale_inplace_real(state->vdsp_phase, width, phase_scale);
        sim_accel_add_scalar_real(state->vdsp_phase, width, phase_bias);

        vDSP_vsqD(state->vdsp_u, 1, state->vdsp_value, 1, len);
        vDSP_vsqD(state->vdsp_v, 1, state->vdsp_work, 1, len);
        vDSP_vaddD(state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
        vvsqrt(state->vdsp_value, state->vdsp_value, &vforce_len);

        for (size_t i = 0U; i < width; ++i) {
            state->vdsp_value[i] =
                sim_bessel_jn_f64(cfg->order, radial_wavenumber * state->vdsp_value[i]);
        }

        if (!is_complex) {
            vvcos(state->vdsp_work, state->vdsp_phase, &vforce_len);
            vDSP_vmulD(state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_value, 1, len);
            sim_accel_copy_scale_real(
                state->vdsp_value, dst_real + row * width, width, output_scale, true);
        } else {
            SimComplexDouble* row_ptr = dst_complex + row * width;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vvsincos(state->vdsp_work, state->vdsp_u, state->vdsp_phase, &vforce_len);
            vDSP_vmulD(state->vdsp_u, 1, state->vdsp_value, 1, state->vdsp_u, 1, len);
            vDSP_vmulD(state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_work, 1, len);
            vDSP_vsmaD(state->vdsp_u, 1, &output_re_scale, row_re, 2, row_re, 2, len);
            if (output_im_scale != 0.0) {
                double neg_output_im_scale = -output_im_scale;
                vDSP_vsmaD(state->vdsp_work, 1, &neg_output_im_scale, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_u, 1, &output_im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_work, 1, &output_re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static void bessel_beam_destroy(void* state_ptr) {
    SimStimulusBesselBeamState* state = (SimStimulusBesselBeamState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block    = NULL;
        state->vdsp_capacity = 0U;
    }
#endif
    free(state);
}

static SimResult bessel_beam_step(void*               state_ptr,
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

    {
        SimStimulusBesselBeamState* state = (SimStimulusBesselBeamState*) state_ptr;
        SimField*                   field;
        bool                        is_complex;
        size_t                      count;
        double                      scale;
        double                      t;

        if (state == NULL || context == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        field = sim_context_field(context, state->config.field_index);
        if (field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        is_complex = sim_field_is_complex(field);
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

        scale = state->config.scale_by_dt ? dt_sub : 1.0;
        t     = sim_context_time(context) + state->config.time_offset;

#if defined(SIM_HAVE_VDSP)
        if (bessel_beam_try_vdsp_linear_rows(state,
                                             field,
                                             is_complex,
                                             sim_field_real_data(field),
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
                double x  = 0.0;
                double y  = 0.0;
                double u  = 0.0;
                double v  = 0.0;
                double re = 0.0;
                double im = 0.0;

                if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) !=
                    SIM_RESULT_OK) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }

                bessel_beam_map_coord(&state->config, x, y, t, &u, &v);
                bessel_beam_eval_uv(&state->config, u, v, t, &re, &im);
                (void) im;

                re *= state->config.amplitude;
                if (isfinite(re)) {
                    dst[i] += scale * re;
                }
            }
        } else {
            SimComplexDouble* dst   = sim_field_complex_data(field);
            double            sin_r = 0.0;
            double            cos_r = 1.0;

            if (dst == NULL) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            if (state->config.rotation != 0.0) {
                sin_r = sin(state->config.rotation);
                cos_r = cos(state->config.rotation);
            }

            for (size_t i = 0U; i < count; ++i) {
                double x      = 0.0;
                double y      = 0.0;
                double u      = 0.0;
                double v      = 0.0;
                double re     = 0.0;
                double im     = 0.0;
                double out_re = 0.0;
                double out_im = 0.0;

                if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) !=
                    SIM_RESULT_OK) {
                    return SIM_RESULT_INVALID_ARGUMENT;
                }

                bessel_beam_map_coord(&state->config, x, y, t, &u, &v);
                bessel_beam_eval_uv(&state->config, u, v, t, &re, &im);

                re *= state->config.amplitude;
                im *= state->config.amplitude;
                out_re = re * cos_r - im * sin_r;
                out_im = re * sin_r + im * cos_r;
                if (isfinite(out_re) && isfinite(out_im)) {
                    dst[i].re += scale * out_re;
                    dst[i].im += scale * out_im;
                }
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_bessel_beam_operator(struct SimContext*                 context,
                                                const SimStimulusBesselBeamConfig* config,
                                                size_t*                            out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    {
        SimStimulusBesselBeamConfig local = { 0 };
        SimStimulusBesselBeamState* state;
        SimOperatorInfo             info;
        SimOperatorConfig           op_config;
        SimSplitPort                port;
        SimSplitAccess              access;
        SimSplitSubstep             substep;
        SimSplitDescriptor          desc = { 0 };
        char                        name[SIM_OPERATOR_NAME_MAX + 1U];
        SimResult                   result;

        if (config != NULL) {
            local = *config;
        }

        bessel_beam_normalize(&local);
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context,
                                             "stimulus_bessel_beam",
                                             (config != NULL),
                                             (config != NULL) ? config->scale_by_dt : true);

        state = (SimStimulusBesselBeamState*) calloc(1U, sizeof(*state));
        if (state == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        state->config = local;
        bessel_beam_refresh_symbolic(state);

        sim_operator_make_unique_name(name, sizeof(name), "stimulus_bessel_beam");

        info                   = sim_operator_info_defaults();
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
        info.abstract_id       = "stimulus_bessel_beam";
        sim_operator_info_set_schema_identity(&info, "stimulus_bessel_beam");
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

        op_config = sim_operator_config_defaults();

        port    = (SimSplitPort){ .context_field_index = state->config.field_index,
                                  .require_complex     = needs_complex };
        access  = (SimSplitAccess){ .port = 0U, .mode = SIM_ACCESS_RW };
        substep = (SimSplitSubstep){ .name              = NULL,
                                     .fn                = bessel_beam_step,
                                     .accesses          = &access,
                                     .access_count      = 1U,
                                     .dt_scale          = 1.0,
                                     .barrier_after     = false,
                                     .error_measure     = NULL,
                                     .required_features = 0U };
        desc    = (SimSplitDescriptor){ .name          = name,
                                        .ports         = &port,
                                        .port_count    = 1U,
                                        .substeps      = &substep,
                                        .substep_count = 1U,
                                        .state         = state,
                                        .symbolic      = bessel_beam_symbolic,
                                        .destroy       = bessel_beam_destroy,
                                        .info          = info,
                                        .config        = op_config,
                                        .scratch       = { 0U, 0U } };

        result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
        if (result != SIM_RESULT_OK) {
            bessel_beam_destroy(state);
        }
        return result;
    }
}

SimResult sim_stimulus_bessel_beam_config(struct SimContext*           context,
                                          size_t                       operator_index,
                                          SimStimulusBesselBeamConfig* out_config) {
    SimOperator*                op;
    SimStimulusBesselBeamState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimStimulusBesselBeamState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_bessel_beam_update(struct SimContext*                 context,
                                          size_t                             operator_index,
                                          const SimStimulusBesselBeamConfig* config) {
    SimOperator*                op;
    SimStimulusBesselBeamState* state;
    SimStimulusBesselBeamConfig local;

    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimStimulusBesselBeamState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local = *config;
    bessel_beam_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "stimulus_bessel_beam", true, config->scale_by_dt);

    state->config = local;
    bessel_beam_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
