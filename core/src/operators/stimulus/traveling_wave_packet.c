#include "oakfield/operators/stimulus/traveling_wave_packet.h"

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

#define STIM_TRAVELING_WAVE_PACKET_MIN_SIGMA 1.0e-6
#define STIM_TRAVELING_WAVE_PACKET_DEFAULT_SIGMA 0.75
#define STIM_TRAVELING_WAVE_PACKET_VDSP_MIN_LEN 64U

typedef struct SimStimulusTravelingWavePacketState {
    SimStimulusTravelingWavePacketConfig config;
    char                                 symbolic[224];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusTravelingWavePacketState;

static void traveling_wave_packet_normalize(SimStimulusTravelingWavePacketConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->sigma_u) ||
        fabs(config->sigma_u) < STIM_TRAVELING_WAVE_PACKET_MIN_SIGMA) {
        config->sigma_u = STIM_TRAVELING_WAVE_PACKET_DEFAULT_SIGMA;
    }
    if (!isfinite(config->sigma_v) ||
        fabs(config->sigma_v) < STIM_TRAVELING_WAVE_PACKET_MIN_SIGMA) {
        config->sigma_v = config->sigma_u;
    }
    config->sigma_u = fabs(config->sigma_u);
    config->sigma_v = fabs(config->sigma_v);

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
    if (!isfinite(config->carrier_u)) {
        config->carrier_u = 0.0;
    }
    if (!isfinite(config->carrier_v)) {
        config->carrier_v = 0.0;
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

static void traveling_wave_packet_refresh_symbolic(SimStimulusTravelingWavePacketState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusTravelingWavePacketConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "traveling_wave_packet A=%.3g sigma=(%.3g,%.3g) k=(%.3g,%.3g)",
                    cfg->amplitude,
                    cfg->sigma_u,
                    cfg->sigma_v,
                    cfg->carrier_u,
                    cfg->carrier_v);
#else
    (void) state;
#endif
}

static const char* traveling_wave_packet_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusTravelingWavePacketState* state =
        (const SimStimulusTravelingWavePacketState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void traveling_wave_packet_map_sample_coord(const SimStimulusTravelingWavePacketConfig* cfg,
                                                   double  sample_x,
                                                   double  sample_y,
                                                   double  t,
                                                   double* out_u,
                                                   double* out_v) {
    const SimStimulusCoordConfig* coord = &cfg->coord;

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
            double dx = sample_x - coord->center_x;
            double dy = sample_y - coord->center_y;

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

static void traveling_wave_packet_map_coord(const SimStimulusTravelingWavePacketConfig* cfg,
                                            double                                      x,
                                            double                                      y,
                                            double                                      t,
                                            double*                                     out_u,
                                            double*                                     out_v) {
    double sample_x = x;
    double sample_y = y;

    sim_stimulus_coord_sample_xy(&cfg->coord, x, y, t, &sample_x, &sample_y);
    traveling_wave_packet_map_sample_coord(cfg, sample_x, sample_y, t, out_u, out_v);
}

static void traveling_wave_packet_eval(const SimStimulusTravelingWavePacketConfig* cfg,
                                       double                                      u,
                                       double                                      v,
                                       double                                      t,
                                       double*                                     out_re,
                                       double*                                     out_im) {
    double theta = cfg->orientation + cfg->orientation_rate * t;
    double s     = sin(theta);
    double c     = cos(theta);

    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du       = u - center_u;
    double dv       = v - center_v;

    double ur       = du * c + dv * s;
    double vr       = -du * s + dv * c;
    double exponent = -0.5 * ((ur * ur) / (cfg->sigma_u * cfg->sigma_u) +
                              (vr * vr) / (cfg->sigma_v * cfg->sigma_v));
    double envelope = exp(exponent);
    double carrier  = cfg->carrier_u * ur + cfg->carrier_v * vr - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(carrier);
    *out_im = envelope * sin(carrier);
}

#if defined(SIM_HAVE_VDSP)
static bool traveling_wave_packet_vdsp_ensure_buffers(SimStimulusTravelingWavePacketState* state,
                                                      size_t                               width) {
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

static bool traveling_wave_packet_linear_map(const SimStimulusTravelingWavePacketConfig* cfg,
                                             double*                                     out_u_x,
                                             double*                                     out_u_y,
                                             double*                                     out_v_x,
                                             double*                                     out_v_y) {
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

static bool traveling_wave_packet_try_vdsp_linear_rows(SimStimulusTravelingWavePacketState* state,
                                                       const SimField*                      field,
                                                       bool              is_complex,
                                                       double*           dst_real,
                                                       SimComplexDouble* dst_complex,
                                                       size_t            count,
                                                       double            scale,
                                                       double            t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL ||
        !traveling_wave_packet_linear_map(&state->config, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_TRAVELING_WAVE_PACKET_VDSP_MIN_LEN ||
        width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!traveling_wave_packet_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusTravelingWavePacketConfig* cfg = &state->config;
    double x0               = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0               = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx               = cfg->coord.spacing_x;
    double dy               = cfg->coord.spacing_y;
    double theta            = cfg->orientation + cfg->orientation_rate * t;
    double s                = sin(theta);
    double c                = cos(theta);
    double center_u         = cfg->center_u + cfg->velocity_u * t;
    double center_v         = cfg->center_v + cfg->velocity_v * t;
    double ur_bias          = -center_u * c - center_v * s;
    double vr_bias          = center_u * s - center_v * c;
    double ur_x             = c * u_x + s * v_x;
    double ur_y             = c * u_y + s * v_y;
    double vr_x             = -s * u_x + c * v_x;
    double vr_y             = -s * u_y + c * v_y;
    double ur_step          = ur_x * dx;
    double vr_step          = vr_x * dx;
    double envelope_scale_u = -0.5 / (cfg->sigma_u * cfg->sigma_u);
    double envelope_scale_v = -0.5 / (cfg->sigma_v * cfg->sigma_v);
    double phase_x          = cfg->carrier_u * ur_x + cfg->carrier_v * vr_x;
    double phase_y          = cfg->carrier_u * ur_y + cfg->carrier_v * vr_y;
    double phase_step       = phase_x * dx;
    double phase_bias =
        cfg->carrier_u * ur_bias + cfg->carrier_v * vr_bias - cfg->omega * t + cfg->phase;
    double output_scale    = scale * cfg->amplitude;
    double output_re_scale = output_scale;
    double output_im_scale = 0.0;

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
        !isfinite(envelope_scale_u) || !isfinite(envelope_scale_v) || !isfinite(phase_x) ||
        !isfinite(phase_y) || !isfinite(phase_step) || !isfinite(phase_bias) ||
        !isfinite(output_re_scale) || !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y    = y0 + (double) row * dy;
        double ur_start    = ur_x * x0 + ur_y * sample_y + ur_bias;
        double vr_start    = vr_x * x0 + vr_y * sample_y + vr_bias;
        double phase_start = phase_x * x0 + phase_y * sample_y + phase_bias;

        if (!isfinite(sample_y) || !isfinite(ur_start) || !isfinite(vr_start) ||
            !isfinite(phase_start)) {
            return false;
        }

        vDSP_vrampD(&ur_start, &ur_step, state->vdsp_u, 1, len);
        vDSP_vrampD(&vr_start, &vr_step, state->vdsp_v, 1, len);
        vDSP_vrampD(&phase_start, &phase_step, state->vdsp_phase, 1, len);

        vDSP_vsqD(state->vdsp_u, 1, state->vdsp_value, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_value, width, envelope_scale_u);
        vDSP_vsqD(state->vdsp_v, 1, state->vdsp_work, 1, len);
        sim_accel_scale_inplace_real(state->vdsp_work, width, envelope_scale_v);
        vDSP_vaddD(state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
        vvexp(state->vdsp_u, state->vdsp_value, &vforce_len);

        if (!is_complex) {
            vvcos(state->vdsp_value, state->vdsp_phase, &vforce_len);
            vDSP_vmulD(state->vdsp_value, 1, state->vdsp_u, 1, state->vdsp_value, 1, len);
            sim_accel_copy_scale_real(
                state->vdsp_value, dst_real + row * width, width, output_scale, true);
        } else {
            SimComplexDouble* row_ptr = dst_complex + row * width;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vvsincos(state->vdsp_work, state->vdsp_value, state->vdsp_phase, &vforce_len);
            vDSP_vmulD(state->vdsp_value, 1, state->vdsp_u, 1, state->vdsp_value, 1, len);
            vDSP_vmulD(state->vdsp_work, 1, state->vdsp_u, 1, state->vdsp_work, 1, len);
            vDSP_vsmaD(state->vdsp_value, 1, &output_re_scale, row_re, 2, row_re, 2, len);
            if (output_im_scale != 0.0) {
                double neg_output_im_scale = -output_im_scale;
                vDSP_vsmaD(state->vdsp_work, 1, &neg_output_im_scale, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_value, 1, &output_im_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_work, 1, &output_re_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static void traveling_wave_packet_destroy(void* state_ptr) {
    SimStimulusTravelingWavePacketState* state = (SimStimulusTravelingWavePacketState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block    = NULL;
        state->vdsp_capacity = 0U;
    }
#endif
    free(state);
}

static SimResult traveling_wave_packet_step(void*               state_ptr,
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

    SimStimulusTravelingWavePacketState* state = (SimStimulusTravelingWavePacketState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
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

    double        scale = state->config.scale_by_dt ? dt_sub : 1.0;
    double        t     = sim_context_time(context) + state->config.time_offset;
    SimFieldPatch patch = { 0 };

#if defined(SIM_HAVE_VDSP)
    if (traveling_wave_packet_try_vdsp_linear_rows(state,
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

    if (sim_field_patch_full(sim_field_width(field), sim_field_height(field), &patch) !=
        SIM_RESULT_OK) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!is_complex) {
        double* dst = (double*) sim_field_data(field);
        if (dst == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }

        for (size_t row_offset = 0U; row_offset < patch.height; ++row_offset) {
            SimStimulusCoordRow row       = { 0 };
            size_t              dst_index = (patch.y0 + row_offset) * patch.field_width + patch.x0;
            double              sample_x  = 0.0;
            double              sample_y  = 0.0;

            if (sim_stimulus_coord_patch_row(&state->config.coord, &patch, row_offset, t, &row) !=
                SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sample_x = row.sample_x;
            sample_y = row.sample_y;

            for (size_t col = 0U; col < row.width; ++col) {
                double u  = 0.0;
                double v  = 0.0;
                double re = 0.0;
                double im = 0.0;

                traveling_wave_packet_map_sample_coord(
                    &state->config, sample_x, sample_y, t, &u, &v);
                traveling_wave_packet_eval(&state->config, u, v, t, &re, &im);
                (void) im;

                re *= state->config.amplitude;
                if (isfinite(re)) {
                    dst[dst_index] += scale * re;
                }

                dst_index += 1U;
                sample_x += row.sample_x_step;
                sample_y += row.sample_y_step;
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

        for (size_t row_offset = 0U; row_offset < patch.height; ++row_offset) {
            SimStimulusCoordRow row       = { 0 };
            size_t              dst_index = (patch.y0 + row_offset) * patch.field_width + patch.x0;
            double              sample_x  = 0.0;
            double              sample_y  = 0.0;

            if (sim_stimulus_coord_patch_row(&state->config.coord, &patch, row_offset, t, &row) !=
                SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            sample_x = row.sample_x;
            sample_y = row.sample_y;

            for (size_t col = 0U; col < row.width; ++col) {
                double u      = 0.0;
                double v      = 0.0;
                double re     = 0.0;
                double im     = 0.0;
                double out_re = 0.0;
                double out_im = 0.0;

                traveling_wave_packet_map_sample_coord(
                    &state->config, sample_x, sample_y, t, &u, &v);
                traveling_wave_packet_eval(&state->config, u, v, t, &re, &im);

                re *= state->config.amplitude;
                im *= state->config.amplitude;
                out_re = re * cos_r - im * sin_r;
                out_im = re * sin_r + im * cos_r;
                if (isfinite(out_re) && isfinite(out_im)) {
                    dst[dst_index].re += scale * out_re;
                    dst[dst_index].im += scale * out_im;
                }

                dst_index += 1U;
                sample_x += row.sample_x_step;
                sample_y += row.sample_y_step;
            }
        }
    }

    return SIM_RESULT_OK;
}

SimResult
sim_add_stimulus_traveling_wave_packet_operator(struct SimContext*                          context,
                                                const SimStimulusTravelingWavePacketConfig* config,
                                                size_t* out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusTravelingWavePacketConfig local = { 0 };
    local.sigma_u                              = STIM_TRAVELING_WAVE_PACKET_DEFAULT_SIGMA;
    local.sigma_v                              = STIM_TRAVELING_WAVE_PACKET_DEFAULT_SIGMA;
    if (config != NULL) {
        local = *config;
    }

    traveling_wave_packet_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_traveling_wave_packet",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusTravelingWavePacketState* state =
        (SimStimulusTravelingWavePacketState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    traveling_wave_packet_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_traveling_wave_packet");

    SimOperatorInfo info   = sim_operator_info_defaults();
    info.category          = SIM_OPERATOR_CATEGORY_POTENTIAL;
    info.warp_level        = SIM_WARP_LEVEL_NONE;
    info.is_noise          = false;
    info.is_spectral       = false;
    info.is_local          = true;
    info.is_nonlocal       = false;
    info.is_linear         = false;
    info.is_warp           = false;
    info.is_differentiable = true;
    info.preserves_real    = true;
    info.preferred_dt      = 0.0;
    info.abstract_id       = "stimulus_traveling_wave_packet";
    sim_operator_info_set_schema_identity(&info, "stimulus_traveling_wave_packet");
    info.algebraic_flags       = SIM_OPERATOR_ALG_NONE;
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

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = traveling_wave_packet_step,
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
                                .symbolic      = traveling_wave_packet_symbolic,
                                .destroy       = traveling_wave_packet_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        traveling_wave_packet_destroy(state);
    }
    return result;
}

SimResult
sim_stimulus_traveling_wave_packet_config(struct SimContext*                    context,
                                          size_t                                operator_index,
                                          SimStimulusTravelingWavePacketConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusTravelingWavePacketState* state =
        (SimStimulusTravelingWavePacketState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult
sim_stimulus_traveling_wave_packet_update(struct SimContext* context,
                                          size_t             operator_index,
                                          const SimStimulusTravelingWavePacketConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusTravelingWavePacketState* state =
        (SimStimulusTravelingWavePacketState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusTravelingWavePacketConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    traveling_wave_packet_normalize(&local);
    state->config = local;
    traveling_wave_packet_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
