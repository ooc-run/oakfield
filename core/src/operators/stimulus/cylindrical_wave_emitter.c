#include "oakfield/operators/stimulus/cylindrical_wave_emitter.h"

#include "operators/common/operator_utils.h"

#include "sim_accel.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_context.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define STIM_CYLINDRICAL_WAVE_MIN_RADIUS 1.0e-6
#define STIM_CYLINDRICAL_WAVE_DEFAULT_SOFTENING_RADIUS 0.1
#define STIM_CYLINDRICAL_WAVE_DEFAULT_RADIAL_WAVENUMBER 8.0
#define STIM_CYLINDRICAL_WAVE_VDSP_MIN_LEN 64U

typedef struct SimStimulusCylindricalWaveEmitterState {
    SimStimulusCylindricalWaveEmitterConfig config;
    char                                    symbolic[256];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusCylindricalWaveEmitterState;

static void cylindrical_wave_emitter_normalize(SimStimulusCylindricalWaveEmitterConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->radial_wavenumber)) {
        config->radial_wavenumber = STIM_CYLINDRICAL_WAVE_DEFAULT_RADIAL_WAVENUMBER;
    }
    config->radial_wavenumber = fabs(config->radial_wavenumber);

    if (!isfinite(config->attenuation)) {
        config->attenuation = 0.0;
    }
    config->attenuation = fabs(config->attenuation);

    if (!isfinite(config->softening_radius) ||
        config->softening_radius < STIM_CYLINDRICAL_WAVE_MIN_RADIUS) {
        config->softening_radius = STIM_CYLINDRICAL_WAVE_DEFAULT_SOFTENING_RADIUS;
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

static void
cylindrical_wave_emitter_refresh_symbolic(SimStimulusCylindricalWaveEmitterState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    {
        const SimStimulusCylindricalWaveEmitterConfig* cfg = &state->config;
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "cylindrical_wave_emitter A=%.3g k_r=%.3g alpha=%.3g a=%.3g",
                        cfg->amplitude,
                        cfg->radial_wavenumber,
                        cfg->attenuation,
                        cfg->softening_radius);
    }
#else
    (void) state;
#endif
}

static const char* cylindrical_wave_emitter_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusCylindricalWaveEmitterState* state =
        (const SimStimulusCylindricalWaveEmitterState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void cylindrical_wave_emitter_map_coord(const SimStimulusCylindricalWaveEmitterConfig* cfg,
                                               double                                         x,
                                               double                                         y,
                                               double                                         t,
                                               double*                                        out_u,
                                               double* out_v) {
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

static void cylindrical_wave_emitter_eval(const SimStimulusCylindricalWaveEmitterConfig* cfg,
                                          size_t                                         rank,
                                          double                                         u,
                                          double                                         v,
                                          double                                         t,
                                          double*                                        out_re,
                                          double*                                        out_im) {
    double center_u = cfg->center_u + cfg->velocity_u * t;
    double center_v = cfg->center_v + cfg->velocity_v * t;
    double du       = u - center_u;
    double dv       = v - center_v;
    double rho      = (rank > 1U) ? hypot(du, dv) : fabs(du);
    double radius   = hypot(rho, cfg->softening_radius);
    double envelope = exp(-cfg->attenuation * radius) / sqrt(radius);
    double phase    = cfg->radial_wavenumber * radius - cfg->omega * t + cfg->phase;

    *out_re = envelope * cos(phase);
    *out_im = envelope * sin(phase);
}

#if defined(SIM_HAVE_VDSP)
static bool
cylindrical_wave_emitter_vdsp_ensure_buffers(SimStimulusCylindricalWaveEmitterState* state,
                                             size_t                                  width) {
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

static bool cylindrical_wave_emitter_linear_map(const SimStimulusCylindricalWaveEmitterConfig* cfg,
                                                double* out_u_x,
                                                double* out_u_y,
                                                double* out_v_x,
                                                double* out_v_y) {
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

static bool
cylindrical_wave_emitter_try_vdsp_linear_rows(SimStimulusCylindricalWaveEmitterState* state,
                                              const SimField*                         field,
                                              bool                                    is_complex,
                                              double*                                 dst_real,
                                              SimComplexDouble*                       dst_complex,
                                              size_t                                  count,
                                              double                                  scale,
                                              double                                  t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL ||
        !cylindrical_wave_emitter_linear_map(&state->config, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_CYLINDRICAL_WAVE_VDSP_MIN_LEN ||
        width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!cylindrical_wave_emitter_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusCylindricalWaveEmitterConfig* cfg        = &state->config;
    bool                                           rank_is_2d = (field->layout.rank > 1U);
    double x0                  = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double y0                  = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx                  = cfg->coord.spacing_x;
    double dy                  = cfg->coord.spacing_y;
    double center_u            = cfg->center_u + cfg->velocity_u * t;
    double center_v            = cfg->center_v + cfg->velocity_v * t;
    double u_step              = u_x * dx;
    double v_step              = v_x * dx;
    double softening_radius_sq = cfg->softening_radius * cfg->softening_radius;
    double phase_scale         = cfg->radial_wavenumber;
    double phase_bias          = -cfg->omega * t + cfg->phase;
    double envelope_scale      = -cfg->attenuation;
    double output_scale        = scale * cfg->amplitude;
    double output_re_scale     = output_scale;
    double output_im_scale     = 0.0;

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

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(center_u) ||
        !isfinite(center_v) || !isfinite(u_step) || !isfinite(v_step) ||
        !isfinite(softening_radius_sq) || !isfinite(phase_scale) || !isfinite(phase_bias) ||
        !isfinite(envelope_scale) || !isfinite(output_re_scale) || !isfinite(output_im_scale)) {
        return false;
    }
    if (output_re_scale == 0.0 && (!is_complex || output_im_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double u_start  = u_x * x0 + u_y * sample_y - center_u;
        double v_start  = v_x * x0 + v_y * sample_y - center_v;

        if (!isfinite(sample_y) || !isfinite(u_start) || !isfinite(v_start)) {
            return false;
        }

        vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
        if (rank_is_2d) {
            vDSP_vrampD(&v_start, &v_step, state->vdsp_v, 1, len);
            vDSP_vsqD(state->vdsp_u, 1, state->vdsp_value, 1, len);
            vDSP_vsqD(state->vdsp_v, 1, state->vdsp_work, 1, len);
            vDSP_vaddD(state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
            vvsqrt(state->vdsp_work, state->vdsp_value, &vforce_len);
        } else {
            vDSP_vabsD(state->vdsp_u, 1, state->vdsp_work, 1, len);
        }

        vDSP_vsqD(state->vdsp_work, 1, state->vdsp_value, 1, len);
        if (softening_radius_sq != 0.0) {
            sim_accel_add_scalar_real(state->vdsp_value, width, softening_radius_sq);
        }
        vvsqrt(state->vdsp_u, state->vdsp_value, &vforce_len);

        vDSP_vsmulD(state->vdsp_u, 1, &phase_scale, state->vdsp_phase, 1, len);
        if (phase_bias != 0.0) {
            sim_accel_add_scalar_real(state->vdsp_phase, width, phase_bias);
        }

        vDSP_vsmulD(state->vdsp_u, 1, &envelope_scale, state->vdsp_value, 1, len);
        vvexp(state->vdsp_value, state->vdsp_value, &vforce_len);
        vvsqrt(state->vdsp_work, state->vdsp_u, &vforce_len);
        for (size_t col = 0U; col < width; ++col) {
            state->vdsp_value[col] /= state->vdsp_work[col];
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

static void cylindrical_wave_emitter_destroy(void* state_ptr) {
    SimStimulusCylindricalWaveEmitterState* state =
        (SimStimulusCylindricalWaveEmitterState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block    = NULL;
        state->vdsp_capacity = 0U;
    }
#endif
    free(state);
}

static SimResult cylindrical_wave_emitter_step(void*               state_ptr,
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

    SimStimulusCylindricalWaveEmitterState* state =
        (SimStimulusCylindricalWaveEmitterState*) state_ptr;
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

    double scale = state->config.scale_by_dt ? dt_sub : 1.0;
    double t     = sim_context_time(context) + state->config.time_offset;

#if defined(SIM_HAVE_VDSP)
    if (cylindrical_wave_emitter_try_vdsp_linear_rows(state,
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

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            cylindrical_wave_emitter_map_coord(&state->config, x, y, t, &u, &v);
            cylindrical_wave_emitter_eval(&state->config, field->layout.rank, u, v, t, &re, &im);
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

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            cylindrical_wave_emitter_map_coord(&state->config, x, y, t, &u, &v);
            cylindrical_wave_emitter_eval(&state->config, field->layout.rank, u, v, t, &re, &im);

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

    return SIM_RESULT_OK;
}

SimResult sim_add_stimulus_cylindrical_wave_emitter_operator(
    struct SimContext*                             context,
    const SimStimulusCylindricalWaveEmitterConfig* config,
    size_t*                                        out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    {
        SimStimulusCylindricalWaveEmitterConfig local = { 0 };
        SimStimulusCylindricalWaveEmitterState* state;
        SimOperatorInfo                         info;
        SimOperatorConfig                       op_config;
        SimSplitPort                            port;
        SimSplitAccess                          access;
        SimSplitSubstep                         substep;
        SimSplitDescriptor                      desc = { 0 };
        char                                    name[SIM_OPERATOR_NAME_MAX + 1U];
        SimResult                               result;

        local.radial_wavenumber = STIM_CYLINDRICAL_WAVE_DEFAULT_RADIAL_WAVENUMBER;
        local.softening_radius  = STIM_CYLINDRICAL_WAVE_DEFAULT_SOFTENING_RADIUS;
        if (config != NULL) {
            local = *config;
        }

        cylindrical_wave_emitter_normalize(&local);
        local.scale_by_dt =
            sim_operator_resolve_scale_by_dt(context,
                                             "stimulus_cylindrical_wave_emitter",
                                             (config != NULL),
                                             (config != NULL) ? config->scale_by_dt : true);

        state = (SimStimulusCylindricalWaveEmitterState*) calloc(1U, sizeof(*state));
        if (state == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }

        state->config = local;
        cylindrical_wave_emitter_refresh_symbolic(state);

        sim_operator_make_unique_name(name, sizeof(name), "stimulus_cylindrical_wave_emitter");

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
        info.abstract_id       = "stimulus_cylindrical_wave_emitter";
        sim_operator_info_set_schema_identity(&info, "stimulus_cylindrical_wave_emitter");
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
                                     .fn                = cylindrical_wave_emitter_step,
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
                                        .symbolic      = cylindrical_wave_emitter_symbolic,
                                        .destroy       = cylindrical_wave_emitter_destroy,
                                        .info          = info,
                                        .config        = op_config,
                                        .scratch       = { 0U, 0U } };

        result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
        if (result != SIM_RESULT_OK) {
            cylindrical_wave_emitter_destroy(state);
        }
        return result;
    }
}

SimResult
sim_stimulus_cylindrical_wave_emitter_config(struct SimContext* context,
                                             size_t             operator_index,
                                             SimStimulusCylindricalWaveEmitterConfig* out_config) {
    SimOperator*                            op;
    SimStimulusCylindricalWaveEmitterState* state;

    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimStimulusCylindricalWaveEmitterState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_cylindrical_wave_emitter_update(
    struct SimContext*                             context,
    size_t                                         operator_index,
    const SimStimulusCylindricalWaveEmitterConfig* config) {
    SimOperator*                            op;
    SimStimulusCylindricalWaveEmitterState* state;
    SimStimulusCylindricalWaveEmitterConfig local;

    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    state = (SimStimulusCylindricalWaveEmitterState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    local = *config;
    cylindrical_wave_emitter_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "stimulus_cylindrical_wave_emitter", true, config->scale_by_dt);

    state->config = local;
    cylindrical_wave_emitter_refresh_symbolic(state);
    return SIM_RESULT_OK;
}
