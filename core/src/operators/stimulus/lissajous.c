#include "oakfield/operators/stimulus/lissajous.h"
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

#define STIM_LISSAJOUS_MIN_WIDTH 1.0e-6
#define STIM_LISSAJOUS_VDSP_MIN_LEN 64U

typedef struct SimStimulusLissajousState {
    SimStimulusLissajousConfig config;
    char                       symbolic[224];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta;
    double* vdsp_sin;
    double* vdsp_band;
    double* vdsp_cos;
    size_t  vdsp_capacity;
#endif
} SimStimulusLissajousState;

static void lissajous_normalize(SimStimulusLissajousConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (!isfinite(config->wavenumber_x)) {
        config->wavenumber_x = 3.0;
    }
    if (!isfinite(config->wavenumber_y)) {
        config->wavenumber_y = 2.0;
    }
    if (!isfinite(config->omega_x)) {
        config->omega_x = 0.0;
    }
    if (!isfinite(config->omega_y)) {
        config->omega_y = 0.0;
    }
    if (!isfinite(config->phase_x)) {
        config->phase_x = 0.0;
    }
    if (!isfinite(config->phase_y)) {
        config->phase_y = 0.0;
    }
    if (!isfinite(config->coupling)) {
        config->coupling = 1.0;
    }
    if (!isfinite(config->bias)) {
        config->bias = 0.0;
    }
    if (!isfinite(config->line_width)) {
        config->line_width = 0.25;
    }
    config->line_width = fabs(config->line_width);
    if (config->line_width < STIM_LISSAJOUS_MIN_WIDTH) {
        config->line_width = 0.25;
    }
    if (!isfinite(config->time_offset)) {
        config->time_offset = 0.0;
    }
    if (!isfinite(config->rotation)) {
        config->rotation = 0.0;
    }

    sim_stimulus_coord_normalize(&config->coord);
}

static void lissajous_refresh_symbolic(SimStimulusLissajousState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusLissajousConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "lissajous A=%.3g k=(%.3g,%.3g) w=%.3g c=%.3g",
                    cfg->amplitude,
                    cfg->wavenumber_x,
                    cfg->wavenumber_y,
                    cfg->line_width,
                    cfg->coupling);
#else
    (void) state;
#endif
}

static const char* lissajous_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusLissajousState* state = (const SimStimulusLissajousState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void lissajous_eval(const SimStimulusLissajousConfig* cfg,
                           double                            ux,
                           double                            uy,
                           double                            t,
                           double*                           out_re,
                           double*                           out_im) {
    double theta_x = cfg->wavenumber_x * ux - cfg->omega_x * t + cfg->phase_x;
    double theta_y = cfg->wavenumber_y * uy - cfg->omega_y * t + cfg->phase_y;
    double delta   = sin(theta_x) - cfg->coupling * sin(theta_y) - cfg->bias;
    double width   = cfg->line_width;
    double band    = exp(-0.5 * (delta * delta) / (width * width));
    double carrier = 0.5 * (theta_x + theta_y);

    *out_re = band * cos(carrier);
    *out_im = band * sin(carrier);
}

static void lissajous_destroy(void* state_ptr) {
    SimStimulusLissajousState* state = (SimStimulusLissajousState*) state_ptr;
    if (state != NULL) {
#if defined(SIM_HAVE_VDSP)
        free(state->vdsp_block);
#endif
    }
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool lissajous_vdsp_ensure_buffers(SimStimulusLissajousState* state, size_t width) {
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
    state->vdsp_sin      = resized + width;
    state->vdsp_band     = resized + width * 2U;
    state->vdsp_cos      = resized + width * 3U;
    return true;
}

static bool
lissajous_linear_map(const SimStimulusLissajousConfig* cfg, double* out_u_x, double* out_u_y) {
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
        default:
            break;
    }

    return false;
}

static bool lissajous_vdsp_eval_row(SimStimulusLissajousState* state,
                                    size_t                     width,
                                    double                     theta_x_start,
                                    double                     theta_x_step,
                                    double                     theta_y_start,
                                    double                     theta_y_step,
                                    double                     coupling,
                                    double                     bias,
                                    double                     line_width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (!lissajous_vdsp_ensure_buffers(state, width)) {
        return false;
    }
    if (!isfinite(theta_x_start) || !isfinite(theta_x_step) || !isfinite(theta_y_start) ||
        !isfinite(theta_y_step) || !isfinite(coupling) || !isfinite(bias) ||
        !isfinite(line_width)) {
        return false;
    }

    const vDSP_Length len               = (vDSP_Length) width;
    const int         vforce_len        = (int) width;
    const double      neg_half_width_sq = -0.5 / (line_width * line_width);
    const double      neg_coupling      = -coupling;
    const double      neg_bias          = -bias;

    vDSP_vrampD(&theta_x_start, &theta_x_step, state->vdsp_theta, 1, len);
    vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
    sim_accel_copy_scale_real(state->vdsp_sin, state->vdsp_band, width, 1.0, false);

    vDSP_vrampD(&theta_y_start, &theta_y_step, state->vdsp_theta, 1, len);
    vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
    vDSP_vsmaD(state->vdsp_sin, 1, &neg_coupling, state->vdsp_band, 1, state->vdsp_band, 1, len);
    if (neg_bias != 0.0) {
        sim_accel_add_scalar_real(state->vdsp_band, width, neg_bias);
    }

    vDSP_vsqD(state->vdsp_band, 1, state->vdsp_band, 1, len);
    sim_accel_scale_inplace_real(state->vdsp_band, width, neg_half_width_sq);
    vvexp(state->vdsp_band, state->vdsp_band, &vforce_len);

    {
        double carrier_start = 0.5 * (theta_x_start + theta_y_start);
        double carrier_step  = 0.5 * (theta_x_step + theta_y_step);
        vDSP_vrampD(&carrier_start, &carrier_step, state->vdsp_theta, 1, len);
    }
    vvsincos(state->vdsp_sin, state->vdsp_cos, state->vdsp_theta, &vforce_len);
    vDSP_vmulD(state->vdsp_band, 1, state->vdsp_cos, 1, state->vdsp_cos, 1, len);
    vDSP_vmulD(state->vdsp_band, 1, state->vdsp_sin, 1, state->vdsp_sin, 1, len);
    return true;
}

static bool lissajous_try_vdsp_linear_rows(SimStimulusLissajousState* state,
                                           const SimField*            field,
                                           bool                       is_complex,
                                           double*                    dst_real,
                                           SimComplexDouble*          dst_complex,
                                           size_t                     count,
                                           double                     scale,
                                           double                     t) {
    if (state == NULL || field == NULL) {
        return false;
    }
    if ((!is_complex && dst_real == NULL) || (is_complex && dst_complex == NULL)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t rank = field->layout.rank;
    if (rank == 0U || rank > 2U) {
        return false;
    }

    const SimStimulusLissajousConfig* cfg       = &state->config;
    bool                              separable = (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE);
    double                            u_x       = 0.0;
    double                            u_y       = 0.0;

    if (!separable && !lissajous_linear_map(cfg, &u_x, &u_y)) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_LISSAJOUS_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!lissajous_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    double line_width = cfg->line_width;
    if (!isfinite(line_width) || line_width < STIM_LISSAJOUS_MIN_WIDTH) {
        line_width = 0.25;
    }

    double sample_x0  = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double sample_y0  = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double dx         = cfg->coord.spacing_x;
    double dy         = cfg->coord.spacing_y;
    double real_scale = cfg->amplitude * scale;
    double imag_scale = 0.0;

    if (is_complex) {
        double sin_r = 0.0;
        double cos_r = 1.0;
        if (cfg->rotation != 0.0) {
            sin_r = sin(cfg->rotation);
            cos_r = cos(cfg->rotation);
        }
        real_scale = cfg->amplitude * scale * cos_r;
        imag_scale = cfg->amplitude * scale * sin_r;
    }

    if (!isfinite(sample_x0) || !isfinite(sample_y0) || !isfinite(dx) || !isfinite(dy) ||
        !isfinite(real_scale) || !isfinite(imag_scale)) {
        return false;
    }
    if (real_scale == 0.0 && (!is_complex || imag_scale == 0.0)) {
        return true;
    }

    const vDSP_Length len = (vDSP_Length) width;
    for (size_t row = 0U; row < height; ++row) {
        double sample_y      = sample_y0 + (double) row * dy;
        double theta_x_start = 0.0;
        double theta_x_step  = 0.0;
        double theta_y_start = 0.0;
        double theta_y_step  = 0.0;

        if (separable) {
            theta_x_start = cfg->wavenumber_x * sample_x0 - cfg->omega_x * t + cfg->phase_x;
            theta_x_step  = cfg->wavenumber_x * dx;
            theta_y_start = cfg->wavenumber_y * sample_y - cfg->omega_y * t + cfg->phase_y;
            theta_y_step  = 0.0;
        } else {
            double u_start = u_x * sample_x0 + u_y * sample_y;
            double u_step  = u_x * dx;
            theta_x_start  = cfg->wavenumber_x * u_start - cfg->omega_x * t + cfg->phase_x;
            theta_x_step   = cfg->wavenumber_x * u_step;
            theta_y_start  = cfg->wavenumber_y * u_start - cfg->omega_y * t + cfg->phase_y;
            theta_y_step   = cfg->wavenumber_y * u_step;
        }

        if (!lissajous_vdsp_eval_row(state,
                                     width,
                                     theta_x_start,
                                     theta_x_step,
                                     theta_y_start,
                                     theta_y_step,
                                     cfg->coupling,
                                     cfg->bias,
                                     line_width)) {
            return false;
        }

        size_t offset = row * width;
        if (!is_complex) {
            double* row_ptr = dst_real + offset;
            vDSP_vsmaD(state->vdsp_cos, 1, &real_scale, row_ptr, 1, row_ptr, 1, len);
        } else {
            SimComplexDouble* row_ptr = dst_complex + offset;
            double*           row_re  = &row_ptr[0].re;
            double*           row_im  = &row_ptr[0].im;

            vDSP_vsmaD(state->vdsp_cos, 1, &real_scale, row_re, 2, row_re, 2, len);
            if (imag_scale != 0.0) {
                double neg_imag = -imag_scale;
                vDSP_vsmaD(state->vdsp_sin, 1, &neg_imag, row_re, 2, row_re, 2, len);
                vDSP_vsmaD(state->vdsp_cos, 1, &imag_scale, row_im, 2, row_im, 2, len);
            }
            vDSP_vsmaD(state->vdsp_sin, 1, &real_scale, row_im, 2, row_im, 2, len);
        }
    }

    return true;
}
#endif

static SimResult lissajous_step(void*               state_ptr,
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

    SimStimulusLissajousState* state = (SimStimulusLissajousState*) state_ptr;
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

    double scale     = state->config.scale_by_dt ? dt_sub : 1.0;
    double t         = sim_context_time(context) + state->config.time_offset;
    bool   separable = (state->config.coord.mode == SIM_STIMULUS_COORD_SEPARABLE);

#if defined(SIM_HAVE_VDSP)
    if (lissajous_try_vdsp_linear_rows(state,
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

            double ux = 0.0;
            double uy = 0.0;
            if (separable) {
                ux = sample_x;
                uy = sample_y;
            } else {
                double u = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                ux       = u;
                uy       = u;
            }

            double re = 0.0;
            double im = 0.0;
            lissajous_eval(&state->config, ux, uy, t, &re, &im);
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

            double ux = 0.0;
            double uy = 0.0;
            if (separable) {
                ux = sample_x;
                uy = sample_y;
            } else {
                double u = sim_stimulus_coord_u(&state->config.coord, x, y, t);
                ux       = u;
                uy       = u;
            }

            double re = 0.0;
            double im = 0.0;
            lissajous_eval(&state->config, ux, uy, t, &re, &im);

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

SimResult sim_add_stimulus_lissajous_operator(struct SimContext*                context,
                                              const SimStimulusLissajousConfig* config,
                                              size_t*                           out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusLissajousConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    lissajous_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_lissajous",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusLissajousState* state = (SimStimulusLissajousState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    lissajous_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_lissajous");

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
    info.abstract_id       = "stimulus_lissajous";
    sim_operator_info_set_schema_identity(&info, "stimulus_lissajous");
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
                                .fn                = lissajous_step,
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
                                .symbolic      = lissajous_symbolic,
                                .destroy       = lissajous_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        lissajous_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_lissajous_config(struct SimContext*          context,
                                        size_t                      operator_index,
                                        SimStimulusLissajousConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLissajousState* state = (SimStimulusLissajousState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_lissajous_update(struct SimContext*                context,
                                        size_t                            operator_index,
                                        const SimStimulusLissajousConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLissajousState* state = (SimStimulusLissajousState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusLissajousConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    lissajous_normalize(&local);
    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, sim_operator_schema_key_or(op, "stimulus_lissajous"), true, local.scale_by_dt);
    state->config = local;
    lissajous_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
