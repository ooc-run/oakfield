#include "oakfield/operators/stimulus/wave_modes.h"

#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"
#include "oakfield/sim_context.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(SIM_HAVE_VDSP)
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define STIM_WAVE_MODES_MIN_EXTENT 1.0e-6
#define STIM_WAVE_MODES_DEFAULT_EXTENT 2.0
#define STIM_WAVE_MODES_DEFAULT_SPEED 1.0
#define STIM_WAVE_MODES_MAX_MODE 64U
#define STIM_WAVE_MODES_VDSP_MIN_LEN 64U

typedef struct SimStimulusWaveModesState {
    SimStimulusWaveModesConfig config;
    char                       symbolic[224];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_theta_u;
    double* vdsp_theta_v;
    double* vdsp_value_u;
    double* vdsp_value_v;
    size_t  vdsp_capacity;
#endif
} SimStimulusWaveModesState;

static void wave_modes_normalize(SimStimulusWaveModesConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (config->mode_u == 0U) {
        config->mode_u = 2U;
    }
    if (config->mode_v == 0U) {
        config->mode_v = 3U;
    }
    if (config->mode_u > STIM_WAVE_MODES_MAX_MODE) {
        config->mode_u = STIM_WAVE_MODES_MAX_MODE;
    }
    if (config->mode_v > STIM_WAVE_MODES_MAX_MODE) {
        config->mode_v = STIM_WAVE_MODES_MAX_MODE;
    }
    if (!isfinite(config->extent_u) || fabs(config->extent_u) < STIM_WAVE_MODES_MIN_EXTENT) {
        config->extent_u = STIM_WAVE_MODES_DEFAULT_EXTENT;
    }
    if (!isfinite(config->extent_v) || fabs(config->extent_v) < STIM_WAVE_MODES_MIN_EXTENT) {
        config->extent_v = STIM_WAVE_MODES_DEFAULT_EXTENT;
    }
    config->extent_u = fabs(config->extent_u);
    config->extent_v = fabs(config->extent_v);

    if (!isfinite(config->wave_speed) || fabs(config->wave_speed) < STIM_WAVE_MODES_MIN_EXTENT) {
        config->wave_speed = STIM_WAVE_MODES_DEFAULT_SPEED;
    }
    config->wave_speed = fabs(config->wave_speed);

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

static double wave_modes_modal_omega(const SimStimulusWaveModesConfig* cfg, size_t rank) {
    double mu = (double) cfg->mode_u / cfg->extent_u;

    if (rank <= 1U) {
        return cfg->wave_speed * M_PI * mu;
    }

    double mv = (double) cfg->mode_v / cfg->extent_v;
    return cfg->wave_speed * M_PI * hypot(mu, mv);
}

static void wave_modes_refresh_symbolic(SimStimulusWaveModesState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusWaveModesConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "wave_modes A=%.3g m=%u n=%u L=(%.3g,%.3g) c=%.3g",
                    cfg->amplitude,
                    cfg->mode_u,
                    cfg->mode_v,
                    cfg->extent_u,
                    cfg->extent_v,
                    cfg->wave_speed);
#else
    (void) state;
#endif
}

static const char* wave_modes_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusWaveModesState* state = (const SimStimulusWaveModesState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void wave_modes_map_coord(const SimStimulusWaveModesConfig* cfg,
                                 double                            x,
                                 double                            y,
                                 double                            t,
                                 double*                           out_u,
                                 double*                           out_v) {
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

static double wave_modes_dirichlet_mode(unsigned int mode, double coord, double extent) {
    return sin(M_PI * (double) mode * (coord / extent + 0.5));
}

static void wave_modes_eval(const SimStimulusWaveModesConfig* cfg,
                            size_t                            rank,
                            double                            u,
                            double                            v,
                            double                            t,
                            double*                           out_re,
                            double*                           out_im) {
    double spatial = wave_modes_dirichlet_mode(cfg->mode_u, u, cfg->extent_u);
    if (rank > 1U) {
        spatial *= wave_modes_dirichlet_mode(cfg->mode_v, v, cfg->extent_v);
    }

    double carrier = -wave_modes_modal_omega(cfg, rank) * t + cfg->phase;
    *out_re        = spatial * cos(carrier);
    *out_im        = spatial * sin(carrier);
}

static void wave_modes_destroy(void* state_ptr) {
    SimStimulusWaveModesState* state = (SimStimulusWaveModesState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block = NULL;
    }
#endif
    free(state);
}

#if defined(SIM_HAVE_VDSP)
static bool wave_modes_vdsp_ensure_buffers(SimStimulusWaveModesState* state, size_t width) {
    if (state == NULL || width == 0U) {
        return false;
    }
    if (state->vdsp_block != NULL && state->vdsp_capacity >= width) {
        return true;
    }
    if (width > SIZE_MAX / (4U * sizeof(double))) {
        return false;
    }

    double* block = (double*) realloc(state->vdsp_block, width * 4U * sizeof(double));
    if (block == NULL) {
        return false;
    }

    state->vdsp_block    = block;
    state->vdsp_capacity = width;
    state->vdsp_theta_u  = block;
    state->vdsp_theta_v  = block + width;
    state->vdsp_value_u  = block + width * 2U;
    state->vdsp_value_v  = block + width * 3U;
    return true;
}

static bool wave_modes_linear_map(const SimStimulusWaveModesConfig* cfg,
                                  double*                           out_u_x,
                                  double*                           out_u_y,
                                  double*                           out_v_x,
                                  double*                           out_v_y) {
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

static bool wave_modes_try_vdsp_linear_rows(SimStimulusWaveModesState* state,
                                            const SimField*            field,
                                            bool                       is_complex,
                                            double*                    dst_real,
                                            SimComplexDouble*          dst_complex,
                                            size_t                     count,
                                            double                     scale,
                                            double                     t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL ||
        !wave_modes_linear_map(&state->config, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_WAVE_MODES_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!wave_modes_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusWaveModesConfig* cfg = &state->config;
    double                            x0  = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double                            y0  = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double                            dx  = cfg->coord.spacing_x;
    double                            dy  = cfg->coord.spacing_y;
    double carrier = -wave_modes_modal_omega(cfg, field->layout.rank) * t + cfg->phase;
    double u_scale = M_PI * (double) cfg->mode_u / cfg->extent_u;
    double v_scale =
        (field->layout.rank > 1U) ? (M_PI * (double) cfg->mode_v / cfg->extent_v) : 0.0;
    double u_phase_bias = 0.5 * M_PI * (double) cfg->mode_u;
    double v_phase_bias = 0.5 * M_PI * (double) cfg->mode_v;
    double u_step       = u_scale * u_x * dx;
    double v_step       = v_scale * v_x * dx;
    double out_re_coef  = scale * cfg->amplitude * cos(carrier);
    double out_im_coef  = 0.0;

    if (is_complex) {
        double phase = carrier + cfg->rotation;
        out_re_coef  = scale * cfg->amplitude * cos(phase);
        out_im_coef  = scale * cfg->amplitude * sin(phase);
    }

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(u_step) ||
        !isfinite(v_step) || !isfinite(out_re_coef) || !isfinite(out_im_coef)) {
        return false;
    }
    if (out_re_coef == 0.0 && (!is_complex || out_im_coef == 0.0)) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;

    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double u_start  = u_scale * (u_x * x0 + u_y * sample_y) + u_phase_bias;
        if (!isfinite(u_start)) {
            return false;
        }

        vDSP_vrampD(&u_start, &u_step, state->vdsp_theta_u, 1, len);
        vvsin(state->vdsp_value_u, state->vdsp_theta_u, &vforce_len);

        if (field->layout.rank > 1U) {
            double v_start = v_scale * (v_x * x0 + v_y * sample_y) + v_phase_bias;
            if (!isfinite(v_start)) {
                return false;
            }
            vDSP_vrampD(&v_start, &v_step, state->vdsp_theta_v, 1, len);
            vvsin(state->vdsp_value_v, state->vdsp_theta_v, &vforce_len);
            vDSP_vmulD(state->vdsp_value_u, 1, state->vdsp_value_v, 1, state->vdsp_value_u, 1, len);
        }

        size_t offset = row * width;
        if (!is_complex) {
            double* row_ptr = dst_real + offset;
            vDSP_vsmaD(state->vdsp_value_u, 1, &out_re_coef, row_ptr, 1, row_ptr, 1, len);
            continue;
        }

        SimComplexDouble* row_ptr = dst_complex + offset;
        double*           row_re  = &row_ptr[0].re;
        double*           row_im  = &row_ptr[0].im;
        vDSP_vsmaD(state->vdsp_value_u, 1, &out_re_coef, row_re, 2, row_re, 2, len);
        vDSP_vsmaD(state->vdsp_value_u, 1, &out_im_coef, row_im, 2, row_im, 2, len);
    }

    return true;
}
#endif

static SimResult wave_modes_step(void*               state_ptr,
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

    SimStimulusWaveModesState* state = (SimStimulusWaveModesState*) state_ptr;
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
    if (wave_modes_try_vdsp_linear_rows(state,
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
            double x  = 0.0;
            double y  = 0.0;
            double u  = 0.0;
            double v  = 0.0;
            double re = 0.0;
            double im = 0.0;

            if (sim_stimulus_coord_xy(&state->config.coord, field, i, &x, &y) != SIM_RESULT_OK) {
                return SIM_RESULT_INVALID_ARGUMENT;
            }

            wave_modes_map_coord(&state->config, x, y, t, &u, &v);
            wave_modes_eval(&state->config, field->layout.rank, u, v, t, &re, &im);
            (void) im;

            re *= state->config.amplitude;
            if (isfinite(re)) {
                dst[i] += scale * re;
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

            wave_modes_map_coord(&state->config, x, y, t, &u, &v);
            wave_modes_eval(&state->config, field->layout.rank, u, v, t, &re, &im);

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

SimResult sim_add_stimulus_wave_modes_operator(struct SimContext*                context,
                                               const SimStimulusWaveModesConfig* config,
                                               size_t*                           out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusWaveModesConfig local = { 0 };
    local.mode_u                     = 2U;
    local.mode_v                     = 3U;
    local.wave_speed                 = STIM_WAVE_MODES_DEFAULT_SPEED;
    if (config != NULL) {
        local = *config;
    }

    wave_modes_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_wave_modes",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusWaveModesState* state = (SimStimulusWaveModesState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    wave_modes_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_wave_modes");

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
    info.abstract_id       = "stimulus_wave_modes";
    sim_operator_info_set_schema_identity(&info, "stimulus_wave_modes");
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
                                .fn                = wave_modes_step,
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
                                .symbolic      = wave_modes_symbolic,
                                .destroy       = wave_modes_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        wave_modes_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_wave_modes_config(struct SimContext*          context,
                                         size_t                      operator_index,
                                         SimStimulusWaveModesConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusWaveModesState* state = (SimStimulusWaveModesState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_wave_modes_update(struct SimContext*                context,
                                         size_t                            operator_index,
                                         const SimStimulusWaveModesConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusWaveModesState* state = (SimStimulusWaveModesState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusWaveModesConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    wave_modes_normalize(&local);
    state->config = local;
    wave_modes_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
