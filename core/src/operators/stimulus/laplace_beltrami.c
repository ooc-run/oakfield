#include "oakfield/operators/stimulus/laplace_beltrami.h"

#include "operators/common/operator_utils.h"

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

#define STIM_LB_MIN_EXTENT 1.0e-6
#define STIM_LB_DEFAULT_EXTENT 2.0
#define STIM_LB_DEFAULT_MODE_U 2
#define STIM_LB_DEFAULT_MODE_V 3
#define STIM_LB_MAX_MODE 64
#define STIM_LB_VDSP_MIN_LEN 64U

typedef struct SimStimulusLaplaceBeltramiState {
    SimStimulusLaplaceBeltramiConfig config;
    char                             symbolic[256];
#if defined(SIM_HAVE_VDSP)
    double* vdsp_block;
    double* vdsp_u;
    double* vdsp_v;
    double* vdsp_phase;
    double* vdsp_value;
    double* vdsp_work;
    size_t  vdsp_capacity;
#endif
} SimStimulusLaplaceBeltramiState;

static const char* laplace_beltrami_manifold_key(SimStimulusLaplaceBeltramiManifold manifold) {
    switch (manifold) {
        case SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS:
            return "flat_torus";
        case SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER:
            return "cylinder";
        case SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE:
        default:
            return "rectangle";
    }
}

static int laplace_beltrami_clamp_mode(int value, int fallback, bool allow_zero) {
    int mode = value;
    if (mode > STIM_LB_MAX_MODE) {
        mode = STIM_LB_MAX_MODE;
    } else if (mode < -STIM_LB_MAX_MODE) {
        mode = -STIM_LB_MAX_MODE;
    }

    if (!allow_zero && mode == 0) {
        mode = fallback;
    }
    return mode;
}

static void laplace_beltrami_normalize(SimStimulusLaplaceBeltramiConfig* config) {
    if (config == NULL) {
        return;
    }

    if (!isfinite(config->amplitude)) {
        config->amplitude = 0.0;
    }
    if (config->manifold < SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE ||
        config->manifold > SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER) {
        config->manifold = SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE;
    }

    config->mode_u =
        laplace_beltrami_clamp_mode(config->mode_u,
                                    STIM_LB_DEFAULT_MODE_U,
                                    config->manifold != SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE);
    config->mode_v =
        laplace_beltrami_clamp_mode(config->mode_v,
                                    STIM_LB_DEFAULT_MODE_V,
                                    config->manifold == SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS);

    if (config->manifold == SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE) {
        if (config->mode_u < 0) {
            config->mode_u = -config->mode_u;
        }
        if (config->mode_v < 0) {
            config->mode_v = -config->mode_v;
        }
        if (config->mode_u == 0) {
            config->mode_u = STIM_LB_DEFAULT_MODE_U;
        }
        if (config->mode_v == 0) {
            config->mode_v = STIM_LB_DEFAULT_MODE_V;
        }
    } else if (config->manifold == SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER) {
        if (config->mode_v < 0) {
            config->mode_v = -config->mode_v;
        }
        if (config->mode_v == 0) {
            config->mode_v = STIM_LB_DEFAULT_MODE_V;
        }
    }

    if (!isfinite(config->extent_u) || fabs(config->extent_u) < STIM_LB_MIN_EXTENT) {
        config->extent_u = STIM_LB_DEFAULT_EXTENT;
    }
    if (!isfinite(config->extent_v) || fabs(config->extent_v) < STIM_LB_MIN_EXTENT) {
        config->extent_v = config->extent_u;
    }
    config->extent_u = fabs(config->extent_u);
    config->extent_v = fabs(config->extent_v);

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

static void laplace_beltrami_refresh_symbolic(SimStimulusLaplaceBeltramiState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimStimulusLaplaceBeltramiConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "laplace_beltrami A=%.3g M=%s modes=(%d,%d) L=(%.3g,%.3g)",
                    cfg->amplitude,
                    laplace_beltrami_manifold_key(cfg->manifold),
                    cfg->mode_u,
                    cfg->mode_v,
                    cfg->extent_u,
                    cfg->extent_v);
#else
    (void) state;
#endif
}

static const char* laplace_beltrami_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimStimulusLaplaceBeltramiState* state =
        (const SimStimulusLaplaceBeltramiState*) state_ptr;
    return (state != NULL) ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void laplace_beltrami_map_coord(const SimStimulusLaplaceBeltramiConfig* cfg,
                                       double                                  x,
                                       double                                  y,
                                       double                                  t,
                                       double*                                 out_u,
                                       double*                                 out_v) {
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

static double laplace_beltrami_dirichlet_mode(int mode, double coord, double extent) {
    int n = mode;
    if (n < 0) {
        n = -n;
    }
    if (n == 0) {
        n = 1;
    }
    return sin(M_PI * (double) n * (coord / extent + 0.5));
}

static double laplace_beltrami_periodic_phase(int mode, double coord, double extent) {
    return 2.0 * M_PI * (double) mode * coord / extent;
}

static void laplace_beltrami_eval(const SimStimulusLaplaceBeltramiConfig* cfg,
                                  double                                  u,
                                  double                                  v,
                                  double                                  t,
                                  bool                                    include_v,
                                  double*                                 out_re,
                                  double*                                 out_im) {
    double spatial_re = 0.0;
    double spatial_im = 0.0;
    double temporal   = -cfg->omega * t + cfg->phase;

    if (cfg == NULL || out_re == NULL || out_im == NULL) {
        return;
    }

    switch (cfg->manifold) {
        case SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS: {
            double arg = laplace_beltrami_periodic_phase(cfg->mode_u, u, cfg->extent_u);
            if (include_v) {
                arg += laplace_beltrami_periodic_phase(cfg->mode_v, v, cfg->extent_v);
            }
            spatial_re = cos(arg);
            spatial_im = sin(arg);
            break;
        }
        case SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER: {
            double arg = laplace_beltrami_periodic_phase(cfg->mode_u, u, cfg->extent_u);
            double envelope =
                include_v ? laplace_beltrami_dirichlet_mode(cfg->mode_v, v, cfg->extent_v) : 1.0;
            spatial_re = envelope * cos(arg);
            spatial_im = envelope * sin(arg);
            break;
        }
        case SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE:
        default: {
            double value = laplace_beltrami_dirichlet_mode(cfg->mode_u, u, cfg->extent_u);
            if (include_v) {
                value *= laplace_beltrami_dirichlet_mode(cfg->mode_v, v, cfg->extent_v);
            }
            spatial_re = value;
            spatial_im = 0.0;
            break;
        }
    }

    *out_re = spatial_re * cos(temporal) - spatial_im * sin(temporal);
    *out_im = spatial_re * sin(temporal) + spatial_im * cos(temporal);
}

#if defined(SIM_HAVE_VDSP)
static bool laplace_beltrami_vdsp_ensure_buffers(SimStimulusLaplaceBeltramiState* state,
                                                 size_t                           width) {
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

static bool laplace_beltrami_linear_map(const SimStimulusLaplaceBeltramiConfig* cfg,
                                        double*                                 out_u_x,
                                        double*                                 out_u_y,
                                        double*                                 out_v_x,
                                        double*                                 out_v_y) {
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

static bool laplace_beltrami_try_vdsp_rows(SimStimulusLaplaceBeltramiState* state,
                                           const SimField*                  field,
                                           bool                             is_complex,
                                           double*                          dst_real,
                                           SimComplexDouble*                dst_complex,
                                           size_t                           count,
                                           bool                             include_v,
                                           double                           scale,
                                           double                           t) {
    double u_x = 0.0;
    double u_y = 0.0;
    double v_x = 0.0;
    double v_y = 0.0;

    if (state == NULL || field == NULL ||
        !laplace_beltrami_linear_map(&state->config, &u_x, &u_y, &v_x, &v_y)) {
        return false;
    }
    if (!field->layout.contiguous || field->storage != SIM_FIELD_STORAGE_ROW_MAJOR) {
        return false;
    }

    size_t width  = sim_field_width(field);
    size_t height = sim_field_height(field);
    if (width == 0U || height == 0U || width < STIM_LB_VDSP_MIN_LEN || width > count) {
        return false;
    }

    size_t plane = width * height;
    if (plane == 0U || plane != count || width > (size_t) INT_MAX) {
        return false;
    }
    if (!laplace_beltrami_vdsp_ensure_buffers(state, width)) {
        return false;
    }

    const SimStimulusLaplaceBeltramiConfig* cfg = &state->config;
    double                                  x0  = cfg->coord.origin_x - cfg->coord.velocity_x * t;
    double                                  y0  = cfg->coord.origin_y - cfg->coord.velocity_y * t;
    double                                  dx  = cfg->coord.spacing_x;
    double                                  dy  = cfg->coord.spacing_y;
    double                                  u_step          = u_x * dx;
    double                                  v_step          = v_x * dx;
    double                                  amplitude_scale = scale * cfg->amplitude;
    double                                  temporal_real   = -cfg->omega * t + cfg->phase;
    double temporal_complex = temporal_real + (is_complex ? cfg->rotation : 0.0);

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(dx) || !isfinite(dy) || !isfinite(u_step) ||
        !isfinite(v_step) || !isfinite(amplitude_scale) || !isfinite(temporal_real) ||
        !isfinite(temporal_complex)) {
        return false;
    }
    if (amplitude_scale == 0.0) {
        return true;
    }

    const vDSP_Length len        = (vDSP_Length) width;
    const int         vforce_len = (int) width;
    double            one        = 1.0;

    for (size_t row = 0U; row < height; ++row) {
        double sample_y = y0 + (double) row * dy;
        double u_start  = u_x * x0 + u_y * sample_y;
        double v_start  = v_x * x0 + v_y * sample_y;

        if (!isfinite(sample_y) || !isfinite(u_start) || !isfinite(v_start)) {
            return false;
        }

        vDSP_vrampD(&u_start, &u_step, state->vdsp_u, 1, len);
        vDSP_vrampD(&v_start, &v_step, state->vdsp_v, 1, len);

        switch (cfg->manifold) {
            case SIM_STIMULUS_LAPLACE_BELTRAMI_FLAT_TORUS: {
                double phase_u_scale = 2.0 * M_PI * (double) cfg->mode_u / cfg->extent_u;
                double phase_v_scale = 2.0 * M_PI * (double) cfg->mode_v / cfg->extent_v;
                double phase_bias    = temporal_complex;
                if (!isfinite(phase_u_scale) || !isfinite(phase_v_scale) || !isfinite(phase_bias)) {
                    return false;
                }

                vDSP_vsmulD(state->vdsp_u, 1, &phase_u_scale, state->vdsp_phase, 1, len);
                if (include_v) {
                    vDSP_vsmulD(state->vdsp_v, 1, &phase_v_scale, state->vdsp_work, 1, len);
                    vDSP_vaddD(
                        state->vdsp_phase, 1, state->vdsp_work, 1, state->vdsp_phase, 1, len);
                }
                if (phase_bias != 0.0) {
                    sim_accel_add_scalar_real(state->vdsp_phase, width, phase_bias);
                }

                if (!is_complex) {
                    vvcos(state->vdsp_value, state->vdsp_phase, &vforce_len);
                    sim_accel_copy_scale_real(
                        state->vdsp_value, dst_real + row * width, width, amplitude_scale, true);
                } else {
                    SimComplexDouble* row_ptr = dst_complex + row * width;
                    double*           row_re  = &row_ptr[0].re;
                    double*           row_im  = &row_ptr[0].im;

                    vvsincos(state->vdsp_work, state->vdsp_value, state->vdsp_phase, &vforce_len);
                    vDSP_vsmaD(state->vdsp_value, 1, &amplitude_scale, row_re, 2, row_re, 2, len);
                    vDSP_vsmaD(state->vdsp_work, 1, &amplitude_scale, row_im, 2, row_im, 2, len);
                }
                break;
            }
            case SIM_STIMULUS_LAPLACE_BELTRAMI_CYLINDER: {
                double phase_u_scale = 2.0 * M_PI * (double) cfg->mode_u / cfg->extent_u;
                int    n_v           = cfg->mode_v;
                double env_scale;
                double env_bias;
                double phase_bias = temporal_complex;

                if (n_v < 0) {
                    n_v = -n_v;
                }
                if (n_v == 0) {
                    n_v = 1;
                }
                env_scale = M_PI * (double) n_v / cfg->extent_v;
                env_bias  = 0.5 * M_PI * (double) n_v;

                if (!isfinite(phase_u_scale) || !isfinite(env_scale) || !isfinite(env_bias) ||
                    !isfinite(phase_bias)) {
                    return false;
                }

                if (include_v) {
                    vDSP_vsmulD(state->vdsp_v, 1, &env_scale, state->vdsp_phase, 1, len);
                    if (env_bias != 0.0) {
                        sim_accel_add_scalar_real(state->vdsp_phase, width, env_bias);
                    }
                    vvsin(state->vdsp_value, state->vdsp_phase, &vforce_len);
                } else {
                    vDSP_vfillD(&one, state->vdsp_value, 1, len);
                }

                vDSP_vsmulD(state->vdsp_u, 1, &phase_u_scale, state->vdsp_phase, 1, len);
                if (phase_bias != 0.0) {
                    sim_accel_add_scalar_real(state->vdsp_phase, width, phase_bias);
                }

                if (!is_complex) {
                    vvcos(state->vdsp_work, state->vdsp_phase, &vforce_len);
                    vDSP_vmulD(
                        state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
                    sim_accel_copy_scale_real(
                        state->vdsp_value, dst_real + row * width, width, amplitude_scale, true);
                } else {
                    SimComplexDouble* row_ptr = dst_complex + row * width;
                    double*           row_re  = &row_ptr[0].re;
                    double*           row_im  = &row_ptr[0].im;

                    vvsincos(state->vdsp_work, state->vdsp_u, state->vdsp_phase, &vforce_len);
                    vDSP_vmulD(state->vdsp_u, 1, state->vdsp_value, 1, state->vdsp_u, 1, len);
                    vDSP_vmulD(state->vdsp_work, 1, state->vdsp_value, 1, state->vdsp_work, 1, len);
                    vDSP_vsmaD(state->vdsp_u, 1, &amplitude_scale, row_re, 2, row_re, 2, len);
                    vDSP_vsmaD(state->vdsp_work, 1, &amplitude_scale, row_im, 2, row_im, 2, len);
                }
                break;
            }
            case SIM_STIMULUS_LAPLACE_BELTRAMI_RECTANGLE:
            default: {
                int    n_u = cfg->mode_u;
                int    n_v = cfg->mode_v;
                double phase_u_scale;
                double phase_v_scale;
                double phase_u_bias;
                double phase_v_bias;

                if (n_u < 0) {
                    n_u = -n_u;
                }
                if (n_u == 0) {
                    n_u = 1;
                }
                if (n_v < 0) {
                    n_v = -n_v;
                }
                if (n_v == 0) {
                    n_v = 1;
                }

                phase_u_scale = M_PI * (double) n_u / cfg->extent_u;
                phase_v_scale = M_PI * (double) n_v / cfg->extent_v;
                phase_u_bias  = 0.5 * M_PI * (double) n_u;
                phase_v_bias  = 0.5 * M_PI * (double) n_v;

                if (!isfinite(phase_u_scale) || !isfinite(phase_v_scale) ||
                    !isfinite(phase_u_bias) || !isfinite(phase_v_bias)) {
                    return false;
                }

                vDSP_vsmulD(state->vdsp_u, 1, &phase_u_scale, state->vdsp_phase, 1, len);
                if (phase_u_bias != 0.0) {
                    sim_accel_add_scalar_real(state->vdsp_phase, width, phase_u_bias);
                }
                vvsin(state->vdsp_value, state->vdsp_phase, &vforce_len);

                if (include_v) {
                    vDSP_vsmulD(state->vdsp_v, 1, &phase_v_scale, state->vdsp_phase, 1, len);
                    if (phase_v_bias != 0.0) {
                        sim_accel_add_scalar_real(state->vdsp_phase, width, phase_v_bias);
                    }
                    vvsin(state->vdsp_work, state->vdsp_phase, &vforce_len);
                    vDSP_vmulD(
                        state->vdsp_value, 1, state->vdsp_work, 1, state->vdsp_value, 1, len);
                }

                if (!is_complex) {
                    double output_scale = amplitude_scale * cos(temporal_real);
                    if (output_scale != 0.0) {
                        sim_accel_copy_scale_real(
                            state->vdsp_value, dst_real + row * width, width, output_scale, true);
                    }
                } else {
                    double output_re_scale = amplitude_scale * cos(temporal_complex);
                    double output_im_scale = amplitude_scale * sin(temporal_complex);
                    sim_accel_accumulate_real_to_complex(state->vdsp_value,
                                                         dst_complex + row * width,
                                                         width,
                                                         output_re_scale,
                                                         output_im_scale);
                }
                break;
            }
        }
    }

    return true;
}
#endif

static void laplace_beltrami_destroy(void* state_ptr) {
    SimStimulusLaplaceBeltramiState* state = (SimStimulusLaplaceBeltramiState*) state_ptr;
#if defined(SIM_HAVE_VDSP)
    if (state != NULL) {
        free(state->vdsp_block);
        state->vdsp_block    = NULL;
        state->vdsp_capacity = 0U;
    }
#endif
    free(state);
}

static SimResult laplace_beltrami_step(void*               state_ptr,
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

    SimStimulusLaplaceBeltramiState* state = (SimStimulusLaplaceBeltramiState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimField* field = sim_context_field(context, state->config.field_index);
    if (field == NULL || field->layout.rank == 0U || field->layout.rank > 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool   is_complex = sim_field_is_complex(field);
    size_t count      = 0U;
    bool   include_v  = (field->layout.rank > 1U);

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
    if (laplace_beltrami_try_vdsp_rows(state,
                                       field,
                                       is_complex,
                                       sim_field_real_data(field),
                                       sim_field_complex_data(field),
                                       count,
                                       include_v,
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

            laplace_beltrami_map_coord(&state->config, x, y, t, &u, &v);
            laplace_beltrami_eval(&state->config, u, v, t, include_v, &re, &im);
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

            laplace_beltrami_map_coord(&state->config, x, y, t, &u, &v);
            laplace_beltrami_eval(&state->config, u, v, t, include_v, &re, &im);

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

SimResult sim_add_stimulus_laplace_beltrami_operator(struct SimContext* context,
                                                     const SimStimulusLaplaceBeltramiConfig* config,
                                                     size_t* out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimStimulusLaplaceBeltramiConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    }

    laplace_beltrami_normalize(&local);
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context,
                                         "stimulus_laplace_beltrami",
                                         (config != NULL),
                                         (config != NULL) ? config->scale_by_dt : true);

    SimStimulusLaplaceBeltramiState* state =
        (SimStimulusLaplaceBeltramiState*) calloc(1U, sizeof(*state));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    state->config = local;
    laplace_beltrami_refresh_symbolic(state);

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "stimulus_laplace_beltrami");

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
    info.abstract_id       = "stimulus_laplace_beltrami";
    sim_operator_info_set_schema_identity(&info, "stimulus_laplace_beltrami");
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

    SimSplitPort    port    = { .context_field_index = state->config.field_index,
                                .require_complex     = needs_complex };
    SimSplitAccess  access  = { .port = 0, .mode = SIM_ACCESS_RW };
    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = laplace_beltrami_step,
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
                                .symbolic      = laplace_beltrami_symbolic,
                                .destroy       = laplace_beltrami_destroy,
                                .info          = info,
                                .config        = op_config,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        laplace_beltrami_destroy(state);
    }
    return result;
}

SimResult sim_stimulus_laplace_beltrami_config(struct SimContext*                context,
                                               size_t                            operator_index,
                                               SimStimulusLaplaceBeltramiConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLaplaceBeltramiState* state = (SimStimulusLaplaceBeltramiState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_stimulus_laplace_beltrami_update(struct SimContext* context,
                                               size_t             operator_index,
                                               const SimStimulusLaplaceBeltramiConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimStimulusLaplaceBeltramiState* state = (SimStimulusLaplaceBeltramiState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimStimulusLaplaceBeltramiConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    laplace_beltrami_normalize(&local);
    state->config = local;
    laplace_beltrami_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
