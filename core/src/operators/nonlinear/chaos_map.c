#include "oakfield/operators/nonlinear/chaos_map.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHAOS_SYMBOLIC_CAPACITY 192
#define CHAOS_EPSILON_MIN 1.0e-12

typedef struct SimChaosMapOperatorState {
    SimChaosMapOperatorConfig config;
    char                      symbolic[CHAOS_SYMBOLIC_CAPACITY];
} SimChaosMapOperatorState;

static SimResult chaos_map_describe_io_fields(const SimField*         input_field,
                                              const SimField*         output_field,
                                              bool*                   out_needs_complex,
                                              SimFieldRepresentation* out_output_repr) {
    if (input_field == NULL || output_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const bool input_complex  = sim_field_is_complex(input_field);
    const bool output_complex = sim_field_is_complex(output_field);
    if (input_complex != output_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (input_complex) {
        if (input_field->element_size != sizeof(SimComplexDouble) ||
            output_field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else if (input_field->element_size != sizeof(double) ||
               output_field->element_size != sizeof(double)) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (sim_field_element_count(&input_field->layout) !=
        sim_field_element_count(&output_field->layout)) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (out_needs_complex != NULL) {
        *out_needs_complex = input_complex;
    }
    if (out_output_repr != NULL) {
        *out_output_repr = sim_field_representation(output_field);
    }

    return SIM_RESULT_OK;
}

static void
chaos_map_fill_info(SimOperatorInfo* info, SimFieldRepresentation output_repr, bool needs_complex) {
    if (info == NULL) {
        return;
    }

    *info                   = sim_operator_info_defaults();
    info->category          = SIM_OPERATOR_CATEGORY_NONLINEAR;
    info->warp_level        = SIM_WARP_LEVEL_LEVEL2;
    info->is_noise          = false;
    info->is_spectral       = false;
    info->is_local          = true;
    info->is_nonlocal       = false;
    info->is_linear         = false;
    info->is_warp           = false;
    info->is_differentiable = false;
    info->preserves_real    = !needs_complex;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "chaos_map";
    sim_operator_info_set_schema_identity(info, "chaos_map");
    info->algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info->representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info->representation.value_kind =
        (output_repr.value_kind != SIM_FIELD_VALUE_UNKNOWN)
            ? output_repr.value_kind
            : (needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR);
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = info->preserves_real;
}

static const char* chaos_map_name(SimChaosMapType map_type) {
    switch (map_type) {
        case SIM_CHAOS_MAP_IKEDA:
            return "ikeda";
        case SIM_CHAOS_MAP_EXPONENTIAL:
            return "exponential";
        case SIM_CHAOS_MAP_QUADRATIC:
            return "quadratic";
        case SIM_CHAOS_MAP_HENON:
            return "henon";
        case SIM_CHAOS_MAP_LOZI:
            return "lozi";
        case SIM_CHAOS_MAP_TINKERBELL:
            return "tinkerbell";
        case SIM_CHAOS_MAP_STANDARD:
        default:
            return "standard";
    }
}

static const char* chaos_kick_mode_name(SimChaosKickMode mode) {
    switch (mode) {
        case SIM_CHAOS_DRIFT_KICK:
            return "drift_kick";
        case SIM_CHAOS_KICK_DRIFT_KICK:
            return "symmetric";
        case SIM_CHAOS_KICK_DRIFT:
        default:
            return "kick_drift";
    }
}

static const char* chaos_wrap_mode_name(SimChaosWrapMode mode) {
    switch (mode) {
        case SIM_CHAOS_WRAP_PERIODIC:
            return "periodic";
        case SIM_CHAOS_WRAP_CLAMP:
            return "clamp";
        case SIM_CHAOS_WRAP_MIRROR:
            return "mirror";
        case SIM_CHAOS_WRAP_NONE:
        default:
            return "none";
    }
}

static const char* chaos_escape_mode_name(SimChaosEscapeMode mode) {
    switch (mode) {
        case SIM_CHAOS_ESCAPE_CLAMP:
            return "clamp";
        case SIM_CHAOS_ESCAPE_RESET:
            return "reset";
        case SIM_CHAOS_ESCAPE_NAN:
            return "nan";
        case SIM_CHAOS_ESCAPE_NONE:
        default:
            return "none";
    }
}

static void chaos_map_normalize_config(SimChaosMapOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    if (config->map_type < SIM_CHAOS_MAP_STANDARD || config->map_type > SIM_CHAOS_MAP_TINKERBELL) {
        config->map_type = SIM_CHAOS_MAP_STANDARD;
    }

    if (config->kick_mode < SIM_CHAOS_KICK_DRIFT || config->kick_mode > SIM_CHAOS_KICK_DRIFT_KICK) {
        config->kick_mode = SIM_CHAOS_KICK_DRIFT;
    }

    if (config->iterations_per_step == 0U) {
        config->iterations_per_step = 1U;
    }

    if (!isfinite(config->blend)) {
        config->blend = 1.0;
    }
    if (config->blend < 0.0) {
        config->blend = 0.0;
    }
    if (config->blend > 1.0) {
        config->blend = 1.0;
    }

    if (!isfinite(config->k)) {
        config->k = 1.0;
    }
    if (!isfinite(config->angle_scale) || fabs(config->angle_scale) < CHAOS_EPSILON_MIN) {
        config->angle_scale = 1.0;
    }

    if (!isfinite(config->ikeda_u)) {
        config->ikeda_u = 0.9;
    }
    if (!isfinite(config->ikeda_a)) {
        config->ikeda_a = 0.4;
    }
    if (!isfinite(config->ikeda_b)) {
        config->ikeda_b = 6.0;
    }
    if (!isfinite(config->ikeda_offset_re)) {
        config->ikeda_offset_re = 1.0;
    }
    if (!isfinite(config->ikeda_offset_im)) {
        config->ikeda_offset_im = 0.0;
    }

    if (!isfinite(config->exp_scale_re)) {
        config->exp_scale_re = 1.0;
    }
    if (!isfinite(config->exp_scale_im)) {
        config->exp_scale_im = 0.0;
    }
    if (!isfinite(config->exp_c_re)) {
        config->exp_c_re = 0.0;
    }
    if (!isfinite(config->exp_c_im)) {
        config->exp_c_im = 0.0;
    }

    if (!isfinite(config->quad_a_re)) {
        config->quad_a_re = 1.0;
    }
    if (!isfinite(config->quad_a_im)) {
        config->quad_a_im = 0.0;
    }
    if (!isfinite(config->quad_b_re)) {
        config->quad_b_re = 0.0;
    }
    if (!isfinite(config->quad_b_im)) {
        config->quad_b_im = 0.0;
    }
    if (!isfinite(config->quad_c_re)) {
        config->quad_c_re = 0.0;
    }
    if (!isfinite(config->quad_c_im)) {
        config->quad_c_im = 0.0;
    }

    if (!isfinite(config->henon_a)) {
        config->henon_a = 1.4;
    }
    if (!isfinite(config->henon_b)) {
        config->henon_b = 0.3;
    }
    if (!isfinite(config->henon_x_gain)) {
        config->henon_x_gain = 0.0;
    }
    if (!isfinite(config->henon_y_gain)) {
        config->henon_y_gain = 1.0;
    }
    if (!isfinite(config->henon_offset_re)) {
        config->henon_offset_re = 1.0;
    }
    if (!isfinite(config->henon_offset_im)) {
        config->henon_offset_im = 0.0;
    }

    if (!isfinite(config->lozi_a)) {
        config->lozi_a = 1.7;
    }
    if (!isfinite(config->lozi_b)) {
        config->lozi_b = 0.5;
    }
    if (!isfinite(config->lozi_x_gain)) {
        config->lozi_x_gain = 0.0;
    }
    if (!isfinite(config->lozi_y_gain)) {
        config->lozi_y_gain = 1.0;
    }
    if (!isfinite(config->lozi_offset_re)) {
        config->lozi_offset_re = 1.0;
    }
    if (!isfinite(config->lozi_offset_im)) {
        config->lozi_offset_im = 0.0;
    }
    if (!isfinite(config->lozi_abs_epsilon) || config->lozi_abs_epsilon < 0.0) {
        config->lozi_abs_epsilon = 0.0;
    }

    if (!isfinite(config->tinkerbell_a)) {
        config->tinkerbell_a = 0.9;
    }
    if (!isfinite(config->tinkerbell_b)) {
        config->tinkerbell_b = -0.6013;
    }
    if (!isfinite(config->tinkerbell_c)) {
        config->tinkerbell_c = 2.0;
    }
    if (!isfinite(config->tinkerbell_d)) {
        config->tinkerbell_d = 0.5;
    }
    if (!isfinite(config->tinkerbell_x2_gain)) {
        config->tinkerbell_x2_gain = 1.0;
    }
    if (!isfinite(config->tinkerbell_y2_gain)) {
        config->tinkerbell_y2_gain = -1.0;
    }
    if (!isfinite(config->tinkerbell_xy_gain)) {
        config->tinkerbell_xy_gain = 2.0;
    }
    if (!isfinite(config->tinkerbell_offset_re)) {
        config->tinkerbell_offset_re = 0.0;
    }
    if (!isfinite(config->tinkerbell_offset_im)) {
        config->tinkerbell_offset_im = 0.0;
    }

    if (config->wrap_mode_re < SIM_CHAOS_WRAP_NONE ||
        config->wrap_mode_re > SIM_CHAOS_WRAP_MIRROR) {
        config->wrap_mode_re = SIM_CHAOS_WRAP_NONE;
    }
    if (config->wrap_mode_im < SIM_CHAOS_WRAP_NONE ||
        config->wrap_mode_im > SIM_CHAOS_WRAP_MIRROR) {
        config->wrap_mode_im = SIM_CHAOS_WRAP_NONE;
    }

    if (!isfinite(config->wrap_min_re)) {
        config->wrap_min_re = 0.0;
    }
    if (!isfinite(config->wrap_max_re)) {
        config->wrap_max_re = 2.0 * M_PI;
    }
    if (!isfinite(config->wrap_min_im)) {
        config->wrap_min_im = 0.0;
    }
    if (!isfinite(config->wrap_max_im)) {
        config->wrap_max_im = 2.0 * M_PI;
    }

    if (config->escape_mode < SIM_CHAOS_ESCAPE_NONE || config->escape_mode > SIM_CHAOS_ESCAPE_NAN) {
        config->escape_mode = SIM_CHAOS_ESCAPE_NONE;
    }
    if (!isfinite(config->escape_radius) || config->escape_radius < 0.0) {
        config->escape_radius = 0.0;
    }
    if (!isfinite(config->escape_reset_re)) {
        config->escape_reset_re = 0.0;
    }
    if (!isfinite(config->escape_reset_im)) {
        config->escape_reset_im = 0.0;
    }
}

static void chaos_map_refresh_symbolic(SimChaosMapOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimChaosMapOperatorConfig* cfg = &state->config;
    (void) snprintf(state->symbolic,
                    sizeof(state->symbolic),
                    "chaos_map=%s iter=%u k=%.3g u=%.3g wrap=(%s,%s)",
                    chaos_map_name(cfg->map_type),
                    cfg->iterations_per_step,
                    cfg->k,
                    cfg->ikeda_u,
                    chaos_wrap_mode_name(cfg->wrap_mode_re),
                    chaos_wrap_mode_name(cfg->wrap_mode_im));
#else
    (void) state;
#endif
}

static const char* chaos_map_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimChaosMapOperatorState* state = (const SimChaosMapOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static void chaos_map_destroy(void* state_ptr) {
    SimChaosMapOperatorState* state = (SimChaosMapOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state);
}

static double
chaos_wrap_value(double value, SimChaosWrapMode mode, double min_value, double max_value) {
    if (!isfinite(value)) {
        return value;
    }
    if (mode == SIM_CHAOS_WRAP_NONE) {
        return value;
    }
    if (!isfinite(min_value) || !isfinite(max_value) || max_value <= min_value) {
        return value;
    }

    double range = max_value - min_value;
    if (range <= 0.0) {
        return value;
    }

    switch (mode) {
        case SIM_CHAOS_WRAP_PERIODIC: {
            double t = fmod(value - min_value, range);
            if (t < 0.0) {
                t += range;
            }
            return min_value + t;
        }
        case SIM_CHAOS_WRAP_CLAMP:
            if (value < min_value) {
                return min_value;
            }
            if (value > max_value) {
                return max_value;
            }
            return value;
        case SIM_CHAOS_WRAP_MIRROR: {
            double period = 2.0 * range;
            double t      = fmod(value - min_value, period);
            if (t < 0.0) {
                t += period;
            }
            if (t > range) {
                t = period - t;
            }
            return min_value + t;
        }
        default:
            return value;
    }
}

static SimResult chaos_map_prepare_param_field(const SimContext*        context,
                                               size_t                   field_index,
                                               size_t                   count,
                                               const double**           out_real,
                                               const SimComplexDouble** out_complex) {
    if (out_real != NULL) {
        *out_real = NULL;
    }
    if (out_complex != NULL) {
        *out_complex = NULL;
    }

    if (field_index == SIZE_MAX) {
        return SIM_RESULT_OK;
    }

    SimField* field = sim_context_field((SimContext*) context, field_index);
    if (field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_element_count(&field->layout) != count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (sim_field_is_complex(field)) {
        const SimComplexDouble* data = sim_field_complex_data_const(field);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (out_complex != NULL) {
            *out_complex = data;
        }
    } else {
        if (field->element_size != sizeof(double)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
        const double* data = sim_field_real_data_const(field);
        if (data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (out_real != NULL) {
            *out_real = data;
        }
    }

    return SIM_RESULT_OK;
}

static double chaos_param_real(const double*           real_data,
                               const SimComplexDouble* complex_data,
                               size_t                  index,
                               double                  fallback) {
    if (real_data != NULL) {
        return real_data[index];
    }
    if (complex_data != NULL) {
        return complex_data[index].re;
    }
    return fallback;
}

static SimComplexDouble chaos_param_complex(const double*           real_data,
                                            const SimComplexDouble* complex_data,
                                            size_t                  index,
                                            SimComplexDouble        fallback) {
    if (complex_data != NULL) {
        return complex_data[index];
    }
    if (real_data != NULL) {
        SimComplexDouble value = { real_data[index], 0.0 };
        return value;
    }
    return fallback;
}

static void chaos_apply_escape(const SimChaosMapOperatorConfig* cfg, double* x, double* y) {
    if (cfg == NULL || x == NULL || y == NULL) {
        return;
    }

    if (cfg->escape_mode == SIM_CHAOS_ESCAPE_NONE || !(cfg->escape_radius > 0.0)) {
        return;
    }

    double radius = hypot(*x, *y);
    if (!(radius > cfg->escape_radius)) {
        return;
    }

    switch (cfg->escape_mode) {
        case SIM_CHAOS_ESCAPE_CLAMP: {
            double scale = cfg->escape_radius / radius;
            *x *= scale;
            *y *= scale;
            break;
        }
        case SIM_CHAOS_ESCAPE_RESET:
            *x = cfg->escape_reset_re;
            *y = cfg->escape_reset_im;
            break;
        case SIM_CHAOS_ESCAPE_NAN:
            *x = NAN;
            *y = NAN;
            break;
        case SIM_CHAOS_ESCAPE_NONE:
        default:
            break;
    }
}

static SimResult
chaos_map_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;
    (void) dt;

    SimChaosMapOperatorState* state = (SimChaosMapOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimChaosMapOperatorConfig* cfg = &state->config;

    SimField* input_field  = sim_context_field(context, cfg->input_field);
    SimField* output_field = sim_context_field(context, cfg->output_field);
    if (input_field == NULL || output_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool      needs_complex = false;
    SimResult io_rc = chaos_map_describe_io_fields(input_field, output_field, &needs_complex, NULL);
    if (io_rc != SIM_RESULT_OK) {
        return io_rc;
    }

    size_t count = sim_field_element_count(&input_field->layout);
    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    const SimComplexDouble* input_complex_data  = NULL;
    SimComplexDouble*       output_complex_data = NULL;
    const double*           input_real_data     = NULL;
    double*                 output_real_data    = NULL;

    if (needs_complex) {
        input_complex_data  = sim_field_complex_data_const(input_field);
        output_complex_data = sim_field_complex_data(output_field);
        if (input_complex_data == NULL || output_complex_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    } else {
        input_real_data  = sim_field_real_data_const(input_field);
        output_real_data = sim_field_real_data(output_field);
        if (input_real_data == NULL || output_real_data == NULL) {
            return SIM_RESULT_INVALID_ARGUMENT;
        }
    }

    const double*           k_real    = NULL;
    const SimComplexDouble* k_complex = NULL;
    const double*           u_real    = NULL;
    const SimComplexDouble* u_complex = NULL;
    const double*           a_real    = NULL;
    const SimComplexDouble* a_complex = NULL;
    const double*           b_real    = NULL;
    const SimComplexDouble* b_complex = NULL;
    const double*           c_real    = NULL;
    const SimComplexDouble* c_complex = NULL;
    const double*           d_real    = NULL;
    const SimComplexDouble* d_complex = NULL;

    SimResult prep =
        chaos_map_prepare_param_field(context, cfg->k_field, count, &k_real, &k_complex);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }
    prep = chaos_map_prepare_param_field(context, cfg->u_field, count, &u_real, &u_complex);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }
    prep = chaos_map_prepare_param_field(context, cfg->a_field, count, &a_real, &a_complex);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }
    prep = chaos_map_prepare_param_field(context, cfg->b_field, count, &b_real, &b_complex);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }
    prep = chaos_map_prepare_param_field(context, cfg->c_field, count, &c_real, &c_complex);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }
    prep = chaos_map_prepare_param_field(context, cfg->d_field, count, &d_real, &d_complex);
    if (prep != SIM_RESULT_OK) {
        return prep;
    }

    SimComplexDouble exp_c_default  = { cfg->exp_c_re, cfg->exp_c_im };
    SimComplexDouble exp_scale      = { cfg->exp_scale_re, cfg->exp_scale_im };
    SimComplexDouble quad_a_default = { cfg->quad_a_re, cfg->quad_a_im };
    SimComplexDouble quad_b_default = { cfg->quad_b_re, cfg->quad_b_im };
    SimComplexDouble quad_c_default = { cfg->quad_c_re, cfg->quad_c_im };

    for (size_t i = 0U; i < count; ++i) {
        double x0 = needs_complex ? input_complex_data[i].re : input_real_data[i];
        double y0 = needs_complex ? input_complex_data[i].im : 0.0;
        double x  = x0;
        double y  = y0;

        double           k            = chaos_param_real(k_real, k_complex, i, cfg->k);
        double           ikeda_u      = chaos_param_real(u_real, u_complex, i, cfg->ikeda_u);
        double           ikeda_a      = chaos_param_real(a_real, a_complex, i, cfg->ikeda_a);
        double           ikeda_b      = chaos_param_real(b_real, b_complex, i, cfg->ikeda_b);
        SimComplexDouble exp_c        = chaos_param_complex(c_real, c_complex, i, exp_c_default);
        SimComplexDouble quad_a       = chaos_param_complex(a_real, a_complex, i, quad_a_default);
        SimComplexDouble quad_b       = chaos_param_complex(b_real, b_complex, i, quad_b_default);
        SimComplexDouble quad_c       = chaos_param_complex(c_real, c_complex, i, quad_c_default);
        double           henon_a      = chaos_param_real(a_real, a_complex, i, cfg->henon_a);
        double           henon_b      = chaos_param_real(b_real, b_complex, i, cfg->henon_b);
        double           lozi_a       = chaos_param_real(a_real, a_complex, i, cfg->lozi_a);
        double           lozi_b       = chaos_param_real(b_real, b_complex, i, cfg->lozi_b);
        double           tinkerbell_a = chaos_param_real(a_real, a_complex, i, cfg->tinkerbell_a);
        double           tinkerbell_b = chaos_param_real(b_real, b_complex, i, cfg->tinkerbell_b);
        double           tinkerbell_c = chaos_param_real(c_real, c_complex, i, cfg->tinkerbell_c);
        double           tinkerbell_d = chaos_param_real(d_real, d_complex, i, cfg->tinkerbell_d);

        for (unsigned int iter = 0U; iter < cfg->iterations_per_step; ++iter) {
            switch (cfg->map_type) {
                case SIM_CHAOS_MAP_STANDARD: {
                    if (cfg->kick_mode == SIM_CHAOS_KICK_DRIFT) {
                        y = y + k * sin(cfg->angle_scale * x);
                        x = x + y;
                    } else if (cfg->kick_mode == SIM_CHAOS_DRIFT_KICK) {
                        x = x + y;
                        y = y + k * sin(cfg->angle_scale * x);
                    } else {
                        y = y + 0.5 * k * sin(cfg->angle_scale * x);
                        x = x + y;
                        y = y + 0.5 * k * sin(cfg->angle_scale * x);
                    }
                    break;
                }
                case SIM_CHAOS_MAP_IKEDA: {
                    double r2 = x * x + y * y;
                    double t  = ikeda_a - ikeda_b / (1.0 + r2);
                    double ct = cos(t);
                    double st = sin(t);
                    double nx = cfg->ikeda_offset_re + ikeda_u * (x * ct - y * st);
                    double ny = cfg->ikeda_offset_im + ikeda_u * (x * st + y * ct);
                    x         = nx;
                    y         = ny;
                    break;
                }
                case SIM_CHAOS_MAP_EXPONENTIAL: {
                    double complex z      = (x + I * y) * (exp_scale.re + I * exp_scale.im);
                    double complex z_next = cexp(z) + (exp_c.re + I * exp_c.im);
                    x                     = creal(z_next);
                    y                     = cimag(z_next);
                    break;
                }
                case SIM_CHAOS_MAP_QUADRATIC: {
                    double complex z      = x + I * y;
                    double complex a      = quad_a.re + I * quad_a.im;
                    double complex b      = quad_b.re + I * quad_b.im;
                    double complex c      = quad_c.re + I * quad_c.im;
                    double complex z_next = a * z * z + b * z + c;
                    x                     = creal(z_next);
                    y                     = cimag(z_next);
                    break;
                }
                case SIM_CHAOS_MAP_HENON: {
                    double nx = cfg->henon_offset_re + cfg->henon_x_gain * x +
                                cfg->henon_y_gain * y - henon_a * x * x;
                    double ny = cfg->henon_offset_im + henon_b * x;
                    x         = nx;
                    y         = ny;
                    break;
                }
                case SIM_CHAOS_MAP_LOZI: {
                    double abs_x = fabs(x);
                    if (cfg->lozi_abs_epsilon > 0.0) {
                        double eps = cfg->lozi_abs_epsilon;
                        abs_x      = sqrt(x * x + eps * eps);
                    }
                    double nx = cfg->lozi_offset_re + cfg->lozi_x_gain * x + cfg->lozi_y_gain * y -
                                lozi_a * abs_x;
                    double ny = cfg->lozi_offset_im + lozi_b * x;
                    x         = nx;
                    y         = ny;
                    break;
                }
                case SIM_CHAOS_MAP_TINKERBELL: {
                    double x2 = x * x;
                    double y2 = y * y;
                    double xy = x * y;
                    double nx = cfg->tinkerbell_offset_re + cfg->tinkerbell_x2_gain * x2 +
                                cfg->tinkerbell_y2_gain * y2 + tinkerbell_a * x + tinkerbell_b * y;
                    double ny = cfg->tinkerbell_offset_im + cfg->tinkerbell_xy_gain * xy +
                                tinkerbell_c * x + tinkerbell_d * y;
                    x         = nx;
                    y         = ny;
                    break;
                }
                default:
                    break;
            }

            chaos_apply_escape(cfg, &x, &y);

            if (cfg->wrap_mode_re != SIM_CHAOS_WRAP_NONE) {
                x = chaos_wrap_value(x, cfg->wrap_mode_re, cfg->wrap_min_re, cfg->wrap_max_re);
            }
            if (cfg->wrap_mode_im != SIM_CHAOS_WRAP_NONE) {
                y = chaos_wrap_value(y, cfg->wrap_mode_im, cfg->wrap_min_im, cfg->wrap_max_im);
            }
        }

        if (cfg->blend < 1.0) {
            double alpha = cfg->blend;
            x            = x0 + alpha * (x - x0);
            y            = y0 + alpha * (y - y0);
        }

        if (!isfinite(x) || !isfinite(y)) {
            continue;
        }

        if (needs_complex) {
            output_complex_data[i].re = x;
            output_complex_data[i].im = y;
        } else {
            output_real_data[i] = x;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult chaos_map_step(void*               state_ptr,
                                struct SimContext*  context,
                                struct SimOperator* self,
                                size_t              substep_index,
                                double              dt_sub,
                                void*               scratch,
                                size_t              scratch_size) {
    (void) substep_index;
    (void) dt_sub;
    (void) scratch;
    (void) scratch_size;
    return chaos_map_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_chaos_map_operator(struct SimContext*               context,
                                     const SimChaosMapOperatorConfig* config,
                                     size_t*                          out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimChaosMapOperatorState* state =
        (SimChaosMapOperatorState*) calloc(1U, sizeof(SimChaosMapOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimChaosMapOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.input_field          = 0U;
        local.output_field         = 0U;
        local.map_type             = SIM_CHAOS_MAP_STANDARD;
        local.kick_mode            = SIM_CHAOS_KICK_DRIFT;
        local.iterations_per_step  = 1U;
        local.blend                = 1.0;
        local.k                    = 1.0;
        local.angle_scale          = 1.0;
        local.ikeda_u              = 0.9;
        local.ikeda_a              = 0.4;
        local.ikeda_b              = 6.0;
        local.ikeda_offset_re      = 1.0;
        local.ikeda_offset_im      = 0.0;
        local.exp_scale_re         = 1.0;
        local.exp_scale_im         = 0.0;
        local.exp_c_re             = 0.0;
        local.exp_c_im             = 0.0;
        local.quad_a_re            = 1.0;
        local.quad_a_im            = 0.0;
        local.quad_b_re            = 0.0;
        local.quad_b_im            = 0.0;
        local.quad_c_re            = 0.0;
        local.quad_c_im            = 0.0;
        local.henon_a              = 1.4;
        local.henon_b              = 0.3;
        local.henon_x_gain         = 0.0;
        local.henon_y_gain         = 1.0;
        local.henon_offset_re      = 1.0;
        local.henon_offset_im      = 0.0;
        local.lozi_a               = 1.7;
        local.lozi_b               = 0.5;
        local.lozi_x_gain          = 0.0;
        local.lozi_y_gain          = 1.0;
        local.lozi_offset_re       = 1.0;
        local.lozi_offset_im       = 0.0;
        local.lozi_abs_epsilon     = 0.0;
        local.tinkerbell_a         = 0.9;
        local.tinkerbell_b         = -0.6013;
        local.tinkerbell_c         = 2.0;
        local.tinkerbell_d         = 0.5;
        local.tinkerbell_x2_gain   = 1.0;
        local.tinkerbell_y2_gain   = -1.0;
        local.tinkerbell_xy_gain   = 2.0;
        local.tinkerbell_offset_re = 0.0;
        local.tinkerbell_offset_im = 0.0;
        local.k_field              = SIZE_MAX;
        local.u_field              = SIZE_MAX;
        local.a_field              = SIZE_MAX;
        local.b_field              = SIZE_MAX;
        local.c_field              = SIZE_MAX;
        local.d_field              = SIZE_MAX;
        local.wrap_mode_re         = SIM_CHAOS_WRAP_NONE;
        local.wrap_mode_im         = SIM_CHAOS_WRAP_NONE;
        local.wrap_min_re          = 0.0;
        local.wrap_max_re          = 2.0 * M_PI;
        local.wrap_min_im          = 0.0;
        local.wrap_max_im          = 2.0 * M_PI;
        local.escape_mode          = SIM_CHAOS_ESCAPE_NONE;
        local.escape_radius        = 0.0;
        local.escape_reset_re      = 0.0;
        local.escape_reset_im      = 0.0;
    }

    chaos_map_normalize_config(&local);
    state->config = local;
    chaos_map_refresh_symbolic(state);

    SimField* input_field  = sim_context_field(context, state->config.input_field);
    SimField* output_field = sim_context_field(context, state->config.output_field);
    if (input_field == NULL || output_field == NULL) {
        chaos_map_destroy(state);
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool                   needs_complex = false;
    SimFieldRepresentation output_repr   = { 0 };
    SimResult              io_rc =
        chaos_map_describe_io_fields(input_field, output_field, &needs_complex, &output_repr);
    if (io_rc != SIM_RESULT_OK) {
        chaos_map_destroy(state);
        return io_rc;
    }

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "chaos_map");

    SimOperatorInfo info = sim_operator_info_defaults();
    chaos_map_fill_info(&info, output_repr, needs_complex);

    SimSplitPort ports[2] = {
        { .context_field_index = state->config.input_field, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = chaos_map_step,
                                .accesses          = accesses,
                                .access_count      = 2U,
                                .dt_scale          = 1.0,
                                .barrier_after     = false,
                                .error_measure     = NULL,
                                .required_features = 0U };

    SimSplitDescriptor desc = { .name          = name,
                                .ports         = ports,
                                .port_count    = 2U,
                                .substeps      = &substep,
                                .substep_count = 1U,
                                .state         = state,
                                .symbolic      = chaos_map_symbolic,
                                .destroy       = chaos_map_destroy,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        chaos_map_destroy(state);
    }

    return result;
}

SimResult sim_chaos_map_config(struct SimContext*         context,
                               size_t                     operator_index,
                               SimChaosMapOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimChaosMapOperatorState* state = (SimChaosMapOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_chaos_map_update(struct SimContext*               context,
                               size_t                           operator_index,
                               const SimChaosMapOperatorConfig* config) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimChaosMapOperatorState* state = (SimChaosMapOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimChaosMapOperatorConfig local = state->config;
    if (config != NULL) {
        local = *config;
    }

    chaos_map_normalize_config(&local);
    state->config = local;
    chaos_map_refresh_symbolic(state);

    return SIM_RESULT_OK;
}
