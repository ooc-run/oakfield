#include "oakfield/operators/nonlinear/hysteretic.h"
#include "operators/common/operator_utils.h"

#include "oakfield/field.h"
#include "oakfield/sim_context.h"
#include "oakfield/operator_identity.h"
#include "oakfield/operator_split.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HYST_SYMBOLIC_CAPACITY 200
#define HYST_WIDTH_MIN 1.0e-9

typedef struct SimHystereticOperatorState {
    SimHystereticOperatorConfig config;
    double*                     output_state;
    double*                     z_state;
    double*                     x_prev;
    size_t                      capacity;
    bool                        initialized;
    char                        symbolic[HYST_SYMBOLIC_CAPACITY];
    double*                     snapshot_output;
    double*                     snapshot_z;
    double*                     snapshot_x_prev;
    size_t                      snapshot_capacity;
    bool                        snapshot_initialized;
    bool                        snapshot_has_bouc_wen;
} SimHystereticOperatorState;

static const char* hysteretic_mode_name(SimHystereticMode mode) {
    switch (mode) {
        case SIM_HYSTERETIC_MODE_SCHMITT:
            return "schmitt";
        case SIM_HYSTERETIC_MODE_PLAY:
            return "play";
        case SIM_HYSTERETIC_MODE_BOUC_WEN:
            return "bouc_wen";
        default:
            return "schmitt";
    }
}

static const char* hysteretic_threshold_mode_name(SimHystereticThresholdMode mode) {
    switch (mode) {
        case SIM_HYSTERETIC_THRESHOLD_CENTER_WIDTH:
            return "center_width";
        case SIM_HYSTERETIC_THRESHOLD_BOUNDS:
        default:
            return "bounds";
    }
}

static const char* hysteretic_input_mode_name(SimHystereticInputMode mode) {
    switch (mode) {
        case SIM_HYSTERETIC_INPUT_ABS:
            return "abs";
        case SIM_HYSTERETIC_INPUT_SQUARED:
            return "squared";
        case SIM_HYSTERETIC_INPUT_DIRECT:
        default:
            return "direct";
    }
}

static void hysteretic_normalize_config(SimHystereticOperatorConfig* cfg) {
    if (cfg == NULL) {
        return;
    }

    if (cfg->mode < SIM_HYSTERETIC_MODE_SCHMITT || cfg->mode > SIM_HYSTERETIC_MODE_BOUC_WEN) {
        cfg->mode = SIM_HYSTERETIC_MODE_SCHMITT;
    }
    if (cfg->threshold_mode != SIM_HYSTERETIC_THRESHOLD_BOUNDS &&
        cfg->threshold_mode != SIM_HYSTERETIC_THRESHOLD_CENTER_WIDTH) {
        cfg->threshold_mode = SIM_HYSTERETIC_THRESHOLD_BOUNDS;
    }
    if (cfg->input_mode < SIM_HYSTERETIC_INPUT_DIRECT ||
        cfg->input_mode > SIM_HYSTERETIC_INPUT_SQUARED) {
        cfg->input_mode = SIM_HYSTERETIC_INPUT_DIRECT;
    }

    if (!isfinite(cfg->input_gain)) {
        cfg->input_gain = 1.0;
    }
    if (!isfinite(cfg->input_bias)) {
        cfg->input_bias = 0.0;
    }

    if (!isfinite(cfg->threshold_low)) {
        cfg->threshold_low = -0.5;
    }
    if (!isfinite(cfg->threshold_high)) {
        cfg->threshold_high = 0.5;
    }
    if (!isfinite(cfg->threshold_center)) {
        cfg->threshold_center = 0.0;
    }
    if (!isfinite(cfg->threshold_width) || cfg->threshold_width <= 0.0) {
        cfg->threshold_width = fabs(cfg->threshold_high - cfg->threshold_low);
    }
    if (!(cfg->threshold_width > 0.0)) {
        cfg->threshold_width = 1.0;
    }

    if (cfg->threshold_mode == SIM_HYSTERETIC_THRESHOLD_CENTER_WIDTH) {
        double half         = 0.5 * cfg->threshold_width;
        cfg->threshold_low  = cfg->threshold_center - half;
        cfg->threshold_high = cfg->threshold_center + half;
    } else if (cfg->threshold_high <= cfg->threshold_low) {
        cfg->threshold_high = cfg->threshold_low + fmax(cfg->threshold_width, HYST_WIDTH_MIN);
    }

    if (!isfinite(cfg->output_low)) {
        cfg->output_low = 0.0;
    }
    if (!isfinite(cfg->output_high)) {
        cfg->output_high = 1.0;
    }
    if (cfg->output_high == cfg->output_low) {
        cfg->output_high = cfg->output_low + 1.0;
    }
    if (cfg->output_low > cfg->output_high) {
        double tmp       = cfg->output_low;
        cfg->output_low  = cfg->output_high;
        cfg->output_high = tmp;
    }

    if (!isfinite(cfg->state_min)) {
        cfg->state_min = -DBL_MAX;
    }
    if (!isfinite(cfg->state_max)) {
        cfg->state_max = DBL_MAX;
    }
    if (cfg->state_min > cfg->state_max) {
        double tmp     = cfg->state_min;
        cfg->state_min = cfg->state_max;
        cfg->state_max = tmp;
    }

    if (!isfinite(cfg->smooth) || cfg->smooth < 0.0) {
        cfg->smooth = 0.0;
    } else if (cfg->smooth > 1.0) {
        cfg->smooth = 1.0;
    }

    if (!isfinite(cfg->rate_limit) || cfg->rate_limit < 0.0) {
        cfg->rate_limit = 0.0;
    }

    cfg->accumulate            = cfg->accumulate ? true : false;
    cfg->scale_by_dt           = cfg->scale_by_dt ? true : false;
    cfg->initialize_from_input = cfg->initialize_from_input ? true : false;

    if (!isfinite(cfg->initial_output)) {
        cfg->initial_output = 0.0;
    }
    if (!isfinite(cfg->initial_input)) {
        cfg->initial_input = 0.0;
    }
    if (!isfinite(cfg->initial_z)) {
        cfg->initial_z = 0.0;
    }

    if (!isfinite(cfg->play_radius) || cfg->play_radius <= 0.0) {
        double width = cfg->threshold_high - cfg->threshold_low;
        if (!(width > 0.0)) {
            width = cfg->threshold_width;
        }
        if (!(width > 0.0)) {
            width = 1.0;
        }
        cfg->play_radius = 0.5 * width;
    }
    if (cfg->play_radius < HYST_WIDTH_MIN) {
        cfg->play_radius = HYST_WIDTH_MIN;
    }

    if (!isfinite(cfg->bw_alpha)) {
        cfg->bw_alpha = 0.1;
    }
    if (cfg->bw_alpha < 0.0) {
        cfg->bw_alpha = 0.0;
    } else if (cfg->bw_alpha > 1.0) {
        cfg->bw_alpha = 1.0;
    }

    if (!isfinite(cfg->bw_A)) {
        cfg->bw_A = 1.0;
    }
    if (!isfinite(cfg->bw_beta)) {
        cfg->bw_beta = 0.5;
    }
    if (!isfinite(cfg->bw_gamma)) {
        cfg->bw_gamma = 0.5;
    }
    if (!isfinite(cfg->bw_n) || cfg->bw_n < 1.0) {
        cfg->bw_n = 1.0;
    }

    if (!isfinite(cfg->bw_z_clamp) || cfg->bw_z_clamp <= 0.0) {
        cfg->bw_z_clamp = 0.0;
    }
    if (!isfinite(cfg->bw_xdot_clamp) || cfg->bw_xdot_clamp <= 0.0) {
        cfg->bw_xdot_clamp = 0.0;
    }

    if (!isfinite(cfg->output_gain)) {
        cfg->output_gain = 1.0;
    }
    if (!isfinite(cfg->output_bias)) {
        cfg->output_bias = 0.0;
    }
}

static void hysteretic_refresh_symbolic(SimHystereticOperatorState* state) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    if (state == NULL) {
        return;
    }

    const SimHystereticOperatorConfig* cfg = &state->config;
    if (cfg->mode == SIM_HYSTERETIC_MODE_BOUC_WEN) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "hysteretic mode=%s input=%s alpha=%.3g beta=%.3g gamma=%.3g n=%.3g",
                        hysteretic_mode_name(cfg->mode),
                        hysteretic_input_mode_name(cfg->input_mode),
                        cfg->bw_alpha,
                        cfg->bw_beta,
                        cfg->bw_gamma,
                        cfg->bw_n);
    } else if (cfg->mode == SIM_HYSTERETIC_MODE_PLAY) {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "hysteretic mode=%s input=%s r=%.3g smooth=%.3g",
                        hysteretic_mode_name(cfg->mode),
                        hysteretic_input_mode_name(cfg->input_mode),
                        cfg->play_radius,
                        cfg->smooth);
    } else {
        (void) snprintf(state->symbolic,
                        sizeof(state->symbolic),
                        "hysteretic mode=%s input=%s thr=%s low=%.3g high=%.3g out=[%.3g,%.3g]",
                        hysteretic_mode_name(cfg->mode),
                        hysteretic_input_mode_name(cfg->input_mode),
                        hysteretic_threshold_mode_name(cfg->threshold_mode),
                        cfg->threshold_low,
                        cfg->threshold_high,
                        cfg->output_low,
                        cfg->output_high);
    }
#else
    (void) state;
#endif
}

static const char* hysteretic_symbolic(const void* state_ptr) {
#if OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    const SimHystereticOperatorState* state = (const SimHystereticOperatorState*) state_ptr;
    return state ? state->symbolic : NULL;
#else
    (void) state_ptr;
    return NULL;
#endif
}

static SimResult hysteretic_describe_fields(const SimField* input_field,
                                            const SimField* output_field,
                                            bool*           out_complex,
                                            size_t*         out_scalar_count) {
    if (input_field == NULL || output_field == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    size_t input_count  = sim_field_element_count(&input_field->layout);
    size_t output_count = sim_field_element_count(&output_field->layout);
    if (input_count != output_count) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    bool input_complex  = sim_field_is_complex(input_field);
    bool output_complex = sim_field_is_complex(output_field);
    if (input_complex != output_complex) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    if (input_complex) {
        if (input_field->element_size != sizeof(SimComplexDouble) ||
            output_field->element_size != sizeof(SimComplexDouble)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    } else {
        if (input_field->element_size != sizeof(double) ||
            output_field->element_size != sizeof(double)) {
            return SIM_RESULT_TYPE_MISMATCH;
        }
    }

    if (out_complex != NULL) {
        *out_complex = input_complex;
    }
    if (out_scalar_count != NULL) {
        *out_scalar_count = input_count * (input_complex ? 2U : 1U);
    }

    return SIM_RESULT_OK;
}

static void hysteretic_fill_info(SimOperatorInfo* info, bool needs_complex) {
    if (info == NULL) {
        return;
    }

    *info                   = sim_operator_info_defaults();
    info->category          = SIM_OPERATOR_CATEGORY_NONLINEAR;
    info->warp_level        = SIM_WARP_LEVEL_NONE;
    info->is_noise          = false;
    info->is_spectral       = false;
    info->is_local          = true;
    info->is_nonlocal       = false;
    info->is_linear         = false;
    info->is_warp           = false;
    info->is_differentiable = false;
    info->preserves_real    = !needs_complex;
    info->preferred_dt      = 0.0;
    info->abstract_id       = "hysteretic";
    sim_operator_info_set_schema_identity(info, "hysteretic");
    info->algebraic_flags       = SIM_OPERATOR_ALG_NONE;
    info->representation.domain = SIM_FIELD_DOMAIN_PHYSICAL;
    info->representation.value_kind =
        needs_complex ? SIM_FIELD_VALUE_COMPLEX_SCALAR : SIM_FIELD_VALUE_REAL_SCALAR;
    info->representation.requires_complex_input          = needs_complex;
    info->representation.requires_complex_representation = needs_complex;
    info->representation.preserves_real_subspace         = info->preserves_real;
}

static void hysteretic_release(void* state_ptr) {
    SimHystereticOperatorState* state = (SimHystereticOperatorState*) state_ptr;
    if (state == NULL) {
        return;
    }
    free(state->output_state);
    free(state->z_state);
    free(state->x_prev);
    free(state->snapshot_output);
    free(state->snapshot_z);
    free(state->snapshot_x_prev);
    free(state);
}

static SimResult hysteretic_prepare_state(SimHystereticOperatorState* state, size_t count) {
    bool need_bouc_wen = (state->config.mode == SIM_HYSTERETIC_MODE_BOUC_WEN);

    if (count == 0U) {
        free(state->output_state);
        free(state->z_state);
        free(state->x_prev);
        state->output_state = NULL;
        state->z_state      = NULL;
        state->x_prev       = NULL;
        state->capacity     = 0U;
        state->initialized  = false;
        return SIM_RESULT_OK;
    }

    if (state->capacity != count) {
        free(state->output_state);
        free(state->z_state);
        free(state->x_prev);
        state->output_state = (double*) calloc(count, sizeof(double));
        state->z_state      = NULL;
        state->x_prev       = NULL;
        if (state->output_state == NULL) {
            state->capacity = 0U;
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->capacity    = count;
        state->initialized = false;
    }

    if (need_bouc_wen) {
        if (state->z_state == NULL || state->x_prev == NULL) {
            free(state->z_state);
            free(state->x_prev);
            state->z_state = (double*) calloc(count, sizeof(double));
            state->x_prev  = (double*) calloc(count, sizeof(double));
            if (state->z_state == NULL || state->x_prev == NULL) {
                free(state->z_state);
                free(state->x_prev);
                state->z_state     = NULL;
                state->x_prev      = NULL;
                state->initialized = false;
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            state->initialized = false;
        }
    } else {
        if (state->z_state != NULL || state->x_prev != NULL) {
            free(state->z_state);
            free(state->x_prev);
            state->z_state     = NULL;
            state->x_prev      = NULL;
            state->initialized = false;
        }
    }

    return SIM_RESULT_OK;
}

static inline double hysteretic_transform_input(const SimHystereticOperatorConfig* cfg,
                                                double                             value) {
    double x = cfg->input_gain * value + cfg->input_bias;
    switch (cfg->input_mode) {
        case SIM_HYSTERETIC_INPUT_ABS:
            return fabs(x);
        case SIM_HYSTERETIC_INPUT_SQUARED:
            return x * x;
        case SIM_HYSTERETIC_INPUT_DIRECT:
        default:
            return x;
    }
}

static inline double hysteretic_clamp_state(const SimHystereticOperatorConfig* cfg, double value) {
    if (value < cfg->state_min) {
        return cfg->state_min;
    }
    if (value > cfg->state_max) {
        return cfg->state_max;
    }
    return value;
}

static void
hysteretic_initialize_state(SimHystereticOperatorState* state, const double* input, size_t count) {
    const SimHystereticOperatorConfig* cfg  = &state->config;
    double                             low  = cfg->threshold_low;
    double                             high = cfg->threshold_high;

    for (size_t i = 0U; i < count; ++i) {
        double x = (input != NULL) ? hysteretic_transform_input(cfg, input[i]) : 0.0;
        double y = cfg->initial_output;

        if (cfg->initialize_from_input && input != NULL) {
            switch (cfg->mode) {
                case SIM_HYSTERETIC_MODE_SCHMITT:
                    if (x >= high) {
                        y = cfg->output_high;
                    } else if (x <= low) {
                        y = cfg->output_low;
                    } else {
                        y = cfg->output_low;
                    }
                    break;
                case SIM_HYSTERETIC_MODE_PLAY:
                    y = x;
                    break;
                case SIM_HYSTERETIC_MODE_BOUC_WEN:
                    y = cfg->bw_alpha * x + (1.0 - cfg->bw_alpha) * cfg->initial_z;
                    break;
                default:
                    break;
            }
        }

        state->output_state[i] = hysteretic_clamp_state(cfg, y);
        if (cfg->mode == SIM_HYSTERETIC_MODE_BOUC_WEN && state->z_state && state->x_prev) {
            state->z_state[i] = cfg->initial_z;
            state->x_prev[i]  = cfg->initialize_from_input ? x : cfg->initial_input;
        }
    }

    if (cfg->mode != SIM_HYSTERETIC_MODE_BOUC_WEN && state->z_state && state->x_prev) {
        (void) memset(state->z_state, 0, count * sizeof(double));
        (void) memset(state->x_prev, 0, count * sizeof(double));
    }

    state->initialized = true;
}

static SimResult
hysteretic_save(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;
    SimHystereticOperatorState* state = (SimHystereticOperatorState*) userdata;
    if (state == NULL || state->capacity == 0U || state->output_state == NULL) {
        return SIM_RESULT_OK;
    }

    if (state->snapshot_capacity < state->capacity) {
        double* next_output =
            (double*) realloc(state->snapshot_output, state->capacity * sizeof(double));
        if (next_output == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->snapshot_output   = next_output;
        state->snapshot_capacity = state->capacity;
    }
    (void) memcpy(state->snapshot_output, state->output_state, state->capacity * sizeof(double));

    state->snapshot_has_bouc_wen = (state->z_state != NULL && state->x_prev != NULL);
    if (state->snapshot_has_bouc_wen) {
        if (state->snapshot_z == NULL || state->snapshot_x_prev == NULL ||
            state->snapshot_capacity < state->capacity) {
            double* next_z = (double*) realloc(state->snapshot_z, state->capacity * sizeof(double));
            if (next_z == NULL) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            state->snapshot_z = next_z;

            double* next_x =
                (double*) realloc(state->snapshot_x_prev, state->capacity * sizeof(double));
            if (next_x == NULL) {
                return SIM_RESULT_OUT_OF_MEMORY;
            }
            state->snapshot_x_prev = next_x;
        }
        (void) memcpy(state->snapshot_z, state->z_state, state->capacity * sizeof(double));
        (void) memcpy(state->snapshot_x_prev, state->x_prev, state->capacity * sizeof(double));
    }

    state->snapshot_initialized = state->initialized;
    return SIM_RESULT_OK;
}

static SimResult
hysteretic_restore(struct SimContext* context, struct SimOperator* self, void* userdata) {
    (void) context;
    (void) self;
    SimHystereticOperatorState* state = (SimHystereticOperatorState*) userdata;
    if (state == NULL || state->capacity == 0U || state->output_state == NULL) {
        return SIM_RESULT_OK;
    }
    if (state->snapshot_output == NULL || state->snapshot_capacity < state->capacity) {
        return SIM_RESULT_OK;
    }

    (void) memcpy(state->output_state, state->snapshot_output, state->capacity * sizeof(double));

    if (state->z_state != NULL && state->x_prev != NULL && state->snapshot_has_bouc_wen) {
        if (state->snapshot_z != NULL && state->snapshot_x_prev != NULL) {
            (void) memcpy(state->z_state, state->snapshot_z, state->capacity * sizeof(double));
            (void) memcpy(state->x_prev, state->snapshot_x_prev, state->capacity * sizeof(double));
        }
        state->initialized = state->snapshot_initialized;
    } else if (state->config.mode == SIM_HYSTERETIC_MODE_BOUC_WEN) {
        state->initialized = false;
    } else {
        state->initialized = state->snapshot_initialized;
    }

    return SIM_RESULT_OK;
}

static SimResult
hysteretic_apply(void* state_ptr, struct SimContext* context, struct SimOperator* self, double dt) {
    (void) self;
    SimHystereticOperatorState* state = (SimHystereticOperatorState*) state_ptr;
    if (state == NULL || context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    const SimHystereticOperatorConfig* cfg          = &state->config;
    SimField*                          input_field  = sim_context_field(context, cfg->input_field);
    SimField*                          output_field = sim_context_field(context, cfg->output_field);
    size_t                             scalar_count = 0U;
    SimResult field_rc = hysteretic_describe_fields(input_field, output_field, NULL, &scalar_count);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }
    if (scalar_count == 0U) {
        return SIM_RESULT_OK;
    }

    SimResult prep_rc = hysteretic_prepare_state(state, scalar_count);
    if (prep_rc != SIM_RESULT_OK) {
        return prep_rc;
    }

    const double* input  = (const double*) sim_field_data(input_field);
    double*       output = (double*) sim_field_data(output_field);
    if (input == NULL || output == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (!state->initialized) {
        hysteretic_initialize_state(state, input, scalar_count);
    }

    double dt_effective = (dt > 0.0 && isfinite(dt)) ? dt : 0.0;
    double inv_dt       = (dt_effective > 0.0) ? (1.0 / dt_effective) : 0.0;
    double smooth       = cfg->smooth;
    double rate_limit   = cfg->rate_limit;
    double rate_step =
        (rate_limit > 0.0) ? rate_limit * ((dt_effective > 0.0) ? dt_effective : 1.0) : 0.0;
    double low       = cfg->threshold_low;
    double high      = cfg->threshold_high;
    double out_low   = cfg->output_low;
    double out_high  = cfg->output_high;
    double out_mid   = 0.5 * (out_low + out_high);
    double add_scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    for (size_t i = 0U; i < scalar_count; ++i) {
        double x      = hysteretic_transform_input(cfg, input[i]);
        double y_prev = state->output_state[i];
        double y_next;

        switch (cfg->mode) {
            case SIM_HYSTERETIC_MODE_SCHMITT: {
                bool is_high = (y_prev >= out_mid);
                if (is_high) {
                    if (x <= low) {
                        y_next = out_low;
                    } else {
                        y_next = out_high;
                    }
                } else {
                    if (x >= high) {
                        y_next = out_high;
                    } else {
                        y_next = out_low;
                    }
                }
                break;
            }
            case SIM_HYSTERETIC_MODE_PLAY: {
                double r = cfg->play_radius;
                if (x - y_prev > r) {
                    y_next = x - r;
                } else if (y_prev - x > r) {
                    y_next = x + r;
                } else {
                    y_next = y_prev;
                }
                break;
            }
            case SIM_HYSTERETIC_MODE_BOUC_WEN: {
                double z      = (state->z_state != NULL) ? state->z_state[i] : 0.0;
                double x_prev = (state->x_prev != NULL) ? state->x_prev[i] : x;
                double xdot   = (dt_effective > 0.0) ? (x - x_prev) * inv_dt : 0.0;

                if (cfg->bw_xdot_clamp > 0.0) {
                    if (xdot > cfg->bw_xdot_clamp) {
                        xdot = cfg->bw_xdot_clamp;
                    } else if (xdot < -cfg->bw_xdot_clamp) {
                        xdot = -cfg->bw_xdot_clamp;
                    }
                }

                double abs_z        = fabs(z);
                double abs_z_pow    = (cfg->bw_n == 1.0) ? abs_z : pow(abs_z, cfg->bw_n);
                double abs_z_pow_n1 = 1.0;
                if (cfg->bw_n > 1.0) {
                    abs_z_pow_n1 = (abs_z > 0.0) ? pow(abs_z, cfg->bw_n - 1.0) : 0.0;
                }

                double z_dot = cfg->bw_A * xdot - cfg->bw_beta * fabs(xdot) * abs_z_pow_n1 * z -
                               cfg->bw_gamma * xdot * abs_z_pow;
                if (dt_effective > 0.0) {
                    z += dt_effective * z_dot;
                }

                if (cfg->bw_z_clamp > 0.0) {
                    if (z > cfg->bw_z_clamp) {
                        z = cfg->bw_z_clamp;
                    } else if (z < -cfg->bw_z_clamp) {
                        z = -cfg->bw_z_clamp;
                    }
                }

                if (state->z_state != NULL) {
                    state->z_state[i] = z;
                }
                if (state->x_prev != NULL) {
                    state->x_prev[i] = x;
                }

                y_next = cfg->bw_alpha * x + (1.0 - cfg->bw_alpha) * z;
                break;
            }
            default:
                y_next = y_prev;
                break;
        }

        if (smooth > 0.0) {
            y_next = (1.0 - smooth) * y_next + smooth * y_prev;
        }

        if (rate_step > 0.0) {
            double delta = y_next - y_prev;
            if (delta > rate_step) {
                delta = rate_step;
            } else if (delta < -rate_step) {
                delta = -rate_step;
            }
            y_next = y_prev + delta;
        }

        y_next                 = hysteretic_clamp_state(cfg, y_next);
        state->output_state[i] = y_next;

        double out_value = cfg->output_gain * y_next + cfg->output_bias;
        if (cfg->accumulate) {
            output[i] += add_scale * out_value;
        } else {
            output[i] = out_value;
        }
    }

    return SIM_RESULT_OK;
}

static SimResult hysteretic_step(void*               state_ptr,
                                 struct SimContext*  context,
                                 struct SimOperator* self,
                                 size_t              substep_index,
                                 double              dt_sub,
                                 void*               scratch,
                                 size_t              scratch_size) {
    (void) substep_index;
    (void) scratch;
    (void) scratch_size;
    return hysteretic_apply(state_ptr, context, self, dt_sub);
}

SimResult sim_add_hysteretic_operator(struct SimContext*                 context,
                                      const SimHystereticOperatorConfig* config,
                                      size_t*                            out_index) {
    if (context == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimHystereticOperatorState* state =
        (SimHystereticOperatorState*) calloc(1U, sizeof(SimHystereticOperatorState));
    if (state == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    SimHystereticOperatorConfig local = { 0 };
    if (config != NULL) {
        local = *config;
    } else {
        local.input_field           = 0U;
        local.output_field          = 0U;
        local.mode                  = SIM_HYSTERETIC_MODE_SCHMITT;
        local.threshold_mode        = SIM_HYSTERETIC_THRESHOLD_BOUNDS;
        local.input_mode            = SIM_HYSTERETIC_INPUT_DIRECT;
        local.input_gain            = 1.0;
        local.input_bias            = 0.0;
        local.threshold_low         = -0.5;
        local.threshold_high        = 0.5;
        local.threshold_center      = 0.0;
        local.threshold_width       = 1.0;
        local.output_low            = -1.0;
        local.output_high           = 1.0;
        local.state_min             = -1.0e6;
        local.state_max             = 1.0e6;
        local.smooth                = 0.0;
        local.rate_limit            = 0.0;
        local.accumulate            = false;
        local.scale_by_dt           = true;
        local.initialize_from_input = true;
        local.initial_output        = 0.0;
        local.initial_input         = 0.0;
        local.initial_z             = 0.0;
        local.play_radius           = 0.0;
        local.bw_alpha              = 0.1;
        local.bw_A                  = 1.0;
        local.bw_beta               = 0.5;
        local.bw_gamma              = 0.5;
        local.bw_n                  = 2.0;
        local.bw_z_clamp            = 0.0;
        local.bw_xdot_clamp         = 0.0;
        local.output_gain           = 1.0;
        local.output_bias           = 0.0;
    }

    local.scale_by_dt = sim_operator_resolve_scale_by_dt(
        context, "hysteretic", (config != NULL), (config != NULL) ? config->scale_by_dt : true);
    hysteretic_normalize_config(&local);
    state->config = local;
    hysteretic_refresh_symbolic(state);

    SimField* input_field   = sim_context_field(context, state->config.input_field);
    SimField* output_field  = sim_context_field(context, state->config.output_field);
    bool      needs_complex = false;
    SimResult field_rc =
        hysteretic_describe_fields(input_field, output_field, &needs_complex, NULL);
    if (field_rc != SIM_RESULT_OK) {
        hysteretic_release(state);
        return field_rc;
    }

    char name[SIM_OPERATOR_NAME_MAX + 1U];
    sim_operator_make_unique_name(name, sizeof(name), "hysteretic");

    SimOperatorInfo info = sim_operator_info_defaults();
    hysteretic_fill_info(&info, needs_complex);

    SimSplitPort ports[2] = {
        { .context_field_index = state->config.input_field, .require_complex = needs_complex },
        { .context_field_index = state->config.output_field, .require_complex = needs_complex }
    };

    SimSplitAccess accesses[2] = { { .port = 0, .mode = SIM_ACCESS_READ },
                                   { .port = 1, .mode = SIM_ACCESS_RW } };

    SimSplitSubstep substep = { .name              = NULL,
                                .fn                = hysteretic_step,
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
                                .symbolic      = hysteretic_symbolic,
                                .save_state    = hysteretic_save,
                                .restore_state = hysteretic_restore,
                                .destroy       = hysteretic_release,
                                .info          = info,
                                .scratch       = { 0U, 0U } };

    SimResult result = sim_split_register(context, &desc, NULL, 0U, out_index, NULL);
    if (result != SIM_RESULT_OK) {
        hysteretic_release(state);
    }

    return result;
}

SimResult sim_hysteretic_config(struct SimContext*           context,
                                size_t                       operator_index,
                                SimHystereticOperatorConfig* out_config) {
    if (context == NULL || out_config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimHystereticOperatorState* state = (SimHystereticOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    *out_config = state->config;
    return SIM_RESULT_OK;
}

SimResult sim_hysteretic_update(struct SimContext*                 context,
                                size_t                             operator_index,
                                const SimHystereticOperatorConfig* config) {
    if (context == NULL || config == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    SimOperator* op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    SimHystereticOperatorState* state = (SimHystereticOperatorState*) sim_split_state(op);
    if (state == NULL) {
        return SIM_RESULT_INVALID_STATE;
    }

    SimHystereticOperatorConfig local = *config;
    local.scale_by_dt =
        sim_operator_resolve_scale_by_dt(context, "hysteretic", true, config->scale_by_dt);
    hysteretic_normalize_config(&local);

    SimField* input_field  = sim_context_field(context, local.input_field);
    SimField* output_field = sim_context_field(context, local.output_field);
    SimResult field_rc     = hysteretic_describe_fields(input_field, output_field, NULL, NULL);
    if (field_rc != SIM_RESULT_OK) {
        return field_rc;
    }

    state->config      = local;
    state->initialized = false;
    hysteretic_refresh_symbolic(state);

    sim_scheduler_plan_invalidate(&context->scheduler);
    return SIM_RESULT_OK;
}
