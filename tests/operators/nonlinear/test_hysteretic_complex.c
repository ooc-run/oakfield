/*
 * Migrated nonlinear operator coverage for hysteretic complex contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool approx_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= tol * scale;
}

static double transform_input(const SimHystereticOperatorConfig *cfg, double value) {
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

static double clamp_state(const SimHystereticOperatorConfig *cfg, double value) {
    if (value < cfg->state_min) {
        return cfg->state_min;
    }
    if (value > cfg->state_max) {
        return cfg->state_max;
    }
    return value;
}

static void initialize_expected_state(const SimHystereticOperatorConfig *cfg, const double *input,
                                      double *output_state, double *z_state, double *x_prev,
                                      size_t scalar_count) {
    for (size_t i = 0U; i < scalar_count; ++i) {
        double x = (input != NULL) ? transform_input(cfg, input[i]) : 0.0;
        double y = cfg->initial_output;

        if (cfg->initialize_from_input && input != NULL) {
            switch (cfg->mode) {
            case SIM_HYSTERETIC_MODE_SCHMITT:
                if (x >= cfg->threshold_high) {
                    y = cfg->output_high;
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

        output_state[i] = clamp_state(cfg, y);
        z_state[i] = cfg->initial_z;
        x_prev[i] = cfg->initialize_from_input ? x : cfg->initial_input;
    }
}

static void apply_expected_bouc_wen(const SimHystereticOperatorConfig *cfg, const double *input,
                                    double *output_field, double *output_state, double *z_state,
                                    double *x_prev, size_t scalar_count, double dt) {
    double dt_effective = (dt > 0.0 && isfinite(dt)) ? dt : 0.0;
    double inv_dt = (dt_effective > 0.0) ? (1.0 / dt_effective) : 0.0;
    double smooth = cfg->smooth;
    double rate_limit = cfg->rate_limit;
    double rate_step =
        (rate_limit > 0.0) ? rate_limit * ((dt_effective > 0.0) ? dt_effective : 1.0) : 0.0;
    double add_scale = cfg->scale_by_dt ? fmax(dt, 0.0) : 1.0;

    for (size_t i = 0U; i < scalar_count; ++i) {
        double x = transform_input(cfg, input[i]);
        double y_prev = output_state[i];
        double z = z_state[i];
        double x_last = x_prev[i];
        double xdot = (dt_effective > 0.0) ? (x - x_last) * inv_dt : 0.0;

        if (cfg->bw_xdot_clamp > 0.0) {
            if (xdot > cfg->bw_xdot_clamp) {
                xdot = cfg->bw_xdot_clamp;
            } else if (xdot < -cfg->bw_xdot_clamp) {
                xdot = -cfg->bw_xdot_clamp;
            }
        }

        double abs_z = fabs(z);
        double abs_z_pow = (cfg->bw_n == 1.0) ? abs_z : pow(abs_z, cfg->bw_n);
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

        double y_next = cfg->bw_alpha * x + (1.0 - cfg->bw_alpha) * z;
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

        y_next = clamp_state(cfg, y_next);
        output_state[i] = y_next;
        z_state[i] = z;
        x_prev[i] = x;

        double out_value = cfg->output_gain * y_next + cfg->output_bias;
        if (cfg->accumulate) {
            output_field[i] += add_scale * out_value;
        } else {
            output_field[i] = out_value;
        }
    }
}

static bool check_scalars(const double *got, const double *expected, size_t scalar_count,
                          const char *label) {
    for (size_t i = 0U; i < scalar_count; ++i) {
        if (!approx_equal(got[i], expected[i], 1.0e-12)) {
            fprintf(stderr, "FAIL: %s mismatch at %zu got=%.17g expected=%.17g\n", label, i, got[i],
                    expected[i]);
            return false;
        }
    }
    return true;
}

int main(void) {
    enum { kCount = 3, kScalarCount = 6 };
    static const SimComplexDouble input_step1[] = {
        {-0.6, 0.4},
        {0.25, -0.75},
        {-0.15, 0.9},
    };
    static const SimComplexDouble input_step2[] = {
        {0.7, -0.2},
        {-0.45, 0.55},
        {0.05, -1.1},
    };

    SimContext ctx;
    SimField input = {0};
    SimField output = {0};
    size_t shape[1] = {(size_t)kCount};
    size_t input_index = 0U;
    size_t output_index = 0U;
    size_t op_index = 0U;
    double expected_output[kScalarCount];
    double expected_state[kScalarCount];
    double expected_z[kScalarCount];
    double expected_x_prev[kScalarCount];
    const size_t scalar_count = (size_t)kScalarCount;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 1;
    }

    if (sim_field_init(&input, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&output, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    memcpy(sim_field_data(&input), input_step1, sizeof(input_step1));
    memset(sim_field_data(&output), 0, sim_field_bytes(&output));

    if (sim_context_add_field(&ctx, &input, &input_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimHystereticOperatorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_field = input_index;
    cfg.output_field = output_index;
    cfg.mode = SIM_HYSTERETIC_MODE_BOUC_WEN;
    cfg.threshold_mode = SIM_HYSTERETIC_THRESHOLD_BOUNDS;
    cfg.input_mode = SIM_HYSTERETIC_INPUT_ABS;
    cfg.input_gain = 0.75;
    cfg.input_bias = -0.1;
    cfg.threshold_low = -0.5;
    cfg.threshold_high = 0.5;
    cfg.threshold_center = 0.0;
    cfg.threshold_width = 1.0;
    cfg.output_low = -1.0;
    cfg.output_high = 1.0;
    cfg.state_min = -1.5;
    cfg.state_max = 1.5;
    cfg.smooth = 0.0;
    cfg.rate_limit = 0.0;
    cfg.accumulate = false;
    cfg.scale_by_dt = false;
    cfg.initialize_from_input = false;
    cfg.initial_output = 0.05;
    cfg.initial_input = -0.2;
    cfg.initial_z = 0.1;
    cfg.play_radius = 0.0;
    cfg.bw_alpha = 0.35;
    cfg.bw_A = 0.8;
    cfg.bw_beta = 0.25;
    cfg.bw_gamma = 0.15;
    cfg.bw_n = 2.0;
    cfg.bw_z_clamp = 0.0;
    cfg.bw_xdot_clamp = 0.0;
    cfg.output_gain = 1.1;
    cfg.output_bias = -0.03;

    if (sim_add_hysteretic_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_hysteretic_operator\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: operator lookup\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimOperatorInfo info = sim_operator_info(op);
    if (info.category != SIM_OPERATOR_CATEGORY_NONLINEAR ||
        info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
        !info.representation.requires_complex_input ||
        !info.representation.requires_complex_representation || info.preserves_real ||
        info.representation.preserves_real_subspace) {
        fprintf(stderr, "FAIL: complex metadata mismatch\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimHystereticOperatorConfig fetched;
    memset(&fetched, 0, sizeof(fetched));
    if (sim_hysteretic_config(&ctx, op_index, &fetched) != SIM_RESULT_OK ||
        fetched.mode != SIM_HYSTERETIC_MODE_BOUC_WEN ||
        fetched.input_mode != SIM_HYSTERETIC_INPUT_ABS ||
        !approx_equal(fetched.bw_alpha, cfg.bw_alpha, 1.0e-12)) {
        fprintf(stderr, "FAIL: sim_hysteretic_config\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    initialize_expected_state(&cfg, (const double *)input_step1, expected_state, expected_z,
                              expected_x_prev, scalar_count);
    memset(expected_output, 0, sizeof(expected_output));

    sim_context_set_timestep(&ctx, 0.2);
    if (sim_context_prepare_plan(&ctx) != SIM_RESULT_OK ||
        sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: first execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    apply_expected_bouc_wen(&cfg, (const double *)input_step1, expected_output, expected_state,
                            expected_z, expected_x_prev, scalar_count, 0.2);

    if (!check_scalars((const double *)sim_field_data(sim_context_field(&ctx, output_index)),
                       expected_output, scalar_count, "step1")) {
        sim_context_destroy(&ctx);
        return 1;
    }

    memcpy(sim_field_data(sim_context_field(&ctx, input_index)), input_step2, sizeof(input_step2));
    if (sim_context_prepare_plan(&ctx) != SIM_RESULT_OK ||
        sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: second execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    apply_expected_bouc_wen(&cfg, (const double *)input_step2, expected_output, expected_state,
                            expected_z, expected_x_prev, scalar_count, 0.2);

    if (!check_scalars((const double *)sim_field_data(sim_context_field(&ctx, output_index)),
                       expected_output, scalar_count, "step2")) {
        sim_context_destroy(&ctx);
        return 1;
    }

    sim_context_destroy(&ctx);
    return 0;
}
