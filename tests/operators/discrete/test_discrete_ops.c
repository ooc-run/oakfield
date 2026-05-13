// Tests for coordinate, elementwise math, and mask operators.
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static bool nearly_equal(double a, double b, double eps) {
    double diff = fabs(a - b);
    double scale = fmax(fabs(a), fabs(b));
    if (scale < eps) {
        scale = 1.0;
    }
    return diff <= eps * scale;
}

static bool run_coordinate_index(void) {
    SimContext context = {0};
    SimField output = {0};
    SimResult result;
    bool context_ready = false;
    bool output_ready = false;
    size_t output_index = SIZE_MAX;
    size_t shape[1] = {5U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate: field init\n");
        goto cleanup;
    }
    output_ready = true;

    if (sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate: add field\n");
        goto cleanup;
    }
    output_ready = false;

    SimCoordinateOperatorConfig config;
    (void)memset(&config, 0, sizeof(config));
    config.output_field = output_index;
    config.mode = SIM_COORD_MODE_INDEX;
    config.normalize = SIM_COORD_NORMALIZE_NONE;
    config.gain = 1.0;
    config.bias = 0.0;
    config.accumulate = false;
    config.scale_by_dt = false;

    result = sim_add_coordinate_operator(&context, &config, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate: add operator (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate: execute (%d)\n", (int)result);
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&context, output_index);
        const double *values = sim_field_real_data_const(out_field);
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (!nearly_equal(values[i], (double)i, 1.0e-9)) {
                fprintf(stderr, "[FAIL] coordinate: index %zu value %.6g\n", i, values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return false;
}

static double coordinate_raw_expected_value(const SimCoordinateOperatorConfig *config,
                                            const SimField *field, size_t index, double t) {
    double x = 0.0;
    double y = 0.0;

    if (config == NULL || field == NULL) {
        return 0.0;
    }
    if (config->mode == SIM_COORD_MODE_INDEX) {
        return (double)index;
    }
    if (sim_stimulus_coord_xy(&config->coord, field, index, &x, &y) != SIM_RESULT_OK) {
        return 0.0;
    }

    if (config->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
        return (config->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) ? (x + y) : (x * y);
    }

    return sim_stimulus_coord_u(&config->coord, x, y, t);
}

static bool coordinate_expected_range(const SimCoordinateOperatorConfig *config,
                                      const SimField *field, double t, double *out_min,
                                      double *out_max) {
    size_t count = 0U;

    if (config == NULL || field == NULL || out_min == NULL || out_max == NULL) {
        return false;
    }

    count = sim_field_element_count(&field->layout);
    if (count == 0U || config->normalize == SIM_COORD_NORMALIZE_NONE) {
        return false;
    }

    if (config->mode == SIM_COORD_MODE_INDEX) {
        if (count <= 1U) {
            return false;
        }

        *out_min = 0.0;
        *out_max = (double)(count - 1U);
        return true;
    }

    {
        double min_value = HUGE_VAL;
        double max_value = -HUGE_VAL;

        for (size_t i = 0U; i < count; ++i) {
            double value = coordinate_raw_expected_value(config, field, i, t);
            if (!isfinite(value)) {
                continue;
            }
            if (value < min_value) {
                min_value = value;
            }
            if (value > max_value) {
                max_value = value;
            }
        }

        if (!isfinite(min_value) || !isfinite(max_value) || max_value <= min_value) {
            return false;
        }

        *out_min = min_value;
        *out_max = max_value;
        return true;
    }
}

static double coordinate_expected_value(const SimCoordinateOperatorConfig *config,
                                        const SimField *field, size_t index, double t,
                                        bool normalize, double min_value, double max_value) {
    double value = coordinate_raw_expected_value(config, field, index, t);

    if (normalize) {
        value = (value - min_value) / (max_value - min_value);
        if (config->normalize == SIM_COORD_NORMALIZE_CENTERED) {
            value -= 0.5;
        } else if (config->normalize == SIM_COORD_NORMALIZE_SIGNED) {
            value = value * 2.0 - 1.0;
        }
    }

    return value * config->gain + config->bias;
}

static bool check_coordinate_output(const char *label, SimContext *context, size_t output_index,
                                    const SimCoordinateOperatorConfig *config, double base_time) {
    SimField *out_field = sim_context_field(context, output_index);
    bool normalize = false;
    double min_value = 0.0;
    double max_value = 0.0;
    if (out_field == NULL || config == NULL) {
        fprintf(stderr, "[FAIL] coordinate/%s: missing field or config\n", label);
        return false;
    }

    const double *values = sim_field_real_data_const(out_field);
    size_t count = sim_field_element_count(&out_field->layout);
    double t = base_time + config->time_offset;

    normalize = coordinate_expected_range(config, out_field, t, &min_value, &max_value);

    for (size_t i = 0U; i < count; ++i) {
        double expected =
            coordinate_expected_value(config, out_field, i, t, normalize, min_value, max_value);
        if (!nearly_equal(values[i], expected, 1.0e-9)) {
            fprintf(stderr, "[FAIL] coordinate/%s: index %zu value %.12g expected %.12g\n", label,
                    i, values[i], expected);
            return false;
        }
    }

    return true;
}

static bool run_coordinate_chart_modes(void) {
    typedef struct CoordinateCase {
        const char *label;
        double base_time;
        double time_offset;
        SimStimulusCoordConfig coord;
    } CoordinateCase;

    static const CoordinateCase cases[] = {{.label = "axis_y",
                                            .base_time = 0.0,
                                            .time_offset = 0.0,
                                            .coord = {.mode = SIM_STIMULUS_COORD_AXIS,
                                                      .axis = SIM_STIMULUS_AXIS_Y,
                                                      .origin_x = -0.75,
                                                      .origin_y = 0.5,
                                                      .spacing_x = 0.4,
                                                      .spacing_y = 0.3}},
                                           {.label = "angle",
                                            .base_time = 0.0,
                                            .time_offset = 0.35,
                                            .coord = {.mode = SIM_STIMULUS_COORD_ANGLE,
                                                      .angle = 0.6,
                                                      .origin_x = -1.0,
                                                      .origin_y = 0.75,
                                                      .spacing_x = 0.5,
                                                      .spacing_y = 0.25,
                                                      .velocity_x = 0.1,
                                                      .velocity_y = -0.2}},
                                           {.label = "radial",
                                            .base_time = 0.0,
                                            .time_offset = -0.2,
                                            .coord = {.mode = SIM_STIMULUS_COORD_RADIAL,
                                                      .origin_x = -0.5,
                                                      .origin_y = -0.25,
                                                      .spacing_x = 0.6,
                                                      .spacing_y = 0.5,
                                                      .center_x = 0.25,
                                                      .center_y = 0.1,
                                                      .velocity_x = -0.15,
                                                      .velocity_y = 0.05}},
                                           {.label = "polar",
                                            .base_time = 0.0,
                                            .time_offset = 0.1,
                                            .coord = {.mode = SIM_STIMULUS_COORD_POLAR,
                                                      .origin_x = -0.2,
                                                      .origin_y = -0.6,
                                                      .spacing_x = 0.55,
                                                      .spacing_y = 0.45,
                                                      .center_x = 0.3,
                                                      .center_y = -0.15,
                                                      .velocity_x = 0.08,
                                                      .velocity_y = 0.04}},
                                           {.label = "azimuth",
                                            .base_time = 0.0,
                                            .time_offset = 0.25,
                                            .coord = {.mode = SIM_STIMULUS_COORD_AZIMUTH,
                                                      .origin_x = -0.6,
                                                      .origin_y = -0.3,
                                                      .spacing_x = 0.4,
                                                      .spacing_y = 0.7,
                                                      .center_x = -0.1,
                                                      .center_y = 0.5,
                                                      .velocity_x = 0.12,
                                                      .velocity_y = -0.07}},
                                           {.label = "elliptic",
                                            .base_time = 0.0,
                                            .time_offset = -0.15,
                                            .coord = {.mode = SIM_STIMULUS_COORD_ELLIPTIC,
                                                      .angle = 0.4,
                                                      .origin_x = -0.8,
                                                      .origin_y = -0.4,
                                                      .spacing_x = 0.35,
                                                      .spacing_y = 0.5,
                                                      .center_x = 0.1,
                                                      .center_y = -0.2,
                                                      .velocity_x = -0.05,
                                                      .velocity_y = 0.09,
                                                      .ellipse_u = 1.7,
                                                      .ellipse_v = 0.8}},
                                           {.label = "spiral",
                                            .base_time = 0.0,
                                            .time_offset = 0.5,
                                            .coord = {.mode = SIM_STIMULUS_COORD_SPIRAL,
                                                      .origin_x = -0.3,
                                                      .origin_y = -0.2,
                                                      .spacing_x = 0.45,
                                                      .spacing_y = 0.55,
                                                      .center_x = 0.2,
                                                      .center_y = -0.1,
                                                      .velocity_x = 0.03,
                                                      .velocity_y = -0.06,
                                                      .spiral_arms = 1.5,
                                                      .spiral_pitch = 0.9,
                                                      .spiral_phase = 0.25,
                                                      .spiral_angular_velocity = 0.2}},
                                           {.label = "separable_add",
                                            .base_time = 0.0,
                                            .time_offset = 0.0,
                                            .coord = {.mode = SIM_STIMULUS_COORD_SEPARABLE,
                                                      .combine = SIM_STIMULUS_SEPARABLE_ADD,
                                                      .origin_x = -0.25,
                                                      .origin_y = 0.4,
                                                      .spacing_x = 0.6,
                                                      .spacing_y = 0.2}}};

    SimContext context = {0};
    SimField output = {0};
    SimResult result;
    bool context_ready = false;
    bool output_ready = false;
    size_t output_index = SIZE_MAX;
    size_t operator_index = SIZE_MAX;
    size_t shape[2] = {4U, 3U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate/chart: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&output, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate/chart: field init\n");
        goto cleanup;
    }
    output_ready = true;

    if (sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate/chart: add field\n");
        goto cleanup;
    }
    output_ready = false;

    for (size_t case_index = 0U; case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
        SimCoordinateOperatorConfig config;
        CoordinateCase current = cases[case_index];

        (void)memset(&config, 0, sizeof(config));
        config.output_field = output_index;
        config.mode = SIM_COORD_MODE_COORD;
        config.normalize = SIM_COORD_NORMALIZE_NONE;
        config.coord = current.coord;
        config.gain = 1.0;
        config.bias = 0.0;
        config.time_offset = current.time_offset;
        config.accumulate = false;
        config.scale_by_dt = false;

        sim_stimulus_coord_normalize(&config.coord);

        if (case_index == 0U) {
            result = sim_add_coordinate_operator(&context, &config, &operator_index);
            if (result != SIM_RESULT_OK) {
                fprintf(stderr, "[FAIL] coordinate/%s: add operator (%d)\n", current.label,
                        (int)result);
                goto cleanup;
            }
        } else {
            result = sim_coordinate_update(&context, operator_index, &config);
            if (result != SIM_RESULT_OK) {
                fprintf(stderr, "[FAIL] coordinate/%s: update operator (%d)\n", current.label,
                        (int)result);
                goto cleanup;
            }
        }

        result = sim_context_execute(&context);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] coordinate/%s: execute (%d)\n", current.label, (int)result);
            goto cleanup;
        }

        if (!check_coordinate_output(current.label, &context, output_index, &config,
                                     current.base_time)) {
            goto cleanup;
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return false;
}

static bool run_coordinate_normalized_modes(void) {
    typedef struct CoordinateNormalizeCase {
        const char *label;
        SimCoordinateMode mode;
        SimCoordinateNormalizeMode normalize;
        double base_time;
        double time_offset;
        double gain;
        double bias;
        SimStimulusCoordConfig coord;
    } CoordinateNormalizeCase;

    static const CoordinateNormalizeCase cases[] = {
        {.label = "unit_radial",
         .mode = SIM_COORD_MODE_COORD,
         .normalize = SIM_COORD_NORMALIZE_UNIT,
         .base_time = 0.0,
         .time_offset = 0.18,
         .gain = 1.0,
         .bias = 0.0,
         .coord = {.mode = SIM_STIMULUS_COORD_RADIAL,
                   .origin_x = -1.25,
                   .origin_y = -0.9,
                   .spacing_x = 0.4,
                   .spacing_y = 0.2,
                   .center_x = 0.35,
                   .center_y = -0.15,
                   .velocity_x = 0.07,
                   .velocity_y = -0.04}},
        {.label = "centered_separable_affine",
         .mode = SIM_COORD_MODE_COORD,
         .normalize = SIM_COORD_NORMALIZE_CENTERED,
         .base_time = 0.0,
         .time_offset = -0.12,
         .gain = 1.75,
         .bias = -0.2,
         .coord = {.mode = SIM_STIMULUS_COORD_SEPARABLE,
                   .combine = SIM_STIMULUS_SEPARABLE_ADD,
                   .origin_x = -0.8,
                   .origin_y = 0.3,
                   .spacing_x = 0.35,
                   .spacing_y = 0.27,
                   .velocity_x = -0.05,
                   .velocity_y = 0.08}},
        {.label = "signed_index_affine",
         .mode = SIM_COORD_MODE_INDEX,
         .normalize = SIM_COORD_NORMALIZE_SIGNED,
         .base_time = 0.0,
         .time_offset = 0.0,
         .gain = 0.6,
         .bias = 0.1,
         .coord = {0}}};

    SimContext context = {0};
    SimField output = {0};
    SimResult result;
    bool context_ready = false;
    bool output_ready = false;
    size_t output_index = SIZE_MAX;
    size_t operator_index = SIZE_MAX;
    size_t shape[2] = {65U, 7U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate/normalize: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&output, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate/normalize: field init\n");
        goto cleanup;
    }
    output_ready = true;

    if (sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] coordinate/normalize: add field\n");
        goto cleanup;
    }
    output_ready = false;

    for (size_t case_index = 0U; case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
        SimCoordinateOperatorConfig config;
        CoordinateNormalizeCase current = cases[case_index];

        (void)memset(&config, 0, sizeof(config));
        config.output_field = output_index;
        config.mode = current.mode;
        config.normalize = current.normalize;
        config.coord = current.coord;
        config.gain = current.gain;
        config.bias = current.bias;
        config.time_offset = current.time_offset;
        config.accumulate = false;
        config.scale_by_dt = false;

        sim_stimulus_coord_normalize(&config.coord);

        if (case_index == 0U) {
            result = sim_add_coordinate_operator(&context, &config, &operator_index);
            if (result != SIM_RESULT_OK) {
                fprintf(stderr, "[FAIL] coordinate/%s: add operator (%d)\n", current.label,
                        (int)result);
                goto cleanup;
            }
        } else {
            result = sim_coordinate_update(&context, operator_index, &config);
            if (result != SIM_RESULT_OK) {
                fprintf(stderr, "[FAIL] coordinate/%s: update operator (%d)\n", current.label,
                        (int)result);
                goto cleanup;
            }
        }

        result = sim_context_execute(&context);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] coordinate/%s: execute (%d)\n", current.label, (int)result);
            goto cleanup;
        }

        if (!check_coordinate_output(current.label, &context, output_index, &config,
                                     current.base_time)) {
            goto cleanup;
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return false;
}

static bool run_elementwise_math_mod(void) {
    SimContext context = {0};
    SimField input = {0};
    SimField output = {0};
    SimResult result;
    bool context_ready = false;
    bool input_ready = false;
    bool output_ready = false;
    size_t input_index = SIZE_MAX;
    size_t output_index = SIZE_MAX;
    size_t shape[1] = {5U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: input init\n");
        goto cleanup;
    }
    input_ready = true;

    if (sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: output init\n");
        goto cleanup;
    }
    output_ready = true;

    {
        double *data = sim_field_real_data(&input);
        for (size_t i = 0U; i < shape[0]; ++i) {
            data[i] = (double)i;
        }
    }

    if (sim_context_add_field(&context, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: add input\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: add output\n");
        goto cleanup;
    }
    output_ready = false;

    SimElementwiseMathOperatorConfig config;
    (void)memset(&config, 0, sizeof(config));
    config.lhs_field = input_index;
    config.rhs_field = input_index;
    config.output_field = output_index;
    config.mode = SIM_ELEMENTWISE_MATH_MOD;
    config.rhs_source = SIM_ELEMENTWISE_MATH_RHS_CONSTANT;
    config.rhs_constant = 2.0;
    config.epsilon = 1.0e-6;
    config.accumulate = false;
    config.scale_by_dt = false;

    result = sim_add_elementwise_math_operator(&context, &config, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: add operator (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] math: execute (%d)\n", (int)result);
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&context, output_index);
        const double *values = sim_field_real_data_const(out_field);
        const double expected[] = {0.0, 1.0, 0.0, 1.0, 0.0};
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (!nearly_equal(values[i], expected[i], 1.0e-9)) {
                fprintf(stderr, "[FAIL] math: index %zu value %.6g\n", i, values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return false;
}

static bool run_integer_coordinate_and_math(void) {
    SimContext context = {0};
    SimField coordinate = {0};
    SimField modulo = {0};
    SimField eq_mask = {0};
    SimField gt_mask = {0};
    SimResult result;
    bool context_ready = false;
    bool coordinate_ready = false;
    bool modulo_ready = false;
    bool eq_mask_ready = false;
    bool gt_mask_ready = false;
    size_t coordinate_index = SIZE_MAX;
    size_t modulo_index = SIZE_MAX;
    size_t eq_mask_index = SIZE_MAX;
    size_t gt_mask_index = SIZE_MAX;
    size_t shape[1] = {5U};
    const uint64_t base = UINT64_C(9007199254740997);
    const uint64_t expected_mod[] = {2U, 3U, 4U, 5U, 6U};
    const double expected_eq[] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const double expected_gt[] = {0.0, 1.0, 1.0, 1.0, 1.0};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer math: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init_typed(&coordinate, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&modulo, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init(&eq_mask, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&gt_mask, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer math: field init\n");
        goto cleanup;
    }
    coordinate_ready = modulo_ready = eq_mask_ready = gt_mask_ready = true;

    if (sim_context_add_field(&context, &coordinate, &coordinate_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &modulo, &modulo_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &eq_mask, &eq_mask_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &gt_mask, &gt_mask_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer math: add field\n");
        goto cleanup;
    }
    coordinate_ready = modulo_ready = eq_mask_ready = gt_mask_ready = false;

    {
        SimCoordinateOperatorConfig cfg;
        (void)memset(&cfg, 0, sizeof(cfg));
        cfg.output_field = coordinate_index;
        cfg.mode = SIM_COORD_MODE_INDEX;
        cfg.normalize = SIM_COORD_NORMALIZE_NONE;
        cfg.gain = 1.0;
        cfg.bias = 0.0;
        cfg.scale_by_dt = false;
        cfg.exact_gain_enabled = true;
        cfg.exact_gain_raw = 1U;
        cfg.exact_bias_enabled = true;
        cfg.exact_bias_raw = base;
        result = sim_add_coordinate_operator(&context, &cfg, NULL);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] integer math: add coordinate (%d)\n", (int)result);
            goto cleanup;
        }
    }

    {
        SimElementwiseMathOperatorConfig cfg;
        (void)memset(&cfg, 0, sizeof(cfg));
        cfg.lhs_field = coordinate_index;
        cfg.rhs_field = coordinate_index;
        cfg.output_field = modulo_index;
        cfg.mode = SIM_ELEMENTWISE_MATH_MOD;
        cfg.rhs_source = SIM_ELEMENTWISE_MATH_RHS_CONSTANT;
        cfg.rhs_constant = 7.0;
        cfg.scale_by_dt = false;
        result = sim_add_elementwise_math_operator(&context, &cfg, NULL);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] integer math: add mod (%d)\n", (int)result);
            goto cleanup;
        }
    }

    {
        SimElementwiseMathOperatorConfig cfg;
        (void)memset(&cfg, 0, sizeof(cfg));
        cfg.lhs_field = modulo_index;
        cfg.rhs_field = modulo_index;
        cfg.output_field = eq_mask_index;
        cfg.mode = SIM_ELEMENTWISE_MATH_EQ;
        cfg.rhs_source = SIM_ELEMENTWISE_MATH_RHS_CONSTANT;
        cfg.rhs_constant = 0.0;
        cfg.true_value = 1.0;
        cfg.false_value = 0.0;
        cfg.scale_by_dt = false;
        result = sim_add_elementwise_math_operator(&context, &cfg, NULL);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] integer math: add eq (%d)\n", (int)result);
            goto cleanup;
        }
    }

    {
        SimElementwiseMathOperatorConfig cfg;
        (void)memset(&cfg, 0, sizeof(cfg));
        cfg.lhs_field = modulo_index;
        cfg.rhs_field = modulo_index;
        cfg.output_field = gt_mask_index;
        cfg.mode = SIM_ELEMENTWISE_MATH_GT;
        cfg.rhs_source = SIM_ELEMENTWISE_MATH_RHS_CONSTANT;
        cfg.rhs_constant = 2.0;
        cfg.true_value = 1.0;
        cfg.false_value = 0.0;
        cfg.scale_by_dt = false;
        result = sim_add_elementwise_math_operator(&context, &cfg, NULL);
        if (result != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] integer math: add gt (%d)\n", (int)result);
            goto cleanup;
        }
    }

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer math: execute (%d)\n", (int)result);
        goto cleanup;
    }

    {
        const uint64_t *coord_values =
            sim_field_u64_data_const(sim_context_field(&context, coordinate_index));
        const uint64_t *mod_values =
            sim_field_u64_data_const(sim_context_field(&context, modulo_index));
        const double *eq_values =
            sim_field_real_data_const(sim_context_field(&context, eq_mask_index));
        const double *gt_values =
            sim_field_real_data_const(sim_context_field(&context, gt_mask_index));
        if (coord_values == NULL || mod_values == NULL || eq_values == NULL || gt_values == NULL) {
            fprintf(stderr, "[FAIL] integer math: output data missing\n");
            goto cleanup;
        }
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (coord_values[i] != base + i) {
                fprintf(stderr, "[FAIL] integer math: coord[%zu]=%llu\n", i,
                        (unsigned long long)coord_values[i]);
                goto cleanup;
            }
            if (mod_values[i] != expected_mod[i]) {
                fprintf(stderr, "[FAIL] integer math: mod[%zu]=%llu\n", i,
                        (unsigned long long)mod_values[i]);
                goto cleanup;
            }
            if (!nearly_equal(eq_values[i], expected_eq[i], 1.0e-9)) {
                fprintf(stderr, "[FAIL] integer math: eq[%zu]=%.6g\n", i, eq_values[i]);
                goto cleanup;
            }
            if (!nearly_equal(gt_values[i], expected_gt[i], 1.0e-9)) {
                fprintf(stderr, "[FAIL] integer math: gt[%zu]=%.6g\n", i, gt_values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (coordinate_ready) {
        sim_field_destroy(&coordinate);
    }
    if (modulo_ready) {
        sim_field_destroy(&modulo);
    }
    if (eq_mask_ready) {
        sim_field_destroy(&eq_mask);
    }
    if (gt_mask_ready) {
        sim_field_destroy(&gt_mask);
    }
    return false;
}

static bool run_mask_apply(void) {
    SimContext context = {0};
    SimField input = {0};
    SimField mask = {0};
    SimField output = {0};
    SimResult result;
    bool context_ready = false;
    bool input_ready = false;
    bool mask_ready = false;
    bool output_ready = false;
    size_t input_index = SIZE_MAX;
    size_t mask_index = SIZE_MAX;
    size_t output_index = SIZE_MAX;
    size_t shape[1] = {4U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init(&input, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: input init\n");
        goto cleanup;
    }
    input_ready = true;

    if (sim_field_init(&mask, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: mask init\n");
        goto cleanup;
    }
    mask_ready = true;

    if (sim_field_init(&output, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: output init\n");
        goto cleanup;
    }
    output_ready = true;

    {
        double *input_data = sim_field_real_data(&input);
        double *mask_data = sim_field_real_data(&mask);
        input_data[0] = 1.0;
        input_data[1] = 2.0;
        input_data[2] = 3.0;
        input_data[3] = 4.0;
        mask_data[0] = 0.0;
        mask_data[1] = 1.0;
        mask_data[2] = 0.0;
        mask_data[3] = 1.0;
    }

    if (sim_context_add_field(&context, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: add input\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_context_add_field(&context, &mask, &mask_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: add mask\n");
        goto cleanup;
    }
    mask_ready = false;

    if (sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: add output\n");
        goto cleanup;
    }
    output_ready = false;

    SimMaskOperatorConfig config;
    (void)memset(&config, 0, sizeof(config));
    config.input_field = input_index;
    config.mask_field = mask_index;
    config.output_field = output_index;
    config.mode = SIM_MASK_MODE_APPLY;
    config.threshold = 0.5;
    config.feather = 0.0;
    config.fill_value = 0.0;
    config.accumulate = false;
    config.scale_by_dt = false;

    result = sim_add_mask_operator(&context, &config, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: add operator (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] mask: execute (%d)\n", (int)result);
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&context, output_index);
        const double *values = sim_field_real_data_const(out_field);
        const double expected[] = {0.0, 2.0, 0.0, 4.0};
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (!nearly_equal(values[i], expected[i], 1.0e-9)) {
                fprintf(stderr, "[FAIL] mask: index %zu value %.6g\n", i, values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (mask_ready) {
        sim_field_destroy(&mask);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return false;
}

static bool run_mask_apply_integer(void) {
    SimContext context = {0};
    SimField input = {0};
    SimField mask = {0};
    SimField output = {0};
    SimResult result;
    bool context_ready = false;
    bool input_ready = false;
    bool mask_ready = false;
    bool output_ready = false;
    size_t input_index = SIZE_MAX;
    size_t mask_index = SIZE_MAX;
    size_t output_index = SIZE_MAX;
    size_t shape[1] = {4U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init_typed(&input, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: input init\n");
        goto cleanup;
    }
    input_ready = true;

    if (sim_field_init(&mask, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: mask init\n");
        goto cleanup;
    }
    mask_ready = true;

    if (sim_field_init_typed(&output, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: output init\n");
        goto cleanup;
    }
    output_ready = true;

    {
        uint64_t *input_data = sim_field_u64_data(&input);
        double *mask_data = sim_field_real_data(&mask);
        input_data[0] = 11U;
        input_data[1] = 13U;
        input_data[2] = 17U;
        input_data[3] = 19U;
        mask_data[0] = 0.0;
        mask_data[1] = 1.0;
        mask_data[2] = 0.0;
        mask_data[3] = 1.0;
    }

    if (sim_context_add_field(&context, &input, &input_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: add input\n");
        goto cleanup;
    }
    input_ready = false;

    if (sim_context_add_field(&context, &mask, &mask_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: add mask\n");
        goto cleanup;
    }
    mask_ready = false;

    if (sim_context_add_field(&context, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: add output\n");
        goto cleanup;
    }
    output_ready = false;

    SimMaskOperatorConfig config;
    (void)memset(&config, 0, sizeof(config));
    config.input_field = input_index;
    config.mask_field = mask_index;
    config.output_field = output_index;
    config.mode = SIM_MASK_MODE_APPLY;
    config.threshold = 0.5;
    config.feather = 0.0;
    config.fill_value = 0.0;
    config.accumulate = false;
    config.scale_by_dt = false;

    result = sim_add_mask_operator(&context, &config, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: add operator (%d)\n", (int)result);
        goto cleanup;
    }

    result = sim_context_execute(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integer mask: execute (%d)\n", (int)result);
        goto cleanup;
    }

    {
        SimField *out_field = sim_context_field(&context, output_index);
        const uint64_t *values = sim_field_u64_data_const(out_field);
        const uint64_t expected[] = {0U, 13U, 0U, 19U};
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (values[i] != expected[i]) {
                fprintf(stderr, "[FAIL] integer mask: index %zu value %llu\n", i,
                        (unsigned long long)values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (input_ready) {
        sim_field_destroy(&input);
    }
    if (mask_ready) {
        sim_field_destroy(&mask);
    }
    if (output_ready) {
        sim_field_destroy(&output);
    }
    return false;
}

static bool run_segmented_sieve_mark(void) {
    SimContext context = {0};
    SimField candidate = {0};
    SimField primes = {0};
    SimField flags = {0};
    size_t shape[1] = {32U};
    size_t primes_shape[1] = {3U};
    bool context_ready = false;
    bool candidate_ready = false;
    bool primes_ready = false;
    bool flags_ready = false;
    size_t candidate_index = SIZE_MAX;
    size_t primes_index = SIZE_MAX;
    size_t flags_index = SIZE_MAX;
    const uint8_t expected[] = {0U, 0U, 1U, 1U, 0U, 1U, 0U, 1U, 0U, 0U, 0U, 1U, 0U, 1U, 0U, 0U,
                                0U, 1U, 0U, 1U, 0U, 0U, 0U, 1U, 0U, 0U, 0U, 0U, 0U, 1U, 0U, 1U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init_typed(&candidate, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&primes, 1U, primes_shape, sim_scalar_domain_u32(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&flags, 1U, shape, sim_scalar_domain_u8(), SIM_FIELD_STORAGE_ROW_MAJOR,
                             NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark: field init\n");
        goto cleanup;
    }
    candidate_ready = true;
    primes_ready = true;
    flags_ready = true;

    {
        uint64_t *candidate_data = sim_field_u64_data(&candidate);
        uint32_t *prime_data = sim_field_u32_data(&primes);
        uint8_t *flag_data = sim_field_u8_data(&flags);
        for (size_t i = 0U; i < shape[0]; ++i) {
            candidate_data[i] = (uint64_t)i;
            flag_data[i] = (i >= 2U) ? 1U : 0U;
        }
        prime_data[0] = 2U;
        prime_data[1] = 3U;
        prime_data[2] = 5U;
    }

    if (sim_context_add_field(&context, &candidate, &candidate_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &primes, &primes_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &flags, &flags_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark: add field\n");
        goto cleanup;
    }
    candidate_ready = false;
    primes_ready = false;
    flags_ready = false;

    {
        SimSegmentedSieveMarkBatchOperatorConfig config = {0};
        config.candidate_field = candidate_index;
        config.primes_field = primes_index;
        config.flags_field = flags_index;
        if (sim_add_segmented_sieve_mark_batch_operator(&context, &config, NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] segmented sieve mark: add batch operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark: execute\n");
        goto cleanup;
    }

    {
        const uint8_t *values = sim_field_u8_data_const(sim_context_field(&context, flags_index));
        for (size_t i = 0U; i < shape[0]; ++i) {
            if (values[i] != expected[i]) {
                fprintf(stderr, "[FAIL] segmented sieve mark: index %zu value %llu\n", i,
                        (unsigned long long)values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (candidate_ready) {
        sim_field_destroy(&candidate);
    }
    if (primes_ready) {
        sim_field_destroy(&primes);
    }
    if (flags_ready) {
        sim_field_destroy(&flags);
    }
    return false;
}

static bool run_segmented_sieve_mark_float_flags_rejected(void) {
    SimContext context = {0};
    SimField candidate = {0};
    SimField primes = {0};
    SimField flags = {0};
    size_t shape[1] = {16U};
    size_t primes_shape[1] = {3U};
    bool context_ready = false;
    bool candidate_ready = false;
    bool primes_ready = false;
    bool flags_ready = false;
    size_t candidate_index = SIZE_MAX;
    size_t primes_index = SIZE_MAX;
    size_t flags_index = SIZE_MAX;
    SimResult result = SIM_RESULT_OK;

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark float flags rejected: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init_typed(&candidate, 1U, shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&primes, 1U, primes_shape, sim_scalar_domain_u32(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init(&flags, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark float flags rejected: field init\n");
        goto cleanup;
    }
    candidate_ready = true;
    primes_ready = true;
    flags_ready = true;

    {
        uint64_t *candidate_data = sim_field_u64_data(&candidate);
        uint32_t *prime_data = sim_field_u32_data(&primes);
        double *flag_data = sim_field_real_data(&flags);
        for (size_t i = 0U; i < shape[0]; ++i) {
            candidate_data[i] = (uint64_t)i;
            flag_data[i] = (i >= 2U) ? 1.0 : 0.0;
        }
        prime_data[0] = 2U;
        prime_data[1] = 3U;
        prime_data[2] = 5U;
    }

    if (sim_context_add_field(&context, &candidate, &candidate_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &primes, &primes_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &flags, &flags_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark float flags rejected: add field\n");
        goto cleanup;
    }
    candidate_ready = false;
    primes_ready = false;
    flags_ready = false;

    {
        SimSegmentedSieveMarkBatchOperatorConfig config = {0};
        config.candidate_field = candidate_index;
        config.primes_field = primes_index;
        config.flags_field = flags_index;
        result = sim_add_segmented_sieve_mark_batch_operator(&context, &config, NULL);
        if (result != SIM_RESULT_TYPE_MISMATCH) {
            fprintf(stderr,
                    "[FAIL] segmented sieve mark float flags rejected: expected TYPE_MISMATCH (got "
                    "%d)\n",
                    (int)result);
            goto cleanup;
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (candidate_ready) {
        sim_field_destroy(&candidate);
    }
    if (primes_ready) {
        sim_field_destroy(&primes);
    }
    if (flags_ready) {
        sim_field_destroy(&flags);
    }
    return false;
}

static bool run_segmented_sieve_mark_sparse_candidates(void) {
    SimContext context = {0};
    SimField candidate = {0};
    SimField primes = {0};
    SimField flags = {0};
    size_t candidate_shape[1] = {7U};
    size_t primes_shape[1] = {4U};
    bool context_ready = false;
    bool candidate_ready = false;
    bool primes_ready = false;
    bool flags_ready = false;
    size_t candidate_index = SIZE_MAX;
    size_t primes_index = SIZE_MAX;
    size_t flags_index = SIZE_MAX;
    static const uint64_t candidate_values[] = {2U, 3U, 4U, 9U, 25U, 49U, 51U};
    static const uint8_t expected_flags[] = {1U, 1U, 0U, 0U, 0U, 0U, 0U};

    if (sim_context_init(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark sparse: sim_context_init\n");
        return false;
    }
    context_ready = true;

    if (sim_field_init_typed(&candidate, 1U, candidate_shape, sim_scalar_domain_u64(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&primes, 1U, primes_shape, sim_scalar_domain_u32(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK ||
        sim_field_init_typed(&flags, 1U, candidate_shape, sim_scalar_domain_u8(),
                             SIM_FIELD_STORAGE_ROW_MAJOR, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark sparse: field init\n");
        goto cleanup;
    }
    candidate_ready = true;
    primes_ready = true;
    flags_ready = true;

    {
        uint64_t *candidate_data = sim_field_u64_data(&candidate);
        uint32_t *prime_data = sim_field_u32_data(&primes);
        uint8_t *flag_data = sim_field_u8_data(&flags);

        memcpy(candidate_data, candidate_values, sizeof(candidate_values));
        for (size_t i = 0U; i < candidate_shape[0]; ++i) {
            flag_data[i] = 1U;
        }
        prime_data[0] = 2U;
        prime_data[1] = 3U;
        prime_data[2] = 5U;
        prime_data[3] = 7U;
    }

    if (sim_context_add_field(&context, &candidate, &candidate_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &primes, &primes_index) != SIM_RESULT_OK ||
        sim_context_add_field(&context, &flags, &flags_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark sparse: add field\n");
        goto cleanup;
    }
    candidate_ready = false;
    primes_ready = false;
    flags_ready = false;

    {
        SimSegmentedSieveMarkBatchOperatorConfig config = {0};
        config.candidate_field = candidate_index;
        config.primes_field = primes_index;
        config.flags_field = flags_index;
        if (sim_add_segmented_sieve_mark_batch_operator(&context, &config, NULL) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] segmented sieve mark sparse: add batch operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&context) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] segmented sieve mark sparse: execute\n");
        goto cleanup;
    }

    {
        const uint8_t *values = sim_field_u8_data_const(sim_context_field(&context, flags_index));
        for (size_t i = 0U; i < candidate_shape[0]; ++i) {
            if (values[i] != expected_flags[i]) {
                fprintf(stderr, "[FAIL] segmented sieve mark sparse: index %zu value %llu\n", i,
                        (unsigned long long)values[i]);
                goto cleanup;
            }
        }
    }

    sim_context_destroy(&context);
    return true;

cleanup:
    if (context_ready) {
        sim_context_destroy(&context);
    }
    if (candidate_ready) {
        sim_field_destroy(&candidate);
    }
    if (primes_ready) {
        sim_field_destroy(&primes);
    }
    if (flags_ready) {
        sim_field_destroy(&flags);
    }
    return false;
}

int main(void) {
    bool ok = true;

    ok = run_coordinate_index() && ok;
    ok = run_coordinate_chart_modes() && ok;
    ok = run_coordinate_normalized_modes() && ok;
    ok = run_elementwise_math_mod() && ok;
    ok = run_integer_coordinate_and_math() && ok;
    ok = run_mask_apply() && ok;
    ok = run_mask_apply_integer() && ok;
    ok = run_segmented_sieve_mark() && ok;
    ok = run_segmented_sieve_mark_float_flags_rejected() && ok;
    ok = run_segmented_sieve_mark_sparse_candidates() && ok;

    if (ok) {
        printf("[PASS] discrete operators\n");
        return 0;
    }

    printf("[FAIL] discrete operators\n");
    return 1;
}
