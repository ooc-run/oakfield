/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/math/xi.h>
#include <oakfield/math/zeta.h>
#include <oakfield/sim.h>

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int approx(double a, double b, double eps) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= eps * scale;
}

static int status_usable(SimZetaStatus status) {
    return status != SIM_ZETA_STATUS_INVALID_ARGUMENT && status != SIM_ZETA_STATUS_SINGULAR &&
           status != SIM_ZETA_STATUS_NUMERIC_FAILURE;
}

typedef struct PlaneChartMap {
    SimPlaneSamplingFrame frame;
    SimPlaneChartConfig chart;
    double primary_min;
    double primary_max;
    double secondary_min;
    double secondary_max;
    int valid;
} PlaneChartMap;

static int build_chart_frame_2d(const size_t *shape, const SimStimulusZetaPlaneSliceConfig *cfg,
                                SimPlaneSamplingFrame *out_frame) {
    if (shape == NULL || cfg == NULL || out_frame == NULL || shape[0] == 0U || shape[1] == 0U) {
        return 0;
    }

    *out_frame = (SimPlaneSamplingFrame){
        .origin_x = (shape[1] > 1U) ? -0.5 : 0.0,
        .origin_y = (shape[0] > 1U) ? 0.5 : 0.0,
        .spacing_x = (shape[1] > 1U) ? (1.0 / (double)(shape[1] - 1U)) : 1.0,
        .spacing_y = (shape[0] > 1U) ? (-1.0 / (double)(shape[0] - 1U)) : 1.0,
        .center_x = cfg->chart_center_x,
        .center_y = cfg->chart_center_y,
        .velocity_x = 0.0,
        .velocity_y = 0.0,
    };
    sim_plane_sampling_frame_normalize(out_frame);
    return 1;
}

static int build_chart_frame_1d(size_t width, const SimStimulusZetaPlaneSliceConfig *cfg,
                                SimPlaneSamplingFrame *out_frame) {
    if (cfg == NULL || out_frame == NULL || width == 0U) {
        return 0;
    }

    *out_frame = (SimPlaneSamplingFrame){
        .origin_x = (width > 1U) ? -0.5 : 0.0,
        .origin_y = 0.0,
        .spacing_x = (width > 1U) ? (1.0 / (double)(width - 1U)) : 1.0,
        .spacing_y = 1.0,
        .center_x = cfg->chart_center_x,
        .center_y = cfg->chart_center_y,
        .velocity_x = 0.0,
        .velocity_y = 0.0,
    };
    sim_plane_sampling_frame_normalize(out_frame);
    return 1;
}

static SimPlaneChartConfig
plane_chart_config_from_slice(const SimStimulusZetaPlaneSliceConfig *cfg) {
    SimPlaneChartConfig chart = {
        .kind = cfg->chart_kind,
        .secondary_wrap = (cfg->chart_kind == SIM_PLANE_CHART_CARTESIAN)
                              ? SIM_PLANE_CHART_WRAP_NONE
                              : SIM_PLANE_CHART_WRAP_SIGNED_ANGLE,
        .rotation = cfg->chart_rotation,
        .ellipse_u = cfg->chart_ellipse_u,
        .ellipse_v = cfg->chart_ellipse_v,
        .spiral_arms = cfg->chart_spiral_arms,
        .spiral_pitch = cfg->chart_spiral_pitch,
        .spiral_phase = cfg->chart_spiral_phase,
        .spiral_angular_velocity = cfg->chart_spiral_angular_velocity,
    };
    sim_plane_chart_normalize(&chart);
    return chart;
}

static double normalize_chart_range(double value, double min_value, double max_value) {
    double span = max_value - min_value;

    if (!isfinite(value) || !isfinite(min_value) || !isfinite(max_value) || fabs(span) <= 1.0e-9) {
        return 0.0;
    }

    return ((value - min_value) / span) - 0.5;
}

static int project_chart_axis(SimPlaneChartCoord coord, SimPlaneProjectionKind projection_kind,
                              double *out_value) {
    SimPlaneProjectionConfig projection = {.kind = projection_kind};
    SimPlaneProjectionValue value = {0.0, 0.0, false};

    sim_plane_projection_normalize(&projection);
    if (projection.kind == SIM_PLANE_PROJECTION_FULL) {
        projection.kind = SIM_PLANE_PROJECTION_PRIMARY;
    }

    if (out_value == NULL ||
        sim_plane_projection_eval(&projection, coord, &value) != SIM_PLANE_CHART_STATUS_OK ||
        !isfinite(value.primary)) {
        return 0;
    }

    *out_value = value.primary;
    return 1;
}

static int projection_range(const PlaneChartMap *map, SimPlaneProjectionKind projection_kind,
                            double *out_min, double *out_max) {
    SimPlaneProjectionConfig projection = {.kind = projection_kind};

    if (map == NULL || out_min == NULL || out_max == NULL) {
        return 0;
    }

    sim_plane_projection_normalize(&projection);
    if (projection.kind == SIM_PLANE_PROJECTION_FULL) {
        projection.kind = SIM_PLANE_PROJECTION_PRIMARY;
    }

    if (projection.kind == SIM_PLANE_PROJECTION_SECONDARY) {
        *out_min = map->secondary_min;
        *out_max = map->secondary_max;
    } else {
        *out_min = map->primary_min;
        *out_max = map->primary_max;
    }
    return 1;
}

static int build_chart_map_2d(const size_t *shape, const SimStimulusZetaPlaneSliceConfig *cfg,
                              PlaneChartMap *out_map) {
    if (shape == NULL || cfg == NULL || out_map == NULL ||
        !build_chart_frame_2d(shape, cfg, &out_map->frame)) {
        return 0;
    }

    out_map->chart = plane_chart_config_from_slice(cfg);
    out_map->primary_min = HUGE_VAL;
    out_map->primary_max = -HUGE_VAL;
    out_map->secondary_min = HUGE_VAL;
    out_map->secondary_max = -HUGE_VAL;
    out_map->valid = 0;

    for (size_t row = 0U; row < shape[0]; ++row) {
        for (size_t col = 0U; col < shape[1]; ++col) {
            double x = out_map->frame.origin_x + (double)col * out_map->frame.spacing_x;
            double y = out_map->frame.origin_y + (double)row * out_map->frame.spacing_y;
            SimPlaneChartCoord coord = {0.0, 0.0};

            if (sim_plane_chart_eval(&out_map->frame, &out_map->chart, x, y, 0.0, &coord) !=
                    SIM_PLANE_CHART_STATUS_OK ||
                !isfinite(coord.primary) || !isfinite(coord.secondary)) {
                return 0;
            }

            if (coord.primary < out_map->primary_min) {
                out_map->primary_min = coord.primary;
            }
            if (coord.primary > out_map->primary_max) {
                out_map->primary_max = coord.primary;
            }
            if (coord.secondary < out_map->secondary_min) {
                out_map->secondary_min = coord.secondary;
            }
            if (coord.secondary > out_map->secondary_max) {
                out_map->secondary_max = coord.secondary;
            }
            out_map->valid = 1;
        }
    }

    return out_map->valid;
}

static int build_chart_map_1d(size_t width, const SimStimulusZetaPlaneSliceConfig *cfg,
                              PlaneChartMap *out_map) {
    if (cfg == NULL || out_map == NULL || !build_chart_frame_1d(width, cfg, &out_map->frame)) {
        return 0;
    }

    out_map->chart = plane_chart_config_from_slice(cfg);
    out_map->primary_min = HUGE_VAL;
    out_map->primary_max = -HUGE_VAL;
    out_map->secondary_min = HUGE_VAL;
    out_map->secondary_max = -HUGE_VAL;
    out_map->valid = 0;

    for (size_t col = 0U; col < width; ++col) {
        double x = out_map->frame.origin_x + (double)col * out_map->frame.spacing_x;
        SimPlaneChartCoord coord = {0.0, 0.0};

        if (sim_plane_chart_eval(&out_map->frame, &out_map->chart, x, 0.0, 0.0, &coord) !=
                SIM_PLANE_CHART_STATUS_OK ||
            !isfinite(coord.primary) || !isfinite(coord.secondary)) {
            return 0;
        }

        if (coord.primary < out_map->primary_min) {
            out_map->primary_min = coord.primary;
        }
        if (coord.primary > out_map->primary_max) {
            out_map->primary_max = coord.primary;
        }
        if (coord.secondary < out_map->secondary_min) {
            out_map->secondary_min = coord.secondary;
        }
        if (coord.secondary > out_map->secondary_max) {
            out_map->secondary_max = coord.secondary;
        }
        out_map->valid = 1;
    }

    return out_map->valid;
}

static int plane_coord(const size_t *shape, const SimStimulusZetaPlaneSliceConfig *cfg, size_t row,
                       size_t col, double *out_sigma, double *out_t) {
    PlaneChartMap map = {0};
    SimPlaneChartCoord coord = {0.0, 0.0};
    double x = 0.0;
    double y = 0.0;

    if (shape == NULL || cfg == NULL || out_sigma == NULL || out_t == NULL ||
        !build_chart_map_2d(shape, cfg, &map)) {
        return 0;
    }

    x = map.frame.origin_x + (double)col * map.frame.spacing_x;
    y = map.frame.origin_y + (double)row * map.frame.spacing_y;
    if (sim_plane_chart_eval(&map.frame, &map.chart, x, y, 0.0, &coord) !=
        SIM_PLANE_CHART_STATUS_OK) {
        return 0;
    }

    {
        double sigma_value = 0.0;
        double t_value = 0.0;
        double sigma_min = 0.0;
        double sigma_max = 0.0;
        double t_min = 0.0;
        double t_max = 0.0;
        double sigma_u = 0.0;
        double t_v = 0.0;

        if (!project_chart_axis(coord, cfg->sigma_projection, &sigma_value) ||
            !project_chart_axis(coord, cfg->t_projection, &t_value) ||
            !projection_range(&map, cfg->sigma_projection, &sigma_min, &sigma_max) ||
            !projection_range(&map, cfg->t_projection, &t_min, &t_max)) {
            return 0;
        }

        sigma_u = normalize_chart_range(sigma_value, sigma_min, sigma_max);
        t_v = normalize_chart_range(t_value, t_min, t_max);
        if (cfg->sigma_flip) {
            sigma_u = -sigma_u;
        }
        if (cfg->t_flip) {
            t_v = -t_v;
        }

        *out_sigma = cfg->sigma_center + cfg->sigma_span * sigma_u;
        *out_t = cfg->t_center + cfg->t_span * t_v;
    }
    return 1;
}

static int plane_coord_1d(size_t width, const SimStimulusZetaPlaneSliceConfig *cfg, size_t col,
                          double *out_sigma, double *out_t) {
    PlaneChartMap map = {0};
    SimPlaneChartCoord coord = {0.0, 0.0};
    double x = 0.0;

    if (cfg == NULL || out_sigma == NULL || out_t == NULL ||
        !build_chart_map_1d(width, cfg, &map)) {
        return 0;
    }

    x = map.frame.origin_x + (double)col * map.frame.spacing_x;
    if (sim_plane_chart_eval(&map.frame, &map.chart, x, 0.0, 0.0, &coord) !=
        SIM_PLANE_CHART_STATUS_OK) {
        return 0;
    }

    {
        double sigma_value = 0.0;
        double t_value = 0.0;
        double sigma_min = 0.0;
        double sigma_max = 0.0;
        double t_min = 0.0;
        double t_max = 0.0;
        double sigma_u = 0.0;
        double t_v = 0.0;

        if (!project_chart_axis(coord, cfg->sigma_projection, &sigma_value) ||
            !project_chart_axis(coord, cfg->t_projection, &t_value) ||
            !projection_range(&map, cfg->sigma_projection, &sigma_min, &sigma_max) ||
            !projection_range(&map, cfg->t_projection, &t_min, &t_max)) {
            return 0;
        }

        sigma_u = normalize_chart_range(sigma_value, sigma_min, sigma_max);
        t_v = normalize_chart_range(t_value, t_min, t_max);
        if (cfg->sigma_flip) {
            sigma_u = -sigma_u;
        }
        if (cfg->t_flip) {
            t_v = -t_v;
        }

        *out_sigma = cfg->sigma_center + cfg->sigma_span * sigma_u;
        *out_t = cfg->t_center + cfg->t_span * t_v;
    }
    return 1;
}

static int eval_reference(const SimStimulusZetaPlaneSliceConfig *cfg, double sigma, double t,
                          SimComplexDouble *out_value) {
    SimComplexDouble s = {sigma, t};

    if (cfg->family == SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI) {
        if (cfg->render_mode == SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT) {
            SimXiContext xi_context = sim_xi_context_default();
            SimComplexBall ball = sim_xi_eval_ball(s, &xi_context);
            if (!status_usable(ball.status)) {
                return 0;
            }
            *out_value = ball.center;
        } else {
            SimXiContext xi_context = sim_xi_context_interactive();
            SimXiResult result = sim_xi_eval(s, &xi_context);
            if (!status_usable(result.status)) {
                return 0;
            }
            *out_value = result.value;
        }
    } else {
        if (cfg->render_mode == SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT) {
            SimZetaContext zeta_context = sim_zeta_context_default();
            SimComplexBall ball = sim_zeta_eval_ball(s, &zeta_context);
            if (!status_usable(ball.status)) {
                return 0;
            }
            *out_value = ball.center;
        } else {
            SimZetaContext zeta_context = sim_zeta_context_interactive();
            SimZetaResult result = sim_zeta_eval(s, &zeta_context);
            if (!status_usable(result.status)) {
                return 0;
            }
            *out_value = result.value;
        }
    }

    return isfinite(out_value->re) && isfinite(out_value->im);
}

static double project_value(const SimStimulusZetaPlaneSliceConfig *cfg, SimComplexDouble value) {
    double complex z = value.re + value.im * I;
    double magnitude = cabs(z);

    switch (cfg->view_mode) {
    case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_IM:
        return cimag(z);
    case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ABS:
        return magnitude;
    case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS:
        return log(fmax(magnitude, cfg->log_floor));
    case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ARG:
        return (magnitude <= cfg->log_floor) ? 0.0 : carg(z);
    case SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_RE:
    default:
        return creal(z);
    }
}

static int run_exact_real_case(void) {
    const size_t shape[2] = {5U, 7U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: real case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: real case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_real_data(&field), 0, shape[0] * shape[1] * sizeof(double));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: real case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 0.75;
        cfg.sigma_center = 2.0;
        cfg.t_center = 0.0;
        cfg.sigma_span = 1.25;
        cfg.t_span = 24.0;
        cfg.log_floor = 1.0e-12;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ABS;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: real case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: real case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const double *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: real case config\n");
            goto cleanup;
        }

        data = sim_field_real_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: real case output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                double sigma = 0.0;
                double t = 0.0;
                SimComplexDouble value = {0.0, 0.0};
                double expected = 0.0;
                size_t idx = row * shape[1] + col;

                if (!plane_coord(shape, &cfg, row, col, &sigma, &t)) {
                    fprintf(stderr, "FAIL: real case plane_coord\n");
                    goto cleanup;
                }
                if (eval_reference(&cfg, sigma, t, &value)) {
                    expected = cfg.amplitude * project_value(&cfg, value);
                }

                if (!approx(data[idx], expected, 1.0e-11)) {
                    fprintf(stderr,
                            "FAIL: real case mismatch at (%zu,%zu) got=%.17g expected=%.17g\n", row,
                            col, data[idx], expected);
                    goto cleanup;
                }
            }
        }

        for (size_t col = 0U; col < shape[1]; ++col) {
            size_t top = col;
            size_t bottom = (shape[0] - 1U) * shape[1] + col;
            if (!approx(data[top], data[bottom], 1.0e-10)) {
                fprintf(stderr, "FAIL: real case conjugation symmetry at col=%zu\n", col);
                goto cleanup;
            }
        }
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static int run_interactive_real_case(void) {
    const size_t shape[2] = {4U, 5U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: interactive case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: interactive case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_real_data(&field), 0, shape[0] * shape[1] * sizeof(double));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: interactive case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 0.55;
        cfg.sigma_center = 0.75;
        cfg.t_center = 18.0;
        cfg.sigma_span = 1.0;
        cfg.t_span = 6.0;
        cfg.log_floor = 1.0e-10;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: interactive case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: interactive case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const double *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: interactive case config\n");
            goto cleanup;
        }

        data = sim_field_real_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: interactive case output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                double sigma = 0.0;
                double t = 0.0;
                SimComplexDouble value = {0.0, 0.0};
                double expected = 0.0;
                size_t idx = row * shape[1] + col;

                if (!plane_coord(shape, &cfg, row, col, &sigma, &t)) {
                    fprintf(stderr, "FAIL: interactive case plane_coord\n");
                    goto cleanup;
                }
                if (eval_reference(&cfg, sigma, t, &value)) {
                    expected = cfg.amplitude * project_value(&cfg, value);
                }

                if (!approx(data[idx], expected, 1.0e-11)) {
                    fprintf(
                        stderr,
                        "FAIL: interactive case mismatch at (%zu,%zu) got=%.17g expected=%.17g\n",
                        row, col, data[idx], expected);
                    goto cleanup;
                }
            }
        }
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static int run_exact_real_1d_case(void) {
    const size_t shape[1] = {6U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: 1d case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: 1d case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_real_data(&field), 0, shape[0] * sizeof(double));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: 1d case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 0.65;
        cfg.sigma_center = 0.5;
        cfg.t_center = 14.134725;
        cfg.sigma_span = 1.5;
        cfg.t_span = 6.0;
        cfg.log_floor = 1.0e-10;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ABS;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: 1d case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: 1d case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const double *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: 1d case config\n");
            goto cleanup;
        }

        data = sim_field_real_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: 1d case output missing\n");
            goto cleanup;
        }

        for (size_t col = 0U; col < shape[0]; ++col) {
            double sigma = 0.0;
            double t = 0.0;
            SimComplexDouble value = {0.0, 0.0};
            double expected = 0.0;

            if (!plane_coord_1d(shape[0], &cfg, col, &sigma, &t)) {
                fprintf(stderr, "FAIL: 1d case plane_coord\n");
                goto cleanup;
            }
            if (eval_reference(&cfg, sigma, t, &value)) {
                expected = cfg.amplitude * project_value(&cfg, value);
            }

            if (!approx(data[col], expected, 1.0e-11)) {
                fprintf(stderr, "FAIL: 1d case mismatch at %zu got=%.17g expected=%.17g\n", col,
                        data[col], expected);
                goto cleanup;
            }
        }
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static int run_polar_real_case(void) {
    const size_t shape[2] = {5U, 6U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;
    int saw_noncartesian = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: polar case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: polar case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_real_data(&field), 0, shape[0] * shape[1] * sizeof(double));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: polar case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 0.7;
        cfg.sigma_center = 0.5;
        cfg.t_center = 14.0;
        cfg.sigma_span = 1.8;
        cfg.t_span = 5.0;
        cfg.log_floor = 1.0e-10;
        cfg.chart_kind = SIM_PLANE_CHART_POLAR;
        cfg.chart_center_x = 0.1;
        cfg.chart_center_y = -0.05;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: polar case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: polar case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const double *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: polar case config\n");
            goto cleanup;
        }

        data = sim_field_real_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: polar case output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                double sigma = 0.0;
                double t = 0.0;
                double cart_sigma = 0.0;
                double cart_t = 0.0;
                SimComplexDouble value = {0.0, 0.0};
                double expected = 0.0;
                size_t idx = row * shape[1] + col;
                double u = (shape[1] > 1U) ? ((double)col / (double)(shape[1] - 1U) - 0.5) : 0.0;
                double v = (shape[0] > 1U) ? (0.5 - (double)row / (double)(shape[0] - 1U)) : 0.0;

                if (!plane_coord(shape, &cfg, row, col, &sigma, &t)) {
                    fprintf(stderr, "FAIL: polar case plane_coord\n");
                    goto cleanup;
                }
                cart_sigma = cfg.sigma_center + cfg.sigma_span * u;
                cart_t = cfg.t_center + cfg.t_span * v;
                if (!approx(sigma, cart_sigma, 1.0e-9) || !approx(t, cart_t, 1.0e-9)) {
                    saw_noncartesian = 1;
                }

                if (eval_reference(&cfg, sigma, t, &value)) {
                    expected = cfg.amplitude * project_value(&cfg, value);
                }

                if (!approx(data[idx], expected, 1.0e-11)) {
                    fprintf(stderr,
                            "FAIL: polar case mismatch at (%zu,%zu) got=%.17g expected=%.17g\n",
                            row, col, data[idx], expected);
                    goto cleanup;
                }
            }
        }
    }

    if (!saw_noncartesian) {
        fprintf(stderr, "FAIL: polar case never departed from cartesian sampling\n");
        goto cleanup;
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static int run_projection_real_case(void) {
    const size_t shape[2] = {3U, 4U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;
    int saw_projection_remap = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection real case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection real case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_real_data(&field), 0, shape[0] * shape[1] * sizeof(double));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection real case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 0.9;
        cfg.sigma_center = 0.5;
        cfg.t_center = 20.0;
        cfg.sigma_span = 1.4;
        cfg.t_span = 5.0;
        cfg.log_floor = 1.0e-9;
        cfg.sigma_projection = SIM_PLANE_PROJECTION_SECONDARY;
        cfg.t_projection = SIM_PLANE_PROJECTION_PRIMARY;
        cfg.sigma_flip = true;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_ZETA;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: projection real case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection real case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const double *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: projection real case config\n");
            goto cleanup;
        }

        data = sim_field_real_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: projection real case output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                double sigma = 0.0;
                double t = 0.0;
                double default_sigma = 0.0;
                double default_t = 0.0;
                SimComplexDouble value = {0.0, 0.0};
                double expected = 0.0;
                size_t idx = row * shape[1] + col;
                double u = (shape[1] > 1U) ? ((double)col / (double)(shape[1] - 1U) - 0.5) : 0.0;
                double v = (shape[0] > 1U) ? (0.5 - (double)row / (double)(shape[0] - 1U)) : 0.0;

                if (!plane_coord(shape, &cfg, row, col, &sigma, &t)) {
                    fprintf(stderr, "FAIL: projection real case plane_coord\n");
                    goto cleanup;
                }
                default_sigma = cfg.sigma_center + cfg.sigma_span * u;
                default_t = cfg.t_center + cfg.t_span * v;
                if (!approx(sigma, default_sigma, 1.0e-9) || !approx(t, default_t, 1.0e-9)) {
                    saw_projection_remap = 1;
                }
                if (eval_reference(&cfg, sigma, t, &value)) {
                    expected = cfg.amplitude * project_value(&cfg, value);
                }

                if (!approx(data[idx], expected, 1.0e-11)) {
                    fprintf(stderr,
                            "FAIL: projection real case mismatch at (%zu,%zu) got=%.17g "
                            "expected=%.17g\n",
                            row, col, data[idx], expected);
                    goto cleanup;
                }
            }
        }
    }

    if (!saw_projection_remap) {
        fprintf(stderr, "FAIL: projection real case never departed from default axis mapping\n");
        goto cleanup;
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static int run_exact_complex_case(void) {
    const size_t shape[2] = {4U, 6U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: complex case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: complex case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_complex_data(&field), 0, shape[0] * shape[1] * sizeof(SimComplexDouble));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: complex case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 1.1;
        cfg.sigma_center = 0.5;
        cfg.t_center = 14.0;
        cfg.sigma_span = 1.5;
        cfg.t_span = 4.0;
        cfg.log_floor = 1.0e-9;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_LOG_ABS;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_EXACT;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: complex case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: complex case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const SimComplexDouble *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: complex case config\n");
            goto cleanup;
        }

        data = sim_field_complex_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: complex case output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                double sigma = 0.0;
                double t = 0.0;
                SimComplexDouble value = {0.0, 0.0};
                double expected = 0.0;
                size_t idx = row * shape[1] + col;

                if (!plane_coord(shape, &cfg, row, col, &sigma, &t)) {
                    fprintf(stderr, "FAIL: complex case plane_coord\n");
                    goto cleanup;
                }
                if (eval_reference(&cfg, sigma, t, &value)) {
                    expected = cfg.amplitude * project_value(&cfg, value);
                }

                if (!approx(data[idx].re, expected, 1.0e-8) ||
                    !approx(data[idx].im, 0.0, 1.0e-12)) {
                    fprintf(stderr,
                            "FAIL: complex case mismatch at (%zu,%zu) got=(%.17g, %.17g) "
                            "expected=(%.17g, 0)\n",
                            row, col, data[idx].re, data[idx].im, expected);
                    goto cleanup;
                }
            }
        }
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static int run_projection_complex_case(void) {
    const size_t shape[2] = {4U, 5U};
    SimContext ctx = {0};
    SimField field = {0};
    size_t field_index = 0U;
    size_t op_index = 0U;
    int ok = 0;
    int saw_projection_remap = 0;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection complex case context init\n");
        return 0;
    }

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection complex case field init\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(sim_field_complex_data(&field), 0, shape[0] * shape[1] * sizeof(SimComplexDouble));

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection complex case add field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        cfg.field_index = field_index;
        cfg.amplitude = 0.8;
        cfg.sigma_center = 0.5;
        cfg.t_center = 14.134725;
        cfg.sigma_span = 1.2;
        cfg.t_span = 3.0;
        cfg.log_floor = 1.0e-9;
        cfg.chart_kind = SIM_PLANE_CHART_POLAR;
        cfg.t_projection = SIM_PLANE_PROJECTION_PRIMARY;
        cfg.t_flip = true;
        cfg.family = SIM_STIMULUS_ZETA_PLANE_SLICE_FAMILY_XI;
        cfg.view_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_VIEW_ARG;
        cfg.render_mode = SIM_STIMULUS_ZETA_PLANE_SLICE_RENDER_INTERACTIVE;

        if (sim_add_stimulus_zeta_plane_slice_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: projection complex case add operator\n");
            goto cleanup;
        }
    }

    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: projection complex case execute\n");
        goto cleanup;
    }

    {
        SimStimulusZetaPlaneSliceConfig cfg = {0};
        const SimComplexDouble *data;

        if (sim_stimulus_zeta_plane_slice_config(&ctx, op_index, &cfg) != SIM_RESULT_OK) {
            fprintf(stderr, "FAIL: projection complex case config\n");
            goto cleanup;
        }

        data = sim_field_complex_data_const(sim_context_field(&ctx, field_index));
        if (data == NULL) {
            fprintf(stderr, "FAIL: projection complex case output missing\n");
            goto cleanup;
        }

        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                double sigma = 0.0;
                double t = 0.0;
                double default_sigma = 0.0;
                double default_t = 0.0;
                SimComplexDouble value = {0.0, 0.0};
                double expected = 0.0;
                size_t idx = row * shape[1] + col;
                double u = (shape[1] > 1U) ? ((double)col / (double)(shape[1] - 1U) - 0.5) : 0.0;
                double v = (shape[0] > 1U) ? (0.5 - (double)row / (double)(shape[0] - 1U)) : 0.0;

                if (!plane_coord(shape, &cfg, row, col, &sigma, &t)) {
                    fprintf(stderr, "FAIL: projection complex case plane_coord\n");
                    goto cleanup;
                }
                default_sigma = cfg.sigma_center + cfg.sigma_span * u;
                default_t = cfg.t_center + cfg.t_span * v;
                if (!approx(sigma, default_sigma, 1.0e-9) || !approx(t, default_t, 1.0e-9)) {
                    saw_projection_remap = 1;
                }
                if (eval_reference(&cfg, sigma, t, &value)) {
                    expected = cfg.amplitude * project_value(&cfg, value);
                }

                if (!approx(data[idx].re, expected, 1.0e-11) ||
                    !approx(data[idx].im, 0.0, 1.0e-12)) {
                    fprintf(stderr,
                            "FAIL: projection complex case mismatch at (%zu,%zu) got=(%.17g, "
                            "%.17g) expected=(%.17g, 0)\n",
                            row, col, data[idx].re, data[idx].im, expected);
                    goto cleanup;
                }
            }
        }
    }

    if (!saw_projection_remap) {
        fprintf(stderr, "FAIL: projection complex case never departed from default axis mapping\n");
        goto cleanup;
    }

    ok = 1;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    if (!run_exact_real_case()) {
        return 1;
    }
    if (!run_interactive_real_case()) {
        return 1;
    }
    if (!run_exact_real_1d_case()) {
        return 1;
    }
    if (!run_polar_real_case()) {
        return 1;
    }
    if (!run_projection_real_case()) {
        return 1;
    }
    if (!run_exact_complex_case()) {
        return 1;
    }
    if (!run_projection_complex_case()) {
        return 1;
    }
    return 0;
}
