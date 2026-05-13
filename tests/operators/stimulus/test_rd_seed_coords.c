/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define TEST_RD_SEED_EPS 1.0e-12

typedef struct {
    uint64_t state;
    uint64_t inc;
} test_rd_seed_pcg32_t;

typedef struct {
    double u_min;
    double u_max;
    double v_min;
    double v_max;
    double u_span;
    double v_span;
    double avg_span;
} TestRDSeedBounds;

static uint32_t test_rd_seed_pcg32_random(test_rd_seed_pcg32_t *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void test_rd_seed_pcg32_seed(test_rd_seed_pcg32_t *rng, uint64_t initstate,
                                    uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    (void)test_rd_seed_pcg32_random(rng);
    rng->state += initstate;
    (void)test_rd_seed_pcg32_random(rng);
}

static double test_rd_seed_uniform(test_rd_seed_pcg32_t *rng) {
    return ldexp(test_rd_seed_pcg32_random(rng), -32);
}

static double array_sum_abs(const double *data, size_t count) {
    double sum = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        sum += fabs(data[i]);
    }
    return sum;
}

static double array_max_abs_diff(const double *a, const double *b, size_t count) {
    double max_diff = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

static void test_rd_seed_map_coord(const SimStimulusCoordConfig *coord, double x, double y,
                                   double t, double *out_u, double *out_v) {
    double u = x;
    double v = y;
    sim_stimulus_coord_sample_xy(coord, x, y, t, &u, &v);
    if (coord == NULL) {
        *out_u = u;
        *out_v = v;
        return;
    }

    switch (coord->mode) {
    case SIM_STIMULUS_COORD_AXIS:
        if (coord->axis == SIM_STIMULUS_AXIS_Y) {
            double tmp = u;
            u = v;
            v = tmp;
        }
        break;
    case SIM_STIMULUS_COORD_ANGLE: {
        double c = cos(coord->angle);
        double s = sin(coord->angle);
        double sample_x = u;
        double sample_y = v;
        u = sample_x * c + sample_y * s;
        v = -sample_x * s + sample_y * c;
        break;
    }
    case SIM_STIMULUS_COORD_RADIAL:
    case SIM_STIMULUS_COORD_POLAR: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        u = hypot(dx, dy);
        v = atan2(dy, dx);
        break;
    }
    case SIM_STIMULUS_COORD_AZIMUTH: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        u = atan2(dy, dx);
        v = hypot(dx, dy);
        break;
    }
    case SIM_STIMULUS_COORD_ELLIPTIC: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        sim_stimulus_coord_elliptic_polar(coord, dx, dy, &u, &v);
        break;
    }
    case SIM_STIMULUS_COORD_SPIRAL: {
        double dx = 0.0;
        double dy = 0.0;
        sim_stimulus_coord_centered_xy(coord, x, y, t, &dx, &dy);
        double r = hypot(dx, dy);
        double th = atan2(dy, dx);
        u = coord->spiral_pitch * r + coord->spiral_arms * th + coord->spiral_phase +
            coord->spiral_angular_velocity * t;
        v = r;
        break;
    }
    case SIM_STIMULUS_COORD_SEPARABLE:
    default:
        break;
    }

    *out_u = u;
    *out_v = v;
}

static double test_rd_seed_clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double test_rd_seed_sigmoid(double value) {
    if (value >= 40.0) {
        return 1.0;
    }
    if (value <= -40.0) {
        return 0.0;
    }
    return 1.0 / (1.0 + exp(-value));
}

static int test_rd_seed_build_tables(const SimStimulusRDSeedConfig *cfg, double **out_u,
                                     double **out_v, double **out_phase) {
    *out_u = NULL;
    *out_v = NULL;
    *out_phase = NULL;
    if (cfg == NULL || cfg->seed_count == 0U) {
        return 1;
    }

    double *seed_u = (double *)malloc((size_t)cfg->seed_count * sizeof(double));
    double *seed_v = (double *)malloc((size_t)cfg->seed_count * sizeof(double));
    double *seed_phase = (double *)malloc((size_t)cfg->seed_count * sizeof(double));
    if (seed_u == NULL || seed_v == NULL || seed_phase == NULL) {
        free(seed_u);
        free(seed_v);
        free(seed_phase);
        return 0;
    }

    test_rd_seed_pcg32_t rng;
    test_rd_seed_pcg32_seed(&rng, cfg->seed, cfg->seed ^ 0x9E3779B97F4A7C15ULL);
    for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
        seed_u[i] = test_rd_seed_uniform(&rng);
        seed_v[i] = test_rd_seed_uniform(&rng);
        seed_phase[i] = 2.0 * M_PI * test_rd_seed_uniform(&rng);
    }

    *out_u = seed_u;
    *out_v = seed_v;
    *out_phase = seed_phase;
    return 1;
}

static int test_rd_seed_measure_bounds(const SimStimulusRDSeedConfig *cfg, size_t rank,
                                       const size_t *shape, double t,
                                       TestRDSeedBounds *out_bounds) {
    size_t count = 1U;
    bool first = true;
    for (size_t i = 0U; i < rank; ++i) {
        count *= shape[i];
    }

    for (size_t i = 0U; i < count; ++i) {
        double x = 0.0;
        double y = cfg->coord.origin_y;
        double u = 0.0;
        double v = 0.0;

        if (rank == 1U) {
            x = cfg->coord.origin_x + (double)i * cfg->coord.spacing_x;
        } else {
            size_t row = i / shape[1];
            size_t col = i % shape[1];
            x = cfg->coord.origin_x + (double)col * cfg->coord.spacing_x;
            y = cfg->coord.origin_y + (double)row * cfg->coord.spacing_y;
        }

        test_rd_seed_map_coord(&cfg->coord, x, y, t, &u, &v);
        if (!isfinite(u) || !isfinite(v)) {
            continue;
        }

        if (first) {
            out_bounds->u_min = out_bounds->u_max = u;
            out_bounds->v_min = out_bounds->v_max = v;
            first = false;
        } else {
            if (u < out_bounds->u_min) {
                out_bounds->u_min = u;
            }
            if (u > out_bounds->u_max) {
                out_bounds->u_max = u;
            }
            if (v < out_bounds->v_min) {
                out_bounds->v_min = v;
            }
            if (v > out_bounds->v_max) {
                out_bounds->v_max = v;
            }
        }
    }

    if (first) {
        out_bounds->u_min = out_bounds->u_max = 0.0;
        out_bounds->v_min = out_bounds->v_max = 0.0;
    }
    out_bounds->u_span = out_bounds->u_max - out_bounds->u_min;
    out_bounds->v_span = out_bounds->v_max - out_bounds->v_min;
    if (out_bounds->u_span <= TEST_RD_SEED_EPS) {
        out_bounds->u_span = 1.0;
    }
    if (out_bounds->v_span <= TEST_RD_SEED_EPS) {
        out_bounds->v_span = 1.0;
    }
    out_bounds->avg_span = 0.5 * (out_bounds->u_span + out_bounds->v_span);
    if (out_bounds->avg_span <= TEST_RD_SEED_EPS) {
        out_bounds->avg_span = 1.0;
    }
    return 1;
}

static double test_rd_seed_eval_pattern(const SimStimulusRDSeedConfig *cfg,
                                        const TestRDSeedBounds *bounds, const double *seed_u,
                                        const double *seed_v, const double *seed_phase, double u,
                                        double v, double t) {
    double phase_drive = cfg->phase - cfg->omega * t;
    double u_span = bounds->u_span;
    double v_span = bounds->v_span;
    double avg_span = bounds->avg_span;
    double raw = 0.0;

    switch (cfg->mode) {
    case SIM_STIMULUS_RD_SEED_SPOTS: {
        double sigma = avg_span / (6.0 * cfg->scale);
        if (sigma <= TEST_RD_SEED_EPS) {
            sigma = avg_span * 0.02 + TEST_RD_SEED_EPS;
        }
        double inv_two_sigma2 = 0.5 / (sigma * sigma);
        double sum = 0.0;
        for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
            double center_u = bounds->u_min + seed_u[i] * u_span;
            double center_v = bounds->v_min + seed_v[i] * v_span;
            double du = u - center_u;
            double dv = v - center_v;
            sum += exp(-(du * du + dv * dv) * inv_two_sigma2);
        }
        raw = sum / (double)cfg->seed_count;
        break;
    }
    case SIM_STIMULUS_RD_SEED_STRIPES: {
        double k = cfg->scale * (2.0 * M_PI / avg_span);
        double sum = 0.0;
        for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
            double angle = seed_phase[i];
            double proj = cos(angle) * u + sin(angle) * v;
            double theta = k * proj + phase_drive + 2.0 * M_PI * seed_v[i];
            sum += cos(theta);
        }
        raw = 0.5 + 0.5 * (sum / (double)cfg->seed_count);
        break;
    }
    case SIM_STIMULUS_RD_SEED_LABYRINTH: {
        double k = cfg->scale * (2.0 * M_PI / avg_span);
        double sum = 0.0;
        for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
            double angle = seed_phase[i];
            double proj = cos(angle) * u + sin(angle) * v;
            double theta = k * proj + phase_drive + 2.0 * M_PI * seed_v[i];
            sum += sin(theta) + 0.45 * cos(1.7 * theta + seed_u[i] * M_PI);
        }
        raw = 0.5 + 0.5 * tanh(1.25 * (sum / sqrt((double)cfg->seed_count)));
        break;
    }
    case SIM_STIMULUS_RD_SEED_RINGS:
    default: {
        double k = cfg->scale * (2.0 * M_PI / avg_span);
        double sum = 0.0;
        for (unsigned int i = 0U; i < cfg->seed_count; ++i) {
            double center_u = bounds->u_min + seed_u[i] * u_span;
            double center_v = bounds->v_min + seed_v[i] * v_span;
            double r = hypot(u - center_u, v - center_v);
            double theta = k * r + phase_drive + seed_phase[i];
            double envelope = exp(-0.75 * r / avg_span);
            sum += (0.5 + 0.5 * cos(theta)) * envelope;
        }
        raw = sum / (double)cfg->seed_count;
        break;
    }
    }

    raw = test_rd_seed_clamp01(raw);
    if (cfg->sharpness > TEST_RD_SEED_EPS) {
        raw = test_rd_seed_sigmoid(cfg->sharpness * (raw - cfg->threshold));
    }
    return test_rd_seed_clamp01(raw);
}

static int run_rd_seed_2d_persistent_case(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent sim_context_init\n");
        return 0;
    }

    size_t shape[2] = {48U, 64U};
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRDSeedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.9;
    cfg.bias = -0.2;
    cfg.scale = 8.0;
    cfg.threshold = 0.53;
    cfg.sharpness = 15.0;
    cfg.seed_count = 28U;
    cfg.seed = 7U;
    cfg.mode = SIM_STIMULUS_RD_SEED_LABYRINTH;
    cfg.omega = 0.6;
    cfg.phase = 0.2;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_SPIRAL;
    cfg.coord.origin_x = -2.0;
    cfg.coord.origin_y = -1.5;
    cfg.coord.spacing_x = 0.06;
    cfg.coord.spacing_y = 0.06;
    cfg.coord.center_x = 0.1;
    cfg.coord.center_y = -0.2;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.coord.spiral_arms = 2.0;
    cfg.coord.spiral_pitch = 1.2;
    cfg.coord.spiral_phase = 0.1;
    cfg.coord.spiral_angular_velocity = 0.3;

    size_t op_index = 0U;
    if (sim_add_stimulus_rd_seed_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent operator lookup\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent first evaluate\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *out = sim_context_field(&ctx, field_index);
    if (out == NULL) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent output field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double *data = (double *)sim_field_data(out);
    if (data == NULL) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent output data\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = sim_field_bytes(out) / sizeof(double);
    if (count == 0U) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent empty output\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double sum_abs = array_sum_abs(data, count);
    if (!(sum_abs > 1.0e-6)) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent first write too small\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double *snapshot = (double *)malloc(count * sizeof(double));
    if (snapshot == NULL) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent snapshot alloc\n");
        sim_context_destroy(&ctx);
        return 0;
    }
    memcpy(snapshot, data, count * sizeof(double));

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent second evaluate\n");
        free(snapshot);
        sim_context_destroy(&ctx);
        return 0;
    }

    double second_diff = array_max_abs_diff(data, snapshot, count);
    if (!(second_diff > 1.0e-6)) {
        fprintf(stderr,
                "FAIL: rd_seed_2d_persistent second evaluate did not accumulate (diff=%.3g)\n",
                second_diff);
        free(snapshot);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRDSeedConfig updated;
    memset(&updated, 0, sizeof(updated));
    if (sim_stimulus_rd_seed_config(&ctx, op_index, &updated) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent config fetch\n");
        free(snapshot);
        sim_context_destroy(&ctx);
        return 0;
    }
    updated.seed += 17U;
    updated.amplitude = 0.55;

    if (sim_stimulus_rd_seed_update(&ctx, op_index, &updated) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent update\n");
        free(snapshot);
        sim_context_destroy(&ctx);
        return 0;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent evaluate after update\n");
        free(snapshot);
        sim_context_destroy(&ctx);
        return 0;
    }

    double update_diff = array_max_abs_diff(data, snapshot, count);
    if (!(update_diff > 1.0e-6)) {
        fprintf(stderr, "FAIL: rd_seed_2d_persistent update did not change output (diff=%.3g)\n",
                update_diff);
        free(snapshot);
        sim_context_destroy(&ctx);
        return 0;
    }

    free(snapshot);
    sim_context_destroy(&ctx);
    fprintf(stdout, "[rd_seed_2d_persistent] ok\n");
    return 1;
}

static int run_rd_seed_1d_case(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_1d sim_context_init\n");
        return 0;
    }

    size_t shape[1] = {257U};
    SimField field = {0};
    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_1d sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_1d sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRDSeedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 1.0;
    cfg.bias = 0.0;
    cfg.scale = 9.0;
    cfg.threshold = 0.45;
    cfg.sharpness = 18.0;
    cfg.seed_count = 18U;
    cfg.seed = 123U;
    cfg.mode = SIM_STIMULUS_RD_SEED_SPOTS;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -1.5;
    cfg.coord.spacing_x = 0.012;

    size_t op_index = 0U;
    if (sim_add_stimulus_rd_seed_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_1d add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_1d evaluate\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *out = sim_context_field(&ctx, field_index);
    if (out == NULL) {
        fprintf(stderr, "FAIL: rd_seed_1d output field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    const double *data = (const double *)sim_field_data(out);
    if (data == NULL) {
        fprintf(stderr, "FAIL: rd_seed_1d output data\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t count = sim_field_bytes(out) / sizeof(double);
    if (count == 0U || !(array_sum_abs(data, count) > 1.0e-6)) {
        fprintf(stderr, "FAIL: rd_seed_1d insufficient write\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    for (size_t i = 0U; i < count; ++i) {
        if (!isfinite(data[i])) {
            fprintf(stderr, "FAIL: rd_seed_1d non-finite sample at %zu\n", i);
            sim_context_destroy(&ctx);
            return 0;
        }
    }

    sim_context_destroy(&ctx);
    fprintf(stdout, "[rd_seed_1d_spots] ok\n");
    return 1;
}

static int run_rd_seed_complex_reference_case(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference sim_context_init\n");
        return 0;
    }

    size_t shape[2] = {4U, 96U};
    SimField field = {0};
    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference sim_field_init\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimComplexDouble *initial = sim_field_complex_data(&field);
    if (initial == NULL) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference sim_field_complex_data\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }
    memset(initial, 0, sim_field_bytes(&field));

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference sim_context_add_field\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRDSeedConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.72;
    cfg.bias = -0.14;
    cfg.scale = 7.8;
    cfg.threshold = 0.49;
    cfg.sharpness = 14.0;
    cfg.seed_count = 30U;
    cfg.seed = 41U;
    cfg.mode = SIM_STIMULUS_RD_SEED_RINGS;
    cfg.omega = -0.35;
    cfg.phase = 0.27;
    cfg.rotation = 0.29;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.41;
    cfg.coord.origin_x = -1.44;
    cfg.coord.origin_y = -0.36;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;

    size_t op_index = 0U;
    if (sim_add_stimulus_rd_seed_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRDSeedConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_rd_seed_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference config fetch\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double *seed_u = NULL;
    double *seed_v = NULL;
    double *seed_phase = NULL;
    if (!test_rd_seed_build_tables(&normalized, &seed_u, &seed_v, &seed_phase)) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference seed tables\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    TestRDSeedBounds bounds;
    if (!test_rd_seed_measure_bounds(&normalized, 2U, shape, normalized.time_offset, &bounds)) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference bounds\n");
        free(seed_u);
        free(seed_v);
        free(seed_phase);
        sim_context_destroy(&ctx);
        return 0;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL || op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference evaluate\n");
        free(seed_u);
        free(seed_v);
        free(seed_phase);
        sim_context_destroy(&ctx);
        return 0;
    }

    const SimComplexDouble *data =
        sim_field_complex_data_const(sim_context_field(&ctx, field_index));
    if (data == NULL) {
        fprintf(stderr, "FAIL: rd_seed_complex_reference output data\n");
        free(seed_u);
        free(seed_v);
        free(seed_phase);
        sim_context_destroy(&ctx);
        return 0;
    }

    {
        double sin_r = sin(normalized.rotation);
        double cos_r = cos(normalized.rotation);
        for (size_t row = 0U; row < shape[0]; ++row) {
            for (size_t col = 0U; col < shape[1]; ++col) {
                size_t idx = row * shape[1] + col;
                double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
                double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
                double u = 0.0;
                double v = 0.0;
                double seed = 0.0;
                double re = 0.0;
                double im = 0.0;

                test_rd_seed_map_coord(&normalized.coord, x, y, normalized.time_offset, &u, &v);
                seed = test_rd_seed_eval_pattern(&normalized, &bounds, seed_u, seed_v, seed_phase,
                                                 u, v, normalized.time_offset);
                re = normalized.bias + normalized.amplitude * seed;

                if (fabs(data[idx].re - re * cos_r) > 1.0e-9 ||
                    fabs(data[idx].im - re * sin_r) > 1.0e-9) {
                    fprintf(stderr, "FAIL: rd_seed_complex_reference mismatch at %zu\n", idx);
                    free(seed_u);
                    free(seed_v);
                    free(seed_phase);
                    sim_context_destroy(&ctx);
                    return 0;
                }
            }
        }
    }

    free(seed_u);
    free(seed_v);
    free(seed_phase);
    sim_context_destroy(&ctx);
    fprintf(stdout, "[rd_seed_complex_reference] ok\n");
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= run_rd_seed_2d_persistent_case();
    ok &= run_rd_seed_1d_case();
    ok &= run_rd_seed_complex_reference_case();
    return ok ? 0 : 1;
}
