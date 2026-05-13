/*
 * Migrated stimulus coordinate/mode contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_RFF_MAX_FEATURES 256U

typedef struct {
    uint64_t state;
    uint64_t inc;
} test_rff_pcg32_t;

static uint32_t test_rff_pcg32_random(test_rff_pcg32_t *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + (rng->inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void test_rff_pcg32_seed(test_rff_pcg32_t *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    (void)test_rff_pcg32_random(rng);
    rng->state += initstate;
    (void)test_rff_pcg32_random(rng);
}

static double test_rff_uniform(test_rff_pcg32_t *rng) {
    return ldexp(test_rff_pcg32_random(rng), -32);
}

static int add_real_field_2d(SimContext *ctx, size_t ny, size_t nx, size_t *out_field_index) {
    size_t shape[2] = {ny, nx};
    SimField field = {0};
    if (sim_field_init(&field, 2, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        return 0;
    }

    double *raw = (double *)sim_field_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    if (sim_context_add_field(ctx, &field, out_field_index) != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return 0;
    }

    return 1;
}

static int add_complex_field_2d(SimContext *ctx, size_t ny, size_t nx, size_t *out_field_index) {
    size_t shape[2] = {ny, nx};
    SimField field = {0};
    if (sim_field_init(&field, 2, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        return 0;
    }

    SimComplexDouble *raw = sim_field_complex_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    if (sim_context_add_field(ctx, &field, out_field_index) != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return 0;
    }

    return 1;
}

static int add_real_field_1d(SimContext *ctx, size_t nx, size_t *out_field_index) {
    size_t shape[1] = {nx};
    SimField field = {0};
    if (sim_field_init(&field, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        return 0;
    }

    double *raw = (double *)sim_field_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    if (sim_context_add_field(ctx, &field, out_field_index) != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return 0;
    }

    return 1;
}

static int eval_operator(SimContext *ctx, size_t op_index, const char *label) {
    SimOperator *op = sim_operator_registry_get(&ctx->world.operators, op_index);
    if (op == NULL || op->evaluate == NULL) {
        fprintf(stderr, "[%s] operator evaluate hook unavailable\n", label);
        return 0;
    }
    if (op->evaluate(ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "[%s] operator evaluation failed\n", label);
        return 0;
    }
    return 1;
}

static int compare_real_fields(SimContext *ctx, size_t field_a, size_t field_b, double tol,
                               const char *label) {
    SimField *fa = sim_context_field(ctx, field_a);
    SimField *fb = sim_context_field(ctx, field_b);
    if (fa == NULL || fb == NULL) {
        fprintf(stderr, "[%s] missing comparison fields\n", label);
        return 0;
    }

    size_t bytes_a = sim_field_bytes(fa);
    size_t bytes_b = sim_field_bytes(fb);
    if (bytes_a != bytes_b || fa->element_size != sizeof(double) ||
        fb->element_size != sizeof(double)) {
        fprintf(stderr, "[%s] incompatible field layouts\n", label);
        return 0;
    }

    const double *a = (const double *)sim_field_data(fa);
    const double *b = (const double *)sim_field_data(fb);
    if (a == NULL || b == NULL) {
        fprintf(stderr, "[%s] missing comparison data\n", label);
        return 0;
    }

    size_t count = bytes_a / sizeof(double);
    for (size_t i = 0U; i < count; ++i) {
        double err = fabs(a[i] - b[i]);
        if (err > tol) {
            fprintf(stderr, "[%s] mismatch idx=%zu a=%.12g b=%.12g err=%.3g tol=%.3g\n", label, i,
                    a[i], b[i], err, tol);
            return 0;
        }
    }

    return 1;
}

static int expect_uniform_real_field(SimContext *ctx, size_t field_index, double expected,
                                     double tol, const char *label) {
    SimField *field = sim_context_field(ctx, field_index);
    if (field == NULL || field->element_size != sizeof(double)) {
        fprintf(stderr, "[%s] missing uniform field\n", label);
        return 0;
    }

    const double *data = (const double *)sim_field_data(field);
    size_t count = sim_field_bytes(field) / sizeof(double);
    if (data == NULL) {
        fprintf(stderr, "[%s] missing field data\n", label);
        return 0;
    }

    for (size_t i = 0U; i < count; ++i) {
        double err = fabs(data[i] - expected);
        if (err > tol) {
            fprintf(stderr, "[%s] mismatch idx=%zu got=%.12g expected=%.12g err=%.3g tol=%.3g\n",
                    label, i, data[i], expected, err, tol);
            return 0;
        }
    }

    return 1;
}

static void test_random_fourier_eval_base(const SimStimulusRandomFourierConfig *cfg,
                                          const double *k_values, const double *phi_values,
                                          const double *weight_values, double feature_norm,
                                          double u, double t, double *out_re, double *out_im) {
    double re_sum = 0.0;
    double im_sum = 0.0;
    double weight_base = (cfg->feature_count > 0U) ? (1.0 / sqrt((double)cfg->feature_count)) : 0.0;

    for (unsigned int m = 0U; m < cfg->feature_count; ++m) {
        double theta = k_values[m] * u - cfg->omega * t + phi_values[m];
        double local_weight = weight_base;
        if (cfg->spectral_slope != 0.0) {
            local_weight *= weight_values[m] / feature_norm;
        }
        re_sum += local_weight * cos(theta);
        im_sum += local_weight * sin(theta);
    }

    *out_re = re_sum;
    *out_im = im_sum;
}

static int compare_random_fourier_reference(SimContext *ctx, size_t field_index,
                                            const SimStimulusRandomFourierConfig *cfg, double tol,
                                            const char *label) {
    SimField *field = sim_context_field(ctx, field_index);
    if (field == NULL || cfg == NULL || field->element_size != sizeof(double)) {
        fprintf(stderr, "[%s] invalid random-fourier reference inputs\n", label);
        return 0;
    }

    const double *data = (const double *)sim_field_data(field);
    if (data == NULL) {
        fprintf(stderr, "[%s] missing random-fourier data\n", label);
        return 0;
    }
    if (cfg->feature_count == 0U || cfg->feature_count > TEST_RFF_MAX_FEATURES) {
        fprintf(stderr, "[%s] unsupported feature count %u\n", label, cfg->feature_count);
        return 0;
    }

    double k_values[TEST_RFF_MAX_FEATURES];
    double phi_values[TEST_RFF_MAX_FEATURES];
    double weight_values[TEST_RFF_MAX_FEATURES];
    double feature_norm = 1.0;
    double sum_sq = 0.0;
    double dk = cfg->k_max - cfg->k_min;

    test_rff_pcg32_t rng;
    test_rff_pcg32_seed(&rng, cfg->seed, cfg->seed ^ 0x9E3779B97F4A7C15ULL);
    for (unsigned int i = 0U; i < cfg->feature_count; ++i) {
        double u = test_rff_uniform(&rng);
        double v = test_rff_uniform(&rng);
        double k = (dk > 0.0) ? (cfg->k_min + dk * u) : cfg->k_min;
        double w = 1.0;
        if (cfg->spectral_slope != 0.0) {
            double freq = fabs(k);
            if (freq > 1.0e-9) {
                w = pow(freq, -0.5 * cfg->spectral_slope);
            }
        }
        k_values[i] = k;
        phi_values[i] = 2.0 * M_PI * v;
        weight_values[i] = w;
        sum_sq += w * w;
    }
    if (sum_sq > 1.0e-9) {
        feature_norm = sqrt(sum_sq / (double)cfg->feature_count);
    }

    double t = sim_context_time(ctx) + cfg->time_offset;
    size_t count = sim_field_bytes(field) / sizeof(double);
    for (size_t i = 0U; i < count; ++i) {
        size_t ix = 0U;
        size_t iy = 0U;
        if (sim_field_index_to_xy(field, i, &ix, &iy) != SIM_RESULT_OK) {
            fprintf(stderr, "[%s] index_to_xy failed at %zu\n", label, i);
            return 0;
        }

        double x = cfg->coord.origin_x + (double)ix * cfg->coord.spacing_x;
        double y = cfg->coord.origin_y + (double)iy * cfg->coord.spacing_y;
        double sample_x = x - cfg->coord.velocity_x * t;
        double sample_y = y - cfg->coord.velocity_y * t;
        double base_re = 0.0;
        double base_im = 0.0;

        if (cfg->use_wavevector) {
            double dot = cfg->kx * sample_x + cfg->ky * sample_y;
            double base_k = hypot(cfg->kx, cfg->ky);
            int has_base = (base_k > 1.0e-9);
            for (unsigned int m = 0U; m < cfg->feature_count; ++m) {
                double spatial =
                    has_base ? ((k_values[m] / base_k) * dot) : (k_values[m] * sample_x);
                double theta = spatial - cfg->omega * t + phi_values[m];
                double local_weight = (1.0 / sqrt((double)cfg->feature_count));
                if (cfg->spectral_slope != 0.0) {
                    local_weight *= weight_values[m] / feature_norm;
                }
                base_re += local_weight * cos(theta);
                base_im += local_weight * sin(theta);
            }
        } else if (cfg->coord.mode == SIM_STIMULUS_COORD_SEPARABLE) {
            double fx_re = 0.0;
            double fx_im = 0.0;
            double fy_re = 0.0;
            double fy_im = 0.0;
            test_random_fourier_eval_base(cfg, k_values, phi_values, weight_values, feature_norm,
                                          sample_x, t, &fx_re, &fx_im);
            test_random_fourier_eval_base(cfg, k_values, phi_values, weight_values, feature_norm,
                                          sample_y, t, &fy_re, &fy_im);
            if (cfg->coord.combine == SIM_STIMULUS_SEPARABLE_ADD) {
                base_re = fx_re + fy_re;
                base_im = fx_im + fy_im;
            } else {
                base_re = fx_re * fy_re - fx_im * fy_im;
                base_im = fx_re * fy_im + fx_im * fy_re;
            }
        } else {
            double u = sample_x;
            if (cfg->coord.mode == SIM_STIMULUS_COORD_AXIS) {
                u = (cfg->coord.axis == SIM_STIMULUS_AXIS_Y) ? sample_y : sample_x;
            } else if (cfg->coord.mode == SIM_STIMULUS_COORD_ANGLE) {
                double s = sin(cfg->coord.angle);
                double c = cos(cfg->coord.angle);
                u = sample_x * c + sample_y * s;
            }
            test_random_fourier_eval_base(cfg, k_values, phi_values, weight_values, feature_norm, u,
                                          t, &base_re, &base_im);
        }

        double expected = cfg->amplitude * base_re;
        double err = fabs(data[i] - expected);
        if (err > tol) {
            fprintf(stderr,
                    "[%s] reference mismatch idx=%zu got=%.12g exp=%.12g err=%.3g tol=%.3g\n",
                    label, i, data[i], expected, err, tol);
            return 0;
        }
    }

    return 1;
}

static int test_digamma_square_deformation_helpers(void) {
    const double amplitude = 0.61;
    const double tol = 1.0e-12;
    const double samples[] = {-3.2, -0.75, 0.0, 0.5, 1.9, 4.1};

    for (size_t i = 0U; i < sizeof(samples) / sizeof(samples[0]); ++i) {
        double legacy =
            sim_digamma_square_base_real(amplitude, samples[i], SIM_DIGAMMA_BACKEND_12_TAIL, tol);
        double deformed = sim_digamma_square_base_deformed_real(amplitude, 0.25, samples[i],
                                                                SIM_DIGAMMA_BACKEND_12_TAIL, tol);
        if (fabs(legacy - deformed) > 1.0e-12) {
            fprintf(stderr, "[digamma-helper] mismatch inner=%.12g legacy=%.12g deformed=%.12g\n",
                    samples[i], legacy, deformed);
            return 0;
        }
    }

    return 1;
}

static int test_digamma_square_shape_outputs(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-shapes] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_default = 0U;
    if (!add_real_field_2d(&ctx, 3U, 4U, &field_default)) {
        fprintf(stderr, "[digamma-shapes] failed to add fields\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.amplitude = 0.55;
    cfg.wavenumber = 0.0;
    cfg.omega = 0.0;
    cfg.phase = 0.42;
    cfg.harmonics = 3.5;
    cfg.coord.spacing_x = 1.0;
    cfg.coord.spacing_y = 1.0;
    cfg.a = 0.31;

    size_t op_default = 0U;

    cfg.field_index = field_default;
    cfg.shape = SIM_DIGAMMA_SQUARE_WAVEFORM_DEFAULT;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg, &op_default) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-shapes] failed to add default operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_digamma_square_config(&ctx, op_default, &normalized) != SIM_RESULT_OK ||
        fabs(normalized.a - cfg.a) > 1.0e-12) {
        fprintf(stderr, "[digamma-shapes] failed to round-trip deformation parameter\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_default, "digamma_default_shape")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    double expected_default = sim_digamma_square_base_deformed_real(
        cfg.amplitude, cfg.a, cfg.harmonics * cos(cfg.phase), SIM_DIGAMMA_BACKEND_12_TAIL, 1.0e-12);

    int ok = 1;
    ok &= expect_uniform_real_field(&ctx, field_default, expected_default, 1.0e-12,
                                    "digamma_default_shape");

    SimStimulusDigammaSquareConfig cfg_default_a;
    memset(&cfg_default_a, 0, sizeof(cfg_default_a));
    cfg_default_a.field_index = field_default;
    cfg_default_a.amplitude = 0.1;
    cfg_default_a.wavenumber = 0.0;
    cfg_default_a.omega = 0.0;
    cfg_default_a.coord.spacing_x = 1.0;
    size_t op_default_a = 0U;
    if (ok && sim_add_stimulus_digamma_square_operator(&ctx, &cfg_default_a, &op_default_a) ==
                  SIM_RESULT_OK) {
        memset(&normalized, 0, sizeof(normalized));
        if (sim_stimulus_digamma_square_config(&ctx, op_default_a, &normalized) != SIM_RESULT_OK ||
            fabs(normalized.a - 0.25) > 1.0e-12) {
            fprintf(stderr, "[digamma-shapes] expected default a=0.25\n");
            ok = 0;
        }
    } else if (ok) {
        fprintf(stderr, "[digamma-shapes] failed to add default-a operator\n");
        ok = 0;
    }

    sim_context_destroy(&ctx);
    return ok;
}

static int test_digamma_square_wavevector(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_auto = 0U;
    size_t field_axis = 0U;
    size_t field_sep = 0U;
    if (!add_real_field_2d(&ctx, 4U, 5U, &field_auto) ||
        !add_real_field_2d(&ctx, 4U, 5U, &field_axis) ||
        !add_real_field_2d(&ctx, 4U, 5U, &field_sep)) {
        fprintf(stderr, "[digamma] failed to add fields\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig base;
    memset(&base, 0, sizeof(base));
    base.amplitude = 0.4;
    base.wavenumber = 1.3;
    base.omega = 0.2;
    base.phase = -0.3;
    base.harmonics = 4.0;
    base.tolerance = 1.0e-12;
    base.scale_by_dt = false;
    base.coord.origin_x = -0.6;
    base.coord.origin_y = 0.3;
    base.coord.spacing_x = 0.35;
    base.coord.spacing_y = 0.6;
    base.kx = 0.73;
    base.ky = -0.19;

    SimStimulusDigammaSquareConfig cfg_auto = base;
    cfg_auto.field_index = field_auto;
    cfg_auto.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_auto.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg_auto.use_wavevector = false;

    size_t op_auto = 0U;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg_auto, &op_auto) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma] failed to add auto-enable operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_digamma_square_config(&ctx, op_auto, &normalized) != SIM_RESULT_OK ||
        !normalized.use_wavevector) {
        fprintf(stderr, "[digamma] expected use_wavevector auto-enable from kx/ky\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig cfg_axis = base;
    cfg_axis.field_index = field_axis;
    cfg_axis.use_wavevector = true;
    cfg_axis.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_axis.coord.axis = SIM_STIMULUS_AXIS_X;

    SimStimulusDigammaSquareConfig cfg_sep = base;
    cfg_sep.field_index = field_sep;
    cfg_sep.use_wavevector = true;
    cfg_sep.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg_sep.coord.combine = SIM_STIMULUS_SEPARABLE_ADD;

    size_t op_axis = 0U;
    size_t op_sep = 0U;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg_axis, &op_axis) != SIM_RESULT_OK ||
        sim_add_stimulus_digamma_square_operator(&ctx, &cfg_sep, &op_sep) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma] failed to add comparison operators\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_axis, "digamma_axis") ||
        !eval_operator(&ctx, op_sep, "digamma_sep")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    int ok = compare_real_fields(&ctx, field_axis, field_sep, 1.0e-10, "digamma_wavevector_mode");
    sim_context_destroy(&ctx);
    return ok;
}

static int test_digamma_square_wide_wavevector_reference(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-wide] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_index = 0U;
    if (!add_real_field_2d(&ctx, 4U, 96U, &field_index)) {
        fprintf(stderr, "[digamma-wide] failed to add field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.37;
    cfg.kx = 0.83;
    cfg.ky = -0.27;
    cfg.omega = 0.28;
    cfg.phase = -0.31;
    cfg.harmonics = 4.6;
    cfg.a = 0.31;
    cfg.backend = SIM_DIGAMMA_BACKEND_12_TAIL;
    cfg.tolerance = 1.0e-12;
    cfg.shape = SIM_DIGAMMA_SQUARE_WAVEFORM_TRIANGLE;
    cfg.use_wavevector = true;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -0.62;
    cfg.coord.origin_y = 0.24;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.time_offset = 0.07;

    size_t op_index = 0U;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-wide] failed to add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_digamma_square_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-wide] failed to fetch normalized config\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_index, "digamma_wide")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *field = sim_context_field(&ctx, field_index);
    if (field == NULL || field->element_size != sizeof(double)) {
        fprintf(stderr, "[digamma-wide] missing field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    const double *data = (const double *)sim_field_data(field);
    if (data == NULL) {
        fprintf(stderr, "[digamma-wide] missing field data\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t width = 96U;
    size_t height = 4U;
    double t = normalized.time_offset;
    const double tol = 1.0e-9;
    for (size_t row = 0U; row < height; ++row) {
        for (size_t col = 0U; col < width; ++col) {
            size_t idx = row * width + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double sample_x = x - normalized.coord.velocity_x * t;
            double sample_y = y - normalized.coord.velocity_y * t;
            double phase = normalized.kx * sample_x + normalized.ky * sample_y + normalized.phase -
                           normalized.omega * t;
            double sin_phase = sin(phase);
            double cos_phase = cos(phase);
            double inner = normalized.harmonics * cos_phase;
            double base;

            if (fabs(normalized.a - 0.25) < 1.0e-6) {
                base = sim_digamma_square_base_real(normalized.amplitude, inner, normalized.backend,
                                                    normalized.tolerance);
            } else {
                base =
                    sim_digamma_square_base_deformed_real(normalized.amplitude, normalized.a, inner,
                                                          normalized.backend, normalized.tolerance);
            }

            double expected = base * base * sin_phase;
            double err = fabs(data[idx] - expected);
            if (err > tol) {
                fprintf(
                    stderr,
                    "[digamma-wide] mismatch idx=%zu got=%.12g expected=%.12g err=%.3g tol=%.3g\n",
                    idx, data[idx], expected, err, tol);
                sim_context_destroy(&ctx);
                return 0;
            }
        }
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int test_digamma_square_complex_wavevector_reference(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-complex] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_index = 0U;
    if (!add_complex_field_2d(&ctx, 4U, 96U, &field_index)) {
        fprintf(stderr, "[digamma-complex] failed to add field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.29;
    cfg.kx = 0.71;
    cfg.ky = -0.22;
    cfg.omega = -0.17;
    cfg.phase = 0.26;
    cfg.harmonics = 3.9;
    cfg.a = 0.28;
    cfg.backend = SIM_DIGAMMA_BACKEND_12_TAIL;
    cfg.tolerance = 1.0e-12;
    cfg.shape = SIM_DIGAMMA_SQUARE_WAVEFORM_SAWTOOTH;
    cfg.use_wavevector = true;
    cfg.rotation = 0.23;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -0.58;
    cfg.coord.origin_y = 0.18;
    cfg.coord.spacing_x = 0.05;
    cfg.coord.spacing_y = 0.08;
    cfg.coord.velocity_x = 0.03;
    cfg.coord.velocity_y = -0.02;
    cfg.time_offset = -0.06;

    size_t op_index = 0U;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-complex] failed to add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_digamma_square_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-complex] failed to fetch normalized config\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_index, "digamma_complex")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *field = sim_context_field(&ctx, field_index);
    if (field == NULL || field->element_size != sizeof(SimComplexDouble)) {
        fprintf(stderr, "[digamma-complex] missing field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    const SimComplexDouble *data = sim_field_complex_data_const(field);
    if (data == NULL) {
        fprintf(stderr, "[digamma-complex] missing field data\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    size_t width = 96U;
    size_t height = 4U;
    double t = normalized.time_offset;
    double rot_cos = cos(normalized.rotation);
    double rot_sin = sin(normalized.rotation);
    double tol = 1.0e-9;
    for (size_t row = 0U; row < height; ++row) {
        for (size_t col = 0U; col < width; ++col) {
            size_t idx = row * width + col;
            double x = normalized.coord.origin_x + (double)col * normalized.coord.spacing_x;
            double y = normalized.coord.origin_y + (double)row * normalized.coord.spacing_y;
            double sample_x = x - normalized.coord.velocity_x * t;
            double sample_y = y - normalized.coord.velocity_y * t;
            double phase = normalized.kx * sample_x + normalized.ky * sample_y + normalized.phase -
                           normalized.omega * t;
            double sin_phase = sin(phase);
            double cos_phase = cos(phase);
            double inner = normalized.harmonics * cos_phase;
            SimComplexDouble base;
            double expected_re;
            double expected_im;

            if (fabs(normalized.a - 0.25) < 1.0e-6) {
                base = sim_digamma_square_base_complex(normalized.amplitude,
                                                       (SimComplexDouble){inner, 0.0},
                                                       normalized.backend, normalized.tolerance);
            } else {
                base = sim_digamma_square_base_deformed_complex(
                    normalized.amplitude, normalized.a, (SimComplexDouble){inner, 0.0},
                    normalized.backend, normalized.tolerance);
            }

            expected_re = (base.re * sin_phase) * rot_cos - (base.im * sin_phase) * rot_sin;
            expected_im = (base.re * sin_phase) * rot_sin + (base.im * sin_phase) * rot_cos;
            if (fabs(data[idx].re - expected_re) > tol || fabs(data[idx].im - expected_im) > tol) {
                fprintf(stderr,
                        "[digamma-complex] mismatch idx=%zu got=(%.12g, %.12g) expected=(%.12g, "
                        "%.12g)\n",
                        idx, data[idx].re, data[idx].im, expected_re, expected_im);
                sim_context_destroy(&ctx);
                return 0;
            }
        }
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int test_digamma_square_axis_velocity(void) {
    static const size_t kCount = 8U;
    static const double kTol = 1.0e-10;

    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-velocity] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_index = 0U;
    if (!add_real_field_1d(&ctx, kCount, &field_index)) {
        fprintf(stderr, "[digamma-velocity] failed to add field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusDigammaSquareConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.32;
    cfg.wavenumber = 1.1;
    cfg.omega = 0.17;
    cfg.phase = -0.21;
    cfg.harmonics = 3.4;
    cfg.backend = SIM_DIGAMMA_BACKEND_12_TAIL;
    cfg.tolerance = 1.0e-12;
    cfg.a = 0.29;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg.coord.origin_x = -0.4;
    cfg.coord.spacing_x = 0.2;
    cfg.coord.velocity_x = 0.15;
    cfg.time_offset = 0.37;

    size_t op_index = 0U;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[digamma-velocity] failed to add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_index, "digamma_velocity")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    SimField *field = sim_context_field(&ctx, field_index);
    if (field == NULL || field->element_size != sizeof(double)) {
        fprintf(stderr, "[digamma-velocity] failed to get field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    const double *data = (const double *)sim_field_data(field);
    if (data == NULL) {
        fprintf(stderr, "[digamma-velocity] missing field data\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    double delta = cfg.wavenumber * cfg.coord.spacing_x;
    bool phase_constant = fabs(delta) < 1.0e-12;
    double t = cfg.time_offset;

    for (size_t i = 0U; i < kCount; ++i) {
        double x = cfg.coord.origin_x + (double)i * cfg.coord.spacing_x;
        double sample_x = x - cfg.coord.velocity_x * t;
        double phase = cfg.wavenumber * sample_x + cfg.phase - cfg.omega * t;
        double basis = cos(phase);
        double base = sim_digamma_square_base_deformed_real(
            cfg.amplitude, cfg.a, cfg.harmonics * basis, cfg.backend, cfg.tolerance);
        double expected = phase_constant ? base : 0.5 * base;
        double err = fabs(data[i] - expected);
        if (err > kTol) {
            fprintf(stderr,
                    "[digamma-velocity] mismatch idx=%zu got=%.12g expected=%.12g err=%.3g "
                    "tol=%.3g\n",
                    i, data[i], expected, err, kTol);
            sim_context_destroy(&ctx);
            return 0;
        }
    }

    sim_context_destroy(&ctx);
    return 1;
}

static int test_random_fourier_wavevector(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[rff] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_auto = 0U;
    size_t field_axis = 0U;
    size_t field_sep = 0U;
    if (!add_real_field_2d(&ctx, 4U, 96U, &field_auto) ||
        !add_real_field_2d(&ctx, 4U, 96U, &field_axis) ||
        !add_real_field_2d(&ctx, 4U, 96U, &field_sep)) {
        fprintf(stderr, "[rff] failed to add fields\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRandomFourierConfig base;
    memset(&base, 0, sizeof(base));
    base.amplitude = 0.7;
    base.k_min = 0.4;
    base.k_max = 1.8;
    base.omega = -0.35;
    base.spectral_slope = 0.5;
    base.feature_count = 24U;
    base.seed = 12345U;
    base.scale_by_dt = false;
    base.coord.origin_x = -0.2;
    base.coord.origin_y = 0.8;
    base.coord.spacing_x = 0.4;
    base.coord.spacing_y = 0.55;
    base.kx = 0.91;
    base.ky = -0.26;

    SimStimulusRandomFourierConfig cfg_auto = base;
    cfg_auto.field_index = field_auto;
    cfg_auto.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_auto.coord.axis = SIM_STIMULUS_AXIS_X;
    cfg_auto.use_wavevector = false;

    size_t op_auto = 0U;
    if (sim_add_stimulus_random_fourier_operator(&ctx, &cfg_auto, &op_auto) != SIM_RESULT_OK) {
        fprintf(stderr, "[rff] failed to add auto-enable operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRandomFourierConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_random_fourier_config(&ctx, op_auto, &normalized) != SIM_RESULT_OK ||
        !normalized.use_wavevector) {
        fprintf(stderr, "[rff] expected use_wavevector auto-enable from kx/ky\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRandomFourierConfig cfg_axis = base;
    cfg_axis.field_index = field_axis;
    cfg_axis.use_wavevector = true;
    cfg_axis.coord.mode = SIM_STIMULUS_COORD_AXIS;
    cfg_axis.coord.axis = SIM_STIMULUS_AXIS_X;

    SimStimulusRandomFourierConfig cfg_sep = base;
    cfg_sep.field_index = field_sep;
    cfg_sep.use_wavevector = true;
    cfg_sep.coord.mode = SIM_STIMULUS_COORD_SEPARABLE;
    cfg_sep.coord.combine = SIM_STIMULUS_SEPARABLE_MULTIPLY;

    size_t op_axis = 0U;
    size_t op_sep = 0U;
    if (sim_add_stimulus_random_fourier_operator(&ctx, &cfg_axis, &op_axis) != SIM_RESULT_OK ||
        sim_add_stimulus_random_fourier_operator(&ctx, &cfg_sep, &op_sep) != SIM_RESULT_OK) {
        fprintf(stderr, "[rff] failed to add comparison operators\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_axis, "rff_axis") || !eval_operator(&ctx, op_sep, "rff_sep")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    int ok = compare_real_fields(&ctx, field_axis, field_sep, 1.0e-10, "rff_wavevector_mode");
    if (ok) {
        memset(&normalized, 0, sizeof(normalized));
        if (sim_stimulus_random_fourier_config(&ctx, op_axis, &normalized) != SIM_RESULT_OK) {
            fprintf(stderr, "[rff] failed to fetch normalized axis config\n");
            ok = 0;
        } else {
            ok &= compare_random_fourier_reference(&ctx, field_axis, &normalized, 1.0e-9,
                                                   "rff_wavevector_reference");
        }
    }
    sim_context_destroy(&ctx);
    return ok;
}

static int test_random_fourier_angle_reference(void) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[rff-angle] failed to init context\n");
        return 0;
    }

    sim_context_set_timestep(&ctx, 0.1f);

    size_t field_index = 0U;
    if (!add_real_field_2d(&ctx, 5U, 80U, &field_index)) {
        fprintf(stderr, "[rff-angle] failed to add field\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRandomFourierConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 0.58;
    cfg.k_min = 0.3;
    cfg.k_max = 1.6;
    cfg.omega = 0.21;
    cfg.spectral_slope = 0.75;
    cfg.feature_count = 32U;
    cfg.seed = 45678U;
    cfg.scale_by_dt = false;
    cfg.coord.mode = SIM_STIMULUS_COORD_ANGLE;
    cfg.coord.angle = 0.47;
    cfg.coord.origin_x = -0.35;
    cfg.coord.origin_y = 0.22;
    cfg.coord.spacing_x = 0.09;
    cfg.coord.spacing_y = 0.13;
    cfg.coord.velocity_x = 0.04;
    cfg.coord.velocity_y = -0.03;
    cfg.time_offset = -0.18;

    size_t op_index = 0U;
    if (sim_add_stimulus_random_fourier_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[rff-angle] failed to add operator\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    SimStimulusRandomFourierConfig normalized;
    memset(&normalized, 0, sizeof(normalized));
    if (sim_stimulus_random_fourier_config(&ctx, op_index, &normalized) != SIM_RESULT_OK) {
        fprintf(stderr, "[rff-angle] failed to fetch normalized config\n");
        sim_context_destroy(&ctx);
        return 0;
    }

    if (!eval_operator(&ctx, op_index, "rff_angle")) {
        sim_context_destroy(&ctx);
        return 0;
    }

    int ok = compare_random_fourier_reference(&ctx, field_index, &normalized, 1.0e-9,
                                              "rff_angle_reference");
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    int ok = 1;
    ok &= test_digamma_square_deformation_helpers();
    ok &= test_digamma_square_shape_outputs();
    ok &= test_digamma_square_wavevector();
    ok &= test_digamma_square_wide_wavevector_reference();
    ok &= test_digamma_square_complex_wavevector_reference();
    ok &= test_digamma_square_axis_velocity();
    ok &= test_random_fourier_wavevector();
    ok &= test_random_fourier_angle_reference();
    return ok ? 0 : 1;
}
