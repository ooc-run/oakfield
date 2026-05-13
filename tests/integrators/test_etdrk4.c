#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CubicReactionState {
    size_t field_index;
    double beta_re;
    double beta_im;
} CubicReactionState;

static bool nearly_equal(double a, double b, double rel_tol, double abs_tol) {
    double diff = fabs(a - b);
    double scale = fmax(fmax(fabs(a), fabs(b)), 1.0);
    return diff <= fmax(abs_tol, rel_tol * scale);
}

static bool fill_complex_field(SimField *field) {
    SimComplexDouble *data = sim_field_complex_data(field);
    size_t count = 0U;

    if (field == NULL || data == NULL) {
        return false;
    }

    count = sim_field_element_count(&field->layout);
    for (size_t i = 0U; i < count; ++i) {
        double x = (double)i;
        data[i].re = 0.35 * cos(0.31 * x) + 0.15 * sin(0.73 * x);
        data[i].im = 0.25 * sin(0.19 * x) - 0.10 * cos(0.57 * x);
    }

    return true;
}

static bool init_zero_complex_context(SimContext *ctx, size_t count, size_t *out_field_index,
                                      bool spectral_domain) {
    size_t shape[1] = {0U};
    SimField field = {0};
    SimComplexDouble *data = NULL;
    SimResult result = SIM_RESULT_OK;

    if (ctx == NULL || out_field_index == NULL || count == 0U) {
        return false;
    }

    shape[0] = count;
    if (sim_context_init(ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_field_init failed\n");
        sim_context_destroy(ctx);
        return false;
    }

    if (spectral_domain) {
        field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
        field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }

    data = sim_field_complex_data(&field);
    if (data == NULL) {
        fprintf(stderr, "[FAIL] sim_field_complex_data failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(ctx);
        return false;
    }
    memset(data, 0, count * sizeof(*data));

    result = sim_context_add_field(ctx, &field, out_field_index);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_add_field failed (%d)\n", (int)result);
        sim_field_destroy(&field);
        sim_context_destroy(ctx);
        return false;
    }

    sim_context_set_timestep(ctx, 0.02);
    return true;
}

static bool init_complex_context(SimContext *ctx, size_t *out_field_index, bool spectral_domain) {
    size_t shape[1] = {16U};
    SimField field = {0};
    SimResult result = SIM_RESULT_OK;

    if (ctx == NULL || out_field_index == NULL) {
        return false;
    }

    if (sim_context_init(ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_init failed\n");
        return false;
    }

    if (sim_field_init(&field, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_field_init failed\n");
        sim_context_destroy(ctx);
        return false;
    }

    if (spectral_domain) {
        field.repr.domain = SIM_FIELD_DOMAIN_SPECTRAL;
        field.repr.value_kind = SIM_FIELD_VALUE_COMPLEX_SCALAR;
    }

    if (!fill_complex_field(&field)) {
        fprintf(stderr, "[FAIL] fill_complex_field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(ctx);
        return false;
    }

    result = sim_context_add_field(ctx, &field, out_field_index);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_add_field failed (%d)\n", (int)result);
        sim_field_destroy(&field);
        sim_context_destroy(ctx);
        return false;
    }

    sim_context_set_timestep(ctx, 0.02);
    return true;
}

static bool complex_fields_match(const SimField *lhs, const SimField *rhs, double rel_tol,
                                 double abs_tol) {
    const SimComplexDouble *a = NULL;
    const SimComplexDouble *b = NULL;
    size_t count = 0U;

    if (lhs == NULL || rhs == NULL) {
        return false;
    }

    if (lhs->element_size != sizeof(SimComplexDouble) ||
        rhs->element_size != sizeof(SimComplexDouble) || lhs->layout.rank != rhs->layout.rank) {
        return false;
    }

    count = sim_field_element_count(&lhs->layout);
    if (count != sim_field_element_count(&rhs->layout)) {
        return false;
    }

    a = sim_field_complex_data_const(lhs);
    b = sim_field_complex_data_const(rhs);
    if (a == NULL || b == NULL) {
        return false;
    }

    for (size_t i = 0U; i < count; ++i) {
        if (!nearly_equal(a[i].re, b[i].re, rel_tol, abs_tol) ||
            !nearly_equal(a[i].im, b[i].im, rel_tol, abs_tol)) {
            fprintf(stderr,
                    "[FAIL] complex field mismatch at %zu: got (%.17g, %.17g) expected (%.17g, "
                    "%.17g)\n",
                    i, a[i].re, a[i].im, b[i].re, b[i].im);
            return false;
        }
    }

    return true;
}

static bool create_context_integrator(const char *name, SimContext *ctx, size_t field_index,
                                      double dt, Integrator *out) {
    IntegratorRegistry registry = {0};
    IntegratorConfig config = {0};
    SimResult result = SIM_RESULT_OK;

    if (name == NULL || ctx == NULL || out == NULL) {
        return false;
    }

    config.drift = integrator_context_drift;
    config.userdata = ctx;
    config.target_field_index = field_index;
    config.initial_dt = dt;
    config.min_dt = dt;
    config.max_dt = dt;
    config.tolerance = 1.0e-8;
    config.safety = 0.9;
    config.adaptive = false;

    if (integrator_registry_init(&registry) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integrator_registry_init failed\n");
        return false;
    }

    result = integrator_registry_create(&registry, name, &config, out);
    integrator_registry_destroy(&registry);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] integrator_registry_create('%s') failed (%d)\n", name, (int)result);
        return false;
    }

    return true;
}

static SimResult cubic_reaction_eval(SimContext *context, SimOperator *self, void *userdata) {
    CubicReactionState *state = (CubicReactionState *)userdata;
    SimField *field = NULL;
    SimComplexDouble *data = NULL;
    size_t count = 0U;
    double dt = 0.0;
    (void)self;

    if (context == NULL || state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    field = sim_context_field(context, state->field_index);
    if (field == NULL) {
        return SIM_RESULT_NOT_FOUND;
    }

    data = sim_field_complex_data(field);
    if (data == NULL) {
        return SIM_RESULT_TYPE_MISMATCH;
    }

    dt = sim_context_timestep(context);
    if (!(dt > 0.0)) {
        dt = 1.0;
    }

    count = sim_field_element_count(&field->layout);
    for (size_t i = 0U; i < count; ++i) {
        double re = data[i].re;
        double im = data[i].im;
        double mag2 = re * re + im * im;
        double drift_re = mag2 * (state->beta_re * re - state->beta_im * im);
        double drift_im = mag2 * (state->beta_re * im + state->beta_im * re);
        data[i].re += dt * drift_re;
        data[i].im += dt * drift_im;
    }

    return SIM_RESULT_OK;
}

static void cubic_reaction_destroy(void *userdata) { free(userdata); }

static bool add_cubic_reaction_operator(SimContext *ctx, size_t field_index) {
    CubicReactionState *state = NULL;
    SimOperatorDescriptor desc = {0};
    size_t field_indices[1] = {field_index};

    if (ctx == NULL) {
        return false;
    }

    state = (CubicReactionState *)calloc(1U, sizeof(CubicReactionState));
    if (state == NULL) {
        fprintf(stderr, "[FAIL] calloc for CubicReactionState failed\n");
        return false;
    }

    state->field_index = field_index;
    state->beta_re = -0.08;
    state->beta_im = 0.21;

    desc.name = "test_cubic_reaction";
    desc.evaluate = cubic_reaction_eval;
    desc.destroy = cubic_reaction_destroy;
    desc.userdata = state;
    desc.info = sim_operator_info_defaults();
    desc.info.category = SIM_OPERATOR_CATEGORY_REACTION;
    desc.info.warp_level = SIM_WARP_LEVEL_NONE;
    desc.info.is_differentiable = true;
    desc.info.abstract_id = "test_cubic_reaction";
    desc.read_mask = 1ULL << field_index;
    desc.write_mask = 1ULL << field_index;
    desc.read_indices = field_indices;
    desc.read_index_count = 1U;
    desc.write_indices = field_indices;
    desc.write_index_count = 1U;

    if (sim_context_register_operator(ctx, &desc, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_context_register_operator for cubic reaction failed\n");
        free(state);
        return false;
    }

    return true;
}

static bool build_phase_rotate_context(SimContext *ctx, size_t *out_field_index) {
    PhaseRotateOperatorConfig config = {0};

    if (!init_complex_context(ctx, out_field_index, false)) {
        return false;
    }

    config.field_index = *out_field_index;
    config.phase_rate = 0.35;

    if (sim_add_phase_rotate_operator(ctx, &config, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_phase_rotate_operator failed\n");
        sim_context_destroy(ctx);
        return false;
    }

    return true;
}

static bool build_nonlinear_context(SimContext *ctx, size_t *out_field_index) {
    if (!init_complex_context(ctx, out_field_index, false)) {
        return false;
    }

    if (!add_cubic_reaction_operator(ctx, *out_field_index)) {
        sim_context_destroy(ctx);
        return false;
    }

    return true;
}

static bool build_forced_sine_context(SimContext *ctx, size_t *out_field_index,
                                      PhaseRotateOperatorConfig *out_phase,
                                      SimStimulusSinusoidalConfig *out_stimulus) {
    PhaseRotateOperatorConfig phase = {0};
    SimStimulusSinusoidalConfig stimulus = {0};

    if (!init_zero_complex_context(ctx, 1U, out_field_index, false)) {
        return false;
    }

    phase.field_index = *out_field_index;
    phase.phase_rate = 0.45;
    if (sim_add_phase_rotate_operator(ctx, &phase, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_phase_rotate_operator failed for forced sine context\n");
        sim_context_destroy(ctx);
        return false;
    }

    stimulus.field_index = *out_field_index;
    stimulus.amplitude = 0.18;
    stimulus.wavenumber = 0.0;
    stimulus.omega = 1.35;
    stimulus.phase = 0.4;
    stimulus.rotation = 0.35;
    stimulus.fixed_clock = false;
    stimulus.scale_by_dt = true;
    stimulus.coord.mode = SIM_STIMULUS_COORD_AXIS;
    stimulus.coord.axis = SIM_STIMULUS_AXIS_X;
    stimulus.coord.origin_x = 0.0;
    stimulus.coord.spacing_x = 1.0;

    if (sim_add_stimulus_sine_operator(ctx, &stimulus, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_stimulus_sine_operator failed for forced sine context\n");
        sim_context_destroy(ctx);
        return false;
    }

    if (out_phase != NULL) {
        *out_phase = phase;
    }
    if (out_stimulus != NULL) {
        *out_stimulus = stimulus;
    }

    return true;
}

static bool build_linear_spectral_fusion_context(SimContext *ctx, size_t *out_field_index) {
    LinearSpectralFusionOperatorConfig config = {0};

    if (!init_complex_context(ctx, out_field_index, false)) {
        return false;
    }

    config.field_index = *out_field_index;
    config.viscosity = 0.32;
    config.alpha = 1.5;
    config.dissipation_spacing = 0.2;
    config.dispersion_coefficient = 0.45;
    config.dispersion_order = 2.0;
    config.dispersion_reference_k = 0.15;
    config.dispersion_spacing = 0.2;
    config.phase_rate = 0.28;

    if (sim_add_linear_spectral_fusion_operator(ctx, &config, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_linear_spectral_fusion_operator failed\n");
        sim_context_destroy(ctx);
        return false;
    }

    return true;
}

static bool build_random_fourier_forcing_context(SimContext *ctx, size_t *out_field_index) {
    PhaseRotateOperatorConfig phase = {0};
    SimStimulusRandomFourierConfig rff = {0};

    if (!init_zero_complex_context(ctx, 16U, out_field_index, false)) {
        return false;
    }

    phase.field_index = *out_field_index;
    phase.phase_rate = 0.22;
    if (sim_add_phase_rotate_operator(ctx, &phase, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_phase_rotate_operator failed for random fourier context\n");
        sim_context_destroy(ctx);
        return false;
    }

    rff.field_index = *out_field_index;
    rff.amplitude = 0.14;
    rff.k_min = 0.3;
    rff.k_max = 1.4;
    rff.omega = 0.85;
    rff.coord.mode = SIM_STIMULUS_COORD_AXIS;
    rff.coord.axis = SIM_STIMULUS_AXIS_X;
    rff.coord.origin_x = -0.25;
    rff.coord.spacing_x = 0.35;
    rff.time_offset = 0.12;
    rff.nominal_dt = 0.0;
    rff.spectral_slope = 0.4;
    rff.feature_count = 18U;
    rff.seed = 123456789ULL;
    rff.use_wavevector = false;
    rff.fixed_clock = false;
    rff.scale_by_dt = true;

    if (sim_add_stimulus_random_fourier_operator(ctx, &rff, NULL) != SIM_RESULT_OK) {
        fprintf(stderr,
                "[FAIL] sim_add_stimulus_random_fourier_operator failed for ETDRK4 context\n");
        sim_context_destroy(ctx);
        return false;
    }

    return true;
}

static bool build_moire_forcing_context(SimContext *ctx, size_t *out_field_index) {
    PhaseRotateOperatorConfig phase = {0};
    SimStimulusMoireConfig moire = {0};

    if (!init_zero_complex_context(ctx, 16U, out_field_index, false)) {
        return false;
    }

    phase.field_index = *out_field_index;
    phase.phase_rate = 0.19;
    if (sim_add_phase_rotate_operator(ctx, &phase, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_phase_rotate_operator failed for moire context\n");
        sim_context_destroy(ctx);
        return false;
    }

    moire.field_index = *out_field_index;
    moire.amplitude = 0.11;
    moire.wavenumber_a = 0.9;
    moire.wavenumber_b = 1.05;
    moire.omega_a = 0.6;
    moire.omega_b = 0.72;
    moire.phase_a = 0.1;
    moire.phase_b = -0.2;
    moire.coord.mode = SIM_STIMULUS_COORD_AXIS;
    moire.coord.axis = SIM_STIMULUS_AXIS_X;
    moire.coord.origin_x = -0.4;
    moire.coord.spacing_x = 0.25;
    moire.time_offset = 0.08;
    moire.rotation = 0.27;
    moire.use_wavevectors = false;
    moire.scale_by_dt = true;

    if (sim_add_stimulus_moire_operator(ctx, &moire, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_stimulus_moire_operator failed for ETDRK4 context\n");
        sim_context_destroy(ctx);
        return false;
    }

    return true;
}

static bool build_scale_context(SimContext *ctx, size_t *out_field_index) {
    PhaseRotateOperatorConfig phase = {0};
    SimScaleOperatorConfig scale = {0};

    if (!init_complex_context(ctx, out_field_index, false)) {
        return false;
    }

    phase.field_index = *out_field_index;
    phase.phase_rate = 0.31;
    if (sim_add_phase_rotate_operator(ctx, &phase, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_phase_rotate_operator failed for scale context\n");
        sim_context_destroy(ctx);
        return false;
    }

    scale.input_field = *out_field_index;
    scale.output_field = *out_field_index;
    scale.scale = -0.14;
    scale.accumulate = true;
    scale.scale_by_dt = true;

    if (sim_add_scale_operator(ctx, &scale, NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] sim_add_scale_operator failed for ETDRK4 context\n");
        sim_context_destroy(ctx);
        return false;
    }

    return true;
}

static bool test_etdrk4_matches_phase_rotate(void) {
    const double dt = 0.03125;
    SimContext etd_ctx = {0};
    SimContext ref_ctx = {0};
    Integrator etd = {0};
    size_t etd_field_index = 0U;
    size_t ref_field_index = 0U;
    bool ok = false;

    if (!build_phase_rotate_context(&etd_ctx, &etd_field_index) ||
        !build_phase_rotate_context(&ref_ctx, &ref_field_index)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &etd_ctx, etd_field_index, dt, &etd)) {
        goto cleanup;
    }

    if (integrator_step_context(&etd, &etd_ctx, sim_context_backend(&etd_ctx), dt) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] ETDRK4 step_context failed for phase rotate test\n");
        goto cleanup;
    }
    sim_context_accept_step(&etd_ctx, dt);

    sim_context_set_timestep(&ref_ctx, dt);
    if (sim_context_prepare_plan(&ref_ctx) != SIM_RESULT_OK ||
        sim_context_execute(&ref_ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] direct context execution failed for phase rotate reference\n");
        goto cleanup;
    }

    if (!complex_fields_match(sim_context_field(&etd_ctx, etd_field_index),
                              sim_context_field(&ref_ctx, ref_field_index), 1.0e-11, 1.0e-12)) {
        fprintf(stderr, "[FAIL] ETDRK4 phase rotate comparison failed\n");
        goto cleanup;
    }

    if (!nearly_equal(integrator_last_step(&etd), dt, 1.0e-12, 1.0e-12)) {
        fprintf(stderr, "[FAIL] ETDRK4 last_step mismatch for phase rotate test\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    sim_context_destroy(&etd_ctx);
    sim_context_destroy(&ref_ctx);
    return ok;
}

static bool test_etdrk4_matches_linear_spectral_fusion(void) {
    const double dt = 0.025;
    const size_t steps = 4U;
    SimContext etd_ctx = {0};
    SimContext ref_ctx = {0};
    Integrator etd = {0};
    size_t etd_field_index = 0U;
    size_t ref_field_index = 0U;
    bool ok = false;

    if (!build_linear_spectral_fusion_context(&etd_ctx, &etd_field_index) ||
        !build_linear_spectral_fusion_context(&ref_ctx, &ref_field_index)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &etd_ctx, etd_field_index, dt, &etd)) {
        goto cleanup;
    }

    for (size_t step_index = 0U; step_index < steps; ++step_index) {
        if (integrator_step_context(&etd, &etd_ctx, sim_context_backend(&etd_ctx), dt) !=
            SIM_RESULT_OK) {
            fprintf(stderr,
                    "[FAIL] ETDRK4 step_context failed for linear_spectral_fusion step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&etd_ctx, dt);

        sim_context_set_timestep(&ref_ctx, dt);
        if (sim_context_prepare_plan(&ref_ctx) != SIM_RESULT_OK ||
            sim_context_execute(&ref_ctx) != SIM_RESULT_OK) {
            fprintf(stderr,
                    "[FAIL] direct context execution failed for linear_spectral_fusion step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ref_ctx, dt);
    }

    if (!complex_fields_match(sim_context_field(&etd_ctx, etd_field_index),
                              sim_context_field(&ref_ctx, ref_field_index), 1.0e-10, 1.0e-11)) {
        fprintf(stderr, "[FAIL] ETDRK4 linear_spectral_fusion comparison failed\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    sim_context_destroy(&etd_ctx);
    sim_context_destroy(&ref_ctx);
    return ok;
}

static bool test_etdrk4_matches_fine_rk4_for_nonlinear_only(void) {
    const double dt = 0.02;
    const double reference_dt = 0.0005;
    const size_t steps = 6U;
    const size_t reference_steps = (size_t)((dt * (double)steps) / reference_dt);
    SimContext etd_ctx = {0};
    SimContext rk_ctx = {0};
    Integrator etd = {0};
    Integrator rk = {0};
    size_t etd_field_index = 0U;
    size_t rk_field_index = 0U;
    bool ok = false;

    if (!build_nonlinear_context(&etd_ctx, &etd_field_index) ||
        !build_nonlinear_context(&rk_ctx, &rk_field_index)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &etd_ctx, etd_field_index, dt, &etd) ||
        !create_context_integrator("rk4", &rk_ctx, rk_field_index, reference_dt, &rk)) {
        goto cleanup;
    }

    for (size_t step_index = 0U; step_index < steps; ++step_index) {
        if (integrator_step_context(&etd, &etd_ctx, sim_context_backend(&etd_ctx), dt) !=
            SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] ETDRK4 step_context failed on nonlinear-only step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&etd_ctx, dt);
    }

    for (size_t step_index = 0U; step_index < reference_steps; ++step_index) {
        if (integrator_step_context(&rk, &rk_ctx, sim_context_backend(&rk_ctx), reference_dt) !=
            SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] RK4 reference step_context failed on nonlinear-only step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&rk_ctx, reference_dt);
    }

    if (!complex_fields_match(sim_context_field(&etd_ctx, etd_field_index),
                              sim_context_field(&rk_ctx, rk_field_index), 5.0e-6, 5.0e-7)) {
        fprintf(stderr, "[FAIL] ETDRK4 nonlinear-only comparison failed against fine RK4\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    integrator_destroy(&rk);
    sim_context_destroy(&etd_ctx);
    sim_context_destroy(&rk_ctx);
    return ok;
}

static bool test_etdrk4_supports_time_dependent_forcing(void) {
    const double dt = 0.04;
    const size_t steps = 8U;
    const double reference_dt = 5.0e-5;
    const size_t reference_steps = (size_t)((dt * (double)steps) / reference_dt);
    SimContext ctx = {0};
    SimContext ref_ctx = {0};
    Integrator etd = {0};
    size_t field_index = 0U;
    size_t ref_field_index = 0U;
    bool ok = false;

    if (!build_forced_sine_context(&ctx, &field_index, NULL, NULL) ||
        !build_forced_sine_context(&ref_ctx, &ref_field_index, NULL, NULL)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &ctx, field_index, dt, &etd)) {
        goto cleanup;
    }

    for (size_t step_index = 0U; step_index < steps; ++step_index) {
        if (integrator_step_context(&etd, &ctx, sim_context_backend(&ctx), dt) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] ETDRK4 step_context failed for forced sine step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ctx, dt);
    }

    for (size_t step_index = 0U; step_index < reference_steps; ++step_index) {
        sim_context_set_timestep(&ref_ctx, reference_dt);
        if (sim_context_prepare_plan(&ref_ctx) != SIM_RESULT_OK ||
            sim_context_execute(&ref_ctx) != SIM_RESULT_OK) {
            fprintf(stderr,
                    "[FAIL] direct context execution failed for forced sine reference step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ref_ctx, reference_dt);
    }

    if (!complex_fields_match(sim_context_field(&ctx, field_index),
                              sim_context_field(&ref_ctx, ref_field_index), 3.0e-3, 3.0e-4)) {
        fprintf(stderr, "[FAIL] ETDRK4 forced sine comparison failed against fine reference\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    sim_context_destroy(&ctx);
    sim_context_destroy(&ref_ctx);
    return ok;
}

static bool test_etdrk4_supports_random_fourier_forcing(void) {
    const double dt = 0.02;
    const size_t steps = 6U;
    const double reference_dt = 5.0e-5;
    const size_t reference_steps = (size_t)((dt * (double)steps) / reference_dt);
    SimContext ctx = {0};
    SimContext ref_ctx = {0};
    Integrator etd = {0};
    size_t field_index = 0U;
    size_t ref_field_index = 0U;
    bool ok = false;

    if (!build_random_fourier_forcing_context(&ctx, &field_index) ||
        !build_random_fourier_forcing_context(&ref_ctx, &ref_field_index)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &ctx, field_index, dt, &etd)) {
        goto cleanup;
    }

    for (size_t step_index = 0U; step_index < steps; ++step_index) {
        if (integrator_step_context(&etd, &ctx, sim_context_backend(&ctx), dt) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] ETDRK4 step_context failed for random fourier step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ctx, dt);
    }

    for (size_t step_index = 0U; step_index < reference_steps; ++step_index) {
        sim_context_set_timestep(&ref_ctx, reference_dt);
        if (sim_context_prepare_plan(&ref_ctx) != SIM_RESULT_OK ||
            sim_context_execute(&ref_ctx) != SIM_RESULT_OK) {
            fprintf(
                stderr,
                "[FAIL] direct context execution failed for random fourier reference step %zu\n",
                step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ref_ctx, reference_dt);
    }

    if (!complex_fields_match(sim_context_field(&ctx, field_index),
                              sim_context_field(&ref_ctx, ref_field_index), 5.0e-3, 5.0e-4)) {
        fprintf(stderr, "[FAIL] ETDRK4 random fourier comparison failed against fine reference\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    sim_context_destroy(&ctx);
    sim_context_destroy(&ref_ctx);
    return ok;
}

static bool test_etdrk4_supports_moire_forcing(void) {
    const double dt = 0.02;
    const size_t steps = 6U;
    const double reference_dt = 5.0e-5;
    const size_t reference_steps = (size_t)((dt * (double)steps) / reference_dt);
    SimContext ctx = {0};
    SimContext ref_ctx = {0};
    Integrator etd = {0};
    size_t field_index = 0U;
    size_t ref_field_index = 0U;
    bool ok = false;

    if (!build_moire_forcing_context(&ctx, &field_index) ||
        !build_moire_forcing_context(&ref_ctx, &ref_field_index)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &ctx, field_index, dt, &etd)) {
        goto cleanup;
    }

    for (size_t step_index = 0U; step_index < steps; ++step_index) {
        if (integrator_step_context(&etd, &ctx, sim_context_backend(&ctx), dt) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] ETDRK4 step_context failed for moire step %zu\n", step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ctx, dt);
    }

    for (size_t step_index = 0U; step_index < reference_steps; ++step_index) {
        sim_context_set_timestep(&ref_ctx, reference_dt);
        if (sim_context_prepare_plan(&ref_ctx) != SIM_RESULT_OK ||
            sim_context_execute(&ref_ctx) != SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] direct context execution failed for moire reference step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&ref_ctx, reference_dt);
    }

    if (!complex_fields_match(sim_context_field(&ctx, field_index),
                              sim_context_field(&ref_ctx, ref_field_index), 4.0e-3, 4.0e-4)) {
        fprintf(stderr, "[FAIL] ETDRK4 moire comparison failed against fine reference\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    sim_context_destroy(&ctx);
    sim_context_destroy(&ref_ctx);
    return ok;
}

static bool test_etdrk4_supports_same_field_scale_operator(void) {
    const double dt = 0.02;
    const double reference_dt = 0.0005;
    const size_t steps = 6U;
    const size_t reference_steps = (size_t)((dt * (double)steps) / reference_dt);
    SimContext etd_ctx = {0};
    SimContext rk_ctx = {0};
    Integrator etd = {0};
    Integrator rk = {0};
    size_t etd_field_index = 0U;
    size_t rk_field_index = 0U;
    bool ok = false;

    if (!build_scale_context(&etd_ctx, &etd_field_index) ||
        !build_scale_context(&rk_ctx, &rk_field_index)) {
        goto cleanup;
    }

    if (!create_context_integrator("etdrk4", &etd_ctx, etd_field_index, dt, &etd) ||
        !create_context_integrator("rk4", &rk_ctx, rk_field_index, reference_dt, &rk)) {
        goto cleanup;
    }

    for (size_t step_index = 0U; step_index < steps; ++step_index) {
        if (integrator_step_context(&etd, &etd_ctx, sim_context_backend(&etd_ctx), dt) !=
            SIM_RESULT_OK) {
            fprintf(stderr, "[FAIL] ETDRK4 step_context failed on same-field scale step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&etd_ctx, dt);
    }

    for (size_t step_index = 0U; step_index < reference_steps; ++step_index) {
        if (integrator_step_context(&rk, &rk_ctx, sim_context_backend(&rk_ctx), reference_dt) !=
            SIM_RESULT_OK) {
            fprintf(stderr,
                    "[FAIL] RK4 reference step_context failed on same-field scale step %zu\n",
                    step_index);
            goto cleanup;
        }
        sim_context_accept_step(&rk_ctx, reference_dt);
    }

    if (!complex_fields_match(sim_context_field(&etd_ctx, etd_field_index),
                              sim_context_field(&rk_ctx, rk_field_index), 5.0e-6, 5.0e-7)) {
        fprintf(stderr, "[FAIL] ETDRK4 same-field scale comparison failed against fine RK4\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    integrator_destroy(&etd);
    integrator_destroy(&rk);
    sim_context_destroy(&etd_ctx);
    sim_context_destroy(&rk_ctx);
    return ok;
}

int main(void) {
    bool ok = true;

    ok &= test_etdrk4_matches_phase_rotate();
    ok &= test_etdrk4_matches_linear_spectral_fusion();
    ok &= test_etdrk4_matches_fine_rk4_for_nonlinear_only();
    ok &= test_etdrk4_supports_time_dependent_forcing();
    ok &= test_etdrk4_supports_random_fourier_forcing();
    ok &= test_etdrk4_supports_moire_forcing();
    ok &= test_etdrk4_supports_same_field_scale_operator();

    return ok ? 0 : 1;
}
