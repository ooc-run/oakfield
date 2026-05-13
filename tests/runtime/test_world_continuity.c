#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

int main(void) {
    SimContext context;
    SimField field = {0};
    size_t shape[1] = {8U};
    SimResult result = sim_context_init(&context);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_init failed: %d\n", (int)result);
        return 1;
    }

    result = sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "sim_field_init failed: %d\n", (int)result);
        sim_context_destroy(&context);
        return 1;
    }

    size_t field_index = SIZE_MAX;
    result = sim_context_add_field(&context, &field, &field_index);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_add_field failed: %d\n", (int)result);
        sim_field_destroy(&field);
        sim_context_destroy(&context);
        return 1;
    }

    SimOperatorConfig fetched;
    if (sim_world_field_continuity_override(&context.world, field_index, &fetched)) {
        fprintf(stderr, "unexpected default continuity override\n");
        sim_context_destroy(&context);
        return 1;
    }

    SimOperatorConfig override_cfg = sim_operator_config_defaults();
    override_cfg.continuity = SIM_CONTINUITY_CLAMPED;
    override_cfg.clamp_min = -2.0;
    override_cfg.clamp_max = 2.0;
    override_cfg.continuity_tol = 1.0e-4;
    {
        double spacing_vals[2] = {0.1, 0.2};
        sim_operator_config_set_spacing(&override_cfg, spacing_vals, 2U);
    }

    result =
        sim_world_set_field_continuity_override(&context.world, field_index, true, &override_cfg);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "set_field_continuity_override (enable) failed: %d\n", (int)result);
        sim_context_destroy(&context);
        return 1;
    }

    if (!sim_world_field_continuity_override(&context.world, field_index, &fetched)) {
        fprintf(stderr, "continuity override missing after enable\n");
        sim_context_destroy(&context);
        return 1;
    }

    if (fetched.continuity != override_cfg.continuity ||
        fabs(fetched.clamp_min - override_cfg.clamp_min) > 1.0e-6 ||
        fabs(fetched.clamp_max - override_cfg.clamp_max) > 1.0e-6 ||
        fabs(fetched.continuity_tol - override_cfg.continuity_tol) > 1.0e-6 ||
        fetched.boundary != override_cfg.boundary ||
        fetched.spacing_rank != override_cfg.spacing_rank ||
        fabs(fetched.spacing[0] - override_cfg.spacing[0]) > 1.0e-6 ||
        fabs(fetched.spacing[1] - override_cfg.spacing[1]) > 1.0e-6) {
        fprintf(stderr, "continuity override roundtrip mismatch\n");
        sim_context_destroy(&context);
        return 1;
    }

    result = sim_world_set_field_continuity_override(&context.world, field_index, false, NULL);
    if (result != SIM_RESULT_OK) {
        fprintf(stderr, "set_field_continuity_override (disable) failed: %d\n", (int)result);
        sim_context_destroy(&context);
        return 1;
    }

    if (sim_world_field_continuity_override(&context.world, field_index, NULL)) {
        fprintf(stderr, "continuity override still active after disable\n");
        sim_context_destroy(&context);
        return 1;
    }

    result = sim_world_set_field_continuity_override(&context.world, field_index + 1U, true,
                                                     &override_cfg);
    if (result != SIM_RESULT_INVALID_ARGUMENT) {
        fprintf(stderr, "expected invalid argument for out-of-range field override\n");
        sim_context_destroy(&context);
        return 1;
    }

    bool has_invalid = sim_world_field_continuity_override(&context.world, field_index + 1U, NULL);
    if (has_invalid) {
        fprintf(stderr, "unexpected override reported for out-of-range field\n");
        sim_context_destroy(&context);
        return 1;
    }

    sim_context_destroy(&context);
    return 0;
}
