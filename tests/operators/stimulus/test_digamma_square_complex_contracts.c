/*
 * Migrated stimulus complex-output contract coverage from the legacy suite.
 */
#include <oakfield/sim.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

static bool add_field(SimContext *context, size_t rank, const size_t *shape, size_t element_size,
                      size_t *out_index) {
    SimField field = {0};
    if (sim_field_init(&field, rank, shape, element_size, SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        return false;
    }
    if (sim_context_add_field(context, &field, out_index) != SIM_RESULT_OK) {
        sim_field_destroy(&field);
        return false;
    }
    return true;
}

static bool load_info(SimContext *context, size_t operator_index, SimOperatorInfo *out_info) {
    SimOperator *op = sim_operator_registry_get(&context->world.operators, operator_index);
    if (op == NULL || out_info == NULL) {
        return false;
    }
    *out_info = sim_operator_info(op);
    return true;
}

static bool expect_complex_metadata(const char *key, SimOperatorInfo info) {
    if (!info.preserves_real || info.representation.value_kind != SIM_FIELD_VALUE_COMPLEX_SCALAR ||
        !info.representation.requires_complex_input ||
        !info.representation.requires_complex_representation ||
        !info.representation.preserves_real_subspace) {
        fprintf(stderr, "[FAIL] complex metadata mismatch for %s\n", key);
        return false;
    }
    return true;
}

static bool check_plain_complex_metadata(void) {
    SimContext ctx = {0};
    size_t shape[1] = {8U};
    size_t field_index = SIZE_MAX;
    size_t operator_index = SIZE_MAX;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK ||
        !add_field(&ctx, 1U, shape, sizeof(SimComplexDouble), &field_index)) {
        return false;
    }

    SimStimulusDigammaSquareConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.amplitude = 0.25;
    cfg.a = 0.35;
    cfg.wavenumber = 0.5;
    cfg.omega = 1.0;
    cfg.rotation = 0.2;
    if (sim_add_stimulus_digamma_square_operator(&ctx, &cfg, &operator_index) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add stimulus_digamma_square operator\n");
        goto cleanup;
    }

    {
        SimOperatorInfo info;
        if (!load_info(&ctx, operator_index, &info) ||
            !expect_complex_metadata("stimulus_digamma_square", info)) {
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

static bool check_warp_complex_metadata(void) {
    SimContext ctx = {0};
    size_t shape[1] = {8U};
    size_t field_index = SIZE_MAX;
    size_t warp_index = SIZE_MAX;
    size_t operator_index = SIZE_MAX;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK ||
        !add_field(&ctx, 1U, shape, sizeof(SimComplexDouble), &field_index) ||
        !add_field(&ctx, 1U, shape, sizeof(SimComplexDouble), &warp_index)) {
        return false;
    }

    SimStimulusDigammaSquareConfig cfg = {0};
    cfg.field_index = field_index;
    cfg.warp_field_index = warp_index;
    cfg.use_warp = true;
    cfg.amplitude = 0.25;
    cfg.a = 0.35;
    cfg.wavenumber = 0.5;
    cfg.omega = 1.0;
    cfg.rotation = 0.2;
    if (sim_add_digamma_square_operator(&ctx, field_index, &cfg, &operator_index) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] add digamma_square operator with warp field\n");
        goto cleanup;
    }

    {
        SimOperatorInfo info;
        if (!load_info(&ctx, operator_index, &info) ||
            !expect_complex_metadata("stimulus_digamma_square", info)) {
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    bool ok = true;

    ok = ok && check_plain_complex_metadata();
    ok = ok && check_warp_complex_metadata();

    return ok ? 0 : 1;
}
