#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static bool setup_complex_mix_context(SimContext *ctx, size_t count, double dt, size_t *out_lhs,
                                      size_t *out_rhs, size_t *out_dst) {
    SimField lhs = {0};
    SimField rhs = {0};
    SimField dst = {0};
    size_t shape[1] = {count};
    size_t lhs_index = 0U;
    size_t rhs_index = 0U;
    size_t dst_index = 0U;
    SimResult rc = SIM_RESULT_OK;

    if (sim_context_init(ctx) != SIM_RESULT_OK) {
        return false;
    }

    rc = sim_field_init(&lhs, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                        NULL);
    if (rc != SIM_RESULT_OK) {
        sim_context_destroy(ctx);
        return false;
    }
    rc = sim_field_init(&rhs, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                        NULL);
    if (rc != SIM_RESULT_OK) {
        sim_field_destroy(&lhs);
        sim_context_destroy(ctx);
        return false;
    }
    rc = sim_field_init(&dst, 1U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                        NULL);
    if (rc != SIM_RESULT_OK) {
        sim_field_destroy(&lhs);
        sim_field_destroy(&rhs);
        sim_context_destroy(ctx);
        return false;
    }

    SimComplexDouble *lhs_values = sim_field_complex_data(&lhs);
    SimComplexDouble *rhs_values = sim_field_complex_data(&rhs);
    SimComplexDouble *dst_values = sim_field_complex_data(&dst);
    if (lhs_values == NULL || rhs_values == NULL || dst_values == NULL) {
        sim_field_destroy(&lhs);
        sim_field_destroy(&rhs);
        sim_field_destroy(&dst);
        sim_context_destroy(ctx);
        return false;
    }

    for (size_t i = 0U; i < count; ++i) {
        double x = (double)i / (double)count;
        lhs_values[i].re = 0.80 * cos(2.0 * M_PI * x) + 0.15 * sin(6.0 * M_PI * x);
        lhs_values[i].im = 0.35 * sin(4.0 * M_PI * x) - 0.20 * cos(10.0 * M_PI * x);
        rhs_values[i].re = 0.55 * sin(3.0 * M_PI * x) + 0.10 * cos(5.0 * M_PI * x);
        rhs_values[i].im = 0.25 * cos(7.0 * M_PI * x) + 0.05 * sin(9.0 * M_PI * x);
        dst_values[i].re = 0.12 * sin(11.0 * M_PI * x);
        dst_values[i].im = -0.09 * cos(13.0 * M_PI * x);
    }

    if (sim_context_add_field(ctx, &lhs, &lhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(ctx, &rhs, &rhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(ctx, &dst, &dst_index) != SIM_RESULT_OK) {
        sim_field_destroy(&lhs);
        sim_field_destroy(&rhs);
        sim_field_destroy(&dst);
        sim_context_destroy(ctx);
        return false;
    }

    sim_context_set_timestep(ctx, (float)dt);
    if (out_lhs != NULL) {
        *out_lhs = lhs_index;
    }
    if (out_rhs != NULL) {
        *out_rhs = rhs_index;
    }
    if (out_dst != NULL) {
        *out_dst = dst_index;
    }
    return true;
}

static bool run_metal_mix_kernel_update(size_t count) {
    const double dt = 0.0125;
    const size_t pre_steps = 4U;
    const size_t post_steps = 6U;
    const double tolerance_max = 2.5e-3;

    SimBackend backend = {0};
    backend.type = SIM_BACKEND_TYPE_METAL;
    backend_init(&backend);
    if (backend.last_error != SIM_RESULT_OK) {
        return true;
    }

    SimContext split_ctx = {0};
    SimContext kernel_ctx = {0};
    size_t split_lhs = 0U;
    size_t split_rhs = 0U;
    size_t split_dst = 0U;
    size_t kernel_lhs = 0U;
    size_t kernel_rhs = 0U;
    size_t kernel_dst = 0U;
    size_t split_op = 0U;
    size_t kernel_op = 0U;
    bool ok = false;

    if (!setup_complex_mix_context(&split_ctx, count, dt, &split_lhs, &split_rhs, &split_dst)) {
        goto cleanup;
    }
    if (!setup_complex_mix_context(&kernel_ctx, count, dt, &kernel_lhs, &kernel_rhs, &kernel_dst)) {
        goto cleanup;
    }
    sim_context_set_backend(&kernel_ctx, &backend);

    SimMetalMixOperatorConfig cfg = {.lhs_field = split_lhs,
                                     .rhs_field = split_rhs,
                                     .output_field = split_dst,
                                     .lhs_gain = 0.85,
                                     .rhs_gain = -0.40,
                                     .mix = 0.20,
                                     .bias = 0.07,
                                     .mode = SIM_MIXER_MODE_LINEAR,
                                     .accumulate = true,
                                     .scale_by_dt = true};

    if (sim_add_metal_mix_operator(&split_ctx, &cfg, &split_op) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] metal_mix split registration failed\n");
        goto cleanup;
    }

    cfg.lhs_field = kernel_lhs;
    cfg.rhs_field = kernel_rhs;
    cfg.output_field = kernel_dst;
    if (sim_add_metal_mix_operator(&kernel_ctx, &cfg, &kernel_op) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] metal_mix kernel registration failed\n");
        goto cleanup;
    }

    SimOperator *kernel_op_ptr = sim_operator_registry_get(&kernel_ctx.world.operators, kernel_op);
    if (kernel_op_ptr == NULL || kernel_op_ptr->kernel == NULL) {
        fprintf(stderr, "[FAIL] metal_mix kernel path not registered\n");
        goto cleanup;
    }

    if (sim_context_prepare_plan(&split_ctx) != SIM_RESULT_OK ||
        sim_context_prepare_plan(&kernel_ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    for (size_t i = 0U; i < pre_steps; ++i) {
        if (sim_context_execute(&split_ctx) != SIM_RESULT_OK ||
            sim_context_execute(&kernel_ctx) != SIM_RESULT_OK) {
            goto cleanup;
        }
    }

    SimMetalMixOperatorConfig update_split = cfg;
    update_split.lhs_field = split_lhs;
    update_split.rhs_field = split_rhs;
    update_split.output_field = split_dst;
    update_split.mode = SIM_MIXER_MODE_CROSSFADE;
    update_split.mix = 0.68;
    update_split.lhs_gain = 1.05;
    update_split.rhs_gain = 0.55;
    update_split.bias = -0.03;
    update_split.accumulate = true;
    update_split.scale_by_dt = false;

    if (sim_metal_mix_update(&split_ctx, split_op, &update_split) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] metal_mix split update failed\n");
        goto cleanup;
    }

    SimMetalMixOperatorConfig update_kernel = update_split;
    update_kernel.lhs_field = kernel_lhs;
    update_kernel.rhs_field = kernel_rhs;
    update_kernel.output_field = kernel_dst;
    if (sim_metal_mix_update(&kernel_ctx, kernel_op, &update_kernel) != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] metal_mix kernel update failed\n");
        goto cleanup;
    }

    for (size_t i = 0U; i < post_steps; ++i) {
        if (sim_context_execute(&split_ctx) != SIM_RESULT_OK ||
            sim_context_execute(&kernel_ctx) != SIM_RESULT_OK) {
            goto cleanup;
        }
    }

    SimField *split_field = sim_context_field(&split_ctx, split_dst);
    SimField *kernel_field = sim_context_field(&kernel_ctx, kernel_dst);
    if (split_field == NULL || kernel_field == NULL) {
        goto cleanup;
    }

    SimComplexDouble *split_values = sim_field_complex_data(split_field);
    SimComplexDouble *kernel_values = sim_field_complex_data(kernel_field);
    if (split_values == NULL || kernel_values == NULL) {
        goto cleanup;
    }

    double max_err = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        double err_re = fabs(split_values[i].re - kernel_values[i].re);
        double err_im = fabs(split_values[i].im - kernel_values[i].im);
        double err = fmax(err_re, err_im);
        if (err > max_err) {
            max_err = err;
        }
    }

    if (max_err > tolerance_max) {
        fprintf(stderr, "[FAIL] metal_mix kernel/update mismatch: max_err=%.12g\n", max_err);
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_context_destroy(&split_ctx);
    sim_context_destroy(&kernel_ctx);
    backend_destroy(&backend);
    return ok;
}

static bool run_metal_mix_real_case(size_t count) {
    SimContext ctx = {0};
    SimField lhs = {0};
    SimField rhs = {0};
    SimField dst = {0};
    size_t shape[1] = {count};
    size_t lhs_index = 0U;
    size_t rhs_index = 0U;
    size_t dst_index = 0U;
    size_t op_index = 0U;
    const double dt = 0.05;
    const double tol = 1.0e-12;
    bool ok = false;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        return false;
    }

    if (sim_field_init(&lhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&rhs, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK ||
        sim_field_init(&dst, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
            SIM_RESULT_OK) {
        goto cleanup;
    }

    double *lhs_values = sim_field_data(&lhs);
    double *rhs_values = sim_field_data(&rhs);
    double *dst_values = sim_field_data(&dst);
    if (lhs_values == NULL || rhs_values == NULL || dst_values == NULL) {
        goto cleanup;
    }

    for (size_t i = 0U; i < count; ++i) {
        double x = (double)i / (double)count;
        lhs_values[i] = 0.9 * cos(2.0 * M_PI * x) + 0.15 * sin(5.0 * M_PI * x);
        rhs_values[i] = -0.4 * sin(3.0 * M_PI * x) + 0.2 * cos(7.0 * M_PI * x);
        dst_values[i] = 0.25 * sin(11.0 * M_PI * x) - 0.05 * cos(13.0 * M_PI * x);
    }

    if (sim_context_add_field(&ctx, &lhs, &lhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &rhs, &rhs_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &dst, &dst_index) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimMetalMixOperatorConfig cfg = {.lhs_field = lhs_index,
                                     .rhs_field = rhs_index,
                                     .output_field = dst_index,
                                     .lhs_gain = 1.15,
                                     .rhs_gain = -0.45,
                                     .mix = 0.70,
                                     .bias = 0.08,
                                     .mode = SIM_MIXER_MODE_CROSSFADE,
                                     .accumulate = false,
                                     .scale_by_dt = true};

    if (sim_add_metal_mix_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        goto cleanup;
    }

    sim_context_set_timestep(&ctx, (float)dt);
    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        goto cleanup;
    }

    SimField *dst_field = sim_context_field(&ctx, dst_index);
    if (dst_field == NULL) {
        goto cleanup;
    }

    double *result = sim_field_data(dst_field);
    if (result == NULL) {
        goto cleanup;
    }

    for (size_t i = 0U; i < count; ++i) {
        double left = cfg.lhs_gain * lhs_values[i];
        double right = cfg.rhs_gain * rhs_values[i];
        double expected = (1.0 - cfg.mix) * left + cfg.mix * right + cfg.bias;
        if (fabs(result[i] - expected) > tol) {
            fprintf(stderr, "[FAIL] metal_mix real mismatch at %zu: got %.15g expected %.15g\n", i,
                    result[i], expected);
            goto cleanup;
        }
    }

    ok = true;

cleanup:
    sim_context_destroy(&ctx);
    return ok;
}

int main(void) {
    if (!run_metal_mix_real_case(257U)) {
        return 1;
    }
    if (!run_metal_mix_kernel_update(192U)) {
        return 1;
    }

    printf("[PASS] metal_mix tests\n");
    return 0;
}
