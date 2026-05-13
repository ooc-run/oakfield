/*
 * Migrated diffusion operator coverage for fractional-memory complex contracts.
 */
#include <math.h>
#include <oakfield/sim.h>
#include <stdio.h>
#include <string.h>

static int nearly_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= tol * scale;
}

int main(void) {
    SimContext ctx;
    SimField field = {0};
    size_t shape[1] = {16};
    size_t field_index = 0U;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to init context\n");
        return 1;
    }

    if (sim_field_init(&field, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "Failed to init field\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    // Promote to complex
    if (sim_field_promote_inplace_to_complex(&field) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to promote field to complex\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return 1;
    }

    // Initialize complex data with distinct scaling between imag and real
    SimComplexDouble *z = sim_field_complex_data(&field);
    size_t N = sim_field_element_count(&field.layout);
    for (size_t i = 0; i < N; ++i) {
        z[i].re = (double)(i + 1);
        z[i].im = 2.0 * (double)(i + 1);
    }

    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to add field to context\n");
        sim_field_destroy(&field); // ownership may have moved only on success
        sim_context_destroy(&ctx);
        return 1;
    }

    FractionalMemoryOperatorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.order = 0.5;      // fractional order
    cfg.gain = 0.2;       // apply noticeable change
    cfg.memory_steps = 8; // maintain reasonable history length

    size_t op_index = 0;
    if (sim_add_fractional_memory_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "Failed to register fractional memory operator\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    // Run several steps to exercise history usage
    const int steps = 12;
    for (int s = 0; s < steps; ++s) {
        if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
            fprintf(stderr, "Context execute failed at step %d\n", s);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    // Fetch updated field
    SimField *f2 = sim_context_field(&ctx, field_index);
    if (!f2) {
        fprintf(stderr, "Failed to retrieve field after execution\n");
        sim_context_destroy(&ctx);
        return 1;
    }
    const SimComplexDouble *out = sim_field_complex_data_const(f2);

    // Check that:
    // 1) Values changed from initialization
    // 2) Imag remains approximately 2x Real for each element (componentwise linearity)
    int changed = 0;
    for (size_t i = 0; i < N; ++i) {
        if (!nearly_equal(out[i].re, (double)(i + 1), 1e-6) ||
            !nearly_equal(out[i].im, 2.0 * (double)(i + 1), 1e-6)) {
            changed = 1; // any element changed suffices
            break;
        }
    }
    if (!changed) {
        fprintf(stderr, "Fractional memory operator produced no change\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (size_t i = 0; i < N; ++i) {
        double expect = 2.0 * out[i].re;
        if (!nearly_equal(out[i].im, expect, 1e-12)) {
            fprintf(stderr, "Complex proportionality broken at i=%zu: re=%g im=%g\n", i, out[i].re,
                    out[i].im);
            sim_context_destroy(&ctx);
            return 1;
        }
    }

    sim_context_destroy(&ctx);
    return 0;
}
