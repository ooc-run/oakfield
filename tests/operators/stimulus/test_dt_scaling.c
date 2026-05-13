#include <math.h>
#include <oakfield/sim.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static double nearly_equal(double a, double b, double rel_tol) {
    double diff = fabs(a - b);
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= rel_tol * scale;
}

static double accumulate_sinusoidal_signal(double dt) {
    SimContext ctx;
    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_init failed\n");
        return NAN;
    }

    sim_context_set_timestep(&ctx, (float)dt);

    size_t shape[1] = {1U};
    SimField field = {0};
    if (sim_field_init(&field, 1, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        fprintf(stderr, "sim_field_init failed\n");
        sim_context_destroy(&ctx);
        return NAN;
    }

    double *raw = (double *)sim_field_data(&field);
    if (raw != NULL) {
        memset(raw, 0, sim_field_bytes(&field));
    }

    size_t field_index = 0U;
    if (sim_context_add_field(&ctx, &field, &field_index) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_context_add_field failed\n");
        sim_field_destroy(&field);
        sim_context_destroy(&ctx);
        return NAN;
    }

    SimStimulusSinusoidalConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.field_index = field_index;
    cfg.amplitude = 3.0;
    cfg.wavenumber = 0.0;
    cfg.omega = 0.0;
    cfg.phase = 0.5 * M_PI; /* ensures sin argument is unity */
    cfg.scale_by_dt = false;

    size_t op_index = 0U;
    if (sim_add_stimulus_sine_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "sim_add_stimulus_sine_operator failed\n");
        sim_context_destroy(&ctx);
        return NAN;
    }

    SimOperator *op = sim_operator_registry_get(&ctx.world.operators, op_index);
    if (op == NULL) {
        fprintf(stderr, "operator lookup failed\n");
        sim_context_destroy(&ctx);
        return NAN;
    }

    if (op->evaluate(&ctx, op, op->userdata) != SIM_RESULT_OK) {
        fprintf(stderr, "sinusoidal stimulus evaluate failed\n");
        sim_context_destroy(&ctx);
        return NAN;
    }

    SimField *out_field = sim_context_field(&ctx, field_index);
    double value = NAN;
    if (out_field != NULL) {
        double *data = (double *)sim_field_data(out_field);
        if (data != NULL) {
            value = data[0];
        }
    }

    sim_context_destroy(&ctx);
    return value;
}

int main(void) {
    const double dt_small = 0.02;
    const double dt_large = 0.1;

    double signal_small = accumulate_sinusoidal_signal(dt_small);
    double signal_large = accumulate_sinusoidal_signal(dt_large);

    if (!isfinite(signal_small) || !isfinite(signal_large)) {
        fprintf(stderr, "Failed to evaluate sinusoidal stimulus responses (small=%g large=%g)\n",
                signal_small, signal_large);
        return 1;
    }

    if (!nearly_equal(signal_small, signal_large, 1.0e-6)) {
        fprintf(stderr, "Sinusoidal signal should be dt-independent but differed (%.9g vs %.9g)\n",
                signal_small, signal_large);
        return 1;
    }

    return 0;
}
