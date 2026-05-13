/*
 * Migrated advection operator coverage for complex divergence contracts.
 */
#include <oakfield/sim.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static size_t wrap_index(ptrdiff_t coord, size_t extent) {
    ptrdiff_t wrapped = coord % (ptrdiff_t)extent;
    if (wrapped < 0) {
        wrapped += (ptrdiff_t)extent;
    }
    return (size_t)wrapped;
}

static SimComplexDouble sample_periodic(const SimComplexDouble *data, size_t height, size_t width,
                                        ptrdiff_t y, ptrdiff_t x) {
    SimComplexDouble value = {0.0, 0.0};
    if (data == NULL || height == 0U || width == 0U) {
        return value;
    }

    size_t iy = wrap_index(y, height);
    size_t ix = wrap_index(x, width);
    return data[iy * width + ix];
}

static SimComplexDouble derivative_periodic(const SimComplexDouble *data, size_t height,
                                            size_t width, size_t y, size_t x, size_t axis,
                                            SimDivergenceStencil stencil, double spacing) {
    SimComplexDouble result = {0.0, 0.0};
    if (data == NULL || spacing == 0.0) {
        return result;
    }

    ptrdiff_t dy = (axis == 0U) ? 1 : 0;
    ptrdiff_t dx = (axis == 1U) ? 1 : 0;

    SimComplexDouble f0 = sample_periodic(data, height, width, (ptrdiff_t)y, (ptrdiff_t)x);
    SimComplexDouble fp1 =
        sample_periodic(data, height, width, (ptrdiff_t)y + dy, (ptrdiff_t)x + dx);
    SimComplexDouble fm1 =
        sample_periodic(data, height, width, (ptrdiff_t)y - dy, (ptrdiff_t)x - dx);
    SimComplexDouble fp2 =
        sample_periodic(data, height, width, (ptrdiff_t)y + 2 * dy, (ptrdiff_t)x + 2 * dx);
    SimComplexDouble fm2 =
        sample_periodic(data, height, width, (ptrdiff_t)y - 2 * dy, (ptrdiff_t)x - 2 * dx);

    switch (stencil) {
    case SIM_DIVERGENCE_STENCIL_FORWARD_1:
        result.re = (fp1.re - f0.re) / spacing;
        result.im = (fp1.im - f0.im) / spacing;
        break;
    case SIM_DIVERGENCE_STENCIL_BACKWARD_1:
        result.re = (f0.re - fm1.re) / spacing;
        result.im = (f0.im - fm1.im) / spacing;
        break;
    case SIM_DIVERGENCE_STENCIL_FORWARD_2:
        result.re = (-3.0 * f0.re + 4.0 * fp1.re - fp2.re) / (2.0 * spacing);
        result.im = (-3.0 * f0.im + 4.0 * fp1.im - fp2.im) / (2.0 * spacing);
        break;
    case SIM_DIVERGENCE_STENCIL_BACKWARD_2:
        result.re = (3.0 * f0.re - 4.0 * fm1.re + fm2.re) / (2.0 * spacing);
        result.im = (3.0 * f0.im - 4.0 * fm1.im + fm2.im) / (2.0 * spacing);
        break;
    case SIM_DIVERGENCE_STENCIL_CENTRAL_4:
        result.re = (-fp2.re + 8.0 * fp1.re - 8.0 * fm1.re + fm2.re) / (12.0 * spacing);
        result.im = (-fp2.im + 8.0 * fp1.im - 8.0 * fm1.im + fm2.im) / (12.0 * spacing);
        break;
    case SIM_DIVERGENCE_STENCIL_CENTRAL_2:
    default:
        result.re = (fp1.re - fm1.re) / (2.0 * spacing);
        result.im = (fp1.im - fm1.im) / (2.0 * spacing);
        break;
    }

    return result;
}

static int approx(double a, double b, double eps) { return fabs(a - b) <= eps; }

int main(void) {
    const size_t height = 6U;
    const size_t width = 7U;
    const size_t shape[2] = {height, width};
    const double spacing_x = 0.5;
    const double spacing_y = 0.75;
    const double eps = 1.0e-10;

    SimContext ctx;
    SimField field_x = {0};
    SimField field_y = {0};
    SimField output = {0};
    size_t field_x_index = SIZE_MAX;
    size_t field_y_index = SIZE_MAX;
    size_t output_index = SIZE_MAX;
    size_t op_index = SIZE_MAX;

    if (sim_context_init(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_init\n");
        return 1;
    }

    if (sim_field_init(&field_x, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&field_y, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK ||
        sim_field_init(&output, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_field_init\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimComplexDouble *vx = sim_field_complex_data(&field_x);
    SimComplexDouble *vy = sim_field_complex_data(&field_y);
    if (vx == NULL || vy == NULL) {
        fprintf(stderr, "FAIL: sim_field_complex_data\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (size_t y = 0U; y < height; ++y) {
        for (size_t x = 0U; x < width; ++x) {
            size_t index = y * width + x;
            double theta_x = (2.0 * M_PI * (double)x) / (double)width;
            double theta_y = (2.0 * M_PI * (double)y) / (double)height;
            double theta_xy = theta_x + theta_y;

            vx[index].re = sin(theta_x) + 0.25 * cos(2.0 * theta_y);
            vx[index].im = cos(theta_x) - 0.5 * sin(theta_y);
            vy[index].re = 0.5 * cos(theta_xy) + 0.25 * sin(theta_y);
            vy[index].im = sin(2.0 * theta_y) - 0.25 * cos(theta_x);
        }
    }

    if (sim_context_add_field(&ctx, &field_x, &field_x_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &field_y, &field_y_index) != SIM_RESULT_OK ||
        sim_context_add_field(&ctx, &output, &output_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_add_field\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimDivergenceOperatorConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_field_x = field_x_index;
    cfg.input_field_y = field_y_index;
    cfg.output_field = output_index;
    cfg.spacing_x = spacing_x;
    cfg.spacing_y = spacing_y;
    cfg.axis_x = 1U;
    cfg.axis_y = 0U;
    cfg.stencil = SIM_DIVERGENCE_STENCIL_CENTRAL_4;
    cfg.boundary = SIM_IR_BOUNDARY_PERIODIC;
    cfg.accumulate = false;
    cfg.scale_by_dt = false;

    if (sim_add_divergence_operator(&ctx, &cfg, &op_index) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_add_divergence_operator\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimDivergenceOperatorConfig fetched;
    memset(&fetched, 0, sizeof(fetched));
    if (sim_divergence_config(&ctx, op_index, &fetched) != SIM_RESULT_OK ||
        fetched.stencil != SIM_DIVERGENCE_STENCIL_CENTRAL_4 ||
        fetched.boundary != SIM_IR_BOUNDARY_PERIODIC) {
        fprintf(stderr, "FAIL: sim_divergence_config\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_divergence_update(&ctx, op_index, &fetched) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_divergence_update\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    if (sim_context_prepare_plan(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_prepare_plan\n");
        sim_context_destroy(&ctx);
        return 1;
    }
    if (sim_context_execute(&ctx) != SIM_RESULT_OK) {
        fprintf(stderr, "FAIL: sim_context_execute\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    SimField *out_field = sim_context_field(&ctx, output_index);
    if (out_field == NULL) {
        fprintf(stderr, "FAIL: output field lookup\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    const SimComplexDouble *out_data = sim_field_complex_data_const(out_field);
    if (out_data == NULL) {
        fprintf(stderr, "FAIL: output complex data\n");
        sim_context_destroy(&ctx);
        return 1;
    }

    for (size_t y = 0U; y < height; ++y) {
        for (size_t x = 0U; x < width; ++x) {
            size_t index = y * width + x;
            SimComplexDouble dx =
                derivative_periodic(vx, height, width, y, x, 1U, cfg.stencil, spacing_x);
            SimComplexDouble dy =
                derivative_periodic(vy, height, width, y, x, 0U, cfg.stencil, spacing_y);
            double expected_re = dx.re + dy.re;
            double expected_im = dx.im + dy.im;

            if (!approx(out_data[index].re, expected_re, eps) ||
                !approx(out_data[index].im, expected_im, eps)) {
                fprintf(stderr,
                        "FAIL: mismatch at (%zu,%zu) got=(%.12g, %.12g) expected=(%.12g, %.12g)\n",
                        y, x, out_data[index].re, out_data[index].im, expected_re, expected_im);
                sim_context_destroy(&ctx);
                return 1;
            }
        }
    }

    sim_context_destroy(&ctx);
    return 0;
}
