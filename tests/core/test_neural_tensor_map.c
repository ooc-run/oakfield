/*
 * Covers the tensor mapping layer used to exchange field data with neural
 * operators, including zero-copy, copy-fallback, affine, and complex paths.
 */
#include <oakfield/neural_tensor_map.h>
#include <oakfield/sim.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool nearly_equal(double a, double b, double tol) {
    double diff = fabs(a - b);
    double scale = fmax(fabs(a), fabs(b));
    if (scale < 1.0) {
        scale = 1.0;
    }
    return diff <= tol * scale;
}

static size_t tensor_offset(const SimNeuralTensorView *view, const size_t *indices) {
    size_t offset = 0U;
    if (view == NULL || indices == NULL) {
        return 0U;
    }
    for (size_t axis = 0U; axis < view->rank; ++axis) {
        offset += indices[axis] * view->strides[axis];
    }
    return offset;
}

static bool test_rank1_zero_copy_implicit_channel(void) {
    size_t shape[1] = {8U};
    SimField field = {0};
    SimNeuralTensorMapping mapping = {0};
    SimResult rc;
    bool ok = false;

    if (sim_field_init(&field, 1U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        return false;
    }

    rc = sim_neural_tensor_map_input(&field, NULL, &mapping);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] rank1 map_input failed (%d)\n", (int)rc);
        goto cleanup;
    }
    if (!mapping.metrics.zero_copy || mapping.metrics.used_copy_fallback) {
        fprintf(stderr, "[FAIL] expected zero-copy mapping for rank1 default policy\n");
        goto cleanup;
    }
    if (mapping.view.rank != 3U || mapping.view.shape[0] != 1U || mapping.view.shape[1] != 8U ||
        mapping.view.shape[2] != 1U || mapping.view.channel_axis != 2U ||
        mapping.view.spatial_rank != 1U) {
        fprintf(stderr, "[FAIL] rank1 canonical tensor shape mismatch\n");
        goto cleanup;
    }
    if (mapping.view.value_type != SIM_NEURAL_TENSOR_VALUE_REAL_F64 ||
        mapping.view.data != sim_field_data(&field)) {
        fprintf(stderr, "[FAIL] rank1 value type/data alias mismatch\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    sim_neural_tensor_mapping_release(&mapping);
    sim_field_destroy(&field);
    return ok;
}

static bool test_rank2_reorder_copy_with_affine(void) {
    size_t shape[2] = {3U, 4U};
    SimField field = {0};
    SimNeuralTensorMapConfig config = sim_neural_tensor_map_config_defaults();
    SimNeuralTensorMapping mapping = {0};
    SimResult rc;
    bool ok = false;

    if (sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, NULL) !=
        SIM_RESULT_OK) {
        return false;
    }

    {
        double *data = sim_field_real_data(&field);
        if (data == NULL) {
            goto cleanup;
        }
        for (size_t c = 0U; c < shape[0]; ++c) {
            for (size_t x = 0U; x < shape[1]; ++x) {
                data[c * shape[1] + x] = 10.0 * (double)c + (double)x;
            }
        }
    }

    config.channel_axis = 0U;
    config.channels_last = true;
    config.affine.enabled = true;
    config.affine.scale = 2.0;
    config.affine.bias = -1.0;

    rc = sim_neural_tensor_map_input(&field, &config, &mapping);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] reorder map_input failed (%d)\n", (int)rc);
        goto cleanup;
    }
    if (mapping.metrics.zero_copy || !mapping.metrics.used_copy_fallback) {
        fprintf(stderr, "[FAIL] expected fallback copy for reordered rank2 mapping\n");
        goto cleanup;
    }
    if (mapping.metrics.bytes_copied_from_field != sim_field_bytes(&field)) {
        fprintf(stderr, "[FAIL] unexpected copy byte metric (%zu vs %zu)\n",
                mapping.metrics.bytes_copied_from_field, sim_field_bytes(&field));
        goto cleanup;
    }
    if (mapping.view.rank != 3U || mapping.view.shape[0] != 1U || mapping.view.shape[1] != 4U ||
        mapping.view.shape[2] != 3U || mapping.view.channel_axis != 2U ||
        mapping.view.value_type != SIM_NEURAL_TENSOR_VALUE_REAL_F64) {
        fprintf(stderr, "[FAIL] reordered tensor metadata mismatch\n");
        goto cleanup;
    }

    {
        const double *tensor = (const double *)mapping.view.data;
        if (tensor == NULL) {
            goto cleanup;
        }
        for (size_t x = 0U; x < shape[1]; ++x) {
            for (size_t c = 0U; c < shape[0]; ++c) {
                size_t indices[3] = {0U, x, c};
                size_t off = tensor_offset(&mapping.view, indices);
                double expected = (10.0 * (double)c + (double)x) * 2.0 - 1.0;
                if (!nearly_equal(tensor[off], expected, 1.0e-12)) {
                    fprintf(stderr, "[FAIL] reordered tensor[%zu,%zu]=%.12g expected %.12g\n", x, c,
                            tensor[off], expected);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_neural_tensor_mapping_release(&mapping);
    sim_field_destroy(&field);
    return ok;
}

static bool test_complex_split_output_unmap(void) {
    size_t shape[2] = {2U, 2U};
    SimField field = {0};
    SimNeuralTensorMapConfig config = sim_neural_tensor_map_config_defaults();
    SimNeuralTensorMapping mapping = {0};
    SimResult rc;
    bool ok = false;

    if (sim_field_init(&field, 2U, shape, sizeof(SimComplexDouble), SIM_FIELD_STORAGE_ROW_MAJOR,
                       NULL) != SIM_RESULT_OK) {
        return false;
    }

    config.channel_axis = SIM_NEURAL_TENSOR_CHANNEL_AXIS_AUTO;
    config.channels_last = true;
    config.complex_mode = SIM_NEURAL_TENSOR_COMPLEX_SPLIT_CHANNELS;
    config.affine.enabled = true;
    config.affine.scale = 2.0;
    config.affine.bias = 1.0;

    rc = sim_neural_tensor_map_output(&field, &config, &mapping);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex split map_output failed (%d)\n", (int)rc);
        goto cleanup;
    }
    if (mapping.metrics.zero_copy || !mapping.metrics.used_copy_fallback) {
        fprintf(stderr, "[FAIL] expected fallback copy for complex split output mapping\n");
        goto cleanup;
    }
    if (mapping.view.value_type != SIM_NEURAL_TENSOR_VALUE_REAL_F64 || mapping.view.rank != 3U ||
        mapping.view.shape[0] != 1U || mapping.view.shape[1] != 2U || mapping.view.shape[2] != 4U ||
        mapping.view.channel_axis != 2U) {
        fprintf(stderr, "[FAIL] complex split tensor view metadata mismatch\n");
        goto cleanup;
    }

    {
        double *tensor = (double *)mapping.view.data;
        if (tensor == NULL) {
            goto cleanup;
        }
        for (size_t x = 0U; x < 2U; ++x) {
            for (size_t c = 0U; c < 2U; ++c) {
                size_t idx_re[3] = {0U, x, 2U * c};
                size_t idx_im[3] = {0U, x, 2U * c + 1U};
                tensor[tensor_offset(&mapping.view, idx_re)] =
                    10.0 * (double)x + 2.0 * (double)c + 0.25;
                tensor[tensor_offset(&mapping.view, idx_im)] =
                    -5.0 * (double)x - 2.0 * (double)c - 0.5;
            }
        }
    }

    rc = sim_neural_tensor_unmap_output(&mapping);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] complex split unmap failed (%d)\n", (int)rc);
        goto cleanup;
    }
    if (mapping.metrics.bytes_copied_to_field != sim_field_bytes(&field)) {
        fprintf(stderr, "[FAIL] complex split bytes_copied_to_field mismatch\n");
        goto cleanup;
    }

    {
        const SimComplexDouble *out = sim_field_complex_data_const(&field);
        if (out == NULL) {
            goto cleanup;
        }
        for (size_t x = 0U; x < 2U; ++x) {
            for (size_t c = 0U; c < 2U; ++c) {
                size_t off = x * 2U + c;
                double re_src = 10.0 * (double)x + 2.0 * (double)c + 0.25;
                double im_src = -5.0 * (double)x - 2.0 * (double)c - 0.5;
                double re_exp = re_src * 2.0 + 1.0;
                double im_exp = im_src * 2.0 + 1.0;
                if (!nearly_equal(out[off].re, re_exp, 1.0e-12) ||
                    !nearly_equal(out[off].im, im_exp, 1.0e-12)) {
                    fprintf(stderr, "[FAIL] complex output[%zu,%zu]=(%g,%g) expected (%g,%g)\n", x,
                            c, out[off].re, out[off].im, re_exp, im_exp);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_neural_tensor_mapping_release(&mapping);
    sim_field_destroy(&field);
    return ok;
}

static bool test_noncontiguous_forces_copy(void) {
    size_t shape[2] = {2U, 2U};
    size_t strides[2] = {3U, 1U};
    SimFieldLayout layout = {.rank = 2U, .shape = shape, .strides = strides, .contiguous = false};
    double backing[6] = {1.0, 2.0, 0.0, 3.0, 4.0, 0.0};
    SimField wrapped = {0};
    SimNeuralTensorMapping mapping = {0};
    SimResult rc;
    bool ok = false;

    if (sim_field_wrap(&wrapped, &layout, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR, backing) !=
        SIM_RESULT_OK) {
        return false;
    }

    rc = sim_neural_tensor_map_input(&wrapped, NULL, &mapping);
    if (rc != SIM_RESULT_OK) {
        fprintf(stderr, "[FAIL] noncontiguous map_input failed (%d)\n", (int)rc);
        goto cleanup;
    }
    if (mapping.metrics.zero_copy || !mapping.metrics.used_copy_fallback) {
        fprintf(stderr, "[FAIL] noncontiguous layout should force fallback copy\n");
        goto cleanup;
    }
    if (mapping.metrics.bytes_copied_from_field !=
        sim_field_element_count(&wrapped.layout) * sizeof(double)) {
        fprintf(stderr, "[FAIL] noncontiguous copy metric mismatch\n");
        goto cleanup;
    }

    {
        const double *tensor = (const double *)mapping.view.data;
        if (tensor == NULL) {
            goto cleanup;
        }
        for (size_t x = 0U; x < 2U; ++x) {
            for (size_t c = 0U; c < 2U; ++c) {
                size_t idx[3] = {0U, x, c};
                size_t off = tensor_offset(&mapping.view, idx);
                double expected = backing[x * 3U + c];
                if (!nearly_equal(tensor[off], expected, 1.0e-12)) {
                    fprintf(stderr, "[FAIL] noncontiguous tensor[%zu,%zu]=%.12g expected %.12g\n",
                            x, c, tensor[off], expected);
                    goto cleanup;
                }
            }
        }
    }

    ok = true;

cleanup:
    sim_neural_tensor_mapping_release(&mapping);
    sim_field_destroy(&wrapped);
    return ok;
}

int main(void) {
    bool ok = true;
    ok &= test_rank1_zero_copy_implicit_channel();
    ok &= test_rank2_reorder_copy_with_affine();
    ok &= test_complex_split_output_unmap();
    ok &= test_noncontiguous_forces_copy();
    return ok ? 0 : 1;
}
