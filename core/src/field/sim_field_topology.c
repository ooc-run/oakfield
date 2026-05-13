#include "oakfield/sim_field_topology.h"

#include "oakfield/sim_field_stats_runtime.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float sim_field_topology_clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static double sim_field_topology_wrap_phase_delta(double from_phase, double to_phase) {
    double delta = fmod((to_phase - from_phase) + M_PI, 2.0 * M_PI);
    if (delta < 0.0) {
        delta += 2.0 * M_PI;
    }
    return delta - M_PI;
}

static bool sim_field_topology_load_sample(const SimFieldView* view,
                                           size_t              index,
                                           double*             out_re,
                                           double*             out_im) {
    if (view == NULL || out_re == NULL || out_im == NULL || view->data == NULL ||
        index >= view->count) {
        return false;
    }

    switch (view->type) {
        case SIM_FIELD_DOUBLE: {
            const double* src = (const double*) view->data;
            *out_re           = src[index];
            *out_im           = 0.0;
            return true;
        }
        case SIM_FIELD_COMPLEX_DOUBLE: {
            const SimComplexDouble* src = (const SimComplexDouble*) view->data;
            *out_re                    = src[index].re;
            *out_im                    = src[index].im;
            return true;
        }
        default:
            break;
    }

    return false;
}

static int sim_field_topology_popcount4(uint8_t value) {
    int count = 0;
    while (value != 0U) {
        count += (value & 1U) ? 1 : 0;
        value >>= 1U;
    }
    return count;
}

void sim_field_topology_config_default(SimFieldTopologyConfig* out_config) {
    if (out_config == NULL) {
        return;
    }

    out_config->seam_phase_threshold      = (float) (M_PI * 0.90);
    out_config->seam_ambiguity_band       = (float) (M_PI * 0.08);
    out_config->charge_ambiguity_threshold = 0.20f;
    out_config->magnitude_epsilon         = 1.0e-12f;
    out_config->enable_core_localization  = true;
}

bool sim_field_topology_dimensions(const struct SimField* field,
                                   size_t*                out_width,
                                   size_t*                out_height) {
    const size_t* shape;
    size_t        width;
    size_t        height;
    size_t        element_count;

    if (out_width != NULL) {
        *out_width = 0U;
    }
    if (out_height != NULL) {
        *out_height = 0U;
    }

    if (field == NULL || field->layout.rank == 0U || field->layout.shape == NULL) {
        return false;
    }

    shape = field->layout.shape;
    width = shape[field->layout.rank - 1U];
    if (width == 0U) {
        return false;
    }

    element_count = sim_field_element_count(&field->layout);
    if (element_count == 0U) {
        return false;
    }

    height = element_count / width;
    if (height == 0U) {
        height = 1U;
    }

    if (out_width != NULL) {
        *out_width = width;
    }
    if (out_height != NULL) {
        *out_height = height;
    }
    return true;
}

bool sim_field_topology_extract(const struct SimField*     field,
                                SimFieldTopologyCell*      out_cells,
                                size_t                     cell_capacity,
                                SimFieldTopologySummary*   out_summary,
                                const SimFieldTopologyConfig* config,
                                size_t*                    out_width,
                                size_t*                    out_height) {
    SimFieldTopologyConfig cfg_storage;
    SimFieldTopologySummary summary      = { 0 };
    const SimFieldTopologyConfig* cfg    = config;
    SimFieldView               view      = { 0 };
    size_t                     width     = 0U;
    size_t                     height    = 0U;
    size_t                     cell_count = 0U;

    if (!sim_field_topology_dimensions(field, &width, &height)) {
        return false;
    }

    if (cfg == NULL) {
        sim_field_topology_config_default(&cfg_storage);
        cfg = &cfg_storage;
    }

    cell_count = width * height;
    if (out_cells != NULL) {
        if (cell_capacity < cell_count) {
            return false;
        }
        (void) memset(out_cells, 0, cell_count * sizeof(*out_cells));
    }

    if (out_width != NULL) {
        *out_width = width;
    }
    if (out_height != NULL) {
        *out_height = height;
    }

    if (field == NULL) {
        if (out_summary != NULL) {
            *out_summary = summary;
        }
        return false;
    }

    view = sim_field_view_from_field((SimField*) field);
    if (view.data == NULL || view.count == 0U) {
        if (out_summary != NULL) {
            *out_summary = summary;
        }
        return false;
    }

    if (view.type != SIM_FIELD_COMPLEX_DOUBLE || width < 2U || height < 2U) {
        if (out_summary != NULL) {
            *out_summary = summary;
        }
        return true;
    }

    summary.valid = true;

    for (size_t y = 0U; y < height; ++y) {
        for (size_t x = 0U; x < width; ++x) {
            const size_t cell_index = y * width + x;
            SimFieldTopologyCell cell = { 0 };

            if ((x + 1U) >= width || (y + 1U) >= height) {
                if (out_cells != NULL) {
                    out_cells[cell_index] = cell;
                }
                continue;
            }

            const size_t idx00 = y * width + x;
            const size_t idx10 = y * width + (x + 1U);
            const size_t idx11 = (y + 1U) * width + (x + 1U);
            const size_t idx01 = (y + 1U) * width + x;
            double       re00  = 0.0;
            double       im00  = 0.0;
            double       re10  = 0.0;
            double       im10  = 0.0;
            double       re11  = 0.0;
            double       im11  = 0.0;
            double       re01  = 0.0;
            double       im01  = 0.0;
            bool         ok    = true;

            ok = ok && sim_field_topology_load_sample(&view, idx00, &re00, &im00);
            ok = ok && sim_field_topology_load_sample(&view, idx10, &re10, &im10);
            ok = ok && sim_field_topology_load_sample(&view, idx11, &re11, &im11);
            ok = ok && sim_field_topology_load_sample(&view, idx01, &re01, &im01);

            if (!ok || !isfinite(re00) || !isfinite(im00) || !isfinite(re10) || !isfinite(im10) ||
                !isfinite(re11) || !isfinite(im11) || !isfinite(re01) || !isfinite(im01)) {
                cell.flags = SIM_FIELD_TOPOLOGY_CELL_VALID | SIM_FIELD_TOPOLOGY_CELL_AMBIGUOUS |
                             SIM_FIELD_TOPOLOGY_CELL_NONFINITE;
                summary.ambiguous_cell_count += 1U;
                if (out_cells != NULL) {
                    out_cells[cell_index] = cell;
                }
                continue;
            }

            {
                const double p00 = atan2(im00, re00);
                const double p10 = atan2(im10, re10);
                const double p11 = atan2(im11, re11);
                const double p01 = atan2(im01, re01);
                const double raw_bottom = p10 - p00;
                const double raw_right  = p11 - p10;
                const double raw_top    = p01 - p11;
                const double raw_left   = p00 - p01;
                const double wrapped_bottom = sim_field_topology_wrap_phase_delta(p00, p10);
                const double wrapped_right  = sim_field_topology_wrap_phase_delta(p10, p11);
                const double wrapped_top    = sim_field_topology_wrap_phase_delta(p11, p01);
                const double wrapped_left   = sim_field_topology_wrap_phase_delta(p01, p00);
                const double winding =
                    wrapped_bottom + wrapped_right + wrapped_top + wrapped_left;
                const double raw_charge = winding / (2.0 * M_PI);
                const double m00 = hypot(re00, im00);
                const double m10 = hypot(re10, im10);
                const double m11 = hypot(re11, im11);
                const double m01 = hypot(re01, im01);
                const double local_min = fmin(fmin(m00, m10), fmin(m11, m01));
                const double local_max = fmax(fmax(m00, m10), fmax(m11, m01));
                const double safe_max  =
                    (local_max > (double) cfg->magnitude_epsilon) ? local_max
                                                                  : (double) cfg->magnitude_epsilon;
                const double seam_threshold = (double) cfg->seam_phase_threshold;
                const double seam_band      = (double) cfg->seam_ambiguity_band;
                double       nearest_charge = 0.0;
                double       charge_error   = 0.0;
                double       charge_conf    = 0.0;
                double       contrast_conf  = 0.0;
                bool         ambiguous      = false;

                cell.flags = SIM_FIELD_TOPOLOGY_CELL_VALID;
                if (raw_charge > 0.5) {
                    cell.charge = 1;
                } else if (raw_charge < -0.5) {
                    cell.charge = -1;
                } else {
                    cell.charge = 0;
                }

                nearest_charge = (double) cell.charge;
                charge_error   = fabs(raw_charge - nearest_charge);
                charge_conf    = 1.0 - fmin(1.0, charge_error / 0.5);
                contrast_conf  = 1.0 - fmin(1.0, local_min / safe_max);

                if (fabs(raw_bottom) >= seam_threshold) {
                    cell.seam_mask |= SIM_FIELD_TOPOLOGY_SEAM_BOTTOM;
                }
                if (fabs(raw_right) >= seam_threshold) {
                    cell.seam_mask |= SIM_FIELD_TOPOLOGY_SEAM_RIGHT;
                }
                if (fabs(raw_top) >= seam_threshold) {
                    cell.seam_mask |= SIM_FIELD_TOPOLOGY_SEAM_TOP;
                }
                if (fabs(raw_left) >= seam_threshold) {
                    cell.seam_mask |= SIM_FIELD_TOPOLOGY_SEAM_LEFT;
                }

                ambiguous = ambiguous || (charge_error > (double) cfg->charge_ambiguity_threshold);
                ambiguous = ambiguous || !isfinite(raw_charge) || !isfinite(local_min) ||
                            !isfinite(local_max) || (local_max <= (double) cfg->magnitude_epsilon);
                ambiguous =
                    ambiguous || (fabs(fabs(raw_bottom) - seam_threshold) <= seam_band) ||
                    (fabs(fabs(raw_right) - seam_threshold) <= seam_band) ||
                    (fabs(fabs(raw_top) - seam_threshold) <= seam_band) ||
                    (fabs(fabs(raw_left) - seam_threshold) <= seam_band);

                cell.confidence =
                    (float) sim_field_topology_clampf((float) (0.65 * charge_conf + 0.35 * contrast_conf),
                                                      0.0f,
                                                      1.0f);

                if (cfg->enable_core_localization && cell.charge != 0) {
                    const double eps = fmax((double) cfg->magnitude_epsilon, 1.0e-15);
                    const double w00 = 1.0 / fmax(m00, eps);
                    const double w10 = 1.0 / fmax(m10, eps);
                    const double w11 = 1.0 / fmax(m11, eps);
                    const double w01 = 1.0 / fmax(m01, eps);
                    const double wsum = w00 + w10 + w11 + w01;

                    if (wsum > eps) {
                        const double px = (w10 + w11) / wsum;
                        const double py = (w01 + w11) / wsum;
                        cell.core_offset_x =
                            sim_field_topology_clampf((float) (px - 0.5), -0.5f, 0.5f);
                        cell.core_offset_y =
                            sim_field_topology_clampf((float) (py - 0.5), -0.5f, 0.5f);
                        if (contrast_conf > 0.05) {
                            cell.flags |= SIM_FIELD_TOPOLOGY_CELL_CORE_OFFSET_VALID;
                        }
                    }
                }

                if (ambiguous) {
                    cell.flags |= SIM_FIELD_TOPOLOGY_CELL_AMBIGUOUS;
                    cell.confidence *= 0.5f;
                    summary.ambiguous_cell_count += 1U;
                }

                if (cell.charge > 0) {
                    summary.positive_singularities += 1U;
                } else if (cell.charge < 0) {
                    summary.negative_singularities += 1U;
                }
                summary.seam_edge_count += (size_t) sim_field_topology_popcount4(cell.seam_mask);
            }

            if (out_cells != NULL) {
                out_cells[cell_index] = cell;
            }
        }
    }

    if (out_summary != NULL) {
        *out_summary = summary;
    }
    return true;
}

void sim_field_topology_cell_pack_rgba8(const SimFieldTopologyCell* cell, uint8_t out_texel[4]) {
    const SimFieldTopologyCell zero = { 0 };
    const SimFieldTopologyCell* src = (cell != NULL) ? cell : &zero;
    const float charge01            =
        sim_field_topology_clampf(((float) src->charge + 1.0f) * 0.5f, 0.0f, 1.0f);
    const uint8_t charge_byte = (uint8_t) lrintf(charge01 * 255.0f);
    const uint8_t seam_and_flags =
        (uint8_t) ((((src->flags & 0x0Fu) << 4) & 0xF0u) | (src->seam_mask & 0x0Fu));
    uint8_t core_x = 128U;
    uint8_t core_y = 128U;

    if ((src->flags & SIM_FIELD_TOPOLOGY_CELL_CORE_OFFSET_VALID) != 0U) {
        core_x = (uint8_t) lrintf(
            sim_field_topology_clampf(src->core_offset_x + 0.5f, 0.0f, 1.0f) * 255.0f);
        core_y = (uint8_t) lrintf(
            sim_field_topology_clampf(src->core_offset_y + 0.5f, 0.0f, 1.0f) * 255.0f);
    }

    out_texel[0] = charge_byte;
    out_texel[1] = seam_and_flags;
    out_texel[2] = core_x;
    out_texel[3] = core_y;
}
