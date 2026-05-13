#include <oakfield/sim.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool expect(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "[FAIL] %s\n", (message != NULL) ? message : "expectation failed");
        return false;
    }
    return true;
}

typedef struct RowAccumulator {
    double row_sums[2];
    size_t row_count;
} RowAccumulator;

static SimResult accumulate_rows(size_t row_index, void *row_data, size_t row_length,
                                 void *userdata) {
    RowAccumulator *acc = (RowAccumulator *)userdata;
    const double *row = (const double *)row_data;
    double sum = 0.0;

    if (acc == NULL || row == NULL || row_index >= 2U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    for (size_t i = 0U; i < row_length; ++i) {
        sum += row[i];
    }

    acc->row_sums[row_index] = sum;
    acc->row_count += 1U;
    return SIM_RESULT_OK;
}

static bool test_patch_geometry(void) {
    SimFieldPatch full = {0};
    SimFieldPatch partial = {0};
    SimFieldPatch edge = {0};

    return expect(sim_field_patch_full(8U, 5U, &full) == SIM_RESULT_OK, "full patch failed") &&
           expect(sim_field_patch_is_valid(&full), "full patch should be valid") &&
           expect(full.x0 == 0U && full.y0 == 0U && full.width == 8U && full.height == 5U,
                  "full patch geometry mismatch") &&
           expect((full.flags & SIM_FIELD_PATCH_FLAG_COVERS_FULL_ROWS) != 0U,
                  "full patch should cover rows") &&
           expect((full.flags & SIM_FIELD_PATCH_FLAG_IS_FULL_FIELD) != 0U,
                  "full patch should cover the field") &&
           expect((full.flags & SIM_FIELD_PATCH_FLAG_TOUCHES_BOUNDARY) != 0U,
                  "full patch should touch the boundary") &&
           expect(sim_field_patch_from_xywh(8U, 5U, 2U, 1U, 3U, 2U, &partial) == SIM_RESULT_OK,
                  "partial patch failed") &&
           expect(partial.flags == SIM_FIELD_PATCH_FLAG_NONE, "partial patch flags mismatch") &&
           expect(sim_field_patch_from_xywh(8U, 5U, 0U, 2U, 4U, 2U, &edge) == SIM_RESULT_OK,
                  "edge patch failed") &&
           expect((edge.flags & SIM_FIELD_PATCH_FLAG_TOUCHES_BOUNDARY) != 0U,
                  "edge patch should touch the boundary") &&
           expect(sim_field_patch_from_xywh(8U, 5U, 7U, 0U, 2U, 1U, &edge) ==
                      SIM_RESULT_INVALID_ARGUMENT,
                  "out-of-bounds patch should be rejected");
}

static bool test_patch_intersection_and_tiles(void) {
    SimFieldPatch lhs = {0};
    SimFieldPatch rhs = {0};
    SimFieldPatch intersection = {0};
    SimFieldPatch tiles[6];
    size_t tile_count = 0U;

    memset(tiles, 0, sizeof(tiles));

    return expect(sim_field_patch_from_xywh(8U, 6U, 1U, 1U, 4U, 3U, &lhs) == SIM_RESULT_OK,
                  "lhs patch failed") &&
           expect(sim_field_patch_from_xywh(8U, 6U, 3U, 0U, 3U, 4U, &rhs) == SIM_RESULT_OK,
                  "rhs patch failed") &&
           expect(sim_field_patch_intersect(&lhs, &rhs, &intersection),
                  "patches should intersect") &&
           expect(intersection.x0 == 3U && intersection.y0 == 1U && intersection.width == 2U &&
                      intersection.height == 3U,
                  "intersection geometry mismatch") &&
           expect(sim_field_patch_split_tiles(&lhs, 2U, 2U, NULL, 0U) == 4U,
                  "tile count mismatch") &&
           ((tile_count = sim_field_patch_split_tiles(&lhs, 2U, 2U, tiles, 6U)), true) &&
           expect(tile_count == 4U, "tile split count mismatch") &&
           expect(tiles[0].x0 == 1U && tiles[0].y0 == 1U && tiles[0].width == 2U &&
                      tiles[0].height == 2U,
                  "tile[0] mismatch") &&
           expect(tiles[1].x0 == 3U && tiles[1].y0 == 1U && tiles[1].width == 2U &&
                      tiles[1].height == 2U,
                  "tile[1] mismatch") &&
           expect(tiles[2].x0 == 1U && tiles[2].y0 == 3U && tiles[2].width == 2U &&
                      tiles[2].height == 1U,
                  "tile[2] mismatch") &&
           expect(tiles[3].x0 == 3U && tiles[3].y0 == 3U && tiles[3].width == 2U &&
                      tiles[3].height == 1U,
                  "tile[3] mismatch");
}

static bool test_normalized_region_resolution(void) {
    SimFieldPatch patch = {0};

    return expect(sim_field_patch_from_normalized_region(8U, 4U, 0.25, 0.25, 0.75, 0.75, &patch) ==
                      SIM_RESULT_OK,
                  "normalized region patch failed") &&
           expect(patch.x0 == 2U && patch.y0 == 1U && patch.width == 4U && patch.height == 2U,
                  "normalized region geometry mismatch") &&
           expect(sim_field_patch_from_normalized_region(8U, 4U, 0.99, 0.0, 1.0, 1.0, &patch) ==
                      SIM_RESULT_OK,
                  "edge normalized region patch failed") &&
           expect(patch.x0 == 7U && patch.width == 1U, "edge normalized region width mismatch") &&
           expect(sim_field_patch_from_normalized_region(8U, 4U, 0.5, 0.5, 0.5, 0.75, &patch) ==
                      SIM_RESULT_INVALID_ARGUMENT,
                  "empty normalized region should be rejected");
}

static bool test_patch_view_from_field(void) {
    SimField field = {0};
    SimFieldPatch patch = {0};
    SimFieldPatchView view = {0};
    RowAccumulator rows = {{0.0, 0.0}, 0U};
    size_t shape[2] = {3U, 5U};
    size_t offset = 0U;
    double *data = NULL;

    if (!expect(sim_field_init(&field, 2U, shape, sizeof(double), SIM_FIELD_STORAGE_ROW_MAJOR,
                               NULL) == SIM_RESULT_OK,
                "field init failed")) {
        return false;
    }

    data = (double *)sim_field_data(&field);
    for (size_t i = 0U; i < 15U; ++i) {
        data[i] = (double)i;
    }

    if (!expect(sim_field_patch_from_xywh(5U, 3U, 1U, 1U, 3U, 2U, &patch) == SIM_RESULT_OK,
                "patch from xywh failed") ||
        !expect(sim_field_patch_view_from_field(&field, &patch, false, &view) == SIM_RESULT_OK,
                "patch view creation failed") ||
        !expect(sim_field_patch_view_is_valid(&view), "patch view should be valid") ||
        !expect(view.buffer_view.layout.rank == 2U, "patch view rank mismatch") ||
        !expect(view.buffer_view.layout.shape[0] == 2U && view.buffer_view.layout.shape[1] == 3U,
                "patch view shape mismatch") ||
        !expect(view.row_stride == 5U, "patch view row stride mismatch") ||
        !expect(view.buffer_view.data == &data[6], "patch view base pointer mismatch") ||
        !expect(
            sim_buffer_view_offset_for_indices(&view.buffer_view, (size_t[]){1U, 2U}, 2U, &offset),
            "patch view offset lookup failed") ||
        !expect(offset == 7U, "patch view offset mismatch") ||
        !expect(*((double *)view.buffer_view.data + offset) == 13.0,
                "patch view sample mismatch") ||
        !expect(sim_field_patch_iter_rows(&view, accumulate_rows, &rows) == SIM_RESULT_OK,
                "patch row iteration failed") ||
        !expect(rows.row_count == 2U, "patch row iteration count mismatch") ||
        !expect(rows.row_sums[0] == 21.0 && rows.row_sums[1] == 36.0, "patch row sums mismatch")) {
        sim_field_destroy(&field);
        return false;
    }

    sim_field_destroy(&field);
    return true;
}

int main(void) {
    bool ok = true;

    ok = ok && test_patch_geometry();
    ok = ok && test_patch_intersection_and_tiles();
    ok = ok && test_normalized_region_resolution();
    ok = ok && test_patch_view_from_field();

    return ok ? 0 : 1;
}
