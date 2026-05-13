#include "oakfield/sim_field_topology_runtime.h"

#include <stdlib.h>
#include <string.h>

static void sim_field_topology_runtime_workspace_free(SimFieldTopologyWorkspace* workspace) {
    if (workspace == NULL) {
        return;
    }

    free(workspace->labels);
    free(workspace->visited);
    free(workspace->seam_chain);
    free(workspace->ambiguity);
    workspace->labels     = NULL;
    workspace->visited    = NULL;
    workspace->seam_chain = NULL;
    workspace->ambiguity  = NULL;
    workspace->capacity   = 0U;
}

static SimResult
sim_field_topology_runtime_workspace_resize(SimFieldTopologyWorkspace* workspace, size_t capacity) {
    uint32_t* new_labels     = NULL;
    uint8_t*  new_visited    = NULL;
    uint8_t*  new_seam_chain = NULL;
    uint8_t*  new_ambiguity  = NULL;

    if (workspace == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (capacity == 0U) {
        sim_field_topology_runtime_workspace_free(workspace);
        return SIM_RESULT_OK;
    }

    if (workspace->capacity >= capacity && workspace->labels != NULL && workspace->visited != NULL &&
        workspace->seam_chain != NULL && workspace->ambiguity != NULL) {
        return SIM_RESULT_OK;
    }

    new_labels     = (uint32_t*) calloc(capacity, sizeof(uint32_t));
    new_visited    = (uint8_t*) calloc(capacity, sizeof(uint8_t));
    new_seam_chain = (uint8_t*) calloc(capacity, sizeof(uint8_t));
    new_ambiguity  = (uint8_t*) calloc(capacity, sizeof(uint8_t));
    if (new_labels == NULL || new_visited == NULL || new_seam_chain == NULL ||
        new_ambiguity == NULL) {
        free(new_labels);
        free(new_visited);
        free(new_seam_chain);
        free(new_ambiguity);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    sim_field_topology_runtime_workspace_free(workspace);
    workspace->labels     = new_labels;
    workspace->visited    = new_visited;
    workspace->seam_chain = new_seam_chain;
    workspace->ambiguity  = new_ambiguity;
    workspace->capacity   = capacity;
    return SIM_RESULT_OK;
}

static size_t sim_field_topology_runtime_sanitize_cadence(size_t cadence_steps) {
    return (cadence_steps == 0U) ? 1U : cadence_steps;
}

void sim_field_topology_runtime_init(SimFieldTopologyRuntimeState* state) {
    if (state == NULL) {
        return;
    }

    (void) memset(state, 0, sizeof(*state));
    state->enabled            = false;
    state->dirty              = true;
    state->cadence_steps      = 1U;
    state->last_computed_step = SIZE_MAX;
    sim_field_topology_config_default(&state->config);
}

void sim_field_topology_runtime_reset(SimFieldTopologyRuntimeState* state) {
    if (state == NULL) {
        return;
    }

    state->valid              = false;
    state->dirty              = true;
    state->last_computed_step = SIZE_MAX;
    state->summary            = (SimFieldTopologySummary) { 0 };
    if (state->cells != NULL && state->cell_capacity > 0U) {
        (void) memset(state->cells, 0, state->cell_capacity * sizeof(*state->cells));
    }
    if (state->workspace.ambiguity != NULL && state->workspace.capacity > 0U) {
        (void) memset(state->workspace.ambiguity, 0, state->workspace.capacity);
    }
}

void sim_field_topology_runtime_mark_dirty(SimFieldTopologyRuntimeState* state) {
    if (state == NULL) {
        return;
    }
    state->dirty = true;
}

SimResult sim_field_topology_runtime_resize(SimFieldTopologyRuntimeState* state,
                                            size_t                        width,
                                            size_t                        height) {
    const size_t cell_count = width * height;

    if (state == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (cell_count == 0U) {
        free(state->cells);
        state->cells         = NULL;
        state->width         = 0U;
        state->height        = 0U;
        state->cell_capacity = 0U;
        state->valid         = false;
        state->dirty         = true;
        sim_field_topology_runtime_workspace_free(&state->workspace);
        return SIM_RESULT_OK;
    }

    if (state->cell_capacity < cell_count || state->cells == NULL) {
        SimFieldTopologyCell* new_cells =
            (SimFieldTopologyCell*) realloc(state->cells, cell_count * sizeof(SimFieldTopologyCell));
        if (new_cells == NULL) {
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        state->cells = new_cells;
        if (cell_count > state->cell_capacity) {
            size_t old_capacity = state->cell_capacity;
            (void) memset(state->cells + old_capacity,
                          0,
                          (cell_count - old_capacity) * sizeof(SimFieldTopologyCell));
        }
        state->cell_capacity = cell_count;
    }

    if (sim_field_topology_runtime_workspace_resize(&state->workspace, cell_count) !=
        SIM_RESULT_OK) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    if (state->width != width || state->height != height) {
        state->width  = width;
        state->height = height;
        state->valid  = false;
        state->dirty  = true;
        state->generation += 1U;
    }

    return SIM_RESULT_OK;
}

void sim_field_topology_runtime_free(SimFieldTopologyRuntimeState* state) {
    if (state == NULL) {
        return;
    }

    free(state->cells);
    state->cells = NULL;
    sim_field_topology_runtime_workspace_free(&state->workspace);
    sim_field_topology_runtime_init(state);
}

bool sim_field_topology_runtime_recompute(SimFieldTopologyRuntimeState* state,
                                          const struct SimField*        field,
                                          size_t                        step_index) {
    size_t width  = 0U;
    size_t height = 0U;

    if (state == NULL || field == NULL) {
        return false;
    }

    if (!state->enabled) {
        state->valid = false;
        state->summary = (SimFieldTopologySummary) { 0 };
        state->dirty = true;
        return false;
    }

    state->request_count += 1U;

    if (!sim_field_topology_dimensions(field, &width, &height)) {
        state->valid = false;
        state->summary = (SimFieldTopologySummary) { 0 };
        state->dirty = true;
        return false;
    }

    if (sim_field_topology_runtime_resize(state, width, height) != SIM_RESULT_OK) {
        state->valid = false;
        state->summary = (SimFieldTopologySummary) { 0 };
        return false;
    }

    state->cadence_steps = sim_field_topology_runtime_sanitize_cadence(state->cadence_steps);
    if (!state->dirty && state->last_computed_step != SIZE_MAX && step_index >= state->last_computed_step &&
        (step_index - state->last_computed_step) < state->cadence_steps) {
        state->skip_count += 1U;
        return state->valid;
    }

    if (state->workspace.ambiguity != NULL && state->workspace.capacity >= (width * height)) {
        (void) memset(state->workspace.ambiguity, 0, width * height);
    }

    if (!sim_field_topology_extract(
            field, state->cells, width * height, &state->summary, &state->config, NULL, NULL)) {
        state->valid = false;
        state->summary = (SimFieldTopologySummary) { 0 };
        return false;
    }

    state->valid              = state->summary.valid;
    state->dirty              = false;
    state->last_computed_step = step_index;
    state->recompute_count += 1U;
    state->generation += 1U;

    if (state->workspace.ambiguity != NULL && state->workspace.capacity >= (width * height)) {
        for (size_t i = 0U; i < width * height; ++i) {
            state->workspace.ambiguity[i] =
                (state->cells[i].flags & SIM_FIELD_TOPOLOGY_CELL_AMBIGUOUS) != 0U ? 1U : 0U;
        }
    }

    return state->valid;
}

bool sim_field_topology_runtime_pack_rgba8(SimFieldTopologyRuntimeState* state,
                                           uint8_t*                      dest,
                                           size_t                        capacity,
                                           size_t*                       out_width,
                                           size_t*                       out_height,
                                           size_t*                       out_bytes) {
    size_t required = 0U;

    if (out_width != NULL) {
        *out_width = 0U;
    }
    if (out_height != NULL) {
        *out_height = 0U;
    }
    if (out_bytes != NULL) {
        *out_bytes = 0U;
    }

    if (state == NULL || !state->valid || state->cells == NULL || state->width == 0U ||
        state->height == 0U || dest == NULL) {
        return false;
    }

    required = state->width * state->height * 4U;
    if (capacity < required) {
        return false;
    }

    for (size_t i = 0U; i < state->width * state->height; ++i) {
        sim_field_topology_cell_pack_rgba8(&state->cells[i], &dest[i * 4U]);
    }

    state->pack_count += 1U;
    if (out_width != NULL) {
        *out_width = state->width;
    }
    if (out_height != NULL) {
        *out_height = state->height;
    }
    if (out_bytes != NULL) {
        *out_bytes = required;
    }
    return true;
}
