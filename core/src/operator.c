/**
 * @file operator.c
 * @brief Operator registry, metadata normalization, and dependency plan resolver.
 *
 * Operators wrap executable callbacks, symbolic/kernel metadata, field access
 * declarations, and dependency lists. This implementation assigns stable
 * identities, resolves schemas and IR opcodes, builds hazard-aware execution
 * plans, and maintains caller-owned payload semantics for registered operators.
 */
#include "oakfield/operator.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

static unsigned long long g_operator_guid_counter = 0ULL;

static uint64_t sim_operator_guid_epoch(void) {
    static uint64_t epoch   = 0ULL;
    uint64_t        current = __sync_val_compare_and_swap(&epoch, 0ULL, 0ULL);
    if (current != 0ULL) {
        return current;
    }

    uint64_t seed = 0ULL;
#if defined(__APPLE__)
    seed = mach_absolute_time();
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        seed = ((uint64_t) ts.tv_sec << 32) ^ (uint64_t) ts.tv_nsec;
    }
#endif
    if (seed == 0ULL) {
        seed = (uint64_t) time(NULL);
    }

    seed ^= (uint64_t) (uintptr_t) &epoch;
    seed ^= (uint64_t) (uintptr_t) &g_operator_guid_counter;
    seed ^= 0x9E3779B97F4A7C15ULL;

    if (seed == 0ULL) {
        seed = 0xD1CEB00B5ULL;
    }

    if (__sync_bool_compare_and_swap(&epoch, 0ULL, seed)) {
        return seed;
    }

    return __sync_val_compare_and_swap(&epoch, 0ULL, 0ULL);
}

static uint64_t sim_operator_next_guid(void) {
    uint64_t counter = __sync_add_and_fetch(&g_operator_guid_counter, 1ULL);
    uint64_t epoch   = sim_operator_guid_epoch();
    uint64_t guid    = counter ^ (epoch << 17);
    if (guid == 0ULL) {
        guid = counter ^ 0x9E3779B97F4A7C15ULL;
    }
    return guid;
}

static SimContinuityMode sim_operator_resolve_continuity(SimContinuityMode mode) {
    switch (mode) {
        case SIM_CONTINUITY_NONE:
        case SIM_CONTINUITY_STRICT:
        case SIM_CONTINUITY_CLAMPED:
        case SIM_CONTINUITY_LIMITED:
            return mode;
        default:
            return SIM_CONTINUITY_NONE;
    }
}

static bool sim_operator_ir_opcode_is_valid(SimIROpcode opcode) {
    switch (opcode) {
        case OAK_OP_DIFF:
        case OAK_OP_CONV:
        case OAK_OP_DISP:
        case OAK_OP_DIFFUSE:
        case OAK_OP_WARP:
        case OAK_OP_NOISE:
        case OAK_OP_FLOW:
        case OAK_OP_MISC:
        case OAK_OP_CORE:
            return true;
        default:
            return false;
    }
}

static SimIROpcode sim_operator_ir_opcode_for_identity(const SimOperatorInfo* info,
                                                       const char*            schema_key) {
    if (schema_key != NULL) {
        if (strcmp(schema_key, "dispersion") == 0) {
            return OAK_OP_DISP;
        }
        if (strcmp(schema_key, "spatial_derivative") == 0) {
            return OAK_OP_DIFF;
        }
        if (strcmp(schema_key, "analytic_warp") == 0) {
            return OAK_OP_WARP;
        }
        if (strcmp(schema_key, "minimal_convolution") == 0 || strcmp(schema_key, "sieve") == 0) {
            return OAK_OP_CONV;
        }
    }

    if (info == NULL) {
        return OAK_OP_MISC;
    }
    if (info->is_noise) {
        return OAK_OP_NOISE;
    }

    switch (info->category) {
        case SIM_OPERATOR_CATEGORY_DIFFUSION:
            return OAK_OP_DIFFUSE;
        case SIM_OPERATOR_CATEGORY_NOISE:
            return OAK_OP_NOISE;
        case SIM_OPERATOR_CATEGORY_ADVECTION:
        case SIM_OPERATOR_CATEGORY_COUPLING:
        case SIM_OPERATOR_CATEGORY_THERMOSTAT:
        case SIM_OPERATOR_CATEGORY_POTENTIAL:
            return OAK_OP_FLOW;
        case SIM_OPERATOR_CATEGORY_UTILITY:
        case SIM_OPERATOR_CATEGORY_MEASUREMENT:
        case SIM_OPERATOR_CATEGORY_REACTION:
            return OAK_OP_CORE;
        case SIM_OPERATOR_CATEGORY_NONLINEAR:
        case SIM_OPERATOR_CATEGORY_UNKNOWN:
        default:
            return OAK_OP_MISC;
    }
}

static const char* sim_operator_descriptor_schema_key(const SimOperatorDescriptor* descriptor) {
    if (descriptor == NULL) {
        return NULL;
    }
    if (descriptor->info.schema_key != NULL && descriptor->info.schema_key[0] != '\0') {
        return descriptor->info.schema_key;
    }
    return NULL;
}

void sim_operator_set_schema_key(SimOperator* op, const char* schema_key) {
    if (op == NULL) {
        return;
    }

    op->schema_key[0] = '\0';
    op->info.schema_key = NULL;
    if (schema_key == NULL || schema_key[0] == '\0') {
        return;
    }

    (void) strncpy(op->schema_key, schema_key, SIM_OPERATOR_SCHEMA_KEY_MAX);
    op->schema_key[SIM_OPERATOR_SCHEMA_KEY_MAX] = '\0';
    op->info.schema_key = op->schema_key;
}

void sim_operator_set_catalog_metadata(SimOperator* op, const void* metadata) {
    if (op == NULL) {
        return;
    }
    op->catalog_metadata = metadata;
}

const void* sim_operator_catalog_metadata(const SimOperator* op) {
    return (op != NULL) ? op->catalog_metadata : NULL;
}

const char* sim_continuity_mode_name(SimContinuityMode mode) {
    switch (mode) {
        case SIM_CONTINUITY_NONE:
            return "none";
        case SIM_CONTINUITY_STRICT:
            return "strict";
        case SIM_CONTINUITY_CLAMPED:
            return "clamped";
        case SIM_CONTINUITY_LIMITED:
            return "limited";
        default:
            break;
    }
    return "none";
}

bool sim_continuity_mode_from_string(const char* text, SimContinuityMode* out_mode) {
    SimContinuityMode mode = SIM_CONTINUITY_NONE;
    if (text == NULL) {
        return false;
    }

    if (strcasecmp(text, "none") == 0) {
        mode = SIM_CONTINUITY_NONE;
    } else if (strcasecmp(text, "strict") == 0) {
        mode = SIM_CONTINUITY_STRICT;
    } else if (strcasecmp(text, "clamped") == 0) {
        mode = SIM_CONTINUITY_CLAMPED;
    } else if (strcasecmp(text, "limited") == 0) {
        mode = SIM_CONTINUITY_LIMITED;
    } else {
        return false;
    }

    if (out_mode != NULL) {
        *out_mode = mode;
    }
    return true;
}

const char* sim_continuity_limiter_name(SimContinuityLimiterStrategy strategy) {
    switch (strategy) {
        case SIM_LIMITER_STRATEGY_HARD_CLIP:
            return "hard-clip";
        case SIM_LIMITER_STRATEGY_SOFT_CLIP:
            return "soft-clip";
        case SIM_LIMITER_STRATEGY_SIGMOID_COMPRESSION:
            return "sigmoid";
        case SIM_LIMITER_STRATEGY_ADAPTIVE_BIAS:
            return "adaptive-bias";
        case SIM_LIMITER_STRATEGY_STAT_AWARE:
            return "stat-aware";
        case SIM_LIMITER_STRATEGY_PER_FIELD_OVERRIDE:
            return "per-field-override";
        default:
            break;
    }
    return "unknown";
}

const char* sim_boundary_policy_name(SimIRBoundaryPolicy policy) {
    switch (policy) {
        case SIM_IR_BOUNDARY_NEUMANN:
            return "neumann";
        case SIM_IR_BOUNDARY_DIRICHLET:
            return "dirichlet";
        case SIM_IR_BOUNDARY_PERIODIC:
            return "periodic";
        case SIM_IR_BOUNDARY_REFLECTIVE:
            return "reflective";
        default:
            break;
    }
    return "neumann";
}

bool sim_boundary_policy_from_string(const char* text, SimIRBoundaryPolicy* out_policy) {
    SimIRBoundaryPolicy policy = SIM_IR_BOUNDARY_NEUMANN;

    if (text == NULL) {
        return false;
    }

    if (strcasecmp(text, "neumann") == 0) {
        policy = SIM_IR_BOUNDARY_NEUMANN;
    } else if (strcasecmp(text, "dirichlet") == 0) {
        policy = SIM_IR_BOUNDARY_DIRICHLET;
    } else if (strcasecmp(text, "periodic") == 0) {
        policy = SIM_IR_BOUNDARY_PERIODIC;
    } else if (strcasecmp(text, "reflective") == 0 || strcasecmp(text, "reflect") == 0) {
        policy = SIM_IR_BOUNDARY_REFLECTIVE;
    } else {
        return false;
    }

    if (out_policy != NULL) {
        *out_policy = policy;
    }
    return true;
}

void sim_operator_config_set_spacing(SimOperatorConfig* config,
                                     const double*      spacing,
                                     size_t             rank) {
    if (config == NULL) {
        return;
    }
    if (spacing == NULL || rank == 0U) {
        config->spacing_rank = 0U;
        memset(config->spacing, 0, sizeof(config->spacing));
        return;
    }
    if (rank > SIM_OPERATOR_MAX_SPACING_DIMS) {
        rank = SIM_OPERATOR_MAX_SPACING_DIMS;
    }
    for (size_t i = 0; i < rank; ++i) {
        config->spacing[i] = spacing[i];
    }
    for (size_t i = rank; i < SIM_OPERATOR_MAX_SPACING_DIMS; ++i) {
        config->spacing[i] = 0.0;
    }
    config->spacing_rank = (uint8_t) rank;
}

void sim_operator_config_normalize(SimOperatorConfig* config) {
    if (config == NULL) {
        return;
    }

    config->continuity = sim_operator_resolve_continuity(config->continuity);

    if (!isfinite((double) config->clamp_min) || !isfinite((double) config->clamp_max) ||
        config->clamp_min >= config->clamp_max) {
        config->clamp_min = 0.0;
        config->clamp_max = 0.0;
    }

    if (!isfinite((double) config->continuity_tol) || config->continuity_tol < 0.0) {
        config->continuity_tol = 0.0;
    }

    if (!isfinite(config->norm_budget) || config->norm_budget < 0.0) {
        config->norm_budget = 0.0;
    }
    if (!isfinite(config->norm_budget_softness) || config->norm_budget_softness < 0.0) {
        config->norm_budget_softness = 0.0;
    }

    switch (config->boundary) {
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_PERIODIC:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            break;
        default:
            config->boundary = SIM_IR_BOUNDARY_NEUMANN;
            break;
    }

    if (config->spacing_rank > SIM_OPERATOR_MAX_SPACING_DIMS) {
        config->spacing_rank = SIM_OPERATOR_MAX_SPACING_DIMS;
    }
    for (uint8_t i = 0U; i < config->spacing_rank; ++i) {
        if (!isfinite((double) config->spacing[i]) || config->spacing[i] <= 0.0) {
            config->spacing[i] = 0.0;
        }
    }

    if (config->representation_mode_override_enabled) {
        switch (config->representation_mode_override) {
            case SIM_REPRESENTATION_MODE_STRICT:
            case SIM_REPRESENTATION_MODE_RELAXED:
            case SIM_REPRESENTATION_MODE_EXPLORATION:
                break;
            default:
                config->representation_mode_override_enabled = false;
                config->representation_mode_override         = SIM_REPRESENTATION_MODE_RELAXED;
                break;
        }
    }
}

SimOperatorConfig sim_operator_config_defaults(void) {
    SimOperatorConfig cfg = { .continuity                           = SIM_CONTINUITY_NONE,
                              .clamp_min                            = 0.0,
                              .clamp_max                            = 0.0,
                              .continuity_tol                       = 0.0,
                              .boundary                             = SIM_IR_BOUNDARY_NEUMANN,
                              .spacing_rank                         = 0U,
                              .spacing                              = { 0.0, 0.0, 0.0, 0.0 },
                              .norm_budget                          = 0.0,
                              .norm_budget_softness                 = 0.0,
                              .representation_mode_override_enabled = false,
                              .representation_mode_override = SIM_REPRESENTATION_MODE_RELAXED };
    return cfg;
}

bool sim_operator_config_get(const SimOperator* op, SimOperatorConfig* out_config) {
    if (op == NULL || out_config == NULL) {
        return false;
    }

    *out_config = op->config;
    return true;
}

void sim_operator_config_set(struct SimOperator* op, const SimOperatorConfig* config) {
    if (op == NULL) {
        return;
    }

    if (config == NULL) {
        op->config = sim_operator_config_defaults();
    } else {
        op->config = *config;
    }

    sim_operator_config_normalize(&op->config);
}

static void sim_operator_kernel_builder_free(SimIRBuilder* builder) {
    if (builder == NULL) {
        return;
    }
    free(builder->nodes);
    free(builder->constants_data);
    free(builder->constants_offsets);
    free(builder->constants_components);
    free(builder);
}

static void sim_operator_kernel_destroy(SimOperatorKernel* kernel) {
    if (kernel == NULL) {
        return;
    }
    /* If we allocated a builder copy during kernel init, free it here */
    if (kernel->kernel.builder != NULL) {
        sim_operator_kernel_builder_free((SimIRBuilder*) kernel->kernel.builder);
        kernel->kernel.builder = NULL;
    }
    free(kernel->binding_map);
    free(kernel->bindings);
    free(kernel->output_map);
    free(kernel->outputs);
    free(kernel->params);
    free(kernel);
}

static SimResult sim_operator_kernel_init(SimOperatorKernel**                out_kernel,
                                          const SimOperatorKernelDescriptor* descriptor) {
    SimOperatorKernel* kernel;
    size_t             i;

    if (out_kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    *out_kernel = NULL;

    if (descriptor == NULL) {
        return SIM_RESULT_OK;
    }

    if (descriptor->builder == NULL || descriptor->bindings == NULL ||
        descriptor->binding_count == 0U || descriptor->outputs == NULL ||
        descriptor->output_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    kernel = (SimOperatorKernel*) calloc(1U, sizeof(SimOperatorKernel));
    if (kernel == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    kernel->bindings =
        (SimKernelIRBinding*) calloc(descriptor->binding_count, sizeof(SimKernelIRBinding));
    kernel->binding_map = (SimOperatorKernelBindingDescriptor*) calloc(
        descriptor->binding_count, sizeof(SimOperatorKernelBindingDescriptor));
    kernel->outputs =
        (SimKernelIROutput*) calloc(descriptor->output_count, sizeof(SimKernelIROutput));
    kernel->output_map = (SimOperatorKernelOutputDescriptor*) calloc(
        descriptor->output_count, sizeof(SimOperatorKernelOutputDescriptor));
    if (kernel->bindings == NULL || kernel->binding_map == NULL || kernel->outputs == NULL ||
        kernel->output_map == NULL) {
        sim_operator_kernel_destroy(kernel);
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    kernel->binding_count = descriptor->binding_count;
    kernel->output_count  = descriptor->output_count;
    kernel->param_count   = descriptor->param_count;

    if (descriptor->param_count > 0U) {
        kernel->params = (double*) calloc(descriptor->param_count, sizeof(double));
        if (kernel->params == NULL) {
            sim_operator_kernel_destroy(kernel);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        if (descriptor->params != NULL) {
            memcpy(kernel->params, descriptor->params, descriptor->param_count * sizeof(double));
        }
    }

    for (i = 0U; i < descriptor->binding_count; ++i) {
        kernel->binding_map[i]          = descriptor->bindings[i];
        kernel->bindings[i].field_index = descriptor->bindings[i].ir_field_index;
        kernel->bindings[i].field       = NULL;
    }

    for (i = 0U; i < descriptor->output_count; ++i) {
        kernel->output_map[i]          = descriptor->outputs[i];
        kernel->outputs[i].field_index = descriptor->outputs[i].ir_field_index;
        kernel->outputs[i].expression  = descriptor->outputs[i].expression;
    }

    /* Create a copy of builder to make kernel descriptor relocatable and immutable. */
    {
        const SimIRBuilder* src         = descriptor->builder;
        SimIRBuilder*       copy        = (SimIRBuilder*) calloc(1U, sizeof(SimIRBuilder));
        SimResult           copy_result = SIM_RESULT_OK;
        if (copy == NULL) {
            sim_operator_kernel_destroy(kernel);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        copy->count            = src->count;
        copy->capacity         = src->capacity;
        copy->default_boundary = src->default_boundary;
        if (src->capacity > 0U) {
            copy->nodes = (SimIRNode*) malloc(src->capacity * sizeof(SimIRNode));
            if (copy->nodes == NULL) {
                copy_result = SIM_RESULT_OUT_OF_MEMORY;
                goto copy_fail;
            }
            /* Copy only count nodes; uninitialized entries not needed */
            memcpy(copy->nodes, src->nodes, src->count * sizeof(SimIRNode));
        }

        /* Copy constants pool if present */
        if (src->constants_count > 0U) {
            size_t entry_capacity = src->constants_capacity;
            copy->constants_count = src->constants_count;
            if (entry_capacity < copy->constants_count) {
                entry_capacity = copy->constants_count;
            }
            copy->constants_capacity   = entry_capacity;
            copy->constants_offsets    = (size_t*) malloc(entry_capacity * sizeof(size_t));
            copy->constants_components = (size_t*) malloc(entry_capacity * sizeof(size_t));
            if (copy->constants_offsets == NULL || copy->constants_components == NULL) {
                copy_result = SIM_RESULT_OUT_OF_MEMORY;
                goto copy_fail;
            }
            memcpy(copy->constants_offsets,
                   src->constants_offsets,
                   copy->constants_count * sizeof(size_t));
            memcpy(copy->constants_components,
                   src->constants_components,
                   copy->constants_count * sizeof(size_t));
            /* compute total values to copy */
            size_t total_values = src->constants_data_used;
            if (total_values > 0U) {
                size_t data_capacity = src->constants_data_capacity;
                if (data_capacity < total_values) {
                    data_capacity = total_values;
                }
                copy->constants_data_capacity = data_capacity;
                copy->constants_data_used     = total_values;
                copy->constants_data          = (double*) malloc(data_capacity * sizeof(double));
                if (copy->constants_data == NULL) {
                    copy_result = SIM_RESULT_OUT_OF_MEMORY;
                    goto copy_fail;
                }
                memcpy(copy->constants_data, src->constants_data, total_values * sizeof(double));
#if defined(SIM_DEBUG_KERNEL_CONSTANTS)
                /* Print copied constants for debugging */
                fprintf(
                    stderr,
                    "[DEBUG] sim_operator_kernel_init: copy.constants_count=%zu total_values=%zu\n",
                    copy->constants_count,
                    total_values);
                for (size_t k = 0U; k < total_values; ++k) {
                    fprintf(
                        stderr, "[DEBUG] copy.const_data[%zu] = %g\n", k, copy->constants_data[k]);
                }
#endif
            }
        }
        kernel->kernel.builder = copy;
        /* Debug: print kernel builder copy info */
        if (copy != NULL) {
#if defined(SIM_DEBUG_KERNEL_CONSTANTS)
            fprintf(
                stderr,
                "[DEBUG] sim_operator_kernel_init: kernel builder copied nodes=%zu constants=%zu\n",
                copy->count,
                copy->constants_count);
#endif
        }
        goto copy_done;

    copy_fail:
        sim_operator_kernel_builder_free(copy);
        sim_operator_kernel_destroy(kernel);
        return copy_result;
    copy_done:;
    }
    kernel->kernel.bindings          = kernel->bindings;
    kernel->kernel.binding_count     = descriptor->binding_count;
    kernel->kernel.outputs           = kernel->outputs;
    kernel->kernel.output_count      = descriptor->output_count;
    kernel->kernel.params            = kernel->params;
    kernel->kernel.param_count       = kernel->param_count;
    kernel->kernel.required_features = descriptor->required_features;
    kernel->kernel.complex_semantics = descriptor->complex_semantics;

    *out_kernel = kernel;
    return SIM_RESULT_OK;
}

static void sim_operator_info_apply_ir_metadata(SimOperatorInfo*    info,
                                                const SimIRBuilder* builder) {
    if (info == NULL || builder == NULL || builder->nodes == NULL) {
        return;
    }

    double max_order    = 0.0;
    double max_stencil  = 0.0;
    double max_constant = 0.0;
    bool   saw_diff     = false;

    for (size_t i = 0U; i < builder->count; ++i) {
        const SimIRNode* node = &builder->nodes[i];
        if (node == NULL) {
            continue;
        }
        if (node->type == SIM_IR_NODE_DIFF) {
            saw_diff = true;
            if (node->data.diff.order > max_order) {
                max_order = (double) node->data.diff.order;
            }
            if (node->data.diff.stencil_order > max_stencil) {
                max_stencil = (double) node->data.diff.stencil_order;
            }
            if (node->data.diff.consistency_constant > max_constant) {
                max_constant = node->data.diff.consistency_constant;
            }
            /* If boundary hint is unset, inherit from first differential node. */
            if (info->representation.boundary == SIM_IR_BOUNDARY_NEUMANN &&
                node->data.diff.boundary != SIM_IR_BOUNDARY_NEUMANN) {
                info->representation.boundary = node->data.diff.boundary;
            }
        }
    }

    if (saw_diff) {
        if (info->approximation.spatial_order <= 0.0) {
            info->approximation.spatial_order = max_order;
        }
        if (info->approximation.stencil_order <= 0.0) {
            info->approximation.stencil_order = max_stencil;
        }
        if (info->approximation.error_constant <= 0.0) {
            info->approximation.error_constant = max_constant;
        }
    }
}

static void sim_operator_reset(SimOperator* op) {
    if (op == NULL) {
        return;
    }

    if (op->destroy != NULL) {
        op->destroy(op->userdata);
    }

    sim_operator_kernel_destroy(op->kernel);
    op->kernel = NULL;
    free(op->dependencies);
    op->dependencies     = NULL;
    op->dependency_count = 0U;
    free(op->read_indices);
    op->read_indices     = NULL;
    op->read_index_count = 0U;
    free(op->write_indices);
    op->write_indices     = NULL;
    op->write_index_count = 0U;
    op->evaluate          = NULL;
    op->destroy           = NULL;
    op->userdata          = NULL;
    op->name[0]           = '\0';
    op->schema_key[0]     = '\0';
    op->config            = sim_operator_config_defaults();
    op->catalog_metadata  = NULL;
}

SimResult sim_operator_registry_init(SimOperatorRegistry* registry) {
    if (registry == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    registry->records  = NULL;
    registry->count    = 0U;
    registry->capacity = 0U;

    return SIM_RESULT_OK;
}

void sim_operator_registry_destroy(SimOperatorRegistry* registry) {
    size_t i;

    if (registry == NULL) {
        return;
    }

    for (i = 0; i < registry->count; ++i) {
        sim_operator_reset(&registry->records[i]);
    }

    free(registry->records);
    registry->records  = NULL;
    registry->count    = 0U;
    registry->capacity = 0U;
}

static SimResult sim_operator_grow(SimOperatorRegistry* registry) {
    size_t       new_capacity;
    SimOperator* new_records;

    if (registry->count < registry->capacity) {
        return SIM_RESULT_OK;
    }

    new_capacity = (registry->capacity == 0U) ? 4U : (registry->capacity * 2U);
    new_records  = (SimOperator*) realloc(registry->records, new_capacity * sizeof(SimOperator));
    if (new_records == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }

    registry->records  = new_records;
    registry->capacity = new_capacity;
    return SIM_RESULT_OK;
}

static SimResult sim_operator_assign(SimOperator* op, const SimOperatorDescriptor* descriptor) {
    size_t          i;
    SimOperatorInfo info;

    if (op == NULL || descriptor == NULL || descriptor->name == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    if (descriptor->evaluate == NULL && descriptor->kernel == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    (void) memset(op, 0, sizeof(*op));

    /* assign a mixed epoch+counter GUID for tracing */
    op->guid = sim_operator_next_guid();

    (void) strncpy(op->name, descriptor->name, SIM_OPERATOR_NAME_MAX);
    op->name[SIM_OPERATOR_NAME_MAX] = '\0';

    op->evaluate      = descriptor->evaluate;
    op->save_state    = descriptor->save_state;
    op->restore_state = descriptor->restore_state;
    op->destroy       = descriptor->destroy;
    op->userdata      = descriptor->userdata;
    op->save_state    = descriptor->save_state;
    op->restore_state = descriptor->restore_state;
    info              = descriptor->info;
    sim_operator_info_normalize(&info);
    op->info = info;
    sim_operator_config_set(op, &descriptor->config);
    op->read_mask  = descriptor->read_mask;
    op->write_mask = descriptor->write_mask;
    if (descriptor->read_index_count > 0U && descriptor->read_indices != NULL) {
        op->read_indices = (size_t*) calloc(descriptor->read_index_count, sizeof(size_t));
        if (op->read_indices == NULL) {
            sim_operator_reset(op);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        memcpy(op->read_indices,
               descriptor->read_indices,
               descriptor->read_index_count * sizeof(size_t));
        op->read_index_count = descriptor->read_index_count;
    }
    if (descriptor->write_index_count > 0U && descriptor->write_indices != NULL) {
        op->write_indices = (size_t*) calloc(descriptor->write_index_count, sizeof(size_t));
        if (op->write_indices == NULL) {
            sim_operator_reset(op);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        memcpy(op->write_indices,
               descriptor->write_indices,
               descriptor->write_index_count * sizeof(size_t));
        op->write_index_count = descriptor->write_index_count;
    }
    op->required_features = descriptor->required_features;
    op->catalog_metadata  = descriptor->catalog_metadata;
    op->config_adapter    = descriptor->config_adapter;
    op->graph_ir_view     = descriptor->graph_ir_view;
    sim_operator_set_schema_key(op, sim_operator_descriptor_schema_key(descriptor));

    if (descriptor->kernel != NULL) {
        SimResult kernel_result = sim_operator_kernel_init(&op->kernel, descriptor->kernel);
        if (kernel_result != SIM_RESULT_OK) {
            sim_operator_reset(op);
            return kernel_result;
        }
        /* Tag kernel IR nodes with the core-owned semantic opcode. */
        if (op->kernel != NULL && op->kernel->kernel.builder != NULL) {
            SimIROpcode     opcode          = sim_operator_ir_opcode(op);
            SimResult       opcode_result   = SIM_RESULT_INVALID_ARGUMENT;
            const KernelIR* kernel          = &op->kernel->kernel;
            SimIRBuilder*   mutable_builder = (SimIRBuilder*) op->kernel->kernel.builder;
            if (kernel->output_count > 0U && kernel->outputs != NULL) {
                SimIRNodeId* roots =
                    (SimIRNodeId*) calloc(kernel->output_count, sizeof(SimIRNodeId));
                if (roots != NULL) {
                    for (size_t i = 0U; i < kernel->output_count; ++i) {
                        roots[i] = kernel->outputs[i].expression;
                    }
                    opcode_result = sim_ir_builder_apply_opcode_reachable(
                        mutable_builder, opcode, true, roots, kernel->output_count);
                    free(roots);
                } else {
                    opcode_result = SIM_RESULT_OUT_OF_MEMORY;
                }
            }

            if (opcode_result != SIM_RESULT_OK) {
                sim_ir_builder_apply_opcode(mutable_builder, opcode, true);
            }
            sim_operator_info_apply_ir_metadata(&op->info, op->kernel->kernel.builder);
        }
    }

    if (descriptor->dependency_count > 0U) {
        op->dependencies = (size_t*) calloc(descriptor->dependency_count, sizeof(size_t));
        if (op->dependencies == NULL) {
            sim_operator_reset(op);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        for (i = 0; i < descriptor->dependency_count; ++i) {
            op->dependencies[i] = descriptor->dependencies[i];
        }
        op->dependency_count = descriptor->dependency_count;
    }

    return SIM_RESULT_OK;
}

SimResult sim_operator_registry_register(SimOperatorRegistry*         registry,
                                         const SimOperatorDescriptor* descriptor,
                                         size_t*                      out_index) {
    SimResult    result;
    SimOperator* slot;

    if (registry == NULL || descriptor == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    result = sim_operator_grow(registry);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    slot = &registry->records[registry->count];

    result = sim_operator_assign(slot, descriptor);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    if (out_index != NULL) {
        *out_index = registry->count;
    }

    registry->count += 1U;
    return SIM_RESULT_OK;
}

SimOperator* sim_operator_registry_get(SimOperatorRegistry* registry, size_t index) {
    if (registry == NULL || index >= registry->count) {
        return NULL;
    }
    return &registry->records[index];
}

void sim_operator_dependencies(const SimOperator* op, const size_t** out_deps, size_t* out_count) {
    if (out_deps != NULL) {
        *out_deps = (op != NULL) ? op->dependencies : NULL;
    }
    if (out_count != NULL) {
        *out_count = (op != NULL) ? op->dependency_count : 0U;
    }
}

static SimResult sim_operator_plan_allocate(SimOperatorPlan* plan, size_t count) {
    if (plan == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    plan->order = NULL;
    plan->count = 0U;

    if (count == 0U) {
        return SIM_RESULT_OK;
    }

    plan->order = (size_t*) calloc(count, sizeof(size_t));
    if (plan->order == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    plan->count = count;
    return SIM_RESULT_OK;
}

static void sim_operator_plan_reset(SimOperatorPlan* plan) {
    if (plan == NULL) {
        return;
    }
    free(plan->order);
    plan->order = NULL;
    plan->count = 0U;
}

typedef struct {
    size_t   from;
    size_t   to;
    uint64_t from_guid;
    uint64_t to_guid;
} SimHazardEdge;

typedef struct {
    size_t* indices;
    size_t  count;
    size_t  capacity;
} SimReaderList;

static void sim_reader_list_init(SimReaderList* list) {
    if (list == NULL) {
        return;
    }
    list->indices  = NULL;
    list->count    = 0U;
    list->capacity = 0U;
}

static void sim_reader_list_destroy(SimReaderList* list) {
    if (list == NULL) {
        return;
    }
    free(list->indices);
    list->indices  = NULL;
    list->count    = 0U;
    list->capacity = 0U;
}

static bool sim_reader_list_append(SimReaderList* list, size_t value) {
    size_t  new_capacity;
    size_t* new_data;

    if (list == NULL) {
        return false;
    }

    if (list->count == list->capacity) {
        new_capacity = (list->capacity == 0U) ? 4U : (list->capacity * 2U);
        new_data     = (size_t*) realloc(list->indices, new_capacity * sizeof(size_t));
        if (new_data == NULL) {
            return false;
        }
        list->indices  = new_data;
        list->capacity = new_capacity;
    }

    list->indices[list->count++] = value;
    return true;
}

static bool sim_hazard_append(const SimOperatorRegistry* registry,
                              SimHazardEdge**            edges,
                              size_t*                    count,
                              size_t*                    capacity,
                              size_t                     from,
                              size_t                     to) {
    size_t         new_capacity;
    SimHazardEdge* new_edges;

    if (edges == NULL || count == NULL || capacity == NULL) {
        return false;
    }

    if (*count == *capacity) {
        new_capacity = (*capacity == 0U) ? 8U : (*capacity * 2U);
        new_edges    = (SimHazardEdge*) realloc(*edges, new_capacity * sizeof(SimHazardEdge));
        if (new_edges == NULL) {
            return false;
        }
        *edges    = new_edges;
        *capacity = new_capacity;
    }

    (*edges)[*count].from = from;
    (*edges)[*count].to   = to;
    if (registry != NULL) {
        const SimOperator* from_op = (from < registry->count) ? &registry->records[from] : NULL;
        const SimOperator* to_op   = (to < registry->count) ? &registry->records[to] : NULL;
        (*edges)[*count].from_guid = (from_op != NULL) ? from_op->guid : 0ULL;
        (*edges)[*count].to_guid   = (to_op != NULL) ? to_op->guid : 0ULL;
    } else {
        (*edges)[*count].from_guid = 0ULL;
        (*edges)[*count].to_guid   = 0ULL;
    }
    *count += 1U;
    return true;
}

static bool sim_hazard_contains(const SimHazardEdge* edges, size_t count, size_t from, size_t to) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (edges[i].from == from && edges[i].to == to) {
            return true;
        }
    }
    return false;
}

static void
sim_operator_log_emit(SimOperatorCycleLogFn logger, void* userdata, const char* message) {
    if (logger != NULL) {
        logger(message, userdata);
    } else {
        fprintf(stderr, "%s\n", message);
    }
}

static void
sim_operator_log_printf(SimOperatorCycleLogFn logger, void* userdata, const char* format, ...) {
    char    buffer[512];
    va_list args;
    va_start(args, format);
    (void) vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    sim_operator_log_emit(logger, userdata, buffer);
}

static void sim_operator_log_cycle(const SimOperatorRegistry* registry,
                                   const size_t*              indegree,
                                   const size_t*              out_degree,
                                   size_t                     node_count,
                                   const SimHazardEdge*       hazard_edges,
                                   size_t                     hazard_count,
                                   SimOperatorCycleLogFn      logger,
                                   void*                      logger_userdata) {
    size_t   i;
    bool     any         = false;
    uint8_t* problematic = NULL;

    if (registry == NULL || indegree == NULL || node_count == 0U) {
        return;
    }

    problematic = (uint8_t*) calloc(node_count, sizeof(uint8_t));
    if (problematic == NULL) {
        sim_operator_log_emit(
            logger,
            logger_userdata,
            "[ERROR] sim_operator_resolve_plan: cycle detected (detail allocation failed)");
        return;
    }

    sim_operator_log_printf(
        logger,
        logger_userdata,
        "[ERROR] sim_operator_resolve_plan: detected dependency cycle (%zu operators remain)",
        node_count);

    for (i = 0U; i < node_count; ++i) {
        if (indegree[i] == 0U) {
            continue;
        }

        const SimOperator* op        = (i < registry->count) ? &registry->records[i] : NULL;
        const char*        name      = (op != NULL && op->name[0] != '\0') ? op->name : "<unnamed>";
        uint64_t           guid      = (op != NULL) ? op->guid : 0ULL;
        size_t             outgoing  = (out_degree != NULL) ? out_degree[i] : 0U;
        size_t             dep_count = (op != NULL) ? op->dependency_count : 0U;
        const char*        aid       = sim_operator_abstract_id(op);

        sim_operator_log_printf(logger,
                                logger_userdata,
                                "    op[%zu] guid=%" PRIu64
                                " name=%s aid=%s indegree=%zu out_degree=%zu deps=%zu",
                                i,
                                (uint64_t) guid,
                                name,
                                (aid != NULL) ? aid : "<none>",
                                indegree[i],
                                outgoing,
                                dep_count);

        if (op != NULL && dep_count > 0U) {
            size_t dep_idx;
            char   line[512];
            int    written = snprintf(line, sizeof(line), "        depends on:");
            for (dep_idx = 0U; dep_idx < dep_count; ++dep_idx) {
                size_t             dep = op->dependencies[dep_idx];
                const SimOperator* dep_op =
                    (dep < registry->count) ? &registry->records[dep] : NULL;
                const char* dep_name =
                    (dep_op != NULL && dep_op->name[0] != '\0') ? dep_op->name : "<unnamed>";
                uint64_t dep_guid = (dep_op != NULL) ? dep_op->guid : 0ULL;
                if (written > 0 && (size_t) written < sizeof(line)) {
                    written += snprintf(
                        line + written,
                        (written < (int) sizeof(line)) ? sizeof(line) - (size_t) written : 0U,
                        " %zu(guid=%" PRIu64 ":%s)",
                        dep,
                        (uint64_t) dep_guid,
                        dep_name);
                }
            }
            sim_operator_log_emit(logger, logger_userdata, line);
        }

        if (op != NULL) {
            problematic[i] = 1U;
            any            = true;
        }
    }

    if (any && hazard_edges != NULL && hazard_count > 0U) {
        sim_operator_log_emit(logger, logger_userdata, "    hazard edges contributing to cycle:");
        for (i = 0U; i < hazard_count; ++i) {
            size_t from = hazard_edges[i].from;
            size_t to   = hazard_edges[i].to;
            if (from >= node_count || to >= node_count) {
                continue;
            }
            if (!problematic[from] && !problematic[to]) {
                continue;
            }
            const SimOperator* from_op = (from < registry->count) ? &registry->records[from] : NULL;
            const SimOperator* to_op   = (to < registry->count) ? &registry->records[to] : NULL;
            const char*        from_name =
                (from_op != NULL && from_op->name[0] != '\0') ? from_op->name : "<unnamed>";
            const char* to_name =
                (to_op != NULL && to_op->name[0] != '\0') ? to_op->name : "<unnamed>";
            sim_operator_log_printf(logger,
                                    logger_userdata,
                                    "        %zu(guid=%" PRIu64 ":%s) -> %zu(guid=%" PRIu64 ":%s)",
                                    from,
                                    (uint64_t) hazard_edges[i].from_guid,
                                    from_name,
                                    to,
                                    (uint64_t) hazard_edges[i].to_guid,
                                    to_name);
        }
    }

    free(problematic);
}

static bool sim_resolve_add_hazard(const SimOperatorRegistry* registry,
                                   SimHazardEdge**            edges,
                                   size_t*                    count,
                                   size_t*                    capacity,
                                   size_t                     from,
                                   size_t                     to,
                                   size_t*                    indegree,
                                   size_t*                    out_degree,
                                   size_t*                    total_edges) {
    const SimOperator* target;

    if (registry == NULL || edges == NULL || count == NULL || capacity == NULL ||
        indegree == NULL || out_degree == NULL || total_edges == NULL) {
        return false;
    }

    if (from == (size_t) (-1) || to == (size_t) (-1) || from == to) {
        return true;
    }

    target = &registry->records[to];
    if (target != NULL) {
        size_t dep_idx;
        for (dep_idx = 0U; dep_idx < target->dependency_count; ++dep_idx) {
            if (target->dependencies[dep_idx] == from) {
                return true;
            }
        }
    }

    if (sim_hazard_contains(*edges, *count, from, to)) {
        return true;
    }

    if (!sim_hazard_append(registry, edges, count, capacity, from, to)) {
        return false;
    }

    indegree[to] += 1U;
    out_degree[from] += 1U;
    *total_edges += 1U;
    return true;
}

SimResult sim_operator_resolve_plan_with_logger(const SimOperatorRegistry* registry,
                                                SimOperatorPlan*           plan,
                                                SimOperatorCycleLogFn      logger,
                                                void*                      logger_userdata) {
    size_t         i;
    size_t         n;
    size_t         queue_head      = 0U;
    size_t         queue_tail      = 0U;
    size_t         order_index     = 0U;
    size_t         total_edges     = 0U;
    size_t*        indegree        = NULL;
    size_t*        out_degree      = NULL;
    size_t*        adjacency       = NULL;
    size_t*        offsets         = NULL;
    size_t*        cursor          = NULL;
    size_t*        queue           = NULL;
    SimHazardEdge* hazard_edges    = NULL;
    size_t         hazard_count    = 0U;
    size_t         hazard_capacity = 0U;
    SimReaderList* readers         = NULL;
    size_t*        last_writer     = NULL;
    size_t         field_slots     = 0U;
    SimResult      result          = SIM_RESULT_OK;

    if (registry == NULL || plan == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    sim_operator_plan_reset(plan);

    n = registry->count;

    result = sim_operator_plan_allocate(plan, n);
    if (result != SIM_RESULT_OK) {
        return result;
    }

    if (n == 0U) {
        return SIM_RESULT_OK;
    }

    indegree   = (size_t*) calloc(n, sizeof(size_t));
    out_degree = (size_t*) calloc(n, sizeof(size_t));

    if (indegree == NULL || out_degree == NULL) {
        result = SIM_RESULT_OUT_OF_MEMORY;
        goto cleanup;
    }

    /* Determine max field index referenced by any operator (mask or extended indices). */
    {
        bool   have_fields = false;
        size_t max_field   = 0U;
        for (i = 0U; i < n; ++i) {
            const SimOperator* op   = &registry->records[i];
            uint64_t           mask = op->read_mask | op->write_mask;
            if (mask != 0ULL) {
                size_t highest = 63U - (size_t) __builtin_clzll(mask);
                if (!have_fields || highest > max_field) {
                    max_field = highest;
                }
                have_fields = true;
            }
            for (size_t r = 0U; r < op->read_index_count; ++r) {
                size_t field = op->read_indices[r];
                if (!have_fields || field > max_field) {
                    max_field = field;
                }
                have_fields = true;
            }
            for (size_t w = 0U; w < op->write_index_count; ++w) {
                size_t field = op->write_indices[w];
                if (!have_fields || field > max_field) {
                    max_field = field;
                }
                have_fields = true;
            }
        }
        if (have_fields) {
            field_slots = max_field + 1U;
            readers     = (SimReaderList*) calloc(field_slots, sizeof(SimReaderList));
            last_writer = (size_t*) malloc(field_slots * sizeof(size_t));
            if (readers == NULL || last_writer == NULL) {
                result = SIM_RESULT_OUT_OF_MEMORY;
                goto cleanup;
            }
            for (i = 0U; i < field_slots; ++i) {
                sim_reader_list_init(&readers[i]);
                last_writer[i] = (size_t) (-1);
            }
        }
    }

    hazard_capacity = (n > 0U) ? (n * 4U) : 1U;
    hazard_edges    = (SimHazardEdge*) calloc(hazard_capacity, sizeof(SimHazardEdge));
    if (hazard_edges == NULL) {
        result = SIM_RESULT_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (i = 0U; i < n; ++i) {
        const SimOperator* op = &registry->records[i];
        size_t             dep_idx;
        for (dep_idx = 0U; dep_idx < op->dependency_count; ++dep_idx) {
            size_t dep = op->dependencies[dep_idx];
            if (dep >= n) {
                result = SIM_RESULT_INVALID_ARGUMENT;
                goto cleanup;
            }
            indegree[i] += 1U;
            out_degree[dep] += 1U;
            total_edges += 1U;
        }

        if (field_slots > 0U && (op->read_mask != 0ULL || op->write_mask != 0ULL ||
                                 op->read_index_count > 0U || op->write_index_count > 0U)) {
            uint64_t reads        = op->read_mask;
            uint64_t writes       = op->write_mask;
            uint8_t* seen         = (uint8_t*) calloc(field_slots, sizeof(uint8_t));
            bool     hazard_error = false;
            if (seen == NULL) {
                result = SIM_RESULT_OUT_OF_MEMORY;
                goto cleanup;
            }
            while (reads != 0ULL) {
                unsigned long bit   = __builtin_ctzll(reads);
                size_t        field = (size_t) bit;
                reads &= ~(1ULL << bit);
                if (field < field_slots && (seen[field] & 0x1U) == 0U) {
                    seen[field] |= 0x1U;
                    if (last_writer[field] != (size_t) (-1)) {
                        if (!sim_resolve_add_hazard(registry,
                                                    &hazard_edges,
                                                    &hazard_count,
                                                    &hazard_capacity,
                                                    last_writer[field],
                                                    i,
                                                    indegree,
                                                    out_degree,
                                                    &total_edges)) {
                            result       = SIM_RESULT_OUT_OF_MEMORY;
                            hazard_error = true;
                            break;
                        }
                    }
                    if (!sim_reader_list_append(&readers[field], i)) {
                        result       = SIM_RESULT_OUT_OF_MEMORY;
                        hazard_error = true;
                        break;
                    }
                }
            }
            if (hazard_error) {
                free(seen);
                goto cleanup;
            }
            for (size_t r = 0U; r < op->read_index_count; ++r) {
                size_t field = op->read_indices[r];
                if (field < field_slots && (seen[field] & 0x1U) == 0U) {
                    seen[field] |= 0x1U;
                    if (last_writer[field] != (size_t) (-1)) {
                        if (!sim_resolve_add_hazard(registry,
                                                    &hazard_edges,
                                                    &hazard_count,
                                                    &hazard_capacity,
                                                    last_writer[field],
                                                    i,
                                                    indegree,
                                                    out_degree,
                                                    &total_edges)) {
                            result       = SIM_RESULT_OUT_OF_MEMORY;
                            hazard_error = true;
                            break;
                        }
                    }
                    if (!sim_reader_list_append(&readers[field], i)) {
                        result       = SIM_RESULT_OUT_OF_MEMORY;
                        hazard_error = true;
                        break;
                    }
                }
            }
            if (hazard_error) {
                free(seen);
                goto cleanup;
            }

            while (writes != 0ULL) {
                unsigned long bit   = __builtin_ctzll(writes);
                size_t        field = (size_t) bit;
                writes &= ~(1ULL << bit);
                if (field < field_slots && (seen[field] & 0x2U) == 0U) {
                    seen[field] |= 0x2U;
                    size_t reader_index;
                    if (last_writer[field] != (size_t) (-1)) {
                        if (!sim_resolve_add_hazard(registry,
                                                    &hazard_edges,
                                                    &hazard_count,
                                                    &hazard_capacity,
                                                    last_writer[field],
                                                    i,
                                                    indegree,
                                                    out_degree,
                                                    &total_edges)) {
                            result       = SIM_RESULT_OUT_OF_MEMORY;
                            hazard_error = true;
                            break;
                        }
                    }
                    for (reader_index = 0U; reader_index < readers[field].count; ++reader_index) {
                        if (!sim_resolve_add_hazard(registry,
                                                    &hazard_edges,
                                                    &hazard_count,
                                                    &hazard_capacity,
                                                    readers[field].indices[reader_index],
                                                    i,
                                                    indegree,
                                                    out_degree,
                                                    &total_edges)) {
                            result       = SIM_RESULT_OUT_OF_MEMORY;
                            hazard_error = true;
                            break;
                        }
                    }
                    readers[field].count = 0U;
                    last_writer[field]   = i;
                    if (hazard_error) {
                        break;
                    }
                }
            }
            if (hazard_error) {
                free(seen);
                goto cleanup;
            }
            for (size_t w = 0U; w < op->write_index_count; ++w) {
                size_t field = op->write_indices[w];
                if (field < field_slots && (seen[field] & 0x2U) == 0U) {
                    seen[field] |= 0x2U;
                    size_t reader_index;
                    if (last_writer[field] != (size_t) (-1)) {
                        if (!sim_resolve_add_hazard(registry,
                                                    &hazard_edges,
                                                    &hazard_count,
                                                    &hazard_capacity,
                                                    last_writer[field],
                                                    i,
                                                    indegree,
                                                    out_degree,
                                                    &total_edges)) {
                            result       = SIM_RESULT_OUT_OF_MEMORY;
                            hazard_error = true;
                            break;
                        }
                    }
                    for (reader_index = 0U; reader_index < readers[field].count; ++reader_index) {
                        if (!sim_resolve_add_hazard(registry,
                                                    &hazard_edges,
                                                    &hazard_count,
                                                    &hazard_capacity,
                                                    readers[field].indices[reader_index],
                                                    i,
                                                    indegree,
                                                    out_degree,
                                                    &total_edges)) {
                            result       = SIM_RESULT_OUT_OF_MEMORY;
                            hazard_error = true;
                            break;
                        }
                    }
                    readers[field].count = 0U;
                    last_writer[field]   = i;
                    if (hazard_error) {
                        break;
                    }
                }
            }
            if (hazard_error) {
                free(seen);
                goto cleanup;
            }
            free(seen);
        }
    }

    offsets   = (size_t*) calloc(n + 1U, sizeof(size_t));
    cursor    = (size_t*) calloc(n, sizeof(size_t));
    adjacency = (size_t*) calloc(total_edges > 0U ? total_edges : 1U, sizeof(size_t));
    queue     = (size_t*) calloc(n, sizeof(size_t));

    if (offsets == NULL || cursor == NULL || adjacency == NULL || queue == NULL) {
        result = SIM_RESULT_OUT_OF_MEMORY;
        goto cleanup;
    }

    offsets[0] = 0U;
    for (i = 0U; i < n; ++i) {
        offsets[i + 1U] = offsets[i] + out_degree[i];
    }

    for (i = 0U; i < n; ++i) {
        const SimOperator* op = &registry->records[i];
        size_t             dep_idx;
        for (dep_idx = 0U; dep_idx < op->dependency_count; ++dep_idx) {
            size_t dep        = op->dependencies[dep_idx];
            size_t offset     = offsets[dep] + cursor[dep];
            adjacency[offset] = i;
            cursor[dep] += 1U;
        }
    }

    for (i = 0U; i < hazard_count; ++i) {
        size_t from = hazard_edges[i].from;
        size_t to   = hazard_edges[i].to;
        if (from >= n || to >= n) {
            continue;
        }
        size_t offset     = offsets[from] + cursor[from];
        adjacency[offset] = to;
        cursor[from] += 1U;
    }

    for (i = 0U; i < n; ++i) {
        if (indegree[i] == 0U) {
            queue[queue_tail++] = i;
        }
    }

    while (queue_head < queue_tail) {
        size_t node = queue[queue_head++];
        size_t edge_index;

        plan->order[order_index++] = node;

        for (edge_index = offsets[node]; edge_index < offsets[node + 1U]; ++edge_index) {
            size_t neighbor = adjacency[edge_index];
            if (indegree[neighbor] == 0U) {
                continue;
            }
            indegree[neighbor] -= 1U;
            if (indegree[neighbor] == 0U) {
                queue[queue_tail++] = neighbor;
            }
        }
    }

    if (order_index != n) {
        sim_operator_log_cycle(
            registry, indegree, out_degree, n, hazard_edges, hazard_count, logger, logger_userdata);
        result = SIM_RESULT_DEPENDENCY_ERROR;
        goto cleanup;
    }

    plan->count = n;

cleanup:
    free(indegree);
    free(out_degree);
    free(adjacency);
    free(offsets);
    free(cursor);
    free(queue);
    free(hazard_edges);
    if (readers != NULL) {
        for (i = 0U; i < field_slots; ++i) {
            sim_reader_list_destroy(&readers[i]);
        }
        free(readers);
    }
    free(last_writer);

    if (result != SIM_RESULT_OK) {
        sim_operator_plan_destroy(plan);
    }

    return result;
}

SimResult sim_operator_resolve_plan(const SimOperatorRegistry* registry, SimOperatorPlan* plan) {
    return sim_operator_resolve_plan_with_logger(registry, plan, NULL, NULL);
}

void sim_operator_plan_destroy(SimOperatorPlan* plan) {
    sim_operator_plan_reset(plan);
}

void* sim_operator_payload(SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    return op->userdata;
}

const char* sim_operator_name(const SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    return op->name;
}

SimOperatorInfo sim_operator_info_defaults(void) {
    SimOperatorInfo info = { .category          = SIM_OPERATOR_CATEGORY_UNKNOWN,
                             .warp_level        = SIM_WARP_LEVEL_NONE,
                             .schema_key        = NULL,
                             .has_ir_opcode     = false,
                             .ir_opcode         = OAK_OP_MISC,
                             .is_noise          = false,
                             .is_differentiable = true,
                             .is_spectral       = false,
                             .is_local          = false,
                             .is_nonlocal       = false,
                             .is_linear         = false,
                             .is_warp           = false,
                             .preserves_real    = false,
                             .preferred_dt      = 0.0,
                             .abstract_id       = NULL,
                             .abstract_id_fn    = NULL,
                             .algebraic_flags   = SIM_OPERATOR_ALG_NONE,
                             .determinism_flags = SIM_DET_NONE,
                             .neural            = { .enabled = false,
                                                    .determinism_policy = SIM_NEURAL_DETERMINISM_INHERIT,
                                                    .device_requirement = SIM_NEURAL_DEVICE_ANY,
                                                    .precision_mode = SIM_NEURAL_PRECISION_DEFAULT,
                                                    .shape = {
                                                        .min_rank = 1U,
                                                        .max_rank = 0U,
                                                        .channel_axis = SIM_NEURAL_CHANNEL_AXIS_AUTO,
                                                        .min_channels = 0U,
                                                        .max_channels = 0U,
                                                        .channels_last = true,
                                                        .allow_complex_input = true,
                                                    } },
                             .representation    = { .domain     = SIM_FIELD_DOMAIN_PHYSICAL,
                                                    .value_kind = SIM_FIELD_VALUE_REAL_SCALAR,
                                                    .requires_complex_input          = false,
                                                    .preserves_real_subspace         = false,
                                                    .requires_complex_representation = false,
                                                    .boundary          = SIM_IR_BOUNDARY_NEUMANN,
                                                    .spacing_hint_rank = 0U,
                                                    .spacing_hint      = { 0.0, 0.0, 0.0, 0.0 } },
                             .approximation     = { .spatial_order  = 0.0,
                                                    .stencil_order  = 0.0,
                                                    .error_constant = 0.0,
                                                    .temporal_order = 0.0 },
                             .invariants        = { { SIM_OPERATOR_INVARIANT_NONE, 0.0 } },
                             .invariant_count   = 0U };

    return info;
}

void sim_operator_info_normalize(SimOperatorInfo* info) {
    if (info == NULL) {
        return;
    }

    SimOperatorInfo defaults = sim_operator_info_defaults();

    if (info->schema_key != NULL && info->schema_key[0] == '\0') {
        info->schema_key = NULL;
    }

    if (!info->has_ir_opcode || !sim_operator_ir_opcode_is_valid(info->ir_opcode)) {
        info->has_ir_opcode = false;
        info->ir_opcode     = defaults.ir_opcode;
    }

    /* Fill spacing hint padding and clamp rank. */
    if (info->representation.spacing_hint_rank > SIM_OPERATOR_MAX_SPACING_DIMS) {
        info->representation.spacing_hint_rank = SIM_OPERATOR_MAX_SPACING_DIMS;
    }
    for (uint8_t i = info->representation.spacing_hint_rank; i < SIM_OPERATOR_MAX_SPACING_DIMS;
         ++i) {
        info->representation.spacing_hint[i] = 0.0;
    }

    /* Validate boundary */
    switch (info->representation.boundary) {
        case SIM_IR_BOUNDARY_NEUMANN:
        case SIM_IR_BOUNDARY_DIRICHLET:
        case SIM_IR_BOUNDARY_PERIODIC:
        case SIM_IR_BOUNDARY_REFLECTIVE:
            break;
        default:
            info->representation.boundary = defaults.representation.boundary;
            break;
    }

    switch (info->representation.domain) {
        case SIM_FIELD_DOMAIN_PHYSICAL:
        case SIM_FIELD_DOMAIN_SPECTRAL:
        case SIM_FIELD_DOMAIN_HYBRID:
            break;
        default:
            info->representation.domain = defaults.representation.domain;
            break;
    }

    switch (info->representation.value_kind) {
        case SIM_FIELD_VALUE_REAL_SCALAR:
        case SIM_FIELD_VALUE_COMPLEX_SCALAR:
        case SIM_FIELD_VALUE_COMPLEX_IMAG_ZERO_CONSTRAINT:
        case SIM_FIELD_VALUE_COMPLEX_REAL_CONSTRAINT:
            break;
        default:
            info->representation.value_kind = defaults.representation.value_kind;
            break;
    }

    switch (info->neural.determinism_policy) {
        case SIM_NEURAL_DETERMINISM_INHERIT:
        case SIM_NEURAL_DETERMINISM_STRICT:
        case SIM_NEURAL_DETERMINISM_BEST_EFFORT:
        case SIM_NEURAL_DETERMINISM_OFF:
            break;
        default:
            info->neural.determinism_policy = defaults.neural.determinism_policy;
            break;
    }

    switch (info->neural.device_requirement) {
        case SIM_NEURAL_DEVICE_ANY:
        case SIM_NEURAL_DEVICE_CPU_ONLY:
        case SIM_NEURAL_DEVICE_ACCELERATOR_PREFERRED:
        case SIM_NEURAL_DEVICE_ACCELERATOR_REQUIRED:
            break;
        default:
            info->neural.device_requirement = defaults.neural.device_requirement;
            break;
    }

    switch (info->neural.precision_mode) {
        case SIM_NEURAL_PRECISION_DEFAULT:
        case SIM_NEURAL_PRECISION_FP32:
        case SIM_NEURAL_PRECISION_FP64:
        case SIM_NEURAL_PRECISION_MIXED:
        case SIM_NEURAL_PRECISION_FP16:
        case SIM_NEURAL_PRECISION_BF16:
            break;
        default:
            info->neural.precision_mode = defaults.neural.precision_mode;
            break;
    }

    if (info->neural.shape.min_rank == 0U) {
        info->neural.shape.min_rank = defaults.neural.shape.min_rank;
    }
    if (info->neural.shape.max_rank > 0U && info->neural.shape.max_rank < info->neural.shape.min_rank) {
        info->neural.shape.max_rank = info->neural.shape.min_rank;
    }
    if (info->neural.shape.channel_axis != SIM_NEURAL_CHANNEL_AXIS_AUTO &&
        info->neural.shape.channel_axis > 15U) {
        info->neural.shape.channel_axis = defaults.neural.shape.channel_axis;
    }
    if (info->neural.shape.max_channels > 0U &&
        info->neural.shape.max_channels < info->neural.shape.min_channels) {
        info->neural.shape.max_channels = info->neural.shape.min_channels;
    }

    /* Clamp invariant count and wipe trailing slots. */
    if (info->invariant_count > SIM_OPERATOR_MAX_INVARIANTS) {
        info->invariant_count = SIM_OPERATOR_MAX_INVARIANTS;
    }
    for (uint8_t i = info->invariant_count; i < SIM_OPERATOR_MAX_INVARIANTS; ++i) {
        info->invariants[i] = defaults.invariants[0];
    }
}

void sim_operator_info_set_identity(SimOperatorInfo* info,
                                    const char*      schema_key,
                                    SimIROpcode      ir_opcode) {
    if (info == NULL) {
        return;
    }
    info->schema_key = (schema_key != NULL && schema_key[0] != '\0') ? schema_key : NULL;
    if (sim_operator_ir_opcode_is_valid(ir_opcode)) {
        info->ir_opcode     = ir_opcode;
        info->has_ir_opcode = true;
    } else {
        info->ir_opcode     = OAK_OP_MISC;
        info->has_ir_opcode = false;
    }
}

void sim_operator_info_set_schema_identity(SimOperatorInfo* info, const char* schema_key) {
    if (info == NULL) {
        return;
    }
    sim_operator_info_set_identity(
        info, schema_key, sim_operator_ir_opcode_for_identity(info, schema_key));
}

const char* sim_operator_abstract_id(const SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    if (op->info.abstract_id_fn != NULL) {
        const char* id = op->info.abstract_id_fn(op);
        if (id != NULL) {
            return id;
        }
    }
    if (op->info.abstract_id != NULL) {
        return op->info.abstract_id;
    }
    return op->name;
}

const char* sim_operator_schema_key(const SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    if (op->schema_key[0] != '\0') {
        return op->schema_key;
    }
    if (op->info.schema_key != NULL && op->info.schema_key[0] != '\0') {
        return op->info.schema_key;
    }
    return NULL;
}

const char* sim_operator_schema_key_or(const SimOperator* op, const char* fallback) {
    const char* key = sim_operator_schema_key(op);
    return (key != NULL && key[0] != '\0') ? key : fallback;
}

SimIROpcode sim_operator_ir_opcode(const SimOperator* op) {
    if (op == NULL) {
        return OAK_OP_MISC;
    }
    if (op->info.has_ir_opcode && sim_operator_ir_opcode_is_valid(op->info.ir_opcode)) {
        return op->info.ir_opcode;
    }
    return sim_operator_ir_opcode_for_identity(&op->info, sim_operator_schema_key(op));
}

const SimOperatorRepresentation* sim_operator_representation(const SimOperator* op) {
    if (op == NULL) {
        return NULL;
    }
    return &op->info.representation;
}

const char* sim_operator_representation_domain_name(const SimOperator* op) {
    const SimOperatorRepresentation* repr = sim_operator_representation(op);
    return sim_field_domain_name(repr ? repr->domain : SIM_FIELD_DOMAIN_UNKNOWN);
}

const char* sim_operator_representation_value_kind_name(const SimOperator* op) {
    const SimOperatorRepresentation* repr = sim_operator_representation(op);
    return sim_field_value_kind_name(repr ? repr->value_kind : SIM_FIELD_VALUE_UNKNOWN);
}

const struct SimGraphIR* sim_operator_graph_ir(const SimOperator* op) {
    if (op == NULL || op->graph_ir_view == NULL) {
        return NULL;
    }
    return op->graph_ir_view(op, op->userdata);
}

SimOperatorInfo sim_operator_info(const SimOperator* op) {
    SimOperatorInfo default_info = sim_operator_info_defaults();

    if (op == NULL) {
        return default_info;
    }
    return op->info;
}
