/**
 * @file operator_split.c
 * @brief Implementation for declarative split operators with hazard-aware scheduling.
 */

#include "oakfield/operator_split.h"
#include "oakfield/sim_context.h"
#include "oakfield/field.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <errno.h>

#define SIM_SPLIT_SHARED_MAGIC 0x53504C54u       /* 'SPLT' */
#define SIM_SPLIT_OPERATOR_REF_MAGIC 0x53505246u /* 'SPRF' */

typedef struct SimSplitShared {
    uint32_t            magic;
    SimSplitDescriptor* desc;
    uint64_t*           substep_reads;
    uint64_t*           substep_writes;
    size_t              refcount;
    size_t              scratch_size;
    size_t              scratch_alignment;
    pthread_key_t       scratch_tls;
    bool                scratch_tls_ready;
    struct SimContext*  context;
    bool                scratch_accounted;
} SimSplitShared;

typedef struct SimSplitOperatorRef {
    uint32_t        magic;
    size_t          substep_index;
    SimSplitShared* shared;
} SimSplitOperatorRef;

static SimSplitOperatorRef* sim_split_operator_ref(void* userdata) {
    SimSplitOperatorRef* ref = (SimSplitOperatorRef*) userdata;
    if (ref == NULL || ref->magic != SIM_SPLIT_OPERATOR_REF_MAGIC) {
        return NULL;
    }
    return ref;
}

static SimSplitShared* sim_split_shared_from_userdata(void* userdata, size_t* out_substep_index) {
    if (out_substep_index != NULL) {
        *out_substep_index = 0U;
    }
    if (userdata == NULL) {
        return NULL;
    }

    SimSplitOperatorRef* ref = sim_split_operator_ref(userdata);
    if (ref != NULL) {
        if (out_substep_index != NULL) {
            *out_substep_index = ref->substep_index;
        }
        return ref->shared;
    }

    SimSplitShared* shared = (SimSplitShared*) userdata;
    if (shared->magic != SIM_SPLIT_SHARED_MAGIC) {
        return NULL;
    }
    return shared;
}

static void sim_split_release_scratch_budget(SimSplitShared* shared) {
    if (shared == NULL || !shared->scratch_accounted) {
        return;
    }
    if (shared->context != NULL && shared->scratch_size > 0U) {
        sim_context_release_scratch(shared->context, shared->scratch_size);
    }
    shared->scratch_accounted = false;
}

static void sim_split_tls_destructor(void* ptr) {
    free(ptr);
}

static bool sim_split_append_unique(size_t* array, size_t* count, size_t capacity, size_t value) {
    if (array == NULL || count == NULL || *count > capacity) {
        return false;
    }
    for (size_t i = 0U; i < *count; ++i) {
        if (array[i] == value) {
            return true;
        }
    }
    if (*count == capacity) {
        return false;
    }
    array[*count] = value;
    *count += 1U;
    return true;
}

static SimResult
sim_split_save_state(struct SimContext* ctx, struct SimOperator* self, void* userdata) {
    SimSplitShared* shared = sim_split_shared_from_userdata(userdata, NULL);
    if (shared == NULL || shared->desc == NULL || shared->desc->save_state == NULL) {
        return SIM_RESULT_OK;
    }
    return shared->desc->save_state(ctx, self, shared->desc->state);
}

static SimResult
sim_split_restore_state(struct SimContext* ctx, struct SimOperator* self, void* userdata) {
    SimSplitShared* shared = sim_split_shared_from_userdata(userdata, NULL);
    if (shared == NULL || shared->desc == NULL || shared->desc->restore_state == NULL) {
        return SIM_RESULT_OK;
    }
    return shared->desc->restore_state(ctx, self, shared->desc->state);
}

static void sim_split_free_descriptor(SimSplitDescriptor* copy) {
    if (copy == NULL) {
        return;
    }

    if (copy->substeps != NULL) {
        SimSplitSubstep* owned_substeps = (SimSplitSubstep*) copy->substeps;
        for (size_t i = 0U; i < copy->substep_count; ++i) {
            if (owned_substeps[i].accesses != NULL) {
                free((void*) owned_substeps[i].accesses);
                owned_substeps[i].accesses = NULL;
            }
        }
        free(owned_substeps);
        copy->substeps = NULL;
    }

    if (copy->ports != NULL) {
        free((void*) copy->ports);
        copy->ports = NULL;
    }

    free(copy);
}

static SimSplitDescriptor* sim_split_clone_descriptor(const SimSplitDescriptor* desc) {
    SimSplitDescriptor* copy;

    if (desc == NULL) {
        return NULL;
    }

    copy = (SimSplitDescriptor*) calloc(1U, sizeof(SimSplitDescriptor));
    if (copy == NULL) {
        return NULL;
    }

    *copy               = *desc;
    copy->save_state    = desc->save_state;
    copy->restore_state = desc->restore_state;
    copy->ports         = NULL;
    copy->substeps      = NULL;

    if (desc->port_count > 0U && desc->ports != NULL) {
        SimSplitPort* ports_copy = (SimSplitPort*) calloc(desc->port_count, sizeof(SimSplitPort));
        if (ports_copy == NULL) {
            free(copy);
            return NULL;
        }
        memcpy(ports_copy, desc->ports, desc->port_count * sizeof(SimSplitPort));
        copy->ports = ports_copy;
    } else {
        copy->port_count = 0U;
    }

    if (desc->substep_count > 0U && desc->substeps != NULL) {
        SimSplitSubstep* substeps_copy =
            (SimSplitSubstep*) calloc(desc->substep_count, sizeof(SimSplitSubstep));
        if (substeps_copy == NULL) {
            if (copy->ports != NULL) {
                free((void*) copy->ports);
                copy->ports = NULL;
            }
            free(copy);
            return NULL;
        }

        for (size_t i = 0U; i < desc->substep_count; ++i) {
            const SimSplitSubstep* orig = &desc->substeps[i];
            substeps_copy[i]            = *orig;
            substeps_copy[i].accesses   = NULL;
            if (orig->access_count > 0U && orig->accesses != NULL) {
                SimSplitAccess* accesses_copy =
                    (SimSplitAccess*) calloc(orig->access_count, sizeof(SimSplitAccess));
                if (accesses_copy == NULL) {
                    /* free any prior allocations before returning */
                    for (size_t j = 0U; j < i; ++j) {
                        free((void*) substeps_copy[j].accesses);
                        substeps_copy[j].accesses = NULL;
                    }
                    free(substeps_copy);
                    if (copy->ports != NULL) {
                        free((void*) copy->ports);
                        copy->ports = NULL;
                    }
                    free(copy);
                    return NULL;
                }
                memcpy(accesses_copy, orig->accesses, orig->access_count * sizeof(SimSplitAccess));
                substeps_copy[i].accesses     = accesses_copy;
                substeps_copy[i].access_count = orig->access_count;
            } else {
                substeps_copy[i].access_count = 0U;
            }
        }

        copy->substeps = substeps_copy;
    } else {
        copy->substep_count = 0U;
    }

    return copy;
}

static void* sim_split_acquire_scratch(SimSplitShared* shared) {
    if (shared == NULL || shared->scratch_size == 0U) {
        return NULL;
    }

    if (!shared->scratch_tls_ready) {
        if (pthread_key_create(&shared->scratch_tls, sim_split_tls_destructor) != 0) {
            return NULL;
        }
        shared->scratch_tls_ready = true;
    }

    void* buf = pthread_getspecific(shared->scratch_tls);
    if (buf == NULL) {
        size_t alignment = (shared->scratch_alignment > 0U) ? shared->scratch_alignment : 64U;
        if (alignment < sizeof(void*)) {
            alignment = sizeof(void*);
        }
        void* allocated = NULL;
#if defined(_POSIX_VERSION)
        if (posix_memalign(&allocated, alignment, shared->scratch_size) != 0) {
            return NULL;
        }
#else
        allocated = aligned_alloc(alignment, shared->scratch_size);
        if (allocated == NULL) {
            return NULL;
        }
#endif
        (void) pthread_setspecific(shared->scratch_tls, allocated);
        buf = allocated;
    }

    return buf;
}

static SimResult sim_split_eval(struct SimContext* ctx, struct SimOperator* self, void* userdata) {
    size_t          substep_index = 0U;
    SimSplitShared* shared        = sim_split_shared_from_userdata(userdata, &substep_index);
    double          base_dt;
    double          dt_sub;
    SimResult       r;

    if (ctx == NULL || self == NULL || shared == NULL || shared->desc == NULL) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }
    if (substep_index >= shared->desc->substep_count) {
        return SIM_RESULT_INVALID_STATE;
    }

    const SimSplitSubstep* sub = &shared->desc->substeps[substep_index];

    base_dt = sim_context_timestep(ctx);

    if (base_dt <= 0.0) {
        base_dt = 1.0e-12;
    }
    dt_sub = base_dt * ((sub->dt_scale > 0.0) ? sub->dt_scale
                                              : 1.0); /* per-substep dt forwarded to integrator */

    void* scratch = sim_split_acquire_scratch(shared);

    r = sub->fn(
        shared->desc->state, ctx, self, substep_index, dt_sub, scratch, shared->scratch_size);

    if (shared->desc->state != NULL && sub->error_measure != NULL) {
        double err = sub->error_measure(shared->desc->state);
        sim_split_notify_integrator(ctx, dt_sub, err);
    }

    return r;
}

static void sim_split_destroy(void* userdata) {
    SimSplitOperatorRef* ref    = sim_split_operator_ref(userdata);
    SimSplitShared*      shared = sim_split_shared_from_userdata(userdata, NULL);
    if (ref != NULL) {
        ref->magic  = 0U;
        ref->shared = NULL;
        free(ref);
    }
    if (shared == NULL || shared->magic != SIM_SPLIT_SHARED_MAGIC) {
        return;
    }

    if (shared->refcount > 0U) {
        shared->refcount -= 1U;
    }
    if (shared->refcount > 0U) {
        return;
    }

    if (shared->desc != NULL && shared->desc->destroy != NULL) {
        shared->desc->destroy(shared->desc->state);
    }

    sim_split_release_scratch_budget(shared);
    sim_split_free_descriptor(shared->desc);
    shared->desc = NULL;

    free(shared->substep_reads);
    free(shared->substep_writes);

    if (shared->scratch_tls_ready) {
        /* Clean up the current thread’s scratch eagerly; TLS destructor handles other threads. */
        void* buf = pthread_getspecific(shared->scratch_tls);
        if (buf != NULL) {
            free(buf);
            (void) pthread_setspecific(shared->scratch_tls, NULL);
        }
        (void) pthread_key_delete(shared->scratch_tls);
    }

    shared->magic = 0U;
    free(shared);
}

SimResult sim_split_register(struct SimContext*        context,
                             const SimSplitDescriptor* desc,
                             const size_t*             dependencies,
                             size_t                    dependency_count,
                             size_t*                   out_first,
                             size_t*                   out_last) {
    size_t          i;
    size_t          previous_index = (size_t) (-1);
    SimResult       r;
    SimSplitShared* shared;

    if (context == NULL || desc == NULL || desc->name == NULL || desc->substeps == NULL ||
        desc->substep_count == 0U) {
        return SIM_RESULT_INVALID_ARGUMENT;
    }

    shared = (SimSplitShared*) calloc(1U, sizeof(SimSplitShared));
    if (shared == NULL) {
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    shared->magic = SIM_SPLIT_SHARED_MAGIC;

    shared->desc = sim_split_clone_descriptor(desc);
    if (shared->desc == NULL) {
        free(shared);
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    const SimSplitDescriptor* desc_copy = shared->desc;

    shared->refcount          = desc_copy->substep_count;
    shared->scratch_size      = desc_copy->scratch.bytes_per_worker;
    shared->scratch_alignment = desc_copy->scratch.alignment;
    shared->context           = context;
    shared->scratch_accounted = false;
    if (shared->scratch_size > 0U && context->memory_limits.max_scratch_bytes_per_operator > 0U &&
        shared->scratch_size > context->memory_limits.max_scratch_bytes_per_operator) {
        sim_split_free_descriptor(shared->desc);
        free(shared);
        return SIM_RESULT_OUT_OF_MEMORY;
    }
    if (shared->scratch_size > 0U) {
        SimResult scratch_result = sim_context_reserve_scratch(context, shared->scratch_size);
        if (scratch_result != SIM_RESULT_OK) {
            sim_split_free_descriptor(shared->desc);
            free(shared);
            return scratch_result;
        }
        shared->scratch_accounted = true;
    }
    if (desc_copy->substep_count > 0U) {
        shared->substep_reads  = (uint64_t*) calloc(desc_copy->substep_count, sizeof(uint64_t));
        shared->substep_writes = (uint64_t*) calloc(desc_copy->substep_count, sizeof(uint64_t));
        if (shared->substep_reads == NULL || shared->substep_writes == NULL) {
            free(shared->substep_reads);
            free(shared->substep_writes);
            sim_split_release_scratch_budget(shared);
            sim_split_free_descriptor(shared->desc);
            free(shared);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
    }

    /* Port binding invariants:
     *   - Field indices must be valid in the context.
     *   - Representation requirements are enforced here (no implicit promotion).
     */
    for (i = 0U; i < desc_copy->port_count; ++i) {
        size_t    field_index  = desc_copy->ports[i].context_field_index;
        bool      need_complex = desc_copy->ports[i].require_complex;
        SimField* f            = sim_context_field(context, field_index);
        if (f == NULL) {
            free(shared->substep_reads);
            free(shared->substep_writes);
            sim_split_release_scratch_budget(shared);
            sim_split_free_descriptor(shared->desc);
            free(shared);
            return SIM_RESULT_INVALID_ARGUMENT;
        }
        if (need_complex) {
            if (!sim_field_is_complex(f)) {
                free(shared->substep_reads);
                free(shared->substep_writes);
                sim_split_release_scratch_budget(shared);
                sim_split_free_descriptor(shared->desc);
                free(shared);
                return SIM_RESULT_TYPE_MISMATCH;
            }
        }
    }

    for (i = 0U; i < desc_copy->substep_count; ++i) {
        char                   name_buf[SIM_OPERATOR_NAME_MAX + 1U];
        SimOperatorDescriptor  d;
        SimSplitOperatorRef*   ref = NULL;
        size_t                 dep_array[4];
        size_t                 dep_count = 0U;
        uint64_t               rmask = 0ULL, wmask = 0ULL;
        size_t*                read_fields  = NULL;
        size_t*                write_fields = NULL;
        size_t                 read_count   = 0U;
        size_t                 write_count  = 0U;
        const SimSplitSubstep* ss           = &desc_copy->substeps[i];

        (void) memset(&d, 0, sizeof(d));

        if (desc_copy->port_count > 0U) {
            read_fields  = (size_t*) calloc(desc_copy->port_count, sizeof(size_t));
            write_fields = (size_t*) calloc(desc_copy->port_count, sizeof(size_t));
            if ((read_fields == NULL && desc_copy->port_count > 0U) ||
                (write_fields == NULL && desc_copy->port_count > 0U)) {
                free(read_fields);
                free(write_fields);
                free(shared->substep_reads);
                free(shared->substep_writes);
                sim_split_release_scratch_budget(shared);
                sim_split_free_descriptor(shared->desc);
                free(shared);
                return SIM_RESULT_OUT_OF_MEMORY;
            }
        }

        for (size_t a = 0U; a < ss->access_count; ++a) {
            size_t port = ss->accesses[a].port;
            if (port >= desc_copy->port_count) {
                free(read_fields);
                free(write_fields);
                free(shared->substep_reads);
                free(shared->substep_writes);
                sim_split_release_scratch_budget(shared);
                sim_split_free_descriptor(shared->desc);
                free(shared);
                return SIM_RESULT_INVALID_ARGUMENT;
            }
            size_t field_index = desc_copy->ports[port].context_field_index;
            if (field_index < 64U) {
                if (ss->accesses[a].mode == SIM_ACCESS_READ) {
                    rmask |= (1ULL << field_index);
                } else if (ss->accesses[a].mode == SIM_ACCESS_WRITE) {
                    wmask |= (1ULL << field_index);
                } else {
                    rmask |= (1ULL << field_index);
                    wmask |= (1ULL << field_index);
                }
            }
            if (ss->accesses[a].mode == SIM_ACCESS_READ || ss->accesses[a].mode == SIM_ACCESS_RW) {
                if (!sim_split_append_unique(
                        read_fields, &read_count, desc_copy->port_count, field_index)) {
                    free(read_fields);
                    free(write_fields);
                    free(shared->substep_reads);
                    free(shared->substep_writes);
                    sim_split_release_scratch_budget(shared);
                    sim_split_free_descriptor(shared->desc);
                    free(shared);
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
            }
            if (ss->accesses[a].mode == SIM_ACCESS_WRITE || ss->accesses[a].mode == SIM_ACCESS_RW) {
                if (!sim_split_append_unique(
                        write_fields, &write_count, desc_copy->port_count, field_index)) {
                    free(read_fields);
                    free(write_fields);
                    free(shared->substep_reads);
                    free(shared->substep_writes);
                    sim_split_release_scratch_budget(shared);
                    sim_split_free_descriptor(shared->desc);
                    free(shared);
                    return SIM_RESULT_OUT_OF_MEMORY;
                }
            }
        }

        /* Optional barrier: treat all bound ports as both read and written to block reordering. */
        if (ss->barrier_after) {
            for (size_t p = 0U; p < desc_copy->port_count; ++p) {
                size_t field_index = desc_copy->ports[p].context_field_index;
                if (field_index < 64U) {
                    rmask |= (1ULL << field_index);
                    wmask |= (1ULL << field_index);
                }
                (void) sim_split_append_unique(
                    read_fields, &read_count, desc_copy->port_count, field_index);
                (void) sim_split_append_unique(
                    write_fields, &write_count, desc_copy->port_count, field_index);
            }
        }
        shared->substep_reads[i]  = rmask;
        shared->substep_writes[i] = wmask;

        (void) memset(&d, 0, sizeof(d));
        (void) snprintf(name_buf, sizeof(name_buf), "%s#%zu", desc->name, i);
        d.name     = name_buf;
        d.evaluate = sim_split_eval;
        d.destroy  = sim_split_destroy;
        ref        = (SimSplitOperatorRef*) calloc(1U, sizeof(*ref));
        if (ref == NULL) {
            free(read_fields);
            free(write_fields);
            free(shared->substep_reads);
            free(shared->substep_writes);
            sim_split_release_scratch_budget(shared);
            sim_split_free_descriptor(shared->desc);
            free(shared);
            return SIM_RESULT_OUT_OF_MEMORY;
        }
        ref->magic         = SIM_SPLIT_OPERATOR_REF_MAGIC;
        ref->substep_index = i;
        ref->shared        = shared;
        d.userdata         = ref;
        d.info             = desc_copy->info;
        if (d.info.abstract_id == NULL) {
            d.info.abstract_id = desc_copy->name;
        }
        if (d.info.abstract_id_fn == NULL && desc_copy->symbolic != NULL) {
            d.info.abstract_id_fn = sim_split_symbolic;
        }
        d.config            = desc_copy->config;
        d.read_mask         = rmask;
        d.write_mask        = wmask;
        d.read_indices      = (read_count > 0U) ? read_fields : NULL;
        d.read_index_count  = read_count;
        d.write_indices     = (write_count > 0U) ? write_fields : NULL;
        d.write_index_count = write_count;
        d.required_features = ss->required_features;
        d.catalog_metadata  = desc_copy->catalog_metadata;
        d.save_state        = (desc_copy->save_state != NULL) ? sim_split_save_state : NULL;
        d.restore_state     = (desc_copy->restore_state != NULL) ? sim_split_restore_state : NULL;

        if (i == 0U && dependency_count > 0U && dependencies != NULL) {
            d.dependencies     = dependencies;
            d.dependency_count = dependency_count;
        } else if (i > 0U) {
            dep_array[dep_count++] = previous_index;
            d.dependencies         = dep_array;
            d.dependency_count     = dep_count;
        }

        r = sim_context_register_operator(context, &d, &previous_index);
        free(read_fields);
        free(write_fields);
        if (r != SIM_RESULT_OK) {
            free(ref);
            free(shared->substep_reads);
            free(shared->substep_writes);
            sim_split_release_scratch_budget(shared);
            sim_split_free_descriptor(shared->desc);
            free(shared);
            return r;
        }
        if (out_first != NULL && i == 0U) {
            *out_first = previous_index;
        }
    }

    if (out_last != NULL) {
        *out_last = previous_index;
    }
    return SIM_RESULT_OK;
}

const char* sim_split_symbolic(const struct SimOperator* op) {
#if !OAKFIELD_ENABLE_SYMBOLIC_KERNELS
    (void) op;
    return NULL;
#endif
    if (op == NULL || op->userdata == NULL) {
        return NULL;
    }
    SimSplitShared* shared = sim_split_shared_from_userdata(op->userdata, NULL);
    if (shared == NULL || shared->desc == NULL || shared->desc->symbolic == NULL) {
        return NULL;
    }
    return shared->desc->symbolic(shared->desc->state);
}

void* sim_split_state(struct SimOperator* op) {
    if (op == NULL || op->userdata == NULL) {
        return NULL;
    }
    SimSplitShared* shared = sim_split_shared_from_userdata(op->userdata, NULL);
    if (shared == NULL || shared->desc == NULL) {
        return NULL;
    }
    return shared->desc->state;
}

void* sim_operator_state(struct SimOperator* op) {
    void* split_state = sim_split_state(op);
    if (split_state != NULL) {
        return split_state;
    }
    return sim_operator_payload(op);
}
