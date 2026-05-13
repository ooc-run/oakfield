/**
 * @file operator_split.h
 * @brief Declarative complex-first operator splitting (substep expansion at plan time).
 *
 * A split descriptor registers a sequence of ordinary SimOperator instances
 * that share one user state pointer. Access declarations become read/write
 * hazards for the scheduler, optional scratch is reserved per split operator,
 * and later substeps depend on earlier substeps to preserve sequence order.
 */
#ifndef OAKFIELD_OPERATOR_SPLIT_H
#define OAKFIELD_OPERATOR_SPLIT_H

#include "operator.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;
struct SimOperator;

/**
 * @brief Field access mode declared by a split-operator substep.
 */
typedef enum SimAccessMode {
    SIM_ACCESS_READ = 0,  /**< Substep reads the bound port field. */
    SIM_ACCESS_WRITE = 1, /**< Substep writes the bound port field. */
    SIM_ACCESS_RW = 2     /**< Substep both reads and writes the bound port field. */
} SimAccessMode;

/**
 * @brief Per-substep access to a descriptor port.
 */
typedef struct SimSplitAccess {
    uint16_t port; /**< Index into SimSplitDescriptor::ports. */
    uint8_t mode;  /**< SimAccessMode value. */
} SimSplitAccess;

/**
 * @brief One executable step inside a split operator sequence.
 */
typedef struct SimSplitSubstep {
    const char *name; /**< Optional label for tooling. */
    /**
     * @brief Execute the substep.
     *
     * @param state Shared descriptor state pointer.
     * @param ctx Active simulation context.
     * @param container_or_self Operator wrapper for the current substep.
     * @param substep_index Zero-based substep index.
     * @param dt_sub Context timestep scaled by dt_scale.
     * @param scratch Optional per-thread scratch buffer, or NULL.
     * @param scratch_size Size of @p scratch in bytes.
     * @return #SIM_RESULT_OK or an error propagated to context execution.
     */
    SimResult (*fn)(void *state, struct SimContext *ctx, struct SimOperator *container_or_self,
                    size_t substep_index, double dt_sub, void *scratch,
                    size_t scratch_size); /**< Execute the substep callback. */
    const SimSplitAccess *accesses;       /**< Per-substep access declarations. */
    size_t access_count;                  /**< Number of entries in accesses. */
    double dt_scale;    /**< Positive scale applied to context dt; otherwise 1.0. */
    bool barrier_after; /**< Treat all ports as read/write for future ordering. */
    double (*error_measure)(void *state); /**< Optional error metric reported to the integrator. */
    uint64_t required_features;           /**< Backend features required for this substep. */
} SimSplitSubstep;

/**
 * @brief Binding between a split port and a context field.
 */
typedef struct SimSplitPort {
    size_t context_field_index; /**< Bound context field index. */
    bool require_complex;       /**< Require complex field storage at registration time. */
} SimSplitPort;

/**
 * @brief Optional per-thread scratch request for split substeps.
 */
typedef struct SimSplitScratchRequest {
    size_t bytes_per_worker; /**< Scratch bytes per worker/thread; zero disables scratch. */
    size_t alignment;        /**< Alignment requirement; defaults to at least 64 when zero. */
} SimSplitScratchRequest;

/**
 * @brief Full declarative description of a split operator sequence.
 *
 * The descriptor, port array, substep array, and access arrays are cloned during
 * registration, but the state pointer remains caller-provided shared state. When
 * the registered substep operators are destroyed, the optional destroy callback
 * receives that shared state once.
 */
typedef struct SimSplitDescriptor {
    const char *name;                /**< Container name; must be non-NULL. */
    const SimSplitPort *ports;       /**< Bound ports, copied during registration. */
    size_t port_count;               /**< Number of entries in ports. */
    const SimSplitSubstep *substeps; /**< Substeps, copied during registration. */
    size_t substep_count;            /**< Number of substeps; must be nonzero. */
    void *state;                     /**< Shared user state passed to substeps and hooks. */
    const char *(*symbolic)(const void *state); /**< Optional symbolic form callback. */
    SimResult (*save_state)(struct SimContext *ctx, struct SimOperator *self,
                            void *userdata); /**< Optional snapshot hook. */
    SimResult (*restore_state)(struct SimContext *ctx, struct SimOperator *self,
                               void *userdata); /**< Optional restore hook. */
    void (*destroy)(void *state);               /**< Optional shared-state teardown. */
    SimOperatorInfo info;           /**< Metadata copied onto each generated operator. */
    SimOperatorConfig config;       /**< Configuration copied onto each generated operator. */
    const void *catalog_metadata;   /**< Optional catalog/runtime metadata. */
    SimSplitScratchRequest scratch; /**< Optional scratch requirements. */
} SimSplitDescriptor;

/**
 * @brief Register a split operator which expands into sequential substeps.
 *
 * Registers one operator per substep. The first substep depends on the provided
 * dependencies; each subsequent substep depends on the previous substep. Field
 * access declarations are converted into scheduler read/write masks and index
 * arrays. Complex requirements are checked before registration completes.
 *
 * @param context Target simulation context.
 * @param desc Split operator description; arrays are copied, state is not copied.
 * @param dependencies Optional dependency list for the first substep.
 * @param dependency_count Length of @p dependencies.
 * @param[out] out_first Optional receiver for the first generated operator index.
 * @param[out] out_last Optional receiver for the final generated operator index.
 * @return #SIM_RESULT_OK, #SIM_RESULT_INVALID_ARGUMENT, #SIM_RESULT_TYPE_MISMATCH,
 *         #SIM_RESULT_OUT_OF_MEMORY, or a result propagated by operator registration.
 */
SimResult sim_split_register(struct SimContext *context, const SimSplitDescriptor *desc,
                             const size_t *dependencies, size_t dependency_count, size_t *out_first,
                             size_t *out_last);

/**
 * @brief Return the symbolic form string for any generated substep operator.
 *
 * @param op Generated split operator.
 * @return Symbolic string from the descriptor callback, or NULL when unavailable.
 */
const char *sim_split_symbolic(const struct SimOperator *op);

/**
 * @brief Return the shared descriptor state for any generated substep operator.
 *
 * @param op Generated split operator.
 * @return Shared state pointer, or NULL when @p op is not a split operator.
 */
void *sim_split_state(struct SimOperator *op);

/**
 * @brief Return operator state regardless of split or kernel-backed registration.
 *
 * @param op Operator to inspect.
 * @return Split shared state when present, otherwise sim_operator_payload(op).
 */
void *sim_operator_state(struct SimOperator *op);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_OPERATOR_SPLIT_H */
