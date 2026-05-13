/**
 * @file operator.h
 * @brief Operator abstraction, registry, and dependency resolution for libsimcore.
 * @ingroup oakfield_operators
 *
 * @details Operators are registered into a SimContext with explicit read/write
 * field access, optional dependencies, metadata, and configuration. The
 * scheduler uses those declarations to build deterministic execution order,
 * while operator descriptors keep ownership boundaries clear: descriptors and
 * arrays are copied as needed, but user payload lifetime remains the caller's
 * responsibility unless a concrete operator documents otherwise.
 */
#ifndef OAKFIELD_OPERATOR_H
#define OAKFIELD_OPERATOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef OAKFIELD_ENABLE_SYMBOLIC_KERNELS
#define OAKFIELD_ENABLE_SYMBOLIC_KERNELS 1
#endif

#ifndef OAKFIELD_ENABLE_ZETA_CORE
#define OAKFIELD_ENABLE_ZETA_CORE 1
#endif

#include "field.h"
#include "oakfield/backend.h"
#include "operator_identity.h"

struct SimOperatorKernel;
struct SimGraphIR;

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;
struct SimOperator;
struct SimOperatorConfig;
struct SimOperatorConfigAdapter;

/**
 * @brief Callback used to report dependency-cycle diagnostics.
 */
typedef void (*SimOperatorCycleLogFn)(const char* message, void* userdata);

/** Maximum length for operator identifiers (excluding null terminator). */
#define SIM_OPERATOR_NAME_MAX 63U
/** Maximum length for stable operator schema keys (excluding null terminator). */
#define SIM_OPERATOR_SCHEMA_KEY_MAX 127U

/**
 * @brief Operator category for grouping and scheduling hints.
 */
typedef enum SimOperatorCategory {
    SIM_OPERATOR_CATEGORY_UNKNOWN = 0, /**< Uncategorized operator. */
    SIM_OPERATOR_CATEGORY_DIFFUSION,   /**< Diffusion, dissipation, linear damping. */
    SIM_OPERATOR_CATEGORY_ADVECTION,   /**< Transport, advection, warp. */
    SIM_OPERATOR_CATEGORY_REACTION,    /**< Nonlinear reaction terms (e.g., cubic). */
    SIM_OPERATOR_CATEGORY_POTENTIAL,   /**< External potential application. */
    SIM_OPERATOR_CATEGORY_MEASUREMENT, /**< Measurement, projection, collapse. */
    SIM_OPERATOR_CATEGORY_COUPLING,    /**< Field coupling, sieve operators. */
    SIM_OPERATOR_CATEGORY_NOISE,       /**< Stochastic noise injection. */
    SIM_OPERATOR_CATEGORY_BOUNDARY,    /**< Boundary condition enforcement. */
    SIM_OPERATOR_CATEGORY_THERMOSTAT,  /**< Thermostat regulation operators. */
    SIM_OPERATOR_CATEGORY_UTILITY,     /**< Utility operators (copy, scale, etc.). */
    SIM_OPERATOR_CATEGORY_NONLINEAR    /**< General nonlinear operators. */
} SimOperatorCategory;

/**
 * @brief Numerical continuity and stability policy.
 */
typedef enum SimContinuityMode {
    SIM_CONTINUITY_NONE    = 0, /**< Disable all guards; raw kernel output. */
    SIM_CONTINUITY_STRICT  = 1, /**< Preserve analytic continuity; may emit NaN/Inf. */
    SIM_CONTINUITY_CLAMPED = 2, /**< Clip to finite range near singularities. */
    SIM_CONTINUITY_LIMITED = 3  /**< Blend asymptotic/analytic forms near threshold. */
} SimContinuityMode;

const char* sim_continuity_mode_name(SimContinuityMode mode);
bool        sim_continuity_mode_from_string(const char* text, SimContinuityMode* out_mode);

/**
 * @brief Limiter/clipper strategies engaged by continuity guards.
 */
typedef enum SimContinuityLimiterStrategy {
    SIM_LIMITER_STRATEGY_HARD_CLIP = 1u << 0, /**< Clamp values to configured hard bounds. */
    SIM_LIMITER_STRATEGY_SOFT_CLIP = 1u << 1, /**< Smoothly compress values near bounds. */
    SIM_LIMITER_STRATEGY_SIGMOID_COMPRESSION =
        1u << 2, /**< Use sigmoid compression for singular neighborhoods. */
    SIM_LIMITER_STRATEGY_ADAPTIVE_BIAS =
        1u << 3, /**< Shift probe bias adaptively near singularities. */
    SIM_LIMITER_STRATEGY_STAT_AWARE =
        1u << 4, /**< Incorporate field statistics into limiter decisions. */
    SIM_LIMITER_STRATEGY_PER_FIELD_OVERRIDE = 1u << 5 /**< Permit per-field limiter overrides. */
} SimContinuityLimiterStrategy;

const char* sim_continuity_limiter_name(SimContinuityLimiterStrategy strategy);
const char* sim_boundary_policy_name(SimIRBoundaryPolicy policy);
bool        sim_boundary_policy_from_string(const char* text, SimIRBoundaryPolicy* out_policy);
void        sim_operator_config_set_spacing(struct SimOperatorConfig* config,
                                            const double*             spacing,
                                            size_t                    rank);

/** Maximum dimensions tracked for spacing metadata. */
#define SIM_OPERATOR_MAX_SPACING_DIMS 4U

/**
 * @brief Algebraic property bitmask for operators.
 */
typedef enum SimOperatorAlgebraicFlags {
    SIM_OPERATOR_ALG_NONE                = 0u,      /**< No algebraic properties declared. */
    SIM_OPERATOR_ALG_LINEAR              = 1u << 0, /**< Purely linear mapping. */
    SIM_OPERATOR_ALG_AFFINE              = 1u << 1, /**< Affine (linear + bias). */
    SIM_OPERATOR_ALG_SELF_ADJOINT        = 1u << 2, /**< Self-adjoint / symmetric. */
    SIM_OPERATOR_ALG_PROJECTION          = 1u << 3, /**< Idempotent projection. */
    SIM_OPERATOR_ALG_COMMUTES_WITH_NOISE = 1u
                                           << 4, /**< Commutes with stochastic noise injection. */
    SIM_OPERATOR_ALG_COMMUTES_WITH_BOUNDARY = 1u << 5 /**< Commutes with boundary enforcement. */
} SimOperatorAlgebraicFlags;

/**
 * @brief Representation metadata attached to an operator.
 */
typedef struct SimOperatorRepresentation {
    SimFieldDomain    domain;     /**< Preferred domain. */
    SimFieldValueKind value_kind; /**< Output value kind classification. */
    bool requires_complex_input;  /**< True if inputs must be at least complex-valued. */
    bool preserves_real_subspace; /**< True if real input stays in the real subspace. */
    bool
        requires_complex_representation; /**< True if storage must be complex even for real math. */
    SimIRBoundaryPolicy boundary;        /**< Boundary policy hint for spatial operators. */
    uint8_t             spacing_hint_rank; /**< Optional spacing hint rank. */
    double
        spacing_hint[SIM_OPERATOR_MAX_SPACING_DIMS]; /**< Optional per-dimension spacing hints. */
} SimOperatorRepresentation;

/**
 * @brief Approximation metadata describing discretization quality.
 */
typedef struct SimOperatorApproximation {
    double spatial_order; /**< Spatial order of accuracy (0 when unspecified). */
    double stencil_order; /**< Finite-difference stencil order p (0 when unspecified). */
    double
        error_constant; /**< Consistency constant C_p for truncation error (0 when unspecified). */
    double temporal_order; /**< Temporal integration order (0 when unspecified). */
} SimOperatorApproximation;

/**
 * @brief Representation mode governing determinism guarantees.
 *
 * STRICT: deterministic, snapshot-friendly, IR/CPU parity, rewind-safe.
 * RELAXED: mostly deterministic; permits limited conveniences.
 * EXPLORATION: best-effort; stateful or non-replayable behavior permitted.
 */
typedef enum SimRepresentationMode {
    SIM_REPRESENTATION_MODE_STRICT = 0, /**< Deterministic, replayable representation mode. */
    SIM_REPRESENTATION_MODE_RELAXED,    /**< Mostly deterministic representation mode. */
    SIM_REPRESENTATION_MODE_EXPLORATION /**< Best-effort exploratory representation mode. */
} SimRepresentationMode;

/**
 * @brief Determinism capability flags declared by operators.
 */
typedef enum SimDeterminismFlags {
    SIM_DET_NONE                   = 0U,      /**< No determinism guarantees declared. */
    SIM_DET_PURE_TIME              = 1U << 0, /**< Time is derived purely from runtime params. */
    SIM_DET_REWIND_SAFE            = 1U << 1, /**< Operator behavior is rewind-safe. */
    SIM_DET_NO_STATEFUL_NODES      = 1U << 2, /**< Operator does not rely on stateful nodes. */
    SIM_DET_DETERMINISTIC_RNG_ONLY = 1U << 3  /**< RNG usage is deterministic and seed-driven. */
} SimDeterminismFlags;

/**
 * @brief Determinism policy for neural inference operators.
 */
typedef enum SimNeuralDeterminismPolicy {
    SIM_NEURAL_DETERMINISM_INHERIT = 0, /**< Defer to context/operator determinism mode. */
    SIM_NEURAL_DETERMINISM_STRICT,      /**< Require strict deterministic inference path. */
    SIM_NEURAL_DETERMINISM_BEST_EFFORT, /**< Prefer deterministic path but allow fallback. */
    SIM_NEURAL_DETERMINISM_OFF          /**< Determinism not required for inference. */
} SimNeuralDeterminismPolicy;

/**
 * @brief Device placement requirement for neural inference operators.
 */
typedef enum SimNeuralDeviceRequirement {
    SIM_NEURAL_DEVICE_ANY = 0,               /**< Any available device is acceptable. */
    SIM_NEURAL_DEVICE_CPU_ONLY,              /**< Must run on CPU. */
    SIM_NEURAL_DEVICE_ACCELERATOR_PREFERRED, /**< Prefer accelerator, fallback to CPU. */
    SIM_NEURAL_DEVICE_ACCELERATOR_REQUIRED   /**< Require accelerator device. */
} SimNeuralDeviceRequirement;

/**
 * @brief Numeric precision policy for neural inference operators.
 */
typedef enum SimNeuralPrecisionMode {
    SIM_NEURAL_PRECISION_DEFAULT = 0, /**< Runtime/provider default precision. */
    SIM_NEURAL_PRECISION_FP32,        /**< 32-bit floating-point inference. */
    SIM_NEURAL_PRECISION_FP64,        /**< 64-bit floating-point inference. */
    SIM_NEURAL_PRECISION_MIXED,       /**< Backend-selected mixed precision. */
    SIM_NEURAL_PRECISION_FP16,        /**< 16-bit floating-point inference. */
    SIM_NEURAL_PRECISION_BF16         /**< bfloat16 inference. */
} SimNeuralPrecisionMode;

/** Automatic channel-axis selection for neural shape constraints. */
#define SIM_NEURAL_CHANNEL_AXIS_AUTO 255U

/**
 * @brief Shape and channel constraints used by neural operators.
 */
typedef struct SimNeuralShapeConstraints {
    uint8_t  min_rank;      /**< Minimum accepted field rank (0 treated as 1). */
    uint8_t  max_rank;      /**< Maximum accepted field rank (0 disables upper bound). */
    uint8_t  channel_axis;  /**< Explicit channel axis or @ref SIM_NEURAL_CHANNEL_AXIS_AUTO. */
    uint32_t min_channels;  /**< Minimum accepted channels on selected axis (0 disables). */
    uint32_t max_channels;  /**< Maximum accepted channels on selected axis (0 disables). */
    bool     channels_last; /**< Auto-axis policy: use last axis when true, axis 0 otherwise. */
    bool     allow_complex_input; /**< Whether complex-valued fields are accepted. */
} SimNeuralShapeConstraints;

/**
 * @brief Neural operator contract metadata.
 */
typedef struct SimOperatorNeuralContract {
    bool                       enabled; /**< True when operator includes neural inference path. */
    SimNeuralDeterminismPolicy determinism_policy; /**< Inference determinism requirement. */
    SimNeuralDeviceRequirement device_requirement; /**< Device placement policy. */
    SimNeuralPrecisionMode     precision_mode;     /**< Precision policy for inference. */
    SimNeuralShapeConstraints  shape;              /**< Rank/channel constraints. */
} SimOperatorNeuralContract;

/**
 * @brief Clock selection for time-sensitive operators.
 *
 * Pure deterministic time: t = f(params) (rewind-safe, replayable, KernelIR-friendly).
 * True accumulated time: t_{n+1} = t_n + dt_n (requires state, rewind policy-dependent).
 *
 * CLOCK_FROM_TIME_PARAM: t = time param + offset (pure).
 * CLOCK_FROM_STEP_PURE: t = step_index * nominal_dt + offset (pure).
 * CLOCK_ACCUMULATED_STATEFUL: t accumulates with state (non-replayable).
 */
typedef enum SimClockMode {
    SIM_CLOCK_FROM_TIME_PARAM = 0, /**< Derive time from runtime time parameter. */
    SIM_CLOCK_FROM_STEP_PURE,      /**< Derive time from step index and nominal dt. */
    SIM_CLOCK_ACCUMULATED_STATEFUL /**< Accumulate time inside operator state. */
} SimClockMode;

/** Maximum invariants tracked inline per operator. */
#define SIM_OPERATOR_MAX_INVARIANTS 4U

/**
 * @brief Supported invariant kinds for bookkeeping/diagnostics.
 */
typedef enum SimOperatorInvariantKind {
    SIM_OPERATOR_INVARIANT_NONE = 0, /**< No invariant declared. */
    SIM_OPERATOR_INVARIANT_L2_NORM,  /**< L2 norm should remain within tolerance. */
    SIM_OPERATOR_INVARIANT_L1_NORM,  /**< L1 norm should remain within tolerance. */
    SIM_OPERATOR_INVARIANT_ENERGY,   /**< Energy should remain within tolerance. */
    SIM_OPERATOR_INVARIANT_MASS,     /**< Mass/sum should remain within tolerance. */
    SIM_OPERATOR_INVARIANT_MEAN,     /**< Mean value should remain within tolerance. */
    SIM_OPERATOR_INVARIANT_VARIANCE  /**< Variance should remain within tolerance. */
} SimOperatorInvariantKind;

/**
 * @brief Invariant declaration with tolerance.
 */
typedef struct SimOperatorInvariant {
    SimOperatorInvariantKind kind;      /**< Invariant type. */
    double                   tolerance; /**< Acceptable deviation (absolute). */
} SimOperatorInvariant;

/**
 * @brief Metadata describing operator characteristics.
 */
typedef struct SimOperatorInfo {
    SimOperatorCategory category;      /**< Operator category. */
    SimWarpLevel        warp_level;    /**< Warp classification level. */
    const char*         schema_key;    /**< Stable identity independent of rich schema metadata. */
    bool                has_ir_opcode; /**< True when core metadata supplied @ref ir_opcode. */
    SimIROpcode         ir_opcode;     /**< Core-owned semantic opcode for KernelIR tagging. */
    bool                is_noise;      /**< Whether operator involves randomness. */
    bool        is_differentiable; /**< Whether operator is differentiable (for AD/sensitivity). */
    bool        is_spectral;       /**< Requires FFT/spectral domain. */
    bool        is_local;          /**< Stencil or pointwise operator. */
    bool        is_nonlocal;       /**< Convolution, diffusion, or other nonlocal operator. */
    bool        is_linear;         /**< Enables reordering optimizations. */
    bool        is_warp;           /**< Nonlinear transform classified as a warp. */
    bool        preserves_real;    /**< True if operator keeps real fields real. */
    double      preferred_dt;      /**< Suggested timestep, or 0.0 for no preference. */
    const char* abstract_id;       /**< Optional abstract identifier (functional identity). */
    const char* (*abstract_id_fn)(
        const struct SimOperator* op);     /**< Callback to fetch abstract id. */
    uint32_t            algebraic_flags;   /**< Bitmask of algebraic properties. */
    SimDeterminismFlags determinism_flags; /**< Determinism guarantees promised by the operator. */
    SimOperatorNeuralContract neural;      /**< Neural inference contract metadata. */
    SimOperatorRepresentation representation; /**< Representation hints. */
    SimOperatorApproximation  approximation;  /**< Approximation order metadata. */
    SimOperatorInvariant      invariants[SIM_OPERATOR_MAX_INVARIANTS]; /**< Declared invariants. */
    uint8_t                   invariant_count; /**< Number of invariants populated. */
} SimOperatorInfo;

/**
 * @brief Operator runtime configuration shared between descriptors and instances.
 */
typedef struct SimOperatorConfig {
    SimContinuityMode   continuity;     /**< Singular-domain handling policy. */
    double              clamp_min;      /**< Lower clamp bound for limited continuity. */
    double              clamp_max;      /**< Upper clamp bound for limited continuity. */
    double              continuity_tol; /**< Offset tolerance for continuity probes. */
    SimIRBoundaryPolicy boundary;       /**< Boundary handling policy for spatial ops. */
    uint8_t             spacing_rank;   /**< Number of spatial spacing entries populated. */
    double spacing[SIM_OPERATOR_MAX_SPACING_DIMS]; /**< Optional per-dimension spacing metadata. */
    double norm_budget;                            /**< Optional L2 budget (0 disables). */
    double norm_budget_softness; /**< Placeholder for future soft-limit behaviour. */
    bool   representation_mode_override_enabled; /**< True when per-operator representation override
                                                  is set. */
    SimRepresentationMode
        representation_mode_override; /**< Per-operator representation override. */
} SimOperatorConfig;

/**
 * @brief Function signature for operator evaluation.
 */
typedef SimResult (*SimOperatorEvalFn)(struct SimContext*  context,
                                       struct SimOperator* self,
                                       void*               userdata);

/**
 * @brief Function signature for operator teardown callbacks.
 */
typedef void (*SimOperatorDestroyFn)(void* userdata);

/**
 * @brief Optional hook exposing a read-only GraphIR lowering view.
 */
typedef const struct SimGraphIR* (*SimOperatorGraphIRViewFn)(const struct SimOperator* self,
                                                             void*                     userdata);

/**
 * @brief Operator instance stored by the registry.
 */
typedef struct SimOperator {
    char     name[SIM_OPERATOR_NAME_MAX + 1U];             /**< Null-terminated operator name. */
    char     schema_key[SIM_OPERATOR_SCHEMA_KEY_MAX + 1U]; /**< Stable copied schema key. */
    uint64_t guid;                              /**< Monotonic unique identifier for tracing. */
    SimOperatorEvalFn         evaluate;         /**< Evaluation callback. */
    SimOperatorEvalFn         save_state;       /**< Optional snapshot callback (drift sandbox). */
    SimOperatorEvalFn         restore_state;    /**< Optional restore callback (drift sandbox). */
    SimOperatorDestroyFn      destroy;          /**< Optional teardown callback. */
    void*                     userdata;         /**< User payload. */
    size_t*                   dependencies;     /**< Owned dependency array. */
    size_t                    dependency_count; /**< Number of dependencies. */
    struct SimOperatorKernel* kernel;           /**< Optional kernel-backed execution. */
    SimOperatorInfo           info;             /**< Operator metadata. */
    SimOperatorConfig         config;           /**< Continuity/clamp configuration. */
    uint64_t                  read_mask; /**< Optional bitmask of context fields read (up to 64). */
    uint64_t write_mask;          /**< Optional bitmask of context fields written (up to 64). */
    size_t*  read_indices;        /**< Optional extended read set (all field indices). */
    size_t   read_index_count;    /**< Length of @ref read_indices. */
    size_t*  write_indices;       /**< Optional extended write set (all field indices). */
    size_t   write_index_count;   /**< Length of @ref write_indices. */
    uint64_t required_features;   /**< Backend features required by this operator (evaluate fallback
                                     allowed). */
    const void* catalog_metadata; /**< Optional catalog/runtime metadata. */
    const struct SimOperatorConfigAdapter*
                             config_adapter; /**< Optional config adapter for dynamic params. */
    SimOperatorGraphIRViewFn graph_ir_view;  /**< Optional GraphIR lowering view. */
} SimOperator;

/**
 * @brief Maps an IR field reference to a context field index.
 */
typedef struct SimOperatorKernelBindingDescriptor {
    size_t ir_field_index;      /**< Field index used inside the IR builder. */
    size_t context_field_index; /**< Field slot inside the context. */
} SimOperatorKernelBindingDescriptor;

/**
 * @brief Describes an output expression emitted by a kernel-backed operator.
 */
typedef struct SimOperatorKernelOutputDescriptor {
    size_t      ir_field_index; /**< Destination binding field index. */
    SimIRNodeId expression;     /**< Root node identifier for the expression. */
} SimOperatorKernelOutputDescriptor;

/**
 * @brief Descriptor for registering a kernel-backed operator.
 */
typedef struct SimOperatorKernelDescriptor {
    const SimIRBuilder* builder; /**< IR builder storing the referenced nodes. */
    const SimOperatorKernelBindingDescriptor* bindings;      /**< Binding descriptors. */
    size_t                                    binding_count; /**< Number of binding descriptors. */
    const SimOperatorKernelOutputDescriptor*  outputs;       /**< Output descriptors. */
    size_t                                    output_count;  /**< Number of output descriptors. */
    const double*                             params; /**< Optional initial parameter values. */
    size_t                                    param_count; /**< Number of runtime parameters. */
    uint64_t                  required_features; /**< Backend feature mask required to launch. */
    SimKernelComplexSemantics complex_semantics; /**< Complex-lane interpretation contract. */
} SimOperatorKernelDescriptor;

/**
 * @brief Runtime state for a kernel-backed operator.
 */
typedef struct SimOperatorKernel {
    KernelIR                            kernel;        /**< Kernel launch package. */
    SimOperatorKernelBindingDescriptor* binding_map;   /**< Owned binding descriptors. */
    SimKernelIRBinding*                 bindings;      /**< Mutable bindings used at runtime. */
    size_t                              binding_count; /**< Number of bindings. */
    SimOperatorKernelOutputDescriptor*  output_map;    /**< Owned output descriptors. */
    SimKernelIROutput*                  outputs;       /**< Mutable outputs used at runtime. */
    size_t                              output_count;  /**< Number of outputs. */
    double*                             params;        /**< Owned runtime parameter storage. */
    size_t                              param_count;   /**< Number of parameters in @ref params. */
} SimOperatorKernel;

/**
 * @brief Description used when registering a new operator.
 */
typedef struct SimOperatorDescriptor {
    const char*          name;         /**< Unique operator identifier. */
    SimOperatorConfig    config;       /**< Optional per-operator configuration overrides. */
    SimOperatorEvalFn    evaluate;     /**< Evaluation callback. */
    SimOperatorDestroyFn destroy;      /**< Optional teardown callback. */
    void*                userdata;     /**< Context passed to callbacks. */
    const size_t*        dependencies; /**< Array of operator indices this operator depends on. */
    size_t               dependency_count;     /**< Length of @ref dependencies. */
    const SimOperatorKernelDescriptor* kernel; /**< Optional kernel-backed execution descriptor. */
    SimOperatorInfo                    info;   /**< Operator metadata. */
    uint64_t      read_mask;         /**< Optional bitmask of context fields read (up to 64). */
    uint64_t      write_mask;        /**< Optional bitmask of context fields written (up to 64). */
    const size_t* read_indices;      /**< Optional extended read set (all field indices). */
    size_t        read_index_count;  /**< Length of @ref read_indices. */
    const size_t* write_indices;     /**< Optional extended write set (all field indices). */
    size_t        write_index_count; /**< Length of @ref write_indices. */
    uint64_t      required_features; /**< Backend features required to execute (for hazard sync). */
    const void*   catalog_metadata;  /**< Optional catalog/runtime metadata descriptor. */
    SimOperatorEvalFn save_state;    /**< Optional snapshot hook for drift sandbox. */
    SimOperatorEvalFn restore_state; /**< Optional restore hook for drift sandbox. */
    const struct SimOperatorConfigAdapter*
                             config_adapter; /**< Optional config adapter for dynamic params. */
    SimOperatorGraphIRViewFn graph_ir_view;  /**< Optional GraphIR lowering view. */
} SimOperatorDescriptor;

/**
 * @brief Operator registry storing owned operator instances.
 */
typedef struct SimOperatorRegistry {
    SimOperator* records;  /**< Contiguous operator array. */
    size_t       count;    /**< Number of active operators. */
    size_t       capacity; /**< Allocated capacity. */
} SimOperatorRegistry;

/**
 * @brief Execution plan obtained via dependency resolution.
 */
typedef struct SimOperatorPlan {
    size_t* order; /**< Topologically sorted operator indices. */
    size_t  count; /**< Number of scheduled operators. */
} SimOperatorPlan;

/**
 * @brief Initialize an empty registry.
 *
 * @param[out] registry Target registry.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_operator_registry_init(SimOperatorRegistry* registry);

/**
 * @brief Release all operators contained in a registry.
 *
 * @param registry Registry to tear down.
 */
void sim_operator_registry_destroy(SimOperatorRegistry* registry);

/**
 * @brief Register a new operator.
 *
 * @param registry Target registry.
 * @param descriptor Operator descriptor; all strings are copied.
 * @param[out] out_index Optional pointer receiving the assigned index.
 * @return #SIM_RESULT_OK on success or an error code.
 */
SimResult sim_operator_registry_register(SimOperatorRegistry*         registry,
                                         const SimOperatorDescriptor* descriptor,
                                         size_t*                      out_index);

/**
 * @brief Lookup an operator by index.
 *
 * @param registry Registry instance.
 * @param index Operator index.
 * @return Pointer to the operator, or NULL if not found.
 */
SimOperator* sim_operator_registry_get(SimOperatorRegistry* registry, size_t index);

/**
 * @brief Access operator dependency list.
 *
 * @param op Operator instance.
 * @param[out] out_deps Pointer receiving dependency array (owned by @p op).
 * @param[out] out_count Pointer receiving dependency count.
 */
void sim_operator_dependencies(const SimOperator* op, const size_t** out_deps, size_t* out_count);

/**
 * @brief Resolve an execution plan for the registry.
 *
 * @param registry Registry instance.
 * @param[out] plan Execution plan to populate.
 * @param logger Optional callback for dependency-cycle diagnostics.
 * @param logger_userdata Opaque pointer forwarded to @p logger.
 * @return #SIM_RESULT_OK on success or #SIM_RESULT_DEPENDENCY_ERROR when cycles exist.
 */
SimResult sim_operator_resolve_plan_with_logger(const SimOperatorRegistry* registry,
                                                SimOperatorPlan*           plan,
                                                SimOperatorCycleLogFn      logger,
                                                void*                      logger_userdata);

SimResult sim_operator_resolve_plan(const SimOperatorRegistry* registry, SimOperatorPlan* plan);

/**
 * @brief Release resources owned by a plan.
 *
 * @param plan Plan to destroy; may be NULL.
 */
void sim_operator_plan_destroy(SimOperatorPlan* plan);

/**
 * @brief Retrieve the internal user payload.
 *
 * @param op Operator instance.
 * @return User-provided pointer.
 */
void* sim_operator_payload(SimOperator* op);

/**
 * @brief Retrieve the operator name.
 *
 * @param op Operator instance.
 * @return Null-terminated operator name.
 */
const char* sim_operator_name(const SimOperator* op);

/**
 * @brief Retrieve operator metadata.
 *
 * @param op Operator instance.
 * @return Operator metadata structure.
 */
SimOperatorInfo sim_operator_info(const SimOperator* op);

/**
 * @brief Convenience helpers for working with operator configuration payloads.
 */
void              sim_operator_config_normalize(SimOperatorConfig* config);
SimOperatorConfig sim_operator_config_defaults(void);
SimOperatorInfo   sim_operator_info_defaults(void);
void              sim_operator_info_normalize(SimOperatorInfo* info);
void              sim_operator_info_set_identity(SimOperatorInfo* info,
                                                 const char*      schema_key,
                                                 SimIROpcode      ir_opcode);
void        sim_operator_info_set_schema_identity(SimOperatorInfo* info, const char* schema_key);
const char* sim_operator_abstract_id(const struct SimOperator* op);
void        sim_operator_set_schema_key(struct SimOperator* op, const char* schema_key);
void        sim_operator_set_catalog_metadata(struct SimOperator* op, const void* metadata);
const void* sim_operator_catalog_metadata(const struct SimOperator* op);
const char* sim_operator_schema_key(const struct SimOperator* op);
const char* sim_operator_schema_key_or(const struct SimOperator* op, const char* fallback);
SimIROpcode sim_operator_ir_opcode(const struct SimOperator* op);
const SimOperatorRepresentation* sim_operator_representation(const struct SimOperator* op);
const char*              sim_operator_representation_domain_name(const struct SimOperator* op);
const char*              sim_operator_representation_value_kind_name(const struct SimOperator* op);
const struct SimGraphIR* sim_operator_graph_ir(const struct SimOperator* op);
bool sim_operator_config_get(const SimOperator* op, SimOperatorConfig* out_config);
void sim_operator_config_set(struct SimOperator* op, const SimOperatorConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_OPERATOR_H */
