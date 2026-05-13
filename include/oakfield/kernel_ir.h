/**
 * @file kernel_ir.h
 * @brief Intermediate representation facilities for libsimcore operator fusion.
 * @ingroup oakfield_kernel_ir
 *
 * @details KernelIR is a typed expression graph used by operators and backends
 * to describe pointwise numeric work independently of a concrete execution
 * engine. Builders own node and constant-pool storage; backend launch packages
 * borrow builders, field bindings, outputs, and runtime parameters only for the
 * duration of a launch.
 */
#ifndef OAKFIELD_KERNEL_IR_H
#define OAKFIELD_KERNEL_IR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "field.h"
#include "math/special_functions.h"
#include "operator_identity.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Identifier for an IR node. */
typedef size_t SimIRNodeId;

/** Sentinel value returned when node creation fails. */
#define SIM_IR_INVALID_NODE ((SimIRNodeId)SIZE_MAX)
/* Sentinel for a missing constant pool entry */
#define SIM_IR_INVALID_CONSTANT_INDEX ((size_t)SIZE_MAX)

/**
 * @brief Enumerates the value category for an IR expression.
 */
typedef enum SimIRValueKind {
    SIM_IR_VALUE_SCALAR = 0, /**< Scalar-valued expression. */
    SIM_IR_VALUE_VECTOR      /**< Vector-valued expression with @ref SimIRType::components lanes. */
} SimIRValueKind;

/**
 * @brief Type descriptor carried by each IR node.
 */
typedef struct SimIRType {
    SimIRValueKind kind;           /**< Value kind (scalar/vector). */
    size_t components;             /**< Component count (>=1); scalar nodes store 1. */
    SimScalarDomain scalar_domain; /**< Scalar algebra domain for each lane. */
} SimIRType;

/**
 * @brief Exact scalar payload used by scalar-domain-aware evaluators.
 */
typedef union SimIRDomainPayload {
    SimComplexDouble as_complex; /**< Complex payload. */
    double as_f64;               /**< Real floating payload. */
    int64_t as_i64;              /**< Signed integer payload. */
    uint64_t as_u64;             /**< Unsigned integer payload. */
} SimIRDomainPayload;

/**
 * @brief Scalar-domain-aware evaluation result.
 */
typedef struct SimIRDomainValue {
    SimScalarDomain domain;   /**< Scalar domain of the evaluated root expression. */
    SimIRDomainPayload value; /**< Exact value payload matching @ref domain. */
} SimIRDomainValue;

/**
 * @brief Maximum rank supported by shape descriptors.
 */
typedef struct SimIRShape {
    size_t rank;     /**< Number of valid dimensions in @ref shape. */
    size_t shape[4]; /**< Extent per dimension, up to the supported rank. */
    bool has_shape;  /**< True when shape/rank contain inferred metadata. */
} SimIRShape;

/**
 * @brief Boundary handling policy for differential operators.
 */
typedef enum SimIRBoundaryPolicy {
    SIM_IR_BOUNDARY_NEUMANN = 0, /**< Zero-gradient boundary (mirrors center value). */
    SIM_IR_BOUNDARY_DIRICHLET,   /**< Fixed-value boundary (assumed zero when unspecified). */
    SIM_IR_BOUNDARY_PERIODIC,    /**< Wrap indices across the domain. */
    SIM_IR_BOUNDARY_REFLECTIVE   /**< Reflect indices to remain in bounds. */
} SimIRBoundaryPolicy;

/**
 * @brief Finite-difference stencil selection for differential operators.
 */
typedef enum SimIRDiffMethod {
    SIM_IR_DIFF_METHOD_AUTO = 0, /**< Adaptive: central when possible, else one-sided. */
    SIM_IR_DIFF_METHOD_CENTRAL, /**< Central difference (uses boundary substitution when needed). */
    SIM_IR_DIFF_METHOD_FORWARD, /**< Forward difference. */
    SIM_IR_DIFF_METHOD_BACKWARD /**< Backward difference. */
} SimIRDiffMethod;

/**
 * @brief Noise law describing the stochastic calculus interpretation.
 */
typedef enum SimIRNoiseLaw {
    SIM_IR_NOISE_LAW_ITO = 0,     /**< Itô interpretation (default). */
    SIM_IR_NOISE_LAW_STRATONOVICH /**< Stratonovich interpretation. */
} SimIRNoiseLaw;

/**
 * @brief Probability distribution used when sampling stochastic nodes.
 */
typedef enum SimIRNoiseDistribution {
    SIM_IR_NOISE_DISTRIBUTION_UNIFORM = 0, /**< Uniform samples in [-amplitude, amplitude]. */
    SIM_IR_NOISE_DISTRIBUTION_GAUSSIAN     /**< Gaussian samples with stddev = amplitude. */
} SimIRNoiseDistribution;

/**
 * @brief Runtime parameter identifiers referenced by PARAM nodes.
 */
typedef enum SimIRParamKind {
    SIM_IR_PARAM_DT = 0,     /**< Current timestep (seconds). */
    SIM_IR_PARAM_STEP_INDEX, /**< Monotonic step index (integer-valued). */
    SIM_IR_PARAM_SQRT_DT,    /**< Square root of timestep (seconds^0.5). */
    SIM_IR_PARAM_TIME        /**< Current simulation time (seconds). */
} SimIRParamKind;

/**
 * @brief Optional metadata for creating differential nodes.
 */
typedef struct SimIRDiffSpec {
    SimIRNodeId operand;          /**< Operand node identifier. */
    size_t axis;                  /**< Axis index (0-based). */
    double dx;                    /**< Grid spacing along @ref axis. */
    double scale;                 /**< Scaling factor applied after discretization. */
    size_t order;                 /**< Order of the derivative (defaults to 1). */
    size_t stencil_order;         /**< Accuracy order p of the finite-difference stencil. */
    SimIRDiffMethod method;       /**< Stencil method selection. */
    double consistency_constant;  /**< Consistency constant C_p for truncation error bounds. */
    SimIRBoundaryPolicy boundary; /**< Boundary handling policy. */
    SimIRType result_type;        /**< Resulting value type (defaults to operand type). */
} SimIRDiffSpec;

/**
 * @brief Supported analytic warp profiles for dedicated IR nodes.
 */
typedef enum SimIRWarpProfile {
    SIM_IR_WARP_PROFILE_DIGAMMA = 0,          /**< Digamma warp (psi, 12-tail default). */
    SIM_IR_WARP_PROFILE_TRIGAMMA = 1,         /**< Trigamma warp (psi\_1). */
    SIM_IR_WARP_PROFILE_DIGAMMA_7_TAIL = 2,   /**< Digamma (7-term Stirling tail). */
    SIM_IR_WARP_PROFILE_DIGAMMA_5_TAIL = 3,   /**< Digamma (5-term Stirling tail). */
    SIM_IR_WARP_PROFILE_DIGAMMA_ADAPTIVE = 4, /**< Digamma (adaptive-tail). */
    SIM_IR_WARP_PROFILE_DIGAMMA_MORTICI = 5   /**< Digamma (Mortici speedy). */
} SimIRWarpProfile;

/**
 * @brief Built-in call identifiers for unary analytic functions.
 */
typedef enum SimIRCallKind {
    SIM_IR_CALL_SIN = 0, /**< Sine function. */
    SIM_IR_CALL_COS,     /**< Cosine function. */
    SIM_IR_CALL_EXP,     /**< Exponential function. */
    SIM_IR_CALL_ABS,     /**< Absolute value. */
    SIM_IR_CALL_LOG,     /**< Natural logarithm. */
    SIM_IR_CALL_TANH,    /**< Hyperbolic tangent. */
    SIM_IR_CALL_SINH,    /**< Hyperbolic sine. */
    SIM_IR_CALL_SIGN     /**< Sign (copysign(1, x)). */
} SimIRCallKind;

/**
 * @brief Continuity guard metadata attached to warp specifications.
 */
typedef struct SimIRWarpGuard {
    int mode;         /**< Continuity mode (mirrors @ref SimContinuityMode). */
    double clamp_min; /**< Lower clamp bound when guards are enabled. */
    double clamp_max; /**< Upper clamp bound when guards are enabled. */
    double tolerance; /**< Offset tolerance used for guard fallbacks. */
} SimIRWarpGuard;

/**
 * @brief Metadata for creating analytic warp nodes.
 */
typedef struct SimIRWarpSpec {
    SimIRNodeId operand;      /**< Operand node identifier. */
    double bias;              /**< Bias applied before sampling the warp. */
    double delta;             /**< Symmetric offset for evaluating the warp. */
    double lambda;            /**< Scaling factor applied to the warp response. */
    double tolerance;         /**< Optional tolerance for adaptive warps (<=0 uses default). */
    SimIRWarpProfile profile; /**< Warp profile selector. */
    SimWarpLevel warp_class;  /**< Optional classification hint for code generation. */
    SimIRWarpGuard guard;     /**< Continuity guard policy for runtime sampling. */
    SimIRType result_type;    /**< Resulting value type (defaults to scalar). */
} SimIRWarpSpec;

/**
 * @brief Optional metadata for creating stochastic noise nodes.
 */
typedef struct SimIRNoiseSpec {
    uint32_t seed;                       /**< Seed for deterministic sampling streams. */
    double amplitude;                    /**< Amplitude or standard deviation for the noise. */
    double variance;                     /**< Variance parameter σ² carried for SDE estimates. */
    SimIRNoiseLaw law;                   /**< Stochastic calculus interpretation. */
    SimIRNoiseDistribution distribution; /**< Probability distribution for samples. */
    SimIRType value_type;                /**< Resulting value type (defaults to scalar). */
} SimIRNoiseSpec;

/**
 * @brief Enumerates IR node categories.
 */
typedef enum SimIRNodeType {
    SIM_IR_NODE_CONSTANT = 0,   /**< Scalar literal. */
    SIM_IR_NODE_FIELD_REF,      /**< Reference to a field. */
    SIM_IR_NODE_ADD,            /**< Binary addition. */
    SIM_IR_NODE_SUB,            /**< Binary subtraction. */
    SIM_IR_NODE_MUL,            /**< Binary multiplication. */
    SIM_IR_NODE_DIV,            /**< Binary division. */
    SIM_IR_NODE_POW,            /**< Binary power (pow for real, cpow for complex). */
    SIM_IR_NODE_DIFF,           /**< Differential term (finite difference placeholder). */
    SIM_IR_NODE_NOISE,          /**< Stochastic noise term. */
    SIM_IR_NODE_WARP,           /**< Analytic warp response. */
    SIM_IR_NODE_PARAM,          /**< Runtime parameter lookup (e.g., dt). */
    SIM_IR_NODE_COMPLEX_PACK,   /**< Pack real/imag scalars into a complex vector. */
    SIM_IR_NODE_COMPLEX_ROTATE, /**< Complex phase rotation by a scalar angle. */
    SIM_IR_NODE_INDEX,          /**< Linear element index within the field. */
    SIM_IR_NODE_CALL,           /**< Unary analytic call node (sin/cos/exp/abs/log). */
    SIM_IR_NODE_FLOOR,          /**< Floor operation on a scalar operand. */
    SIM_IR_NODE_MOD,            /**< Modulo operation on scalar operands. */
    SIM_IR_NODE_COORD,          /**< Coordinate lookup along an axis. */
    SIM_IR_NODE_STATEFUL        /**< Stateful callback-driven node (CPU-only). */
} SimIRNodeType;

struct KernelIR;
/**
 * @brief Callback signature for stateful IR nodes.
 */
typedef SimResult (*SimIRStatefulEvalFn)(void *userdata, const struct KernelIR *kernel,
                                         size_t element_index, size_t component, double *out_value);

/**
 * @brief Generic IR node representation.
 */
#ifndef SIM_IR_SMALL_CONSTANT_CAPACITY
#define SIM_IR_SMALL_CONSTANT_CAPACITY 4U
#endif

/**
 * @brief Node record in a symbolic kernel IR expression graph.
 */
typedef struct SimIRNode {
    SimIRNodeType type;        /**< Node classification. */
    SimIRNodeId id;            /**< Unique identifier within the builder. */
    SimIRType value_type;      /**< Declared value type for the node. */
    SimIROpcode opcode;        /**< Semantic operator category opcode. */
    SimIRShape inferred_shape; /**< Inferred shape descriptor for the node. */
    SimWarpLevel warp_class;   /**< Warp classification hint (defaults to NONE). */
    bool is_local; /**< True when the node is pointwise (Lipschitz), false for non-local ops. */
    union {
        struct {
            double scalar;            /**< Literal value for constant nodes. */
            int64_t signed_scalar;    /**< Exact signed integer literal. */
            uint64_t unsigned_scalar; /**< Exact unsigned integer literal. */
            bool exact_integer;       /**< True when an exact integer literal is present. */
            size_t constant_index;    /**< Index into builder constant pool for vector constants. */
            double small[SIM_IR_SMALL_CONSTANT_CAPACITY]; /**< Inline per-lane constant storage for
                                                             small vectors. */
        } constant;                                       /**< Constant literal payload. */
        size_t field;                                     /**< Field identifier for references. */
        struct {
            SimIRNodeId lhs; /**< Left operand node. */
            SimIRNodeId rhs; /**< Right operand node. */
        } binary;            /**< Binary operation payload. */
        struct {
            SimIRNodeId operand;    /**< Operand node identifier. */
            size_t axis;            /**< Axis along which the derivative is taken (0=x). */
            double dx;              /**< Grid spacing along @ref axis. */
            double scale;           /**< Scaling applied to the differential term. */
            size_t order;           /**< Derivative order (>=1). */
            size_t stencil_order;   /**< Accuracy order p for the stencil. */
            SimIRDiffMethod method; /**< Stencil method selection. */
            double
                consistency_constant; /**< Consistency constant C_p for truncation error bounds. */
            SimIRBoundaryPolicy boundary; /**< Boundary policy. */
        } diff;                           /**< Differential term payload. */
        struct {
            uint32_t seed;                       /**< Pseudorandom seed. */
            double amplitude;                    /**< Noise magnitude. */
            double variance;                     /**< Variance parameter σ². */
            SimIRNoiseLaw law;                   /**< Stochastic calculus interpretation. */
            SimIRNoiseDistribution distribution; /**< Sampling distribution. */
        } noise;                                 /**< Noise payload. */
        struct {
            SimIRNodeId operand;      /**< Operand node identifier. */
            double bias;              /**< Additive bias prior to sampling. */
            double delta;             /**< Symmetric evaluation offset. */
            double lambda;            /**< Scaling applied to the response. */
            double tolerance;         /**< Optional tolerance for adaptive warps. */
            SimIRWarpProfile profile; /**< Warp profile selector. */
            SimIRWarpGuard guard;     /**< Guard metadata controlling continuity handling. */
        } warp;                       /**< Analytic warp payload. */
        struct {
            SimIRParamKind param; /**< Runtime parameter identifier. */
        } param;                  /**< Parameter payload. */
        struct {
            SimIRNodeId operand; /**< Complex operand (re, im). */
            SimIRNodeId angle;   /**< Scalar rotation angle (radians). */
        } complex_rotate;        /**< Complex rotation payload. */
        struct {
            SimIRNodeId real; /**< Real component. */
            SimIRNodeId imag; /**< Imaginary component. */
        } complex_pack;       /**< Complex pack payload. */
        struct {
            SimIRNodeId operand; /**< Operand node identifier. */
            SimIRCallKind kind;  /**< Built-in call identifier. */
        } call;                  /**< Unary call payload. */
        struct {
            SimIRNodeId operand; /**< Operand node identifier. */
        } unary;                 /**< Unary operand payload. */
        struct {
            size_t field; /**< Field identifier providing shape/stride metadata. */
            size_t axis;  /**< Axis index for coordinate lookup. */
        } coord;          /**< Coordinate lookup payload. */
        struct {
            SimIRStatefulEvalFn eval; /**< Callback to produce the node value. */
            void *userdata;           /**< User context passed to @ref eval. */
            const char *label;        /**< Optional label for MathView rendering. */
        } stateful;                   /**< Stateful callback payload. */
    } data;                           /**< Type-specific payload. */
} SimIRNode;

struct SimWarpSampleSpec;

/**
 * @brief Metadata for creating stateful IR nodes.
 */
typedef struct SimIRStatefulSpec {
    SimIRStatefulEvalFn eval; /**< Callback to produce the node value. */
    void *userdata;           /**< User context passed to @ref eval. */
    const char *label;        /**< Optional label for MathView rendering. */
    SimIRType value_type;     /**< Resulting value type (defaults to scalar). */
} SimIRStatefulSpec;

SimResult sim_ir_warp_sample_response(const struct SimWarpSampleSpec *spec,
                                      SimIRWarpProfile profile, double tolerance,
                                      SimSpecialFallbackFn fallback, void *fallback_userdata,
                                      double *out_response);

/**
 * @brief Monotonic arena for IR node allocation.
 */
typedef struct SimIRBuilder {
    SimIRNode *nodes;                     /**< Node storage. */
    size_t count;                         /**< Number of valid entries in @ref nodes. */
    size_t capacity;                      /**< Allocated capacity for @ref nodes. */
    SimIRBoundaryPolicy default_boundary; /**< Default boundary policy for helper-built diffs. */
    /* Constant pool to store per-lane vector constants.
     * Flattened layout of all vector constant values. */
    double *constants_data;         /**< Flattened vector-constant lane storage. */
    size_t *constants_offsets;      /**< Per-constant offset into @ref constants_data. */
    size_t *constants_components;   /**< Per-constant component count. */
    size_t constants_count;         /**< Number of vector constants in the pool. */
    size_t constants_capacity;      /**< Entry capacity for offsets/components arrays. */
    size_t constants_data_capacity; /**< Value capacity for constants_data (number of doubles). */
    size_t constants_data_used;     /**< Number of values populated in constants_data. */
} SimIRBuilder;

/**
 * @brief Initialize an empty IR builder.
 *
 * @param[out] builder Builder instance to initialize.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_ir_builder_init(SimIRBuilder *builder);

/**
 * @brief Set the default boundary policy used by shorthand differential helpers.
 *
 * @param builder Target builder.
 * @param boundary Boundary policy; falls back to NEUMANN when invalid.
 */
void sim_ir_builder_set_default_boundary(SimIRBuilder *builder, SimIRBoundaryPolicy boundary);

/**
 * @brief Reset builder counts while retaining allocated storage.
 *
 * Clears nodes/constants counts but keeps capacity for reuse.
 */
void sim_ir_builder_reset(SimIRBuilder *builder);

/**
 * @brief Apply a semantic opcode to nodes stored in a builder.
 *
 * @param builder Target builder.
 * @param opcode Opcode to assign (falls back to OAK_OP_MISC when invalid).
 * @param preserve_existing When true, only nodes tagged OAK_OP_MISC are updated.
 */
void sim_ir_builder_apply_opcode(SimIRBuilder *builder, SimIROpcode opcode, bool preserve_existing);

/**
 * @brief Apply a semantic opcode to reachable nodes starting from the provided roots.
 *
 * @param builder Target builder.
 * @param opcode Opcode to assign (falls back to OAK_OP_MISC when invalid).
 * @param preserve_existing When true, only nodes tagged OAK_OP_MISC are updated.
 * @param roots Array of root node identifiers to traverse.
 * @param root_count Number of roots in @p roots.
 */
SimResult sim_ir_builder_apply_opcode_reachable(SimIRBuilder *builder, SimIROpcode opcode,
                                                bool preserve_existing, const SimIRNodeId *roots,
                                                size_t root_count);

/**
 * @brief Mark nodes reachable from a set of roots.
 *
 * @param builder Builder instance.
 * @param roots Root node identifiers to traverse.
 * @param root_count Number of roots in @p roots.
 * @param out_reachable Byte array to receive reachability flags (size >= builder->count).
 * @param reachable_count Size of @p out_reachable in bytes.
 */
SimResult sim_ir_collect_reachable(const SimIRBuilder *builder, const SimIRNodeId *roots,
                                   size_t root_count, unsigned char *out_reachable,
                                   size_t reachable_count);

/**
 * @brief Initialize a diff spec with safe defaults.
 *
 * Sets order/stencil_order=1, scale=1, dx=1, boundary=builder default (if provided) or NEUMANN,
 * and result_type to scalar.
 */
void sim_ir_diff_spec_init(SimIRDiffSpec *spec, const SimIRBuilder *builder);

/**
 * @brief Release resources held by an IR builder.
 *
 * @param builder Builder to destroy; may be NULL.
 */
void sim_ir_builder_destroy(SimIRBuilder *builder);

/**
 * @brief Retrieve an immutable pointer to a node.
 *
 * @param builder Builder instance.
 * @param id Node identifier.
 * @return Pointer to the node or NULL if @p id is invalid.
 */
const SimIRNode *sim_ir_builder_get(const SimIRBuilder *builder, SimIRNodeId id);

/**
 * @brief Query the value type of a node.
 */
SimIRType sim_ir_builder_node_type(const SimIRBuilder *builder, SimIRNodeId id);

/**
 * @brief Retrieve the warp classification associated with a node.
 */
SimWarpLevel sim_ir_builder_node_warp_class(const SimIRBuilder *builder, SimIRNodeId id);

/**
 * @brief Assign a warp classification to a node for downstream code generation.
 */
SimResult sim_ir_builder_set_node_warp_class(SimIRBuilder *builder, SimIRNodeId id,
                                             SimWarpLevel warp_class);

/**
 * @brief Create a constant node with explicit type annotation.
 */
SimIRNodeId sim_ir_builder_constant_typed(SimIRBuilder *builder, double value, SimIRType type);

/**
 * @brief Create an exact signed integer constant node with explicit type annotation.
 */
SimIRNodeId sim_ir_builder_constant_i64_typed(SimIRBuilder *builder, int64_t value, SimIRType type);

/**
 * @brief Create an exact unsigned integer constant node with explicit type annotation.
 */
SimIRNodeId sim_ir_builder_constant_u64_typed(SimIRBuilder *builder, uint64_t value,
                                              SimIRType type);

/**
 * @brief Create a vector constant node with explicit type annotation.
 */
SimIRNodeId sim_ir_builder_constant_vector_typed(SimIRBuilder *builder, const double *values,
                                                 size_t components, SimIRType type);

/**
 * @brief Create a complex constant node from real/imag components.
 */
SimIRNodeId sim_ir_builder_constant_complex(SimIRBuilder *builder, double real, double imag);

/**
 * @brief Add a per-lane constant vector to the builder and return a node id.
 *
 * Values are copied into the builder's constant pool; the returned node is a constant
 * node with its value_type set to vector(components).
 */
SimIRNodeId sim_ir_builder_constant_vector(SimIRBuilder *builder, const double *values,
                                           size_t components);

/**
 * @brief Query whether a kernel contains true complex-domain nodes whose semantics are not
 * representable by the simple pointwise GPU emitters.
 *
 * Kernels that opt into componentwise complex-lane semantics are treated as supported.
 */
bool sim_ir_kernel_has_unsupported_complex_semantics(const struct KernelIR *kernel);

/**
 * @brief Create a constant node.
 *
 * @param builder Target builder.
 * @param value Literal value.
 * @return Identifier of the created node or #SIM_IR_INVALID_NODE on failure.
 */
SimIRNodeId sim_ir_builder_constant(SimIRBuilder *builder, double value);

/**
 * @brief Create a runtime parameter node.
 */
SimIRNodeId sim_ir_builder_param(SimIRBuilder *builder, SimIRParamKind param);

/**
 * @brief Create a linear element-index node.
 */
SimIRNodeId sim_ir_builder_index(SimIRBuilder *builder);

/**
 * @brief Create a unary analytic call node (elementwise for vector values).
 */
SimIRNodeId sim_ir_builder_call(SimIRBuilder *builder, SimIRCallKind kind, SimIRNodeId operand);

/**
 * @brief Create a floor node.
 */
SimIRNodeId sim_ir_builder_floor(SimIRBuilder *builder, SimIRNodeId operand);

/**
 * @brief Create a modulo node.
 */
SimIRNodeId sim_ir_builder_mod(SimIRBuilder *builder, SimIRNodeId lhs, SimIRNodeId rhs);

/**
 * @brief Create a coordinate lookup node.
 */
SimIRNodeId sim_ir_builder_coord(SimIRBuilder *builder, size_t field_id, size_t axis);
/**
 * @brief Create a field reference node with an explicit value type.
 */
SimIRNodeId sim_ir_builder_field_ref_typed(SimIRBuilder *builder, size_t field_id, SimIRType type);

/**
 * @brief Create a field reference node.
 *
 * @param builder Target builder.
 * @param field_id Identifier of the referenced field.
 * @return New node identifier or #SIM_IR_INVALID_NODE on failure.
 */
SimIRNodeId sim_ir_builder_field_ref(SimIRBuilder *builder, size_t field_id);

/**
 * @brief Create a binary operation node.
 *
 * @param builder Target builder.
 * @param type Operation type; must be one of the binary node kinds.
 * @param lhs Left operand.
 * @param rhs Right operand.
 * @return Node identifier or #SIM_IR_INVALID_NODE.
 */
SimIRNodeId sim_ir_builder_binary(SimIRBuilder *builder, SimIRNodeType type, SimIRNodeId lhs,
                                  SimIRNodeId rhs);

/**
 * @brief Create a power node (pow for real, cpow for complex).
 */
SimIRNodeId sim_ir_builder_pow(SimIRBuilder *builder, SimIRNodeId lhs, SimIRNodeId rhs);

/**
 * @brief Create a differential term node using an extended specification.
 */
SimIRNodeId sim_ir_builder_diff_spec(SimIRBuilder *builder, const SimIRDiffSpec *spec);

/**
 * @brief Create a differential term node.
 *
 * @param builder Target builder.
 * @param operand Operand node identifier.
 * @param axis Axis index for the derivative.
 * @param dx Grid spacing along @p axis.
 * @param scale Scaling factor applied to the differential.
 * @return Node identifier or #SIM_IR_INVALID_NODE.
 */
SimIRNodeId sim_ir_builder_diff(SimIRBuilder *builder, SimIRNodeId operand, size_t axis, double dx,
                                double scale);

/**
 * @brief Create a stochastic noise node with extended metadata.
 */
SimIRNodeId sim_ir_builder_noise_spec(SimIRBuilder *builder, const SimIRNoiseSpec *spec);

/**
 * @brief Create a stateful callback node with extended metadata.
 */
SimIRNodeId sim_ir_builder_stateful_spec(SimIRBuilder *builder, const SimIRStatefulSpec *spec);

/**
 * @brief Create a stateful callback node with a scalar result type.
 */
SimIRNodeId sim_ir_builder_stateful(SimIRBuilder *builder, SimIRStatefulEvalFn eval, void *userdata,
                                    const char *label);

/**
 * @brief Create a stochastic noise node.
 *
 * @param builder Target builder.
 * @param seed Seed value for deterministic noise streams.
 * @param amplitude Noise amplitude.
 * @return Node identifier or #SIM_IR_INVALID_NODE.
 */
SimIRNodeId sim_ir_builder_noise(SimIRBuilder *builder, uint32_t seed, double amplitude);

/**
 * @brief Create an analytic warp node using the extended specification.
 */
SimIRNodeId sim_ir_builder_warp_spec(SimIRBuilder *builder, const SimIRWarpSpec *spec);

/**
 * @brief Create an analytic warp node with explicit parameters.
 */
SimIRNodeId sim_ir_builder_warp(SimIRBuilder *builder, SimIRNodeId operand,
                                SimIRWarpProfile profile, double bias, double delta, double lambda);

/**
 * @brief Pack real/imag scalar nodes into a complex-valued vector.
 */
SimIRNodeId sim_ir_builder_complex_pack(SimIRBuilder *builder, SimIRNodeId real, SimIRNodeId imag);

/**
 * @brief Create a complex rotation node using a scalar angle.
 */
SimIRNodeId sim_ir_builder_complex_rotate(SimIRBuilder *builder, SimIRNodeId operand,
                                          SimIRNodeId angle);

/**
 * @brief Construct a scalar value type descriptor.
 */
SimIRType sim_ir_type_scalar(void);

/**
 * @brief Construct a scalar value type descriptor for an explicit scalar domain.
 */
SimIRType sim_ir_type_scalar_domain_typed(SimScalarDomain domain);

/**
 * @brief Construct a vector value type descriptor with the requested lane count.
 */
SimIRType sim_ir_type_vector(size_t components);

/**
 * @brief Construct a complex scalar value type descriptor (2 lanes).
 */
SimIRType sim_ir_type_complex(void);

/**
 * @brief Return true if the provided type describes a scalar expression.
 */
bool sim_ir_type_is_scalar(SimIRType type);

/**
 * @brief Compare two type descriptors for equality.
 */
bool sim_ir_type_equal(SimIRType lhs, SimIRType rhs);

/**
 * @brief Return the scalar-domain descriptor attached to an IR type.
 */
SimScalarDomain sim_ir_type_scalar_domain(SimIRType type);

/**
 * @brief Evaluate a warp profile at the provided sample.
 */
double sim_ir_warp_profile_eval(SimIRWarpProfile profile, double x, double tolerance);

/**
 * @brief Compute the symmetric warp difference f(x+δ) − f(x−δ).
 */
double sim_ir_warp_difference(SimIRWarpProfile profile, double sample, double delta,
                              double tolerance);

/**
 * @brief Resolve a real-valued field reference during IR evaluation.
 */
typedef SimResult (*SimIRFieldEvalFn)(void *userdata, size_t field_id, SimIRType type,
                                      double *out_value);

/**
 * @brief Evaluate a real-valued differential node during IR evaluation.
 */
typedef SimResult (*SimIRDiffEvalFn)(void *userdata, const SimIRBuilder *builder,
                                     const SimIRNode *node, SimIRNodeId operand,
                                     double operand_value, double *out_value);

/**
 * @brief Sample a real-valued stochastic node during IR evaluation.
 */
typedef SimResult (*SimIRNoiseEvalFn)(void *userdata, const SimIRNode *node, double *out_value);

/**
 * @brief Resolve a real-valued runtime parameter during IR evaluation.
 */
typedef SimResult (*SimIRParamEvalFn)(void *userdata, SimIRParamKind param, double *out_value);

/**
 * @brief Resolve a complex-valued field reference during IR evaluation.
 */
typedef SimResult (*SimIRFieldEvalFnComplex)(void *userdata, size_t field_id, SimIRType type,
                                             SimComplexDouble *out_value);

/**
 * @brief Evaluate a complex-valued differential node during IR evaluation.
 */
typedef SimResult (*SimIRDiffEvalFnComplex)(void *userdata, const SimIRBuilder *builder,
                                            const SimIRNode *node, SimIRNodeId operand,
                                            SimComplexDouble operand_value,
                                            SimComplexDouble *out_value);

/**
 * @brief Sample a complex-valued stochastic node during IR evaluation.
 */
typedef SimResult (*SimIRNoiseEvalFnComplex)(void *userdata, const SimIRNode *node,
                                             SimComplexDouble *out_value);

/**
 * @brief Callback surface for evaluating complex-valued IR expressions.
 */
typedef struct SimIREvaluatorComplex {
    SimIRFieldEvalFnComplex field_value_c; /**< Complex field resolver. */
    SimIRDiffEvalFnComplex differential_c; /**< Complex diff resolver. */
    SimIRNoiseEvalFnComplex noise_c;       /**< Complex noise resolver. */
    SimIRParamEvalFn param_value;          /**< Runtime parameter resolver. */
    void *userdata;                        /**< Caller-supplied callback context. */
} SimIREvaluatorComplex;

/**
 * @brief Callback surface for evaluating real-valued IR expressions.
 */
typedef struct SimIREvaluator {
    SimIRFieldEvalFn
        field_value; /**< Resolves field references. Assumed uniformly Lipschitz in user proofs. */
    SimIRDiffEvalFn
        differential; /**< Evaluates differential nodes under the linear-boundedness hypothesis. */
    SimIRNoiseEvalFn noise;       /**< Samples stochastic nodes with bounded second moment. */
    SimIRParamEvalFn param_value; /**< Runtime parameter resolver. */
    void *userdata; /**< Caller-supplied context carrying the corresponding constants. */
} SimIREvaluator;

/**
 * @brief Resolve a scalar-domain-aware field reference during IR evaluation.
 */
typedef SimResult (*SimIRFieldEvalFnDomain)(void *userdata, size_t field_id, SimIRType type,
                                            SimScalarDomain domain, SimIRDomainValue *out_value);

/**
 * @brief Evaluate a scalar-domain-aware differential node during IR evaluation.
 */
typedef SimResult (*SimIRDiffEvalFnDomain)(void *userdata, const SimIRBuilder *builder,
                                           const SimIRNode *node, SimIRNodeId operand,
                                           SimScalarDomain domain, SimIRDomainValue operand_value,
                                           SimIRDomainValue *out_value);

/**
 * @brief Sample a scalar-domain-aware stochastic node during IR evaluation.
 */
typedef SimResult (*SimIRNoiseEvalFnDomain)(void *userdata, const SimIRNode *node,
                                            SimScalarDomain domain, SimIRDomainValue *out_value);

/**
 * @brief Resolve a scalar-domain-aware runtime parameter during IR evaluation.
 */
typedef SimResult (*SimIRParamEvalFnDomain)(void *userdata, SimIRParamKind param,
                                            SimScalarDomain domain, SimIRDomainValue *out_value);

/**
 * @brief Callback surface for evaluating scalar-domain-aware IR expressions.
 */
typedef struct SimIREvaluatorDomain {
    SimIRFieldEvalFnDomain field_value; /**< Scalar-domain-aware field resolver. */
    SimIRDiffEvalFnDomain differential; /**< Scalar-domain-aware differential resolver. */
    SimIRNoiseEvalFnDomain noise;       /**< Scalar-domain-aware noise resolver. */
    SimIRParamEvalFnDomain param_value; /**< Scalar-domain-aware runtime parameter resolver. */
    void *userdata;                     /**< Caller-supplied context. */
} SimIREvaluatorDomain;

/**
 * @brief Evaluate an IR expression using a unified scalar-domain evaluator surface.
 *
 * This API selects real vs complex execution semantics from `SimIRType.scalar_domain`.
 * Existing `sim_ir_evaluate*` paths remain as compatibility wrappers/fast paths.
 */
SimResult sim_ir_evaluate_domain(const SimIRBuilder *builder, SimIRNodeId root,
                                 const SimIREvaluatorDomain *evaluator,
                                 SimIRDomainValue *out_value);

/**
 * @brief Evaluate an IR expression recursively using user-provided callbacks.
 */
SimResult sim_ir_evaluate(const SimIRBuilder *builder, SimIRNodeId root,
                          const SimIREvaluator *evaluator, double *out_value);

/**
 * @brief Evaluate an IR expression recursively using complex-valued callbacks.
 */
SimResult sim_ir_evaluate_complex(const SimIRBuilder *builder, SimIRNodeId root,
                                  const SimIREvaluatorComplex *evaluator,
                                  SimComplexDouble *out_value);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_KERNEL_IR_H */
