/**
 * @file graph_ir.h
 * @brief Graph-level IR for composing pointwise kernels with FFT transforms.
 */
#ifndef OAKFIELD_GRAPH_IR_H
#define OAKFIELD_GRAPH_IR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "oakfield/field.h"
#include "oakfield/operator.h"

#ifdef __cplusplus
extern "C"
{
#endif

    struct SimContext;

    /** Identifier for a GraphIR node. */
    typedef size_t SimGraphIRNodeId;

/** Sentinel value for invalid node IDs. */
#define SIM_GRAPH_IR_INVALID_NODE ((SimGraphIRNodeId)SIZE_MAX)

    /**
     * @brief GraphIR node kinds.
     */
    typedef enum SimGraphIRNodeKind
    {
        SIM_GRAPH_IR_NODE_POINTWISE_KERNEL = 0, /**< Pointwise KernelIR execution. */
        SIM_GRAPH_IR_NODE_FFT_FORWARD,          /**< FFT forward (physical -> spectral). */
        SIM_GRAPH_IR_NODE_FFT_INVERSE,          /**< FFT inverse (spectral -> physical). */
        SIM_GRAPH_IR_NODE_PROMOTE_COMPLEX,      /**< Promote real storage to complex. */
        SIM_GRAPH_IR_NODE_CAST_COPY,            /**< Copy/cast between compatible fields. */
        SIM_GRAPH_IR_NODE_CANONICALIZE_REAL_CONSTRAINT /**< Enforce Hermitian symmetry. */
    } SimGraphIRNodeKind;

    /**
     * @brief GraphIR input reference kinds.
     */
    typedef enum SimGraphIRInputKind
    {
        SIM_GRAPH_IR_INPUT_NONE = 0,
        SIM_GRAPH_IR_INPUT_FIELD,
        SIM_GRAPH_IR_INPUT_NODE
    } SimGraphIRInputKind;

    /**
     * @brief GraphIR output reference kinds.
     */
    typedef enum SimGraphIROutputKind
    {
        SIM_GRAPH_IR_OUTPUT_NONE = 0,
        SIM_GRAPH_IR_OUTPUT_FIELD,
        SIM_GRAPH_IR_OUTPUT_TEMP
    } SimGraphIROutputKind;

    /**
     * @brief GraphIR input reference.
     */
    typedef struct SimGraphIRInputRef
    {
        SimGraphIRInputKind kind;
        size_t field_index;
        SimGraphIRNodeId node_id;
    } SimGraphIRInputRef;

    /**
     * @brief GraphIR output reference.
     */
    typedef struct SimGraphIROutputRef
    {
        SimGraphIROutputKind kind;
        size_t field_index;
    } SimGraphIROutputRef;

    /**
     * @brief Representation requirements for a node input.
     */
    typedef struct SimGraphIRInputRequirement
    {
        SimFieldDomain domain;
        SimFieldValueKind value_kind;
    } SimGraphIRInputRequirement;

    /**
     * @brief Representation guarantees for a node output.
     */
    typedef struct SimGraphIROutputGuarantee
    {
        SimFieldDomain domain;
        SimFieldValueKind value_kind;
        bool preserves_real_subspace;
        bool preserves_real_constraint;
    } SimGraphIROutputGuarantee;

    /**
     * @brief Time semantics for a node.
     */
    typedef enum SimGraphIRTimeSource
    {
        SIM_GRAPH_IR_TIME_NONE = 0,
        SIM_GRAPH_IR_TIME_PARAM,
        SIM_GRAPH_IR_TIME_STEP_PURE,
        SIM_GRAPH_IR_TIME_ACCUMULATED
    } SimGraphIRTimeSource;

    /**
     * @brief Timestep sourcing for a node.
     */
    typedef enum SimGraphIRDTSrc
    {
        SIM_GRAPH_IR_DT_NONE = 0,
        SIM_GRAPH_IR_DT_PARAM,
        SIM_GRAPH_IR_DT_NOMINAL
    } SimGraphIRDTSrc;

    /**
     * @brief Purity/rewind semantics for a node.
     */
    typedef struct SimGraphIRPurity
    {
        SimGraphIRTimeSource time_source;
        SimGraphIRDTSrc dt_source;
        bool has_state;
        bool reset_on_rewind;
    } SimGraphIRPurity;

    /**
     * @brief Full representation contract for a GraphIR node.
     */
    typedef struct SimGraphIRRepresentationContract
    {
        SimGraphIRInputRequirement input;
        SimGraphIROutputGuarantee output;
        SimGraphIRPurity purity;
    } SimGraphIRRepresentationContract;

    /**
     * @brief Descriptor for a pointwise kernel node.
     */
    typedef struct SimGraphIRPointwiseDescriptor
    {
        const SimOperatorKernelDescriptor *kernel;
    } SimGraphIRPointwiseDescriptor;

    /**
     * @brief Descriptor for canonicalizing Hermitian symmetry.
     */
    typedef struct SimGraphIRCanonicalizeDescriptor
    {
        double tolerance;
    } SimGraphIRCanonicalizeDescriptor;

    /**
     * @brief Node descriptor used for graph construction.
     */
    typedef struct SimGraphIRNodeDesc
    {
        SimGraphIRNodeKind kind;
        SimGraphIRInputRef input;
        SimGraphIROutputRef output;
        SimGraphIRRepresentationContract contract;
        union
        {
            SimGraphIRPointwiseDescriptor pointwise;
            SimGraphIRCanonicalizeDescriptor canonicalize;
        } config;
    } SimGraphIRNodeDesc;

    /**
     * @brief Read-only node view exposed for GraphIR inspection.
     */
    typedef struct SimGraphIRNodeView
    {
        SimGraphIRNodeKind kind;
        SimGraphIRInputRef input;
        SimGraphIROutputRef output;
        SimGraphIRRepresentationContract contract;
        size_t pointwise_binding_count;
        size_t pointwise_output_count;
        size_t pointwise_param_count;
        bool   has_canonicalize_tolerance;
        double canonicalize_tolerance;
    } SimGraphIRNodeView;

    /**
     * @brief GraphIR edge connecting two nodes.
     */
    typedef struct SimGraphIREdge
    {
        SimGraphIRNodeId from;
        SimGraphIRNodeId to;
    } SimGraphIREdge;

    /**
     * @brief Graph-level IR storage.
     */
    typedef struct SimGraphIR
    {
        struct SimGraphIRNode *nodes;
        size_t node_count;
        size_t node_capacity;

        SimGraphIREdge *edges;
        size_t edge_count;
        size_t edge_capacity;

        void (*rewind_fn)(struct SimGraphIR *graph, size_t step_index, void *userdata);
        void *rewind_userdata;
    } SimGraphIR;

    /**
     * @brief Compilation/execution context for GraphIR.
     */
    typedef struct SimGraphIRCompileContext
    {
        struct SimContext *context;
        SimRepresentationMode representation_mode;
        bool exploration_mode;
    } SimGraphIRCompileContext;

    /**
     * @brief Initialize a GraphIR instance.
     */
    void sim_graph_ir_init(SimGraphIR *graph);

    /**
     * @brief Destroy a GraphIR instance and release resources.
     */
    void sim_graph_ir_destroy(SimGraphIR *graph);

    /**
     * @brief Add a node to the GraphIR graph.
     */
    SimResult sim_graph_ir_add_node(SimGraphIR *graph,
                                    const SimGraphIRNodeDesc *desc,
                                    SimGraphIRNodeId *out_node_id);

    /**
     * @brief Add an edge between two nodes.
     */
    SimResult sim_graph_ir_add_edge(SimGraphIR *graph, SimGraphIRNodeId from, SimGraphIRNodeId to);

    /**
     * @brief Fuse adjacent pointwise kernel nodes when compatible.
     */
    SimResult sim_graph_ir_fuse_pointwise(SimGraphIR *graph);

    /**
     * @brief Validate a GraphIR graph against the provided context/mode.
     */
    SimResult sim_graph_ir_validate(SimGraphIR *graph, const SimGraphIRCompileContext *ctx);

    /**
     * @brief Execute a GraphIR graph.
     */
    SimResult sim_graph_ir_execute(SimGraphIR *graph, const SimGraphIRCompileContext *ctx);

    /**
     * @brief Notify GraphIR of a rewind event and reset stateful nodes as needed.
     */
    void sim_graph_ir_notify_rewind(SimGraphIR *graph, size_t step_index);

    /**
     * @brief Configure a graph-level rewind callback.
     */
    void sim_graph_ir_set_rewind_callback(SimGraphIR *graph,
                                          void (*rewind_fn)(SimGraphIR *graph, size_t step_index, void *userdata),
                                          void *userdata);

    /**
     * @brief Initialize a compile context from an existing simulation context.
     */
    void sim_graph_ir_compile_context_init(SimGraphIRCompileContext *ctx, struct SimContext *context);

    /**
     * @brief Return a stable string name for a GraphIR node kind.
     */
    const char *sim_graph_ir_node_kind_name(SimGraphIRNodeKind kind);

    /**
     * @brief Return a stable string name for a GraphIR input kind.
     */
    const char *sim_graph_ir_input_kind_name(SimGraphIRInputKind kind);

    /**
     * @brief Return a stable string name for a GraphIR output kind.
     */
    const char *sim_graph_ir_output_kind_name(SimGraphIROutputKind kind);

    /**
     * @brief Return a stable string name for a GraphIR time source.
     */
    const char *sim_graph_ir_time_source_name(SimGraphIRTimeSource source);

    /**
     * @brief Return a stable string name for a GraphIR dt source.
     */
    const char *sim_graph_ir_dt_source_name(SimGraphIRDTSrc source);

    /**
     * @brief Return the number of nodes recorded in a GraphIR graph.
     */
    size_t sim_graph_ir_node_count(const SimGraphIR *graph);

    /**
     * @brief Return the number of edges recorded in a GraphIR graph.
     */
    size_t sim_graph_ir_edge_count(const SimGraphIR *graph);

    /**
     * @brief Read a GraphIR node view by node id.
     *
     * @return true when the requested node exists.
     */
    bool sim_graph_ir_node_view(const SimGraphIR *graph,
                                SimGraphIRNodeId  node_id,
                                SimGraphIRNodeView *out_view);

    /**
     * @brief Read a GraphIR edge by edge index.
     *
     * @return true when the requested edge exists.
     */
    bool sim_graph_ir_edge_view(const SimGraphIR *graph,
                                size_t            edge_index,
                                SimGraphIREdge   *out_edge);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_GRAPH_IR_H */
