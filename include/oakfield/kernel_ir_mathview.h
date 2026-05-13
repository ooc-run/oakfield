/**
 * @file kernel_ir_mathview.h
 * @brief MathView rendering helpers for KernelIR graphs.
 */
#ifndef OAKFIELD_KERNEL_IR_MATHVIEW_H
#define OAKFIELD_KERNEL_IR_MATHVIEW_H

#include "kernel_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SIM_IR_MATHVIEW_SCHEMA_VERSION "1.0"
#define SIM_IR_MATHVIEW_COMPLEX_SEMANTICS "cartesian"
#define SIM_IR_MATHVIEW_COMPLEX_BRANCH "principal"

/**
 * @brief Render a KernelIR expression into a canonical MathView string.
 *
 * The canonical string includes a schema header with complex semantics.
 *
 * @param builder Builder containing the referenced nodes.
 * @param root Root node identifier.
 * @param buffer Output buffer (may be NULL to query required length).
 * @param capacity Capacity of @p buffer in bytes.
 * @param[out] out_length Receives the required length (without NUL) when non-NULL.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_OUT_OF_MEMORY if @p buffer is too small.
 */
SimResult sim_ir_mathview_render(const SimIRBuilder *builder, SimIRNodeId root, char *buffer,
                                 size_t capacity, size_t *out_length);

/**
 * @brief Render a KernelIR expression into a Math IR JSON AST.
 *
 * @param builder Builder containing the referenced nodes.
 * @param root Root node identifier.
 * @param buffer Output buffer (may be NULL to query required length).
 * @param capacity Capacity of @p buffer in bytes.
 * @param[out] out_length Receives the required length (without NUL) when non-NULL.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_OUT_OF_MEMORY if @p buffer is too small.
 */
SimResult sim_ir_mathview_render_json(const SimIRBuilder *builder, SimIRNodeId root, char *buffer,
                                      size_t capacity, size_t *out_length);

/**
 * @brief Render a KernelIR expression into a LaTeX outline string.
 *
 * @param builder Builder containing the referenced nodes.
 * @param root Root node identifier.
 * @param buffer Output buffer (may be NULL to query required length).
 * @param capacity Capacity of @p buffer in bytes.
 * @param[out] out_length Receives the required length (without NUL) when non-NULL.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_OUT_OF_MEMORY if @p buffer is too small.
 */
SimResult sim_ir_mathview_render_latex(const SimIRBuilder *builder, SimIRNodeId root, char *buffer,
                                       size_t capacity, size_t *out_length);

/**
 * @brief Hash a KernelIR expression based on its canonical MathView string.
 *
 * @param builder Builder containing the referenced nodes.
 * @param root Root node identifier.
 * @param[out] out_hash Receives the computed 64-bit hash.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_ir_mathview_hash(const SimIRBuilder *builder, SimIRNodeId root, uint64_t *out_hash);

/**
 * @brief Hash a KernelIR expression based on its canonical MathView string.
 *
 * Uses SHA-256 over the canonical string (including header).
 *
 * @param builder Builder containing the referenced nodes.
 * @param root Root node identifier.
 * @param[out] out_hash Receives the 32-byte SHA-256 digest.
 * @return #SIM_RESULT_OK on success.
 */
SimResult sim_ir_mathview_hash_sha256(const SimIRBuilder *builder, SimIRNodeId root,
                                      unsigned char out_hash[32]);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_KERNEL_IR_MATHVIEW_H */
