/**
 * @file operator_identity.h
 * @brief Core-owned operator identity metadata shared by execution and IR.
 */
#ifndef OAKFIELD_CORE_OPERATOR_IDENTITY_H
#define OAKFIELD_CORE_OPERATOR_IDENTITY_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Public classification for analytic warp operators.
 */
typedef enum SimWarpLevel {
    SIM_WARP_LEVEL_NONE = 0, /**< Not a warp or not classified. */
    SIM_WARP_LEVEL_LEVEL0,   /**< Pure geometric/diffeomorphic warp. */
    SIM_WARP_LEVEL_LEVEL1,   /**< Analytic monotonic warp. */
    SIM_WARP_LEVEL_LEVEL2    /**< Potentially non-monotonic or singular warp. */
} SimWarpLevel;

/**
 * @brief Semantic operator category opcodes for KernelIR nodes.
 */
typedef enum SimIROpcode {
    OAK_OP_DIFF = 0, /**< Differential operator category. */
    OAK_OP_CONV,     /**< Convolution operator category. */
    OAK_OP_DISP,     /**< Dispersion operator category. */
    OAK_OP_DIFFUSE,  /**< Diffusion/dissipation operator category. */
    OAK_OP_WARP,     /**< Analytic warp operator category. */
    OAK_OP_NOISE,    /**< Stochastic/noise operator category. */
    OAK_OP_FLOW,     /**< Flow/advection operator category. */
    OAK_OP_MISC,     /**< Miscellaneous operator category. */
    OAK_OP_CORE      /**< Core/runtime operator category. */
} SimIROpcode;

/**
 * @brief Return the canonical lowercase name for a semantic opcode.
 */
const char *sim_ir_opcode_name(SimIROpcode opcode);

/**
 * @brief Parse a semantic opcode name into its enum value.
 *
 * The parser accepts canonical names such as "diffuse" and optional
 * "oak_op_" prefixes.
 */
bool sim_ir_opcode_from_string(const char *text, SimIROpcode *out_opcode);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_CORE_OPERATOR_IDENTITY_H */
