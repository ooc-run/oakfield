/**
 * @file chaos_map.h
 * @brief Discrete chaotic map operator for real or complex state fields.
 */
#ifndef OAKFIELD_CHAOS_MAP_H
#define OAKFIELD_CHAOS_MAP_H

#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Supported chaotic map families.
 */
typedef enum SimChaosMapType {
    SIM_CHAOS_MAP_STANDARD = 0, /**< Chirikov standard map. */
    SIM_CHAOS_MAP_IKEDA,        /**< Ikeda map. */
    SIM_CHAOS_MAP_EXPONENTIAL,  /**< Exponential complex map. */
    SIM_CHAOS_MAP_QUADRATIC,    /**< Quadratic complex map. */
    SIM_CHAOS_MAP_HENON,        /**< Henon map. */
    SIM_CHAOS_MAP_LOZI,         /**< Lozi map. */
    SIM_CHAOS_MAP_TINKERBELL    /**< Tinkerbell map. */
} SimChaosMapType;

/**
 * @brief Standard map kick/drift ordering.
 */
typedef enum SimChaosKickMode {
    SIM_CHAOS_KICK_DRIFT = 0, /**< Apply kick before drift. */
    SIM_CHAOS_DRIFT_KICK,     /**< Apply drift before kick. */
    SIM_CHAOS_KICK_DRIFT_KICK /**< Symmetric kick-drift-kick ordering. */
} SimChaosKickMode;

/**
 * @brief Wrap mode for state components.
 */
typedef enum SimChaosWrapMode {
    SIM_CHAOS_WRAP_NONE = 0, /**< Leave state components unwrapped. */
    SIM_CHAOS_WRAP_PERIODIC, /**< Wrap state components periodically. */
    SIM_CHAOS_WRAP_CLAMP,    /**< Clamp state components to configured bounds. */
    SIM_CHAOS_WRAP_MIRROR    /**< Mirror state components at configured bounds. */
} SimChaosWrapMode;

/**
 * @brief Escape handling when state diverges.
 */
typedef enum SimChaosEscapeMode {
    SIM_CHAOS_ESCAPE_NONE = 0, /**< Do not apply escape handling. */
    SIM_CHAOS_ESCAPE_CLAMP,    /**< Clamp escaped state back into bounds. */
    SIM_CHAOS_ESCAPE_RESET,    /**< Reset escaped state to configured fallback. */
    SIM_CHAOS_ESCAPE_NAN       /**< Mark escaped state as NaN. */
} SimChaosEscapeMode;

/**
 * @brief Configuration for discrete chaotic map operators.
 *
 * The state is stored in a complex field: re = x, im = y (or momentum for standard map).
 */
typedef struct SimChaosMapOperatorConfig {
    size_t input_field;               /**< Input complex field index. */
    size_t output_field;              /**< Output complex field index. */
    SimChaosMapType map_type;         /**< Map family selection. */
    SimChaosKickMode kick_mode;       /**< Kick/drift ordering (standard map). */
    unsigned int iterations_per_step; /**< Iterations executed per simulation step. */
    double blend;                     /**< Blend new state with old (1 = full update). */

    /* Standard map parameters */
    double k;           /**< Standard map chaos parameter. */
    double angle_scale; /**< Scaling for the sine argument. */

    /* Ikeda map parameters */
    double ikeda_u;         /**< Ikeda contraction factor. */
    double ikeda_a;         /**< Ikeda phase bias. */
    double ikeda_b;         /**< Ikeda nonlinearity strength. */
    double ikeda_offset_re; /**< Ikeda offset real component. */
    double ikeda_offset_im; /**< Ikeda offset imaginary component. */

    /* Exponential map parameters */
    double exp_scale_re; /**< Exponential map scale real component. */
    double exp_scale_im; /**< Exponential map scale imaginary component. */
    double exp_c_re;     /**< Exponential map constant real component. */
    double exp_c_im;     /**< Exponential map constant imaginary component. */

    /* Quadratic map parameters */
    double quad_a_re; /**< Quadratic coefficient (real). */
    double quad_a_im; /**< Quadratic coefficient (imag). */
    double quad_b_re; /**< Linear coefficient (real). */
    double quad_b_im; /**< Linear coefficient (imag). */
    double quad_c_re; /**< Constant coefficient (real). */
    double quad_c_im; /**< Constant coefficient (imag). */

    /* Henon map parameters */
    double henon_a;         /**< Henon quadratic coefficient. */
    double henon_b;         /**< Henon coupling coefficient. */
    double henon_x_gain;    /**< Henon x linear gain. */
    double henon_y_gain;    /**< Henon y gain in x update. */
    double henon_offset_re; /**< Henon x offset. */
    double henon_offset_im; /**< Henon y offset. */

    /* Lozi map parameters */
    double lozi_a;           /**< Lozi absolute-value coefficient. */
    double lozi_b;           /**< Lozi coupling coefficient. */
    double lozi_x_gain;      /**< Lozi x linear gain. */
    double lozi_y_gain;      /**< Lozi y gain in x update. */
    double lozi_offset_re;   /**< Lozi x offset. */
    double lozi_offset_im;   /**< Lozi y offset. */
    double lozi_abs_epsilon; /**< Lozi absolute-value smoothing (0 = sharp). */

    /* Tinkerbell map parameters */
    double tinkerbell_a;         /**< Tinkerbell a coefficient. */
    double tinkerbell_b;         /**< Tinkerbell b coefficient. */
    double tinkerbell_c;         /**< Tinkerbell c coefficient. */
    double tinkerbell_d;         /**< Tinkerbell d coefficient. */
    double tinkerbell_x2_gain;   /**< Tinkerbell x^2 gain. */
    double tinkerbell_y2_gain;   /**< Tinkerbell y^2 gain. */
    double tinkerbell_xy_gain;   /**< Tinkerbell x*y gain. */
    double tinkerbell_offset_re; /**< Tinkerbell x offset. */
    double tinkerbell_offset_im; /**< Tinkerbell y offset. */

    /* Optional per-element parameter fields (SIZE_MAX disables). */
    size_t k_field; /**< Optional field for standard map K. */
    size_t u_field; /**< Optional field for Ikeda u. */
    size_t a_field; /**< Optional field for map parameter a. */
    size_t b_field; /**< Optional field for map parameter b. */
    size_t c_field; /**< Optional field for map parameter c. */
    size_t d_field; /**< Optional field for map parameter d. */

    /* Wrapping configuration */
    SimChaosWrapMode wrap_mode_re; /**< Wrap mode for real component. */
    SimChaosWrapMode wrap_mode_im; /**< Wrap mode for imaginary component. */
    double wrap_min_re;            /**< Wrap minimum for real component. */
    double wrap_max_re;            /**< Wrap maximum for real component. */
    double wrap_min_im;            /**< Wrap minimum for imaginary component. */
    double wrap_max_im;            /**< Wrap maximum for imaginary component. */

    /* Escape handling */
    SimChaosEscapeMode escape_mode; /**< Escape behavior when |z| exceeds escape_radius. */
    double escape_radius;           /**< Divergence radius (<=0 disables). */
    double escape_reset_re;         /**< Reset real component. */
    double escape_reset_im;         /**< Reset imaginary component. */
} SimChaosMapOperatorConfig;

/**
 * @brief Register a chaos map operator with the provided configuration.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional chaos map configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from argument validation,
 *         field compatibility checks, allocation, or split registration.
 */
SimResult sim_add_chaos_map_operator(struct SimContext *context,
                                     const SimChaosMapOperatorConfig *config, size_t *out_index);

/**
 * @brief Retrieve the configuration currently bound to a chaos map operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_chaos_map_operator().
 * @param[out] out_config Receives the normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no chaos-map state.
 */
SimResult sim_chaos_map_config(struct SimContext *context, size_t operator_index,
                               SimChaosMapOperatorConfig *out_config);

/**
 * @brief Update an existing chaos map operator in-place.
 *
 * Passing NULL for @p config keeps the current configuration and reapplies
 * normalization. A successful update refreshes symbolic metadata.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the chaos map operator to update.
 * @param config Optional replacement chaos map configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for a NULL
 *         context, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no chaos-map state.
 */
SimResult sim_chaos_map_update(struct SimContext *context, size_t operator_index,
                               const SimChaosMapOperatorConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_CHAOS_MAP_H */
