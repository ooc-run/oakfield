/**
 * @file spectral_lines.h
 * @brief Spectral-line stimulus: pure frequency spikes and multi-line harmonics.
 *
 * Generates sums of spatial harmonics with optional temporal oscillation.
 * For complex fields this yields single-sided spectral lines; for real fields
 * the spectrum is Hermitian-symmetric.
 */
#ifndef OAKFIELD_STIMULUS_SPECTRAL_LINES_H
#define OAKFIELD_STIMULUS_SPECTRAL_LINES_H

#include "coords.h"
#include "oakfield/operator_split.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SimContext;

/**
 * @brief Harmonic twist strategy for spectral-line sums.
 */
typedef enum SimStimulusSpectralLinesTwistKind {
    SIM_SPECTRAL_LINES_TWIST_NONE = 0,    /**< No harmonic twist; χ(n)=1. */
    SIM_SPECTRAL_LINES_TWIST_ALTERNATING, /**< Alternating sign twist; χ(n)=(-1)^n. */
    SIM_SPECTRAL_LINES_TWIST_DIRICHLET    /**< Dirichlet-character-style twist χ(n mod q). */
} SimStimulusSpectralLinesTwistKind;

/**
 * @brief Preset table family used for Dirichlet-style spectral-line twists.
 */
typedef enum SimStimulusSpectralLinesTwistPreset {
    SIM_SPECTRAL_LINES_TWIST_PRESET_PRINCIPAL = 0, /**< Principal character modulo q. */
    SIM_SPECTRAL_LINES_TWIST_PRESET_CHI4,          /**< Real χ4 character (odd residues mod 4). */
    SIM_SPECTRAL_LINES_TWIST_PRESET_QUADRATIC,     /**< Quadratic/Jacobi character (odd q). */
    SIM_SPECTRAL_LINES_TWIST_PRESET_TABLE          /**< Residue table χ[r], r in [0, q). */
} SimStimulusSpectralLinesTwistPreset;

/**
 * @brief Configuration for harmonic spectral-line stimulus fields.
 */
typedef struct SimStimulusSpectralLinesConfig {
    size_t field_index;           /**< Target field index. */
    double amplitude;             /**< Base amplitude for the first harmonic. */
    double wavenumber;            /**< Base spatial wavenumber k0 (rad / unit). */
    double kx;                    /**< Optional wavevector X component (rad / unit). */
    double ky;                    /**< Optional wavevector Y component (rad / unit). */
    double omega;                 /**< Base temporal frequency ω0 (rad / s). */
    double phase;                 /**< Base phase offset φ0 (radians). */
    SimStimulusCoordConfig coord; /**< Spatial coordinate mapping configuration. */
    double time_offset;           /**< Additional time shift applied before evaluation. */
    double nominal_dt;            /**< Nominal dt when fixed_clock is enabled. */
    unsigned int harmonic_count;  /**< Number of harmonics (>= 1). */
    double harmonic_power;        /**< Power-law exponent p in 1 / n^p amplitude scaling. */
    SimStimulusSpectralLinesTwistKind twist_kind; /**< Harmonic twist mode. */
    unsigned int twist_q;                         /**< Modulus q for DIRICHLET twist mode. */
    unsigned int twist_k; /**< Optional character index (Mathematica-style). */
    SimStimulusSpectralLinesTwistPreset twist_preset; /**< DIRICHLET preset selector. */
    bool twist_zero_non_units;   /**< Zero χ on non-units for table mode (gcd(n,q) != 1). */
    bool twist_table_is_complex; /**< When true, χ-table imaginary values are used. */
    bool use_wavevector;         /**< When true, use (kx,ky) instead of wavenumber+coord. */
    bool fixed_clock;            /**< Lock evolution to nominal_dt instead of adaptive dt. */
    bool scale_by_dt;            /**< Scale writes by dt when true; else dt-independent. */
} SimStimulusSpectralLinesConfig;

/**
 * @brief Register a spectral-line harmonic stimulus operator.
 *
 * The implementation copies and normalizes @p config, prepares any harmonic
 * twist coefficients, and registers the operator on the target field.
 *
 * @param context Simulation context that will own the operator.
 * @param config Optional spectral-lines configuration; NULL selects normalized defaults.
 * @param[out] out_index Optional destination for the registered operator index.
 * @return #SIM_RESULT_OK on success, or an error code from validation, allocation,
 *         coefficient setup, or split-operator registration.
 */
SimResult sim_add_stimulus_spectral_lines_operator(struct SimContext *context,
                                                   const SimStimulusSpectralLinesConfig *config,
                                                   size_t *out_index);

/**
 * @brief Copy the current spectral-lines configuration from a registered operator.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index returned by sim_add_stimulus_spectral_lines_operator().
 * @param[out] out_config Receives the operator's normalized configuration.
 * @return #SIM_RESULT_OK on success, #SIM_RESULT_INVALID_ARGUMENT for NULL
 *         pointers, #SIM_RESULT_NOT_FOUND for a missing operator, or
 *         #SIM_RESULT_INVALID_STATE when the operator has no stimulus state.
 */
SimResult sim_stimulus_spectral_lines_config(struct SimContext *context, size_t operator_index,
                                             SimStimulusSpectralLinesConfig *out_config);

/**
 * @brief Replace or renormalize a registered spectral-lines configuration.
 *
 * Passing NULL for @p config keeps the existing configuration and reapplies
 * normalization. A successful update refreshes twist coefficients as needed and
 * invalidates the scheduler plan.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the spectral-lines operator to update.
 * @param config Optional replacement configuration.
 * @return #SIM_RESULT_OK on success, or an error code if lookup, coefficient
 *         setup, or state validation fails.
 */
SimResult sim_stimulus_spectral_lines_update(struct SimContext *context, size_t operator_index,
                                             const SimStimulusSpectralLinesConfig *config);

/**
 * @brief Install an explicit Dirichlet-style twist table on a spectral-lines operator.
 *
 * The real table @p chi_re is required and must contain @p q entries. The
 * imaginary table @p chi_im is optional; when supplied, complex twists are used.
 * A successful call replaces the operator's twist table and invalidates cached
 * coefficients.
 *
 * @param context Simulation context containing the operator.
 * @param operator_index Index of the spectral-lines operator to update.
 * @param q Modulus and table length; must be greater than zero.
 * @param chi_re Real twist values indexed by residue class.
 * @param chi_im Optional imaginary twist values indexed by residue class.
 * @param zero_non_units When true, residues with gcd(residue, q) != 1 are zeroed.
 * @return #SIM_RESULT_OK on success, or an error code if arguments, lookup,
 *         allocation, or coefficient setup fails.
 */
SimResult sim_stimulus_spectral_lines_set_twist_table(struct SimContext *context,
                                                      size_t operator_index, unsigned int q,
                                                      const double *chi_re, const double *chi_im,
                                                      bool zero_non_units);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_STIMULUS_SPECTRAL_LINES_H */
