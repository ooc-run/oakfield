/**
 * @file special_functions.h
 * @brief Analytic special functions, finite ladder helpers, and q-analogs.
 * @ingroup oakfield_special_math
 *
 * Implementations:
 *  - Real & complex, float & double precision.
 *  - Daily drivers: fixed 12/7/5-term Stirling tails.
 *  - Accuracy-driven: tail runs until |next term| < tol.
 *  - Speedy: Mortici-style log-shift rational approximants (few flops).
 *  - Finite ladder φ and q-φ via digamma/trigamma differences.
 *
 * Math references:
 *  - Asymptotic expansions and reflection: NIST DLMF §5.11 and §5.15.
 *  - Reflection identities: ψ(1−z)−ψ(z)=π cot(πz),  ψ₁(1−z)+ψ₁(z)=π² csc²(πz).
 *
 * Implementation notes:
 *  - Real reflection avoids libm trig: custom sinpi/cospi via quadrant reduction and
 *    polynomial kernels with FMA.
 *  - Complex path uses stable exponentials for sin(πz)/cos(πz) to avoid overflow.
 *  - Optional MPFR/MPC backends (define SIM_HAVE_MPFR/SIM_HAVE_MPC) provide true
 *    arbitrary precision; otherwise, "AP" variants run the Bernoulli tail to tolerance
 *    in long double / double complex.
 */

#ifndef SIM_MATH_SPECIAL_FUNCTIONS_H
#define SIM_MATH_SPECIAL_FUNCTIONS_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oakfield/field.h" /* expected to define SimComplexDouble; we provide fallbacks below */

/* Compile-time diagnostics toggle: enable lightweight sampling of
 * special-function health metrics (reflection usage, Stirling tails, etc.).
 * Can be overridden by defining SIM_DIAGNOSTICS at compile time. */
#ifndef SIM_DIAGNOSTICS
#define SIM_DIAGNOSTICS 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SIM_HAVE_SIMCOMPLEXFLOAT
/** @brief Fallback ABI-compatible complex float if not in field.h */
typedef struct {
    float re, im; /**< Real and imaginary components. */
} SimComplexFloat;
#endif

#ifndef SIM_HAVE_SIMCOMPLEXDOUBLE
/** @brief Fallback ABI-compatible complex double if not in field.h */
typedef struct {
    double re, im; /**< Real and imaginary components. */
} SimComplexDouble;
#endif

/**
 * @brief Backend choice for digamma/trigamma evaluation.
 */
typedef enum SimDigammaBackend {
    SIM_DIGAMMA_BACKEND_12_TAIL  = 0, /**< 12-term Stirling tail digamma/trigamma. */
    SIM_DIGAMMA_BACKEND_7_TAIL   = 1, /**< 7-term Stirling tail digamma/trigamma. */
    SIM_DIGAMMA_BACKEND_5_TAIL   = 2, /**< 5-term Stirling tail digamma/trigamma. */
    SIM_DIGAMMA_BACKEND_ADAPTIVE = 3, /**< Adaptive-tail digamma/trigamma. */
    SIM_DIGAMMA_BACKEND_MORTICI  = 4, /**< Mortici speedy digamma/trigamma. */
} SimDigammaBackend;

#ifndef SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT
#define SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT 0.25
#endif

/*==============================================================================
 *                           DIGAMMA (psi)
 *============================================================================*/

/** @name Digamma daily drivers (real, double)
 *  @brief ψ(x) using Stirling tail truncated after N=12,7,5 Bernoulli pairs.
 *  @note Reflection + recurrence handle x<=0; asymptotics for large x.
 *  @sa DLMF 5.11 & 5.15 for expansions and reflection.
 *  @{ */
double sim_digamma_f64_12(double x);
double sim_digamma_f64_7(double x);
double sim_digamma_f64_5(double x);
/** @} */

/** @name Digamma daily drivers (real, float) */
float sim_digamma_f32_12(float x);
float sim_digamma_f32_7(float x);
float sim_digamma_f32_5(float x);

/** @name Digamma (complex, double) — analytic continuation */
SimComplexDouble sim_digamma_c64_12(SimComplexDouble z);
SimComplexDouble sim_digamma_c64_7(SimComplexDouble z);
SimComplexDouble sim_digamma_c64_5(SimComplexDouble z);

/** @name Digamma (complex, float) — analytic continuation */
SimComplexFloat sim_digamma_c32_12(SimComplexFloat z);
SimComplexFloat sim_digamma_c32_7(SimComplexFloat z);
SimComplexFloat sim_digamma_c32_5(SimComplexFloat z);

/**
 * @brief Speedy Mortici-style digamma (complex, double).
 * @note Reflection + recurrence for stability; mirrors the real Mortici coefficients.
 */
SimComplexDouble sim_digamma_c64_mortici(SimComplexDouble z);
SimComplexFloat  sim_digamma_c32_mortici(SimComplexFloat z);

/**
 * @brief Digamma with adaptive tail: run Bernoulli series until |next term| < tol.
 * @param x   Real argument (double).
 * @param tol Absolute tolerance on next Bernoulli term (e.g., 1e-16).
 * @return ψ(x) with tail stopped adaptively.
 */
double sim_digamma_f64_tail(double x, double tol);

/**
 * @brief Complex digamma with adaptive tail.
 * @param z   Complex argument (double complex ABI).
 * @param tol Absolute tolerance on next Bernoulli term.
 */
SimComplexDouble sim_digamma_c64_tail(SimComplexDouble z, double tol);

/**
 * @brief Speedy Mortici-style digamma (real, double).
 * @details For x>0 uses ψ(x) ≈ log(x + δ) − 1/(2x) with
 *          δ = 1/2 + 1/(24x) − 1/(48x²) + 23/(5760 x³),
 *          matching terms through O(x⁻⁴) in the asymptotics; reflection+recurrence elsewhere.
 * @note Designed for throughput over last‑bit accuracy; use `sim_digamma_f64_12` for best accuracy.
 */
double sim_digamma_f64_mortici(double x);

/** @brief Speedy Mortici-style digamma (real, float). */
float sim_digamma_f32_mortici(float x);

/* Backward-compatible aliases matching your current API (default = 12-tail, double): */
double           sim_special_digamma(double x);
SimComplexDouble sim_special_digamma_complex(SimComplexDouble z);

/*==============================================================================
 *                      DIGAMMA WAVEFORM HELPERS (SQUARE)
 *============================================================================*/

/**
 * @brief Legacy digamma-square base with the quarter-shift deformation.
 * @param amplitude     Output amplitude multiplier.
 * @param inner_radians Normalized carrier sample u.
 * @param backend       Digamma backend selector.
 * @param tolerance     Adaptive-tail tolerance when backend=adaptive.
 */
double sim_digamma_square_base_real(double            amplitude,
                                    double            inner_radians,
                                    SimDigammaBackend backend,
                                    double            tolerance);

/**
 * @brief Deformable digamma-square base using notebook-style shift parameter a.
 * @details Computes
 *          A * (1 + (cos(inner) - cos(2*pi*a)) * (psi(a+u) - psi(1-a+u)) / pi),
 *          where inner = 2*pi*u.
 */
double sim_digamma_square_base_deformed_real(double            amplitude,
                                             double            a,
                                             double            inner_radians,
                                             SimDigammaBackend backend,
                                             double            tolerance);

SimComplexDouble sim_digamma_square_base_complex(double            amplitude,
                                                 SimComplexDouble  inner_radians,
                                                 SimDigammaBackend backend,
                                                 double            tolerance);

SimComplexDouble sim_digamma_square_base_deformed_complex(double            amplitude,
                                                          double            a,
                                                          SimComplexDouble  inner_radians,
                                                          SimDigammaBackend backend,
                                                          double            tolerance);

/*==============================================================================
 *                           TRIGAMMA (psi_1)
 *============================================================================*/

/** @name Trigamma daily drivers (real, double) */
double sim_trigamma_f64_12(double x);
double sim_trigamma_f64_7(double x);
double sim_trigamma_f64_5(double x);

/** @name Trigamma daily drivers (real, float) */
float sim_trigamma_f32_12(float x);
float sim_trigamma_f32_7(float x);
float sim_trigamma_f32_5(float x);

/** @name Trigamma (complex, double) — analytic continuation */
SimComplexDouble sim_trigamma_c64_12(SimComplexDouble z);
SimComplexDouble sim_trigamma_c64_7(SimComplexDouble z);
SimComplexDouble sim_trigamma_c64_5(SimComplexDouble z);

/** @name Trigamma (complex, float) — analytic continuation */
SimComplexFloat sim_trigamma_c32_12(SimComplexFloat z);
SimComplexFloat sim_trigamma_c32_7(SimComplexFloat z);
SimComplexFloat sim_trigamma_c32_5(SimComplexFloat z);

/**
 * @brief Trigamma with adaptive tail (real).
 * @param x   Real argument (double).
 * @param tol Absolute tolerance on next Bernoulli term.
 */
double sim_trigamma_f64_tail(double x, double tol);

/**
 * @brief Trigamma with adaptive tail (complex).
 * @param z   Complex argument (double ABI).
 * @param tol Absolute tolerance on next Bernoulli term.
 */
SimComplexDouble sim_trigamma_c64_tail(SimComplexDouble z, double tol);

/**
 * @brief Speedy trigamma (real, double): ψ₁(x) ≈ 1/x + 1/(2x²) + 1/(6x³) for x>0,
 *        with reflection+recurrence elsewhere. Matches asymptotic coefficients to O(x⁻³).
 */
double sim_trigamma_f64_mortici(double x);

/** @brief Speedy trigamma (real, float). */
float sim_trigamma_f32_mortici(float x);

#define sim_digamma_f64_speedy sim_digamma_f64_mortici
#define sim_digamma_f32_speedy sim_digamma_f32_mortici
#define sim_trigamma_f64_speedy sim_trigamma_f64_mortici
#define sim_trigamma_f32_speedy sim_trigamma_f32_mortici

/* Backward-compatible aliases matching your current API (default = 12-tail, double): */
double           sim_special_trigamma(double x);
SimComplexDouble sim_special_trigamma_complex(SimComplexDouble z);

/*==============================================================================
 *                           TETRAGAMMA (psi_2)
 *============================================================================*/

/** @name Tetragamma daily drivers (real, double) */
double sim_tetragamma_f64_12(double x);
double sim_tetragamma_f64_7(double x);
double sim_tetragamma_f64_5(double x);

/** @name Tetragamma daily drivers (real, float) */
float sim_tetragamma_f32_12(float x);
float sim_tetragamma_f32_7(float x);
float sim_tetragamma_f32_5(float x);

/** @name Tetragamma (complex, double) — analytic continuation */
SimComplexDouble sim_tetragamma_c64_12(SimComplexDouble z);
SimComplexDouble sim_tetragamma_c64_7(SimComplexDouble z);
SimComplexDouble sim_tetragamma_c64_5(SimComplexDouble z);

/** @name Tetragamma (complex, float) — analytic continuation */
SimComplexFloat sim_tetragamma_c32_12(SimComplexFloat z);
SimComplexFloat sim_tetragamma_c32_7(SimComplexFloat z);
SimComplexFloat sim_tetragamma_c32_5(SimComplexFloat z);

/* Backward-compatible aliases */
double           sim_special_tetragamma(double x);
SimComplexDouble sim_special_tetragamma_complex(SimComplexDouble z);

/*==============================================================================
 *                 GENERALIZED HARMONIC NUMBERS (MGHN CORE)
 *============================================================================*/

/**
 * @brief Generalized harmonic ladder H_K(a) = Σ_{k=0}^{K-1} 1/(a+k).
 * @details Analytic continuation: H_K(a) = ψ(a+K) − ψ(a).
 *          For large a with small K, uses a stable asymptotic difference
 *          (log1p + Bernoulli tail) to avoid cancellation.
 * @param a  Real shift (typically a = λ + ε). Requires a>0.
 * @param K  Ladder depth (K>=0). For K==0 returns 0.
 */
double sim_ghn_HK(double a, int K);

/** @brief First derivative H'_K(a) = ψ₁(a+K) − ψ₁(a). (Negative for a>0,K>0.) */
double sim_ghn_HK_d1(double a, int K);

/** @brief Second derivative H''_K(a) = ψ₂(a+K) − ψ₂(a). (Positive for a>0,K>0.) */
double sim_ghn_HK_d2(double a, int K);

/** @brief Complex generalized harmonic ladder using ψ difference. */
SimComplexDouble sim_ghn_HK_complex(SimComplexDouble a, int K);

/** @brief Complex first derivative using trigamma difference. */
SimComplexDouble sim_ghn_HK_d1_complex(SimComplexDouble a, int K);

/** @brief Complex second derivative using tetragamma difference. */
SimComplexDouble sim_ghn_HK_d2_complex(SimComplexDouble a, int K);

/*==============================================================================
 *                 FINITE LADDER φ AND q-DEFORMED φ
 *============================================================================*/

/**
 * @brief Finite ladder helper φ(λ,ε;K) = Σ_{k=0}^{K-1} λ/(λ+ε+k).
 * @details Numerically stabilized via ψ-difference:
 *          φ = λ [ ψ(λ+ε+K) − ψ(λ+ε) ]  (valid ∀ complex λ,ε with poles excluded).
 *          This removes inner loops, improves accuracy, and exposes derivatives.
 */
double sim_hyperexp_phi(double lambda, double epsilon, int K);

/** @brief ∂φ/∂λ = [ψ(λ+ε+K) − ψ(λ+ε)] + λ[ψ₁(λ+ε+K) − ψ₁(λ+ε)]. */
double sim_hyperexp_phi_deriv(double lambda, double epsilon, int K);

/** @brief Complex φ using the same ψ-difference identity. */
SimComplexDouble sim_hyperexp_phi_complex(SimComplexDouble lambda, SimComplexDouble epsilon, int K);

/**
 * @brief q-deformed φ_q(λ,ε;K,q) := Σ_{k=0}^{K-1} λ / (λ + ε q^k).
 * @ingroup oakfield_experimental_q
 * @details A geometric (q-analogue) ladder; derivative wrt λ:
 *          ∂φ_q/∂λ = Σ ε q^k / (λ + ε q^k)^2.
 * @note Choose q∈(0,1) for decaying ladder; q>1 allowed but can amplify roundoff.
 * @warning Experimental API. q-method domains, convergence behavior, and
 * fallback policy may change before the math surface is stable.
 */
double sim_qhyperexp_phi(double lambda, double epsilon, int K, double q);
double sim_qhyperexp_phi_deriv(double lambda, double epsilon, int K, double q);

/**
 * @brief Experimental complex q-φ and derivative.
 * @ingroup oakfield_experimental_q
 *
 * q-method domains, convergence behavior, and fallback policy may change
 * before the math surface is stable.
 */
SimComplexDouble
sim_qhyperexp_phi_complex(SimComplexDouble lambda, SimComplexDouble epsilon, int K, double q);
SimComplexDouble
sim_qhyperexp_phi_deriv_complex(SimComplexDouble lambda, SimComplexDouble epsilon, int K, double q);

/*==============================================================================
 *                                Q-ANALOGS
 *============================================================================*/

/**
 * @ingroup oakfield_experimental_q
 * @warning Experimental API. q-analog helpers are included for exploration.
 * Their input-domain checks, convergence limits, fallback behavior, and
 * precision guarantees may change before they are declared stable.
 */

/** @brief Experimental Jackson q-number [x]_q = (1 - q^x) / (1 - q) with stable q→1 limit. */
double sim_q_number(double x, double q);

/** @brief Experimental complex Jackson q-number for SimComplexDouble argument. */
SimComplexDouble sim_q_number_complex(SimComplexDouble z, double q);

/**
 * @brief Experimental q-zeta (Hurwitz-style) ζ_q(s, a) = Σ_{n>=0} q^{(n+a)(s-1)} / [n+a]_q^s.
 * @note Convergent for q∈(0,1) and s>1 with a>0. Returns NAN outside this domain.
 */
double sim_q_zeta(double s, double a, double q);

/** @brief Experimental q-digamma ψ_q(z) with analytic continuation via Jackson series. */
double sim_q_digamma(double x, double q);

/** @brief Experimental complex q-digamma ψ_q(z) for SimComplexDouble argument. */
SimComplexDouble sim_q_digamma_complex(SimComplexDouble z, double q);

#if SIM_DIAGNOSTICS
/**
 * @brief Per-thread snapshot of special-function diagnostics.
 *
 * These counters are populated opportunistically inside the special-function
 * helpers when SIM_DIAGNOSTICS!=0 and can be sampled once per integration
 * step to feed higher-level runtime health tracking.
 */
typedef struct SimSpecialDiagnosticsSnapshot {
    uint64_t reflection_count;         /**< Number of reflection-path evaluations. */
    uint64_t recurrence_shift_samples; /**< Calls that performed recurrence shifts. */
    double   max_recurrence_shift;     /**< Maximum |Δz| accumulated via recurrence. */
    uint64_t stirling_tail_samples;    /**< Calls that evaluated a Stirling tail. */
    double   max_stirling_tail;        /**< Maximum |tail| observed in Stirling series. */
    uint64_t pole_proximity_samples;   /**< Samples within the monitored pole window. */
    double   min_pole_distance;        /**< Smallest distance to a pole encountered. */
} SimSpecialDiagnosticsSnapshot;

/**
 * @brief Capture the current thread-local diagnostics snapshot.
 *
 * @param[out] out   Destination for the snapshot (required).
 * @param reset      When true, clears the underlying TLS counters after copy.
 */
void sim_special_diagnostics_snapshot(SimSpecialDiagnosticsSnapshot* out, bool reset);
#endif

/**
 * @brief Fault categories for special-function helpers when evaluation fails.
 */
typedef enum SimSpecialFault {
    SIM_SPECIAL_FAULT_NONE = 0,        /**< No error. */
    SIM_SPECIAL_FAULT_DOMAIN,          /**< Argument outside supported domain. */
    SIM_SPECIAL_FAULT_SINGULARITY,     /**< Encountered pole / denominator singularity. */
    SIM_SPECIAL_FAULT_ITERATION_LIMIT, /**< Failed to converge within iteration budget. */
    SIM_SPECIAL_FAULT_NUMERIC          /**< General numeric breakdown (overflow/NaN). */
} SimSpecialFault;

/**
 * @brief Structured report describing why a special-function helper failed.
 */
typedef struct SimSpecialEvalReport {
    SimSpecialFault  fault;           /**< Fault classification. */
    const char*      function;        /**< Name of the helper reporting the fault. */
    SimComplexDouble input;           /**< Primary input argument (real stored in .re). */
    double           q_param;         /**< q parameter when applicable (else NAN). */
    double           aux_param;       /**< Auxiliary real parameter (e.g., shift a). */
    double           exponent_param;  /**< Exponent parameter (e.g., s for ζ_q). */
    int              iteration_count; /**< Iterations consumed before exit. */
    double           residual;        /**< Magnitude of last term or residual. */
    double           tolerance;       /**< Target tolerance used for the evaluation. */
} SimSpecialEvalReport;

/** @brief Callback signature used to provide fallback values on failure. */
typedef SimResult (*SimSpecialFallbackFn)(void*                       userdata,
                                          const SimSpecialEvalReport* report,
                                          SimComplexDouble*           value_out);

/** @brief Convert a fault enum to a human-readable string. */
const char* sim_special_fault_name(SimSpecialFault fault);

/** @name Safe classical special-function helpers with fallback hooks */
SimResult sim_digamma_safe(double                x,
                           SimSpecialFallbackFn  fallback,
                           void*                 userdata,
                           SimSpecialEvalReport* report,
                           double*               out_value);

SimResult sim_trigamma_safe(double                x,
                            SimSpecialFallbackFn  fallback,
                            void*                 userdata,
                            SimSpecialEvalReport* report,
                            double*               out_value);

SimResult sim_tetragamma_safe(double                x,
                              SimSpecialFallbackFn  fallback,
                              void*                 userdata,
                              SimSpecialEvalReport* report,
                              double*               out_value);

SimResult sim_hyperexp_phi_safe(double                lambda,
                                double                epsilon,
                                int                   K,
                                SimSpecialFallbackFn  fallback,
                                void*                 userdata,
                                SimSpecialEvalReport* report,
                                double*               out_value);

SimResult sim_hyperexp_phi_deriv_safe(double                lambda,
                                      double                epsilon,
                                      int                   K,
                                      SimSpecialFallbackFn  fallback,
                                      void*                 userdata,
                                      SimSpecialEvalReport* report,
                                      double*               out_value);

/** @name Experimental safe q-number helpers with fallback hooks */
SimResult sim_q_number_safe(double                x,
                            double                q,
                            SimSpecialFallbackFn  fallback,
                            void*                 userdata,
                            SimSpecialEvalReport* report,
                            double*               out_value);

SimResult sim_q_number_complex_safe(SimComplexDouble      z,
                                    double                q,
                                    SimSpecialFallbackFn  fallback,
                                    void*                 userdata,
                                    SimSpecialEvalReport* report,
                                    SimComplexDouble*     out_value);

/** @name Experimental safe q-zeta helpers with fallback hooks */
SimResult sim_q_zeta_safe(double                s,
                          double                a,
                          double                q,
                          SimSpecialFallbackFn  fallback,
                          void*                 userdata,
                          SimSpecialEvalReport* report,
                          double*               out_value);

/** @name Experimental safe q-digamma helpers with fallback hooks */
SimResult sim_q_digamma_safe(double                x,
                             double                q,
                             SimSpecialFallbackFn  fallback,
                             void*                 userdata,
                             SimSpecialEvalReport* report,
                             double*               out_value);

SimResult sim_q_digamma_complex_safe(SimComplexDouble      z,
                                     double                q,
                                     SimSpecialFallbackFn  fallback,
                                     void*                 userdata,
                                     SimSpecialEvalReport* report,
                                     SimComplexDouble*     out_value);

/*==============================================================================
 *                     OPTIONAL ARBITRARY PRECISION (MPFR/MPC)
 *============================================================================*/

#ifdef SIM_HAVE_MPFR
#include <mpfr.h>
/**
 * @brief MPFR digamma ψ with target precision (in bits) and tail cut by tol.
 * @param x   Input (mpfr_t).
 * @param y   Output ψ(x).
 * @param tol Absolute tolerance for Bernoulli tail (mpfr).
 * @note Uses MPFR native psi when available; else evaluates via Stirling tail
 *       until next term < tol. See SIM_HAVE_MPC for complex.
 */
void sim_digamma_mpfr(mpfr_t y, const mpfr_t x, const mpfr_t tol);

/** @brief MPFR trigamma ψ₁ with adaptive tail. */
void sim_trigamma_mpfr(mpfr_t y, const mpfr_t x, const mpfr_t tol);

#ifdef SIM_HAVE_MPC
#include <mpc.h>
/** @brief MPC complex digamma with adaptive tail and reflection. */
void sim_digamma_mpc(mpc_t y, const mpc_t z, const mpfr_t tol);
/** @brief MPC complex trigamma with adaptive tail and reflection. */
void sim_trigamma_mpc(mpc_t y, const mpc_t z, const mpfr_t tol);
#endif
#endif /* SIM_HAVE_MPFR */

#ifdef __cplusplus
}
#endif
#endif /* SIM_MATH_SPECIAL_FUNCTIONS_H */
