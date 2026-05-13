/**
 * @file fourier.h
 * @brief Bandlimited Fourier waveform kernels (BLIT / PolyBLEP / miniBLEP).
 *
 * This header provides reusable kernels for research-grade saw, square, and triangle
 * oscillators:
 *  - BLIT (Dirichlet kernel) + integration for analytical Fourier-series control.
 *  - PolyBLEP / PolyBLAMP for real-time anti-aliased steps and slope changes.
 *  - miniBLEP / miniBLAMP (windowed-sinc table) for higher-stopband rejection.
 *
 * All routines use double precision. Complex variants use SimComplexDouble for ABI
 * stability and mirror the real APIs.
 */

#ifndef FOURIER_H
#define FOURIER_H

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oakfield/field.h"

/**
 * @brief Skip hot-path argument clamps when inputs are validated by caller.
 *
 * 0 = clamp/guard (default), 1 = assume phase/duty in-range; debug builds assert.
 */
#ifndef SIM_FOURIER_ASSUME_VALID
#define SIM_FOURIER_ASSUME_VALID 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SIM_HAVE_SIMCOMPLEXDOUBLE
/**
 * @brief Fallback ABI-compatible complex double if not defined in field.h.
 */
typedef struct SimComplexDouble {
    double re; /**< Real component. */
    double im; /**< Imaginary component. */
} SimComplexDouble;
#endif

/**
 * @brief Two-sided BLIT support window (in units of phase increment): wider -> cleaner stopband.
 * @details The miniBLEP/miniBLAMP kernels are stretched over @ref SIM_FOURIER_MINIBLEP_SPAN
 *          phase increments but capped at @ref SIM_FOURIER_MINIBLEP_MAX_WINDOW to avoid
 *          consuming an entire period near Nyquist.
 */
#ifndef SIM_FOURIER_MINIBLEP_SPAN
#define SIM_FOURIER_MINIBLEP_SPAN 4.0
#endif

/** @brief Maximum miniBLEP window in normalized phase (0..1). */
#ifndef SIM_FOURIER_MINIBLEP_MAX_WINDOW
#define SIM_FOURIER_MINIBLEP_MAX_WINDOW 0.5
#endif

/** @brief Small leak to keep BLIT integrators numerically bounded. */
#ifndef SIM_FOURIER_INTEGRATOR_LEAK
#define SIM_FOURIER_INTEGRATOR_LEAK 1e-9
#endif

/*==============================================================================
 *                           BLIT (Dirichlet kernel)
 *============================================================================*/

/**
 * @brief Normalized Dirichlet kernel D_M(φ) = sin(Mφ/2)/(M sin(φ/2)).
 * @param phase_radians Instantaneous phase in radians (wrap to 2π is optional).
 * @param harmonic_count N = number of positive harmonics below Nyquist.
 *
 * Implementation detail:
 *  - We internally build an odd Dirichlet length M = 2N + 1 to preserve 2π periodicity.
 *  - DC correction terms use this M (see @ref sim_fourier_blit).
 * @return BLIT sample with DC gain of 1.
 */
double sim_fourier_dirichlet(double phase_radians, int harmonic_count);

/**
 * @brief Complex Dirichlet kernel.
 */
SimComplexDouble sim_fourier_dirichlet_complex(SimComplexDouble phase_radians, int harmonic_count);

/**
 * @brief DC-corrected BLIT for saw/triangle integration: D_M(φ) − 1/M.
 */
double sim_fourier_blit(double phase_radians, int harmonic_count);

/**
 * @brief Complex BLIT (DC-corrected).
 */
SimComplexDouble sim_fourier_blit_complex(SimComplexDouble phase_radians, int harmonic_count);

/**
 * @brief Bandlimited saw from BLIT integration.
 * @param phase_radians Phase in radians.
 * @param phase_increment_radians Per-sample phase advance (2π f / fs).
 * @param harmonic_count Number of partials (see @ref sim_fourier_dirichlet).
 * @param[in,out] state Single-pole integrator state (initialize to 0).
 * @return Saw sample (approximately unity peak-to-peak).
 */
double sim_fourier_saw_blit(double phase_radians, double phase_increment_radians,
                            int harmonic_count, double *state);

/**
 * @brief Complex BLIT saw (integrates complex BLIT).
 */
SimComplexDouble sim_fourier_saw_blit_complex(SimComplexDouble phase_radians,
                                              double phase_increment_radians, int harmonic_count,
                                              SimComplexDouble *state);

/**
 * @brief Bandlimited square via phase-shifted BLIT difference and integration.
 * @param phase_radians Current oscillator phase.
 * @param phase_increment_radians Phase increment per output sample.
 * @param harmonic_count Number of harmonics to include.
 * @param duty Duty cycle in [0,1]; 0.5 yields a symmetric square.
 * @param state Integrator state for the BLIT accumulator.
 */
double sim_fourier_square_blit(double phase_radians, double phase_increment_radians,
                               int harmonic_count, double duty, double *state);

/**
 * @brief Complex BLIT square.
 */
SimComplexDouble sim_fourier_square_blit_complex(SimComplexDouble phase_radians,
                                                 double phase_increment_radians, int harmonic_count,
                                                 double duty, SimComplexDouble *state);

/**
 * @brief Triangle from double-integrated BLIT square (velocity + position states).
 * @param phase_radians Current oscillator phase.
 * @param phase_increment_radians Phase increment per output sample.
 * @param harmonic_count Number of harmonics to include.
 * @param velocity_state First integrator (square accumulator), init to 0.
 * @param position_state Second integrator (triangle accumulator), init to 0.
 */
double sim_fourier_triangle_blit(double phase_radians, double phase_increment_radians,
                                 int harmonic_count, double *velocity_state,
                                 double *position_state);

/**
 * @brief Complex BLIT triangle (double integration).
 */
SimComplexDouble sim_fourier_triangle_blit_complex(SimComplexDouble phase_radians,
                                                   double phase_increment_radians,
                                                   int harmonic_count,
                                                   SimComplexDouble *velocity_state,
                                                   SimComplexDouble *position_state);

/*==============================================================================
 *                           PolyBLEP / PolyBLAMP
 *============================================================================*/

/**
 * @brief Two-point PolyBLEP step correction (subtract from naive saw/square).
 * @param phase Normalized phase in [0,1).
 * @param dphase Normalized phase increment per sample (f / fs).
 */
double sim_fourier_polyblep(double phase, double dphase);

/**
 * @brief Complex PolyBLEP (imaginary part = 0).
 */
SimComplexDouble sim_fourier_polyblep_complex(double phase, double dphase);

/**
 * @brief PolyBLAMP (integral of PolyBLEP) for slope/kink smoothing.
 */
double sim_fourier_polyblamp(double phase, double dphase);

/**
 * @brief Complex PolyBLAMP.
 */
SimComplexDouble sim_fourier_polyblamp_complex(double phase, double dphase);

/*==============================================================================
 *                           miniBLEP / miniBLAMP
 *============================================================================*/

/**
 * @brief Table-driven miniBLEP (windowed-sinc step) for high stopband rejection.
 * @param phase Normalized phase in [0,1).
 * @param dphase Normalized phase increment (f / fs).
 * @return Correction to subtract from a naive discontinuity.
 */
double sim_fourier_miniblep(double phase, double dphase);

/** @brief Complex miniBLEP. */
SimComplexDouble sim_fourier_miniblep_complex(double phase, double dphase);

/**
 * @brief miniBLAMP (integrated miniBLEP) for slope discontinuities (triangle).
 */
double sim_fourier_miniblamp(double phase, double dphase);

/** @brief Complex miniBLAMP. */
SimComplexDouble sim_fourier_miniblamp_complex(double phase, double dphase);

#ifdef __cplusplus
}
#endif

#endif /* FOURIER_H */
