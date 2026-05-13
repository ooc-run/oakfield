/**
 * @file digamma_square.c
 * @brief Digamma-based square-wave deformation helpers.
 *
 * These helpers build square-like waveforms from differences of digamma values.
 * A deformation parameter controls the pair of shifted poles; it is normalized
 * into a safe interval before evaluation to avoid singular endpoint behavior.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Dispatch a real digamma evaluation through the selected backend.
 *
 * @param backend Backend selector for fixed-tail, adaptive-tail, or Mortici mode.
 * @param x Real digamma argument.
 * @param tolerance Adaptive-tail tolerance when requested; ignored otherwise.
 * @return Real digamma value from the selected backend.
 */
static inline double sim_digamma_backend_eval_real(SimDigammaBackend backend,
                                                   double            x,
                                                   double            tolerance) {
    double tol = (isfinite(tolerance) && tolerance > 0.0) ? tolerance : 1.0e-12;

    switch (backend) {
        case SIM_DIGAMMA_BACKEND_7_TAIL:
            return sim_digamma_f64_7(x);
        case SIM_DIGAMMA_BACKEND_5_TAIL:
            return sim_digamma_f64_5(x);
        case SIM_DIGAMMA_BACKEND_ADAPTIVE:
            return sim_digamma_f64_tail(x, tol);
        case SIM_DIGAMMA_BACKEND_MORTICI:
            return sim_digamma_f64_mortici(x);
        case SIM_DIGAMMA_BACKEND_12_TAIL:
        default:
            return sim_digamma_f64_12(x);
    }
}

/**
 * @brief Dispatch a complex digamma evaluation through the selected backend.
 *
 * @param backend Backend selector for fixed-tail, adaptive-tail, or Mortici mode.
 * @param z Complex digamma argument.
 * @param tolerance Adaptive-tail tolerance when requested; ignored otherwise.
 * @return Complex digamma value from the selected backend.
 */
static inline SimComplexDouble sim_digamma_backend_eval_complex(SimDigammaBackend backend,
                                                                SimComplexDouble  z,
                                                                double            tolerance) {
    double tol = (isfinite(tolerance) && tolerance > 0.0) ? tolerance : 1.0e-12;

    switch (backend) {
        case SIM_DIGAMMA_BACKEND_7_TAIL:
            return sim_digamma_c64_7(z);
        case SIM_DIGAMMA_BACKEND_5_TAIL:
            return sim_digamma_c64_5(z);
        case SIM_DIGAMMA_BACKEND_ADAPTIVE:
            return sim_digamma_c64_tail(z, tol);
        case SIM_DIGAMMA_BACKEND_MORTICI:
            return sim_digamma_c64_mortici(z);
        case SIM_DIGAMMA_BACKEND_12_TAIL:
        default:
            return sim_digamma_c64_12(z);
    }
}

/**
 * @brief Normalize and clamp the deformation parameter away from singular endpoints.
 *
 * The parameter is reduced modulo one, mirrored into `[0, 0.5]`, and clamped
 * away from 0 and 0.5 by `SIM_DIGAMMA_SQUARE_A_EPS`. Non-finite input falls
 * back to `SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT`.
 *
 * @param a Raw deformation parameter.
 * @return Normalized safe deformation in `(0, 0.5)`.
 */
static inline double sim_digamma_square_normalize_a(double a) {
    if (!isfinite(a)) {
        return SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT;
    }

    double normalized = fmod(a, 1.0);
    if (normalized < 0.0) {
        normalized += 1.0;
    }
    if (normalized > 0.5) {
        normalized = 1.0 - normalized;
    }
    if (normalized < SIM_DIGAMMA_SQUARE_A_EPS) {
        normalized = SIM_DIGAMMA_SQUARE_A_EPS;
    } else if (normalized > 0.5 - SIM_DIGAMMA_SQUARE_A_EPS) {
        normalized = 0.5 - SIM_DIGAMMA_SQUARE_A_EPS;
    }

    return normalized;
}

/**
 * @brief Evaluate the deformed real digamma-square base waveform.
 *
 * Computes a cosine-weighted difference of `psi(a + u)` and `psi(1 - a + u)`,
 * scaled by `amplitude`. The `inner_radians` parameter is treated as the
 * normalized carrier sample `u`; the local variable `inner` converts it to
 * `2*pi*u` for the cosine term.
 *
 * @param amplitude Output amplitude multiplier.
 * @param a Deformation parameter before normalization.
 * @param inner_radians Normalized carrier sample `u`.
 * @param backend Digamma backend selector.
 * @param tolerance Adaptive-tail tolerance when `backend` requests it.
 * @return Deformed real square-like value, or NaN if a backend fails.
 */
double sim_digamma_square_base_deformed_real(double            amplitude,
                                             double            a,
                                             double            inner_radians,
                                             SimDigammaBackend backend,
                                             double            tolerance) {
    double a_norm    = sim_digamma_square_normalize_a(a);
    double u         = inner_radians;
    double inner     = (2.0 * M_PI) * u;
    double psi_lo    = sim_digamma_backend_eval_real(backend, a_norm + u, tolerance);
    double psi_hi    = sim_digamma_backend_eval_real(backend, 1.0 - a_norm + u, tolerance);
    double cos_shift = cos(2.0 * M_PI * a_norm);

    if (!isfinite(psi_lo) || !isfinite(psi_hi)) {
        return NAN;
    }

    return amplitude * (1.0 + (cos(inner) - cos_shift) * ((psi_lo - psi_hi) * M_INV_PI));
}

/**
 * @brief Evaluate the legacy real digamma-square base waveform.
 *
 * Uses `SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT` for the deformation parameter.
 *
 * @param amplitude Output amplitude multiplier.
 * @param inner_radians Normalized carrier sample `u`.
 * @param backend Digamma backend selector.
 * @param tolerance Adaptive-tail tolerance when `backend` requests it.
 * @return Real square-like value, or NaN if a backend fails.
 */
double sim_digamma_square_base_real(double             amplitude,
                                    double             inner_radians,
                                    SimDigammaBackend  backend,
                                    double             tolerance) {
    return sim_digamma_square_base_deformed_real(
        amplitude, SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT, inner_radians, backend, tolerance);
}

/**
 * @brief Evaluate the deformed complex digamma-square base waveform.
 *
 * Complex input is interpreted as a complex normalized carrier sample `u`; the
 * same normalized deformation parameter is used for both shifted digamma calls.
 *
 * @param amplitude Output amplitude multiplier.
 * @param a Deformation parameter before normalization.
 * @param inner_radians Complex normalized carrier sample.
 * @param backend Digamma backend selector.
 * @param tolerance Adaptive-tail tolerance when `backend` requests it.
 * @return Deformed complex square-like value.
 */
SimComplexDouble sim_digamma_square_base_deformed_complex(double            amplitude,
                                                          double            a,
                                                          SimComplexDouble  inner_radians,
                                                          SimDigammaBackend backend,
                                                          double            tolerance) {
    double         a_norm        = sim_digamma_square_normalize_a(a);
    double complex u             = inner_radians.re + I * inner_radians.im;
    double complex inner_complex = (2.0 * M_PI) * u;
    SimComplexDouble psi_lo = sim_digamma_backend_eval_complex(
        backend, (SimComplexDouble) { a_norm + creal(u), cimag(u) }, tolerance);
    SimComplexDouble psi_hi = sim_digamma_backend_eval_complex(
        backend, (SimComplexDouble) { 1.0 - a_norm + creal(u), cimag(u) }, tolerance);
    double         cos_shift    = cos(2.0 * M_PI * a_norm);
    double complex dpsi_over_pi = ((psi_lo.re - psi_hi.re) + I * (psi_lo.im - psi_hi.im)) * M_INV_PI;
    double complex value =
        amplitude * (1.0 + (ccos(inner_complex) - cos_shift) * dpsi_over_pi);

    return (SimComplexDouble) { creal(value), cimag(value) };
}

/**
 * @brief Evaluate the legacy complex digamma-square base waveform.
 *
 * Uses `SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT` for the deformation parameter.
 *
 * @param amplitude Output amplitude multiplier.
 * @param inner_radians Complex normalized carrier sample.
 * @param backend Digamma backend selector.
 * @param tolerance Adaptive-tail tolerance when `backend` requests it.
 * @return Complex square-like value.
 */
SimComplexDouble sim_digamma_square_base_complex(double            amplitude,
                                                 SimComplexDouble  inner_radians,
                                                 SimDigammaBackend backend,
                                                 double            tolerance) {
    return sim_digamma_square_base_deformed_complex(amplitude,
                                                    SIM_DIGAMMA_SQUARE_DEFORMATION_DEFAULT,
                                                    inner_radians,
                                                    backend,
                                                    tolerance);
}
