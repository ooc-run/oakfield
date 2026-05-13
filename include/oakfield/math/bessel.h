/**
 * @file bessel.h
 * @brief Integer-order cylindrical Bessel J_n helpers for beam-style operators.
 *
 * The implementation combines:
 *  - A convergent power series for moderate arguments.
 *  - Large-argument asymptotics for oscillatory tails.
 *  - Integer-order parity handling for negative orders and arguments.
 *
 * This keeps common Bessel-beam style operators self-contained without relying on
 * platform-specific special-function extensions.
 */
#ifndef OAKFIELD_MATH_BESSEL_H
#define OAKFIELD_MATH_BESSEL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Cylindrical Bessel J_0(x) for double-precision real x.
 *
 * @param x Real argument.
 * @return Approximate J_0(x).
 */
double sim_bessel_j0_f64(double x);

/**
 * @brief Cylindrical Bessel J_1(x) for double-precision real x.
 *
 * @param x Real argument.
 * @return Approximate J_1(x).
 */
double sim_bessel_j1_f64(double x);

/**
 * @brief Cylindrical Bessel J_n(x) for integer order n and double-precision real x.
 *
 * @param order Integer Bessel order.
 * @param x Real argument.
 * @return Approximate J_order(x).
 */
double sim_bessel_jn_f64(int order, double x);

/**
 * @brief Cylindrical Bessel J_n(x) for integer order n and single-precision real x.
 *
 * @param order Integer Bessel order.
 * @param x Real argument.
 * @return Approximate J_order(x) rounded to float.
 */
float sim_bessel_jn_f32(int order, float x);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_BESSEL_H */
