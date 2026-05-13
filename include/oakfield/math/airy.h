/**
 * @file airy.h
 * @brief Real Airy Ai helpers for beam-style stimulus operators.
 *
 * The implementation combines:
 *  - Power-series evaluation around the origin using the Airy ODE recurrence.
 *  - Positive-tail asymptotics for exponentially small values.
 *  - Negative-tail oscillatory asymptotics for large-magnitude negative inputs.
 *
 * This gives a stable daily-driver Ai(x) for the moderate argument ranges used by
 * stimulus operators without pulling in heavyweight external special-function
 * dependencies.
 */
#ifndef OAKFIELD_MATH_AIRY_H
#define OAKFIELD_MATH_AIRY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Airy Ai(x) for double-precision real x.
 *
 * @param x Real argument.
 * @return Approximate Airy Ai value.
 */
double sim_airy_ai_f64(double x);

/**
 * @brief Airy Ai(x) for single-precision real x.
 *
 * @param x Real argument.
 * @return Approximate Airy Ai value rounded to float.
 */
float sim_airy_ai_f32(float x);

#ifdef __cplusplus
}
#endif

#endif /* OAKFIELD_MATH_AIRY_H */
