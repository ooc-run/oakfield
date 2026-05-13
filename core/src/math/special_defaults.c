/**
 * @file special_defaults.c
 * @brief Public default aliases for special-function helpers.
 *
 * These wrappers preserve older API entry points while choosing the current
 * default backend: double-precision, 12-term Stirling tails for classical
 * digamma/trigamma/tetragamma.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Default real digamma alias using the 12-term tail backend.
 *
 * @param x Real argument.
 * @return `sim_digamma_f64_12(x)`.
 */
double sim_special_digamma(double x) {
    return sim_digamma_f64_12(x);
}

/**
 * @brief Default real trigamma alias using the 12-term tail backend.
 *
 * @param x Real argument.
 * @return `sim_trigamma_f64_12(x)`.
 */
double sim_special_trigamma(double x) {
    return sim_trigamma_f64_12(x);
}

/**
 * @brief Default real tetragamma alias using the 12-term tail backend.
 *
 * @param x Real argument.
 * @return `sim_tetragamma_f64_12(x)`.
 */
double sim_special_tetragamma(double x) {
    return sim_tetragamma_f64_12(x);
}

/**
 * @brief Backward-compatible complex digamma alias using the 12-term tail backend.
 *
 * @param z Complex argument.
 * @return `sim_digamma_c64_12(z)`.
 */
SimComplexDouble sim_special_digamma_complex(SimComplexDouble z) {
    return sim_digamma_c64_12(z);
}

/**
 * @brief Backward-compatible complex trigamma alias using the 12-term tail backend.
 *
 * @param z Complex argument.
 * @return `sim_trigamma_c64_12(z)`.
 */
SimComplexDouble sim_special_trigamma_complex(SimComplexDouble z) {
    return sim_trigamma_c64_12(z);
}

/**
 * @brief Backward-compatible complex tetragamma alias using the 12-term tail backend.
 *
 * @param z Complex argument.
 * @return `sim_tetragamma_c64_12(z)`.
 */
SimComplexDouble sim_special_tetragamma_complex(SimComplexDouble z) {
    return sim_tetragamma_c64_12(z);
}
