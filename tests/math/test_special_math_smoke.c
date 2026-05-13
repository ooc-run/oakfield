/*
 * Adapted from tests/test_special_functions_ulp.c for the standalone 
 * package. Zeta/Xi and experimental q-method coverage live in separately
 * labeled tests.
 */
#include <oakfield/sim.h>
#include <oakfield/math/airy.h>
#include <oakfield/math/bessel.h>

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t ordered_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & (1ULL << 63)) != 0ULL) {
        bits = ~bits + 1ULL;
    } else {
        bits |= (1ULL << 63);
    }
    return bits;
}

static uint64_t ulp_diff(double lhs, double rhs) {
    uint64_t ordered_lhs = ordered_bits(lhs);
    uint64_t ordered_rhs = ordered_bits(rhs);
    return (ordered_lhs > ordered_rhs) ? (ordered_lhs - ordered_rhs) : (ordered_rhs - ordered_lhs);
}

static bool check_ulp(const char* label, double value, double expected, uint64_t max_ulp) {
    uint64_t diff = ulp_diff(value, expected);
    if (diff > max_ulp) {
        fprintf(stderr,
                "[FAIL] %s: value=%.17g expected=%.17g ulp_diff=%" PRIu64 " max=%" PRIu64 "\n",
                label,
                value,
                expected,
                diff,
                max_ulp);
        return false;
    }
    return true;
}

static bool check_relative(const char* label, double value, double expected, double tolerance) {
    double scale = fmax(1.0, fmax(fabs(value), fabs(expected)));
    double diff  = fabs(value - expected);
    if (diff > tolerance * scale) {
        fprintf(stderr,
                "[FAIL] %s: value=%.17g expected=%.17g rel=%.17g max=%.17g\n",
                label,
                value,
                expected,
                diff / scale,
                tolerance);
        return false;
    }
    return true;
}

static double hyperexp_deriv_direct(double lambda, double epsilon, int terms) {
    double sum = 0.0;
    for (int k = 0; k < terms; ++k) {
        double shifted = epsilon + (double) k;
        double denom   = lambda + shifted;
        sum += shifted / (denom * denom);
    }
    return sum;
}

int main(void) {
    bool ok = true;

    /* Reference constants sourced from high-precision mpmath. */
    const double digamma_1            = -0.577215664901532860606512090082402431;
    const double digamma_half         = -1.963510026021423479440976332998755439;
    const double digamma_three_halves = 0.036489973978576520166902461303949594;
    const double trigamma_1           = 1.64493406684822643647241516664602519;
    const double trigamma_half        = 4.93480220054467930941724549993807569;
    const double trigamma_2           = 0.644934066848226436472415166646025189;

    ok = check_ulp("digamma(1)", sim_special_digamma(1.0), digamma_1, 4U) && ok;
    ok = check_ulp("digamma(0.5)", sim_special_digamma(0.5), digamma_half, 4U) && ok;
    ok = check_ulp("digamma(1.5)", sim_special_digamma(1.5), digamma_three_halves, 32U) && ok;
    ok = check_ulp("trigamma(1)", sim_special_trigamma(1.0), trigamma_1, 4U) && ok;
    ok = check_ulp("trigamma(0.5)", sim_special_trigamma(0.5), trigamma_half, 4U) && ok;
    ok = check_ulp("trigamma(2)", sim_special_trigamma(2.0), trigamma_2, 4U) && ok;

    ok = check_relative("hyperexp_phi_deriv direct",
                        sim_hyperexp_phi_deriv(0.62, 0.08, 7),
                        hyperexp_deriv_direct(0.62, 0.08, 7),
                        1.0e-12) &&
         ok;

    ok = check_relative("airy Ai(0)", sim_airy_ai_f64(0.0), 0.35502805388781723926, 1.0e-14) && ok;
    ok = check_relative("bessel J0(0)", sim_bessel_j0_f64(0.0), 1.0, 1.0e-15) && ok;
    ok = check_relative("bessel J1(0)", sim_bessel_j1_f64(0.0), 0.0, 1.0e-15) && ok;
    ok = check_relative("bessel J2(0)", sim_bessel_jn_f64(2, 0.0), 0.0, 1.0e-15) && ok;

    return ok ? 0 : 1;
}
