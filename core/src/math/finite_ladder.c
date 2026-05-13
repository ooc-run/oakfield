/**
 * @file finite_ladder.c
 * @brief Finite hyperexponential ladder helpers using digamma differences.
 *
 * The core real helper evaluates `phi(lambda, epsilon; K) =
 * lambda * (psi(lambda + epsilon + K) - psi(lambda + epsilon))` and its
 * derivatives. Short ladders are summed directly with compensation; longer
 * ladders use recurrence shifts and asymptotic polygamma differences.
 */

#include "internal/polygamma_internal.h"

/**
 * @brief Pair of digamma and trigamma finite differences.
 *
 * `dpsi` stores `psi(a + K) - psi(a)` and `dpsi1` stores
 * `psi_1(a + K) - psi_1(a)`.
 */
typedef struct {
    double dpsi;
    double dpsi1;
} SimDigammaDelta;

/**
 * @brief Real finite ladder value and lambda/epsilon derivatives.
 *
 * The derivatives correspond to `phi = lambda * dpsi` with respect to
 * `lambda` and `epsilon`.
 */
typedef struct {
    double phi;
    double dphi_dlambda;
    double dphi_depsilon;
} SimFiniteLadderEval;

/**
 * @brief Directly sum short finite differences with compensation.
 *
 * @param a Starting offset in the denominator.
 * @param K Number of finite terms.
 * @param dpsi Receives `sum 1/(a+k)`.
 * @param dpsi1 Receives `-sum 1/(a+k)^2`.
 * @return true on finite accumulation, false if a singular/non-finite term occurs.
 */
static inline bool sim_finite_ladder_delta_direct(double a, int K, double* dpsi, double* dpsi1) {
    double sum1 = 0.0;
    double c1   = 0.0;
    double sum2 = 0.0;
    double c2   = 0.0;
    double ak   = a;
    for (int k = 0; k < K; ++k, ak += 1.0) {
        double inv = 1.0 / ak;
        if (!isfinite(inv))
            return false;

        double y1 = inv - c1;
        double t1 = sum1 + y1;
        c1        = (t1 - sum1) - y1;
        sum1      = t1;

        double term2 = -(inv * inv);
        double y2    = term2 - c2;
        double t2    = sum2 + y2;
        c2           = (t2 - sum2) - y2;
        sum2         = t2;
    }
    *dpsi  = sum1;
    *dpsi1 = sum2;
    return true;
}

/**
 * @brief Evaluate digamma/trigamma finite differences with direct and asymptotic paths.
 *
 * The routine sums very short ladders directly, shifts small starting offsets
 * upward term by term, then evaluates the remaining long tail through
 * log/asymptotic differences to reduce cancellation.
 *
 * @param a Starting offset.
 * @param K Number of finite terms; must be positive.
 * @return Difference pair, with NaNs when inputs are invalid or hit poles.
 */
static inline SimDigammaDelta sim_digamma_delta(double a, int K) {
    SimDigammaDelta out = { NAN, NAN };
    if (K <= 0 || !isfinite(a))
        return out;

    double a_end = a + (double) K;
    if ((a <= 0.0 && sim_is_int(a)) || (a_end <= 0.0 && sim_is_int(a_end)))
        return out;

    if (K <= 12) {
        if (!sim_finite_ladder_delta_direct(a, K, &out.dpsi, &out.dpsi1))
            return out;
        return out;
    }

    double dpsi      = 0.0;
    double dpsi1     = 0.0;
    double a_shift   = a;
    int    k_left    = K;
    double shift_mag = 0.0;
#if !SIM_DIAGNOSTICS
    (void) shift_mag;
#endif

    while (k_left > 0 && a_shift < SIM_STIRLING_SHIFT_THRESHOLD) {
        double inv = 1.0 / a_shift;
        if (!isfinite(inv))
            return out;

        dpsi += inv;
        dpsi1 -= inv * inv;
        a_shift += 1.0;
        k_left -= 1;
        shift_mag += 1.0;

        if (k_left <= 12) {
            double tail_dpsi  = 0.0;
            double tail_dpsi1 = 0.0;
            if (!sim_finite_ladder_delta_direct(a_shift, k_left, &tail_dpsi, &tail_dpsi1))
                return (SimDigammaDelta) { NAN, NAN };
            out.dpsi  = dpsi + tail_dpsi;
            out.dpsi1 = dpsi1 + tail_dpsi1;
#if SIM_DIAGNOSTICS
            if (shift_mag > 0.0)
                sim_special_diag_track_recurrence(shift_mag);
#endif
            return out;
        }
    }

#if SIM_DIAGNOSTICS
    if (shift_mag > 0.0)
        sim_special_diag_track_recurrence(shift_mag);
#endif

    if (k_left == 0) {
        out.dpsi  = dpsi;
        out.dpsi1 = dpsi1;
        return out;
    }

    double a1   = a_shift;
    double a2   = a_shift + k_left;
    double inv1 = 1.0 / a1;
    double inv2 = 1.0 / a2;
    if (!isfinite(inv1) || !isfinite(inv2))
        return (SimDigammaDelta) { NAN, NAN };

    double inv1_sq = inv1 * inv1;
    double inv2_sq = inv2 * inv2;

    double delta_log   = log1p(((double) k_left) / a1);
    double delta_dpsi  = delta_log - 0.5 * (inv2 - inv1);
    double delta_dpsi1 = (inv2 - inv1) + 0.5 * (inv2_sq - inv1_sq);

    double pow1          = inv1_sq;
    double pow2          = inv2_sq;
    double series_dpsi   = 0.0;
    int    digamma_terms = (int) (sizeof(C2) / sizeof(C2[0]));
    for (int n = 1; n <= digamma_terms; ++n) {
        double diff = pow2 - pow1;
        delta_dpsi  = FMA(C2[n - 1], diff, delta_dpsi);
        series_dpsi = FMA(C2[n - 1], diff, series_dpsi);
        pow1 *= inv1_sq;
        pow2 *= inv2_sq;
    }

    double powt1        = inv1_sq * inv1;
    double powt2        = inv2_sq * inv2;
    double series_dpsi1 = 0.0;
    int    trig_terms   = (int) (sizeof(B2) / sizeof(B2[0]));
    for (int n = 1; n <= trig_terms; ++n) {
        double diff  = powt2 - powt1;
        delta_dpsi1  = FMA(B2[n - 1], diff, delta_dpsi1);
        series_dpsi1 = FMA(B2[n - 1], diff, series_dpsi1);
        powt1 *= inv1_sq;
        powt2 *= inv2_sq;
    }

#if SIM_DIAGNOSTICS
    sim_special_diag_track_stirling_tail(fmax(fabs(series_dpsi), fabs(series_dpsi1)));
#else
    (void) series_dpsi;
    (void) series_dpsi1;
#endif

    out.dpsi  = dpsi + delta_dpsi;
    out.dpsi1 = dpsi1 + delta_dpsi1;
    return out;
}

/**
 * @brief Evaluate phi and both first derivatives for the real finite ladder.
 *
 * @param lambda Numerator parameter.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms.
 * @return Value and derivatives, or NaNs for invalid inputs.
 */
static inline SimFiniteLadderEval sim_finite_ladder_phi_all(double lambda, double epsilon, int K) {
    SimFiniteLadderEval out = { NAN, NAN, NAN };
    if (!isfinite(lambda) || !isfinite(epsilon) || K <= 0)
        return out;

    SimDigammaDelta delta = sim_digamma_delta(lambda + epsilon, K);
    if (!isfinite(delta.dpsi) || !isfinite(delta.dpsi1))
        return out;

    out.phi           = lambda * delta.dpsi;
    out.dphi_dlambda  = delta.dpsi + lambda * delta.dpsi1;
    out.dphi_depsilon = lambda * delta.dpsi1;
    return out;
}

/**
 * @brief Approximate finite differences using Mortici-style fast polygamma backends.
 *
 * @param a Starting offset.
 * @param K Number of finite terms.
 * @return Approximate difference pair, or NaNs when invalid.
 */
static inline SimDigammaDelta sim_digamma_delta_mortici(double a, int K) {
    SimDigammaDelta out = { NAN, NAN };
    if (K <= 0 || !isfinite(a))
        return out;

    double a_end = a + (double) K;
    if ((a <= 0.0 && sim_is_int(a)) || (a_end <= 0.0 && sim_is_int(a_end)))
        return out;

    double psi0 = sim_digamma_f64_mortici(a);
    double psi1 = sim_digamma_f64_mortici(a_end);
    double tau0 = sim_trigamma_f64_mortici(a);
    double tau1 = sim_trigamma_f64_mortici(a_end);
    if (!isfinite(psi0) || !isfinite(psi1) || !isfinite(tau0) || !isfinite(tau1))
        return out;

    out.dpsi  = psi1 - psi0;
    out.dpsi1 = tau1 - tau0;
    return out;
}

/**
 * @brief Fast real finite ladder phi using Mortici-style approximations.
 *
 * @param lambda Numerator parameter.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms.
 * @return Approximate phi value, or NaN for invalid inputs.
 */
double sim_hyperexp_phi_fast(double lambda, double epsilon, int K) {
    if (!isfinite(lambda) || !isfinite(epsilon) || K <= 0)
        return NAN;
    SimDigammaDelta delta = sim_digamma_delta_mortici(lambda + epsilon, K);
    if (!isfinite(delta.dpsi))
        return NAN;
    return lambda * delta.dpsi;
}

/**
 * @brief Fast real finite ladder phi with lambda and epsilon derivatives.
 *
 * @param lambda Numerator parameter.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms.
 * @return Approximate value and derivatives, or NaNs for invalid inputs.
 */
SimFiniteLadderEval sim_hyperexp_phi_and_deriv_fast(double lambda, double epsilon, int K) {
    SimFiniteLadderEval out = { NAN, NAN, NAN };
    if (!isfinite(lambda) || !isfinite(epsilon) || K <= 0)
        return out;
    SimDigammaDelta delta = sim_digamma_delta_mortici(lambda + epsilon, K);
    if (!isfinite(delta.dpsi) || !isfinite(delta.dpsi1))
        return out;

    out.phi           = lambda * delta.dpsi;
    out.dphi_dlambda  = delta.dpsi + lambda * delta.dpsi1;
    out.dphi_depsilon = lambda * delta.dpsi1;
    return out;
}

/**
 * @brief Evaluate the real finite ladder phi.
 *
 * @param lambda Numerator parameter.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms.
 * @return Phi value, or NaN for invalid inputs.
 */
double sim_hyperexp_phi(double lambda, double epsilon, int K) {
    SimFiniteLadderEval eval = sim_finite_ladder_phi_all(lambda, epsilon, K);
    return eval.phi;
}

/**
 * @brief Evaluate the lambda derivative of the real finite ladder phi.
 *
 * @param lambda Numerator parameter.
 * @param epsilon Denominator offset.
 * @param K Number of finite ladder terms.
 * @return Lambda derivative, or NaN for invalid inputs.
 */
double sim_hyperexp_phi_deriv(double lambda, double epsilon, int K) {
    SimFiniteLadderEval eval = sim_finite_ladder_phi_all(lambda, epsilon, K);
    return eval.dphi_dlambda;
}

/**
 * @brief Evaluate the complex finite ladder phi.
 *
 * Short complex ladders are summed directly with compensation. Longer ladders
 * shift away from small real parts, then use complex digamma differences; a
 * direct fallback is used when the difference is too small relative to the
 * operands.
 *
 * @param lambda Complex numerator parameter.
 * @param epsilon Complex denominator offset.
 * @param K Number of finite ladder terms.
 * @return Complex phi value, or `{NAN, NAN}` for invalid inputs.
 */
SimComplexDouble
sim_hyperexp_phi_complex(SimComplexDouble lambda, SimComplexDouble epsilon, int K) {
    SimComplexDouble out = { NAN, NAN };
    if (K <= 0)
        return out;

    double complex l = lambda.re + I * lambda.im;
    double complex e = epsilon.re + I * epsilon.im;
    double complex a = l + e;

    if (K <= 12) {
        double complex sum = 0.0 + 0.0 * I;
        double complex c   = 0.0 + 0.0 * I;
        double complex ak  = a;
        for (int k = 0; k < K; k++, ak += 1.0) {
            double complex term = l / ak;
            double complex y    = term - c;
            double complex t    = sum + y;
            c                   = (t - sum) - y;
            sum                 = t;
        }
        return (SimComplexDouble) { creal(sum), cimag(sum) };
    }

    double complex shift_accum = 0.0 + 0.0 * I;
    double complex a_shift     = a;
    int            Kshift      = K;

    while (creal(a_shift) < 1.0) {
        shift_accum += l / a_shift;
        a_shift += 1.0;
        Kshift -= 1;
        if (Kshift <= 12) {
            double complex sum = shift_accum;
            double complex c   = 0.0;
            double complex ak  = a_shift;
            for (int k = 0; k < Kshift; k++, ak += 1.0) {
                double complex term = l / ak;
                double complex y    = term - c;
                double complex t    = sum + y;
                c                   = (t - sum) - y;
                sum                 = t;
            }
            return (SimComplexDouble) { creal(sum), cimag(sum) };
        }
    }

    SimComplexDouble psi0 = sim_digamma_c64_12((SimComplexDouble) { creal(a_shift), cimag(a_shift) });
    SimComplexDouble psi1 =
        sim_digamma_c64_12((SimComplexDouble) { creal(a_shift) + Kshift, cimag(a_shift) });
    double complex dpsi  = (psi1.re + I * psi1.im) - (psi0.re + I * psi0.im);
    double         base  = fmax(cabs(psi0.re + I * psi0.im), cabs(psi1.re + I * psi1.im));
    double         tol   = 1e-12 * fmax(1.0, base);

    if (cabs(dpsi) < tol) {
        double complex sum = shift_accum;
        double complex c   = 0.0;
        double complex ak  = a_shift;
        for (int k = 0; k < Kshift; k++, ak += 1.0) {
            double complex term = l / ak;
            double complex y    = term - c;
            double complex t    = sum + y;
            c                   = (t - sum) - y;
            sum                 = t;
        }
        return (SimComplexDouble) { creal(sum), cimag(sum) };
    }

    double complex res = shift_accum + l * dpsi;
    return (SimComplexDouble) { creal(res), cimag(res) };
}

/**
 * @brief Evaluate the complex lambda derivative of finite ladder phi.
 *
 * Uses direct compensated summation for short/near-cancelling cases and
 * digamma/trigamma differences for longer stable cases.
 *
 * @param lambda Complex numerator parameter.
 * @param epsilon Complex denominator offset.
 * @param K Number of finite ladder terms.
 * @return Complex lambda derivative, or `{NAN, NAN}` for invalid inputs.
 */
SimComplexDouble
sim_hyperexp_phi_deriv_complex_opt(SimComplexDouble lambda, SimComplexDouble epsilon, int K) {
    SimComplexDouble out = { NAN, NAN };
    if (K <= 0)
        return out;

    double complex l = lambda.re + I * lambda.im;
    double complex e = epsilon.re + I * epsilon.im;
    double complex a = l + e;

    if (K <= 12) {
        double complex sum = 0.0;
        double complex c   = 0.0;
        double complex ak  = a;
        for (int k = 0; k < K; k++, ak += 1.0) {
            double complex term = (e + k) / (ak * ak);
            double complex y    = term - c;
            double complex t    = sum + y;
            c                   = (t - sum) - y;
            sum                 = t;
        }
        return (SimComplexDouble) { creal(sum), cimag(sum) };
    }

    double complex shift_accum = 0.0;
    double complex a_shift     = a;
    int            Kshift      = K;
    int            shift_idx   = 0;

    while (creal(a_shift) < 1.0) {
        shift_accum += (e + shift_idx) / (a_shift * a_shift);
        a_shift += 1.0;
        shift_idx += 1;
        Kshift -= 1;

        if (Kshift <= 12) {
            double complex sum = shift_accum;
            double complex c   = 0.0;
            double complex ak  = a_shift;
            for (int k = 0; k < Kshift; k++, ak += 1.0) {
                double complex term = (e + shift_idx + k) / (ak * ak);
                double complex y    = term - c;
                double complex t    = sum + y;
                c                   = (t - sum) - y;
                sum                 = t;
            }
            return (SimComplexDouble) { creal(sum), cimag(sum) };
        }
    }

    SimComplexDouble psi0 = sim_digamma_c64_12((SimComplexDouble) { creal(a_shift), cimag(a_shift) });
    SimComplexDouble psi1 =
        sim_digamma_c64_12((SimComplexDouble) { creal(a_shift) + Kshift, cimag(a_shift) });
    double complex dpsi  = (psi1.re + I * psi1.im) - (psi0.re + I * psi0.im);
    double         basep = fmax(cabs(psi0.re + I * psi0.im), cabs(psi1.re + I * psi1.im));
    double         tolp  = 1e-12 * fmax(1.0, basep);

    SimComplexDouble tau0 =
        sim_trigamma_c64_12((SimComplexDouble) { creal(a_shift), cimag(a_shift) });
    SimComplexDouble tau1 =
        sim_trigamma_c64_12((SimComplexDouble) { creal(a_shift) + Kshift, cimag(a_shift) });
    double complex dtau  = (tau1.re + I * tau1.im) - (tau0.re + I * tau0.im);
    double         baset = fmax(cabs(tau0.re + I * tau0.im), cabs(tau1.re + I * tau1.im));
    double         tolt  = 1e-12 * fmax(1.0, baset);

    if (cabs(dpsi) < tolp && cabs(dtau) < tolt) {
        double complex sum = shift_accum;
        double complex c   = 0.0;
        double complex ak  = a_shift;
        for (int k = 0; k < Kshift; k++, ak += 1.0) {
            double complex term = (e + shift_idx + k) / (ak * ak);
            double complex y    = term - c;
            double complex t    = sum + y;
            c                   = (t - sum) - y;
            sum                 = t;
        }
        return (SimComplexDouble) { creal(sum), cimag(sum) };
    }

    double complex result = shift_accum + dpsi + l * dtau;
    return (SimComplexDouble) { creal(result), cimag(result) };
}
