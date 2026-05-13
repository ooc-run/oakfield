/**
 * @file airy.c
 * @brief Airy Ai approximations using a central series and asymptotic tails.
 *
 * The implementation favors a compact, dependency-free approximation for real
 * arguments: a power series near the origin, an exponentially decaying leading
 * term for large positive inputs, and an oscillatory leading term for large
 * negative inputs.
 */

#include "oakfield/math/airy.h"

#include <math.h>
#include <stddef.h>

#define SIM_AIRY_SERIES_TERMS 48U
#define SIM_AIRY_ASYMPTOTIC_THRESHOLD 8.0
#define SIM_AIRY_AI0 0.35502805388781723926
#define SIM_AIRY_AIP0 -0.25881940379280679841

typedef struct SimAiryRationalPiece {
    double        a;
    double        b;
    const double* p;
    size_t        p_count;
    const double* q;
    size_t        q_count;
} SimAiryRationalPiece;

static const double SIM_AIRY_AI_M12_M8_P[] = {
    -0.077164247192830188,   -0.18749129725606173,    -0.094713479657837787,
    -0.012088402038172148,   0.060762642071246063,    0.20453015599687449,
    0.041232261359827931,    -0.051122567058103516,   -0.010947891197258655,
    0.0044552602209973974,   0.00081895585310037984,  -0.00016915847026767011,
    -2.2515720366118458e-05, 2.4924657443669504e-06,  1.7565554322419549e-07,
};
static const double SIM_AIRY_AI_M12_M8_Q[] = {
    1.0,                    0.75672412899157937,    0.40550000327600455,
    0.10578245885332484,    0.035770854037956767,   0.007180063201198612,
    0.0018732999067215422,  0.00029905078894766884, 6.5670400387409601e-05,
    8.2374337973927613e-06, 1.6073095397879463e-06, 1.4692498679750355e-07,
    2.6681230947082331e-08, 1.3816123987384183e-09, 2.4620020947668133e-10,
};

static const double SIM_AIRY_AI_M8_M4_P[] = {
    0.088107443127869209,   0.087574987330312729,   0.015150590397042241,
    -0.13211828957639973,   -0.2374211872743367,    -0.039999441678794011,
    0.038659858981871381,   0.0075175376393245893,  -0.0021998960903161354,
    -0.0003184611358323145, 6.4425811349653052e-05, 4.2622566255026636e-06,
    -7.8801144671936468e-07,
};
static const double SIM_AIRY_AI_M8_M4_Q[] = {
    1.0,                    0.93150901589414059,    0.40561178014020033,
    0.11458106051868956,    0.030829502101039442,   0.0062608232601578028,
    0.0012382491717064248,  0.00018974651086051646, 2.9168187298057952e-05,
    3.287045480943412e-06,  3.9447766896837686e-07, 2.7182637282433908e-08,
    2.4251215003906915e-09,
};

static const double SIM_AIRY_AI_M4_0_P[] = {
    0.32306827768328072,     0.57098793446738383,    0.16587788313580926,
    -0.093979119924217466,   -0.022162986066455139,  0.008555339158671094,
    0.00057030437277736865,  -0.00035796656634766672, 1.3266277437733467e-05,
    5.0157261471661389e-06,  -4.4948511830842705e-07,
};
static const double SIM_AIRY_AI_M4_0_Q[] = {
    1.0,                    1.0834123085930041,     0.4376893739619877,
    0.12538792921924949,    0.029914831079235985,   0.0056513465405953235,
    0.00091718125335069078, 0.00011793033467060669, 1.2924154245303908e-05,
    9.8115750891478208e-07, 5.921734081369366e-08,
};

static const double SIM_AIRY_AI_0_4_P[] = {
    0.02878046461582201,     -0.033556423481993322, 0.0072690526212670983,
    0.00062997450184297369,  -0.00067982682776030585, 0.00014266265829668943,
    -7.2521279652826465e-06, -2.3639553663486284e-06, 5.4416539229102553e-07,
    -4.9910052235009422e-08, 1.8652518481033097e-09,
};
static const double SIM_AIRY_AI_0_4_Q[] = {
    1.0,                    1.1410363716950542,     0.42613449976981327,
    0.11314167482925945,    0.023323524463124523,   0.0037683942652092946,
    0.00048858020061302615, 4.9480906466146597e-05, 3.8857725225570294e-06,
    2.1133784151266612e-07, 7.4279524594498943e-09,
};

static const double SIM_AIRY_AI_4_8_P[] = {
    1.3982864847438626e-05,  -2.1677998981278402e-05, 1.0691932220269235e-05,
    -3.5195596535192174e-06, 7.8189013619019583e-07,  -1.1133174476735663e-07,
    7.6535302713424082e-09,  4.51945034274855e-10,    -1.648641217124189e-10,
    1.6439785485249094e-11,  -6.5423342310922394e-13,
};
static const double SIM_AIRY_AI_4_8_Q[] = {
    1.0,                    1.4228737779056364,     0.6376758815429554,
    0.20143799147545338,    0.047498999639829967,   0.0085840484732774263,
    0.0011948372037326124,  0.00012606502528921408, 9.6401848807739957e-06,
    4.8267807423612815e-07, 1.2032580203820006e-08,
};

static const SimAiryRationalPiece SIM_AIRY_AI_PIECES[] = {
    { -12.0, -8.0, SIM_AIRY_AI_M12_M8_P, sizeof(SIM_AIRY_AI_M12_M8_P) / sizeof(SIM_AIRY_AI_M12_M8_P[0]),
      SIM_AIRY_AI_M12_M8_Q, sizeof(SIM_AIRY_AI_M12_M8_Q) / sizeof(SIM_AIRY_AI_M12_M8_Q[0]) },
    { -8.0, -4.0, SIM_AIRY_AI_M8_M4_P, sizeof(SIM_AIRY_AI_M8_M4_P) / sizeof(SIM_AIRY_AI_M8_M4_P[0]),
      SIM_AIRY_AI_M8_M4_Q, sizeof(SIM_AIRY_AI_M8_M4_Q) / sizeof(SIM_AIRY_AI_M8_M4_Q[0]) },
    { -4.0, 0.0, SIM_AIRY_AI_M4_0_P, sizeof(SIM_AIRY_AI_M4_0_P) / sizeof(SIM_AIRY_AI_M4_0_P[0]),
      SIM_AIRY_AI_M4_0_Q, sizeof(SIM_AIRY_AI_M4_0_Q) / sizeof(SIM_AIRY_AI_M4_0_Q[0]) },
    { 0.0, 4.0, SIM_AIRY_AI_0_4_P, sizeof(SIM_AIRY_AI_0_4_P) / sizeof(SIM_AIRY_AI_0_4_P[0]),
      SIM_AIRY_AI_0_4_Q, sizeof(SIM_AIRY_AI_0_4_Q) / sizeof(SIM_AIRY_AI_0_4_Q[0]) },
    { 4.0, 8.0, SIM_AIRY_AI_4_8_P, sizeof(SIM_AIRY_AI_4_8_P) / sizeof(SIM_AIRY_AI_4_8_P[0]),
      SIM_AIRY_AI_4_8_Q, sizeof(SIM_AIRY_AI_4_8_Q) / sizeof(SIM_AIRY_AI_4_8_Q[0]) },
};

static double sim_airy_eval_cheb_f64(const double* c, size_t n, double t) {
    double sum;
    double t_prev;
    double t_cur;

    if (n == 0U) {
        return 0.0;
    }
    sum = c[0];
    if (n == 1U) {
        return sum;
    }

    t_prev = 1.0;
    t_cur  = t;
    sum += c[1] * t_cur;
    for (size_t i = 2U; i < n; ++i) {
        double t_next = 2.0 * t * t_cur - t_prev;
        sum += c[i] * t_next;
        t_prev = t_cur;
        t_cur  = t_next;
    }
    return sum;
}

static double sim_airy_ai_piece_f64(const SimAiryRationalPiece* piece, double x) {
    double t = (2.0 * x - piece->a - piece->b) / (piece->b - piece->a);
    double p = sim_airy_eval_cheb_f64(piece->p, piece->p_count, t);
    double q = sim_airy_eval_cheb_f64(piece->q, piece->q_count, t);
    return p / q;
}

/**
 * @brief Evaluate Ai(x) near the origin from the power-series recurrence.
 *
 * The coefficients satisfy the Airy differential equation `y'' - x y = 0`
 * with initial values Ai(0) and Ai'(0). Horner evaluation is used after the
 * coefficient table is generated.
 *
 * @param x Real input in the central approximation region.
 * @return Power-series approximation to Ai(x).
 */
static double sim_airy_ai_series_f64(double x) {
    double coeffs[SIM_AIRY_SERIES_TERMS] = { 0.0 };
    coeffs[0]                            = SIM_AIRY_AI0;
    coeffs[1]                            = SIM_AIRY_AIP0;
    coeffs[2]                            = 0.0;

    for (size_t n = 1U; (n + 2U) < SIM_AIRY_SERIES_TERMS; ++n) {
        double denom   = (double) (n + 2U) * (double) (n + 1U);
        coeffs[n + 2U] = coeffs[n - 1U] / denom;
    }

    double sum = coeffs[SIM_AIRY_SERIES_TERMS - 1U];
    for (size_t i = SIM_AIRY_SERIES_TERMS - 1U; i-- > 0U;) {
        sum = sum * x + coeffs[i];
    }
    return sum;
}

/**
 * @brief Evaluate the Airy Ai function in double precision.
 *
 * NaN inputs are propagated. Positive infinity maps to the limiting value zero,
 * while negative infinity produces NaN through the indeterminate oscillatory
 * limit. Finite inputs are routed to either the central series or leading
 * asymptotic forms based on `SIM_AIRY_ASYMPTOTIC_THRESHOLD`.
 *
 * @param x Real argument.
 * @return Approximate Ai(x) in double precision.
 */
double sim_airy_ai_f64(double x) {
    if (isnan(x)) {
        return x;
    }
    if (isinf(x)) {
        return signbit(x) ? (x - x) : 0.0;
    }

    if (x >= -12.0 && x <= SIM_AIRY_ASYMPTOTIC_THRESHOLD) {
        size_t piece_count = sizeof(SIM_AIRY_AI_PIECES) / sizeof(SIM_AIRY_AI_PIECES[0]);
        for (size_t i = 0U; i < piece_count; ++i) {
            const SimAiryRationalPiece* piece = &SIM_AIRY_AI_PIECES[i];
            if (x <= piece->b) {
                return sim_airy_ai_piece_f64(piece, x);
            }
        }
    }

    if (x > SIM_AIRY_ASYMPTOTIC_THRESHOLD) {
        double root = sqrt(x);
        double xi   = (2.0 / 3.0) * x * root;
        return 0.5 * exp(-xi) / (sqrt(M_PI) * pow(x, 0.25));
    }

    if (x < -12.0) {
        double ax   = -x;
        double root = sqrt(ax);
        double xi   = (2.0 / 3.0) * ax * root + 0.25 * M_PI;
        return sin(xi) / (sqrt(M_PI) * pow(ax, 0.25));
    }

    return sim_airy_ai_series_f64(x);
}

/**
 * @brief Evaluate the Airy Ai function in float precision.
 *
 * This wrapper evaluates the double-precision implementation and casts the
 * result to `float`, keeping all domain behavior shared with
 * `sim_airy_ai_f64()`.
 *
 * @param x Real single-precision argument.
 * @return Approximate Ai(x) rounded to float.
 */
float sim_airy_ai_f32(float x) {
    return (float) sim_airy_ai_f64((double) x);
}
