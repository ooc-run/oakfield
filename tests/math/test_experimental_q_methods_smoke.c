#include <oakfield/math.h>

#include <math.h>
#include <stdio.h>

static int expect_true(const char* label, int ok) {
    if (!ok) {
        fprintf(stderr, "experimental q-method smoke check failed: %s\n", label);
        return 1;
    }
    return 0;
}

int main(void) {
    const double q = 0.85;
    double       q_number;
    double       q_phi;
    double       q_zeta;
    double       q_digamma;
    double       safe_value = NAN;
    SimSpecialEvalReport report;
    SimResult status;
    int failures = 0;

    q_number  = sim_q_number(5.0, q);
    q_phi     = sim_qhyperexp_phi(0.62, 0.08, 7, q);
    q_zeta    = sim_q_zeta(2.5, 1.0, q);
    q_digamma = sim_q_digamma(1.5, q);

    failures += expect_true("q-number finite", isfinite(q_number));
    failures += expect_true("q-phi finite", isfinite(q_phi));
    failures += expect_true("q-zeta finite", isfinite(q_zeta));
    failures += expect_true("q-digamma finite", isfinite(q_digamma));
    failures += expect_true("q-number q->1 limit", fabs(sim_q_number(5.0, 1.0) - 5.0) < 1.0e-12);

    status = sim_q_zeta_safe(2.5, 1.0, q, NULL, NULL, &report, &safe_value);
    failures += expect_true("safe q-zeta status", status == SIM_RESULT_OK);
    failures += expect_true("safe q-zeta value", isfinite(safe_value));
    failures += expect_true("safe q-zeta report", report.fault == SIM_SPECIAL_FAULT_NONE);

    safe_value = 0.0;
    status     = sim_q_zeta_safe(0.5, 1.0, q, NULL, NULL, &report, &safe_value);
    failures += expect_true("safe q-zeta domain status", status != SIM_RESULT_OK);
    failures += expect_true("safe q-zeta domain report", report.fault == SIM_SPECIAL_FAULT_DOMAIN);

    return failures == 0 ? 0 : 1;
}
