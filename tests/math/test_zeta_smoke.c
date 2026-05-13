#include <oakfield/math/xi.h>
#include <oakfield/math/zeta.h>

#include <math.h>
#include <stdio.h>

static int expect_true(const char* label, int ok) {
    if (!ok) {
        fprintf(stderr, "zeta smoke check failed: %s\n", label);
        return 1;
    }
    return 0;
}

int main(void) {
    const SimComplexDouble s2 = { 2.0, 0.0 };
    SimZetaContext zeta_context;
    SimZetaResult zeta;
    SimComplexBall zeta_ball;
    SimXiContext xi_context;
    SimXiResult xi;
    int failures = 0;

    zeta_context = sim_zeta_context_default();
    zeta = sim_zeta_eval(s2, &zeta_context);
    failures += expect_true("zeta status", zeta.status == SIM_ZETA_STATUS_OK);
    failures += expect_true("zeta finite", isfinite(zeta.value.re) && isfinite(zeta.value.im));
    failures += expect_true("zeta(2) real part", fabs(zeta.value.re - 1.64493406685) < 1.0e-8);

    zeta_ball = sim_zeta_eval_ball(s2, &zeta_context);
    failures += expect_true("zeta ball status", zeta_ball.status == SIM_ZETA_STATUS_OK);
    failures += expect_true("zeta ball finite", isfinite(zeta_ball.center.re) && isfinite(zeta_ball.radius));

    xi_context = sim_xi_context_interactive();
    xi = sim_xi_eval((SimComplexDouble) { 0.5, 0.0 }, &xi_context);
    failures += expect_true("xi status", xi.status == SIM_ZETA_STATUS_OK);
    failures += expect_true("xi finite", isfinite(xi.value.re) && isfinite(xi.value.im));

    return failures == 0 ? 0 : 1;
}
