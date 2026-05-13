#include <oakfield/math.h>
#include <oakfield/math/airy.h>
#include <oakfield/math/bessel.h>

#include <stdio.h>

int main(void) {
    double digamma = sim_special_digamma(1.0);
    double trigamma = sim_special_trigamma(1.0);
    double phi = sim_hyperexp_phi(0.62, 0.08, 7);
    double phi_deriv = sim_hyperexp_phi_deriv(0.62, 0.08, 7);
    double airy = sim_airy_ai_f64(0.0);
    double bessel = sim_bessel_j0_f64(0.5);

    printf("stable special math:\n");
    printf("  digamma(1)=%.15f\n", digamma);
    printf("  trigamma(1)=%.15f\n", trigamma);
    printf("  phi(0.62,0.08;7)=%.15f derivative=%.15f\n", phi, phi_deriv);
    printf("  airy_ai(0)=%.15f bessel_j0(0.5)=%.15f\n", airy, bessel);
    return 0;
}
