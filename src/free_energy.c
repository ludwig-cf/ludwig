/*****************************************************************************
 *
 *  free_energy
 *
 *  This is the symmetric phi^4 free energy:
 *
 *  F[\phi] = (1/2) A \phi^2 + (1/4) B \phi^4 + (1/2) \kappa (\nabla\phi)^2
 *
 *  The first two terms represent the bulk free energy, while the
 *  final term penalises curvature in the interface. For a complete
 *  description see Kendon et al., J. Fluid Mech., 440, 147 (2001).
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "runtime.h"
#include "free_energy.h"

static double A     = -1.0;
static double B     = +1.0;
static double kappa = +1.0;

/*****************************************************************************
 *
 *  free_energy_init
 *
 *  Get the user's parameters, if present. 
 *
 *****************************************************************************/

void init_free_energy() {

  int n;

  n = RUN_get_double_parameter("A", &A);
  n = RUN_get_double_parameter("B", &B);
  n = RUN_get_double_parameter("K", &kappa);

  if (A > 0.0) {
    fatal("The free energy parameter A must be negative\n");
  }
  if (B < 0.0) {
    fatal("The free energy parameter B must be positive\n");
  }
  if (kappa < 0.0) {
    fatal("The free energy parameter kappa must be positive\n");
  }

  info("\nSymmetric phi^4 free energy:\n");
  info("Bulk parameter A      = %f\n", A);
  info("Bulk parameter B      = %f\n", B);
  info("Surface penalty kappa = %f\n", kappa);
  info("Surface tension       = %f\n", surface_tension());
  info("Interfacial width     = %f\n", interfacial_width());

  return;
}

/*****************************************************************************
 *
 *  free_energy_A
 *  free_energy_B
 *  free_energy_K
 *
 *****************************************************************************/

double free_energy_A() {
  return A;
}
double free_energy_B() {
  return B;
}
double free_energy_K() {
  return kappa;
}

/*****************************************************************************
 *
 *  surface_tension
 *
 *  Return the theoretical surface tension for the model.
 *
 *****************************************************************************/

double surface_tension() {

  return sqrt(-8.0*kappa*A*A*A / (9.0*B*B));
}

/*****************************************************************************
 *
 *  interfacial_width
 *
 *  Return the theoretical interfacial width. Note that there is a
 *  typo in 6.17 of Kendon et al (2001); the factor of 2 should be in
 *  numerator.
 *
 *****************************************************************************/

double interfacial_width() {

  return sqrt(-2.0*kappa / A);
}
