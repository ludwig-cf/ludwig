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
 *  $Id: free_energy.c,v 1.3.2.1 2007-04-30 15:05:03 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "runtime.h"
#include "utilities.h"
#include "free_energy.h"

static double A_     = -1.0;
static double B_     = +1.0;
static double kappa_ = +1.0;

/*****************************************************************************
 *
 *  free_energy_init
 *
 *  Get the user's parameters, if present. 
 *
 *****************************************************************************/

void init_free_energy() {

  int n;

  n = RUN_get_double_parameter("A", &A_);
  n = RUN_get_double_parameter("B", &B_);
  n = RUN_get_double_parameter("K", &kappa_);

#ifdef _SINGLE_FLUID_
#else
  if (A_ > 0.0) {
    fatal("The free energy parameter A must be negative\n");
  }
  if (B_ < 0.0) {
    fatal("The free energy parameter B must be positive\n");
  }
  if (kappa_ < 0.0) {
    fatal("The free energy parameter kappa must be positive\n");
  }

  info("\nSymmetric phi^4 free energy:\n");
  info("Bulk parameter A      = %f\n", A_);
  info("Bulk parameter B      = %f\n", B_);
  info("Surface penalty kappa = %f\n", kappa_);
  info("Surface tension       = %f\n", surface_tension());
  info("Interfacial width     = %f\n", interfacial_width());
#endif

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
  return A_;
}
double free_energy_B() {
  return B_;
}
double free_energy_K() {
  return kappa_;
}

/*****************************************************************************
 *
 *  surface_tension
 *
 *  Return the theoretical surface tension for the model.
 *
 *****************************************************************************/

double surface_tension() {

  return sqrt(-8.0*kappa_*A_*A_*A_ / (9.0*B_*B_));
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

  return sqrt(-2.0*kappa_ / A_);
}

/*****************************************************************************
 *
 *  chemical_potential
 *
 *  Return the chemical potential given \phi and \nabla^2 \phi.
 *
 *****************************************************************************/

double chemical_potential(const double phi, const double delsq_phi) {

  double mu = phi*(A_ + B_*phi*phi) - kappa_*delsq_phi;

  return mu;
}

/*****************************************************************************
 *
 *  chemical_stress
 *
 *  Return the chemical stress tensor given \phi and related quantities.
 *
 *****************************************************************************/

void chemical_stress(double p[3][3], const double phi,
		     const double grad_phi[3], const double delsq_phi) {

  int i, j;
  double bulk = 0.5*phi*phi*(A_ + 1.5*B_*phi*phi);
  double grad_phi_sq = dot_product(grad_phi, grad_phi);
  extern const double d_[3][3]; /* Pending Refactor util etc. */ 

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      p[i][j] = (bulk - kappa_*(phi*delsq_phi + 0.5*grad_phi_sq))*d_[i][j]
	+ kappa_*grad_phi[i]*grad_phi[j];
    }
  }

  return;
}
