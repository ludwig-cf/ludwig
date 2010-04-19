/*****************************************************************************
 *
 *  polar_active.c
 *
 *  Free energy for polar active gel.
 *
 *    f = (A/2) P_a P_a + (B/4) (P_a P_a)^2 + (kappa1/2) (d_a P_b)^2
 *      + (kappa2/2) (d_a P_b P_c)^2
 *
 *  This is an implemetation of a free energy with vector order
 *  parameter.
 *
 *  $Id: polar_active.c,v 1.1.2.5 2010-04-19 10:32:36 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "phi.h"
#include "phi_gradients.h"
#include "polar_active.h"

static double a_;
static double b_;
static double kappa1_;
static double kappa2_;

static double zeta_ = 0.0;

/*****************************************************************************
 *
 *  polar_active_parameters_set
 *
 *****************************************************************************/

void polar_active_parameters_set(const double a, const double b,
				 const double k1, const double k2) {
  a_ = a;
  b_ = b;
  kappa1_ = k1;
  kappa2_ = k2;

  return;
}

/*****************************************************************************
 *
 *  polar_active_free_energy_density
 *
 *  The free energy density is:
 *
 *    f = (A/2) P_a P_a + (B/4) (P_a P_a)^2 + (kappa1/2) (d_a P_b)^2
 *      + (kappa2/2) (d_a P_b P_c)^2
 *
 *****************************************************************************/

double polar_active_free_energy_density(const int index) {

  int ia, ib, ic;

  double e;
  double p2;
  double dp1, dp2;
  double p[3];
  double dp[3][3];
  double dpp[3][3][3];

  phi_vector(index, p);
  phi_gradients_vector_gradient(index, dp);
  phi_gradients_grad_dyadic(index, dpp);

  p2  = 0.0;
  dp1 = 0.0;
  dp2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
    for (ib = 0; ib < 3; ib++) {
      dp1 += dp[ia][ib]*dp[ia][ib];
      for (ic = 0; ic < 3; ic++) {
	dp2 += dpp[ia][ib][ic]*dpp[ia][ib][ic];
      }
    }
  }

  e = 0.5*a_*p2 + 0.25*b_*p2*p2 + 0.5*kappa1_*dp1 + 0.5*kappa2_*dp2;

  return e;
}

/*****************************************************************************
 *
 *  polar_active_chemical_stress
 *
 *  The stress is
 *
 *  S_ab = (1/2) (P_a h_b - P_b h_a) - (1/2) lambda (P_a h_b - P_b h_a)
 *         - zeta P_a P_b - d_a P_c d_b P_c
 * 
 *  This is antisymmetric. Note that extra minus sign added at
 *  the end to allow the force on the Navier Stokes to be
 *  computed as F_a = - d_b S_ab.
 *
 *****************************************************************************/

void polar_active_chemical_stress(const int index, double s[3][3]) {

  int ia, ib, ic;

  double sum;
  double lambda;
  double p[3];
  double h[3];
  double dp[3][3];

  lambda = fe_v_lambda();

  phi_vector(index, p);
  phi_gradients_vector_gradient(index, dp);
  polar_active_molecular_field(index, h);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	sum += dp[ia][ic]*dp[ib][ic];
      }
      s[ia][ib] = 0.5*(p[ia]*h[ib] - p[ib]*h[ia])
	- 0.5*lambda*(p[ia]*h[ib] + p[ib]*h[ia])
	- kappa1_*sum - zeta_*p[ia]*p[ib];
    }
  }

  /* Add a negative sign so that the force on the fluid may be
   * computed as f_a = -d_b s_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -s[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  polar_active_molecular_field
 *
 *  H_a = A P_a + B (P_b)^2 P_a - 2 kappa1 \nabla^2 P_a
 *        + 2 kappa2 P_c \nabla^2 P_c P_a 
 *  
 *****************************************************************************/

void polar_active_molecular_field(const int index, double h[3]) {

  int ia, ib;

  double p2;
  double dp2;
  double p[3];
  double dsqp[3];
  double dsqpp[3][3];

  phi_vector(index, p);
  phi_gradients_vector_delsq(index, dsqp);
  phi_gradients_delsq_dyadic(index, dsqpp);

  p2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    dp2 = 0.0;
    for (ib = 0; ib < 3; ib++) {
      dp2 += p[ib]*dsqpp[ib][ia];
    }
    h[ia] = -a_*p[ia] + -b_*p2*p[ia] + kappa1_*dsqp[ia] + 2.0*kappa2_*dp2;
  }

  return;
}

/*****************************************************************************
 *
 *  polar_active_zeta_set
 *
 *****************************************************************************/

void polar_active_zeta_set(const double zeta_new) {

  zeta_ = zeta_new;
  return;
}

/*****************************************************************************
 *
 *  polar_active_zeta
 *
 *****************************************************************************/

double polar_active_zeta(void) {

  return zeta_;
}

/*****************************************************************************
 *
 *  polar_active_region
 *
 *  Returns 1 in the 'region' and zero outside. The 'region' is a
 *  spherical volume of radius ractive, centred at the centre of
 *  the grid (assumed to be cubic).
 *
 *  The radius of the 'region' is hardwired to (L/2) - 4 lattice
 *  units to allow a clear inactive region at the edges.
 *
 *****************************************************************************/

double polar_active_region(const int index) {

  int noffset[3];
  int coords[3];

  double x, y, z;
  double ractive;
  double active;

  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, coords);

  x = 1.0*(noffset[X] + coords[X]) - (Lmin(X) + 0.5*L(X));
  y = 1.0*(noffset[Y] + coords[Y]) - (Lmin(Y) + 0.5*L(Y));
  z = 1.0*(noffset[Z] + coords[Z]) - (Lmin(Z) + 0.5*L(Z));

  active = 1.0;
  ractive = 0.5*L(X) - 4.0;
  if ((x*x + y*y + z*z) > ractive*ractive) active = 0.0;

  return active;
}
