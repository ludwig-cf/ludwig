/*****************************************************************************
 *
 *  gelx.c
 *
 *  Free energy for polar active gel (formerly gel X).
 *  This is an implemetation of a free energy with vector order
 *  parameter.
 *
 *  $Id: gelx.c,v 1.1.2.2 2010-03-21 13:38:15 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "phi.h"
#include "gelx.h"

static double a_;
static double b_;
static double kappa1_;
static double kappa2_;

static double zeta_ = 0.0;

/*****************************************************************************
 *
 *  gelx_parameters_set
 *
 *****************************************************************************/

void gelx_parameters_set(const double a, const double b, const double k1,
			 double k2) {

  a_ = a;
  b_ = b;
  kappa1_ = k1;
  kappa2_ = k2;

  /* No kappa2 term yet. */
  assert(kappa2_ == 0.0);

  return;
}

/*****************************************************************************
 *
 *  gelx_free_energy_density
 *
 *  The free energy density is:
 *
 *    f = (A/2) P_a P_a + (B/4) (P_a P_a)^2 + (kappa1/2) (d_a P_b)^2
 *      + (kappa2/2) (d_a P_b P_c)^2
 *
 *****************************************************************************/

double gelx_free_energy_density(const int index) {

  int ia, ib, ic;

  double e;
  double p2;
  double dp1, dp2;
  double p[3];
  double dp[3][3];
  double dpp[3][3][3];

  phi_get_q_vector(index, p);
  phi_get_q_gradient_vector(index, dp);
  /* PENDING dpp currently zero as no kappa2 */

  p2  = 0.0;
  dp1 = 0.0;
  dp2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
    for (ib = 0; ib < 3; ib++) {
      dp1 += dp[ia][ib]*dp[ia][ib];
      for (ic = 0; ic < 3; ic++) {
	dpp[ia][ib][ic] = 0.0;
	dp2 += dpp[ia][ib][ic]*dpp[ia][ib][ic];
      }
    }
  }

  e = 0.5*a_*p2 + 0.25*b_*p2*p2 + 0.5*kappa1_*dp1 + 0.5*kappa2_*dp2;

  return e;
}

/*****************************************************************************
 *
 *  gelx_chemical_stress
 *
 *****************************************************************************/

void gelx_chemical_stress(const int index, double s[3][3]) {

  int ia, ib, ic;

  double sum;
  double lambda;
  double p[3];
  double h[3];
  double dp[3][3];

  lambda = fe_v_lambda();

  phi_get_q_vector(index, p);
  phi_get_q_gradient_vector(index, dp);
  gelx_molecular_field(index, h);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	sum += dp[ia][ic]*dp[ib][ic];
      }
      s[ia][ib] = 0.5*(p[ia]*h[ib] - p[ib]*h[ia])
	- 0.5*lambda*(p[ia]*h[ib] + p[ib]*h[ia])
	- kappa1_*sum
	- zeta_*p[ia]*p[ib];
    }
  }

  /* Add a negative sign so that the force on the fluid may be
   * computed as f_a -d_b s_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -s[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gelx_molecular_field
 *
 *  H_a = A P_a + B (P_b)^2 P_a - 2 kappa1 \nabla^2 P_a
 *        + 2 kappa2 P_c \nabla^2 P_c P_a 
 *  
 *****************************************************************************/

void gelx_molecular_field(const int index, double h[3]) {

  int ia, ib;

  double p2;
  double dp2;
  double p[3];
  double dsqp[3];
  double dsqpp[3][3];

  phi_get_q_vector(index, p);
  phi_get_q_delsq_vector(index, dsqp);
  /* PENDING delsq P_a P_b as kappa2 = 0 */

  p2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    dp2 = 0.0;
    for (ib = 0; ib < 3; ib++) {
      dsqpp[ib][ia] = 0.0;
      dp2 += p[ib]*dsqpp[ib][ia];
    }
    h[ia] = -a_*p[ia] + -b_*p2*p[ia] + kappa1_*dsqp[ia] + 2.0*kappa2_*dp2;
  }

  return;
}

/*****************************************************************************
 *
 *  gelx_zeta_set
 *
 *****************************************************************************/

void gelx_zeta_set(const double zeta_new) {

  zeta_ = zeta_new;
  return;
}

/*****************************************************************************
 *
 *  gelx_zeta
 *
 *****************************************************************************/

double gelx_zeta(void) {

  return zeta_;
}
