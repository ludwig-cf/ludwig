/*****************************************************************************
 *
 *  gelx.c
 *
 *  Free energy for polar acive gel (formerly gel X).
 *
 *  $Id: gelx.c,v 1.1.2.1 2010-03-04 14:06:46 kevin Exp $
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

static double lambda_;
static double zeta_;

/*****************************************************************************
 *
 *  gelx_parameters_set
 *
 *****************************************************************************/

void gelx_parameters_set(const double a, const double b, const double k) {

  a_ = a;
  b_ = b;
  kappa1_ = k;

  return;
}

/*****************************************************************************
 *
 *  gelx_free_energy_density
 *
 *****************************************************************************/

double gelx_free_energy_density(const int index) {

  int ia, ib;

  double e;
  double bulk;
  double grad;
  double p[3];
  double dp[3][3];

  phi_get_q_vector(index, p);
  phi_get_q_gradient_vector(index, dp);

  bulk = 0.0;
  grad = 0.0;

  for (ia = 0; ia < 3; ia++) {
    bulk += p[ia]*p[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      grad += dp[ia][ib]*dp[ia][ib];
    }
  }

  e = 0.5*a_*bulk + 0.25*b_*bulk*bulk + kappa1_*grad;

  return e;
}

/*****************************************************************************
 *
 *  gelx_chemical_stress
 *
 *****************************************************************************/

void gelx_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;

  double p[3];
  double h[3];

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.5*(p[ia]*h[ib] - p[ib]*h[ia])
	- 0.5*lambda_*(p[ia]*h[ib] + p[ib]*h[ia]) - zeta_*p[ia]*p[ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  gelx_molecular_field
 *
 *  H_a = A P_a + B (P_b)^2 P_a - 2 kappa \nabla^2 P_a
 *  
 *****************************************************************************/

void gelx_molecular_field(const int index, double h[3]) {

  int ia;

  double p[3];
  double dp[3];
  double sum;

  phi_get_q_vector(index, p);
  phi_get_q_delsq_vector(index, dp);

  sum = 0.0;
  for (ia = 0; ia < 3; ia++) {
    sum += p[ia]*p[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    h[ia] = a_*p[ia] + b_*sum*p[ia] - 2.0*kappa1_*dp[ia];
  }

  return;
}

