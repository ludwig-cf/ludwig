/*****************************************************************************
 *
 *  polar_active.c
 *
 *  Free energy for polar active gel.
 *
 *    f = (A/2) P_a P_a + (B/4) (P_a P_a)^2 + (kappa1/2) (d_a P_b)^2
 *      + (delta kappa1 / 2) (e_abc d_b P_c)^2
 *      + (kappa2/2) (d_a P_b P_c)^2
 *
 *  This is an implemetation of a free energy with vector order
 *  parameter.
 *
 *  For the time being, we demand delta = kappa2 = zero; this is until
 *  a full implementation of the final two terms is available.
 *
 *  I note that the Liquid crystal term (1/2) kappa_2 (d_a P_b P_c)^2
 *  may be computed as
 *         (1/2) kappa_2 (P_b d_a P_c + P_c d_a P_b)^2
 *
 *  in which case the term in the molecular field remains
 *        h_a = + 2 kappa_2 P_b (\nabla^2) P_b P_a
 *  which may be equated to
 *        h_a = + 2 kappa_2 [2 P_b d_c P_a d_c P_b + P_b P_b d^2 P_a
 *                           + P_a P_b d^2 P_b]
 *  and so can be computed from P_a, d_b P_a, and d^2 P_a (only).
 *
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "field.h"
#include "field_grad.h"
#include "polar_active.h"
#include "util.h"

static double a_;                 /* Free energy parameter */
static double b_;                 /* Free energy parameter */
static double kappa1_;            /* Free energy elastic constant */
static double delta_;             /* Free energy elastic constant */
static double kappa2_;            /* Free energy elastic constant */

static double zeta_ = 0.0;        /* 'Activity' parameter */

static field_t * p_ = NULL;           /* A reference to the order parameter */
static field_grad_t * grad_p_ = NULL; /* Ditto for gradients */

/*****************************************************************************
 *
 *  polar_active_p_set
 *
 *  Attach a reference to the order parameter field object and the
 *  associated field gradient object.
 *
 *****************************************************************************/

int polar_active_p_set(field_t * p, field_grad_t * p_grad) {

  assert(p);
  assert(p_grad);

  p_ = p;
  grad_p_ = p_grad;

  return 0;
}

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
  delta_  = 0.0;
  kappa2_ = 0.0;

  assert(delta_ == 0.0);
  assert(kappa2_ == 0.0);

  return;
}

/*****************************************************************************
 *
 *  polar_active_free_energy_density
 *
 *  The free energy density is:
 *
 *    f = (A/2) P_a P_a + (B/4) (P_a P_a)^2 + (kappa1/2) (d_a P_b)^2
 *      + (delta kappa1 / 2) (e_abc d_b P_c)^2
 *      + (kappa2/2) (d_a P_b P_c)^2
 *
 *****************************************************************************/

double polar_active_free_energy_density(const int index) {

  int ia, ib, ic;

  double e;
  double p2;
  double dp1, dp3;
  double p[3];
  double dp[3][3];
  double sum;

  field_vector(p_, index, p);
  field_grad_vector_grad(grad_p_, index, dp);

  p2  = 0.0;
  dp1 = 0.0;
  dp3 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
    sum = 0.0;
    for (ib = 0; ib < 3; ib++) {
      dp1 += dp[ia][ib]*dp[ia][ib];
      for (ic = 0; ic < 3; ic++) {
        sum += e_[ia][ib][ic]*dp[ib][ic];
      }
    }
    dp3 += sum*sum;
  }

  e = 0.5*a_*p2 + 0.25*b_*p2*p2 + 0.5*kappa1_*dp1 + 0.5*delta_*kappa1_*dp3;

  return e;
}

/*****************************************************************************
 *
 *  polar_active_chemical_stress
 *
 *  The stress is
 *
 *  S_ab = (1/2) (P_a h_b - P_b h_a)
 *       - lambda [(1/2)(P_a h_b - P_b h_a) - (1/3)P_c h_c d_ab]
 *       - zeta [P_a P_b - (1/3) P_c P_c d_ab]
 *       - kappa1 d_a P_c d_b P_c
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
  double pdoth;
  double p2;
  double p[3];
  double h[3];
  double dp[3][3];

  const double r3 = (1.0/3.0);

  lambda = fe_v_lambda();

  field_vector(p_, index, p);
  field_grad_vector_grad(grad_p_, index, dp);
  polar_active_molecular_field(index, h);

  p2 = 0.0;
  pdoth = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
    pdoth +=  p[ia]*h[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	sum += dp[ia][ic]*dp[ib][ic];
      }
      s[ia][ib] = 0.5*(p[ia]*h[ib] - p[ib]*h[ia])
	- lambda*(0.5*(p[ia]*h[ib] + p[ib]*h[ia]) - r3*d_[ia][ib]*pdoth)
	- kappa1_*sum - zeta_*(p[ia]*p[ib] - r3*d_[ia][ib]*p2);
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
 *  H_a = - A P_a - B (P_b)^2 P_a + kappa1 \nabla^2 P_a
 *        + 2 kappa2 P_c \nabla^2 P_c P_a 
 *  
 *****************************************************************************/

void polar_active_molecular_field(const int index, double h[3]) {

  int ia;

  double p2;
  double p[3];
  double dsqp[3];

  field_vector(p_, index, p);
  field_grad_vector_delsq(grad_p_, index, dsqp);

  p2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    h[ia] = -a_*p[ia] + -b_*p2*p[ia] + kappa1_*dsqp[ia];
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

