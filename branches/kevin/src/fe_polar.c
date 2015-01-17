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
 *  $Id: polar_active.c 2574 2014-12-31 04:20:43Z stratford $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>

#include "util.h"
#include "fe_polar.h"

struct fe_polar_s {
  fe_polar_param_t * param;
  field_t * p;
  field_grad_t * dp;
  fe_polar_t * target;
};

int fe_polar_fed_cb(fe_t * fe, int index, double * fed);
int fe_polar_hvector_cb(fe_t * fe, int index, double h[3]);
int fe_polar_str_cb(fe_t * fe, int index, double s[3][3]);

/*****************************************************************************
 *
 *  fe_polar_create
 *
 *****************************************************************************/

__host__ int fe_polar_create(fe_t * fe, field_t * p, field_grad_t * dp,
			     fe_polar_t ** pobj) {

  fe_polar_t * obj = NULL;

  assert(fe);
  assert(p);
  assert(dp);

  obj = (fe_polar_t *) calloc(1, sizeof(fe_polar_t));
  if (obj == NULL) fatal("calloc(fe_polar_t) failed\n");

  obj->param = (fe_polar_param_t *) calloc(1, sizeof(fe_polar_param_t));
  if (obj->param == NULL) fatal("calloc(fe_polar_param_t) failed\n");

  obj->p = p;
  obj->dp = dp;
  fe_register_cb(fe, obj, fe_polar_fed_cb, NULL, fe_polar_str_cb, NULL,
		 fe_polar_hvector_cb, NULL);
  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_free
 *
 *****************************************************************************/

__host__ int fe_polar_free(fe_polar_t * fe) {

  assert(fe);

  free(fe->param);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_param_set
 *
 *****************************************************************************/

__host__ int fe_polar_param_set(fe_polar_t * fe, fe_polar_param_t values) {

  assert(fe);

  *fe->param = values;

  /* Enforced */
  fe->param->delta  = 0.0;
  fe->param->kappa2 = 0.0;

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_param
 *
 *****************************************************************************/

__host__ __device__ int fe_polar_param(fe_polar_t * fe,
				       fe_polar_param_t * values) {
  assert(fe);
  assert(values);

  *values = *fe->param;

  return 0;
}

/*****************************************************************************
 *
 * fe_polar_fed_cb
 *
 *****************************************************************************/

__host__ __device__ int fe_polar_fed_cb(fe_t * fe, int index, double * fed) {

  fe_polar_t * fp;

  assert(fe);

  fe_child(fe, (void **) &fp);

  return fe_polar_fed(fp, index, fed);
}

/*****************************************************************************
 *
 *  fe_polar_fed
 *
 *  The free energy density is:
 *
 *    f = (A/2) P_a P_a + (B/4) (P_a P_a)^2 + (kappa1/2) (d_a P_b)^2
 *      + (delta kappa1 / 2) (e_abc d_b P_c)^2
 *      + (kappa2/2) (d_a P_b P_c)^2
 *
 *****************************************************************************/

__host__ __device__ int fe_polar_fed(fe_polar_t * fe, int index, double * fed) {
  int ia, ib, ic;
  double p2;
  double dp1, dp3;
  double p[3];
  double dp[3][3];
  double sum;

  assert(fe);
  assert(fed);

  field_vector(fe->p, index, p);
  field_grad_vector_grad(fe->dp, index, dp);

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

  *fed = 0.5*fe->param->a*p2
    + 0.25*fe->param->b*p2*p2
    + 0.5*fe->param->kappa1*dp1
    + 0.5*fe->param->delta*fe->param->kappa1*dp3;

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_str_cb
 *
 *****************************************************************************/

__host__ __device__ int fe_polar_str_cb(fe_t * fe, int index, double s[3][3]) {

  fe_polar_t * fp;

  assert(fe);

  fe_child(fe, (void **) &fp);

  return fe_polar_stress(fp, index, s);
}

/*****************************************************************************
 *
 *  fe_polar_stress
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

__host__ __device__ int fe_polar_stress(fe_polar_t * fe, int index,
					double s[3][3]) {
  int ia, ib, ic;

  double lambda, kappa1, zeta;
  double sum;
  double pdoth;
  double p2;
  double p[3];
  double h[3];
  double dp[3][3];

  const double r3 = (1.0/3.0);

  assert(fe);

  field_vector(fe->p, index, p);
  field_grad_vector_grad(fe->dp, index, dp);
  fe_polar_mol_field(fe, index, h);

  lambda = fe->param->lambda;
  kappa1 = fe->param->kappa1;
  zeta   = fe->param->zeta;

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
	- kappa1*sum - zeta*(p[ia]*p[ib] - r3*d_[ia][ib]*p2);
    }
  }

  /* Add a negative sign so that the force on the fluid may be
   * computed as f_a = -d_b s_ab */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -s[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_hvector_cb
 *
 *****************************************************************************/

__host__ __device__ int fe_polar_hvector_cb(fe_t * fe, int index, double h[3]) {
  fe_polar_t * fp;

  assert(fe);

  fe_child(fe, (void **) &fp);

  return fe_polar_mol_field(fp, index, h);
}


/*****************************************************************************
 *
 *  fe_polar_mol_field
 *
 *  H_a = - A P_a - B (P_b)^2 P_a + kappa1 \nabla^2 P_a
 *        + 2 kappa2 P_c \nabla^2 P_c P_a 
 *  
 *****************************************************************************/

__host__ __device__ int fe_polar_mol_field(fe_polar_t * fe, int index,
					   double h[3]) {
  int ia;
  double p2;
  double p[3];
  double dsqp[3];

  assert(fe);

  field_vector(fe->p, index, p);
  field_grad_vector_delsq(fe->dp, index, dsqp);

  p2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    p2 += p[ia]*p[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    h[ia] = -fe->param->a*p[ia] + -fe->param->b*p2*p[ia]
      + fe->param->kappa1*dsqp[ia];
  }

  return 0;
}
