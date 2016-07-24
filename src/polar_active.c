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
 *  There is an additional parameter lambda in the stress which is a
 *  material parameter:
 *     | lambda | < 1 => flow aligning
 *     | lambda | > 1 => flow tumbling
 *
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "polar_active.h"
#include "util.h"

struct fe_polar_s {
  fe_t super;               /* Superclass */
  fe_polar_param_t * param; /* Parameters */
  field_t * p;              /* Vector order parameter */
  field_grad_t * dp;        /* Gradients thereof */
  fe_polar_t * target;      /* Device pointer */
};

static __constant__ fe_polar_param_t const_param;

static fe_vt_t fe_polar_hvt = {
  (fe_free_ft)      fe_polar_free,
  (fe_target_ft)    fe_polar_target,
  (fe_fed_ft)       fe_polar_fed,
  (fe_mu_ft)        NULL,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_polar_stress,
  (fe_hvector_ft)   fe_polar_mol_field,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL
};

static  __constant__ fe_vt_t fe_polar_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_polar_fed,
  (fe_mu_ft)        NULL,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_polar_stress,
  (fe_hvector_ft)   fe_polar_mol_field,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL
};


/*****************************************************************************
 *
 *  fe_polar_create
 *
 *****************************************************************************/

__host__ int fe_polar_create(field_t * p, field_grad_t * dp, fe_polar_t ** fe) {

  int ndevice;
  fe_polar_t * obj = NULL;

  obj = (fe_polar_t *) calloc(1, sizeof(fe_polar_t));
  if (obj == NULL) fatal("calloc(fe_polar_t) failed\n");

  obj->param = (fe_polar_param_t *) calloc(1, sizeof(fe_polar_param_t));
  if (obj->param == NULL) fatal("calloc(fe_polar_param_t) failed\n");

  obj->p = p;
  obj->dp = dp;
  obj->super.func = &fe_polar_hvt;
  obj->super.id = FE_POLAR;

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    assert(0); /* Arrange device implementation */
    fe_polar_param_t * p;
    fe_vt_t * vt;
    targetCalloc((void **) &obj->target, sizeof(fe_polar_t));
    targetConstAddress((void **) &p, const_param);
    copyToTarget(&obj->target->param, p, sizeof(fe_polar_param_t *));
    targetConstAddress((void **) &vt, fe_polar_dvt);
    copyToTarget(&obj->target->super.func, &vt, sizeof(fe_vt_t *));
  }

  *fe = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_free
 *
 *****************************************************************************/

__host__ int fe_polar_free(fe_polar_t * fe) {

  assert(fe);

  if (fe->target != fe) targetFree(fe->target);

  free(fe->param);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_target
 *
 *****************************************************************************/

__host__ int fe_polar_target(fe_polar_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_param_set
 *
 *****************************************************************************/

__host__ int fe_polar_param_set(fe_polar_t * fe, fe_polar_param_t values) {

  assert(fe);

  *(fe->param) = values;

  fe->param->delta = 0.0;
  fe->param->kappa2 = 0.0;

  assert(fe->param->delta == 0.0);
  assert(fe->param->kappa2 == 0.0);

  return 0;
}

/*****************************************************************************
 *
 *  fe_polar_param
 *
 *****************************************************************************/

__host__ int fe_polar_param(fe_polar_t * fe, fe_polar_param_t * values) {

  assert(fe);
  assert(values);

  *values = *fe->param;

  return 0;
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

__host__ __device__
int fe_polar_fed(fe_polar_t * fe, int index, double * fed) {

  int ia, ib, ic;

  double p2;
  double dp1, dp3;
  double p[3];
  double dp[3][3];
  double sum;
  LEVI_CIVITA_CHAR(e);

  assert(fe);

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
        sum += e[ia][ib][ic]*dp[ib][ic];
      }
    }
    dp3 += sum*sum;
  }

  *fed = 0.5*fe->param->a*p2 + 0.25*fe->param->b*p2*p2
    + 0.5*fe->param->kappa1*dp1
    + 0.5*fe->param->delta*fe->param->kappa1*dp3;

  return 0;
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

__host__ __device__
int fe_polar_stress(fe_polar_t * fe, int index, double s[3][3]) {

  int ia, ib, ic;

  double sum;
  double pdoth;
  double p2;
  double p[3];
  double h[3];
  double dp[3][3];

  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  field_vector(fe->p, index, p);
  field_grad_vector_grad(fe->dp, index, dp);
  fe_polar_mol_field(fe, index, h);

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
	- fe->param->lambda*(0.5*(p[ia]*h[ib] + p[ib]*h[ia])
			     - r3*d[ia][ib]*pdoth)
	- fe->param->kappa1*sum
	- fe->param->zeta*(p[ia]*p[ib] - r3*d[ia][ib]*p2);
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
 *  fe_polar_mol_field
 *
 *  H_a = - A P_a - B (P_b)^2 P_a + kappa1 \nabla^2 P_a
 *        + 2 kappa2 P_c \nabla^2 P_c P_a 
 *  
 *****************************************************************************/

__host__ __device__
int fe_polar_mol_field(fe_polar_t * fe, int index, double h[3]) {

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

/*****************************************************************************
 *
 *  fe_polar_active_region
 *
 *  Returns 1 in the 'region' and zero outside. The 'region' is a
 *  spherical volume centred at the centre of the grid.
 *
 *****************************************************************************/

__host__
int polar_active_region(fe_polar_t * fe, int index) {

  int noffset[3];
  int coords[3];
  int active;

  double r;
  double x, y, z;

  assert(fe);

  r = fe->param->radius;
  coords_nlocal_offset(noffset);
  coords_index_to_ijk(index, coords);

  x = 1.0*(noffset[X] + coords[X]) - (Lmin(X) + 0.5*L(X));
  y = 1.0*(noffset[Y] + coords[Y]) - (Lmin(Y) + 0.5*L(Y));
  z = 1.0*(noffset[Z] + coords[Z]) - (Lmin(Z) + 0.5*L(Z));

  active = 1;
  if ((x*x + y*y + z*z) > r*r) active = 0;

  return active;
}
