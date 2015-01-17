/****************************************************************************
 *
 *  fe_symmetric.c
 *
 *  Implementation of the symmetric \phi^4 free energy functional:
 *
 *  F[\phi] = (1/2) A \phi^2 + (1/4) B \phi^4 + (1/2) \kappa (\nabla\phi)^2
 *
 *  The first two terms represent the bulk free energy, while the
 *  final term penalises curvature in the interface. For a complete
 *  description see Kendon et al., J. Fluid Mech., 440, 147 (2001).
 *
 *  The usual mode of operation is to take a = -b < 0 and k > 0.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2014 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "util.h"
#include "fe_s.h"
#include "fe_symmetric.h"

struct fe_symmetric_s {
  fe_symmetric_param_t * param;
  field_t * phi;
  field_grad_t * dphi;
  fe_symmetric_t * target; /* placeholder awaiting implementation */
};

/* "Superclass" call backs */

int fe_symmetric_fed_cb(fe_t * fe, int index, double * fed);
int fe_symmetric_mu_cb(fe_t * fe, int index, double * mu);
int fe_symmetric_str_cb(fe_t * fe, int index, double s[3][3]);

/****************************************************************************
 *
 *  fe_symmetric_create
 *
 *  fe is the "superclass" free energy pointer.
 *
 ****************************************************************************/

__host__ int fe_symmetric_create(fe_t * fe, field_t * phi, field_grad_t * dphi,
				 fe_symmetric_t ** p) {

  fe_symmetric_t * obj = NULL;

  assert(fe);
  assert(phi);
  assert(dphi);

  obj = (fe_symmetric_t *) calloc(1, sizeof(fe_symmetric_t));
  if (obj == NULL) fatal("calloc(fe_symmetric_t) failed\n");

  obj->param = (fe_symmetric_param_t *) calloc(1, sizeof(fe_symmetric_param_t));
  if (obj->param == NULL) fatal("calloc(fe_symetric_param_t failed\n");

  obj->phi = phi;
  obj->dphi = dphi;
  fe_register_cb(fe, obj, fe_symmetric_fed_cb, fe_symmetric_mu_cb,
		 fe_symmetric_str_cb, NULL, NULL, NULL);
  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_free
 *
 *****************************************************************************/

__host__ int fe_symmetric_free(fe_symmetric_t * fe) {

  assert(fe);

  free(fe->param);
  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_param_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

__host__ int fe_symmetric_param_set(fe_symmetric_t * fe,
				    fe_symmetric_param_t values) {
  assert(fe);

  fe->param->a = values.a;
  fe->param->b = values.b;
  fe->param->kappa = values.kappa;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_param
 *
 *****************************************************************************/

__host__ __device__
int fe_symmetric_param(fe_symmetric_t * fe, fe_symmetric_param_t * values) {

  assert(fe);

  values->a = fe->param->a;
  values->b = fe->param->b;
  values->kappa = fe->param->kappa;

  return 0;
}

/****************************************************************************
 *
 *  symmetric_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2 and a < 0.
 *
 ****************************************************************************/

__host__ __device__
int fe_symmetric_interfacial_tension(fe_symmetric_t * fe, double * sigma) {

  double a, b, kappa;

  assert(fe);

  a = fe->param->a;
  b = fe->param->b;
  kappa = fe->param->kappa;

  *sigma = sqrt(-8.0*kappa*a*a*a/(9.0*b*b));

  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_interfacial_width
 *
 ****************************************************************************/

__host__ __device__
int fe_symmetric_interfacial_width(fe_symmetric_t * fe, double * xi) {

  assert(fe);

  *xi = sqrt(-2.0*fe->param->kappa/fe->param->a);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_fed_cb
 *
 *****************************************************************************/

__host__ __device__
int fe_symmetric_fed_cb(fe_t * fe, int index, double * fed) {

  fe_symmetric_t * fs;

  assert(fe);

  fe_child(fe, (void **) &fs);

  return fe_symmetric_fed(fs, index, fed);
}

/****************************************************************************
 *
 *  symmetric_free_energy_density
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

__host__ __device__ int fe_symmetric_fed(fe_symmetric_t * fe, int index,
					 double * fed) {
  double phi;
  double dphi[3];

  assert(fe);

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, dphi);

  *fed = (0.5*fe->param->a + 0.25*fe->param->b*phi*phi)*phi*phi
    + 0.5*fe->param->kappa*dot_product(dphi, dphi);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_mu_cb
 *
 *****************************************************************************/

__host__ __device__
int fe_symmetric_mu_cb(fe_t * fe, int index, double * mu) {

  fe_symmetric_t * fs;

  assert(fe);

  fe_child(fe, (void **) &fs);

  return fe_symmetric_mu(fs, index, mu);
}

/****************************************************************************
 *
 *  fe_symmetric_mu
 *
 *  The chemical potential
 *
 *     \mu = \delta F / \delta \phi
 *         = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *
 ****************************************************************************/

__host__ __device__
int fe_symmetric_mu(fe_symmetric_t * fe, int index, double * mu) {

  double phi;
  double delsq_phi;

  assert(fe);

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_delsq(fe->dphi, index, &delsq_phi);

  *mu = fe->param->a*phi + fe->param->b*phi*phi*phi
    - fe->param->kappa*delsq_phi;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_str_cb
 *
 *****************************************************************************/

__host__ __device__
int fe_symmetric_str_cb(fe_t * fe, int index, double s[3][3]) {

  fe_symmetric_t * fs;

  assert(fe);

  fe_child(fe, (void **) &fs);

  return fe_symmetric_str(fs, index, s);
}

/****************************************************************************
 *
 *  fe_symmetric_str
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

__host__ __device__
int fe_symmetric_str(fe_symmetric_t * fe, int index,  double s[3][3]) {

  int ia, ib;
  double kappa;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  assert(fe);

  kappa = fe->param->kappa;

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, grad_phi);
  field_grad_scalar_delsq(fe->dphi, index, &delsq_phi);

  p0 = 0.5*fe->param->a*phi*phi + 0.75*fe->param->b*phi*phi*phi*phi
    - kappa*phi*delsq_phi - 0.5*kappa*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+ kappa*grad_phi[ia]*grad_phi[ib];
    }
  }

  return 0;
}
