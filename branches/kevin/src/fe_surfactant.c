/****************************************************************************
 *
 *  fe_surfactant.c
 *
 *  Implementation of the surfactant free energy described by
 *  van der Graff and van der Sman [reference].
 *
 *  Two order parameters are required:
 *
 *  [0] \phi is compositional order parameter (cf symmetric free energy)
 *  [1] \psi is surfactant concentration (strictly 0 < psi < 1)
 *
 *  The free energy density is:
 *
 *    F = F_\phi + F_\psi + F_surf + F_add
 *
 *  with
 *
 *    F_phi  = symmetric phi^4 free energy
 *    F_psi  = kT [\psi ln \psi + (1 - \psi) ln (1 - \psi)] 
 *    F_surf = - (1/2)\epsilon\psi (grad \phi)^2
 *             - (1/2)\beta \psi^2 (grad \phi)^2
 *    F_add  = + (1/2) W \psi \phi^2
 *
 *  The beta term allows one to get at the Frumkin isotherm and has
 *  been added here.
 *
 *
 *  This could b rewritten as a function of two separate order parameters:
 *  one for phi, and a separate field for psi which is subject to Langmuir
 *  Hinchwood dynamics.
 *
 *  $Id: surfactant.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2015 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "util.h"
#include "fe_s.h"
#include "fe_surfactant.h"

/* Some values might be
 * a_       = -0.0208333;
 * b_       = +0.0208333;
 * kappa_   = +0.12;
 * 
 * kt_      = 0.00056587;
 * epsilon_ = 0.03;
 * beta_    = 0.0;
 * w_       = 0.0;
 */

struct fe_surfactant_s {
  fe_surfactant_param_t * param;
  field_t * pair;                  /* Single field with two components */
  field_grad_t * dp;               /* Gradients thereof */
  fe_surfactant_t * target;
};

/* "Superclass" call back functions */

int fe_surfactant_fed_cb(fe_t * fe, int index, double * fed);
int fe_surfactant_mu_cb(fe_t * fe, int index, double * mu);
int fe_surfactant_str_cb(fe_t * fe, int index, double s[3][3]);

/****************************************************************************
 *
 *  fe_surfactant_create
 *
 *  A single static instance
 *
 ****************************************************************************/

__host__ int fe_surfactant_create(fe_t * fe, field_t * pair,
				  field_grad_t * dp, fe_surfactant_t ** p) {

  fe_surfactant_t * obj = NULL;

  assert(pair);
  assert(dp);

  obj = (fe_surfactant_t *) calloc(1, sizeof(fe_surfactant_t));
  if (obj == NULL) fatal("calloc(fe_surfactant_t) failed\n");

  obj->param = (fe_surfactant_param_t *)
    calloc(1, sizeof(fe_surfactant_param_t));
  if (obj->param == NULL) fatal("calloc(fe_surfactant_param_t) failed\n");

  obj->pair = pair;
  obj->dp   = dp;

  fe_register_cb(fe, obj, fe_surfactant_fed_cb, fe_surfactant_mu_cb,
		 fe_surfactant_str_cb, NULL, NULL, NULL);
  *p = obj;

  return 0;
}

/****************************************************************************
 *
 *  fe_surfactant_free
 *
 ****************************************************************************/

__host__ int fe_surfactant_free(fe_surfactant_t * fe) {

  assert(fe);

  free(fe->param);
  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_surfactant_param_set
 *
 ****************************************************************************/

__host__ int fe_surfactant_param_set(fe_surfactant_t * fe,
				     fe_surfactant_param_t values) {

  assert(fe);

  *fe->param = values;

  return 0;
}

/*****************************************************************************
 *
 *  fe_surfactant_param
 *
 *****************************************************************************/

__host__ __device__ int fe_surfactant_param(fe_surfactant_t * fe,
					    fe_surfactant_param_t * values) {
  assert(fe);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_surfactant_sigma
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

__host__ __device__ int fe_surfactant_sigma(fe_surfactant_t * fe,
					    double * sigma0) {
  double a, b, kappa;

  assert(fe);
  assert(sigma0);

  a = fe->param->a;
  b = fe->param->b;
  kappa = fe->param->kappa;

  *sigma0 = sqrt(-8.0*kappa*a*a*a/(9.0*b*b));

  return 0;
}

/****************************************************************************
 *
 *  fe_surfactant_xi0
 *
 ****************************************************************************/

__host__ __device__ int fe_surfactant_xi0(fe_surfactant_t * fe,
					  double * xi0) {
  assert(fe);
  assert(xi0);

  *xi0 = sqrt(-2.0*fe->param->kappa/fe->param->a);

  return 0;
}

/****************************************************************************
 *
 *  fe_surfactant_langmuir_isotherm
 *
 *  The Langmuir isotherm psi_c is given by
 *  
 *  ln psi_c = (1/2) epsilon / (kT xi_0^2)
 *
 *  and can be a useful reference. The situation is more complex if
 *  beta is not zero (Frumpkin isotherm).
 *
 ****************************************************************************/ 

__host__ int fe_surfactant_langmuir_isotherm(fe_surfactant_t * fe,
					     double * psi_c) {
  double xi0;

  assert(fe);

  fe_surfactant_xi0(fe, &xi0);
  *psi_c = exp(0.5*fe->param->epsilon / (fe->param->kt*xi0*xi0));

  return 0;
}

/*****************************************************************************
 *
 *  fe_surfactant_fed_cb
 *
 *****************************************************************************/

__host__ __device__ int fe_surfactant_fed_cb(fe_t * fe, int index,
					     double * fed) {

  fe_surfactant_t * fs;

  assert(fe);

  fe_child(fe, (void **) &fs);

  return fe_surfactant_fed(fs, index, fed);
}

/****************************************************************************
 *
 *  fe_surfactant_fed
 *
 *  This is:
 *     (1/2)A \phi^2 + (1/4)B \phi^4 + (1/2) kappa (\nabla\phi)^2
 *   + kT [ \psi ln \psi + (1 - \psi) ln (1 - \psi) ]
 *   - (1/2) \epsilon\psi (\nabla\phi)^2 - (1/2) \beta \psi^2 (\nabla\phi)^2
 *   + (1/2)W\psi\phi^2
 *
 ****************************************************************************/

__host__ __device__ int fe_surfactant_fed(fe_surfactant_t * fe, int index,
					  double * fed) {

  double field[2];
  double phi;
  double psi;
  double grad[2][3];
  double delsq[2];
  double dphisq;

  assert(fe);

  field_scalar_array(fe->pair, index, field);
  field_grad_pair_grad(fe->dp, index, grad);
  field_grad_pair_delsq(fe->dp, index, delsq);
  phi = field[0];
  psi = field[1];

  dphisq = dot_product(grad[0], grad[0]);

  /* We have the symmetric piece followed by terms in psi */

  *fed = 0.5*fe->param->a*phi*phi + 0.25*fe->param->b*phi*phi*phi*phi
    + 0.5*fe->param->kappa*dphisq;

  assert(psi > 0.0);
  assert(psi < 1.0);

  *fed += fe->param->kt*(psi*log(psi) + (1.0 - psi)*log(1.0 - psi))
    - 0.5*fe->param->epsilon*psi*dphisq
    - 0.5*fe->param->beta*psi*psi*dphisq
    + 0.5*fe->param->w*psi*phi*phi;

  return 0;
}

/*****************************************************************************
 *
 *  fe_surfactant_mu_cb
 *
 *****************************************************************************/

__host__ __device__ int fe_surfactant_mu_cb(fe_t * fe, int index,
					    double * mu) {

  fe_surfactant_t * fs;

  assert(fe);

  fe_child(fe, (void **) &fs);

  return fe_surfactant_mu(fs, index, mu);
}

/****************************************************************************
 *
 *  fe_surfactant_mu
 * 
 *  Two chemical potentials are present:
 *
 *  \mu_\phi = A\phi + B\phi^3 - kappa \nabla^2 \phi
 *           + W\phi \psi
 *           + \epsilon (\psi \nabla^2\phi + \nabla\phi . \nabla\psi)
 *           + \beta (\psi^2 \nabla^2\phi + 2\psi \nabla\phi . \nabla\psi) 
 * 
 *  \mu_\psi = kT (ln \psi - ln (1 - \psi) + (1/2) W \phi^2
 *           - (1/2) \epsilon (\nabla \phi)^2
 *           - \beta \psi (\nabla \phi)^2
 *
 ****************************************************************************/

__host__ __device__ int fe_surfactant_mu(fe_surfactant_t * fe, int index,
					 double * mu) {
  double phi;
  double psi;
  double field[2];
  double grad[2][3];
  double delsq[2];

  assert(fe);
  assert(mu);

  field_scalar_array(fe->pair, index, field);
  field_grad_pair_grad(fe->dp, index, grad);
  field_grad_pair_delsq(fe->dp, index, delsq);
  phi = field[0];
  psi = field[1];

  /* mu_phi */

  mu[0] = fe->param->a*phi + fe->param->b*phi*phi*phi
    - fe->param->kappa*delsq[0]
    + fe->param->w*phi*psi
    + fe->param->epsilon*(psi*delsq[0] + dot_product(grad[0], grad[1]))
    + fe->param->beta*psi*(psi*delsq[0] + 2.0*dot_product(grad[0], grad[1]));

  /* mu_psi */

  assert(psi > 0.0);
  assert(psi < 1.0);

  mu[1] = fe->param->kt*(log(psi) - log(1.0 - psi))
    + 0.5*fe->param->w*phi*phi
    - 0.5*fe->param->epsilon*dot_product(grad[0], grad[0])
    - fe->param->beta*psi*dot_product(grad[0], grad[0]);

  return 0;
}

/*****************************************************************************
 *
 *  fe_surfactant_str_cb
 *
 *****************************************************************************/

__host__ __device__ int fe_surfactant_str_cb(fe_t * fe, int index,
					     double s[3][3]) {
  fe_surfactant_t * fs;

  assert(fe);

  fe_child(fe, (void **) &fs);

  return fe_surfactant_str(fs, index, s);
}

/****************************************************************************
 *
 *  fe_surfactant_str
 *
 *  Thermodynamoc stress S_ab = p0 delta_ab + P_ab
 *
 *  p0 = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *     - (1/2) kappa (\nabla phi)^2
 *     - kT ln(1 - \psi)
 *     + W \psi \phi^2
 *     + \epsilon \phi \nabla_a \phi \nabla_a \psi
 *     + \epsilon \phi \psi \nabla^2 \phi
 *     + 2 \beta \phi \psi \nabla_a\phi \nabla_a\psi
 *     + \beta\phi\psi^2 \nabla^2 \phi
 *     - (1/2) \beta\psi^2 (\nabla\phi)^2  
 *
 *  P_ab = (\kappa - \epsilon\psi - \beta\psi^2) \nabla_a \phi \nabla_b \phi
 *
 ****************************************************************************/

__host__ __device__ int fe_surfactant_str(fe_surfactant_t * fe, int index,
					  double s[3][3]) {
  int ia, ib;
  double field[2];
  double phi;
  double psi;
  double delsq[2];
  double grad[2][3];
  double p0;

  assert(fe);

  field_scalar_array(fe->pair, index, field);
  field_grad_pair_grad(fe->dp, index, grad);
  field_grad_pair_delsq(fe->dp, index, delsq);
  phi = field[0];
  psi = field[1];

  assert(psi < 1.0);

  p0 = 0.5*fe->param->a*phi*phi + 0.75*fe->param->b*phi*phi*phi*phi
    - fe->param->kappa*(phi*delsq[0] - 0.5*dot_product(grad[0], grad[0]))
    - fe->param->kt*log(1.0 - psi)
    + fe->param->w*psi*phi*phi
    + fe->param->epsilon*phi*(dot_product(grad[0], grad[1]) + psi*delsq[0])
    + fe->param->beta*psi*(2.0*phi*dot_product(grad[0], grad[1])
			   + phi*psi*delsq[0]
			   - 0.5*psi*dot_product(grad[0], grad[0]));

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+
	(fe->param->kappa - fe->param->epsilon*psi - fe->param->beta*psi*psi)*
	grad[0][ia]*grad[0][ib];
    }
  }

  return 0;
}
