/****************************************************************************
 *
 *  two_symm_oft.c
 *
 *  Implementation of the temperature-dependent symm mixture with an additional
 *  symm mixture which I use to represent a surfactant.
 *
 *  psi currently is not T-dependent but could be in the future.
 *
 *  Two order parameters are required:
 *
 *  [0] \phi is compositional order parameter (cf symmetric free energy)
 *  [1] \psi also (cf symmetric free energy)
 *
 *  The free energy density is:
 *
 *    F = F_\phi + F_\psi 
 *
 *  with
 *
 *    F_phi  = symmetric phi^4 free energy
 *    F_psi  = symmetric phi^4 free energy
 *

 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field.h"
#include "two_symm_oft.h"

/* Virtual function table (host) */

static fe_vt_t fe_two_symm_oft_hvt = {
  (fe_free_ft)      fe_two_symm_oft_free,     /* Virtual destructor */
  (fe_target_ft)    fe_two_symm_oft_target,   /* Return target pointer */
  (fe_fed_ft)       fe_two_symm_oft_fed,      /* Free energy density */
  (fe_mu_ft)        fe_two_symm_oft_mu,       /* Chemical potential */
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_two_symm_oft_str,      /* Total stress */
  (fe_str_ft)       fe_two_symm_oft_str,      /* Symmetric stress */
  (fe_str_ft)       NULL,             /* Antisymmetric stress (not relevant) */
  (fe_hvector_ft)   NULL,             /* Not relevant */
  (fe_htensor_ft)   NULL,             /* Not relevant */
  (fe_htensor_v_ft) NULL,             /* Not reelvant */
  (fe_stress_v_ft)  fe_two_symm_oft_str_v,    /* Total stress (vectorised version) */
  (fe_stress_v_ft)  fe_two_symm_oft_str_v,    /* Symmetric part (vectorised) */
  (fe_stress_v_ft)  NULL              /* Antisymmetric part */
};


static __constant__ fe_two_symm_oft_param_t const_param;

/****************************************************************************
 *
 *  fe_two_symm_oft_create
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_create(pe_t * pe, cs_t * cs, field_t * phi,
		    field_grad_t * dphi, field_t * temperature, fe_two_symm_oft_param_t param,
		    fe_two_symm_oft_t ** fe) {
  int ndevice;
  fe_two_symm_oft_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(fe);
  assert(phi);
  assert(dphi);
  assert(temperature);

  obj = (fe_two_symm_oft_t *) calloc(1, sizeof(fe_two_symm_oft_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_two_symm_oft_t) failed\n");

  obj->param = (fe_two_symm_oft_param_t *) calloc(1, sizeof(fe_two_symm_oft_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_two_symm_oft_param_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->temperature = temperature;
  obj->super.func = &fe_two_symm_oft_hvt;
  obj->super.id = FE_TWO_SYMM_OFT;

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe_two_symm_oft_param_set(obj, param);
    obj->target = obj;
  }
  else {
    fe_two_symm_oft_param_t * tmp;
    tdpMalloc((void **) &obj->target, sizeof(fe_two_symm_oft_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, tmp, sizeof(fe_two_symm_oft_param_t *),
	      tdpMemcpyHostToDevice);
    /* Now copy. */
    assert(0); /* No implementation */
  }

  *fe = obj;

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_free
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_free(fe_two_symm_oft_t * fe) {

  int ndevice;

  assert(fe);

  tdpGetDeviceCount(&ndevice);
  if (ndevice > 0) tdpFree(fe->target);

  free(fe->param);
  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_info
 *
 *  Some information on parameters etc.
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_info(fe_two_symm_oft_t * fe) {

  double phi_sigma, psi_sigma, phi_xi, psi_xi;
  pe_t * pe = NULL;

  assert(fe);

  pe = fe->pe;

  fe_two_symm_oft_sigma(fe, &phi_sigma, &psi_sigma);
  fe_two_symm_oft_xi0(fe, &phi_xi, &psi_xi);

  pe_info(pe, "Surfactant free energy parameters:\n");

  pe_info(pe, "Bulk parameter phi_A      = %12.5e\n", fe->param->phi_a);
  pe_info(pe, "Bulk parameter phi_B      = %12.5e\n", fe->param->phi_b);
  pe_info(pe, "Surface penalty phi_kappa0 = %12.5e\n", fe->param->phi_kappa0);
  pe_info(pe, "Surface penalty phi_kappa1 = %12.5e\n", fe->param->phi_kappa1);

  pe_info(pe, "Bulk parameter psi_A      = %12.5e\n", fe->param->psi_a);
  pe_info(pe, "Bulk parameter psi_B      = %12.5e\n", fe->param->psi_b);
  pe_info(pe, "Surface penalty psi_kappa = %12.5e\n", fe->param->psi_kappa);

  pe_info(pe, "Wetting term c      = %12.5e\n", fe->param->c);
  pe_info(pe, "Wetting term h      = %12.5e\n", fe->param->h);

  pe_info(pe, "\n");
  pe_info(pe, "Derived quantities\n");
  pe_info(pe, "Phi interfacial tension   = %12.5e\n", phi_sigma);
  pe_info(pe, "Psi interfacial tension   = %12.5e\n", psi_sigma);
  pe_info(pe, "Phi interfacial width     = %12.5e\n", phi_xi);
  pe_info(pe, "Psi interfacial width     = %12.5e\n", psi_xi);

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_target
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_target(fe_two_symm_oft_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_param_set
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_param_set(fe_two_symm_oft_t * fe, fe_two_symm_oft_param_t vals) {

  assert(fe);

  *fe->param = vals;

  return 0;
}

/*****************************************************************************
 *
 *  fe_two_symm_oft_param
 *
 *****************************************************************************/

__host__ int fe_two_symm_oft_param(fe_two_symm_oft_t * fe, fe_two_symm_oft_param_t * values) {
  assert(fe);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_sigma
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_sigma(fe_two_symm_oft_t * fe,  double * phi_sigma, double * psi_sigma) {

  double phi_a, phi_b, phi_kappa0;
  double psi_a, psi_b, psi_kappa;

  assert(fe);
  assert(phi_sigma);
  assert(psi_sigma);

  phi_a = fe->param->phi_a;
  phi_b = fe->param->phi_b;
  phi_kappa0 = fe->param->phi_kappa0; 

  psi_a = fe->param->psi_a;
  psi_b = fe->param->psi_b;
  psi_kappa = fe->param->psi_kappa;

  *phi_sigma = sqrt(-8.0*phi_kappa0*phi_a*phi_a*phi_a/(9.0*phi_b*phi_b));
  *psi_sigma = sqrt(-8.0*psi_kappa*psi_a*psi_a*psi_a/(9.0*psi_b*psi_b));

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_xi0
 *
 *  Interfacial width.
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_xi0(fe_two_symm_oft_t * fe, double * phi_xi, double * psi_xi) {

  assert(fe);
  assert(phi_xi);
  assert(phi_xi);

  *phi_xi = sqrt(-2.0*fe->param->phi_kappa0/fe->param->phi_a);
  *psi_xi = sqrt(-2.0*fe->param->psi_kappa/fe->param->psi_a);

  return 0;
}


/****************************************************************************
 *
 *  fe_two_symm_oft_fed
 *
 *  This is:
 *     (1/2) phi_A \phi^2 + (1/4) phi_B \phi^4 + (1/2) (phi_kappa0 + phi_kappa1 * T) (\nabla\phi)^2
 *   + (1/2) psi_A \psi^2 + (1/4) psi_B \psi^4 + (1/2) psi_kappa (\nabla\psi)^2
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_fed(fe_two_symm_oft_t * fe, int index, double * fed) {

  double field[2];
  double phi;
  double psi;
  double temperature;
  double dphi[2][3];
  double dphisq;
  double phi_kappa_oft;

  assert(fe);

  field_scalar_array(fe->phi, index, field);
  field_scalar(fe->temperature, index, &temperature);
  field_grad_pair_grad(fe->dphi, index, dphi);

  phi = field[0];
  psi = field[1];

  dphisq = dphi[0][X]*dphi[0][X] + dphi[0][Y]*dphi[0][Y]
         + dphi[0][Z]*dphi[0][Z];

  /* We have the symmetric piece followed by terms in psi */
 
  phi_kappa_oft = fe->param->phi_kappa0 + fe->param->phi_kappa1*temperature;

  *fed = 0.5*fe->param->phi_a*phi*phi + 0.25*fe->param->phi_b*phi*phi*phi*phi
    + 0.5*phi_kappa_oft*dphisq;

  return 0;
}

/****************************************************************************
 *
 *  fe_two_symm_oft_mu
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

__host__ int fe_two_symm_oft_mu(fe_two_symm_oft_t * fe, int index, double * mu) {

  double phi;
  double psi;
  double temperature;
  double field[2];
  double grad[2][3];
  double delsq[2];
  double phi_kappa_oft;

  assert(fe);
  assert(mu); assert(mu + 1);

  field_scalar_array(fe->phi, index, field);
  field_scalar(fe->temperature, index, &temperature);
  field_grad_pair_grad(fe->dphi, index, grad);
  field_grad_pair_delsq(fe->dphi, index, delsq);

  phi = field[0];
  psi = field[1];

  /* mu_phi */
  phi_kappa_oft = fe->param->phi_kappa0 + fe->param->phi_kappa1*temperature;
  mu[0] = fe->param->phi_a*phi + fe->param->phi_b*phi*phi*phi
    - phi_kappa_oft*delsq[0];

  /* mu_psi */
  mu[1] = fe->param->psi_a*psi + fe->param->psi_b*psi*psi*psi
    - fe->param->psi_kappa*delsq[1];

  return 0;

}

/****************************************************************************
 *
 *  fe_two_symm_oft_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 *  p0 = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *     - (1/2) kappa (\nabla phi)^2
 *     + same with psi
 *
 *  P_ab = \kappa \nabla_a \phi \nabla_b \phi + same with psi
 *
 ****************************************************************************/

__host__ int fe_two_symm_oft_str(fe_two_symm_oft_t * fe, int index, double s[3][3]) {

  int ia, ib;
  double field[2];
  double phi;
  double psi;
  double temperature;
  double delsq[2];
  double grad[2][3];
  double p0;
  double phi_kappa_oft;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  field_scalar_array(fe->phi, index, field);
  field_scalar(fe->temperature, index, &temperature);
  field_grad_pair_grad(fe->dphi, index, grad);
  field_grad_pair_delsq(fe->dphi, index, delsq);

  phi = field[0];
  psi = field[1];

  phi_kappa_oft = fe->param->phi_kappa0 + fe->param->phi_kappa1*temperature;

  p0 = 0.5*fe->param->phi_a*phi*phi + 0.75*fe->param->phi_b*phi*phi*phi*phi
    - phi_kappa_oft*(phi*delsq[0] - 0.5*dot_product(grad[0], grad[0]))
     + 0.5*fe->param->psi_a*psi*psi + 0.75*fe->param->psi_b*psi*psi*psi*psi
    - fe->param->psi_kappa*(psi*delsq[1] - 0.5*dot_product(grad[1], grad[1]));

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d[ia][ib] 
      +	phi_kappa_oft*grad[0][ia]*grad[0][ib]
      + fe->param->psi_kappa*grad[1][ia]*grad[1][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_two_symm_oft_str_v
 *
 *  Stress (vectorised version) Currently a patch-up.
 *
 *****************************************************************************/

int fe_two_symm_oft_str_v(fe_two_symm_oft_t * fe, int index, double s[3][3][NSIMDVL]) {

  int ia, ib;
  int iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_two_symm_oft_str(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return 0;
}
