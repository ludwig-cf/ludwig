/****************************************************************************
 *
 *  symmetric_oft.c
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
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Early versions date to Desplat, Pagonabarraga and Bladon
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
#include "field_s.h"
#include "symmetric_oft.h"

static __constant__ fe_symm_oft_param_t const_param;

static fe_vt_t fe_symm_oft_hvt = {
  (fe_free_ft)      fe_symm_oft_free,
  (fe_target_ft)    fe_symm_oft_target,
  (fe_fed_ft)       fe_symm_oft_fed,
  (fe_mu_ft)        fe_symm_oft_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_symm_oft_str,
  (fe_str_ft)       fe_symm_oft_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_symm_oft_str_v,
  (fe_stress_v_ft)  fe_symm_oft_str_v,
  (fe_stress_v_ft)  NULL
};


static  __constant__ fe_vt_t fe_symm_oft_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_symm_oft_fed,
  (fe_mu_ft)        fe_symm_oft_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_symm_oft_str,
  (fe_str_ft)       fe_symm_oft_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_symm_oft_str_v,
  (fe_stress_v_ft)  fe_symm_oft_str_v,
  (fe_stress_v_ft)  NULL
};

/*
==> what is this __constant__ for ?
==> why are some functions duplicated ?
*/

/****************************************************************************
 *
 *  fe_symm_oft_create
 *
 *  fe is the "superclass" free energy pointer.
 *
 ****************************************************************************/

__host__ int fe_symm_oft_create(pe_t * pe, cs_t * cs, field_t * phi,
			    field_grad_t * dphi, field_t * temperature, io_options_t * options, fe_symm_oft_t ** p) {

  int ndevice;
  fe_symm_oft_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(phi);
  assert(dphi);
  assert(temperature);
  assert(options);

  obj = (fe_symm_oft_t *) calloc(1, sizeof(fe_symm_oft_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_symm_oft_t) failed\n");

  obj->param = (fe_symm_oft_param_t *) calloc(1, sizeof(fe_symm_oft_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_symm_oft_param_t failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->temperature = temperature;
  obj->options = options;
  obj->super.func = &fe_symm_oft_hvt;
  obj->super.id = FE_SYMM_OFT;

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    fe_symm_oft_param_t * tmp = NULL;
    fe_vt_t * vt;
    tdpMalloc((void **) &obj->target, sizeof(fe_symm_oft_t));
    tdpMemset(obj->target, 0, sizeof(fe_symm_oft_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, &tmp, sizeof(fe_symm_oft_param_t *),
	      tdpMemcpyHostToDevice);
    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_symm_oft_dvt));
    tdpMemcpy(&obj->target->super.func, &vt, sizeof(fe_vt_t *),
	      tdpMemcpyHostToDevice);

    tdpMemcpy(&obj->target->phi, &phi->target, sizeof(field_t *),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->dphi, &dphi->target, sizeof(field_grad_t *),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->temperature, &temperature->target, sizeof(field_t *),
	      tdpMemcpyHostToDevice);
  }

  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_oft_free
 *
 *****************************************************************************/

__host__ int fe_symm_oft_free(fe_symm_oft_t * fe) {

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
 *  fe_symm_oft_target
 *
 *  Return (fe_t *) host copy of target pointer.
 *
 ****************************************************************************/

__host__ int fe_symm_oft_target(fe_symm_oft_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_oft_kernel_commit
 *
 ****************************************************************************/

__host__ int fe_symm_oft_kernel_commit(fe_symm_oft_t * fe) {

  assert(fe);

  tdpMemcpyToSymbol(tdpSymbol(const_param), fe->param, sizeof(fe_symm_oft_param_t),
		    0, tdpMemcpyHostToDevice);

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_oft_param_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

__host__ int fe_symm_oft_param_set(fe_symm_oft_t * fe, fe_symm_oft_param_t values) {

  assert(fe);

  *fe->param = values;

  fe_symm_oft_kernel_commit(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_oft_param
 *
 *****************************************************************************/

__host__ __device__
int fe_symm_oft_param(fe_symm_oft_t * fe, fe_symm_oft_param_t * values) {

  assert(fe);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_oft_fed
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_oft_fed(fe_symm_oft_t * fe, int index, double * fed) {

  double temperature;
  double phi;
  double dphi[3];
  double aoft;
  double kappaoft;

  assert(fe);

  field_scalar(fe->temperature, index, &temperature);
  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, dphi);

  aoft = fe->param->a*temperature; /* linear dependance on T */
  kappaoft = fe->param->kappa*temperature; /* idem for K(T) */

  *fed = (0.5*aoft + 0.25*fe->param->b*phi*phi)*phi*phi
    + 0.5*kappaoft*dot_product(dphi, dphi);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_oft_mu
 *
 *****************************************************************************/

__host__ __device__
int fe_symm_oft_mu(fe_symm_oft_t * fe, int index, double * mu) {

  double phi;
  double delsq;
  double temperature;
  double aoft;
  double kappaoft;
  
  phi = fe->phi->data[addr_rank0(fe->phi->nsites, index)];
  delsq = fe->dphi->delsq[addr_rank0(fe->phi->nsites, index)];
  temperature = fe->temperature->data[addr_rank0(fe->temperature->nsites, index)];

  aoft = fe->param->a * temperature; /* A(T) is linear wrt T */
  kappaoft = fe->param->kappa * temperature; /* idem for K(T) */

  *mu = aoft*phi + fe->param->b*phi*phi*phi - kappaoft*delsq;

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_oft_str
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_oft_str(fe_symm_oft_t * fe, int index,  double s[3][3]) {

  int ia, ib;
  double entropy;

  assert(fe);

  entropy = fe->param->entropy;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      if (ia == 0 && ib == 0) {
	s[ia][ib] = entropy;
      }
      else {
        s[ia][ib] = 0;
      }
    }
  }

  return 0;
}

