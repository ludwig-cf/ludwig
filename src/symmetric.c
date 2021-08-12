/****************************************************************************
 *
 *  symmetric.c
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
#include "field_grad_s.h"
#include "symmetric.h"

/* Defaults */

#define FE_DEFUALT_PARAM_A      -0.003125
#define FE_DEFAULT_PARAM_B      +0.003125
#define FE_DEFAULT_PARAM_KAPPA  +0.002

static __constant__ fe_symm_param_t const_param;

static fe_vt_t fe_symm_hvt = {
  (fe_free_ft)      fe_symm_free,
  (fe_target_ft)    fe_symm_target,
  (fe_fed_ft)       fe_symm_fed,
  (fe_mu_ft)        fe_symm_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_symm_str,
  (fe_str_ft)       fe_symm_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_symm_str_v,
  (fe_stress_v_ft)  fe_symm_str_v,
  (fe_stress_v_ft)  NULL
};

static  __constant__ fe_vt_t fe_symm_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_symm_fed,
  (fe_mu_ft)        fe_symm_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_symm_str,
  (fe_str_ft)       fe_symm_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_symm_str_v,
  (fe_stress_v_ft)  fe_symm_str_v,
  (fe_stress_v_ft)  NULL
};

/****************************************************************************
 *
 *  fe_symm_create
 *
 *  fe is the "superclass" free energy pointer.
 *
 ****************************************************************************/

__host__ int fe_symm_create(pe_t * pe, cs_t * cs, field_t * phi,
			    field_grad_t * dphi, fe_symm_t ** p) {

  int ndevice;
  fe_symm_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(phi);
  assert(dphi);

  obj = (fe_symm_t *) calloc(1, sizeof(fe_symm_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_symm_t) failed\n");

  obj->param = (fe_symm_param_t *) calloc(1, sizeof(fe_symm_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_symm_param_t failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->super.func = &fe_symm_hvt;
  obj->super.id = FE_SYMMETRIC;

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    fe_symm_param_t * tmp = NULL;
    fe_vt_t * vt;
    tdpMalloc((void **) &obj->target, sizeof(fe_symm_t));
    tdpMemset(obj->target, 0, sizeof(fe_symm_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, &tmp, sizeof(fe_symm_param_t *),
	      tdpMemcpyHostToDevice);
    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_symm_dvt));
    tdpMemcpy(&obj->target->super.func, &vt, sizeof(fe_vt_t *),
	      tdpMemcpyHostToDevice);

    tdpMemcpy(&obj->target->phi, &phi->target, sizeof(field_t *),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&obj->target->dphi, &dphi->target, sizeof(field_grad_t *),
	      tdpMemcpyHostToDevice);
  }

  *p = obj;

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_free
 *
 *****************************************************************************/

__host__ int fe_symm_free(fe_symm_t * fe) {

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
 *  fe_symm_target
 *
 *  Return (fe_t *) host copy of target pointer.
 *
 ****************************************************************************/

__host__ int fe_symm_target(fe_symm_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_kernel_commit
 *
 ****************************************************************************/

__host__ int fe_symm_kernel_commit(fe_symm_t * fe) {

  assert(fe);

  tdpMemcpyToSymbol(tdpSymbol(const_param), fe->param, sizeof(fe_symm_param_t),
		    0, tdpMemcpyHostToDevice);

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_param_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

__host__ int fe_symm_param_set(fe_symm_t * fe, fe_symm_param_t values) {

  assert(fe);

  *fe->param = values;

  fe_symm_kernel_commit(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_param
 *
 *****************************************************************************/

__host__ __device__
int fe_symm_param(fe_symm_t * fe, fe_symm_param_t * values) {

  assert(fe);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2 and a < 0.
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_interfacial_tension(fe_symm_t * fe, double * sigma) {

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
 *  fe_symm_interfacial_width
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_interfacial_width(fe_symm_t * fe, double * xi) {

  assert(fe);

  *xi = sqrt(-2.0*fe->param->kappa/fe->param->a);

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_fed
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_fed(fe_symm_t * fe, int index, double * fed) {

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
 *  fe_symm_mu
 *
 *****************************************************************************/

__host__ __device__
int fe_symm_mu(fe_symm_t * fe, int index, double * mu) {

  double phi;
  double delsq;

  phi = fe->phi->data[addr_rank0(fe->phi->nsites, index)];
  delsq = fe->dphi->delsq[addr_rank0(fe->phi->nsites, index)];

  *mu = fe->param->a*phi + fe->param->b*phi*phi*phi - fe->param->kappa*delsq;

  return 0;
}

/****************************************************************************
 *
 *  fe_symm_str
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

__host__ __device__
int fe_symm_str(fe_symm_t * fe, int index,  double s[3][3]) {

  int ia, ib;
  double kappa;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;
  double d_ab;

  assert(fe);

  kappa = fe->param->kappa;

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, grad_phi);
  field_grad_scalar_delsq(fe->dphi, index, &delsq_phi);

  p0 = 0.5*fe->param->a*phi*phi + 0.75*fe->param->b*phi*phi*phi*phi
    - kappa*phi*delsq_phi - 0.5*kappa*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      d_ab = (ia == ib);
      s[ia][ib] = p0*d_ab + kappa*grad_phi[ia]*grad_phi[ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_str_v
 *
 *****************************************************************************/

__host__ __device__
void fe_symm_str_v(fe_symm_t * fe, int index, double s[3][3][NSIMDVL]) {

  int ia, ib;
  int iv;
  double a;
  double b;
  double kappa;
  double phi;
  double delsq;
  double grad[3][NSIMDVL];
  double p0;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);
  assert(fe->phi);
  assert(fe->dphi);
  assert(fe->param);

  a = fe->param->a;
  b = fe->param->b;
  kappa = fe->param->kappa;

  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL) {
      grad[ia][iv] = fe->dphi->grad[addr_rank2(fe->dphi->nsite,1,3,index+iv,0,ia)];
    }
  }

  for_simd_v(iv, NSIMDVL) {
    phi = fe->phi->data[addr_rank1(fe->phi->nsites, 1, index + iv, 0)];
    delsq = fe->dphi->delsq[addr_rank1(fe->dphi->nsite, 1, index + iv, 0)];

    p0 = 0.5*a*phi*phi + 0.75*b*phi*phi*phi*phi - kappa*phi*delsq 
      - 0.5*kappa*(grad[X][iv]*grad[X][iv] + grad[Y][iv]*grad[Y][iv]
		   + grad[Z][iv]*grad[Z][iv]);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = p0*d[ia][ib] + kappa*grad[ia][iv]*grad[ib][iv];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_symm_theta_to_h
 *
 *  If theta is an angle in degrees, then the resulting h will satisfy
 *  0.5 [(1+h)^3/2 - (1-h)^3/2] = cos(theta).
 *
 *  That is, H = h sqrt(kappa*B) should give the desired wetting
 *  angle theta from theory.
 *
 *  h +ve is returned. ierr should always be zero.
 *
 *****************************************************************************/

__host__ int fe_symm_theta_to_h(double theta, double * h) {

  int ierr = 0;
  PI_DOUBLE(pi);

  /* Computation must be performed as complex, but the result must have
   * zero imaginary part for any (real) angle. */
  /* Courtesy of Wolfram alpha. */

  /* Programming note: for C/C++ use
   *   1. "double _Complex" not "double complex";
   *   2. no initialisers;
   *   3. c*() functions cpow() etc everywhere. */

  double _Complex a1;
  double _Complex a2;
  double _Complex a3;
  double _Complex z;

  a1 = ccos(pi*theta/180.0);

  z  = cpow(a1, 8) - 6.0*cpow(a1, 6) + 12.0*cpow(a1, 4) - 8.0*cpow(a1, 2);
  a2 = csqrt(z);

  z  = -2.0*cpow(a1, 4) - 10.0*cpow(a1, 2) + 2.0*a2 + 1.0;
  a3 = cpow(z, 1.0/3.0);

  z  = csqrt(4.0*cpow(a1, 2)/a3 + a3 + 1.0/a3 - 2.0);

  /* May not quite make DBL_EPSILON depending on argument ... */
  if (fabs(cimag(z) > 2.0*DBL_EPSILON)) ierr = -1;

  *h = creal(z);

  return ierr;
}

/*****************************************************************************
 *
 *  fe_symm_h_to_costheta
 *
 *  For h = H sqrt(1.0/kappa B), return the associated cos(theta)
 *  where theta is the theoretical wetting angle.
 *
 *  cos(theta) = 0.5 [-(1-h)^3/2 + (1+h)^3/2]
 *
 *  abs(h) >  1                     => costheta is complex
 *  abs(h) >  sqrt(2sqrt(3) - 3)    => |costheta| >  1
 *  abs(h) <= sqrt(2sqrt(3) - 3)    => |costheta| <= 1, i.e., a valid angle
 *
 *  So, evaluate, and check the return code (0 for valid).
 *
 *****************************************************************************/

__host__ int fe_symm_h_to_costheta(double h, double * costheta) {

  int ierr = 0;

  if (abs(h) > 1.0) {
    ierr = -1;
    *costheta = -999.999;
  }
  else {
    *costheta = 0.5*(-pow(1.0 - h, 1.5) + pow(1.0 + h, 1.5));
    if (abs(h) > sqrt(2.0*sqrt(3.0) - 3.0)) ierr = -1;
  }

  return ierr;
}
