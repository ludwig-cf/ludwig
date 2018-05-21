/****************************************************************************
 *
 *  brazovskii.c
 *
 *  This is the implementation of the Brazovskii free energy:
 *
 *  F[phi] = (1/2) A phi^2 + (1/4) B phi^4 + (1/2) kappa (\nabla \phi)^2
 *                                         + (1/2) C (\nabla^2 \phi)^2
 *
 *  so there is one additional term compared with the symmetric phi^4
 *  case. The original reference is S. A. Brazovskii, Sov. Phys. JETP,
 *  {\bf 41} 85 (1975). One can see also, for example, Xu et al, PRE
 *  {\bf 74} 011505 (2006) for details.
 *
 *  Parameters:
 *
 *  One should have b, c > 0 for stability purposes. Then for a < 0
 *  and kappa > 0 one gets two homogenous phases with
 *  phi = +/- sqrt(-a/b) cf. the symmetric case.
 *
 *  Negative kappa favours the presence of interfaces, and lamellae
 *  can form. Approximately, the lamellar phase can be described by
 *  phi ~= A sin(k_0 x) in the traverse direction, where
 *  A^2 = 4 (1 + kappa^2/4cb)/3 and k_0 = sqrt(-kappa/2c). 
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "util.h"
#include "brazovskii.h"

struct fe_brazovskii_s {
  fe_t super;
  pe_t * pe;                        /* Parallel environment */
  cs_t * cs;                        /* Coordinate system */
  fe_brazovskii_param_t * param;    /* Parameters */
  field_t * phi;                    /* Reference to order parameter field */
  field_grad_t * dphi;              /* Reference to gradient field */
  fe_brazovskii_t * target;         /* Device copy */
};

static __constant__ fe_brazovskii_param_t const_param;


static fe_vt_t fe_braz_hvt = {
  (fe_free_ft)      fe_brazovskii_free,
  (fe_target_ft)    fe_brazovskii_target,
  (fe_fed_ft)       fe_brazovskii_fed,
  (fe_mu_ft)        fe_brazovskii_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_brazovskii_str,
  (fe_str_ft)       fe_brazovskii_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_brazovskii_str_v,
  (fe_stress_v_ft)  fe_brazovskii_str_v,
  (fe_stress_v_ft)  NULL
};

static  __constant__ fe_vt_t fe_braz_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_brazovskii_fed,
  (fe_mu_ft)        fe_brazovskii_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_brazovskii_str,
  (fe_str_ft)       fe_brazovskii_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_brazovskii_str_v,
  (fe_stress_v_ft)  fe_brazovskii_str_v,
  (fe_stress_v_ft)  NULL
};


/*****************************************************************************
 *
 *  fe_brazovskii_create
 *
 *****************************************************************************/

__host__ int fe_brazovskii_create(pe_t * pe, cs_t * cs, field_t * phi,
				  field_grad_t * dphi,
				  fe_brazovskii_t ** p) {

  int ndevice;
  fe_brazovskii_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(phi);
  assert(dphi);

  obj = (fe_brazovskii_t *) calloc(1, sizeof(fe_brazovskii_t));
  if (obj == NULL) pe_fatal(pe, "calloc(fe_brazovskii_t) failed\n");

  obj->param =
    (fe_brazovskii_param_t *) calloc(1, sizeof(fe_brazovskii_param_t));
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_brazovskii_param_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->super.func = &fe_braz_hvt;
  obj->super.id = FE_BRAZOVSKII;

  /* Allocate device memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    fe_brazovskii_param_t * tmp;
    fe_vt_t * vt;
    tdpMalloc((void **) &obj->target, sizeof(fe_brazovskii_t));
    tdpMemset(obj->target, 0, sizeof(fe_brazovskii_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, &tmp, sizeof(fe_brazovskii_t *),
	      tdpMemcpyHostToDevice);
    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_braz_dvt));
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
 *  fe_brazovskii_free
 *
 *****************************************************************************/

__host__ int fe_brazovskii_free(fe_brazovskii_t * fe) {

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
 *  fe_brazovskii_target
 *
 ****************************************************************************/

__host__ int fe_brazovskii_target(fe_brazovskii_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_brazovskii_param_set
 *
 *  No constraints on the parameters are enforced, but see comments
 *  above.
 *
 ****************************************************************************/

__host__ int fe_brazovskii_param_set(fe_brazovskii_t * fe,
				     fe_brazovskii_param_t values) {
  assert(fe);

  *fe->param = values;

  return 0;
}

/*****************************************************************************
 *
 *  fe_brazovskii_param
 *
 *****************************************************************************/

__host__ __device__ int fe_brazovskii_param(fe_brazovskii_t * fe,
					    fe_brazovskii_param_t * values) {
  assert(fe);
  assert(values);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_brazovskii_amplitude
 *
 *  Return the single-mode approximation amplitude.
 *
 ****************************************************************************/

__host__ __device__ int fe_brazovskii_amplitude(fe_brazovskii_t * fe,
						double * a0) {
  double b, c, kappa;

  assert(fe);

  b = fe->param->b;
  c = fe->param->c;
  kappa = fe->param->kappa;

  *a0 = sqrt((4.0/3.0)*(1.0 + kappa*kappa/(4.0*b*c)));

  return 0;
}

/****************************************************************************
 *
 *  fe_brazovskii_wavelength
 *
 *  Return the single-mode approximation wavelength 2\pi / k_0.
 *
 ****************************************************************************/

__host__ __device__ int fe_brazovskii_wavelength(fe_brazovskii_t * fe,
						 double * lambda) {
  assert(fe);

  *lambda = 2.0*4.0*atan(1.0) / sqrt(-fe->param->kappa/(2.0*fe->param->c));

  return 0;
}

/****************************************************************************
 *
 *  fe_brazovskii_fed
 *
 *  The free energy density at position index.
 *
 ****************************************************************************/

__host__ __device__
int fe_brazovskii_fed(fe_brazovskii_t * fe, int index, double * fed) {

  double phi;
  double dphi[3];
  double delsq;

  assert(fe);

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, dphi);
  field_grad_scalar_delsq(fe->dphi, index, &delsq);

  *fed = 0.5*fe->param->a*phi*phi + 0.25*fe->param->b*phi*phi*phi*phi
    + 0.5*fe->param->kappa*dot_product(dphi, dphi)
    + 0.5*fe->param->c*delsq*delsq;

  return 0;
}

/****************************************************************************
 *
 *  fe_brazovskii_mu
 *
 *  The chemical potential at position index
 *
 *     \mu = \delta F / \delta \phi
 *         = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *         + c (\nabla^2)(\nabla^2 \phi)
 *
 ****************************************************************************/

__host__ __device__
int fe_brazovskii_mu(fe_brazovskii_t * fe, int index, double * mu) {

  double phi;
  double del2_phi;
  double del4_phi;

  assert(fe);

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_delsq(fe->dphi, index, &del2_phi);
  field_grad_scalar_delsq_delsq(fe->dphi, index, &del4_phi);

  *mu = fe->param->a*phi + fe->param->b*phi*phi*phi
    - fe->param->kappa*del2_phi + fe->param->c*del4_phi;

  return 0;
}

/****************************************************************************
 *
 *  fe_brazovskii_str
 *
 *  Return the chemical stress tensor for given position index.
 *
 ****************************************************************************/

__host__ __device__
int fe_brazovskii_str(fe_brazovskii_t * fe, int index,  double s[3][3]) {

  int ia, ib;
  double c, kappa;
  double phi;
  double del2_phi;
  double del4_phi;
  double grad_phi[3];
  double grad_del2_phi[3];
  double p0;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  c = fe->param->c;
  kappa = fe->param->kappa;

  field_scalar(fe->phi, index, &phi);
  field_grad_scalar_grad(fe->dphi, index, grad_phi);
  field_grad_scalar_delsq(fe->dphi, index, &del2_phi);
  field_grad_scalar_delsq_delsq(fe->dphi, index, &del4_phi);
  field_grad_scalar_grad_delsq(fe->dphi, index, grad_del2_phi);

  /* Isotropic part and tensor part */

  p0 = 0.5*fe->param->a*phi*phi + 0.75*fe->param->b*phi*phi*phi*phi
    - kappa*phi*del2_phi
    + 0.5*kappa*dot_product(grad_phi, grad_phi)
    + c*phi*del4_phi + 0.5*c*del2_phi*del2_phi
    + c*dot_product(grad_phi, grad_del2_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d[ia][ib] + kappa*grad_phi[ia]*grad_phi[ib]
      - c*(grad_phi[ia]*grad_del2_phi[ib] + grad_phi[ib]*grad_del2_phi[ia]);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_brazovskii_str_v
 *
 *  May require optimisation.
 *
 *****************************************************************************/

__host__ __device__ void fe_brazovskii_str_v(fe_brazovskii_t * fe, int index,
					     double s[3][3][NSIMDVL]) {

  int ia, ib, iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_brazovskii_str(fe, index+iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return;
}
