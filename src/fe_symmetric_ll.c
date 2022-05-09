/****************************************************************************
 *
 *  fe_symmetric_ll.c
 *
 *  Two binary mixture free energy in one structure.
 *  Adapted from fe_ternary.c
 *
 *  Two order parameters are required:
 *
 *  [0] FE_PHI \phi is compositional order parameter
 *  [1] FE_PSI \psi is compositional order parameter
 *  [2] FE_RHO is 'spectating' at the moment
 * 
 *  The free energy density is Cahn Hilliard binary mixture
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "fe_symmetric_ll.h"

/* Memory order for e.g., field[2], mu[3] */
#define FE_PHI 0
#define FE_PSI 1
#define FE_RHO 2

/* Virtual function table (host) */

static fe_vt_t fe_symmetric_ll_hvt = {
    (fe_free_ft)      fe_symmetric_ll_free,  /* Virtual destructor */
    (fe_target_ft)    fe_symmetric_ll_target,/* Return target pointer */
    (fe_fed_ft)       fe_symmetric_ll_fed,   /* Free energy density */
    (fe_mu_ft)        fe_symmetric_ll_mu,    /* Chemical potential */
    (fe_mu_solv_ft)   NULL,
    (fe_str_ft)       fe_symmetric_ll_str,   /* Total stress */
    (fe_str_ft)       fe_symmetric_ll_str,   /* Symmetric stress */
    (fe_str_ft)       NULL,             /* Antisymmetric stress (not used) */
    (fe_hvector_ft)   NULL,             /* Not relevant */
    (fe_htensor_ft)   NULL,             /* Not relevant */
    (fe_htensor_v_ft) NULL,             /* Not reelvant */
    (fe_stress_v_ft)  fe_symmetric_ll_str_v, /* Total stress (vectorised version) */
    (fe_stress_v_ft)  fe_symmetric_ll_str_v, /* Symmetric part (vectorised) */
    (fe_stress_v_ft)  NULL              /* Antisymmetric part (not used) */
};

static __constant__ fe_vt_t fe_symmetric_ll_dvt = {
    (fe_free_ft)      NULL,             /* Virtual destructor */
    (fe_target_ft)    NULL,             /* Return target pointer */
    (fe_fed_ft)       fe_symmetric_ll_fed,   /* Free energy density */
    (fe_mu_ft)        fe_symmetric_ll_mu,    /* Chemical potential */
    (fe_mu_solv_ft)   NULL,
    (fe_str_ft)       fe_symmetric_ll_str,   /* Total stress */
    (fe_str_ft)       fe_symmetric_ll_str,   /* Symmetric stress */
    (fe_str_ft)       NULL,             /* Antisymmetric stress (not used) */
    (fe_hvector_ft)   NULL,             /* Not relevant */
    (fe_htensor_ft)   NULL,             /* Not relevant */
    (fe_htensor_v_ft) NULL,             /* Not reelvant */
    (fe_stress_v_ft)  fe_symmetric_ll_str_v, /* Total stress (vectorised version) */
    (fe_stress_v_ft)  fe_symmetric_ll_str_v, /* Symmetric part (vectorised) */
    (fe_stress_v_ft)  NULL              /* Antisymmetric part (not used) */
};

static __constant__ fe_symmetric_ll_param_t const_param;

/****************************************************************************
 *
 *  fe_symmetric_ll_create
 *
 ****************************************************************************/

int fe_symmetric_ll_create(pe_t * pe, cs_t * cs, field_t * phi,
                      field_grad_t * dphi, fe_symmetric_ll_param_t param,
                      fe_symmetric_ll_t ** fe) {
  int ndevice;
  fe_symmetric_ll_t * obj = NULL;
    
  assert(pe);
  assert(cs);
  assert(fe);
  assert(phi);
  assert(dphi);
    
  obj = (fe_symmetric_ll_t *) calloc(1, sizeof(fe_symmetric_ll_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_surf1_t) failed\n");
    
  obj->param = (fe_symmetric_ll_param_t *) calloc(1, sizeof(fe_symmetric_ll_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_symmetric_ll_param_t) fail\n");
    
  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->super.func = &fe_symmetric_ll_hvt;
  obj->super.id = FE_SYMMETRIC_LL;
    
  /* Allocate target memory, or alias */
    
  tdpGetDeviceCount(&ndevice);
    
  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    tdpAssert(tdpMalloc((void **) &obj->target, sizeof(fe_symmetric_ll_t)));
    tdpAssert(tdpMemset(obj->target, 0, sizeof(fe_symmetric_ll_t)));

    /* Device function table */
    {
      fe_vt_t * vt = NULL;
      tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_symmetric_ll_dvt));
      tdpAssert(tdpMemcpy(&obj->target->super.func, &vt, sizeof(fe_vt_t *),
			  tdpMemcpyHostToDevice));
    }

    /* Constant symbols */
    {
      fe_symmetric_ll_param_t * tmp = NULL;
      tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
      tdpAssert(tdpMemcpy(&obj->target->param, &tmp,
			  sizeof(fe_symmetric_ll_param_t *),
			  tdpMemcpyHostToDevice));
    }

    /* Order parameter and gradient */
    tdpAssert(tdpMemcpy(&obj->target->phi, &phi->target, sizeof(field_t *),
			tdpMemcpyHostToDevice));
    tdpAssert(tdpMemcpy(&obj->target->dphi, &dphi->target,
			sizeof(field_grad_t *), tdpMemcpyHostToDevice));
  }
    
  fe_symmetric_ll_param_set(obj, param);
  *fe = obj;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_free
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_free(fe_symmetric_ll_t * fe) {
    
  int ndevice;
    
  assert(fe);
    
  tdpGetDeviceCount(&ndevice);
  if (ndevice > 0) tdpAssert(tdpFree(fe->target));
    
  free(fe->param);
  free(fe);
    
  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_info
 *
 *  Some information on parameters etc.
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_info(fe_symmetric_ll_t * fe) {

  double sigma[2];
  double kappa1, kappa2;

  pe_t * pe = NULL;
    
  assert(fe);
    
  pe = fe->pe;
  kappa1 = fe->param->kappa1;  
  kappa2 = fe->param->kappa2;  

  fe_symmetric_ll_sigma(fe, sigma);
    
  pe_info(pe, "Symmetric_ll free energy parameters:\n");
  pe_info(pe, "Gradient penalty kappa1 = %12.5e\n", fe->param->kappa1);
  pe_info(pe, "Gradient penalty kappa2 = %12.5e\n", fe->param->kappa2);
    
  pe_info(pe, "\n");
  pe_info(pe, "Derived quantities\n");
  pe_info(pe, "Interfacial tension 1 = %12.5e\n",  sigma[0]);
  pe_info(pe, "Interfacial tension 2 = %12.5e\n",  sigma[1]);

  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_target
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_target(fe_symmetric_ll_t * fe, fe_t ** target) {
    
  assert(fe);
  assert(target);
    
  *target = (fe_t *) fe->target;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_param_set
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_param_set(fe_symmetric_ll_t * fe, fe_symmetric_ll_param_t vals) {

  assert(fe);

  *fe->param = vals;

  tdpMemcpyToSymbol(tdpSymbol(const_param), fe->param,
		    sizeof(fe_symmetric_ll_param_t), 0, tdpMemcpyHostToDevice);
  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_ll_param
 *
 *****************************************************************************/

__host__ int fe_symmetric_ll_param(fe_symmetric_ll_t * fe, fe_symmetric_ll_param_t * values) {

  assert(fe);
    
  *values = *fe->param;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_sigma
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_sigma(fe_symmetric_ll_t * fe,  double * sigma) {
    
  double a1, b1, kappa1;
  double a2, b2, kappa2;
    
  assert(fe);
  assert(sigma);
    
  a1  = fe->param->a1;
  b1  = fe->param->b1;
  kappa1 = fe->param->kappa1;

  a2  = fe->param->a2;
  b2  = fe->param->b2;
  kappa2 = fe->param->kappa2;
    
  sigma[0] = sqrt(-8.0*kappa1*a1*a1*a1/(9.0*b1*b1));
  sigma[1] = sqrt(-8.0*kappa2*a2*a2*a2/(9.0*b2*b2));
    
  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_xi0
 *
 *  Interfacial width.    interfical width = alpha
 *
 ****************************************************************************/

__host__ int fe_symmetric_ll_xi0(fe_symmetric_ll_t * fe, double * xi0) {
    
  assert(fe);
  assert(xi0);
    
  *xi0 = sqrt(-2.0*fe->param->kappa1/fe->param->a1);

    
  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_fed
 *
 *  The free energy density is:
 *
 *   (1/32) kappa1 (\rho + \phi - \psi)^2 (2 + \psi - \rho - \phi)^2
 * + (1/8)  kappa1 \alpha^2 (\nabla\rho + \nabla\phi - \nalba\psi)^2
 * + (1/32) kappa2 (\rho - \phi - \psi)^2 (2 + \psi - \rho + \phi)^2
 * + (1/8)  kappa2 (\alpha)^2 (\nabla\rho - \nabla\phi - \nalba\psi)^2
 * + (1/2)  kappa3 (\psi)^2 (1 - \psi)^2
 * + (1/2)  kappa3 (\alpha)^2 (\nalba\psi)^2
 *
 *
 ****************************************************************************/

__host__ __device__ int fe_symmetric_ll_fed(fe_symmetric_ll_t * fe, int index,
				       double * fed) {
    
  int ia;
  double field[2];
  double phi;
  double psi;
  double grad[2][3];
  double a1, b1, kappa1;
  double a2, b2, kappa2;

  assert(fe);
    
  a1 = fe->param->a1;
  b1 = fe->param->b1;
  kappa1 = fe->param->kappa1;

  a2 = fe->param->a2;
  b2 = fe->param->b2;
  kappa2 = fe->param->kappa2;

  field_scalar_array(fe->phi, index, field);
    
  phi = field[FE_PHI];
  psi = field[FE_PSI];
    
  field_grad_pair_grad(fe->dphi, index, grad);
    
  *fed = (0.5*a1 + 0.25*b1*phi*phi)*phi*phi
    + 0.5*kappa1*dot_product(grad[FE_PHI], grad[FE_PHI])
    +    (0.5*a2 + 0.25*b2*psi*psi)*psi*psi
    + 0.5*kappa2*dot_product(grad[FE_PSI], grad[FE_PSI]);

  return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_mu
 *
 *  Three chemical potentials are present:
 *
 *   \mu_\rho 
 * = 1/8 kappa1  (rho + phi - psi)(rho + phi - psi - 2)(rho + phi - psi - 1)
 * - 1/8 kappa2  (rho - phi - psi)(rho - phi - psi - 2)(rho - phi - psi - 1)
 * + 1/4 alpha^2 (kappa1 + kappa2)(\Delta\psi - \Delta\phi)
 * + 1/4 alpha^2 (kappa2 - kappa1)(\Delta\rho)
 *
 *
 *    \mu_\phi
 * = 1/8 kappa1  (rho + phi - psi)(rho + phi - psi - 2)(rho + phi - psi - 1)
 * - 1/8 kappa2  (rho - phi - pai)(rho - phi - psi - 2)(rho - phi - psi - 1)
 * + 1/4 alpha^2 (kappa2 - kappa1)(\Delta\rho -\Delta\psi)
 * - 1/4 alpha^2 (kappa2 + kappa1)(\Delta\phi)
 *
 *    \mu_\psi
 * = 1/8 kappa1  (rho + phi - psi)(rho + phi - psi - 2)(rho + phi - psi - 1)
 * - 1/8 kappa2  (rho - phi - psi)(rho - phi - psi - 2)(rho - phi - psi - 1)
 * +     kappa3  psi(\psi - 1)(2psi - 1)
 * + 1/4 alpha^2 (kappa1 + kappa2)(\Delta\rho) - (kappa2 - kappa1)(\Delta\phi)
 * - 1/4 alpha^2 (kappa2 + kappa1 + 4*kappa3)(\Delta\psi)
 *
 ****************************************************************************/

__host__ __device__ int fe_symmetric_ll_mu(fe_symmetric_ll_t * fe, int index,
				      double * mu) {
    double phi;
    double psi;
    double field[2];
    double delsq[2];
    double a1, b1, kappa1;
    double a2, b2, kappa2;
    
    assert(fe);
    assert(mu);
     
    a1 = fe->param->a1;
    b1 = fe->param->b1;
    kappa1 = fe->param->kappa1;

    a2 = fe->param->a2;
    b2 = fe->param->b2;
    kappa2 = fe->param->kappa2;

    field_scalar_array(fe->phi, index, field);

    phi = field[FE_PHI];
    psi = field[FE_PSI];
    
    field_grad_pair_delsq(fe->dphi, index, delsq);
    
    /* mu_phi */
    mu[FE_PHI] = a1*phi + b1*phi*phi*phi - kappa1*delsq[FE_PHI];

    /* mu_psi */
    mu[FE_PSI] = a2*psi + b2*psi*psi*psi - kappa2*delsq[FE_PSI];

    return 0;
}

/****************************************************************************
 *
 *  fe_symmetric_ll_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 ****************************************************************************/

__host__ __device__ int fe_symmetric_ll_str(fe_symmetric_ll_t * fe, int index,
				       double s[3][3]) {
    int ia, ib;
    double field[2];
    double phi;
    double psi;
    double delsq[3];
    double grad[2][3];
    double p0, p0_phi, p0_psi;
    double a1, b1, kappa1;
    double a2, b2, kappa2;
    double d_ab;
    KRONECKER_DELTA_CHAR(d);
    
    assert(fe);
     
    a1 = fe->param->a1;
    b1 = fe->param->b1;
    kappa1 = fe->param->kappa1;

    a2 = fe->param->a2;
    b2 = fe->param->b2;
    kappa2 = fe->param->kappa2;

    field_scalar_array(fe->phi, index, field);
    phi = field[FE_PHI];
    psi = field[FE_PSI];
    
    field_grad_pair_grad(fe->dphi, index, grad);
    field_grad_pair_delsq(fe->dphi, index, delsq);

    p0_phi = 0.5*a1*phi*phi + 0.75*b1*phi*phi*phi*phi
    - kappa1*phi*delsq[FE_PHI] - 0.5*kappa1*dot_product(grad[FE_PHI], grad[FE_PHI]);

    p0_psi = 0.5*a2*psi*psi + 0.75*b2*psi*psi*psi*psi
    - kappa2*psi*delsq[FE_PSI] - 0.5*kappa2*dot_product(grad[FE_PSI], grad[FE_PSI]);

    p0 = p0_phi + p0_psi;

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        d_ab = (ia == ib);
        s[ia][ib] = p0*d_ab + kappa1*grad[FE_PHI][ia]*grad[FE_PHI][ib] + kappa2*grad[FE_PSI][ia]*grad[FE_PSI][ib];
      }
    }


    return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_ll_str_v
 *
 *  Stress (vectorised version) Currently a patch-up.
 *
 *****************************************************************************/

__host__ __device__ int fe_symmetric_ll_str_v(fe_symmetric_ll_t * fe, int index,
					 double s[3][3][NSIMDVL]) {

  int ia, ib;
  int iv;
  double s1[3][3];
    
  assert(fe);
    
  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_symmetric_ll_str(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return 0;
}
