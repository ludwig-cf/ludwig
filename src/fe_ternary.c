/****************************************************************************
 *
 *  fe_ternary.c
 *
 *  Implementation of the surfactant free energy described by
 *  Semprebon et al, Phys Rev E 93, 033305 (2016)
 *
 *  Two order parameters are required:
 *
 *  [0] FE_PHI \phi is compositional order parameter
 *  [1] FE_PSI \psi is surfactant concentration
 *  [2] FE_RHO is 'spectating' at the moment
 * 
 *  The free energy density is:
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
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
#include "fe_ternary.h"

/* Memory order for e.g., field[2], mu[3] */
#define FE_PHI 0
#define FE_PSI 1
#define FE_RHO 2

/* Virtual function table (host) */

static fe_vt_t fe_ternary_hvt = {
    (fe_free_ft)      fe_ternary_free,  /* Virtual destructor */
    (fe_target_ft)    fe_ternary_target,/* Return target pointer */
    (fe_fed_ft)       fe_ternary_fed,   /* Free energy density */
    (fe_mu_ft)        fe_ternary_mu,    /* Chemical potential */
    (fe_mu_solv_ft)   NULL,
    (fe_str_ft)       fe_ternary_str,   /* Total stress */
    (fe_str_ft)       fe_ternary_str,   /* Symmetric stress */
    (fe_str_ft)       NULL,             /* Antisymmetric stress (not used) */
    (fe_hvector_ft)   NULL,             /* Not relevant */
    (fe_htensor_ft)   NULL,             /* Not relevant */
    (fe_htensor_v_ft) NULL,             /* Not reelvant */
    (fe_stress_v_ft)  fe_ternary_str_v, /* Total stress (vectorised version) */
    (fe_stress_v_ft)  fe_ternary_str_v, /* Symmetric part (vectorised) */
    (fe_stress_v_ft)  NULL              /* Antisymmetric part (no used) */
};

static __constant__ fe_ternary_param_t const_param;

/****************************************************************************
 *
 *  fe_ternary_create
 *
 ****************************************************************************/

int fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi,
                      field_grad_t * dphi, fe_ternary_param_t param,
                      fe_ternary_t ** fe) {
  int ndevice;
  fe_ternary_t * obj = NULL;
    
  assert(pe);
  assert(cs);
  assert(fe);
  assert(phi);
  assert(dphi);
    
  obj = (fe_ternary_t *) calloc(1, sizeof(fe_ternary_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_surf1_t) failed\n");
    
  obj->param = (fe_ternary_param_t *) calloc(1, sizeof(fe_ternary_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_ternary_param_t) fail\n");
    
  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->super.func = &fe_ternary_hvt;
  obj->super.id = FE_TERNARY;
    
  /* Allocate target memory, or alias */
    
  tdpGetDeviceCount(&ndevice);
    
  if (ndevice == 0) {
    fe_ternary_param_set(obj, param);
    obj->target = obj;
  }
  else {
    fe_ternary_param_t * tmp;
    tdpMalloc((void **) &obj->target, sizeof(fe_ternary_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, tmp, sizeof(fe_ternary_param_t *),
	      tdpMemcpyHostToDevice);
    /* Now copy. */
    assert(0); /* No implementation */
  }
    
  *fe = obj;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_free
 *
 ****************************************************************************/

__host__ int fe_ternary_free(fe_ternary_t * fe) {
    
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
 *  fe_ternary_info
 *
 *  Some information on parameters etc.
 *
 ****************************************************************************/

__host__ int fe_ternary_info(fe_ternary_t * fe) {

  int wet;
  double sigma[3];
  double theta[3];
  double h1, h2, h3;

  pe_t * pe = NULL;
    
  assert(fe);
    
  pe = fe->pe;
    
  fe_ternary_sigma(fe, sigma);
    
  pe_info(pe, "Ternary free energy parameters:\n");
  pe_info(pe, "Surface penalty kappa1 = %12.5e\n", fe->param->kappa1);
  pe_info(pe, "Surface penalty kappa2 = %12.5e\n", fe->param->kappa2);
  pe_info(pe, "Surface penalty kappa3 = %12.5e\n", fe->param->kappa3);
  pe_info(pe, "Interface width alpha  = %12.5e\n", fe->param->alpha);
    
  pe_info(pe, "\n");
  pe_info(pe, "Derived quantities\n");
  pe_info(pe, "Interfacial tension 12 = %12.5e\n",  sigma[0]);
  pe_info(pe, "Interfacial tension 23 = %12.5e\n",  sigma[1]);
  pe_info(pe, "Interfacial tension 13 = %12.5e\n",  sigma[2]);

  /* Todo: check for equilibrium possible here? */

  fe_ternary_angles(fe, theta);
  pe_info(pe, "Equilibrium angle    1 = %12.5e\n", theta[0]);
  pe_info(pe, "Equilibrium angle    2 = %12.5e\n", theta[1]);
  pe_info(pe, "Equilibrium angle    3 = %12.5e\n", theta[2]);

  /* Wetting (if appropriate) */
  h1 = fe->param->h1;
  h2 = fe->param->h2;
  h3 = fe->param->h3;
  wet = (h1 > 0.0 || h2 > 0.0 || h3 > 0.0);

  if (wet) {
    printf("\n");
    printf("Solid wetting parameters:\n");
    pe_info(pe, "Wetting parameter   h1 = %12.5e\n", h1);
    pe_info(pe, "Wetting parameter   h2 = %12.5e\n", h2);
    pe_info(pe, "Wetting parameter   h3 = %12.5e\n", h3);
    fe_ternary_wetting_angles(fe, theta);
    pe_info(pe, "Wetting angle theta_12 = %12.5e\n", theta[0]);
    pe_info(pe, "Wetting angle theta_23 = %12.5e\n", theta[1]);
    pe_info(pe, "Wetting angle theta_31 = %12.5e\n", theta[2]);
  }

  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_target
 *
 ****************************************************************************/

__host__ int fe_ternary_target(fe_ternary_t * fe, fe_t ** target) {
    
  assert(fe);
  assert(target);
    
  *target = (fe_t *) fe->target;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_param_set
 *
 ****************************************************************************/

__host__ int fe_ternary_param_set(fe_ternary_t * fe, fe_ternary_param_t vals) {
    
  assert(fe);
    
  *fe->param = vals;
    
  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_param
 *
 *****************************************************************************/

__host__ int fe_ternary_param(fe_ternary_t * fe, fe_ternary_param_t * values) {

  assert(fe);
    
  *values = *fe->param;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_sigma
 *  Interfacial tensions s_12, s_23, s_13
 *
 ****************************************************************************/

__host__ int fe_ternary_sigma(fe_ternary_t * fe,  double * sigma) {
    
  double alpha, kappa1, kappa2, kappa3;
    
  assert(fe);
  assert(sigma);
    
  alpha  = fe->param->alpha;
  kappa1 = fe->param->kappa1;
  kappa2 = fe->param->kappa2;
  kappa3 = fe->param->kappa3;
    
  sigma[0] = alpha*(kappa1 + kappa2)/6.0;
  sigma[1] = alpha*(kappa2 + kappa3)/6.0;
  sigma[2] = alpha*(kappa1 + kappa3)/6.0;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_angles
 *
 *  Return the three equilibrium angles: theta_1, theta_2, theta_3
 *  at the fluid three-phase contact. (In degrees.)
 *
 *
 *  Using the sine rule, and the cosine rule, we have, e.g.,
 *
 *   cos(pi - theta_1) = [(s_12^2 + s_13^2) - s_23^2] / 2s_12 s_13
 *
 *  with tensions s_12, s_13, and s_23.
 *
 ****************************************************************************/

__host__ int fe_ternary_angles(fe_ternary_t * fe, double * theta) {

  double sigma[3];
  double a1, a2, a3;
  double d1, d2;
  PI_DOUBLE(pi);

  assert(fe);
  assert(theta);
    
  fe_ternary_sigma(fe, sigma);

  d1 = sigma[1]*sigma[1] - (sigma[0]*sigma[0] + sigma[2]*sigma[2]);
  d2 = 2.0*sigma[0]*sigma[2];
  a1 = acos(d1/d2)*180.0/pi;

  d1 = sigma[2]*sigma[2] - (sigma[0]*sigma[0] + sigma[1]*sigma[1]);
  d2 = 2.0*sigma[0]*sigma[1];
  a2 = acos(d1/d2)*180.0/pi;

  d1 = sigma[0]*sigma[0] - (sigma[1]*sigma[1] + sigma[2]*sigma[2]);
  d2 = 2.0*sigma[1]*sigma[2];
  a3 = acos(d1/d2)*180.0/pi;

  theta[0] = a1;
  theta[1] = a2;
  theta[2] = a3;

  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_wetting_angles
 *
 *  Return the three angles: theta_12, thera_23, theta_31
 *  when solid wetting parameters h1, h2, h3 are available.
 *
 ****************************************************************************/

__host__ int fe_ternary_wetting_angles(fe_ternary_t * fe, double * angle) { 

  double a, h;
  double kappa1, kappa2, kappa3;
  double f1, factor[3];
  PI_DOUBLE(pi);

  a = fe->param->alpha;
  kappa1 = fe->param->kappa1;
  kappa2 = fe->param->kappa2;
  kappa3 = fe->param->kappa3;

  h = fe->param->h1;
  f1 = pow(a*kappa1 + 4.0*h, 1.5) - pow(a*kappa1 - 4.0*h, 1.5);
  factor[0] = f1/sqrt(a*kappa1);

  h = fe->param->h2;
  f1 = pow(a*kappa2 + 4.0*h, 1.5) - pow(a*kappa2 - 4.0*h, 1.5);
  factor[1] = f1/sqrt(a*kappa2);

  h = fe->param->h3;
  f1 = pow(a*kappa3 + 4.0*h, 1.5) - pow(a*kappa3 - 4.0*h, 1.5);
  factor[2] = f1/sqrt(a*kappa3);

  /* angles: 12, 23, 31 */
  angle[0] = acos((factor[0] - factor[1])/(2.0*(kappa1 + kappa2)))*180.0/pi;
  angle[1] = acos((factor[1] - factor[2])/(2.0*(kappa2 + kappa3)))*180.0/pi;
  angle[2] = acos((factor[2] - factor[0])/(2.0*(kappa3 + kappa1)))*180.0/pi;

  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_xi0
 *
 *  Interfacial width.    interfical width = alpha
 *
 ****************************************************************************/

__host__ int fe_ternary_xi0(fe_ternary_t * fe, double * xi0) {
    
  assert(fe);
  assert(xi0);
    
  *xi0 = fe->param->alpha;
    
  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_fed
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

__host__ __device__ int fe_ternary_fed(fe_ternary_t * fe, int index,
				       double * fed) {
    
  int ia;
  double field[2];
  double phi;
  double psi;
  double rho;
  double grad[2][3];
  double drho;       /* temporary gradient of rho */
  double d3, dsum;
  double s1, s2, fe1, fe2;
  double kappa1, kappa2, kappa3, alpha2;

  assert(fe);
    
  kappa1 = fe->param->kappa1;
  kappa2 = fe->param->kappa2;
  kappa3 = fe->param->kappa3;
  alpha2 = fe->param->alpha*fe->param->alpha;

  field_scalar_array(fe->phi, index, field);
    
  rho = 1.0;
  phi = field[FE_PHI];
  psi = field[FE_PSI];
    
  drho = 0.0;
  field_grad_pair_grad(fe->dphi, index, grad);
    
  dsum = 0.0;
  for (ia = 0; ia < 3; ia++) {
    d3 = drho + grad[FE_PHI][ia] - grad[FE_PSI][ia];
    dsum += d3*d3;
  }

  s1  = rho + phi - psi;
  s2  = 2.0 + psi - rho - phi;
  fe1 = 0.03125*kappa1*s1*s1*s2*s2 + 0.125*alpha2*kappa1*dsum;
    
  dsum = 0.0;
  for (ia = 0; ia < 3; ia++) {
    d3 = drho - grad[FE_PHI][ia] - grad[FE_PSI][ia];
    dsum += d3*d3;
  }

  s1  = rho - phi - psi;
  s2  = 2.0 + psi - rho + phi;
  fe2 = 0.03125*kappa2*s1*s1*s2*s2 + 0.125*alpha2*kappa2*dsum;

  s1 = 0.5*kappa3*psi*psi*(1.0 - psi)*(1.0 - psi);
  s2 = 0.5*alpha2*kappa3*dot_product(grad[FE_PSI], grad[FE_PSI]);

  *fed = fe1 + fe2 + s1 + s2;

  return 0;
}

/****************************************************************************
 *
 *  fe_ternary_mu
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

__host__ int fe_ternary_mu(fe_ternary_t * fe, int index, double * mu) {
    
    double phi;
    double psi;
    double rho;
    double field[2];
    double delsq[2];
    double delsq_rho; /* laplace operator*/
    double kappa1, kappa2, kappa3, alpha2;
    double krhorho, kphipsi, kpsipsi;
    double s1, s2;
    
    assert(fe);
    assert(mu);
    
    kappa1 = fe->param->kappa1;
    kappa2 = fe->param->kappa2;
    kappa3 = fe->param->kappa3;
    alpha2 = fe->param->alpha*fe->param->alpha;

    krhorho = 0.25*alpha2*(kappa1 + kappa2);
    kphipsi = 0.25*alpha2*(kappa2 - kappa1);
    kpsipsi = 0.25*alpha2*(kappa1 + kappa2 + 4.0*kappa3);

    field_scalar_array(fe->phi, index, field);

    rho = 1.0;    
    phi = field[FE_PHI];
    psi = field[FE_PSI];
    
    field_grad_pair_delsq(fe->dphi, index, delsq);

    delsq_rho = 0.0;
    
    /* mu_phi */

    s1 = (rho + phi - psi)*(rho + phi - psi - 2.0)*(rho + phi - psi - 1.0);
    s2 = (rho - phi - psi)*(rho - phi - psi - 2.0)*(rho - phi - psi - 1.0);

    mu[FE_PHI] = 0.125*kappa1*s1 - 0.125*kappa2*s2
               + kphipsi*(delsq_rho - delsq[FE_PSI]) - krhorho*delsq[FE_PHI];

    /* mu_psi */
    s1 = (rho + phi - psi)*(rho + phi - psi - 2.0)*(rho + phi - psi - 1.0);
    s2 = (rho - phi - psi)*(rho - phi - psi - 2.0)*(rho - phi - psi - 1.0);

    mu[FE_PSI] = -0.125*kappa1*s1 - 0.125*kappa2*s2 
               + kappa3*psi*(psi - 1.0)*(2.0*psi - 1.0)
               + krhorho*delsq_rho - kphipsi*delsq[FE_PHI]
               - kpsipsi*delsq[FE_PSI];

    /* mu_rho */
    s1 = (rho + phi - psi)*(rho + phi - psi - 2.0)*(rho + phi - psi - 1.0);
    s2 = (rho - phi - psi)*(rho - phi - psi - 2.0)*(rho - phi - psi - 1.0);

    mu[FE_RHO] = 0.125*kappa1*s1 - 0.125*kappa2*s2
               + krhorho*(delsq[FE_PSI] - delsq[FE_PHI]) + kphipsi*delsq_rho;

    return 0;
}

/****************************************************************************
 *
 *  fe_ternary_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 ****************************************************************************/

__host__ int fe_ternary_str(fe_ternary_t * fe, int index, double s[3][3]) {
    
    int ia, ib;
    double field[2];
    double phi, phi2, dphi[3], dphi2;
    double psi, psi2, dpsi[3], dpsi2;
    double rho, rho2, drho[3], drho2;
    double delsq[3];
    double grad[2][3];
    double p0;
    double p1, p2, p3, p4, p5, p6;
    double kappa1, kappa2, kappa3, alpha2;
    double krhorho, kphiphi, kpsipsi;
    double krhophi, krhopsi, kphipsi;
    double drhodphi, drhodpsi, dphidpsi;
    KRONECKER_DELTA_CHAR(d);
    
    assert(fe);

    kappa1 = fe->param->kappa1;
    kappa2 = fe->param->kappa2;
    kappa3 = fe->param->kappa3;
    alpha2 = fe->param->alpha*fe->param->alpha;

    krhorho = 0.25*alpha2*(kappa1 + kappa2);
    kphiphi = krhorho;
    kpsipsi = 0.25*alpha2*(kappa1 + kappa2 + 4.0*kappa3);
    krhophi = 0.25*alpha2*(kappa1 - kappa2);
    krhopsi = - krhorho;
    kphipsi = - krhophi;

    field_scalar_array(fe->phi, index, field);
    
    rho = 1.0;
    phi = field[FE_PHI];
    psi = field[FE_PSI];
    rho2 = rho*rho;
    phi2 = phi*phi;
    psi2 = psi*psi;
    
    field_grad_pair_grad(fe->dphi, index, grad);
    field_grad_pair_delsq(fe->dphi, index, delsq);

    dphi[X] = grad[FE_PHI][X]; dpsi[X] = grad[FE_PSI][X];
    dphi[Y] = grad[FE_PHI][Y]; dpsi[Y] = grad[FE_PSI][Y];
    dphi[Z] = grad[FE_PHI][Z]; dpsi[Z] = grad[FE_PSI][Z];

    drho[X] = 0.0; drho[Y] = 0.0; drho[Z] = 0.0;
    delsq[FE_RHO] = 0.0;

    /* Construct the bulk isotropic term p0 (with 4 pieces) */

    p1 = (kappa1 + kappa2)*
      (0.09375*(rho2*rho2 + phi2*phi2)
       + 0.5625*(rho2*phi2 + rho2*psi2 + phi2*psi2) 
       - 0.3750*rho*psi*(rho2 + psi2) 
       + 0.75*(rho2*psi  - rho*phi2 - rho*psi2 + phi2*psi)
       - 0.25*rho2*rho + 0.125*rho2 + 0.125*phi2 - 0.25*rho*psi
       - 1.125*rho*phi2*psi);

    p2 = (kappa1 - kappa2)*
      (0.375*(rho2*rho*phi + rho*phi2*phi - phi2*phi*psi - phi*psi2*psi)
       -0.25*phi2*phi - 0.75*(rho2*phi + phi*psi2) + 0.25*(rho*phi - phi*psi)
       + 1.125*rho*phi*psi2 - 1.125*rho2*phi*psi + 1.5*rho*phi*psi);

    p3 = 0.25*(kappa1 + kappa2 - 8.0*kappa3)*psi2*psi;
    p4 = (kappa1 + kappa2 + 16.0*kappa3)*(0.09375*psi2 + 0.125)*psi2;

    p0 = p1 + p2 + p3 + p4;

    /* Additional terms in d_\alpha\beta (6 terms) */

    drho2    = drho[X]*drho[X] + drho[Y]*drho[Y] + drho[Z]*drho[Z];
    dphi2    = dphi[X]*dphi[X] + dphi[Y]*dphi[Y] + dphi[Z]*dphi[Z];
    dpsi2    = dpsi[X]*dpsi[X] + dpsi[Y]*dpsi[Y] + dpsi[Z]*dpsi[Z];

    drhodphi = drho[X]*dphi[X] + drho[Y]*dphi[Y] + drho[Z]*dphi[Z];
    drhodpsi = drho[X]*dpsi[X] + drho[Y]*dpsi[Y] + drho[Z]*dpsi[Z];
    dphidpsi = dphi[X]*dpsi[X] + dphi[Y]*dpsi[Y] + dphi[Z]*dpsi[Z];

    p1 = 0.5*drho2 + rho*delsq[FE_RHO];
    p2 = 0.5*dphi2 + phi*delsq[FE_PHI];
    p3 = 0.5*dpsi2 + psi*delsq[FE_PSI];
    p4 = drhodphi  + rho*delsq[FE_PHI] + phi*delsq[FE_RHO];
    p5 = drhodpsi  + rho*delsq[FE_PSI] + psi*delsq[FE_RHO];
    p6 = dphidpsi  + phi*delsq[FE_PSI] + psi*delsq[FE_PHI];

    /* Final stress */
    
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {

	s[ia][ib] = p0*d[ia][ib]
	          + krhorho*(drho[ia]*drho[ib] - p1*d[ia][ib])
	          + kphiphi*(dphi[ia]*dphi[ib] - p2*d[ia][ib])
                  + kpsipsi*(dpsi[ia]*dpsi[ib] - p3*d[ia][ib])
                  + krhophi*(drho[ia]*dphi[ib]
                           + dphi[ia]*drho[ib] - p4*d[ia][ib])
                  + krhopsi*(drho[ia]*dpsi[ib]
                           + dpsi[ia]*drho[ib] - p5*d[ia][ib])
                  + kphipsi*(dphi[ia]*dpsi[ib]
                           + dpsi[ia]*dphi[ib] - p6*d[ia][ib]);
      }
    }

    return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_str_v
 *
 *  Stress (vectorised version) Currently a patch-up.
 *
 *****************************************************************************/

__host__ int fe_ternary_str_v(fe_ternary_t * fe, int index,
			      double s[3][3][NSIMDVL]) {

  int ia, ib;
  int iv;
  double s1[3][3];
    
  assert(fe);
    
  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_ternary_str(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return 0;
}
