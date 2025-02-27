/*****************************************************************************
 *
 *  lc_droplet.c
 *
 *  Routines related to liquid crystal droplet free energy
 *  and molecular field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Juho Lituvuori  (juho.lintuvuori@u-bordeaux.fr)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "blue_phase.h"
#include "symmetric.h"
#include "kernel.h"
#include "lc_droplet.h"

#define NGRAD_ 27
static const int bs_cv[NGRAD_][3] = {{ 0, 0, 0},
                                 {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},
                                 {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
                                 {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
                                 { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},
                                 { 0, 0,-1},             { 0, 0, 1},
                                 { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
                                 { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},
                                 { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
                                 { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}};

static fe_vt_t fe_drop_hvt = {
  (fe_free_ft)      fe_lc_droplet_free,
  (fe_target_ft)    fe_lc_droplet_target,
  (fe_fed_ft)       fe_lc_droplet_fed,
  (fe_mu_ft)        fe_lc_droplet_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_droplet_stress,
  (fe_str_ft)       fe_lc_droplet_str_symm,
  (fe_str_ft)       fe_lc_droplet_str_anti,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_droplet_mol_field,
  (fe_htensor_v_ft) fe_lc_droplet_mol_field_v,
  (fe_stress_v_ft)  fe_lc_droplet_stress_v,
  (fe_stress_v_ft)  fe_lc_droplet_str_symm_v,
  (fe_stress_v_ft)  fe_lc_droplet_str_anti_v
};

static __constant__ fe_vt_t fe_drop_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_lc_droplet_fed,
  (fe_mu_ft)        fe_lc_droplet_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_droplet_stress,
  (fe_str_ft)       fe_lc_droplet_str_symm,
  (fe_str_ft)       fe_lc_droplet_str_anti,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_droplet_mol_field,
  (fe_htensor_v_ft) fe_lc_droplet_mol_field_v,
  (fe_stress_v_ft)  fe_lc_droplet_stress_v,
  (fe_stress_v_ft)  fe_lc_droplet_str_symm_v,
  (fe_stress_v_ft)  fe_lc_droplet_str_anti_v
};

__host__ __device__
int fe_lc_droplet_anchoring_h(fe_lc_droplet_t * fe, int index, double h[3][3]);
__host__ __device__
int fe_lc_droplet_active_stress(const fe_lc_droplet_param_t * fp, double phi,
				double q[3][3], double s[3][3]);

__global__ void fe_lc_droplet_bf_kernel(kernel_3d_t k3d,
					fe_lc_droplet_t * fe,
					hydro_t * hydro);

static __constant__ fe_lc_droplet_param_t const_param;
static __constant__ fe_lc_param_t const_lc;

/*****************************************************************************
 *
 *  fe_lc_droplet_create
 *
 *****************************************************************************/

__host__ int fe_lc_droplet_create(pe_t * pe, cs_t * cs, fe_lc_t * lc,
				  fe_symm_t * symm,
				  fe_lc_droplet_t ** p) {

  int ndevice;
  fe_lc_droplet_t * fe = NULL;

  assert(pe);
  assert(cs);
  assert(lc);
  assert(symm);

  fe = (fe_lc_droplet_t *) calloc(1, sizeof(fe_lc_droplet_t));
  assert(fe);
  if (fe == NULL) pe_fatal(pe, "calloc(fe_lc_droplet_t) failed\n");

  fe->param =
    (fe_lc_droplet_param_t *) calloc(1, sizeof(fe_lc_droplet_param_t));
  assert(fe->param);
  if (fe->param == NULL) pe_fatal(pe, "calloc(fe_lc_droplet_param_t) failed\n");

  fe->pe = pe;
  fe->cs = cs;
  fe->lc = lc;
  fe->symm = symm;

  fe->super.func = &fe_drop_hvt;
  fe->super.id = FE_LC_DROPLET;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_lc_droplet_param_t * tmp;
    fe_vt_t * vt;
    tdpAssert( tdpMalloc((void **) &fe->target, sizeof(fe_lc_droplet_t)) );
    tdpAssert( tdpMemset(fe->target, 0, sizeof(fe_lc_droplet_t)) );
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpAssert( tdpMemcpy(&fe->target->param, &tmp,
			 sizeof(fe_lc_droplet_param_t *),
			 tdpMemcpyHostToDevice) );
    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_drop_dvt));
    tdpAssert( tdpMemcpy(&fe->target->super.func, &vt, sizeof(fe_vt_t *),
			 tdpMemcpyHostToDevice) );

    tdpAssert( tdpMemcpy(&fe->target->lc, &lc->target, sizeof(fe_lc_t *),
			 tdpMemcpyHostToDevice) );
    tdpAssert( tdpMemcpy(&fe->target->symm, &symm->target, sizeof(fe_symm_t *),
			 tdpMemcpyHostToDevice) );

    {
      /* Provide constant memory for lc parameters */
      /* If symm->param are required, must be added */
      /* Should perhaps aggregate to fe_lc_droplet_param_t itself */
      fe_lc_param_t * tmp_lc = NULL;
      fe_lc_t * x = NULL;
      tdpGetSymbolAddress((void **) &tmp_lc, tdpSymbol(const_lc));
      tdpAssert(tdpMemcpy(&x, &fe->target->lc, sizeof(fe_lc_param_t *),
			  tdpMemcpyDeviceToHost));
      tdpAssert(tdpMemcpy(&x->param, &tmp_lc, sizeof(fe_lc_param_t *),
			  tdpMemcpyHostToDevice));
    }
  }

  *p = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_free
 *
 *****************************************************************************/

__host__ int fe_lc_droplet_free(fe_lc_droplet_t * fe) {

  assert(fe);

  /* Free constituent parts, then self... */
  fe_lc_free(fe->lc);
  fe_symm_free(fe->symm);

  if (fe->target != fe) tdpAssert(tdpFree(fe->target));

  free(fe->param);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_param
 *
 *****************************************************************************/

__host__
int fe_lc_droplet_param(fe_lc_droplet_t * fe, fe_lc_droplet_param_t * param) {

  assert(fe);
  assert(param);

  *param = *fe->param;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_param_set
 *
 *****************************************************************************/

__host__ int fe_lc_droplet_param_set(fe_lc_droplet_t * fe,
				     fe_lc_droplet_param_t param) {

  assert(fe);

  *fe->param = param;

  tdpMemcpyToSymbol(tdpSymbol(const_param), fe->param,
		    sizeof(fe_lc_droplet_param_t), 0, tdpMemcpyHostToDevice);

  tdpMemcpyToSymbol(tdpSymbol(const_lc), fe->lc->param, sizeof(fe_lc_param_t),
		    0, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_target
 *
 *****************************************************************************/

__host__ int fe_lc_droplet_target(fe_lc_droplet_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/*****************************************************************************
 *
 *  lc_droplet_free_energy_density
 *
 *  Return the free energy density at lattice site index
 *
 *  f = f_symmetric + f_lc + f_anchoring
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_droplet_fed(fe_lc_droplet_t * fe, int index,
					  double * fed) {

  int ia, ib;
  double gamma;
  double q[3][3];
  double dphi[3];
  double dq[3][3][3];
  double fed_symm, fed_lc, fed_anch;

  assert(fe);

  fe_symm_fed(fe->symm, index, &fed_symm);

  field_tensor(fe->lc->q, index, q);
  field_grad_scalar_grad(fe->symm->dphi, index, dphi);
  field_grad_tensor_grad(fe->lc->dq, index, dq);

  fe_lc_droplet_gamma(fe, index, &gamma);
  fe_lc_compute_fed(fe->lc, gamma, q, dq, &fed_lc);

  fed_anch = 0.0;
  for (ia = 0; ia < 3; ia++){
    for (ib = 0; ib < 3; ib++){
      fed_anch += q[ia][ib]*dphi[ia]*dphi[ib];
    }
  }

  *fed = fed_symm + fed_lc + fe->param->w*fed_anch;

  return 0;
}

/*****************************************************************************
 *
 *  lc_droplet_gamma
 *
 *  gamma = gamma0 + delta * (1 + phi). Remember phi = phi(r).
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_droplet_gamma(fe_lc_droplet_t * fe, int index,
					    double * gamma) {

  double phi;
  assert(fe);
  assert(gamma);

  field_scalar(fe->symm->phi, index, &phi);

  *gamma = fe->param->gamma0 + fe->param->delta*(1.0 + phi);

  return 0;
}

/*****************************************************************************
 *
 *  lc_droplet_molecular_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_droplet_mol_field(fe_lc_droplet_t * fe,
						int index, double h[3][3]) {

  int ia,ib;
  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double h1[3][3], h2[3][3];
  double gamma;

  assert(fe);

  fe_lc_droplet_gamma(fe, index, &gamma);

  field_tensor(fe->lc->q, index, q);
  field_grad_tensor_grad(fe->lc->dq, index, dq);
  field_grad_tensor_delsq(fe->lc->dq, index, dsq);

  fe_lc_compute_h(fe->lc, gamma, q, dq, dsq, h1);

  fe_lc_droplet_anchoring_h(fe, index, h2);

  for (ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      h[ia][ib] = h1[ia][ib] + h2[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_mol_field_v
 *
 *  Vectorisation needs attention.
 *
 *****************************************************************************/

__host__ __device__ void fe_lc_droplet_mol_field_v(fe_lc_droplet_t * fe,
						   int index,
						   double h[3][3][NSIMDVL]) {
  int ia, ib, iv;
  double h1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_lc_droplet_mol_field(fe, index+iv, h1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	h[ia][ib][iv] = h1[ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_anchoring_h
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_droplet_anchoring_h(fe_lc_droplet_t * fe,
						  int index, double h[3][3]) {

  int ia, ib;
  double dphi[3];
  double delsq_phi;
  double dphi2;
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  field_grad_scalar_grad(fe->symm->dphi, index, dphi);
  field_grad_scalar_delsq(fe->symm->dphi, index, &delsq_phi);

  dphi2 = dphi[X]*dphi[X] + dphi[Y]*dphi[Y] + dphi[Z]*dphi[Z];

  for (ia = 0; ia < 3; ia++){
    for (ib = 0; ib < 3; ib++){
      h[ia][ib] = -fe->param->w*(dphi[ia]*dphi[ib] - r3*d[ia][ib]*dphi2);
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  lc_droplet_chemical_potential
 *
 *  Return the chemical potential at lattice site index.
 *
 *  This is d F / d phi = -A phi + B phi^3 - kappa \nabla^2 phi (symmetric)
 *                      - (1/6) A_0 \delta Q_ab^2
 *                      - (1/3) A_0 \delta  Q_ab Q_bc Q_ca
 *                      + (1/4) A_0 \delta  (Q_ab^2)^2
 *                      - 2W [ d_a phi d_b Q_ab + Q_ab d_a d_b phi ]
 *
 *****************************************************************************/

__host__ __device__ int fe_lc_droplet_mu(fe_lc_droplet_t * fe, int index,
					 double * mu) {

  int ia, ib, ic;
  double q[3][3];
  double dphi[3];
  double dq[3][3][3];
  double dabphi[3][3];
  double q2, q3;
  double wmu;
  double a0;
  double delta;
  const double r3 = (1.0/3.0);

  fe_symm_mu(fe->symm, index, mu);

  field_tensor(fe->lc->q, index, q);
  field_grad_tensor_grad(fe->lc->dq, index, dq);
  field_grad_scalar_grad(fe->symm->dphi, index, dphi);
  field_grad_scalar_dab(fe->symm->dphi, index, dabphi);

  q2 = 0.0;

  /* Q_ab^2 */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  /* Q_ab Q_bc Q_ca */

  q3 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	/* We use here the fact that q[ic][ia] = q[ia][ic] */
	q3 += q[ia][ib]*q[ib][ic]*q[ia][ic];
      }
    }
  }

  wmu = 0.0;
  for (ia = 0; ia < 3; ia++){
    for (ib = 0; ib < 3; ib++){
      wmu += (dphi[ia]*dq[ib][ia][ib] + q[ia][ib]*dabphi[ia][ib]);
    }
  }

  a0 = fe->lc->param->a0;
  delta = fe->param->delta;

  *mu += -0.5*r3*a0*delta*q2 - r3*a0*delta*q3 + 0.25*a0*delta*q2*q2
    - 2.0*fe->param->w*wmu;

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_stress
 *
 *  Return the chemical stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_droplet_stress(fe_lc_droplet_t * fe, int index, double sth[3][3]) {

  int ia, ib;
  double s1[3][3];
  double s2[3][3];

  assert(fe);

  fe_lc_droplet_str_symm(fe, index, s1);
  fe_lc_droplet_str_anti(fe, index, s2);

  for (ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      sth[ia][ib] = s1[ia][ib] + s2[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_stress_v
 *
 *  Vectorisation needs attention with called routines.
 *
 *****************************************************************************/

__host__ __device__ void fe_lc_droplet_stress_v(fe_lc_droplet_t * fe,
						int index,
						double sth[3][3][NSIMDVL]) {
  int ia, ib, iv;
  double s1[3][3];
  double s2[3][3];

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_lc_droplet_str_symm(fe, index+iv, s1);
    fe_lc_droplet_str_anti(fe, index+iv, s2);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	sth[ia][ib][iv] = s1[ia][ib] + s2[ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_str_symm
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_droplet_str_symm(fe_lc_droplet_t * fe, int index, double sth[3][3]){

  double q[3][3];
  double h[3][3];
  int ia, ib, ic;
  double qh;
  double xi;
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);

  xi = fe->lc->param->xi;

  /* No redshift at the moment */

  field_tensor(fe->lc->q, index, q);
  fe_lc_droplet_mol_field(fe, index, h);

  qh = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      qh += q[ia][ib]*h[ia][ib];
    }
  }

  /* The term in the isotropic pressure, plus that in qh */
  /* we have, for now, ignored the isotropic contribution
   * po = rho*T - lc_droplet_free_energy_density(index); */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] = 2.0*xi*(q[ia][ib] + r3*d[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -xi*h[ia][ic]*(q[ib][ic] + r3*d[ib][ic])
	  -xi*(q[ia][ic] + r3*d[ia][ic])*h[ib][ic];
      }
    }
  }

  /* Put active stress here (even if zero). */

  {
    double phi = 0.0;
    double sa[3][3] = {0};

    field_scalar(fe->symm->phi, index, &phi);
    fe_lc_droplet_active_stress(fe->param, phi, q, sa);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] += sa[ia][ib];
      }
    }
  }

  /* Additional minus sign. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_active_stress
 *
 *  The active stress is
 *
 *  S_ab = [ zeta_1 Q_ab - (1/3) zeta_0 d_ab ] f(phi)
 *
 *  where f(phi) = 0.5(1 + phi) makes phi = +1 the active phase.
 *
 ****************************************************************************/

__host__ __device__
int fe_lc_droplet_active_stress(const fe_lc_droplet_param_t * fp, double phi,
				 double q[3][3], double s[3][3]) {
  assert(fp);

  {
    double r3   = (1.0/3.0);
    double fphi = 0.5*(1.0 + phi);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	double d_ab = 1.0*(ia == ib);
	s[ia][ib] = fphi*(-r3*fp->zeta0*d_ab - fp->zeta1*q[ia][ib]);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * fe_lc_droplet_str_symm_v
 *
 *****************************************************************************/

__host__ __device__
void fe_lc_droplet_str_symm_v(fe_lc_droplet_t * fe, int index,
			      double sth[3][3][NSIMDVL]) {
  int ia, ib, iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_lc_droplet_str_symm(fe, index+iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	sth[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_str_anti
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_droplet_str_anti(fe_lc_droplet_t * fe, int index, double sth[3][3]) {

  int ia, ib, ic;
  double q[3][3], dq[3][3][3];
  double h[3][3], dsq[3][3];
  double gamma;

  assert(fe);

  /* No redshift at the moment */

  field_tensor(fe->lc->q, index, q);
  fe_lc_droplet_mol_field(fe, index, h);

  fe_lc_droplet_gamma(fe, index, &gamma);

  field_grad_tensor_grad(fe->lc->dq, index, dq);
  field_grad_tensor_delsq(fe->lc->dq, index, dsq);

  fe_lc_droplet_mol_field(fe, index, h);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] = 0.0;
    }
  }
  /* The antisymmetric piece q_ac h_cb - h_ac q_cb. We can
     rewrite it as q_ac h_bc - h_ac q_bc. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] += q[ia][ic]*h[ib][ic] - h[ia][ic]*q[ib][ic];
      }
    }
  }

  /*  Additional minus sign. */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * fe_lc_droplet_str_anti_v
 *
 *****************************************************************************/

__host__ __device__
void fe_lc_droplet_str_anti_v(fe_lc_droplet_t * fe, int index,
			      double s[3][3][NSIMDVL]) {
  int ia, ib, iv;
  double s1[3][3];

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_lc_droplet_str_anti(fe, index+iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_bodyforce
 *
 *  The driver for the bodyforce calculation.
 *
 *****************************************************************************/

__host__ int fe_lc_droplet_bodyforce(fe_lc_droplet_t * fe, hydro_t * hydro) {

  int nlocal[3] = {0};

  assert(fe);
  assert(hydro);

  cs_nlocal(fe->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(fe->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(fe_lc_droplet_bf_kernel, nblk, ntpb, 0, 0,
		    k3d, fe->target, hydro->target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_bf_kernel
 *
 *  This computes and stores the force on the fluid via
 *
 *    f_a = - H_gn \nabla_a Q_gn - phi \nabla_a mu
 *
 *  this is appropriate for the LC droplets including symmtric and blue_phase
 *  free energies. Additional force terms are included in the stress tensor.
 *
 *  The gradient of the chemical potential is computed as
 *
 *    grad_x mu = 0.5*(mu(i+1) - mu(i-1)) etc
 *
 ****************************************************************************/

__global__ void fe_lc_droplet_bf_kernel(kernel_3d_t k3d,
					fe_lc_droplet_t * fe,
					hydro_t * hydro) {
  int kindex = 0;

  assert(fe);
  assert(hydro);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ia, ib;
    int index0, indexm1, indexp1;
    double mum1, mup1;
    double force[3];

    double h[3][3];
    double q[3][3];
    double dq[3][3][3];
    double phi;

    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    index0 = kernel_3d_cs_index(&k3d, ic, jc, kc);

    field_scalar(fe->symm->phi, index0, &phi);
    field_tensor(fe->lc->q, index0, q);

    field_grad_tensor_grad(fe->lc->dq, index0, dq);
    fe_lc_droplet_mol_field(fe, index0, h);

    indexm1 = kernel_3d_cs_index(&k3d, ic-1, jc, kc);
    indexp1 = kernel_3d_cs_index(&k3d, ic+1, jc, kc);

    fe_lc_droplet_mu(fe, indexm1, &mum1);
    fe_lc_droplet_mu(fe, indexp1, &mup1);

    /* X */

    force[X] = - phi*0.5*(mup1 - mum1);

    for (ia = 0; ia < 3; ia++ ) {
      for(ib = 0; ib < 3; ib++ ) {
	force[X] -= h[ia][ib]*dq[X][ia][ib];
      }
    }

    /* Y */

    indexm1 = kernel_3d_cs_index(&k3d, ic, jc-1, kc);
    indexp1 = kernel_3d_cs_index(&k3d, ic, jc+1, kc);

    fe_lc_droplet_mu(fe, indexm1, &mum1);
    fe_lc_droplet_mu(fe, indexp1, &mup1);

    force[Y] = -phi*0.5*(mup1 - mum1);

    for (ia = 0; ia < 3; ia++ ) {
      for(ib = 0; ib < 3; ib++ ) {
	force[Y] -= h[ia][ib]*dq[Y][ia][ib];
      }
    }

    /* Z */

    indexm1 = kernel_3d_cs_index(&k3d, ic, jc, kc-1);
    indexp1 = kernel_3d_cs_index(&k3d, ic, jc, kc+1);

    fe_lc_droplet_mu(fe, indexm1, &mum1);
    fe_lc_droplet_mu(fe, indexp1, &mup1);

    force[Z] = -phi*0.5*(mup1 - mum1);

    for (ia = 0; ia < 3; ia++ ) {
      for(ib = 0; ib < 3; ib++ ) {
	force[Z] -= h[ia][ib]*dq[Z][ia][ib];
      }
    }

    /* Store the force on lattice */

    hydro_f_local_add(hydro, index0, force);
  }

  return;
}

/*****************************************************************************
 *
 *  fe_lc_droplet_bodyforce_wall
 *
 *  This computes and stores the force on the fluid via
 *
 *    f_a = - H_gn \nabla_a Q_gn - phi \nabla_a mu
 *
 *  This is appropriate for the LC droplets including symmtric and blue_phase
 *  free energies. Additional force terms are included in the stress tensor.
 *
 *  The routine takes care of gradients at walls by extrapolating the
 *  gradients from the adjacent fluid site. The gradients are computed on a
 *  27-point stencil basis.
 *
 *****************************************************************************/

int fe_lc_droplet_bodyforce_wall(fe_lc_droplet_t * fe, lees_edw_t * le,
			    hydro_t * hydro, map_t * map, wall_t * wall) {

  int ic, jc, kc;
  int ic1, jc1, kc1, isite[NGRAD_];
  int ia, ib, p;
  int index0;
  int nhalo;
  int nlocal[3];
  int status;

  double mu0, mup;
  double force[3] = {0.0, 0.0, 0.0};
  double h[3][3];
  double q[3][3];
  double dq[3][3][3];
  double phi;
  double count[NGRAD_];
  double gradt[NGRAD_];
  double gradn[3];

  cs_t * cs = NULL;

  assert(fe);
  assert(le);
  assert(hydro);
  assert(map);
  assert(cs);

  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal(le, nlocal);
  assert(nhalo >= 2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = cs_index(cs, ic, jc, kc);
	map_status(map, index0, &status);

	if (status == MAP_FLUID) {

	  field_scalar(fe->symm->phi, index0, &phi);
	  field_tensor(fe->lc->q, index0, q);

	  field_grad_tensor_grad(fe->lc->dq, index0, dq);
	  fe_lc_droplet_mol_field(fe, index0, h);

	  /* Set solid/fluid flag to determine dry links */
	  for (p = 1; p < NGRAD_; p++) {

	    ic1 = ic + bs_cv[p][X];
	    jc1 = jc + bs_cv[p][Y];
	    kc1 = kc + bs_cv[p][Z];

	    isite[p] = cs_index(cs, ic1, jc1, kc1);
	    map_status(map, isite[p], &status);
	    if (status != MAP_FLUID) isite[p] = -1;

	  }

	  for (ia = 0; ia < 3; ia++) {
	    count[ia] = 0.0;
	    gradn[ia] = 0.0;
	  }

	  /* Estimate normal gradient using FD on wet links */
	  for (p = 1; p < NGRAD_; p++) {

	    if (isite[p] == -1) continue;

	    fe_lc_droplet_mu(fe, isite[p], &mup);
	    fe_lc_droplet_mu(fe, index0, &mu0);

	    gradt[p] = mup - mu0;

	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += bs_cv[p][ia]*gradt[p];
	      count[ia] += bs_cv[p][ia]*bs_cv[p][ia];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    if (count[ia] > 0.0) gradn[ia] /= count[ia];
	  }

	  /* Estimate gradient at boundaries */
	  for (p = 1; p < NGRAD_; p++) {

	    if (isite[p] == -1) {

	      gradt[p] = (bs_cv[p][X]*gradn[X] +
			  bs_cv[p][Y]*gradn[Y] +
			  bs_cv[p][Z]*gradn[Z]);

	    }
	  }

	  /* Accumulate the final gradients */
	  for (ia = 0; ia < 3; ia++) {
	    count[ia] = 0.0;
	    gradn[ia] = 0.0;
	  }

	  for (p = 1; p < NGRAD_; p++) {
	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += bs_cv[p][ia]*gradt[p];
	      count[ia] += bs_cv[p][ia]*bs_cv[p][ia];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    if (count[ia] > 0.0) gradn[ia] /= count[ia];
	  }

	  force[X] = -phi*gradn[X];
	  force[Y] = -phi*gradn[Y];
	  force[Z] = -phi*gradn[Z];

	  for (ia = 0; ia < 3; ia++ ) {
	    for(ib = 0; ib < 3; ib++ ) {
	      force[X] -= h[ia][ib]*dq[X][ia][ib];
	      force[Y] -= h[ia][ib]*dq[Y][ia][ib];
	      force[Z] -= h[ia][ib]*dq[Z][ia][ib];
	    }
	  }

	  /* Store the force on lattice */
	  hydro_f_local_add(hydro, index0, force);

	}

      }
    }
  }

  return 0;
}
