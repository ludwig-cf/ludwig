/*****************************************************************************
 *
 *  lc_droplet.c
 *
 *  Routines related to liquid crystal droplet free energy
 *  and molecular field.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
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
#include "hydro_s.h" 
#include "kernel.h"
#include "lc_droplet.h"

static fe_vt_t fe_drop_hvt = {
  (fe_free_ft)      fe_lc_droplet_free,
  (fe_target_ft)    fe_lc_droplet_target,
  (fe_fed_ft)       fe_lc_droplet_fed,
  (fe_mu_ft)        fe_lc_droplet_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_droplet_stress,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_droplet_mol_field,
  (fe_htensor_v_ft) fe_lc_droplet_mol_field_v,
  (fe_stress_v_ft)  fe_lc_droplet_stress_v
};

static __constant__ fe_vt_t fe_drop_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_lc_droplet_fed,
  (fe_mu_ft)        fe_lc_droplet_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_lc_droplet_stress,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   fe_lc_droplet_mol_field,
  (fe_htensor_v_ft) fe_lc_droplet_mol_field_v,
  (fe_stress_v_ft)  fe_lc_droplet_stress_v
};

__host__ __device__
int fe_lc_droplet_anchoring_h(fe_lc_droplet_t * fe, int index, double h[3][3]);

__host__ __device__
int fe_lc_droplet_symmetric_stress(fe_lc_droplet_t * fe, int index,
				   double sth[3][3]);
__host__ __device__
int fe_lc_droplet_antisymmetric_stress(fe_lc_droplet_t * fe, int index,
				       double sth[3][3]);

__global__ void fe_lc_droplet_bf_kernel(kernel_ctxt_t * ktx,
					fe_lc_droplet_t * fe,
					hydro_t * hydro);


static __constant__ fe_lc_droplet_param_t const_param;

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
  if (fe == NULL) pe_fatal(pe, "calloc(fe_lc_droplet_t) failed\n");

  fe->param = (fe_lc_droplet_param_t *) calloc(1, sizeof(fe_lc_droplet_param_t));
  if (fe->param == NULL) pe_fatal(pe, "calloc(fe_lc_droplet_param_t) failed\n");

  fe->pe = pe;
  fe->cs = cs;
  fe->lc = lc;
  fe->symm = symm;

  fe->super.func = &fe_drop_hvt;
  fe->super.id = FE_LC_DROPLET;

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_lc_droplet_param_t * tmp;
    fe_vt_t * vt;
    tdpMalloc((void **) &fe->target, sizeof(fe_lc_droplet_t));
    tdpMemset(fe->target, 0, sizeof(fe_lc_droplet_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&fe->target->param, &tmp, sizeof(fe_lc_droplet_param_t *),
	      tdpMemcpyHostToDevice);
    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_drop_dvt));
    tdpMemcpy(&fe->target->super.func, &vt, sizeof(fe_vt_t *),
	      tdpMemcpyHostToDevice);

    tdpMemcpy(&fe->target->lc, &lc->target, sizeof(fe_lc_t *),
	      tdpMemcpyHostToDevice);
    tdpMemcpy(&fe->target->symm, &symm->target, sizeof(fe_symm_t *),
	      tdpMemcpyHostToDevice);
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

  if (fe->target != fe) tdpFree(fe->target);

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

int fe_lc_droplet_fed(fe_lc_droplet_t * fe, int index, double * fed) {

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

__host__ __device__ int fe_lc_droplet_stress(fe_lc_droplet_t * fe,
					     int index, double sth[3][3]) {

  int ia, ib;
  double s1[3][3];
  double s2[3][3];

  assert(fe);

  fe_lc_droplet_symmetric_stress(fe, index, s1);
  fe_lc_droplet_antisymmetric_stress(fe, index, s2);

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
    fe_lc_droplet_symmetric_stress(fe, index+iv, s1);
    fe_lc_droplet_antisymmetric_stress(fe, index+iv, s2);
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
 *  fe_lc_droplet_symmetric_stress
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_droplet_symmetric_stress(fe_lc_droplet_t * fe, int index,
				   double sth[3][3]){

  double q[3][3];
  double h[3][3];
  int ia, ib, ic;
  double qh;
  double xi, zeta;
  const double r3 = (1.0/3.0);
  KRONECKER_DELTA_CHAR(d);

  xi = fe->lc->param->xi;
  zeta = fe->lc->param->zeta;
  
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

  /* Additional active stress -zeta*(q_ab - 1/3 d_ab)Â */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] -= zeta*(q[ia][ib] + r3*d[ia][ib]);
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
 *  fe_lc_droplet_antisymmetric_stress
 *
 *****************************************************************************/

__host__ __device__
int fe_lc_droplet_antisymmetric_stress(fe_lc_droplet_t * fe, int index,
				       double sth[3][3]) {

  int ia, ib, ic;
  double q[3][3];
  double h[3][3];

  assert(fe);

  /* No redshift at the moment */
  
  field_tensor(fe->lc->q, index, q);
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
 *  fe_lc_droplet_bodyforce
 *
 *  The driver for the bodyforce calculation.
 *
 *****************************************************************************/

__host__ int fe_lc_droplet_bodyforce(fe_lc_droplet_t * fe, hydro_t * hydro) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(fe);
  assert(hydro);

  cs_nlocal(fe->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(fe->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(fe_lc_droplet_bf_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, fe->target, hydro->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

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
 *  free energies. Additonal force terms are included in the stress tensor.
 *
 *  The gradient of the chemical potential is computed as
 *
 *    grad_x mu = 0.5*(mu(i+1) - mu(i-1)) etc
 *
 ****************************************************************************/

__global__ void fe_lc_droplet_bf_kernel(kernel_ctxt_t * ktx,
					fe_lc_droplet_t * fe,
					hydro_t * hydro) {
  int kindex;
  int kiterations;

  assert(ktx);
  assert(fe);
  assert(hydro);

  kiterations = kernel_iterations(ktx);

  __target_simt_for(kindex, kiterations, 1) {

    int ic, jc, kc;
    int ia, ib;
    int index0, indexm1, indexp1;
    double mum1, mup1;
    double force[3];

    double h[3][3];
    double q[3][3];
    double dq[3][3][3];
    double phi;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index0 = kernel_coords_index(ktx, ic, jc, kc);

    field_scalar(fe->symm->phi, index0, &phi);
    field_tensor(fe->lc->q, index0, q);

    field_grad_tensor_grad(fe->lc->dq, index0, dq);
    fe_lc_droplet_mol_field(fe, index0, h);

    indexm1 = kernel_coords_index(ktx, ic-1, jc, kc);
    indexp1 = kernel_coords_index(ktx, ic+1, jc, kc);

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

    indexm1 = kernel_coords_index(ktx, ic, jc-1, kc);
    indexp1 = kernel_coords_index(ktx, ic, jc+1, kc);

    fe_lc_droplet_mu(fe, indexm1, &mum1);
    fe_lc_droplet_mu(fe, indexp1, &mup1);

    force[Y] = -phi*0.5*(mup1 - mum1);
 
    for (ia = 0; ia < 3; ia++ ) {
      for(ib = 0; ib < 3; ib++ ) {
	force[Y] -= h[ia][ib]*dq[Y][ia][ib];
      }
    }

    /* Z */

    indexm1 = kernel_coords_index(ktx, ic, jc, kc-1);
    indexp1 = kernel_coords_index(ktx, ic, jc, kc+1);

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
