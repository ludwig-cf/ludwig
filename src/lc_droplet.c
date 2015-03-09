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
 *  (c) 2012 The University of Edinburgh
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
#include "leesedwards.h"
#include "lc_droplet.h"


static double gamma0_; /* \gamma(\phi) = gamma0_ + delta(1 + \phi)*/
static double delta_;  /* For above */
static double W_;      /* Coupling or anchoring constant */

static field_t * q_ = NULL;
static field_grad_t * grad_q_ = NULL;
static field_t * phi_ = NULL;
static field_grad_t * grad_phi_ = NULL;

/*****************************************************************************
 *
 *  lc_droplet_phi_set
 *
 *  Attach a reference to the order parameter field object, and the
 *  associated gradient object.
 *
 *****************************************************************************/

int lc_droplet_phi_set(field_t * phi, field_grad_t * dphi) {

  assert(phi);
  assert(dphi);
  
  phi_ = phi;
  grad_phi_ = dphi;
  symmetric_phi_set(phi, dphi);
  
  return 0;
}

/*****************************************************************************
 *
 *  lc_droplet_q_set
 *
 *  Attach a reference to the order parameter tensor object, and the
 *  associated gradient object.
 *
 *****************************************************************************/

int lc_droplet_q_set(field_t * q, field_grad_t * dq) {

  assert(q);
  assert(dq);
  
  q_ = q;
  grad_q_ = dq;
  blue_phase_q_set(q, dq);
  
  return 0;
}
    

/*****************************************************************************
 *
 *  lc_droplet_set_parameters
 *
 *  Set the parameters.
 *
 *****************************************************************************/

void lc_droplet_set_parameters(double gamma0, double delta, double W) {

  gamma0_ = gamma0;
  delta_ = delta;
  W_ = W;

  return;
}

/*****************************************************************************
 *
 *  lc_droplet_get_gamma0
 *
 *****************************************************************************/

double lc_droplet_get_gamma0(void) {

  return gamma0_;
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

double lc_droplet_free_energy_density(const int index) {
  
  double f;
  double phi;
  double q[3][3];
  double dphi[3];
  double gamma;
  
  assert(phi_);
  assert(q_);
  assert(grad_phi_);
    
  field_scalar(phi_, index, &phi);
  field_tensor(q_, index, q);

  field_grad_scalar_grad(grad_phi_, index, dphi);
  
  gamma = lc_droplet_gamma_calculate(phi);
  blue_phase_set_gamma(gamma);
  
  f = symmetric_free_energy_density(index);
  f += blue_phase_free_energy_density(index);
  f += lc_droplet_anchoring_energy_density(q, dphi);

  return f;
}

/*****************************************************************************
 *
 *  lc_droplet_gamma
 *
 *  calculate: gamma = gamma0 + delta * (1 + phi)
 *
 *****************************************************************************/

double lc_droplet_gamma_calculate(const double phi) {
  
  double gamma;
  
  gamma = gamma0_ + delta_ * (1.0 + phi);
  
  return gamma;
}

/*****************************************************************************
 *
 *  lc_droplet_anchoring_energy_density
 *
 *  Return the free energy density of anchoring at the interface
 *
 *  f = W Q dphi dphi
 *
 *****************************************************************************/

double lc_droplet_anchoring_energy_density(double q[3][3], double dphi[3]) {
  
  double f;
  int ia, ib;
  
  f = 0.0;
  
  for (ia = 0; ia < 3; ia++){
    for (ib = 0; ib < 3; ib++){
      f += q[ia][ib] * dphi[ia] * dphi[ib];
    }
  }
  f = W_ * f;
  
  return f;
}
  
/*****************************************************************************
 *
 *  lc_droplet_molecular_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *****************************************************************************/

void lc_droplet_molecular_field(const int index, double h[3][3]) {
  
  double h1[3][3], h2[3][3];
  int ia,ib;
  double phi;
  double gamma;
  
  assert(phi_);
  
  field_scalar(phi_, index, &phi);
  gamma = lc_droplet_gamma_calculate(phi);

  blue_phase_set_gamma(gamma);
  
  blue_phase_molecular_field(index, h1);
  
  lc_droplet_anchoring_molecular_field(index, h2);
    
  for (ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      h[ia][ib] = h1[ia][ib] + h2[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lc_droplet_anchoring_molecular_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *****************************************************************************/

void lc_droplet_anchoring_molecular_field(const int index, double h[3][3]) {
  
  double dphi[3];
  double delsq_phi;
  int ia, ib;
  double dphi2;
  
  field_grad_scalar_grad(grad_phi_, index, dphi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);
  
  dphi2 = dphi[X]*dphi[X] + dphi[Y]*dphi[Y] + dphi[Z]*dphi[Z];

  for (ia = 0; ia < 3; ia++){
    for (ib = 0; ib < 3; ib++){
      h[ia][ib] = -W_*(dphi[ia]*dphi[ib] - r3_*d_[ia][ib]*dphi2);
    }
  }

  return;
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

double lc_droplet_chemical_potential(const int index, const int nop) {
  
  double mu;
  double q[3][3];
  double dphi[3];
  double dq[3][3][3];
  double dabphi[3][3];
  double q2, q3;
  double wmu;
  double a0;
  int ia, ib, ic;
  
  mu = 0;
  
  mu = symmetric_chemical_potential(index, nop);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_scalar_grad(grad_phi_, index, dphi);
  field_grad_scalar_dab(grad_phi_, index, dabphi);
  
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
  
  a0 = blue_phase_a0();
  mu += -0.5*r3_*a0*delta_*q2 - r3_*a0*delta_*q3 + 0.25*a0*delta_*q2*q2
    - 2.0*W_*wmu;

  return mu;
}

/*****************************************************************************
 *
 *  lc_droplet_bodyforce
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
 *  Lees-Edwards planes are allowed for.
 *
 *****************************************************************************/
  
void lc_droplet_bodyforce(hydro_t * hydro) {

  double h[3][3];
  double q[3][3];
  double dq[3][3][3];
  double phi;
    
  int ic, jc, kc, icm1, icp1;
  int ia, ib;
  int index0, indexm1, indexp1;
  int nhalo;
  int nlocal[3];
  int zs, ys;
  double mum1, mup1;
  double force[3];
  
  assert(phi_);
  
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  assert(nhalo >= 2);

  /* Memory strides */

  zs = 1;
  ys = (nlocal[Z] + 2*nhalo)*zs;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);
	
	field_scalar(phi_, index0, &phi);
	field_tensor(q_, index0, q);
  
	field_grad_tensor_grad(grad_q_, index0, dq);
	lc_droplet_molecular_field(index0, h);
  
        indexm1 = le_site_index(icm1, jc, kc);
        indexp1 = le_site_index(icp1, jc, kc);

        mum1 = lc_droplet_chemical_potential(indexm1, 0);
        mup1 = lc_droplet_chemical_potential(indexp1, 0);
	
	/* X */

	force[X] = - phi*0.5*(mup1 - mum1);
	
	for (ia = 0; ia < 3; ia++ ) {
	  for(ib = 0; ib < 3; ib++ ) {
	    force[X] -= h[ia][ib]*dq[X][ia][ib];
	  }
	}
	
	/* Y */

	mum1 = lc_droplet_chemical_potential(index0 - ys, 0);
        mup1 = lc_droplet_chemical_potential(index0 + ys, 0);

        force[Y] = -phi*0.5*(mup1 - mum1);
	
	for (ia = 0; ia < 3; ia++ ) {
	  for(ib = 0; ib < 3; ib++ ) {
	    force[Y] -= h[ia][ib]*dq[Y][ia][ib];
	  }
	}

	/* Z */

	mum1 = lc_droplet_chemical_potential(index0 - zs, 0);
        mup1 = lc_droplet_chemical_potential(index0 + zs, 0);

        force[Z] = -phi*0.5*(mup1 - mum1);
	
	for (ia = 0; ia < 3; ia++ ) {
	  for(ib = 0; ib < 3; ib++ ) {
	    force[Z] -= h[ia][ib]*dq[Z][ia][ib];
	  }
	}

	/* Store the force on lattice */

	hydro_f_local_add(hydro, index0, force);
	
      }
    }
  }
  return;
}

/*****************************************************************************
 *
 *  lc_droplet_chemical_stress
 *
 *  Return the chemical stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/

void lc_droplet_chemical_stress(const int index, double sth[3][3]) {

  double s1[3][3];
  double s2[3][3];
  int ia, ib;

  lc_droplet_symmetric_stress(index, s1);
  lc_droplet_antisymmetric_stress(index, s2);

  for (ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      sth[ia][ib] = s1[ia][ib] + s2[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lc_droplet_symmetric_stress
 *
 *****************************************************************************/

void lc_droplet_symmetric_stress(const int index, double sth[3][3]){

  double q[3][3], dq[3][3][3];
  double h[3][3], dsq[3][3];
  double phi;
  int ia, ib, ic;
  double qh;
  double gamma;
  double xi_, zeta_;
  
  xi_ = blue_phase_get_xi();
  zeta_ = blue_phase_get_zeta();
  
  /* No redshift at the moment */
  
  field_scalar(phi_, index, &phi);
  field_tensor(q_, index, q);

  gamma = lc_droplet_gamma_calculate(phi);
  blue_phase_set_gamma(gamma);
  
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_tensor_delsq(grad_q_, index, dsq);

  blue_phase_compute_h(q, dq, dsq, h);
  
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
      sth[ia][ib] = 2.0*xi_*(q[ia][ib] + r3_*d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	sth[ia][ib] +=
	  -xi_*h[ia][ic]*(q[ib][ic] + r3_*d_[ib][ic])
	  -xi_*(q[ia][ic] + r3_*d_[ia][ic])*h[ib][ic];
      }
    }
  }

  /* Additional active stress -zeta*(q_ab - 1/3 d_ab) */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] -= zeta_*(q[ia][ib] + r3_*d_[ia][ib]);
    }
  }

  /* This is the minus sign. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lc_droplet_antisymmetric_stress
 *
 *****************************************************************************/

void lc_droplet_antisymmetric_stress(const int index, double sth[3][3]) {

  double q[3][3], dq[3][3][3];
  double h[3][3], dsq[3][3];
  double phi;
  int ia, ib, ic;
  double gamma;
  
  /* No redshift at the moment */
  
  field_scalar(phi_, index, &phi);
  field_tensor(q_, index, q);

  gamma = lc_droplet_gamma_calculate(phi);
  blue_phase_set_gamma(gamma);
  
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_tensor_delsq(grad_q_, index, dsq);

  blue_phase_compute_h(q, dq, dsq, h);

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
  
  /*  This is the minus sign. */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return;
}

