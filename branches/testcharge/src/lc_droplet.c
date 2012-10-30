/*****************************************************************************
 *
 *  lc_droplet.c
 *
 *  Routines related to liquid crystal droplet free energy
 *  and molecular field.
 *
 *  $Id:               $
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
#include "field.h"
#include "field_grad.h"
#include "blue_phase.h"
#include "symmetric.h"
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
 *  lc_droplet_set_gamma0
 *
 *  Set the gamma0.
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
  
  field_grad_scalar_grad(grad_phi_, index, dphi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);
  
  for (ia = 0; ia < 3; ib++){
    for (ib = 0; ib < 3; ib++){
      h[ia][ib] = -W_*(dphi[ia]*dphi[ib] - r3_*d_[ia][ib]*delsq_phi);
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
  
  mu = 0;
  
  mu = symmetric_chemical_potential(index, nop);
  mu += lc_droplet_chemical_potential_lc(index);

  return mu;
}
  
/*****************************************************************************
 *
 *  lc_droplet_chemical_potential_lc
 *
 *  Return the chemical potential at lattice site index.
 *
 *****************************************************************************/

double lc_droplet_chemical_potential_lc(const int index) {
  
  double q[3][3];
  double dphi[3];
  double dq[3][3][3];
  double dabphi[3][3];
  double q2, q3;
  double mu;
  double wmu;
  double a0;
  int ia, ib, ic;
  
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
  mu = -0.5*r3_*a0*delta_*q2 - r3_*a0*delta_*q3 + 0.25*a0*delta_*q2*q2
    - 2.0*W_*wmu;
  
  return mu;
}

/*****************************************************************************
 *
 *  lc_droplet_chemical_stress
 *
 *  Return the chemical stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/

void lc_droplet_chemical_stress(const int index, double sth[3][3]) {

  double q[3][3];
  double dphi[3];
  double s1[3][3];
  double s2[3][3];
  double gamma;
  double phi;
  int ia, ib;
  
  assert(phi_);
  
  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, dphi);
  field_tensor(q_, index, q);

  gamma = lc_droplet_gamma_calculate(phi);
  blue_phase_set_gamma(gamma);

  /* This will be returned with additional -ve sign */
  blue_phase_chemical_stress(index, s1);

  /* This will have a +ve sign */
  symmetric_chemical_stress(index, s2);

  /* The whole thing is ... */
  /* This has an additional -ve sign so the divergence of the stress
   * can be computed with the correct sign (see blue phase). */

  for (ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      sth[ia][ib] = -1.0*(-s1[ia][ib] + s2[ia][ib]
			  - 2.0*W_*dphi[ia]*dphi[ib]*q[ia][ib]);
    }
  }

  return;
}
  
