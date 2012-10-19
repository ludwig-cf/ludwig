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
static double delta_;
static double W_;

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
}

/*****************************************************************************
 *
 *  lc_droplet_chemical_potential
 *
 *  Return the chemical potential at lattice site index.
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
  double q2, q3;
  double mu;
  double a0;
  int ia, ib, ic;
  
  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_scalar_grad(grad_phi_, index, dphi);
  
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
  
  mu = 0.0;
  for (ia = 0; ia < 3; ia++){
    for (ib = 0; ib < 3; ib++){
      mu -= W_*(dphi[ia]*dq[ib][ia][ib] + dphi[ib]*dq[ia][ia][ib]);
    }
  }
  
  a0 = blue_phase_a0();
  mu -= a0*delta_*r3_*0.5*q2;
  mu -= a0*delta_*r3_*q3;
  mu -= a0*delta_*0.25*q2*q2;
  
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
  
  double s1[3][3];
  double s2[3][3];
  double gamma;
  double phi;
  int ia, ib;
  
  assert(phi_);
  
  field_scalar(phi_, index, &phi);
  gamma = lc_droplet_gamma_calculate(phi);
  blue_phase_set_gamma(gamma);
  
  /*the stresses needs to be sorted*/
  blue_phase_chemical_stress(index, s1);
  symmetric_chemical_stress(index, s2);
  
  for (ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      sth[ia][ib] = s1[ia][ib] + s2[ia][ib];
    }
  }
}
  
