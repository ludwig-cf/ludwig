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
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 ****************************************************************************/

#define INCLUDED_FROM_TARGET


#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "targetDP.h"

#include "phi.h"
#include "phi_gradients.h"
#include "util.h"
#include "symmetric.h"



#define HOST
#ifdef CUDA
#define HOST extern "C"
#endif


//static double a_     = -0.003125;
static TARGET_CONST double a_     = -0.003125;
static TARGET_CONST double b_     = +0.003125;
static TARGET_CONST double kappa_ = +0.002;

/****************************************************************************
 *
 *  symmetric_free_energy_parameters_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

HOST void symmetric_free_energy_parameters_set_target(double a, double b, double kappa) {

  //a_ = a;
  //b_ = b;
  //kappa_ = kappa;

  copyConstantDoubleToTarget(&a_, &a, sizeof(double));
  copyConstantDoubleToTarget(&b_, &b, sizeof(double));
  copyConstantDoubleToTarget(&kappa_, &kappa, sizeof(double));

  //TO DO TARGET
  //fe_kappa_set(kappa);


  return;
}

// /****************************************************************************
//  *
//  *  symmetric_a
//  *
//  ****************************************************************************/

// double symmetric_a(void) {

//   return a_;
// }

// /****************************************************************************
//  *
//  *  symmetric_b
//  *
//  ****************************************************************************/

// double symmetric_b(void) {

//   return b_;
// }

// /****************************************************************************
//  *
//  *  symmetric_interfacial_tension
//  *
//  *  Assumes phi^* = (-a/b)^1/2
//  *
//  ****************************************************************************/

// double symmetric_interfacial_tension(void) {

//   double sigma;

//   sigma = sqrt(-8.0*kappa_*a_*a_*a_/(9.0*b_*b_));

//   return sigma;
// }

// /****************************************************************************
//  *
//  *  symmetric_interfacial_width
//  *
//  ****************************************************************************/

// double symmetric_interfacial_width(void) {

//   double xi;

//   xi = sqrt(-2.0*kappa_/a_);

//   return xi;
// }

// /****************************************************************************
//  *
//  *  symmetric_free_energy_density
//  *
//  *  The free energy density is as above.
//  *
//  ****************************************************************************/

// double symmetric_free_energy_density(const int index) {

//   double phi;
//   double dphi[3];
//   double e;

//   phi = phi_get_phi_site(index);
//   phi_gradients_grad(index, dphi);

//   e = 0.5*a_*phi*phi + 0.25*b_*phi*phi*phi*phi
// 	+ 0.5*kappa_*dot_product(dphi, dphi);

//   return e;
// }

// /****************************************************************************
//  *
//  *  symmetric_isotropic_pressure
//  *
//  *  This ignores the term in the density (assumed to be uniform).
//  *
//  ****************************************************************************/

// double symmetric_isotropic_pressure(const int index) {

//   double phi;
//   double delsq_phi;
//   double grad_phi[3];
//   double p0;

//   phi = phi_get_phi_site(index);
//   phi_gradients_grad(index, grad_phi);
//   delsq_phi = phi_gradients_delsq(index);

//   p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
//     - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

//   return p0;
// }


// type for chemical potential function
// TODO This is generic fine here
//typedef double (*cp_fntype)(const int index, const int nop, double* t_phi, double* t_delsqphi);
//typedef double (*mu_fntype)(const int, const int, const double*, const double*);


/****************************************************************************
 *
 *  symmetric_chemical_potential
 *
 *  The chemical potential \mu = \delta F / \delta \phi
 *                             = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *
 ****************************************************************************/

TARGET double symmetric_chemical_potential_target(const int index, const int nop, const double* t_phi, const double* t_delsqphi) {

  double phi;
  double delsq_phi;
  double mu;

  //TODO
  //assert(nop == 0);

  //phi = phi_get_phi_site(index);
  //delsq_phi = phi_gradients_delsq(index);

  phi=t_phi[index];
  delsq_phi=t_delsqphi[index];

  mu = a_*phi + b_*phi*phi*phi - kappa_*delsq_phi;

  return mu;
}

// pointer to above target function. 
TARGET mu_fntype p_symmetric_chemical_potential_target = symmetric_chemical_potential_target;



HOST void get_chemical_potential_target(mu_fntype* t_chemical_potential){

  mu_fntype h_chemical_potential; //temp host copy of fn addess

  //get host copy of function pointer
  copyConstantMufnFromTarget(&h_chemical_potential, &p_symmetric_chemical_potential_target,sizeof(mu_fntype) );

  //and put back on target, now in an accessible location
  copyToTarget( t_chemical_potential, &h_chemical_potential,sizeof(mu_fntype));

  return;


}


/****************************************************************************
 *
 *  symmetric_chemical_stress
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

void symmetric_chemical_stress_target(const int index, double s[3][3*NILP]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  int vecIndex=0;
  //HACK
  TARGET_ILP(vecIndex){

  phi = phi_get_phi_site(index+vecIndex);
  phi_gradients_grad(index+vecIndex, grad_phi);
  delsq_phi = phi_gradients_delsq(index+vecIndex);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib+vecIndex] = p0*d_[ia][ib]	+ kappa_*grad_phi[ia]*grad_phi[ib];
    }
  }

  }

  return;
}

// pointer to above target function. 
TARGET pth_fntype p_symmetric_chemical_stress_target = symmetric_chemical_stress_target;


HOST void get_chemical_stress_target(pth_fntype* t_chemical_stress){

  pth_fntype h_chemical_stress; //temp host copy of fn addess

  //get host copy of function pointer
  copyConstantPthfnFromTarget(&h_chemical_stress, &p_symmetric_chemical_stress_target,sizeof(pth_fntype) );

  //and put back on target, now in an accessible location
  copyToTarget( t_chemical_stress, &h_chemical_stress,sizeof(pth_fntype));

  return;


}
