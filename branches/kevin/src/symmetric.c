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

#include <assert.h>
#include <math.h>

#include "util.h"
#include "symmetric.h"
#include "field_s.h"
#include "field_grad_s.h"

static TARGET_CONST double a_     = -0.003125;
static TARGET_CONST double b_     = +0.003125;
static TARGET_CONST double kappa_ = +0.002;

static field_t * phi_ = NULL;
static field_grad_t * grad_phi_ = NULL;


// flag to track whether this module has been initiated
static char symmetric_flag=0;
HOST char symmetric_in_use(){ return symmetric_flag; }


/****************************************************************************
 *
 *  symmetric_phi_set
 *
 *  Attach a reference to the order parameter field object, and the
 *  associated gradient object.
 *
 ****************************************************************************/

HOST int symmetric_phi_set(field_t * phi, field_grad_t * dphi) {

  assert(phi);
  assert(dphi);

  phi_ = phi;
  grad_phi_ = dphi;

  return 0;
}




/****************************************************************************
 *
 *  symmetric_free_energy_parameters_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

HOST void symmetric_free_energy_parameters_set(double a, double b, double kappa) {

  //  a_ = a;
  //b_ = b;
  //kappa_ = kappa;

  copyConstantDoubleToTarget(&a_, &a, sizeof(double));
  copyConstantDoubleToTarget(&b_, &b, sizeof(double));
  copyConstantDoubleToTarget(&kappa_, &kappa, sizeof(double));


  fe_kappa_set(kappa);

  symmetric_flag=1;

  return;
}

/****************************************************************************
 *
 *  symmetric_a
 *
 ****************************************************************************/

HOST double symmetric_a(void) {

  double atmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));

  return atmp;

}

/****************************************************************************
 *
 *  symmetric_b
 *
 ****************************************************************************/

HOST double symmetric_b(void) {

  double btmp;
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));

  return btmp;
}


/****************************************************************************
 *
 *  symmetric_phi
 *
 ****************************************************************************/

HOST void symmetric_phi(double** address_of_ptr) {

  *address_of_ptr = phi_->data;

  return;
  
}

/****************************************************************************
 *
 *  symmetric_t_phi
 *
 ****************************************************************************/

HOST void symmetric_t_phi(double** address_of_ptr) {

  *address_of_ptr = phi_->t_data;

  return;
  
}

/****************************************************************************
 *
 *  symmetric_gradphi
 *
 ****************************************************************************/

HOST void symmetric_gradphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->grad;
  
  return;
}


/****************************************************************************
 *
 *  symmetric_t_gradphi
 *
 ****************************************************************************/

HOST void symmetric_t_gradphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->t_grad;
  
  return;
}

/****************************************************************************
 *
 *  symmetric_delsqphi
 *
 ****************************************************************************/

HOST void symmetric_delsqphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->delsq;

  return;

}


/****************************************************************************
 *
 *  symmetric_t_delsqphi
 *
 ****************************************************************************/

HOST void symmetric_t_delsqphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->t_delsq;

  return;

}


/****************************************************************************
 *
 *  symmetric_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

HOST double symmetric_interfacial_tension(void) {

  double sigma;

  double atmp, btmp,kappatmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));
  copyConstantDoubleFromTarget(&kappatmp, &kappa_, sizeof(double));

  sigma = sqrt(-8.0*kappatmp*atmp*atmp*atmp/(9.0*btmp*btmp));

  return sigma;
}

/****************************************************************************
 *
 *  symmetric_interfacial_width
 *
 ****************************************************************************/

HOST double symmetric_interfacial_width(void) {

  double xi;

  double atmp, btmp,kappatmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));
  copyConstantDoubleFromTarget(&kappatmp, &kappa_, sizeof(double));

  xi = sqrt(-2.0*kappatmp/atmp);

  return xi;
}

/****************************************************************************
 *
 *  symmetric_free_energy_density
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

HOST double symmetric_free_energy_density(const int index) {

  double phi;
  double dphi[3];
  double e;

  double atmp, btmp,kappatmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));
  copyConstantDoubleFromTarget(&kappatmp, &kappa_, sizeof(double));


  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, dphi);

  e = 0.5*atmp*phi*phi + 0.25*btmp*phi*phi*phi*phi
	+ 0.5*kappatmp*dot_product(dphi, dphi);

  return e;
}

/****************************************************************************
 *
 *  symmetric_chemical_potential
 *
 *  The chemical potential \mu = \delta F / \delta \phi
 *                             = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *
 ****************************************************************************/

HOST double symmetric_chemical_potential(const int index, const int nop) {

  double phi;
  double delsq_phi;
  double mu;

  double atmp, btmp,kappatmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));
  copyConstantDoubleFromTarget(&kappatmp, &kappa_, sizeof(double));

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  mu = atmp*phi + btmp*phi*phi*phi - kappatmp*delsq_phi;

  return mu;
}


//TODO currently we have duplicate HOST and TARGET versions of potential and stress routines 

//TODO vectorise


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
 *  symmetric_isotropic_pressure
 *
 *  This ignores the term in the density (assumed to be uniform).
 *
 ****************************************************************************/

HOST double symmetric_isotropic_pressure(const int index) {

  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  double atmp, btmp,kappatmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));
  copyConstantDoubleFromTarget(&kappatmp, &kappa_, sizeof(double));

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  p0 = 0.5*atmp*phi*phi + 0.75*btmp*phi*phi*phi*phi
    - kappatmp*phi*delsq_phi - 0.5*kappatmp*dot_product(grad_phi, grad_phi);

  return p0;
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

HOST void symmetric_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  double atmp, btmp,kappatmp;
  copyConstantDoubleFromTarget(&atmp, &a_, sizeof(double));
  copyConstantDoubleFromTarget(&btmp, &b_, sizeof(double));
  copyConstantDoubleFromTarget(&kappatmp, &kappa_, sizeof(double));

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  p0 = 0.5*atmp*phi*phi + 0.75*btmp*phi*phi*phi*phi
    - kappatmp*phi*delsq_phi - 0.5*kappatmp*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+ kappatmp*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}

TARGET void symmetric_chemical_stress_target(const int index, double s[3][3*NILP], const double* t_phi,  const double* t_gradphi, const double* t_delsqphi) {
  
  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  //initialisation for targetDP instruction level parallelism

  int vecIndex=0;

  TARGET_ILP(vecIndex) {
    
    //phi = phi_get_phi_site(index+vecIndex);
    //phi_gradients_grad(index+vecIndex, grad_phi);
    //  delsq_phi = phi_gradients_delsq(index+vecIndex);
    
    for (ia=0;ia<3;ia++){
      
      grad_phi[ia]=t_gradphi[3*(index+vecIndex)+ia];
      
    }
    
    phi=t_phi[index+vecIndex];
    delsq_phi=t_delsqphi[index+vecIndex];
    
    
    //    p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    //- kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);
    
    p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
      - kappa_*phi*delsq_phi 
      - 0.5*kappa_
      *(grad_phi[0]*grad_phi[0]+grad_phi[1]*grad_phi[1]
	+grad_phi[2]*grad_phi[2]);
    
    
    
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ILPIDX(ib)] = p0*tc_d_[ia][ib]	+ kappa_*grad_phi[ia]*grad_phi[ib];
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
