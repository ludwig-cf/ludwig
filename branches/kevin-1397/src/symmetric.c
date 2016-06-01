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
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "symmetric.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "coords.h"
#include "leesedwards.h"

static double a_     = -0.003125;
static double b_     = +0.003125;
static double kappa_ = +0.002;

static __targetConst__ double t_a_     = -0.003125;
static __targetConst__ double t_b_     = +0.003125;
static __targetConst__ double t_kappa_ = +0.002;

static field_t * phi_ = NULL;
static field_grad_t * grad_phi_ = NULL;


/* flag to track whether this module has been initiated*/
static char symmetric_flag=0;

__targetHost__ char symmetric_in_use(){ return symmetric_flag; }

/* Global objects */


__target__
double symmetric_chemical_potential_target(const int index, const int nop,
					   const double * t_phi,
					   const double * t_delsqphi);
__target__ void symmetric_chemical_stress_target(const int index,
						 double s[3][3*NSIMDVL],
						 const double* t_phi, 
						 const double* t_gradphi,
						 const double* t_delsqphi);

__target__ mu_fntype p_symmetric_chemical_potential_target = symmetric_chemical_potential_target;

__target__ pth_fntype p_symmetric_chemical_stress_target = symmetric_chemical_stress_target;


/****************************************************************************
 *
 *  symmetric_phi_set
 *
 *  Attach a reference to the order parameter field object, and the
 *  associated gradient object.
 *
 ****************************************************************************/

__targetHost__ int symmetric_phi_set(field_t * phi, field_grad_t * dphi) {

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

__targetHost__
void symmetric_free_energy_parameters_set(double a, double b, double kappa) {

  a_ = a;
  b_ = b;
  kappa_ = kappa;

  copyConstToTarget(&t_a_, &a, sizeof(double));
  copyConstToTarget(&t_b_, &b, sizeof(double));
  copyConstToTarget(&t_kappa_, &kappa, sizeof(double));

  fe_kappa_set(kappa);

  symmetric_flag=1;

  return;
}

/****************************************************************************
 *
 *  symmetric_a
 *
 ****************************************************************************/

__targetHost__ double symmetric_a(void) {

  return a_;

}

/****************************************************************************
 *
 *  symmetric_b
 *
 ****************************************************************************/

__targetHost__ double symmetric_b(void) {

  return b_;

}

/****************************************************************************
 *
 *  symmetric_phi
 *
 ****************************************************************************/

__targetHost__ void symmetric_phi(double** address_of_ptr) {

  *address_of_ptr = phi_->data;

  return;
  
}

/****************************************************************************
 *
 *  symmetric_t_phi
 *
 ****************************************************************************/

__targetHost__ void symmetric_t_phi(double** address_of_ptr) {

  *address_of_ptr = phi_->t_data;

  return;
  
}

/****************************************************************************
 *
 *  symmetric_gradphi
 *
 ****************************************************************************/

__targetHost__ void symmetric_gradphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->grad;
  
  return;
}

/****************************************************************************
 *
 *  symmetric_t_gradphi
 *
 ****************************************************************************/

__targetHost__ void symmetric_t_gradphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->t_grad;
  
  return;
}

/****************************************************************************
 *
 *  symmetric_delsqphi
 *
 ****************************************************************************/

__targetHost__ void symmetric_delsqphi(double** address_of_ptr) {

  *address_of_ptr = grad_phi_->delsq;

  return;

}

/****************************************************************************
 *
 *  symmetric_t_delsqphi
 *
 ****************************************************************************/

__targetHost__ void symmetric_t_delsqphi(double** address_of_ptr) {

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

__targetHost__ double symmetric_interfacial_tension(void) {

  double sigma;


  sigma = sqrt(-8.0*kappa_*a_*a_*a_/(9.0*b_*b_));

  return sigma;
}

/****************************************************************************
 *
 *  symmetric_interfacial_width
 *
 ****************************************************************************/

__targetHost__ double symmetric_interfacial_width(void) {

  double xi;

  xi = sqrt(-2.0*kappa_/a_);

  return xi;
}

/****************************************************************************
 *
 *  symmetric_free_energy_density
 *
 *  The free energy density is as above.
 *
 ****************************************************************************/

__targetHost__ double symmetric_free_energy_density(const int index) {

  double phi;
  double dphi[3];
  double e;


  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, dphi);

  e = 0.5*a_*phi*phi + 0.25*b_*phi*phi*phi*phi
	+ 0.5*kappa_*dot_product(dphi, dphi);

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

__targetHost__
double symmetric_chemical_potential(const int index, const int nop) {

  double phi;
  double delsq_phi;
  double mu;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  mu = a_*phi + b_*phi*phi*phi - kappa_*delsq_phi;

  return mu;
}

__target__
double symmetric_chemical_potential_target(const int index, const int nop,
					   const double * t_phi,
					   const double * t_delsqphi) {
  double phi;
  double delsq_phi;
  double mu;

  phi = t_phi[addr_rank0(le_nsites(), index)];
  delsq_phi = t_delsqphi[addr_rank0(le_nsites(), index)];

  mu = t_a_*phi + t_b_*phi*phi*phi - t_kappa_*delsq_phi;

  return mu;
}


__targetHost__
void get_chemical_potential_target(mu_fntype* t_chemical_potential) {

  mu_fntype h_chemical_potential;

  /* get host copy of function pointer*/
  copyConstFromTarget(&h_chemical_potential,
		      &p_symmetric_chemical_potential_target,
		      sizeof(mu_fntype));

  /* and put back on target, now in an accessible location*/
  copyToTarget(t_chemical_potential,
	       &h_chemical_potential, sizeof(mu_fntype));

  return;
}


/****************************************************************************
 *
 *  symmetric_isotropic_pressure
 *
 *  This ignores the term in the density (assumed to be uniform).
 *
 ****************************************************************************/

__targetHost__ double symmetric_isotropic_pressure(const int index) {

  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

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

__targetHost__ void symmetric_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  assert(phi_);
  assert(grad_phi_);

  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+ kappa_*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}

__target__
void symmetric_chemical_stress_target(const int index,
				      double s[3][3*NSIMDVL],
				      const double* t_phi, 
				      const double* t_gradphi,
				      const double* t_delsqphi) {
  int ia, ib;
  int iv;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  __targetILP__(iv) {
    
    for (ia = 0; ia < 3; ia++) {
      grad_phi[ia] = t_gradphi[vaddr_rank2(le_nsites(), 1, 3, index, 0, ia, iv)];
    }

    phi = t_phi[vaddr_rank1(le_nsites(), 1, index, 0, iv)];
    delsq_phi = t_delsqphi[vaddr_rank1(le_nsites(), 1, index, 0, iv)];

    p0 = 0.5*t_a_*phi*phi + 0.75*t_b_*phi*phi*phi*phi
      - t_kappa_*phi*delsq_phi 
      - 0.5*t_kappa_*(grad_phi[X]*grad_phi[X] + grad_phi[Y]*grad_phi[Y]
		      + grad_phi[Z]*grad_phi[Z]);
    
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib*VVL+ iv] = p0*tc_d_[ia][ib]
	  + t_kappa_*grad_phi[ia]*grad_phi[ib];
      }
    }
  }

  return;
}


__targetHost__ void get_chemical_stress_target(pth_fntype* t_chemical_stress) {

  pth_fntype h_chemical_stress;

  /* get host copy of function pointer */
  copyConstFromTarget(&h_chemical_stress,
		      &p_symmetric_chemical_stress_target,
		      sizeof(pth_fntype));

  /* and put back on target, now in an accessible location*/
  copyToTarget( t_chemical_stress, &h_chemical_stress, sizeof(pth_fntype));

  return;
}
