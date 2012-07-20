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

#ifdef OLD_PHI
#include "phi.h"
#include "phi_gradients.h"
#else
#include "field.h"
#include "field_grad.h"
#endif

#include "util.h"
#include "symmetric.h"

static double a_     = -0.003125;
static double b_     = +0.003125;
static double kappa_ = +0.002;

#ifdef OLD_PHI
#else
static field_t * phi_ = NULL;
static field_grad_t * grad_phi_ = NULL;

/****************************************************************************
 *
 *  symmetric_phi_set
 *
 *  Attach a reference to the order parameter field object, and the
 *  associated gradient object.
 *
 ****************************************************************************/

int symmetric_phi_set(field_t * phi, field_grad_t * dphi) {

  assert(phi);
  assert(dphi);

  phi_ = phi;
  grad_phi_ = dphi;

  return 0;
}
#endif

/****************************************************************************
 *
 *  symmetric_free_energy_parameters_set
 *
 *  No constraints are placed on the parameters here, so the caller is
 *  responsible for making sure they are sensible.
 *
 ****************************************************************************/

void symmetric_free_energy_parameters_set(double a, double b, double kappa) {

  a_ = a;
  b_ = b;
  kappa_ = kappa;
  fe_kappa_set(kappa);

  return;
}

/****************************************************************************
 *
 *  symmetric_a
 *
 ****************************************************************************/

double symmetric_a(void) {

  return a_;
}

/****************************************************************************
 *
 *  symmetric_b
 *
 ****************************************************************************/

double symmetric_b(void) {

  return b_;
}

/****************************************************************************
 *
 *  symmetric_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

double symmetric_interfacial_tension(void) {

  double sigma;

  sigma = sqrt(-8.0*kappa_*a_*a_*a_/(9.0*b_*b_));

  return sigma;
}

/****************************************************************************
 *
 *  symmetric_interfacial_width
 *
 ****************************************************************************/

double symmetric_interfacial_width(void) {

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

double symmetric_free_energy_density(const int index) {

  double phi;
  double dphi[3];
  double e;

#ifdef OLD_PHI
  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, dphi);
#else
  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, dphi);
#endif

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

double symmetric_chemical_potential(const int index, const int nop) {

  double phi;
  double delsq_phi;
  double mu;

#ifdef OLD_PHI
  assert(nop == 0);

  phi = phi_get_phi_site(index);
  delsq_phi = phi_gradients_delsq(index);
#else
  field_scalar(phi_, index, &phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);
#endif

  mu = a_*phi + b_*phi*phi*phi - kappa_*delsq_phi;

  return mu;
}

/****************************************************************************
 *
 *  symmetric_isotropic_pressure
 *
 *  This ignores the term in the density (assumed to be uniform).
 *
 ****************************************************************************/

double symmetric_isotropic_pressure(const int index) {

  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

#ifdef OLD_PHI
  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, grad_phi);
  delsq_phi = phi_gradients_delsq(index);
#else
  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);
#endif

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

void symmetric_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

#ifdef OLD_PHI
  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, grad_phi);
  delsq_phi = phi_gradients_delsq(index);
#else
  field_scalar(phi_, index, &phi);
  field_grad_scalar_grad(grad_phi_, index, grad_phi);
  field_grad_scalar_delsq(grad_phi_, index, &delsq_phi);
#endif

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+ kappa_*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}
