/****************************************************************************
 *
 *  brazovskii.c
 *
 *  This is the implementation of the Brazovskii free energy:
 *
 *  F[phi] = (1/2) A phi^2 + (1/4) B phi^4 + (1/2) kappa (\nabla \phi)^2
 *                                         + (1/2) C (\nabla^2 \phi)^2
 *
 *  so there is one additional term compared with the symmetric phi^4
 *  case. The original reference is S. A. Brazovskii, Sov. Phys. JETP,
 *  {\bf 41} 85 (1975). One can see also, for example, Xu et al, PRE
 *  {\bf 74} 011505 (2006) for details.
 *
 *  Parameters:
 *
 *  One should have b, c > 0 for stability purposes. Then for a < 0
 *  and kappa > 0 one gets two homogenous phases with
 *  phi = +/- sqrt(-a/b) cf. the symmetric case.
 *
 *  Negative kappa favours the presence of interfaces, and lamellae
 *  can form. Approximately, the lamellar phase can be described by
 *  phi ~= A sin(k_0 x) in the traverse direction, where
 *  A^2 = 4 (1 + kappa^2/4cb)/3 and k_0 = sqrt(-kappa/2c). 
 *
 *  $Id: brazovskii.c,v 1.1.2.2 2010-04-02 07:56:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "phi.h"
#include "phi_gradients.h"
#include "util.h"
#include "brazovskii.h"

static double a_     = -0.00;
static double b_     = +0.00;
static double kappa_ = +0.00;
static double c_     = -0.00;

/****************************************************************************
 *
 *  brazovskii_free_energy_parameters_set
 *
 *  No constrints on the parameters are enforced, but see comments
 *  above.
 *
 ****************************************************************************/

void brazovskii_free_energy_parameters_set(double a, double b, double kappa,
					   double c) {
  a_ = a;
  b_ = b;
  kappa_ = kappa;
  c_ = c;

  return;
}

/****************************************************************************
 *
 *  brazovskii_amplitude
 *
 *  Return the single-mode approximation amplitude.
 *
 ****************************************************************************/

double brazovskii_amplitude(void) {

  double a;

  a = sqrt(4.0*(1.0 + kappa_*kappa_/(4.0*b_*c_))/3.0);

  return a;
}

/****************************************************************************
 *
 *  brazovskii_wavelength
 *
 *  Return the single-mode approximation wavelength 2\pi / k_0.
 *
 ****************************************************************************/

double brazovskii_wavelength(void) {

  double lambda;

  lambda = 2.0*4.0*atan(1.0) / sqrt(-kappa_/(2.0*c_));

  return lambda;
}

/****************************************************************************
 *
 *  brazovskii_free_energy_density
 *
 *  The free energy density.
 *
 ****************************************************************************/

double brazovskii_free_energy_density(const int index) {

  double phi;
  double dphi[3];
  double delsq;
  double e;

  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, dphi);
  delsq = phi_gradients_delsq(index);

  e = 0.5*a_*phi*phi* + 0.25*b_*phi*phi*phi*phi
    + 0.5*kappa_*dot_product(dphi, dphi) + 0.5*c_*delsq*delsq;

  return e;
}

/****************************************************************************
 *
 *  brazovskii_chemical_potential
 *
 *  The chemical potential \mu = \delta F / \delta \phi
 *                             = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *                                               + c (\nabla^2)(\nabla^2 \phi)
 *
 ****************************************************************************/

double brazovskii_chemical_potential(const int index, const int nop) {

  double phi;
  double del2_phi;
  double del4_phi;
  double mu;

  assert(nop == 0);

  phi      = phi_get_phi_site(index);
  del2_phi = phi_gradients_delsq(index);
  del4_phi = phi_gradients_delsq_delsq(index);

  mu = a_*phi + b_*phi*phi*phi - kappa_*del2_phi + c_*del4_phi;

  return mu;
}

/****************************************************************************
 *
 *  brazovskii_isotropic_pressure
 *
 *  This ignores the term in the density (assumed to be uniform).
 *
 ****************************************************************************/

double brazovskii_isotropic_pressure(const int index) {

  double phi;
  double del2_phi;
  double del4_phi;
  double grad_phi[3];
  double grad_del2_phi[3];
  double p0;

  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, grad_phi);
  del2_phi = phi_gradients_delsq(index);

  del4_phi = phi_gradients_delsq_delsq(index);
  phi_gradients_grad_delsq(index, grad_del2_phi);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi - kappa_*phi*del2_phi
    + 0.5*kappa_*dot_product(grad_phi, grad_phi) + c_*phi*del4_phi
    + 0.5*c_*del2_phi*del2_phi+ c_*dot_product(grad_phi, grad_del2_phi);

  return p0;
}

/****************************************************************************
 *
 *  brazovskii_chemical_stress
 *
 *  Return the chemical stress tensor for given position index.
 *
 ****************************************************************************/

void brazovskii_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double del2_phi;
  double del4_phi;
  double grad_phi[3];
  double grad_del2_phi[3];
  double p0;

  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, grad_phi);
  del2_phi = phi_gradients_delsq(index);

  del4_phi = phi_gradients_delsq_delsq(index);
  phi_gradients_grad_delsq(index, grad_del2_phi);

  /* Isotropic part and tensor part */

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi - kappa_*phi*del2_phi
    + 0.5*kappa_*dot_product(grad_phi, grad_phi) + c_*phi*del4_phi
    + 0.5*c_*del2_phi*del2_phi + c_*dot_product(grad_phi, grad_del2_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib] + kappa_*grad_phi[ia]*grad_phi[ib]
      - c_*(grad_phi[ia]*grad_del2_phi[ib] + grad_phi[ib]*grad_del2_phi[ia]);
    }
  }

  return;
}
