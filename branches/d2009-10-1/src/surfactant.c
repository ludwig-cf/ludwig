/****************************************************************************
 *
 *  surfactant.c
 *
 *  Implementation of the surfactant free energy described by
 *  van der Graff and van der Sman TODO
 *
 *  Two order parameters are required:
 *
 *  [0] \phi is compositional order parameter (cf symmetric free energy)
 *  [1] \psi is surfactant concentration (strictly 0 < psi < 1)
 *
 *  The free energy density is:
 *
 *    F = F_\phi + F_\psi + F_surf + F_add
 *
 *  with
 *
 *    F_phi  = symmetric phi^4 free energy
 *    F_psi  = kT [\psi ln \psi + (1 - \psi) ln (1 - \psi)] 
 *    F_surf = - (1/2)\epsilon\psi (grad \phi)^2
 *             - (1/2)\beta \psi^2 (grad \phi)^2
 *    F_add  = + (1/2) W \psi \phi^2
 *
 *  The beta term allows one to get at the Frumkin isotherm and has
 *  been added here.
 *
 *  $Id: surfactant.c,v 1.1.2.1 2009-11-04 09:55:34 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "phi.h"
#include "util.h"

static double a_       = -0.0208333;
static double b_       = +0.0208333;
static double kappa_   = +0.12;

static double kt_      = 0.00056587;
static double epsilon_ = 0.03;
static double beta_    = 0.0;
static double w_       = 0.0;

/****************************************************************************
 *
 *  surfactant_fluid_parameters_set
 *
 ****************************************************************************/

void surfactant_fluid_parameters_set(double a, double b, double kappa) {

  a_ = a;
  b_ = b;
  kappa_ = kappa;

  return;
}

/****************************************************************************
 *
 *  surfactant_parameters_set
 *
 ****************************************************************************/

void surfactant_parameters_set(double kt, double e, double beta, double w) {

  kt_ = kt;
  epsilon_ = e;
  beta_ = beta;
  w_ = w;

  return;
}


/****************************************************************************
 *
 *  surfactant_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

double surfactant_interfacial_tension(void) {

  double sigma;

  sigma = sqrt(-8.0*a_*a_*a_/(9.0*b_*b_));

  return sigma;
}

/****************************************************************************
 *
 *  surfactant_interfacial_width
 *
 ****************************************************************************/

double surfactant_interfacial_width(void) {

  double xi;

  xi = sqrt(-2.0*kappa_/a_);

  return xi;
}

/****************************************************************************
 *
 *  surfactant_langmuir_isotherm
 *
 *  The Langmuir isotherm psi_c is given by
 *  
 *  ln psi_c = (1/2) epsilon / (kT xi_0^2)
 *
 *  and can be a useful reference. The situation is more complex if
 *  beta is not zero.
 *
 ****************************************************************************/ 

double surfactant_langmuir_isotherm(void) {

  double psi_c;
  double xi0;
  
  xi0 = surfactant_interfacial_width();
  psi_c = exp(0.5*epsilon_ / (kt_*xi0*xi0));

  return psi_c;
}

/****************************************************************************
 *
 *  surfactant_free_energy_density
 *
 *  This is:
 *     (1/2)A \phi^2 + (1/4)B \phi^4 + (1/2) kappa (\nabla\phi)^2
 *   + kT [ \psi ln \psi + (1 - \psi) ln (1 - \psi) ]
 *   - (1/2) \epsilon\psi (\nabla\phi)^2 - (1/2) \beta \psi^2 (\nabla\phi)^2
 *   + (1/2)W\psi\phi^2
 *
 ****************************************************************************/

double surfactant_free_energy_density(const int index) {

  double e;
  double phi;
  double psi;
  double dphi[3];
  double dphisq;

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  phi_get_grad_phi_site(index, dphi);

  dphisq = dot_product(dphi, dphi);

  /* We have the symmetric piece followed by terms in psi */

  e = 0.5*a_*phi*phi + 0.25*b_*phi*phi*phi*phi + 0.5*kappa_*dphisq;

  assert(psi > 0.0);
  assert(psi < 1.0);

  e += kt_*(psi*log(psi) + (1.0 - psi)*log(1.0 - psi))
    -0.5*epsilon_*psi*dphisq - 0.5*beta_*psi*psi*dphisq + 0.5*w_*psi*phi*phi;

  return e;
}

/****************************************************************************
 *
 *  surfactant_chemical_potential
 *
 *  \mu_\phi = A\phi + B\phi^3 - kappa \nabla^2 \phi
 *           + W\phi \psi
 *           + \epsilon (\psi \nabla^2\phi + \nabla\phi . \nabla\psi)
 *           + \beta (\psi^2 \nabla^2\phi + 2\psi \nabla\phi . \nabla\psi) 
 * 
 *  \mu_\psi = kT (ln \psi - ln (1 - \psi) + (1/2) W \phi^2
 *           - (1/2) \epsilon (\nabla \phi)^2
 *           - \beta \psi (\nabla \phi)^2
 *
 ****************************************************************************/

double surfactant_chemical_potential(const int index, const int nop) {

  double phi;
  double psi;
  double dphi[3];
  double dpsi[3];
  double delsq_phi;
  double mu;

  assert(nop == 0 || nop == 1);

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  phi_op_get_grad_phi_site(index, 0, dphi);

  /* There's a rather ugly switch here... */

  if (nop == 0) {
    /* mu_phi */
    delsq_phi = phi_op_get_delsq_phi_site(index, 0);
    phi_op_get_grad_phi_site(index, 1, dpsi);

    mu = a_*phi + b_*phi*phi*phi - kappa_*delsq_phi
      + w_*phi*psi
      + epsilon_*(psi*delsq_phi + dot_product(dphi, dpsi))
      + beta_*psi*(psi*delsq_phi + 2.0*dot_product(dphi, dpsi));
  }
  else {
    /* mu_psi */
    assert(psi > 0.0);
    assert(psi < 1.0);

    mu = kt_*(log(psi) - log(1.0-psi))
      + 0.5*w_*phi*phi
      - 0.5*epsilon_*dot_product(dphi, dphi)
      - beta_*psi*dot_product(dphi, dphi);
  }

  return mu;
}

/****************************************************************************
 *
 *  surfactant_isotropic_pressure
 *
 *  See below.
 *
 ****************************************************************************/

double surfactant_isotropic_pressure(const int index) {

  double phi;
  double psi;
  double delsq_phi;
  double dphi[3];
  double dpsi[3];
  double p0;

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  delsq_phi = phi_op_get_delsq_phi_site(index, 0);
  phi_op_get_grad_phi_site(index, 0, dphi);
  phi_op_get_grad_phi_site(index, 1, dpsi);

  assert(psi < 1.0);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(dphi, dphi)
    - kt_*log(1.0 - psi) + w_*psi*phi*phi
    + epsilon_*phi*(dot_product(dphi, dpsi) + psi*delsq_phi)
    + beta_*psi*(2.0*phi*dot_product(dphi, dpsi) + phi*psi*delsq_phi
		 - 0.5*psi*dot_product(dphi, dphi));

  return p0;
}

/****************************************************************************
 *
 *  surfactant_chemical_stress
 *
 *  S_ab = p0 delta_ab + P_ab
 *
 *  p0 = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *     - (1/2) kappa (\nabla phi)^2
 *     - kT ln(1 - \psi)
 *     + W \psi \phi^2
 *     + \epsilon \phi \nabla_a \phi \nabla_a \psi
 *     + \epsilon \phi \psi \nabla^2 \phi
 *     + 2 \beta \phi \psi \nabla_a\phi \nabla_a\psi
 *     + \beta\phi\psi^2 \nabla^2 \phi
 *     - (1/2) \beta\psi^2 (\nabla\phi)^2  
 *
 *  P_ab = (\kappa - \epsilon\psi - \beta\psi^2) \nabla_a \phi \nabla_b \phi
 *
 ****************************************************************************/

void surfactant_chemical_stress(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double psi;
  double delsq_phi;
  double dphi[3];
  double dpsi[3];
  double p0;

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  delsq_phi = phi_op_get_delsq_phi_site(index, 0);
  phi_op_get_grad_phi_site(index, 0, dphi);
  phi_op_get_grad_phi_site(index, 1, dpsi);

  assert(psi < 1.0);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(dphi, dphi)
    - kt_*log(1.0 - psi) + w_*psi*phi*phi
    + epsilon_*phi*(dot_product(dphi, dpsi) + psi*delsq_phi)
    + beta_*psi*(2.0*phi*dot_product(dphi, dpsi) + phi*psi*delsq_phi
		 - 0.5*psi*dot_product(dphi, dphi));

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]
	+ (kappa_ - epsilon_*psi - beta_*psi*psi)*dphi[ia]*dphi[ib];
    }
  }

  return;
}
