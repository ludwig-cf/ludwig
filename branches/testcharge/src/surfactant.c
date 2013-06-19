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
 *  $Id: surfactant.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "surfactant.h"

/* Some values might be
 * a_       = -0.0208333;
 * b_       = +0.0208333;
 * kappa_   = +0.12;
 * 
 * kt_      = 0.00056587;
 * epsilon_ = 0.03;
 * beta_    = 0.0;
 * w_       = 0.0;
 */

typedef struct fe_surfactant_s fe_surfactant_t;

struct fe_surfactant_s {

  field_t * phi;         /* Single field object with two components */
  field_grad_t * grad;   /* gradients thereof */

  double a;              /* Symmetric a */
  double b;              /* Symmetric b */
  double kappa;          /* Symmetric kappa */

  double kt;             /* Not to be confused with physics_kt(). */
  double epsilon;        /* Surfactant epsilon */
  double beta;           /* Frumpkin isotherm */
  double w;              /* Surfactant w */

};

static fe_surfactant_t * fe = NULL;


/****************************************************************************
 *
 *  fe_surfactant_create
 *
 *  A single static instance
 *
 ****************************************************************************/

int fe_surfactant_create(field_t * phi, field_grad_t * grad) {

  assert(fe == NULL);
  assert(phi);
  assert(grad);

  fe = calloc(1, sizeof(fe_surfactant_t));
  if (fe == NULL) fatal("calloc(fe_surfactant) failed\n");

  fe->phi = phi;
  fe->grad = grad;

  return 0;
}

/****************************************************************************
 *
 *  fe_surfactant_free
 *
 ****************************************************************************/

void fe_surfactant_free(void) {

  if (fe) free(fe);
  fe = NULL;

  return;
}

/****************************************************************************
 *
 *  surfactant_fluid_parameters_set
 *
 ****************************************************************************/

int surfactant_fluid_parameters_set(double a, double b, double kappa) {

  assert(fe);

  fe->a = a;
  fe->b = b;
  fe->kappa = kappa;

  return 0;
}

/****************************************************************************
 *
 *  surfactant_parameters_set
 *
 ****************************************************************************/

int surfactant_parameters_set(double kt, double e, double beta, double w) {

  assert(fe);

  fe->kt = kt;
  fe->epsilon = e;
  fe->beta = beta;
  fe->w = w;

  return 0;
}


/****************************************************************************
 *
 *  surfactant_interfacial_tension
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

double surfactant_interfacial_tension(void) {

  double a, b;
  double sigma;

  assert(fe);

  a = fe->a;
  b = fe->b;

  sigma = sqrt(-8.0*fe->kappa*a*a*a/(9.0*b*b));

  return sigma;
}

/****************************************************************************
 *
 *  surfactant_interfacial_width
 *
 ****************************************************************************/

double surfactant_interfacial_width(void) {

  double xi;

  assert(fe);

  xi = sqrt(-2.0*fe->kappa/fe->a);

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

  assert(fe);

  xi0 = surfactant_interfacial_width();
  psi_c = exp(0.5*fe->epsilon / (fe->kt*xi0*xi0));

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
  double field[2];
  double phi;
  double psi;
  double dphi[3] = {0.0, 0.0, 0.0};
  double dphisq = 0.0;

  assert(fe);
  assert(0); /* Gradient needs work */

  field_scalar_array(fe->phi, index, field);
  phi = field[0];
  psi = field[1];

  dphisq = dot_product(dphi, dphi);

  /* We have the symmetric piece followed by terms in psi */

  e = 0.5*fe->a*phi*phi + 0.25*fe->b*phi*phi*phi*phi + 0.5*fe->kappa*dphisq;

  assert(psi > 0.0);
  assert(psi < 1.0);

  e += fe->kt*(psi*log(psi) + (1.0 - psi)*log(1.0 - psi))
    - 0.5*fe->epsilon*psi*dphisq
    - 0.5*fe->beta*psi*psi*dphisq + 0.5*fe->w*psi*phi*phi;

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
  double field[2];
  double dphi[3] = {0.0, 0.0, 0.0};
  double dpsi[3] = {0.0, 0.0, 0.0};
  double delsq_phi = 0.0;
  double mu;

  assert(nop == 0 || nop == 1);

  assert(fe);
  assert(0); /* Gradient needs work */

  field_scalar_array(fe->phi, index, field);
  phi = field[0];
  psi = field[1];

  /* There's a rather ugly switch here... */

  if (nop == 0) {
    /* mu_phi */

    mu = fe->a*phi + fe->b*phi*phi*phi - fe->kappa*delsq_phi
      + fe->w*phi*psi
      + fe->epsilon*(psi*delsq_phi + dot_product(dphi, dpsi))
      + fe->beta*psi*(psi*delsq_phi + 2.0*dot_product(dphi, dpsi));
  }
  else {
    /* mu_psi */
    assert(psi > 0.0);
    assert(psi < 1.0);

    mu = fe->kt*(log(psi) - log(1.0 - psi))
      + 0.5*fe->w*phi*phi
      - 0.5*fe->epsilon*dot_product(dphi, dphi)
      - fe->beta*psi*dot_product(dphi, dphi);
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

  double phi = 0.0;
  double psi = 0.0;
  double delsq_phi = 0.0;
  double dphi[3] = {0.0, 0.0, 0.0};
  double dpsi[3] = {0.0, 0.0, 0.0};
  double p0;

  assert(fe);
  assert(0); /* Sort values */

  assert(psi < 1.0);

  p0 = 0.5*fe->a*phi*phi + 0.75*fe->b*phi*phi*phi*phi
    - fe->kappa*phi*delsq_phi - 0.5*fe->kappa*dot_product(dphi, dphi)
    - fe->kt*log(1.0 - psi) + fe->w*psi*phi*phi
    + fe->epsilon*phi*(dot_product(dphi, dpsi) + psi*delsq_phi)
    + fe->beta*psi*(2.0*phi*dot_product(dphi, dpsi) + phi*psi*delsq_phi
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

  assert(fe);
  assert(0);

  assert(psi < 1.0);

  p0 = 0.5*fe->a*phi*phi + 0.75*fe->b*phi*phi*phi*phi
    - fe->kappa*phi*delsq_phi - 0.5*fe->kappa*dot_product(dphi, dphi)
    - fe->kt*log(1.0 - psi) + fe->w*psi*phi*phi
    + fe->epsilon*phi*(dot_product(dphi, dpsi) + psi*delsq_phi)
    + fe->beta*psi*(2.0*phi*dot_product(dphi, dpsi) + phi*psi*delsq_phi
		    - 0.5*psi*dot_product(dphi, dphi));

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]
	+ (fe->kappa - fe->epsilon*psi - fe->beta*psi*psi)*dphi[ia]*dphi[ib];
    }
  }

  return;
}
