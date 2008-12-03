/*****************************************************************************
 *
 *  free_energy
 *
 *  This is the symmetric phi^4 free energy:
 *
 *  F[\phi] = (1/2) A \phi^2 + (1/4) B \phi^4 + (1/2) \kappa (\nabla\phi)^2
 *
 *  The first two terms represent the bulk free energy, while the
 *  final term penalises curvature in the interface. For a complete
 *  description see Kendon et al., J. Fluid Mech., 440, 147 (2001).
 *
 *  An option for free energy owing to Brazovskii is in the process
 *  of testing. This adds one term to the above:
 *
 *             (1/2) C (\nabla^2 \phi)^2 
 *
 *  $Id: free_energy.c,v 1.8 2008-12-03 20:31:13 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <string.h>

#include "pe.h"
#include "phi.h"
#include "runtime.h"
#include "utilities.h"
#include "free_energy.h"

static double A_     = -0.003125;
static double B_     = +0.003125;
static double C_     =  0.000;
static double kappa_ = +0.002;
static int    is_brazovskii_ = 0;

/* Surfactant model of van der Sman and van der Graaf in development */

static double D_ = 0.0;
static double epsilon_ = 0.0;
static double W_ = 0.0;

/* The choice of free energy is currently determined by the value of C_ */ 
static void (* fe_chemical_stress)(const int, double [3][3]);
static void fe_chemical_stress_zero(const int, double [3][3]);
static void fe_chemical_stress_symmetric(const int, double [3][3]);
static void fe_chemical_stress_brazovskii(const int, double [3][3]);

static double (** fe_chemical_potential)(const int);
static double fe_chemical_potential_brazovskii(const int);
static double fe_chemical_potential_sman_phi(const int);
static double fe_chemical_potential_sman_psi(const int);
static void fe_chemical_stress_sman(const int, double [3][3]);

/*****************************************************************************
 *
 *  free_energy_init
 *
 *  Get the user's parameters, if present. 
 *
 *****************************************************************************/

void init_free_energy() {

  int n;
  char description[128];

  n = RUN_get_double_parameter("A", &A_);
  n = RUN_get_double_parameter("B", &B_);
  n = RUN_get_double_parameter("K", &kappa_);
  n = RUN_get_double_parameter("C", &C_);

  n = RUN_get_string_parameter("free_energy", description, 128);

  if (nop_ > 0) {
    fe_chemical_potential = malloc(nop_*sizeof(void *));
    if (fe_chemical_potential == NULL) {
      fatal("malloc(fe_chemical_potential failed\n");
    }
  }

#ifdef _SINGLE_FLUID_
  fe_chemical_stress = fe_chemical_stress_zero;
#else
  if (n == 0) {
    info("[Default] Free energy: symmetric\n");
    strcpy(description, "symmetric");
  }
  else {
    info("[User   ] Free energy: %s\n", description);
  }

  if (strcmp(description, "symmetric") == 0) {
    if (A_ > 0.0) {
      fatal("The free energy parameter A must be negative\n");
    }
    if (B_ < 0.0) {
      fatal("The free energy parameter B must be positive\n");
    }
    if (kappa_ < 0.0) {
      fatal("The free energy parameter kappa must be positive\n");
    }

    assert(nop_ == 1);
    info("\nSymmetric phi^4 free energy:\n");
    info("Bulk parameter A      = %f\n", A_);
    info("Bulk parameter B      = %f\n", B_);
    info("Surface penalty kappa = %f\n", kappa_);
    info("Surface tension       = %f\n", surface_tension());
    info("Interfacial width     = %f\n", interfacial_width());
    fe_chemical_stress = fe_chemical_stress_symmetric;
    fe_chemical_potential[0] = fe_chemical_potential_brazovskii;
  }
  else if (strcmp(description, "brazovskii") == 0) {
    info("\nBrazovskii free energy:\n");
    info("Bulk parameter A      = %f\n", A_);
    info("Bulk parameter B      = %f\n", B_);
    info("Ext. parameter C      = %f\n", C_);
    info("Surface penalty kappa = %f\n", kappa_);
    info("Surface tension       = %f\n", surface_tension());
    info("Interfacial width     = %f\n", interfacial_width());
    fe_chemical_stress = fe_chemical_stress_brazovskii;
    fe_chemical_potential[0] = fe_chemical_potential_brazovskii;
    is_brazovskii_ = 1;
  }
  else if (strcmp(description, "surfactant") == 0) {

    assert(nop_ == 2);

    n = RUN_get_double_parameter("epsilon", &epsilon_);
    n = RUN_get_double_parameter("W", &W_);
    n = RUN_get_double_parameter("D", &D_);

    info("Surfactant model free energy\n");
    info("Bulk parameter A      = %f\n", A_);
    info("Bulk parameter B      = %f\n", B_);
    info("Ext. parameter C      = %f\n", C_);
    info("Surface penalty kappa = %f\n", kappa_);
    info("Surface tension       = %f\n", surface_tension());
    info("Interfacial width     = %f\n", interfacial_width());
    info("Scale energy D        = %f\n", D_);
    info("Surface adsorption e  = %f\n", epsilon_);
    info("Enthalpic term W      = %f\n", W_);
    assert(nop_ == 2);
    fe_chemical_stress = fe_chemical_stress_sman;
    fe_chemical_potential[0] = fe_chemical_potential_sman_phi;
    fe_chemical_potential[1] = fe_chemical_potential_sman_psi;
  }
  else {
    fatal("Unrecognised free energy\n");
  }
#endif

  return;
}

/*****************************************************************************
 *
 *  free_energy_A
 *  free_energy_B
 *  free_energy_K
 *
 *****************************************************************************/

double free_energy_A() {
  return A_;
}
double free_energy_B() {
  return B_;
}
double free_energy_K() {
  return kappa_;
}

void free_energy_set_A(double a) {
  assert(a < 0.0);
  A_ = a;
  return;
}

void free_energy_set_B(double b) {
  assert(b > 0.0);
  B_ = b;
  return;
}

void free_energy_set_kappa(double k) {
  assert(k > 0.0);
  kappa_ = k;
  return;
}

/*****************************************************************************
 *
 *  free_energy_is_brazovskii
 *
 *****************************************************************************/

int free_energy_is_brazovskii() {

  return is_brazovskii_;
}

/*****************************************************************************
 *
 *  surface_tension
 *
 *  Return the theoretical surface tension for the model.
 *
 *****************************************************************************/

double surface_tension() {

  return sqrt(-8.0*kappa_*A_*A_*A_ / (9.0*B_*B_));
}

/*****************************************************************************
 *
 *  interfacial_width
 *
 *  Return the theoretical interfacial width. Note that there is a
 *  typo in 6.17 of Kendon et al (2001); the factor of 2 should be in
 *  numerator.
 *
 *****************************************************************************/

double interfacial_width() {

  return sqrt(-2.0*kappa_ / A_);
}

/*****************************************************************************
 *
 *  free_energy_get_chemical_potential
 *
 *  Return the chemical potential at given position index.
 *
 *****************************************************************************/

double free_energy_get_chemical_potential(const int index) {

  assert(nop_ >= 1);
  return fe_chemical_potential[0](index);
}

/*****************************************************************************
 *
 *  free_energy_chemical_potential
 *
 *  As above, but allows for multiple order parameters.
 *
 *****************************************************************************/

double free_energy_chemical_potential(const int index, const int nop) {

  assert(nop <= nop_);
  return fe_chemical_potential[nop](index);
}

/*****************************************************************************
 *
 *  free_energy_get_chemical_stress
 *
 *  Driver function to return the chemical stress tensor for given
 *  position index.
 *
 *****************************************************************************/

void free_energy_get_chemical_stress(const int index, double p[3][3]) {

  assert(fe_chemical_stress);
  fe_chemical_stress(index, p);

  return;
}

/*****************************************************************************
 *
 *  fe_chemical_stress_symmetric
 *
 *  Return the chemical stress tensor for given position index.
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 *****************************************************************************/

static void fe_chemical_stress_symmetric(const int index, double p[3][3]) {

  int ia, ib;
  double phi, bulk, delsq_phi, grad_phi_sq;
  double grad_phi[3];
  extern const double d_[3][3]; /* Pending Refactor util etc. */ 

  phi = phi_get_phi_site(index);
  phi_get_grad_phi_site(index, grad_phi);
  delsq_phi = phi_get_delsq_phi_site(index);

  bulk = 0.5*phi*phi*(A_ + 1.5*B_*phi*phi);
  grad_phi_sq = dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = (bulk - kappa_*(phi*delsq_phi + 0.5*grad_phi_sq))*d_[ia][ib]
	+ kappa_*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_chemical_stress_brazovskii
 *
 *  Return the chemical stress tensor for given position index.
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 *****************************************************************************/

static void fe_chemical_stress_brazovskii(const int index, double p[3][3]) {

  int ia, ib;
  double phi, delsq_phi, grad_phi_sq;
  double delsq_delsq_phi;
  double bulk_symmetric, brazovskii;
  double grad_phi[3];
  double grad_delsq_phi[3];
  extern const double d_[3][3]; /* Pending Refactor util etc. */ 

  phi = phi_get_phi_site(index);
  phi_get_grad_phi_site(index, grad_phi);
  delsq_phi = phi_get_delsq_phi_site(index);

  grad_phi_sq = dot_product(grad_phi, grad_phi);
  bulk_symmetric = 0.5*phi*phi*(A_ + 1.5*B_*phi*phi)
    - kappa_*(phi*delsq_phi + 0.5*grad_phi_sq);

  /* Extra terms in Brazovskii */
  delsq_delsq_phi = phi_get_delsq_delsq_phi_site(index);
  phi_get_grad_delsq_phi_site(index, grad_delsq_phi);

  brazovskii = phi*delsq_delsq_phi + 0.5*delsq_phi*delsq_phi
    + dot_product(grad_phi, grad_delsq_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = (bulk_symmetric + C_*brazovskii)*d_[ia][ib]
	        + kappa_*grad_phi[ia]*grad_phi[ib]
      - C_*(grad_phi[ia]*grad_delsq_phi[ib] + grad_phi[ib]*grad_delsq_phi[ia]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_chemical_stress_zero
 *
 *  No free version.
 *
 *****************************************************************************/

void fe_chemical_stress_zero(const int index, double pchem[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pchem[ia][ib] = 0.0;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  free_energy_get_isotropic_pressure
 *
 *  Return the isotropic part of the pressure tensor.
 *  P_0 = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nabla phi)^2]
 *
 *****************************************************************************/

double free_energy_get_isotropic_pressure(const int index) {

  double p0, phi, bulk, delsq_phi, grad_phi_sq;
  double grad_phi[3];

  phi = phi_get_phi_site(index);
  phi_get_grad_phi_site(index, grad_phi);
  delsq_phi = phi_get_delsq_phi_site(index);

  bulk = 0.5*phi*phi*(A_ + 1.5*B_*phi*phi);
  grad_phi_sq = dot_product(grad_phi, grad_phi);
  p0 = bulk - kappa_*(phi*delsq_phi + 0.5*grad_phi_sq);

  return p0;
}

/*****************************************************************************
 *
 *  free_energy_density
 *
 *  Return the free energy density
 *  E = (1/2) A phi^2 + (1/4) B phi^4 + (1/2) kappa (\nabla phi)^2
 *    + (1/2) C (\nabla^2 phi)^2
 *
 *****************************************************************************/

double free_energy_density(const int index) {

  double e, bulk;
  double phi, dphi[3], delsq;

  phi = phi_get_phi_site(index);
  phi_get_grad_phi_site(index, dphi);
  delsq = phi_get_delsq_phi_site(index);

  bulk = phi*phi*(A_ + 0.5*B_*phi*phi);
  e = 0.5*(bulk + kappa_*dot_product(dphi, dphi) + C_*delsq*delsq);

  return e;
}

/*****************************************************************************
 *
 *  fe_chemical_potential_brazovskii
 *
 *  Return the chemical potential at given position index.
 *  The symmetric is a special case of the Brazovskii with C = 0.
 *
 *****************************************************************************/

double fe_chemical_potential_brazovskii(const int index) {

  double phi, delsq_phi, delsq_sq_phi, mu;

  phi = phi_get_phi_site(index);
  delsq_phi = phi_get_delsq_phi_site(index);
  delsq_sq_phi = phi_get_delsq_delsq_phi_site(index);

  mu = phi*(A_ + B_*phi*phi) - kappa_*delsq_phi + C_*delsq_sq_phi;

  return mu;
}

/*****************************************************************************
 *
 *  fe_chemical_potential_sman_phi
 *
 *  Chemical potential for compositional part of surfactant model of
 *  van der Sman and van der Graaf.
 *
 *****************************************************************************/

static double fe_chemical_potential_sman_phi(const int index) {

  double mu;
  double phi, psi, delsq_phi;
  double dphi[3];
  double dpsi[3];

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  delsq_phi = phi_op_get_delsq_phi_site(index, 0);
  phi_op_get_grad_phi_site(index, 0, dphi);
  phi_op_get_grad_phi_site(index, 1, dpsi);

  mu = phi*(A_ + B_*phi*phi) - kappa_*delsq_phi + W_*phi*psi
    + epsilon_*(psi*delsq_phi + dot_product(dphi, dpsi));

  return mu;
}

/*****************************************************************************
 *
 *  fe_chemical_potential_sman_psi
 *
 *  Surfactant part of the chemical potentail for the surfactant model
 *  of van der Sman and van der Graaf.
 * 
 *  mu_\psi = D (ln \psi - ln (1 - \psi) + (1/2) W \phi^2
 *          - (1/2) \epsilon (\nabla \phi)^2
 *
 *****************************************************************************/

static double fe_chemical_potential_sman_psi(const int index) {

  double mu;
  double phi, psi;
  double dphi[3];

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  phi_op_get_grad_phi_site(index, 0, dphi);

  assert(psi > 0.0);
  assert(psi < 1.0);

  mu = D_*(log(psi) - log(1.0-psi)) + 0.5*W_*phi*phi
    - 0.5*epsilon_*dot_product(dphi, dphi);

  return mu;
}

/*****************************************************************************
 *
 *  fe_chemical_stress_sman
 *
 *  The chemical stress of the van der Sman and van der Graaf model.
 *
 *  P_ab = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *       - D ln(1 - \psi) + W \psi \phi^2
 *       + \epsilon \phi \nabla_a \phi \nabla_b \psi
 *       + \epsilon \phi \psi \nabla^2 \phi
 *       + (\kappa - \epsilon\psi) \nabla_a \phi \nabla_b \phi
 *
 *****************************************************************************/

static void fe_chemical_stress_sman(const int index, double p[3][3]) {

  int ia, ib;
  double phi, psi, delsq_phi, p0;
  double dphi[3];
  double dpsi[3];
  extern const double d_[3][3]; /* Pending Refactor util etc. */ 

  phi = phi_op_get_phi_site(index, 0);
  psi = phi_op_get_phi_site(index, 1);
  delsq_phi = phi_op_get_delsq_phi_site(index, 0);
  phi_op_get_grad_phi_site(index, 0, dphi);
  phi_op_get_grad_phi_site(index, 1, dpsi);

  assert(psi < 1.0);

  p0 = 0.5*phi*phi*(A_ + 1.5*B_*phi*phi)
    - kappa_*(phi*delsq_phi + 0.5*dot_product(dphi, dphi))
    - D_*log(1.0 - psi) + W_*psi*phi*phi
    + epsilon_*(dot_product(dphi, dpsi) + phi*psi*delsq_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = p0*d_[ia][ib]
	+ (kappa_ - epsilon_*psi)*dphi[ia]*dphi[ib];
    }
  }

  return;
}
