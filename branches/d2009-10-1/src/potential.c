/*****************************************************************************
 *
 *  potential.c
 *
 *  Conservative potentials (soft-sphere, Leonard-Jones etc).
 *
 *  Provides routines for pairwise energies and forces.
 *
 *  $Id: potential.c,v 1.5.16.1 2010-06-02 14:10:48 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "runtime.h"
#include "physics.h"
#include "util.h"
#include "potential.h"


const double ENERGY_HARD_SPHERE = 100000.0;

static struct soft_sphere_potential_struct {
  int on;
  double epsilon;
  double sigma;
  double nu;
  double cutoff;
} soft_sphere;

static struct leonard_jones_potential_struct {
  int on;
  double sigma;
  double epsilon;
  double cutoff;
} leonard_jones;

static struct yukawa_struct {
  int on;
  double epsilon;
  double kappa;
  double cutoff;
} yukawa;

/*****************************************************************************
 *
 *  soft_sphere_init
 *
 *  Initialise the parameters for the soft-sphere interaction between
 *  colloids.
 *
 *****************************************************************************/

void soft_sphere_init() {

  int n;

  info("\nColloid-colloid soft-sphere potential\n");

  soft_sphere.on = 0;

  n = RUN_get_int_parameter("soft_sphere_on", &soft_sphere.on);
  info((n == 0) ? "[Default] " : "[User   ] ");
  info("Soft sphere potential is switched %s\n",
       (soft_sphere.on == 0) ? "off" : "on");

  if (soft_sphere.on) {
    n = RUN_get_double_parameter("soft_sphere_epsilon", &soft_sphere.epsilon);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere energy (epsilon) is %f (%f kT)\n", soft_sphere.epsilon,
	 soft_sphere.epsilon/get_kT());

    n = RUN_get_double_parameter("soft_sphere_sigma", &soft_sphere.sigma);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere width (sigma) is %f\n", soft_sphere.sigma);

    n = RUN_get_double_parameter("soft_sphere_nu", &soft_sphere.nu);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere exponent (nu) is %f\n", soft_sphere.nu);
    if (soft_sphere.nu <= 0.0) fatal("Please check nu is positive\n");

    n = RUN_get_double_parameter("soft_sphere_cutoff", &soft_sphere.cutoff);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Soft-sphere cutoff range is %f\n", soft_sphere.cutoff);
  }

  return;
}

/*****************************************************************************
 *
 *  leonard_jones_init
 *
 *  Initialise the parameters for the Leonard-Jones interaction between
 *  colloids.
 *
 *****************************************************************************/

void leonard_jones_init() {

  int n;

  info("\nColloid-colloid Leonard Jones potential\n");

  leonard_jones.on = 0;

  n = RUN_get_int_parameter("leonard_jones_on", &leonard_jones.on);
  info((n == 0) ? "[Default] " : "[User   ] ");
  info("Leonard Jones potential is switched %s\n",
       (leonard_jones.on == 0) ? "off" : "on");

  if (leonard_jones.on) {
    n = RUN_get_double_parameter("lj_epsilon", &leonard_jones.epsilon);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Leonard Jones energy (epsilon) is %f (%f kT)\n",
	 leonard_jones.epsilon, leonard_jones.epsilon/get_kT());

    n = RUN_get_double_parameter("lj_sigma", &leonard_jones.sigma);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Leonard Jones width (sigma) is %f\n", leonard_jones.sigma);

    n = RUN_get_double_parameter("lj_cutoff", &leonard_jones.cutoff);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Leonard Jones cutoff range is %f (%3.2f x sigma)\n",
	 leonard_jones.cutoff, leonard_jones.cutoff/leonard_jones.sigma);
  }

  return;
}

/*****************************************************************************
 *
 *  yukawa_init
 *
 *****************************************************************************/

void yukawa_init() {

  int n;

  info("\nColloid-colloid Yukawa potential\n");

  yukawa.on = 0;

  n = RUN_get_int_parameter("yukawa_on", &yukawa.on);
  info((n == 0) ? "[Default] " : "[User   ] ");
  info("Yukawa potential is switched %s\n",
       (yukawa.on == 0) ? "off" : "on");

  if (yukawa.on) {
    n = RUN_get_double_parameter("yukawa_epsilon", &yukawa.epsilon);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Yukawa energy (epsilon) is %f (%f kT)\n",
	 yukawa.epsilon, yukawa.epsilon/get_kT());

    n = RUN_get_double_parameter("yukawa_kappa", &yukawa.kappa);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Yukawa width (kappa) is %f\n", yukawa.kappa);

    n = RUN_get_double_parameter("yukawa_cutoff", &yukawa.cutoff);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Yukawa cutoff range is %f (%3.2f x kappa)\n",
	 yukawa.cutoff, yukawa.cutoff/yukawa.kappa);
  }

  return;
}

/*****************************************************************************
 *
 *  soft_sphere_energy
 *
 *  Return the energy of interaction between two particles with
 *  (surface-surface) separation h.
 *
 *****************************************************************************/

double soft_sphere_energy(const double h) {

  double e = 0.0;

  if (soft_sphere.on) {
    double hc = soft_sphere.cutoff;
    double nu = soft_sphere.nu;

    if (h > 0 && h < hc) {
      e = pow(h, -nu) - pow(hc, -nu)*(1.0 - (h-hc)*nu/hc);
      e = e*soft_sphere.epsilon*pow(soft_sphere.sigma, nu);
    }
  }

  return e;
}

/*****************************************************************************
 *
 *  soft_sphere_force
 *
 *  Return the magnitude of the 'soft-sphere' interaction force between
 *  two particles with (surface-surface) separation h.
 *
 ****************************************************************************/

double soft_sphere_force(const double h) {

  double f = 0.0;

  if (soft_sphere.on) {
    double hc = soft_sphere.cutoff;
    double nu = soft_sphere.nu;

    if (h > 0 && h < hc) {
      f = pow(h, -(nu+1)) - pow(hc, -(nu+1));
      f = f*soft_sphere.epsilon*pow(soft_sphere.sigma, nu)*nu;
    }
  }

  return f;
}

/*****************************************************************************
 *
 *  leonard_jones_energy
 *
 *  Return the potential at centre-centre separation r.
 *
 *****************************************************************************/

double leonard_jones_energy(const double r) {

  double e = 0.0;

  if (leonard_jones.on && r < leonard_jones.cutoff) {
    double sigmar = pow(leonard_jones.sigma/r, 6.0);
    e = 4.0*leonard_jones.epsilon*(sigmar*sigmar - sigmar);
  }

  return e;
}

/*****************************************************************************
 *
 *  hard_sphere_energy
 *
 *  Return energy of hard-sphere interaction at separation h.
 *
 *****************************************************************************/

double hard_sphere_energy(const double h) {

  double e = 0.0;

  if (h <= 0.0) e = ENERGY_HARD_SPHERE;

  return e;
}


/*****************************************************************************
 *
 *  yukawa_potential
 *
 *  Return Yukawa potenial as function of centre-centre separation r.
 *  u(r) = epsilon*exp(-kappa*r)/r
 *
 *****************************************************************************/

double yukawa_potential(double r) {

  double epsilon = yukawa.epsilon;
  double kappa   = yukawa.kappa;
  double rc      = yukawa.cutoff;
  double u0, u0_rc, du0_rc, e = 0.0;

  if (yukawa.on && r < rc) {
    u0 = epsilon*exp(-kappa*r)/r;
    u0_rc = epsilon*exp(-kappa*rc)/rc;
    du0_rc = -u0_rc*(kappa*rc + 1.0)/rc;

    e = u0 - u0_rc - (r - rc)*du0_rc;
  }

  return e;
}

/*****************************************************************************
 *
 *  yukawa_force
 *
 *  Return magnitude of the force for centre-centre separation r.
 *
 *****************************************************************************/

double yukawa_force(double r) {

  double epsilon = yukawa.epsilon;
  double kappa   = yukawa.kappa;
  double rc      = yukawa.cutoff;
  double u0, u0_rc, f = 0.0;

  if (yukawa.on && r < rc) {
    u0 = epsilon*exp(-kappa*r)/r;
    u0_rc = epsilon*exp(-kappa*rc)/rc;
    f = (u0*(kappa*r + 1.0)/r - u0_rc*(kappa*rc + 1.0)/rc)/r;
  }

  return f;
}

/*****************************************************************************
 *
 *  get_max_potential_range
 *
 *  Return the maximum range of potential cutoffs.
 *
 *****************************************************************************/

double get_max_potential_range() {

  double rmax = 0.0;

  rmax = dmax(rmax, soft_sphere.cutoff);
  rmax = dmax(rmax, leonard_jones.cutoff);
  rmax = dmax(rmax, yukawa.cutoff);

  return rmax;
}
