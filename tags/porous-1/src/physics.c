/*****************************************************************************
 *
 *  physics.c
 *
 *  Basic physical quantities for fluid.
 *
 *  $Id: physics.c,v 1.2.4.1 2008-03-21 09:20:18 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include "pe.h"
#include "runtime.h"
#include "physics.h"


static double eta_shear = 1.0/6.0;   /* Shear viscosity */
static double eta_bulk  = 1.0/6.0;   /* Bulk viscosity */
static double kT_ = 0.0;             /* Isothermal "temperature" */

static double rho0 = 1.0;            /* Average simulation density */
static double phi0 = 0.0;            /* Average order parameter    */

static double g_[3] = {0.0, 0.0, 0.0}; /* External gravitational force */

/*****************************************************************************
 *
 *  init_physics
 *
 *  Set physical parameters
 *
 *****************************************************************************/

void init_physics() {

  int p;

  p = RUN_get_double_parameter("viscosity", &eta_shear);
  eta_bulk = eta_shear;

  p = RUN_get_double_parameter("viscosity_bulk", &eta_bulk);
  p = RUN_get_double_parameter("phi0", &phi0);
  p = RUN_get_double_parameter("rho0", &rho0);

  info("\nExternal gravitational force\n");
  p = RUN_get_double_parameter_vector("colloid_gravity", g_);
  info("[%s] gravity = %g %g %g\n", (p == 0) ? "Default" : "User   ",
       g_[0], g_[1], g_[2]);

  return;
}


/*****************************************************************************
 *
 *  get_eta_shear
 *
 *  Return the shear viscosity.
 *
 *****************************************************************************/

double get_eta_shear() {

  return eta_shear;
}

void set_eta(double eta) {
  eta_shear = eta;
  eta_bulk = eta;
  return;
}

/*****************************************************************************
 *
 *  get_eta_bulk
 *
 *  Return the bulk viscosity
 *
 *****************************************************************************/

double get_eta_bulk() {

  return eta_bulk;
}

/*****************************************************************************
 *
 *  get_kT
 *
 *  Access function for the isothermal temperature.
 *
 *****************************************************************************/

double get_kT() {

  return kT_;
}

/*****************************************************************************
 *
 *  set_kT
 *
 *****************************************************************************/

void set_kT(const double t) {

  if (t < 0.0) fatal("Trying to set kT < 0 (=%f)\n"); 
  kT_ = t;

  return;
}

/*****************************************************************************
 *
 *  get_rho0
 *
 *  Access function for the mean fluid density.
 *
 *****************************************************************************/

double get_rho0() {

  return rho0;
}

/*****************************************************************************
 *
 *  get_phi0
 *
 *  Access function for the mean order parameter.
 *
 *****************************************************************************/

double get_phi0() {

  return phi0;
}

/*****************************************************************************
 *
 *  get_gravity
 *
 *****************************************************************************/

void get_gravity(double gravity[3]) {

  int i;

  for (i = 0; i < 3; i++) gravity[i] = g_[i];

  return;
}
