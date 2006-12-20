/*****************************************************************************
 *
 *  physics.c
 *
 *  Basic physical quantities for fluid.
 *
 *  $Id: physics.c,v 1.1 2006-12-20 16:53:19 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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
