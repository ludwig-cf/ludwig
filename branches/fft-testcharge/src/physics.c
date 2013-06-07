/*****************************************************************************
 *
 *  physics.c
 *
 *  Basic physical quantities for fluid.
 *
 *  $Id: physics.c,v 1.4 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include "pe.h"
#include "runtime.h"
#include "physics.h"


static double eta_shear = 1.0/6.0;   /* Shear viscosity */
static double eta_bulk  = 1.0/6.0;   /* Bulk viscosity */
static double kt_ = 0.0;             /* Isothermal "temperature" */

static double rho0 = 1.0;            /* Average simulation density */
static double phi0 = 0.0;            /* Average order parameter    */

static double bodyforce_[3] = {0.0, 0.0, 0.0};

/*****************************************************************************
 *
 *  init_physics
 *
 *  Set physical parameters
 *
 *****************************************************************************/

void init_physics() {

  int p;
  double vector[3];

  p = RUN_get_double_parameter("viscosity", &eta_shear);
  eta_bulk = eta_shear;

  p = RUN_get_double_parameter("viscosity_bulk", &eta_bulk);
  p = RUN_get_double_parameter("phi0", &phi0);

  p = RUN_get_double_parameter_vector("force", vector);

  if (p != 0) {
    bodyforce_[0] = vector[0];
    bodyforce_[1] = vector[1];
    bodyforce_[2] = vector[2];
  }

  p = RUN_get_double_parameter("temperature", &kt_);

  info("\n");
  info("Fluid properties\n");
  info("----------------\n");
  info("Mean density:      %12.5e\n", rho0);
  info("Shear viscosity    %12.5e\n", eta_shear);
  info("Bulk viscosity     %12.5e\n", eta_bulk);
  info("Temperature        %12.5e\n", kt_);
  info("Body force density %12.5e %12.5e %12.5e\n", 
       bodyforce_[0], bodyforce_[1], bodyforce_[2]); 

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
 *  get_kt
 *
 *  Access function for the isothermal temperature.
 *
 *****************************************************************************/

double get_kT() {

  return kt_;
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
 *  fluid_body_force
 *
 *****************************************************************************/

void fluid_body_force(double f[3]) {

  f[0] = bodyforce_[0];
  f[1] = bodyforce_[1];
  f[2] = bodyforce_[2];

  return;
}

/*****************************************************************************
 *
 *  fluid_body_force_set
 *
 *****************************************************************************/

void fluid_body_force_set(const double f[3]) {

  bodyforce_[0] = f[0];
  bodyforce_[1] = f[1];
  bodyforce_[2] = f[2];

  return;
}

/*****************************************************************************
 *
 *  fluid_kt
 *
 *****************************************************************************/

double fluid_kt(void) {

  return kt_;
}
