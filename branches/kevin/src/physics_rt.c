/*****************************************************************************
 *
 *  physics_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2013 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "physics.h"
#include "physics_rt.h"

/*****************************************************************************
 *
 *  physics_info
 *
 *****************************************************************************/

int physics_info(physics_t * phys) {

  double rho0;
  double eta1, eta2;
  double kt;
  double f0[3], e0[3], b0[3];

  physics_rho0(&rho0);
  physics_eta_shear(&eta1);
  physics_eta_bulk(&eta2);
  physics_kt(&kt);
  physics_fbody(f0);
  physics_e0(e0);
  physics_b0(b0);

  info("\n");
  info("System properties\n");
  info("----------------\n");
  info("Mean fluid density:          %12.5e\n", rho0);
  info("Shear viscosity              %12.5e\n", eta1);
  info("Bulk viscosity               %12.5e\n", eta2);
  info("Temperature                  %12.5e\n", kt);
  info("External body force density  %12.5e %12.5e %12.5e\n",
       f0[0], f0[1], f0[2]);
  info("External electric field      %12.5e %12.5e %12.5e\n",
       e0[0], e0[1], e0[2]);
  info("External magnetic field      %12.5e %12.5e %12.5e\n",
       b0[0], b0[1], b0[2]);

  return 0;
}

/*****************************************************************************
 *
 *  physics_init_rt
 *
 *****************************************************************************/

int physics_init_rt(rt_t * rt, physics_t * phys) {

  double kt;
  double eta;
  double rho0;
  double phi0;
  double vector[3];

  assert(rt);

  /* Bulk viscosity defaults to shear value */

  if (rt_double_parameter(rt, "viscosity", &eta)) {
    physics_eta_shear_set(eta);
    physics_eta_bulk_set(eta);
  }

  if (rt_double_parameter(rt, "viscosity_bulk", &eta)) {
    physics_eta_bulk_set(eta);
  }

  if (rt_double_parameter(rt, "temperature", &kt)) {
    physics_kt_set(kt);
  }

  if (rt_double_parameter(rt, "fluid_rho0", &rho0)) {
    physics_rho0_set(rho0);
  }

  if (rt_double_parameter(rt, "phi0", &phi0)) {
    physics_phi0_set(phi0);
  }

  if (rt_double_parameter_vector(rt, "force", vector)) {
    physics_fbody_set(vector);
  }

  if (rt_double_parameter_vector(rt, "magnetic_b0", vector)) {
    physics_b0_set(vector);
  }

  if (rt_double_parameter_vector(rt, "electric_e0", vector)) {
    physics_e0_set(vector);
  }

  return 0;
}
