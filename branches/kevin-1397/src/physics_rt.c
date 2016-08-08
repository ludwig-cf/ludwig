/*****************************************************************************
 *
 *  physics_rt.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "physics.h"

/*****************************************************************************
 *
 *  physics_info
 *
 *****************************************************************************/

__host__ int physics_info(pe_t * pe, physics_t * phys) {

  double rho0;
  double eta1, eta2;
  double kt;
  double f0[3], e0[3], b0[3];
  double e0_frequency;

  assert(pe);
  assert(phys);

  physics_rho0(phys, &rho0);
  physics_eta_shear(phys, &eta1);
  physics_eta_bulk(phys, &eta2);
  physics_kt(phys, &kt);
  physics_fbody(phys, f0);
  physics_e0(phys, e0);
  physics_e0_frequency(phys, &e0_frequency);
  physics_b0(phys, b0);

  pe_info(pe, "\n");
  pe_info(pe, "System properties\n");
  pe_info(pe, "----------------\n");
  pe_info(pe, "Mean fluid density:          %12.5e\n", rho0);
  pe_info(pe, "Shear viscosity              %12.5e\n", eta1);
  pe_info(pe, "Bulk viscosity               %12.5e\n", eta2);
  pe_info(pe, "Temperature                  %12.5e\n", kt);
  pe_info(pe, "External body force density  %12.5e %12.5e %12.5e\n",
	  f0[0], f0[1], f0[2]);
  pe_info(pe, "External E-field amplitude   %12.5e %12.5e %12.5e\n",
	  e0[0], e0[1], e0[2]);
  pe_info(pe, "External E-field frequency   %12.5e\n", e0_frequency);
  pe_info(pe, "External magnetic field      %12.5e %12.5e %12.5e\n",
	  b0[0], b0[1], b0[2]);

  return 0;
}

/*****************************************************************************
 *
 *  physics_init_rt
 *
 *****************************************************************************/

__host__ int physics_init_rt(rt_t * rt, physics_t * phys) {

  double kt;
  double eta;
  double rho0;
  double phi0;
  double vector[3];
  double frequency;

  assert(phys);

  /* Bulk viscosity defaults to shear value */

  if (rt_double_parameter(rt, "viscosity", &eta)) {
    physics_eta_shear_set(phys, eta);
    physics_eta_bulk_set(phys, eta);
  }

  if (rt_double_parameter(rt, "viscosity_bulk", &eta)) {
    physics_eta_bulk_set(phys, eta);
  }

  if (rt_double_parameter(rt, "temperature", &kt)) {
    physics_kt_set(phys, kt);
  }

  if (rt_double_parameter(rt, "fluid_rho0", &rho0)) {
    physics_rho0_set(phys, rho0);
  }

  if (rt_double_parameter(rt, "phi0", &phi0)) {
    physics_phi0_set(phys, phi0);
  }

  if (rt_double_parameter_vector(rt, "force", vector)) {
    physics_fbody_set(phys, vector);
  }

  if (rt_double_parameter_vector(rt, "magnetic_b0", vector)) {
    physics_b0_set(phys, vector);
  }

  if (rt_double_parameter_vector(rt, "electric_e0", vector)) {
    physics_e0_set(phys, vector);
  }

  if (rt_double_parameter(rt, "electric_e0_frequency", &frequency)) {
    physics_e0_frequency_set(phys, frequency);
  }

  return 0;
}
