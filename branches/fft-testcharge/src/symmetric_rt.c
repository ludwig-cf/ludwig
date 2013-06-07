/****************************************************************************
 *
 *  symmetric_rt.c
 *
 *  Run time initialisation for the symmetric phi^4 free energy.
 *
 *  $Id$
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
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "free_energy.h"
#include "symmetric.h"

#include "physics.h"
#include "ran.h"
#include "util.h"

static int symmetric_init_drop(field_t * fphi, double radius, double xi0);
static int symmetric_init_block(field_t * phi, double xi0);
static int symmetric_init_bath(field_t * phi);

/****************************************************************************
 *
 *  symmetric_run_time
 *
 ****************************************************************************/

void symmetric_run_time(void) {

  int n;
  double a;
  double b;
  double kappa;

  info("Symmetric phi^4 free energy selected.\n");
  info("\n");

  /* Parameters */

  n = RUN_get_double_parameter("A", &a);
  n = RUN_get_double_parameter("B", &b);
  n = RUN_get_double_parameter("K", &kappa);

  info("Parameters:\n");
  info("Bulk parameter A      = %12.5e\n", a);
  info("Bulk parameter B      = %12.5e\n", b);
  info("Surface penalty kappa = %12.5e\n", kappa);

  symmetric_free_energy_parameters_set(a, b, kappa);

  info("Surface tension       = %12.5e\n", symmetric_interfacial_tension());
  info("Interfacial width     = %12.5e\n", symmetric_interfacial_width());

  /* Set free energy function pointers. */

  fe_density_set(symmetric_free_energy_density);
  fe_chemical_potential_set(symmetric_chemical_potential);
  fe_isotropic_pressure_set(symmetric_isotropic_pressure);
  fe_chemical_stress_set(symmetric_chemical_stress);

  return;
}

/*****************************************************************************
 *
 *  symmetric_rt_initial_conditions
 *
 *****************************************************************************/

int symmetric_rt_initial_conditions(field_t * phi) {

  int p;
  int ic, jc, kc, index;
  int ntotal[3], nlocal[3];
  int offset[3];
  double phi0, phi1;
  double noise0 = 0.1; /* Default value. */
  char value[BUFSIZ];

  assert(phi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);
  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  phi0 = get_phi0();
  RUN_get_double_parameter("noise", &noise0);

  /* Default initialisation (always?) Note serial nature of this,
   * which could be replaced. */

  for (ic = 1; ic <= ntotal[X]; ic++) {
    for (jc = 1; jc <= ntotal[Y]; jc++) {
      for (kc = 1; kc <= ntotal[Z]; kc++) {

	phi1 = phi0 + noise0*(ran_serial_uniform() - 0.5);

	/* For computation with single fluid and no noise */
	/* Only set values if within local box */
	if ( (ic > offset[X]) && (ic <= offset[X] + nlocal[X]) &&
	     (jc > offset[Y]) && (jc <= offset[Y] + nlocal[Y]) &&
	     (kc > offset[Z]) && (kc <= offset[Z] + nlocal[Z]) ) {

	    index = coords_index(ic-offset[X], jc-offset[Y], kc-offset[Z]);
	    field_scalar_set(phi, index, phi1);
	    
	}
      }
    }
  }

  p = RUN_get_string_parameter("phi_initialisation", value, BUFSIZ);

  if (p != 0 && strcmp(value, "block") == 0) {
    info("Initialisng phi as block\n");
    symmetric_init_block(phi, symmetric_interfacial_width());
  }

  if (p != 0 && strcmp(value, "bath") == 0) {
    info("Initialising phi for bath\n");
    symmetric_init_bath(phi);
  }

  if (p != 0 && strcmp(value, "drop") == 0) {
    info("Initialising droplet\n");
    /* Could do with a drop radius */
    symmetric_init_drop(phi, 0.4*L(X), symmetric_interfacial_width());
  }

  if (p != 0 && strcmp(value, "from_file") == 0) {
    info("Initial order parameter requested from file\n");
    info("Reading phi from serial file\n");

    fatal("Not reading from file\n");
    /* need to do something! */
  }

  return 0;
}

/*****************************************************************************
 *
 *  symmetric_init_drop
 *
 *****************************************************************************/

static int symmetric_init_drop(field_t * fphi, double radius, double xi0) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double centre[3];
  double phi, r, rxi0;

  assert(fphi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  rxi0 = 1.0/xi0;

  centre[X] = 0.5*L(X);
  centre[Y] = 0.5*L(Y);
  centre[Z] = 0.5*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        position[X] = 1.0*(noffset[X] + ic) - centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - centre[Z];

        r = sqrt(dot_product(position, position));

        phi = tanh(rxi0*(r - radius));
	field_scalar_set(fphi, index, phi);
      }
    }
  }

  return 0;
}


/*****************************************************************************
 *
 * symmetric_init_block
 *
 *  Initialise two blocks with interfaces at z = Lz/4 and z = 3Lz/4.
 *
 *****************************************************************************/

static int symmetric_init_block(field_t * phi, double xi0) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z1, z2;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z1 = 0.25*L(Z);
  z2 = 0.75*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;

	if (z > 0.5*L(Z)) {
	  phi0 = tanh((z-z2)/xi0);
	}
	else {
	  phi0 = -tanh((z-z1)/xi0);
	}

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  symmetric_init_bath
 *
 *  Initialise one interface at z = Lz/8. This is inended for
 *  capillary rise in systems with z not periodic.
 *
 *****************************************************************************/

static int symmetric_init_bath(field_t * phi) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z0;
  double phi0, xi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z0 = 0.25*L(Z);
  xi0 = 1.13;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;
	phi0 = tanh((z-z0)/xi0);

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}
