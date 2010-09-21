
/* This has become a dumping ground for things that need to
 * be refactored */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"

#include "io_harness.h"
#include "blue_phase.h"
#include "physics.h"
#include "phi.h"
#include "model.h"
#include "lattice.h"
#include "ran.h"
#include "phi_lb_coupler.h"
#include "phi_cahn_hilliard.h"
#include "phi_stats.h"
#include "symmetric.h"
#include "control.h"
#include "colloids_Q_tensor.h"

#include "communicate.h"

void MODEL_init( void ) {

  int     i,j,k,ind;
  int     N[3];
  int     offset[3];
  double   phi;
  double   phi0;
  char     filename[FILENAME_MAX];
  double  noise0 = 0.1;   /* Initial noise amplitude    */

  phi0 = get_phi0();

  coords_nlocal(N);
  coords_nlocal_offset(offset);

  /* Now setup the rest of the simulation */


  phi_init();

  ind = RUN_get_string_parameter("phi_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_phi);
    info("Setting phi I/O format to ASCII\n");
  }

  scalar_q_io_init();

  ind = RUN_get_string_parameter("qs_dir_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_scalar_q_);
    info("Setting qs_dir I/O format to ASCII\n");
  }

  hydrodynamics_init();
  
  ind = RUN_get_string_parameter("vel_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_velocity_);
    info("Setting velocity I/O format to ASCII\n"); 
  }

  /*
   * A number of options are offered to start a simulation:
   * 1. Read distribution functions site from file, then simply calculate
   *    all other properties (rho/phi/gradients/velocities)
   * 6. set rho/phi/velocity to default values, automatically set etc.
   */

  RUN_get_double_parameter("noise", &noise0);

  if (phi_nop()){

    /* Initialise phi with initial value +- noise */

    for(i=1; i<=N_total(X); i++)
      for(j=1; j<=N_total(Y); j++)
	for(k=1; k<=N_total(Z); k++) {

	  phi = phi0 + noise0*(ran_serial_uniform() - 0.5);

	  /* For computation with single fluid and no noise */
	  /* Only set values if within local box */
	  if((i>offset[X]) && (i<=offset[X] + N[X]) &&
	     (j>offset[Y]) && (j<=offset[Y] + N[Y]) &&
	     (k>offset[Z]) && (k<=offset[Z] + N[Z]))
	    {
	      ind = coords_index(i-offset[X], j-offset[Y], k-offset[Z]);

	      phi_lb_coupler_phi_set(ind, phi);
	    }
	}
  }

  if (phi_nop()) {

    ind = RUN_get_string_parameter("phi_initialisation", filename,
				   FILENAME_MAX);

    if (ind != 0 && strcmp(filename, "block") == 0) {
      info("Initialisng phi as block\n");
      phi_init_block(symmetric_interfacial_width());
    }

    if (ind != 0 && strcmp(filename, "bath") == 0) {
      info("Initialising phi for bath\n");
      phi_init_bath();
    }

    /* Assumes symmetric free energy */
    if (ind != 0 && strcmp(filename, "drop") == 0) {
      info("Initialising droplet\n");
      phi_lb_init_drop(0.4*L(X), symmetric_interfacial_width());
    }

    if (ind != 0 && strcmp(filename, "from_file") == 0) {
      info("Initial order parameter requested from file\n");
      info("Reading phi from serial file\n");
      io_info_set_processor_independent(io_info_phi);
      io_read("phi-init", io_info_phi);
      io_info_set_processor_dependent(io_info_phi);

      if (distribution_ndist() > 1) {
	/* Set the distribution from initial phi */
	for (i = 1; i <= N[X]; i++) {
	  for (j = 1; j <= N[Y]; j++) {
	    for (k = 1; k <= N[Z]; k++) {
	    
	      ind = coords_index(i, j, k);
	      phi = phi_get_phi_site(ind);
	      distribution_zeroth_moment_set_equilibrium(ind, 1, phi);
	    }
	  }
	}
      }
    }
  }

  /* BLUEPHASE */
  blue_phase_twist_init(0.3333333);
  /*blue_set_random_q_init(50, 78, 50, 78, 50, 78);*/
  /* blue_phase_O8M_init(-0.2);*/
  /* blue_phase_O2_init(0.3);*/
}
