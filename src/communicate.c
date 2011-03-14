
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

  int i, j, k, ind;
  int N[3];
  int offset[3];
  int io_grid_default[3] = {1, 1, 1};
  int io_grid[3];

  double   phi;
  double   phi0;
  char     filename[FILENAME_MAX];
  double  noise0 = 0.1;   /* Initial noise amplitude    */

  struct io_info_t * io_info;

  coords_nlocal(N);
  coords_nlocal_offset(offset);

  /* Now setup the rest of the simulation */

  RUN_get_int_parameter_vector("default_io_grid", io_grid_default);

  for (i = 0; i < 3; i++) {
    io_grid[i] = io_grid_default[i];
  }
  RUN_get_int_parameter_vector("phi_io_grid", io_grid);

  io_info = io_info_create_with_grid(io_grid);
  phi_io_info_set(io_info);

  phi0 = get_phi0();

  phi_init();

  ind = RUN_get_string_parameter("phi_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_phi);
    info("Setting phi I/O format to ASCII\n");
  }


  for (i = 0; i < 3; i++) {
    io_grid[i] = io_grid_default[i];
  }
  RUN_get_int_parameter_vector("qs_dir_io_grid", io_grid);

  io_info = io_info_create_with_grid(io_grid);
  scalar_q_io_info_set(io_info);

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

  if (phi_nop() == 1) {

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

  if (phi_nop() == 5) {

    /* BLUEPHASE initialisation */
    RUN_get_string_parameter("lc_q_initialisation", filename, FILENAME_MAX);
    RUN_get_double_parameter("lc_q_init_amplitude", &phi0);

    if (strcmp(filename, "twist") == 0) {
      info("Initialising Q_ab to cholesteric (amplitude %14.7e)\n", phi0);
      blue_phase_twist_init(phi0);
    }

    if (strcmp(filename, "o8m") == 0) {
      info("Initialising Q_ab using O8M (amplitude %14.7e)\n", phi0);
      blue_phase_O8M_init(phi0);
    }

    if (strcmp(filename, "o2") == 0) {
      info("Initialising Q_ab using O2 (amplitude %14.7e)\n", phi0);
      blue_phase_O2_init(phi0);
    }

    RUN_get_string_parameter("lc_anchoring", filename, FILENAME_MAX);

    if (strcmp(filename, "normal") == 0) {
      info("Using normal anchoring boundary conditions\n");
      colloids_q_tensor_anchoring_set(ANCHORING_NORMAL);
    }

    if (strcmp(filename, "planar") == 0) {
      info("Using planar anchoring boundary conditions\n");
      colloids_q_tensor_anchoring_set(ANCHORING_PLANAR);
    }

    if (strcmp(filename, "planar") == 0 ||  strcmp(filename, "normal") == 0) {
      phi0 = 0.0;
      RUN_get_double_parameter("lc_anchoring_strength", &phi0);
      colloids_q_tensor_w_set(phi0);
      info("Anchoring strength w = %14.7e\n", colloids_q_tensor_w());
    }
    
  }

}
