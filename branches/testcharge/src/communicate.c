
/* This has become a dumping ground for things that need to
 * be refactored */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"

#include "io_harness.h"
#include "physics.h"
#include "phi.h"
#include "model.h"
#include "ran.h"
#include "phi_lb_coupler.h"
#include "phi_cahn_hilliard.h"
#include "phi_stats.h"
#include "symmetric.h"

#include "communicate.h"

void MODEL_init( void ) {

  int i, j, k, ind;
  int N[3];
  int offset[3];
  int io_grid_default[3] = {1, 1, 1};
  int io_grid[3];
  int form;

  double   phi;
  double   phi0;
  char    value[BUFSIZ];
  double  noise0 = 0.1;   /* Initial noise amplitude    */

  io_info_t * iohandler;

  coords_nlocal(N);
  coords_nlocal_offset(offset);

  /* phi */

  RUN_get_int_parameter_vector("default_io_grid", io_grid_default);
  for (i = 0; i < 3; i++) {
    io_grid[i] = io_grid_default[i];
  }
  RUN_get_int_parameter_vector("phi_io_grid", io_grid);

  form = IO_FORMAT_DEFAULT;
  ind = RUN_get_string_parameter("phi_format", value, BUFSIZ);
  if (ind != 0 && strcmp(value, "ASCII") == 0) {
    form = IO_FORMAT_ASCII;
    info("Setting phi I/O format to ASCII\n");
  }

  phi_init();
  phi_init_io_info(io_grid, form, form);

  info("\n");
  info("Other I/O\n");
  info("---------\n");

  info("phi I/O format:             %s\n", value);
  info("phi I/O decomposition:      %d %d %d\n", io_grid[0], io_grid[1],
       io_grid[2]);

  /*
   * A number of options are offered to start a simulation:
   * 1. Read distribution functions site from file, then simply calculate
   *    all other properties (rho/phi/gradients/velocities)
   * 6. set rho/phi/velocity to default values, automatically set etc.
   */

  phi0 = get_phi0();
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

    ind = RUN_get_string_parameter("phi_initialisation", value, BUFSIZ);

    if (ind != 0 && strcmp(value, "block") == 0) {
      info("Initialisng phi as block\n");
      phi_init_block(symmetric_interfacial_width());
    }

    if (ind != 0 && strcmp(value, "bath") == 0) {
      info("Initialising phi for bath\n");
      phi_init_bath();
    }

    /* Assumes symmetric free energy */
    if (ind != 0 && strcmp(value, "drop") == 0) {
      info("Initialising droplet\n");
      phi_lb_init_drop(0.4*L(X), symmetric_interfacial_width());
    }

    if (ind != 0 && strcmp(value, "from_file") == 0) {
      phi_io_info(&iohandler);
      info("Initial order parameter requested from file\n");
      info("Reading phi from serial file\n");
      io_info_set_processor_independent(iohandler);
      io_read("phi-init", iohandler);
      io_info_set_processor_dependent(iohandler);

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

}
