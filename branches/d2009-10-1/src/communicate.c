
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
#include "lattice.h"
#include "ran.h"
#include "phi_lb_coupler.h"
#include "phi_cahn_hilliard.h"
#include "control.h"

#include "utilities.h"
#include "communicate.h"

static char         input_config[256] = "EMPTY";
static char         output_config[256] = "config.out";


void MODEL_init( void ) {

  int     i,j,k,ind;
  int     N[3];
  int     offset[3];
  double   phi;
  double   phi0;
  char     filename[FILENAME_MAX];
  double  noise0 = 0.1;   /* Initial noise amplitude    */

  phi0 = get_phi0();

  get_N_local(N);
  get_N_offset(offset);

  /* Now setup the rest of the simulation */

  /* Order parameter */

  ind = RUN_get_string_parameter("phi_finite_difference", filename,
				 FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "yes") == 0) {
    phi_set_finite_difference();
    info("Switching order parameter to finite difference\n");
    distribution_ndist_set(1);
  }
  else {
    info("Order parameter is via lattice Boltzmann\n");
    distribution_ndist_set(2);
    /* Only Cahn-Hilliard (binary) by this method */
    i = RUN_get_string_parameter("free_energy", filename, FILENAME_MAX);
    if (i == 1 && strcmp(filename, "symmetric") != 0) {
      fatal("Trying to run full LB: check free energy?\n");
    }
  }

  /* Distributions */

  init_site();

  phi_init();

  ind = RUN_get_string_parameter("phi_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_phi);
    info("Setting phi I/O format to ASCII\n");
  }

  ind = RUN_get_string_parameter("reduced_halo", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "yes") == 0) {
    info("\nUsing reduced halos\n\n");
    distribution_halo_set_reduced();
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

  /* Option 1: read distribution functions from file */

  ind = RUN_get_string_parameter("input_config", filename, FILENAME_MAX);

  if (ind != 0) {

    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), filename);

    /* Read distribution functions - sets both */
    io_read(filename, io_info_distribution_);
  } 
  else if (phi_nop()){
      /* 
       * Provides same initial conditions for rho/phi regardless of the
       * decomposition. 
       */
      
      /* Initialise lattice with constant density */
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
		ind = get_site_index(i-offset[X], j-offset[Y], k-offset[Z]);

		phi_lb_coupler_phi_set(ind, phi);
	      }
	  }
  }

  /* BLUEPHASE */
  /* blue_phase_twist_init(0.3333333);*/
  /* blue_phase_O8M_init(-0.2);*/
  /* blue_phase_O2_init(0.3);*/

  /* I/O old COM_init stuff */
  strcpy(output_config, "config.out");

  RUN_get_string_parameter("input_config", input_config, 256);
  RUN_get_string_parameter("output_config", output_config, 256);


}

/*****************************************************************************
 *
 *  get_output_config_filename
 *
 *  Return conifguration file name stub for time "step"
 *
 *****************************************************************************/

void get_output_config_filename(char * stub, const int step) {

  sprintf(stub, "%s%8.8d", output_config, step);

  return;
}

/*****************************************************************************
 *
 *  get_input_config_filename
 *
 *  Return configuration file name (where for historical reasons,
 *  input_config holds the whole name). "step is ignored.
 *
 *****************************************************************************/

void get_input_config_filename(char * stub, const int step) {

  /* But use this... */
  sprintf(stub, "%s", input_config);

  return;
}
