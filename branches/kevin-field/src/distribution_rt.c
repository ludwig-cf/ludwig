/*****************************************************************************
 *
 *  distribution_rt.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "io_harness.h"
#include "distribution_rt.h"

static void distribution_rt_2d_kelvin_helmholtz(void);

/*****************************************************************************
 *
 *  distribution_run_time
 *
 *****************************************************************************/

void distribution_run_time(void) {

  int nreduced;
  int io_grid[3] = {1, 1, 1};
  char string[FILENAME_MAX];
  char memory = ' '; 

  struct io_info_t * io_info;

  RUN_get_string_parameter("free_energy", string, FILENAME_MAX);
  if (strcmp(string, "symmetric_lb") == 0) distribution_ndist_set(2);

  nreduced = 0;
  RUN_get_string_parameter("reduced_halo", string, FILENAME_MAX);
  if (strcmp(string, "yes") == 0) nreduced = 1;

  RUN_get_int_parameter_vector("distribution_io_grid", io_grid);
  io_info = io_info_create_with_grid(io_grid);
  distribution_io_info_set(io_info);

  RUN_get_string_parameter("distribution_io_format_input", string,
			   FILENAME_MAX);

  /* Append R to the record if the model is the reverse implementation */ 
  if (distribution_order() == MODEL_R) memory = 'R';

  info("\n");
  info("Lattice Boltzmann distributions\n");
  info("-------------------------------\n");

  info("Model:            d%dq%d %c\n", NDIM, NVEL, memory);
  info("SIMD vector len:  %d\n", SIMDVL);
  info("Number of sets:   %d\n", distribution_ndist());
  info("Halo type:        %s\n", (nreduced == 1) ? "reduced" : "full");

  if (strcmp("BINARY_SERIAL", string) == 0) {
    info("Input format:     binary single serial file\n");
    io_info_set_processor_independent(io_info);
  }
  else {
    info("Input format:     binary\n");
  }

  info("Output format:    binary\n");
  info("I/O grid:         %d %d %d\n", io_grid[0], io_grid[1], io_grid[2]);

  distribution_init();

  io_write_metadata("dist", distribution_io_info());
  if (nreduced == 1) distribution_halo_set_reduced();

  return;
}

/*****************************************************************************
 *
 *  distribution_rt_initial_conditions
 *
 *****************************************************************************/

void distribution_rt_initial_conditions(void) {

  char key[FILENAME_MAX];

  RUN_get_string_parameter("distribution_initialisation", key, FILENAME_MAX);

  if (strcmp("2d_kelvin_helmholtz", key) == 0) {
    distribution_rt_2d_kelvin_helmholtz();
  }

  return;
}

/*****************************************************************************
 *
 *  distribution_rt_2d_kelvin_helmholtz
 *
 *  A test problem put forward by Brown and Minion
 *  J. Comp. Phys. \textbf{122} 165--183 (1995)
 *
 *  The system in (x, y) is scaled to 0 <= x,y < 1 and then
 *
 *      u_x = U tanh( kappa (y - 1/4))   y <= 1/2
 *      u_x = U tanh( kappa (3/4 - y))   y > 1/2
 *
 *      u_y = delta U sin(2 pi (x + 1/4))
 *
 *      where U is a maximum velocity, kappa is the (inverse) width of
 *      the initial shear layer, and delta is a perturbation.
 *
 *   The (x + 1/4) is just to move the area of interest into the
 *   middle of the system.
 *   
 *   For example, a 'thin' shear layer might have kappa = 80 and delta = 0.05
 *   with Re = rho U L / eta = 10^4.
 *
 *   U must satisfy the LB mach number constraint. See also Dellar
 *   J. Comp. Phys. \textbf{190} 351--370 (2003).
 *
 *****************************************************************************/

static void distribution_rt_2d_kelvin_helmholtz(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double rho = 1.0;
  double u0 = 0.04;
  double delta = 0.05;
  double kappa = 80.0;
  double u[3];

  double x, y;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = (1.0*(noffset[X] + ic) - Lmin(X))/L(X);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = (1.0*(noffset[Y] + jc) - Lmin(Y))/L(Y);

      if (y >  0.5) u[X] = u0*tanh(kappa*(0.75 - y));
      if (y <= 0.5) u[X] = u0*tanh(kappa*(y - 0.25));
      u[Y] = u0*delta*sin(2.0*pi_*(x + 0.25));
      u[Z] = 0.0;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
        distribution_rho_u_set_equilibrium(index, rho, u);
      }
    }
  }

  info("\n");
  info("Initial distribution: 2d kelvin helmholtz\n");
  info("Velocity magnitude:   %14.7e\n", u0);
  info("Shear layer kappa:    %14.7e\n", kappa);
  info("Perturbation delta:   %14.7e\n", delta);
  info("\n");

  return;
}
