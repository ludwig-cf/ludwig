/*****************************************************************************
 *
 *  distribution_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util.h"
#include "io_harness.h"
#include "distribution_rt.h"

static int lb_rt_2d_kelvin_helmholtz(lb_t * lb, coords_t * cs);
static int lb_rt_2d_shear_wave(lb_t * lb, coords_t * cs);
static int lb_init_uniform(lb_t * lb, coords_t *cs, double rho0, double u0[3]);
static int lb_init_poiseuille(lb_t * lb, coords_t * cs, double rho0,
			      const double umax[3]);

/*****************************************************************************
 *
 *  lb_run_time
 *
 *****************************************************************************/

int lb_run_time(lb_t * lb, rt_t * rt, coords_t * cs) {

  int ndist;
  int nreduced;
  int io_grid[3] = {1, 1, 1};
  char string[FILENAME_MAX];
  char memory = ' '; 

  io_info_t * io_info = NULL;

  assert(lb);
  assert(rt);
  assert(cs);

  nreduced = 0;
  rt_string_parameter(rt, "reduced_halo", string, FILENAME_MAX);
  if (strcmp(string, "yes") == 0) nreduced = 1;

  rt_int_parameter_vector(rt, "distribution_io_grid", io_grid);
  io_info_create_with_grid(cs, io_grid, &io_info);
  lb_io_info_set(lb, io_info);

  rt_string_parameter(rt, "distribution_io_format_input", string,
		      FILENAME_MAX);

  /* Append R to the record if the model is the reverse implementation */ 
  if (lb_order(lb) == MODEL_R) memory = 'R';
  lb_ndist(lb, &ndist);

  info("\n");
  info("Lattice Boltzmann distributions\n");
  info("-------------------------------\n");

  info("Model:            d%dq%d %c\n", NDIM, NVEL, memory);
  info("SIMD vector len:  %d\n", SIMDVL);
  info("Number of sets:   %d\n", ndist);
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

  lb_init(lb);

  io_info_metadata_filestub_set(io_info, "dist");
  if (nreduced == 1) lb_halo_set(lb, LB_HALO_REDUCED);

  return 0;
}

/*****************************************************************************
 *
 *  lb_rt_initial_conditions
 *
 *****************************************************************************/

int lb_rt_initial_conditions(lb_t * lb, rt_t * rt, coords_t * cs,
			     physics_t * phys) {

  char key[FILENAME_MAX];
  double rho0;
  double u0[3] = {0.0, 0.0, 0.0};

  assert(lb);
  assert(rt);
  assert(cs);
  assert(phys);
  physics_rho0(&rho0);

  /* Default */

  lb_init_rest_f(lb, rho0);

  rt_string_parameter(rt, "distribution_initialisation", key, FILENAME_MAX);

  if (strcmp("2d_kelvin_helmholtz", key) == 0) {
    lb_rt_2d_kelvin_helmholtz(lb, cs);
  }

  if (strcmp("2d_shear_wave", key) == 0) {
    lb_rt_2d_shear_wave(lb, cs);
  }

  if (strcmp("3d_uniform_u", key) == 0) {
    rt_double_parameter_vector(rt, "distribution_uniform_u", u0);
    lb_init_uniform(lb, cs, rho0, u0);
  }

  if (strcmp("1d_poiseuille", key) == 0) {
    rt_double_parameter_vector(rt, "distribution_poiseuille_umax", u0);
    lb_init_poiseuille(lb, cs, rho0, u0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_rt_2d_kelvin_helmholtz
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

static int lb_rt_2d_kelvin_helmholtz(lb_t * lb, coords_t * cs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double rho = 1.0;
  double u0 = 0.04;
  double delta = 0.05;
  double kappa = 80.0;
  double u[3];
  double lmin[3];
  double ltot[3];

  double x, y;

  assert(lb);
  assert(cs);

  coords_lmin(cs, lmin);
  coords_ltot(cs, ltot);
  coords_nlocal(cs, nlocal);
  coords_nlocal_offset(cs, noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = (1.0*(noffset[X] + ic) - lmin[X])/ltot[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = (1.0*(noffset[Y] + jc) - lmin[Y])/ltot[Y];

      if (y >  0.5) u[X] = u0*tanh(kappa*(0.75 - y));
      if (y <= 0.5) u[X] = u0*tanh(kappa*(y - 0.25));
      u[Y] = u0*delta*sin(2.0*pi_*(x + 0.25));
      u[Z] = 0.0;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(cs, ic, jc, kc);
        lb_1st_moment_equilib_set(lb, index, rho, u);
      }
    }
  }

  info("\n");
  info("Initial distribution: 2d kelvin helmholtz\n");
  info("Velocity magnitude:   %14.7e\n", u0);
  info("Shear layer kappa:    %14.7e\n", kappa);
  info("Perturbation delta:   %14.7e\n", delta);
  info("\n");

  return 0;
}

/*****************************************************************************
 *
 *  lb_rt_2d_shear_wave
 *
 *  The system in (x, y) is scaled to 0 <= x,y < 1 and then
 *
 *      u_x = U sin( kappa y)
 *
 *      where U is a maximum velocity, kappa is the (inverse) width of
 *      the initial shear layer.
 *
 *****************************************************************************/

static int lb_rt_2d_shear_wave(lb_t * lb, coords_t * cs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double rho = 1.0;
  double u0 = 0.04;
  double kappa;
  double u[3];

  double lmin[3];
  double ltot[3];
  double y;

  assert(lb);
  assert(cs);

  coords_lmin(cs, lmin);
  coords_ltot(cs, ltot);
  coords_nlocal(cs, nlocal);
  coords_nlocal_offset(cs, noffset);

  kappa = 2.0*pi_;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = (1.0*(noffset[Y] + jc) - lmin[Y])/ltot[Y];

      u[X] = u0*sin(kappa * y);
      u[Y] = 0.0;
      u[Z] = 0.0;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(cs, ic, jc, kc);
        lb_1st_moment_equilib_set(lb, index, rho, u);
      }
    }
  }

  info("\n");
  info("Initial distribution: 2d shear wave\n");
  info("Velocity magnitude:   %14.7e\n", u0);
  info("Shear layer kappa:    %14.7e\n", kappa);
  info("\n");

  return 0;
}

/*****************************************************************************
 *
 *  lb_init_uniform
 *
 *  Set the initial distribution consistent with fixed (rho_0, u_0).
 *
 *****************************************************************************/

static int lb_init_uniform(lb_t * lb, coords_t * cs, double rho0,
			   double u0[3]) {

  int ic, jc, kc, index;
  int nlocal[3];

  assert(lb);
  assert(cs);

  coords_nlocal(cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(cs, ic, jc, kc);
	lb_1st_moment_equilib_set(lb, index, rho0, u0);

      }
    }
  }

  info("\n");
  info("Initial distribution: 3d uniform desnity/velocity\n");
  info("Density:              %14.7e\n", rho0);
  info("Velocity:             %14.7e %14.7e %14.7e\n", u0[X], u0[Y], u0[Z]);
  info("\n");

  return 0;
}

/*****************************************************************************
 *
 *  lb_init_poiseuille
 *
 *  A 1-d Poiseuille parabolic profile based on, e.g.,
 *    u(x) ~ umax[X] x (Lx - x)
 *
 *  The umax[3] should have only one non-zero component.
 *
 *****************************************************************************/

static int lb_init_poiseuille(lb_t * lb, coords_t * cs, double rho0,
			      const double umax[3]) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double u0[3];
  double x, y, z;
  double lmin[3];
  double ltot[3];

  assert(lb);
  assert(cs);

  coords_lmin(cs, lmin);
  coords_ltot(cs, ltot);
  coords_nlocal(cs, nlocal);
  coords_nlocal_offset(cs, noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {

    /* The - Lmin() in each direction centres the profile symmetrically,
     * and the 4/L^2 normalises to umax at centre */

    x = 1.0*(noffset[X] + ic) - lmin[X];
    u0[X] = umax[X]*x*(ltot[X] - x)*4.0/(ltot[X]*ltot[X]);

    for (jc = 1; jc <= nlocal[Y]; jc++) {

      y = 1.0*(noffset[Y] + jc) - lmin[Y];
      u0[Y] = umax[Y]*y*(ltot[Y] - y)*4.0/(ltot[Y]*ltot[Y]);

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	z = 1.0*(noffset[Z] + kc) - lmin[Z];
	u0[Z] = umax[Z]*z*(ltot[Z] - z)*4.0/(ltot[Z]*ltot[Z]);

	index = coords_index(cs, ic, jc, kc);
	lb_1st_moment_equilib_set(lb, index, rho0, u0);

      }
    }
  }

  info("\n");
  info("Initial distribution: 1d Poiseuille profile\n");
  info("Density:              %14.7e\n", rho0);
  info("Velocity (max):       %14.7e %14.7e %14.7e\n", umax[X], umax[Y],
       umax[Z]);
  info("\n");

  return 0;
}
