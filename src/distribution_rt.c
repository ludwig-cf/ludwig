/*****************************************************************************
 *
 *  distribution_rt.c
 *
 *  Wrangle run time input to control model distribution behaviour.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "runtime.h"
#include "coords.h"
#include "io_info_args_rt.h"
#include "distribution_rt.h"

struct kh_2d_param_s {
  double rho0;          /* Mean density */
  double u0;            /* Maximum velocity one dimension */
  double delta;         /* Initial perturbation */
  double kappa;         /* Shear layer "thinckness" (an inverse length) */
};

typedef struct kh_2d_param_s kh_2d_param_t;

static int lb_rt_2d_kelvin_helmholtz(lb_t * lb, const kh_2d_param_t * kh);
static int lb_rt_2d_shear_wave(lb_t * lb);
static int lb_init_uniform(lb_t * lb, double rho0, double u0[3]);
static int lb_init_poiseuille(lb_t * lb, double rho0,
			      const double umax[3]);

static int lb_ndist_rt(rt_t * rt);

/*****************************************************************************
 *
 *  lb_ndist_rt
 *
 *  Determine the number of distributions required at run time.
 *
 *****************************************************************************/

int lb_ndist_rt(rt_t * rt) {

  int ndist = 1;

  assert(rt);

  {
    /* The only exception ... */
    char description[BUFSIZ] = {0};
    rt_string_parameter(rt, "free_energy", description, BUFSIZ);
    if (strcmp(description, "symmetric_lb") == 0) ndist = 2;
  }

  return ndist;
}

/*****************************************************************************
 *
 *  lb_run_time
 *
 *  Some old keys are trapped with advice before accepting new keys.
 *
 *  Miscellaneous input keys
 *  I/O
 *
 *  If no options at all are specfied in the input, io_options_t information
 *  defaults to those values in io_options.h.
 *
 *  Obtain io_options_t information for distributions "lb_io" from input
 *
 *    -> "default_io_mode"     "single/multiple"    input and output
 *    -> "default_io_format"   "ascii/binary"       input and output
 *    -> "default_io_grid"     "X_Y_Z"              input and output
 *    -> "default_io_freq"     "123456789"          output only
 *
 *    which may be overridden by specific requests...
 *
 *    -> "lb_input_io_mode"    "single/multiple"
 *    -> "lb_input_io_format"  "ascii/binary"
 *
 *    and analogues for "lb_output_io_*"
 *
 *  Obtain io_options_t information for scalar density "rho_io"
 *  as for "lb_io"
 *
 *****************************************************************************/

int lb_run_time_prev(pe_t * pe, cs_t * cs, rt_t * rt, lb_t ** lb);

int lb_run_time(pe_t * pe, cs_t * cs, rt_t * rt, lb_t ** lb) {

  if (1) {
    lb_run_time_prev(pe, cs, rt, lb);
  }
  else {

    /* I/O */

    io_info_args_t lb_info  = io_info_args_default();
    io_info_args_t rho_info = io_info_args_default();

    io_info_args_rt(pe, rt, "lb", IO_INFO_READ_WRITE, &lb_info);

    /* density is output only */

    io_info_args_rt(pe, rt, "rho", IO_INFO_WRITE_ONLY, &rho_info);

  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_run_time
 *
 *  Input key                         default
 *  -----------------------------------------
 *  "distribution_io_grid"            {1,1,1}
 *  "distribution_io_format_input"    "binary" 
 *
 *  "rho_io_wanted"                   false
 *  "rho_io_grid"                     {1,1,1}
 *  "rho_io_format"                   "binary"
 *
 *****************************************************************************/

int lb_run_time_prev(pe_t * pe, cs_t * cs, rt_t * rt, lb_t ** lb) {

  int ndist;
  int io_grid[3] = {1, 1, 1};
  char string[FILENAME_MAX] = "";
  char memory = ' ';
  int lb_form_in = IO_FORMAT_DEFAULT;
  int lb_form_out = IO_FORMAT_DEFAULT;
  int rho_wanted;

  io_info_t * io_info = NULL;
  io_info_t * io_rho = NULL;

  io_mode_enum_t lb_input_opt_mode = io_mode_default();
  lb_data_options_t options = lb_data_options_default();

  assert(pe);
  assert(cs);
  assert(rt);
  assert(lb);

  /* Number of distributions */

  ndist = lb_ndist_rt(rt);
  assert(ndist == 1 || ndist == 2);

  /* Halo options */

  options.ndim  = NDIM;
  options.nvel  = NVEL;
  options.ndist = ndist;

  /* Halo options */
  {
    char htype[BUFSIZ] = {0};
    int havetype = rt_string_parameter(rt, "lb_halo_scheme", htype, BUFSIZ);
    if (strcmp(htype, "lb_halo_target") == 0) {
      options.halo = LB_HALO_TARGET;
    }
    else if (strcmp(htype, "lb_halo_openmp_full") == 0) {
      options.halo = LB_HALO_OPENMP_FULL;
    }
    else if (strcmp(htype, "lb_halo_openmp_reduced") == 0) {
      options.halo = LB_HALO_OPENMP_REDUCED;
    }
    else if (havetype) {
      pe_fatal(pe, "lb_halo_scheme not recognised\n");
    }

    /* I'm going to trap this silently here - which is slightly
     * better than having the wrong halo. I avoid a message so
     * not as to disrupt the regression tests. */
    {
      int ndevice = 0;
      tdpGetDeviceCount(&ndevice);
      if (ndevice > 0) options.halo = LB_HALO_TARGET;
    }

    options.reportimbalance = rt_switch(rt, "lb_halo_report_imbalance");

    if (lb_data_options_valid(&options) == 0) {
      pe_fatal(pe, "lb_data_options are invalid. Please check halo.\n");
    }
  }


  /* I//O Grid */

  rt_int_parameter_vector(rt, "distribution_io_grid", io_grid);

  rt_string_parameter(rt,"distribution_io_format_input", string,
		      FILENAME_MAX);

  /* Append R to the record if the model is the reverse implementation */ 

  /* TODO: r -> "AOS" or "SOA" or "AOSOA" */
  if (DATA_MODEL == DATA_MODEL_SOA) memory = 'R';

  pe_info(pe, "\n");
  pe_info(pe, "Lattice Boltzmann distributions\n");
  pe_info(pe, "-------------------------------\n");

  pe_info(pe, "Model:            d%dq%d %c\n", NDIM, NVEL, memory);
  pe_info(pe, "SIMD vector len:  %d\n", NSIMDVL);
  pe_info(pe, "Number of sets:   %d\n", ndist);

  if (options.halo == LB_HALO_TARGET) { 
    pe_info(pe, "Halo type:        %s\n", "lb_halo_target (full halo)");
  }
  if (options.halo == LB_HALO_OPENMP_FULL) {
    pe_info(pe, "Halo type:        %s\n", "lb_halo_openmp_full (host)");
  }
  if (options.halo == LB_HALO_OPENMP_REDUCED) {
    pe_info(pe, "Halo type:        %s\n", "lb_halo_openmp_reduced (host)");
  }
  if (options.reportimbalance) {
    pe_info(pe, "Imbalance time:   %s\n", "reported");
  }

  if (strcmp("BINARY_SERIAL", string) == 0) {
    pe_info(pe, "Input format:     binary single serial file\n");
    lb_input_opt_mode = IO_MODE_SINGLE;
  }
  else if (strcmp(string, "ASCII") == 0) {
    lb_form_in = IO_FORMAT_ASCII;
    lb_form_out = IO_FORMAT_ASCII;
    pe_info(pe, "Input format:     ASCII\n");
    pe_info(pe, "Output format:    ASCII\n");
  }
  else {
    pe_info(pe, "Input format:     binary\n");
    pe_info(pe, "Output format:    binary\n");
  }

  pe_info(pe, "I/O grid:         %d %d %d\n", io_grid[X], io_grid[Y], io_grid[Z]);

  /* Density io_info:
   *
   * rho_io_wanted           switch to indicate output wanted [no]
   * rho_io_freq             output frequency
   * rho_io_grid             grid
   * rho_io_format           ASCII/BINARY (default BINARY)
   * */


  int rho_form_in = IO_FORMAT_BINARY;
  int rho_form_out = IO_FORMAT_BINARY;
  io_info_args_t rhoio = io_info_args_default();

  rho_wanted = rt_switch(rt, "rho_io_wanted");
  rt_int_parameter_vector(rt, "rho_io_grid", rhoio.grid);
  rt_string_parameter(rt,"rho_io_format", string, FILENAME_MAX);

  if (strcmp(string, "ASCII") == 0) {
    rho_form_in = IO_FORMAT_ASCII;
    rho_form_out = IO_FORMAT_ASCII;
  }

  if (rho_wanted) {
    pe_info(pe, "Fluid density output\n");
    if (rho_form_out == IO_FORMAT_ASCII)  {
      pe_info(pe, "Output format:    ASCII\n");
    }
    if (rho_form_out == IO_FORMAT_BINARY) {
      pe_info(pe, "Output format:    binary\n");
    }
    pe_info(pe, "I/O grid:         %d %d %d\n",
	    rhoio.grid[X], rhoio.grid[Y], rhoio.grid[Z]);
  }


  /* Initialise */

  lb_data_create(pe, cs, &options, lb);

  /* distribution i/o */
  {
    io_info_args_t tmp = io_info_args_default();
    tmp.grid[X] = io_grid[X];
    tmp.grid[Y] = io_grid[Y];
    tmp.grid[Z] = io_grid[Z];
    io_info_create(pe, cs, &tmp, &io_info);
  }

  io_info_metadata_filestub_set(io_info, "dist");
  if (lb_input_opt_mode == IO_MODE_SINGLE) {
    io_info_set_processor_independent(io_info);
  }
  lb_io_info_set(*lb, io_info, lb_form_in, lb_form_out);

  /* rho i/o */

  io_info_create(pe, cs, &rhoio, &io_rho);
  io_info_metadata_filestub_set(io_rho, "rho");
  lb_io_rho_set(*lb, io_rho, rho_form_in, rho_form_out);

  return 0;
}

/*****************************************************************************
 *
 *  lb_rt_initial_conditions
 *
 *****************************************************************************/

int lb_rt_initial_conditions(pe_t * pe, rt_t * rt, lb_t * lb,
			     physics_t * phys) {

  char key[BUFSIZ] = "";
  double rho0;

  assert(pe);
  assert(rt);
  assert(lb);
  assert(phys);

  physics_rho0(phys, &rho0);

  /* Default */

  lb_init_rest_f(lb, rho0);

  rt_string_parameter(rt, "distribution_initialisation", key, BUFSIZ);

  if (strcmp("2d_kelvin_helmholtz", key) == 0) {

    double u0 = 0.0;
    double delta = 0.0;
    double kappa = 0.0;
    kh_2d_param_t kh = {0};

    rt_key_required(rt, "2d_kelvin_helmholtz_u0", RT_FATAL);
    rt_key_required(rt, "2d_kelvin_helmholtz_delta", RT_FATAL);
    rt_key_required(rt, "2d_kelvin_helmholtz_kappa", RT_FATAL);

    rt_double_parameter(rt, "2d_kelvin_helmholtz_u0", &u0);
    rt_double_parameter(rt, "2d_kelvin_helmholtz_delta", &delta);
    rt_double_parameter(rt, "2d_kelvin_helmholtz_kappa", &kappa);

    kh.rho0 = rho0;
    kh.u0   = u0;
    kh.delta = delta;
    kh.kappa = kappa;

    lb_rt_2d_kelvin_helmholtz(lb, &kh);

    pe_info(pe, "\n");
    pe_info(pe, "Initial distribution: 2d kelvin helmholtz\n");
    pe_info(pe, "Velocity magnitude:   %14.7e\n", kh.u0);
    pe_info(pe, "Shear layer kappa:    %14.7e\n", kh.kappa);
    pe_info(pe, "Perturbation delta:   %14.7e\n", kh.delta);
    pe_info(pe, "\n");
  }

  if (strcmp("2d_shear_wave", key) == 0) {
    lb_rt_2d_shear_wave(lb);
  }

  if (strcmp("3d_uniform_u", key) == 0) {

    double u0[3] = {0.0, 0.0, 0.0};

    rt_double_parameter_vector(rt, "distribution_uniform_u", u0);
    lb_init_uniform(lb, rho0, u0);

    pe_info(pe, "\n");
    pe_info(pe, "Initial distribution: 3d uniform desnity/velocity\n");
    pe_info(pe, "Density:              %14.7e\n", rho0);
    pe_info(pe, "Velocity:             %14.7e %14.7e %14.7e\n",
	    u0[X], u0[Y], u0[Z]);
    pe_info(pe, "\n");
  }

  if (strcmp("1d_poiseuille", key) == 0) {

    double u0[3] = {0.0, 0.0, 0.0};

    rt_double_parameter_vector(rt, "distribution_poiseuille_umax", u0);
    lb_init_poiseuille(lb, rho0, u0);

    pe_info(pe, "\n");
    pe_info(pe, "Initial distribution: 1d Poiseuille profile\n");
    pe_info(pe, "Density:              %14.7e\n", rho0);
    pe_info(pe, "Velocity (max):       %14.7e %14.7e %14.7e\n",
	    u0[X], u0[Y], u0[Z]);
    pe_info(pe, "\n");
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

static int lb_rt_2d_kelvin_helmholtz(lb_t * lb, const kh_2d_param_t * kh) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double u[3];
  double lmin[3];
  double ltot[3];

  double x, y;
  PI_DOUBLE(pi);

  assert(lb);

  cs_nlocal(lb->cs, nlocal);
  cs_nlocal_offset(lb->cs, noffset);
  cs_lmin(lb->cs, lmin);
  cs_ltot(lb->cs, ltot);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = (1.0*(noffset[X] + ic) - lmin[X])/ltot[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = (1.0*(noffset[Y] + jc) - lmin[Y])/ltot[Y];

      if (y >  0.5) u[X] = kh->u0*tanh(kh->kappa*(0.75 - y));
      if (y <= 0.5) u[X] = kh->u0*tanh(kh->kappa*(y - 0.25));
      u[Y] = kh->u0*kh->delta*sin(2.0*pi*(x + 0.25));
      u[Z] = 0.0;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(lb->cs, ic, jc, kc);
        lb_1st_moment_equilib_set(lb, index, kh->rho0, u);
      }
    }
  }

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

static int lb_rt_2d_shear_wave(lb_t * lb) {

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
  PI_DOUBLE(pi);

  assert(lb);

  cs_nlocal(lb->cs, nlocal);
  cs_nlocal_offset(lb->cs, noffset);
  cs_lmin(lb->cs, lmin);
  cs_ltot(lb->cs, ltot);

  kappa = 2.0*pi;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = (1.0*(noffset[Y] + jc) - lmin[Y])/ltot[Y];

      u[X] = u0*sin(kappa * y);
      u[Y] = 0.0;
      u[Z] = 0.0;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(lb->cs, ic, jc, kc);
        lb_1st_moment_equilib_set(lb, index, rho, u);
      }
    }
  }

  pe_info(lb->pe, "\n");
  pe_info(lb->pe, "Initial distribution: 2d shear wave\n");
  pe_info(lb->pe, "Velocity magnitude:   %14.7e\n", u0);
  pe_info(lb->pe, "Shear layer kappa:    %14.7e\n", kappa);
  pe_info(lb->pe, "\n");

  return 0;
}

/*****************************************************************************
 *
 *  lb_init_uniform
 *
 *  Set the initial distribution consistent with fixed (rho_0, u_0).
 *
 *****************************************************************************/

static int lb_init_uniform(lb_t * lb, double rho0, double u0[3]) {

  int ic, jc, kc, index;
  int nlocal[3];

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(lb->cs, ic, jc, kc);
	lb_1st_moment_equilib_set(lb, index, rho0, u0);

      }
    }
  }

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

static int lb_init_poiseuille(lb_t * lb, double rho0, const double umax[3]) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];

  double lmin[3];
  double ltot[3];
  double u0[3];
  double x, y, z;

  assert(lb);

  cs_nlocal(lb->cs, nlocal);
  cs_nlocal_offset(lb->cs, noffset);
  cs_lmin(lb->cs, lmin);
  cs_ltot(lb->cs, ltot);

  for (ic = 1; ic <= nlocal[X]; ic++) {

    /* The - lmin[] in each direction centres the profile symmetrically,
     * and the 4/L^2 normalises to umax at centre */

    x = 1.0*(noffset[X] + ic) - lmin[X];
    u0[X] = umax[X]*x*(ltot[X] - x)*4.0/(ltot[X]*ltot[X]);

    for (jc = 1; jc <= nlocal[Y]; jc++) {

      y = 1.0*(noffset[Y] + jc) - lmin[Y];
      u0[Y] = umax[Y]*y*(ltot[Y] - y)*4.0/(ltot[Y]*ltot[Y]);

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	z = 1.0*(noffset[Z] + kc) - lmin[Z];
	u0[Z] = umax[Z]*z*(ltot[Z] - z)*4.0/(ltot[Z]*ltot[Z]);

	index = cs_index(lb->cs, ic, jc, kc);
	lb_1st_moment_equilib_set(lb, index, rho0, u0);

      }
    }
  }

  return 0;
}
