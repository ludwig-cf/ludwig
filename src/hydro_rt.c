/*****************************************************************************
 *
 *  hydro_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "io_info_args_rt.h"
#include "hydro_rt.h"

static int hydro_do_init(pe_t * pe, rt_t * rt, cs_t * cs, lees_edw_t * le,
			 hydro_t ** phydro);

/*****************************************************************************
 *
 *  hydro_rt
 *
 ****************************************************************************/

int hydro_rt(pe_t * pe, rt_t * rt, cs_t * cs, lees_edw_t * le,
	     hydro_t ** phydro) {

  int hswitch = 1;
  char value[BUFSIZ] = "";

  assert(pe);
  assert(rt);
  assert(phydro);

  if (rt_string_parameter(rt, "hydrodynamics", value, BUFSIZ)) {
    if (strcmp(value, "off") == 0) hswitch = 0;
    if (strcmp(value, "0") == 0) hswitch = 0;
    if (strcmp(value, "no") == 0) hswitch = 0;
  }

  pe_info(pe, "\n");
  pe_info(pe, "Hydrodynamics\n");
  pe_info(pe, "-------------\n");
  pe_info(pe, "Hydrodynamics: %s\n", (hswitch) ? "on" : "off");

  if (hswitch) hydro_do_init(pe, rt, cs, le, phydro);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_do_init
 *
 *  Some work on the I/O options is required:
 *    - should be specified as part of the options.
 *  Halo options should be extended to include density.
 *
 *****************************************************************************/

static int hydro_do_init(pe_t * pe, rt_t * rt, cs_t * cs, lees_edw_t * le,
			 hydro_t ** phydro) {

  hydro_options_t opts = hydro_options_default();
  hydro_t * obj = NULL;

  char value[BUFSIZ] = "";
  int io_grid[3] = {1, 1, 1};
  int io_format_in  = IO_FORMAT_DEFAULT;
  int io_format_out = IO_FORMAT_DEFAULT;

  assert(rt);
  assert(phydro);

  /* Halo scheme options (all fields have the same option!) */

  if (rt_string_parameter(rt, "hydro_halo_scheme", value, BUFSIZ)) {
    field_halo_enum_t haloscheme = FIELD_HALO_TARGET;
    /* The output is only provided if the key is present to
     * prevent the regression tests getting upset. */
    if (strcmp(value, "hydro_u_halo_target") == 0) {
      /* Should be the default */
      haloscheme = FIELD_HALO_TARGET;
      pe_info(pe, "Hydro halo:    %s\n", "hydro_u_halo_target");
    }
    else if (strcmp(value, "hydro_u_halo_openmp") == 0) {
      haloscheme = FIELD_HALO_OPENMP;
      pe_info(pe, "Hydro halo:    %s\n", "hydro_u_halo_openmp");    
    }
    else if (strcmp(value, "hydro_u_halo_host") == 0) {
      haloscheme = FIELD_HALO_HOST;
      pe_info(pe, "Hydro halo:    %s\n", "hydro_u_halo_host");
    }
    else {
      pe_fatal(pe, "hydro_halo_scheme is present but not recongnised\n");
    }
    opts = hydro_options_haloscheme(haloscheme);
  }

  /* First touch option (all or nothing) */

  if (rt_switch(rt, "hydro_data_use_first_touch")) {
    opts.rho.usefirsttouch   = 1;
    opts.u.usefirsttouch     = 1;
    opts.force.usefirsttouch = 1;
    opts.eta.usefirsttouch   = 1;
    pe_info(pe, "Hydro data:    first touch\n");
  }

  /* User i/o options */
  io_info_args_rt(rt, RT_FATAL, "rho", IO_INFO_READ_WRITE, &opts.rho.iodata);
  io_info_args_rt(rt, RT_FATAL, "vel", IO_INFO_READ_WRITE, &opts.u.iodata);

  hydro_create(pe, cs, le, &opts, &obj);
  assert(obj);

  /* Old-style input (to be removed) */
  rt_int_parameter_vector(rt, "default_io_grid", io_grid);
  rt_string_parameter(rt, "vel_format", value, BUFSIZ);

  if (strcmp(value, "ASCII") == 0) {
    io_format_in = IO_FORMAT_ASCII;
    io_format_out = IO_FORMAT_ASCII;
  }

  hydro_init_io_info(obj, io_grid, io_format_in, io_format_out);

  *phydro = obj;

  return 0;
}
