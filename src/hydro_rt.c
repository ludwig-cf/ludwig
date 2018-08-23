/*****************************************************************************
 *
 *  hydro_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
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
 *  Note that input format is really irrelevant for velocity, as it
 *  is never read from file.
 *
 *****************************************************************************/

static int hydro_do_init(pe_t * pe, rt_t * rt, cs_t * cs, lees_edw_t * le,
			 hydro_t ** phydro) {

  hydro_t * obj = NULL;

  char value[BUFSIZ] = "";
  int nhcomm = 1; /* Always create with halo width one */
  int io_grid[3] = {1, 1, 1};
  int io_format_in  = IO_FORMAT_DEFAULT;
  int io_format_out = IO_FORMAT_DEFAULT;

  assert(rt);
  assert(phydro);

  hydro_create(pe, cs, le, nhcomm, &obj);
  assert(obj);

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
