/*****************************************************************************
 *
 *  hydro_rt.c
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "hydro_rt.h"

static int hydro_do_init(rt_t * rt, coords_t * cs, hydro_t ** phydro);

/*****************************************************************************
 *
 *  hydro_rt
 *
 ****************************************************************************/

int hydro_rt(rt_t * rt, coords_t * cs, hydro_t ** phydro) {

  int hswitch = 1;
  char value[BUFSIZ];

  assert(rt);
  assert(cs);
  assert(phydro);

  if (rt_string_parameter(rt, "hydrodynamics", value, BUFSIZ)) {
    if (strcmp(value, "off") == 0) hswitch = 0;
    if (strcmp(value, "0") == 0) hswitch = 0;
    if (strcmp(value, "no") == 0) hswitch = 0;
  }

  info("\n");
  info("Hydrodynamics\n");
  info("-------------\n");
  info("Hydrodynamics: %s\n", (hswitch) ? "on" : "off");

  if (hswitch) hydro_do_init(rt, cs, phydro);

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

static int hydro_do_init(rt_t * rt, coords_t * cs, hydro_t ** phydro) {

  hydro_t * obj = NULL;

  char value[BUFSIZ];
  int nhcomm = 1; /* Always create with halo width one */
  int io_grid[3] = {1, 1, 1};
  int io_format_in  = IO_FORMAT_DEFAULT;
  int io_format_out = IO_FORMAT_DEFAULT;

  assert(rt);
  assert(cs);
  assert(phydro);

  hydro_create(cs, nhcomm, &obj);
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
