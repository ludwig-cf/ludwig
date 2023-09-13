/*****************************************************************************
 *
 *  map_rt.c
 *
 *  map_t initialisation at run time.
 *
 *  A number of possibilities exist:
 *    1. General data from file, wetting data can be included;
 *    2. Some simple geometric structures; uniform wetting available
 *       via appropriate free energy/gradient considerations.
 *       Not to be confused with wall initialisations, which update
 *       the map status, but are separate (see wall_rt.c).
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "map_init.h"
#include "map_rt.h"

enum map_initial_status {MAP_INIT_FLUID_ONLY,
			 MAP_INIT_CIRCLE_XY,
			 MAP_INIT_SQUARE_XY,
			 MAP_INIT_WALL_X,
			 MAP_INIT_WALL_Y,
			 MAP_INIT_WALL_Z,
			 MAP_INIT_SIMPLE_CUBIC,
			 MAP_INIT_FACE_CENTRED_CUBIC,
			 MAP_INIT_BODY_CENTRED_CUBIC,
			 MAP_INIT_INVALID};

typedef struct map_options_s map_options_t;

struct map_options_s {
  int ndata;
  int geometry;
  int acell;      /* cubic lattice constant; an integer */
};

__host__ int map_init_options(pe_t * pe, cs_t * cs,
			      const map_options_t * options, map_t ** map);
__host__ int map_init_option_acell(pe_t * pe, cs_t * cs, rt_t * rt);
__host__ int map_init_options_parse(pe_t * pe, cs_t * cs, rt_t * rt,
				    map_options_t * options);

/*****************************************************************************
 *
 *  map_init_rt
 *
 *   A map structure is always created, even if only fluid is present.
 *
 *****************************************************************************/

int map_init_rt(pe_t * pe, cs_t * cs, rt_t * rt, map_t ** map) {

  assert(pe);
  assert(cs);
  assert(rt);
  assert(map);

  {
    int is_file = rt_switch(rt, "porous_media_file");

    if (is_file) {
      /* This is compatible with previous versions */
      map_init_porous_media_from_file(pe, cs, rt, map);
    }
    else {
      /* Anything else including simple geometries specified in input */
      /* The default (no porous media) must allow that colloids have
       * access to C, H, that is, ndata = 2 */
      /* Colloids plus porous media is a non-feature at this time. */

      map_options_t options = {0};
      options.ndata = 2;

      map_init_options_parse(pe, cs, rt, &options);
      map_init_options(pe, cs, &options, map);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_porous_media_from_file
 *
 *  The file must have stub "capillary", e.g., "capillary.001-001"
 *  for serial.
 *
 *****************************************************************************/

int map_init_porous_media_from_file(pe_t * pe, cs_t * cs, rt_t * rt,
				    map_t ** pmap) {

  int ndata = 0;
  int have_ndata = 0;
  int form_in = IO_FORMAT_DEFAULT;
  int form_out = IO_FORMAT_DEFAULT;
  int grid[3] = {1, 1, 1};

  char format[BUFSIZ] = "";

  io_info_t * iohandler = NULL;
  map_t * map = NULL;

  assert(pe);
  assert(rt);

  have_ndata = rt_int_parameter(rt, "porous_media_ndata", &ndata);

  if (have_ndata) {
    /* This is now the preferred mechanism */
  }
  else {

    /* Work out ndata from the key. This method will be removed in future. */
    char status[BUFSIZ] = "";

    rt_string_parameter(rt, "porous_media_type", status, BUFSIZ);

    ndata = 0;
    if (strcmp(status, "status_only") == 0) ndata = 0;
    if (strcmp(status, "status_with_h") == 0) ndata = 1;
    if (strcmp(status, "status_with_sigma") == 0) ndata = 1;
    if (strcmp(status, "status_with_c_h") == 0) ndata = 2;

    if (strcmp(status, "status_with_h") == 0) {
      /* This is not to be used as it not implemented correctly. */
      pe_info(pe, "porous_media_type    status_with_h\n");
      pe_info(pe, "Please use status_with_c_h (and set C = 0) instead\n");
      pe_fatal(pe, "Will not continue.\n");
    }
  }

  rt_string_parameter(rt, "porous_media_format", format, BUFSIZ);

  if (strcmp(format, "ASCII") == 0) form_in = IO_FORMAT_ASCII_SERIAL;
  if (strcmp(format, "BINARY") == 0) form_in = IO_FORMAT_BINARY_SERIAL;
  if (strcmp(format, "BINARY_SERIAL") == 0) form_in = IO_FORMAT_BINARY_SERIAL;

  rt_int_parameter_vector(rt, "porous_media_io_grid", grid);

  pe_info(pe, "\n");
  pe_info(pe, "Porous media\n");
  pe_info(pe, "------------\n");
  pe_info(pe, "Porous media file stub:       %s\n", "capillary");
  pe_info(pe, "Porous media file data items: %d\n", ndata);
  pe_info(pe, "Porous media format (serial): %s\n", format);
  pe_info(pe, "Porous media io grid:         %d %d %d\n",
	  grid[X], grid[Y], grid[Z]);

  map_create(pe, cs, ndata, &map);
  map_init_io_info(map, grid, form_in, form_out);
  map_io_info(map, &iohandler);

  io_info_set_processor_independent(iohandler);
  io_read_data(iohandler, "capillary", map);
  map_pm_set(map, 1);

  map_halo(map);

  *pmap = map;

  return 0;
}

/*****************************************************************************
 *
 *  map_init_options
 *
 *****************************************************************************/

__host__ int map_init_options(pe_t * pe, cs_t * cs,
			      const map_options_t * options, map_t ** pmap) {

  map_t * map = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(pmap);

  map_create(pe, cs, options->ndata, &map);

  switch (options->geometry) {
  case MAP_INIT_FLUID_ONLY:
    /* Do nothing; it's default in map_create() */
    break;
  case MAP_INIT_CIRCLE_XY:
    map_init_status_circle_xy(map);
    break;
  case MAP_INIT_SQUARE_XY: /* to include "rectangles" */
    map_init_status_wall(map, X);
    map_init_status_wall(map, Y);
    break;
  case MAP_INIT_WALL_X:
    map_init_status_wall(map, X);
    break;
  case MAP_INIT_WALL_Y:
    map_init_status_wall(map, Y);
    break;
  case MAP_INIT_WALL_Z:
    map_init_status_wall(map, Z);
    break;
  case MAP_INIT_SIMPLE_CUBIC:
    map_init_status_simple_cubic(map, options->acell);
    break;
  case MAP_INIT_FACE_CENTRED_CUBIC:
    map_init_status_face_centred_cubic(map, options->acell);
    break;
  case MAP_INIT_BODY_CENTRED_CUBIC:
    map_init_status_body_centred_cubic(map, options->acell);
    break;
  default:
    pe_fatal(pe, "Internal error: unrecognised porous media geometry\n");
  }

  map_halo(map);

  *pmap = map;

  return 0;
}

/*****************************************************************************
 *
 *  map_init_options_parse
 *
 *****************************************************************************/

__host__ int map_init_options_parse(pe_t * pe, cs_t * cs, rt_t * rt,
				    map_options_t * options) {

  assert(pe);
  assert(cs);
  assert(rt);
  assert(options);

  {
    char ptype[BUFSIZ] = "";
    int is_init = rt_string_parameter(rt, "porous_media_init", ptype, BUFSIZ);

    if (is_init) {

      if (strncmp(ptype, "circle_xy", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_CIRCLE_XY;
      }
      else if (strncmp(ptype, "square_xy", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_SQUARE_XY;
      }
      else if (strncmp(ptype, "wall_x", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_WALL_X;
      }
      else if (strncmp(ptype, "wall_y", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_WALL_Y;
      }
      else if (strncmp(ptype, "wall_z", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_WALL_Z;
      }
      else if (strncmp(ptype, "simple_cubic", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_SIMPLE_CUBIC;
	options->acell = map_init_option_acell(pe, cs, rt);
      }
      else if (strncmp(ptype, "face_centred_cubic", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_FACE_CENTRED_CUBIC;
	options->acell = map_init_option_acell(pe, cs, rt);
      }
      else if (strncmp(ptype, "body_centred_cubic", BUFSIZ) == 0) {
	options->ndata = 0;
	options->geometry = MAP_INIT_BODY_CENTRED_CUBIC;
	options->acell = map_init_option_acell(pe, cs, rt);
      }
      else {
	/* Not recognised */
	pe_info(pe, "Input: porous_media_init not recognised %s\n\n", ptype);
	pe_fatal(pe, "Please check and try again.\n");
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_options_acell
 *
 *  Look for a cubic lattice constant and check it is a multiple of
 *  all three of ntotal[].
 *
 *****************************************************************************/

__host__ int map_init_option_acell(pe_t * pe, cs_t * cs, rt_t * rt) {

  int acell = 0;
  int have_acell = 0;

  assert(pe);
  assert(cs);
  assert(rt);

  have_acell = rt_int_parameter(rt, "porous_media_acell", &acell);

  if (have_acell == 0) {
    /* Fail */
    pe_fatal(pe, "Key porous_media_acell expected but not present\n");
  }
  else {
    /* Check (and possibly fail) */
    int ntotal[3] = {0};
    cs_ntotal(cs, ntotal);

    if (acell < 1) pe_fatal(pe, "acell must be positive\n");
    if (ntotal[X] % acell) pe_fatal(pe, "acell must divide Nx exactly\n");
    if (ntotal[Y] % acell) pe_fatal(pe, "acell must divide Ny exactly\n");
    if (ntotal[Z] % acell) pe_fatal(pe, "acell must divide Nz exactly\n");
    /* ok ... */
  }

  return acell;
}
