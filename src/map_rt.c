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
 *  (c) 2021-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "io_info_args_rt.h"
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

typedef struct map_keys_s map_keys_t;

struct map_keys_s {
  int ndata;
  int geometry;
  int acell;      /* cubic lattice constant; an integer */
};

__host__ int map_init_options(pe_t * pe, cs_t * cs,
			      const map_keys_t * options, map_t ** map);
__host__ int map_init_option_acell(pe_t * pe, cs_t * cs, rt_t * rt);
__host__ int map_init_options_parse(pe_t * pe, cs_t * cs, rt_t * rt,
				    map_keys_t * options);

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

      map_keys_t options = {0};
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
 *  A file map-000000000.001-001 must be present.
 *
 *****************************************************************************/

int map_init_porous_media_from_file(pe_t * pe, cs_t * cs, rt_t * rt,
				    map_t ** pmap) {

  int ndata = 0;
  int have_data = 0;

  char status[BUFSIZ] = "";
  map_t * map = NULL;

  assert(pe);
  assert(rt);

  have_data = rt_string_parameter(rt, "porous_media_data", status, BUFSIZ);

  if (have_data) {

    /* Default ndata = 0 */
    if (strcmp(status, "status_only") == 0) ndata = 0;
    if (strcmp(status, "status_with_h") == 0) ndata = 1;
    if (strcmp(status, "status_with_sigma") == 0) ndata = 1;
    if (strcmp(status, "status_with_c_h") == 0) ndata = 2;

  }

  /* Initialise the map structure */

  {
    int ifail = 0;
    map_options_t opts = map_options_ndata(ndata);

    io_info_args_rt(rt, RT_FATAL, "porous_media", IO_INFO_READ_WRITE,
		    &opts.iodata);

    ifail = map_create(pe, cs, &opts, &map);
    ifail = map_io_read(map, 0);

    if (ifail != 0) {
      pe_exit(pe, "Error reading map file for porous media input\n");
    }

    map_pm_set(map, 1);
    map_halo(map);
  }

  /* Report */

  pe_info(pe, "\n");
  pe_info(pe, "Porous media\n");
  pe_info(pe, "------------\n");
  pe_info(pe, "Porous media from file:       %s\n", "yes");
  pe_info(pe, "Porous media file data items: %d\n", map->ndata);
  pe_info(pe, "Porous media format:          %s\n",
	  io_record_format_to_string(map->options.iodata.input.iorformat));
  pe_info(pe, "Porous media io grid:         %d %d %d\n",
	  map->options.iodata.input.iogrid[X],
	  map->options.iodata.input.iogrid[Y],
	  map->options.iodata.input.iogrid[Z]);

  *pmap = map;

  return 0;
}

/*****************************************************************************
 *
 *  map_init_options
 *
 *****************************************************************************/

__host__ int map_init_options(pe_t * pe, cs_t * cs,
			      const map_keys_t * options, map_t ** pmap) {

  assert(pe);
  assert(cs);
  assert(options);
  assert(pmap);

  map_options_t opts = map_options_ndata(options->ndata);
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);

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
				    map_keys_t * options) {

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
