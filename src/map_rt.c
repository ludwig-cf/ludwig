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
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "map_rt.h"

enum map_initial_status {MAP_INIT_FLUID_ONLY,
			 MAP_INIT_CIRCLE_XY,
			 MAP_INIT_SQUARE_XY,
			 MAP_INIT_WALL_X,
			 MAP_INIT_WALL_Y,
			 MAP_INIT_WALL_Z,
			 MAP_INIT_INVALID};

typedef struct map_options_s map_options_t;

struct map_options_s {
  int ndata;
  int geometry;
};

__host__ int map_init_options(pe_t * pe, cs_t * cs,
			      const map_options_t * options, map_t ** map);
__host__ int map_init_options_parse(pe_t * pe, rt_t * rt,
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
    char filestub[FILENAME_MAX] = ""; /* not used (yet) */
    int is_file = rt_string_parameter(rt, "porous_media_file", filestub,
				      FILENAME_MAX);
    if (is_file) {
      /* This is compatible with previous versions */
      map_init_porous_media_from_file(pe, cs, rt, map);
    }
    else {
      /* Anything else including simple geometries specified in input */
      map_options_t options = {};
      map_init_options_parse(pe, rt, &options);
      map_init_options(pe, cs, &options, map);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_porous_media_file
 *
 *  The case of "from_file".
 *
 *****************************************************************************/

int map_init_porous_media_from_file(pe_t * pe, cs_t * cs, rt_t * rt,
				    map_t ** pmap) {

  int is_porous_media = 0;
  int ndata = 2;           /* Default is to allow C,H e.g. for colloids */
  int form_in = IO_FORMAT_DEFAULT;
  int form_out = IO_FORMAT_DEFAULT;
  int grid[3] = {1, 1, 1};

  char status[BUFSIZ] = "";
  char format[BUFSIZ] = "";
  char filename[FILENAME_MAX];

  io_info_t * iohandler = NULL;
  map_t * map = NULL;

  assert(pe);
  assert(rt);

  is_porous_media = rt_string_parameter(rt, "porous_media_file", filename,
					FILENAME_MAX);
  if (is_porous_media) {

    rt_string_parameter(rt, "porous_media_type", status, BUFSIZ);

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

    rt_string_parameter(rt, "porous_media_format", format, BUFSIZ);

    if (strcmp(format, "ASCII") == 0) form_in = IO_FORMAT_ASCII_SERIAL;
    if (strcmp(format, "BINARY") == 0) form_in = IO_FORMAT_BINARY_SERIAL;
    if (strcmp(format, "BINARY_SERIAL") == 0) form_in = IO_FORMAT_BINARY_SERIAL;

    rt_int_parameter_vector(rt, "porous_media_io_grid", grid);

    pe_info(pe, "\n");
    pe_info(pe, "Porous media\n");
    pe_info(pe, "------------\n");
    pe_info(pe, "Porous media file requested:  %s\n", filename);
    pe_info(pe, "Porous media file type:       %s\n", status);
    pe_info(pe, "Porous media format (serial): %s\n", format);
    pe_info(pe, "Porous media io grid:         %d %d %d\n",
	    grid[X], grid[Y], grid[Z]);
  }

  map_create(pe, cs, ndata, &map);
  map_init_io_info(map, grid, form_in, form_out);
  map_io_info(map, &iohandler);

  if (is_porous_media) {
    io_info_set_processor_independent(iohandler);
    io_read_data(iohandler, filename, map);
    map_pm_set(map, 1);
  }
  map_halo(map);

  *pmap = map;

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_circle_xy
 *
 *  Centre at (Lx/2, Ly/2) with solid boundary at L = 1 and L = L in (x,y).
 *  We could insist that the system is square.
 *
 *****************************************************************************/

int map_init_status_circle_xy(pe_t * pe, cs_t * cs, map_t * map) {

  int ntotal[3] = {};
  int nlocal[3] = {};
  int noffset[3] = {};
  double x0, y0, r0;

  assert(pe);
  assert(cs);
  assert(map);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  /* Check (x,y) is square and assign a centre and radius */

  if (ntotal[X] != ntotal[Y]) {
    pe_fatal(pe, "map_init_status_circle_xy must have Lx == Ly\n");
  }

  x0 = 0.5*(1 + ntotal[X]); /* ok for even, odd ntotal[X] */
  y0 = 0.5*(1 + ntotal[Y]);
  r0 = 0.5*(ntotal[X] - 2);

  /* Assign status */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    double x = (noffset[X] + ic) - x0;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      double y = (noffset[Y] + jc) - y0;

      double r = x*x + y*y;
      char status = MAP_BOUNDARY;

      if (r <= r0*r0) status = MAP_FLUID;

      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	map->status[addr_rank0(map->nsite, index)] = status;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_status_wall
 *
 *  Initialise boundary at L = 1 and L = L in given direction.
 *  Note we do not set any fluid sites.
 *
 *****************************************************************************/

__host__ int map_init_status_wall(pe_t * pe, cs_t * cs, int id, map_t * map) {

  int ntotal[3]  = {};
  int nlocal[3]  = {};
  int noffset[3] = {};

  assert(pe);
  assert(cs);
  assert(id == X || id == Y || id == Z);
  assert(map);

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    int ix = noffset[X] + ic;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      int iy = noffset[Y] + jc;
      for (int kc = 1; kc <= nlocal[Y]; kc++) {
	int iz = noffset[Z] + kc;
	int index = cs_index(cs, ic, jc, kc);

	if (id == X && (ix == 1 || ix == ntotal[X])) { 
	  map->status[addr_rank0(map->nsite, index)] = MAP_BOUNDARY;
	}
	if (id == Y && (iy == 1 || iy == ntotal[Y])) { 
	  map->status[addr_rank0(map->nsite, index)] = MAP_BOUNDARY;
	}
	if (id == Z && (iz == 1 || iz == ntotal[Z])) { 
	  map->status[addr_rank0(map->nsite, index)] = MAP_BOUNDARY;
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_options
 *
 *****************************************************************************/

__host__ int map_init_options(pe_t * pe, cs_t * cs,
			      const map_options_t * options, map_t ** map) {
  assert(pe);
  assert(cs);
  assert(options);
  assert(map);

  map_create(pe, cs, options->ndata, map);

  switch (options->geometry) {
  case MAP_INIT_FLUID_ONLY:
    /* Do nothing; it's default in map_create() */
    break;
  case MAP_INIT_CIRCLE_XY:
    map_init_status_circle_xy(pe, cs, *map);
    break;
  case MAP_INIT_SQUARE_XY: /* to include "rectangles" */
    map_init_status_wall(pe, cs, X, *map);
    map_init_status_wall(pe, cs, Y, *map);
    break;
  case MAP_INIT_WALL_X:
    map_init_status_wall(pe, cs, X, *map);
    break;
  case MAP_INIT_WALL_Y:
    map_init_status_wall(pe, cs, Y, *map);
    break;
  case MAP_INIT_WALL_Z:
    map_init_status_wall(pe, cs, Z, *map);
    break;
  default:
    pe_fatal(pe, "Internal error: unrecognised porous media geometry\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_options_parse
 *
 *****************************************************************************/

__host__ int map_init_options_parse(pe_t * pe, rt_t * rt,
				    map_options_t * options) {

  assert(pe);
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
      else {
	/* Not recognised */
	pe_info(pe, "Input: porous_media_init not recognised %s\n\n", ptype);
	pe_fatal(pe, "Please check and try again.\n");
      }
    }
  }

  return 0;
}
