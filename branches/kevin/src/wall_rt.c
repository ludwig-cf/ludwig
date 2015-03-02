/*****************************************************************************
 *
 *  wall_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "wall_rt.h"

/*****************************************************************************
 *
 *  wall_rt_init
 *
 ****************************************************************************/

int wall_rt_init(rt_t * rt, coords_t * cs, lb_t * lb, map_t * map,
		 wall_t ** wall) {

  double ux_bot = 0.0;
  double ux_top = 0.0;
  wall_param_t p = {0};

  assert(rt);
  assert(cs);
  assert(lb);
  assert(map);
  assert(wall);

  /* We may have porous media, or wall boundaries, but not both. */

  map_pm(map, &p.isporousmedia);
  rt_int_parameter_vector(rt, "boundary_walls", &p.isboundary[X]);

  p.iswall = (p.isboundary[X] || p.isboundary[Y] || p.isboundary[Z]);

  if (p.iswall && p.isporousmedia) {
    fatal("You have both porous media and walls in the input\n");
  }

  /* Run through input parameters */

  if (p.iswall) {
    rt_double_parameter(rt, "boundary_speed_bottom", &ux_bot);
    rt_double_parameter(rt, "boundary_speed_top", &ux_top);

    /* To be updated in input */

    p.ubot[X] = ux_bot; p.ubot[Y] = 0.0; p.ubot[Z] = 0.0;
    p.utop[X] = ux_top; p.utop[Y] = 0.0; p.utop[Z] = 0.0;

    rt_double_parameter(rt, "boundary_lubrication_rcnormal", &p.lubr_rc[X]);
    p.lubr_rc[Y] = p.lubr_rc[X];
    p.lubr_rc[Z] = p.lubr_rc[X];
  }

  /* Allocate */

  if (p.isporousmedia) {
    wall_create(cs, map, lb, wall);
    wall_commit(*wall, p);
  }

  if (p.iswall) {
    wall_create(cs, map, lb, wall);
    wall_commit(*wall, p);

    info("\n");
    info("Boundary walls\n");
    info("--------------\n");
    info("Boundary walls:                  %1s %1s %1s\n",
	 (p.isboundary[X] == 1) ? "X" : "-",
	 (p.isboundary[Y] == 1) ? "Y" : "-",
	 (p.isboundary[Z] == 1) ? "Z" : "-");
    info("Boundary speed u_x (bottom):    %14.7e\n", p.ubot[X]);
    info("Boundary speed u_x (top):       %14.7e\n", p.utop[X]);
    info("Boundary normal lubrication rc: %14.7e\n", p.lubr_rc[X]);
  }

  return 0;
}
