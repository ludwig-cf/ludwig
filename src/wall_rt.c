/*****************************************************************************
 *
 *  wall_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015-2016 The University of Edinburgh
 *
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

int wall_rt_init(pe_t * pe, cs_t * cs, rt_t * rt, lb_t * lb, map_t * map,
		 wall_t ** wall) {

  double ux_bot = 0.0;
  double ux_top = 0.0;
  int vector[3]; 
  wall_param_t p = {0};

  assert(rt);
  assert(cs);
  assert(lb);
  assert(map);
  assert(wall);

  /* We may have porous media, or wall boundaries, but not usually
     both. Some care may be required. */

  map_pm(map, &p.isporousmedia);
  rt_int_parameter_vector(rt, "boundary_walls", &p.isboundary[X]);

  p.iswall = (p.isboundary[X] || p.isboundary[Y] || p.isboundary[Z]);

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
	//Distance to colloid -->
    rt_double_parameter(rt, "boundary_distance_floor", &p.limit_colloid_floor[X]);
    p.limit_colloid_floor[Y] = p.limit_colloid_floor[X];
    p.limit_colloid_floor[Z] = p.limit_colloid_floor[X];
    rt_double_parameter(rt, "boundary_distance_ceil", &p.limit_colloid_ceil[X]);
    rt_int_parameter_vector(rt, "size", vector);
    p.limit_colloid_ceil[Z] = vector[Z]-p.limit_colloid_ceil[X];
    p.limit_colloid_ceil[Y] = vector[Y]-p.limit_colloid_ceil[X];
    p.limit_colloid_ceil[X] = vector[X]-p.limit_colloid_ceil[X];
    //<-- Distance to colloid
  }

  /* Allocate */

  wall_create(pe, cs, map, lb, wall);

  if (p.iswall || p.isporousmedia) {
    wall_commit(*wall, p);
    wall_info(*wall);
  }

  return 0;
}
