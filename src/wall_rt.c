/*****************************************************************************
 *
 *  wall_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015-2022 The University of Edinburgh
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
  wall_slip_t ws = {0};
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

    rt_double_parameter(rt, "boundary_lubrication_dhnormal", &p.lubr_dh[X]);
    p.lubr_dh[Y] = p.lubr_dh[X];
    p.lubr_dh[Z] = p.lubr_dh[X];
  }

  /* Slip properties [optional] */

  if (p.iswall) {
    double sbot[3] = {0.0, 0.0, 0.0};
    double stop[3] = {0.0, 0.0, 0.0};

    rt_double_parameter_vector(rt, "boundary_walls_slip_fraction_bot", sbot);
    rt_double_parameter_vector(rt, "boundary_walls_slip_fraction_top", stop);

    ws = wall_slip(sbot, stop);
    if (wall_slip_valid(&ws) == 0) {
      /* Could move elsewhere */
      pe_info(pe,  "Wall slip parameters must be 0 <= s <= 1 everywhere\n");
      pe_fatal(pe, "Please check and try again!\n");
    }

    /* Allow user to force use of slip (even if s = 0 everywhere) */
    if (rt_switch(rt, "boundary_walls_slip_active")) ws.active = 1;

    p.slip = ws;
  }

  /* Allocate */

  wall_create(pe, cs, map, lb, wall);

  if (p.iswall || p.isporousmedia) {
    wall_commit(*wall, &p);
    wall_info(*wall);
  }

  return 0;
}
