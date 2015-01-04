/*****************************************************************************
 *
 *  leesedwards_rt.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "leesedwards_rt.h"

/*****************************************************************************
 *
 *  le_rt
 *
 *  Runtime initialisation of LE object/parameters.
 *
 *****************************************************************************/

int le_rt(rt_t * rt, coords_t * coords, le_t ** ple) {

  int nplane = 0;
  int nt0 = 0;
  int period = 0;
  double uy = 0.0;

  le_t * le = NULL;

  assert(rt);
  assert(coords);
  assert(ple);

  le_create(coords, &le);
  assert(le);

  rt_int_parameter(rt, "N_LE_plane", &nplane);
  rt_double_parameter(rt, "LE_plane_vel", &uy);
  rt_int_parameter(rt, "LE_time_offset", &nt0);
  rt_int_parameter(rt, "LE_oscillation_period", &period);

  le_nplane_set(le, nplane);
  le_plane_uy_set(le, uy);
  le_toffset_set(le, nt0);

  if (period > 0) {
    le_oscillatory_set(le, period);
  }

  le_commit(le);
  le_info(le);

  *ple = le;

  return 0;
}
