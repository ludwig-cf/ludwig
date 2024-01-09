/*****************************************************************************
 *
 *  coords_rt.c
 *
 *  Run time stuff for the coordinate system.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "coords_rt.h"

/*****************************************************************************
 *
 *  coords_init_rt
 *
 *****************************************************************************/

int coords_init_rt(pe_t * pe, rt_t * rt, cs_t * cs) {

  int n;
  int reorder;
  int vector[3];

  assert(pe);
  assert(rt);
  assert(cs);

  {
    int ntotal[3] = {0, 0, 0};
    rt_int_parameter_vector(rt, "size", ntotal);
    if (ntotal[X] < 1 || ntotal[Y] < 1 || ntotal[Z] < 1) {
      pe_info(pe, "System size must be > 0 in all three co-ordinates\n");
      pe_fatal(pe, "Please check and try again\n");
    }
    cs_ntotal_set(cs, ntotal);
  }

  n = rt_int_parameter_vector(rt, "periodicity", vector);
  if (n != 0) cs_periodicity_set(cs, vector);

  /* Look for a user-defined decomposition */

  n = rt_int_parameter_vector(rt, "grid", vector);
  if (n != 0) cs_decomposition_set(cs, vector);

  n = rt_int_parameter(rt, "reorder", &reorder);
  if (n != 0) cs_reorder_set(cs, reorder);

  cs_init(cs);
  cs_info(cs);

  return 0;
}
