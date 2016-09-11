/*****************************************************************************
 *
 *  coords_rt.c
 *
 *  Run time stuff for the coordinate system.
 *
 *  $Id: coords_rt.c,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2016 The University of Edinburgh
 *
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

  n = rt_int_parameter_vector(rt, "size", vector);
  cs_ntotal_set(cs, vector);

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
