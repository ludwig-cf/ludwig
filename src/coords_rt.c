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

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "coords_rt.h"

/*****************************************************************************
 *
 *  coords_run_time
 *
 *****************************************************************************/

int coords_run_time(pe_t * pe, rt_t * rt) {

  int n;
  int reorder;
  int vector[3];

  pe_info(pe, "\n");
  pe_info(pe, "System details\n");
  pe_info(pe, "--------------\n");

  n = rt_int_parameter_vector(rt, "size", vector);
  coords_ntotal_set(vector);

  n = rt_int_parameter_vector(rt, "periodicity", vector);
  if (n != 0) coords_periodicity_set(vector);

  /* Look for a user-defined decomposition */

  n = rt_int_parameter_vector(rt, "grid", vector);
  if (n != 0) coords_decomposition_set(vector);

  n = rt_int_parameter(rt, "reorder", &reorder);
  if (n != 0) coords_reorder_set(reorder);

  coords_init();
  coords_info();

  return 0;
}
