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
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"

/*****************************************************************************
 *
 *  coords_run_time
 *
 *****************************************************************************/

void coords_run_time(void) {

  int n;
  int reorder;
  int vector[3];

  info("\n");
  info("System details\n");
  info("--------------\n");

  n = RUN_get_int_parameter_vector("size", vector);
  coords_ntotal_set(vector);

  n = RUN_get_int_parameter_vector("periodicity", vector);
  if (n != 0) coords_periodicity_set(vector);

  /* Look for a user-defined decomposition */

  n = RUN_get_int_parameter_vector("grid", vector);
  if (n != 0) coords_decomposition_set(vector);

  n = RUN_get_int_parameter("reorder", &reorder);
  if (n != 0) coords_reorder_set(reorder);

  coords_init();
  coords_info();

  return;
}
