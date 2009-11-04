/*****************************************************************************
 *
 *  coords_rt.c
 *
 *  Run time stuff for the coordinate system.
 *
 *  $Id: coords_rt.c,v 1.1.2.1 2009-11-04 18:33:43 kevin Exp $
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
  info("--------------\n\n");

  n = RUN_get_int_parameter_vector("size", vector);

  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("Lattice size is (%d, %d, %d)\n", vector[X], vector[Y], vector[Z]);

  coords_ntotal_set(vector);

  n = RUN_get_int_parameter_vector("periodicity", vector);

  info((n == 0) ? "[Default] " : "[User   ] "); 
  info("periodic boundaries set to (%d, %d, %d)\n", vector[X], vector[Y],
       vector[Z]);

  coords_periodicity_set(vector);

  /* Look for a user-defined decomposition */

  n = RUN_get_int_parameter_vector("grid", vector);

  if (n != 0) {
    coords_decomposition_set(vector);
  }

  if (n != 0 && is_ok_decomposition()) {
    /* The user decomposition is selected */
    info("[User   ] Processor decomposition is (%d, %d, %d)\n",
	 vector[X], vector[Y], vector[Z]);
  }
  else {
    /* Look for a default */
    default_decomposition();
    info("[Default] Processor decomposition is (%d, %d, %d)\n",
	 cart_size(X), cart_size(Y), cart_size(Z));
  }

  n = RUN_get_int_parameter("reorder", &reorder);
  if (n != 0) coords_reorder_set(reorder);

  return;
}
