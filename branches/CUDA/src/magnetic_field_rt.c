/*****************************************************************************
 *
 *  magnetic_field_rt.c
 *
 *  Run time initialisation of external magnetic field quantities.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "magnetic_field.h"
#include "magnetic_field_rt.h"

/*****************************************************************************
 *
 *  magnetic_field_runtime
 *
 *****************************************************************************/

void magnetic_field_runtime(void) {

  int n;
  double b0[3];

  n = RUN_get_double_parameter_vector("magnetic_b0", b0);

  if (n == 1) magnetic_field_b0_set(b0);

  magnetic_field_b0(b0);
  info("\n");
  info("Uniform external magnetic field: %12.5e %12.5e %12.5e\n\n",
       b0[X], b0[Y], b0[Z]);

  return;
}
