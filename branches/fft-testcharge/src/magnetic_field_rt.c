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

  int nb, ne;
  double b0[3];
  double e0[3];

  nb = RUN_get_double_parameter_vector("magnetic_b0", b0);
  if (nb == 1) magnetic_field_b0_set(b0);

  ne = RUN_get_double_parameter_vector("electric_e0", e0);
  if (ne == 1) electric_field_e0_set(e0);

  if (nb || ne) {
    magnetic_field_b0(b0);
    electric_field_e0(e0);

    info("\n");
    info("Uniform external fields\n");
    info("---------------\n");
    info("Magnetic field: %14.7e %14.7e %14.7e\n", b0[X], b0[Y], b0[Z]);
    info("Electric field: %14.7e %14.7e %14.7e\n", e0[X], e0[Y], e0[Z]);
  }

  return;
}
