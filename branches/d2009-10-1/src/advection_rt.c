/*****************************************************************************
 *
 *  advection_rt.c
 *
 *  Look at awitches associated with advection.
 *
 *  $Id: advection_rt.c,v 1.1.2.1 2010-03-30 14:10:16 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "advection.h"
#include "advection_rt.h"

/*****************************************************************************
 *
 *  advection_run_time
 *
 *****************************************************************************/

void advection_run_time(void) {

  int n;
  int order;
  char key1[FILENAME_MAX];
  char key2[FILENAME_MAX];

  RUN_get_string_parameter("free_energy", key1, FILENAME_MAX);
  RUN_get_string_parameter("phi_finite_difference", key2, FILENAME_MAX);

  if (strcmp(key1, "none") != 0 && strcmp(key2, "yes") == 0) {

    info("Advection scheme order: ");

    n = RUN_get_int_parameter("fd_advection_scheme_order", &order);

    if (n == 0) {
      info("%2d (default)\n", advection_order());
    }
    else {
      info("%d\n", order);
      advection_order_set(order);
    }
  }

  return;
}
