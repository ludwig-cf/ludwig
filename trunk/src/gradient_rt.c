/*****************************************************************************
 *
 *  gradient_rt.c
 *
 *  Set the gradient routine. 
 *
 *  $Id$
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
#include "gradient.h"
#include "gradient_2d_5pt_fluid.h"
#include "gradient_3d_7pt_fluid.h"
#include "gradient_3d_7pt_solid.h"
#include "gradient_3d_27pt_fluid.h"
#include "gradient_3d_27pt_solid.h"
#include "gradient_rt.h"

/*****************************************************************************
 *
 *  gradient_run_time
 *
 *****************************************************************************/

void gradient_run_time(void) {

  int n;
  char key1[FILENAME_MAX];

  n = RUN_get_string_parameter("free_energy", key1, FILENAME_MAX);

  if (n == 0 || strcmp(key1, "none") == 0) return;

  n = RUN_get_string_parameter("fd_gradient_calculation", key1, FILENAME_MAX);

  if (n == 0) {
    info("You must specify the key fd_gradient_calculation\n");
    fatal("Please check and try again\n");
  }
  else {
    info("Gradient calcaulation: ");
    if (strcmp(key1, "2d_5pt_fluid") == 0) {
      info("2d_5pt_fluid\n");
      gradient_2d_5pt_fluid_init();
    }
    else if (strcmp(key1, "3d_7pt_fluid") == 0) {
      info("3d_7pt_fluid\n");
      gradient_3d_7pt_fluid_init();
    }
    else if (strcmp(key1, "3d_7pt_solid") == 0) {
      info("3d_7pt_solid\n");
      gradient_3d_7pt_solid_init();
    }
    else if (strcmp(key1, "3d_27pt_fluid") == 0) {
      info("3d_27pt_fluid\n");
      gradient_3d_27pt_fluid_init();
    }
    else if (strcmp(key1, "3d_27pt_solid") == 0) {
      info("3d_27pt_solid\n");
      gradient_3d_27pt_solid_init();
    }
    else {
      /* Not recognised */
      info("\nfd_gradient_calculation %s not recognised\n", key1);
      fatal("Please check and try again\n");
    }
  }

  info("\n");

  return;
}
