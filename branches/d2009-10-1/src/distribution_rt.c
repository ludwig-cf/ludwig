/*****************************************************************************
 *
 *  distribution_rt.c
 *
 *  $Id: distribution_rt.c,v 1.1.2.1 2010-08-04 17:48:39 kevin Exp $
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
#include "model.h"
#include "io_harness.h"
#include "distribution_rt.h"

/*****************************************************************************
 *
 *  distribution_run_time
 *
 *****************************************************************************/

void distribution_run_time(void) {

  int nreduced;
  int io_grid[3] = {1, 1, 1};
  char string[FILENAME_MAX];

  struct io_info_t * io_info;

  RUN_get_string_parameter("free_energy", string, FILENAME_MAX);
  if (strcmp(string, "symmetric_lb") == 0) distribution_ndist_set(2);

  nreduced = 0;
  RUN_get_string_parameter("reduced_halo", string, FILENAME_MAX);
  if (strcmp(string, "yes") == 0) nreduced = 1;

  RUN_get_int_parameter_vector("distribution_io_grid", io_grid);
  io_info = io_info_create_with_grid(io_grid);
  distribution_io_info_set(io_info);


  info("\n");
  info("Lattice Boltzmann distributions\n");
  info("-------------------------------\n");

  info("Model:            d%dq%d\n", NDIM, NVEL);
  info("Number of sets:   %d\n", distribution_ndist());
  info("Halo type:        %s\n", (nreduced == 1) ? "reduced" : "full");
  info("Input format:     binary\n");
  info("Output format:    binary\n");
  info("I/O grid:         %d %d %d\n", io_grid[0], io_grid[1], io_grid[2]);
  info("\n");

  distribution_init();

  io_write_metadata("dist", distribution_io_info());
  if (nreduced == 1) distribution_halo_set_reduced();

  return;
}
