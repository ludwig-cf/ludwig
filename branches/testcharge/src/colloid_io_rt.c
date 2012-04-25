/*****************************************************************************
 *
 *  colloid_io_rt.c
 *
 *  Run time colloid I/O settings.
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

#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "cio.h"

/*****************************************************************************
 *
 *  colloid_io_run_time
 *
 *****************************************************************************/

void colloid_io_run_time(void) {

  int  nuser;
  int io_grid[3] = {1, 1, 1};
  char tmp[256];

  RUN_get_int_parameter_vector("default_io_grid", io_grid);
  RUN_get_int_parameter_vector("colloid_io_grid", io_grid);

  colloid_io_init(io_grid);

  /* Defaults */

  colloid_io_format_input_ascii_set();
  colloid_io_format_output_ascii_set();

  info("\n");
  info("Colloid I/O settings\n");
  info("--------------------\n");
  info("Decomposition:  %2d %2d %2d\n", io_grid[0], io_grid[1], io_grid[2]);
  
  nuser = RUN_get_string_parameter("colloid_io_format_input", tmp, 256);

  if (nuser == 0) {
    info("Input format:       ascii serial\n");
  }

  if (strncmp("ASCII",  tmp, 5) == 0 ) {
    colloid_io_format_input_ascii_set();
    info("Input format:  ascii\n");
  }

  if (strncmp("ASCII_SERIAL",  tmp, 12) == 0 ) {
    colloid_io_format_input_ascii_set();
    colloid_io_format_input_serial_set();
    info("Input file:    serial single file\n");
  }

  if (strncmp("BINARY", tmp, 6) == 0 ) {
    colloid_io_format_input_binary_set();
    info("Input format:  binary\n");
  }

  if (strncmp("BINARY_SERIAL", tmp, 13) == 0 ) {
    colloid_io_format_input_binary_set();
    colloid_io_format_input_serial_set();
    info("Input file:    serial single file\n");
  }

  nuser = RUN_get_string_parameter("colloid_io_format_output", tmp, 256);

  if (nuser == 0) {
    info("Output format:      ascii\n");
  }

  if (strncmp("ASCII",  tmp, 5) == 0 ) {
    colloid_io_format_output_ascii_set();
    info("Output format: ascii\n");
  }

  if (strncmp("BINARY", tmp, 6) == 0 ) {
    colloid_io_format_output_binary_set();
    info("Output format: binary\n");
  }

  info("\n");

  return;
}
