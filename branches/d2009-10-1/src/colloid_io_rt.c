/*****************************************************************************
 *
 *  colloid_io_rt.c
 *
 *  Run time colloid I/O settings.
 *
 *  $Id: colloid_io_rt.c,v 1.1.2.1 2010-05-12 15:16:46 kevin Exp $
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
 *  The decomposition is pending.
 *
 *****************************************************************************/

void colloid_io_run_time(void) {

  int  nuser;
  char tmp[256];

  /* Defaults */

  colloid_io_format_input_ascii_serial_set();
  colloid_io_format_output_ascii_set();

  info("\n");
  info("Colloid I/O settings\n");
  info("--------------------\n");

  nuser = RUN_get_string_parameter("colloid_io_format_input", tmp, 256);

  if (nuser == 0) {
    info("Input format:       ascii serial\n");
  }

  if (strncmp("ASCII",  tmp, 5) == 0 ) {
    colloid_io_format_input_ascii_set();
    info("Input format:  ascii\n");
  }

  if (strncmp("ASCII_SERIAL",  tmp, 12) == 0 ) {
    colloid_io_format_input_ascii_serial_set();
    info("Input format:  ascii serial\n");
  }

  if (strncmp("BINARY", tmp, 6) == 0 ) {
    colloid_io_format_input_binary_set();
    info("Input format:  binary\n");
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
