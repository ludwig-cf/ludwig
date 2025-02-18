/*****************************************************************************
 *
 *  io_info_args.c
 *
 *  Default container values.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "io_info_args.h"

/*****************************************************************************
 *
 *  io_info_args_default
 *
 *****************************************************************************/

io_info_args_t io_info_args_default(void) {

  io_info_args_t args = {.input  = io_options_default(),
			 .output = io_options_default(),
			 .grid   = {1, 1, 1},
			 .iofreq = 0};
  return args;
}

/*****************************************************************************
 *
 *  io_info_args_iogrid_valid
 *
 *  Sanity only.
 *
 *****************************************************************************/

int io_info_args_iogrid_valid(int iogrid[3]) {

  int valid = 1;

  if (iogrid[0] < 1) valid = 0;
  if (iogrid[1] < 1) valid = 0;
  if (iogrid[2] < 1) valid = 0;

  return valid;
}
