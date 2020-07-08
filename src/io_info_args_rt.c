/*****************************************************************************
 *
 *  io_info_args_rt.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020 The University of Edinburgh
 *
 *  Contribuiting authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "io_info_args_rt.h"

/*****************************************************************************
 *
 *  io_info_args_rt
 *
 *  Look for a series of key in the input to determine the io arguments
 *  for the type with stub name "stub".
 *
 *****************************************************************************/

__host__ int io_info_args_rt(pe_t * pe, rt_t * rt, const char * stub,
			     io_info_rw_enum_t rw, io_info_args_t * args) {

  int iogrid[3] = {0};
  int sz = 3*sizeof(int);
  char stubi[BUFSIZ] = {0};
  char stubo[BUFSIZ] = {0};

  assert(pe);
  assert(rt);
  assert(stub);
  assert(args);

  assert(strlen(stub) < BUFSIZ/2);
  assert(0); /* io_options_rt() must not overwrite args */

  sprintf(stubi, "%s_input", stub);
  sprintf(stubo, "%s_output", stub);

  /* User defaults */
  /* Specific options for this info type */

  io_options_rt(pe, rt, "default",        &args->input);
  io_options_rt(pe, rt, "default",        &args->output);
  io_options_rt(pe, rt, "default_input",  &args->input);
  io_options_rt(pe, rt, "default_output", &args->output);
  io_options_rt(pe, rt, stubi,            &args->input);
  io_options_rt(pe, rt, stubo,            &args->output);

  /* Now the io grid (shared between input and output) */

  io_info_args_rt_iogrid(pe, rt, "default_io_grid", iogrid);
  if (io_info_args_iogrid_valid(iogrid)) memcpy(args->grid, iogrid, sz);

  sprintf(stubi, "%s_io_grid", stub);
  io_info_args_rt_iogrid(pe, rt, stubi, iogrid);
  if (io_info_args_iogrid_valid(iogrid)) memcpy(args->grid, iogrid, sz);

  return 0;
}

/*****************************************************************************
 *
 *  io_info_args_rt_iogrid
 *
 *  Could return an error flag here if an invalid grid.
 *
 *****************************************************************************/

__host__ int io_info_args_rt_iogrid(pe_t * pe, rt_t * rt, const char * key,
			            int iogrid[3]) {

  int key_present = 0;
  int user_grid[3] = {0};

  assert(pe);
  assert(rt);
  assert(key);

  key_present = rt_int_parameter_vector(rt, key, user_grid);

  if (key_present && io_info_args_iogrid_valid(user_grid) == 0) {
    pe_info(pe, "I/O grid key present but invalid\n");
    pe_info(pe, "key: %s\n", key);
    pe_info(pe, "value: %d %d %d\n", user_grid[0], user_grid[1], user_grid[2]);
    pe_info(pe, "Must be greater than zero in each dimension\n");
    pe_fatal(pe, "Please check the input file and try again!\n");
  }

  iogrid[0] = user_grid[0];
  iogrid[1] = user_grid[1];
  iogrid[2] = user_grid[2];

  return 0;
}
