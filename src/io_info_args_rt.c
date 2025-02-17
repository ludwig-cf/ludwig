/*****************************************************************************
 *
 *  io_info_args_rt.c
 *
 *  Initialisation of the container for i/o information.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020-2025 The University of Edinburgh
 *
 *  Contribuiting authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

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

int io_info_args_rt(rt_t * rt, rt_enum_t lv, const char * stub,
		    io_info_rw_enum_t rw, io_info_args_t * args) {

  assert(rt);
  assert(stub);
  assert(args);
  assert(strlen(stub) < BUFSIZ/2);

  /* User defaults */
  /* Specific options for this info type */

  if (rw == IO_INFO_READ_WRITE || rw == IO_INFO_READ_ONLY) {
    io_info_args_rt_input(rt, lv, stub, args);
  }

  if (rw == IO_INFO_READ_WRITE || rw == IO_INFO_WRITE_ONLY) {
    io_info_args_rt_output(rt, lv, stub, args);
  }

  /* Now the io grid (shared between input and output) */

  {
    char key[BUFSIZ] = {0};

    sprintf(key, "%s_io_grid", stub);
    io_info_args_rt_iogrid(rt, lv, "default_io_grid", args->grid);
    io_info_args_rt_iogrid(rt, lv, key, args->grid);
  }

  args->input.iogrid[0]  = args->grid[0];
  args->input.iogrid[1]  = args->grid[1];
  args->input.iogrid[2]  = args->grid[2];
  args->output.iogrid[0] = args->grid[0];
  args->output.iogrid[1] = args->grid[1];
  args->output.iogrid[2] = args->grid[2];

  /* i/o frequency (output only) */

  {
    char key[BUFSIZ] = {0};

    sprintf(key, "%s_io_freq", stub);
    io_info_args_rt_iofreq(rt, lv, "default_io_freq", &args->iofreq);
    io_info_args_rt_iofreq(rt, lv, key, &args->iofreq);
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_info_args_rt_input
 *
 *****************************************************************************/

int io_info_args_rt_input(rt_t * rt, rt_enum_t lv, const char * stub,
			  io_info_args_t * args) {

  char stub_input[BUFSIZ] = {0};

  assert(rt);
  assert(stub);
  assert(args);

  assert(strlen(stub) < BUFSIZ/2);

  sprintf(stub_input, "%s_input", stub);

  /* This is in order of increasing precedence ... */
  io_options_rt(rt, lv, "default",       &args->input);
  io_options_rt(rt, lv, "default_input", &args->input);
  io_options_rt(rt, lv, stub,            &args->input);
  io_options_rt(rt, lv, stub_input,      &args->input);

  return 0;
}

/*****************************************************************************
 *
 *  io_info_args_rt_output
 *
 *****************************************************************************/

int io_info_args_rt_output(rt_t * rt, rt_enum_t lv, const char * stub,
			   io_info_args_t * args) {

  char stub_output[BUFSIZ] = {0};

  assert(rt);
  assert(stub);
  assert(args);

  assert(strlen(stub) < BUFSIZ/2);

  sprintf(stub_output, "%s_output", stub);

  /* In order of increasing precedence .. */
  io_options_rt(rt, lv, "default",        &args->output);
  io_options_rt(rt, lv, "default_output", &args->output);
  io_options_rt(rt, lv, stub,             &args->output);
  io_options_rt(rt, lv, stub_output,      &args->output);

  return 0;
}

/*****************************************************************************
 *
 *  io_info_args_rt_iogrid
 *
 *****************************************************************************/

int io_info_args_rt_iogrid(rt_t * rt, rt_enum_t lv, const char * key,
			   int grid[3]) {

  int key_present = 0;
  int iogrid[3] = {0};   /* invalid */
  int ifail = -1;        /* Return -1 for no key; +1 for invalid key; 0 ok */

  assert(rt);
  assert(key);

  key_present = rt_int_parameter_vector(rt, key, iogrid);

  if (key_present) {
    if (io_info_args_iogrid_valid(iogrid) == 1) {
      grid[0] = iogrid[0];
      grid[1] = iogrid[1];
      grid[2] = iogrid[2];
      ifail = 0;
    }
    else {
      rt_vinfo(rt, lv, "I/O grid key present but invalid\n");
      rt_vinfo(rt, lv, "key: %s\n", key);
      rt_vinfo(rt, lv, "value: %d %d %d\n", iogrid[0], iogrid[1], iogrid[2]);
      rt_vinfo(rt, lv, "Must be greater than zero in each dimension\n");
      rt_fatal(rt, lv, "Please check the input file and try again!\n");
      ifail = +1;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_info_args_rt_iofreq
 *
 *  Obtain a valid i/o frequency, if present.
 *
 *****************************************************************************/

int io_info_args_rt_iofreq(rt_t * rt, rt_enum_t lv, const char * key,
			   int * iofreq) {

  int ifail = -1;      /* -1 for no key; 0 for valid key; +1 if invalid */
  int ival  =  0;
  int key_present = 0;

  key_present = rt_int_parameter(rt, key, &ival);

  if (key_present) {
    if (ival >= 0) {
      *iofreq = ival;
      ifail = 0;
    }
    else {
      rt_vinfo(rt, lv, "I/O freq key present but is invalid\n");
      rt_vinfo(rt, lv, "key: %s\n", key);
      rt_fatal(rt, lv, "The value must be a non-negative integer.\n");
      rt_fatal(rt, lv, "Please check the input file and try again!\n");
      ifail = 1;
    }
  }

  return ifail;
}
