/*****************************************************************************
 *
 *  test_io_options_rt.c
 *
 *  Parse run time i/o options.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "io_options_rt.h"

__host__ int test_io_options_rt_mode(pe_t * pe);
__host__ int test_io_options_rt_rformat(pe_t * pe);
__host__ int test_io_options_rt_default(pe_t * pe);
__host__ int test_io_options_rt(pe_t * pe);

/*****************************************************************************
 *
 *  test_io_options_rt_suite
 *
 *****************************************************************************/

__host__ int test_io_options_rt_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_io_options_rt_mode(pe);
  test_io_options_rt_rformat(pe);
  test_io_options_rt_default(pe);
  test_io_options_rt(pe);

  pe_info(pe, "PASS     ./unit/test_io_options_rt\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_options_rt_mode
 *
 *****************************************************************************/

__host__ int test_io_options_rt_mode(pe_t * pe) {

  rt_t * rt = NULL;
  io_mode_enum_t mode = io_mode_default();

  assert(pe);

  rt_create(pe, &rt);
  rt_add_key_value(rt, "example_input_io_mode", "SINGLE");
  rt_add_key_value(rt, "example_output_io_mode", "MULTIPLE");

  io_options_rt_mode(pe, rt, "example_input_io_mode", &mode);
  assert(mode == IO_MODE_SINGLE);

  io_options_rt_mode(pe, rt, "example_output_io_mode", &mode);
  assert(mode == IO_MODE_MULTIPLE);

  rt_free(rt);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_options_rt_rformat
 *
 *****************************************************************************/

__host__ int test_io_options_rt_rformat(pe_t * pe) {

  rt_t * rt = NULL;
  io_record_format_enum_t iorformat = io_record_format_default();

  assert(pe);

  rt_create(pe, &rt);

  rt_add_key_value(rt, "lb_input_io_format", "ASCII");
  rt_add_key_value(rt, "lb_output_io_format", "BINARY");

  io_options_rt_record_format(pe, rt, "lb_input_io_format", &iorformat);
  assert(iorformat == IO_RECORD_ASCII);

  io_options_rt_record_format(pe, rt, "lb_output_io_format", &iorformat);
  assert(iorformat == IO_RECORD_BINARY);

  rt_free(rt);

  return 0;
}

/*****************************************************************************
 *
 *  test_io_options_rt_default
 *
 *  Check we get the defaults if there is no user input.
 *
 *****************************************************************************/

__host__ int test_io_options_rt_default(pe_t * pe) {

  rt_t * rt = NULL;
  io_options_t opts = {};
  io_options_t defs = io_options_default();

  assert(pe);

  rt_create(pe, &rt);

  io_options_rt(pe, rt, "default", &opts);

  assert(opts.mode             == defs.mode);
  assert(opts.iorformat        == defs.iorformat);
  assert(opts.metadata_version == defs.metadata_version);
  assert(opts.report           == defs.report);
  assert(opts.asynchronous     == defs.asynchronous);

  rt_free(rt);

  return defs.report; /* Return 0 (and prevent unused variable warning) */
}


/*****************************************************************************
 *
 *  test_io_options_rt
 *
 *  Test some non-default key/values
 *
 *****************************************************************************/

__host__ int test_io_options_rt(pe_t * pe) {

  rt_t * rt = NULL;
  io_options_t opts = {};

  assert(pe);

  rt_create(pe, &rt);

  rt_add_key_value(rt, "default_io_mode",   "multiple");
  rt_add_key_value(rt, "default_io_format", "ascii");
  rt_add_key_value(rt, "default_io_report", "yes"); 

  io_options_rt(pe, rt, "default", &opts);

  assert(opts.mode             == IO_MODE_MULTIPLE);
  assert(opts.iorformat        == IO_RECORD_ASCII);
  assert(opts.metadata_version == IO_METADATA_MULTI_V1);
  assert(opts.report           == 1);
  assert(opts.asynchronous     == 0);
  rt_free(rt);

  return 0;
}

