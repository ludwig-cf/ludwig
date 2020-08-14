/*****************************************************************************
 *
 *  test_io_options.c
 *
 *  For i/o options container.
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

#include "pe.h"
#include "io_options.h"

__host__ int test_io_options_mode_valid(void);
__host__ int test_io_options_record_format_valid(void);
__host__ int test_io_options_metadata_version_valid(void);
__host__ int test_io_options_default(void);

/*****************************************************************************
 *
 *  test_io_options_suite
 *
 *****************************************************************************/

__host__ int test_io_options_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_io_options_mode_valid();
  test_io_options_record_format_valid();
  test_io_options_metadata_version_valid();
  test_io_options_default();

  pe_info(pe, "PASS     ./unit/test_io_options\n");

  pe_free(pe);

  return 0;
}



/*****************************************************************************
 *
 *  test_io_options_mode_valid
 *
 *****************************************************************************/

__host__ int test_io_options_mode_valid(void) {

  io_mode_enum_t mode = IO_MODE_INVALID;

  mode = IO_MODE_SINGLE;
  assert(io_options_mode_valid(mode));

  mode = IO_MODE_MULTIPLE;
  assert(io_options_mode_valid(mode));

  mode = IO_MODE_INVALID;
  assert(io_options_mode_valid(mode) == 0);

  return 0;
}


/*****************************************************************************
 *
 *  test_io_options_record_format_valid
 *
 *****************************************************************************/

__host__ int test_io_options_record_format_valid(void) {

  io_record_format_enum_t iorformat = IO_RECORD_INVALID;

  iorformat = IO_RECORD_ASCII;
  assert(io_options_record_format_valid(iorformat));

  iorformat = IO_RECORD_BINARY;
  assert(io_options_record_format_valid(iorformat));

  iorformat = IO_RECORD_INVALID;
  assert(io_options_record_format_valid(iorformat) == 0);

  return 0;
}


/*****************************************************************************
 *
 *  test_io_options_metadata_version_valid
 *
 *****************************************************************************/

__host__ int test_io_options_metadata_version_valid(void) {

  io_options_t opts = {};

  /* Wrong */
  opts.mode             = IO_MODE_SINGLE;
  opts.metadata_version = IO_METADATA_MULTI_V1;

  assert(io_options_metadata_version_valid(&opts) == 0);

  /* Wrong */
  opts.mode             = IO_MODE_MULTIPLE;
  opts.metadata_version = IO_METADATA_SINGLE_V1;

  assert(io_options_metadata_version_valid(&opts) == 0);

  /* Right */
  opts.mode             = IO_MODE_SINGLE;
  opts.metadata_version = IO_METADATA_SINGLE_V1;

  assert(io_options_metadata_version_valid(&opts));

  /* Right */
  opts.mode             = IO_MODE_MULTIPLE;
  opts.metadata_version = IO_METADATA_MULTI_V1;

  assert(io_options_metadata_version_valid(&opts));

  return 0;
}


/*****************************************************************************
 *
 *  test_io_options_default
 *
 *****************************************************************************/

__host__ int test_io_options_default(void) {

  io_options_t opts = io_options_default();

  assert(io_options_mode_valid(opts.mode));
  assert(io_options_record_format_valid(opts.iorformat));
  assert(io_options_metadata_version_valid(&opts));
  assert(io_options_valid(&opts));

  assert(opts.report == 0);
  assert(opts.asynchronous == 0);

  return 0;
}
