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
 *  (c) 2020-2022 The University of Edinburgh
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
__host__ int test_io_options_with_mode(void);
__host__ int test_io_options_with_format(void);
__host__ int test_io_options_with_iogrid(void);

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
  test_io_options_with_mode();
  test_io_options_with_format();
  test_io_options_with_iogrid();

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

  io_mode_enum_t mode1 = IO_MODE_SINGLE;
  io_mode_enum_t mode2 = IO_MODE_MULTIPLE;
  io_mode_enum_t mode3 = IO_MODE_MPIIO;
  io_mode_enum_t mode9 = IO_MODE_INVALID;
  int isvalid = 0;

  isvalid = io_options_mode_valid(mode1);
  assert(isvalid);

  isvalid = io_options_mode_valid(mode2);
  assert(isvalid);

  isvalid = io_options_mode_valid(mode3);
  assert(isvalid == 1);

  isvalid = io_options_mode_valid(mode9);
  assert(isvalid == 0);

  return isvalid;
}


/*****************************************************************************
 *
 *  test_io_options_record_format_valid
 *
 *****************************************************************************/

__host__ int test_io_options_record_format_valid(void) {

  io_record_format_enum_t iorformat1 = IO_RECORD_ASCII;
  io_record_format_enum_t iorformat2 = IO_RECORD_BINARY;
  io_record_format_enum_t iorformat3 = IO_RECORD_INVALID;
  int isvalid = 0;

  isvalid = io_options_record_format_valid(iorformat1);
  assert(isvalid);

  isvalid = io_options_record_format_valid(iorformat2);
  assert(isvalid);

  isvalid = io_options_record_format_valid(iorformat3);
  assert(isvalid == 0);

  return isvalid;
}


/*****************************************************************************
 *
 *  test_io_options_metadata_version_valid
 *
 *****************************************************************************/

__host__ int test_io_options_metadata_version_valid(void) {

  io_options_t opts = io_options_default();
  int isvalid = 0;

  /* Wrong */
  opts.mode             = IO_MODE_SINGLE;
  opts.metadata_version = IO_METADATA_MULTI_V1;

  isvalid = io_options_metadata_version_valid(&opts);
  assert(isvalid == 0);

  /* Wrong */
  opts.mode             = IO_MODE_MULTIPLE;
  opts.metadata_version = IO_METADATA_SINGLE_V1;

  isvalid = io_options_metadata_version_valid(&opts);
  assert(isvalid == 0);

  /* Right */
  opts.mode             = IO_MODE_SINGLE;
  opts.metadata_version = IO_METADATA_SINGLE_V1;

  assert(io_options_metadata_version_valid(&opts));

  /* Right */
  opts.mode             = IO_MODE_MULTIPLE;
  opts.metadata_version = IO_METADATA_MULTI_V1;

  assert(io_options_metadata_version_valid(&opts));

  /* Right */
  opts.mode             = IO_MODE_MPIIO;
  opts.metadata_version = IO_METADATA_V2;

  assert(io_options_metadata_version_valid(&opts));

  return isvalid;
}


/*****************************************************************************
 *
 *  test_io_options_default
 *
 *****************************************************************************/

__host__ int test_io_options_default(void) {

  io_options_t opts = io_options_default();

  /* If entries are changed in the struct, the tests should be updated... */
  assert(sizeof(io_options_t) == 36);

  assert(io_options_mode_valid(opts.mode));
  assert(io_options_record_format_valid(opts.iorformat));
  assert(io_options_metadata_version_valid(&opts));
  assert(io_options_valid(&opts));

  assert(opts.report == 0);
  assert(opts.asynchronous == 0);
  assert(opts.compression_levl == 0);
  assert(opts.iogrid[0] == 1);
  assert(opts.iogrid[1] == 1);
  assert(opts.iogrid[2] == 1);

  return opts.report;
}

/*****************************************************************************
 *
 *  test_io_options_with_mode
 *
 *****************************************************************************/

__host__ int test_io_options_with_mode(void) {

  int ifail = 0;

  {
    /* IO_MODE_SINGLE */
    io_options_t options = io_options_with_mode(IO_MODE_SINGLE);
    assert(options.mode              == IO_MODE_SINGLE);
    assert(options.iorformat         == IO_RECORD_BINARY);
    assert(options.metadata_version  == IO_METADATA_SINGLE_V1);
    assert(options.report            == 0);
    assert(options.asynchronous      == 0);
    assert(options.compression_levl  == 0);
    assert(options.iogrid[0]         == 1);
    assert(options.iogrid[1]         == 1);
    assert(options.iogrid[2]         == 1);
    ifail = options.report;
  }

  {
    /* IO_MODE_MULTIPLE */
    io_options_t options = io_options_with_mode(IO_MODE_MULTIPLE);
    assert(options.mode              == IO_MODE_MULTIPLE);
    assert(options.iorformat         == IO_RECORD_BINARY);
    assert(options.metadata_version  == IO_METADATA_MULTI_V1);
    assert(options.report            == 0);
    assert(options.asynchronous      == 0);
    assert(options.compression_levl  == 0);
    assert(options.iogrid[0]         == 1);
    assert(options.iogrid[1]         == 1);
    assert(options.iogrid[2]         == 1);
    ifail = options.report;
  }

  {
    /* IO_MODE_MPIIO */
    io_options_t options = io_options_with_mode(IO_MODE_MPIIO);
    assert(options.mode              == IO_MODE_MPIIO);
    assert(options.iorformat         == IO_RECORD_BINARY);
    assert(options.metadata_version  == IO_METADATA_V2);
    assert(options.report            == 1);
    assert(options.asynchronous      == 0);
    assert(options.compression_levl  == 0);
    assert(options.iogrid[0]         == 1);
    assert(options.iogrid[1]         == 1);
    assert(options.iogrid[2]         == 1);
    ifail = options.asynchronous;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_options_with_format
 *
 *****************************************************************************/

__host__ int test_io_options_with_format(void) {

  int ifail = 0;
  io_mode_enum_t mode = IO_MODE_MPIIO;

  {
    io_options_t opts = io_options_with_format(mode, IO_RECORD_ASCII);
    assert(opts.mode      == mode);
    assert(opts.iorformat == IO_RECORD_ASCII);
    if (opts.iorformat != IO_RECORD_ASCII) ifail = -1;
  }

  {
    io_options_t opts = io_options_with_format(mode, IO_RECORD_BINARY);
    assert(opts.mode      == mode);
    assert(opts.iorformat == IO_RECORD_BINARY);
    if (opts.iorformat != IO_RECORD_BINARY) ifail = -2;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_options_with_iogrid
 *
 *****************************************************************************/

__host__ int test_io_options_with_iogrid(void) {

  int ifail = 0;
  io_mode_enum_t mode = IO_MODE_MPIIO;
  io_record_format_enum_t ior = IO_RECORD_ASCII;

  {
    int iogrid[3] = {2, 3, 4};
    io_options_t opts = io_options_with_iogrid(mode, ior, iogrid);
    assert(opts.mode        == mode);
    assert(opts.iorformat   == ior);
    assert(opts.iogrid[0]   == iogrid[0]);
    assert(opts.iogrid[1]   == iogrid[1]);
    assert(opts.iogrid[2]   == iogrid[2]);
    if (opts.iogrid[0]*opts.iogrid[1]*opts.iogrid[2] != 24) ifail = -1;
  }

  return ifail;
}
