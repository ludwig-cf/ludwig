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
 *  (c) 2020-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "io_options.h"

__host__ int test_io_options_mode_valid(void);
__host__ int test_io_options_record_format_valid(void);
__host__ int test_io_options_metadata_version_valid(void);
__host__ int test_io_options_default(void);
__host__ int test_io_options_with_mode(void);
__host__ int test_io_options_with_format(void);
__host__ int test_io_options_with_iogrid(void);
__host__ int test_io_mode_to_string(void);
__host__ int test_io_mode_from_string(void);
__host__ int test_io_record_format_to_string(void);
__host__ int test_io_record_format_from_string(void);
__host__ int test_io_options_to_json(void);
__host__ int test_io_options_from_json(void);

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
  test_io_mode_to_string();
  test_io_mode_from_string();
  test_io_record_format_to_string();
  test_io_record_format_from_string();
  test_io_options_to_json();
  test_io_options_from_json();

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

  io_mode_enum_t mode3 = IO_MODE_MPIIO;
  io_mode_enum_t mode9 = IO_MODE_INVALID;
  int isvalid = 0;

  isvalid = io_options_mode_valid(mode3);
  assert(isvalid == 1);

  isvalid = io_options_mode_valid(mode9);
  assert(isvalid == 0);

  /* nb., a declaration "io_options_t io  = {0};" will not pass
   * muster for C++ (bad conversion to enum_t). One can instead do
   * "io_options_t io = {IO_MODE_INVALID};" providing ... */

  assert(IO_MODE_INVALID == 0);

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

  int ifail = 0;
  io_options_t opts = io_options_default();
  int isvalid = 0;

  /* Right */
  opts.mode             = IO_MODE_MPIIO;
  opts.metadata_version = IO_METADATA_V2;

  isvalid = io_options_metadata_version_valid(&opts);
  if (isvalid == 0) ifail = -1;
  assert(ifail == 0);

  return ifail;
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

/*****************************************************************************
 *
 *  test_io_mode_to_string
 *
 *****************************************************************************/

__host__ int test_io_mode_to_string(void) {

  int ifail = 0;

  /* Test each case in turn */

  {
    const char * str = NULL;
    str = io_mode_to_string(IO_MODE_MPIIO);
    ifail = strcmp(str, "mpiio");
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_mode_from_string
 *
 *****************************************************************************/

__host__ int test_io_mode_from_string(void) {

  int ifail = 0;

  {
    io_mode_enum_t mode = io_mode_from_string("MPIIO");
    if (mode != IO_MODE_MPIIO) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_record_format_to_string
 *
 *****************************************************************************/

__host__ int test_io_record_format_to_string(void) {

  int ifail = 0;

  {
    const char * str = io_record_format_to_string(IO_RECORD_ASCII);
    ifail = strcmp(str, "ascii");
    assert(ifail == 0);
  }

  {
    const char * str = io_record_format_to_string(IO_RECORD_BINARY);
    ifail = strcmp(str, "binary");
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_record_format_from_string
 *
 *****************************************************************************/

__host__ int test_io_record_format_from_string(void) {

  int ifail = 0;

  {
    io_record_format_enum_t ior = io_record_format_from_string("ASCII");
    if (ior != IO_RECORD_ASCII) ifail = -1;
    assert(ifail == 0);
  }

  {
    io_record_format_enum_t ior = io_record_format_from_string("BINARY");
    if (ior != IO_RECORD_BINARY) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_options_to_json
 *
 *****************************************************************************/

__host__ int test_io_options_to_json(void) {

  int ifail = 0;

  io_options_t opts = io_options_default();
  cJSON * json = NULL;

  ifail = io_options_to_json(&opts, &json);
  assert(ifail == 0);

  {
    /* A somewhat circular test which we excuse by saying the test on
     * io_options_from_json() is not also circular */
    io_options_t check = {IO_MODE_INVALID};
    io_options_from_json(json, &check);
    assert(check.mode == io_mode_default());
    assert(check.iorformat == io_record_format_default());
    assert(check.metadata_version == io_metadata_version_default());
    assert(check.report == opts.report);
    assert(check.asynchronous == opts.asynchronous);
    assert(check.compression_levl == opts.compression_levl);
    assert(check.iogrid[0] == 1);
    assert(check.iogrid[1] == 1);
    assert(check.iogrid[2] == 1);
  }

  cJSON_Delete(json);

  return ifail;
}

/*****************************************************************************
 *
 *  test_io_options_from_json
 *
 *****************************************************************************/

__host__ int test_io_options_from_json(void) {

  int ifail = 0;
  const char * jstr = "{\"Mode\": \"single\","
                      "\"Record format\": \"binary\","
                      "\"Metadata version\": 2,"
                      "\"Report\": false, "
                      "\"Asynchronous\": true,"
                      "\"Compression level\": 9,"
                      "\"I/O grid\": [2, 3, 4] }";

  cJSON * json = cJSON_Parse(jstr);
  assert(json);

  {
    /* Convert to options and test */
    io_options_t opts = {IO_MODE_INVALID};
    ifail = io_options_from_json(json, &opts);
    assert(opts.metadata_version == 2);
    assert(opts.report           == 0);
    assert(opts.asynchronous         );
    assert(opts.compression_levl == 9);
    assert(opts.iogrid[0]        == 2);
    assert(opts.iogrid[1]        == 3);
    assert(opts.iogrid[2]        == 4);
  }

  cJSON_Delete(json);

  return ifail;
}
