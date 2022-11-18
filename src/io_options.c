/*****************************************************************************
 *
 *  io_options.c
 *
 *  Routines for io_options_t container.
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
#include "io_options.h"

/* Defaults */

#define IO_MODE_DEFAULT()             IO_MODE_SINGLE
#define IO_RECORD_FORMAT_DEFAULT()    IO_RECORD_BINARY
#define IO_METADATA_VERSION_DEFAULT() IO_METADATA_SINGLE_V1
#define IO_REPORT_DEFAULT()           0
#define IO_ASYNCHRONOUS_DEFAULT()     0
#define IO_COMPRESSION_LEVL_DEFAULT() 0
#define IO_GRID_DEFAULT()             {1, 1, 1}
#define IO_OPTIONS_DEFAULT()         {IO_MODE_DEFAULT(), \
                                      IO_RECORD_FORMAT_DEFAULT(), \
                                      IO_METADATA_VERSION_DEFAULT(),\
                                      IO_REPORT_DEFAULT(), \
                                      IO_ASYNCHRONOUS_DEFAULT(),     \
                                      IO_COMPRESSION_LEVL_DEFAULT(), \
                                      IO_GRID_DEFAULT()}

/*****************************************************************************
 *
 *  io_mode_default
 *
 *****************************************************************************/

__host__ io_mode_enum_t io_mode_default(void) {

  return IO_MODE_DEFAULT();
}

/*****************************************************************************
 *
 *  io_record_format_default
 *
 *****************************************************************************/

__host__ io_record_format_enum_t io_record_format_default(void) {

  return IO_RECORD_FORMAT_DEFAULT();
}


/*****************************************************************************
 *
 *  io_metadata_version_default
 *
 *****************************************************************************/

__host__ io_metadata_version_enum_t io_metadata_version_default(void) {

  return IO_METADATA_VERSION_DEFAULT();
}

/*****************************************************************************
 *
 *  io_options_default
 *
 *****************************************************************************/

__host__ io_options_t io_options_default(void) {

  io_options_t opts = IO_OPTIONS_DEFAULT();

  return opts;
}


/*****************************************************************************
 *
 *  io_options_valid
 *
 *  Return zero if options are invalid.
 *
 *****************************************************************************/

__host__ int io_options_valid(const io_options_t * options) {

  int valid = 0;

  assert(options);

  valid += io_options_mode_valid(options->mode);
  valid += io_options_record_format_valid(options->iorformat);
  valid += io_options_metadata_version_valid(options);

  return valid;
}

/*****************************************************************************
 *
 *  io_options_mode_valid
 *
 *****************************************************************************/

__host__ int io_options_mode_valid(io_mode_enum_t mode) {

  int valid = 0;

  valid += (mode == IO_MODE_SINGLE);
  valid += (mode == IO_MODE_MULTIPLE);
  valid += (mode == IO_MODE_MPIIO);

  return valid;
}


/*****************************************************************************
 *
 *  io_options_record_format_valid
 *
 *  Return non-zero for a valid format.
 *
 *****************************************************************************/

__host__ int io_options_record_format_valid(io_record_format_enum_t ioformat) {

  int valid = 0;

  valid += (ioformat == IO_RECORD_ASCII);
  valid += (ioformat == IO_RECORD_BINARY);

  return valid;
}

/*****************************************************************************
 *
 *  io_options_metadata_version_valid
 *
 *  Return non-zero for a valid metadata version.
 *
 *****************************************************************************/

__host__ int io_options_metadata_version_valid(const io_options_t * options) {

  int valid = 0;

  assert(options);

  /* Should be consistent with mode */

  switch (options->metadata_version) {

  case IO_METADATA_SINGLE_V1:
    valid = (options->mode == IO_MODE_SINGLE);
    break;

  case IO_METADATA_MULTI_V1:
    valid = (options->mode == IO_MODE_MULTIPLE);
    break;

  case IO_METADATA_V2:
    valid = (options->mode == IO_MODE_MPIIO);
    break;

  default:
    ;
  }

  return valid;
}

/*****************************************************************************
 *
 *  io_options_with_mode
 *
 *  Return a default for the given mode
 *
 *****************************************************************************/

__host__ io_options_t io_options_with_mode(io_mode_enum_t mode) {

  io_options_t options = io_options_default();

  switch (mode) {
  case IO_MODE_SINGLE:
    options.mode             = IO_MODE_SINGLE;
    options.iorformat        = IO_RECORD_BINARY;
    options.metadata_version = IO_METADATA_SINGLE_V1;
    /* otherwise defaults */
    break;
  case IO_MODE_MULTIPLE:
    options.mode             = IO_MODE_MULTIPLE;
    options.iorformat        = IO_RECORD_BINARY;
    options.metadata_version = IO_METADATA_MULTI_V1;
    /* otherwise defaults */
    break;
  case IO_MODE_MPIIO:
    options.mode             = IO_MODE_MPIIO;
    options.iorformat        = IO_RECORD_BINARY;
    options.metadata_version = IO_METADATA_V2;
    options.report           = 1;
    options.asynchronous     = 0;
    options.compression_levl = 0;
    break;
  default:
    /* User error ... */
    options.mode             = IO_MODE_INVALID;
  }

  return options;
}

/*****************************************************************************
 *
 *  io_options_with_format
 *
 *  Actually with mode and format ...
 *
 *****************************************************************************/

__host__ io_options_t io_options_with_format(io_mode_enum_t mode,
					     io_record_format_enum_t iorf) {

  io_options_t options = io_options_with_mode(mode);

  options.iorformat = iorf;

  return options;
}

/*****************************************************************************
 *
 *  io_options_with_iogrid
 *
 *****************************************************************************/

__host__ io_options_t io_options_with_iogrid(io_mode_enum_t mode,
					     io_record_format_enum_t iorf,
					     int iogrid[3]) {

  io_options_t options = io_options_with_format(mode, iorf);

  options.iogrid[0] = iogrid[0];
  options.iogrid[1] = iogrid[1];
  options.iogrid[2] = iogrid[2];

  return options;
}
