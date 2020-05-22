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
 *  (c) 2020 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "io_options.h"

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
  valid += io_options_rformat_valid(options->iorformat);
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

  return valid;
}


/*****************************************************************************
 *
 *  io_options_rformat_valid
 *
 *  Return non-zero for a valid format.
 *
 *****************************************************************************/

__host__ int io_options_rformat_valid(io_rformat_enum_t iorformat) {

  int valid = 0;

  valid += (iorformat == IO_RECORD_ASCII);
  valid += (iorformat == IO_RECORD_BINARY);

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

  default:
    ;
  }

  return valid;
}
