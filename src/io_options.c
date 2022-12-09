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
#include <string.h>

#include "io_options.h"
#include "util.h"
#include "util_json.h"

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
  valid += (mode == IO_MODE_ANSI);
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
  case IO_MODE_ANSI:
    options.mode             = IO_MODE_ANSI;
    options.iorformat        = IO_RECORD_BINARY;
    options.metadata_version = IO_METADATA_SINGLE_V1;
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

/*****************************************************************************
 *
 *  io_mode_to_string
 *
 *  Always return a lower-case string.
 *
 *****************************************************************************/

__host__ const char * io_mode_to_string(io_mode_enum_t mode) {

  const char * str = NULL;

  switch (mode) {
  case IO_MODE_SINGLE:
    str = "single";
    break;
  case IO_MODE_MULTIPLE:
    str = "multiple";
    break;
  case IO_MODE_ANSI:
    str = "ansi";
    break;
  case IO_MODE_MPIIO:
    str = "mpiio";
    break;
  default:
    str = "invalid";
  }

  return str;
}

/*****************************************************************************
 *
 *  io_mode_from_string
 *
 *****************************************************************************/

__host__ io_mode_enum_t io_mode_from_string(const char * str) {

  int mode = IO_MODE_INVALID;
  char value[BUFSIZ] = {0};

  assert(str);

  strncpy(value, str, BUFSIZ-1);
  util_str_tolower(value, strlen(value));

  if (strcmp(value, "single")   == 0) mode = IO_MODE_SINGLE;
  if (strcmp(value, "multiple") == 0) mode = IO_MODE_MULTIPLE;
  if (strcmp(value, "ansi")     == 0) mode = IO_MODE_ANSI;
  if (strcmp(value, "mpiio")    == 0) mode = IO_MODE_MPIIO;

  return mode;
}

/*****************************************************************************
 *
 *  io_record_format_to_string
 *
 *****************************************************************************/

const char * io_record_format_to_string(io_record_format_enum_t ior) {

  const char * str = NULL;

  switch (ior) {
  case IO_RECORD_ASCII:
    str = "ascii";
    break;
  case IO_RECORD_BINARY:
    str = "binary";
    break;
  default:
    str = "invalid";
  }

  return str;
}

/*****************************************************************************
 *
 *  io_record_format_from_string
 *
 *****************************************************************************/

io_record_format_enum_t io_record_format_from_string(const char * str) {

  io_record_format_enum_t ior = IO_RECORD_INVALID;
  char value[BUFSIZ] = {0};

  assert(str);

  strncpy(value, str, BUFSIZ-1);
  util_str_tolower(value, strlen(value));

  if (strcmp(value, "ascii")  == 0) ior = IO_RECORD_ASCII;
  if (strcmp(value, "binary") == 0) ior = IO_RECORD_BINARY;

  return ior;
}

/*****************************************************************************
 *
 *  io_options_to_json
 *
 *****************************************************************************/

__host__ int io_options_to_json(const io_options_t * opts, cJSON ** json) {

  int ifail = 0;

  if (json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {

    /* Seven key/value pairs */
    cJSON * myjson = cJSON_CreateObject();
    cJSON * iogrid = cJSON_CreateIntArray(opts->iogrid, 3);

    cJSON_AddStringToObject(myjson, "Mode", io_mode_to_string(opts->mode));
    cJSON_AddStringToObject(myjson, "Record format",
			    io_record_format_to_string(opts->iorformat));
    cJSON_AddNumberToObject(myjson, "Metadata version",
			    opts->metadata_version);
    cJSON_AddBoolToObject(myjson, "Report", opts->report);
    cJSON_AddBoolToObject(myjson, "Asynchronous", opts->asynchronous);
    cJSON_AddNumberToObject(myjson, "Compression level",
			    opts->compression_levl);
    cJSON_AddItemToObject(myjson, "I/O grid", iogrid);

    *json = myjson;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_options_from_json
 *
 *****************************************************************************/

__host__ int io_options_from_json(const cJSON * json, io_options_t * opts) {

  int ifail = 0;

  if (json == NULL || opts == NULL) {
    ifail = -1;
  }
  else {
    /* Note metadata version is just an integer. */
    cJSON * mode = cJSON_GetObjectItemCaseSensitive(json, "Mode");
    cJSON * ior = cJSON_GetObjectItemCaseSensitive(json, "Record format");
    cJSON * meta = cJSON_GetObjectItemCaseSensitive(json, "Metadata version");
    cJSON * report = cJSON_GetObjectItemCaseSensitive(json, "Report");
    cJSON * async = cJSON_GetObjectItemCaseSensitive(json, "Asynchronous");
    cJSON * level= cJSON_GetObjectItemCaseSensitive(json, "Compression level");
    cJSON * iogrid = cJSON_GetObjectItemCaseSensitive(json, "I/O grid");

    if (mode) {
      char * str = cJSON_GetStringValue(mode);
      opts->mode = io_mode_from_string(str);
    }
    if (ior) {
      char * str = cJSON_GetStringValue(ior);
      opts->iorformat = io_record_format_from_string(str);
    }
    if (meta)   opts->metadata_version = cJSON_GetNumberValue(meta);
    if (report) opts->report = cJSON_IsTrue(report);
    if (async)  opts->asynchronous = cJSON_IsTrue(async);
    if (level)  opts->compression_levl = cJSON_GetNumberValue(level);

    /* Errors */
    if (mode   == NULL) ifail += 1;
    if (ior    == NULL) ifail += 2;
    if (meta   == NULL) ifail += 4;
    if (report == NULL) ifail += 8;
    if (async  == NULL) ifail += 16;
    if (level  == NULL) ifail += 32;
    if (3 != util_json_to_int_array(iogrid, opts->iogrid, 3)) ifail += 64;
  }

  return ifail;
}
