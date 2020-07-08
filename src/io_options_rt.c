/*****************************************************************************
 *
 *  io_options_rt.c
 *
 *  Runtime parsing of i/o options.
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
#include <string.h>

#include "util.h"
#include "io_options_rt.h"

/*****************************************************************************
 *
 *  io_options_rt
 *
 *  For keystub e.g., "default" look for the following keys in the rt input:
 *
 *    default_io_mode
 *    default_io_format
 *    default_io_report
 *
 *  The options returned are defaults, or valid user input.
 *
 *****************************************************************************/

__host__ int io_options_rt(pe_t * pe, rt_t * rt, const char * keystub,
			   io_options_t * options) {

  char key[BUFSIZ] = {0};
  io_mode_enum_t mode = IO_MODE_INVALID;
  io_record_format_enum_t iorformat = IO_RECORD_INVALID;
  io_options_t opts = io_options_default();

  assert(pe);
  assert(rt);
  assert(keystub);
  assert(options);


  sprintf(key, "%s_io_mode", keystub);
  io_options_rt_mode(pe, rt, key, &mode);
  if (io_options_mode_valid(mode)) opts.mode = mode;

  sprintf(key, "%s_io_format", keystub);
  io_options_rt_record_format(pe, rt, key, &iorformat);
  if (io_options_record_format_valid(iorformat)) opts.iorformat = iorformat;

  sprintf(key, "%s_io_report", keystub);
  opts.report = rt_switch(rt, key);

  /* Force metadata to be consistent with the mode */

  if (opts.mode == IO_MODE_SINGLE) {
    /* only one choice at the moment */
    opts.metadata_version = IO_METADATA_SINGLE_V1;
  }

  if (opts.mode == IO_MODE_MULTIPLE) {
    /* only one choice again */
    opts.metadata_version = IO_METADATA_MULTI_V1;
  }

  *options = opts;

  return 0;
}

/*****************************************************************************
 *
 *  io_options_rt_mode
 *
 *****************************************************************************/

__host__ int io_options_rt_mode(pe_t * pe, rt_t * rt, const char * key,
				io_mode_enum_t * mode) {
  int key_present = 0;
  char value[BUFSIZ] = {0};
  io_mode_enum_t user_mode = IO_MODE_INVALID;

  assert(pe);
  assert(rt);
  assert(key);
  assert(mode);

  key_present = rt_string_parameter(rt, key, value, BUFSIZ);
  util_str_tolower(value, strlen(value));

  if (strcmp(value, "single")   == 0) user_mode = IO_MODE_SINGLE;
  if (strcmp(value, "multiple") == 0) user_mode = IO_MODE_MULTIPLE;

  if (key_present && io_options_mode_valid(user_mode) == 0) {
    pe_info(pe, "I/O mode key present but value not recognised\n");
    pe_info(pe, "key:   %s\n", key);
    pe_info(pe, "value: %s\n", value);
    pe_info(pe, "Should be either 'single' or 'multiple'\n");
    pe_fatal(pe, "Please check the input file and try again!\n");
  }

  *mode = user_mode;

  return 0;
}

/*****************************************************************************
 *
 *  io_options_rt_rformat
 *
 *  If the key is not present IO_RECORD_INVALID is returned.
 *
 *****************************************************************************/

__host__ int io_options_rt_record_format(pe_t * pe, rt_t * rt,
					 const char * key,
					 io_record_format_enum_t * iorformat) {
  int key_present = 0;
  char value[BUFSIZ] = {0};
  io_record_format_enum_t user_iorformat = IO_RECORD_INVALID;

  assert(pe);
  assert(rt);
  assert(key);
  assert(iorformat);

  key_present = rt_string_parameter(rt, key, value, BUFSIZ);
  util_str_tolower(value, strlen(value));

  if (strcmp(value, "ascii")  == 0) user_iorformat = IO_RECORD_ASCII;
  if (strcmp(value, "binary") == 0) user_iorformat = IO_RECORD_BINARY;

  if (key_present && io_options_record_format_valid(user_iorformat) == 0) {
    pe_info(pe, "I/O record format key present but value not recognised\n");
    pe_info(pe, "key:   %s\n", key);
    pe_info(pe, "value: %s\n", value);
    pe_info(pe, "Should be either 'ascii' or 'binary'\n");
    pe_fatal(pe, "Please check the input file and try again!\n");
  }

  *iorformat = user_iorformat;

  return 0;
}
