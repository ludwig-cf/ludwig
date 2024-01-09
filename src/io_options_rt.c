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
 *  (c) 2020-2022 The University of Edinburgh
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

__host__ int io_options_rt(rt_t * rt, rt_enum_t lv, const char * keystub,
			   io_options_t * options) {

  char key[BUFSIZ] = {0};

  assert(rt);
  assert(keystub);
  assert(options);

  sprintf(key, "%s_io_mode", keystub);
  {
    io_mode_enum_t mode = IO_MODE_INVALID;
    int ierr = io_options_rt_mode(rt, lv, key, &mode);

    /* For a valid mode, the default options are selected. */

    if (ierr == RT_KEY_OK) {
      *options = io_options_with_mode(mode);
    }
  }

  /* Allow default format and report to be updated */
  sprintf(key, "%s_io_format", keystub);
  io_options_rt_record_format(rt, lv, key, &options->iorformat);

  sprintf(key, "%s_io_report", keystub);
  io_options_rt_report(rt, lv, key, &options->report);

  return 0;
}

/*****************************************************************************
 *
 *  io_options_rt_mode
 *
 *  If a valid key/value is present, record a mode and return RT_KEY_OK.
 *
 *****************************************************************************/

__host__ int io_options_rt_mode(rt_t * rt, rt_enum_t lv, const char * key,
				io_mode_enum_t * mode) {

  int ifail = RT_KEY_MISSING;
  int key_present = 0;
  char value[BUFSIZ] = {0};

  assert(rt);
  assert(key);
  assert(mode);

  key_present = rt_string_parameter(rt, key, value, BUFSIZ);

  if (key_present) {
    io_mode_enum_t user_mode = io_mode_from_string(value);

    if (io_options_mode_valid(user_mode) == 1) {
      *mode = user_mode;
      ifail = RT_KEY_OK;
    }
    else {
      rt_vinfo(rt, lv, "I/O mode key present but value not recognised\n");
      rt_vinfo(rt, lv, "key:   %s\n", key);
      rt_vinfo(rt, lv, "value: %s\n", value);
      rt_vinfo(rt, lv, "Should be either 'single', 'multiple' or 'mpiio'\n");
      rt_fatal(rt, lv, "Please check the input file and try again!\n");
      ifail = RT_KEY_INVALID;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_options_rt_rformat
 *
 *  If the key is present and valid, return RT_KEY_OK and set the
 *  iorformat.
 *
 *****************************************************************************/

__host__ int io_options_rt_record_format(rt_t * rt, rt_enum_t lv,
					 const char * key,
					 io_record_format_enum_t * iorformat) {
  int ifail = RT_KEY_MISSING;
  int key_present = 0;
  char str[BUFSIZ] = {0};

  assert(rt);
  assert(key);
  assert(iorformat);

  key_present = rt_string_parameter(rt, key, str, BUFSIZ);

  if (key_present) {
    io_record_format_enum_t user_iorformat = io_record_format_from_string(str);

    if (io_options_record_format_valid(user_iorformat) == 1) {
      ifail = RT_KEY_OK;
      *iorformat = user_iorformat;
    }
    else {
      ifail = RT_KEY_INVALID;
      rt_vinfo(rt, lv, "I/O record format present but value not recognised\n");
      rt_vinfo(rt, lv, "key:   %s\n", key);
      rt_vinfo(rt, lv, "value: %s\n", str);
      rt_vinfo(rt, lv, "Should be either 'ascii' or 'binary'\n");
      rt_fatal(rt, lv, "Please check the input file and try again!\n");
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  io_options_rt_report
 *
 *  Update report accordingly if the switch "key" is present.
 *  Return RT_KEY_OK or RT_KEY_MISSING.
 *
 *****************************************************************************/

__host__ int io_options_rt_report(rt_t * rt, rt_enum_t lv, const char * key,
				  int * report) {

  int ifail = RT_KEY_MISSING;
  int key_present = -1;

  /* A switch is just false or true, so we have to look more carefully */

  key_present = rt_key_present(rt, key);

  if (key_present) {
    ifail = RT_KEY_OK;
    *report = rt_switch(rt, key);
  }

  return ifail;
}
