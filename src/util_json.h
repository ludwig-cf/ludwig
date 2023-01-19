/*****************************************************************************
 *
 *  util_json.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_JSON_H
#define LUDWIG_UTIL_JSON_H

#include "util_cJSON.h"

int util_json_to_int_array(const cJSON * const json, int * array, int sz);
int util_json_to_double_array(const cJSON * const json, double * array, int sz);
int util_json_to_file(const char * filename, const cJSON * const json);
int util_json_from_file(const char * filename, cJSON ** json);

#endif
