/*****************************************************************************
 *
 *  util_json.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_JSON_H
#define LUDWIG_UTIL_JSON_H

#include "util_cJSON.h"

int util_json_to_int_array(const cJSON * const json, int * array, int sz);

#endif
