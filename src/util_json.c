/*****************************************************************************
 *
 *  util_json.c
 *
 *  A couple of useful additions to cJSON.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/


#include <assert.h>
#include <stddef.h>

#include "util_json.h"

/*****************************************************************************
 *
 *  util_json_to_int_array
 *
 *  The number of elements successfully identified is returned.
 *
 *****************************************************************************/

int util_json_to_int_array(const cJSON * const json, int * array, int sz) {

  int icount = 0;

  if (cJSON_IsArray(json) && cJSON_GetArraySize(json) == sz) {
    cJSON * element = NULL;
    cJSON_ArrayForEach(element, json) {
      if (cJSON_IsNumber(element)) {
	array[icount++] = cJSON_GetNumberValue(element);
      }
    }
  }

  return icount;
}

/*****************************************************************************
 *
 *  util_json_to_double_array
 *
 *  Actually the same as the int version except for the argument.
 *
 *****************************************************************************/

int util_json_to_double_array(const cJSON * const json, double * array, int sz) {

  int icount = 0;

  if (cJSON_IsArray(json) && cJSON_GetArraySize(json) == sz) {
    cJSON * element = NULL;
    cJSON_ArrayForEach(element, json) {
      if (cJSON_IsNumber(element)) {
	array[icount++] = cJSON_GetNumberValue(element);
      }
    }
  }

  return icount;
}
