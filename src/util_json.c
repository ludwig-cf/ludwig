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
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "util_fopen.h"
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

/*****************************************************************************
 *
 *  util_json_to_file
 *
 *  Write the stringified version to the file given.
 *
 *****************************************************************************/

int util_json_to_file(const char * filename, const cJSON * json) {

  int ifail = -1;
  char * str = cJSON_Print(json);

  assert(filename);
  assert(json);

  if (str) {
    FILE * fp = util_fopen(filename, "w");
    if (fp) {
      fprintf(fp, "%s", str);
      fclose(fp);
      ifail = 0;
    }
    free(str);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_json_from_file
 *
 *  The file should contain a well-formed json object.
 *  Success means return value is zero AND json != NULL.
 *
 *  json object to be deleted by caller.
 *
 *****************************************************************************/

int util_json_from_file(const char * filename, cJSON ** json) {

  int ifail = 0;
  FILE * fp = util_fopen(filename, "rb");

  assert(filename);
  assert(json && *json == NULL);

  if (fp == NULL) {
    ifail = -1;
  }
  else {
    char * buf = NULL;
    size_t len = 0;
    fseek(fp, 0, SEEK_END);
    len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf = (char *) calloc(len+1, sizeof(char));
    if (buf) {
      size_t nread = fread(buf, 1, len, fp);
      if (nread != len) ifail = +1;
      *json = cJSON_Parse(buf);
      free(buf);
    }
    fclose(fp);
  }

  return ifail;
}
