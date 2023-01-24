/*****************************************************************************
 *
 *  lees_edwards_options.c
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
#include <stdio.h>
#include <string.h>

#include "util.h"
#include "lees_edwards_options.h"

/*****************************************************************************
 *
 *  lees_edw_type_to_string
 *
 *  Being "steady" or "oscillatory".
 *
 *****************************************************************************/

const char * lees_edw_type_to_string(lees_edw_enum_t mytype) {

  const char * str = "INVALID";
  if (mytype == LE_SHEAR_TYPE_STEADY)      str = "STEADY";
  if (mytype == LE_SHEAR_TYPE_OSCILLATORY) str = "OSCILLATORY";

  return str;
}

/*****************************************************************************
 *
 *  lees_edw_type_from_string
 *
 *****************************************************************************/

lees_edw_enum_t lees_edw_type_from_string(const char * str) {

  lees_edw_enum_t mytype = LE_SHEAR_TYPE_INVALID;
  char tmp[BUFSIZ] = {0};

  strncpy(tmp, str, BUFSIZ-1);
  util_str_tolower(tmp, strlen(tmp));

  if (strcmp(tmp, "steady")      == 0) mytype = LE_SHEAR_TYPE_STEADY;
  if (strcmp(tmp, "oscillatory") == 0) mytype = LE_SHEAR_TYPE_OSCILLATORY;

  return mytype;
}

/*****************************************************************************
 *
 *  lees_edw_opts_to_json
 *
 *****************************************************************************/

int lees_edw_opts_to_json(const lees_edw_options_t * opts, cJSON ** json) {

  int ifail = 0;

  assert(opts);

  if (json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {
    int nplane = opts->nplanes;
    cJSON * myjson = cJSON_CreateObject();

    cJSON_AddNumberToObject(myjson, "Number of planes", nplane);
    if (nplane > 0) {
      cJSON_AddStringToObject(myjson, "Shear type",
			      lees_edw_type_to_string(opts->type));
      if (opts->type == LE_SHEAR_TYPE_OSCILLATORY) {
	cJSON_AddNumberToObject(myjson, "Period (timesteps)", opts->period);
      }
      cJSON_AddNumberToObject(myjson, "Reference time", opts->nt0);
      cJSON_AddNumberToObject(myjson, "Plane speed", opts->uy);
    }

    *json = myjson;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  lees_edw_opts_from_json
 *
 *  Acceptable formats include:
 *  { "Number of planes:", 0}                     can omit everything else
 *  { ... , "Shear type": "STEADY", ... }         may omit period
 *  { ... , "Shear type": "OSCILLATORY", ... }    must include period
 *
 *****************************************************************************/

int lees_edw_opts_from_json(const cJSON * json, lees_edw_options_t * opts) {

  int ifail = 0;

  if (json == NULL || opts == NULL) {
    ifail = -1;
  }
  else {
    lees_edw_options_t myopt = {.nplanes = -1, .type = LE_SHEAR_TYPE_INVALID};
    cJSON * np = cJSON_GetObjectItemCaseSensitive(json, "Number of planes");
    cJSON * st = cJSON_GetObjectItemCaseSensitive(json, "Shear type");
    cJSON * sp = cJSON_GetObjectItemCaseSensitive(json, "Period (timesteps)");
    cJSON * rt = cJSON_GetObjectItemCaseSensitive(json, "Reference time");
    cJSON * ps = cJSON_GetObjectItemCaseSensitive(json, "Plane speed");

    if (np) myopt.nplanes = cJSON_GetNumberValue(np);

    if (myopt.nplanes > 0) {
      if (st) {
	myopt.type = lees_edw_type_from_string(cJSON_GetStringValue(st));
	/* Then period only if oscillatory ... */
	if (myopt.type == LE_SHEAR_TYPE_OSCILLATORY && sp) {
	  myopt.period = cJSON_GetNumberValue(sp);
	  if (myopt.period <= 0) ifail += 2;
	}
      }
      if (rt) myopt.nt0 = cJSON_GetNumberValue(rt);
      if (ps) myopt.uy  = cJSON_GetNumberValue(ps);
    }

    /* Error condition */
    if (myopt.nplanes < 0) ifail += 1;

    *opts = myopt;
  }

  return ifail;
}
