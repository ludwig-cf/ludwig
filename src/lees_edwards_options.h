/*****************************************************************************
 *
 *  lees_edwards_options.h
 *
 *  Meta-data to define Lees Edwards configuration.
 *  Only the number of planes is required, as the plane spacing is always
 *  uniform; all the planes have the same "speed".
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

#ifndef LUDWIG_LEES_EDWARDS_OPTIONS_H
#define LUDWIG_LEES_EDWARDS_OPTIONS_H

#include "util_json.h"

typedef struct lees_edw_options_s lees_edw_options_t;

typedef enum lees_edw_enum {
  LE_SHEAR_TYPE_INVALID,
  LE_SHEAR_TYPE_STEADY,            /* Steady shear (default) */
  LE_SHEAR_TYPE_OSCILLATORY        /* Oscillatory shear uy = uy cos(wt) */
} lees_edw_enum_t;

struct lees_edw_options_s {
  int nplanes;                     /* Number of planes */
  lees_edw_enum_t type;            /* Shear type */
  int period;                      /* Oscillatory shear period (timesteps) */
  int nt0;                         /* Reference time (usually t0 = 0) */
  double uy;                       /* "Plane speed"; stricly the velocity
				    * jump crossing the plane. */
};

const char * lees_edw_type_to_string(lees_edw_enum_t mytype);
lees_edw_enum_t lees_edw_type_from_string(const char * str);

int lees_edw_opts_to_json(const lees_edw_options_t * opts, cJSON ** json);
int lees_edw_opts_from_json(const cJSON * json, lees_edw_options_t * opts);

#endif
