/*****************************************************************************
 *
 *  lc_anchoring.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "lc_anchoring.h"

/*****************************************************************************
 *
 *  lc_anchoring_type_from_string
 *
 *  Translate a string to an anchoring type
 *
 ****************************************************************************/

lc_anchoring_enum_t lc_anchoring_type_from_string(const char * string) {

  lc_anchoring_enum_t lc_anchor = LC_ANCHORING_INVALID;

  assert(string);

  if (strcmp(string, "normal") == 0) lc_anchor = LC_ANCHORING_NORMAL;
  if (strcmp(string, "planar") == 0) lc_anchor = LC_ANCHORING_PLANAR;
  if (strcmp(string, "fixed")  == 0) lc_anchor = LC_ANCHORING_FIXED;

  return lc_anchor;
}
