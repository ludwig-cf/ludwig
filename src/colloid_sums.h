/*****************************************************************************
 *
 *  colloid_sums.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_SUMS_H
#define LUDWIG_COLLOID_SUMS_H

#include "colloids.h"

typedef struct colloid_sum_s colloid_sum_t;

typedef enum colloid_sum_enum_type {
  COLLOID_SUM_NULL = 0,
  COLLOID_SUM_STRUCTURE = 1,
  COLLOID_SUM_DYNAMICS = 2,
  COLLOID_SUM_ACTIVE = 3,
  COLLOID_SUM_SUBGRID = 4,
  COLLOID_SUM_CONSERVATION = 5,
  COLLOID_SUM_FORCE_EXT_ONLY = 6,
  COLLOID_SUM_DIAGNOSTIC = 7,
  COLLOID_SUM_MAX = 8} colloid_sum_enum_t;

int colloid_sums_create(colloids_info_t * cinfo, colloid_sum_t ** psum);
void colloid_sums_free(colloid_sum_t * sum);
int colloid_sums_halo(colloids_info_t * cinfo, colloid_sum_enum_t type);
int colloid_sums_1d(colloid_sum_t * sum, int dim, colloid_sum_enum_t type);

#endif
