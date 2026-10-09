/*****************************************************************************
 *
 *  coords_inline.h
 *
 *  __host__ __device__ static inline
 *
 *  To be included via coords.h so that cs_t definition is avaalable.
 *  Note: it is the intention that all HDSI_ function should be here
 *  ultimately; some are still in coords.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_COORDS_INLINE_H_
#define LUDWIG_COORDS_INLINE_H_

#include <assert.h>

#define HDSI_ __host__ __device__ static inline

/*****************************************************************************
 *
 *  cs_index_to_ic
 *
 *****************************************************************************/

HDSI_ int cs_index_to_ic(const cs_t * cs, int index) {

  assert(cs);
  assert(0 <= index && index < cs->param->nsites);

  return ((1 - cs->param->nhalo) + index / cs->param->str[X]);
}

/*****************************************************************************
 *
 *  cs_index_to_jc
 *
 *****************************************************************************/

HDSI_ int cs_index_to_jc(const cs_t * cs, int index) {

  assert(cs);
  assert(0 <= index && index < cs->param->nsites);

  int jc =
      (1 - cs->param->nhalo) + (index % cs->param->str[X]) / cs->param->str[Y];

  return jc;
}

/*****************************************************************************
 *
 *  cs_index_to_kc
 *
 *****************************************************************************/

HDSI_ int cs_index_to_kc(const cs_t * cs, int index) {

  assert(cs);
  assert(0 <= index && index < cs->param->nsites);

  int kc = (1 - cs->param->nhalo) + index % cs->param->str[Y];

  return kc;
}

#undef HDSI_

#endif
