/*****************************************************************************
 *
 *  colloid_link_impl.h
 *
 *  Static inline function implementations. This file is included in
 *  colloid_link.h so that the relevant colloid_link_t definition is
 *  available.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Alexei Borissov (alexei@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_LINK_IMPL_H
#define LUDWIG_COLLOID_LINK_IMPL_H

#include <assert.h>

#define HDSI_ __host__ __device__ static inline

/*****************************************************************************
 *
 *  colloid_links_array_i
 *
 *  Retrieves the i link at given index
 *
 *****************************************************************************/

HDSI_ int colloid_links_array_i(const colloid_links_array_t * array,
                                int                           index) {

  assert(0 <= index && index < array->max_links);
  return array->i[index];
}

/*****************************************************************************
 *
 *  colloid_links_array_j
 *
 *  Retrieves the j link at given index
 *
 *****************************************************************************/

HDSI_ int colloid_links_array_j(const colloid_links_array_t * array,
                                int                           index) {

  assert(0 <= index && index < array->max_links);
  return array->j[index];
}

/***************************************************************************
 *
 *  colloid_links_array__p
 *
 *  Retrieves the p link at given index
 *
 ***************************************************************************/

HDSI_ int colloid_links_array_p(const colloid_links_array_t * array,
                                int                           index) {

  assert(0 <= index && index < array->max_links);
  return array->p[index];
}

/***************************************************************************
 *
 * colloid_links_array_status
 *
 * Retrieves the link status at given index
 *
 ***************************************************************************/

HDSI_ int colloid_links_array_status(const colloid_links_array_t * array,
                                     int                           index) {

  assert(0 <= index && index < array->max_links);

  return array->status[index];
}

/***************************************************************************
 *
 * colloid_links_array_rb
 *
 * Retrieves the rb link at given index
 *
 ***************************************************************************/

HDSI_ void colloid_links_array_rb(const colloid_links_array_t * array,
                                  int index, double rb[3]) {

  assert(0 <= index && index < array->max_links);

  for (int i = 0; i < 3; i++) {
    rb[i] = array->rb[i][index];
  }

  return;
}

#undef HDSI_

#endif
