/*****************************************************************************
 *
 *  util_random_impl.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2026 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_RANDOM_IMPL_H
#define LUDWIG_UTIL_RANDOM_IMPL_H

#include <math.h>

#include "target.h"

__host__ __device__ static inline double util_random_uniform(int * seed);

/*****************************************************************************
 *
 *  util_random_uniform
 *
 *  A simple LCG for "occasional" use where statistics are not an
 *  overriding concern.
 *
 *  The state is one int32_t (> 0 on input), and the updated state
 *  must also be < INT_MAX = 2147483647, ie., suitable for int32_t.
 *
 *****************************************************************************/

__host__ __device__ static inline double util_random_uniform(int * seed) {

  int64_t s = *seed;

  assert(s > 0);

  s     = 1389796 * (s + 0) % 2147483647;
  *seed = (int) s;

  return (1.0 * s / 2147483647);
}

#endif
