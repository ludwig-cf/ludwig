/*****************************************************************************
 *
 *  util_sum.c
 *
 *  Compensated sums. File to be compiled at -O0.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "util_sum.h"

/*****************************************************************************
 *
 *  kahan_add
 *
 *  Add a contribution to a compensated sum.
 *
 *****************************************************************************/

__host__ __device__ void kahan_add(kahan_t * kahan, double val) {

  assert(kahan);

  {
    volatile double y = val - kahan->cs;
    volatile double t = kahan->sum + y;
    kahan->cs  = (t - kahan->sum) - y;
    kahan->sum = t;
  }
}

/*****************************************************************************
 *
 *  klein_zero
 *
 *  Return zero object.
 *
 *****************************************************************************/

__host__ __device__ klein_t klein_zero(void) {

  klein_t sum = {0.0, 0.0, 0.0};

  return sum;
}

/*****************************************************************************
 *
 *  klein_add
 *
 *  Add a contribution to a doubly compensated sum.
 *
 *****************************************************************************/

__host__ __device__ void klein_add(klein_t * klein, double val) {

  assert(klein);

  {
    volatile double c;
    volatile double cc;
    volatile double t = klein->sum + val;
  
    if (fabs(klein->sum) >= fabs(val)) {
      c = (klein->sum - t) + val;
    }
    else {
      c = (val - t) + klein->sum;
    }
    klein->sum = t;
    t = klein->cs + c;
    if (fabs(klein->cs) >= fabs(c)) {
      cc = (klein->cs - t) + c;
    }
    else {
      cc = (c - t) + klein->cs;
    }
    klein->cs = t;
    klein->ccs = klein->ccs + cc;
  }

  return;
}

/*****************************************************************************
 *
 *  klein_sum
 *
 *  Convenience to return doubly compensated sum.
 *
 *****************************************************************************/

__host__ __device__ double klein_sum(const klein_t * klein) {

  assert(klein);

  return klein->sum + klein->cs + klein->ccs;
}

/*****************************************************************************
 *
 *  klein_atomic_add
 *
 *  Add contribution atomically
 *
 *****************************************************************************/

__host__ __device__ void klein_atomic_add(klein_t * sum, double val) {

  assert(sum);

  /* KLUDGE FOR NOW */
  klein_add(sum, val);

  return;
}
