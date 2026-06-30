/*****************************************************************************
 *
 *  util_vector.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023-2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_VECTOR_H
#define LUDWIG_UTIL_VECTOR_H

int    util_vector_orthonormalise(const double a[3], double b[3]);
void   util_vector_basis_to_dcm(const double a[3], const double b[3],
				const double c[3], double r[3][3]);
void   util_vector_dcm_to_euler(const double r[3][3], double * phi,
				double * theta, double * psi);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functions
 *
 *  double util_vector_modulus(const double v[3]);
 *  double util_vector_dot_product(const double a[3], const double b[3]);
 *  int    util_vector_random_util_vector(int seed, double v[3]);
 *
 *****************************************************************************/

#include <math.h>
#include "target.h"

/*****************************************************************************
 *
 *  util_vector_modulus
 *
 *****************************************************************************/

__host__ __device__
static inline double util_vector_modulus(const double a[3]) {

  return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

/*****************************************************************************
 *
 *  util_vector_dot_product
 *
 *****************************************************************************/

__host__ __device__
static inline double util_vector_dot_product(const double a[3],
					     const double b[3]) {
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

/*****************************************************************************
 *
 *  util_vector_cross_product
 *
 *  Computes c = a x b (note order of arguments)
 *
 *****************************************************************************/

__host__ __device__
static inline void util_vector_cross_product(double c[3], const double a[3],
					     const double b[3]) {
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

/*****************************************************************************
 *
 *  util_vector_copy
 *
 *  Copy a to b.
 *
 *****************************************************************************/

__host__ __device__
static inline void util_vector_copy(int n, const double * a, double * b) {

  assert(n > 0);
  assert(a);
  assert(b);

  for (int ia = 0; ia < n; ia++) {
    b[ia] = a[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  util_vector_l2_norm
 *
 *  For vector of length n, compute l2 = sqrt(sum_i a_i^2).
 *
 *****************************************************************************/

__host__ __device__
static inline double util_vector_l2_norm(int n, const double * a) {

  double l2 = 0.0;

  assert(n > 0);
  assert(a);

  for (int ia = 0; ia < n; ia++) {
    l2 += a[ia]*a[ia];
  }

  return sqrt(l2);
}

/*****************************************************************************
 *
 *  util_vector_normalise
 *
 *  For the given vector of length n, compute norm = sum_i a_i^2
 *  and divide each element by sqrt(norm) to normalise.
 *
 *****************************************************************************/

__host__ __device__
static inline void util_vector_normalise(int n, double * a) {

  assert(n > 0);
  assert(a);

  double anorm = util_vector_l2_norm(n, a);

  if (anorm > 0.0) anorm = 1.0/anorm;

  for (int ia = 0; ia < n; ia++) {
    a[ia] = anorm*a[ia];
  }

  return;
}

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

__host__ __device__
static inline double util_random_uniform(int * seed) {

  int64_t s = *seed;

  assert(s > 0);

  s = 1389796*(s + 0) % 2147483647;
  *seed = (int) s;

  return (1.0*s/2147483647);
}

/*****************************************************************************
 *
 *  util_vector_random_unit_vector
 *
 *  See, e.g., https://mathworld.wolfram.com/SpherePointPicking.html (2026).
 *
 *****************************************************************************/

__host__ __device__
static inline int util_vector_random_unit_vector(int seed, double r[3]) {

  int s = seed;

  double u = 1.0 - 2.0*util_random_uniform(&s);          /* -1 < u <= 1   */
  double v = 2.0*4.0*atan(1.0)*util_random_uniform(&s);  /*  0 < v <= 2pi */

  r[X] = sqrt(1.0 - u*u)*cos(v);
  r[Y] = sqrt(1.0 - u*u)*sin(v);
  r[Z] = u;

  return s;
}

#endif
