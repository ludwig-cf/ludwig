/*****************************************************************************
 *
 *  util_vector.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023-2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_VECTOR_H
#define LUDWIG_UTIL_VECTOR_H

double util_vector_l2_norm(int n, const double * a);
void   util_vector_normalise(int n, double * a);
int    util_vector_orthonormalise(const double a[3], double b[3]);
void   util_vector_basis_to_dcm(const double a[3], const double b[3],
				const double c[3], double r[3][3]);
void   util_vector_dcm_to_euler(const double r[3][3], double * phi,
				double * theta, double * psi);

/*****************************************************************************
 *
 *  __host__ __device__ static inline functions
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

#endif
