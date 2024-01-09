/*****************************************************************************
 *
 *  util_vector.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_VECTOR_H
#define LUDWIG_UTIL_VECTOR_H

double util_vector_l2_norm(int n, const double * a);
void   util_vector_normalise(int n, double * a);
void   util_vector_copy(int n, const double * a, double * b);
int    util_vector_orthonormalise(const double a[3], double b[3]);
void   util_vector_basis_to_dcm(const double a[3], const double b[3],
				const double c[3], double r[3][3]);
void   util_vector_dcm_to_euler(const double r[3][3], double * phi,
				double * theta, double * psi);

/* util_vector_dot_product(a, b) */

static inline double util_vector_dot_product(const double a[3],
					     const double b[3]) {
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

/* util_vector_cross_product(c, a, b) provides c = a x b */

static inline void util_vector_cross_product(double c[3], const double a[3],
					     const double b[3]) {
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}

#endif
