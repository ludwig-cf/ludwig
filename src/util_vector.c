/*****************************************************************************
 *
 *  util_vector.c
 *
 *  Some simple vector operations.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023-2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "util_vector.h"

/*****************************************************************************
 *
 *  util_vector_l2_norm
 *
 *  For vector of length n, compute l2 = sqrt(sum_i a_i^2).
 *
 *****************************************************************************/

double util_vector_l2_norm(int n, const double * a) {

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

void util_vector_normalise(int n, double * a) {

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
 *  util_vector_orthonormalise
 *
 *  Orthonormalise vector b to the vector a; a Gram-Schmidt like
 *  procedure: the result is normalised to be a unit vector.
 *
 *  Strictly, neither a nor b should be zero.
 *
 *****************************************************************************/

int util_vector_orthonormalise(const double a[3], double b[3]) {

  int ifail = 0;
  double aa = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
  double ab = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

  if (aa <= 0.0) ifail = -1;

  b[0] = b[0] - (ab/aa)*a[0];
  b[1] = b[1] - (ab/aa)*a[1];
  b[2] = b[2] - (ab/aa)*a[2];

  {
    double bb = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
    double rmod = 1.0/sqrt(bb);
    b[0] = rmod*b[0];
    b[1] = rmod*b[1];
    b[2] = rmod*b[2];
    if (bb <= 0.0) ifail = -2;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_vector_basis_to_dcm
 *
 *  The rotation required to go from a standard (v1, v2, v3) Cartesian
 *  basis to the basis described by the orthonormal (a, b, c) may be
 *  described by the Direction Cosine Matrix.
 *
 *  (a,b,c) should be a set of orthogonal unit vectors i.e., an
 *  orthonormal basis.
 *
 *  E.g., https://en.wikipedia.org/wiki/Euclidean_vector
 *  section on converting between Cartesian bases
 *
 *****************************************************************************/

void util_vector_basis_to_dcm(const double a[3],
			      const double b[3],
			      const double c[3],
			      double r[3][3]) {

  double v1[3] = {1.0, 0.0, 0.0};
  double v2[3] = {0.0, 1.0, 0.0};
  double v3[3] = {0.0, 0.0, 1.0};

  r[0][0] = util_vector_dot_product(v1, a);
  r[0][1] = util_vector_dot_product(v2, a);
  r[0][2] = util_vector_dot_product(v3, a);
  r[1][0] = util_vector_dot_product(v1, b);
  r[1][1] = util_vector_dot_product(v2, b);
  r[1][2] = util_vector_dot_product(v3, b);
  r[2][0] = util_vector_dot_product(v1, c);
  r[2][1] = util_vector_dot_product(v2, c);
  r[2][2] = util_vector_dot_product(v3, c);

  return;
}

/*****************************************************************************
 *
 *  util_vector_dcm_to_euler
 *
 *  The rotation described by the direction cosine matrix r my be
 *  converted to standard z-x-z Euler angles.
 *
 *  -pi < phi <= pi; 0 <= theta <= pi; -pi < psi <= pi on output.
 *
 *****************************************************************************/

void util_vector_dcm_to_euler(const double r[3][3],
			      double * phi,
			      double * theta,
			      double * psi) {
  *theta = acos(r[2][2]);

  if (fabs(fabs(r[2][2]) - 1.0) > DBL_EPSILON) {
    *phi = atan2(r[2][0], -r[2][1]);
    *psi = atan2(r[0][2],  r[1][2]);
  }
  else {
    *phi = atan2(r[0][1],  r[0][0]);
    *psi = 0.0;
  }

  return;
}
