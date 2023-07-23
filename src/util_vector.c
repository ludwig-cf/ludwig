/*****************************************************************************
 *
 *  util_vector.c
 *
 *  Some simple vector operations.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
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
 *  util_vector_copy
 *
 *  Copy a to b.
 *
 *****************************************************************************/

void util_vector_copy(int n, const double * a, double * b) {

  assert(n > 0);
  assert(a);
  assert(b);

  for (int ia = 0; ia < n; ia++) {
    b[ia] = a[ia];
  }

  return;
}
