/*****************************************************************************
 *
 *  test_util.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2014 The University of Edinburgh
 *  
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"

/* For SVD tests, SVD_EPSILON is increased according to matrix elements... */
#define SVD_EPSILON (2.0*DBL_EPSILON)

int util_svd_check(int m, int n, double ** a);

/*****************************************************************************
 *
 *  test_suite_suite
 *
 *****************************************************************************/

int test_util_suite(void) {

  int ifail = 0;
  int m = 3;
  int n = 2;

  double ** a = NULL;
  double b[3] = {1.0, 2.0, 3.0};
  double x[2];

  pe_init_quiet();

  ifail = util_matrix_create(m, n, &a);
  assert(ifail == 0);

  a[0][0] = -1.0;
  a[0][1] = 0.0;
  a[1][0] = 0.0;
  a[1][1] = 3.0;
  a[2][0] = 2.0;
  a[2][1] = -1.0;

  ifail = util_svd_check(m, n, a);
  assert(ifail == 0);

  ifail = util_svd_solve(m, n, a, b, x);
  assert(ifail == 0);

  util_matrix_free(m, &a);

  info("PASS     ./unit/test_util\n");
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  svd_check
 *
 *****************************************************************************/

int util_svd_check(int m, int n, double **a) {

  int i, j, k;
  int ifail = 0;
  double sum;
  double svd_epsilon;     /* Test tolerance for this a matrix */

  double ** u = NULL;     /* m x n matrix u (copy of a on input to svd) */
  double * w = NULL;      /* Singular values w[n] */
  double ** v = NULL;     /* n x n matrix v */

  ifail += util_matrix_create(m, n, &u);
  ifail += util_matrix_create(n, n, &v);
  ifail += util_vector_create(n, &w);

  /* Copy input matrix. Use largest fabs(a[i][j]) to set a tolerance */

  sum = 0.0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      u[i][j] = a[i][j];
      sum = dmax(sum, fabs(a[i][j]));
    }
  }
  svd_epsilon = sum*SVD_EPSILON;

  ifail += util_svd(m, n, u, w, v);
  if (ifail > 0) printf("SVD routine failed\n");

  /* Assert u is orthonormal \sum_{k=0}^{m-1} u_ki u_kj = delta_ij
   * for 0 <= i, j < n: */

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      sum = 0.0;
      for (k = 0; k < m; k++) {
	sum += u[k][j]*u[k][i];
      }
      if (fabs(sum - 1.0*(i == j)) > svd_epsilon) ifail += 1;
    }
  }
  if (ifail > 0) printf("U not orthonormal\n");

  /* Assert v is orthonormal \sum_{k=0}^{n-1} v_ki v_kj = delta_ij
   * for <= 0 i, j, < n: */

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      sum = 0.0;
      for (k = 0; k < n; k++) {
	sum += v[k][j]*v[k][i];
      }
      if (fabs(sum - 1.0*(i == j)) > svd_epsilon) ifail += 1;
    }
  }
  if (ifail > 0) printf("V not orthonormal\n");

  /* Assert u w v^t = a, ie., the decomposition is correct. */

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      sum = 0.0;
      for (k = 0; k < n; k++) {
	sum += u[i][k]*w[k]*v[j][k];
      }
      if (fabs(sum - a[i][j]) > svd_epsilon) ifail += 1;
    }
  }
  if (ifail > 0) printf("Decomposition incorrect\n");

  util_vector_free(&w);
  util_matrix_free(n, &v);
  util_matrix_free(m, &u);

  return ifail;
}
