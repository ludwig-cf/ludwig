/*****************************************************************************
 *
 *  test_util_vector.c
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
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util_vector.h"

int test_util_vector_l2_norm(void);
int test_util_vector_normalise(void);
int test_util_vector_copy(void);

/*****************************************************************************
 *
 *  test_util_vector_suite
 *
 *****************************************************************************/

int test_util_vector_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_vector_l2_norm();
  test_util_vector_normalise();
  test_util_vector_copy();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_vector_l2_norm
 *
 *****************************************************************************/

int test_util_vector_l2_norm(void) {

  int ifail = 0;

  {
    int n = 2;
    double a[2] = {3.0, 4.0};
    double l2 = util_vector_l2_norm(n, a);
    if (fabs(l2 - 5.0) >= DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_vector_normalise
 *
 *****************************************************************************/

int test_util_vector_normalise(void) {

  int ifail = 0;

  {
    int n = 7;
    double a[7] = {-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0};
    util_vector_normalise(n, a);
    printf("NORMAL %14.7e %14.7e %14.7e\n", a[0], a[1], a[2]);
    if (fabs(a[0] + 0.5) > DBL_EPSILON) ifail += 1;
    if (fabs(a[1] - 0.0) > DBL_EPSILON) ifail += 1;
    if (fabs(a[2] - 0.5) > DBL_EPSILON) ifail += 1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_vector_copy
 *
 *****************************************************************************/

int test_util_vector_copy(void) {

  int ifail = 0;

  {
    int n = 3;
    double a[3] = {1.0, 2.0, 4.0};
    double b[3] = {0};
    util_vector_copy(n, a, b);
    assert(fabs(a[0] - 1.0) < DBL_EPSILON);
    assert(fabs(a[1] - 2.0) < DBL_EPSILON);
    assert(fabs(a[2] - 4.0) < DBL_EPSILON);
    if (fabs(b[0] - a[0]) > DBL_EPSILON) ifail += 1;
    if (fabs(b[1] - a[1]) > DBL_EPSILON) ifail += 1;
    if (fabs(b[2] - a[2]) > DBL_EPSILON) ifail += 1;
  }

  return ifail;
}
