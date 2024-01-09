/*****************************************************************************
 *
 *  test_util.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "util_ellipsoid.h"
#include "tests.h"

/* For RNG tests */
#define NLARGE         10000000
#define STAT_TOLERANCE 0.001

int util_random_unit_vector_check(void);
int util_jacobi_check(void);
int util_dpythag_check(void);
int util_str_tolower_check(void);
int util_rectangle_conductance_check(void);

/*****************************************************************************
 *
 *  test_suite_suite
 *
 *****************************************************************************/

int test_util_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  util_random_unit_vector_check();

  util_jacobi_check();
  util_dpythag_check();
  util_str_tolower_check();
  util_rectangle_conductance_check();

  pe_info(pe, "PASS     ./unit/test_util\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  util_random_unit_vector_check
 *
 *  Check a known case, and some simple statistics.
 *
 *****************************************************************************/

int util_random_unit_vector_check(void) {

  int n;
  int state = 1;
  double rhat[3];
  double rvar;
  double rmin, rmax, rmean[3];

  rmin = 0.0;
  rmax = 0.0;
  rmean[0] = 0.0;
  rmean[1] = 0.0;
  rmean[2] = 0.0;

  for (n = 0; n < NLARGE; n++) {
    util_random_unit_vector(&state, rhat);
    rvar = rhat[0]*rhat[0] + rhat[1]*rhat[1] + rhat[2]*rhat[2];
    /* The algorithm is ok to about 5x10^-16 */
    test_assert(fabs(rvar - 1.0) < 4.0*DBL_EPSILON);
    rmean[0] += rhat[0];
    rmean[1] += rhat[1];
    rmean[2] += rhat[2];
    rmin = dmin(rmin, rhat[0]);
    rmin = dmin(rmin, rhat[1]);
    rmin = dmin(rmin, rhat[2]);
    rmax = dmax(rmax, rhat[0]);
    rmax = dmax(rmax, rhat[1]);
    rmax = dmax(rmax, rhat[2]);
  }


  test_assert(rmin >= -1.0);
  test_assert(rmax <= +1.0);

  rmean[0] /= NLARGE;
  rmean[1] /= NLARGE;
  rmean[2] /= NLARGE;

  test_assert(fabs(rmean[0]) < STAT_TOLERANCE);
  test_assert(fabs(rmean[1]) < STAT_TOLERANCE);
  test_assert(fabs(rmean[2]) < STAT_TOLERANCE);

  return 0;
}

/*****************************************************************************
 *
 *  util_jacobi_check
 *
 *****************************************************************************/

int util_jacobi_check(void) {

  int ifail = 0;

  {
    double a[3][3] = {0};
    double evals[3] = {0};
    double evecs[3][3] = {0};

    ifail = util_jacobi(a, evals, evecs);
    assert(ifail == 0);
    if (evals[0] != 0.0)    ifail = -1;
    if (evals[1] != 0.0)    ifail = -2;
    if (evals[2] != 0.0)    ifail = -3;
    if (evecs[0][0] != 1.0) ifail = -4;
    if (evecs[1][1] != 1.0) ifail = -5;
    if (evecs[2][2] != 1.0) ifail = -6;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_dpythag_check
 *
 *****************************************************************************/

int util_dpythag_check(void) {

  int ifail = 0;

  {
    double a = 0.0;
    ifail = util_dpythag(3.0, 4.0, &a);
    if (fabs(a - 5.0) > DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  {
    double a = -1.0;
    ifail = util_dpythag(0.0, 0.0, &a);
    if (a != 0.0) ifail = -1;
    assert(ifail == 0);
  }

  {
    double a = 0.0;
    ifail = util_dpythag(12.0, 5.0, &a);
    if (fabs(a - 13.0) > DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_str_tolower_check
 *
 *  Don't stray into tests of standard tolower()
 *
 *****************************************************************************/

int util_str_tolower_check(void) {

  char s1[BUFSIZ] = {0};

  /* basic */
  strncpy(s1, "TesT", 5);
  util_str_tolower(s1, strlen(s1));
  assert(strncmp(s1, "test", 4) == 0);

  /* maxlen < len */
  strncpy(s1, "AbCD", 5);
  util_str_tolower(s1, 3);
  assert(strncmp(s1, "abcD", 4) == 0);

  /* a longer example */
  strncpy(s1, "__12345ABCDE__", 15);
  util_str_tolower(s1, strlen(s1));
  assert(strncmp(s1, "__12345abcde__", 14) == 0);

  return 0;
}

/*****************************************************************************
 *
 *  util_rectangle_conductance_check
 *
 *****************************************************************************/

int util_rectangle_conductance_check(void) {

  int ierr = 0;
  double c = 0.0;

  {
    /* w must be the larger */
    double h = 1.0;
    double w = 2.0;

    ierr = util_rectangle_conductance(w, h, &c);
    assert(ierr == 0);
    ierr = util_rectangle_conductance(h, w, &c); /* Wrong! */
    assert(ierr != 0);
  }

  {
    double h = 2.0;
    double w = 2.0;
    ierr = util_rectangle_conductance(w, h, &c);
    assert(ierr == 0);
  }


  {
    /* Value used for some regression tests */
    double h = 30.0;
    double w = 62.0;
    ierr = util_rectangle_conductance(w, h, &c);
    assert(ierr == 0);
    assert(fabs(c - 97086.291)/97086.291 < FLT_EPSILON);
  }

  return ierr;
}
