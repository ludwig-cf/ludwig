/*****************************************************************************
 *
 *  test_util_vector.c
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

#include "pe.h"
#include "util_vector.h"

int test_util_vector_dot_product(void);
int test_util_vector_cross_product(void);
int test_util_vector_l2_norm(void);
int test_util_vector_normalise(void);
int test_util_vector_copy(void);
int test_util_vector_orthonormalise(void);
int test_util_vector_basis_to_dcm(void);
int test_util_vector_dcm_to_euler(void);

/*****************************************************************************
 *
 *  test_util_vector_suite
 *
 *****************************************************************************/

int test_util_vector_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_vector_dot_product();
  test_util_vector_cross_product();
  test_util_vector_l2_norm();
  test_util_vector_normalise();
  test_util_vector_copy();
  test_util_vector_orthonormalise();
  test_util_vector_basis_to_dcm();
  test_util_vector_dcm_to_euler();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_vector_dot_product
 *
 *****************************************************************************/

int test_util_vector_dot_product(void) {

  int ifail = 0;

  {
    double a[3] = {2.0, 3.0, 4.0};
    double b[3] = {-0.5, -1.0/3.0, 0.75};
    double dot = util_vector_dot_product(a, b);
    assert(fabs(dot - 1.0) < DBL)EPSILON);
    if (fabs(dot - 1.0) > DBL_EPSILON) ifail -= 1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_vector_cross_product
 *
 *****************************************************************************/

int test_util_vector_cross_product(void) {

  int ifail = 0;

  {
    double a[3] = {1.0, 0.0, 0.0};
    double b[3] = {0.0, 1.0, 0.0};
    double c[3] = {0};
    util_vector_cross_product(c, a, b);
    assert(fabs(c[0] - 0.0) < DBL_EPSILON);
    assert(fabs(c[1] - 0.0) < DBL_EPSILON);
    assert(fabs(c[2] - 1.0) < DBL_EPSILON);
  }

  return ifail;
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

/*****************************************************************************
 *
 *  test_util_vector_orthonormalise
 *
 *****************************************************************************/

int test_util_vector_orthonormalise(void) {

  int ifail = 0;

  {
    double a[3] = {0.0, 0.0, -1.0};
    double b[3] = {0.0, 0.0,  1.0};
    ifail = util_vector_orthonormalise(a, b);
    assert(ifail == -2);
  }

  {
    double a[3] = {1.0, 0.0, 0.0};
    double b[3] = {1.0, 1.0, 0.0};
    ifail = util_vector_orthonormalise(a, b);
    assert(ifail == 0);
    assert(fabs(b[0] - 0.0) < DBL_EPSILON);
    assert(fabs(b[1] - 1.0) < DBL_EPSILON);
    assert(fabs(b[2] = 0.0) < DBL_EPSILON);
  }

  {
    double a[3] = {0.0, 2.0, 0.0};
    double b[3] = {1.0, 1.0, 1.0};
    ifail = util_vector_orthonormalise(a, b);
    assert(ifail == 0);
    assert(fabs(b[0] - 1.0/sqrt(2.0)) < DBL_EPSILON);
    assert(fabs(b[1] -          0.0 ) < DBL_EPSILON);
    assert(fabs(b[2] - 1.0/sqrt(2.0)) < DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_vector_basis_to_dcm
 *
 *  One could also test that the tranpose of the DCM is its inverse.
 *
 *****************************************************************************/

int test_util_vector_basis_to_dcm(void) {

  int ifail = 0;

  {
    /* The (a,b,c) should be orthonormal */
    double dcm[3][3] = {0};
    double a[3] = {1.0, 0.0, 0.0};
    double b[3] = {0.0, 1.0, 0.0};
    double c[3] = {0.0, 0.0, 1.0};
    util_vector_basis_to_dcm(a, b, c, dcm);
    assert(fabs(dcm[0][0] - 1.0) < DBL_EPSILON);
    assert(fabs(dcm[0][1] - 0.0) < DBL_EPSILON);
    assert(fabs(dcm[0][2] - 0.0) < DBL_EPSILON);
    assert(fabs(dcm[1][0] - 0.0) < DBL_EPSILON);
    assert(fabs(dcm[1][1] - 1.0) < DBL_EPSILON);
    assert(fabs(dcm[1][2] - 0.0) < DBL_EPSILON);
    assert(fabs(dcm[2][0] - 0.0) < DBL_EPSILON);
    assert(fabs(dcm[2][1] - 0.0) < DBL_EPSILON);
    assert(fabs(dcm[2][2] - 1.0) < DBL_EPSILON);
  }

  {
    double dcm[3][3] = {0};
    double a[3] = {1.0, 0.0, 0.0};
    double b[3] = {0.0, 1.0/sqrt(2.), 1.0/sqrt(2.0)};
    double c[3] = {0};
    util_vector_cross_product(c, a, b);
    util_vector_basis_to_dcm(a, b, c, dcm);

    assert(fabs(dcm[0][0] - 1.0)           < DBL_EPSILON);
    assert(fabs(dcm[0][1] - 0.0)           < DBL_EPSILON);
    assert(fabs(dcm[0][2] - 0.0)           < DBL_EPSILON);
    assert(fabs(dcm[1][0] - 0.0)           < DBL_EPSILON);
    assert(fabs(dcm[1][1] - 1.0/sqrt(2.0)) < DBL_EPSILON);
    assert(fabs(dcm[1][2] - 1.0/sqrt(2.0)) < DBL_EPSILON);
    assert(fabs(dcm[2][0] - 0.0)           < DBL_EPSILON);
    assert(fabs(dcm[2][1] + 1.0/sqrt(2.0)) < DBL_EPSILON);
    assert(fabs(dcm[2][2] - 1.0/sqrt(2.0)) < DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_vector_dcm_to_euler
 *
 *****************************************************************************/

int test_util_vector_dcm_to_euler(void) {

  int ifail = 0;
  double r2 = sqrt(2.0);
  double pi = 4.0*atan(1.0);

  {
    /* theta = acos(r_zz), phi = atan2(), psi = atan2() ... */
    double dcm[3][3] = {{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 1.0/r2}};
    double phi   = -1.0;
    double theta = -1.0;
    double psi   = -1.0;
    util_vector_dcm_to_euler(dcm, &phi, &theta, &psi);
    assert(fabs(theta - pi/4.0) < DBL_EPSILON);
    assert(fabs(phi   - pi/2.0) < DBL_EPSILON);
    assert(fabs(psi   - pi/2.0) < DBL_EPSILON);
  }

  {
    /* theta = 0, phi = atan2(r_xy, r_xx), psi = 0 */
    double dcm[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    double phi   = -1.0;
    double theta = -1.0;
    double psi   = -1.0;
    util_vector_dcm_to_euler(dcm, &phi, &theta, &psi);
    assert(fabs(theta - 0.0)    < DBL_EPSILON);
    assert(fabs(phi   - pi/2.0) < DBL_EPSILON);
    assert(fabs(psi   - 0.0)    < DBL_EPSILON);
  }

  return ifail;
}
