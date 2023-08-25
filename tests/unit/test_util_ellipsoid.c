/*****************************************************************************
 *
 *  test_util_ellipsoid.c
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

#include "pe.h"
#include "util.h"
#include "util_ellipsoid.h"

int test_util_q4_product(void);
int test_util_q4_from_euler_angles(void);
int test_util_q4_to_euler_angles(void);
int test_util_q4_rotate_vector(void);
int test_util_q4_from_omega(void);

/*****************************************************************************
 *
 *  test_util_ellispoid_suite
 *
 *****************************************************************************/

int test_util_ellipsoid_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_q4_product();
  test_util_q4_rotate_vector();
  test_util_q4_from_omega();
  test_util_q4_from_euler_angles();
  test_util_q4_to_euler_angles();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_q4_product
 *
 *****************************************************************************/

int test_util_q4_product(void) {

  int ifail = 0;

  {
    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4] = {5.0, 6.0, 7.0, 8.0};
    double c[4] = {0};

    util_q4_product(a, b, c);

    if (fabs(c[0] - -60.0) > FLT_EPSILON) ifail = -1;
    if (fabs(c[1] -  12.0) > FLT_EPSILON) ifail = -1;
    if (fabs(c[2] -  30.0) > FLT_EPSILON) ifail = -1;
    if (fabs(c[3] -  24.0) > FLT_EPSILON) ifail = -1;
    assert(ifail == 0);

    util_q4_product(b, a, c);

    if (fabs(c[0] - -60.0) > FLT_EPSILON) ifail = -1;
    if (fabs(c[1] -  20.0) > FLT_EPSILON) ifail = -1;
    if (fabs(c[2] -  14.0) > FLT_EPSILON) ifail = -1;
    if (fabs(c[3] -  32.0) > FLT_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_rotate_vector
 *
 *****************************************************************************/

int test_util_q4_rotate_vector(void) {

  int ifail = 0;

  {
    double q[4] = {1.0, 0.0, 0.0, 0.0};
    double a[3] = {1.0, 1.0, 1.0};
    double b[3] = {0.0, 0.0, 0.0};

    util_q4_rotate_vector(q, a, b);
    if (fabs(a[0] - b[0]) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(a[0] - b[0]) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(a[0] - b[0]) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_from_omega
 *
 *****************************************************************************/

int test_util_q4_from_omega(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    /* Zero case */
    double omega[3] = {0};
    double dt       = 0.5;
    double q[4]     = {0};

    util_q4_from_omega(omega, dt, q);
    if (fabs(q[0] - 1.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[1] - 0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[2] - 0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[3] - 0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
  }

  {
    /* A test for q[0] */
    double omega[3] = {1.0, 0.0, 0.0};
    double dt       = pi;
    double q[4]     = {0};

    util_q4_from_omega(omega, dt, q);
    if (fabs(q[0] - -1.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[1] -  0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[2] -  0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[3] -  0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
  }

  {
    /* A test for q[1] */
    double omega[3] = {1.0, 0.0, 0.0};
    double dt       = 0.5*pi;
    double q[4]     = {0};

    util_q4_from_omega(omega, dt, q);
    if (fabs(q[0] -  0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[1] -  1.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[2] -  0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
    if (fabs(q[3] -  0.0) > DBL_EPSILON) ifail -= 1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_from_euler_angles
 *
 *****************************************************************************/

int test_util_q4_from_euler_angles(void) {

  int ifail = 0;

  {
    double phi   = 0.0;
    double theta = 0.0;
    double psi   = 0.0;
    double q[4]  = {0};

    util_q4_from_euler_angles(phi, theta, psi, q);
    assert(fabs(q[0] - 1.0) < DBL_EPSILON);
    assert(fabs(q[1] - 0.0) < DBL_EPSILON);
    assert(fabs(q[2] - 0.0) < DBL_EPSILON);
    assert(fabs(q[3] - 0.0) < DBL_EPSILON);
  }

  {
    double phi   = 4.0*atan(1.0);
    double theta = 0.0;
    double psi   = 0.0;
    double q[4]  = {0};

    util_q4_from_euler_angles(phi, theta, psi, q);
    assert(fabs(q[0] - 0.0) < DBL_EPSILON);
    assert(fabs(q[1] - 0.0) < DBL_EPSILON);
    assert(fabs(q[2] - 0.0) < DBL_EPSILON);
    assert(fabs(q[3] - 1.0) < DBL_EPSILON);
  }

  {
    double phi   = 4.0*atan(1.0);
    double theta = 4.0*atan(1.0);
    double psi   = 0.0;
    double q[4]  = {0};

    util_q4_from_euler_angles(phi, theta, psi, q);
    assert(fabs(q[0] - 0.0) < DBL_EPSILON);
    assert(fabs(q[1] - 0.0) < DBL_EPSILON);
    assert(fabs(q[2] - 1.0) < DBL_EPSILON);
    assert(fabs(q[3] - 0.0) < DBL_EPSILON);
  }

  {
    double phi   = 0.0;
    double theta = 4.0*atan(1.0);
    double psi   = 4.0*atan(1.0);
    double q[4]  = {0};

    util_q4_from_euler_angles(phi, theta, psi, q);
    assert(fabs(q[0] - 0.0) < DBL_EPSILON);
    assert(fabs(q[1] - 0.0) < DBL_EPSILON);
    assert(fabs(q[2] + 1.0) < DBL_EPSILON);
    assert(fabs(q[3] - 0.0) < DBL_EPSILON);
  }

  {
    double phi   = 4.0*atan(1.0);
    double theta = 4.0*atan(1.0);
    double psi   = 4.0*atan(1.0);
    double q[4]  = {0};

    util_q4_from_euler_angles(phi, theta, psi, q);
    assert(fabs(q[0] - 0.0) < DBL_EPSILON);
    assert(fabs(q[1] - 1.0) < DBL_EPSILON);
    assert(fabs(q[2] - 0.0) < DBL_EPSILON);
    assert(fabs(q[3] - 0.0) < DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_to_euler_angles
 *
 *****************************************************************************/

int test_util_q4_to_euler_angles(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    double phi   = -999.999;
    double theta = -999.999;
    double psi   = -999.999;
    double q[4]  = {-999.999, -999.999, -999.999, -999.999};

    ifail = util_q4_to_euler_angles(q, &phi, &theta, &psi);
    assert(ifail != 0);
  }

  {
    double phi   = -999.999;
    double theta = -999.999;
    double psi   = -999.999;
    double q[4]  = {1.0, 0.0, 0.0, 0.0};

    ifail = util_q4_to_euler_angles(q, &phi, &theta, &psi);
    assert(ifail == 0);
    assert(fabs(phi   - 0.0) < DBL_EPSILON);
    assert(fabs(theta - 0.0) < DBL_EPSILON);
    assert(fabs(psi   - 0.0) < DBL_EPSILON);
  }

  {
    double phi   = -999.999;
    double theta = -999.999;
    double psi   = -999.999;
    double q[4]  = {0.0, 1.0, 0.0, 0.0};

    ifail = util_q4_to_euler_angles(q, &phi, &theta, &psi);
    assert(ifail == 0);
    assert(fabs(phi   - 0.0) < DBL_EPSILON);
    assert(fabs(theta - pi)  < DBL_EPSILON);
    assert(fabs(psi   - 0.0) < DBL_EPSILON);
  }

  {
    double phi   = -999.999;
    double theta = -999.999;
    double psi   = -999.999;
    double q[4]  = {0.0, 0.0, 1.0, 0.0};

    ifail = util_q4_to_euler_angles(q, &phi, &theta, &psi);
    assert(ifail == 0);
    assert(fabs(phi   - pi)  < FLT_EPSILON);
    assert(fabs(theta - pi)  < FLT_EPSILON);
    assert(fabs(psi   - 0.0) < DBL_EPSILON);
  }

  {
    double phi   = -999.999;
    double theta = -999.999;
    double psi   = -999.999;
    double q[4]  = {0.0, 0.0, 0.0, 1.0};

    ifail = util_q4_to_euler_angles(q, &phi, &theta, &psi);
    assert(ifail == 0);
    if (fabs(phi   - pi)  > FLT_EPSILON) ifail = -1;
    if (fabs(theta - 0.0) > DBL_EPSILON) ifail = -1;
    if (fabs(psi   - 0.0) > DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  {
    double phi   = -999.999;
    double theta = -999.999;
    double psi   = -999.999;
    double q[4]  = {0.0, 0.5, 0.5, 0.0};

    ifail = util_q4_to_euler_angles(q, &phi, &theta, &psi);
    assert(ifail == 0);

    if (fabs(phi   - 0.0)    > DBL_EPSILON) ifail = -1;
    if (fabs(theta - 0.5*pi) > FLT_EPSILON) ifail = -1;
    if (fabs(psi   - 0.0)    > DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}
