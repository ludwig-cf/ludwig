/******************************************************************************
 *
 *  test_magnetic.c
 *
 *  Test of uniform magnetic field interface.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "tests.h"
#include "coords.h"
#include "magnetic_field.h"

static void test_magnetic_field_uniform(void);
static void cross_product(const double a[3], const double b[3], double c[3]);

int main (int argc, char ** argv) {

  test_magnetic_field_uniform();

  return 0;
}


/*****************************************************************************
 *
 *  test_magnetic_field_uniform
 *
 *****************************************************************************/

static void test_magnetic_field_uniform(void) {

  const double b1[3] = {0.0, 0.0, -1.0};
  const double b2[3] = {1.0, 2.0, 3.0};
  double mu[3] = {-1.0, 0.5, 2.0};

  double b0[3] = {1.0, 1.0, 1.0};
  double force[3];
  double torque[3];
  double t[3];

  info("Testing uniform magnetic field...");

  /* Default field must be zero */

  magnetic_field_b0(b0);
  
  test_assert(fabs(b0[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(b0[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(b0[Z]) < TEST_DOUBLE_TOLERANCE);

  /* Set a field and check it is returned correctly. */

  magnetic_field_b0_set(b1);
  magnetic_field_b0(b0);

  test_assert(fabs(b0[X] - b1[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(b0[Y] - b1[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(b0[Z] - b1[Z]) < TEST_DOUBLE_TOLERANCE);

  magnetic_field_force(mu, force);

  test_assert(fabs(force[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(force[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(force[Z]) < TEST_DOUBLE_TOLERANCE);

  magnetic_field_torque(mu, torque);

  cross_product(mu, b1, t);

  test_assert(fabs(torque[X] - t[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(torque[X] - t[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(torque[X] - t[X]) < TEST_DOUBLE_TOLERANCE);

  magnetic_field_b0_set(b2);
  magnetic_field_b0(b0);

  test_assert(fabs(b2[X] - b0[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(b2[Y] - b0[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(b2[Z] - b0[Z]) < TEST_DOUBLE_TOLERANCE);

  magnetic_field_force(mu, force);

  test_assert(fabs(force[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(force[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(force[Z]) < TEST_DOUBLE_TOLERANCE);

  magnetic_field_torque(mu, torque);

  cross_product(mu, b2, t);

  test_assert(fabs(torque[X] - t[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(torque[X] - t[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(torque[X] - t[X]) < TEST_DOUBLE_TOLERANCE);

  info("ok\n\n");
  
  return;
}

/*****************************************************************************
 *
 * cross_product
 *
 *****************************************************************************/

static void cross_product(const double a[3], const double b[3], double c[3]) {

  c[X] = a[Y]*b[Z] - a[Z]*b[Y];
  c[Y] = a[Z]*b[X] - a[X]*b[Z];
  c[Z] = a[X]*b[Y] - a[Y]*b[X];

  return;
}
