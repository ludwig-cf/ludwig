/*****************************************************************************
 *
 *  test_util_ellipsoid.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023-2024 The University of Edinburgh
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
#include "util_vector.h"
#include "util_ellipsoid.h"

int test_util_q4_product(void);
int test_util_q4_from_euler_angles(void);
int test_util_q4_to_euler_angles(void);
int test_util_q4_rotate_vector(void);
int test_util_q4_from_omega(void);
int test_util_q4_is_inside_ellipsoid(void);
int test_util_q4_inertia_tensor(void);

int test_util_q4_r(void);
int test_util_q4_distance_to_tangent_plane(void);

int test_util_ellipsoid_is_sphere(void);
int test_util_ellipsoid_euler_from_vectors(void);
int test_util_ellipsoid_prolate_settling_velocity(void);
int test_util_spheroid_surface_normal(void);
int test_util_spheroid_surface_tangent(void);
int test_util_discrete_volume_ellipsoid(void);

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
  test_util_q4_is_inside_ellipsoid();
  test_util_q4_inertia_tensor();

  test_util_q4_r();
  test_util_q4_distance_to_tangent_plane();

  test_util_ellipsoid_is_sphere();
  test_util_ellipsoid_euler_from_vectors();
  test_util_ellipsoid_prolate_settling_velocity();
  test_util_spheroid_surface_normal();
  test_util_spheroid_surface_tangent();
  test_util_discrete_volume_ellipsoid();

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

/*****************************************************************************
 *
 *  test_util_q4_is_inside_ellipsoid
 *
 *****************************************************************************/

int test_util_q4_is_inside_ellipsoid(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    /* Inside */
    double q[4]     = {1.0, 0.0, 0.0, 0.0};
    double elabc[3] = {7.5, 2.5, 2.5};
    double r[3]     = {6.5, 0.0, 0.0};
    ifail = util_q4_is_inside_ellipsoid(q, elabc, r);
    assert(ifail);
  }

  {
    double q[4]     = {0};
    double elabc[3] = {7.5, 2.5, 2.5};
    double r[3]     = {6.5, 0.0, 0.0};
    /* Rotate ellipsoid enough that r is outside */
    util_q4_from_euler_angles(pi/6.0, 0.0, 0.0, q);
    ifail = util_q4_is_inside_ellipsoid(q, elabc, r);
    assert(ifail == 0);
  }

  {
    /* Not inside: easy ... */
    double q[4] = {1.0, 0.0, 0.0, 0.0};
    double a[3] = {7.5, 2.5, 2.5};
    double r[3] = {7.6, 0.0, 0.0};
    ifail = util_q4_is_inside_ellipsoid(q, a, r);
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_ellipsoid_is_sphere
 *
 *****************************************************************************/

int test_util_ellipsoid_is_sphere(void) {

  int ifail = 0;

  {
    /* Yes */
    double elabc[3] = {7.5, 7.5, 7.5};
    ifail = util_ellipsoid_is_sphere(elabc);
    assert(ifail != 0);
  }

  {
    /* No. */
    double elabc[3] = {7.5, 2.5, 2.5};
    ifail = util_ellipsoid_is_sphere(elabc);
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_ellipsoid_euler_from_vectors
 *
 *****************************************************************************/

int test_util_ellipsoid_euler_from_vectors(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    /* Fail a = 0 */
    double a[3]     = {0.0, 0.0, 0.0};
    double b[3]     = {1.0, 0.0, 0.0};
    double euler[3] = {0.0, 0.0, 0.0};
    ifail = util_ellipsoid_euler_from_vectors(a, b, euler);
    assert(ifail != 0);
  }

  {
    /* Fail b = 0 */
    double a[3]     = {0.0, 1.0, 0.0};
    double b[3]     = {0.0, 0.0, 0.0};
    double euler[3] = {0.0, 0.0, 0.0};
    ifail = util_ellipsoid_euler_from_vectors(a, b, euler);
    assert(ifail != 0);
  }

  {
    /* Fail linear dependency */
    double a[3]     = {1.0, 0.0, 0.0};
    double b[3]     = {2.0, 0.0, 0.0};
    double euler[3] = {0.0, 0.0, 0.0};
    ifail = util_ellipsoid_euler_from_vectors(a, b, euler);
    assert(ifail != 0);
  }

  {
    /* This one is ok. */
    double a[3]     = {0.0, 0.0, 1.0};
    double b[3]     = {0.0, 1.0, 0.0};
    double euler[3] = {0.0, 0.0, 0.0};
    ifail = util_ellipsoid_euler_from_vectors(a, b, euler);
    assert(ifail == 0);
    if (fabs(euler[0] + 0.5*pi) > DBL_EPSILON) ifail = -1;
    if (fabs(euler[1] - 0.5*pi) > DBL_EPSILON) ifail = -1;
    if (fabs(euler[2] - 0.5*pi) > DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_ellipsoid_prolate_settling_velocity
 *
 *****************************************************************************/

int test_util_ellipsoid_prolate_settling_velocity(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    /* Ellipsoid */
    double a = 1.0;
    double b = 0.5;
    double eta  = 1.0/(6.0*pi);
    double f    = 1.0;
    double u[2] = {0};

    ifail = util_ellipsoid_prolate_settling_velocity(a, b, eta, f, u);
    assert(ifail == 0);

    assert(fabs(u[0] - 1.6612110) < FLT_EPSILON);
    assert(fabs(u[1] - 1.4504325) < FLT_EPSILON);
  }

  {
    /* Limit of a sphere */
    double a = 1.0;
    double b = 1.0;
    double eta  = 1.0/(6.0*pi);
    double f    = 1.0;
    double u[2] = {0};

    ifail = util_ellipsoid_prolate_settling_velocity(a, b, eta, f, u);
    assert(ifail == 0);

    assert(fabs(u[0] - 1.0) < DBL_EPSILON);
    assert(fabs(u[1] - 1.0) < DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_spheroid_surface_normal
 *
 *  This is a test for the normal, but we also check the tangent is
 *  consistent with the normal.
 *
 *****************************************************************************/

int test_util_spheroid_surface_normal(void) {

  int ifail = 0;

  /* Position exactly at surface ... */
  {
    double elabc[3] = {7.5, 2.5, 2.5};
    double m[3]     = {1.0, 0.0, 0.0};
    double r[3]     = {7.5, 0.0, 0.0};
    double rn[3]    = {0};

    ifail = util_spheroid_surface_normal(elabc, m, r, rn);
    assert(ifail == 0);
    /* This coming in at around 2 epsilon ... */
    assert(fabs(rn[0] - 1.0) < 4.0*DBL_EPSILON);
    assert(fabs(rn[1] - 0.0) <     DBL_EPSILON);
    assert(fabs(rn[2] - 0.0) <     DBL_EPSILON);

    ifail = util_spheroid_surface_tangent(elabc, m, r, rn);
    assert(ifail == -999);
  }

  /* New position slightly outside ... */
  {
    double elabc[3] = {7.5, 2.5, 2.5};
    double m[3]     = {1.0, 0.0, 0.0};
    double r[3]     = {7.6, 0.0, 0.0};
    double rn[3]    = {0};

    ifail = util_spheroid_surface_normal(elabc, m, r, rn);
    assert(ifail == 0);
    assert(rn[0] > 0.0); /* Not a unit vector at this time */
    assert(fabs(rn[1] - 0.0) < DBL_EPSILON);
    assert(fabs(rn[2] - 0.0) < DBL_EPSILON);
    /* Tangent still wrong */
    ifail = util_spheroid_surface_tangent(elabc, m, r, rn);
    assert(ifail == -999);
  }

  /* Position slightly inside ... */
  {
    double elabc[3] = {7.5, 2.5, 2.5};
    double m[3]     = {1.0, 0.0, 0.0};
    double r[3]     = {7.3, 0.0, 0.0};
    double rn[3]    = {0};

    ifail = util_spheroid_surface_normal(elabc, m, r, rn);
    assert(ifail == 0);
    assert(rn[0] > 0.0); /* Not a unit vector */
    assert(fabs(rn[1] - 0.0) < DBL_EPSILON);
    assert(fabs(rn[2] - 0.0) < DBL_EPSILON);
    /* Tangent wrong */
    ifail = util_spheroid_surface_tangent(elabc, m, r, rn);
    assert(ifail == -999);
  }

  /* Position along short direction... */
  {
    double elabc[3] = {7.5, 2.5, 2.5};
    double m[3]     = {1.0, 0.0, 0.0};
    double r[3]     = {0.0, 2.5, 0.0};
    double rn[3]    = {0};
    double rt[3]    = {0};

    ifail = util_spheroid_surface_normal(elabc, m, r, rn);
    assert(ifail == 0);
    assert(fabs(rn[0] - 0.0) <     DBL_EPSILON);
    assert(fabs(rn[1] - 1.0) < 4.0*DBL_EPSILON);
    assert(fabs(rn[2] - 0.0) <     DBL_EPSILON);

    ifail = util_spheroid_surface_tangent(elabc, m, r, rt);
    assert(ifail == 0);
    {
      double dot = dot_product(rn, rt);
      if (fabs(dot) > DBL_EPSILON) ifail = -1;
      assert(ifail == 0);
    }
  }

  /* General position exactly on surface... */
  {
    double a        = 7.5;
    double b        = 2.5;
    double c        = 2.5;
    double elabc[3] = {  a,   b,   c};
    double m[3]     = {1.0, 0.0, 0.0};
    double r[3]     = {3.0, 2.0, 0.0};
    double rn[3]    = {0};
    double rt[3]    = {0};

    /* force (x/a)^2 + (y/b)^2 + (z/c)^2 = 1 */
    r[2] = c*sqrt(1.0 - r[0]*r[0]/(a*a) - r[1]*r[1]/(b*b));

    ifail = util_spheroid_surface_normal(elabc, m, r, rn);
    ifail = util_spheroid_surface_tangent(elabc, m, r, rt);
    assert(ifail == 0);

    /* Work out true normal 2(x/a^2, y/b^2, z/c^2) unit vector */
    {
      double u[3] = {2.0*r[0]/(a*a), 2.0*r[1]/(b*b), 2.0*r[2]/(c*c)};
      util_vector_normalise(3, u);
      assert(fabs(rn[0] - u[0]) < DBL_EPSILON);
      assert(fabs(rn[1] - u[1]) < DBL_EPSILON);
      assert(fabs(rn[2] - u[2]) < DBL_EPSILON);
    }
    {
      double dot = dot_product(rn, rt);
      if (fabs(dot) > DBL_EPSILON) ifail = -1;
      assert(ifail == 0);
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_spheroid_surface_tangent
 *
 *****************************************************************************/

int test_util_spheroid_surface_tangent(void) {

  int ifail = 0;

  /* Position along short direction... */
  {
    double elabc[3] = {7.5, 2.5, 2.5};
    double m[3]     = {1.0, 0.0, 0.0};
    double r[3]     = {0.0, 2.5, 0.0};
    double rt[3]    = {0};

    ifail = util_spheroid_surface_tangent(elabc, m, r, rt);
    assert(ifail == 0);
    assert(fabs(rt[0] - -1.0) < DBL_EPSILON);
    assert(fabs(rt[1] -  0.0) < DBL_EPSILON);
    assert(fabs(rt[2] -  0.0) < DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_inertia_tensor
 *
 *****************************************************************************/

int test_util_q4_inertia_tensor(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  {
    /* No prizes for this one ... */
    double moment[3] = {1.0, 2.0, 3.0};
    double q[4]      = {1.0, 0.0, 0.0, 0.0};
    double mi[3][3]  = {0};
    util_q4_inertia_tensor(q, moment, mi);
    assert(fabs(mi[0][0] - 1.0) < DBL_EPSILON);
    assert(fabs(mi[0][1] - 0.0) < DBL_EPSILON);
    assert(fabs(mi[0][2] - 0.0) < DBL_EPSILON);
    assert(fabs(mi[1][0] - 0.0) < DBL_EPSILON);
    assert(fabs(mi[1][1] - 2.0) < DBL_EPSILON);
    assert(fabs(mi[1][2] - 0.0) < DBL_EPSILON);
    assert(fabs(mi[2][0] - 0.0) < DBL_EPSILON);
    assert(fabs(mi[2][1] - 0.0) < DBL_EPSILON);
    assert(fabs(mi[2][2] - 3.0) < DBL_EPSILON);
  }

  {
    /* Rotate the quaternion ... */
    double moment[3] = {1.0, 2.0, 3.0};
    double q[4]      = {0};
    double mi[3][3]  = {0};
    double phi       = pi/2.0;
    double theta     = pi/3.0;
    double psi       = pi/4.0;
    util_q4_from_euler_angles(phi, theta, psi, q);
    util_q4_inertia_tensor(q, moment, mi);
    /* Sample of ... */
    assert(fabs(mi[0][0] - 2.6250) < 8.0*DBL_EPSILON);
    assert(fabs(mi[0][1] - 0.2500) < 8.0*DBL_EPSILON);
    assert(fabs(mi[1][0] - 0.2500) < 8.0*DBL_EPSILON);
    assert(fabs(mi[1][1] - 1.5000) < 8.0*DBL_EPSILON);
    assert(fabs(mi[2][2] - 1.8750) < 8.0*DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_discrete_volume_ellipsoid
 *
 *****************************************************************************/

int test_util_discrete_volume_ellipsoid(void) {

  int ifail = 0;

  {
    /* prolate example: vol = 3 units */
    double r0[3]  = {1.00, 1.00, 1.00};
    double abc[3] = {1.01, 0.25, 0.25};
    double q4[4]  = {1.0, 0.0, 0.0, 0.0};
    int ivol = 0;
    double rvol = 0.0;
    ivol = util_discrete_volume_ellipsoid(abc, r0, q4, &rvol);
    if (ivol != 3) ifail = -1;
    assert(ifail == 0);
    assert(fabs(rvol - 1.0*ivol) < DBL_EPSILON);
  }

  {
    /* oblate example: vol = 4 units */
    double r0[3]  = {1.00, 1.50, 1.50};
    double abc[3] = {0.25, 0.71, 0.71};    /* slightly larger than sqrt(2) */
    double q4[4]  = {1.0, 0.0, 0.0, 0.0};
    int ivol = 0;
    double rvol = 0.0;
    ivol = util_discrete_volume_ellipsoid(abc, r0, q4, &rvol);
    if (ivol != 4) ifail = -1;
    assert(ifail == 0);
    assert(fabs(rvol - 1.0*ivol) < DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_r
 *
 *  Quaternion to rotation matrix.
 *
 *****************************************************************************/

int test_util_q4_r(void) {

  int ifail = 0;
  PI_DOUBLE(pi);

  /* trivial case */
  {
    double q[4] = {1.0, 0.0, 0.0, 0.0};
    double r[3][3] = {0};

    util_q4_to_r(q, r);

    assert(fabs(r[0][0] - 1.0) < DBL_EPSILON);
    assert(fabs(r[0][1] - 0.0) < DBL_EPSILON);
    assert(fabs(r[0][2] - 0.0) < DBL_EPSILON);
    assert(fabs(r[1][0] - 0.0) < DBL_EPSILON);
    assert(fabs(r[1][1] - 1.0) < DBL_EPSILON);
    assert(fabs(r[1][2] - 0.0) < DBL_EPSILON);
    assert(fabs(r[2][0] - 0.0) < DBL_EPSILON);
    assert(fabs(r[2][1] - 0.0) < DBL_EPSILON);
    assert(fabs(r[2][2] - 1.0) < DBL_EPSILON);
  }

  /* Rotate around z */
  {
    double phi = pi/6.0;
    double theta = 0.0;
    double psi = 0.0;
    double q[4] = {0};
    double r[3][3] = {0};

    ifail = util_q4_from_euler_angles(phi, theta, psi, q);
    assert(ifail == 0);

    util_q4_to_r(q, r);

    assert(fabs(r[0][0] - cos(phi)) < DBL_EPSILON);
    assert(fabs(r[0][1] + sin(phi)) < DBL_EPSILON);
    assert(fabs(r[0][2] - 0.0     ) < DBL_EPSILON);
    assert(fabs(r[1][0] - sin(phi)) < DBL_EPSILON);
    assert(fabs(r[1][1] - cos(phi)) < DBL_EPSILON);
    assert(fabs(r[1][2] - 0.0     ) < DBL_EPSILON);
    assert(fabs(r[2][0] - 0.0     ) < DBL_EPSILON);
    assert(fabs(r[2][1] - 0.0     ) < DBL_EPSILON);
    assert(fabs(r[2][2] - 1.0     ) < DBL_EPSILON);
  }

  /* Rotate around x (no z rotation) */
  {
    double phi = 0.0;
    double theta = pi/3.0;
    double psi = 0.0;
    double q[4] = {0};
    double r[3][3] = {0};

    ifail = util_q4_from_euler_angles(phi, theta, psi, q);
    assert(ifail == 0);

    util_q4_to_r(q, r);

    assert(fabs(r[0][0] - 1.0       ) < 2.0*DBL_EPSILON); /* v. close */
    assert(fabs(r[0][1] - 0.0       ) <     DBL_EPSILON);
    assert(fabs(r[0][2] - 0.0       ) <     DBL_EPSILON);
    assert(fabs(r[1][0] - 0.0       ) <     DBL_EPSILON);
    assert(fabs(r[1][1] - cos(theta)) <     DBL_EPSILON);
    assert(fabs(r[1][2] + sin(theta)) <     DBL_EPSILON);
    assert(fabs(r[2][0] - 0.0       ) <     DBL_EPSILON);
    assert(fabs(r[2][1] - sin(theta)) <     DBL_EPSILON);
    assert(fabs(r[2][2] - cos(theta)) <     DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_util_q4_distance_to_tangent_plane
 *
 *****************************************************************************/

int test_util_q4_distance_to_tangent_plane(void) {

  int ifail = 0;

  double a = 3.0;
  double b = 4.0;
  double c = 5.0;
  double abc[3] = {a, b, c};

  PI_DOUBLE(pi);

  /* Principal axes aligned with lab frame. */
  {
    double q[4] = {1.0, 0.0, 0.0, 0.0};

    /* x-y plane */
    {
      double d = -1.0;
      double nhat[3] = {0.0, 0.0, 1.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - c) < DBL_EPSILON);
    }

    /* y-z plane */
    {
      double d = -1.0;
      double nhat[3] = {1.0, 0.0, 0.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - a) < DBL_EPSILON);
    }

    /* x-z plane */
    {
      double d = -1.0;
      double nhat[3] = {0.0, 1.0, 0.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - b) < DBL_EPSILON);
    }
  }


  /* z-rotation of principal axes (30^o)*/
  {
    double phi   = pi/6.0;
    double theta = 0.0;
    double psi   = 0.0;
    double q[4] = {0};

    ifail = util_q4_from_euler_angles(phi, theta, psi, q);
    assert(ifail == 0);

    /* x-y plane (should be unchanged) */
    {
      double d = -1.0;
      double nhat[3] = {0.0, 0.0, -1.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - c) < DBL_EPSILON);
    }
    /* y-z plane */
    {
      double d = -1.0;
      double d0 = a*a*cos(phi)*cos(phi) + b*b*sin(phi)*sin(phi);
      double nhat[3] = {-1.0, 0.0, 0.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - sqrt(d0)) < DBL_EPSILON);
    }
  }


  /* z-rotation of principal axes (60^o)*/
  {
    double phi   = pi/3.0;
    double theta = 0.0;
    double psi   = 0.0;
    double q[4] = {0};

    ifail = util_q4_from_euler_angles(phi, theta, psi, q);
    assert(ifail == 0);

    /* x-y plane (should be unchanged) */
    {
      double d = -1.0;
      double nhat[3] = {0.0, 0.0, -1.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - c) < FLT_EPSILON);
    }
    /* y-z plane */
    {
      double d = -1.0;
      double d0 = a*a*cos(phi)*cos(phi) + b*b*sin(phi)*sin(phi);
      double nhat[3] = {-1.0, 0.0, 0.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - sqrt(d0)) < DBL_EPSILON);
    }
  }


  /* z-rotation of principal axes (90^o)*/
  {
    double phi   = pi/2.0;
    double theta = 0.0;
    double psi   = 0.0;
    double q[4] = {0};

    ifail = util_q4_from_euler_angles(phi, theta, psi, q);
    assert(ifail == 0);

    {
      double d = -1.0;
      double nhat[3] = {+1.0, 0.0, 0.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - b) < FLT_EPSILON);
    }
  }

  /* Rotation about z- and x'- axes. */
  /* This differentiates q and q^*, unlike the previous cases. */
  {
    double phi   = pi/2.0;
    double theta = pi/2.0;
    double psi   = 0.0;
    double q[4] = {0};

    ifail = util_q4_from_euler_angles(phi, theta, psi, q);
    assert(ifail == 0);

    {
      double d = -1.0;
      double nhat[3] = {+1.0, 0.0, 0.0};
      d = util_q4_distance_to_tangent_plane(abc, q, nhat);
      assert(fabs(d - c) < DBL_EPSILON);
    }
  }

  return ifail;
}
