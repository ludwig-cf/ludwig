/*****************************************************************************
 *
 *  test_fe_lc_droplet.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "lc_droplet.h"

/* Forward declarations some additional testable routines */

int fe_lc_droplet_active_stress(const fe_lc_droplet_param_t * fp, double phi,
				double q[3][3], double s[3][3]);

/* Tests */

int test_fe_lc_droplet_active_stress(pe_t * pe);

/*****************************************************************************
 *
 *  test_fe_lc_droplet_suite
 *
 *****************************************************************************/

int test_fe_lc_droplet_suite() {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_fe_lc_droplet_active_stress(pe);

  pe_info(pe, "PASS     ./unit/test_fe_lc_droplet\n");
  pe_free(pe);
  
  return 0;
}

/*****************************************************************************
 *
 *  test_fe_lc_droplet_active_stress
 *
 *****************************************************************************/

__host__ int test_fe_lc_droplet_active_stress(pe_t * pe) {

  assert(pe);
  
  /* zeta_0 dependence */

  {
    double zeta0 = 2.0;
    double zeta1 = 0.0;
    double phi   = 1.0;
    fe_lc_droplet_param_t fp = {.zeta0 = zeta0, .zeta1 = zeta1};

    double q[3][3] = {0};
    double s[3][3] = {0};

    fe_lc_droplet_active_stress(&fp, phi, q, s);

    assert(fabs(s[X][X] - (-2.0/3.0)) < DBL_EPSILON);
    assert(fabs(s[X][Y] - ( 0.0    )) < DBL_EPSILON);
    assert(fabs(s[X][Z] - ( 0.0    )) < DBL_EPSILON);

    assert(fabs(s[Y][X] - ( 0.0    )) < DBL_EPSILON);
    assert(fabs(s[Y][Y] - (-2.0/3.0)) < DBL_EPSILON);
    assert(fabs(s[Y][Z] - ( 0.0    )) < DBL_EPSILON);

    assert(fabs(s[Z][X] - ( 0.0    )) < DBL_EPSILON);
    assert(fabs(s[Z][Y] - ( 0.0    )) < DBL_EPSILON);
    assert(fabs(s[Z][Z] - (-2.0/3.0)) < DBL_EPSILON);    
  }

  /* zeta_1 q_ab dependence; requires q_ab  */

  {
    double zeta0 = 0.0;
    double zeta1 = 2.0;
    double phi   = 1.0;
    fe_lc_droplet_param_t fp = {.zeta0 = zeta0, .zeta1 = zeta1};

    double q[3][3] = { {1., 2., 3.,}, {2., 4., 5.}, {3., 5., -3.} };
    double s[3][3] = {0};

    fe_lc_droplet_active_stress(&fp, phi, q, s);

    assert(fabs(s[X][X] - (-q[X][X]*zeta1)) < DBL_EPSILON);
    assert(fabs(s[X][Y] - (-q[X][Y]*zeta1)) < DBL_EPSILON);
    assert(fabs(s[X][Z] - (-q[X][Z]*zeta1)) < DBL_EPSILON);

    assert(fabs(s[Y][X] - (-q[Y][X]*zeta1)) < DBL_EPSILON);
    assert(fabs(s[Y][Y] - (-q[Y][Y]*zeta1)) < DBL_EPSILON);
    assert(fabs(s[Y][Z] - (-q[Y][Z]*zeta1)) < DBL_EPSILON);

    assert(fabs(s[Z][X] - (-q[Z][X]*zeta1)) < DBL_EPSILON);
    assert(fabs(s[Z][Y] - (-q[Z][Y]*zeta1)) < DBL_EPSILON);
    assert(fabs(s[Z][Z] - (-q[Z][Z]*zeta1)) < DBL_EPSILON);
  }
  
  /* phi dependence */

  {
    double zeta0 = 0.0;
    double zeta1 = 1.0;
    fe_lc_droplet_param_t fp = {.zeta0 = zeta0, .zeta1 = zeta1};

    double q[3][3] = { {1., 1., 1.,}, {1., 1., 1.}, {1., 1., 1.} };
    double s[3][3] = {0};

    double phi     = -1.0;

    fe_lc_droplet_active_stress(&fp, phi, q, s);

    assert(fabs(s[X][X] - 0.0) < DBL_EPSILON);
    assert(fabs(s[X][Y] - 0.0) < DBL_EPSILON);
    assert(fabs(s[X][Z] - 0.0) < DBL_EPSILON);
    assert(fabs(s[Y][X] - 0.0) < DBL_EPSILON);
    assert(fabs(s[Y][Y] - 0.0) < DBL_EPSILON);
    assert(fabs(s[Y][Z] - 0.0) < DBL_EPSILON);
    assert(fabs(s[Z][X] - 0.0) < DBL_EPSILON);
    assert(fabs(s[Z][Y] - 0.0) < DBL_EPSILON);
    assert(fabs(s[Z][Z] - 0.0) < DBL_EPSILON);

    phi = 0.0;
    fe_lc_droplet_active_stress(&fp, phi, q, s);

    assert(fabs(s[X][X] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[X][Y] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[X][Z] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[Y][X] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[Y][Y] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[Y][Z] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[Z][X] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[Z][Y] - (-0.5)) < DBL_EPSILON);
    assert(fabs(s[Z][Z] - (-0.5)) < DBL_EPSILON);
  }

  return 0;
}
