/*****************************************************************************
 *
 *  test_fe_symmetric.c
 *
 *  Symmetric free energy tests (at last).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authots:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "symmetric.h"

int test_fe_symm_theta_to_h(pe_t * pe);
int test_fe_symm_h_to_costheta(pe_t * pe);

/*****************************************************************************
 *
 *  test_fe_symmetric_suite
 *
 *****************************************************************************/

int test_fe_symmetric_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_fe_symm_theta_to_h(pe);
  test_fe_symm_h_to_costheta(pe);

  pe_info(pe, "PASS     ./unit/test_fe_symmetric\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_symm_theta_to_h
 *
 *****************************************************************************/

int test_fe_symm_theta_to_h(pe_t * pe) {

  int ierr = 0;

  {
    /* 90 degrees is neutral (h = 0) */
    double theta = 90.0;
    double h = 0.0;

    ierr += fe_symm_theta_to_h(theta, &h);
    assert(ierr == 0);
    assert(fabs(h - 0.0) < DBL_EPSILON);
  }

  {
    /* 180 degrees => costheta = 1 => h = sqrt(2sqrt(3) - 3) */
    double theta = 180.0;
    double h = 0.0;

    ierr += fe_symm_theta_to_h(theta, &h);
    assert(ierr == 0);
    assert(fabs(h - sqrt(2.0*sqrt(3.0) - 3.0)) < 2.0*DBL_EPSILON);
  }

  return ierr;
}

/*****************************************************************************
 *
 *  test_fe_symm_h_to_costheta
 *
 *****************************************************************************/

int test_fe_symm_h_to_costheta(pe_t * pe) {

  int ierr = 0;

  {
    /* h = -sqrt(2sqrt(3) - 3) => costheta = -1 */
    double h = -sqrt(2.0*sqrt(3.0) - 3.0);
    double costheta = 0.0;

    ierr += fe_symm_h_to_costheta(h, &costheta);
    assert(ierr == 0);
    /* Won't quite make DBL_EPSILON */
    assert(fabs(costheta - (-1.0)) < 2.0*DBL_EPSILON);
  }

  {
    /* h = 0 is neutral costheta = 0 */
    double h = 0.0;
    double costheta = -1.0;

    ierr += fe_symm_h_to_costheta(h, &costheta);
    assert(ierr == 0);
    assert(fabs(costheta - 0.0) < DBL_EPSILON);
  }

  {
    /* h = 2.0 is invalid */
    double h = 2.0;
    double costheta = 0.0;
    int ibad = 0;

    ibad = fe_symm_h_to_costheta(h, &costheta);
    assert(ibad != 0);
    if (ibad == 0) ierr += 1;
  }

  return ierr;
}
