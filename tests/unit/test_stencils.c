/*****************************************************************************
 *
 *  test_stencils.c
 *
 *  Generic tests for the various finite difference stencils.
 *
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
#include "stencil.h"

int test_stencil_create(int npoints);

/*****************************************************************************
 *
 *  test_stencils_suite
 *
 *****************************************************************************/

int test_stencils_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_stencil_create(7);
  test_stencil_create(19);
  test_stencil_create(27);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_stencil_create
 *
 *  These are generic tests related to the Laplacian and Gradient
 *  computations. Specific errors should be captured by the
 *  relevant specific test.
 *
 *****************************************************************************/

int test_stencil_create(int npoints) {

  int ifail = 0;
  stencil_t * s = NULL;

  ifail = stencil_create(npoints, &s);
  assert(ifail == 0);
  assert(s);
  assert(s->npoints == npoints);

  /* Laplacian central weight */

  {
    double sum = 0.0;
    for (int p = 1; p < s->npoints; p++) {
      sum += s->wlaplacian[p];
    }
    if (fabs(s->wlaplacian[0] + sum) > DBL_EPSILON) ifail = -1;
    assert(ifail == 0);
  }

  /* Gradient */

  /* d_x */
  {
    double fgrad[3] = {0};
    for (int p = 0; p < s->npoints; p++) {
      double f = 1.0*s->cv[p][X];
      fgrad[X] += s->wgradients[p]*s->cv[p][X]*f;
      fgrad[Y] += s->wgradients[p]*s->cv[p][Y]*f;
      fgrad[Z] += s->wgradients[p]*s->cv[p][Z]*f;
    }
    if (fabs(fgrad[X] - 1.0) > DBL_EPSILON) ifail = -1;
    if (fabs(fgrad[Y] - 0.0) > DBL_EPSILON) ifail = -2;
    if (fabs(fgrad[Z] - 0.0) > DBL_EPSILON) ifail = -4;
    assert(ifail == 0);
  }

  /* d_y */
  {
    double fgrad[3] = {0};
    for (int p = 0; p < s->npoints; p++) {
      double f = 1.0*s->cv[p][Y];
      fgrad[X] += s->wgradients[p]*s->cv[p][X]*f;
      fgrad[Y] += s->wgradients[p]*s->cv[p][Y]*f;
      fgrad[Z] += s->wgradients[p]*s->cv[p][Z]*f;      
    }
    if (fabs(fgrad[X] - 0.0) > DBL_EPSILON) ifail = -1;
    if (fabs(fgrad[Y] - 1.0) > DBL_EPSILON) ifail = -2;
    if (fabs(fgrad[Z] - 0.0) > DBL_EPSILON) ifail = -4;
    assert(ifail == 0);
  }

  /* d_z */
  {
    double fgrad[3] = {0};
    for (int p = 0; p < s->npoints; p++) {
      double f = 1.0*s->cv[p][Z];
      fgrad[X] += s->wgradients[p]*s->cv[p][X]*f;
      fgrad[Y] += s->wgradients[p]*s->cv[p][Y]*f;
      fgrad[Z] += s->wgradients[p]*s->cv[p][Z]*f;      
    }
    if (fabs(fgrad[X] - 0.0) > DBL_EPSILON) ifail = -1;
    if (fabs(fgrad[Y] - 0.0) > DBL_EPSILON) ifail = -2;
    if (fabs(fgrad[Z] - 1.0) > DBL_EPSILON) ifail = -4;
    assert(ifail == 0);
  }

  stencil_free(&s);

  return ifail;
}

