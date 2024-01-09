/*****************************************************************************
 *
 *  test_stencil_d3q27.c
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
#include "stencil_d3q27.h"

int test_stencil_d3q27_create(void);

/*****************************************************************************
 *
 *  test_stencil_d3q27_suite
 *
 *****************************************************************************/

int test_stencil_d3q27_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_stencil_d3q27_create();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_stencil_d3q27_create
 *
 *****************************************************************************/

int test_stencil_d3q27_create(void) {

  int ifail = 0;
  stencil_t * s = NULL;

  ifail = stencil_d3q27_create(&s);
  assert(ifail == 0);
  assert(s);
  assert(s->ndim == 3);
  assert(s->npoints == 27);
  assert(s->cv);
  assert(s->wlaplacian);
  assert(s->wgradients);

  if (s->wlaplacian[0] != 152.0) ifail = -1;
  if (s->wgradients[0] !=   0.0) ifail = -1;
  assert(ifail == 0);

  ifail = stencil_free(&s);
  assert(ifail == 0);
  assert(s == NULL);

  return ifail;
}
