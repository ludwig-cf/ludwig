/*****************************************************************************
 *
 *  lb_d3q27.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "lb_d3q27.h"

__host__ int test_lb_d3q27_create(void);

/*****************************************************************************
 *
 *  test_lb_d3q27_suite
 *
 *****************************************************************************/

__host__ int test_lb_d3q27_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_d3q27_create();

  pe_info(pe, "PASS     ./unit/test_lb_d3q27\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_d3q27_create
 *
 *****************************************************************************/

__host__ int test_lb_d3q27_create(void) {

  lb_model_t model = {};

  lb_d3q27_create(&model);

  assert(model.ndim == 3);
  assert(model.nvel == 27);
  assert(model.cv);
  assert(model.wv);
  assert(model.na);
  assert(model.ma);

  assert(fabs(model.cs2 - 1.0/3.0) < DBL_EPSILON);

  lb_model_free(&model);

  assert(model.nvel == 0);
  assert(model.cv   == NULL);
  assert(model.wv   == NULL);
  assert(model.na   == NULL);
  assert(model.ma   == NULL);

  return 0;
}
