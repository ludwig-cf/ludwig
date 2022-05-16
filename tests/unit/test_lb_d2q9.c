/*****************************************************************************
 *
 *  test_lb_d2q9.c
 *
 *  For general tests, see test_lb_model.c
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

#include "pe.h"
#include "lb_d2q9.h"

__host__ int test_lb_d2q9_create(void);

/*****************************************************************************
 *
 *  test_lb_d2q9_suite
 *
 *****************************************************************************/

__host__ int test_lb_d2q9_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_d2q9_create();

  pe_info(pe, "PASS     ./unit/test_lb_d2q9\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_d2q9_create
 *
 *****************************************************************************/

__host__ int test_lb_d2q9_create(void) {

  lb_model_t model = {0};

  lb_d2q9_create(&model);

  assert(model.ndim == 2);
  assert(model.nvel == 9);
  assert(model.cv);
  assert(model.wv);
  assert(model.na);
  assert(model.ma);

  for (int p = 0; p < model.nvel; p++) {
    assert(model.cv[p][Z] == 0);
  }
  
  lb_model_free(&model);

  assert(model.nvel == 0);
  assert(model.cv   == NULL);
  assert(model.wv   == NULL);
  assert(model.na   == NULL);
  assert(model.ma   == NULL);

  return 0;
}
