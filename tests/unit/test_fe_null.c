/*****************************************************************************
 *
 *  test_fe_null.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "fe_null.h"

int test_fe_null_create(pe_t * pe);
int test_fe_null_fed(pe_t * pe);
int test_fe_null_mu(pe_t * pe);
int test_fe_null_str(pe_t * pe);
int test_fe_null_str_v(pe_t * pe);

/*****************************************************************************
 *
 *  test_fe_null_suite
 *
 *****************************************************************************/

int test_fe_null_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_fe_null_create(pe);
  test_fe_null_fed(pe);
  test_fe_null_mu(pe);
  test_fe_null_str(pe);
  test_fe_null_str_v(pe);

  pe_info(pe, "PASS     ./unit/test_fe_null\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_null_create
 *
 *****************************************************************************/

int test_fe_null_create(pe_t * pe) {

  fe_null_t * fe = NULL;

  fe_null_create(pe, &fe);

  assert(fe);
  assert(fe->super.id == FE_NULL);
  assert(fe->super.func);

  /* If an implementation is added, update the tests. */
  assert(fe->super.func->free);
  assert(fe->super.func->target);
  assert(fe->super.func->fed);
  assert(fe->super.func->mu);
  assert(fe->super.func->mu_solv == NULL);
  assert(fe->super.func->stress);
  assert(fe->super.func->str_symm);
  assert(fe->super.func->str_anti == NULL);
  assert(fe->super.func->hvector == NULL);
  assert(fe->super.func->htensor == NULL);
  assert(fe->super.func->htensor_v == NULL);
  assert(fe->super.func->stress_v);
  assert(fe->super.func->str_symm_v);
  assert(fe->super.func->str_anti_v == NULL);

  fe_null_free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_null_fed
 *
 *****************************************************************************/

int test_fe_null_fed(pe_t * pe) {

  fe_null_t * fe = NULL;

  assert(pe);

  fe_null_create(pe, &fe);

  {
    int index = 1;
    double fed = 1.0;

    fe_null_fed(fe, index, &fed);
    assert(fabs(fed - 0.0) < DBL_EPSILON);
  }

  fe_null_free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_null_mu
 *
 *****************************************************************************/

int test_fe_null_mu(pe_t * pe) {

  fe_null_t * fe = NULL;

  assert(pe);

  fe_null_create(pe, &fe);

  {
    int index = 1;
    double mu = 1.0;

    fe_null_mu(fe, index, &mu);
    assert(fabs(mu - 0.0) < DBL_EPSILON);
  }

  fe_null_free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_null_str
 *
 *****************************************************************************/

int test_fe_null_str(pe_t * pe) {

  fe_null_t * fe = NULL;

  assert(pe);

  fe_null_create(pe, &fe);

  {
    int index = 1;
    double s[3][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};

    fe_null_str(fe, index, s);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	assert(fabs(s[ia][ib] - 0.0) < DBL_EPSILON);
      }
    }
  }

  fe_null_free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_null_str_v
 *
 *****************************************************************************/

int test_fe_null_str_v(pe_t * pe) {

  fe_null_t * fe = NULL;

  assert(pe);

  fe_null_create(pe, &fe);

  {
    int index = 1;
    double s[3][3][NSIMDVL] = {0};

    fe_null_str_v(fe, index, s);

    for (int ia = 0; ia < 3; ia++) {
      for (int ib = 0; ib < 3; ib++) {
	for (int iv = 0; iv < NSIMDVL; iv++) {
	  assert(fabs(s[ia][ib][iv] - 0.0) < DBL_EPSILON);
	}
      }
    }
  }

  fe_null_free(fe);

  return 0;
}

