/*****************************************************************************
 *
 *  test_lb_model.c
 *
 *  Tests that all model should pass.
 *
 *
 *   Edinburgh Soft Matter and Statistical Physics Group and
 *   Edinburgh Parallel Computing Centre
 *
 *   (c) 2021 The University of Edinburgh
 *
 *   Contributing authors:
 *   Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "lb_model.h"


int test_lb_model_create(int nvel);
int test_lb_model_cv(const lb_model_t * model);
int test_lb_model_wv(const lb_model_t * model);
int test_lb_model_ma(const lb_model_t * model);

/*****************************************************************************
 *
 *  test_lb_model_suite
 *
 *****************************************************************************/

int test_lb_model_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_model_create(9);
  test_lb_model_create(15);
  test_lb_model_create(19);

  pe_info(pe, "PASS     ./unit/test_lb_model\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_model_create
 *
 *****************************************************************************/

int test_lb_model_create(int nvel) {

  lb_model_t model = {};

  lb_model_create(nvel, &model);

  test_lb_model_cv(&model);
  test_lb_model_wv(&model);
  test_lb_model_ma(&model);

  lb_model_free(&model);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_model_cv
 *
 *****************************************************************************/

int test_lb_model_cv(const lb_model_t * model) {

  int ifail = 0;

  assert(model);

  /* Zero is first */
  assert(model->cv[0][X] == 0);
  assert(model->cv[0][Y] == 0);
  assert(model->cv[0][Z] == 0);

  /* Check \sum_p cv_pa = 0 */

  {
    int8_t sum[3] = {};

    for (int p = 0; p < model->nvel; p++) {
      sum[X] += model->cv[p][X];
      sum[Y] += model->cv[p][Y];
      sum[Z] += model->cv[p][Z];
    }
    assert(sum[X] == 0);
    assert(sum[Y] == 0);
    assert(sum[Z] == 0);
    ifail += (sum[X] + sum[Y] + sum[Z]);
  }

  /* Check cv[p][] = -cv[NVEL-p][] (p != 0)  */

  for (int p = 1; p < model->nvel; p++) {
    assert(model->cv[p][X] == -model->cv[model->nvel-p][X]);
    assert(model->cv[p][Y] == -model->cv[model->nvel-p][Y]);
    assert(model->cv[p][Z] == -model->cv[model->nvel-p][Z]);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_model_wv
 *
 *****************************************************************************/

int test_lb_model_wv(const lb_model_t * model) {

  assert(model);

  /* Sum of quadrature weights, velocities */

  {
    double sumwv = 0.0;
    double sumcv[3] = {};

    for (int p = 0; p < model->nvel; p++) {
      sumwv    += model->wv[p];
      sumcv[X] += model->cv[p][X];
      sumcv[Y] += model->cv[p][Y];
      sumcv[Z] += model->cv[p][Z];
    }
    /* This can be close... may require 2*epsilon */
    assert(fabs(sumwv    - 1.0) <= DBL_EPSILON);
    assert(fabs(sumcv[X] - 0.0) <= DBL_EPSILON);
    assert(fabs(sumcv[Y] - 0.0) <= DBL_EPSILON);
    assert(fabs(sumcv[Z] - 0.0) <= DBL_EPSILON);
  }

  /* Quadratic terms \sum_p wv[p] c_pi c_pj = cs2 d_ij */

  {
    for (int ia = 0; ia < model->ndim; ia++) {
      for (int ib = 0; ib < model->ndim; ib++) {
	double dij = (ia == ib);
	double sum = 0.0;
	for (int p = 0; p < model->nvel; p++) {
	  sum += model->wv[p]*model->cv[p][ia]*model->cv[p][ib];
	}
	assert(fabs(sum - dij*model->cs2) < DBL_EPSILON);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_model_ma
 *
 *****************************************************************************/

int test_lb_model_ma(const lb_model_t * model) {

  assert(model);

  /* Check normalisers \sum_p na[i]*wv[p]*ma[i][p]*ma[j][p] = dij. */

  for (int i = 0; i < model->nvel; i++) {
    for (int j = 0; j < model->nvel; j++) {
      double dij = (i == j);
      double sum = 0.0;
      for (int p = 0; p < model->nvel; p++) {
	double ** ma = model->ma;
	sum += model->na[i]*model->wv[p]*ma[i][p]*ma[j][p];
      }
      /* Just too tight to make DBL_EPSILON ... */
      assert(fabs(sum - dij) < 2*DBL_EPSILON);
    }
  }

  /* Inverse independent check. */

  {
    int nvel = model->nvel;
    double ** mi = NULL;

    util_matrix_create(nvel, nvel, &mi);

    for (int p = 0; p < nvel; p++) {
      for (int q = 0; q < nvel; q++) {
	mi[p][q] = model->ma[p][q];
      }
    }

    util_matrix_invert(nvel, mi);

    for (int p = 0; p < nvel; p++) {
      for (int q = 0; q < nvel; q++) {
	/* This element of the inverse should be ... */
	double mipq = model->wv[p]*model->na[q]*model->ma[q][p];
	assert(fabs(mipq - mi[p][q]) < DBL_EPSILON);
      }
    }

    util_matrix_free(nvel, &mi);
  }

  return 0;
}
