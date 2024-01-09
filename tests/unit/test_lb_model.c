/*****************************************************************************
 *
 *  test_lb_model.c
 *
 *  Tests that all models should pass.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "lb_model.h"

int test_lb_model_nhydro(void);
int test_lb_model_create(int nvel);
int test_lb_model_cv(const lb_model_t * model);
int test_lb_model_wv(const lb_model_t * model);
int test_lb_model_na(const lb_model_t * model);
int test_lb_model_ma(const lb_model_t * model);
int test_lb_model_hydrodynamic_modes(const lb_model_t * model);

/*****************************************************************************
 *
 *  test_lb_model_suite
 *
 *****************************************************************************/

int test_lb_model_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_model_nhydro();
  test_lb_model_create(9);
  test_lb_model_create(15);
  test_lb_model_create(19);
  test_lb_model_create(27);

  pe_info(pe, "PASS     ./unit/test_lb_model\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_model_nhydro
 *
 *****************************************************************************/

int test_lb_model_nhydro(void) {

  assert(lb_model_nhydro(2) == 6);
  assert(lb_model_nhydro(3) == 10);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_model_create
 *
 *****************************************************************************/

int test_lb_model_create(int nvel) {

  lb_model_t model = {0};

  lb_model_create(nvel, &model);

  test_lb_model_cv(&model);
  test_lb_model_wv(&model);
  test_lb_model_na(&model);
  test_lb_model_ma(&model);

  test_lb_model_hydrodynamic_modes(&model);

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
    int8_t sum[3] = {0};

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

  /* Check cv[p][] = -cv[nvel-p][] (p != 0)  */

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

  int ierr = 0;
  KRONECKER_DELTA_CHAR(d_);

  assert(model);

  /* Sum of quadrature weights, velocities */

  {
    double sumwv = 0.0;
    double sumcv[3] = {0};

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

    if (fabs(sumwv - 1.0) > DBL_EPSILON) ierr += 1;
  }

  /* Quadratic terms \sum_p wv[p] c_pi c_pj = cs2 d_ij */

  {
    for (int ia = 0; ia < model->ndim; ia++) {
      for (int ib = 0; ib < model->ndim; ib++) {
	double sum = 0.0;
	for (int p = 0; p < model->nvel; p++) {
	  sum += model->wv[p]*model->cv[p][ia]*model->cv[p][ib];
	}
	assert(fabs(sum - d_[ia][ib]*model->cs2) < DBL_EPSILON);
	if (fabs(sum - d_[ia][ib]*model->cs2) >= DBL_EPSILON) ierr += 1;
      }
    }
  }

  /* Third moment \sum_i w_i cia c_ib c_ig = 0 */

  {
    for (int ia = 0; ia < model->ndim; ia++) {
      for (int ib = 0; ib < model->ndim; ib++) {
	for (int ig = 0; ig < model->ndim; ig++) {
	  double sum = 0.0;
	  for (int p = 0; p < model->nvel; p++) {
	    int8_t (*cv)[3] = model->cv;
	    sum += model->wv[p]*cv[p][ia]*cv[p][ib]*cv[p][ig];
	  }
	  assert(fabs(sum - 0.0) < DBL_EPSILON);
	  if (fabs(sum) >= DBL_EPSILON) ierr += 1;
	}
      }
    }
  }

  /* Fourth order moment:
   * \sum_i w_i c_ia c_ib c_ig c_ih = c_s^4 (d_ab d_gh + d_ag d_bh + d_ah d_bg)
   * with d_ab Kronecker delta. */

  {

    for (int ia = 0; ia < model->ndim; ia++) {
      for (int ib = 0; ib < model->ndim; ib++) {
	for (int ig = 0; ig < model->ndim; ig++) {
	  for (int ih = 0; ih < model->ndim; ih++) {
	    double delta = d_[ia][ib]*d_[ig][ih] + d_[ia][ig]*d_[ib][ih]
                         + d_[ia][ih]*d_[ib][ig];
	    double expect = model->cs2*model->cs2*delta;

	    double sum = 0.0;
	    for (int p = 0; p < model->nvel; p++) {
	      int8_t (*cv)[3] = model->cv;
	      sum += model->wv[p]*cv[p][ia]*cv[p][ib]*cv[p][ig]*cv[p][ih];
	    }

	    assert(fabs(sum - expect) < DBL_EPSILON);
	    ierr += (fabs(sum -expect) > DBL_EPSILON);
	  }
	}
      }
    }
  }

  return ierr;
}

/*****************************************************************************
 *
 *  test_lb_model_na
 *
 *****************************************************************************/

int test_lb_model_na(const lb_model_t * model) {

  int ifail = 0;

  assert(model);

  /* The normalisers are related to the weighted inner product of the modes */

  for (int m = 0; m < model->nvel; m++) {
    double wip = 0.0;
    for (int p = 0; p < model->nvel; p++) {
      wip += model->wv[p]*model->ma[m][p]*model->ma[m][p];
    }
    assert(fabs(1.0/wip - model->na[m]) < DBL_EPSILON);
    ifail += (fabs(1.0/wip - model->na[m]) > DBL_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_model_ma
 *
 *****************************************************************************/

int test_lb_model_ma(const lb_model_t * model) {

  int ierr = 0;

  assert(model);

  /* Check condition \sum_p na[i]*wv[p]*ma[i][p]*ma[j][p] = dij. */
  /* The modes must all be orthonormal wrt one another. */

  for (int i = 0; i < model->nvel; i++) {
    for (int j = 0; j < model->nvel; j++) {
      double dij = (i == j);
      double sum = 0.0;
      for (int p = 0; p < model->nvel; p++) {
	double ** ma = model->ma;
	sum += model->na[i]*model->wv[p]*ma[i][p]*ma[j][p];
      }

      /* Too tight to make DBL_EPSILON ... */
      /* although d2q9, d3q15, d3q19 will make 2*DBL_EPISLON */

      assert(fabs(sum - dij) < 5.0*DBL_EPSILON);
      ierr += (fabs(sum - dij) > 5.0*DBL_EPSILON);
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
	ierr += (fabs(mipq - mi[p][q]) > DBL_EPSILON);
      }
    }

    util_matrix_free(nvel, &mi);
  }

  return ierr;
}

/*****************************************************************************
 *
 *  test_lb_model_hydrodynamic_modes
 *
 *****************************************************************************/

int test_lb_model_hydrodynamic_modes(const lb_model_t * model) {

  int ifail = 0;

  assert(model);

  /* The hydrodynamic modes are always the same independent of model
   * and must be in the right order */

  /* One density */

  for (int p = 0; p < model->nvel; p++) {
    assert(fabs(model->ma[0][p] - 1.0) < DBL_EPSILON);
  }

  /* ndim velocities */

  for (int p = 0; p < model->nvel; p++) {
    for (int q = 0; q < model->ndim; q++) {
      assert(fabs(model->ma[1+q][p] - model->cv[p][q]) < DBL_EPSILON);
    }
  }

  /* Upper triangle of stresses */
  {
    double cs2 = model->cs2;

    for (int p = 0; p < model->nvel; p++) {
      int k = 1 + model->ndim;
      for (int i = 0; i < model->ndim; i++) {
	for (int j = i; j < model->ndim; j++) {
	  double dij = (i == j);
	  double sij = model->cv[p][i]*model->cv[p][j] - cs2*dij;
	  assert(fabs(model->ma[k][p] - sij) < DBL_EPSILON);
	  if (fabs(model->ma[k][p] - sij) > DBL_EPSILON) ifail += 1;
	  k += 1;
	}
      }
    }
  }

  return ifail;
}
