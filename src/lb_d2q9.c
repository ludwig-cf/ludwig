/*****************************************************************************
 *
 *  lb_d2q9.c
 *
 *  D2Q9 definition.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "lb_d2q9.h"

static int lb_d2q9_matrix_ma(lb_model_t * model);

/*****************************************************************************
 *
 *  lb_d2q9_create
 *
 *****************************************************************************/

int lb_d2q9_create(lb_model_t * model) {

  assert(model);

  *model = (lb_model_t) {};

  model->ndim = 2;
  model->nvel = NVEL_D2Q9;
  model->cv   = (int8_t (*)[3]) calloc(NVEL_D2Q9, sizeof(int8_t[3]));
  model->wv   = (double *)      calloc(NVEL_D2Q9, sizeof(double));
  model->na   = (double *)      calloc(NVEL_D2Q9, sizeof(double));
  model->ma   = (double **)     calloc(NVEL_D2Q9, sizeof(double *));
  model->cs2  = (1.0/3.0);

  if (model->cv == NULL) goto err;
  if (model->wv == NULL) goto err;
  if (model->na == NULL) goto err;
  if (model->ma == NULL) goto err;

  {
    LB_CV_D2Q9(cv);
    LB_WEIGHTS_D2Q9(wv);
    LB_NORMALISERS_D2Q9(na);

    for (int p = 0; p < model->nvel; p++) {
      for (int ia = 0; ia < 3; ia++) {
	model->cv[p][ia] = cv[p][ia];
      }
      model->wv[p] = wv[p];
      model->na[p] = na[p];
    }
  }

  /* Further allocate matrix elements */
  model->ma[0] = (double *) calloc(NVEL_D2Q9*NVEL_D2Q9, sizeof(double));
  if (model->ma[0] == NULL) goto err;

  for (int p = 1; p < model->nvel; p++) {
    model->ma[p] = model->ma[p-1] + NVEL_D2Q9;
  }

  lb_d2q9_matrix_ma(model);

  return 0;

 err:

  lb_model_free(model);

  return -1;
}

/*****************************************************************************
 *
 *  lb_d2q9_matrix_ma
 *
 *  Hydrodynamic modes as usual 1 + NDIM + NDIM*(NDIM+1)/2.
 *
 *  One scalar ghost mode, plus associated vector
 *
 *  chi    = 1/2 (9c^4 - 15c^2 + 2)     eigenvector ma[6]
 *  jchi_x = chi c_x                    eigenvector ma[7]
 *  jchi_y = ch1 c_y                    eigenvector ma[8]
 *
 *****************************************************************************/

static int lb_d2q9_matrix_ma(lb_model_t * model) {

  assert(model);
  assert(model->ma);
  assert(model->ma[0]);

  /* It's convenient to assign the elements columnwise */

  for (int p = 0; p < model->nvel; p++) {

    assert(model->cv[p][Z] == 0);

    double rho = 1.0;
    double cx  = rho*model->cv[p][X];
    double cy  = rho*model->cv[p][Y];
    double sxx = cx*cx - model->cs2;
    double sxy = cx*cy;
    double syy = cy*cy - model->cs2;

    double cs2 = cx*cx + cy*cy;
    double chi = 0.5*(9.0*cs2*cs2 - 15.0*cs2 + 2);

    model->ma[0][p] = rho;
    model->ma[1][p] = cx;
    model->ma[2][p] = cy;
    model->ma[3][p] = sxx;
    model->ma[4][p] = sxy;
    model->ma[5][p] = syy;
    model->ma[6][p] = chi;
    model->ma[7][p] = chi*cx;
    model->ma[8][p] = chi*cy;
  }

  return 0;
}
