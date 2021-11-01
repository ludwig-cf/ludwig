/*****************************************************************************
 *
 *  lb_d3q19.c
 *
 *  D3Q19 model
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

#include "lb_d3q19.h"

static int lb_d3q19_matrix_ma(lb_model_t * model);

/*****************************************************************************
 *
 *  lb_d3q19_create
 *
 *****************************************************************************/

int lb_d3q19_create(lb_model_t * model) {

  assert(model);

  *model = (lb_model_t) {};

  model->ndim = 3;
  model->nvel = NVEL_D3Q19;
  model->cv   = (int8_t (*)[3]) calloc(NVEL_D3Q19, sizeof(int8_t[3]));
  model->wv   = (double *)      calloc(NVEL_D3Q19, sizeof(double));
  model->na   = (double *)      calloc(NVEL_D3Q19, sizeof(double));
  model->ma   = (double **)     calloc(NVEL_D3Q19, sizeof(double *));
  model->cs2  = (1.0/3.0);

  if (model->cv == NULL) goto err;
  if (model->wv == NULL) goto err;
  if (model->na == NULL) goto err;
  if (model->ma == NULL) goto err;

  {
    LB_CV_D3Q19(cv);
    LB_WEIGHTS_D3Q19(wv);
    LB_NORMALISERS_D3Q19(na);

    for (int p = 0; p < model->nvel; p++) {
      for (int ia = 0; ia < 3; ia++) {
	model->cv[p][ia] = cv[p][ia];
      }
      model->wv[p] = wv[p];
      model->na[p] = na[p];
    }
  }

  /* Further allocate matrix elements */

  model->ma[0] = (double *) calloc(NVEL_D3Q19*NVEL_D3Q19, sizeof(double));
  if (model->ma[0] == NULL) goto err;

  for (int p = 1; p < model->nvel; p++) {
    model->ma[p] = model->ma[p-1] + NVEL_D3Q19;
  }

  lb_d3q19_matrix_ma(model);

  return 0;

 err:

  lb_model_free(model);

  return -1;
}

/*****************************************************************************
 *
 *  lb_d3q19_matrix_ma
 *
 *  Hydrodynamic modes as usual:  1 + ndim + ndim*(ndim+1)/2
 *
 *  There are three scalar ghost modes:
 *
 *   chi1  (2cs^2 - 3)(3c_z^2 - cs^2)           mode[10]
 *   chi2  (2cs^2 - 3)(c_y^2 - c_x^2)           mode[14]
 *   chi3  3cs^4 - 6cs^2 + 1                    mode[18]
 *
 *   and two associated vectors jchi1 and jchi2.
 *
 *****************************************************************************/

static int lb_d3q19_matrix_ma(lb_model_t * model) {

  assert(model);
  assert(model->ma);
  assert(model->ma[0]);

  for (int p = 0; p < model->nvel; p++) {

    double rho  = 1.0;
    double cx   = rho*model->cv[p][X];
    double cy   = rho*model->cv[p][Y];
    double cz   = rho*model->cv[p][Z];
    double sxx  = cx*cx - model->cs2;
    double sxy  = cx*cy;
    double sxz  = cx*cz;
    double syy  = cy*cy - model->cs2;
    double syz  = cy*cz;
    double szz  = cz*cz - model->cs2;

    double cs2  = cx*cx + cy*cy + cz*cz;
    double chi1 = (2.0*cs2 - 3.0)*(3.0*cz*cz - cs2);
    double chi2 = (2.0*cs2 - 3.0)*(cy*cy - cx*cx);
    double chi3 = 3.0*cs2*cs2 - 6.0*cs2 + 1;

    model->ma[ 0][p] = rho;
    model->ma[ 1][p] = cx;
    model->ma[ 2][p] = cy;
    model->ma[ 3][p] = cz;
    model->ma[ 4][p] = sxx;
    model->ma[ 5][p] = sxy;
    model->ma[ 6][p] = sxz;
    model->ma[ 7][p] = syy;
    model->ma[ 8][p] = syz;
    model->ma[ 9][p] = szz;
    model->ma[10][p] = chi1;
    model->ma[11][p] = chi1*cx;
    model->ma[12][p] = chi1*cy;
    model->ma[13][p] = chi1*cz;
    model->ma[14][p] = chi2;
    model->ma[15][p] = chi2*cx;
    model->ma[16][p] = chi2*cy;
    model->ma[17][p] = chi2*cz;
    model->ma[18][p] = chi3;
  }

  return 0;
}
