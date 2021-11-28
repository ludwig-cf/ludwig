/*****************************************************************************
 *
 *  lb_d3q27.c
 *
 *  D3Q27 definition. Not yet complete.
 *
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

#include "lb_d3q27.h"

static int lb_d3q27_matrix_ma(lb_model_t * model);

/*****************************************************************************
 *
 *  lb_d3q27_create
 *
 *****************************************************************************/

int lb_d3q27_create(lb_model_t * model) {

  assert(model);

  *model = (lb_model_t) {};

  model->nvel = NVEL_D3Q27;
  model->cv   = (int8_t (*)[3]) calloc(NVEL_D3Q27, sizeof(int8_t[3]));
  model->wv   = (double *)      calloc(NVEL_D3Q27, sizeof(double));
  model->na   = (double *)      calloc(NVEL_D3Q27, sizeof(double));
  model->ma   = (double **)     calloc(NVEL_D3Q27, sizeof(double *));
  model->cs2  = 1.0/3.0;

  if (model->cv == NULL) goto err;
  if (model->wv == NULL) goto err;
  if (model->na == NULL) goto err;
  if (model->ma == NULL) goto err;

  {
    LB_CV_D3Q27(cv);
    LB_WEIGHTS_D3Q27(wv);

    for (int p = 0; p < model->nvel; p++) {
      for (int ia = 0; ia < 3; ia++) {
	model->cv[p][ia] = cv[p][ia];
      }
      model->wv[p] = wv[p];
    }
  }

  /* Matrix elements */

  model->ma[0] = (double *) calloc(NVEL_D3Q27*NVEL_D3Q27, sizeof(double));
  if (model->ma[0] == NULL) goto err;

  for (int p = 1; p < model->nvel; p++) {
    model->ma[p] = model->ma[p-1] + NVEL_D3Q27;
  }

  lb_d3q27_matrix_ma(model);

  /* Normalisers: Compute weighted inner product ... */

  for (int p = 0; p < model->nvel; p++) {
    double sum = 0.0;
    for (int ia = 0; ia < model->nvel; ia++) {
      sum += model->wv[ia]*model->ma[p][ia]*model->ma[p][ia];
    }
    model->na[p] = 1.0/sum;
  }

  return 0;

 err:

  lb_model_free(model);

  return -1;
}

/*****************************************************************************
 *
 *  lb_d3q27_matrix_ma
 *
 *  Hydrodynamic modes:
 *
 *  Being zeroth, first and second order Hermite polynomials in
 *  three diemnsions:
 *
 *  [0]    rho         H_i          1
 *  [1]    rho u_x     H_ix         rho c_ix
 *  [2]    rho u_y     H_iy         rho c_iy
 *  [3]    rho u_z     H_iy         rho c_iz
 *  [4]    S_xx        H_ixx        c_ix c_ix - c_s^2
 *  [5]    S_xy        H_ixy        c_ix c_iy
 *  [6]    S_xz        H_ixz        c_ix c_iz
 *  [7]    S_yy        H_iyy        c_iy c_iy - c_s^2
 *  [8]    S_yz        H_iyz        c_iy c_iz
 *  [9]    S_zz        H_izz        c_iz c_iz - c_s^2
 *
 *  Non-hydrodynamic modes:
 *
 *  Six 3rd order polynomials H_ixxy, etc
 *  [10]               H_ixxy       (c_ix c_ix - cs2) c_iy
 *  [11]               H_ixxz       (c_ix c_ix - cs2) c_iz
 *  [12]               H_iyyx       (c_iy c_iy - cs2) c_ix
 *  [13]               H_iyyz       ...
 *  [14]               H_izzx       ...
 *  [15]               H_izzy       ...
 *
 *  One 3rd order polynomial
 *  [16]               H_ixyz       c_ix c_iy c_iz
 *
 *  Three 4th order polynomials H_ixxyy, etc
 *  [17]               H_ixxyy      (c_ix c_ix - cs2)*(c_iy c_iy - cs2)
 *  [18]               H_iyyzz      (c_iy c_iy - cs2)*(c_iz c_iz - cs2)
 *  [19]               H_izzxx      ...
 *
 *  Four 4th order polynomials H_ixxyz, etc (ORTHOGONALISED against H_i)
 *  [20]               H_ixxyz      c_ix c_ix c_iy c_iz - cs2 c_iy c_iz - cs4
 *  [21]               H_iyyzx      ...
 *  [22]               H_izzxy      ...
 *
 *  Three 5th order polynomials H_ixxyyz, etc
 *  [23]               H_ixxyyz     H_ixxyy c_iz
 *  [24]               H_iyyzzx     H_iyyzz c_ix
 *  [25]               H_izzxxy     H_izzxx c_iy
 *
 *  Finally, one sixth order polynomial
 *  [26]               H_ixxyyzz    H_ixxyy (c_iz c_iz - cs2)
 *
 *
 *  See, e.g., Coreixas et al. PRE 96 033306 (2017) for a broad
 *  description of construction of models from Hermite tensors.
 *
 *****************************************************************************/

static int lb_d3q27_matrix_ma(lb_model_t * model) {

  assert(model);
  assert(model->ma);
  assert(model->ma[0]);

  for (int p = 0; p < model->nvel; p++) {

    double cs2  = model->cs2;

    double rho  = 1.0;
    double cx   = rho*model->cv[p][X];
    double cy   = rho*model->cv[p][Y];
    double cz   = rho*model->cv[p][Z];

    model->ma[ 0][p] = rho;
    model->ma[ 1][p] = cx;
    model->ma[ 2][p] = cy;
    model->ma[ 3][p] = cz;
    model->ma[ 4][p] = cx*cx - cs2;
    model->ma[ 5][p] = cx*cy;
    model->ma[ 6][p] = cx*cz;
    model->ma[ 7][p] = cy*cy - cs2;
    model->ma[ 8][p] = cy*cz;
    model->ma[ 9][p] = cz*cz - cs2;
    model->ma[10][p] = 3.0*(cx*cx - cs2)*cy;
    model->ma[11][p] = 3.0*(cx*cx - cs2)*cz;
    model->ma[12][p] = 3.0*(cy*cy - cs2)*cz;
    model->ma[13][p] = 3.0*(cy*cy - cs2)*cx;
    model->ma[14][p] = 3.0*(cz*cz - cs2)*cx;
    model->ma[15][p] = 3.0*(cz*cz - cs2)*cy;
    model->ma[16][p] = cx*cy*cz;
    model->ma[17][p] = 9.0*(cx*cx - cs2)*(cy*cy - cs2);
    model->ma[18][p] = 9.0*(cy*cy - cs2)*(cz*cz - cs2);
    model->ma[19][p] = 9.0*(cz*cz - cs2)*(cx*cx - cs2);
    model->ma[20][p] = 9.0*(cx*cx*cy*cz - cs2*cy*cz + cs2*cs2) - 1.0;
    model->ma[21][p] = 9.0*(cy*cy*cz*cx - cs2*cz*cx + cs2*cs2) - 1.0;
    model->ma[22][p] = 9.0*(cz*cz*cx*cy - cs2*cx*cy + cs2*cs2) - 1.0;
    model->ma[23][p] = 9.0*(cx*cx - cs2)*(cy*cy - cs2)*cz;
    model->ma[24][p] = 9.0*(cy*cy - cs2)*(cz*cz - cs2)*cx;
    model->ma[25][p] = 9.0*(cz*cz - cs2)*(cx*cx - cs2)*cy;
    model->ma[26][p] = 27.0*(cx*cx - cs2)*(cy*cy - cs2)*(cz*cz - cs2);
  }

  return 0;
}
