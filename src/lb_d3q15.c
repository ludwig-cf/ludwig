/*****************************************************************************
 *
 *  lb_d3q15.c
 *
 *  D3Q15 model.
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

#include "lb_d3q15.h"

static int lb_d3q15_matrix_ma(lb_model_t * model);

/*****************************************************************************
 *
 *  lb_d3q15_create
 *
 *****************************************************************************/

int lb_d3q15_create(lb_model_t * model) {

  assert(model);

  *model = (lb_model_t) {};

  model->ndim = 3;
  model->nvel = NVEL_D3Q15;
  model->cv   = (int8_t (*)[3]) calloc(NVEL_D3Q15, sizeof(int8_t[3]));
  model->wv   = (double *)      calloc(NVEL_D3Q15, sizeof(double));
  model->na   = (double *)      calloc(NVEL_D3Q15, sizeof(double));
  model->ma   = (double **)     calloc(NVEL_D3Q15, sizeof(double *));
  model->cs2  = (1.0/3.0);

  if (model->cv == NULL) goto err;
  if (model->wv == NULL) goto err;
  if (model->na == NULL) goto err;
  if (model->ma == NULL) goto err;

  {
    LB_CV_D3Q15(cv);
    LB_WEIGHTS_D3Q15(wv);

    for (int p = 0; p < model->nvel; p++) {
      for (int ia = 0; ia < 3; ia++) {
	model->cv[p][ia] = cv[p][ia];
      }
      model->wv[p] = wv[p];
    }
  }

  /* Further allocate matrix elements */
  model->ma[0] = (double *) calloc(NVEL_D3Q15*NVEL_D3Q15, sizeof(double));
  if (model->ma[0] == NULL) goto err;

  for (int p = 1; p < model->nvel; p++) {
    model->ma[p] = model->ma[p-1] + NVEL_D3Q15;
  }

  lb_d3q15_matrix_ma(model);

  /* Normalisers */

  for (int p = 0; p < model->nvel; p++) {
    double sum = 0.0;
    for(int ia = 0; ia < model->nvel; ia++) {
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
 *  lb_d3q15_matrix_ma
 *
 *  Hydrodynamic modes
 *
 *  These correspond to zeroth, first and second order Hermite polynomials
 *  in three dimensions:
 *
 *  [0]  rho       H_i       1
 *  [1]  rho u_x   H_ix      rho c_ix
 *  [2]  rho u_y   H_iy      rho c_iy
 *  [3]  rho u_z   H_iz      rho c_iz
 *  [4]  S_xx      H_ixx     c_ix c_ix - cs^2
 *  [5]  S_xy      H_ixy     c_ix c_iy
 *  [6]  S_xz      H_ixz     c_ix c_iz
 *  [7]  S_yy      H_iyy     c_iy c_iy - cs^2
 *  [8]  S_yz      H_iyz     c_iy c_iz
 *  [9]  S_zz      H_izz     c_iz c_iz - cs^2
 *
 *  Non-hydrodynamic modes
 *
 *  One   3rd order polynomial  H_ixyz
 *  Three 3rd order polynomials H_izzx, H_ixxy, H_iyyz
 *  One   4th order polynomial  H_ixxyy (orthogonalised against H_izz)
 *
 *  [10] H_ixyz
 *  [11] H_izzx
 *  [12] H_ixxy
 *  [13] H_iyyz
 *  [14] H_ixxyy
 *
 *  Note that the third order polynomials H_ixxx etc are not admitted in
 *  this single speed model. The fourth order polynomial may be replaced
 *  by H_iyyzz or H_ixxzz (they are all the same when orthogonalised).
 *
 *  This basis was first introduced for purposes of studying isothermal
 *  fluctuations, see Adhikari et al. 2005. At that time, the
 *  non-hydrodynamic modes were written slightly differerntly,
 *  with two scalar ghost modes chi1 and chi2 and one vector:
 *
 *  [10]  0.5*(-3.0*c^2*c^2 + 9.0*c^2 - 4.0)   "chi1"
 *  [11]  chi1 c_ix
 *  [12]  ch11 c_iy
 *  [13]  chi1 c_iz
 *  [14]  c_ix c_iy c_iz                       "chi2"
 *
 *  where c^2 = cx*cx + cy*cy + cz*cz.
 *  It can be confirmed that these discrete non-hydrodynamic modes are
 *  the same as those above (to within a factor of -1 for chi1).
 *
 *****************************************************************************/

static int lb_d3q15_matrix_ma(lb_model_t * model) {

  assert(model);
  assert(model->ma);
  assert(model->ma[0]);

  for (int p = 0; p < model->nvel; p++) {

    double cs2 = model->cs2;

    double rho = 1.0;
    double cx  = rho*model->cv[p][X];
    double cy  = rho*model->cv[p][Y];
    double cz  = rho*model->cv[p][Z];

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

    model->ma[10][p] = cx*cy*cz;
    model->ma[11][p] = 3.0*(cz*cz - cs2)*cx;
    model->ma[12][p] = 3.0*(cx*cx - cs2)*cy;
    model->ma[13][p] = 3.0*(cy*cy - cs2)*cz;
    model->ma[14][p] = 9.0*(cx*cx - cs2)*(cy*cy - cs2) - 3.0*(cz*cz - cs2);
  }

  return 0;
}
