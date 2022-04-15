/*****************************************************************************
 *
 *  lb_d2q9.c
 *
 *  D2Q9 definition.
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

  *model = (lb_model_t) {0};

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

    for (int p = 0; p < model->nvel; p++) {
      for (int ia = 0; ia < 3; ia++) {
	model->cv[p][ia] = cv[p][ia];
      }
      model->wv[p] = wv[p];
    }
  }

  /* Further allocate matrix elements */
  model->ma[0] = (double *) calloc(NVEL_D2Q9*NVEL_D2Q9, sizeof(double));
  if (model->ma[0] == NULL) goto err;

  for (int p = 1; p < model->nvel; p++) {
    model->ma[p] = model->ma[p-1] + NVEL_D2Q9;
  }

  lb_d2q9_matrix_ma(model);

  /* Normalisers */

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
 *  lb_d2q9_matrix_ma
 *
 *  The rows of the transformation matrix are as follows.
 *
 *  Hydrodynamic modes
 *
 *  These correspond to zeroth, first and second order Hermite polynomials
 *  in two dimensions:
 *
 *  [0]   rho       H_i               1
 *  [1]   rho u_x   H_ix              rho c_ix
 *  [2]   rho u_y   H_iy              rho c_iy
 *  [3]   S_xx      H_ixx             c_ix c_ix - c_s^2
 *  [4]   S_xy      H_ixy             c_ix c_iy
 *  [5]   S_yy      H_iyy             c_iy c_iy - c_s^2
 *
 *  Non-hydrodynamic modes (two third order, and one fourth order polynomial)
 *
 *  [7]   chi c_x  H_iyyx             (c_iy c_iy - c_s^2) c_ix
 *  [8]   chi c_y  H_ixxy             (c_ix c_ix - c_s^2) c_iy
 *  [6]   chi      H_ixxyy            (c_ix c_ix - c_s^2)(c_iy c_iy - c_s^2)
 *
 *  Historically, this basis was introduced for the purposes of consistent
 *  isothermal fluctuations see Adhikari et al. 2005 following the a mode
 *  description e.g., d'Humieres et al.
 *  Originally, the ghost modes were written in slightly different form
 *  (cf. Dellar 2002), one scalar and one vector mode:
 *
 *  [6] chi    = 1/2 (9c^4 - 15c^2 + 2)
 *  [7] jchi_x = chi c_ix
 *  [8] jchi_y = ch1 c_iy
 *
 *  with c^2 = c_ix*c_ix + c_iy*c_iy. These are the same as those above in the
 *  discrete picture (see note on order below).
 *
 *  For a more recent analysis of basis sets and their expansion in
 *  terms of the Hermite tensors, see, e.g., Coreixas et al.
 *  PRE 96 033306 (2017).
 *
 *****************************************************************************/

static int lb_d2q9_matrix_ma(lb_model_t * model) {

  assert(model);
  assert(model->ma);
  assert(model->ma[0]);

  /* It's convenient to assign the elements columnwise */

  for (int p = 0; p < model->nvel; p++) {

    assert(model->cv[p][Z] == 0);

    double cs2 = model->cs2;

    double rho = 1.0;
    double cx  = rho*model->cv[p][X];
    double cy  = rho*model->cv[p][Y];

    model->ma[0][p] = rho;
    model->ma[1][p] = cx;
    model->ma[2][p] = cy;
    model->ma[3][p] = cx*cx - cs2;
    model->ma[4][p] = cx*cy;
    model->ma[5][p] = cy*cy - cs2;

    /* This labelling 7,8,6 is retained to keep results relating to
     * fluctuations numerically the same as those in the original
     * description discussed above. */

    model->ma[7][p] = 6.0*(cy*cy - cs2)*cx;
    model->ma[8][p] = 6.0*(cx*cx - cs2)*cy;
    model->ma[6][p] = 9.0*(cx*cx - cs2)*(cy*cy - cs2);
  }

  return 0;
}
