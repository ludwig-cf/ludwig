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
  model->cs2  = 1.0/3.0;

  if (model->cv == NULL) goto err;
  if (model->wv == NULL) goto err;

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

  return 0;

 err:

  lb_model_free(model);

  return -1;
}
