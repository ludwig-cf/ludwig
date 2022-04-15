/*****************************************************************************
 *
 *  lb_model.c
 *
 *  Appropriate model details at run time.
 *
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
#include "lb_d3q15.h"
#include "lb_d3q19.h"
#include "lb_d3q27.h"

#include "lb_model.h"

/*****************************************************************************
 *
 *  lb_model_is_available
 *
 *****************************************************************************/

int lb_model_is_available(int nvel) {

  int available = 0;

  available += (nvel == NVEL_D2Q9);
  available += (nvel == NVEL_D3Q15);
  available += (nvel == NVEL_D3Q19);
  available += (nvel == NVEL_D3Q27);

  return available;
}

/*****************************************************************************
 *
 *  lb_model_nhydro
 *
 *****************************************************************************/

int lb_model_nhydro(int ndim) {

  /* The number of hydrodynamic modes is (rho, u_a, S_ab): */

  int nhydro = 1 + ndim + ndim*(ndim + 1)/2;

  return nhydro;
}

/*****************************************************************************
 *
 *  lb_model_create
 *
 *  Really just a factory method as f(nvel)
 *
 *****************************************************************************/

int lb_model_create(int nvel, lb_model_t * model) {

  int ierr = 0;

  assert(model);

  switch (nvel) {
  case (NVEL_D2Q9):
    lb_d2q9_create(model);
    break;
  case (NVEL_D3Q15):
    lb_d3q15_create(model);
    break;
  case (NVEL_D3Q19):
    lb_d3q19_create(model);
    break;
  case (NVEL_D3Q27):
    lb_d3q27_create(model);
    break;
  default:
    /* Error */
    ierr = -1;
  }

  return ierr;
}

/*****************************************************************************
 *
 *  lb_model_free
 *
 *****************************************************************************/

int lb_model_free(lb_model_t * model) {

  assert(model);

  if (model->ma) {
    if (model->ma[0]) free(model->ma[0]);
    free(model->ma);
  }

  if (model->na) free(model->na);
  if (model->cv) free(model->cv);
  if (model->wv) free(model->wv);

  *model = (lb_model_t) {0};

  return 0;
}
