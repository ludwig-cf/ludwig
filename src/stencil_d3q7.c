/*****************************************************************************
 *
 *  stencil_d3q7.c
 *
 *  A 7-point stencil in three dimensions.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023-2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "stencil_d3q7.h"

/*****************************************************************************
 *
 *  stencil_d3q7_create
 *
 *  The LB weights are {2.0/8.0, 1.0/8.0, ...}
 *  Laplacian. The weights should be {-6.0, +1.0, ...} so a factor of 8.
 *  Gradient.  The weights should be { 0.0, +0.5, ...} so a factor of 4.
 *
 *****************************************************************************/

int stencil_d3q7_create(stencil_t ** stencil) {

  stencil_t * s = NULL;

  s = (stencil_t *) calloc(1, sizeof(stencil_t));
  assert(s);

  s->ndim          = 3;
  s->npoints       = NVEL_D3Q7;
  s->cv            = (int8_t **) calloc(NVEL_D3Q7, sizeof(int8_t *));
  s->wlaplacian    = (double *)  calloc(NVEL_D3Q7, sizeof(double));
  s->wgradients    = (double *)  calloc(NVEL_D3Q7, sizeof(double));

  if (s->cv == NULL) goto err;
  if (s->wlaplacian == NULL) goto err;
  if (s->wgradients == NULL) goto err;

  s->cv[0] = (int8_t *) calloc(s->ndim*NVEL_D3Q7, sizeof(int8_t));
  if (s->cv[0] == NULL) goto err;

  for (int p = 1; p < s->npoints; p++) {
    s->cv[p] = s->cv[p-1] + s->ndim;
  }

  /* Set velocities/weights/stencil */
  {
    double wlap0 = 0.0;
    LB_CV_D3Q7(cv);
    LB_WEIGHTS_D3Q7(wv);
    for (int p = 0; p < s->npoints; p++) {
      for (int ia = 0; ia < s->ndim; ia++) {
	s->cv[p][ia] = cv[p][ia];
      }
      s->wlaplacian[p] = -8.0*wv[p];       /* 2/8, 1/8 -> 2, 1 */
      s->wgradients[p] = +4.0*wv[p];       /*      1/8 -> 1/2  */
      if (p > 0) wlap0 += s->wlaplacian[p];
    }
    /* We must have wlaplacian[0] = -\sum_{p=1} wlaplacian[p]. */
    /* No contribution to the gradient from the central point. */
    s->wlaplacian[0] = -wlap0;
    s->wgradients[0] = 0.0;
  }

  *stencil = s;
  return 0;

 err:

  free(s->wgradients);
  free(s->wlaplacian);
  free(s->cv);
  free(s);

  *stencil = NULL;

  return -1;
}
