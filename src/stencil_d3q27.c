/*****************************************************************************
 *
 *  stencil_d3q27.c
 *
 *  Finite difference stencils based on lb_d3q27.h weights.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "stencil_d3q27.h"

/*****************************************************************************
 *
 *  stencil_d3q27_create
 *
 *****************************************************************************/

int stencil_d3q27_create(stencil_t ** stencil) {

  stencil_t * s = NULL;

  assert(stencil);

  s = (stencil_t *) calloc(1, sizeof(stencil_t));
  assert(s);

  s->ndim          = 3;
  s->npoints       = NVEL_D3Q27;
  s->cv            = (int8_t **) calloc(NVEL_D3Q27, sizeof(int8_t *));
  s->wlaplacian    = (double *)  calloc(NVEL_D3Q27, sizeof(double));
  s->wgradients    = (double *)  calloc(NVEL_D3Q27, sizeof(double));

  if (s->cv == NULL) goto err;
  if (s->wlaplacian == NULL) goto err;
  if (s->wgradients == NULL) goto err;

  s->cv[0] = (int8_t *) calloc(s->ndim*s->npoints, sizeof(int8_t));
  if (s->cv[0] == NULL) goto err;

  for (int p = 1; p < s->npoints; p++) {
    s->cv[p] = s->cv[p-1] + s->ndim;
  }

  /* Set velocities/weights/stencil */
  {
    double wlap0 = 0.0;
    LB_CV_D3Q27(cv);
    LB_WEIGHTS_D3Q27(wv);
    for (int p = 0; p < s->npoints; p++) {
      for (int ia = 0; ia < s->ndim; ia++) {
        s->cv[p][ia] = cv[p][ia];
      }
      s->wlaplacian[p] = -216.0*wv[p];
      s->wgradients[p] =    3.0*wv[p];
      if (p > 0) wlap0 += s->wlaplacian[p];
    }
    /* Central point */
    s->wlaplacian[0] = -wlap0;
    s->wgradients[0] = 0.0;
  }

  *stencil = s;
  return 0;

 err:

  if (s->wgradients) free(s->wgradients);
  if (s->wlaplacian) free(s->wlaplacian);
  if (s->cv) free(s->cv);

  *stencil = NULL;

  return -1;
}
