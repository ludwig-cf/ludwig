/*****************************************************************************
 *
 *  stencils.c
 *
 *  Factory method for stencils.
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

#include "stencil_d3q7.h"
#include "stencil_d3q19.h"
#include "stencil_d3q27.h"

/*****************************************************************************
 *
 *  stencil_create
 *
 *****************************************************************************/

int stencil_create(int npoints, stencil_t ** s) {

  int ifail = 0;

  switch (npoints) {
  case NVEL_D3Q7:
    ifail = stencil_d3q7_create(s);
    break;
  case NVEL_D3Q19:
    ifail = stencil_d3q19_create(s);
    break;
  case NVEL_D3Q27:
    ifail = stencil_d3q27_create(s);
    break;
  default:
    ifail = -1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  stencil_free
 *
 *****************************************************************************/

int stencil_free(stencil_t ** s) {

  assert(s);
  assert(*s);

  stencil_finalise(*s);
  free(*s);
  *s = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  stencil_finalise
 *
 *****************************************************************************/

int stencil_finalise(stencil_t * s) {

  assert(s);

  free(s->wgradients);
  free(s->wlaplacian);
  free(s->cv[0]);
  free(s->cv);

  return 0;
}
