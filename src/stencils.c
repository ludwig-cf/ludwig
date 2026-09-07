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

/*****************************************************************************
 *
 *  stencil_opposite
 *
 *****************************************************************************/

int stencil_opp(const stencil_t * s, int p) {

  assert(s);
  assert(p >= 0);
  assert(p < s->npoints);

  switch (s->npoints) {

  case NVEL_D3Q7: {
    static const int opp_d3q7[NVEL_D3Q7] = {
      0, 6, 5, 4, 3, 2, 1
    };
    return opp_d3q7[p];
  }

  case NVEL_D3Q19: {
    static const int opp_d3q19[NVEL_D3Q19] = {
       0,
      18, 17, 16, 15, 14, 13, 12, 11, 10,
       9,  8,  7,  6,  5,  4,  3,  2,  1
    };
    return opp_d3q19[p];
  }

  case NVEL_D3Q27: {
    static const int opp_d3q27[NVEL_D3Q27] = {
       0,
      26, 25, 24, 23, 22, 21, 20, 19, 18,
      17, 16, 15, 14, 13, 12, 11, 10,  9,
       8,  7,  6,  5,  4,  3,  2,  1
    };
    return opp_d3q27[p];
  }

  default:
    assert(0);
  }

  return -1;
}