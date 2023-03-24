/****************************************************************************
 *
 *  psi_gradients.c
 *
 *  Currently just routines for the electric field, aka, the gradient
 *  of the potential.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *  (c) 2014-2023 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "coords.h"
#include "psi_gradients.h"

/*****************************************************************************
 *
 *  psi_electric_field
 *
 *  Return the electric field associated with the current potential.
 *  E_a = - \nabla_a \psi
 *
 *****************************************************************************/

int psi_electric_field(psi_t * psi, int index, double e[3]) {

  int ijk[3] = {0};
  cs_t * cs = NULL;
  stencil_t * s = NULL;

  assert(psi);

  cs = psi->cs;
  s  = psi->stencil;
  assert(cs);
  assert(s);

  cs_index_to_ijk(cs, index, ijk);

  e[X] = 0;
  e[Y] = 0;
  e[Z] = 0;

  for (int p = 1; p < s->npoints; p++) {

    int8_t cx = s->cv[p][X];
    int8_t cy = s->cv[p][Y];
    int8_t cz = s->cv[p][Z];

    int index1 = cs_index(cs, ijk[X] + cx, ijk[Y] + cy, ijk[Z] + cz);
    double psi0 = psi->psi->data[addr_rank0(psi->nsites, index1)];

    /* E_a = -\nabla_a psi */
    e[X] -= s->wgradients[p]*cx*psi0;
    e[Y] -= s->wgradients[p]*cy*psi0;
    e[Z] -= s->wgradients[p]*cz*psi0;
  }

  return 0;
}
