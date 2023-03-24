/*****************************************************************************
 *
 *  stencil.h
 *
 *  Lattice Boltzmann velocities/weights re-interpreted as a
 *  finite-difference stencil.
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

#ifndef LUDWIG_STENCIL_H
#define LUDWIG_STENCIL_H

#include <math.h>
#include "cartesian.h"     /* not used explicitly here, but required ... */

typedef struct stencil_s stencil_t;

struct stencil_s {
  int8_t ndim;             /* 2 or 3 */
  int8_t npoints;          /* or velocities */
  int8_t ** cv;            /* cv[npoints][ndim] */
  double * wlaplacian;     /* weights for Laplacian */
  double * wgradients;     /* weights for Gradient */
};

int stencil_create(int npoints, stencil_t ** s);
int stencil_free(stencil_t ** s);
int stencil_finalise(stencil_t * s);

/* Could be part of the lattice Boltzmann definition, but used with stencils */
/* Table for 1/sqrt(cx^2 + cy^2 + cz^2) indexed by c^2 */
/* We set rcs[0] = 0, rcs[1] = 1, etc... */

#define LB_RCS_TABLE(rcs)						\
  const double rcs[4] = {0.0, 1.0, 1.0/sqrt(2.0), 1.0/sqrt(3.0)};
  
#endif
