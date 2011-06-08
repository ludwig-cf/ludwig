/*****************************************************************************
 *
 *  propagation_ode.c
 *
 *  Continuous time propagation.
 *
 *  An experiemnt based on Boghosian et al PRIVATE communication.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "model.h"

void propagation_ode_d2q9_euler(void);
void propagation_ode_d2q9_rk2(void);

/*****************************************************************************
 *
 *  propagation_ode
 *
 *  Driver routine.
 *
 *****************************************************************************/

void propagation_ode(void) {

  assert(NVEL == 9);
  propagation_ode_d2q9_rk2();

  return;
}

/*****************************************************************************
 *
 *  propagation_ode_d2q9_euler
 *
 *  Euler forward step for the ODE
 *
 *    d f_i (r; t) / dt = (1/2) [ f_i(r - c_i dt; t) - f_i(r + c_i dt; t) ]
 *
 *  where the f_i on the right hand side are post-collision values.
 *
 *****************************************************************************/

void propagation_ode_d2q9_euler(void) {

  int ic, jc, kc, index, p;
  int nlocal[3];
  int nhalo;
  int xstr, ystr, zstr;

  double dt = 1.0;
  double * fdash;

  extern double * f_;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  assert(distribution_ndist() == 1); /* No implementation for binary. */
  assert(nlocal[Z] == 1); /* 2-d */

  kc = 1;

  zstr = 1*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo); 

  fdash = (double *) malloc(NVEL*xstr*(nlocal[X] + 2*nhalo)*sizeof(double));
  if (fdash == NULL) fatal("malloc(fdash) failed\n");

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      p = NVEL*index + 0;
      fdash[p] = 0.0;
      p = NVEL*index + 1;
      fdash[p] = 0.5*(f_[p - xstr - ystr] - f_[p + xstr + ystr]);
      p = NVEL*index + 2;
      fdash[p] = 0.5*(f_[p - xstr       ] - f_[p + xstr       ]);
      p = NVEL*index + 3;
      fdash[p] = 0.5*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);
      p = NVEL*index + 4;
      fdash[p] = 0.5*(f_[p        - ystr] - f_[p        + ystr]);

      p = NVEL*index + 5;
      fdash[p] = 0.5*(f_[p        + ystr] - f_[p        - ystr]);
      p = NVEL*index + 6;
      fdash[p] = 0.5*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);
      p = NVEL*index + 7;
      fdash[p] = 0.5*(f_[p + xstr       ] - f_[p - xstr       ]);
      p = NVEL*index + 8;
      fdash[p] = 0.5*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);
    }
  }

  /* Update */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      for (p = 0; p < NVEL; p++) {
	f_[NVEL*index + p] += dt*fdash[NVEL*index + p];
      }
    }
  }

  free(fdash);

  return;
}

/*****************************************************************************
 *
 *  propagation_ode_d2q9_rk2
 *
 *  As above, but using Runge Kutta 2.
 *
 *****************************************************************************/

void propagation_ode_d2q9_rk2(void) {

  int ic, jc, kc, index, p;
  int nlocal[3]; 
  int nhalo; 
  int xstr, ystr, zstr; 
  int ihalo, ireal; 

  double dt = 1.0; 
  double * fdash; 
 
  extern double * f_; 
 
  nhalo = coords_nhalo(); 
  coords_nlocal(nlocal); 
 
  assert(distribution_ndist() == 1); /* No implementation for binary. */
  assert(nlocal[Z] == 1); /* 2-d */ 
  assert(pe_size() == 1); /* Halo swap for fdash not parallel. */

  kc = 1;

  zstr = 1*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  fdash = (double *) malloc(NVEL*xstr*(nlocal[X] + 2*nhalo)*sizeof(double));
  if (fdash == NULL) fatal("malloc(fdash) failed\n");

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      p = NVEL*index + 0;
      fdash[p] = 0.0;
      p = NVEL*index + 1;
      fdash[p] = f_[p] + 0.25*dt*(f_[p - xstr - ystr] - f_[p + xstr + ystr]);
      p = NVEL*index + 2;
      fdash[p] = f_[p] + 0.25*dt*(f_[p - xstr       ] - f_[p + xstr       ]);
      p = NVEL*index + 3;
      fdash[p] = f_[p] + 0.25*dt*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);
      p = NVEL*index + 4;
      fdash[p] = f_[p] + 0.25*dt*(f_[p        - ystr] - f_[p        + ystr]);

      p = NVEL*index + 5;
      fdash[p] = f_[p] + 0.25*dt*(f_[p        + ystr] - f_[p        - ystr]);
      p = NVEL*index + 6;
      fdash[p] = f_[p] + 0.25*dt*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);
      p = NVEL*index + 7;
      fdash[p] = f_[p] + 0.25*dt*(f_[p + xstr       ] - f_[p - xstr       ]);
      p = NVEL*index + 8;
      fdash[p] = f_[p] + 0.25*dt*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);
    }
  }

  /* Halo swap for fdash required */

  for (jc = 1; jc <= nlocal[Y]; jc++) {

    ihalo = NVEL*coords_index(0, jc, kc);
    ireal = NVEL*coords_index(nlocal[X], jc, kc);
    memcpy(fdash + ihalo, fdash + ireal, NVEL*sizeof(double));

    ihalo = NVEL*coords_index(nlocal[X]+1, jc, kc);
    ireal = NVEL*coords_index(1, jc, kc);
    memcpy(fdash + ihalo, fdash + ireal, NVEL*sizeof(double));
  }
  for (ic = 0; ic <= nlocal[X] + 1; ic++) {

    ihalo = NVEL*coords_index(ic, 0, kc);
    ireal = NVEL*coords_index(ic, nlocal[Y], kc);
    memcpy(fdash + ihalo, fdash + ireal, NVEL*sizeof(double));

    ihalo = NVEL*coords_index(ic, nlocal[Y] + 1, kc);
    ireal = NVEL*coords_index(ic, 1, kc);
    memcpy(fdash + ihalo, fdash + ireal, NVEL*sizeof(double));
  }

  /* Update */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      p = NVEL*index + 0;
      f_[p] += 0.0;
      p = NVEL*index + 1;
      f_[p] += 0.5*dt*(fdash[p - xstr - ystr] - fdash[p + xstr + ystr]);
      p = NVEL*index + 2; 
      f_[p] += 0.5*dt*(fdash[p - xstr       ] - fdash[p + xstr       ]);
      p = NVEL*index + 3;
      f_[p] += 0.5*dt*(fdash[p - xstr + ystr] - fdash[p + xstr - ystr]);
      p = NVEL*index + 4;
      f_[p] += 0.5*dt*(fdash[p        - ystr] - fdash[p        + ystr]);

      p = NVEL*index + 5;
      f_[p] += 0.5*dt*(fdash[p        + ystr] - fdash[p        - ystr]);
      p = NVEL*index + 6;
      f_[p] += 0.5*dt*(fdash[p + xstr - ystr] - fdash[p - xstr + ystr]);
      p = NVEL*index + 7;
      f_[p] += 0.5*dt*(fdash[p + xstr       ] - fdash[p - xstr       ]);
      p = NVEL*index + 8;
      f_[p] += 0.5*dt*(fdash[p + xstr + ystr] - fdash[p - xstr - ystr]);
    }
  }

  free(fdash);

  return;
}

