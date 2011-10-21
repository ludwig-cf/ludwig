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
#include "propagation_ode.h"

double * fdash_;
static int ndist_ = 2;

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

/* depreciated */
void propagation_ode_d2q9_euler(void) {

  int ic, jc, kc, index, p;
  int nlocal[3];
  int nhalo;
  int xstr, ystr, zstr;

  double dt = 1.0;

  extern double * f_;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  assert(distribution_ndist() == 1); /* No implementation for binary. */
  assert(nlocal[Z] == 1); /* 2-d */

  kc = 1;

  zstr = 1*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo); 

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      p = NVEL*index + 0;
      fdash_[p] = 0.0;
      p = NVEL*index + 1;
      fdash_[p] = 0.5*(f_[p - xstr - ystr] - f_[p + xstr + ystr]);
      p = NVEL*index + 2;
      fdash_[p] = 0.5*(f_[p - xstr       ] - f_[p + xstr       ]);
      p = NVEL*index + 3;
      fdash_[p] = 0.5*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);
      p = NVEL*index + 4;
      fdash_[p] = 0.5*(f_[p        - ystr] - f_[p        + ystr]);

      p = NVEL*index + 5;
      fdash_[p] = 0.5*(f_[p        + ystr] - f_[p        - ystr]);
      p = NVEL*index + 6;
      fdash_[p] = 0.5*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);
      p = NVEL*index + 7;
      fdash_[p] = 0.5*(f_[p + xstr       ] - f_[p - xstr       ]);
      p = NVEL*index + 8;
      fdash_[p] = 0.5*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);
    }
  }

  /* Update */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      for (p = 0; p < NVEL; p++) {
	f_[NVEL*index + p] += dt*fdash_[NVEL*index + p];
      }
    }
  }

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

  int ic, jc, kc, index, p, pdash;
  int nlocal[3]; 
  int nhalo; 
  int xstr, ystr, zstr; 

  double dt = 1.0;  /* We can choose time steps smaller than unity, */
  int ndt=0, idt=0; /* but it must be a rational number. */ 
 
  extern double * f_;
  extern double * fdash_;
 
  nhalo = coords_nhalo(); 
  coords_nlocal(nlocal); 

  zstr = ndist_*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);
 
  kc = 1;

  ndt=(int)(1.0/dt);

  while(idt<ndt){

     for (ic = 1; ic <= nlocal[X]; ic++) {
       for (jc = 1; jc <= nlocal[Y]; jc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist_*NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[pdash] = 0.0;
	 p = ndist_*NVEL*index + 1;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr - ystr] - f_[p + xstr + ystr]);
	 p = ndist_*NVEL*index + 2;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr       ] - f_[p + xstr       ]);
	 p = ndist_*NVEL*index + 3;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);
	 p = ndist_*NVEL*index + 4;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        - ystr] - f_[p        + ystr]);

	 p = ndist_*NVEL*index + 5;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        + ystr] - f_[p        - ystr]);
	 p = ndist_*NVEL*index + 6;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);
	 p = ndist_*NVEL*index + 7;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr       ] - f_[p - xstr       ]);
	 p = ndist_*NVEL*index + 8;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);
       }
     }

     /* Halo exchange */
     distribution_halo();

     /* Update */

     for (ic = 1; ic <= nlocal[X]; ic++) {
       for (jc = 1; jc <= nlocal[Y]; jc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist_*NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[p] += 0.0;
	 p = ndist_*NVEL*index + 1;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr - ystr] - f_[pdash + xstr + ystr]);
	 p = ndist_*NVEL*index + 2; 
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr       ] - f_[pdash + xstr       ]);
	 p = ndist_*NVEL*index + 3;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr + ystr] - f_[pdash + xstr - ystr]);
	 p = ndist_*NVEL*index + 4;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        - ystr] - f_[pdash        + ystr]);

	 p = ndist_*NVEL*index + 5;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        + ystr] - f_[pdash        - ystr]);
	 p = ndist_*NVEL*index + 6;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr - ystr] - f_[pdash - xstr + ystr]);
	 p = ndist_*NVEL*index + 7;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr       ] - f_[pdash - xstr       ]);
	 p = ndist_*NVEL*index + 8;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr + ystr] - f_[pdash - xstr - ystr]);
       }
     }

   idt++;
   }

   return;
}

void propagation_ode_init(void){

  int nlocal[3]; 

  coords_nlocal(nlocal);
  distribution_ndist_set(2);

  assert(distribution_ndist() == 2); /* Implementation with binary distribution */
  assert(nlocal[Z] <= 3); /* 2-d parallel */ 

  return;
}

/* depreciated */
void propagation_ode_halo(void){

  int ic, jc, kc;
  int nlocal[3]; 
  int nhalo; 
  int ihalo, ireal; 

  extern double * fdash_;
 
  nhalo = coords_nhalo(); 
  coords_nlocal(nlocal); 

  kc = 1.0;

  /* Only serial case is implemented here below */
  /* The x-direction (YZ plane) */

  if (cart_size(X) == 1) {
    if (is_periodic(X)) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          ihalo = ndist_*NVEL*coords_index(0, jc, kc);
          ireal = ndist_*NVEL*coords_index(nlocal[X], jc, kc);
          memcpy(fdash_ + ihalo, fdash_ + ireal, ndist_*NVEL*sizeof(double));

          ihalo = ndist_*NVEL*coords_index(nlocal[X]+1, jc, kc);
          ireal = ndist_*NVEL*coords_index(1, jc, kc);
          memcpy(fdash_ + ihalo, fdash_ + ireal, ndist_*NVEL*sizeof(double));
        }
      }
    }
  }
  
  /* The y-direction (XZ plane) */

  if (cart_size(Y) == 1) {
    if (is_periodic(Y)) {
      for (ic = 0; ic <= nlocal[X] + 1; ic++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {

          ihalo = ndist_*NVEL*coords_index(ic, 0, kc);
          ireal = ndist_*NVEL*coords_index(ic, nlocal[Y], kc);
          memcpy(fdash_ + ihalo, fdash_ + ireal, ndist_*NVEL*sizeof(double));

          ihalo = ndist_*NVEL*coords_index(ic, nlocal[Y] + 1, kc);
          ireal = ndist_*NVEL*coords_index(ic, 1, kc);
          memcpy(fdash_ + ihalo, fdash_ + ireal, ndist_*NVEL*sizeof(double));
        }
      }
    }
  }
  return;
}

/* depreciated */
void propagation_ode_finish(void){

   free(fdash_);

  return;
}
