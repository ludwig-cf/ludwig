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
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "propagation_ode.h"
#include "control.h"
#include "runtime.h"

static int  ndist_ = 2;
static void propagation_ode_d2q9_euler(void);
static void propagation_ode_d2q9_rk2(void);
void propagation_ode_integrator_set(const int);
static int  integrator_type = RK2;
static double dt_ode;

/*****************************************************************************
 *
 *  propagation_ode
 *
 *  Driver routine.
 *
 *****************************************************************************/

void propagation_ode(void) {

  assert(NVEL == 9);
  if (integrator_type == EULER) propagation_ode_d2q9_euler();
  if (integrator_type == RK2) propagation_ode_d2q9_rk2();

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

  int ic, jc, kc, index, p, pdash;
  int nlocal[3]; 
  int nhalo; 
  int xstr, ystr, zstr; 

  double dt;
  extern double * f_;
 
  nhalo = coords_nhalo(); 
  coords_nlocal(nlocal); 

  zstr = ndist_*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  dt = propagation_ode_get_tstep();
 
  kc = 1;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

       p = ndist_*NVEL*index + 0;
       pdash = p+NVEL;
       f_[pdash] = 0.0;
       p = ndist_*NVEL*index + 1;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p - xstr - ystr] - f_[p + xstr + ystr]);
       p = ndist_*NVEL*index + 2;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p - xstr       ] - f_[p + xstr       ]);
       p = ndist_*NVEL*index + 3;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);
       p = ndist_*NVEL*index + 4;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p        - ystr] - f_[p        + ystr]);

       p = ndist_*NVEL*index + 5;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p        + ystr] - f_[p        - ystr]);
       p = ndist_*NVEL*index + 6;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);
       p = ndist_*NVEL*index + 7;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p + xstr       ] - f_[p - xstr       ]);
       p = ndist_*NVEL*index + 8;
       pdash = p+NVEL;
       f_[pdash] = 0.5*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);

    }
  }

  /* Update */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      for (p = 0; p < NVEL; p++) {
	f_[ndist_*NVEL*index + p] += dt*f_[ndist_*NVEL*index + p + NVEL];
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
  double dt;
  extern double * f_;
 
  nhalo = coords_nhalo(); 
  coords_nlocal(nlocal); 

  zstr = ndist_*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  dt = propagation_ode_get_tstep();
 
  kc = 1;

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

   return;
}

void propagation_ode_init(void) { 

  int n;
  char integrator[FILENAME_MAX];
  int nlocal[3]; 

  coords_nlocal(nlocal);
  distribution_ndist_set(2);

  assert(distribution_ndist() == 2); /* Implementation with binary distribution */
  assert(nlocal[Z] <= 3); /* 2-d parallel */ 

  n = RUN_get_string_parameter("propagation_ode_integrator", integrator, FILENAME_MAX);

  if (strcmp(integrator, "euler") == 0) {
        propagation_ode_integrator_set(EULER);
  }

  if (strcmp(integrator, "rk2") == 0) {
        propagation_ode_integrator_set(RK2);
  }

  n = RUN_get_double_parameter("propagation_ode_tstep", &dt_ode);
  assert(n == 1);

  info("\n");
  info("Continuous-time-LB propagation\n");
  info("------------------------------\n");
  info("Time step size:  %g\n", dt_ode);
  info("Integrator type: %s\n", integrator);


  return;
}

void propagation_ode_integrator_set(const int type) {

  assert(type == EULER || type == RK2);

  integrator_type = type;
  return;
}

double propagation_ode_get_tstep() {
  return dt_ode;
}
