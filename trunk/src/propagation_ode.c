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


static void propagation_ode_d2q9_rk2(void);
static void propagation_ode_d3q19_rk2(void);
static void propagation_ode_d2q9_rk4(void);
static void propagation_ode_d3q19_rk4(void);
void propagation_ode_integrator_set(const int);
static int  integrator_type;
static double dt_ode;

/*****************************************************************************
 *
 *  propagation_ode
 *
 *  Driver routine.
 *
 *****************************************************************************/

void propagation_ode(void) {

  assert(NVEL == 9 || NVEL == 19);
  switch (NDIM){
    case 2:
      if (integrator_type == RK2) propagation_ode_d2q9_rk2();
      if (integrator_type == RK4) propagation_ode_d2q9_rk4();
      break;
    case 3:
      if (integrator_type == RK2) propagation_ode_d3q19_rk2();
      if (integrator_type == RK4) propagation_ode_d3q19_rk4();
      break;
  }
  return;
}

/*****************************************************************************
 *
 *  propagation_ode_d2q9_rk2
 *
 *  RK2 forward step for the ODE
 * 
 *  d f_i (r; t) / dt = (1/2) [ f_i(r - c_i dt; t) - f_i(r + c_i dt; t) ]
 *                    = g(t_n, f_i(t_n))
 *
 *  k1 = dt g(t_n, f_i(t_n)) 
 *  k2 = dt g(t_n + dt/2, f_i(t_n)+ k1/2)
 *  f_i(r; t_n+1)  = f_i (r; t_n) + k2
 *
 *  where the f_i on the right hand side are post-collision values.
 *
 *****************************************************************************/

void propagation_ode_d2q9_rk2(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash;
   int nlocal[3]; 
   int nhalo; 
   int xstr, ystr, zstr; 
   double dt;
   extern double *f_;

   nhalo = coords_nhalo(); 
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   kc = 1;

   /* f' */

   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pdash = p+NVEL;
       f_[pdash] = 0.0;

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr - ystr] - f_[p + xstr + ystr]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr       ] - f_[p + xstr       ]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p        - ystr] - f_[p        + ystr]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p        + ystr] - f_[p        - ystr]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr       ] - f_[p - xstr       ]);

       p++;
       pdash = p+NVEL;
       f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);
     }
   }

   /* Halo exchange f' (and f) */
   distribution_halo();

   /* Update f */

   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pdash = p+NVEL;
       f_[p] += 0.0;

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash - xstr - ystr] - f_[pdash + xstr + ystr]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash - xstr       ] - f_[pdash + xstr       ]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash - xstr + ystr] - f_[pdash + xstr - ystr]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash        - ystr] - f_[pdash        + ystr]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash        + ystr] - f_[pdash        - ystr]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash + xstr - ystr] - f_[pdash - xstr + ystr]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash + xstr       ] - f_[pdash - xstr       ]);

       p++;
       pdash = p+NVEL;
       f_[p] += 0.5*dt*(f_[pdash + xstr + ystr] - f_[pdash - xstr - ystr]);
     }
   }

   return;
}

/*****************************************************************************
 *
 *  propagation_ode_d2q9_rk4
 *
 *  RK4 forward step for the ODE
 * 
 *  d f_i (r; t) / dt = (1/2) [ f_i(r - c_i dt; t) - f_i(r + c_i dt; t) ]
 *                    = g(t_n, f_i(t_n))
 *
 *  k1 = dt g(t_n, f_i(t_n)) 
 *  k2 = dt g(t_n + dt/2, f_i(t_n) + k1/2)
 *  k3 = dt g(t_n + dt/2, f_i(t_n) + k2/2)
 *  k4 = dt g(t_n + dt/2, f_i(t_n) + k3)
 * 
 *  f_i(r; t_n+1)  = f_i (r; t_n) + k1/6 + k2/3 + k3/3 + k4/6
 *
 *  where the f_i on the right hand side are post-collision values.
 *
 *****************************************************************************/

void propagation_ode_d2q9_rk4(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash, pt;
   int nlocal[3]; 
   int nhalo; 
   int nsites; // including halo regions 
   int xstr, ystr, zstr; 
   int u;
   double dt;
   extern double *f_;
   double onethird=0.33333333333333333333, onesixth=0.166666666666666666666;
   double *ftemp, *ftilde;
   double k1, k2, k3, k4;
 
   nhalo = coords_nhalo(); 
   nsites = coords_nsites();
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();

   ftemp  = (double *) malloc(nsites*NVEL*sizeof (double));
   ftilde = (double *) malloc(nsites*NVEL*sizeof (double));

   /**********************************************/
   /* f' & 1st increment ftilde' = f(t) + 1/6 k1 */

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   kc = 1;

   /* Calculate f' */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       pdash = p+NVEL;
       f_[pdash] = 0.0;
       ftilde[pt] = f_[p] + 0.0;

       p++;
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p - xstr - ystr] - f_[p + xstr + ystr]); 
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++;
       pt++; 
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p - xstr       ] - f_[p + xstr       ]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++; 
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p - xstr + ystr] - f_[p + xstr - ystr]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p        - ystr] - f_[p        + ystr]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p        + ystr] - f_[p        - ystr]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p + xstr - ystr] - f_[p - xstr + ystr]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p + xstr       ] - f_[p - xstr       ]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       k1 = 0.5*dt*(f_[p + xstr + ystr] - f_[p - xstr - ystr]);
       f_[pdash] = f_[p] + 0.5*k1;
       ftilde[pt] = f_[p] + onesixth*k1;

     }
   }

   /* Halo exchange f' (and f) */
   distribution_halo();


  /***************************************************/
  /* f'' & 2nd increment ftilde'' = ftilde' + 1/3 k2 */

   zstr = NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Copy f' into ftemp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {

       index = coords_index(ic, jc, kc);

       pt = NVEL*index + 0;
       pdash = ndist*NVEL*index + 0 + NVEL;
       ftemp[pt] = f_[pdash];

       for (u = 1; u < NVEL; u++){
	 pt++;
	 pdash++;
	 ftemp[pt] = f_[pdash];
       }

     }
   }

   /* Calculate f'' */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       pdash = p+NVEL;
       f_[pdash] = 0.0;
       ftilde[pt] += 0.0;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt - xstr - ystr] - ftemp[pt + xstr + ystr]); 
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt - xstr       ] - ftemp[pt + xstr       ]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt - xstr + ystr] - ftemp[pt + xstr - ystr]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt        - ystr] - ftemp[pt        + ystr]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt        + ystr] - ftemp[pt        - ystr]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt + xstr - ystr] - ftemp[pt - xstr + ystr]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt + xstr       ] - ftemp[pt - xstr       ]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       k2 = 0.5*dt*(ftemp[pt + xstr + ystr] - ftemp[pt - xstr - ystr]);
       f_[pdash] = f_[p] + 0.5*k2;
       ftilde[pt] += onethird*k2;

     }
   }

   /* Halo exchange f'' (and f) */
   distribution_halo();

   /******************************************************/
   /* f''' & 3nd increment ftilde''' = ftilde'' + 1/3 k3 */

   /* Copy f'' into ftemp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {

       index = coords_index(ic, jc, kc);

       pt = NVEL*index + 0;
       pdash = ndist*NVEL*index + 0 + NVEL;
       ftemp[pt] = f_[pdash];

       for (u = 1; u < NVEL; u++){
	 pt++;
	 pdash++;
	 ftemp[pt] = f_[pdash];
       }

     }
   }
 
   /* Calculate f''' */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       pdash = p+NVEL;
       f_[pdash] = 0.0;
       ftilde[pt] += 0.0;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt - xstr - ystr] - ftemp[pt + xstr + ystr]); 
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt - xstr       ] - ftemp[pt + xstr       ]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt - xstr + ystr] - ftemp[pt + xstr - ystr]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt        - ystr] - ftemp[pt        + ystr]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt        + ystr] - ftemp[pt        - ystr]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt + xstr - ystr] - ftemp[pt - xstr + ystr]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt + xstr       ] - ftemp[pt - xstr       ]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       k3 = 0.5*dt*(ftemp[pt + xstr + ystr] - ftemp[pt - xstr - ystr]);
       f_[pdash] = f_[p] + k3;
       ftilde[pt] += onethird*k3;

     }
   }

   /* Halo exchange f''' (and f) */
   distribution_halo();

   /*********************************************/
   /* ftilde'''' = ftilde''' + 1/6 k4 = f(t+dt) */

   /* Copy f''' into ftemp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {

       index = coords_index(ic, jc, kc);

       pt = NVEL*index + 0;
       pdash = ndist*NVEL*index + 0 + NVEL;
       ftemp[pt] = f_[pdash];

       for (u = 1; u < NVEL; u++){
	 pt++;
	 pdash++;
	 ftemp[pt] = f_[pdash];
       }

     }
   }

   /* Calculate last increment */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       ftilde[pt] += 0.0;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt - xstr - ystr] - ftemp[pt + xstr + ystr]); 
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt - xstr       ] - ftemp[pt + xstr       ]);
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt - xstr + ystr] - ftemp[pt + xstr - ystr]);
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt        - ystr] - ftemp[pt        + ystr]);
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt        + ystr] - ftemp[pt        - ystr]);
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt + xstr - ystr] - ftemp[pt - xstr + ystr]);
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt + xstr       ] - ftemp[pt - xstr       ]);
       ftilde[pt] += onesixth*k4;

       p++;
       pt++;
       k4 = 0.5*dt*(ftemp[pt + xstr + ystr] - ftemp[pt - xstr - ystr]);
       ftilde[pt] += onesixth*k4;

     }
   }



   /* Update f(t->t+dt) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       f_[p] = ftilde[pt];

       for (u = 1; u < NVEL; u++){
	 p++;
	 pt++;
	 f_[p] = ftilde[pt];
       }

     }
   }

   free(ftemp);
   free(ftilde);

   return;
}

void propagation_ode_d3q19_rk2(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash;
   int nlocal[3]; 
   int nhalo; 
   int xstr, ystr, zstr; 
   double dt;
   extern double *f_;

   nhalo = coords_nhalo(); 
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* f' */

   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[pdash] = 0.0;

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr - ystr       ] - f_[p + xstr + ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr        - zstr] - f_[p + xstr        + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr              ] - f_[p + xstr              ]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr        + zstr] - f_[p + xstr        - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p - xstr + ystr       ] - f_[p + xstr - ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        - ystr - zstr] - f_[p        + ystr + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        - ystr] - f_[p        + ystr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        - ystr + zstr] - f_[p        + ystr - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p               - zstr] - f_[p               + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p               + zstr] - f_[p               - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        + ystr - zstr] - f_[p        - ystr + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        + ystr       ] - f_[p        - ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p        + ystr + zstr] - f_[p        - ystr - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr - ystr       ] - f_[p - xstr + ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr        - zstr] - f_[p - xstr        + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr              ] - f_[p - xstr              ]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr        + zstr] - f_[p - xstr        - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[pdash] = f_[p] + 0.25*dt*(f_[p + xstr + ystr       ] - f_[p - xstr - ystr       ]);

       }
     }
   }

   /* Halo exchange f' (and f) */
   distribution_halo();

   /* Update f */

   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[p] += 0.0;

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr - ystr       ] - f_[pdash + xstr + ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr        - zstr] - f_[pdash + xstr        + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr              ] - f_[pdash + xstr              ]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr        + zstr] - f_[pdash + xstr        - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash - xstr + ystr       ] - f_[pdash + xstr - ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        - ystr - zstr] - f_[pdash        + ystr + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        - ystr] - f_[pdash        + ystr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        - ystr + zstr] - f_[pdash        + ystr - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash               - zstr] - f_[pdash               + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash               + zstr] - f_[pdash               - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        + ystr - zstr] - f_[pdash        - ystr + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        + ystr       ] - f_[pdash        - ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash        + ystr + zstr] - f_[pdash        - ystr - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr - ystr       ] - f_[pdash - xstr + ystr       ]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr        - zstr] - f_[pdash - xstr        + zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr              ] - f_[pdash - xstr              ]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr        + zstr] - f_[pdash - xstr        - zstr]);

	 p++;
	 pdash = p+NVEL;
	 f_[p] += 0.5*dt*(f_[pdash + xstr + ystr       ] - f_[pdash - xstr - ystr       ]);

       }
     }
   }

   return;
}

/*****************************************************************************
 *
 *  propagation_ode_d3q19_rk4
 *
 *  RK4 forward step for the ODE
 * 
 *  d f_i (r; t) / dt = (1/2) [ f_i(r - c_i dt; t) - f_i(r + c_i dt; t) ]
 *                    = g(t_n, f_i(t_n))
 *
 *  k1 = dt g(t_n, f_i(t_n)) 
 *  k2 = dt g(t_n + dt/2, f_i(t_n) + k1/2)
 *  k3 = dt g(t_n + dt/2, f_i(t_n) + k2/2)
 *  k4 = dt g(t_n + dt/2, f_i(t_n) + k3)
 * 
 *  f_i(r; t_n+1)  = f_i (r; t_n) + k1/6 + k2/3 + k3/3 + k4/6
 *
 *  where the f_i on the right hand side are post-collision values.
 *
 *****************************************************************************/

void propagation_ode_d3q19_rk4(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash, pt;
   int nlocal[3]; 
   int nhalo; 
   int nsites; // including halo regions 
   int xstr, ystr, zstr; 
   int u;
   double dt;
   extern double *f_;
   double onethird=0.33333333333333333333, onesixth=0.166666666666666666666;
   double *ftemp, *ftilde;
   double k1, k2, k3, k4;
 
   nhalo = coords_nhalo(); 
   nsites = coords_nsites();
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();

   ftemp  = (double *) malloc(nsites*NVEL*sizeof (double));
   ftilde = (double *) malloc(nsites*NVEL*sizeof (double));

   /**********************************************/
   /* f' & 1st increment ftilde' = f(t) + 1/6 k1 */

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Calculate f' */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[pdash] = 0.0;
	 ftilde[pt] = f_[p] + 0.0;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p - xstr - ystr       ] - f_[p + xstr + ystr       ]);  
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++; 
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p - xstr        - zstr] - f_[p + xstr        + zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++; 
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p - xstr              ] - f_[p + xstr              ]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p - xstr        + zstr] - f_[p + xstr        - zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p - xstr + ystr       ] - f_[p + xstr - ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p        - ystr - zstr] - f_[p        + ystr + zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p        - ystr] - f_[p        + ystr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p        - ystr + zstr] - f_[p        + ystr - zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p               - zstr] - f_[p               + zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p               + zstr] - f_[p               - zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p        + ystr - zstr] - f_[p        - ystr + zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p        + ystr       ] - f_[p        - ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p        + ystr + zstr] - f_[p        - ystr - zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p + xstr - ystr       ] - f_[p - xstr + ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p + xstr        - zstr] - f_[p - xstr        + zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p + xstr              ] - f_[p - xstr              ]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p + xstr        + zstr] - f_[p - xstr        - zstr]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k1 = 0.5*dt*(f_[p + xstr + ystr       ] - f_[p - xstr - ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k1;
	 ftilde[pt] = f_[p] + onesixth*k1;

       }
     }
   }

   /* Halo exchange f' (and f) */
   distribution_halo();


  /***************************************************/
  /* f'' & 2nd increment ftilde'' = ftilde' + 1/3 k2 */

   zstr = NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Copy f' into ftemp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {
       for (kc = 0; kc <= nlocal[Z]+1; kc++) {

	 index = coords_index(ic, jc, kc);

	 pt = NVEL*index + 0;
	 pdash = ndist*NVEL*index + 0 + NVEL;
	 ftemp[pt] = f_[pdash];

	 for (u = 1; u < NVEL; u++){
	   pt++;
	   pdash++;
	   ftemp[pt] = f_[pdash];
	 }

       }
     }
   }

   /* Calculate f'' */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[pdash] = 0.0;
	 ftilde[pt] += 0.0;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt - xstr - ystr       ] - ftemp[pt + xstr + ystr       ]);  
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++; 
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt - xstr        - zstr] - ftemp[pt + xstr        + zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++; 
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt - xstr              ] - ftemp[pt + xstr              ]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt - xstr        + zstr] - ftemp[pt + xstr        - zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt - xstr + ystr       ] - ftemp[pt + xstr - ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt        - ystr - zstr] - ftemp[pt        + ystr + zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt        - ystr] - ftemp[pt        + ystr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt       - ystr + zstr] - ftemp[pt        + ystr - zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt               - zstr] - ftemp[pt               + zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt               + zstr] - ftemp[pt               - zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt        + ystr - zstr] - ftemp[pt        - ystr + zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt        + ystr       ] - ftemp[pt        - ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt        + ystr + zstr] - ftemp[pt        - ystr - zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt + xstr - ystr       ] - ftemp[pt - xstr + ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt + xstr        - zstr] - ftemp[pt - xstr        + zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt + xstr              ] - ftemp[pt - xstr              ]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt + xstr        + zstr] - ftemp[pt - xstr        - zstr]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k2 = 0.5*dt*(ftemp[pt + xstr + ystr       ] - ftemp[pt - xstr - ystr       ]);
	 f_[pdash] = f_[p] + 0.5*k2;
	 ftilde[pt] += onethird*k2;

       }
     }
   }

   /* Halo exchange f'' (and f) */
   distribution_halo();

   /******************************************************/
   /* f''' & 3nd increment ftilde''' = ftilde'' + 1/3 k3 */

   /* Copy f'' into ftemp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {
       for (kc = 0; kc <= nlocal[Z]+1; kc++) {

	 index = coords_index(ic, jc, kc);

	 pt = NVEL*index + 0;
	 pdash = ndist*NVEL*index + 0 + NVEL;
	 ftemp[pt] = f_[pdash];

	 for (u = 1; u < NVEL; u++){
	   pt++;
	   pdash++;
	   ftemp[pt] = f_[pdash];
	 }

       }
     }
   }
 
   /* Calculate f''' */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 pdash = p+NVEL;
	 f_[pdash] = 0.0;
	 ftilde[pt] += 0.0;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt - xstr - ystr       ] - ftemp[pt + xstr + ystr       ]);  
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++; 
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt - xstr        - zstr] - ftemp[pt + xstr        + zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++; 
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt - xstr              ] - ftemp[pt + xstr              ]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt - xstr        + zstr] - ftemp[pt + xstr        - zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt - xstr + ystr       ] - ftemp[pt + xstr - ystr       ]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt        - ystr - zstr] - ftemp[pt        + ystr + zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt        - ystr] - ftemp[pt        + ystr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt        - ystr + zstr] - ftemp[pt        + ystr - zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt               - zstr] - ftemp[pt               + zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt               + zstr] - ftemp[pt               - zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt        + ystr - zstr] - ftemp[pt        - ystr + zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt        + ystr       ] - ftemp[pt        - ystr       ]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt        + ystr + zstr] - ftemp[pt        - ystr - zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt + xstr - ystr       ] - ftemp[pt - xstr + ystr       ]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt + xstr        - zstr] - ftemp[pt - xstr        + zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt + xstr              ] - ftemp[pt - xstr              ]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt + xstr        + zstr] - ftemp[pt - xstr        - zstr]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 k3 = 0.5*dt*(ftemp[pt + xstr + ystr       ] - ftemp[pt - xstr - ystr       ]);
	 f_[pdash] = f_[p] + k3;
	 ftilde[pt] += onethird*k3;

       }
     }
   }

   /* Halo exchange f''' (and f) */
   distribution_halo();

   /*********************************************/
   /* ftilde'''' = ftilde''' + 1/6 k4 = f(t+dt) */

   /* Copy f''' into ftemp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {
       for (kc = 0; kc <= nlocal[Z]+1; kc++) {

	 index = coords_index(ic, jc, kc);

	 pt = NVEL*index + 0;
	 pdash = ndist*NVEL*index + 0 + NVEL;
	 ftemp[pt] = f_[pdash];

	 for (u = 1; u < NVEL; u++){
	   pt++;
	   pdash++;
	   ftemp[pt] = f_[pdash];
	 }

       }
     }
   }

   /* Calculate last increment */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 ftilde[pt] += 0.0;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt - xstr - ystr       ] - ftemp[pt + xstr + ystr       ]);  
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++; 
	 k4 = 0.5*dt*(ftemp[pt - xstr        - zstr] - ftemp[pt + xstr        + zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++; 
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt - xstr              ] - ftemp[pt + xstr              ]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt - xstr        + zstr] - ftemp[pt + xstr        - zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt - xstr + ystr       ] - ftemp[pt + xstr - ystr       ]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt        - ystr - zstr] - ftemp[pt        + ystr + zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt        - ystr] - ftemp[pt        + ystr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt        - ystr + zstr] - ftemp[pt        + ystr - zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt               - zstr] - ftemp[pt               + zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt               + zstr] - ftemp[pt               - zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt        + ystr - zstr] - ftemp[pt        - ystr + zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt        + ystr       ] - ftemp[pt        - ystr       ]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt        + ystr + zstr] - ftemp[pt        - ystr - zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt + xstr - ystr       ] - ftemp[pt - xstr + ystr       ]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt + xstr        - zstr] - ftemp[pt - xstr        + zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt + xstr              ] - ftemp[pt - xstr              ]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt + xstr        + zstr] - ftemp[pt - xstr        - zstr]);
	 ftilde[pt] += onesixth*k4;

	 p++;
	 pt++;
	 k4 = 0.5*dt*(ftemp[pt + xstr + ystr       ] - ftemp[pt - xstr - ystr       ]);
	 ftilde[pt] += onesixth*k4;

       }
     }
   }



   /* Update f(t->t+dt) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 f_[p] = ftilde[pt];

	 for (u = 1; u < NVEL; u++){
	   p++;
	   pt++;
	   f_[p] = ftilde[pt];
	 }

       }
     }
   }

   free(ftemp);
   free(ftilde);

   return;
}


void propagation_ode_init(void) { 

  int n;
  char integrator[FILENAME_MAX];
  int nlocal[3]; 

  coords_nlocal(nlocal);

  n = RUN_get_string_parameter("propagation_ode_integrator", integrator, FILENAME_MAX);

  if (strcmp(integrator, "rk2") == 0) {
        propagation_ode_integrator_set(RK2);
	// setting ndist_ for initialisation;
	// it is set back to ndist_==1 in model.c
	distribution_ndist_set(2);
  }

  if (strcmp(integrator, "rk4") == 0) {
        propagation_ode_integrator_set(RK4);
	// setting ndist_ for initialisation;
	// it is set back to ndist_==1 in model.c
	distribution_ndist_set(2);
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

  assert(type == RK2 || type == RK4);

  integrator_type = type;
  return;
}

double propagation_ode_get_tstep() {
  return dt_ode;
}
