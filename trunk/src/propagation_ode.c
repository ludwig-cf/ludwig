/*****************************************************************************
 *
 *  propagation_ode.c
 *
 *  Continuous time propagation.
 *
 *  Based on Boghosian et al PRIVATE communication.
 *
 *  The collision and the streaming step enter via a 
 *  total 'rate of change'. The PDE is solved with by 
 *  means of the Runge-Kutta method.
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
#include "timer.h"
#include "physics.h"

static void propagation_ode_d2q9_rk2(void);
static void propagation_ode_d3q19_rk2(void);
static void propagation_ode_d2q9_rk4(void);
static void propagation_ode_d3q19_rk4(void);
void propagation_ode_integrator_set(const int);
static int  integrator_type;
static double dt_ode;
static double rtau_shear;
static int propagation_ode_feq(double *, double *, double *);

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
 *****************************************************************************/

void propagation_ode_d2q9_rk2(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash, m;
   int nlocal[3]; 
   int nhalo; 
   int xstr, ystr, zstr; 
   double dt;
   extern double *f_;
   double k1, k2;
   double rho[1], u[3];
   double feq[NVEL];

   nhalo = coords_nhalo(); 
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();
   rtau_shear = 1.0 / (3.0*get_eta_shear());

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Calculate f' (rate of change) */

   kc = 1;
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

        // Determine feq for BGK-collision at RK-step using f_[p] 
	rho[0] = distribution_zeroth_moment(index,0);
	distribution_first_moment(index,0,u);
	propagation_ode_feq(rho, u, feq);

	hydrodynamics_set_velocity(index, u);

	p = ndist*NVEL*index + 0;
	pdash = p+NVEL;	
	m = 0;
	k1 = - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p - xstr - ystr] - f_[p + xstr + ystr]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p - xstr       ] - f_[p + xstr       ]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p - xstr + ystr] - f_[p + xstr - ystr]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p        - ystr] - f_[p        + ystr]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p        + ystr] - f_[p        - ystr]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p + xstr - ystr] - f_[p - xstr + ystr]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p + xstr       ] - f_[p - xstr       ]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

	p++;
	pdash = p+NVEL;
	m++;
	k1 = 0.5*(f_[p + xstr + ystr] - f_[p - xstr - ystr]) - rtau_shear *(f_[p] - feq[m]);
	f_[pdash] = f_[p] + 0.5*dt*k1;

     }
   }

   /* Halo exchange f' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);

  /* Calculate 2nd increment and update f(t->t+dt) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      index = coords_index(ic, jc, kc);

      // Determine feq for BGK-collision at RK-step using f_[pdash] 
      rho[0] = distribution_zeroth_moment(index,1);
      distribution_first_moment(index,1,u);
      propagation_ode_feq(rho, u, feq);

      hydrodynamics_set_velocity(index, u);

      p = ndist*NVEL*index + 0;
      pdash = p+NVEL;
      m = 0;
      k2 = - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash - xstr - ystr] - f_[pdash + xstr + ystr]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash - xstr       ] - f_[pdash + xstr       ]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash - xstr + ystr] - f_[pdash + xstr - ystr]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash        - ystr] - f_[pdash        + ystr]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash        + ystr] - f_[pdash        - ystr]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash + xstr - ystr] - f_[pdash - xstr + ystr]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash + xstr       ] - f_[pdash - xstr       ]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

      p++;
      pdash = p+NVEL;
      m++;
      k2 = 0.5*(f_[pdash + xstr + ystr] - f_[pdash - xstr - ystr]) - rtau_shear *(f_[pdash] - feq[m]);
      f_[p] += dt*k2;

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
 *****************************************************************************/

void propagation_ode_d2q9_rk4(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash, pt, m;
   int nlocal[3]; 
   int nhalo; 
   int nsites; // including halo regions 
   int xstr, ystr, zstr; 
   double dt;
   extern double *f_;
   double onethird=0.33333333333333333333, onesixth=0.166666666666666666666;
   double *ftmp, *fcum;
   double rho[1], u[3];
   double feq[NVEL];
   double k1, k2, k3, k4;
 
   nhalo = coords_nhalo(); 
   nsites = coords_nsites();
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();
   rtau_shear = 1.0 / (3.0*get_eta_shear());

   ftmp  = (double *) malloc(nsites*NVEL*sizeof (double));
   fcum = (double *) malloc(nsites*NVEL*sizeof (double));

   /**********************************************/
   /* f' & 1st increment fcum' = f(t) + 1/6 k1 */

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Calculate f' (rate of change) */

   kc = 1;
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       // Determine feq for BGK-collision at RK-step using f_[p] 
       rho[0] = distribution_zeroth_moment(index,0);
       distribution_first_moment(index,0,u);
       propagation_ode_feq(rho, u, feq);

       hydrodynamics_set_velocity(index, u);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       pdash = p+NVEL;
       m = 0;
       k1 = - rtau_shear *(f_[p] - feq[m]); 
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p - xstr - ystr] - f_[p + xstr + ystr]) - rtau_shear *(f_[p] - feq[m]); 
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++; 
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p - xstr       ] - f_[p + xstr       ]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++; 
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p - xstr + ystr] - f_[p + xstr - ystr]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p        - ystr] - f_[p        + ystr]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p        + ystr] - f_[p        - ystr]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p + xstr - ystr] - f_[p - xstr + ystr]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p + xstr       ] - f_[p - xstr       ]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k1 = 0.5*(f_[p + xstr + ystr] - f_[p - xstr - ystr]) - rtau_shear *(f_[p] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k1;
       fcum[pt] = f_[p] + onesixth*dt*k1;

     }
   }

   /* Halo exchange f' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);


  /***************************************************/
  /* f'' & 2nd increment fcum'' = fcum' + 1/3 k2 */

   zstr = NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Copy f' into ftmp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {

       index = coords_index(ic, jc, kc);

       pt = NVEL*index + 0;
       pdash = ndist*NVEL*index + 0 + NVEL;
       ftmp[pt] = f_[pdash];

       for (m = 1; m < NVEL; m++){
	 pt++;
	 pdash++;
	 ftmp[pt] = f_[pdash];
       }

     }
   }

   /* Calculate f'' (rate of change) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       // Determine feq for BGK-collision at RK-step using f_[pdash] 
       rho[0] = distribution_zeroth_moment(index,1);
       distribution_first_moment(index,1,u);
       propagation_ode_feq(rho, u, feq);

       hydrodynamics_set_velocity(index, u);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       pdash = p+NVEL;
       m = 0;
       k2 = - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt - xstr - ystr] - ftmp[pt + xstr + ystr]) - rtau_shear *(ftmp[pt] - feq[m]); 
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt - xstr       ] - ftmp[pt + xstr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt - xstr + ystr] - ftmp[pt + xstr - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt        - ystr] - ftmp[pt        + ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt        + ystr] - ftmp[pt        - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt + xstr - ystr] - ftmp[pt - xstr + ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt + xstr       ] - ftmp[pt - xstr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k2 = 0.5*(ftmp[pt + xstr + ystr] - ftmp[pt - xstr - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + 0.5*dt*k2;
       fcum[pt] += onethird*dt*k2;

     }
   }

   /* Halo exchange f'' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);

   /******************************************************/
   /* f''' & 3nd increment fcum''' = fcum'' + 1/3 k3 */

   /* Copy f'' into ftmp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {

       index = coords_index(ic, jc, kc);

       pt = NVEL*index + 0;
       pdash = ndist*NVEL*index + 0 + NVEL;
       ftmp[pt] = f_[pdash];

       for (m = 1; m < NVEL; m++){
	 pt++;
	 pdash++;
	 ftmp[pt] = f_[pdash];
       }

     }
   }
 
   /* Calculate f''' (rate of change) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       // Determine feq for BGK-collision at RK-step using f_[pdash] 
       rho[0] = distribution_zeroth_moment(index,1);
       distribution_first_moment(index,1,u);
       propagation_ode_feq(rho, u, feq);

       hydrodynamics_set_velocity(index, u);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       pdash = p+NVEL;
       m = 0;
       k3 = - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt - xstr - ystr] - ftmp[pt + xstr + ystr]) - rtau_shear *(ftmp[pt] - feq[m]); 
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt - xstr       ] - ftmp[pt + xstr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt - xstr + ystr] - ftmp[pt + xstr - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt        - ystr] - ftmp[pt        + ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt        + ystr] - ftmp[pt        - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt + xstr - ystr] - ftmp[pt - xstr + ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt + xstr       ] - ftmp[pt - xstr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

       p++;
       pt++;
       pdash = p+NVEL;
       m++;
       k3 = 0.5*(ftmp[pt + xstr + ystr] - ftmp[pt - xstr - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       f_[pdash] = f_[p] + dt*k3;
       fcum[pt] += onethird*dt*k3;

     }
   }

   /* Halo exchange f''' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);

   /*********************************************/
   /* fcum'''' = fcum''' + 1/6 k4 = f(t+dt) */

   /* Copy f''' into ftmp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {

       index = coords_index(ic, jc, kc);

       pt = NVEL*index + 0;
       pdash = ndist*NVEL*index + 0 + NVEL;
       ftmp[pt] = f_[pdash];

       for (m = 1; m < NVEL; m++){
	 pt++;
	 pdash++;
	 ftmp[pt] = f_[pdash];
       }

     }
   }

   /* Calculate last increment */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       // Determine feq for BGK-collision at RK-step using f_[pdash] 
       rho[0] = distribution_zeroth_moment(index,1);
       distribution_first_moment(index,1,u);
       propagation_ode_feq(rho, u, feq);

       hydrodynamics_set_velocity(index, u);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       m = 0;
       k4 = - rtau_shear *(ftmp[pt] - feq[m]); 
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt - xstr - ystr] - ftmp[pt + xstr + ystr]) - rtau_shear *(ftmp[pt] - feq[m]); 
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt - xstr       ] - ftmp[pt + xstr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt - xstr + ystr] - ftmp[pt + xstr - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt        - ystr] - ftmp[pt        + ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt        + ystr] - ftmp[pt        - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt + xstr - ystr] - ftmp[pt - xstr + ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt + xstr       ] - ftmp[pt - xstr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

       p++;
       pt++;
       m++;
       k4 = 0.5*(ftmp[pt + xstr + ystr] - ftmp[pt - xstr - ystr]) - rtau_shear *(ftmp[pt] - feq[m]);
       fcum[pt] += onesixth*dt*k4;

     }
   }



   /* Update f(t->t+dt) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {

       index = coords_index(ic, jc, kc);

       p = ndist*NVEL*index + 0;
       pt = NVEL*index + 0;
       f_[p] = fcum[pt];

       for (m = 1; m < NVEL; m++){
	 p++;
	 pt++;
	 f_[p] = fcum[pt];
       }

     }
   }

   free(ftmp);
   free(fcum);

   return;
}

/*****************************************************************************
 *
 *  propagation_ode_d3q19_rk2
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
 *****************************************************************************/

void propagation_ode_d3q19_rk2(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash, m;
   int nlocal[3]; 
   int nhalo; 
   int xstr, ystr, zstr; 
   double dt;
   extern double *f_;
   double k1, k2;
   double rho[1], u[3];
   double feq[NVEL];

   nhalo = coords_nhalo(); 
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();
   rtau_shear = 1.0 / (3.0*get_eta_shear());

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Calculate f' (rate of change) */

   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

         // Determine feq for BGK-collision at RK-step using f_[p] 
         rho[0] = distribution_zeroth_moment(index,0);
         distribution_first_moment(index,0,u);
         propagation_ode_feq(rho, u, feq);

         hydrodynamics_set_velocity(index, u);

	 p = ndist*NVEL*index + 0;
	 pdash = p+NVEL;
	 m = 0;
	 k1 = - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p - xstr - ystr       ] - f_[p + xstr + ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p - xstr        - zstr] - f_[p + xstr        + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p - xstr              ] - f_[p + xstr              ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p - xstr        + zstr] - f_[p + xstr        - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p - xstr + ystr       ] - f_[p + xstr - ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p        - ystr - zstr] - f_[p        + ystr + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p        - ystr       ] - f_[p        + ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p        - ystr + zstr] - f_[p        + ystr - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p               - zstr] - f_[p               + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p               + zstr] - f_[p               - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p        + ystr - zstr] - f_[p        - ystr + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p        + ystr       ] - f_[p        - ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p        + ystr + zstr] - f_[p        - ystr - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p + xstr - ystr       ] - f_[p - xstr + ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p + xstr        - zstr] - f_[p - xstr        + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p + xstr              ] - f_[p - xstr              ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p + xstr        + zstr] - f_[p - xstr        - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k1 = 0.5*(f_[p + xstr + ystr       ] - f_[p - xstr - ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
       }
     }
   }

   /* Halo exchange f' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);

   /* Calculate 2nd increment and update f(t->t+dt) */

   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

         // Determine feq for BGK-collision at RK-step using f_[pdash] 
         rho[0] = distribution_zeroth_moment(index,1);
         distribution_first_moment(index,1,u);
         propagation_ode_feq(rho, u, feq);

         hydrodynamics_set_velocity(index, u);

	 p = ndist*NVEL*index + 0;
	 pdash = p+NVEL;
         m = 0;
	 k2 = - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash - xstr - ystr       ] - f_[pdash + xstr + ystr       ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash - xstr        - zstr] - f_[pdash + xstr        + zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash - xstr              ] - f_[pdash + xstr              ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash - xstr        + zstr] - f_[pdash + xstr        - zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash - xstr + ystr       ] - f_[pdash + xstr - ystr       ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash        - ystr - zstr] - f_[pdash        + ystr + zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash        - ystr] - f_[pdash        + ystr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash        - ystr + zstr] - f_[pdash        + ystr - zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash               - zstr] - f_[pdash               + zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash               + zstr] - f_[pdash               - zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash        + ystr - zstr] - f_[pdash        - ystr + zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash        + ystr       ] - f_[pdash        - ystr       ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash        + ystr + zstr] - f_[pdash        - ystr - zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash + xstr - ystr       ] - f_[pdash - xstr + ystr       ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash + xstr        - zstr] - f_[pdash - xstr        + zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash + xstr              ] - f_[pdash - xstr              ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash + xstr        + zstr] - f_[pdash - xstr        - zstr]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

	 p++;
	 pdash = p+NVEL;
         m++;
	 k2 = 0.5*(f_[pdash + xstr + ystr       ] - f_[pdash - xstr - ystr       ]) - rtau_shear *(f_[pdash] - feq[m]);
	 f_[p] += dt*k2;

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
 *****************************************************************************/

void propagation_ode_d3q19_rk4(void) {

   int ndist=2; // local scope 
   int ic, jc, kc, index, p, pdash, pt, m;
   int nlocal[3]; 
   int nhalo; 
   int nsites; // including halo regions 
   int xstr, ystr, zstr; 
   double dt;
   extern double *f_;
   double onethird=0.33333333333333333333, onesixth=0.166666666666666666666;
   double *ftmp, *fcum;
   double k1, k2, k3, k4;
   double rho[1], u[3];
   double feq[NVEL];
 
   nhalo = coords_nhalo(); 
   nsites = coords_nsites();
   coords_nlocal(nlocal); 
   dt = propagation_ode_get_tstep();
   rtau_shear = 1.0 / (3.0*get_eta_shear());

   ftmp  = (double *) malloc(nsites*NVEL*sizeof (double));
   fcum = (double *) malloc(nsites*NVEL*sizeof (double));

   /**********************************************/
   /* f' & 1st increment fcum' = f(t) + 1/6 k1 */

   zstr = ndist*NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Calculate f' (rate of change) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

         // Determine feq for BGK-collision at RK-step using f_[p] 
         rho[0] = distribution_zeroth_moment(index,0);
         distribution_first_moment(index,0,u);
         propagation_ode_feq(rho, u, feq);

         hydrodynamics_set_velocity(index, u);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 pdash = p+NVEL;
	 m = 0;
         k1 = - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*dt*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p - xstr - ystr       ] - f_[p + xstr + ystr       ]) - rtau_shear *(f_[p] - feq[m]);  
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*dt*k1;

	 p++;
	 pt++; 
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p - xstr        - zstr] - f_[p + xstr        + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++; 
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p - xstr              ] - f_[p + xstr              ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p - xstr        + zstr] - f_[p + xstr        - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p - xstr + ystr       ] - f_[p + xstr - ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p        - ystr - zstr] - f_[p        + ystr + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p        - ystr       ] - f_[p        + ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p        - ystr + zstr] - f_[p        + ystr - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p               - zstr] - f_[p               + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p               + zstr] - f_[p               - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p        + ystr - zstr] - f_[p        - ystr + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p        + ystr       ] - f_[p        - ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p        + ystr + zstr] - f_[p        - ystr - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p + xstr - ystr       ] - f_[p - xstr + ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p + xstr        - zstr] - f_[p - xstr        + zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p + xstr              ] - f_[p - xstr              ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p + xstr        + zstr] - f_[p - xstr        - zstr]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k1 = 0.5*(f_[p + xstr + ystr       ] - f_[p - xstr - ystr       ]) - rtau_shear *(f_[p] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k1;
	 fcum[pt] = f_[p] + onesixth*k1;

       }
     }
   }

   /* Halo exchange f' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);


  /***************************************************/
  /* f'' & 2nd increment fcum'' = fcum' + 1/3 k2 */

   zstr = NVEL;
   ystr = zstr*(nlocal[Z] + 2*nhalo);
   xstr = ystr*(nlocal[Y] + 2*nhalo);

   /* Copy f' into ftmp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {
       for (kc = 0; kc <= nlocal[Z]+1; kc++) {

	 index = coords_index(ic, jc, kc);

	 pt = NVEL*index + 0;
	 pdash = ndist*NVEL*index + 0 + NVEL;
	 ftmp[pt] = f_[pdash];

	 for (m = 1; m < NVEL; m++){
	   pt++;
	   pdash++;
	   ftmp[pt] = f_[pdash];
	 }

       }
     }
   }

   /* Calculate f'' (rate of change) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

         // Determine feq for BGK-collision at RK-step using f_[pdash] 
         rho[0] = distribution_zeroth_moment(index,1);
         distribution_first_moment(index,1,u);
         propagation_ode_feq(rho, u, feq);

         hydrodynamics_set_velocity(index, u);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 pdash = p+NVEL;
	 m = 0;
	 k2 = - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;


	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt - xstr - ystr       ] - ftmp[pt + xstr + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]); 
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++; 
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt - xstr        - zstr] - ftmp[pt + xstr        + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++; 
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt - xstr              ] - ftmp[pt + xstr              ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt - xstr        + zstr] - ftmp[pt + xstr        - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt - xstr + ystr       ] - ftmp[pt + xstr - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt        - ystr - zstr] - ftmp[pt        + ystr + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt        - ystr       ] - ftmp[pt        + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt       - ystr + zstr] - ftmp[pt        + ystr - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt               - zstr] - ftmp[pt               + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt               + zstr] - ftmp[pt               - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt        + ystr - zstr] - ftmp[pt        - ystr + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt        + ystr       ] - ftmp[pt        - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt        + ystr + zstr] - ftmp[pt        - ystr - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt + xstr - ystr       ] - ftmp[pt - xstr + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt + xstr        - zstr] - ftmp[pt - xstr        + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt + xstr              ] - ftmp[pt - xstr              ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt + xstr        + zstr] - ftmp[pt - xstr        - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k2 = 0.5*(ftmp[pt + xstr + ystr       ] - ftmp[pt - xstr - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + 0.5*dt*k2;
	 fcum[pt] += onethird*k2;

       }
     }
   }

   /* Halo exchange f'' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);

   /******************************************************/
   /* f''' & 3nd increment fcum''' = fcum'' + 1/3 k3 */

   /* Copy f'' into ftmp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {
       for (kc = 0; kc <= nlocal[Z]+1; kc++) {

	 index = coords_index(ic, jc, kc);

	 pt = NVEL*index + 0;
	 pdash = ndist*NVEL*index + 0 + NVEL;
	 ftmp[pt] = f_[pdash];

	 for (m = 1; m < NVEL; m++){
	   pt++;
	   pdash++;
	   ftmp[pt] = f_[pdash];
	 }

       }
     }
   }
 
   /* Calculate f''' (rate of change) */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

         // Determine feq for BGK-collision at RK-step using f_[pdash] 
         rho[0] = distribution_zeroth_moment(index,1);
         distribution_first_moment(index,1,u);
         propagation_ode_feq(rho, u, feq);

         hydrodynamics_set_velocity(index, u);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 pdash = p+NVEL;
	 m = 0;
         k3 = - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt - xstr - ystr       ] - ftmp[pt + xstr + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);  
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++; 
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt - xstr        - zstr] - ftmp[pt + xstr        + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++; 
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt - xstr              ] - ftmp[pt + xstr              ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt - xstr        + zstr] - ftmp[pt + xstr        - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt - xstr + ystr       ] - ftmp[pt + xstr - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt        - ystr - zstr] - ftmp[pt        + ystr + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt        - ystr       ] - ftmp[pt        + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt        - ystr + zstr] - ftmp[pt        + ystr - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt               - zstr] - ftmp[pt               + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt               + zstr] - ftmp[pt               - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt        + ystr - zstr] - ftmp[pt        - ystr + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt        + ystr       ] - ftmp[pt        - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt        + ystr + zstr] - ftmp[pt        - ystr - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt + xstr - ystr       ] - ftmp[pt - xstr + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt + xstr        - zstr] - ftmp[pt - xstr        + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt + xstr              ] - ftmp[pt - xstr              ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt + xstr        + zstr] - ftmp[pt - xstr        - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

	 p++;
	 pt++;
	 pdash = p+NVEL;
	 m++;
	 k3 = 0.5*(ftmp[pt + xstr + ystr       ] - ftmp[pt - xstr - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 f_[pdash] = f_[p] + dt*k3;
	 fcum[pt] += onethird*dt*k3;

       }
     }
   }

   /* Halo exchange f''' (and f) */
   TIMER_start(TIMER_HALO_LATTICE);
   distribution_halo();
   TIMER_stop(TIMER_HALO_LATTICE);

   /*********************************************/
   /* fcum'''' = fcum''' + 1/6 k4 = f(t+dt) */

   /* Copy f''' into ftmp */
   for (ic = 0; ic <= nlocal[X]+1; ic++) {
     for (jc = 0; jc <= nlocal[Y]+1; jc++) {
       for (kc = 0; kc <= nlocal[Z]+1; kc++) {

	 index = coords_index(ic, jc, kc);

	 pt = NVEL*index + 0;
	 pdash = ndist*NVEL*index + 0 + NVEL;
	 ftmp[pt] = f_[pdash];

	 for (m = 1; m < NVEL; m++){
	   pt++;
	   pdash++;
	   ftmp[pt] = f_[pdash];
	 }

       }
     }
   }

   /* Calculate last increment */
   for (ic = 1; ic <= nlocal[X]; ic++) {
     for (jc = 1; jc <= nlocal[Y]; jc++) {
       for (kc = 1; kc <= nlocal[Z]; kc++) {

	 index = coords_index(ic, jc, kc);

         // Determine feq for BGK-collision at RK-step using f_[pdash] 
         rho[0] = distribution_zeroth_moment(index,1);
         distribution_first_moment(index,1,u);
         propagation_ode_feq(rho, u, feq);

         hydrodynamics_set_velocity(index, u);

	 p = ndist*NVEL*index + 0;
	 pt = NVEL*index + 0;
	 m = 0;
         k4 =  - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt - xstr - ystr       ] - ftmp[pt + xstr + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);  
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++; 
	 m++;
	 k4 = 0.5*(ftmp[pt - xstr        - zstr] - ftmp[pt + xstr        + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++; 
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt - xstr              ] - ftmp[pt + xstr              ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt - xstr        + zstr] - ftmp[pt + xstr        - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt - xstr + ystr       ] - ftmp[pt + xstr - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt        - ystr - zstr] - ftmp[pt        + ystr + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt        - ystr       ] - ftmp[pt        + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt        - ystr + zstr] - ftmp[pt        + ystr - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt               - zstr] - ftmp[pt               + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt               + zstr] - ftmp[pt               - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt        + ystr - zstr] - ftmp[pt        - ystr + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt        + ystr       ] - ftmp[pt        - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt        + ystr + zstr] - ftmp[pt        - ystr - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt + xstr - ystr       ] - ftmp[pt - xstr + ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt + xstr        - zstr] - ftmp[pt - xstr        + zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt + xstr              ] - ftmp[pt - xstr              ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt + xstr        + zstr] - ftmp[pt - xstr        - zstr]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

	 p++;
	 pt++;
	 m++;
	 k4 = 0.5*(ftmp[pt + xstr + ystr       ] - ftmp[pt - xstr - ystr       ]) - rtau_shear *(ftmp[pt] - feq[m]);
	 fcum[pt] += onesixth*dt*k4;

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
	 f_[p] = fcum[pt];

	 for (m = 1; m < NVEL; m++){
	   p++;
	   pt++;
	   f_[p] = fcum[pt];
	 }

       }
     }
   }

   free(ftmp);
   free(fcum);

   return;
}

/*****************************************************************************
 *
 *  propagation_ode_init
 *
 *  Runtime initialisation of RK-integrator and continuous time step size.
 *
 ****************************************************************************/
void propagation_ode_init(void) { 

  int n;
  char integrator[FILENAME_MAX];

  n = RUN_get_string_parameter("propagation_ode_integrator", integrator, FILENAME_MAX);

  // setting ndist_ for initialisation;
  distribution_ndist_set(2);

  propagation_ode_integrator_set(RK2);

  if (strcmp(integrator, "rk4") == 0) {
        propagation_ode_integrator_set(RK4);
  }

  n = RUN_get_double_parameter("propagation_ode_tstep", &dt_ode);
  assert(n == 1);

  info("\n");
  info("Continuous-time-LB propagation\n");
  info("------------------------------\n");
  info("Time step size:  %g\n", dt_ode);
  if (integrator_type == RK2) info("Integrator type: rk2\n");
  if (integrator_type == RK4) info("Integrator type: rk4\n");

  return;
}
/*****************************************************************************
 *
 *  propagation_ode_integrator_set
 *
 *  Sets the file pointer.  
 *
 ****************************************************************************/
void propagation_ode_integrator_set(const int type) {

  integrator_type = type;
  return;
}

/*****************************************************************************
 *
 *  propagation_ode_get_tstep
 *
 *  Getter routine for continuous time step.
 *
 ****************************************************************************/
double propagation_ode_get_tstep() {
  return dt_ode;
}

/*****************************************************************************
 *
 *  propagation_ode_feq
 *
 *  Equilibrium distributions for collision stage (BGK).
 *
 ****************************************************************************/
int propagation_ode_feq(double rho[1], double u[3], double feq[NVEL]) {

  int       ia, ib, p;
  double    udotc, sdotq;

  for (p = 0; p < NVEL; p++) {
    udotc = 0.0;
    sdotq = 0.0;
    for (ia = 0; ia < 3; ia++) {
      udotc += u[ia]*cv[p][ia];
      for (ib = 0; ib < 3; ib++) {
	sdotq += q_[p][ia][ib]*u[ia]*u[ib];
      }
    }
    feq[p] = rho[0]*wv[p]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq);
  }

  return 0;
}
