/*****************************************************************************
 *
 *  brownian.c
 *
 *  Brownian Dynamics
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "ran.h"
#include "coords.h"
#include "colloids.h"
#include "ccomms.h"
#include "physics.h"
#include "utilities.h"

void brownian_step_ermak_buckholz(void);
void brownian_step_no_inertia(void);
void brownian_step_no_inertia_test(void);
void brownian_set_random(void);

static double dt_ = 1.0;  /* Time step; fixed here to be 1 LB step */

/*****************************************************************************
 *
 *  do_brownian_dynamics
 *
 *  Perform a number of Brownian Dynamics steps.
 *
 *****************************************************************************/

void do_brownian_dynamics() {

  const int ntime = 16;

  int     ic, jc, kc;
  Colloid * p_colloid;
  int i, n, nt, nmax = 100000;
  double u[3][ntime];
  double r[3][ntime];
  double s[3][ntime];
  double correlator_uu[ntime];
  double correlator_ur[ntime];
  double correlator_rr[ntime];
  double correlator_ss[ntime];
  double mass, scaleu, scalex, beta;
  double dr[3];

  brownian_step_no_inertia_test();
  if (1) return;

  mass  = (4.0/3.0)*PI*pow(2.3, 3);
  beta  = 6.0*PI*get_eta_shear()*2.3/mass;
  scaleu = sqrt(mass/(3.0*get_kT()));
  scalex = beta*scaleu;

  dt_ = 0.1/beta;
  dt_ = 1.0;

  info("beta = %f\n", beta);
  info("scale u is %f\n", scaleu);
  info("scale x is %f\n", scalex);

  for (nt = 0; nt < ntime; nt++) {
    for (i = 0; i < 3; i++) {
      u[i][nt] = 0.0;
      r[i][nt] = 0.0;
      s[i][nt] = 0.0;
    }
    correlator_uu[nt] = 0.0;
    correlator_ur[nt] = 0.0;
    correlator_rr[nt] = 0.0;
    correlator_ss[nt] = 0.0;
  }

  for (n = 1; n <= nmax; n++) {

  /* Set random numbers for each particle */
    brownian_set_random();
    CCOM_halo_particles();
    /* brownian_step_ermak_buckholz();*/
    brownian_step_no_inertia();

    /* diagnostics */

    for (nt = ntime-1; nt >= 1; nt--) {
      for (i = 0; i < 3; i++) {
	u[i][nt] = u[i][nt-1];
	r[i][nt] = r[i][nt-1];
	s[i][nt] = s[i][nt-1];
      }
    }

    for (ic = 1; ic <= Ncell(X); ic++) {
      for (jc = 1; jc <= Ncell(Y); jc++) {
	for (kc = 1; kc <= Ncell(Z); kc++) {

	  p_colloid = CELL_get_head_of_list(ic, jc, kc);

	  while (p_colloid) {
	    r[X][0] = p_colloid->r.x;
	    r[Y][0] = p_colloid->r.y;
	    r[Z][0] = p_colloid->r.z;
	    u[X][0] = p_colloid->v.x;
	    u[Y][0] = p_colloid->v.y;
	    u[Z][0] = p_colloid->v.z;
	    s[X][0] = p_colloid->s[X];
	    s[Y][0] = p_colloid->s[Y];
	    s[Z][0] = p_colloid->s[Z];

	    p_colloid = p_colloid->next;
	  }
	}
      }
    }

    if ( n > ntime) {
      for (nt = 0; nt < ntime; nt++) {
	for (i = 0; i < 3; i++) {
	  correlator_uu[nt] += scaleu*scaleu*u[i][nt]*u[i][0];
	  /* -ve sign here because r[0] is current step and nt previous */
	  dr[i] = -(r[i][nt] - r[i][0]);
	  if (dr[i] > L(i)/2.0) dr[i] -= L(i);
	  if (dr[i] <-L(i)/2.0) dr[i] += L(i);
	  correlator_ur[nt] += scaleu*u[i][nt]*scalex*dr[i];
	  correlator_rr[nt] += dr[i]*dr[i];
	  correlator_ss[nt] += (s[i][nt] - s[i][0])*(s[i][nt] - s[i][0]);
	}
      }
    }

    cell_update();
  }

  info("\nResults Ermak and Buckholz \n");
  for (n = 0; n < ntime; n++) {
    double tau = n*beta*dt_;
    info("%6.3f %6.3f %6.3f ", tau, exp(-tau), correlator_uu[n]/nmax);
    info("%6.3f %6.3f ", 1.0 -  exp(-tau), sqrt(3.0)*correlator_ur[n]/nmax);
    info("%6.3f %6.3f\n", 2.0*(tau - 1.0 + exp(-tau)),
	 scalex*scalex*correlator_rr[n]/nmax);
  }

  /* No inertia test */
  for (n = 0; n < ntime; n++) {
    double tau = n*dt_;
    double difft = get_kT() / (6.0*PI*get_eta_shear()*2.3);
    double diffr = get_kT() / (8.0*PI*get_eta_shear()*2.3*2.3*2.3);
    info("%8.3f %8.3f %8.3f ",  tau, 2.0*tau, correlator_rr[n]/(3.0*difft*nmax));
    info("%8.3f %8.3f\n", 4.0*tau, correlator_ss[n]/(diffr*nmax));
  }

  return;
}

/*****************************************************************************
 *
 *  brownian_step_no_inertia
 *
 *  Brownian dynamics with no velocity; only position is updated.
 *
 *  Rotational and translation parts; see, for example, Meriget etal
 *  J. Chem Phys., 121 6078 (2004).
 *  
 *  r(t + dt) = r(t) +  rgamma_t dt Force + r_random
 *  s(t + dt) = s(t) + (rgamma_r dt Torque + t_random) x s(t)
 *
 *  The translational and rotational friction coefficients are
 *  gamma_t  = 6 pi eta a
 *  gamma_r  = 8.0 pi eta a^3
 *
 *  The variances of the random translational and rotational
 *  contributions are related to the diffusion constants (kT/gamma)
 *  <r_i . r_j> = 2 dt (kT / gamma_t) delta_ij
 *  <t_i . t_j> = 2 dt (kT / gamma_r) delta_ij
 * 
 *  This requires six random variates per time step.
 *
 *****************************************************************************/

void brownian_step_no_inertia() {

  int       ic, jc, kc;
  Colloid * p_colloid;
  double    ran[3];
  double    eta, kT;
  double    sigma, rgamma;

  kT = get_kT();
  eta = get_eta_shear();

  /* Update each particle */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  /* Translational motion */

	  rgamma = 1.0 / (6.0*PI*eta*p_colloid->ah);
	  sigma = sqrt(2.0*dt_*kT*rgamma);

	  ran[X] = sigma*p_colloid->random[0];
	  ran[Y] = sigma*p_colloid->random[1];
	  ran[Z] = sigma*p_colloid->random[2];

	  p_colloid->r.x += dt_*rgamma*p_colloid->force.x + ran[X];
	  p_colloid->r.y += dt_*rgamma*p_colloid->force.y + ran[Y];
	  p_colloid->r.z += dt_*rgamma*p_colloid->force.z + ran[Z];

	  /* Rotational motion */

	  rgamma = 1.0 / (8.0*PI*eta*pow(p_colloid->ah, 3));
	  sigma = sqrt(2.0*dt_*kT*rgamma);

	  ran[X] = dt_*rgamma*p_colloid->torque.x + sigma*p_colloid->random[3];
	  ran[Y] = dt_*rgamma*p_colloid->torque.y + sigma*p_colloid->random[4];
	  ran[Z] = dt_*rgamma*p_colloid->torque.z + sigma*p_colloid->random[5];

	  rotate_vector(p_colloid->s, ran);

	  /* Next colloid */

	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  brownian_step_ermak_buckholz
 *
 *  This is the method of Ermak and Buckholz J. Comp. Phys 35, 169 (1980).
 *  It is valid when the time step is of the same order as the decay time
 *  for the velocity autocorrelation function.
 *
 *  See also Allen and Tildesley.
 *
 *  Only the translational part is present at the moment. This requires
 *  6 random numbers per particle per timestep. The updates for the
 *  position and momentum must be correlated (see e.g., Allen & Tildesley
 *  Appendix G.3)   
 *
 *****************************************************************************/

void brownian_step_ermak_buckholz() {

  int     ic, jc, kc;
  Colloid * p_colloid;

  double ranx, rany, ranz;
  double cx, cy, cz;

  double c0, c1, c2;
  double xi, xidt;
  double sigma_r, sigma_v;
  double c12, c21;
  double rmass, kT;

  kT = get_kT();

  /* Update each particle */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  rmass = 1.0/((4.0/3.0)*PI*pow(p_colloid->ah, 3.0));

	  /* Friction coefficient is xi, and related quantities */

	  xi = 6.0*PI*get_eta_shear()*p_colloid->ah*rmass;
	  xidt = xi*dt_;

	  c0 = exp(-xidt);
	  c1 = (1.0 - c0) / xi;
	  c2 = (dt_ - c1) / xi;

	  /* Velocity update */

	  sigma_v = sqrt(rmass*kT*(1.0 - c0*c0));

	  ranx = p_colloid->random[0];
	  rany = p_colloid->random[1];
	  ranz = p_colloid->random[2];

	  cx = c0*p_colloid->v.x + rmass*c1*p_colloid->force.x + sigma_v*ranx;
	  cy = c0*p_colloid->v.y + rmass*c1*p_colloid->force.y + sigma_v*rany;
	  cz = c0*p_colloid->v.z + rmass*c1*p_colloid->force.z + sigma_v*ranz;

	  p_colloid->v.x = cx;
	  p_colloid->v.y = cy;
	  p_colloid->v.z = cz;

	  /* Generate correlated random pairs */

	  sigma_r = sqrt(rmass*kT*(2.0*xidt - 3.0 + 4.0*c0 - c0*c0)/(xi*xi));
	  c12 = rmass*kT*(1.0 - c0)*(1.0 - c0) / (sigma_v*sigma_r*xi);
	  c21 = sqrt(1.0 - c12*c12);

	  ranx = (c12*ranx + c21*p_colloid->random[3]);
	  rany = (c12*rany + c21*p_colloid->random[4]);
	  ranz = (c12*ranz + c21*p_colloid->random[5]);

	  /* Position update */

	  cx = c1*p_colloid->v.x + rmass*c2*p_colloid->force.x + sigma_r*ranx;
	  cy = c1*p_colloid->v.y + rmass*c2*p_colloid->force.y + sigma_r*rany;
	  cz = c1*p_colloid->v.z + rmass*c2*p_colloid->force.z + sigma_r*ranz;

	  p_colloid->r.x += cx;
	  p_colloid->r.y += cy;
	  p_colloid->r.z += cz;

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  brownian_set_random
 *
 *  Set the table of random numbers for each particle.
 *
 *****************************************************************************/

void brownian_set_random() {

  int     ic, jc, kc;
  Colloid * p_colloid;

  /* Set random numbers for each particle */

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->random[0] = ran_parallel_gaussian();
	  p_colloid->random[1] = ran_parallel_gaussian();
	  p_colloid->random[2] = ran_parallel_gaussian();
	  p_colloid->random[3] = ran_parallel_gaussian();
	  p_colloid->random[4] = ran_parallel_gaussian();
	  p_colloid->random[5] = ran_parallel_gaussian();

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  brownian_step_no_interia_test
 *
 *  We test the single-particle position and orientational
 *  correlators:
 *
 *  < [r(t) - r(0)][r(t) - r(0)] > = 2D_t t I     t >> m / gamma_t
 *  < |s(t) - s(0)|^2 >            = 4D_r t       2D_r t << 1
 *
 *  where I is the identity matrix.
 *
 *  See, for example, Jan Dhont 'An Introduction to dynamics of colloids'
 *  Chapter 2.
 *
 *  The correlators should be good to about 1% after 100,000 iterations.
 *
 *****************************************************************************/

void brownian_step_no_inertia_test() {

  const int ntime = 16;
  const int nmax  = 10000;

  int     ic, jc, kc;
  Colloid * p_colloid;
  int     i, n, nt;
  int     ncell[3];
  double  diffr, difft;
  double  r[3][ntime];
  double  s[3][ntime];
  double  correlator_rr[3][ntime];
  double  correlator_ss[ntime];
  double  dr[3];


  difft = get_kT() / (6.0*PI*get_eta_shear()*2.3);
  diffr = get_kT() / (8.0*PI*get_eta_shear()*2.3*2.3*2.3);

  /* Initialise */

  for (i = 0; i < 3; i++) ncell[i] = Ncell(i);

  for (nt = 0; nt < ntime; nt++) {
    for (i = 0; i < 3; i++) {
      correlator_rr[i][nt] = 0.0;
    }
    correlator_ss[nt] = 0.0;
  }

  /* Start the main iteration loop */

  for (n = 1; n <= nmax; n++) {

    brownian_set_random();

    CCOM_halo_particles();
    brownian_step_no_inertia();
    cell_update();

    /* Rotate the tables of position and orientation backwards
     * and store the current values */

    for (nt = ntime-1; nt >= 1; nt--) {
      for (i = 0; i < 3; i++) {
	r[i][nt] = r[i][nt-1];
	s[i][nt] = s[i][nt-1];
      }
    }

    for (ic = 1; ic <= ncell[X]; ic++) {
      for (jc = 1; jc <= ncell[Y]; jc++) {
	for (kc = 1; kc <= ncell[Z]; kc++) {

	  p_colloid = CELL_get_head_of_list(ic, jc, kc);

	  while (p_colloid) {
	    r[X][0] = p_colloid->r.x;
	    r[Y][0] = p_colloid->r.y;
	    r[Z][0] = p_colloid->r.z;
	    s[X][0] = p_colloid->s[X];
	    s[Y][0] = p_colloid->s[Y];
	    s[Z][0] = p_colloid->s[Z];

	    p_colloid = p_colloid->next;
	  }
	}
      }
    }

    /* Work out the contribution to the correlators */

    if ( n > ntime) {
      for (nt = 0; nt < ntime; nt++) {
	for (i = 0; i < 3; i++) {
	  /* correct for periodic boundaries */
	  dr[i] = (r[i][nt] - r[i][0]);
	  if (dr[i] > L(i)/2.0) dr[i] -= L(i);
	  if (dr[i] <-L(i)/2.0) dr[i] += L(i);
	  correlator_rr[i][nt] += dr[i]*dr[i];
	  correlator_ss[nt] += (s[i][nt] - s[i][0])*(s[i][nt] - s[i][0]);
	}
      }
    }

  }

  /* Results */

  info("\n\n");
  info("           Translational                       Orientational\n");
  info("    Time   theory     <rx>     <ry>     <rz>   ");
  info("theory     <ss>\n");
  
  for (n = 0; n < ntime; n++) {
    double tau = n*dt_;
    info("%8.3f %8.3f ",  tau, 2.0*tau);
    info("%8.3f ", correlator_rr[X][n]/(difft*nmax));
    info("%8.3f ", correlator_rr[Y][n]/(difft*nmax));
    info("%8.3f ", correlator_rr[Z][n]/(difft*nmax));
    info("%8.3f %8.3f\n", 4.0*tau, correlator_ss[n]/(diffr*nmax));
  }

  return;
}
