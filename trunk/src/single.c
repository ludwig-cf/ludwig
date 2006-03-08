/*****************************************************************************
 *
 *  single.c
 *
 *  This is an additional version of the collision/propagation
 *  routines which operate for a single fluid. This provides
 *  a significant saving in computational effort if only one
 *  fluid species is required.
 *
 *  Introduced with the preprocessor option _SINGLE_FLUID_
 *  (which also excludes corresponding functions in model.c).
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk) based on model.c
 *
 *****************************************************************************/

#include "globals.h"

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "cartesian.h"

#ifdef _SINGLE_FLUID_

void MODEL_collide_multirelaxtion(void);
void MODEL_limited_propagate(void);

extern double _q[NVEL][3][3];

/*****************************************************************************
 *
 *  MODEL_collide_multirelaxation
 *
 *  Collision with potentially different relaxation for different modes.
 *  From Ronojoy.
 *
 *  
 *  This routine is currently model independent, except that
 *  it is assumed that p = 0 is the null vector in the set.
 *
 *  The BGK form replaces the multirelaxation as :
 *    usq  = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
 *    feq  = wv[p]*rho*(1.0 + rcs2*udotc + r2rcs4*udotc*udotc - r2rcs2*usq);
 *    f[p] = f[p] - rtau*(f[p] - feq);
 *  with appropriate noise variances if required.
 *
 *****************************************************************************/

void MODEL_collide_multirelaxation() {

  int      N[3];
  int      ic, jc, kc, index;       /* site indices */
  int      p;                       /* velocity index */
  int      i, j;                    /* summed over indices ("alphabeta") */
  int      xfac, yfac;

  Float    u[3];                    /* Velocity */
  Float    s[3][3];                 /* Stress */
  Float    rho, rrho;               /* Density, reciprocal density */
  Float    rtau;                    /* Reciprocal \tau */
  Float *  f;

  Float    udotc;
  Float    sdotq;
  Float    cdotf;

  Float    force[3];                /* External force */

  const Float    rcs2   = 3.0;      /* The constant 1 / c_s^2 */
  const Float    r2rcs4 = 4.5;      /* The constant 1 / 2 c_s^4 */

  Float    fr[NVEL];

  extern FVector * _force;
  extern double    normalise;

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);
  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  rtau = 2.0 / (1.0 + 6.0*gbl.eta);

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = ic*xfac + jc*yfac + kc;

	if (site_map[index] != FLUID) continue;

	f = site[index].f;

	rho  = f[0];
	u[0] = 0.0;
	u[1] = 0.0;
	u[2] = 0.0;

	for (p = 1; p < NVEL; p++) {
	  rho  += f[p];
	  u[0] += f[p]*cv[p][0];
	  u[1] += f[p]*cv[p][1];
	  u[2] += f[p]*cv[p][2];
	}

	rrho = 1.0/rho;
	u[0] *= rrho;
	u[1] *= rrho;
	u[2] *= rrho;

	/* The local body force. */
	/* Note the "global" force (gravity) is still constant. */ 

	force[0] = 0.5*(gbl.force.x + (_force + index)->x);
	force[1] = 0.5*(gbl.force.y + (_force + index)->y);
	force[2] = 0.5*(gbl.force.z + (_force + index)->z);

	/* Compute the velocity, taking account of any body force */

	for (i = 0; i < 3; i++) {
	  u[i] += rrho*force[i];  
	}

	/* Stress */

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    s[i][j] = 0.0;

	    for (p = 0; p < NVEL; p++) {
	      s[i][j] += f[p]*_q[p][i][j];
	    }

	    /* Relax this stress mode */
	    s[i][j] = s[i][j] - rtau*(s[i][j] - rho*u[i]*u[j]);

	    /* Add body force terms to give post-collision stress */
	    s[i][j] = s[i][j] + (2.0-rtau)*(u[i]*force[j] + force[i]*u[j]);
	  }
	}

	/* Now update the distribution */

#ifdef _NOISE_
	RAND_fluctuations(fr);
#endif

	for (p = 0; p < NVEL; p++) {

	  udotc = 0.0;
	  sdotq = 0.0;

	  for (i = 0; i < 3; i++) {
	    udotc += (u[i] + rrho*force[i])*cv[p][i];
	    for (j = 0; j < 3; j++) {
	      sdotq += s[i][j]*_q[p][i][j];
	    }
	  }
 
#ifdef _BGK_
	  /* b acts as u^2 and c as the equilibrium feq */
	  b = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
	  c = wv[p]*rho*(1.0 + rcs2*udotc + r2rcs4*udotc*udotc - 0.5*rcs2*b);
	  f[p] = f[p] - rtau*(f[p] - c);

	  /* Body force (Guo et al.) */

	  b = 0.0;
	  c = 0.0;
	  cdotf = 0.0;

	  for (i = 0; i < 3; i++) {
	    b     += (cv[p][i] - u[i])*fxyz[i];
	    c     += cv[p][i]*u[i];
	    cdotf += cv[p][i]*fxyz[i];
	  }

	  f[p] = f[p] + 0.5*(2.0-rtau)*wv[p]*rcs2*(b + rcs2*c*cdotf);
#else
	  /* Reproject */
	  f[p] = wv[p]*(rho + rho*udotc*rcs2 + sdotq*r2rcs4);

#endif

#ifdef _NOISE_
	  /* Noise added here */
	  f[p] = f[p] + normalise*fr[p];
#endif

	  /* Next p */
	}

	/* Next site */
      }
    }
  }
 
 TIMER_stop(TIMER_COLLIDE);

  return;
}


/*****************************************************************************
 *
 *  MODEL_limited_propagate
 *
 *  Propagation scheme for fluid when particles are present.
 *  It supercedes MODEL_propagate to the extent that it includes
 *  no propagation into the halo regions.
 *
 *  This is the single fluid version.
 *
 *****************************************************************************/

void MODEL_limited_propagate() {

  int i, j, k, ii, jj;
  int xfac, yfac, stride1, stride2;
  int N[3];

  TIMER_start(TIMER_PROPAGATE);

  /* Serial or domain decompostion */

  get_N_local(N);
  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  stride1 =  - xfac + yfac + 1;
  stride2 =  - xfac + yfac - 1;

  /* 1st Block: Basis vectors with x-component 0 or +ve */
  
  for(i = N[X]; i > 0; i--) {
    ii = i*xfac;

    /* y-component 0 or +ve */
    for(j = N[Y]; j > 0; j--) {
      jj = ii + j*yfac;

      for(k = N[Z]; k > 0; k--) {
	site[jj + k].f[7] = site[jj + k - 1].f[7];
	site[jj + k].f[5] = site[jj + k - yfac].f[5];
	site[jj + k].f[6] = site[jj + k - xfac].f[6];
	site[jj + k].f[4] = site[jj + k - xfac - yfac - 1].f[4];
      }

      for(k = 1; k <= N[Z]; k++) {
	site[jj + k].f[3] = site[jj + k - xfac - yfac + 1].f[3];
      }
    }

    /* y-component -ve */
    for(j = 1;j <= N[Y]; j++) {
      jj = ii + j*yfac;

      for(k = 1; k <= N[Z]; k++) {
	site[jj+k].f[1] = site[jj+k+stride1].f[1];
      }
      for(k = N[Z]; k > 0; k--) {
	site[jj+k].f[2] = site[jj+k+stride2].f[2];
      }
    }

  }


  /* 2nd block: Basis vectors with x-component -ve */

  for(i = 1; i <= N[X]; i++) {
    ii = i*xfac;

    /* y-component 0 or -ve */
    for(j = 1; j <= N[Y]; j++) {
      jj = ii + j*yfac;

      for(k = 1; k <= N[Z]; k++) {
	site[jj + k].f[8] = site[jj + k + xfac].f[8];
	site[jj + k].f[9] = site[jj + k + yfac].f[9];
	site[jj + k].f[10] = site[jj + k + 1].f[10];
	site[jj + k].f[11] = site[jj + k + xfac + yfac + 1].f[11];
      }

      for(k = N[Z]; k > 0; k--) {
	site[jj + k].f[12] = site[jj + k + xfac + yfac - 1].f[12];
      }
    }

    /* y-component +ve */
    for(j = N[Y]; j > 0; j--) {
      jj = ii + j*yfac;

      for(k = 1; k <= N[Z]; k++) {
	site[jj + k].f[13] = site[jj + k + xfac - yfac + 1].f[13];
      }

      for(k = N[Z]; k > 0; k--) {
	site[jj + k].f[14] = site[jj + k + xfac - yfac - 1].f[14];
      }
    }

  }

  TIMER_stop(TIMER_PROPAGATE);

  return;
}

#endif /* _SINGLE_FLUID_ */
