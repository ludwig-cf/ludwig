/*****************************************************************************
 *
 *  d3q15.c
 *
 *  D3Q15 model definitions and model-dependent code.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "cartesian.h"

#include "d3q15.h"

#ifndef _D3Q19_

/*****************************************************************************
 *
 *  There are 15 eigenvectors:
 *
 *  rho             (eigenvector implicitly {1})
 *  rho cv[p][X]    (x-component of velocity)
 *  rho cv[p][Y]    (y-component of velocity) 
 *  rho cv[p][Z]    (z-component of velocity)
 *  q[p][X][X]      (xx component of deviatoric stress)
 *  q[p][X][Y]      (xy component of deviatoric stress)
 *  q[p][X][Z]      (xz ...
 *  q[p][Y][Y]      (yy ...
 *  q[p][Y][Z]      (yz ...
 *  q[p][Z][Z]      (zz ...
 *  chi1[p]         (1st ghost mode)
 *  jchi1[p][X]     (x-component of ghost current chi1[p]*cv[p][X])
 *  jchi1[p][Y]     (y-component of ghost current chi1[p]*cv[p][Y])
 *  jchi1[p][Z]     (z-component of ghost current chi1[p]*cv[p][Z])
 *  chi3[p]         (2nd ghost mode cv[p][X]*cv[p][Y]*cv[p][Z])
 *
 *  The associated quadrature weights are:
 *
 *  wv[p]
 *
 *  Note that q[p][i][j] and jchi1[p][i] are computed at run time.
 *
 *****************************************************************************/

const int cv[NVEL][3] = {{ 0, 0, 0},
			 { 1,-1,-1}, { 1,-1, 1}, { 1, 1,-1},
			 { 1, 1, 1}, { 0, 1, 0}, { 1, 0, 0},
			 { 0, 0, 1}, {-1, 0, 0}, { 0,-1, 0},
			 { 0, 0,-1}, {-1,-1,-1}, {-1,-1, 1},
                         {-1, 1,-1}, {-1, 1, 1}};


const double chi1[NVEL] = {-2.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,
			          1.0,  1.0,  1.0, -2.0, -2.0, -2.0, -2.0};
const double chi3[NVEL] = { 0.0,  1.0, -1.0, -1.0,  1.0,  0.0,  0.0,  0.0,
			          0.0,  0.0,  0.0, -1.0,  1.0,  1.0, -1.0};


#define w0 (16.0/72.0)
#define w1 ( 8.0/72.0)
#define w3 ( 1.0/72.0)

const double wv[NVEL] = {w0, w3, w3, w3, w3, w1, w1, w1, w1, w1, w1,
			 w3, w3, w3, w3};


const int BC_Map[NVEL] = {  0,     /* 0th  element - mirror is 0  */
			    14,     /* 1st  element - mirror is 14 */
			    13,     /* 2nd  element - mirror is 13 */
			    12,     /* 3rd  element - mirror is 12 */
			    11,     /* 4th  element - mirror is 11 */
			     9,     /* 5th  element - mirror is 9  */
			     8,     /* 6th  element - mirror is 8  */
			     10,    /* 7th  element - mirror is 10 */
			     6,     /* 8th  element - mirror is 6  */
			     5,     /* 9th  element - mirror is 5  */
			     7,     /* 10th element - mirror is 7  */
			     4,     /* 11th element - mirror is 11 */
			     3,     /* 12th element - mirror is 3  */
			     2,     /* 13th element - mirror is 2  */
			     1 };   /* 14th element - mirror is 1  */

Site * site;

static double jchi1[NVEL][3];

static double var_chi1;  /* Variance for chi1 mode fluctuations */
static double var_jchi1; /* Variance for jchi1 mode fluctuations */
static double var_chi3;  /* Variance for chi3 mode fluctuations */

static void d3q15_propagate_single(void);
static void d3q15_propagate_binary(void);


/*****************************************************************************
 *
 *  propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

void propagation() {

  TIMER_start(TIMER_PROPAGATE);

#ifdef _SINGLE_FLUID_
  d3q15_propagate_single();
#else
  d3q15_propagate_binary();
#endif

  TIMER_stop(TIMER_PROPAGATE);

  return;
}

/*****************************************************************************
 *
 *  d3q15_propagate_single
 *
 *****************************************************************************/

void d3q15_propagate_single() {

  int i, j, k, ii, jj;
  int xfac, yfac, stride1, stride2;
  int N[3];


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

  return;
}


/*****************************************************************************
 *
 *  d3q15_propagate_binary
 *
 *  This is the binary fluid version.
 *
 *****************************************************************************/

void d3q15_propagate_binary() {

  int i, j, k, ii, jj;
  int xfac, yfac, stride1, stride2;
  int N[3];


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
	site[jj + k].g[7] = site[jj + k - 1].g[7];
	site[jj + k].f[5] = site[jj + k - yfac].f[5];
	site[jj + k].g[5] = site[jj + k - yfac].g[5];
	site[jj + k].f[6] = site[jj + k - xfac].f[6];
	site[jj + k].g[6] = site[jj + k - xfac].g[6];
	site[jj + k].f[4] = site[jj + k - xfac - yfac - 1].f[4];
	site[jj + k].g[4] = site[jj + k - xfac - yfac - 1].g[4];
      }

      for(k = 1; k <= N[Z]; k++) {
	site[jj + k].f[3] = site[jj + k - xfac - yfac + 1].f[3];
	site[jj + k].g[3] = site[jj + k - xfac - yfac + 1].g[3];
      }
    }

    /* y-component -ve */
    for(j = 1;j <= N[Y]; j++) {
      jj = ii + j*yfac;

      for(k = 1; k <= N[Z]; k++) {
	site[jj+k].f[1] = site[jj+k+stride1].f[1];
	site[jj+k].g[1] = site[jj+k+stride1].g[1];
      }
      for(k = N[Z]; k > 0; k--) {
	site[jj+k].f[2] = site[jj+k+stride2].f[2];
	site[jj+k].g[2] = site[jj+k+stride2].g[2];
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
	site[jj + k].g[8] = site[jj + k + xfac].g[8];
	site[jj + k].f[9] = site[jj + k + yfac].f[9];
	site[jj + k].g[9] = site[jj + k + yfac].g[9];
	site[jj + k].f[10] = site[jj + k + 1].f[10];
	site[jj + k].g[10] = site[jj + k + 1].g[10];
	site[jj + k].f[11] = site[jj + k + xfac + yfac + 1].f[11];
	site[jj + k].g[11] = site[jj + k + xfac + yfac + 1].g[11];
      }

      for(k = N[Z]; k > 0; k--) {
	site[jj + k].f[12] = site[jj + k + xfac + yfac - 1].f[12];
	site[jj + k].g[12] = site[jj + k + xfac + yfac - 1].g[12];
      }
    }

    /* y-component +ve */
    for(j = N[Y]; j > 0; j--) {
      jj = ii + j*yfac;

      for(k = 1; k <= N[Z]; k++) {
	site[jj + k].f[13] = site[jj + k + xfac - yfac + 1].f[13];
	site[jj + k].g[13] = site[jj + k + xfac - yfac + 1].g[13];
      }

      for(k = N[Z]; k > 0; k--) {
	site[jj + k].f[14] = site[jj + k + xfac - yfac - 1].f[14];
	site[jj + k].g[14] = site[jj + k + xfac - yfac - 1].g[14];
      }
    }

  }

  return;
}

/*****************************************************************************
 *
 *  init_ghosts
 *
 *  Initialise the D3Q15 ghost mode jchi1, and the variances for
 *  ghost fluctuations, which depend on the relaxation time, and
 *  the "temperature".
 *
 *****************************************************************************/

void init_ghosts(const double kT) {

  double tau_ghost = 1.0;
  double var;
  int    i, p;

  /* Set the ghost current eigenvectors jchi1 */

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < 3; i++) {
      jchi1[p][i] = chi1[p]*cv[p][i];
    }
  }

  /* These are the variances for fluctuations in ghosts */

  var = sqrt((tau_ghost + tau_ghost - 1.0)/(tau_ghost*tau_ghost));

  var_chi1  = sqrt(kT)*sqrt(2.0)*var;
  var_jchi1 = sqrt(kT)*sqrt(2.0/3.0)*var;
  var_chi3  = sqrt(kT)*sqrt(1.0/9.0)*var;

  return;
}

/*****************************************************************************
 *
 *  get_ghosts
 *
 *  Work out the model-dependent part of the distribution for the
 *  collision stage.
 *
 *****************************************************************************/

void get_ghosts(double fghost[]) {

  double chi1hat;
  double jchi1hat[3];
  double chi3hat;

  const double r2   = (1.0/2.0);   /* Normaliser for chi1 mode */
  const double c3r2 = (3.0/2.0);   /* Normaliser for jchi1 mode */
  const double c9   = 9.0;         /* Normaliser for chi3 mode */

  int   i, p;

  /* Set fluctuating parts and and tot up the ghost projection. */

  chi1hat      = ran_parallel_gaussian()*var_chi1;
  jchi1hat[0]  = ran_parallel_gaussian()*var_jchi1;
  jchi1hat[1]  = ran_parallel_gaussian()*var_jchi1;
  jchi1hat[2]  = ran_parallel_gaussian()*var_jchi1;
  chi3hat      = ran_parallel_gaussian()*var_chi3;

  for (p = 0; p < NVEL; p++) {

    fghost[p] = r2*chi1hat*chi1[p];

    for (i = 0; i < 3; i++) {
      fghost[p] += c3r2*jchi1hat[i]*jchi1[p][i];
    }

    fghost[p] += c9*chi3[p]*chi3hat;
  }

  return;
}

#endif
