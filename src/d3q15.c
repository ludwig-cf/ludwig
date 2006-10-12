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
			 { 1,  1,  1}, { 1,  1, -1}, { 1,  0,  0},
			 { 1, -1,  1}, { 1, -1, -1}, { 0,  1,  0},
                         { 0,  0,  1}, { 0,  0, -1}, { 0, -1,  0},
			 {-1,  1,  1}, {-1,  1, -1}, {-1,  0,  0},
			 {-1, -1,  1}, {-1, -1, -1}};

const double chi1[NVEL] = {-2.0,
			   -2.0, -2.0,  1.0, -2.0, -2.0,  1.0,  1.0,
			    1.0,  1.0, -2.0, -2.0,  1.0, -2.0, -2.0};

const double chi3[NVEL] = { 0.0,
			    1.0, -1.0,  0.0, -1.0,  1.0,  0.0,  0.0,
			    0.0,  0.0, -1.0,  1.0,  0.0,  1.0, -1.0};


#define w0 (16.0/72.0)
#define w1 ( 8.0/72.0)
#define w3 ( 1.0/72.0)

const double wv[NVEL] = {w0,
			 w3, w3, w1, w3, w3, w1, w1,
			 w1, w1, w3, w3, w1, w3, w3};

static double jchi1[NVEL][3];

static double var_chi1;  /* Variance for chi1 mode fluctuations */
static double var_jchi1; /* Variance for jchi1 mode fluctuations */
static double var_chi3;  /* Variance for chi3 mode fluctuations */

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
