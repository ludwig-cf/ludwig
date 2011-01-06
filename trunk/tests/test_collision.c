/*****************************************************************************
 *
 *  test_collision.c
 *
 *  Tests for collision stage.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "fluctuations.h"
#include "tests.h"

#define NSAMPLE 1000000

static void test_fluctuations_stats1(void);

/*****************************************************************************
 *
 *  main.c
 * 
 *****************************************************************************/

int main (int argc, char ** argv) {

  pe_init(argc, argv);

  test_fluctuations_stats1();

  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_fluctuations_stats1
 *
 *  Compute the moments, which should be correct to fifth order.
 *  The initialisation is copied from collision_init for the time being.
 *
 *****************************************************************************/

static void test_fluctuations_stats1(void) {

  int ntotal[3] = {8, 8, 8};
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  int n, ns, nsites;
  int is_local;

  unsigned int serial[4] = {13, 829, 2441, 22383979};
  unsigned int state[4];

  double * moment6;
  double   rnorm;

  double r[NFLUCTUATION];
  fluctuations_t * f;


  coords_ntotal_set(ntotal);
  coords_init();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  nsites = coords_nsites();

  f = fluctuations_create(nsites);

  for (ic = 1; ic <= ntotal[X]; ic++) {
    for (jc = 1; jc <= ntotal[Y]; jc++) {
      for (kc = 1; kc <= ntotal[Z]; kc++) {
        state[0] = fluctuations_uniform(serial);
        state[1] = fluctuations_uniform(serial);
        state[2] = fluctuations_uniform(serial);
        state[3] = fluctuations_uniform(serial);
        is_local = 1;
        if (ic <= noffset[X] || ic > noffset[X] + nlocal[X]) is_local = 0;
        if (jc <= noffset[Y] || jc > noffset[Y] + nlocal[Y]) is_local = 0;
        if (kc <= noffset[Z] || kc > noffset[Z] + nlocal[Z]) is_local = 0;
        if (is_local) {
          index = coords_index(ic-noffset[X], jc-noffset[Y], kc-noffset[Z]);
          fluctuations_state_set(f, index, state);
        }
      }
    }
  }

  moment6 = (double *) calloc(6*nsites, sizeof(double));
  if (moment6 == NULL) fatal("calloc(moment6) failed\n");

  /* Loop */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = coords_index(ic, jc, kc);

	for (ns = 0; ns < NSAMPLE; ns++) {
	  fluctuations_reap(f, index, r);
	  for (n = 0; n < NFLUCTUATION; n++) {
	    moment6[0*nsites + index] += r[n];
	    moment6[1*nsites + index] += r[n]*r[n];
	    moment6[2*nsites + index] += r[n]*r[n]*r[n];
	    moment6[3*nsites + index] += r[n]*r[n]*r[n]*r[n];
	    moment6[4*nsites + index] += r[n]*r[n]*r[n]*r[n]*r[n];
	    moment6[5*nsites + index] += r[n]*r[n]*r[n]*r[n]*r[n]*r[n];
	  }
	}
	/* Next site. */
      }
    }
  }

  /* Stat. */

  rnorm = 1.0/((double) NSAMPLE*NFLUCTUATION);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = coords_index(ic, jc, kc);

	info("%2d %2d %2d %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n",
	     ic, jc, kc,
	     rnorm*moment6[0*nsites + index], rnorm*moment6[1*nsites + index],
	     rnorm*moment6[2*nsites + index], rnorm*moment6[3*nsites + index],
	     rnorm*moment6[4*nsites + index], rnorm*moment6[5*nsites + index]);
      }
    }
  }

  free(moment6);
  fluctuations_destroy(f);
  coords_finish();

  return;
}
