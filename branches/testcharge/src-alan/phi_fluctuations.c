/*****************************************************************************
 *
 *  phi_fluctuations.c
 *
 *  Order parameter fluctuations.
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

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "fluctuations.h"

static int              fluctuations_on_ = 0;
static fluctuations_t * fl_;

/*****************************************************************************
 *
 *  phi_fluctuations_on
 *
 *****************************************************************************/

int phi_fluctuations_on(void) {

  return fluctuations_on_;
}

/*****************************************************************************
 *
 *  phi_fluctuations_on_set
 *
 *****************************************************************************/

void phi_fluctuations_on_set(int flag) {

  fluctuations_on_ = flag;
  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_init
 *
 *  Parallel initialisation of fluctuation generator. The initial
 *  state of the fluctuation generator for each latttice site is set
 *  according its global position and a 'master' seed.
 *
 *  If the master_seed provided > 0, then use it to control the
 *  first part of the initial state, which has a default
 *  value as below.
 *
 *****************************************************************************/

void phi_fluctuations_init(unsigned int master_seed) {

  int nsites;
  int ic, jc, kc, index;
  int ig, jg, kg;
  int nextra;
  int nlocal[3];
  int noffset[3];
  int ntotal[3];

  unsigned int serial[NFLUCTUATION_STATE] = {13, 12953, 712357, 22383979};
  unsigned int serial_local[NFLUCTUATION_STATE];
  unsigned int state[NFLUCTUATION_STATE];

  assert(NFLUCTUATION_STATE == 4); /* Assumed below */

  if (fluctuations_on_ == 0) return;
  if (master_seed > 0) serial[0] = master_seed;

  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  nsites = coords_nsites();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  nsites = coords_nsites();
  nextra = 1;

  fl_ = fluctuations_create(nsites);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {

    /* (ig, jg, kg) is the global lattice position, used for
     * decomposition-independent initial random state */

    ig = noffset[X] + ic;
    if (ig < 1) ig += ntotal[X];
    if (ig > ntotal[X]) ig -= ntotal[X];
 
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      jg = noffset[Y] + jc;
      if (jg < 1) jg += ntotal[Y];
      if (jg > ntotal[Y]) jg -= ntotal[Y];

      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	kg = noffset[Z] + kc;
	if (kg < 1) kg += ntotal[Z];
	if (kg > ntotal[Z]) kg -= ntotal[Z];

	serial_local[0] = serial[0] + ig;
	serial_local[1] = serial[1] + jg;
	serial_local[2] = serial[2] + kg;
	serial_local[3] = serial[3];

        state[0] = fluctuations_uniform(serial_local);
        state[1] = fluctuations_uniform(serial_local);
        state[2] = fluctuations_uniform(serial_local);
        state[3] = fluctuations_uniform(serial_local);

	index = coords_index(ic, jc, kc);
	fluctuations_state_set(fl_, index, state);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_finalise
 *
 *****************************************************************************/

void phi_fluctuations_finalise(void) {

  if (fluctuations_on_) fluctuations_destroy(fl_);
  fluctuations_on_ = 0;

  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_site
 *
 *  Computes n random fluxes of variance var at each lattice site.
 *  Include extra points up to halo = 1 to allow face fluxes to be
 *  computed.
 *
 *****************************************************************************/

void phi_fluctuations_site(int n, double var, double * jsite) {

  int nlocal[3];
  int ic, jc, kc, index;
  int ia;
  int nextra;

  double reap[NFLUCTUATION];

  coords_nlocal(nlocal);
  nextra = 1;

  assert(fluctuations_on_);
  assert(n < NFLUCTUATION);
  assert(nextra <= coords_nhalo());

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);
	fluctuations_reap(fl_, index, reap);

	for (ia = 0; ia < n; ia++) {
	  jsite[n*index + ia] = var*reap[ia];
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_qab
 *
 *****************************************************************************/

int phi_fluctuations_qab(int index, double var, double xi[5]) {

  int n;
  double reap[NFLUCTUATION];

  assert(fluctuations_on_);

  fluctuations_reap(fl_, index, reap);

  for (n = 0; n < 5; n++) {
    xi[n] = var*reap[n];
  }
  
  return 0;
}
