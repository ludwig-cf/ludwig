/*****************************************************************************
 *
 *  phi_fluctuations.c
 *
 *  Order parameter fluctuations.
 *
 *  $Id: phi_fluctuations.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "ran.h"
#include "coords.h"
#include "leesedwards.h"

static double * siteflux_;

static void phi_fluctuations_siteflux_set(void);
static void phi_fluctuations_siteflux_halo(void);
static void phi_fluctuations_random_flux_set(double * fluxw, double * fluxe,
					     double * fluxy, double * fluxz);

/*****************************************************************************
 *
 *  phi_fluctuations_random_flux
 *
 *  Lees-Edwards will be pending implemtation of general
 *  field transformation function and validation for no LE.
 *  MPI pending validation.
 *
 *****************************************************************************/

void phi_fluctuations_random_flux(double * fluxw, double * fluxe,
				  double * fluxy, double * fluxz) {
  int nsites;

  nsites = le_nsites();

  siteflux_ = (double*) malloc(3*nsites*sizeof(double));
  if (siteflux_ == NULL) fatal("malloc(siteflux_) failed\n");

  phi_fluctuations_siteflux_set();
  phi_fluctuations_siteflux_halo();
  phi_fluctuations_random_flux_set(fluxw, fluxe, fluxy, fluxz);

  free(siteflux_);

  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_siteflux_set
 *
 *****************************************************************************/

static void phi_fluctuations_siteflux_set(void) {

  int nlocal[3];
  int ic, jc, kc, index;
  int ia;

  double mobility;
  double kt;
  double rvar;

  coords_nlocal(nlocal);

  /* assignment of kt, mobility pending refactor of physics.h */

  mobility = 1.0;
  kt = 0.0;
  rvar = sqrt(2.0*mobility*kt);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  siteflux_[3*index + ia] = rvar*ran_parallel_gaussian();
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_siteflux_halo
 *
 *****************************************************************************/

static void phi_fluctuations_siteflux_halo(void) {

  int nlocal[3];
  int ic, jc, kc;
  int index1, index2;
  int ia;

  coords_nlocal(nlocal);

  assert(pe_size() == 1);

  for (jc = 1; jc <= nlocal[Y]; jc++) {
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      index1 = coords_index(0, jc, kc);
      index2 = coords_index(nlocal[X], jc, kc);
      for (ia = 0; ia < 3; ia++) {
	siteflux_[3*index1 + ia] = siteflux_[3*index2 + ia];
      }
      index1 = coords_index(1, jc, kc);
      index2 = coords_index(nlocal[X]+1, jc, kc);
      for (ia = 0; ia < 3; ia++) {
	siteflux_[3*index2 + ia] = siteflux_[3*index1 + ia];
      }
    }
  }

  for (ic = 0; ic <= nlocal[X] + 1; ic++) {
    for (kc = 1; kc <= nlocal[Z]; kc++) {
      index1 = coords_index(ic, 0, kc);
      index2 = coords_index(ic, nlocal[Y], kc);
      for (ia = 0; ia < 3; ia++) {
	siteflux_[3*index1 + ia] = siteflux_[3*index2 + ia];
      }
      index1 = coords_index(ic, 1, kc);
      index2 = coords_index(ic, nlocal[Y] + 1, kc);
      for (ia = 0; ia < 3; ia++) {
	siteflux_[3*index2 + ia] = siteflux_[3*index1 + ia];
      }
    }
  }

  for (ic = 0; ic <= nlocal[X] + 1; ic++) {
    for (jc = 0; jc <= nlocal[Y] + 1; jc++) {
      index1 = coords_index(ic, jc, 0);
      index2 = coords_index(ic, jc, nlocal[Z]);
      for (ia = 0; ia < 3; ia++) {
	siteflux_[3*index1 + ia] = siteflux_[3*index2 + ia];
      }
      index1 = coords_index(ic, jc, 1);
      index2 = coords_index(ic, jc, nlocal[Z] + 1);
      for (ia = 0; ia < 3; ia++) {
	siteflux_[3*index2 + ia] = siteflux_[3*index1 + ia];
      }
    }
  }   

  return;
}

/*****************************************************************************
 *
 *  phi_fluctuations_random_flux_set
 *
 *  Used 'centred difference' to obtain the face fluxes from the
 *  site fluxes.
 *
 *****************************************************************************/

static void phi_fluctuations_random_flux_set(double * fluxw, double * fluxe,
					     double * fluxy, double * fluxz) {
  int nlocal[3];
  int ic, jc, kc;
  int index, index1;

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	index1 = coords_index(ic-1, jc, kc);
	fluxw[index] = 0.5*(siteflux_[3*index1 + X] + siteflux_[3*index + X]);
	index1 = coords_index(ic+1, jc, kc);
	fluxe[index] = 0.5*(siteflux_[3*index + X] + siteflux_[3*index1 + X]);

	index1 = coords_index(ic, jc+1, kc);
	fluxy[index] = 0.5*(siteflux_[3*index + Y] + siteflux_[3*index1 + Y]);

	index1 = coords_index(ic, jc, kc+1);
	fluxz[index] = 0.5*(siteflux_[3*index + Z] + siteflux_[3*index1 + Z]);
      }
    }
  }

  return;
}
