/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  $Id: advection_bcs.c,v 1.1.2.1 2009-12-15 16:19:32 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "site_map.h"
#include "phi.h"

/*****************************************************************************
 *
 *  advection_bcs_no_normal_fluxes
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

void advection_bcs_no_normal_flux(double * fluxe, double * fluxw,
				  double * fluxy, double * fluxz) {

  int nlocal[3];
  int ic, jc, kc, index, n;

  double mask, maskw, maske, masky, maskz;

  assert(fluxe);
  assert(fluxw);
  assert(fluxy);
  assert(fluxz);

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	mask  = (site_map_get_status_index(index)  == FLUID);
	maske = (site_map_get_status(ic+1, jc, kc) == FLUID);
	maskw = (site_map_get_status(ic-1, jc, kc) == FLUID);
	masky = (site_map_get_status(ic, jc+1, kc) == FLUID);
	maskz = (site_map_get_status(ic, jc, kc+1) == FLUID);

	for (n = 0;  n < nop_; n++) {
	  fluxw[nop_*index + n] *= mask*maskw;
	  fluxe[nop_*index + n] *= mask*maske;
	  fluxy[nop_*index + n] *= mask*masky;
	  fluxz[nop_*index + n] *= mask*maskz;
	}

      }
    }
  }

  return;
}
