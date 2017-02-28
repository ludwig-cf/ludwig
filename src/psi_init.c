/*****************************************************************************
 *
 *  psi_init.c
 *
 *  Various initial states for electrokinetics.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-16 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
 
#include "pe.h"
#include "coords.h"
#include "psi_init.h"
#include "psi_s.h"

/*****************************************************************************
 *
 *  psi_init_uniform
 *
 *  Set the charge density for all species to be rho_el everywhere.
 *  The potential is initialised to zero.
 *
 *****************************************************************************/

int psi_init_uniform(psi_t * obj, double rho_el) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nk;

  assert(obj);
  assert(rho_el >= 0.0);

  cs_nlocal(obj->cs, nlocal);
  psi_nk(obj, &nk);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);

	psi_psi_set(obj, index, 0.0);

	for (n = 0; n < nk; n++) {
	  psi_rho_set(obj, index, n, rho_el);
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * psi_init_gouy_chapman
 *
 *  Set rho(x = 1)  = + (1/2NyNz)
 *      rho(x = Lx) = + (1/2NyNz)
 *      rho         = - 1/(NyNz*(Nx-2)) + electrolyte
 *
 *  This sets up the system for Gouy-Chapman.
 *
 *  rho_el is the electrolyte (background) charge density.
 *  sigma is the sufrace charge density at the wall.
 *
 *****************************************************************************/

int psi_init_gouy_chapman(psi_t * obj, map_t * map, double rho_el,
			      double sigma) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  double rho_w, rho_i;
  double ltot[3];

  assert(obj);
  assert(map);

  cs_nlocal(obj->cs, nlocal);
  cs_ltot(obj->cs, ltot);
  cs_cartsz(obj->cs, mpi_cartsz);
  cs_cart_coords(obj->cs, mpi_cartcoords);

  /* wall surface charge density */
  rho_w = sigma;

  /* counter charge density */
  rho_i = rho_w * 2.0 *ltot[Y]*ltot[Z] / (ltot[Y]*ltot[Z]*(ltot[X] - 2.0));

  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);

	psi_psi_set(obj, index, 0.0);
	psi_rho_set(obj, index, 0, rho_el);
	psi_rho_set(obj, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (mpi_cartcoords[X] == 0) {
    ic = 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  if (mpi_cartcoords[X] == mpi_cartsz[X] - 1) {
    ic = nlocal[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  map_halo(map);
  map_pm_set(map, 1);

  return 0;
}

/*****************************************************************************
 *
 *  psi_init_liquid_junction
 *
 *  Set rho(1 <= x <= Lx/2)    = 1.01 * electrolyte
 *      rho(Lx/2+1 <= x <= Lx) = 0.99 * electrolyte
 *
 *  This sets up the system for liquid junction potential.
 *
 *  rho_el is the average electrolyte concentration.
 *  delta_el is the relative difference of the concentrations.
 *
 *****************************************************************************/

int psi_init_liquid_junction(psi_t * obj, double rho_el, double delta_el) {

  int ic, jc, kc, index;
  int ntotal[3];
  int nlocal[3], noff[3];

  assert(obj);

  cs_nlocal(obj->cs, nlocal);
  cs_ntotal(obj->cs, ntotal);
  cs_nlocal_offset(obj->cs, noff);

  /* Set electrolyte densities */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(obj->cs, ic, jc, kc);

	psi_psi_set(obj, index, 0.0);

	if ((1 <= noff[0] + ic) && (noff[0] + ic < ntotal[X]/2)) {
	  psi_rho_set(obj, index, 0, rho_el * (1.0 + delta_el));
	  psi_rho_set(obj, index, 1, rho_el * (1.0 + delta_el));
	}
	else{
	  psi_rho_set(obj, index, 0, rho_el * (1.0 - delta_el));
	  psi_rho_set(obj, index, 1, rho_el * (1.0 - delta_el));
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * psi_init_sigma
 *
 *  This sets up surface charges as specified in the porous media file.
 *
 *****************************************************************************/

int psi_init_sigma(psi_t * psi, map_t * map) {

  int ic, jc, kc, index;
  int nlocal[3];
  double sigma; /* point charge or surface charge density */

  assert(psi);
  assert(map);

  cs_nlocal(psi->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);
	map_data(map, index, &sigma);

	psi_psi_set(psi, index, 0.0);

	if (sigma) {
	  if (sigma > 0) {
	    psi_rho_set(psi, index, 0, sigma);
	    psi_rho_set(psi, index, 1, 0);
	  }
	  if (sigma < 0) {
	    psi_rho_set(psi, index, 0, 0);
	    psi_rho_set(psi, index, 1, sigma);
	  }
	}

      }
    }
  }

  map_halo(map);

  return 0;
}

