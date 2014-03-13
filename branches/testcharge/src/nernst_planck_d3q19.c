/*****************************************************************************
 *
 *  nernst_planck2.c
 *
 *  A solution for the Nerst-Planck equation, which is the advection
 *  diffusion equation for charged species \rho_k in the presence of
 *  a potential \psi.
 *
 *  We have, in the most simple case:
 *
 *  d_t rho_k + div . (rho_k u) = div . D_k (grad rho_k + Z_k rho_k grad psi)
 *
 *  where u is the velocity field, D_k are the diffusion constants for
 *  each species k, and Z_k = (valancy_k e / k_bT) = beta valency_k e.
 *  e is the unit charge.
 *
 *  If the chemical potential is mu_k for species k, the diffusive
 *  flux may be written as
 *
 *    j_k = - D_k rho_k grad (beta mu_k)
 *
 *  with mu_k = mu_k^ideal + mu_k^ex = k_bT ln(rho_k) + valency_k e psi.
 *  (For more complex problems, there may be other terms in the chemical
 *  potential.)
 *
 *  As it is important to conserve charge, we solve in a flux form.
 *  Following Capuani, Pagonabarraga and Frenkel, J. Chem. Phys.
 *  \textbf{121} 973 (2004) we include factors to ensure that the
 *  charge densities follow a Boltzmann distribution in equilbrium.
 *
 *  This writes the flux as
 *    j_k = - D_k exp[beta mu_k^ex] grad (rho_k exp[beta mu_k^ex])
 *
 *  which we approximate at the cell faces by (e.g., for x only)
 *
 *    -D_k (1/2) { exp[-beta mu_k^ex(i)] + exp[-beta mu_k^ex(i+1)] }
 *    * { rho_k(i+1) exp[beta mu_k^ex(i+1)] - rho_k(i) exp[beta mu_k^ex(i)] }
 *
 *  We then compute the divergence of the fluxes to update via an
 *  Euler forward step. The advective fluxes (again computed at the
 *  cells faces) may be added to the diffusive fluxes to solve the
 *  whole thing. Appropraite advective fluxes may be computed via
 *  the advection.h interface.
 *
 *  Solid boundaries simply involve enforcing a no normal flux
 *  condition at the cell face.
 *
 *  The potential and charge species are available via the psi_s
 *  object.
 *
 *  A uniform external electric field may be applied; this is done
 *  by adding a contribution to the potential
 *     psi -> psi - eE.r
 *  which just appears as -eE in the calculation of grad psi.
 *
 *
 *  $Id$
 *
 *  Edinbrugh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinbrugh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "psi_s.h"
#include "advection.h"
#include "advection_bcs.h"
#include "free_energy.h"
#include "physics.h"
#include "nernst_planck_d3q19.h"
#include "d3q19.h"
#include "control.h"

static int nernst_planck_fluxes_d3q19(psi_t * psi, double ** flx, map_t *map);
static int nernst_planck_update_d3q19(psi_t * psi, double ** flx, map_t * map, double dt);

/*****************************************************************************
 *
 *  nernst_planck_driver_d3q19
 *
 *  The hydro object is allowed to be NULL, in which case there is
 *  no advection.
 *
 *  The map object is allowed to be NULL, in which case no boundary
 *  condition corrections are attempted.
 *
 *****************************************************************************/

int nernst_planck_driver_d3q19(psi_t * psi, hydro_t * hydro, map_t * map, double dt) {

  int nk;              /* Number of electrolyte species */
  int nsites;          /* Number of lattice sites */
  int ia;

  double ** flx = NULL;

  psi_nk(psi, &nk);
  nsites = coords_nsites();

  /* Allocate fluxes and initialise to zero */
  flx = (double **) calloc(nsites*nk, sizeof(double));
  for (ia = 0; ia < nsites*nk; ia++) {
    flx[ia] = (double *) calloc(NVEL-1, sizeof(double));
  }
  if (flx == NULL) fatal("calloc(flx) failed\n");

  /* Add diffusive fluxes */
  nernst_planck_fluxes_d3q19(psi, flx, map);
  
  /* Apply no-flux BC */
  if (map) advective_bcs_no_flux_d3q19(nk, flx, map);

  /* Update charges */
  nernst_planck_update_d3q19(psi, flx, map, dt);

  for (ia = 0; ia < nsites*nk; ia++) {
    free(flx[ia]);
  }
  free(flx);

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_fluxes_d3q19
 *
 *****************************************************************************/

static int nernst_planck_fluxes_d3q19(psi_t * psi, double ** flx, map_t * map) {

  int ic, jc, kc; 
  int index0, index1;
  int status0, status1;
  int nlocal[3];
  int n, nk; /* Number of charged species */
  int c;
  double delta[18];

  double eunit;
  double beta;
  double b0, b1;
  double mu0, mu1;
  double rho0, rho1;
  double mu_s0, mu_s1;   /* Solvation chemical potential, from free energy */
  double e0[3];

  assert(psi);
  assert(flx);

  coords_nlocal(nlocal);

  psi_nk(psi, &nk);
  psi_unit_charge(psi, &eunit);
  psi_beta(psi, &beta);

  physics_e0(e0);

  /* Lattice spacing */
  delta[0] = sqrt(2.0);
  delta[1] = sqrt(2.0);
  delta[2] = 1.0;
  delta[3] = sqrt(2.0);
  delta[4] = sqrt(2.0);
  delta[5] = sqrt(2.0);
  delta[6] = 1.0;
  delta[7] = sqrt(2.0);
  delta[8] = 1.0;
  delta[9] = 1.0;
  delta[10] = sqrt(2.0);
  delta[11] = 1.0;
  delta[12] = sqrt(2.0);
  delta[13] = sqrt(2.0);
  delta[14] = sqrt(2.0);
  delta[15] = 1.0;
  delta[16] = sqrt(2.0);
  delta[17] = sqrt(2.0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = coords_index(ic, jc, kc);
        map_status(map, index0, &status0);

        for (c = 1; c < NVEL; c++) {

	  index1 = coords_index(ic + cv[c][X], jc + cv[c][Y], kc + cv[c][Z]);
	  map_status(map, index1, &status1);

	  for (n = 0; n < nk; n++) {

	    fe_mu_solv(index0, n, &mu_s0);
	    mu0 = mu_s0 + psi->valency[n]*eunit*psi->psi[index0];
	    rho0 = psi->rho[nk*index0 + n];

	    fe_mu_solv(index1, n, &mu_s1);
	    mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index1] - cv[c][X]*e0[X] - cv[c][Y]*e0[Y] - cv[c][Z]*e0[Z]);
	    b0 = exp(-beta*(mu1 - mu0));
	    b1 = exp(+beta*(mu1 - mu0));
	    rho1 = psi->rho[nk*(index1) + n]*b1;

	    flx[(nk*index0 + n)][c - 1] -= psi->diffusivity[n]*0.5*(1.0 + b0)*(rho1 - rho0) / delta[c - 1];

	  }
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  nernst_planck_update_d3q19
 *
 *****************************************************************************/

static int nernst_planck_update_d3q19(psi_t * psi, double ** flx, map_t * map, double dt) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nhalo;
  int n, nk;
  int c;
  int status;

  assert(psi);
  assert(flx);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  psi_nk(psi, &nk);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
        map_status(map, index, &status);

        if (status == MAP_FLUID) {
	  for (n = 0; n < nk; n++) {
	    for (c = 1; c < NVEL; c++) {
	      psi->rho[nk*index + n] -= flx[nk*index + n][c - 1] * dt;
	    }
	  }
	}

      }
    }
  }

  return 0;
}
