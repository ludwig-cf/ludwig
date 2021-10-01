/*****************************************************************************
 *
 *  stats_surfactant.c
 *
 *  Some routines to perform analysis of the surfactant model.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "field.h"
#include "physics.h"
#include "util.h"
#include "surfactant.h"
#include "stats_surfactant.h"

/*****************************************************************************
 *
 *  stats_surfactant_1d
 *
 *  For a 1-d model in which we have a block initialisation of the
 *  composition, work out the profile of the free energy, and hence
 *  the current interfacial tension.
 *
 *****************************************************************************/

int stats_surfactant_1d(fe_surf_t * fe) {

  int index;
  int ic, jc = 1, kc = 1; /* One-dimensional system */
  int nlocal[3];
  double e, e0, excess;
  double psi_0, psi_b;
  double sigma, sigma0;
  double phi[2];

  assert(fe);

  cs_nlocal(fe->cs, nlocal);

  /* To compute the surface tension, run through both interfaces
   * and divide the final excess free energy by 2. We also record
   * the maximum surfactant value (psi_0) along the way. */ 

  e0 = 0.0;
  psi_b = 1.0;
  psi_0 = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {

    index = cs_index(fe->cs, ic, jc, kc);

    fe_surf_fed(fe, index, &e);
    e0 = dmin(e0, e);

    field_scalar_array(fe->phi, index, phi);
    psi_b = dmin(psi_b, phi[1]);
    psi_0 = dmax(psi_0, phi[1]);
  }

  /* Compute the excess free energy */

  excess = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {

    index = cs_index(fe->cs, ic, jc, kc);

    fe_surf_fed(fe, index, &e);
    excess += (e - e0);
  }

  /* Compute the fractional reduction in the surface tension
   * below the bare surface value */

  fe_surf_sigma(fe, &sigma0);
  sigma = (0.5*excess - sigma0)/sigma0;

  /* The sqrt(t) is the usual dependance for analysis of the
   * diffusion problem, so is included here. */

  pe_info(fe->pe, "Surfactant: %12.5e %12.5e %12.5e %12.5e %12.5e\n",
	  psi_b, psi_0, sigma0, sigma, 0.5*excess);

  return 0;
}
