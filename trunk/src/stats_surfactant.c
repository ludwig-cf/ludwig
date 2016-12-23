/*****************************************************************************
 *
 *  stats_surfactant.c
 *
 *  Some routines to perform analysis of the surfactant model.
 *
 *  $Id: stats_surfactant.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2016 The University of Edinburgh
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
 *  TODO: surfactant code to be entirely refactored.
 *
 *****************************************************************************/

int stats_surfactant_1d(fe_surfactant1_t * fe) {

  int index;
  int ic = 1, jc = 1, kc;
  int nlocal[3];
  int nt;
  double e, e0;
  double psi_0, psi_b;
  double sigma, sigma0;
  double phi[2];
  physics_t * phys = NULL;

  /* This is not run in parallel, so assert it's serial.
   * We also require surfactant */

  assert(fe);
  assert(0); /* Check nf = 2 in refactored version */
  assert(pe_size() == 1);

  physics_ref(&phys);
  coords_nlocal(nlocal);

  /* We assume z = 1 is a reasonable choice for the background
   * free energy level, which we need to subtract to find the
   * excess. */

  kc = 1;
  index = coords_index(ic, jc, kc);
  fe_surfactant1_fed(fe, index, &e0);

  assert(0); /* phi and psi required from relevant field */
  phi[0] = 0.0;
  phi[1] = 0.0;

  psi_b = phi[1];

  /* To compute the surface tension, run through both interfaces
   * and divide the final excess free energy by 2. We also record
   * the maximum surfactant value (psi_0) along the way. */ 

  sigma = 0.0;
  psi_0 = 0.0;

  for (kc = 1; kc <= nlocal[Z]; kc++) {

    index = coords_index(ic, jc, kc);

    fe_surfactant1_fed(fe, index, &e0);
    e = 0.0; /* Check what e should be. */
    sigma += 0.5*(e - e0);
    /* field_scalar_array(fe->phi, index, phi);*/
    assert(0); /* Incomplete type above*/
    psi_0 = dmax(psi_0, phi[1]);
  }

  /* Compute the fractional reduction in the surface tension
   * below the bare surface value */

  fe_surfactant1_sigma(fe, &sigma0);
  sigma = (sigma - sigma0)/sigma0;

  /* The sqrt(t) is the usual dependance for analysis of the
   * diffusion problem, so is included here. */

  nt = physics_control_timestep(phys);

  info("Surfactant: %d %12.5e %12.5e %12.5e %12.5e\n", nt,
       sqrt(1.0*nt), psi_b, psi_0, sigma);

  return 0;
}
