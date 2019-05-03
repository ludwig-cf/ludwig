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
 *  TODO: surfactant code to be entirely refactored.
 *
 *****************************************************************************/

int stats_surfactant_1d(fe_surf1_t * fe) {

  int index;
  int ic = 1, jc = 1, kc;
  int nlocal[3];
  int nt;
  double e, e0;
  double psi_0, psi_b;
  double sigma, sigma0;
  double phi[2];

  cs_t * cs = NULL;
  physics_t * phys = NULL;

  /* This is not run in parallel, so assert it's serial.
   * We also require surfactant */

  assert(fe);
  assert(0); /* Check nf = 2 in refactored version */
  /* assert(pe_size() == 1);*/

  physics_ref(&phys);
  cs_nlocal(cs, nlocal);

  /* We assume z = 1 is a reasonable choice for the background
   * free energy level, which we need to subtract to find the
   * excess. */

  kc = 1;
  index = cs_index(cs, ic, jc, kc);
  fe_surf1_fed(fe, index, &e0);

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

    index = cs_index(cs, ic, jc, kc);

    fe_surf1_fed(fe, index, &e0);
    e = 0.0; /* Check what e should be. */
    sigma += 0.5*(e - e0);
    /* field_scalar_array(fe->phi, index, phi);*/
    assert(0); /* Incomplete type above*/
    psi_0 = dmax(psi_0, phi[1]);
  }

  /* Compute the fractional reduction in the surface tension
   * below the bare surface value */

  fe_surf1_sigma(fe, &sigma0);
  sigma = (sigma - sigma0)/sigma0;

  /* The sqrt(t) is the usual dependance for analysis of the
   * diffusion problem, so is included here. */

  nt = physics_control_timestep(phys);

  /* Move to fe_surf? */
  pe_info(NULL, "Surfactant: %d %12.5e %12.5e %12.5e %12.5e\n", nt,
	  sqrt(1.0*nt), psi_b, psi_0, sigma);

  return 0;
}
