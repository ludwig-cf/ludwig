/*****************************************************************************
 *
 *  gelx_rt.c
 *
 *  Run time initialisation for active gel free energy.
 *
 *  $Id: gelx_rt.c,v 1.1.2.1 2010-03-04 14:06:46 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "phi.h"
#include "coords.h"
#include "runtime.h"
#include "free_energy.h"
#include "gelx.h"
#include "gelx_rt.h"

/*****************************************************************************
 *
 *  gelx_run_time
 *
 *  Sort out the active gel input parameters.
 *
 *****************************************************************************/

void gelx_run_time(void) {

  /* Vector order parameter (nop = 3) and del^2 required. */

  phi_nop_set(3);
  phi_gradient_level_set(2);
  coords_nhalo_set(2);

  info("Gel X free energy selected.\n");
  info("Vector order parameter nop = 3\n");
  info("Requires up to del^2 derivatives so setting nhalo = 2\n");

  /* PARAMETERS */

  /* Set as required. */

  fe_density_set(gelx_free_energy_density);
  fe_chemical_stress_set(gelx_chemical_stress);

  fatal("Stop here until details are filled in!\n");

  return;
}
