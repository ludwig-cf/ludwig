/****************************************************************************
 *
 *  stats_free_energy.c
 *
 *  Statistics for free energy density.
 *
 *  $Id: stats_free_energy.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#include "pe.h"
#include "coords.h"
#include "control.h"
#include "site_map.h"
#include "free_energy.h"
#include "stats_free_energy.h"

/****************************************************************************
 *
 *  stats_free_energy_density
 *
 *  Tots up the free energy density. This does ignore solid sites,
 *  but does not account for any surface free energy.
 *
 ****************************************************************************/

void stats_free_energy_density(void) {

  int ic, jc, kc, index;
  int nlocal[3];

  double fed;
  double fe_local[2];
  double fe_total[2];
  double rv;

  double (* free_energy_density)(const int index);

  coords_nlocal(nlocal);
  free_energy_density = fe_density_function();

  fe_local[0] = 0.0; /* Total */
  fe_local[1] = 0.0; /* Fluid */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	fed = free_energy_density(index);
	fe_local[0] += fed;
	if (site_map_get_status_index(index) == FLUID) fe_local[1] += fed;
      }
    }
  }

  MPI_Reduce(fe_local, fe_total, 2, MPI_DOUBLE, MPI_SUM, 0, pe_comm());
  rv = 1.0/(L(X)*L(Y)*L(Z));

  info("\nFree energy density - timestep total fluid\n");
  info("[fed] %14d %14.7e %14.7e\n", get_step(), rv*fe_total[0],
       rv*fe_total[1]);

  return;
}
