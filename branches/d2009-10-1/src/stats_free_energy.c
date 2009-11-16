/****************************************************************************
 *
 *  stats_free_energy.c
 *
 *  Statistics for free energy density.
 *
 *  $Id: stats_free_energy.c,v 1.1.2.1 2009-11-16 16:21:38 kevin Exp $
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
#include "site_map.h"
#include "free_energy.h"
#include "stats_free_energy.h"

/****************************************************************************
 *
 *  stats_free_energy_density
 *
 ****************************************************************************/

void stats_free_energy_density(void) {

  int ic, jc, kc, index;
  int nlocal[3];

  double fed;
  double fe_local[2];
  double fe_total[2];

  double (* free_energy_density)(const int index);

  get_N_local(nlocal);
  free_energy_density = fe_density_function();

  fe_local[0] = 0.0; /* Total */
  fe_local[1] = 0.0; /* Fluid */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	fed = free_energy_density(index);
	fe_local[0] += fed;
	if (site_map_get_status_index(index) == FLUID) fe_local[1] += fed;
      }
    }
  }

  MPI_Reduce(fe_local, fe_total, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  info("Free energy density [total, fluid]\n");
  info("[fed] %14.7e %14.7e\n", fe_total[0], fe_total[1]);

  return;
}
