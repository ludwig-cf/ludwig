/*****************************************************************************
 *
 *  stats_distribution.c
 *
 *  Various routines to compute statistics associated with the
 *  distribution (that is, the density).
 *
 *  If there is more than one distribution, it is assumed the relevant
 *  statistics are produced in the order parameter sector.
 *
 *  $Id: stats_distribution.c,v 1.1.2.2 2010-04-02 07:56:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "site_map.h"
#include "stats_distribution.h"

/*****************************************************************************
 *
 *  stats_distribution_print
 *
 *  This routine prints some statistics related to the first distribution
 *  (always assumed to be the density).
 *
 *****************************************************************************/

void stats_distribution_print(void) {

  int ic, jc, kc, index;
  int nlocal[3];

  double stat_local[5];
  double stat_total[5];
  double rho;
  double rhomean;
  double rhovar;

  coords_nlocal(nlocal);

  stat_local[0] = 0.0;       /* Volume */
  stat_local[1] = 0.0;       /* total mass (or density) */
  stat_local[2] = 0.0;       /* variance rho^2 */
  stat_local[3] = +DBL_MAX;  /* min local density */
  stat_local[4] = -DBL_MAX;  /* max local density */

  for (ic = 1;  ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        if (site_map_get_status(ic, jc, kc) != FLUID) continue;
        index = coords_index(ic, jc, kc);

	rho = distribution_zeroth_moment(index, 0);
	stat_local[0] += 1.0;
	stat_local[1] += rho;
	stat_local[2] += rho*rho;
	stat_local[3] = dmin(rho, stat_local[3]);
	stat_local[4] = dmax(rho, stat_local[4]);
      }
    }
  }

  MPI_Reduce(stat_local, stat_total, 3, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(stat_local + 3, stat_total + 3, 1, MPI_DOUBLE, MPI_MIN, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(stat_local + 4, stat_total + 4, 1, MPI_DOUBLE, MPI_MAX, 0,
	     MPI_COMM_WORLD);

  /* Compute mean density, and the variance, and print. We
   * assume the fluid volume (stat_total[0]) is not zero... */ 

  rhomean = stat_total[1]/stat_total[0];
  rhovar  = (stat_total[2]/stat_total[0]) - rhomean*rhomean;

  info("\nScalars - total mean variance min max\n");
  info("[rho] %14.2f %14.11f%14.7e %14.11f%14.11f\n",
       stat_total[1], rhomean, rhovar, stat_total[3], stat_total[4]); 

  return;
}

/*****************************************************************************
 *
 *  stats_distribution_momentum
 *
 *  Returns the fluid momentum (always distribution 0).
 *
 *****************************************************************************/

void stats_distribution_momentum(double g[3]) {

  int ic, jc, kc, index;
  int nlocal[3];

  double g_local[3];
  double g_site[3];

  coords_nlocal(nlocal);

  g_local[X] = 0.0;
  g_local[Y] = 0.0;
  g_local[Z] = 0.0;

  for (ic = 1;  ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        if (site_map_get_status(ic, jc, kc) != FLUID) continue;
        index = coords_index(ic, jc, kc);

	distribution_first_moment(index, 0, g_site);
	g_local[X] += g_site[X];
	g_local[Y] += g_site[Y];
	g_local[Z] += g_site[Z];
      }
    }
  }

  MPI_Reduce(g_local, g, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return;
}
