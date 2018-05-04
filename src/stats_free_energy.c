/****************************************************************************
 *
 *  stats_free_energy.c
 *
 *  Statistics for free energy density.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "control.h"
#include "util.h"
#include "stats_free_energy.h"

/****************************************************************************
 *
 *  stats_free_energy_density
 *
 *  Tots up the free energy density. The loop here totals the fluid,
 *  and there is an additional calculation for different types of
 *  solid surface.
 *
 *  The mechanism to compute surface free energy contributions requires
 *  much reworking and generalisation; it only really covers LC at present.
 *
 ****************************************************************************/

int stats_free_energy_density(pe_t * pe, cs_t * cs, wall_t * wall, fe_t * fe,
			      map_t * map,
			      colloids_info_t * cinfo) {

#define NSTAT 5

  int ic, jc, kc, index;
  int ntstep;
  int nlocal[3];
  int status;
  int ncolloid;

  double fed;
  double fe_local[NSTAT];
  double fe_total[NSTAT];
  double rv;
  double ltot[3];
  physics_t * phys = NULL;
  MPI_Comm comm;

  assert(pe);
  assert(cs);
  assert(map);

  if (fe == NULL) return 0;

  pe_mpi_comm(pe, &comm);

  cs_ltot(cs, ltot);
  cs_nlocal(cs, nlocal);
  colloids_info_ntotal(cinfo, &ncolloid);

  fe_local[0] = 0.0; /* Total free energy (fluid all sites) */
  fe_local[1] = 0.0; /* Fluid only free energy */
  fe_local[2] = 0.0; /* Volume of fluid */
  fe_local[3] = 0.0; /* surface free energy */
  fe_local[4] = 0.0; /* other wall free energy (walls only) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	map_status(map, index, &status);

	fe->func->fed(fe, index, &fed);
	fe_local[0] += fed;

	if (status == MAP_FLUID) {
	    fe_local[1] += fed;
	    fe_local[2] += 1.0;
	}
      }
    }
  }

  /* A robust mechanism is required to get the surface free energy */

  physics_ref(&phys);
  ntstep = physics_control_timestep(phys);

  if (wall_present(wall)) {

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s1 fs_s2 \n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	    ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	    fe_total[3], fe_total[4]);
  }
  else if (ncolloid > 0) {

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s a f_s/a\n");

    if (fe_total[4] > 0.0) {
      /* Area > 0 means the free energy is available */
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	      ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3], fe_total[4], fe_total[3]/fe_total[4]);
    }
    else {
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e\n",
	      ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3]);
    }
  }
  else {
    MPI_Reduce(fe_local, fe_total, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
    rv = 1.0/(ltot[X]*ltot[Y]*ltot[Z]);

    pe_info(pe, "\nFree energy density - timestep total fluid\n");
    pe_info(pe, "[fed] %14d %17.10e %17.10e\n", ntstep, rv*fe_total[0],
	    fe_total[1]/fe_total[2]);
  }

#undef NSTAT

  return 0;
}
