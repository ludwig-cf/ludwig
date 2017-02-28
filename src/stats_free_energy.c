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
 *  (c) 2011-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "control.h"
#include "blue_phase.h"
#include "util.h"
#include "stats_free_energy.h"

static int stats_free_energy_wall(cs_t * cs, wall_t * wall, fe_t * fe,
				  field_t * q, double * fs);
static int stats_free_energy_colloid(fe_t * fe, cs_t * cs,
				     colloids_info_t * cinfo,
				     field_t * q, map_t * map, double * fs);

__host__ int blue_phase_fs(fe_lc_param_t * feparam, const double dn[3],
			   double qs[3][3], char status,
			   double * fe);

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
			      field_t * q, map_t * map,
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

    if (q) stats_free_energy_wall(cs, wall, fe, q, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s1 fs_s2 \n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	    ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	    fe_total[3], fe_total[4]);
  }
  else if (ncolloid > 0) {

    if (q) stats_free_energy_colloid(fe, cs, cinfo, q, map, fe_local + 3);

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

/*****************************************************************************
 *
 *  stats_free_energy_wall
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *  TODO: the rather inelegant wall free energy needs to be
 *        refactored to the more general case.
 *
 *****************************************************************************/

static int stats_free_energy_wall(cs_t * cs, wall_t * wall, fe_t * fe,
				  field_t * q,
				  double * fs) {

  int iswall[3];

  assert(cs);
  assert(wall);
  assert(fe);
  assert(q);

  int stats_free_energy_wallx(cs_t * cs, fe_lc_param_t * fe, field_t * q, double * fs);
  int stats_free_energy_wally(cs_t * cs, fe_lc_param_t * fe, field_t * q, double * fs);
  int stats_free_energy_wallz(cs_t * cs, fe_lc_param_t * fe, field_t * q, double * fs);

  wall_present_dim(wall, iswall);

  if (fe->id == FE_LC) {
    /* Slightly inelegant */
    fe_lc_param_t tmp;
    fe_lc_param_t * param = &tmp;

    fe_lc_param((fe_lc_t *) fe, param);

    if (iswall[X]) stats_free_energy_wallx(cs, param, q, fs);
    if (iswall[Y]) stats_free_energy_wally(cs, param, q, fs);
    if (iswall[Z]) stats_free_energy_wallz(cs, param, q, fs);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_free_energy_wallx
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int stats_free_energy_wallx(cs_t * cs, fe_lc_param_t * fep,
			    field_t * q, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double qs[3][3];
  double fes;

  assert(cs);
  assert(fep);
  assert(q);
  assert(fs);

  fs[0] = 0.0;
  fs[1] = 0.0;

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  dn[Y] = 0.0;
  dn[Z] = 0.0;

  if (mpicoords[X] == 0) {

    ic = 1;
    dn[X] = +1.0;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (mpicoords[X] == mpisz[X] - 1) {

    ic = nlocal[X];
    dn[X] = -1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_free_energy_wally
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int stats_free_energy_wally(cs_t * cs, fe_lc_param_t * fep,
			    field_t * q, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double qs[3][3];
  double fes;

  assert(cs);
  assert(fep);
  assert(q);
  assert(fs);

  fs[0] = 0.0;
  fs[1] = 0.0;

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  dn[X] = 0.0;
  dn[Z] = 0.0;

  if (mpicoords[Y] == 0) {

    jc = 1;
    dn[Y] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (mpicoords[Y] == mpisz[Y] - 1) {

    jc = nlocal[Y];
    dn[Y] = -1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_free_energy_wallz
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int stats_free_energy_wallz(cs_t * cs, fe_lc_param_t * fep,
			    field_t * q, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double qs[3][3];
  double fes;

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(cs);
  assert(fep);
  assert(q);
  assert(fs);

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  dn[X] = 0.0;
  dn[Y] = 0.0;

  if (mpicoords[Z] == 0) {

    kc = 1;
    dn[Z] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (mpicoords[Z] == mpisz[Z] - 1) {

    kc = nlocal[Z];
    dn[Z] = -1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_free_energy_colloid
 *
 *  Return f_s for the local colloid surface (and an area).
 *
 *     fs[0] = free energy (integrated over area)
 *     fs[1] = discrete surface area
 *
 *  TODO: This is really LC_FE only.
 *
 *****************************************************************************/

static int stats_free_energy_colloid(fe_t * fe, cs_t * cs,
				     colloids_info_t * cinfo,
				     field_t * q,
				     map_t * map, double * fs) {

  int ic, jc, kc, index, index1;
  int nhat[3];
  int nlocal[3];
  int status;

  double dn[3];
  double qs[3][3];
  double fes;
  fe_lc_param_t param;
  fe_lc_param_t * fep = &param;

  cs_nlocal(cs, nlocal);

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(q);
  assert(fs);
  assert(map);

  assert(fe->id == FE_LC);
  fe_lc_param((fe_lc_t *) fe, fep);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(q, index, qs);

        nhat[Y] = 0;
        nhat[Z] = 0;

	index1 = cs_index(cs, ic+1, jc, kc);
	map_status(map, index1, &status);

	if (status == MAP_COLLOID) {
          nhat[X] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = cs_index(cs, ic-1, jc, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[X] = +1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	nhat[X] = 0;
	nhat[Z] = 0;

	index1 = cs_index(cs, ic, jc+1, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Y] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = cs_index(cs, ic, jc-1, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Y] = +1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	nhat[X] = 0;
	nhat[Y] = 0;

	index1 = cs_index(cs, ic, jc, kc+1);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Z] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = cs_index(cs, ic, jc, kc-1);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Z] = +1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }
	
      }
    }
  }

  return 0;
}

/*
 * Normal anchoring free energy
 * f_s = (1/2) w_1 ( Q_ab - Q^0_ab )^2  with Q^0_ab prefered orientation.
 *
 * Planar anchoring free energy (Fournier and Galatola EPL (2005).
 * f_s = (1/2) w_1 ( Q^tilde_ab - Q^tidle_perp_ab )^2
 *     + (1/2) w_2 ( Q^tidle^2 - S_0^2 )^2
 *
 * so w_2 must be zero for normal anchoring.
 */

/*****************************************************************************
 *
 *  blue_phase_fs
 *
 *  Compute and return surface free energy area density given
 *    outward normal nhat[3]
 *    fluid Q_ab qs
 *    site map status
 *
 *  TODO:  There is a rather ugly depedency on the order parameter
 *         gradient calculation currently in gradient_3d_7pt_solid which 
 *         needs to be refactored.
 *
 *****************************************************************************/

__host__ int blue_phase_fs(fe_lc_param_t * feparam, const double dn[3],
			   double qs[3][3], char status,
			   double * fe) {

  int ia, ib;
  double w1, w2;
  double q0[3][3];
  double qtilde;
  double amplitude;
  double f1, f2, s0;
  KRONECKER_DELTA_CHAR(d);

  int colloids_q_boundary(fe_lc_param_t * param,
			const double nhat[3], double qs[3][3],
			  double q0[3][3], int map_status);

  assert(status == MAP_BOUNDARY || status == MAP_COLLOID);

  colloids_q_boundary(feparam, dn, qs, q0, status);

  w1 = feparam->w1_coll;
  w2 = feparam->w2_coll;

  if (status == MAP_BOUNDARY) {
    w1 = feparam->w1_wall;
    w2 = feparam->w2_wall;
  }

  fe_lc_amplitude_compute(feparam, &amplitude);
  s0 = 1.5*amplitude;  /* Fournier & Galatola S_0 = (3/2)A */

  /* Free energy density */

  f1 = 0.0;
  f2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      f1 += (qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
      qtilde = qs[ia][ib] + 0.5*amplitude*d[ia][ib];
      f2 += (qtilde*qtilde - s0*s0)*(qtilde*qtilde - s0*s0);
    }
  }

  *fe = 0.5*w1*f1 + 0.5*w2*f2;

  return 0;
}
