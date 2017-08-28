/****************************************************************************
 *
 *  fe_lc_stats.c
 *
 *  Statistics for liquid crystal free energy including surface
 *  free energy terms.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>

#include "fe_lc_stats.h"
#include "util.h"


static int fe_lc_wall(cs_t * cs, wall_t * wall, fe_lc_t * fe, double * fs);
static int fe_lc_colloid(fe_lc_t * fe, colloids_info_t * cinfo,
			 map_t * map, double * fs);

__host__ int blue_phase_fs(fe_lc_param_t * feparam, const double dn[3],
			   double qs[3][3], char status, double * fs);

static int fe_lc_bulk_grad(fe_lc_t * fe, map_t * map, double * fbg);

__host__ int blue_phase_fbg(fe_lc_param_t * feparam, double q[3][3], 
			   double dq[3][3][3], double * fbg);


#define NFE_STAT 5

/****************************************************************************
 *
 *  fe_lc_stats_info
 *
 ****************************************************************************/

int fe_lc_stats_info(pe_t * pe, cs_t * cs, fe_lc_t * fe,
		     wall_t * wall, map_t * map,
		     colloids_info_t * cinfo, int step) {

  int ic, jc, kc, index;
  int nlocal[3];
  int status;
  int ncolloid;

  double fed;
  double fe_local[NFE_STAT];
  double fe_total[NFE_STAT];

  assert(pe);
  assert(cs);
  assert(fe);
  assert(map);

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

	fe_lc_fed(fe, index, &fed);
	fe_local[0] += fed;

	if (status == MAP_FLUID) {
	    fe_local[1] += fed;
	    fe_local[2] += 1.0;
	}
      }
    }
  }

  /* I M P O R T A N T */
  /* The regression test output is sensitive to the form of
   * this output. If you change this, you need to update
   * all the test logs when satisfied on correctness. */

  if (wall_present(wall)) {

    fe_lc_wall(cs, wall, fe, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NFE_STAT, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    pe_info(pe, "\nFree energies - timestep f v f/v f_s1 fs_s2 \n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	    step, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	    fe_total[3], fe_total[4]);
  }
  else if (ncolloid > 0) {

    fe_lc_colloid(fe, cinfo, map, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NFE_STAT, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    pe_info(pe, "\nFree energies - timestep f v f/v f_s a f_s/a\n");

    if (fe_total[4] > 0.0) {
      /* Area > 0 means the free energy is available */
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	      step, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3], fe_total[4], fe_total[3]/fe_total[4]);
    }
    else {
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e\n",
	      step, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3]);
    }
  }
  else {

    fe_lc_bulk_grad(fe, map, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NFE_STAT, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    pe_info(pe, "\nFree energies - timestep f v f/v f_bulk/v f_grad/v redshift\n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n", step, 
	fe_total[1], fe_total[2], fe_total[1]/fe_total[2], fe_total[3]/fe_total[2], fe_total[4]/fe_total[2], fe->param->redshift);
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_wall
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

static int fe_lc_wall(cs_t * cs, wall_t * wall, fe_lc_t * fe, double * fs) {

  int iswall[3];

  assert(cs);
  assert(wall);
  assert(fe);

  int fe_lc_wallx(cs_t * cs, fe_lc_t * fe, double * fs);
  int fe_lc_wally(cs_t * cs, fe_lc_t * fe, double * fs);
  int fe_lc_wallz(cs_t * cs, fe_lc_t * fe, double * fs);

  wall_present_dim(wall, iswall);

  if (iswall[X]) fe_lc_wallx(cs, fe, fs);
  if (iswall[Y]) fe_lc_wally(cs, fe, fs);
  if (iswall[Z]) fe_lc_wallz(cs, fe, fs);

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_wallx
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int fe_lc_wallx(cs_t * cs, fe_lc_t * fe, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double qs[3][3];
  double fes;

  assert(cs);
  assert(fe);
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
	field_tensor(fe->q, index, qs);
	blue_phase_fs(fe->param, dn, qs, MAP_BOUNDARY, &fes);
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
	field_tensor(fe->q, index, qs);
	blue_phase_fs(fe->param, dn, qs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_wally
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int fe_lc_wally(cs_t * cs, fe_lc_t * fe, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double qs[3][3];
  double fes;

  assert(cs);
  assert(fe);
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
	field_tensor(fe->q, index, qs);
	blue_phase_fs(fe->param, dn, qs, MAP_BOUNDARY, &fes);
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
	field_tensor(fe->q, index, qs);
	blue_phase_fs(fe->param, dn, qs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_wallz
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int fe_lc_wallz(cs_t * cs, fe_lc_t * fe, double * fs) {

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
  assert(fe);
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
	field_tensor(fe->q, index, qs);
	blue_phase_fs(fe->param, dn, qs, MAP_BOUNDARY, &fes);
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
	field_tensor(fe->q, index, qs);
	blue_phase_fs(fe->param, dn, qs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_lc_colloid
 *
 *  Return f_s for the local colloid surface (and an area).
 *
 *     fs[0] = free energy (integrated over area)
 *     fs[1] = discrete surface area
 *
 *****************************************************************************/

static int fe_lc_colloid(fe_lc_t * fe, colloids_info_t * cinfo,
			 map_t * map, double * fs) {

  int ic, jc, kc, index, index1;
  int nhat[3];
  int nlocal[3];
  int status;

  double dn[3];
  double qs[3][3];
  double fes;

  coords_nlocal(nlocal);

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(fe);
  assert(map);
  assert(fs);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(fe->q, index, qs);

        nhat[Y] = 0;
        nhat[Z] = 0;

	index1 = coords_index(ic+1, jc, kc);
	map_status(map, index1, &status);

	if (status == MAP_COLLOID) {
          nhat[X] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = coords_index(ic-1, jc, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[X] = +1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	nhat[X] = 0;
	nhat[Z] = 0;

	index1 = coords_index(ic, jc+1, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Y] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = coords_index(ic, jc-1, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Y] = +1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	nhat[X] = 0;
	nhat[Y] = 0;

	index1 = coords_index(ic, jc, kc+1);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Z] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = coords_index(ic, jc, kc-1);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Z] = +1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
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
 *         needs to be refactored. That is: colloids_q_boundary().
 *
 *****************************************************************************/

__host__ int blue_phase_fs(fe_lc_param_t * feparam, const double dn[3],
			   double qs[3][3], char status,
			   double * fs) {

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

  *fs = 0.5*w1*f1 + 0.5*w2*f2;

  return 0;
}
/*****************************************************************************
 *
 *  fe_lc_bulk_grad
 *
 *****************************************************************************/

static int fe_lc_bulk_grad(fe_lc_t * fe,  map_t * map, double * fbg) {

  int ic, jc, kc, index;
  int nlocal[3];
  int status;

  double febg[2];
  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  coords_nlocal(nlocal);

  assert(fe);
  assert(map);
  assert(fbg);

  fbg[0] = 0.0;
  fbg[1] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(fe->q, index, q);
	field_grad_tensor_grad(fe->dq, index, dq);
	field_grad_tensor_delsq(fe->dq, index, dsq);
	fe_lc_compute_h(fe, fe->param->gamma, q, dq, dsq, h);

	blue_phase_fbg(fe->param, q, dq, febg);
	fbg[0] += febg[0];
	fbg[1] += febg[1];
	
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_fbg
 *
 *  This computes statistics for the total, bulk and gradient free energy.
 *
 *****************************************************************************/

__host__ int blue_phase_fbg(fe_lc_param_t * feparam, double q[3][3],
                           double dq[3][3][3], double * febg) {

  int ia, ib, ic, id;
  double q0, redshift, rredshift, a0, gamma, kappa0, kappa1;
  double q2, q3, dq0, dq1, sum;
  const double r3 = 1.0/3.0;
  LEVI_CIVITA_CHAR(e);

  q0 = feparam->q0;
  kappa0 = feparam->kappa0;
  kappa1 = feparam->kappa1;

  // Use current redshift.
  redshift = feparam->redshift;
  rredshift = feparam->rredshift;

  q0 *= rredshift;
  kappa0 *= redshift*redshift;
  kappa1 *= redshift*redshift;

  a0 = feparam->a0;
  gamma = feparam->gamma;

  q2 = 0.0;

  // Q_ab^2

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  // Q_ab Q_bd Q_da

  q3 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	// We use here the fact that q[ic][ia] = q[ia][ic]
	q3 += q[ia][ib]*q[ib][ic]*q[ia][ic];
      }
    }
  }

  // (d_b Q_ab)^2

  dq0 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    sum = 0.0;
    for (ib = 0; ib < 3; ib++) {
      sum += dq[ib][ia][ib];
    }
    dq0 += sum*sum;
  }

  // (e_agd d_g Q_db + 2q_0 Q_ab)^2

  dq1 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (id = 0; id < 3; id++) {
	for (ic = 0; ic < 3; ic++) {
	  sum += e[ia][id][ic]*dq[id][ic][ib];
	}
      }
      sum += 2.0*q0*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  // Contribution bulk
  febg[0] = 0.5*a0*(1.0 - r3*gamma)*q2;
  febg[0] += -r3*a0*gamma*q3;
  febg[0] += 0.25*a0*gamma*q2*q2;

  // Contribution gradient kapp0 and kappa1
  febg[1] = 0.5*kappa0*dq0;
  febg[1] += 0.5*kappa1*dq1;

  return 0;
}

