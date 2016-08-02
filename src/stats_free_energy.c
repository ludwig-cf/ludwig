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
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2016 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "control.h"
#include "wall.h"
#include "free_energy.h"
#include "blue_phase.h"
#include "util.h"
#include "stats_free_energy.h"

static int stats_free_energy_wall(fe_t * fe, field_t * q, double * fs);
static int stats_free_energy_colloid(fe_t * fe, colloids_info_t * cinfo,
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

int stats_free_energy_density(fe_t * fe, field_t * q, map_t * map,
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
  physics_t * phys = NULL;

  assert(map);

  if (fe == NULL) return 0;

  coords_nlocal(nlocal);
  colloids_info_ntotal(cinfo, &ncolloid);

  fe_local[0] = 0.0; /* Total free energy (fluid all sites) */
  fe_local[1] = 0.0; /* Fluid only free energy */
  fe_local[2] = 0.0; /* Volume of fluid */
  fe_local[3] = 0.0; /* surface free energy */
  fe_local[4] = 0.0; /* other wall free energy (walls only) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
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

  if (wall_present()) {

    if (q) stats_free_energy_wall(fe, q, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    info("\nFree energies - timestep f v f/v f_s1 fs_s2 \n");
    info("[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	 ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	 fe_total[3], fe_total[4]);
  }
  else if (ncolloid > 0) {

    if (q) stats_free_energy_colloid(fe, cinfo, q, map, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    info("\nFree energies - timestep f v f/v f_s a f_s/a\n");

    if (fe_total[4] > 0.0) {
      /* Area > 0 means the free energy is available */
      info("[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	   ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	   fe_total[3], fe_total[4], fe_total[3]/fe_total[4]);
    }
    else {
      info("[fe] %14d %17.10e %17.10e %17.10e %17.10e\n",
	   ntstep, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	   fe_total[3]);
    }
  }
  else {
    MPI_Reduce(fe_local, fe_total, 3, MPI_DOUBLE, MPI_SUM, 0, pe_comm());
    rv = 1.0/(L(X)*L(Y)*L(Z));

    info("\nFree energy density - timestep total fluid\n");
    info("[fed] %14d %17.10e %17.10e\n", ntstep, rv*fe_total[0],
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
 *****************************************************************************/

static int stats_free_energy_wall(fe_t * fe, field_t * q, double * fs) {

  assert(fe);
  assert(q);

  int stats_free_energy_wallx(fe_lc_param_t * fe, field_t * q, double * fs);
  int stats_free_energy_wally(fe_lc_param_t * fe, field_t * q, double * fs);
  int stats_free_energy_wallz(fe_lc_param_t * fe, field_t * q, double * fs);

  if (fe->id == FE_LC) {
    /* Slightly inelegant: aka SHIT */
    fe_lc_param_t tmp;
    fe_lc_param_t * param = &tmp;

    fe_lc_param((fe_lc_t *) fe, param);

    if (wall_at_edge(X)) stats_free_energy_wallx(param, q, fs);
    if (wall_at_edge(Y)) stats_free_energy_wally(param, q, fs);
    if (wall_at_edge(Z)) stats_free_energy_wallz(param, q, fs);
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

int stats_free_energy_wallx(fe_lc_param_t * fep, field_t * q, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];

  double dn[3];
  double qs[3][3];
  double fes;

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(fep);
  assert(q);
  assert(fs);

  coords_nlocal(nlocal);

  dn[Y] = 0.0;
  dn[Z] = 0.0;

  if (cart_coords(X) == 0) {

    ic = 1;
    dn[X] = +1.0;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    ic = nlocal[X];
    dn[X] = -1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
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

int stats_free_energy_wally(fe_lc_param_t * fep, field_t * q, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];

  double dn[3];
  double qs[3][3];
  double fes;

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(fep);
  assert(q);
  assert(fs);

  coords_nlocal(nlocal);
  dn[X] = 0.0;
  dn[Z] = 0.0;

  if (cart_coords(Y) == 0) {

    jc = 1;
    dn[Y] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (cart_coords(Y) == cart_size(Y) - 1) {

    jc = nlocal[Y];
    dn[Y] = -1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
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

int stats_free_energy_wallz(fe_lc_param_t * fep, field_t * q, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];

  double dn[3];
  double qs[3][3];
  double fes;

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(fep);
  assert(q);
  assert(fs);

  coords_nlocal(nlocal);
  dn[X] = 0.0;
  dn[Y] = 0.0;

  if (cart_coords(Z) == 0) {

    kc = 1;
    dn[Z] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

        index = coords_index(ic, jc, kc);
	field_tensor(q, index, qs);
	blue_phase_fs(fep, dn, qs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    kc = nlocal[Z];
    dn[Z] = -1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

        index = coords_index(ic, jc, kc);
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
 *****************************************************************************/

static int stats_free_energy_colloid(fe_t * fe,
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

  coords_nlocal(nlocal);

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(q);
  assert(fs);
  assert(map);
  /* SHIT NOT LC_DROPLET */
  fe_lc_param((fe_lc_t *) fe, fep);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(q, index, qs);

        nhat[Y] = 0;
        nhat[Z] = 0;

	index1 = coords_index(ic+1, jc, kc);
	map_status(map, index1, &status);

	if (status == MAP_COLLOID) {
          nhat[X] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = coords_index(ic-1, jc, kc);
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

	index1 = coords_index(ic, jc+1, kc);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Y] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = coords_index(ic, jc-1, kc);
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

	index1 = coords_index(ic, jc, kc+1);
	map_status(map, index1, &status);

        if (status == MAP_COLLOID) {
          nhat[Z] = -1;
          colloids_q_boundary_normal(cinfo, index, nhat, dn);
	  blue_phase_fs(fep, dn, qs, MAP_COLLOID, &fes);
	  fs[0] += fes;
	  fs[1] += 1.0;
        }

	index1 = coords_index(ic, jc, kc-1);
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

  /* SHIT An ugly cross cutting concern currently in gradient_3d_7pt_solid */
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

/*****************************************************************************
 *
 *  blue_phase_stats
 *
 *  This computes statistics for the free energy, and for the
 *  thermodynamic stress, if required. Remember that all the
 *  components of the stress have an additional minus sign cf.
 *  what may be expected.
 *
 *****************************************************************************/
#ifdef OLD_SHIT
int blue_phase_stats(field_t * qf, field_grad_t * dqf, map_t * map,
		     int nstep) {

  int ic, jc, kc, index;
  int ia, ib, id, ig;
  int nlocal[3];
  int status;

  double q0, redshift, rredshift, a0, gamma, kappa0, kappa1;
  double q[3][3], dq[3][3][3], dsq[3][3], h[3][3], sth[3][3];

  double q2, q3, dq0, dq1, sum;

  double elocal[14], etotal[14];        /* Free energy contributions etc */
  double rv;

  FILE * fp_output;

  assert(qf);
  assert(dqf);
  assert(map);

  coords_nlocal(nlocal);
  rv = 1.0/(L(X)*L(Y)*L(Z));

  q0 = blue_phase_q0();
  kappa0 = blue_phase_kappa0();
  kappa1 = blue_phase_kappa1();

  /* Use current redshift. */
  redshift = blue_phase_redshift();
  rredshift = blue_phase_rredshift();

  q0 *= rredshift;
  kappa0 *= redshift*redshift;
  kappa1 *= redshift*redshift;

  a0 = blue_phase_a0();
  gamma = blue_phase_gamma();

  for (ia = 0; ia < 14; ia++) {
    elocal[ia] = 0.0;
  }

  /* Accumulate the sums (all fluid) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(qf, index, q);
	field_grad_tensor_grad(dqf, index, dq);
	field_grad_tensor_delsq(dqf, index, dsq);

	//we are doing this on the host
	blue_phase_set_kernel_constants();
	void* pcon=NULL;
	blue_phase_host_constant_ptr(&pcon);

	blue_phase_compute_h(q, dq, dsq, h, (bluePhaseKernelConstants_t*) pcon);
	blue_phase_compute_stress(q, dq, h, sth, (bluePhaseKernelConstants_t*) pcon);

	q2 = 0.0;

	/* Q_ab^2 */

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    q2 += q[ia][ib]*q[ia][ib];
	  }
	}

	/* Q_ab Q_bd Q_da */

	q3 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    for (id = 0; id < 3; id++) {
	      /* We use here the fact that q[id][ia] = q[ia][id] */
	      q3 += q[ia][ib]*q[ib][id]*q[ia][id];
	    }
	  }
	}

	/* (d_b Q_ab)^2 */

	dq0 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  sum = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    sum += dq[ib][ia][ib];
	  }
	  dq0 += sum*sum;
	}

	/* (e_agd d_g Q_db + 2q_0 Q_ab)^2 */

	dq1 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    sum = 0.0;
	    for (ig = 0; ig < 3; ig++) {
	      for (id = 0; id < 3; id++) {
		sum += e_[ia][ig][id]*dq[ig][id][ib];
	      }
	    }
	    sum += 2.0*q0*q[ia][ib];
	    dq1 += sum*sum;
	  }
	}

	/* Contributions bulk */

	elocal[0] += 0.5*a0*(1.0 - r3_*gamma)*q2;
	elocal[1] += -r3_*a0*gamma*q3;
	elocal[2] += 0.25*a0*gamma*q2*q2;

	/* Contributions gradient kapp0 and kappa1 */

	elocal[3] += 0.5*kappa0*dq0;
	elocal[4] += 0.5*kappa1*dq1;

	/* Nine compoenents of stress */

	elocal[5]  += sth[X][X];
	elocal[6]  += sth[X][Y];
	elocal[7]  += sth[X][Z];
	elocal[8]  += sth[Y][X];
	elocal[9]  += sth[Y][Y];
	elocal[10] += sth[Y][Z];
	elocal[11] += sth[Z][X];
	elocal[12] += sth[Z][Y];
	elocal[13] += sth[Z][Z];
      }
    }
  }

  /* Results to standard out */

  MPI_Reduce(elocal, etotal, 14, MPI_DOUBLE, MPI_SUM, 0, cart_comm());

  for (ia = 0; ia < 14; ia++) {
    etotal[ia] *= rv;
  }

   if (output_to_file_ == 1) {

     /* Note that the reduction is to rank 0 in the Cartesian communicator */
     if (cart_rank() == 0) {

       fp_output = fopen("free_energy.dat", "a");
       if (fp_output == NULL) fatal("fopen(free_energy.dat) failed\n");

       /* timestep, total FE, gradient FE, redhsift */
       fprintf(fp_output, "%d %12.6e %12.6e %12.6e ", nstep, 
	       etotal[0] + etotal[1] + etotal[2] + etotal[3] + etotal[4],
	       etotal[3] + etotal[4], redshift);
       /* Stress xx, xy, xz, ... */
       fprintf(fp_output, "%12.6e %12.6e %12.6e ",
	       etotal[5], etotal[6], etotal[7]);
       fprintf(fp_output, "%12.6e %12.6e %12.6e ",
	       etotal[8], etotal[9], etotal[10]);
       fprintf(fp_output, "%12.6e %12.6e %12.6e\n",
	       etotal[11], etotal[12], etotal[13]);
       
       fclose(fp_output);
     }
   }
   else {

     /* To standard output we send
      * 1. three terms in the bulk free energy
      * 2. two terms in distortion + current redshift
      * 3. total bulk, total distortion, and the grand total */

     info("\n");
     info("[fed1]%14d %14.7e %14.7e %14.7e\n", nstep, etotal[0],
	  etotal[1], etotal[2]);
     info("[fed2]%14d %14.7e %14.7e %14.7e\n", nstep, etotal[3], etotal[4],
	  redshift);
     info("[fed3]%14d %14.7e %14.7e %14.7e\n", nstep,
	  etotal[0] + etotal[1] + etotal[2],
	  etotal[3] + etotal[4],
	  etotal[0] + etotal[1] + etotal[2] + etotal[3] + etotal[4]);
     assert(0); /* SHIT NO TEST */
   }

  return 0;
}

#endif
