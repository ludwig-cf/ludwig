/*****************************************************************************
 *
 *  cmd.c
 *
 *  Colloid Molecular Dynamics
 *
 *  This file contains routines for the initialisation of random
 *  particle configurations at various volume fractions via a
 *  simple "growing" algorithm plus molecular dynamics (a la
 *  Lubachevsky & Stillinger). This could be a serparate main
 *  program linked against the rest of the code.
 *
 *  PARAMETERS
 *
 *  The Lubachevsky Stillinger algorithm seems to be a bit of an
 *  art form, in that getting the right parameters is important,
 *  particularly at high volume fraction, if one wants to get a
 *  solution in a reasonable number of time steps.
 *
 *  So, this code proceeds by using a soft sphere potential with
 *  parameters POT_ALPHA and POT_BETA. The lubrication forces
 *  between the particles should be switched off. The range of
 *  the potential is set to R_SSPH.
 *
 *  POT_ALPHA   0.0004
 *  POT_BETA    -2.0
 *  R_SSPH      0.7
 *
 *  The particles should not be too close in the resultant initial
 *  conditions or else the resulting large forces will cause an
 *  immediate explosion when the LB starts. So, a minimum separation
 *  should be requested.
 *
 *  HMIN_REQUEST 0.25    Should be safe for most purposes
 *
 *  In the MD, the particles should not expand when in close proximity.
 *  There is therefore a minimum separation below which growth is
 *  stopped.
 *
 *  HMIN_GROWTH  0.25
 *
 *  The maximum sane volume fraction is
 *
 *  VF_MAX       0.55
 *
 *  The maximum number of MD steps is
 *
 *  NMAX_ITERATIONS 2000
 *
 *  If a very high volume fraction is required, VF_MAX should be
 *  increased along with NMAX_ITERATIONS. However, the above
 *  parameters should be reasonably robust to volume fraction 0.5.
 *
 *  The lubrication corrections are switched off for the duration
 *  of the MD, but restored to whatever the user input requested
 *  before any LB steps are performed.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "pe.h"
#include "coords.h"
#include "cartesian.h"

#include "globals.h"
#include "utilities.h"
#include "colloids.h"
#include "ccomms.h"
#include "cells.h"
#include "cio.h"

#define  POT_ALPHA       0.0002 /* p1 in soft sphere */
#define  POT_BETA        -12.0   /* p1 in soft sphere */
#define  R_SSPH          0.3    /* soft sphere repulsion for MD */
#define  R_MAXINFLATE    0.0025    /* Maximum growth rate */
#define  HMIN_REQUEST    0.01   /* Should be safe */
#define  HMIN_GROWTH     0.01
#define  VF_MAX          0.55
#define  NMAX_ITERATIONS 200000
#define  NMAX_BROWNIAN   100000

static void CMD_do_md(void);
static void CMD_reset_particles(const double);
static int  CMD_update_colloids(double hmin);
static void CMD_brownian_dynamics_step(void);
static void CMD_brownian_set_random(void);

/*****************************************************************************
 *
 *  CMD_init_volume_fraction
 *
 *  This routine will eventually take
 *       nradius - the number of different radii required
 *       radius  - a list of those radii
 *       flag    - equal number / equal area / equal volume
 *                 flag for nradius > 1
 *
 *****************************************************************************/

void CMD_init_volume_fraction(int nradius, int flag) {

  double v_part;
  double v_system;
  double rtarget;
  double vf_eff;
  int   n_global;
  int   n;

  double r_clus = Global_Colloid.r_clus;
  double r_lu_n = Global_Colloid.r_lu_n;
  double r_ssph = Global_Colloid.r_ssph;

  FVector r0;
  FVector zero;
  int     index0;

  zero = UTIL_fvector_zero();

  /* Look at the local volume and see how many particles
   * are required to give the requested solid volume
   * fraction. */

  info("\nCMD_init_volume_fraction\n");

  if (nradius > 1) {
    info("Only monodisperse at the moment... taking radius = %f\n",
	 Global_Colloid.ah);
  }

  rtarget = Global_Colloid.ah;

  v_system = L(X)*L(Y)*L(Z);
  v_part   = (4.0/3.0)*PI*rtarget*rtarget*rtarget;
  n_global = Global_Colloid.vf*(v_system / v_part);

  /* Report the requested volume fraction and what we actually
   * will get. */

  Global_Colloid.N_colloid = n_global;

  vf_eff = v_part*n_global / v_system;

  info("The requested solid volume fraction is      %f\n", Global_Colloid.vf);
  info("Particle (hydrodynamic) radius requested is %f\n", rtarget);
  info("Number of particles required is             %d\n", n_global);
  info("Actual volume fraction is then              %f\n", vf_eff);

  /* Look at the lubrication cutoff distance and decide a safe
   * initial separation. This will give rise to an effective
   * solid fraction which should not be too high. */

  rtarget = Global_Colloid.ah + 0.5*HMIN_REQUEST;
  v_part  = (4.0/3.0)*PI*rtarget*rtarget*rtarget;
  vf_eff  = n_global*v_part / v_system;

  info("The initial separation minumum requested is %f\n", r_lu_n);
  info("This gives effective solid fraction         %f\n", vf_eff);

  if (vf_eff > VF_MAX) {
    fatal("Requested effective volume fraction too high\n");
  }

  /* For a decomposition-independent start, we generate positions
   * for all particles, but only add those which are actually
   * local. The hydrodynamic radius, which determines interactions,
   * is set to zero. */

  rtarget = Global_Colloid.a0;

  for (n = 1; n <= n_global; n++) {
    r0.x = Lmin(X) + ran_serial_uniform()*L(X);
    r0.y = Lmin(Y) + ran_serial_uniform()*L(Y);
    r0.z = Lmin(Z) + ran_serial_uniform()*L(Z);
    COLL_add_colloid_no_halo(n, rtarget, 0.0, r0, zero, zero);
  }

  /* Set, temporarily, the divergence length for the potential
   * to be r_lu_n, that is, the particles should not be closer
   * than (separation) R_SAFE*r_lu_n in the initial conditions.
   * It is possible to set R_SSPH large enough that the cell
   * lists might miss some interactions, or not get truncated
   * exactly to zero . But we don't care about energetics here.
   * Further, the prefactor for the soft sphere potential
   * could take account of the mass of the particle, which always
   * depends on a0. */

  Global_Colloid.r_clus = 0.0;
  Global_Colloid.r_ssph = R_SSPH;
  Global_Colloid.r_lu_n = 0.0;

  /* Iterate until an acceptable solution is found */

  CMD_do_md();
  CMD_reset_particles(0.0);
  CMD_do_more_md();
  CMD_reset_particles(1.0);

  /* Restore the original user parameters before saving the
   * initial particle data. Make sure the cell list is up-to-date,
   * and set the particle state clean before saving the initial
   * positions. */

  Global_Colloid.r_clus = r_clus;
  Global_Colloid.r_ssph = r_ssph;
  Global_Colloid.r_lu_n = r_lu_n;

  CELL_update_cell_lists();
  /* CMD_reset_particles();*/
  CIO_write_state("config.cds000000");

#ifdef _MPI_
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  return;
}


/*****************************************************************************
 *
 *  CMD_do_md
 *
 *  Molecular dynamics iteration.
 *
 *****************************************************************************/

void CMD_do_md() {

  int too_close, too_small;
  int n = 0;
  double hmin = 0.0;
  double delta_a;
  double delta = 0.0;

  void COLL_zero_forces(void);
  void COLL_set_colloid_gravity(void);

  do {

    if (n++ > NMAX_ITERATIONS) {
      fatal("Hit NMAX_ITERATIONS in molecular dynamcics\n");
    }


    CELL_update_cell_lists();
    CCOM_halo_particles();
    CCOM_sort_halo_lists();

    COLL_zero_forces();
    hmin = COLL_interactions();
    COLL_set_colloid_gravity(); /* Zero force on halo particles! */

    CCOM_halo_sum(CHALO_TYPE1);

    /* Step colloid position/velocity and inflate the radius by
     * an amount delta_a. If hmin is large, then the particles
     * can be inflated so that the separation is reduced to the
     * point where interactions start (R_SSPH). If small, the
     * inflation is restricted to a fraction of hmin. In addition,
     * particle growth is stopped at separations lower than
     * HMIN_GROWTH to allow time for the particles to move apart. */

#ifdef _MPI_
    /* Everyone should agree on hmin */
    {
      double hmin_global = 0.0;
      MPI_Allreduce(&hmin, &hmin_global, 1, MPI_DOUBLE, MPI_MIN, cart_comm());
      hmin = hmin_global;
    }
#endif

    delta_a = dmax(R_MAXINFLATE*hmin, 0.5*(hmin - R_SSPH));
    if (hmin < HMIN_GROWTH) delta_a = 0.0;
    delta += delta_a;

    /* Stopping criterion */

    too_small = CMD_update_colloids(delta_a);
    too_close = (hmin < HMIN_REQUEST);

#ifdef _MPI_
    /* Everyone must agree on the stopping criterion */
    {
      int    flag_global = 0;
      MPI_Allreduce(&too_small, &flag_global, 1, MPI_INT, MPI_MAX,
		    cart_comm());
      too_small = flag_global;
    }
#endif

    if (n % 1000 == 0) {
      info("CMD iteration %d: minimum separation was %f (request %f)\n", n,
	   hmin, HMIN_REQUEST);
      info("CMD iteration %d: inflation total %f\n", n, delta);
      CMD_test_particle_energy(n);
    }

  } while (too_close || too_small);

  info("FINAL ITERATION\n");
  info("CMD iteration %d: minimum separation was %f (request %f)\n", n,
       hmin, HMIN_REQUEST);
  info("CMD iteration %d: inflation total %f\n", n, delta);

  return;
}


/*****************************************************************************
 *
 *  CMD_update_colloids
 *
 *  Update the colloid positions. This is a [very] mimimal
 *  dynamics!
 *
 *  The colloid radius is also inflated by an amount delta_a
 *  to a maximum of the target radius (currently G.ah). The
 *  return value is only zero if all the particles have
 *  reached the target value.
 *
 *****************************************************************************/

int CMD_update_colloids(double delta_a) {

  int     ic, jc, kc;
  IVector ncell = Global_Colloid.Ncell;
  int     too_small = 0;
  Colloid * p_colloid;

  double rmass;

  rmass = 1.0/((4.0/3.0)*PI*Global_Colloid.ah);

  for (ic = 0; ic <= ncell.x + 1; ic++) {
    for (jc = 0; jc <= ncell.y + 1; jc++) {
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->v.x += rmass*p_colloid->force.x;
	  p_colloid->v.y += rmass*p_colloid->force.y;
	  p_colloid->v.z += rmass*p_colloid->force.z;

	  p_colloid->r.x += p_colloid->v.x;
	  p_colloid->r.y += p_colloid->v.y;
	  p_colloid->r.z += p_colloid->v.z;

	  p_colloid->ah  += delta_a;

	  if (p_colloid->ah >= Global_Colloid.ah) {
	    p_colloid->ah = Global_Colloid.ah;
	  }
	  else {
	    too_small = 1;
	  }

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return too_small;
}


/*****************************************************************************
 *
 *  CMD_reset_particles
 *
 *  At the end of the Molecular Dynamics stage, we need to
 *  re-initialise the colloid velocities.
 *
 *  Linear part: taken from Gaussian at current temperature.
 *  Rotational part: zero.
 *
 *  Note that the initalisation of the velocity is not decompsoition
 *  independent.
 *
 *****************************************************************************/

void CMD_reset_particles(const double factor) {

  int     ic, jc, kc;
  IVector ncell = Global_Colloid.Ncell;
  Colloid * p_colloid;

  double xt = 0.0, yt = 0.0, zt = 0.0;
  double vmin = 0.0, vmax = 0.0;
  int nt = 0;

  const double cs = 1.0/sqrt(3.0); /* c_s */

  extern double normalise;

  for (ic = 1; ic <= ncell.x; ic++) {
    for (jc = 1; jc <= ncell.y; jc++) {
      for (kc = 1; kc <= ncell.z; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
	  p_colloid->v.x   = factor*cs*normalise*ran_parallel_gaussian();
	  p_colloid->v.y   = factor*cs*normalise*ran_parallel_gaussian();
	  p_colloid->v.z   = factor*cs*normalise*ran_parallel_gaussian();
	  p_colloid->omega = UTIL_fvector_zero();

	  /* Accumulate the totals */
	  xt += p_colloid->v.x;
	  yt += p_colloid->v.y;
	  zt += p_colloid->v.z;
	  ++nt;

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* We set the total colloid momentum to zero locally
   * (and hence globally) */

  if (nt > 0) {

    double xst = 0.0, yst = 0.0, zst = 0.0;

    xt /= (double) nt;
    yt /= (double) nt;
    zt /= (double) nt;

    for (ic = 1; ic <= ncell.x; ic++) {
      for (jc = 1; jc <= ncell.y; jc++) {
	for (kc = 1; kc <= ncell.z; kc++) {

	  p_colloid = CELL_get_head_of_list(ic, jc, kc);

	  while (p_colloid) {
	    p_colloid->v.x -= xt;
	    p_colloid->v.y -= yt;
	    p_colloid->v.z -= zt;

	    /* Actual velocity stats... */
	    xst += p_colloid->v.x*p_colloid->v.x;
	    yst += p_colloid->v.y*p_colloid->v.y;
	    zst += p_colloid->v.z*p_colloid->v.z;

	    vmin = dmin(p_colloid->v.x, vmin);
	    vmin = dmin(p_colloid->v.y, vmin);
	    vmin = dmin(p_colloid->v.z, vmin);
	    vmax = dmax(p_colloid->v.x, vmax);
	    vmax = dmax(p_colloid->v.x, vmax);
	    vmax = dmax(p_colloid->v.x, vmax);

	    p_colloid = p_colloid->next;
	  }
	}
      }
    }
#ifdef _MPI_
    /* MPI_Reduce(); */
#endif
  }

  CELL_update_cell_lists();
  CCOM_halo_particles();
  CCOM_sort_halo_lists();

  return;
}

/*****************************************************************************
 *
 *  CMD_test_particle_energy
 *
 *****************************************************************************/

CMD_test_particle_energy(const int step) {

  int     ic, jc, kc;
  IVector ncell = Global_Colloid.Ncell;
  Colloid * p_colloid;

  double sum[3], gsum[3];
  double mass;

  mass = (4.0/3.0)*PI*pow(Global_Colloid.ah, 3.0);

  sum[0] = 0.0;
  sum[1] = 0.0;
  sum[2] = 0.0;

  for (ic = 1; ic <= ncell.x; ic++) {
    for (jc = 1; jc <= ncell.y; jc++) {
      for (kc = 1; kc <= ncell.z; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  /* Accumulate the totals */
	  sum[0] += p_colloid->v.x;
	  sum[1] += p_colloid->v.y;
	  sum[2] += p_colloid->v.z;

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

#ifdef _MPI_
  MPI_Reduce(sum, gsum, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  sum[0] = gsum[0];
  sum[1] = gsum[1];
  sum[2] = gsum[2];
#endif

  gsum[0] = sum[0]/Global_Colloid.N_colloid;
  gsum[1] = sum[1]/Global_Colloid.N_colloid;
  gsum[2] = sum[2]/Global_Colloid.N_colloid;

  info("Mean particle velocities: [x, y, z]\n");
  info("[%g, %g, %g]\n", gsum[0], gsum[1], gsum[2]);


  sum[0] = 0.0;
  sum[1] = 0.0;
  sum[2] = 0.0;

  for (ic = 1; ic <= ncell.x; ic++) {
    for (jc = 1; jc <= ncell.y; jc++) {
      for (kc = 1; kc <= ncell.z; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  /* Accumulate the totals */
	  sum[0] += p_colloid->v.x*p_colloid->v.x;
	  sum[1] += p_colloid->v.y*p_colloid->v.y;
	  sum[2] += p_colloid->v.z*p_colloid->v.z;

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

#ifdef _MPI_
  MPI_Reduce(sum, gsum, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  sum[0] = gsum[0];
  sum[1] = gsum[1];
  sum[2] = gsum[2];
#endif

  gsum[0] = sum[0]/Global_Colloid.N_colloid;
  gsum[1] = sum[1]/Global_Colloid.N_colloid;
  gsum[2] = sum[2]/Global_Colloid.N_colloid;

  info("Particle: %d %g %g %g %g\n", step,
       mass*gsum[0], mass*gsum[1], mass*gsum[2],
       mass*(gsum[0] + gsum[1] + gsum[2]));


  return;
}

/*****************************************************************************
 *
 * CMD_do_more_md
 *
 *****************************************************************************/


CMD_do_more_md() {

  int n = 0;
  double hmin;

  void COLL_zero_forces(void);
  void COLL_set_colloid_gravity(void);

  do {

    CMD_brownian_set_random();

    CELL_update_cell_lists();
    CCOM_halo_particles();
    CCOM_sort_halo_lists();

    COLL_zero_forces();
    hmin = COLL_interactions();
    COLL_set_colloid_gravity(); /* Zero force on halo particles! */

    CCOM_halo_sum(CHALO_TYPE1);

    if (n < NMAX_BROWNIAN/10) {
      CMD_update_colloids(0.0);
    }
    else {
      CMD_brownian_dynamics_step();
    }

    if ((n % 1000) == 0) {
      char md_file[64];

#ifdef _MPI_
      /* Get a global hmin */
    {
      double hmin_global = 0.0;
      MPI_Reduce(&hmin, &hmin_global, 1, MPI_DOUBLE, MPI_MIN, 0, cart_comm());
      hmin = hmin_global;
    }
#endif

      info("BD iteration %d: minimum separation was %f\n", n, hmin);
      CMD_test_particle_energy(n);

      sprintf(md_file, "%s%6.6d", "config.bd", n/1000);
      CIO_write_state(md_file);

    }


  } while (n++ < NMAX_BROWNIAN);

  CELL_update_cell_lists();
  CCOM_halo_particles();
  CCOM_sort_halo_lists();

  return;
}

/*****************************************************************************
 *
 *  CMD_brownian_dynamics_step
 *
 *  This updates the position an velocities of all particles.
 *  The algorithm is that of Ermak as described by Allen & Tildesley.
 *
 *  All the particles are assumed to have the same mass.
 *
 *****************************************************************************/

void CMD_brownian_dynamics_step() {

  int     ic, jc, kc;
  IVector ncell = Global_Colloid.Ncell;
  Colloid * p_colloid;

  double ranx, rany, ranz;
  double cx, cy, cz;

  double c0, c1, c2;
  double xi, xidt;
  double sigma_r, sigma_v;
  double c12, c21;
  double rmass, kT;
  double dt;

  extern double normalise;

  dt = 1.0;
  rmass = 1.0/((4.0/3.0)*PI*pow(Global_Colloid.ah, 3.0));

  /* Friction coefficient is xi, and related quantities */

  xi = 6.0*PI*gbl.eta*Global_Colloid.ah*rmass;
  xidt = xi*dt;

  c0 = exp(-xidt);
  c1 = (1.0 - c0) / xidt;
  c2 = (1.0 - c1) / xidt;

  c1 *= dt;
  c2 *= dt*dt;

  /* Random variances */

  kT = normalise*normalise/3.0;

  sigma_r = sqrt(dt*dt*rmass*kT*(2.0 - (3.0 - 4.0*c0 + c0*c0)/xidt) / xidt);
  sigma_v = sqrt(rmass*kT*(1.0 - c0*c0));

  c12 = dt*rmass*kT*(1.0 - c0)*(1.0 - c0) / (xidt*sigma_r*sigma_v);
  c21 = sqrt(1.0 - c12*c12);

  /* Update each particle */

  for (ic = 0; ic <= ncell.x + 1; ic++) {
    for (jc = 0; jc <= ncell.y + 1; jc++) {
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  ranx = p_colloid->random[0];
	  rany = p_colloid->random[1];
	  ranz = p_colloid->random[2];

	  cx = c1*p_colloid->v.x + rmass*c2*p_colloid->force.x + sigma_r*ranx;
	  cy = c1*p_colloid->v.y + rmass*c2*p_colloid->force.y + sigma_r*rany;
	  cz = c1*p_colloid->v.z + rmass*c2*p_colloid->force.z + sigma_r*ranz;

	  p_colloid->r.x += cx;
	  p_colloid->r.y += cy;
	  p_colloid->r.z += cz;

	  /* Generate correlated random pairs */

	  ranx = (c12*ranx + c21*p_colloid->random[3]);
	  rany = (c12*rany + c21*p_colloid->random[4]);
	  ranz = (c12*ranz + c21*p_colloid->random[5]);

	  cx = c0*p_colloid->v.x + rmass*c1*p_colloid->force.x + sigma_v*ranx;
	  cy = c0*p_colloid->v.y + rmass*c1*p_colloid->force.y + sigma_v*rany;
	  cz = c0*p_colloid->v.z + rmass*c1*p_colloid->force.z + sigma_v*ranz;

	  p_colloid->v.x = cx;
	  p_colloid->v.y = cy;
	  p_colloid->v.z = cz;

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}


void CMD_brownian_set_random() {

  int     ic, jc, kc;
  IVector ncell = Global_Colloid.Ncell;
  Colloid * p_colloid;

  /* Set random numbers for each particle */

  for (ic = 1; ic <= ncell.x; ic++) {
    for (jc = 1; jc <= ncell.y; jc++) {
      for (kc = 1; kc <= ncell.z; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->random[0] = ran_parallel_gaussian();
	  p_colloid->random[1] = ran_parallel_gaussian();
	  p_colloid->random[2] = ran_parallel_gaussian();
	  p_colloid->random[3] = ran_parallel_gaussian();
	  p_colloid->random[4] = ran_parallel_gaussian();
	  p_colloid->random[5] = ran_parallel_gaussian();

	  p_colloid = p_colloid->next;
	}
      }
    }
  }


  return;
}
