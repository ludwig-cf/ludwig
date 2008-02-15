/*****************************************************************************
 *
 *  cmd.c
 *
 *  $Id: cmd.c,v 1.14 2008-02-15 14:35:26 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "pe.h"
#include "ran.h"
#include "coords.h"
#include "ccomms.h"
#include "runtime.h"
#include "physics.h"
#include "potential.h"
#include "cio.h"

#include "colloids.h"
#include "interaction.h"

static void CMD_reset_particles(const double);
static void CMD_test_particle_energy(const int);

static void do_monte_carlo(const int);
static void mc_move(const double);
static void mc_init_random(const int, const double, const double);
static void mc_set_proposed_move(const double);
static void mc_check_state(void);
static void mc_mean_square_displacement(void);
static int  mc_metropolis(double);
static int  mc_init_bcc_lattice(const double, const double, const double);
static double mc_total_energy(void);

static double vf_max_ = 0.60;
static double mc_max_exp_ = 100.0;
static double mc_drmax_;
static double mc_max_ah_;

/*****************************************************************************
 *
 *  CMD_init_volume_fraction
 *
 *  Look at the user input and work out how many particles are
 *  required (and eventually of what type).
 *
 *****************************************************************************/

void CMD_init_volume_fraction() {

  double vf;
  double a0, ah;
  int    n, n_global;
  char   cinit[128];

  info("\nParticle initialisation\n");

  /* At the moment, we manage with a single a0, ah */
  n = RUN_get_double_parameter("colloid_a0", &a0);
  n = RUN_get_double_parameter("colloid_ah", &ah);

  RUN_get_string_parameter("colloid_init", cinit, 128);

  if (strcmp(cinit, "fixed_number_monodisperse") == 0) {
    /* Look for colloid_no */
    n = RUN_get_int_parameter("colloid_no", &n_global);
    if (n == 0) {
      info("fixed_number_monodisperse is set but not colloids_no\n");
      fatal("Please check and try again\n");
    }
    info("[User   ] requested %d particles (monodisperse)\n", n_global);
    info("[User   ] nominal radius a0 %f                 \n", a0);
    info("[User   ] hydrodynamic radius %f               \n", ah);
    mc_init_random(n_global, a0, ah);
  }

  if (strcmp(cinit, "fixed_volume_fraction_monodisperse") == 0) {
    /* Look for volume fraction */
    n = RUN_get_double_parameter("colloid_vf", &vf);
    if (n == 0) {
      info("fixed_volume_fraction is set but not colloids_vf\n");
      fatal("Please check and try again\n");
    }

    info("[User   ] requested volume fraction of %f\n", vf);
    n_global = mc_init_bcc_lattice(vf, a0, ah);
  }

  vf = (4.0/3.0)*PI*ah*ah*ah*n_global / (L(X)*L(Y)*L(Z));
  info("[       ] initialised %d particles\n", n_global);
  info("[       ] actual volume fraction of %f\n", vf);

  set_N_colloid(n_global);

  mc_max_ah_ = ah;

  return;
}

/*****************************************************************************
 *
 *  monte_carlo
 *
 *****************************************************************************/

void monte_carlo() {

  int n = 0;

  RUN_get_int_parameter("colloid_mc_steps", &n);

  /* Set maximum Monte-Carlo move */

  mc_drmax_ = dmin(Lcell(X), Lcell(Y));
  mc_drmax_ = dmin(Lcell(Z), mc_drmax_);
  mc_drmax_ = 0.5*(mc_drmax_ - 2.0*mc_max_ah_ - get_max_potential_range());

  if (n > 0) do_monte_carlo(n);

  mc_check_state();

#ifdef _MPI_
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  CIO_write_state("config.cds.init");

  return;
}

/*****************************************************************************
 *
 *  do_monte_carlo
 *
 *  
 *
 *****************************************************************************/

void do_monte_carlo(const int mc_max_iterations) {

  int    ntrials = 0;
  int    naccept = 0;
  int    accepted;
  double drmax, rate;
  double etotal_old, etotal_new;

  drmax = mc_drmax_;

  cell_update();
  CCOM_halo_particles();

  etotal_old = mc_total_energy();

  do {

    /* A proposed move for the next step must be communicated. */
    ntrials++;
    mc_set_proposed_move(drmax);
    CCOM_halo_particles();

    /* Move particles and calculate proposed new energy */
    mc_move(+1.0);
 
    etotal_new = mc_total_energy();
    accepted = mc_metropolis(etotal_new - etotal_old);

    if (accepted) {
      /* The particles really move, so update the cell list */
      etotal_old = etotal_new;
      naccept++;
      cell_update();
    }
    else {
      /* Just role back to the old positions everywhere */
      mc_move(-1.0);
    }

    /* Adapt. */
    rate = naccept / 10.0;
    if ((ntrials % 10) == 0) {
      info("Monte Carlo step %d rate = %d drmax = %f et = %f %f\n", ntrials,
	   naccept, drmax, etotal_old, etotal_new);
      if (rate < 0.45) drmax = drmax*0.95;
      if (rate > 0.55) drmax = drmax*1.05;
      drmax = dmin(drmax, mc_drmax_);
      naccept = 0;
    }

  } while (ntrials < mc_max_iterations);

  return;
}

/*****************************************************************************
 *
 *  mc_metropolis
 *
 *  The Metropolis part of the Monte Carlo.
 *
 *  The argument delta is the total change in energy for the system.
 *  All processes determine the same acceptance criteria via use of
 *  the serial random number stream.
 *
 *****************************************************************************/

int mc_metropolis(double delta) {

  int accept = 0;

  if (delta <= 0.0) {
    /* Always accept */
    accept = 1;
  }
  else {
    /* Accept with probabilty exp(-delta/kt) */
    delta = -delta / get_kT();
    if (delta <= mc_max_exp_ && exp(delta) > ran_serial_uniform()) accept = 1;
  }

  return accept;
}

/*****************************************************************************
 *
 *  mc_set_proposed_move
 *
 *  Generate a random update for real domain particles.
 *
 *****************************************************************************/

void mc_set_proposed_move(const double drmax) {

  int       ic, jc, kc;
  Colloid * p_colloid;

  double mc_move_prob_ = 0.01;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  if (ran_parallel_uniform() < mc_move_prob_) {
	    p_colloid->random[0] = drmax*(2.0*ran_parallel_uniform() - 1.0);
	    p_colloid->random[1] = drmax*(2.0*ran_parallel_uniform() - 1.0);
	    p_colloid->random[2] = drmax*(2.0*ran_parallel_uniform() - 1.0);
	  }
	  else {
	    p_colloid->random[0] = 0.0;
	    p_colloid->random[1] = 0.0;
	    p_colloid->random[2] = 0.0;
	  }

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  mc_move
 *
 *  Actual Monte Carlo position update (all particles).
 *  For MC moves which have been rejected, sign can be set
 *  to -1 to move the particles back where they came from.
 *
 ****************************************************************************/

void mc_move(const double sign) {

  int       ic, jc, kc;
  Colloid * p_colloid;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  p_colloid->r.x += sign*p_colloid->random[0];
	  p_colloid->r.y += sign*p_colloid->random[1];
	  p_colloid->r.z += sign*p_colloid->random[2];

	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  mc_check_state
 *
 *  Check for overlaps in the current state. Stop if there is a problem,
 *  i.e., there appears to be a hard-sphere overlap.
 *
 *****************************************************************************/

void mc_check_state() {

  double e = 0.0;

  e = mc_total_energy();

  info("\nChecking colloid state...\n");
  info("Total energy / NkT: %g\n", e/(get_N_colloid()*get_kT()));
  mc_mean_square_displacement();

  if (e >= ENERGY_HARD_SPHERE/2.0) {
    info("This appears to include at least one hard sphere overlap.\n");
    info("Please check the volume fraction (for fixed numbers),\n");
    info("or try increasing the number of Monte Carlo steps.\n");
    fatal("Stop.\n");
  }

  info("State appears to have no overlaps\n");

  return;
}

/*****************************************************************************
 *
 *  mc_total_energy
 *
 *****************************************************************************/

double mc_total_energy() {

  Colloid * p_c1;
  Colloid * p_c2;

  int    ic, jc, kc, id, jd, kd, dx, dy, dz;
  double etot = 0.0;
  double h;
  double r12[3];
  double hard_sphere_energy(const double);
  double hard_wall_energy(const FVector, const double);

  FVector r_12;
  FVector COLL_fvector_separation(FVector, FVector);

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_c1 = CELL_get_head_of_list(ic, jc, kc);

	while (p_c1) {

	  etot += hard_wall_energy(p_c1->r, p_c1->ah);

	  for (dx = -1; dx <= +1; dx++) {
	    for (dy = -1; dy <= +1; dy++) {
	      for (dz = -1; dz <= +1; dz++) {

		id = ic + dx;
		jd = jc + dy;
		kd = kc + dz;

		p_c2 = CELL_get_head_of_list(id, jd, kd);

		while (p_c2) {

		  if (p_c1->index < p_c2->index) {

		    r_12 = COLL_fvector_separation(p_c1->r, p_c2->r);

		    h = sqrt(r_12.x*r_12.x + r_12.y*r_12.y + r_12.z*r_12.z);
		    h = h - p_c1->ah - p_c2->ah;
		    
		    etot += hard_sphere_energy(h);
		    etot += soft_sphere_energy(h);
		    etot += yukawa_potential(h + p_c1->ah + p_c2->ah);
		    etot += leonard_jones_energy(h);

		  }
		  
		  /* Next colloid */
		  p_c2 = p_c2->next;
		}

		/* Next search cell */
	      }
	    }
	  }

	  /* Next colloid */
	  p_c1 = p_c1->next;
	}

	/* Next cell */
      }
    }
  }

#ifdef _MPI_
 {
   double elocal = etot;
   MPI_Allreduce(&elocal, &etot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 }
#endif

  return etot;
}

/*****************************************************************************
 *
 *  hard_wall_energy
 *
 *  Return particle-hard-wall 'energy' if there is an overlap.
 *
 *****************************************************************************/

double hard_wall_energy(const FVector r, const double ah) {

  double etot = 0.0;

  if (is_periodic(X) == 0) {
    if ((r.x - ah) < Lmin(X)) etot += ENERGY_HARD_SPHERE;
    if ((r.x + ah) > Lmin(X) + L(X)) etot += ENERGY_HARD_SPHERE;
  }

  if (is_periodic(Y) == 0) {
    if ((r.y - ah) < Lmin(Y)) etot += ENERGY_HARD_SPHERE;
    if ((r.y + ah) > Lmin(Y) + L(Y)) etot += ENERGY_HARD_SPHERE;
  }

  if (is_periodic(Z) == 0) {
    if ((r.z - ah) < Lmin(Z)) etot += ENERGY_HARD_SPHERE;
    if ((r.z + ah) > Lmin(Z) + L(Z)) etot += ENERGY_HARD_SPHERE;
  }

  return etot;
}

/*****************************************************************************
 *
 *  mc_init_random
 *
 *  Initialise a fixed number of particles in random positions.
 *  As this is a simple insertion, the maximum practical
 *  volume fraction is very limited.
 *
 *****************************************************************************/

void mc_init_random(const int npart, const double a0, const double ah) {

  int n;
  FVector r0;
  FVector zero = {0.0, 0.0, 0.0};
  double Lex[3];

  /* If boundaries are present, some of the volume must be excluded */
  Lex[X] = ah*(1.0 - is_periodic(X));
  Lex[Y] = ah*(1.0 - is_periodic(Y));
  Lex[Z] = ah*(1.0 - is_periodic(Z));

  for (n = 1; n <= npart; n++) {
    r0.x = Lmin(X) + Lex[X] + ran_serial_uniform()*(L(X) - 2.0*Lex[X]);
    r0.y = Lmin(Y) + Lex[Y] + ran_serial_uniform()*(L(Y) - 2.0*Lex[Y]);
    r0.z = Lmin(Z) + Lex[Z] + ran_serial_uniform()*(L(Z) - 2.0*Lex[Z]);
    COLL_add_colloid_no_halo(n, a0, ah, r0, zero, zero);
  }

  return;
}

/*****************************************************************************
 *
 *  mc_init_bcc_lattice
 *
 *  Initialise the given volume fraction. The positions of the
 *  particles are those of a bcc lattice (not extending through
 *  the periodic, or boundary, region).
 *
 *  Returns the actual number of particles initialised.
 *
 *****************************************************************************/

int mc_init_bcc_lattice(double vf, double a0, double ah) {

  int     n_request, n_max;
  int     nx, ny, nz, ncx, ncy, ncz;
  int     index = 0;
  double  dx, dy, dz, vp;
  FVector r0;
  FVector zero = {0.0, 0.0, 0.0};

  /* How many particles to get this volume fraction? */

  n_request = (ceil) (L(X)*L(Y)*L(Z)*vf / ((4.0/3.0)*PI*ah*ah*ah));

  /* How many will fit? */

  ncx = L(X) / (4.0*(ah+0.0)/sqrt(3.0));
  ncy = L(Y) / (4.0*(ah+0.0)/sqrt(3.0));
  ncz = L(Z) / (4.0*(ah+0.0)/sqrt(3.0));

  dx = L(X)/ncx;
  dy = L(Y)/ncy;
  dz = L(Z)/ncz;

  n_max = ncx*ncy*ncz;
  vp = (double) n_request / (double) n_max;

  /* Allocate and initialise. */

  for (nx = 1; nx <= ncx; nx++) {
    for (ny = 1; ny <= ncy; ny++) {
      for (nz = 1; nz <= ncz; nz++) {

	r0.x = Lmin(X) + dx*(nx-0.5);
	r0.y = Lmin(Y) + dy*(ny-0.5);
	r0.z = Lmin(Z) + dz*(nz-0.5);

	if (ran_serial_uniform() < vp) {
	  index++;
	  COLL_add_colloid_no_halo(index, a0, ah, r0, zero, zero);
	}
      }
    }
  }

  /* Add fcc particles if required. */

  n_request -= index;
  n_max = (ncx-1)*(ncy-1)*(ncz-1);
  vp = (double) n_request / (double) n_max;

  for (nx = 1; nx < ncx; nx++) {
    for (ny = 1; ny < ncy; ny++) {
      for (nz = 1; nz < ncz; nz++) {

	r0.x = Lmin(X) + dx*nx;
	r0.y = Lmin(Y) + dy*ny;
	r0.z = Lmin(Z) + dz*nz;

	if (ran_serial_uniform() < vp) {
	  index++;
	  COLL_add_colloid_no_halo(index, a0, ah, r0, zero, zero);
	}
      }
    }
  }

  return index;
}

/*****************************************************************************
 *
 *  mc_mean_square_displacement
 *
 *  Compute mean square displacement of particles since initialisation
 *
 *****************************************************************************/

void mc_mean_square_displacement() {

  int     ic, jc, kc;
  Colloid * p_colloid;
  double    ds, dxsq = 0.0, dysq = 0.0, dzsq = 0.0;


  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
	  ds = p_colloid->r.x - p_colloid->stats.x;
	  dxsq += ds*ds;
	  ds = p_colloid->r.y - p_colloid->stats.y;
	  dysq += ds*ds;
	  ds = p_colloid->r.z - p_colloid->stats.z;
	  dzsq += ds*ds;

	  p_colloid = p_colloid->next;
	}
	/* Next cell */
      }
    }
  }

  dxsq /= get_N_colloid();
  dysq /= get_N_colloid();
  dzsq /= get_N_colloid();

  info("Mean square displacements (serial only) %f %f %f\n", dxsq, dysq, dzsq);

  return;
}

/*****************************************************************************
 *
 *  CMD_reset_particles
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
  Colloid * p_colloid;

  double xt = 0.0, yt = 0.0, zt = 0.0;
  double vmin = 0.0, vmax = 0.0;
  int nt = 0;

  double var;
  double get_kT(void);

  var = factor*sqrt(get_kT());

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
	  p_colloid->v.x   = var*ran_parallel_gaussian();
	  p_colloid->v.y   = var*ran_parallel_gaussian();
	  p_colloid->v.z   = var*ran_parallel_gaussian();
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

    for (ic = 1; ic <= Ncell(X); ic++) {
      for (jc = 1; jc <= Ncell(Y); jc++) {
	for (kc = 1; kc <= Ncell(Z); kc++) {

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

  cell_update();
  CCOM_halo_particles();

  return;
}

/*****************************************************************************
 *
 *  CMD_test_particle_energy
 *
 *****************************************************************************/

void CMD_test_particle_energy(const int step) {

  int     ic, jc, kc;
  Colloid * p_colloid;

  double sum[3], gsum[3];

  sum[0] = 0.0;
  sum[1] = 0.0;
  sum[2] = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

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

  gsum[0] = sum[0]/get_N_colloid();
  gsum[1] = sum[1]/get_N_colloid();
  gsum[2] = sum[2]/get_N_colloid();

  info("Mean particle velocities: [x, y, z]\n");
  info("[%g, %g, %g]\n", gsum[0], gsum[1], gsum[2]);

  sum[0] = 0.0;
  sum[1] = 0.0;
  sum[2] = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

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

  gsum[0] = sum[0]/get_N_colloid();
  gsum[1] = sum[1]/get_N_colloid();
  gsum[2] = sum[2]/get_N_colloid();

  /* Can multiply by mass */
  info("Particle: %d %g %g %g %g\n", step, gsum[0], gsum[1], gsum[2],
       (gsum[0] + gsum[1] + gsum[2]));


  return;
}
