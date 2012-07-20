/*****************************************************************************
 *
 *  stats_sigma.c
 *
 *  Provides a calibration of the surface tension when the symmetric
 *  free energy is used. This approach uses a droplet in 2d or 3d.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computeing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"

#ifdef OLD_PHI
#include "phi.h"
#else
#include "field.h"
#endif

#include "phi_cahn_hilliard.h"
#include "phi_lb_coupler.h"
#include "symmetric.h"
 
#define NBIN      128
#define NFITMAX   2000
#define XIINIT    2.0
#define XIPROFILE 10.0

struct drop_type {
  double radius;
  double xi0;
  double centre[3];
  double phimax;
  double sigma;
};

typedef struct drop_type drop_t;

static int initialised_ = 0;

static void stats_sigma_init_drop(drop_t drop);
static void stats_sigma_find_drop(drop_t * drop);
static void stats_sigma_find_radius(drop_t * drop);
static void stats_sigma_find_xi0(drop_t * drop);
static void stats_sigma_find_sigma(drop_t * drop);

/*****************************************************************************
 *
 *  stats_sigma_init
 *
 *  1. We initialise a drop of raduis L/4, and initial interfacial
 *     width XIINIT*\xi_0, in the middle of the system.
 *  2. A call to stats_sigma_measure() will compute the current
 *     surface via A METHOD, and the current interfacial width
 *     by a best fit to tanh(), based on a binned average radial
 *     profile of the droplet order parameter.
 *
 *  Notes.
 *
 *  This code does not prescribe any parameters --- it just takes
 *  what it is given. However, there should probably be a small
 *  Cahn number (xi0/R) to prevent significant evaporation. The
 *  drop should be allowed to relax for a time xi0^2 / D where
 *  the diffusivity is D ~ MA, the mobility multiplied by the
 *  free energy scale parameter A.
 *
 *  These values are reported below at run time.
 *
 *****************************************************************************/

void stats_sigma_init(int nswitch) {

  drop_t drop;
  double datum;

  if (nswitch == 0) {
    /* No measurement required. */
    initialised_ = 0;
  }
  else {

    /* Check we have a cubic system, or a square system (2d) */

    if (N_total(X) != N_total(Y)) {
      info("Surface tension calibration expects Lx = Ly\n");
      info("You have: %4d %4d\n", N_total(X), N_total(Y));
      fatal("Please check and try again\n");
    }

    if (N_total(Z) != 1 && (N_total(Z) != N_total(Y))) {
      info("Surface tension calibration expects Lx = Ly = Lz\n");
      info("You have: %4d %4d %4d\n", N_total(X), N_total(Y), N_total(Z));
      fatal("Please check and try again\n");
    }

    /* Initialise the drop properties. */

    initialised_ = 1;
    drop.radius    = L(X)/4.0;
    drop.xi0       = XIINIT*symmetric_interfacial_width();
    drop.centre[X] = L(X)/2.0;
    drop.centre[Y] = L(Y)/2.0;
    drop.centre[Z] = L(Z)/2.0;
    drop.phimax    = sqrt(-symmetric_a()/symmetric_b());

    /* Initialise the order parameter field */

    stats_sigma_init_drop(drop);

    /* Print some information */

    info("\n");
    info("Surface tension calibration via droplet initialised\n");
    info("---------------------------------------------------\n");
    info("Drop radius:     %14.7e\n", drop.radius);
    datum = symmetric_interfacial_width()/drop.radius;
    info("Cahn number:     %14.7e\n", datum);
    datum = -phi_cahn_hilliard_mobility()*symmetric_a();
    info("Diffusivity:     %14.7e\n", datum);
    /* The relevant diffusion time is for the interfacial width ... */
    datum = XIINIT*drop.xi0*XIINIT*drop.xi0/datum;
    info("Diffusion time:  %14.7e\n", datum); 
  }

  return;
}

/*****************************************************************************
 *
 *  stats_sigma_measure
 *
 *  Measure the surface tension and apparent xi and report.
 *  The argument is just the time step.
 *
 *****************************************************************************/

void stats_sigma_measure(int ntime) {

  drop_t drop;

  if (initialised_) {

    stats_sigma_find_drop(&drop);
    stats_sigma_find_radius(&drop);
    stats_sigma_find_xi0(&drop);
    stats_sigma_find_sigma(&drop);

    info("\n");
    info("Surface tension calibration - radius xi0 surface tension\n");
    info("[sigma] %14d %14.7e %14.7e %14.7e\n", ntime, drop.radius, drop.xi0,
	 drop.sigma);
  }

  return;
}

/*****************************************************************************
 *
 *  stats_sigma_init_phi
 *
 *  Initialise the order parameter according to a tanh profile.
 *  Note phi = -drop.phimax in the centre.
 *
 *****************************************************************************/

static void stats_sigma_init_drop(drop_t drop) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];  /* current lattice site position */
  double phi;          /* order parameter value */
  double r;            /* radial distance */
  double rxi0;         /* 1/xi0 */

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  rxi0 = 1.0/drop.xi0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        position[X] = 1.0*(noffset[X] + ic) - drop.centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - drop.centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - drop.centre[Z];

        r = modulus(position);

        phi = drop.phimax*tanh(rxi0*(r - drop.radius));
        phi_lb_coupler_phi_set(index, phi);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  stats_sigma_find_drop
 *
 *  Locate the drop centre by looking for the 'centre of mass'.
 *  Expect phi < 0 at the centre.
 *
 *****************************************************************************/

static void stats_sigma_find_drop(drop_t * drop) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double c[3+1];              /* 3 space dimensions plus counter */
  double ctotal[3+1];
  double phi;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 0; ic <= 3; ic++) {
    c[ic] =0.0;
    ctotal[ic] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

#ifdef OLD_PHI
        phi = phi_get_phi_site(index);
#else
	assert(0);
	/* update interface */
#endif

        if (phi <= 0.0) {
          c[X] += 1.0*(noffset[X] + ic);
          c[Y] += 1.0*(noffset[Y] + jc);
          c[Z] += 1.0*(noffset[Z] + kc);
          c[3] += 1.0;
        }

      }
    }
  }

  MPI_Allreduce(c, ctotal, 4, MPI_DOUBLE, MPI_SUM, cart_comm());

  drop->centre[X] = ctotal[X]/ctotal[3];
  drop->centre[Y] = ctotal[Y]/ctotal[3];
  drop->centre[Z] = ctotal[Z]/ctotal[3];

  return;
}

/*****************************************************************************
 *
 *  stats_sigma_find_radius
 *
 *  Find the radius of the drop; the current centre must be known.
 *  Expect phi < 0 on inside.
 *
 *****************************************************************************/

static void stats_sigma_find_radius(drop_t * drop) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  int ip, jp, kp;

  double result[2];                    /* result accumulator and count */
  double result_total[2];              /* ditto after Allreduce() */
  double phi0, phi1;                   /* order parameter values */ 
  double fraction, r[3];               /* lengths */

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  result[0] = 0.0;
  result[1] = 0.0;
  result_total[0] = 0.0;
  result_total[1] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
#ifdef OLD_PHI
        phi0 = phi_get_phi_site(index);
#else
	assert(0);
	/* update interface */
#endif

        /* Look around at the neighbours */

        for (ip = ic - 1; ip <= ic + 1; ip++) {
          for (jp = jc - 1; jp <= jc + 1; jp++) {
            for (kp = kc - 1; kp <= kc + 1; kp++) {

              /* Avoid self (ic, jc, kc) */
	      if (!(ip || jp || kp)) continue;

              index = coords_index(ip, jp, kp);
#ifdef OLD_PHI
              phi1 = phi_get_phi_site(index);
#else
	      assert(0);
	      /* sort interface*/
#endif
	      /* Look for change in sign */

              if (phi0 < 0.0 && phi1 > 0.0) {
		assert(phi0 != phi1);
                fraction = phi0 / (phi0 - phi1);
                r[X] = 1.0*(noffset[X] + ic) + fraction*(ip-ic);
                r[Y] = 1.0*(noffset[Y] + jc) + fraction*(jp-jc);
                r[Z] = 1.0*(noffset[Z] + kc) + fraction*(kp-kc);
                r[X] -= drop->centre[X];
                r[Y] -= drop->centre[Y];
                r[Z] -= drop->centre[Z];

                result[0] += modulus(r);
                result[1] += 1.0;
              }
            }
          }
        }

	/* Next site */
      }
    }
  }

  MPI_Allreduce(result, result_total, 2, MPI_DOUBLE, MPI_SUM, cart_comm());

  drop->radius = result_total[0]/result_total[1];

  return;
}

/*****************************************************************************
 *
 *  stats_sigma_find_xi0
 *
 *  Compute a (binned) mean radial profile of the order parameter and
 *  use the resulting phi(r) to fit an interfacial width based on the
 *  expected tanh(r/xi) profile.
 *
 *  The current values of drop.centre and drop.radius are used to locate
 *  the required profile.
 *
 *  The observed xi is returned as drop.xi0 on rank 0.
 *
 *****************************************************************************/

static void stats_sigma_find_xi0(drop_t * drop) {

  int nlocal[3], noffset[3];
  int ic, jc, kc, index, n;
  int nfit, nbestfit;

  int nphi[NBIN], nphi_local[NBIN];      /* count for bins */
  double phir[NBIN], phir_local[NBIN];   /* binned profile */

  double rmin;                           /* Minimum radial profile r */
  double rmax;                           /* Maximum radial profile r */
  double r0, dr, r[3];

  double cost, costmin, phi;             /* Best fit calculation */
  double xi0, xi0fit;                    /* Expected and observed xi0 */

  MPI_Comm comm;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  comm = pe_comm();

  /* Set the bin widths based on the expected xi0 */

  xi0 = symmetric_interfacial_width();

  rmin = drop->radius - XIPROFILE*xi0;
  rmax = drop->radius + XIPROFILE*xi0;
  dr = (rmax - rmin)/NBIN;

  for (n = 0; n < NBIN; n++) {
    phir_local[n] = 0.0;
    nphi_local[n] = 0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        /* Work out the displacement from the drop centre
           and hence the bin number */

        r[X] = 1.0*(noffset[X] + ic) - drop->centre[X];
        r[Y] = 1.0*(noffset[Y] + jc) - drop->centre[Y];
        r[Z] = 1.0*(noffset[Z] + kc) - drop->centre[Z];

        r0 = modulus(r);
        n = (r0 - rmin)/dr;

        if (n >= 0 && n < NBIN) {
#ifdef OLD_PHI
          phir_local[n] += phi_get_phi_site(index);
#else
	  assert(0);
	  /* sort interface */
#endif
          nphi_local[n] += 1;
        }

      }
    }
  }

  /* Reduce to rank zero and compute the mean profile. */

  MPI_Reduce(phir_local, phir, NBIN, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(nphi_local, nphi, NBIN, MPI_INT, MPI_SUM, 0, comm);

  for (n = 0; n < NBIN; n++) {
    if (nphi[n] > 0) phir[n] = phir[n]/nphi[n];
  }

  /* Fit the mean order parameter profile to tanh((r - r0)/xi).
     Try values 0 < xi < 2xi_0 and see which is best least squares fit
     to the measured mean phi; the (nfit + 1) is to avoid zero. */

  nbestfit = -1;
  costmin =  DBL_MAX;

  for (nfit = 0; nfit < NFITMAX; nfit++) {
    cost = 0.0;
    xi0fit = 2.0*(nfit + 1)*xi0/NFITMAX;
    for (n = 0; n < NBIN; n++) {
      r0 = rmin + (n + 0.5)*dr;
      phi = tanh((r0 - drop->radius)/xi0fit);
      if (nphi[n] > 0) cost += (phir[n] - phi)*(phir[n] - phi);
    }
    if (cost < costmin) {
      costmin = cost;
      nbestfit = nfit;
    }
  }

  /* This assertion should not fail unless something is very wrong */

  assert(nbestfit > 0);
  xi0fit = 2.0*(nbestfit + 1)*xi0/NFITMAX;
  drop->xi0 = xi0fit;

  return;
}

/*****************************************************************************
 *
 *  stats_sigma_find_sigma
 *
 *  Integrate the excess free energy density to estimate the actual
 *  surface tension of the drop. The current drop.radius is used
 *  to compute the circumference (2d) or area (3d) of the drop.
 *
 *  drop.sigma is updated with the value identified on rank 0.
 *
 *****************************************************************************/

static void stats_sigma_find_sigma(drop_t * drop) {

  int nlocal[3];
  int ic, jc, kc, index;

  double fe;
  double fmin, fmin_local;      /* Minimum free energy */
  double excess, excess_local;  /* Excess free energy */

  MPI_Comm comm;

  coords_nlocal(nlocal);
  comm = pe_comm();

  /* Find the local minimum of the free energy density */

  fmin_local = FLT_MAX;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        fe = symmetric_free_energy_density(index);
	if (fe < fmin_local) fmin_local = fe;

      }
    }
  }

  /* Everyone needs fmin to compute the excess */

  MPI_Allreduce(&fmin_local, &fmin, 1, MPI_DOUBLE, MPI_MIN, comm);

  excess_local = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        fe = symmetric_free_energy_density(index);
        excess_local += (fe - fmin);
      }
    }
  }

  /* Reduce to rank zero for result */

  MPI_Reduce(&excess_local, &excess, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (nlocal[Z] == 1) {
    /* Assume 2d system */
    drop->sigma = excess / (2.0*pi_*drop->radius);
  }
  else {
    drop->sigma = excess / (4.0*pi_*drop->radius*drop->radius);
  }

  return;
}

