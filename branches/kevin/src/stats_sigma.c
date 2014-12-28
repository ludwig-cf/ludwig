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

#include "util.h"
#include "physics.h"
#include "symmetric.h"
#include "stats_sigma.h"
 
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

struct stats_sigma_s {
  coords_t * cs;         /* Reference to coordinate system */
  drop_t drop;           /* Droplet for tension calculation */
};

static int stats_sigma_find_sigma(stats_sigma_t * stat);
static int stats_sigma_init_drop(stats_sigma_t * stat, field_t * phi);
static int stats_sigma_find_drop(stats_sigma_t * stat, field_t * phi);
static int stats_sigma_find_radius(stats_sigma_t * stat, field_t * phi);
static int stats_sigma_find_xi0(stats_sigma_t * stat, field_t * phi);

/*****************************************************************************
 *
 *  stats_sigma_create
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

int stats_sigma_create(coords_t * cs, field_t * phi, stats_sigma_t ** pstat) {

  stats_sigma_t * stat = NULL;
  int ntotal[3];
  double datum;
  double mobility;

  assert(cs);
  assert(phi);

  stat = (stats_sigma_t *) calloc(1, sizeof(stats_sigma_t));
  if (stat == NULL) fatal("calloc(stat_sigma_t) failed\n");

  coords_ntotal(ntotal);
  physics_mobility(&mobility);

  /* Check we have a cubic system, or a square system (2d) */

  if (ntotal[X] != ntotal[Y]) {
    info("Surface tension calibration expects Lx = Ly\n");
    info("You have: %4d %4d\n", ntotal[X], ntotal[Y]);
    fatal("Please check and try again\n");
  }

  if (ntotal[Z] != 1 && (ntotal[Z] != ntotal[Y])) {
    info("Surface tension calibration expects Lx = Ly = Lz\n");
    info("You have: %4d %4d %4d\n", ntotal[X], ntotal[Y], ntotal[Z]);
    fatal("Please check and try again\n");
  }

  stat->cs = cs;
  coords_retain(cs);

  /* Initialise the drop properties. */

  stat->drop.radius    = L(X)/4.0;
  stat->drop.xi0       = XIINIT*symmetric_interfacial_width();
  stat->drop.centre[X] = L(X)/2.0;
  stat->drop.centre[Y] = L(Y)/2.0;
  stat->drop.centre[Z] = L(Z)/2.0;
  stat->drop.phimax    = sqrt(-symmetric_a()/symmetric_b());

  /* Initialise the order parameter field */

  stats_sigma_init_drop(stat, phi);

  /* Print some information */

  info("\n");
  info("Surface tension calibration via droplet initialised\n");
  info("---------------------------------------------------\n");
  info("Drop radius:     %14.7e\n", stat->drop.radius);
  datum = symmetric_interfacial_width()/stat->drop.radius;
  info("Cahn number:     %14.7e\n", datum);
  datum = -mobility*symmetric_a();
  info("Diffusivity:     %14.7e\n", datum);
  /* The relevant diffusion time is for the interfacial width ... */
  datum = XIINIT*stat->drop.xi0*XIINIT*stat->drop.xi0/datum;
  info("Diffusion time:  %14.7e\n", datum); 

  return 0;
}

/*****************************************************************************
 *
 *  stats_sigma_free
 *
 *****************************************************************************/

int stats_sigma_free(stats_sigma_t * stat) {

  assert(stat);

  coords_free(&stat->cs);
  free(stat);

  return 0;
}

/*****************************************************************************
 *
 *  stats_sigma_measure
 *
 *  Measure the surface tension and apparent xi and report.
 *  The argument is just the time step.
 *
 *****************************************************************************/

int stats_sigma_measure(stats_sigma_t * stat, field_t * fphi, int ntime) {

  assert(stat);
  assert(fphi);

  stats_sigma_find_drop(stat, fphi);
  stats_sigma_find_radius(stat, fphi);
  stats_sigma_find_xi0(stat, fphi);
  stats_sigma_find_sigma(stat);

  info("\n");
  info("Surface tension calibration - radius xi0 surface tension\n");
  info("[sigma] %14d %14.7e %14.7e %14.7e\n", ntime, stat->drop.radius,
       stat->drop.xi0, stat->drop.sigma);

  return 0;
}

/*****************************************************************************
 *
 *  stats_sigma_init_phi
 *
 *  Initialise the order parameter according to a tanh profile.
 *  Note phi = -drop.phimax in the centre.
 *
 *****************************************************************************/

static int stats_sigma_init_drop(stats_sigma_t * stat, field_t * fphi) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];  /* current lattice site position */
  double phi;          /* order parameter value */
  double r;            /* radial distance */
  double rxi0;         /* 1/xi0 */

  assert(stat);
  assert(fphi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  rxi0 = 1.0/stat->drop.xi0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        position[X] = 1.0*(noffset[X] + ic) - stat->drop.centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - stat->drop.centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - stat->drop.centre[Z];

        r = modulus(position);

        phi = stat->drop.phimax*tanh(rxi0*(r - stat->drop.radius));
	field_scalar_set(fphi, index, phi);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_sigma_find_drop
 *
 *  Locate the drop centre by looking for the 'centre of mass'.
 *  Expect phi < 0 at the centre.
 *
 *****************************************************************************/

static int stats_sigma_find_drop(stats_sigma_t * stat, field_t * fphi) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double c[3+1];              /* 3 space dimensions plus counter */
  double ctotal[3+1];
  double phi;
  MPI_Comm comm;

  assert(stat);
  assert(fphi);

  coords_cart_comm(stat->cs, &comm);
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
	field_scalar(fphi, index, &phi);

        if (phi <= 0.0) {
          c[X] += 1.0*(noffset[X] + ic);
          c[Y] += 1.0*(noffset[Y] + jc);
          c[Z] += 1.0*(noffset[Z] + kc);
          c[3] += 1.0;
        }

      }
    }
  }

  MPI_Allreduce(c, ctotal, 4, MPI_DOUBLE, MPI_SUM, comm);

  stat->drop.centre[X] = ctotal[X]/ctotal[3];
  stat->drop.centre[Y] = ctotal[Y]/ctotal[3];
  stat->drop.centre[Z] = ctotal[Z]/ctotal[3];

  return 0;
}

/*****************************************************************************
 *
 *  stats_sigma_find_radius
 *
 *  Find the radius of the drop; the current centre must be known.
 *  Expect phi < 0 on inside.
 *
 *****************************************************************************/

static int stats_sigma_find_radius(stats_sigma_t * stat, field_t * fphi) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  int ip, jp, kp;

  double result[2];                    /* result accumulator and count */
  double result_total[2];              /* ditto after Allreduce() */
  double phi0, phi1;                   /* order parameter values */ 
  double fraction, r[3];               /* lengths */
  MPI_Comm comm;


  assert(stat);
  assert(fphi);

  coords_cart_comm(stat->cs, &comm);
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
	field_scalar(fphi, index, &phi0);

        /* Look around at the neighbours */

        for (ip = ic - 1; ip <= ic + 1; ip++) {
          for (jp = jc - 1; jp <= jc + 1; jp++) {
            for (kp = kc - 1; kp <= kc + 1; kp++) {

              /* Avoid self (ic, jc, kc) */
	      if (!(ip || jp || kp)) continue;

              index = coords_index(ip, jp, kp);
	      field_scalar(fphi, index, &phi1);

	      /* Look for change in sign */

              if (phi0 < 0.0 && phi1 > 0.0) {
		assert(phi0 != phi1);
                fraction = phi0 / (phi0 - phi1);
                r[X] = 1.0*(noffset[X] + ic) + fraction*(ip-ic);
                r[Y] = 1.0*(noffset[Y] + jc) + fraction*(jp-jc);
                r[Z] = 1.0*(noffset[Z] + kc) + fraction*(kp-kc);
                r[X] -= stat->drop.centre[X];
                r[Y] -= stat->drop.centre[Y];
                r[Z] -= stat->drop.centre[Z];

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

  MPI_Allreduce(result, result_total, 2, MPI_DOUBLE, MPI_SUM, comm);

  stat->drop.radius = result_total[0]/result_total[1];

  return 0;
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

static int stats_sigma_find_xi0(stats_sigma_t * stat, field_t * fphi) {

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

  assert(stat);
  assert(fphi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  comm = pe_comm();

  /* Set the bin widths based on the expected xi0 */

  xi0 = symmetric_interfacial_width();

  rmin = stat->drop.radius - XIPROFILE*xi0;
  rmax = stat->drop.radius + XIPROFILE*xi0;
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

        r[X] = 1.0*(noffset[X] + ic) - stat->drop.centre[X];
        r[Y] = 1.0*(noffset[Y] + jc) - stat->drop.centre[Y];
        r[Z] = 1.0*(noffset[Z] + kc) - stat->drop.centre[Z];

        r0 = modulus(r);
        n = (r0 - rmin)/dr;

        if (n >= 0 && n < NBIN) {
	  field_scalar(fphi, index, &phi);
	  phir_local[n] += phi;
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
      phi = tanh((r0 - stat->drop.radius)/xi0fit);
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
  stat->drop.xi0 = xi0fit;

  return 0;
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

static int stats_sigma_find_sigma(stats_sigma_t * stat) {

  int nlocal[3];
  int ic, jc, kc, index;

  double fe;
  double fmin, fmin_local;      /* Minimum free energy */
  double excess, excess_local;  /* Excess free energy */

  MPI_Comm comm;

  assert(stat);

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
    stat->drop.sigma = excess / (2.0*pi_*stat->drop.radius);
  }
  else {
    stat->drop.sigma = excess / (4.0*pi_*stat->drop.radius*stat->drop.radius);
  }

  return 0;
}

