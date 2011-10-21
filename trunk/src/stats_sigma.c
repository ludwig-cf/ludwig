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
#include "phi.h"
#include "phi_cahn_hilliard.h"
#include "phi_lb_coupler.h"
#include "symmetric.h"
 
#define NBIN      128
#define NFITMAX   2000
#define XIINIT    2.0
#define XIPROFILE 3.0

struct drop_type {
  double radius;
  double xi0;
  double centre[3];
  double phimax;
};

typedef struct drop_type drop_t;

static int initialised_ = 0;

static void stats_sigma_init_drop(drop_t drop);
static void stats_sigma_find_drop(drop_t * drop);
static void stats_sigma_find_radius(drop_t * drop);
static void stats_sigma_find_sigma(drop_t drop, double results[2]);

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

    /* Check we have a cubic system */

    /* Initialise the drop properties. */

    initialised_ = 1;
    drop.radius    = L(X)/4.0;
    drop.xi0       = 2.0*symmetric_interfacial_width();
    drop.centre[X] = L(X)/2.0;
    drop.centre[Y] = L(Y)/2.0;
    drop.centre[Z] = L(Z)/2.0;
    drop.phimax    = sqrt(-symmetric_a()/symmetric_b());

    /* Initialise the order parameter field */

    stats_sigma_init_drop(drop);

    /* Print some information */

    info("\n");
    info("Surface tension calibration via droplet initialised.\n");
    info("Drop radius:     %14.7e\n", drop.radius);
    datum = symmetric_interfacial_width()/drop.radius;
    info("Cahn number:     %14.7e\n", datum);
    datum = phi_cahn_hilliard_mobility()*symmetric_a();
    info("Diffusivity:     %14.7e\n", datum);
    /* The relevant diffusion time is for the interfacial width ... */
    datum = drop.xi0*drop.xi0/datum;
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
  double results[2];

  if (initialised_) {
    /* do compuation */
    stats_sigma_find_drop(&drop);
    stats_sigma_find_radius(&drop);
    stats_sigma_find_sigma(drop, results);
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

        phi = phi_get_phi_site(index);

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
        phi0 = phi_get_phi_site(index);

        /* Look around at the neighbours */

        for (ip = ic - 1; ip <= ic + 1; ip++) {
          for (jp = jc - 1; jp <= jc + 1; jp++) {
            for (kp = kc - 1; kp <= kc + 1; kp++) {

              /* Avoid self (ic, jc, kc) */
	      if (!(ip || jp || kp)) continue;

              index = coords_index(ip, jp, kp);
              phi1 = phi_get_phi_site(index);

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
 *  stats_sigma_find_sigma
 *
 *  Compute the radial profile of phi and of the free energy density
 *  (binned). phi(r) is used to fit an interfacial width and f(r) is
 *  used to fit a sigma.
 *
 *  results[0] is xi, results[1] is sigma, on exit.
 *
 *****************************************************************************/

static void stats_sigma_find_sigma(drop_t drop, double results[2]) {

  int nlocal[3], noffset[3];
  int ic, jc, kc, index, n;
  double rmin;
  double rmax;
  double r0, dr, r[3];

  double phimean[NBIN], phimeant[NBIN];
  double femean[NBIN], femeant[NBIN];
  double count[NBIN], countt[NBIN];

  int nfit, nbestfit;
  double fmin, fmax, cost, costmin, phi, xi0, xi0fit;

  MPI_Comm comm;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  comm = pe_comm();

  /* Set the bin widths etc */

  xi0 = symmetric_interfacial_width();

  rmin = drop.radius - XIPROFILE*xi0;
  rmax = drop.radius + XIPROFILE*xi0;
  dr = (rmax - rmin)/NBIN;

  for (n = 0; n < NBIN; n++) {
    phimean[n] = 0.0;
    femean[n] = 0.0;
    count[n] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        /* Work out the displacement from the drop centre
           and hence the bin number */

        r[X] = 1.0*(noffset[X] + ic) - drop.centre[X];
        r[Y] = 1.0*(noffset[Y] + jc) - drop.centre[Y];
        r[Z] = 1.0*(noffset[Z] + kc) - drop.centre[Z];

        r0 = modulus(r);
        n = (r0 - rmin)/dr;

        if (n >= 0 && n < NBIN) {
          phimean[n] += phi_get_phi_site(index);
          femean[n]   += symmetric_free_energy_density(index);
          count[n]   += 1.0;
        }
      }
    }
  }

  /* Reduce to rank zero for output. */

  MPI_Reduce(phimean, phimeant, NBIN, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(femean, femeant, NBIN, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(count, countt, NBIN, MPI_DOUBLE, MPI_SUM, 0, comm);

  fmin = DBL_MAX;
  fmax = -DBL_MAX;

  for (n = 0; n < NBIN; n++) {
    phimeant[n] /= countt[n];
    femeant[n] /= countt[n];
    fmin = dmin(fmin, femeant[n]);
    fmax = dmax(fmax, femeant[n]);
  }

  /* Fit the free energy density to sech^4 to get an estimate of the
   * surface tension. Assume there is enough resolution to give a
   * good estimate of fmax-fmin. */

  info("\n");
  info("Free energy density range: %g %g -> %g\n", fmin, fmax, fmax-fmin);

  /* Try values 0 < xi < 2xi_0 and see which is best least squares fit
   * to the measured mean phi; the (nfit + 1) is to avoid zero. */

  nbestfit = -1;
  costmin =  DBL_MAX;

  for (nfit = 0; nfit < NFITMAX; nfit++) {
    cost = 0.0;
    xi0fit = 2.0*(nfit + 1)*xi0/NFITMAX;
    for (n = 0; n < NBIN; n++) {
      r0 = rmin + (n + 0.5)*dr;
      phi = tanh((r0 - drop.radius)/xi0fit);
      if (countt[n] != 0) cost += (phimeant[n] - phi)*(phimeant[n] - phi);
    }
    if (cost < costmin) {
      costmin = cost;
      nbestfit = nfit;
    }
  }

  assert(nbestfit > 0);
  xi0fit = 2.0*(nbestfit + 1)*xi0/NFITMAX;

  info("Fit to interfacial width: %f\n", xi0fit);
  info("Fit to sigma: %g\n", 2.0*sqrt(8.0/9.0)*xi0fit*(fmax-fmin));

  return;
}
