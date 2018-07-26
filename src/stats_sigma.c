/*****************************************************************************
 *
 *  stats_sigma.c
 *
 *  Provides a calibration of the surface tension when the symmetric
 *  free energy is used. This approach uses a droplet in 2d or 3d.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computeing Centre
 *
 *  (c) 2011-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "util.h"
#include "physics.h"
#include "field_phi_init.h"
#include "stats_sigma.h"
 
#define NBIN      128     /* Bins used in radial profile */
#define NFITMAX   2000
#define XIINIT    2.0     /* Factor controlling initial interfacial width */
#define XIPROFILE 10.0

struct drop_type {
  fe_symm_t * fe;
  double radius;
  double xi0;
  double centre[3];
  double phimax;
  double sigma;
};

typedef struct drop_type drop_t;

struct stats_sigma_s {
  pe_t * pe;
  cs_t * cs;
  drop_t drop;
  field_t * phi;
};

static int stats_sigma_find_sigma(stats_sigma_t * stat);
static int stats_sigma_find_drop(stats_sigma_t * stat);
static int stats_sigma_find_radius(stats_sigma_t * stat);
static int stats_sigma_find_xi0(stats_sigma_t * stat);

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

int stats_sigma_create(pe_t * pe, cs_t * cs, fe_symm_t * fe, field_t * phi,
		       stats_sigma_t ** pobj) {

  stats_sigma_t * obj = NULL;
  int ntotal[3];
  double xi0;
  double tdiff;
  double mobility;
  physics_t * phys = NULL;
  fe_symm_param_t param;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (stats_sigma_t *) calloc(1, sizeof(stats_sigma_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(stats_sigma_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->drop.fe = fe;
  obj->phi = phi;

  cs_ntotal(cs, ntotal);
  physics_ref(&phys);
  physics_mobility(phys, &mobility);

  /* Check we have a cubic system, or a square system (2d) */

  if (ntotal[X] != ntotal[Y]) {
    pe_info(pe, "Surface tension calibration expects Lx = Ly\n");
    pe_info(pe, "You have: %4d %4d\n", ntotal[X], ntotal[Y]);
    pe_fatal(pe, "Please check and try again\n");
  }

  if (ntotal[Z] != 1 && (ntotal[Z] != ntotal[Y])) {
    pe_info(pe, "Surface tension calibration expects Lx = Ly = Lz\n");
    pe_info(pe, "You have: %4d %4d %4d\n", ntotal[X], ntotal[Y], ntotal[Z]);
    pe_fatal(pe, "Please check and try again\n");
  }

  /* Initialise the drop properties. */

  fe_symm_param(fe, &param);
  fe_symm_interfacial_width(fe, &xi0);

  obj->drop.radius    = 0.25*ntotal[X];
  obj->drop.xi0       = XIINIT*xi0;
  obj->drop.centre[X] = 0.5*ntotal[X];
  obj->drop.centre[Y] = 0.5*ntotal[Y];
  obj->drop.centre[Z] = 0.5*ntotal[Z];
  obj->drop.phimax    = sqrt(-param.a/param.b);

  /* Initialise the order parameter field */

  field_phi_init_drop(phi, obj->drop.xi0, obj->drop.radius, obj->drop.phimax);

  /* Print some information */
  /* The diffusivity is mobility/A (with A < 0) */
  /* The relevant diffusion time is for the initial interfacial width. */

  tdiff = XIINIT*xi0*XIINIT*xi0/(-mobility/param.a);

  pe_info(pe, "\n");
  pe_info(pe, "Surface tension calibration via droplet initialised\n");
  pe_info(pe, "---------------------------------------------------\n");
  pe_info(pe, "Drop radius:     %14.7e\n", obj->drop.radius);
  pe_info(pe, "Cahn number:     %14.7e\n", xi0/obj->drop.radius);
  pe_info(pe, "Diffusivity:     %14.7e\n", -mobility/param.a);
  pe_info(pe, "Diffusion time:  %14.7e\n", tdiff); 

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  stats_sigma_free
 *
 *****************************************************************************/

int stats_sigma_free(stats_sigma_t * stat) {

  assert(stat);

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

int stats_sigma_measure(stats_sigma_t * stat, int ntime) {

  if (stat) {

    stats_sigma_find_drop(stat);
    stats_sigma_find_radius(stat);
    stats_sigma_find_xi0(stat);
    stats_sigma_find_sigma(stat);

    pe_info(stat->pe, "\n");
    pe_info(stat->pe, "Surface tension calibration - radius xi0 surface tension\n");
    pe_info(stat->pe, "[sigma] %14d %14.7e %14.7e %14.7e\n", ntime,
	    stat->drop.radius, stat->drop.xi0, stat->drop.sigma);
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

static int stats_sigma_find_drop(stats_sigma_t * stat) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double c[3+1];              /* 3 space dimensions plus counter */
  double ctotal[3+1];
  double phi;
  MPI_Comm comm;

  assert(stat);

  cs_nlocal(stat->cs, nlocal);
  cs_nlocal_offset(stat->cs, noffset);
  cs_cart_comm(stat->cs, &comm);

  for (ic = 0; ic <= 3; ic++) {
    c[ic] =0.0;
    ctotal[ic] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(stat->cs, ic, jc, kc);
	field_scalar(stat->phi, index, &phi);

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

static int stats_sigma_find_radius(stats_sigma_t * stat) {

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

  cs_nlocal(stat->cs, nlocal);
  cs_nlocal_offset(stat->cs, noffset);
  cs_cart_comm(stat->cs, &comm);

  result[0] = 0.0;
  result[1] = 0.0;
  result_total[0] = 0.0;
  result_total[1] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(stat->cs, ic, jc, kc);
	field_scalar(stat->phi, index, &phi0);

        /* Look around at the neighbours */

        for (ip = ic - 1; ip <= ic + 1; ip++) {
          for (jp = jc - 1; jp <= jc + 1; jp++) {
            for (kp = kc - 1; kp <= kc + 1; kp++) {

              /* Avoid self (ic, jc, kc) */
	      if (!(ip || jp || kp)) continue;

              index = cs_index(stat->cs, ip, jp, kp);
	      field_scalar(stat->phi, index, &phi1);

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

static int stats_sigma_find_xi0(stats_sigma_t * stat) {

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

  cs_nlocal(stat->cs, nlocal);
  cs_nlocal_offset(stat->cs, noffset);
  pe_mpi_comm(stat->pe, &comm);

  /* Set the bin widths based on the expected xi0 */

  fe_symm_interfacial_width(stat->drop.fe, &xi0);

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

        index = cs_index(stat->cs, ic, jc, kc);

        /* Work out the displacement from the drop centre
           and hence the bin number */

        r[X] = 1.0*(noffset[X] + ic) - stat->drop.centre[X];
        r[Y] = 1.0*(noffset[Y] + jc) - stat->drop.centre[Y];
        r[Z] = 1.0*(noffset[Z] + kc) - stat->drop.centre[Z];

        r0 = modulus(r);
        n = (r0 - rmin)/dr;

        if (n >= 0 && n < NBIN) {
	  field_scalar(stat->phi, index, &phi);
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
  PI_DOUBLE(pi);

  MPI_Comm comm;

  assert(stat);

  cs_nlocal(stat->cs, nlocal);
  pe_mpi_comm(stat->pe, &comm);

  /* Find the local minimum of the free energy density */

  fmin_local = FLT_MAX;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(stat->cs, ic, jc, kc);

	fe_symm_fed(stat->drop.fe, index, &fe);
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

        index = cs_index(stat->cs, ic, jc, kc);
	fe_symm_fed(stat->drop.fe, index, &fe);
        excess_local += (fe - fmin);
      }
    }
  }

  /* Reduce to rank zero for result */

  MPI_Reduce(&excess_local, &excess, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (nlocal[Z] == 1) {
    /* Assume 2d system */
    stat->drop.sigma = excess / (2.0*pi*stat->drop.radius);
  }
  else {
    stat->drop.sigma = excess / (4.0*pi*stat->drop.radius*stat->drop.radius);
  }

  return 0;
}
