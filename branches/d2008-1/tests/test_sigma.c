/*****************************************************************************
 *
 *  test_sigma.c
 *
 *  System test to compute the surface tension via measurement of the
 *  Laplace pressure difference between the inside an outside of a
 *  stationary droplet.
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "collision.h"
#include "utilities.h"
#include "propagation.h"
#include "free_energy.h"
#include "phi.h"
#include "phi_gradients.h"
#include "phi_cahn_hilliard.h"
#include "physics.h"

struct drop_t {
  double radius;
  double zeta;
  double centre[3];
  double phimax;
  double laplace_in, laplace_out;
};

void drop_init(struct drop_t);
void drop_locate_centre(struct drop_t *);
void drop_locate_radius(struct drop_t *);
void drop_locate_profile(struct drop_t);
void drop_laplace_pressure_difference(struct drop_t *);
void drop_relax(void);
double sigma(struct drop_t);

int main (int argc, char ** argv) {

  struct drop_t drop0;

  drop0.radius = L(X)/4.0;
  drop0.zeta = interfacial_width();
  drop0.centre[X] = L(X)/2.0;
  drop0.centre[Y] = L(Y)/2.0;
  drop0.centre[Z] = L(Z)/2.0;
  drop0.phimax    = 1.0;

  pe_init(argc, argv);
  coords_init();

  set_eta(1.41);
  free_energy_set_A(-0.083);
  free_energy_set_B( 0.083);
  free_energy_set_kappa(0.053);

  MODEL_init();
  phi_ch_set_mobility(0.05);

  /* Compute the surface tension for a given parameter set */
  sigma(drop0);

  phi_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  sigma
 *
 *****************************************************************************/

double sigma(struct drop_t drop0) {

  struct drop_t drop1;
  double value;

  drop1 = drop0;

  drop_init(drop0);

  phi_compute_phi_site();
  phi_halo();
  drop_locate_centre(&drop1);
  drop_locate_radius(&drop1);

  phi_gradients_compute();
  drop_laplace_pressure_difference(&drop1);
  drop_locate_profile(drop1);

  info("Initial drop: %.3f %.3f %.3f with radius %.3f\n",
       drop0.centre[X], drop0.centre[Y], drop0.centre[Z], drop0.radius);
  info("Laplace drop: %f %f\n", drop1.laplace_in, drop1.laplace_out);
  info("Sigma: %6f\n", (drop1.laplace_in - drop1.laplace_out)*drop1.radius);

  drop_relax();

  phi_compute_phi_site();
  phi_halo();

  drop_locate_centre(&drop1);
  drop_locate_radius(&drop1);

  phi_gradients_compute();
  drop_laplace_pressure_difference(&drop1);
  drop_locate_profile(drop1);

  info("  Final drop: %.3f %.3f %.3f with radius %.3f\n",
       drop1.centre[X], drop1.centre[Y], drop1.centre[Z], drop1.radius);
  info("Sigma: %6f\n", (drop1.laplace_in - drop1.laplace_out)*drop1.radius);
  info("Sigma theory: %f\n", surface_tension());

  return value;
}

/*****************************************************************************
 *
 *  drop_init
 *
 *****************************************************************************/

void drop_init(struct drop_t drop_in) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc, p;

  double position[3];
  double phi, r, rzeta;

  get_N_local(nlocal);
  get_N_offset(noffset);

  rzeta = 1.0/drop_in.zeta;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);
	position[X] = 1.0*(noffset[X] + ic) - drop_in.centre[X];
	position[Y] = 1.0*(noffset[Y] + jc) - drop_in.centre[Y];
	position[Z] = 1.0*(noffset[Z] + kc) - drop_in.centre[Z];

	r = sqrt(dot_product(position, position));

	phi = drop_in.phimax*tanh(rzeta*(r - drop_in.radius));

	/* Set both phi_site and g to allow for FD or LB */
	phi_set_phi_site(index, phi);

	set_rho(index, 1.0);
	set_g_at_site(index, 0, phi);
	for (p = 1; p < NVEL; p++) {
	  set_g_at_site(index, p, 0.0);
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  drop_locate_centre
 *
 *  We assume there is a droplet in the system. Work out where the
 *  centre is.
 *
 *****************************************************************************/

void drop_locate_centre(struct drop_t * drop_out) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;
  double c[3+1];              /* 3 space dimensions plus counter */
  double ctotal[3+1];
  double phi;

  get_N_local(nlocal);
  get_N_offset(noffset);

  for (ic = 0; ic <= 3; ic++) {
    c[ic] =0.0;
    ctotal[ic] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

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

  drop_out->centre[X] = ctotal[X]/ctotal[3];
  drop_out->centre[Y] = ctotal[Y]/ctotal[3];
  drop_out->centre[Z] = ctotal[Z]/ctotal[3];

  return;
}

/*****************************************************************************
 *
 *  drop_locate_radius
 *
 *  We assume there is a droplet in the system with a given centre.
 *  Estimate its radius.
 *
 *****************************************************************************/

void drop_locate_radius(struct drop_t * drop_out) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc;
  double result[2], result_total[2];

  get_N_local(nlocal);
  get_N_offset(noffset);

  result[0] = 0.0;
  result[1] = 0.0;
  result_total[0] = 0.0;
  result_total[1] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	int index, ip, jp, kp;
	double phi;

	index = get_site_index(ic, jc, kc);
	phi = phi_get_phi_site(index);

	/* Look around at the neighbours */

	for (ip = ic-1; ip <= ic+1; ip++) {
	  for (jp = jc-1; jp <= jc+1; jp++) {
	    for (kp = kc-1; kp <= kc+1; kp++) {

	      double phi1, fraction, r[3];

	      if (!(ip || jp || kp)) continue;
	      index = get_site_index(ip, jp, kp);
	      phi1 = phi_get_phi_site(index);

	      /* Look for change in sign */
	      if (phi < 0.0 && phi1 > 0.0) {
		fraction = phi / (phi - phi1);
		r[X] = 1.0*(noffset[X] + ic) + fraction*(ip-ic);
		r[Y] = 1.0*(noffset[Y] + jc) + fraction*(jp-jc);
		r[Z] = 1.0*(noffset[Z] + kc) + fraction*(kp-kc);
		r[X] -= drop_out->centre[X];
		r[Y] -= drop_out->centre[Y];
		r[Z] -= drop_out->centre[Z];

		result[0] += sqrt(dot_product(r, r));
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

  drop_out->radius = result_total[0]/result_total[1];

  return;
}

/*****************************************************************************
 *
 *  drop_laplace_pressure_difference
 *
 *  Work out the mean pressure inside and outside the droplet.
 *
 *****************************************************************************/

void drop_laplace_pressure_difference(struct drop_t * drop) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double p0, phi;
  double result[4], result_total[4];

  get_N_local(nlocal);
  get_N_offset(noffset);

  result[0] = 0.0;
  result[1] = 0.0;
  result[2] = 0.0;
  result[3] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);
	phi = phi_get_phi_site(index);
	p0 = free_energy_get_isotropic_pressure(index);

	if (phi < -0.1) { /* inside */
	  result[0] += p0;
	  result[1] += 1.0;
	}
	if (phi > 0.1) { /* outside */
	  result[2] += p0;
	  result[3] += 1.0;
	}
	
      }
    }
  }

  MPI_Allreduce(result, result_total, 4, MPI_DOUBLE, MPI_SUM, cart_comm());

  drop->laplace_in  = result_total[0]/result_total[1];
  drop->laplace_out = result_total[2]/result_total[3];

  return;
}

/*****************************************************************************
 *
 *  drop_relax
 *
 *  Relax the droplet by running some steps.
 *
 *****************************************************************************/

void drop_relax() {

  int nt, ntmax = 20;

  info("\nRunning %d steps to relax\n", ntmax);

  for (nt = 0; nt < ntmax; nt++) {
    hydrodynamics_zero_force();
    collide();
    halo_site();
    propagation();
  }

  return;
}

/*****************************************************************************
 *
 *  drop_locate_profile
 *
 *  Construct a radial profile of the order parameter near the droplet
 *  interface. This should show the tanh profile.
 *
 *****************************************************************************/

void drop_locate_profile(struct drop_t drop) {

  int nlocal[3], noffset[3];
  int ic, jc, kc, index, n;
  const int nbin = 32;
  double rmin;
  double rmax;
  double r0, dr, r[3];
  double phimean[nbin], phimeant[nbin];
  double emean[nbin], emeant[nbin];
  double count[nbin], countt[nbin];

  get_N_local(nlocal);
  get_N_offset(noffset);

  /* Set the bin widths etc */

  rmin = drop.radius - 3.0*drop.zeta;
  rmax = drop.radius + 3.0*drop.zeta;
  dr = (rmax - rmin)/nbin;

  for (n = 0; n < nbin; n++) {
    phimean[n] = 0.0;
    emean[n] = 0.0;
    count[n] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	/* Work out the displacement from the drop centre
	   and hence the bin number */

	r[X] = 1.0*(noffset[X] + ic) - drop.centre[X];
	r[Y] = 1.0*(noffset[Y] + jc) - drop.centre[Y];
	r[Z] = 1.0*(noffset[Z] + kc) - drop.centre[Z];
	r0 = sqrt(dot_product(r, r));
	n = (r0-rmin)/dr;

	if (n >= 0 && n < nbin) {
	  phimean[n] += phi_get_phi_site(index);
	  emean[n]   += free_energy_density(index);
	  count[n]   += 1.0;
	}
      }
    }
  }

  MPI_Reduce(phimean, phimeant, nbin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(emean, emeant, nbin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(count, countt, nbin, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  for (n = 0; n < nbin; n++) {
    r0 = rmin + (n + 0.5)*dr;
    info(" %f %f %f\n", r0, phimeant[n]/countt[n], emeant[n]/countt[n]);
  }

  return;
}
