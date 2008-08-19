/*****************************************************************************
 *
 *  phi_cahn_hilliard.c
 *
 *  The time evolution of the order parameter phi is described
 *  by the Cahn Hilliard equation
 *     d_t phi + div (u phi - M grad mu) = 0.
 *
 *  The equation is solved here via finite difference. The velocity
 *  field u is assumed known from the hydrodynamic sector. M is the
 *  order parameter mobility. The chemical potential mu is set via
 *  the choice of free energy.
 *
 *  $Id: phi_cahn_hilliard.c,v 1.1.2.9 2008-08-19 17:04:21 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "site_map.h"
#include "lattice.h"
#include "free_energy.h"
#include "phi.h"

extern double * phi_site;
static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static void phi_ch_upwind(void);
static void phi_ch_upwind_third_order(void);
static void phi_ch_upwind_seventh_order(void);
static void phi_ch_update_forward_step(void);
static void phi_ch_update_forward_step_with_solid(void);

static void (* phi_ch_compute_fluxes)(void) = phi_ch_upwind;
static int signbit_double(double);

static double mobility_; /* Order parameter mobility */

/*****************************************************************************
 *
 *  phi_cahn_hilliard
 *
 *  Compute the fluxes (advective/diffusive) and compute the update
 *  to phi_site[].
 *
 *  Conservation is ensured by face-flux uniqueness. However, in the
 *  x-direction, the fluxes at the east face and west face of a given
 *  cell must be handled spearately to take account of Lees Edwards
 *  boundaries.
 *
 *****************************************************************************/

void phi_cahn_hilliard() {

  int nlocal[3];
  int nsites;

  get_N_local(nlocal);
  nsites = (nlocal[X]+2*nhalo_)*(nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);

  fluxe = (double *) calloc(nsites, sizeof(double));
  fluxw = (double *) calloc(nsites, sizeof(double));
  fluxy = (double *) calloc(nsites, sizeof(double));
  fluxz = (double *) calloc(nsites, sizeof(double));
  if (fluxe == NULL) fatal("calloc(fluxe) failed");
  if (fluxw == NULL) fatal("calloc(fluxw) failed");
  if (fluxy == NULL) fatal("calloc(fluxy) failed");
  if (fluxz == NULL) fatal("calloc(fluxz) failed");


  hydrodynamics_halo_u();
  hydrodynamics_leesedwards_transformation();

  /* phi_ch_upwind_seventh_order();*/
  /* phi_ch_compute_fluxes();*/
  phi_ch_upwind_third_order();
  phi_ch_update_forward_step();

  free(fluxe);
  free(fluxw);
  free(fluxy);
  free(fluxz);

  return;
}

/*****************************************************************************
 *
 *  phi_ch_get_mobility
 *
 *****************************************************************************/

double phi_ch_get_mobility() {
  return mobility_;
}

/*****************************************************************************
 *
 *  phi_ch_set_mobility
 *
 *****************************************************************************/

void phi_ch_set_mobility(const double m) {
  assert(m >= 0.0);
  mobility_ = m;
  return;
}

/*****************************************************************************
 *
 *  phi_ch_set_upwind_order
 *
 ****************************************************************************/

void phi_ch_set_upwind_order(int n) {

  switch (n) {
  case 3:
    phi_ch_compute_fluxes = phi_ch_upwind_third_order;
    break;
  case 7:
    phi_ch_compute_fluxes = phi_ch_upwind_seventh_order;
    break;
  default:
    phi_ch_compute_fluxes = phi_ch_upwind;
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_upwind
 *
 *  The fluxes (advective and diffusive) must be uniquely defined at
 *  the interfaces between the LB cells.
 *
 *  The faces are offset compared with the lattice, so care is needed
 *  with the indexing.
 *
 *****************************************************************************/

static void phi_ch_upwind() {

  int nlocal[3];
  int ic, jc, kc;            /* Counters over faces */
  int index0, index1;
  int icm1, icp1;
  double u0[3], u1[3], u;
  double mu0, mu1;
  double phi0, phi;

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);
	phi0 = phi_site[index0];

	mu0 = free_energy_get_chemical_potential(index0);
	hydrodynamics_get_velocity(index0, u0);

	/* west face (icm1 and ic) */
	index1 = ADDR(icm1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[X] + u1[X]);
	if (u > 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxw[index0] = u*phi - mobility_*(mu0 - mu1);

	/* east face (ic and icp1) */
	index1 = ADDR(icp1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[X] + u1[X]);
	if (u < 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxe[index0] = u*phi - mobility_*(mu1 - mu0);



	/* y direction */
	index1 = le_site_index(ic, jc+1, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[Y] + u1[Y]);
	if (u < 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxy[index0] = u*phi - mobility_*(mu1 - mu0);

	/* z direction */
	index1 = ADDR(ic, jc, kc+1);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[Z] + u1[Z]);
	if (u < 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxz[index0] = u*phi - mobility_*(mu1 - mu0);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_get_fluxes_upwind_third_order
 *
 *  Compute third order advective fluxes.
 *
 *****************************************************************************/

static void phi_ch_upwind_third_order() {

  int nlocal[3];
  int ic, jc, kc;            /* Counters over faces */
  int index0, index1;
  int icm2, icm1, icp1, icp2;
  double u0[3], u1[3], u;
  double mu0, mu1;
  double phi0, phi;

  const double a1 = -0.213933;
  const double a2 =  0.927865;
  const double a3 =  0.286067;

  get_N_local(nlocal);
  assert(nhalo_ >= 2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm2 = le_index_real_to_buffer(ic, -2);
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    icp2 = le_index_real_to_buffer(ic, +2);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);
	phi0 = phi_site[index0];

	mu0 = free_energy_get_chemical_potential(index0);
	hydrodynamics_get_velocity(index0, u0);

	/* west face (icm1 and ic) */
	index1 = ADDR(icm1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[X] + u1[X]);
	if (u > 0.0) {
	  phi = a1*phi_site[ADDR(icm2,jc,kc)] + a2*phi_site[index1] + a3*phi0;
	}
	else {
	  phi = a1*phi_site[ADDR(icp1,jc,kc)] + a2*phi0 + a3*phi_site[index1];
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxw[index0] = u*phi - mobility_*(mu0 - mu1);

	/* east face (ic and icp1) */
	index1 = ADDR(icp1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[X] + u1[X]);
	if (u < 0.0) {
	  phi = a1*phi_site[ADDR(icp2,jc,kc)] + a2*phi_site[index1] + a3*phi0;
	}
	else {
	  phi = a1*phi_site[ADDR(icm1,jc,kc)] + a2*phi0 + a3*phi_site[index1];
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxe[index0] = u*phi - mobility_*(mu1 - mu0);



	/* y direction */
	index1 = le_site_index(ic, jc+1, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[Y] + u1[Y]);
	if (u < 0.0) {
	  phi = a1*phi_site[ADDR(ic,jc+2,kc)] + a2*phi_site[index1] + a3*phi0;
	}
	else {
	  phi = a1*phi_site[ADDR(ic,jc-1,kc)] + a2*phi0 + a3*phi_site[index1];
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxy[index0] = u*phi - mobility_*(mu1 - mu0);

	/* z direction */
	index1 = ADDR(ic, jc, kc+1);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[Z] + u1[Z]);
	if (u < 0.0) {
	  phi = a1*phi_site[ADDR(ic,jc,kc+2)] + a2*phi_site[index1] + a3*phi0;
	}
	else {
	  phi = a1*phi_site[ADDR(ic,jc,kc-1)] + a2*phi0 + a3*phi_site[index1];
	}

	mu1 = free_energy_get_chemical_potential(index1);
	fluxz[index0] = u*phi - mobility_*(mu1 - mu0);
      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  phi_ch_update_forward_step
 *
 *  Update phi_site at each site in turn via the divergence of the
 *  fluxes. This is an Euler forward step:
 *
 *  phi new = phi old - dt*(flux_out - flux_in)
 *
 *  The time step is the LB time step dt = 1.
 *
 *****************************************************************************/

static void phi_ch_update_forward_step() {

  int nlocal[3];
  int ic, jc, kc;
  double dphi;

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	dphi = fluxe[ADDR(ic,jc,kc)] - fluxw[ADDR(ic,jc,kc)]
	  + fluxy[ADDR(ic,jc,kc)] - fluxy[ADDR(ic,jc-1,kc)]
	  + fluxz[ADDR(ic,jc,kc)] - fluxz[ADDR(ic,jc,kc-1)];

	phi_site[ADDR(ic,jc,kc)] -= dphi;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *****************************************************************************/
static void phi_ch_update_forward_step_with_solid() {

  int nlocal[3];
  int ic, jc, kc, index;
  double mask;
  double dphi;

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	if (site_map_get_status_index(index) != FLUID) continue;

	dphi = 0.0;
	mask = (site_map_get_status(ic+1, jc, kc) == FLUID);
	dphi += mask*fluxe[index];
	mask = (site_map_get_status(ic-1, jc, kc) == FLUID);
	dphi -= mask*fluxw[index];

	mask = (site_map_get_status(ic, jc+1, kc) == FLUID);
	dphi += mask*fluxy[index];
	mask = (site_map_get_status(ic, jc-1, kc) == FLUID);
	dphi -= mask*fluxy[ADDR(ic, jc-1, kc)];

	mask = (site_map_get_status(ic, jc, kc+1) == FLUID);
	dphi += mask*fluxz[index];
	mask = (site_map_get_status(ic, jc, kc-1) == FLUID);
	dphi -= mask*fluxz[ADDR(ic, jc, kc-1)];

	phi_site[index] -= dphi;
      }
    }
  }

  return;
}



/*****************************************************************************
 *
 *  phi_ch_upwind_seventh_order
 *
 *  Seventh order upwind advective fluxes require a halo of at
 *  least 4 points.
 *
 *  Side effects:
 *    - the flux array phi_flux_ is overwritten with the fluxes.
 *
 *****************************************************************************/

static void phi_ch_upwind_seventh_order() {

  int nlocal[3];
  int ic, jc, kc;            /* Counters over faces */
  int index0, index1;
  double u0[3], u1[3], u;
  double mu0, mu1;
  double phi;

  int s, up;

  /* Stencil has 7 points with weights axx/7! */
  const double am4 = -36.0;
  const double am3 = +300.0;
  const double am2 = -1212.0;
  const double am1 = +3828.0;
  const double ap0 = +2568.0;
  const double ap1 = -456.0;
  const double ap2 = +48.0;
  const double rfactorial7 = 1.0/(1*2*3*4*5*6*7);
  const double r2 = 1.0/2.0;

  get_N_local(nlocal);
  assert(nhalo_ >= 4);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);
	mu0 = free_energy_get_chemical_potential(index0);
	hydrodynamics_get_velocity(index0, u0);
 
	/* x direction */
	index1 = ADDR(ic+1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = r2*(u0[X] + u1[X]);
	s = 1 - 2*signbit_double(u);
	assert(s == -1 || s == +1);
	up = ic + (s+1)/2;

	phi = rfactorial7*(
	    am4*phi_site[ADDR(up - 4*s, jc, kc)]
	  + am3*phi_site[ADDR(up - 3*s, jc, kc)]
	  + am2*phi_site[ADDR(up - 2*s, jc, kc)]
	  + am1*phi_site[ADDR(up - 1*s, jc, kc)]
	  + ap0*phi_site[ADDR(up      , jc, kc)]
	  + ap1*phi_site[ADDR(up + 1*s, jc, kc)]
	  + ap2*phi_site[ADDR(up + 2*s, jc, kc)]);

	mu1 = free_energy_get_chemical_potential(index1);
	fluxe[index0] = u*phi - mobility_*(mu1 - mu0);

	/* y direction */
	index1 = ADDR(ic, jc+1, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = r2*(u0[Y] + u1[Y]);
	s = 1 - 2*signbit_double(u);
	assert(s == -1 || s == +1);
	up = jc + (s+1)/2;

	phi = rfactorial7*(
	    am4*phi_site[ADDR(ic, up - 4*s, kc)]
	  + am3*phi_site[ADDR(ic, up - 3*s, kc)]
	  + am2*phi_site[ADDR(ic, up - 2*s, kc)]
	  + am1*phi_site[ADDR(ic, up - 1*s, kc)]
	  + ap0*phi_site[ADDR(ic, up      , kc)]
	  + ap1*phi_site[ADDR(ic, up + 1*s, kc)]
	  + ap2*phi_site[ADDR(ic, up + 2*s, kc)]);

	mu1 = free_energy_get_chemical_potential(index1);
	fluxy[index0] = u*phi - mobility_*(mu1 - mu0);

	/* z direction */
	index1 = ADDR(ic, jc, kc+1);
	hydrodynamics_get_velocity(index1, u1);

	u = r2*(u0[Z] + u1[Z]);
	s = 1 - 2*signbit_double(u);
	assert(s == -1 || s == +1);
	up = kc + (s+1)/2;

	phi = rfactorial7*(
	    am4*phi_site[ADDR(ic, jc, up - 4*s)]
	  + am3*phi_site[ADDR(ic, jc, up - 3*s)]
	  + am2*phi_site[ADDR(ic, jc, up - 2*s)]
	  + am1*phi_site[ADDR(ic, jc, up - 1*s)]
	  + ap0*phi_site[ADDR(ic, jc, up      )]
	  + ap1*phi_site[ADDR(ic, jc, up + 1*s)]
	  + ap2*phi_site[ADDR(ic, jc, up + 2*s)]);

	mu1 = free_energy_get_chemical_potential(index1);
	fluxz[index0] = u*phi - mobility_*(mu1 - mu0);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  signbit_double function
 *
 *  Return 0 for +ve or zero argument, 1 for negative.
 *
 ****************************************************************************/

int signbit_double(double u) {

  int sign = 0;

  if (u < 0.0) sign = +1;

  return sign;
}
