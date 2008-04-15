/*****************************************************************************
 *
 *  phi_cahn_hilliard.c
 *
 *  The time evolution of the order parameter phi is described
 *  by the Cahn Hilliard equation
 *     d_t phi + div (u phi - M grad mu) = 0.
 *
 *  The equation is solved here via finite difference.
 *
 *  $Id: phi_cahn_hilliard.c,v 1.1.2.3 2008-04-15 18:06:29 kevin Exp $
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
#include "lattice.h"
#include "free_energy.h"
#include "phi.h"

#if (__STDC_VERSION__ < 199901)
int signbit(double);
#endif

extern double * phi_site;
static double * phi_flux;
static void phi_ch_compute_fluxes_upwind(void);
static void phi_ch_compute_fluxes_utopia(void);
static void phi_ch_compute_fluxes_upwind_seventh_order(void);
static void phi_ch_update_forward_step(void);

static void (* phi_ch_compute_fluxes)(void) = phi_ch_compute_fluxes_upwind;

static double mobility_; /* Order parameter mobility */

/*****************************************************************************
 *
 *  phi_cahn_hilliard
 *
 *  Driver routine.
 *
 *****************************************************************************/

void phi_cahn_hilliard() {

  int nlocal[3];
  int nsites;

  get_N_local(nlocal);
  nsites = (nlocal[X]+2*nhalo_)*(nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_);

  phi_flux = (double *) calloc(3*nsites, sizeof(double));
  if (phi_flux == NULL) fatal("calloc(phi_fluxes) failed");

  hydrodynamics_halo_u();
  phi_ch_compute_fluxes_upwind_seventh_order();
  /* phi_ch_compute_fluxes();*/
  phi_ch_update_forward_step();

  free(phi_flux);

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
 *  phi_ch_set_utopia
 *
 *****************************************************************************/

void phi_ch_set_utopia() {
  phi_ch_compute_fluxes = phi_ch_compute_fluxes_utopia;
  return;
}

/*****************************************************************************
 *
 *  phi_ch_set_upwind
 *
 ****************************************************************************/

void phi_ch_set_upwind() {
  phi_ch_compute_fluxes = phi_ch_compute_fluxes_upwind;
  return;
}

/*****************************************************************************
 *
 *  phi_ch_compute_fluxes_upwind
 *
 *  The fluxes (advective and diffusive) must be uniquely defined at
 *  the interfaces between the LB cells.
 *
 *  The faces are offset compared with the lattice, so care is needed
 *  with the indexing.
 *
 *****************************************************************************/

static void phi_ch_compute_fluxes_upwind() {

  int nlocal[3];
  int ic, jc, kc;            /* Counters over faces */
  int index0, index1;
  double u0[3], u1[3], u;
  double mu0, mu1;
  double phi0, phi;

  get_N_local(nlocal);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = get_site_index(ic, jc, kc);
	phi0 = phi_site[index0];
	mu0 = free_energy_get_chemical_potential(index0);
	hydrodynamics_get_velocity(index0, u0);
 
	/* x direction */
	index1 = get_site_index(ic+1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[X] + u1[X]);
	if (u < 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + X] = u*phi - mobility_*(mu1 - mu0);

	/* y direction */
	index1 = get_site_index(ic, jc+1, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[Y] + u1[Y]);
	if (u < 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + Y] = u*phi - mobility_*(mu1 - mu0);

	/* z direction */
	index1 = get_site_index(ic, jc, kc+1);
	hydrodynamics_get_velocity(index1, u1);

	u = 0.5*(u0[Z] + u1[Z]);
	if (u < 0.0) {
	  phi = phi_site[index1];
	}
	else {
	  phi = phi0;
	}

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + Z] = u*phi - mobility_*(mu1 - mu0);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_compute_fluxes_utopia
 *
 *  The fluxes (advective and diffusive) must be uniquely defined at
 *  the interfaces between the LB cells.
 *
 *  The faces are offset compared with the lattice, so care is needed
 *  with the indexing. In -d the flux[i] is the flux at east face of
 *  cell i.
 *
 *****************************************************************************/

static void phi_ch_compute_fluxes_utopia() {

  int nlocal[3];
  int ic, jc, kc;            /* Counters over faces */
  int index0, index1;
  int ia;
  double u0[3], u1[3], u[3];
  double mu0, mu1;
  double phi;
  double phi_w;
  double phi_ww;
  double phi_c;
  double phi_s, phi_sw, phi_se, phi_nw;
  double phi_wd, phi_cd, phi_wu;
  double phi_sd, phi_nwd, phi_swd, phi_swu;
  double phi_ss, phi_su, phi_sed, phi_cdd, phi_ed, phi_nd;

  double c[3];               /* |Courant numbers| */
  int    s[3];
  int    iup1, iup2, ido1;
  int    jup1, jup2, jdo1;
  int    kup1, kup2, kdo1;

  const double r2 = 1.0/2.0;
  const double r3 = 1.0/3.0;
  const double r4 = 1.0/4.0;
  const double r6 = 1.0/6.0;
  const double r8 = 1.0/8.0;

  get_N_local(nlocal);
  assert(nhalo_ >= 2);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = get_site_index(ic, jc, kc);
	mu0 = free_energy_get_chemical_potential(index0);
	hydrodynamics_get_velocity(index0, u0);
 
	/* x direction */
	index1 = get_site_index(ic+1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	for (ia = 0; ia < 3; ia++) {
	  u[ia] = r2*(u0[ia] + u1[ia]);
	  s[ia] = 1 - 2*signbit(u[ia]);
	  assert(s[ia] == -1 || s[ia] == +1);
	  c[ia] = u[ia]*s[ia];
	}

	ido1 = ic + (s[X]+1)/2;
	iup1 = ido1 - s[X];
	iup2 = iup1 - s[X];

	phi_c   = phi_site[get_site_index(ido1, jc, kc)];
	phi_w   = phi_site[get_site_index(iup1, jc, kc)];
	phi_ww  = phi_site[get_site_index(iup2, jc, kc)];

	phi_s   = phi_site[get_site_index(ido1, jc - s[Y], kc)];
	phi_sw  = phi_site[get_site_index(iup1, jc - s[Y], kc)];
	phi_nw  = phi_site[get_site_index(iup1, jc + s[Y], kc)];

	phi_wd  = phi_site[get_site_index(iup1, jc, kc - s[Z])];
	phi_cd  = phi_site[get_site_index(ido1, jc, kc - s[Z])];
	phi_wu  = phi_site[get_site_index(iup1, jc, kc + s[Z])];

	phi_sd  = phi_site[get_site_index(ido1, jc - s[Y], kc - s[Z])];
	phi_nwd = phi_site[get_site_index(iup1, jc + s[Y], kc - s[Z])];
	phi_swd = phi_site[get_site_index(iup1, jc - s[Y], kc - s[Z])];
	phi_swu = phi_site[get_site_index(iup1, jc - s[Y], kc + s[Z])];

	phi = r2*((phi_w + phi_c)
		  - c[X]*(phi_c - phi_w)
		  - r3*(1.0 - c[X]*c[X])*(phi_c - 2.0*phi_w + phi_ww))
	  - c[Y]*(r2*(phi_w - phi_sw)
		  + (r4 - r3*c[X])*(phi_c - phi_w - phi_s + phi_sw)
		  + (r4 - r6*c[Y])*(phi_nw - 2.0*phi_w + phi_sw))
	  - c[Z]*(r2*(phi_w - phi_wd)
		  + (r4 - r3*c[X])*(phi_c - phi_w - phi_cd + phi_wd)
		  + (r4 - r6*c[Y])*(phi_wu - 2.0*phi_w + phi_wd))
	  + c[Y]*c[Z]*
	  (r3*(phi_w - phi_wd) - r3*(phi_sw - phi_swd)
	   + (r6 - r4*c[X])*(phi_c - phi_w - phi_s + phi_sw
			     - (phi_cd - phi_wd - phi_sd + phi_swd))
	   + (r6 - r8*c[Y])*(phi_nw - 2.0*phi_w + phi_sw
			     - (phi_nwd - 2.0*phi_wd + phi_swd))
	   + (r6 - r8*c[Z])*(phi_wu - 2.0*phi_w + phi_wd
			     - (phi_swu - 2.0*phi_sw + phi_swd)));

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + X] = u[X]*phi - mobility_*(mu1 - mu0);

	/* y direction */
	index1 = get_site_index(ic, jc+1, kc);
	hydrodynamics_get_velocity(index1, u1);

	for (ia = 0; ia < 3; ia++) {
	  u[ia] = r2*(u0[ia] + u1[ia]);
	  s[ia] = 1 - 2*signbit(u[ia]);
	  assert(s[ia] == -1 || s[ia] == +1);
	  c[ia] = u[ia]*s[ia];
	}

	jdo1 = jc + (s[Y]+1)/2;
	jup1 = jdo1 - s[Y];
	jup2 = jup1 - s[Y];

	phi_c   = phi_site[get_site_index(ic, jdo1, kc)];
	phi_s   = phi_site[get_site_index(ic, jup1, kc)];
	phi_ss  = phi_site[get_site_index(ic, jup2, kc)];

	phi_w   = phi_site[get_site_index(ic - s[X], jdo1, kc)];
	phi_sw  = phi_site[get_site_index(ic - s[X], jup1, kc)];
	phi_se  = phi_site[get_site_index(ic + s[X], jup1, kc)];

	phi_cd  = phi_site[get_site_index(ic, jdo1, kc - s[Z])];
	phi_sd  = phi_site[get_site_index(ic, jup1, kc - s[Z])];
	phi_su  = phi_site[get_site_index(ic, jup1, kc + s[Z])];

	phi_swd = phi_site[get_site_index(ic - s[X], jup1, kc - s[Z])];
	phi_swu = phi_site[get_site_index(ic - s[X], jup1, kc + s[Z])];
	phi_wd  = phi_site[get_site_index(ic - s[X], jdo1, kc - s[Z])];
	phi_sed = phi_site[get_site_index(ic + s[X], jup1, kc - s[Z])];

	phi = r2*((phi_c + phi_s)
		  - c[Y]*(phi_c - phi_s)
		  - r3*(1.0 - c[Y]*c[Y])*(phi_c - 2.0*phi_s + phi_ss))
	  - c[X]*(r2*(phi_s - phi_sw)
		  + (r4 - r3*c[Y])*(phi_c - phi_s - phi_w + phi_sw)
		  + (r4 - r3*c[X])*(phi_se - 2.0*phi_s + phi_sw))
	  - c[Z]*(r2*(phi_s - phi_sd)
		  + (r4 - r3*c[Y])*(phi_c - phi_s - phi_cd + phi_sd)
		  + (r4 - r3*c[Z])*(phi_su - 2.0*phi_s + phi_sd))
	  + c[X]*c[Z]*
	  (r3*(phi_s - phi_sd) - r3*(phi_sw - phi_swd)
	   + (r6 - r8*c[X])*(phi_se - 2.0*phi_s + phi_sw
			     - (phi_sed - 2.0*phi_sd + phi_swd))
	   + (r6 - r4*c[Y])*(phi_c - phi_s - phi_w + phi_sw
			     - (phi_cd - phi_sd - phi_wd + phi_swd))
	   + (r6 - r8*c[Z])*(phi_su - 2.0*phi_s + phi_sd
			     - (phi_swu - 2.0*phi_sw + phi_swd)));

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + Y] = u[Y]*phi - mobility_*(mu1 - mu0);

	/* z direction */
	index1 = get_site_index(ic, jc, kc+1);
	hydrodynamics_get_velocity(index1, u1);

	for (ia = 0; ia < 3; ia++) {
	  u[ia] = r2*(u0[ia] + u1[ia]);
	  s[ia] = 1 - 2*signbit(u[ia]);
	  assert(s[ia] == -1 || s[ia] == +1);
	  c[ia] = u[ia]*s[ia];
	}

	kdo1 = kc + (s[Z]+1)/2;
	kup1 = kdo1 - s[Z];
	kup2 = kup1 - s[Z];

	phi_c   = phi_site[get_site_index(ic, jc, kdo1)];
	phi_cd  = phi_site[get_site_index(ic, jc, kup1)];
	phi_cdd = phi_site[get_site_index(ic, jc, kup2)];

	phi_wd  = phi_site[get_site_index(ic - s[X], jc, kup1)];
	phi_w   = phi_site[get_site_index(ic - s[X], jc, kdo1)];
	phi_ed  = phi_site[get_site_index(ic + s[X], jc, kdo1)];

	phi_s   = phi_site[get_site_index(ic, jc - s[Y], kdo1)];
	phi_sd  = phi_site[get_site_index(ic, jc - s[Y], kup1)];
	phi_nd  = phi_site[get_site_index(ic, jc + s[Y], kup1)];

	phi_swd = phi_site[get_site_index(ic - s[X], jc - s[Y], kup1)];
	phi_sed = phi_site[get_site_index(ic + s[X], jc - s[Y], kup1)];
	phi_nwd = phi_site[get_site_index(ic - s[X], jc + s[Y], kup1)];
	phi_sw  = phi_site[get_site_index(ic - s[X], jc - s[Y], kdo1)];

	phi = r2*((phi_c + phi_cd)
		  - c[Z]*(phi_c - phi_cd)
		  - r3*(1.0 - c[Z]*c[Z])*(phi_c - 2.0*phi_cd + phi_cdd))
	  - c[X]*(r2*(phi_cd - phi_wd)
		  + (r4 - r6*c[Z])*(phi_c - phi_cd - phi_w + phi_wd)
		  + (r4 - r6*c[X])*(phi_ed - 2.0*phi_cd + phi_wd))
	  - c[Y]*(r2*(phi_cd - phi_sd)
		  + (r4 - r6*c[Z])*(phi_c - phi_cd - phi_s + phi_sd)
		  + (r4 - r6*c[Y])*(phi_nd - 2.0*phi_cd + phi_sd))
	  + c[X]*c[Y]*
		  (r3*(phi_cd - phi_sd) - r3*(phi_wd - phi_swd)
		   + (r6 - r8*c[X])*(phi_ed - 2.0*phi_cd + phi_wd
				     - (phi_sed - 2.0*phi_sd + phi_swd))
		   + (r6 - r8*c[Y])*(phi_nd - 2.0*phi_cd + phi_sd
				     - (phi_nwd - 2.0*phi_wd + phi_swd))
		   + (r6 - r4*c[Z])*(phi_c - phi_cd - phi_w + phi_wd
				     - (phi_s - phi_sd - phi_sw + phi_swd)));

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + Z] = u[Z]*phi - mobility_*(mu1 - mu0);
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
  int ic, jc, kc, index;
  int xfac, yfac, zfac;
  double dphi;

  get_N_local(nlocal);
  xfac = (nlocal[Y] + 2*nhalo_)*(nlocal[Z] + 2*nhalo_);
  yfac = (nlocal[Z] + 2*nhalo_);
  zfac = 1;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	dphi = phi_flux[3*index + X] - phi_flux[3*(index-xfac) + X]
	     + phi_flux[3*index + Y] - phi_flux[3*(index-yfac) + Y]
	     + phi_flux[3*index + Z] - phi_flux[3*(index-zfac) + Z];

	phi_site[index] -= dphi;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_compute_fluxes_upwind_seventh_order
 *
 *  Seventh order upwind advective fluxes require a halo of at
 *  least 4 points.
 *
 *  Side effects:
 *    - the flux array phi_flux_ is overwritten with the fluxes.
 *
 *****************************************************************************/

static void phi_ch_compute_fluxes_upwind_seventh_order() {

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

	index0 = get_site_index(ic, jc, kc);
	mu0 = free_energy_get_chemical_potential(index0);
	hydrodynamics_get_velocity(index0, u0);
 
	/* x direction */
	index1 = get_site_index(ic+1, jc, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = r2*(u0[X] + u1[X]);
	s = 1 - 2*signbit(u);
	assert(s == -1 || s == +1);
	up = ic + (s+1)/2;

	phi = rfactorial7*(
	    am4*phi_site[get_site_index(up - 4*s, jc, kc)]
	  + am3*phi_site[get_site_index(up - 3*s, jc, kc)]
	  + am2*phi_site[get_site_index(up - 2*s, jc, kc)]
	  + am1*phi_site[get_site_index(up - 1*s, jc, kc)]
	  + ap0*phi_site[get_site_index(up      , jc, kc)]
	  + ap1*phi_site[get_site_index(up + 1*s, jc, kc)]
	  + ap2*phi_site[get_site_index(up + 2*s, jc, kc)]);

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + X] = u*phi - mobility_*(mu1 - mu0);

	/* y direction */
	index1 = get_site_index(ic, jc+1, kc);
	hydrodynamics_get_velocity(index1, u1);

	u = r2*(u0[Y] + u1[Y]);
	s = 1 - 2*signbit(u);
	assert(s == -1 || s == +1);
	up = jc + (s+1)/2;

	phi = rfactorial7*(
	    am4*phi_site[get_site_index(ic, up - 4*s, kc)]
	  + am3*phi_site[get_site_index(ic, up - 3*s, kc)]
	  + am2*phi_site[get_site_index(ic, up - 2*s, kc)]
	  + am1*phi_site[get_site_index(ic, up - 1*s, kc)]
	  + ap0*phi_site[get_site_index(ic, up      , kc)]
	  + ap1*phi_site[get_site_index(ic, up + 1*s, kc)]
	  + ap2*phi_site[get_site_index(ic, up + 2*s, kc)]);

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + Y] = u*phi - mobility_*(mu1 - mu0);

	/* z direction */
	index1 = get_site_index(ic, jc, kc+1);
	hydrodynamics_get_velocity(index1, u1);

	u = r2*(u0[Z] + u1[Z]);
	s = 1 - 2*signbit(u);
	assert(s == -1 || s == +1);
	up = kc + (s+1)/2;

	phi = rfactorial7*(
	    am4*phi_site[get_site_index(ic, jc, up - 4*s)]
	  + am3*phi_site[get_site_index(ic, jc, up - 3*s)]
	  + am2*phi_site[get_site_index(ic, jc, up - 2*s)]
	  + am1*phi_site[get_site_index(ic, jc, up - 1*s)]
	  + ap0*phi_site[get_site_index(ic, jc, up      )]
	  + ap1*phi_site[get_site_index(ic, jc, up + 1*s)]
	  + ap2*phi_site[get_site_index(ic, jc, up + 2*s)]);

	mu1 = free_energy_get_chemical_potential(index1);
	phi_flux[3*index0 + Z] = u*phi - mobility_*(mu1 - mu0);
      }
    }
  }

  return;
}
