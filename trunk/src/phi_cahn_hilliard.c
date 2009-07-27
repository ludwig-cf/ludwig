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
 *  $Id: phi_cahn_hilliard.c,v 1.6 2009-07-27 13:48:34 kevin Exp $
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
#include "advection.h"
#include "lattice.h"
#include "free_energy.h"
#include "phi.h"

extern double * phi_site;
static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

void phi_ch_diffusive_flux(void);
void phi_ch_diffusive_flux_surfactant(void);
static void phi_ch_correct_fluxes_for_solid(void);
static void phi_ch_update_forward_step(void);
static void phi_ch_langmuir_hinshelwood(void);

static double * mobility_;   /* Order parameter mobilities */
static double   lh_kplus_;   /* Langmuir Hinshelwood adsorption rate */
static double   lh_kminus_;  /* Langmuir Hinshelwood desorption rate */
static double   lh_psimax_;  /* Langmuir Hinshelwood monolayer capacity */

static int langmuirh_ = 0; /* Langmuir Hinshelwood flag */
static int advection_ = 1; /* Advection scheme */
static int solid_     = 0; /* Solid present? */

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

  fluxe = (double *) malloc(nop_*nsites*sizeof(double));
  fluxw = (double *) malloc(nop_*nsites*sizeof(double));
  fluxy = (double *) malloc(nop_*nsites*sizeof(double));
  fluxz = (double *) malloc(nop_*nsites*sizeof(double));
  if (fluxe == NULL) fatal("malloc(fluxe) failed");
  if (fluxw == NULL) fatal("malloc(fluxw) failed");
  if (fluxy == NULL) fatal("malloc(fluxy) failed");
  if (fluxz == NULL) fatal("malloc(fluxz) failed");


  hydrodynamics_halo_u();
  hydrodynamics_leesedwards_transformation();

  switch (advection_) {
  case 1:
    advection_upwind(fluxe, fluxw, fluxy, fluxz);
    break;
  case 3:
    advection_upwind_third_order(fluxe, fluxw, fluxy, fluxz);
    break;
  case 5:
    advection_upwind_fifth_order(fluxe, fluxw, fluxy, fluxz);
    break;
  case 7:
    advection_upwind_seventh_order(fluxe, fluxw, fluxy, fluxz);
    break;
  }

  if (nop_ == 2) {
    phi_ch_diffusive_flux_surfactant();
  }
  else {
    phi_ch_diffusive_flux();
  }

  if (langmuirh_) {
    phi_ch_langmuir_hinshelwood();
  }
  else {
    if (solid_) phi_ch_correct_fluxes_for_solid();
  }

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
  assert(mobility_);
  return mobility_[0];
}

/*****************************************************************************
 *
 *  phi_ch_set_mobility
 *
 *****************************************************************************/

void phi_ch_set_mobility(const double m) {

  if (mobility_ == NULL) {
    mobility_ = (double *) malloc(nop_*sizeof(double));
    if (mobility_ == NULL) fatal("malloc(mobility) failed\n");
  }

  assert(m >= 0.0);
  mobility_[0] = m;

  return;
}

/*****************************************************************************
 *
 *  phi_ch_set_op_mobility
 *
 *  As above, but for general order parameter.
 *
 *****************************************************************************/

void phi_ch_op_set_mobility(const double m, const int nop) {

  assert(m >= 0);
  assert(nop < nop_);

  if (mobility_ == NULL) {
    mobility_ = (double *) malloc(nop_*sizeof(double));
    if (mobility_ == NULL) fatal("malloc(mobility) failed\n");
  }

  assert(m >= 0.0);
  mobility_[nop] = m;

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
    advection_ = 3;
    info("Using third order upwind\n");
    break;
  case 5:
    advection_ = 5;
    info("Using fifth order upwind\n");
    break;
  case 7:
    advection_ = 7;
    break;
  default:
    advection_ = 1;
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_set_langmuir_hinshelwood
 *
 *  Set the Langmuir Hinshelwood parameters and sets the flag.
 *
 *****************************************************************************/

void phi_ch_set_langmuir_hinshelwood(double kplus, double kminus,
				     double psimax) {
  lh_kplus_  = kplus;
  lh_kminus_ = kminus;
  lh_psimax_ = psimax;
  langmuirh_ = 1;

  return;
}

/*****************************************************************************
 *
 *  phi_ch_diffusive_flux
 *
 *  Accumulate [add to a previously computed advective flux] the
 *  'diffusive' contribution related to the chemical potential. It's
 *  computed everywhere regardless of fluid/solid status.
 *
 *  This is a two point stencil the in the chemical potential,
 *  and the mobility is constant.
 *
 *****************************************************************************/

void phi_ch_diffusive_flux(void) {

  int nlocal[3];
  int ic, jc, kc, n;
  int index0, index1;
  int icm1, icp1;
  double mu0, mu1;
  double mobility;

  get_N_local(nlocal);
  assert(nhalo_ >= 2);
  assert(mobility_);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);

	for (n = 0; n < nop_; n++) {
	  mobility = mobility_[n];

	  mu0 = free_energy_chemical_potential(index0, n);

	  /* x-direction (between ic-1 and ic) */

	  index1 = ADDR(icm1, jc, kc);
	  mu1 = free_energy_chemical_potential(index1, n);
	  fluxw[nop_*index0 + n] -= mobility*(mu0 - mu1);

	  /* ...and between ic and ic+1 */

	  index1 = ADDR(icp1, jc, kc);
	  mu1 = free_energy_chemical_potential(index1, n);
	  fluxe[nop_*index0 + n] -= mobility*(mu1 - mu0);

	  /* y direction */

	  index1 = le_site_index(ic, jc+1, kc);
	  mu1 = free_energy_chemical_potential(index1, n);
	  fluxy[nop_*index0 + n] -= mobility*(mu1 - mu0);

	  /* z direction */

	  index1 = ADDR(ic, jc, kc+1);
	  mu1 = free_energy_chemical_potential(index1, n);
	  fluxz[nop_*index0 + n] -= mobility*(mu1 - mu0);
	}

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_diffusive_flux_surfactant
 *
 *  Analogue of the above for the surfactant model where the
 *  compositional order parameter (n = 0) mobilty is fixed,
 *  but that for the surfactant (n = 1) varies as
 *
 *  D_\psi = M_\psi \psi ( 1 - \psi).
 *
 *  This is again a two point stencil the in the chemical potential
 *  for each face.
 *
 *****************************************************************************/

void phi_ch_diffusive_flux_surfactant(void) {

  int nlocal[3];
  int ic, jc, kc;
  int index0, index1;
  int icm1, icp1;
  double psi, psi0, mu0, mu1;
  double m_phi, m_psi;

  get_N_local(nlocal);
  assert(nhalo_ >= 2);
  assert(nop_ == 2); /* Surfactant only */
  assert(mobility_);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);

	m_phi = mobility_[0];
	m_psi = mobility_[1];

	mu0 = free_energy_chemical_potential(index0, 0);
	psi0 = phi_site[nop_*index0 + 1];

	/* x-direction (between ic-1 and ic) */

	index1 = ADDR(icm1, jc, kc);
	mu1 = free_energy_chemical_potential(index1, 0);
	fluxw[nop_*index0 + 0] -= m_phi*(mu0 - mu1);

	psi = 0.5*(psi0 + phi_site[nop_*index1 + 1]);
	mu1 = free_energy_chemical_potential(index1, 1);
	fluxw[nop_*index0 + 1] -= m_psi*psi*(1.0 - psi)*(mu0 - mu1);

	/* ...and between ic and ic+1 */

	index1 = ADDR(icp1, jc, kc);
	mu1 = free_energy_chemical_potential(index1, 0);
	fluxe[nop_*index0 + 0] -= m_phi*(mu1 - mu0);

	psi = 0.5*(psi0 + phi_site[nop_*index1 + 1]);
	mu1 = free_energy_chemical_potential(index1, 1);
	fluxe[nop_*index0 + 1] -= m_psi*psi*(1.0 - psi)*(mu0 - mu1);

	/* y direction */

	index1 = le_site_index(ic, jc+1, kc);
	mu1 = free_energy_chemical_potential(index1, 0);
	fluxy[nop_*index0 + 0] -= m_phi*(mu1 - mu0);

	psi = 0.5*(psi0 + phi_site[nop_*index1 + 1]);
	mu1 = free_energy_chemical_potential(index1, 1);
	fluxy[nop_*index0 + 1] -= m_psi*psi*(1.0 - psi)*(mu0 - mu1);

	/* z direction */

	index1 = ADDR(ic, jc, kc+1);
	mu1 = free_energy_chemical_potential(index1, 0);
	fluxz[nop_*index0 + 0] -= m_phi*(mu1 - mu0);

	psi = 0.5*(psi0 + phi_site[nop_*index1 + 1]);
	mu1 = free_energy_chemical_potential(index1, 1);
	fluxz[nop_*index0 + 1] -= m_psi*psi*(1.0 - psi)*(mu0 - mu1);

	/* Next site */
      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  phi_ch_correct_fluxes_for_solid
 *
 *  Set fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

static void phi_ch_correct_fluxes_for_solid(void) {

  int nlocal[3];
  int ic, jc, kc, index, n;

  double mask, maskw, maske, masky, maskz;

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	mask  = (site_map_get_status_index(index)  == FLUID);
	maske = (site_map_get_status(ic+1, jc, kc) == FLUID);
	maskw = (site_map_get_status(ic-1, jc, kc) == FLUID);
	masky = (site_map_get_status(ic, jc+1, kc) == FLUID);
	maskz = (site_map_get_status(ic, jc, kc+1) == FLUID);

	for (n = 0;  n < nop_; n++) {
	  fluxw[nop_*index + n] *= mask*maskw;
	  fluxe[nop_*index + n] *= mask*maske;
	  fluxy[nop_*index + n] *= mask*masky;
	  fluxz[nop_*index + n] *= mask*maskz;
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_langmuir_hinshelwood
 *
 *  This computes the contribution to the fluxes for surfactant models
 *  to include adsorption of surfactant at the solid-fluid
 *  interface at the triple contact line.
 *
 *  The flux is determined by the Langmuir Hinshelwood equation
 *
 *  dpsi_t = (k^+ psi_fluid/h)*A*(1 - psi_solid / psi_solid_max)
 *         - k^- *A*psi_solid/psi_solid_max
 *
 *  Here we treat psi as a concentration (per L^3). The thickness
 *  of the adsorbed layer is h and the area of the discrete interface
 *  is A (here = unity). psi_fluid is concentration (per L^3) and
 *  psi_solid, psi_solid_max are nominally concentration (per L^2).
 *
 *  The k are rate constants (where the desorption k^- is often
 *  considered to be zero but included for completeness).
 *  The combination k^+ A can be treated as a diffusion
 *  constant cf the mobility M.
 *
 *  Further, adsoption should only take place at the triple contact
 *  line, which is dealt with by including a factor of
 *
 *  max(0.0,, 1.0 - phi_fluid^2), with phi the composition. The max()
 *
 *  ensures that |phi| > 1 does not contribute.
 *
 *  The current solid value is stored as the psi order parameter at
 *  the solid sites. It is updated by ensuring all sites are updated
 *  in the update routine.
 *
 *  As we do not allow Lees-Edwards planes here, we can set
 *     fluxw(ic,jc,kc) = fluxe(ic-1,jc,kc)
 *  
 *****************************************************************************/

static void phi_ch_langmuir_hinshelwood(void) {

  int nlocal[3];
  int ic, jc, kc, index, index1;

  double mask, maske, masky, maskz;
  double fluxhm, fluxhp;
  double triple_contact;
  double phi;

  const double rh = 1.0;
  const double rpsimax = 1.0/lh_psimax_;

  get_N_local(nlocal);

  assert(nop_ == 2); /* Surfactant model only */
  assert(le_get_nplane_total() == 0);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	mask  = (site_map_get_status_index(index)  == FLUID);
	maske = (site_map_get_status(ic+1, jc, kc) == FLUID);
	masky = (site_map_get_status(ic, jc+1, kc) == FLUID);
	maskz = (site_map_get_status(ic, jc, kc+1) == FLUID);

	/* Order parameter fluxes are set to zero if either mask
	 * is zero at this interface */

	fluxe[nop_*index + 0] *= mask*maske;
	fluxy[nop_*index + 0] *= mask*masky;
	fluxz[nop_*index + 0] *= mask*maskz;

	/* Surfactant: need to be careful what is solid and
	 * what is fluid here (don't change fluid-fluid fluxes!) */

	fluxe[nop_*index + 1] *= mask*maske;
	fluxy[nop_*index + 1] *= mask*masky;
	fluxz[nop_*index + 1] *= mask*maskz;

	phi = phi_site[nop_*index + 0];
	triple_contact = dmax(0.0, 1.0 - phi*phi);

	index1 = ADDR(ic+1,jc,kc);
	phi = phi_site[nop_*index1 + 0];

	fluxhm = rh*lh_kplus_*phi_site[nop_*index + 1]
	  *(1.0 - rpsimax*phi_site[nop_*index1 + 1])
	  - lh_kminus_*rpsimax*phi_site[nop_*index1 + 1];
	fluxhp = rh*lh_kplus_*phi_site[nop_*index1 + 1]
	  *(1.0 - rpsimax*phi_site[nop_*index + 1])
	  - lh_kminus_*rpsimax*phi_site[nop_*index + 1];

	fluxe[nop_*index + 1] += triple_contact*mask*(1.0 - maske)*fluxhm
	  - dmax(0.0, 1.0 - phi*phi)*maske*(1.0 - mask)*fluxhp;

	index1 = ADDR(ic,jc+1,kc);
	phi = phi_site[nop_*index1 + 0];

	fluxhm = rh*lh_kplus_*phi_site[nop_*index + 1]
	  *(1.0 - rpsimax*phi_site[nop_*index1 + 1])
	  - lh_kminus_*rpsimax*phi_site[nop_*index1 + 1];
	fluxhp = rh*lh_kplus_*phi_site[nop_*index1 + 1]
	  *(1.0 - rpsimax*phi_site[nop_*index + 1])
	  - lh_kminus_*rpsimax*phi_site[nop_*index + 1];

	fluxy[nop_*index + 1] += triple_contact*mask*(1.0 - masky)*fluxhm
	  - dmax(0.0, 1.0 - phi*phi)*masky*(1.0 - mask)*fluxhp;

	index1 = ADDR(ic,jc,kc+1);
	phi = phi_site[nop_*index1 + 0];

	fluxhm = rh*lh_kplus_*phi_site[nop_*index + 1]
	  *(1.0 - rpsimax*phi_site[nop_*index1 + 1])
	  - lh_kminus_*rpsimax*phi_site[nop_*index1 + 1];
	fluxhp = rh*lh_kplus_*phi_site[nop_*index1 + 1]
	  *(1.0 - rpsimax*phi_site[nop_*index + 1])
	  - lh_kminus_*rpsimax*phi_site[nop_*index + 1];

	fluxz[nop_*index + 1] += triple_contact*mask*(1.0 - maskz)*fluxhm
	  - dmax(0.0, 1.0 - phi*phi)*maskz*(1.0 - mask)*fluxhp;
      }
    }
  }

  for (ic = 1; ic < nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic,jc,kc);
	index1 = ADDR(ic-1,jc,kc);
	fluxw[nop_*index + 0] = fluxe[nop_*index1 + 0];
	fluxw[nop_*index + 1] = fluxe[nop_*index1 + 1];
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
 *  The time step is the LB time step dt = 1. All sites are processed
 *  to include solid-stored values in the case of Langmuir-Hinshelwood.
 *  It also avoids a conditional on solid/fluid status.
 *
 *****************************************************************************/

static void phi_ch_update_forward_step() {

  int nlocal[3];
  int ic, jc, kc, index, n;

  get_N_local(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	for (n = 0; n < nop_; n++) {
	  phi_site[nop_*index + n] -= (fluxe[nop_*index + n]
	                             - fluxw[nop_*index + n]
	                             + fluxy[nop_*index + n]
	                             - fluxy[nop_*ADDR(ic, jc-1, kc) + n]
	                             + fluxz[nop_*index + n]
				     - fluxz[nop_*ADDR(ic, jc, kc-1) + n]);
	}
      }
    }
  }

  return;
}
