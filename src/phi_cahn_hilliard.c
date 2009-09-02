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
 *  $Id: phi_cahn_hilliard.c,v 1.8 2009-09-02 07:47:51 kevin Exp $
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
static void phi_ch_le_fix_fluxes(void);
static void phi_ch_le_fix_fluxes_parallel(void);

static double * mobility_;   /* Order parameter mobilities */
static double   lh_kplus_;   /* Langmuir Hinshelwood adsorption rate */
static double   lh_kminus_;  /* Langmuir Hinshelwood desorption rate */
static double   lh_psimax_;  /* Langmuir Hinshelwood monolayer capacity */

static int langmuirh_ = 0; /* Langmuir Hinshelwood flag */
static int advection_ = 1; /* Advection scheme */
static int solid_     = 1; /* Solid present? */

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


  phi_ch_le_fix_fluxes();
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
 *  phi_ch_le_fix_fluxes
 *
 *  Owing to the non-linear calculation for the fluxes,
 *  the LE-interpolated phi field doesn't give a unique
 *  east-west face flux.
 *
 *  This ensures uniqueness, by averaging the relevant
 *  contributions from each side of the plane.
 *
 *****************************************************************************/

static void phi_ch_le_fix_fluxes(void) {

  int nlocal[3]; /* Local system size */
  int ip;        /* Index of the plane */
  int ic;        /* Index x location in real system */
  int jc, kc, n;
  int index, index1;
  int nbuffer;

  double dy;     /* Displacement for current plane */
  double fr;     /* Fractional displacement */
  double t;      /* Time */
  int jdy;       /* Integral part of displacement */
  int j1, j2;    /* j values in real system to interpolate between */

  double * bufferw;
  double * buffere;

  int get_step(void);

  if (cart_size(Y) > 1) {
    /* Parallel */
    phi_ch_le_fix_fluxes_parallel();
  }
  else {
    /* Can do it directly */

    get_N_local(nlocal);

    nbuffer = nop_*nlocal[Y]*nlocal[Z];
    buffere = (double *) malloc(nbuffer*sizeof(double));
    bufferw = (double *) malloc(nbuffer*sizeof(double));
    if (buffere == NULL) fatal("malloc(buffere) failed\n");
    if (bufferw == NULL) fatal("malloc(bufferw) failed\n");

    for (ip = 0; ip < le_get_nplane_local(); ip++) {

      /* -1.0 as zero required for first step; a 'feature' to
       * maintain the regression tests */

      t = 1.0*get_step() - 1.0;

      ic = le_plane_location(ip);

      /* Looking up */
      dy = +t*le_plane_uy(t);
      dy = fmod(dy, L(Y));
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1; jc <= nlocal[Y]; jc++) {

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    index = nop_*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    bufferw[index] = fr*fluxw[nop_*ADDR(ic+1,j1,kc) + n]
	      + (1.0-fr)*fluxw[nop_*ADDR(ic+1,j2,kc) + n];
	  }
	}
      }


      /* Looking down */

      dy = -t*le_plane_uy(t);
      dy = fmod(dy, L(Y));
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1; jc <= nlocal[Y]; jc++) {

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    index = nop_*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    buffere[index] = fr*fluxe[nop_*ADDR(ic,j1,kc) + n]
	      + (1.0-fr)*fluxe[nop_*ADDR(ic,j2,kc) + n];
	  }
	}
      }

      /* Now average the fluxes. */

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop_; n++) {
	    index = nop_*ADDR(ic,jc,kc) + n;
	    index1 = nop_*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    fluxe[index] = 0.5*(fluxe[index] + bufferw[index1]);
	    index = nop_*ADDR(ic+1,jc,kc) + n;
	    fluxw[index] = 0.5*(fluxw[index] + buffere[index1]);
	  }
	}
      }

      /* Next plane */
    }

    free(bufferw);
    free(buffere);
  }

  return;
}

/*****************************************************************************
 *
 *  phi_ch_le_fix_fluxes_parallel
 *
 *  Parallel version of the above, where we need to communicate to
 *  get hold of the appropriate fluxes.
 *
 *****************************************************************************/

static void phi_ch_le_fix_fluxes_parallel(void) {

  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  double * buffere;        /* Interpolation buffer */
  double * bufferw;
  int ip;                  /* Index of the plane */
  int ic;                  /* Index x location in real system */
  int jc, kc, j1, j2;
  int n, n1, n2;
  double dy;               /* Displacement for current transforamtion */
  double fre, frw;         /* Fractional displacements */
  double t;                /* Time */
  int jdy;                 /* Integral part of displacement */

  MPI_Comm le_comm = le_communicator();
  int      nrank_s[2];     /* send ranks */
  int      nrank_r[2];     /* recv ranks */
  const int tag0 = 951254;
  const int tag1 = 951255;

  MPI_Request request[8];
  MPI_Status  status[8];

  int get_step(void);

  get_N_local(nlocal);
  get_N_offset(noffset);

  /* Allocate the temporary buffer */

  n = nop_*(nlocal[Y] + 1)*(nlocal[Z] + 2*nhalo_);
  buffere = (double *) malloc(n*sizeof(double));
  bufferw = (double *) malloc(n*sizeof(double));
  if (buffere == NULL) fatal("malloc(buffere) failed\n");
  if (bufferw == NULL) fatal("malloc(bufferw) failed\n");

  /* -1.0 as zero required for fisrt step; this is a 'feature'
   * to ensure the regression tests stay te same */

  t = 1.0*get_step() - 1.0;

  /* One round of communication for each plane */

  for (ip = 0; ip < le_get_nplane_local(); ip++) {

    ic = le_plane_location(ip);

    /* Work out the displacement-dependent quantities */

    dy = +t*le_plane_uy(t);
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    frw  = dy - jdy;

    /* First (global) j1 required is j1 = (noffset[Y] + 1) - jdy - 1.
     * Modular arithmetic ensures 1 <= j1 <= N_total(Y). */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*N_total(Y)) % N_total(Y);
    assert(j1 > 0);
    assert(j1 <= N_total(Y));

    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position 1. */

    j2 = 1 + (j1 - 1) % nlocal[Y];
    assert(j2 > 0);
    assert(j2 <= nlocal[Y]);

    n1 = nop_*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo_);
    n2 = nop_*j2*(nlocal[Z] + 2*nhalo_);

    /* Post receives, sends (the wait is later). */

    MPI_Irecv(bufferw,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(bufferw+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 1);
    MPI_Issend(fluxw + nop_*ADDR(ic+1,j2,1-nhalo_), n1, MPI_DOUBLE, nrank_s[0],
	       tag0, le_comm, request + 2);
    MPI_Issend(fluxw + nop_*ADDR(ic+1,1,1-nhalo_), n2, MPI_DOUBLE, nrank_s[1],
	       tag1, le_comm, request + 3);


    /* OTHER WAY */

    kc = 1 - nhalo_;

    dy = -t*le_plane_uy(t);
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    fre  = dy - jdy;

    /* First (global) j1 required is j1 = (noffset[Y] + 1) - jdy - 1.
     * Modular arithmetic ensures 1 <= j1 <= N_total(Y). */

    jc = noffset[Y] + 1;
    j1 = 1 + (jc - jdy - 2 + 2*N_total(Y)) % N_total(Y);

    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position nhalo_. */

    j2 = 1 + (j1 - 1) % nlocal[Y];

    n1 = nop_*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo_);
    n2 = nop_*j2*(nlocal[Z] + 2*nhalo_);

    /* Post new receives, sends, and wait for whole lot to finish. */

    MPI_Irecv(buffere,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm,
	      request + 4);
    MPI_Irecv(buffere+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 5);
    MPI_Issend(fluxe + nop_*ADDR(ic,j2,1-nhalo_), n1, MPI_DOUBLE, nrank_s[0],
	       tag0, le_comm, request + 6);
    MPI_Issend(fluxe + nop_*ADDR(ic,1,1-nhalo_), n2, MPI_DOUBLE, nrank_s[1],
	       tag1, le_comm, request + 7);

    MPI_Waitall(8, request, status);

    /* Now we've done all the communication, we can update the fluxes
     * using the average of the local value and interpolated buffer
     * value. */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      j1 = (jc - 1    )*(nlocal[Z] + 2*nhalo_);
      j2 = (jc - 1 + 1)*(nlocal[Z] + 2*nhalo_);
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (n = 0; n < nop_; n++) {
	  fluxe[nop_*ADDR(ic,jc,kc) + n]
	    = 0.5*(fluxe[nop_*ADDR(ic,jc,kc) + n]
		   + frw*bufferw[nop_*(j1 + kc+nhalo_-1) + n]
		   + (1.0-frw)*bufferw[nop_*(j2 + kc+nhalo_-1) + n]);
	  fluxw[nop_*ADDR(ic+1,jc,kc) + n]
	    = 0.5*(fluxw[nop_*ADDR(ic+1,jc,kc) + n]
		   + fre*buffere[nop_*(j1 + kc+nhalo_-1) + n]
		   + (1.0-fre)*buffere[nop_*(j2 + kc+nhalo_-1) + n]);
	}
      }
    }

    /* Next plane */
  }

  free(bufferw);
  free(buffere);

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
