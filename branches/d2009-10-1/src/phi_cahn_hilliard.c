/*****************************************************************************
 *
 *  phi_cahn_hilliard.c
 *
 *  The time evolution of the order parameter phi is described
 *  by the Cahn Hilliard equation
 *
 *     d_t phi + div (u phi - M grad mu) = 0.
 *
 *  The equation is solved here via finite difference. The velocity
 *  field u is assumed known from the hydrodynamic sector. M is the
 *  order parameter mobility. The chemical potential mu is set via
 *  the choice of free energy.
 *
 *  $Id: phi_cahn_hilliard.c,v 1.10.4.7 2010-04-05 10:56:21 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "advection.h"
#include "advection_bcs.h"
#include "lattice.h"
#include "free_energy.h"
#include "phi.h"

extern double * phi_site;
static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static void phi_ch_diffusive_flux(void);
static void phi_ch_update_forward_step(void);
static void phi_ch_le_fix_fluxes(void);
static void phi_ch_le_fix_fluxes_parallel(void);

static double mobility_  = 0.0; /* Order parameter mobility */

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

  int nsites;

  /* Single order parameter only */
  assert(phi_nop() == 1);

  nsites = coords_nsites();

  fluxe = (double *) malloc(nsites*sizeof(double));
  fluxw = (double *) malloc(nsites*sizeof(double));
  fluxy = (double *) malloc(nsites*sizeof(double));
  fluxz = (double *) malloc(nsites*sizeof(double));
  if (fluxe == NULL) fatal("malloc(fluxe) failed");
  if (fluxw == NULL) fatal("malloc(fluxw) failed");
  if (fluxy == NULL) fatal("malloc(fluxy) failed");
  if (fluxz == NULL) fatal("malloc(fluxz) failed");

  hydrodynamics_halo_u();
  hydrodynamics_leesedwards_transformation();

  advection_order_n(fluxe, fluxw, fluxy, fluxz);

  phi_ch_diffusive_flux();

  advection_bcs_wall();

  phi_ch_le_fix_fluxes();
  phi_ch_update_forward_step();

  free(fluxz);
  free(fluxy);
  free(fluxw);
  free(fluxe);

  return;
}

/*****************************************************************************
 *
 *  phi_cahn_hilliard_mobility
 *
 *****************************************************************************/

double phi_cahn_hilliard_mobility() {

  return mobility_;
}

/*****************************************************************************
 *
 *  phi_cahn_hilliard_mobility_set
 *
 *****************************************************************************/

void phi_cahn_hilliard_mobility_set(const double m) {

  mobility_ = m;
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

static void phi_ch_diffusive_flux(void) {

  int nlocal[3];
  int ic, jc, kc;
  int index0, index1;
  int icm1, icp1;
  double mu0, mu1;

  double (* chemical_potential)(const int index, const int nop);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_potential = fe_chemical_potential_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);

	mu0 = chemical_potential(index0, 0);

	/* x-direction (between ic-1 and ic) */

	index1 = le_site_index(icm1, jc, kc);
	mu1 = chemical_potential(index1, 0);
	fluxw[index0] -= mobility_*(mu0 - mu1);

	/* ...and between ic and ic+1 */

	index1 = le_site_index(icp1, jc, kc);
	mu1 = chemical_potential(index1, 0);
	fluxe[index0] -= mobility_*(mu1 - mu0);

	/* y direction */

	index1 = le_site_index(ic, jc+1, kc);
	mu1 = chemical_potential(index1, 0);
	fluxy[index0] -= mobility_*(mu1 - mu0);

	/* z direction */

	index1 = le_site_index(ic, jc, kc+1);
	mu1 = chemical_potential(index1, 0);
	fluxz[index0] -= mobility_*(mu1 - mu0);

	/* Next site */
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
 *  I've retained nop here, as these functions might be useful
 *  for general cases.
 *
 *****************************************************************************/

static void phi_ch_le_fix_fluxes(void) {

  int nop;
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

    nop = phi_nop();
    coords_nlocal(nlocal);

    nbuffer = nop*nlocal[Y]*nlocal[Z];
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
	  for (n = 0; n < nop; n++) {
	    index = nop*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    bufferw[index] = fr*fluxw[nop*le_site_index(ic+1,j1,kc) + n]
	      + (1.0-fr)*fluxw[nop*le_site_index(ic+1,j2,kc) + n];
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
	  for (n = 0; n < nop; n++) {
	    index = nop*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    buffere[index] = fr*fluxe[nop*le_site_index(ic,j1,kc) + n]
	      + (1.0-fr)*fluxe[nop*le_site_index(ic,j2,kc) + n];
	  }
	}
      }

      /* Now average the fluxes. */

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nop; n++) {
	    index = nop*le_site_index(ic,jc,kc) + n;
	    index1 = nop*(nlocal[Z]*(jc-1) + (kc-1)) + n;
	    fluxe[index] = 0.5*(fluxe[index] + bufferw[index1]);
	    index = nop*le_site_index(ic+1,jc,kc) + n;
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

  int      nop;
  int      nhalo;
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
  const int tag0 = 1254;
  const int tag1 = 1255;

  MPI_Request request[8];
  MPI_Status  status[8];

  int get_step(void);

  nop = phi_nop();
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  /* Allocate the temporary buffer */

  n = nop*(nlocal[Y] + 1)*(nlocal[Z] + 2*nhalo);
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

    n1 = nop*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = nop*j2*(nlocal[Z] + 2*nhalo);

    /* Post receives, sends (the wait is later). */

    MPI_Irecv(bufferw,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(bufferw+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 1);
    MPI_Issend(fluxw + nop*le_site_index(ic+1,j2,1-nhalo), n1, MPI_DOUBLE, nrank_s[0],
	       tag0, le_comm, request + 2);
    MPI_Issend(fluxw + nop*le_site_index(ic+1,1,1-nhalo), n2, MPI_DOUBLE, nrank_s[1],
	       tag1, le_comm, request + 3);


    /* OTHER WAY */

    kc = 1 - nhalo;

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
     * j2, and the remaining n2 sites from starting position nhalo. */

    j2 = 1 + (j1 - 1) % nlocal[Y];

    n1 = nop*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = nop*j2*(nlocal[Z] + 2*nhalo);

    /* Post new receives, sends, and wait for whole lot to finish. */

    MPI_Irecv(buffere,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm,
	      request + 4);
    MPI_Irecv(buffere+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 5);
    MPI_Issend(fluxe + nop*le_site_index(ic,j2,1-nhalo), n1, MPI_DOUBLE, nrank_s[0],
	       tag0, le_comm, request + 6);
    MPI_Issend(fluxe + nop*le_site_index(ic,1,1-nhalo), n2, MPI_DOUBLE, nrank_s[1],
	       tag1, le_comm, request + 7);

    MPI_Waitall(8, request, status);

    /* Now we've done all the communication, we can update the fluxes
     * using the average of the local value and interpolated buffer
     * value. */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      j1 = (jc - 1    )*(nlocal[Z] + 2*nhalo);
      j2 = (jc - 1 + 1)*(nlocal[Z] + 2*nhalo);
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (n = 0; n < nop; n++) {
	  fluxe[nop*le_site_index(ic,jc,kc) + n]
	    = 0.5*(fluxe[nop*le_site_index(ic,jc,kc) + n]
		   + frw*bufferw[nop*(j1 + kc+nhalo-1) + n]
		   + (1.0-frw)*bufferw[nop*(j2 + kc+nhalo-1) + n]);
	  fluxw[nop*le_site_index(ic+1,jc,kc) + n]
	    = 0.5*(fluxw[nop*le_site_index(ic+1,jc,kc) + n]
		   + fre*buffere[nop*(j1 + kc+nhalo-1) + n]
		   + (1.0-fre)*buffere[nop*(j2 + kc+nhalo-1) + n]);
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
  int ic, jc, kc, index;
  int ys;
  double wz = 1.0;

  coords_nlocal(nlocal);

  ys = nlocal[Z] + 2*coords_nhalo();

  /* In 2-d systems need to eliminate the z fluxes (no chemical
   * potential computed in halo region for 2d_5pt_fluid) */
  if (nlocal[Z] == 1) wz = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	phi_site[index] -=
	  (+ fluxe[index] - fluxw[index]
	   + fluxy[index] - fluxy[index - ys]
	   + wz*fluxz[index] - wz*fluxz[index - 1]);
      }
    }
  }

  return;
}
