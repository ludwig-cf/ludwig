/*****************************************************************************
 *
 *  phi_force.c
 *
 *  Computes the force on the fluid from the thermodynamic sector
 *  via the divergence of the chemical stress. Its calculation as
 *  a divergence ensures momentum is conserved.
 *
 *  $Id: phi_force.c,v 1.6.4.6 2010-04-02 07:56:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "lattice.h"
#include "phi.h"
#include "site_map.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "timer.h"
#include "wall.h"

static void phi_force_calculation_fluid(void);
static void phi_force_calculation_fluid_solid(void);
static void phi_force_compute_fluxes(void);
static void phi_force_flux_divergence(void);
static void phi_force_fix_fluxes(void);
static void phi_force_fix_fluxes_parallel(void);
static void phi_force_flux(void);
static void phi_force_wall(void);

static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static int  force_required_ = 1;
static void (* phi_force_simple_)(void) = phi_force_calculation_fluid;

/*****************************************************************************
 *
 *  phi_force_required_set
 *
 *****************************************************************************/

void phi_force_required_set(const int flag) {

  force_required_ = flag;
  return;
}

/*****************************************************************************
 *
 *  phi_force_calculation
 *
 *  Driver routine to compute the body force on fluid from phi sector.
 *
 *****************************************************************************/

void phi_force_calculation() {

  if (force_required_ == 0) return;

  TIMER_start(TIMER_FORCE_CALCULATION);

  if (le_get_nplane_total() > 0) {
    /* Must use the flux method for LE planes */
    phi_force_flux();
  }
  else {
    /* Note that this routine does not do accounting for the wall
     * momentum, if required. */
    phi_force_simple_();
  }

  TIMER_stop(TIMER_FORCE_CALCULATION);

  return;
}

/*****************************************************************************
 *
 *  phi_force_set_solid
 *
 *  Set the force calculation method to allow for solid.
 *
 *****************************************************************************/

void phi_force_set_solid(void) {

  assert(0); /* Need to do something about force on particles */
  phi_force_simple_ = phi_force_calculation_fluid_solid;

  return;
}

/*****************************************************************************
 *
 *  phi_force_calculation_fluid
 *
 *  Compute force from thermodynamic sector via
 *    F_alpha = nalba_beta Pth_alphabeta
 *  using a simple six-point stencil.
 *
 *  Side effect: increments the force at each local lattice site in
 *  preparation for the collision stage.
 *
 *****************************************************************************/

static void phi_force_calculation_fluid() {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* Compute differences */
	
	index1 = ADDR(icp1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[X][ia] + pth0[X][ia]);
	}
	index1 = ADDR(icm1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[X][ia] + pth0[X][ia]);
	}

	
	index1 = ADDR(ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	}
	index1 = ADDR(ic, jc-1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	}
	
	index1 = ADDR(ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[Z][ia] + pth0[Z][ia]);
	}
	index1 = ADDR(ic, jc, kc-1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[Z][ia] + pth0[Z][ia]);
	}

	/* Store the force on lattice */

	hydrodynamics_add_force_local(index, force);

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_calculation_fluid_solid
 *
 *  Compute force from thermodynamic sector via
 *    F_alpha = nalba_beta Pth_alphabeta
 *  using a simple six-point stencil.
 *
 *  Side effect: increments the force at each local lattice site in
 *  preparation for the collision stage.
 *
 *****************************************************************************/

static void phi_force_calculation_fluid_solid() {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  int mask;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* Compute differences */
	
	index1 = ADDR(icp1, jc, kc);
	chemical_stress(index1, pth1);
	mask = (site_map_get_status_index(index1) == FLUID);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(mask*pth1[X][ia] + pth0[X][ia]);
	}
	index1 = ADDR(icm1, jc, kc);
	chemical_stress(index1, pth1);
	mask = (site_map_get_status_index(index1) == FLUID);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(mask*pth1[X][ia] + pth0[X][ia]);
	}

	
	index1 = ADDR(ic, jc+1, kc);
	chemical_stress(index1, pth1);
	mask = (site_map_get_status_index(index1) == FLUID);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(mask*pth1[Y][ia] + pth0[Y][ia]);
	}
	index1 = ADDR(ic, jc-1, kc);
	chemical_stress(index1, pth1);
	mask = (site_map_get_status_index(index1) == FLUID);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(mask*pth1[Y][ia] + pth0[Y][ia]);
	}
	
	index1 = ADDR(ic, jc, kc+1);
	chemical_stress(index1, pth1);
	mask = (site_map_get_status_index(index1) == FLUID);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(mask*pth1[Z][ia] + pth0[Z][ia]);
	}
	index1 = ADDR(ic, jc, kc-1);
	chemical_stress(index1, pth1);
	mask = (site_map_get_status_index(index1) == FLUID);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(mask*pth1[Z][ia] + pth0[Z][ia]);
	}

	/* Store the force on lattice */

	hydrodynamics_add_force_local(index, force);

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_flux
 *
 *  Here we compute the momentum fluxes, the divergence of which will
 *  give rise to the force on the fluid.
 *
 *  The flux form is used to ensure conservation, and to allow
 *  the appropriate corrections when LE planes are present.
 *
 *****************************************************************************/

static void phi_force_flux(void) {

  int n;

  n = coords_nsites();

  fluxe = (double *) malloc(3*n*sizeof(double));
  fluxw = (double *) malloc(3*n*sizeof(double));
  fluxy = (double *) malloc(3*n*sizeof(double));
  fluxz = (double *) malloc(3*n*sizeof(double));

  if (fluxe == NULL) fatal("malloc(fluxe) force failed");
  if (fluxw == NULL) fatal("malloc(fluxw) force failed");
  if (fluxy == NULL) fatal("malloc(fluxy) force failed");
  if (fluxz == NULL) fatal("malloc(fluxz) force failed");

  phi_force_compute_fluxes();
  phi_force_fix_fluxes();

  if (wall_present()) phi_force_wall();

  phi_force_flux_divergence();

  free(fluxz);
  free(fluxy);
  free(fluxw);
  free(fluxe);

  return;
}

/*****************************************************************************
 *
 *  phi_force_compute_fluxes
 *
 *  Linearly interpolate the chemical stress to the cell faces to get
 *  the momentum fluxes. 
 *
 *****************************************************************************/

static void phi_force_compute_fluxes(void) {

  int ia, ic, jc, kc, icm1, icp1;
  int index, index1;
  int nlocal[3];
  double pth0[3][3];
  double pth1[3][3];

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  chemical_stress = fe_chemical_stress_function();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = ADDR(ic, jc, kc);

	/* Compute pth at current point */
	chemical_stress(index, pth0);

	/* fluxw_a = (1/2)[P(i, j, k) + P(i-1, j, k)]_xa */
	
	index1 = ADDR(icm1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxw[3*index + ia] = 0.5*(pth1[X][ia] + pth0[X][ia]);
	}

	/* fluxe_a = (1/2)[P(i, j, k) + P(i+1, j, k)_xa */

	index1 = ADDR(icp1, jc, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxe[3*index + ia] = 0.5*(pth1[X][ia] + pth0[X][ia]);
	}

	/* fluxy_a = (1/2)[P(i, j, k) + P(i, j+1, k)]_ya */

	index1 = ADDR(ic, jc+1, kc);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxy[3*index + ia] = 0.5*(pth1[Y][ia] + pth0[Y][ia]);
	}
	
	/* fluxz_a = (1/2)[P(i, j, k) + P(i, j, k+1)]_za */

	index1 = ADDR(ic, jc, kc+1);
	chemical_stress(index1, pth1);
	for (ia = 0; ia < 3; ia++) {
	  fluxz[3*index + ia] = 0.5*(pth1[Z][ia] + pth0[Z][ia]);
	}

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_fix_fluxes
 *
 *  For the Lees-Edwards planes we need to reconcile the east and
 *  west face fluxes (in the same manner as for advective fluxes)
 *  at the planes.
 *
 *****************************************************************************/

static void phi_force_fix_fluxes(void) {

  int nlocal[3]; /* Local system size */
  int ip;        /* Index of the plane */
  int ic;        /* Index x location in real system */
  int jc, kc, ia;
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
    phi_force_fix_fluxes_parallel();
  }
  else {

    coords_nlocal(nlocal);

    nbuffer = 3*nlocal[Y]*nlocal[Z];
    buffere = (double *) malloc(nbuffer*sizeof(double));
    bufferw = (double *) malloc(nbuffer*sizeof(double));
    if (buffere == NULL) fatal("malloc(buffere) force failed\n");
    if (bufferw == NULL) fatal("malloc(bufferw) force failed\n");

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
	  for (ia = 0; ia < 3; ia++) {
	    index = 3*(nlocal[Z]*(jc-1) + (kc-1)) + ia;
	    bufferw[index] = fr*fluxw[3*ADDR(ic+1,j1,kc) + ia]
	      + (1.0-fr)*fluxw[3*ADDR(ic+1,j2,kc) + ia];
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
	  for (ia = 0; ia < 3; ia++) {
	    index = 3*(nlocal[Z]*(jc-1) + (kc-1)) + ia;
	    buffere[index] = fr*fluxe[3*ADDR(ic,j1,kc) + ia]
	      + (1.0-fr)*fluxe[3*ADDR(ic,j2,kc) + ia];
	  }
	}
      }

      /* Now average the fluxes. */

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (ia = 0; ia < 3; ia++) {
	    index = 3*ADDR(ic,jc,kc) + ia;
	    index1 = 3*(nlocal[Z]*(jc-1) + (kc-1)) + ia;
	    fluxe[index] = 0.5*(fluxe[index] + bufferw[index1]);
	    index = 3*ADDR(ic+1,jc,kc) + ia;
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
 *  phi_force_fix_fluxes_parallel
 *
 *  Parallel version of the above, where communication is required to
 *  get hold of the interpolation buffer.
 *
 *****************************************************************************/

static void phi_force_fix_fluxes_parallel(void) {

  int      nhalo;
  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  double * buffere;        /* Interpolation buffer */
  double * bufferw;
  int ip;                  /* Index of the plane */
  int ic;                  /* Index x location in real system */
  int jc, kc, j1, j2, ia;
  int n, n1, n2;
  double dy;               /* Displacement for current transforamtion */
  double fre, frw;         /* Fractional displacements */
  double t;                /* Time */
  int jdy;                 /* Integral part of displacement */

  MPI_Comm le_comm = le_communicator();
  int      nrank_s[2];     /* send ranks */
  int      nrank_r[2];     /* recv ranks */
  const int tag0 = 8200;
  const int tag1 = 8201;

  MPI_Request request[8];
  MPI_Status  status[8];

  int get_step(void);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  /* Allocate the temporary buffer */

  n = 3*(nlocal[Y] + 1)*(nlocal[Z] + 2*nhalo);
  buffere = (double *) malloc(n*sizeof(double));
  bufferw = (double *) malloc(n*sizeof(double));
  if (buffere == NULL) fatal("malloc(buffere) failed\n");
  if (bufferw == NULL) fatal("malloc(bufferw) failed\n");

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

    n1 = 3*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = 3*j2*(nlocal[Z] + 2*nhalo);

    /* Post receives, sends (the wait is later). */

    MPI_Irecv(bufferw,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(bufferw+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 1);
    MPI_Issend(fluxw + 3*ADDR(ic+1,j2,1-nhalo), n1, MPI_DOUBLE, nrank_s[0],
	       tag0, le_comm, request + 2);
    MPI_Issend(fluxw + 3*ADDR(ic+1,1,1-nhalo), n2, MPI_DOUBLE, nrank_s[1],
	       tag1, le_comm, request + 3);

    /* OTHER WAY */

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

    n1 = 3*(nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = 3*j2*(nlocal[Z] + 2*nhalo);

    /* Post new receives, sends, and wait for whole lot to finish. */

    MPI_Irecv(buffere,    n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm,
	      request + 4);
    MPI_Irecv(buffere+n1, n2, MPI_DOUBLE, nrank_r[1], tag1, le_comm,
	      request + 5);
    MPI_Issend(fluxe + 3*ADDR(ic,j2,1-nhalo), n1, MPI_DOUBLE, nrank_s[0],
	       tag0, le_comm, request + 6);
    MPI_Issend(fluxe + 3*ADDR(ic,1,1-nhalo), n2, MPI_DOUBLE, nrank_s[1],
	       tag1, le_comm, request + 7);

    MPI_Waitall(8, request, status);

    /* Now interpolate */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      j1 = (jc - 1    )*(nlocal[Z] + 2*nhalo);
      j2 = (jc - 1 + 1)*(nlocal[Z] + 2*nhalo);
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	for (ia = 0; ia < 3; ia++) {
	  fluxe[3*ADDR(ic,jc,kc) + ia]
	    = 0.5*(fluxe[3*ADDR(ic,jc,kc) + ia]
		   + frw*bufferw[3*(j1 + kc+nhalo-1) + ia]
		   + (1.0-frw)*bufferw[3*(j2 + kc+nhalo-1) + ia]);
	  fluxw[3*ADDR(ic+1,jc,kc) + ia]
	    = 0.5*(fluxw[3*ADDR(ic+1,jc,kc) + ia]
		   + fre*buffere[3*(j1 + kc+nhalo-1) + ia]
		   + (1.0-fre)*buffere[3*(j2 + kc+nhalo-1) + ia]);
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
 *  phi_force_flux_divergence
 *
 *  Take the diverence of the momentum fluxes to get a force on the
 *  fluid site.
 *
 *****************************************************************************/

static void phi_force_flux_divergence(void) {

  int nlocal[3];
  int ic, jc, kc, index, ia;
  double f[3];

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = ADDR(ic, jc, kc);

	for (ia = 0; ia < 3; ia++) {
	  f[ia] = - (fluxe[3*index + ia] - fluxw[3*index + ia]
		     + fluxy[3*index + ia] - fluxy[3*ADDR(ic, jc-1, kc) + ia]
		     + fluxz[3*index + ia] - fluxz[3*ADDR(ic, jc, kc-1) + ia]);
	}

	hydrodynamics_add_force_local(index, f);

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_wall
 *
 *  Account for the net force on the wall.
 *
 *****************************************************************************/

static void phi_force_wall(void) {

  int ic, jc, kc;
  int index, ia;
  int nlocal[3];
  double fw[3];         /* Net force on wall */
  double pth0[3][3];    /* Chemical stress extrpolated to wall */

  void (* chemical_stress)(const int index, double s[3][3]);

  coords_nlocal(nlocal);

  chemical_stress = fe_chemical_stress_function();

  fw[X] = 0.0;
  fw[Y] = 0.0;
  fw[Z] = 0.0;

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic,jc,kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxw[3*index + ia] = pth0[X][ia];
	  fw[ia] -= fluxw[3*index + ia];
	}
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {
    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic,jc,kc);
	chemical_stress(index, pth0);

	for (ia = 0; ia < 3; ia++) {
	  fluxe[3*index + ia] = pth0[X][ia];
	  fw[ia] += fluxe[3*index + ia];
	}
      }
    }
  }

  wall_accumulate_force(fw);

  return;
}
