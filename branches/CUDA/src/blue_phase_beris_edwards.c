/*****************************************************************************
 *
 *  blue_phase_beris_edwards.c
 *
 *  Time evolution for the blue phase tensor order parameter via the
 *  Beris-Edwards equation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "leesedwards.h"
#include "site_map.h"
#include "lattice.h"
#include "phi.h"
#include "colloids.h"
#include "colloids_Q_tensor.h"
#include "wall.h"
#include "advection.h"
#include "advection_bcs.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"
#include "timer.h"

static double Gamma_;     /* Collective rotational diffusion constant */

/* static double * fluxe; */
/* static double * fluxw; */
/* static double * fluxy; */
/* static double * fluxz; */

/* expose these to the outside world for GPU version */
double * fluxe;
double * fluxw;
double * fluxy;
double * fluxz;
double * hs5;

static const double r3 = (1.0/3.0);   /* Fraction 1/3 */
static const int    use_hs_ = 0;      /* Switch for surface term h_s */

static void blue_phase_be_update(double * hs5);
static void blue_phase_be_colloid(double * hs5);
static void blue_phase_be_wallx(double * hs5);
static void blue_phase_be_wally(double * hs5);
static void blue_phase_be_wallz(double * hs5);
static void blue_phase_be_hs(int ic, int jc, int kc, const int nhat[3],
			     const double dn[3], double hs[3][3]);

/*****************************************************************************
 *
 *  blue_phase_beris_edwards
 *
 *  Driver routine for the update.
 *
 *****************************************************************************/

void blue_phase_beris_edwards(void) {

  int nsites;
  int nop;
  //double * hs5;

  /* Set up advective fluxes and do the update. */

  nsites = coords_nsites();
  nop = phi_nop();

  TIMER_start(TIMER_PHI_UPDATE_MALLOC);

#ifndef _GPU_
  fluxe = (double *) malloc(nop*nsites*sizeof(double));
  fluxw = (double *) malloc(nop*nsites*sizeof(double));
  fluxy = (double *) malloc(nop*nsites*sizeof(double));
  fluxz = (double *) malloc(nop*nsites*sizeof(double));
  if (fluxe == NULL) fatal("malloc(fluxe) failed");
  if (fluxw == NULL) fatal("malloc(fluxw) failed");
  if (fluxy == NULL) fatal("malloc(fluxy) failed");
  if (fluxz == NULL) fatal("malloc(fluxz) failed");

  /* Allocate, and initialise to zero, the surface terms (calloc) */

  hs5 = (double *) calloc(nop*nsites, sizeof(double));
  if (hs5 == NULL) fatal("calloc(hs5) failed\n");
#endif

  TIMER_stop(TIMER_PHI_UPDATE_MALLOC);

  
#ifdef _GPU_



  //to do - GPU implement commented out stuff below
  TIMER_start(TIMER_HALO_VELOCITY);
  velocity_halo_gpu();
  TIMER_stop(TIMER_HALO_VELOCITY);
  colloids_fix_swd();
  
  //hydrodynamics_leesedwards_transformation();

  TIMER_start(TIMER_PHI_UPDATE_UPWIND);
  advection_upwind_gpu();
  TIMER_stop(TIMER_PHI_UPDATE_UPWIND);

  TIMER_start(TIMER_PHI_UPDATE_ADVEC);
  advection_bcs_no_normal_flux_gpu();
  TIMER_stop(TIMER_PHI_UPDATE_ADVEC);

  if (use_hs_ && colloids_q_anchoring_method() == ANCHORING_METHOD_TWO) {
    	info("Error: blue_phase_be_surface not yet supported in GPU mode\n");
	exit(1);
    //blue_phase_be_surface(hs5;
  }

  TIMER_start(TIMER_PHI_UPDATE_BE);
  blue_phase_be_update_gpu(hs5);
  TIMER_stop(TIMER_PHI_UPDATE_BE);

#else

  hydrodynamics_halo_u();
  colloids_fix_swd();
  hydrodynamics_leesedwards_transformation();
  advection_upwind(fluxe, fluxw, fluxy, fluxz);
  advection_bcs_no_normal_flux(nop, fluxe, fluxw, fluxy, fluxz);

  if (use_hs_ && colloids_q_anchoring_method() == ANCHORING_METHOD_TWO) {
    blue_phase_be_surface(hs5);
  }

  blue_phase_be_update(hs5);


#endif



#ifndef _GPU_
  free(hs5);
  free(fluxe);
  free(fluxw);
  free(fluxy);
  free(fluxz);
#endif

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_update_fluid
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
 *
 *  Note that solid objects (colloids) are currently treated by evolving
 *  the order parameter inside, but with no hydrodynamics.
 *
 *****************************************************************************/

static void blue_phase_be_update(double * hs) {

  int ic, jc, kc;
  int ia, ib, id;
  int index, indexj, indexk;
  int nlocal[3];
  int nop;

  double q[3][3];
  double w[3][3];
  double d[3][3];
  double h[3][3];
  double s[3][3];
  double omega[3][3];
  double trace_qw;
  double xi;

  const double dt = 1.0;
  double dt_solid;

  coords_nlocal(nlocal);
  nop = phi_nop();
  xi = blue_phase_get_xi();

  assert(nop == 5);

  /* For first anchoring method (only) have evolution at solid sites. */

  dt_solid = 0;
  if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) dt_solid = dt;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

	phi_get_q_tensor(index, q);
	blue_phase_molecular_field(index, h);

	if (site_map_get_status_index(index) != FLUID) {

	  /* Solid: diffusion only. */

	  q[X][X] += dt_solid*Gamma_*h[X][X];
	  q[X][Y] += dt_solid*Gamma_*h[X][Y];
	  q[X][Z] += dt_solid*Gamma_*h[X][Z];
	  q[Y][Y] += dt_solid*Gamma_*h[Y][Y];
	  q[Y][Z] += dt_solid*Gamma_*h[Y][Z];

	}
	else {

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);
	  
	  trace_qw = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    trace_qw += q[ia][ia]*w[ia][ia];
	    for (ib = 0; ib < 3; ib++) {
	      d[ia][ib]     = 0.5*(w[ia][ib] + w[ib][ia]);
	      omega[ia][ib] = 0.5*(w[ia][ib] - w[ib][ia]);
	    }
	  }
	  
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      s[ia][ib] = -2.0*xi*(q[ia][ib] + r3*d_[ia][ib])*trace_qw;
	      for (id = 0; id < 3; id++) {
		s[ia][ib] +=
		  (xi*d[ia][id] + omega[ia][id])*(q[id][ib] + r3*d_[id][ib])
		  + (q[ia][id] + r3*d_[ia][id])*(xi*d[id][ib] - omega[id][ib]);
	      }
	    }
	  }
	     
	  /* Here's the full hydrodynamic update. */
	  
	  indexj = le_site_index(ic, jc-1, kc);
	  indexk = le_site_index(ic, jc, kc-1);

	  q[X][X] += dt*(s[X][X] + Gamma_*(h[X][X] + hs[nop*index + XX])
			 - fluxe[nop*index + XX] + fluxw[nop*index  + XX]
			 - fluxy[nop*index + XX] + fluxy[nop*indexj + XX]
			 - fluxz[nop*index + XX] + fluxz[nop*indexk + XX]);

	  q[X][Y] += dt*(s[X][Y] + Gamma_*(h[X][Y] + hs[nop*index + XY])
			 - fluxe[nop*index + XY] + fluxw[nop*index  + XY]
			 - fluxy[nop*index + XY] + fluxy[nop*indexj + XY]
			 - fluxz[nop*index + XY] + fluxz[nop*indexk + XY]);

	  q[X][Z] += dt*(s[X][Z] + Gamma_*(h[X][Z] + hs[nop*index + XZ])
			 - fluxe[nop*index + XZ] + fluxw[nop*index  + XZ]
			 - fluxy[nop*index + XZ] + fluxy[nop*indexj + XZ]
			 - fluxz[nop*index + XZ] + fluxz[nop*indexk + XZ]);

	  q[Y][Y] += dt*(s[Y][Y] + Gamma_*(h[Y][Y] + hs[nop*index + YY])
			 - fluxe[nop*index + YY] + fluxw[nop*index  + YY]
			 - fluxy[nop*index + YY] + fluxy[nop*indexj + YY]
			 - fluxz[nop*index + YY] + fluxz[nop*indexk + YY]);

	  q[Y][Z] += dt*(s[Y][Z] + Gamma_*(h[Y][Z] + hs[nop*index + YZ])
			 - fluxe[nop*index + YZ] + fluxw[nop*index  + YZ]
			 - fluxy[nop*index + YZ] + fluxy[nop*indexj + YZ]
			 - fluxz[nop*index + YZ] + fluxz[nop*indexk + YZ]);
	}
	
	phi_set_q_tensor(index, q);

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_set_rotational_diffusion
 *
 *****************************************************************************/

void blue_phase_be_set_rotational_diffusion(double gamma) {

  Gamma_ = gamma;
}

/*****************************************************************************
 *
 *  blue_phase_be_get_rotational_diffusion
 *
 *****************************************************************************/

double blue_phase_be_get_rotational_diffusion(void) {

  return Gamma_;
}

/*****************************************************************************
 *
 *  blue_phase_be_surface
 *
 *  Organise the surface terms. Note that hs5 should be initalised to zero
 *  before this point.
 *
 *****************************************************************************/

void blue_phase_be_surface(double * hs5) {

  if (wall_at_edge(X)) blue_phase_be_wallx(hs5);
  if (wall_at_edge(Y)) blue_phase_be_wally(hs5);
  if (wall_at_edge(Z)) blue_phase_be_wallz(hs5);

  if (colloid_ntotal() > 0) blue_phase_be_colloid(hs5);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_colloid
 *
 *  Compute the surface contributions to the molecular field in the
 *  presence of colloids.
 *
 *****************************************************************************/

static void blue_phase_be_colloid(double * hs5) {

  int nlocal[3];
  int ic, jc, kc;
  int index;
  int nop;

  int nhat[3];
  double dn[3];
  double hs[3][3];

  coords_nlocal(nlocal);
  nop = phi_nop();

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	if (site_map_get_status_index(index) != FLUID) continue;

	nhat[Y] = 0;
	nhat[Z] = 0;

	if (site_map_get_status(ic+1, jc, kc) == COLLOID) {
	  nhat[X] = -1;
	  colloids_q_boundary_normal(index, nhat, dn);
	  blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	  hs5[nop*index + XX] += hs[X][X];
	  hs5[nop*index + XY] += hs[X][Y];
	  hs5[nop*index + XZ] += hs[X][Z];
	  hs5[nop*index + YY] += hs[Y][Y];
	  hs5[nop*index + YZ] += hs[Y][Z];
	}

	if (site_map_get_status(ic-1, jc, kc) == COLLOID) {
	  nhat[X] = +1;
	  colloids_q_boundary_normal(index, nhat, dn);	  
	  blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	  hs5[nop*index + XX] += hs[X][X];
	  hs5[nop*index + XY] += hs[X][Y];
	  hs5[nop*index + XZ] += hs[X][Z];
	  hs5[nop*index + YY] += hs[Y][Y];
	  hs5[nop*index + YZ] += hs[Y][Z];
	}

	nhat[X] = 0;
	nhat[Z] = 0;

	if (site_map_get_status(ic, jc+1, kc) == COLLOID) {
	  nhat[Y] = -1;
	  colloids_q_boundary_normal(index, nhat, dn);	  
	  blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	  hs5[nop*index + XX] += hs[X][X];
	  hs5[nop*index + XY] += hs[X][Y];
	  hs5[nop*index + XZ] += hs[X][Z];
	  hs5[nop*index + YY] += hs[Y][Y];
	  hs5[nop*index + YZ] += hs[Y][Z];
	}

	if (site_map_get_status(ic, jc-1, kc) == COLLOID) {
	  nhat[Y] = 1;
	  colloids_q_boundary_normal(index, nhat, dn);	  
	  blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	  hs5[nop*index + XX] += hs[X][X];
	  hs5[nop*index + XY] += hs[X][Y];
	  hs5[nop*index + XZ] += hs[X][Z];
	  hs5[nop*index + YY] += hs[Y][Y];
	  hs5[nop*index + YZ] += hs[Y][Z];
	}

	nhat[X] = 0;
	nhat[Y] = 0;

	if (site_map_get_status(ic, jc, kc+1) == COLLOID) {
	  nhat[Z] = -1;
	  colloids_q_boundary_normal(index, nhat, dn);	  
	  blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	  hs5[nop*index + XX] += hs[X][X];
	  hs5[nop*index + XY] += hs[X][Y];
	  hs5[nop*index + XZ] += hs[X][Z];
	  hs5[nop*index + YY] += hs[Y][Y];
	  hs5[nop*index + YZ] += hs[Y][Z];
	}

	if (site_map_get_status(ic, jc, kc-1) == COLLOID) {
	  nhat[Z] = 1;
	  colloids_q_boundary_normal(index, nhat, dn);	  
	  blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	  hs5[nop*index + XX] += hs[X][X];
	  hs5[nop*index + XY] += hs[X][Y];
	  hs5[nop*index + XZ] += hs[X][Z];
	  hs5[nop*index + YY] += hs[Y][Y];
	  hs5[nop*index + YZ] += hs[Y][Z];
	}

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_wallx
 *
 *  Surface term in molecular field for flat wall.
 *
 *****************************************************************************/

static void blue_phase_be_wallx(double * hs5) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nhat[3];
  int nop;

  double dn[3];
  double hs[3][3];

  coords_nlocal(nlocal);
  nop = phi_nop();

  nhat[Y] = 0;
  nhat[Z] = 0;
  dn[Y] = 0.0;
  dn[Z] = 0.0;

  if (cart_coords(X) == 0) {

    ic = 1;
    nhat[X] = +1;
    dn[X] = +1.0;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	index = coords_index(ic, jc, kc);
	hs5[nop*index + XX] = hs[X][X];
	hs5[nop*index + XY] = hs[X][Y];
	hs5[nop*index + XZ] = hs[X][Z];
	hs5[nop*index + YY] = hs[Y][Y];
	hs5[nop*index + YZ] = hs[Y][Z];
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    ic = nlocal[X];
    nhat[X] = -1;
    dn[X] = -1.0;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	index = coords_index(ic, jc, kc);
	hs5[nop*index + XX] = hs[X][X];
	hs5[nop*index + XY] = hs[X][Y];
	hs5[nop*index + XZ] = hs[X][Z];
	hs5[nop*index + YY] = hs[Y][Y];
	hs5[nop*index + YZ] = hs[Y][Z];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_wally
 *
 *  Surface term in molecular field for flat wall.
 *
 *****************************************************************************/

static void blue_phase_be_wally(double * hs5) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nhat[3];
  int nop;

  double dn[3];
  double hs[3][3];

  coords_nlocal(nlocal);
  nop = phi_nop();

  nhat[X] = 0;
  nhat[Z] = 0;
  dn[X] = 0.0;
  dn[Z] = 0.0;

  if (cart_coords(Y) == 0) {

    jc = 1;
    nhat[Y] = +1;
    dn[Y] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	index = coords_index(ic, jc, kc);
	hs5[nop*index + XX] = hs[X][X];
	hs5[nop*index + XY] = hs[X][Y];
	hs5[nop*index + XZ] = hs[X][Z];
	hs5[nop*index + YY] = hs[Y][Y];
	hs5[nop*index + YZ] = hs[Y][Z];
      }
    }
  }

  if (cart_coords(Y) == cart_size(Y) - 1) {

    jc = nlocal[Y];
    nhat[Y] = -1;
    dn[Y] = -1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	index = coords_index(ic, jc, kc);
	hs5[nop*index + XX] = hs[X][X];
	hs5[nop*index + XY] = hs[X][Y];
	hs5[nop*index + XZ] = hs[X][Z];
	hs5[nop*index + YY] = hs[Y][Y];
	hs5[nop*index + YZ] = hs[Y][Z];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_wallz
 *
 *  Surface term in molecular field for flat wall.
 *
 *****************************************************************************/

static void blue_phase_be_wallz(double * hs5) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nhat[3];
  int nop;

  double dn[3];
  double hs[3][3];

  coords_nlocal(nlocal);
  nop = phi_nop();

  nhat[X] = 0;
  nhat[Y] = 0;
  dn[X] = 0.0;
  dn[Y] = 0.0;

  if (cart_coords(Z) == 0) {

    kc = 1;
    nhat[Z] = +1;
    dn[Z] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	index = coords_index(ic, jc, kc);
	hs5[nop*index + XX] = hs[X][X];
	hs5[nop*index + XY] = hs[X][Y];
	hs5[nop*index + XZ] = hs[X][Z];
	hs5[nop*index + YY] = hs[Y][Y];
	hs5[nop*index + YZ] = hs[Y][Z];
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    kc = nlocal[Z];
    nhat[Z] = -1;
    dn[Z] = -1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	blue_phase_be_hs(ic, jc, kc, nhat, dn, hs);
	index = coords_index(ic, jc, kc);
	hs5[nop*index + XX] = hs[X][X];
	hs5[nop*index + XY] = hs[X][Y];
	hs5[nop*index + XZ] = hs[X][Z];
	hs5[nop*index + YY] = hs[Y][Y];
	hs5[nop*index + YZ] = hs[Y][Z];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_hs
 *
 *  Compute the surface term hs in the molecular field.
 *  We follow here the extrapolation of the Q tensor to the surface
 *  as used in the anchoring gradient routines. This gives us Q^s.
 *
 *  The H_s term depends on the derivative of the surface free energy
 *  with respect to the order parameter
 *
 *       H_s = - d/dQ_ab f_s
 *
 *  which we assume to be -w*(Q^s_ab - Q^0_ab) with w positive, and
 *  Q^s the surface order parameter from extrapolation, and Q^0 the
 *  preferred surface orientation.
 *
 *  The input (ic, jc, kc) is the fluid site, and the surface outward
 *  (repeat outward, pointing into the fluid) normal on the lattice is
 *  nhat. This is used to do the extrapolation.
 *
 *  The 'true' outward normal (floating point) is dn[3]. This is used
 *  compute the surface Q^0_ab. 
 *
 *****************************************************************************/

static void blue_phase_be_hs(int ic, int jc, int kc, const int nhat[3],
			     const double dn[3], double hs[3][3]) {
  int ia, ib;
  int index1;
  char status;

  double w;
  double qs[3][3], q0[3][3];

  w = colloids_q_tensor_w(); /* This is for colloid */

  index1 = coords_index(ic, jc, kc);
  status = site_map_get_status(ic - nhat[X], jc - nhat[Y], kc - nhat[Z]);
  
  /*Check if the status is wall */
  if (status == BOUNDARY ) w = wall_w_get();

  phi_get_q_tensor(index1, qs);
  colloids_q_boundary(dn, qs, q0, status);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0 ; ib < 3; ib++) {
      hs[ia][ib] = -w*(qs[ia][ib] - q0[ia][ib]);
    }
  }

  return;
}
