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
#include "advection.h"
#include "advection_bcs.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"

static double Gamma_;     /* Collective rotational diffusion constant */

static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static const double r3 = (1.0/3.0);

static void blue_phase_be_update(void);

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

  /* Set up advective fluxes and do the update. */

  nsites = coords_nsites();
  nop = phi_nop();

  fluxe = (double *) malloc(nop*nsites*sizeof(double));
  fluxw = (double *) malloc(nop*nsites*sizeof(double));
  fluxy = (double *) malloc(nop*nsites*sizeof(double));
  fluxz = (double *) malloc(nop*nsites*sizeof(double));
  if (fluxe == NULL) fatal("malloc(fluxe) failed");
  if (fluxw == NULL) fatal("malloc(fluxw) failed");
  if (fluxy == NULL) fatal("malloc(fluxy) failed");
  if (fluxz == NULL) fatal("malloc(fluxz) failed");

  hydrodynamics_halo_u();
  hydrodynamics_leesedwards_transformation();
  advection_upwind(fluxe, fluxw, fluxy, fluxz);
  advection_bcs_no_normal_flux(fluxe, fluxw, fluxy, fluxz);
  blue_phase_be_update();

  free(fluxe);
  free(fluxw);
  free(fluxy);
  free(fluxz);

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

static void blue_phase_be_update(void) {

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

  coords_nlocal(nlocal);
  nop = phi_nop();
  xi = blue_phase_get_xi();

  assert(nop == 5);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

	phi_get_q_tensor(index, q);
	blue_phase_molecular_field(index, h);

	if (site_map_get_status_index(index) != FLUID) {

	  /* Solid: diffusion only. */

	  q[X][X] += dt*Gamma_*h[X][X];
	  q[X][Y] += dt*Gamma_*h[X][Y];
	  q[X][Z] += dt*Gamma_*h[X][Z];
	  q[Y][Y] += dt*Gamma_*h[Y][Y];
	  q[Y][Z] += dt*Gamma_*h[Y][Z];

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
	  
	  q[X][X] += dt*(s[X][X] + Gamma_*h[X][X]
			 - fluxe[nop*index + XX] + fluxw[nop*index  + XX]
			 - fluxy[nop*index + XX] + fluxy[nop*indexj + XX]
			 - fluxz[nop*index + XX] + fluxz[nop*indexk + XX]);

	  q[X][Y] += dt*(s[X][Y] + Gamma_*h[X][Y]
			 - fluxe[nop*index + XY] + fluxw[nop*index  + XY]
			 - fluxy[nop*index + XY] + fluxy[nop*indexj + XY]
			 - fluxz[nop*index + XY] + fluxz[nop*indexk + XY]);

	  q[X][Z] += dt*(s[X][Z] + Gamma_*h[X][Z]
			 - fluxe[nop*index + XZ] + fluxw[nop*index  + XZ]
			 - fluxy[nop*index + XZ] + fluxy[nop*indexj + XZ]
			 - fluxz[nop*index + XZ] + fluxz[nop*indexk + XZ]);

	  q[Y][Y] += dt*(s[Y][Y] + Gamma_*h[Y][Y]
			 - fluxe[nop*index + YY] + fluxw[nop*index  + YY]
			 - fluxy[nop*index + YY] + fluxy[nop*indexj + YY]
			 - fluxz[nop*index + YY] + fluxz[nop*indexk + YY]);

	  q[Y][Z] += dt*(s[Y][Z] + Gamma_*h[Y][Z]
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
