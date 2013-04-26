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
#include <string.h>

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

static double Gamma_;     /* Collective rotational diffusion constant */

static double * fluxe;
static double * fluxw;
static double * fluxy;
static double * fluxz;

static const double r3 = (1.0/3.0);   /* Fraction 1/3 */

static void blue_phase_be_update(void);
static void be_rk(double dt, const double * qn);

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
  int euler = 0;
  extern double * phi_site;

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
  colloids_fix_swd();
  hydrodynamics_leesedwards_transformation();

  if (euler) {
    advection_upwind(fluxe, fluxw, fluxy, fluxz);
    advection_bcs_no_normal_flux(nop, fluxe, fluxw, fluxy, fluxz);

    blue_phase_be_update();
  }
  else {

    /* This is RK2 with phi_site used to store the half-way state
     * and the final state.
     *
     *    phi(t + dt) = phi(dt) + dt*RHS[ phi(t + 0.5*dt) ]
     *    with
     *    phi(t + 0.5*dt) = phi(dt) + 0.5*dt*RHS[ phi(t) ]
     *
     *  qn is used to keep a copy of phi(t). The velocity field
     *  is fixed for the whole step. */

    double dt = 1.0;
    double * qn = NULL;

    qn = malloc(nop*nsites*sizeof(double));
    if (qn == NULL) fatal("malloc(qn) failed\n");
    memcpy(qn, phi_site, nop*nsites*sizeof(double));

    advection_fourth_order(fluxe, fluxw, fluxy, fluxz);
    advection_bcs_no_normal_flux(nop, fluxe, fluxw, fluxy, fluxz);

    be_rk(0.5*dt, qn);

    phi_halo();
    phi_gradients_compute();

    advection_fourth_order(fluxe, fluxw, fluxy, fluxz);
    advection_bcs_no_normal_flux(nop, fluxe, fluxw, fluxy, fluxz);

    be_rk(dt, qn);

    free(qn);
  }

  free(fluxe);
  free(fluxw);
  free(fluxy);
  free(fluxz);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_be_update
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
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
#ifdef COMPARE_KLUDGE
    if (cart_coords(Z) == 0 && kc == 1) continue;
    if (cart_coords(Z) == cart_size(Z) - 1 && kc == nlocal[Z]) continue;
#endif
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
	    for (ib = 0; ib < 3; ib++) {
	      trace_qw += q[ia][ib]*w[ib][ia];
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


/*****************************************************************************
 *
 *  RK TEST
 *
 *****************************************************************************/


static void be_rk(double dt, const double * qn) {

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

  coords_nlocal(nlocal);
  nop = phi_nop();
  xi = blue_phase_get_xi();

  assert(nop == 5);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

#ifdef COMPARE_KLUDGE
    if (cart_coords(Z) == 0 && kc == 1) continue;
    if (cart_coords(Z) == cart_size(Z) - 1 && kc == nlocal[Z]) continue;
#endif
	index = le_site_index(ic, jc, kc);

	phi_get_q_tensor(index, q);
	blue_phase_molecular_field(index, h);

	/* Velocity gradient tensor, symmetric and antisymmetric parts */

	hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);

	trace_qw = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    trace_qw += q[ia][ib]*w[ib][ia];
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

	q[X][X] = qn[nop*index + XX] + dt*(s[X][X] + Gamma_*h[X][X]
			 - fluxe[nop*index + XX] + fluxw[nop*index  + XX]
			 - fluxy[nop*index + XX] + fluxy[nop*indexj + XX]
			 - fluxz[nop*index + XX] + fluxz[nop*indexk + XX]);

	q[X][Y] = qn[nop*index + XY] + dt*(s[X][Y] + Gamma_*h[X][Y]
			 - fluxe[nop*index + XY] + fluxw[nop*index  + XY]
			 - fluxy[nop*index + XY] + fluxy[nop*indexj + XY]
			 - fluxz[nop*index + XY] + fluxz[nop*indexk + XY]);

	q[X][Z] = qn[nop*index + XZ] + dt*(s[X][Z] + Gamma_*h[X][Z]
			 - fluxe[nop*index + XZ] + fluxw[nop*index  + XZ]
			 - fluxy[nop*index + XZ] + fluxy[nop*indexj + XZ]
			 - fluxz[nop*index + XZ] + fluxz[nop*indexk + XZ]);

	q[Y][Y] = qn[nop*index + YY] + dt*(s[Y][Y] + Gamma_*h[Y][Y]
			 - fluxe[nop*index + YY] + fluxw[nop*index  + YY]
			 - fluxy[nop*index + YY] + fluxy[nop*indexj + YY]
			 - fluxz[nop*index + YY] + fluxz[nop*indexk + YY]);

	q[Y][Z] = qn[nop*index + YZ] + dt*(s[Y][Z] + Gamma_*h[Y][Z]
			 - fluxe[nop*index + YZ] + fluxw[nop*index  + YZ]
			 - fluxy[nop*index + YZ] + fluxy[nop*indexj + YZ]
			 - fluxz[nop*index + YZ] + fluxz[nop*indexk + YZ]);
	
	phi_set_q_tensor(index, q);

	/* Next site */
      }
    }
  }

  return;
}
