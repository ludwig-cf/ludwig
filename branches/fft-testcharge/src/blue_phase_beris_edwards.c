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
#include "colloids_Q_tensor.h"
#include "advection.h"
#include "advection_bcs.h"
#include "blue_phase.h"
#include "blue_phase_beris_edwards.h"
#include "advection_s.h"

static int blue_phase_be_update(field_t * fq, hydro_t * hydro, advflux_t * f,
				map_t * map);
static double Gamma_;     /* Collective rotational diffusion constant */


/*****************************************************************************
 *
 *  blue_phase_beris_edwards
 *
 *  Driver routine for the update.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *****************************************************************************/

int blue_phase_beris_edwards(field_t * fq, hydro_t * hydro, map_t * map) {

  int nf;
  advflux_t * flux = NULL;

  assert(fq);
  assert(map);

  /* Set up advective fluxes (which default to zero),
   * work out the hydrodynmaic stuff if required, and do the update. */

  field_nf(fq, &nf);
  assert(nf == NQAB);

  advflux_create(nf, &flux);

  if (hydro) {
    hydro_u_halo(hydro); /* Can move this to main to make more obvious? */
    colloids_fix_swd(hydro, map);

    hydro_lees_edwards(hydro);

    advection_x(flux, hydro, fq);
    advection_bcs_no_normal_flux(nf, flux, map);
  }

  blue_phase_be_update(fq, hydro, flux, map);
  advflux_free(flux);

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_be_update
 *
 *  Update q via Euler forward step. Note here we only update the
 *  5 independent elements of the Q tensor.
 *
 *  hydro is allowed to be NULL, in which case we only have relaxational
 *  dynamics.
 *
 *  Note that solid objects (colloids) are currently treated by evolving
 *  the order parameter inside, but with no hydrodynamics.
 *
 *****************************************************************************/

static int blue_phase_be_update(field_t * fq, hydro_t * hydro,
				advflux_t * flux, map_t * map) {
  int ic, jc, kc;
  int ia, ib, id;
  int index, indexj, indexk;
  int nlocal[3];
  int nf;
  int status;

  double q[3][3];
  double w[3][3];
  double d[3][3];
  double h[3][3];
  double s[3][3];
  double omega[3][3];
  double trace_qw;
  double xi;

  const double dt = 1.0;

  assert(fq);
  assert(flux);
  assert(map);

  coords_nlocal(nlocal);
  field_nf(fq, &nf);
  assert(nf == NQAB);

  xi = blue_phase_get_xi();

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);

	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(fq, index, q);
	blue_phase_molecular_field(index, h);

	if (hydro) {

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

	  hydro_u_gradient_tensor(hydro, ic, jc, kc, w);
	  
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
	      s[ia][ib] = -2.0*xi*(q[ia][ib] + r3_*d_[ia][ib])*trace_qw;
	      for (id = 0; id < 3; id++) {
		s[ia][ib] +=
		  (xi*d[ia][id] + omega[ia][id])*(q[id][ib] + r3_*d_[id][ib])
		+ (q[ia][id] + r3_*d_[ia][id])*(xi*d[id][ib] - omega[id][ib]);
	      }
	    }
	  }
	}
	     
	/* Here's the full hydrodynamic update. */
	  
	indexj = le_site_index(ic, jc-1, kc);
	indexk = le_site_index(ic, jc, kc-1);

	q[X][X] += dt*(s[X][X] + Gamma_*h[X][X]
		       - flux->fe[nf*index + XX] + flux->fw[nf*index  + XX]
		       - flux->fy[nf*index + XX] + flux->fy[nf*indexj + XX]
		       - flux->fz[nf*index + XX] + flux->fz[nf*indexk + XX]);

	q[X][Y] += dt*(s[X][Y] + Gamma_*h[X][Y]
		       - flux->fe[nf*index + XY] + flux->fw[nf*index  + XY]
		       - flux->fy[nf*index + XY] + flux->fy[nf*indexj + XY]
		       - flux->fz[nf*index + XY] + flux->fz[nf*indexk + XY]);

	q[X][Z] += dt*(s[X][Z] + Gamma_*h[X][Z]
		       - flux->fe[nf*index + XZ] + flux->fw[nf*index  + XZ]
		       - flux->fy[nf*index + XZ] + flux->fy[nf*indexj + XZ]
		       - flux->fz[nf*index + XZ] + flux->fz[nf*indexk + XZ]);

	q[Y][Y] += dt*(s[Y][Y] + Gamma_*h[Y][Y]
		       - flux->fe[nf*index + YY] + flux->fw[nf*index  + YY]
		       - flux->fy[nf*index + YY] + flux->fy[nf*indexj + YY]
		       - flux->fz[nf*index + YY] + flux->fz[nf*indexk + YY]);

	q[Y][Z] += dt*(s[Y][Z] + Gamma_*h[Y][Z]
		       - flux->fe[nf*index + YZ] + flux->fw[nf*index  + YZ]
		       - flux->fy[nf*index + YZ] + flux->fy[nf*indexj + YZ]
		       - flux->fz[nf*index + YZ] + flux->fz[nf*indexk + YZ]);
	field_tensor_set(fq, index, q);

	/* Next site */
      }
    }
  }

  return 0;
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
