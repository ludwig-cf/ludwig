/*****************************************************************************
 *
 *  psi_force.c
 *
 *  Compute the force on the fluid originating with charge.
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga
 *    Oliver Henrich
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "psi_s.h"
#include "colloids.h"
#include "psi_force.h"

/*****************************************************************************
 *
 *  psi_force_grad_mu
 *
 *  The force density is
 *    f_a = - \sum_k rho_k grad_a mu^ex_k
 *  where mu_ex is the excess chemical potential (above ideal gas part).
 *  So
 *    f_a = - \sum_k rho_k grad_a z_k e psi
 *        = - rho_el grad_a psi
 *
 *  The external electric field term is just f = rho E_0
 *
 *  We allow hydro to be NULL, in which case there is no force.
 *
 ****************************************************************************/

int psi_force_grad_mu(psi_t * psi, hydro_t * hydro) {

  int ic, jc, kc, index;
  int zs, ys, xs;
  int nhalo;
  int nlocal[3];

  double rho_elec;
  double f[3];
  double e0[3];

  if (hydro == NULL) return 0;
  assert(psi);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  assert(nhalo >= 1);

  physics_e0(e0);

  /* Memory strides */
  zs = 1;
  ys = (nlocal[Z] + 2*nhalo)*zs;
  xs = (nlocal[Y] + 2*nhalo)*ys;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	psi_rho_elec(psi, index, &rho_elec);

	/* "Internal" field */

	f[X] = -0.5*rho_elec*(psi->psi[index + xs] - psi->psi[index - xs]);
	f[Y] = -0.5*rho_elec*(psi->psi[index + ys] - psi->psi[index - ys]);
	f[Z] = -0.5*rho_elec*(psi->psi[index + zs] - psi->psi[index - zs]);

	/* External field */

	f[X] += rho_elec*e0[X];
	f[Y] += rho_elec*e0[Y];
	f[Z] += rho_elec*e0[Z];

	hydro_f_local_add(hydro, index, f);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_external_field
 *
 *  Add the contribution to the body force on the fluid from the external
 *  field. At solid points, the result is just ignored at the collision
 *  stage.
 *
 *    f_a = rho_elec E_a
 *
 *  The total net momentum input is assured to be zero if strict
 *  electroneutrality is maintined. (This may involve colloid
 *  contributions coming from body force on particles.)
 *
 *****************************************************************************/

int psi_force_external_field(psi_t * psi, hydro_t * hydro) {

  int ic, jc, kc, index;
  int nlocal[3];

  double rho_elec, e2;
  double f[3];
  double e0[3];

  if (hydro == NULL) return 0;

  physics_e0(e0);
  e2 = e0[X]*e0[X] + e0[Y]*e0[Y] + e0[Z]*e0[Z];

  if (e2 == 0.0) return 0;

  coords_nlocal(nlocal);
  assert(psi);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	psi_rho_elec(psi, index, &rho_elec);

	f[X] = rho_elec*e0[X];
	f[Y] = rho_elec*e0[Y];
	f[Z] = rho_elec*e0[Z];

	hydro_f_local_add(hydro, index, f);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_gradmu_conserve
 *
 *  This routine computes the force on the fluid via the gradient
 *  of the chemical potential.
 *
 *  First, we compute a correction which ensures global net momentum
 *  is unchanged. This must take account of colloids, if present.
 *  (There is no direct force on the colloid in this approach.)
 *
 *  This requires MPI_Allreduce().
 *
 *  The resultant force is accumulated to the hydrodynamic sector.
 *  No action is required if hydro is NULL.
 *
 *  One is relying on overall electroneutrality for this to be a
 *  sensible procedure.
 *
 *****************************************************************************/

int psi_force_gradmu_conserve(psi_t * psi, hydro_t * hydro) {

  int ic, jc, kc, index;
  int zs, ys, xs;
  int nk, ia;
  int nlocal[3];
  int ncell[3];
  int v[2];

  double rho_elec;
  double f[3];
  double flocal[4] = {0.0, 0.0, 0.0, 0.0};
  double fsum[4];
  double e0[3];

  colloid_t * pc = NULL;
  MPI_Comm comm;

  if (hydro == NULL) return 0;
  assert(psi);

  physics_e0(e0);

  coords_nlocal(nlocal);
  coords_strides(&xs, &ys, &zs);
  comm = cart_comm();

  psi_nk(psi, &nk);
  assert(nk == 2);              /* This rountine is not completely general */

  psi_valency(psi, 0, v);
  psi_valency(psi, 1, v + 1);


  /* Compute force without correction. */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

	pc = colloid_at_site_index(index);
	if (pc) continue;

	psi_rho_elec(psi, index, &rho_elec);

        /* "Internal" field */

        f[X] = -0.5*rho_elec*(psi->psi[index + xs] - psi->psi[index - xs]);
        f[Y] = -0.5*rho_elec*(psi->psi[index + ys] - psi->psi[index - ys]);
        f[Z] = -0.5*rho_elec*(psi->psi[index + zs] - psi->psi[index - zs]);

        /* External field */

        f[X] += rho_elec*e0[X];
        f[Y] += rho_elec*e0[Y];
        f[Z] += rho_elec*e0[Z];

	/* Accumulate */

	flocal[X] += f[X];
	flocal[Y] += f[Y];
	flocal[Z] += f[Z];
	flocal[3] += 1.0; /* fluid volume */
      }
    }
  }

  /* Colloid contribution */

  ncell[X] = Ncell(X); ncell[Y] = Ncell(Y); ncell[Z] = Ncell(Z);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	pc = colloids_cell_list(ic, jc, kc);
	while (pc) {
	  for (ia = 0; ia < 3; ia++) {
	    flocal[ia] += pc->s.q0*v[0]*e0[ia];
	    flocal[ia] += pc->s.q1*v[1]*e0[ia];
	  }
	  pc = pc->next;
	}
      }
    }
  }

  MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);

  fsum[X] /= fsum[3];
  fsum[Y] /= fsum[3];
  fsum[Z] /= fsum[3];

  /* Now actually compute the force with the correction and store */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

	pc = colloid_at_site_index(index);
	if (pc) continue;

        psi_rho_elec(psi, index, &rho_elec);

        /* "Internal" field */

        f[X] = -0.5*rho_elec*(psi->psi[index + xs] - psi->psi[index - xs]);
        f[Y] = -0.5*rho_elec*(psi->psi[index + ys] - psi->psi[index - ys]);
        f[Z] = -0.5*rho_elec*(psi->psi[index + zs] - psi->psi[index - zs]);

        /* External field, and correction */

        f[X] += rho_elec*e0[X] - fsum[X];
        f[Y] += rho_elec*e0[Y] - fsum[Y];
        f[Z] += rho_elec*e0[Z] - fsum[Z];

	hydro_f_local_add(hydro, index, f);
      }
    }
  }

  return 0;
}
