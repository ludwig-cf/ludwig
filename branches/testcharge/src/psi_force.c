/*****************************************************************************
 *
 *  psi_force.c
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "hydro.h"
#include "physics.h"
#include "psi_s.h"

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
