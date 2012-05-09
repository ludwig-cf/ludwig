/*****************************************************************************
 *
 *  psi_force.c
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "lattice.h"
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
 *  Electric field term?
 *
 ****************************************************************************/

int psi_force_grad_mu(psi_t * psi) {

  int ic, jc, kc, index;
  int zs, ys, xs;
  int nhalo;
  int nlocal[3];

  double rho_elec;
  double f[3];

  assert(psi);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  assert(nhalo >= 1);

  /* Memory strides */
  zs = 1;
  ys = (nlocal[Z] + 2*nhalo)*zs;
  xs = (nlocal[Y] + 2*nhalo)*ys;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	psi_rho_elec(psi, index, &rho_elec);

	f[X] = -0.5*rho_elec*(psi->psi[index + xs] - psi->psi[index - xs]);
	f[Y] = -0.5*rho_elec*(psi->psi[index + ys] - psi->psi[index - ys]);
	f[Z] = -0.5*rho_elec*(psi->psi[index + zs] - psi->psi[index - zs]);

	hydrodynamics_add_force_local(index, f);
      }
    }
  }

  return 0;
}
