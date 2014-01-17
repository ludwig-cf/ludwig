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
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "psi_s.h"
#include "colloids.h"
#include "free_energy.h"
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

int psi_force_grad_mu(psi_t * psi, hydro_t * hydro, double dt) {

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

	f[X] = -0.5*rho_elec*(psi->psi[index + xs] - psi->psi[index - xs])*dt;
	f[Y] = -0.5*rho_elec*(psi->psi[index + ys] - psi->psi[index - ys])*dt;
	f[Z] = -0.5*rho_elec*(psi->psi[index + zs] - psi->psi[index - zs])*dt;

	/* External field */

	f[X] += rho_elec*e0[X]*dt;
	f[Y] += rho_elec*e0[Y]*dt;
	f[Z] += rho_elec*e0[Z]*dt;

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

int psi_force_external_field(psi_t * psi, hydro_t * hydro, double dt) {

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

	f[X] = rho_elec*e0[X]*dt;
	f[Y] = rho_elec*e0[Y]*dt;
	f[Z] = rho_elec*e0[Z]*dt;

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
 *  Following Capuani et al J. Chem. Phys. 121, 973--986 (2004),
 *  we compute the force in the hydrodynamic sector by considering
 *  the fluxes in the Nernst Planck equation.
 *
 *  If the interfacial fluxes of charge are j_k for charge species
 *  n, we write
 *
 *     j_n = -D_n [grad rho_n + rho_n grad (beta mu_n) ]
 *
 *  where mu_n is the excess chemical potential.
 *
 *  We write a force contribution associated with the flux as
 *     f(i + 1/2, j, k) = \sum_n
 *     [ j_n(i + 1/2, j, k) / D_n - (rho_n(i+1, j, k) - rho_n(i,j,k))/dx ]
 *
 *  From this contribution 1/2 is accumulated to (i,j,k) and -1/2 to
 *  (i+1,j,k) ensuring conservation.
 *
 *  The terms in grad rho_n cancel and one is left with a force
 *      rho grad mu.
 *
 *****************************************************************************/

int psi_force_gradmu_conserve(psi_t * psi, hydro_t * hydro) {

  int ic, jc, kc, index;
  int zs, ys, xs;
  int n, nk, ia;
  int nlocal[3];
  int v[2];

  double rho0, rho1;
  double mu0, mu_s0, mu1, mu_s1;
  double b0, b1;
  double f[3], ftot[3];
  double e0[3];
  double beta;
  double eunit;

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
  psi_unit_charge(psi, &eunit);
  psi_beta(psi, &beta);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	f[X] = 0.0; f[Y] = 0.0; f[Z] = 0.0;
	for (ia = 0; ia < 3; ia++) {
	  ftot[ia] = 0.0;
	}

	for (n = 0; n < nk; n++) {

	  fe_mu_solv(index, n, &mu_s0);
	  mu0 = mu_s0 + psi->valency[n]*eunit*psi->psi[index];
	  rho0 = psi->rho[nk*index + n]*exp(beta*mu0);
	  b0 = exp(-beta*mu0);

	  /* x-direction (between ic and ic+1) */

	  fe_mu_solv(index + xs, n, &mu_s1);
	  mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index + xs] - e0[X]);
	  rho1 = psi->rho[nk*(index + xs) + n]*exp(beta*mu1);
	  b1 = exp(-beta*mu1);

	  f[X] -= 0.5*(b0 + b1)*(rho1 - rho0);
	  f[X] -= (psi->rho[nk*(index + xs) + n] - psi->rho[nk*index + n]);

	  /* x-direction (ic-1 and ic) */

	  fe_mu_solv(index - xs, n, &mu_s1);
	  mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index - xs] - e0[X]);
	  rho1 = psi->rho[nk*(index - xs) + n]*exp(beta*mu1);
	  b1 = exp(-beta*mu1);

	  f[X] += 0.5*(b0 + b1)*(rho1 - rho0);
	  f[X] += (psi->rho[nk*(index - xs) + n] - psi->rho[nk*index + n]);

          /* y-direction (between jc and jc+1) */

          fe_mu_solv(index + ys, n, &mu_s1);
          mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index + ys] - e0[Y]);
          rho1 = psi->rho[nk*(index + ys) + n]*exp(beta*mu1);
          b1 = exp(-beta*mu1);

          f[Y] -= 0.5*(b0 + b1)*(rho1 - rho0);
	  f[Y] -= (psi->rho[nk*(index + ys) + n] - psi->rho[nk*index + n]);

	  /* y-direction (jc-1 and jc) */

          fe_mu_solv(index - ys, n, &mu_s1);
          mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index - ys] - e0[Y]);
          rho1 = psi->rho[nk*(index - ys) + n]*exp(beta*mu1);
          b1 = exp(-beta*mu1);

          f[Y] += 0.5*(b0 + b1)*(rho1 - rho0);
	  f[Y] += (psi->rho[nk*(index - ys) + n] - psi->rho[nk*index + n]);


          /* z-direction (between kc and kc+1) */

          fe_mu_solv(index + zs, n, &mu_s1);
          mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index + zs] - e0[Z]);
          rho1 = psi->rho[nk*(index + zs) + n]*exp(beta*mu1);
          b1 = exp(-beta*mu1);

          f[Z] -= 0.5*(b0 + b1)*(rho1 - rho0);
	  f[Z] -= (psi->rho[nk*(index + zs) + n] - psi->rho[nk*index + n]);

          /* z-direction (kc-1 -> kc) */

          fe_mu_solv(index - zs, n, &mu_s1);
          mu1 = mu_s1 + psi->valency[n]*eunit*(psi->psi[index - zs] - e0[Z]);
          rho1 = psi->rho[nk*(index - zs) + n]*exp(beta*mu1);
          b1 = exp(-beta*mu1);

          f[Z] += 0.5*(b0 + b1)*(rho1 - rho0);
	  f[Z] += (psi->rho[nk*(index - zs) + n] - psi->rho[nk*index + n]);

	  for (ia = 0; ia < 3; ia++) {
	    f[ia] = 0.5*f[ia];
	    ftot[ia] += f[ia];
	  }

	}
      }
    }
  }

  printf("FTOT %14.7e %14.7e %14.7e\n", ftot[X], ftot[Y], ftot[Z]);


  return 0;
}
