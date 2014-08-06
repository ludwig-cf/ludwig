
#define INCLUDED_FROM_TARGET

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "targetDP.h"
#include "phi.h"
#include "phi_gradients.h"
#include "util.h"
#include "symmetric.h"

//TODO set these from symmetric.c in some init routine.
static double a_     = -0.003125;
static double b_     = +0.003125;
static double kappa_ = +0.002;

/****************************************************************************
 *
 *  symmetric_chemical_potential
 *
 *  The chemical potential \mu = \delta F / \delta \phi
 *                             = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *
 ****************************************************************************/

TARGET double symmetric_chemical_potential2(const int index, const int nop) {

  double phi;
  double delsq_phi;
  double mu;

  assert(nop == 0);

  phi = phi_get_phi_site(index);
  delsq_phi = phi_gradients_delsq(index);

  mu = a_*phi + b_*phi*phi*phi - kappa_*delsq_phi;

  return mu;
}

/****************************************************************************
 *
 *  symmetric_chemical_stress
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/

TARGET void symmetric_chemical_stress2(const int index, double s[3][3]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  phi = phi_get_phi_site(index);
  phi_gradients_grad(index, grad_phi);
  delsq_phi = phi_gradients_delsq(index);

  p0 = 0.5*a_*phi*phi + 0.75*b_*phi*phi*phi*phi
    - kappa_*phi*delsq_phi - 0.5*kappa_*dot_product(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_[ia][ib]	+ kappa_*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}
