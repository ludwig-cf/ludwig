/*****************************************************************************
 *
 *  fe_electro_symetric.c
 *
 *  A "coupling" free energy which is a combination of binary
 *  symmetric and electrokinetic parts.
 *
 *  The total chemical potential to enter the Cahn-Hilliard equation
 *  is
 *
 *    mu_phi = mu^mix + mu^solv + mu^el
 *
 *  with mu^mix the usual symmetric contribution, 
 *
 *    mu^solv = (1/2) [ rho(+)Delta mu(+) + rho(-)Delta mu(-) ]
 *
 *  for the two-charge case (+/-), and
 *
 *    mu^el = - (1/2) gamma epsilonbar E^2
 *
 *  where gamma is the dielectric contrast and epsilonbar is the mean
 *  dielectric contant for the two phases. E is the external electric
 *  field.
 *
 *
 *  This follows Rotenberg et al. Coarse-grained simulations of
 *  charge, current and flow in heterogeneous media,
 *  Faraday Discussions \textbf{144} 223--243 (2010).
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2013)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "physics.h"
#include "symmetric.h"
#include "fe_electro.h"
#include "fe_electro_symmetric.h"
#include "psi.h"
#include "psi_gradients.h"

typedef struct fe_electro_symmetric_s fe_es_t;

struct fe_electro_symmetric_s {
  field_t * phi;
  field_grad_t * gradphi;
  psi_t * psi;
  double epsilon1;    /* Dielectric constant phase 1 */
  double epsilon2;    /* Dielectric constant phase 2 */
  double epsilonbar;  /* Mean dielectric */
  double gamma;       /* Dielectric contrast */

  int nk;             /* Number of species - same os psi_nk() */
  double * deltamu;   /* Solvation free energy difference (each species) */
};

/* A static implementation that holds relevant quantities for this model */
static fe_es_t * fe = NULL; 

/*****************************************************************************
 *
 *  fe_es_create
 *
 *  Single static instance: we have F = F[ phi, grad phi, psi ]
 *  where psi provides the electrokinetic quantities.
 *
 *****************************************************************************/

int fe_es_create(field_t * phi, field_grad_t * gradphi, psi_t * psi) {

  assert(fe == NULL);
  assert(phi);
  assert(gradphi);
  assert(psi);

  fe = calloc(1, sizeof(fe_es_t));
  if (fe == NULL) fatal("calloc(fe_es_t) failed\n");

  fe->phi = phi;
  fe->gradphi = gradphi;
  fe->psi = psi;

  psi_nk(psi, &fe->nk);
  fe->deltamu = calloc(fe->nk, sizeof(double));
  if (fe->deltamu == NULL) fatal("calloc(fe->deltamu) failed\n");

  fe_electro_create(psi);
  fe_density_set(fe_es_fed);
  fe_mu_solv_set(fe_es_mu_ion_solv);
  fe_chemical_potential_set(fe_es_mu_phi);
  fe_chemical_stress_set(fe_es_stress_ex);

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_free
 *
 *****************************************************************************/

int fe_es_free(void) {

  if (fe) {
    fe_electro_free();
    free(fe->deltamu);
    free(fe);
  }

  fe = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_fed
 *
 *  We have contributions: f_electro + f_symmetric + f_solvation
 *
 *****************************************************************************/

double fe_es_fed(int index) {

  int n;
  double rho;
  double fsolv;
  double fed;

  assert(fe);

  fed = fe_electro_fed(index);
  fed += symmetric_free_energy_density(index);

  for (n = 0; n < fe->nk; n++) {
    psi_rho(fe->psi, index, n, &rho);
    fe_es_mu_ion_solv(index, n, &fsolv);
    fed += rho*fsolv;
  }

  return fed;
}

/*****************************************************************************
 *
 *  fe_es_mu_phi
 *
 *  The chemical potential mu_phi
 *
 *      mu_phi = mu_phi_mix + mu_phi_solv + mu_phi_el
 *
 *  Note: mu_phi_solv needs to be in agreement with the terms in fe_es_mu_ion()
 *
 *****************************************************************************/

double fe_es_mu_phi(const int index, const int nop) {

  int ia, in;
  double e0[3], e[3];  /* Electric field */
  double rho;          /* Charge density */
  double mu;           /* Result */

  assert(fe);
  assert(nop == 0); /* Only zero if relevant */

  mu = symmetric_chemical_potential(index, 0);

  /* Solvation piece */

  for (in = 0; in < fe->nk; in++) {
    psi_rho(fe->psi, index, in, &rho);
    mu += 0.5*rho*fe->deltamu[in];
  }

  /* Total electric field contribution */

  physics_e0(e0);
  psi_electric_field_d3qx(fe->psi, index, e); 

  for (ia = 0; ia < 3; ia++) {
    e[ia] += e0[ia];
  }

  mu += -0.5*fe->gamma*fe->epsilonbar*(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);

  return mu;
}

/*****************************************************************************
 *
 *  fe_es_mu_ion_solv
 *
 *  Solvation chemical potential, parameterised here as
 *
 *     mu_ion_solv = 1/2 * deltamu * [1 + phi(r)]
 *
 *  Note: this needs to be in agreement with the terms in fe_es_mu_phi()
 *
 *****************************************************************************/

int fe_es_mu_ion_solv(int index, int n, double * mu) {

  double phi;

  assert(fe);
  assert(mu);
  assert(fe->phi);
  assert(n < fe->nk);

  field_scalar(fe->phi, index, &phi);
  *mu = 0.5*fe->deltamu[n]*(1.0 + phi);

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_epsilon_set
 *
 *  Set both the epsilon at the same time, so that the contrast gamma
 *  and mean may also be computed.
 *
 *****************************************************************************/

int fe_es_epsilon_set(double e1, double e2) {

  assert(fe);

  fe->epsilon1 = e1;
  fe->epsilon2 = e2;
  fe->epsilonbar = 0.5*(e1 + e2);
  fe->gamma = (e1 - e2) / (e1 + e2);

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_deltamu_set
 *
 *  The delta mu values should match the order of the species in
 *  the corresponding psi_t object.
 *
 *****************************************************************************/

int fe_es_deltamu_set(int nk, double * deltamu) {

  int n;

  assert(fe);
  assert(nk == fe->nk);
  assert(deltamu);

  for (n = 0; n < nk; n++) {
    fe->deltamu[n] = deltamu[n];
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_var_epsilon
 *
 *  epsilon(r) = epsilonbar [ 1 - gamma phi(r) ]
 *
 *  If phi = +1 then epsilon(r) = epsilon2, as set in the call to
 *  fe_es_epsilon_set().
 *
 *  The function is of signature f_vare_t (see psi_sor.h).
 *
 *****************************************************************************/

int fe_es_var_epsilon(int index, double * epsilon) {

  double phi;

  assert(fe);
  assert(fe->phi);

  field_scalar(fe->phi, index, &phi);
  *epsilon = fe->epsilonbar*(1.0 - fe->gamma*phi);

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_stress_ex
 *
 *  The full stress has three parts:
 *
 *  S^full = S^elec + S^symmetric + S^coupling
 *
 *  S^elec_ab = - [epsilon(r) E_a E_b - (1/2) epsilonbar d_ab E^2)]
 *  S^symmetric = S(phi, grad phi) following symmetric.c
 *
 *  S^coupling =
 *    (1/2) phi d_ab [ epsilonbar gamma E^2 + \sum_k rho_k deltamu_k ]
 *
 *  The field term comes from
 *
 *  S^elec = - [D_a E_b - (1/2) epsilonbar d_ab E^2]
 * 
 *    where D_a is the electric displacement. The functional form of
 *    epsilon(r) agrees with fe_es_var_epsilon() above.
 *
 *  Note that the sign of the electro- and symmetric- parts of the
 *  stress is already accounted for in the relevant functions.
 *
 *  Finally, the true Maxwell stress includes the total electric
 *  field. 
 *
 *****************************************************************************/

void fe_es_stress_ex(const int index, double s[3][3]) {

  int ia, ib;

  double phi, rho;
  double s_couple;
  double s_el;
  double epsloc;
  double e0[3], e[3]; /* External and 'internal' electric field. */
  double e2;
  double eunit, reunit, kt;

  physics_kt(&kt);
  psi_unit_charge(fe->psi, &eunit);
  reunit = 1.0/eunit;

  symmetric_chemical_stress(index, s); 

  /* Coupling part, requires phi, total field */

  field_scalar(fe->phi, index, &phi);
  psi_electric_field_d3qx(fe->psi, index, e);

  physics_e0(e0);

  e2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    e[ia] += e0[ia];
    e[ia] *= kt*reunit;
    e2 += e[ia]*e[ia];
  }

  s_couple = 0.5*phi*fe->epsilonbar*fe->gamma*e2;

  for (ia = 0; ia < fe->nk; ia++) {
    psi_rho(fe->psi, index, ia, &rho);
    s_couple += 0.5*phi*rho*fe->deltamu[ia];
  }

  /* Local permittivity, requires phi */
  fe_es_var_epsilon(index, &epsloc);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s_el = -epsloc*(e[ia]*e[ib] - 0.5*d_[ia][ib]*e2);
      s[ia][ib] += s_el + d_[ia][ib]*s_couple;
    }
  }

  return;
}
