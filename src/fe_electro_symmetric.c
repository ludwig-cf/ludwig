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
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "physics.h"
#include "psi.h"
#include "psi_gradients.h"
#include "fe_electro_symmetric.h"

struct fe_electro_symmetric_s {
  fe_t super;
  psi_t * psi;
  fe_symm_t * fe_symm;     /* Symmetric part */
  fe_electro_t * fe_elec;  /* Electro part */
  fe_es_param_t * param;   /* Constant parameters */

  fe_es_t * target;        /* Device implementation */
};

static __constant__ fe_es_param_t const_param;

static fe_vt_t fe_es_hvt = {
  (fe_free_ft)      fe_es_free,
  (fe_target_ft)    fe_es_target,
  (fe_fed_ft)       fe_es_fed,
  (fe_mu_ft)        fe_es_mu_phi,
  (fe_mu_solv_ft)   fe_es_mu_ion_solv,
  (fe_str_ft)       fe_es_stress_ex,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL
};

static  __constant__ fe_vt_t fe_es_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       NULL,
  (fe_mu_ft)        NULL,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL
};

/*****************************************************************************
 *
 *  fe_es_create
 *
 *  We have F = F[ phi, grad phi, psi ]
 *  where psi provides the electrokinetic quantities.
 *
 *****************************************************************************/

__host__ int fe_es_create(fe_symm_t * symm, fe_electro_t * elec,
			  psi_t * psi, fe_es_t ** pobj) {

  int ndevice;
  fe_es_t * fe = NULL;

  assert(symm);
  assert(elec);
  assert(psi);

  fe = (fe_es_t*) calloc(1, sizeof(fe_es_t));
  if (fe == NULL) fatal("calloc(fe_es_t) failed\n");
  fe->param = (fe_es_param_t *) calloc(1, sizeof(fe_es_param_t));
  if (fe->param == NULL) fatal("calloc(fe_es_param_t) failed\n");

  fe->fe_symm = symm;
  fe->fe_elec = elec;
  fe->psi = psi;

  fe->super.func = &fe_es_hvt;
  fe->super.id = FE_ELECTRO_SYMMETRIC;

  psi_nk(psi, &fe->param->nk);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_vt_t * vt;
    fe_es_param_t * tmp;
    assert(0); /* device implementation pending */
    targetCalloc((void **) &fe->target, sizeof(fe_es_t));
    targetConstAddress((void **) &tmp, const_param);
    copyToTarget(&fe->target->param, tmp, sizeof(fe_es_param_t *));
    targetConstAddress((void **) &vt, fe_es_dvt);
  }

  *pobj = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_free
 *
 *****************************************************************************/

__host__ int fe_es_free(fe_es_t * fe) {

  assert(fe);

  free(fe->param);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_target
 *
 *****************************************************************************/

__host__ int fe_es_target(fe_es_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_fed
 *
 *  We have contributions: f_electro + f_symmetric + f_solvation
 *
 *****************************************************************************/

__host__ int fe_es_fed(fe_es_t * fe, int index, double * fed) {

  int n;
  double rho;
  double fsolv;
  double e1;
  double e2;
  double e3;

  assert(fe);

  fe_electro_fed(fe->fe_elec, index, &e1);
  fe_symm_fed(fe->fe_symm, index, &e2);

  e3 = 0.0;
  for (n = 0; n < fe->param->nk; n++) {
    psi_rho(fe->psi, index, n, &rho);
    fe_es_mu_ion_solv(fe, index, n, &fsolv);
    e3 += rho*fsolv;
  }

  *fed = e1 + e2 + e3;

  return 0;
}

/*****************************************************************************
 *
 *  fe_es_mu_phi
 *
 *  The chemical potential mu_phi
 *
 *      mu_phi = mu_phi_mix + mu_phi_solv + mu_phi_el
 *
 *  Note: mu_phi_solv needs to be in agreement with 
 *        the terms in fe_es_mu_ion()
 *
 *****************************************************************************/

__host__ int fe_es_mu_phi(fe_es_t * fe, int index, double * mu) {

  int in, ia;
  double e[3];         /* Total electric field */
  double e2;
  double rho;          /* Charge density */
  double kt, eunit, reunit;
  physics_t * phys = NULL;

  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_unit_charge(fe->psi, &eunit);
  reunit = 1.0/eunit;

  assert(fe);

  /* Contribution from compositional order parameter */

  fe_symm_mu(fe->fe_symm, index, mu);

  /* Contribution from solvation */

  for (in = 0; in < fe->param->nk; in++) {
    psi_rho(fe->psi, index, in, &rho);
    *mu += 0.5*rho*fe->param->deltamu[in]*kt;
  }

  /* Electric field contribution */
 
  e2 = 0.0;

  psi_electric_field_d3qx(fe->psi, index, e); 

  for (ia = 0; ia < 3; ia++) {
    e[ia] *= kt*reunit;
    e2 += e[ia]*e[ia];
  }

  *mu += 0.5*fe->param->gamma*fe->param->epsilonbar*e2;

  return 0;
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

__host__ int fe_es_mu_ion_solv(fe_es_t * fe, int index, int n, double * mu) {

  double phi;
 
  assert(fe);
  assert(mu);
  assert(n < fe->param->nk);

  field_scalar(fe->fe_symm->phi, index, &phi);
  *mu = 0.5*fe->param->deltamu[n]*(1.0 + phi);

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

__host__ int fe_es_epsilon_set(fe_es_t * fe, double e1, double e2) {

  assert(fe);

  fe->param->epsilon1 = e1;
  fe->param->epsilon2 = e2;
  fe->param->epsilonbar = 0.5*(e1 + e2);
  fe->param->gamma = (e1 - e2) / (e1 + e2);

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

__host__ int fe_es_deltamu_set(fe_es_t * fe, int nk, double * deltamu) {

  int n;

  assert(fe);
  assert(nk == fe->param->nk);
  assert(nk <= PSI_NKMAX);
  assert(deltamu);

  for (n = 0; n < nk; n++) {
    fe->param->deltamu[n] = deltamu[n];
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

__host__ int fe_es_var_epsilon(fe_es_t * fe, int index, double * epsilon) {

  double phi;

  assert(fe);
  assert(epsilon);

  field_scalar(fe->fe_symm->phi, index, &phi);
  *epsilon = fe->param->epsilonbar*(1.0 - fe->param->gamma*phi);

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
 *    (1/2) d_ab [ epsilonbar gamma E^2 + \sum_k rho_k deltamu_k ]
 *
 *  The field term comes from
 *
 *  S^elec = - [D_a E_b - (1/2) epsilonbar d_ab E^2]
 * 
 *    where D_a is the electric displacement. The functional form of
 *    epsilon(r) agrees with fe_es_var_epsilon() above.
 *
 *  Note that the sign of the electro- and symmetric- parts of the
 *  stress is already accounted for in the relevant functions and
 *  that deltamu_k is given in units of kt and must be dressed for
 *  the force calculation.
 *
 *  Finally, the true Maxwell stress includes the total electric
 *  field. 
 *
 *****************************************************************************/

__host__ int fe_es_stress_ex(fe_es_t * fe, int index, double s[3][3]) {

  int ia, ib;

  double phi, rho;
  double s_couple;
  double s_el;
  double epsloc;
  double e[3];     /* Total electric field */
  double e2;
  double kt, eunit, reunit;
  physics_t * phys = NULL;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  physics_ref(&phys);
  physics_kt(phys, &kt);
  psi_unit_charge(fe->psi, &eunit);
  reunit = 1.0/eunit;

  fe_symm_str(fe->fe_symm, index, s); 

  /* Coupling part
     requires phi and total electric field */

  field_scalar(fe->fe_symm->phi, index, &phi);
  psi_electric_field_d3qx(fe->psi, index, e);

  e2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    e[ia] *= kt*reunit;
    e2 += e[ia]*e[ia];
  }
  
  /* Dielectric part */

  s_couple = 0.5*fe->param->epsilonbar*fe->param->gamma*e2;

  /* Solvation part */

  for (ia = 0; ia < fe->param->nk; ia++) {
    psi_rho(fe->psi, index, ia, &rho);
    s_couple += 0.5*rho*fe->param->deltamu[ia]*kt;
  }

  /* Electrostatic part
     local permittivity depends implicitly on phi */

  fe_es_var_epsilon(fe, index, &epsloc);
 
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      s_el = -epsloc*(e[ia]*e[ib] - 0.5*d[ia][ib]*e2);
      s[ia][ib] += s_el + d[ia][ib]*s_couple;

    }
  }

  return 0;
}
