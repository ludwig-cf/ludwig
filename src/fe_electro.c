/*****************************************************************************
 *
 *  fe_electro.c
 *
 *  $Id$
 *
 *  Free energy related to electrokinetics (simple fluid).
 *
 *  We have F = \int dr f[psi, rho_a] where the potential and the
 *  charge species are described following the psi_t object.
 *
 *  The free energy density is
 *
 *  f[psi, rho_a] =
 *  \sum_a rho_a [kT(log(rho_a) - 1) - mu_ref_a + 0.5 Z_a e psi]
 *
 *  with mu_ref a reference chemical potential which we shall take
 *  to be zero. psi is the electric potential.
 *
 *  The related chemical potential is
 *
 *  mu_a = kT log(rho_a) + Z_a e psi
 *
 *  See, e.g., Rotenberg et al. Coarse-grained simualtions of charge,
 *  current and flow in heterogeneous media,
 *  Faraday Discussions \textbf{14}, 223--243 (2010).
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "physics.h"
#include "util.h"
#include "psi_s.h"
#include "psi_gradients.h"
#include "fe_electro.h"

struct fe_electro_s {
  fe_t super;
  psi_t * psi;           /* A reference to the electrokinetic quantities */
  physics_t * param;     /* For external field, temperature */
  double * mu_ref;       /* Reference mu currently unused (i.e., zero). */ 
  fe_electro_t * target; /* Device copy */
};

static fe_vt_t fe_electro_hvt = {
  (fe_free_ft)      fe_electro_free,
  (fe_target_ft)    fe_electro_target,
  (fe_fed_ft)       fe_electro_fed,
  (fe_mu_ft)        fe_electro_mu,
  (fe_mu_solv_ft)   fe_electro_mu_solv,
  (fe_str_ft)       fe_electro_stress_ex,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL
};

static  __constant__ fe_vt_t fe_electro_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_electro_fed,
  (fe_mu_ft)        fe_electro_mu,
  (fe_mu_solv_ft)   fe_electro_mu_solv,
  (fe_str_ft)       fe_electro_stress_ex,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL
};

/*****************************************************************************
 *
 *  fe_electro_create
 *
 *  A single static instance.
 *
 *  Retain a reference to the electrokinetics object psi.
 *
 *  Note: In this model we do not set the chemical potential.
 *        In the gradient method the ionic electrostatic forces 
 *        on the fluid are implicitly calculated through the 
 *        electric charge density and the electric field.
 *
 *****************************************************************************/

__host__ int fe_electro_create(psi_t * psi, fe_electro_t ** pobj) {

  int ndevice;
  fe_electro_t * fe = NULL;

  assert(pobj);
  assert(psi);

  fe = (fe_electro_t *) calloc(1, sizeof(fe_electro_t));
  if (fe == NULL) fatal("calloc() failed\n");

  fe->psi = psi;
  physics_ref(&fe->param);
  fe->super.func = &fe_electro_hvt;
  fe->super.id = FE_ELECTRO;

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_vt_t * vt;
    assert(0);
    /* Device implementation please */
    targetConstAddress(&vt, fe_electro_dvt);
  }

  *pobj = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_free
 *
 *****************************************************************************/

__host__ int fe_electro_free(fe_electro_t * fe) {

  assert(fe);

  if (fe->mu_ref) free(fe->mu_ref);
  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_target
 *
 *****************************************************************************/

__host__ int fe_electro_target(fe_electro_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_fed
 *
 *  The free energy density at position index:
 *
 *  \sum_a  rho_a [ kT(log(rho_a) - 1) + (1/2) Z_a e psi ]
 *
 *  If rho = 0, rho log(rho) gives 0 x -inf = nan, so use
 *  rho*log(rho + DBL_EPSILON); rho < 0 is erroneous.
 *
 *****************************************************************************/

__host__ __device__
int fe_electro_fed(fe_electro_t * fe, int index, double * fed) {

  int n;
  int nk;
  double e;
  double kt;
  double psi;
  double rho;

  assert(fe);
  assert(fe->psi);

  e = 0.0;
  physics_kt(&kt);
  psi_nk(fe->psi, &nk);
  psi_psi(fe->psi, index, &psi);

  for (n = 0; n < nk; n++) {
    psi_rho(fe->psi, index, n, &rho);
    assert(rho >= 0.0); /* For log(rho + epsilon) */
    e += rho*((log(rho + DBL_EPSILON) - 1.0) + 0.5*fe->psi->valency[n]*psi);
  }

  *fed = e;

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_mu
 *
 *  Here is the checmical potential for each species at position index.
 *  Performance sensitive, so we use direct references to fe->psi->rho
 *  etc.
 *
 *****************************************************************************/

__host__ __device__
int fe_electro_mu(fe_electro_t * fe, int index, double * mu) {

  int n;
  double kt;
  double rho;
  double psi;

  assert(fe);
  assert(fe->psi);

  psi_psi(fe->psi, index, &psi);
  physics_kt(&kt);

  for (n = 0; n < fe->psi->nk; n++) {
    psi_rho(fe->psi, index, n, &rho);
    assert(rho >= 0.0); /* For log(rho + epsilon) */
  
    mu[n] = kt*log(rho + DBL_EPSILON) + fe->psi->valency[n]*fe->psi->e*psi;
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_mu_solv
 *
 *  This is a dummy which just returns zero, as an implementation is
 *  required. Physically, there is no solvation chemical potential.
 *
 ****************************************************************************/

__host__ __device__
int fe_electro_mu_solv(fe_electro_t * fe, int index, int k, double * mu) {

  assert(mu);
  *mu = 0.0;

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_stress
 *
 *  The stress is
 *    S_ab = -epsilon ( E_a E_b - (1/2) d_ab E^2) + d_ab kt sum_k rho_k
 *  where epsilon is the (uniform) permittivity.
 *
 *  The last term is the ideal gas contribution which is excluded in the 
 *  excess stress tensor.
 *
 *****************************************************************************/

__host__ __device__
int fe_electro_stress(fe_electro_t * fe, int index, double s[3][3]) {

  int ia, ib, in;
  double epsilon;    /* Permittivity */
  double e[3];       /* Total electric field */
  double e2;         /* Magnitude squared */
  int nk;
  double rho;
  double kt, eunit, reunit;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  physics_kt(&kt);
  psi_nk(fe->psi, &nk);
  psi_unit_charge(fe->psi, &eunit);	 
  reunit = 1.0/eunit;

  psi_epsilon(fe->psi, &epsilon);
  psi_electric_field_d3qx(fe->psi, index, e);

  e2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    e[ia] *= kt*reunit;
    e2 += e[ia]*e[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -epsilon*(e[ia]*e[ib] - 0.5*d[ia][ib]*e2);

      /* Ideal gas contribution */
      for (in = 0; in < nk; in++) {
	psi_rho(fe->psi, index, in, &rho);
	s[ia][ib] += d[ia][ib] * kt * rho;

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_stress_ex
 *
 *  The excess stress is S_ab = -epsilon ( E_a E_b - (1/2) d_ab E^2)
 *  where epsilon is the (uniform) permittivity.
 *
 *****************************************************************************/

__host__ __device__
int fe_electro_stress_ex(fe_electro_t * fe, int index, double s[3][3]) {

  int ia, ib;
  double epsilon;    /* Permittivity */
  double e[3];       /* Total electric field */
  double e2;         /* Magnitude squared */
  double kt, eunit, reunit;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  physics_kt(&kt);
  psi_unit_charge(fe->psi, &eunit);	 
  reunit = 1.0/eunit;

  psi_epsilon(fe->psi, &epsilon);
  psi_electric_field_d3qx(fe->psi, index, e);

  e2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    e[ia] *= kt*reunit;
    e2 += e[ia]*e[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -epsilon*(e[ia]*e[ib] - 0.5*d[ia][ib]*e2);
    }
  }

  return 0;
}
