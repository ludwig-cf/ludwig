/*****************************************************************************
 *
 *  fe_electro.c
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
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2013)
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
#include "fe_electro.h"

typedef struct fe_electro_s fe_electro_t;

struct fe_electro_s {
  psi_t * psi;       /* A reference to the electrokinetic quantities */
  physics_t * param; /* For external field, temperature */
  double * mu_ref;   /* Reference potential currently unused (set to zero). */ 
};

static fe_electro_t * fe = NULL;

/*****************************************************************************
 *
 *  fe_electro_create
 *
 *  A single static instance.
 *
 *  Retain a reference to the electrokinetics object psi.
 *
 *****************************************************************************/

int fe_electro_create(psi_t * psi) {

  assert(fe == NULL);
  assert(psi);

  fe = calloc(1, sizeof(fe_electro_t));
  if (fe == NULL) fatal("calloc() failed\n");

  fe->psi = psi;
  physics_ref(&fe->param);

  fe_density_set(fe_electro_fed);
  fe_chemical_potential_set(fe_electro_mu);
  fe_chemical_stress_set(fe_electro_stress);

  return 0;
}

/*****************************************************************************
 *
 *  fe_electro_free
 *
 *****************************************************************************/

int fe_electro_free(void) {

  assert(fe);

  if (fe->mu_ref) free(fe->mu_ref);
  free(fe);
  fe = NULL;

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

double fe_electro_fed(const int index) {

  int n;
  int nk;
  double kt;
  double fed;
  double psi;
  double rho;
  double epsi;

  assert(fe);
  assert(fe->psi);

  fed = 0.0;
  physics_kt(&kt);
  psi_nk(fe->psi, &nk);
  psi_psi(fe->psi, index, &psi);
  epsi = 0.5*fe->psi->e*psi;

  for (n = 0; n < nk; n++) {

    psi_rho(fe->psi, index, n, &rho);
    assert(rho >= 0.0); /* For log(rho + epsilon) */

    fed += rho*(kt*(log(rho + DBL_EPSILON) - 1.0) + fe->psi->valency[n]*epsi);
  }

  return fed;
}

/*****************************************************************************
 *
 *  fe_electro_mu
 *
 *  Here is the checmical potential for species n at position index.
 *  Performance sensitive, so we use direct references to fe->psi->rho
 *  etc.
 *
 *****************************************************************************/

double fe_electro_mu(const int index, const int n) {

  double mu;
  double kt;
  double rho;

  assert(fe);
  assert(fe->psi);
  assert(n < fe->psi->nk);

  rho = fe->psi->rho[fe->psi->nk*index + n];
  physics_kt(&kt);

  assert(rho >= 0.0); /* For log(rho + epsilon) */
  
  mu = kt*log(rho + DBL_EPSILON)
    + fe->psi->valency[n]*fe->psi->e*fe->psi->psi[index];

  return mu;
}

/*****************************************************************************
 *
 *  fe_electro_stress
 *
 *  The stress is S_ab = -epsilon ( E_a E_b - (1/2) d_ab E^2)
 *  where epsilon is the (uniform) permittivity.
 *
 *  This does not include the external field.
 *
 *****************************************************************************/

void fe_electro_stress(const int index, double s[3][3]) {

  int ia, ib;
  double epsilon;    /* Permittivity */
  double e[3];       /* Electric field */
  double e2;         /* Magnitude squared */
  double e0[3];      /* External field */

  assert(fe);

  psi_epsilon(fe->psi, &epsilon);
  psi_electric_field(fe->psi, index, e);
//  psi_electric_field_d3q19(fe->psi, index, e);
  physics_e0(e0);

  e2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    e[ia] += e0[ia];
    e2 += e[ia]*e[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -epsilon*(e[ia]*e[ib] - 0.5*d_[ia][ib]*e2);
    }
  }

  return;
}
