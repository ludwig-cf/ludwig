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
 *  We may also have an (external) electric field E.
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
 *  (c) The University of Edinburgh (2013)
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "psi_s.h"
#include "fe_electro.h"

typedef struct fe_electro_s fe_electro_t;

struct fe_electro_s {
  psi_t * psi;       /* A reference to the electrokinetic quantities */
  double kt;         /* k_B T */
  double * mu_ref;   /* Reference potential currently unused (set to zero). */ 
  double ex[3];      /* External field */
};

static fe_electro_t * fe = NULL;

/*****************************************************************************
 *
 *  fe_electro_create
 *
 *  A single static instance.
 *
 *  Retain a reference to the electrokinetics object psi.
 *  The Boltzmann factor beta is used to compute a local copy
 *  of kT.
 *
 *****************************************************************************/

int fe_electro_create(psi_t * psi) {

  double beta;

  assert(fe == NULL);
  assert(psi);

  fe = calloc(1, sizeof(fe_electro_t));
  if (fe == NULL) fatal("calloc() failed\n");

  fe->psi = psi;

  psi_beta(psi, &beta);
  fe->kt = 1.0/beta;

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
 *  fe_electro_ext_set
 *
 *****************************************************************************/

int fe_electro_ext_set(double ext_field[3]) {

  assert(fe);

  fe->ex[X] = ext_field[X];
  fe->ex[Y] = ext_field[Y];
  fe->ex[Z] = ext_field[Z];

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
  double fed;
  double psi;
  double rho;
  double epsi;

  assert(fe);
  assert(fe->psi);

  fed = 0.0;
  psi_nk(fe->psi, &nk);
  psi_psi(fe->psi, index, &psi);
  epsi = 0.5*fe->psi->e*psi;

  for (n = 0; n < nk; n++) {

    psi_rho(fe->psi, index, n, &rho);
    assert(rho >= 0.0); /* For log(rho + epsilon) */

    fed += rho*(fe->kt*(log(rho + DBL_EPSILON) - 1.0)
		+ fe->psi->valency[n]*epsi);
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
  double rho;

  assert(fe);
  assert(fe->psi);
  assert(n < fe->psi->nk);

  rho = fe->psi->rho[fe->psi->nk*index + n];

  assert(rho >= 0.0); /* For log(rho + epsilon) */
  
  mu = fe->kt*log(rho + DBL_EPSILON)
    + fe->psi->valency[n]*fe->psi->e*fe->psi->psi[index];

  return mu;
}

/*****************************************************************************
 *
 *  fe_electro_stress
 *
 *  The stress is S_ab = -epsilon ( E_a E_b - (1/2) d_ab E^2)
 *  where epsilon is the (uniform) permeativity.
 *
 *  E_a is the total electric field, which is made up of the
 *  -grad psi and external field contributions.
 * 
 *****************************************************************************/

void fe_electro_stress(const int index, double s[3][3]) {

  int ia, ib;
  double epsilon;    /* Permeativity */
  double etot[3];    /* Total electric field */
  double e2;         /* Magnitude squared */

  assert(fe);

  psi_epsilon(fe->psi, &epsilon);
  psi_electric_field(fe->psi, index, etot);

  /* Add the external field, and compute E^2, and then the stress */

  e2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    etot[ia] += fe->ex[ia];
    e2 += etot[ia]*etot[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = -epsilon*(etot[ia]*etot[ib] - 0.5*d_[ia][ib]*e2);
    }
  }

  return;
}
