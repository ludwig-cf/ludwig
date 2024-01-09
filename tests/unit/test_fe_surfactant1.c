/*****************************************************************************
 *
 *  test_fe_surfactant.c
 *
 *  Unit tests for surfactant free energy.
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "surfactant.h"
#include "tests.h"

__host__ int test_fe_surf_create(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_surf_xi_etc(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_surf_fed(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_surf_mu(pe_t * pe, cs_t * cs, field_t * phi);
__host__ int test_fe_surf_str(pe_t * pe, cs_t * cs, field_t * phi);

/* Some reference parameters */
static fe_surf_param_t pref = {-0.0208333,    /* a */
			       +0.0208333,    /* b */
			       0.12,          /* kappa */
			       0.00056587,    /* kT */
			       0.03,          /* epsilon */ 
			       0.0,           /* beta */
			       0.0};          /* W */

/*****************************************************************************
 *
 *  test_fe_surfactant1_suite
 *
 *****************************************************************************/

__host__ int test_fe_surfactant1_suite(void) {

  const int nf2 = 2;
  const int nhalo = 2;

  int ndevice;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  field_t * phi = NULL;

  tdpGetDeviceCount(&ndevice);

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_fe_surfactant\n");
  }
  else {

    field_options_t opts = field_options_ndata_nhalo(nf2, nhalo);

    cs_create(pe, &cs);
    cs_init(cs);

    field_create(pe, cs, NULL, "surfactant", &opts, &phi);

    test_fe_surf_create(pe, cs, phi);
    test_fe_surf_xi_etc(pe, cs, phi);
    test_fe_surf_fed(pe, cs, phi);
    test_fe_surf_mu(pe, cs, phi);
    test_fe_surf_str(pe, cs, phi);

    field_free(phi);
    cs_free(cs);
  }

  pe_info(pe, "PASS     ./unit/test_fe_surfactant\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_surf_create
 *
 *****************************************************************************/

__host__ int test_fe_surf_create(pe_t * pe, cs_t * cs, field_t * phi) {

  fe_surf_t * fe = NULL;
  fe_surf_param_t param = {0};
  field_grad_t * dphi = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_surf_create(pe, cs, phi, dphi, pref, &fe);

  assert(fe);

  /* Evidence that fe_surf_param_set() is working */
  fe_surf_param(fe, &param);
  test_assert(fabs(param.a       - pref.a)       < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(param.b       - pref.b)       < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(param.kappa   - pref.kappa)   < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(param.kt      - pref.kt)      < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(param.epsilon - pref.epsilon) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(param.beta    - pref.beta)    < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(param.w       - pref.w)       < TEST_DOUBLE_TOLERANCE);

  fe_surf_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_surf_xi_etc
 *
 *****************************************************************************/

__host__ int test_fe_surf_xi_etc(pe_t * pe, cs_t * cs, field_t * phi) {

  double xi0, xi0ref;
  double sigma, sigmaref;
  double psi0, psi0ref;
  double phistar;

  field_grad_t * dphi = NULL;
  fe_surf_t * fe = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_surf_create(pe, cs, phi, dphi, pref, &fe);

  /* Interfacial width */
  fe_surf_xi0(fe, &xi0);
  xi0ref = sqrt(-2.0*pref.kappa/pref.a);
  test_assert(fabs(xi0 - xi0ref) < TEST_DOUBLE_TOLERANCE);

  /* Interfacial tension */
  fe_surf_sigma(fe, &sigma);
  phistar = sqrt(-pref.a/pref.b);
  sigmaref = 4.0*pref.kappa*phistar*phistar/(3.0*xi0);
  test_assert(fabs(sigma - sigmaref) < TEST_DOUBLE_TOLERANCE);

  /* Langmuir isotherm */
  fe_surf_langmuir_isotherm(fe, &psi0);
  psi0ref = exp(0.5*pref.epsilon/(pref.kt*xi0ref*xi0ref));
  test_assert(fabs(psi0 - psi0ref) < TEST_DOUBLE_TOLERANCE);

  fe_surf_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_surf_fed
 *
 *****************************************************************************/

__host__ int test_fe_surf_fed(pe_t * pe, cs_t * cs, field_t * phi) {

  int ifail = 0;
  int index = 0;
  double psi, phisq, phipsi[2];
  double fed, fedref;

  field_grad_t * dphi = NULL;
  fe_surf_t * fe = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  assert(fabs(pref.beta - 0.0) < TEST_DOUBLE_TOLERANCE);
  assert(fabs(pref.w    - 0.0) < TEST_DOUBLE_TOLERANCE);

  field_grad_create(pe, phi, 2, &dphi);
  fe_surf_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients, phi = 0 */
  psi = 0.4;
  phipsi[0] = 0.0; phipsi[1] = psi;
  field_scalar_array_set(phi, index, phipsi);

  fe_surf_fed(fe, index, &fed);
  fedref = pref.kt*(psi*log(psi) + (1.0-psi)*log(1.0-psi));
  assert(fabs(fed - fedref) < TEST_DOUBLE_TOLERANCE);
  if (fabs(fed - fedref) >= TEST_DOUBLE_TOLERANCE) ifail = -1;

  /* No gradients, phi = 0.8 */
  psi = 0.4;
  phisq = 0.8*0.8;
  phipsi[0] = 0.8; phipsi[1] = psi;
  field_scalar_array_set(phi, index, phipsi);

  fe_surf_fed(fe, index, &fed);
  fedref += 0.5*pref.a*phisq + 0.25*pref.b*phisq*phisq;
  assert(fabs(fed - fedref) < TEST_DOUBLE_TOLERANCE);

  /* Interface: gradient in phi */

  {
    const double dphiref[2][3] = {{0.1,0.0,0.0}, {0.0,0.0,0.0}};

    field_grad_pair_grad_set(dphi, index, dphiref);

    fe_surf_fed(fe, index, &fed);
    fedref += 0.5*(pref.kappa - pref.epsilon*psi)*dphiref[0][X]*dphiref[0][X];
    assert(fabs(fed - fedref) < TEST_DOUBLE_TOLERANCE);
  }

  fe_surf_free(fe);
  field_grad_free(dphi);

  return ifail;
}

/*****************************************************************************
 *
 *  test_fe_surf_mu
 *
 *****************************************************************************/

__host__ int test_fe_surf_mu(pe_t * pe, cs_t * cs, field_t * phi) {

  int index = 1;
  double psi, phisq;
  double phipsi[2];
  double dsq[2];
  double muref, mu[2];

  field_grad_t * dphi = NULL;
  fe_surf_t * fe = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_surf_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients mu_phi, mu_psi */
  psi = 0.6;
  phipsi[0] = 0.8; phipsi[1] = psi;
  phisq = phipsi[0]*phipsi[0];
  field_scalar_array_set(phi, index, phipsi);

  fe_surf_mu(fe, index, mu);
  muref = (pref.a + pref.b*phisq + pref.w*psi)*phipsi[0];
  test_assert(fabs(mu[0] - muref) < TEST_DOUBLE_TOLERANCE);

  muref = pref.kt*(log(psi) - log(1.0-psi)) + 0.5*pref.w*phisq;
  test_assert(fabs(mu[1] - muref) < TEST_DOUBLE_TOLERANCE);

  /* Gradients phi */
  {
    const double dpref[2][3] = {{0.0,0.1,0.0}, {0.0,0.0,0.0}};
    dsq[0] = 0.0; dsq[1] = 0.0;

    field_grad_pair_grad_set(dphi, index, dpref);
    field_grad_pair_delsq_set(dphi, index, dsq);

    fe_surf_mu(fe, index, mu);

    muref = (pref.a + pref.b*phisq + pref.w*psi)*phipsi[0];
    test_assert(fabs(mu[0] - muref) < TEST_DOUBLE_TOLERANCE);


    muref = pref.kt*(log(psi) - log(1.0-psi)) + 0.5*pref.w*phisq
      - 0.5*pref.epsilon*dpref[0][Y]*dpref[0][Y];
    test_assert(fabs(mu[1] - muref) < TEST_DOUBLE_TOLERANCE);
  }

  /* Gradients phi, psi */
  {
    const double dpref[2][3] = {{0.0,0.1,0.0}, {0.2,0.3,0.0}};
    dsq[0] = 0.0; dsq[1] = 0.0;

    field_grad_pair_grad_set(dphi, index, dpref);
    field_grad_pair_delsq_set(dphi, index, dsq);

    fe_surf_mu(fe, index, mu);

    muref = (pref.a + pref.b*phisq + pref.w*psi)*phipsi[0]
      + pref.epsilon*dpref[0][Y]*dpref[1][Y];
    test_assert(fabs(mu[0] - muref) < TEST_DOUBLE_TOLERANCE);


    muref = pref.kt*(log(psi) - log(1.0-psi)) + 0.5*pref.w*phisq
      - 0.5*pref.epsilon*dpref[0][Y]*dpref[0][Y];
    test_assert(fabs(mu[1] - muref) < TEST_DOUBLE_TOLERANCE);
  }

  fe_surf_free(fe);
  field_grad_free(dphi);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_surf_str
 *
 *****************************************************************************/

__host__ int test_fe_surf_str(pe_t * pe, cs_t * cs, field_t * phi) {

  int index = 3;
  double s[3][3];
  double psi, phisq;
  double phipsi[2];
  double str;

  field_grad_t * dphi = NULL;
  fe_surf_t * fe = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  field_grad_create(pe, phi, 2, &dphi);
  fe_surf_create(pe, cs, phi, dphi, pref, &fe);

  /* No gradients; diagonal stress */
  psi = 0.6;
  phipsi[0] = -0.7; phipsi[1] = psi;
  phisq = phipsi[0]*phipsi[0];
  field_scalar_array_set(phi, index, phipsi);

  fe_surf_str(fe, index, s);

  str = 0.5*pref.a*phisq + 0.75*pref.b*phisq*phisq - pref.kt*log(1.0-psi)
      + pref.w*psi*phisq;

  test_assert(fabs(str     - s[X][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Y][Y] - s[X][X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(s[Z][Z] - s[X][X]) < TEST_DOUBLE_TOLERANCE);

  fe_surf_free(fe);
  field_grad_free(dphi);

  return 0;
}
