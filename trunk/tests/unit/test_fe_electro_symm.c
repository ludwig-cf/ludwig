/*****************************************************************************
 *
 *  test_fe_electro_symm.c
 *
 *  Electrokinetic + symetric free energy
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "fe_electro_symmetric.h"
#include "tests.h"

static int do_test1(void);

/*****************************************************************************
 *
 *  test_fe_electro_symm_suite
 *
 *****************************************************************************/

int test_fe_electro_symm_suite(void) {

  pe_init_quiet();
  coords_nhalo_set(2);
  coords_init();
  le_init();

  do_test1();

  info("PASS     ./unit/test_fe_electro_symm\n");
  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *****************************************************************************/

static int do_test1(void) {

  int nk = 2;
  int index = 1;
  double dmu[2] = {1.0, 2.0};  /* Solvation free energy differences */
  double dmu_test;

  double epsilon1 = 1.0;
  double epsilon2 = 2.0;
  double ebar = 0.5*(epsilon1 + epsilon2);
  double gamma = (epsilon1 - epsilon2) / (epsilon1 + epsilon2);
  double eps_test, eps_expect;
  double phi0;

  psi_t * psi = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;

  psi_create(nk, &psi);
  assert(psi);

  field_create(1, "phi", &phi);
  assert(phi);
  field_init(phi, 1);

  field_grad_create(phi, 2, &dphi);
  assert(dphi);

  fe_es_create(phi, dphi, psi);

  /* Check delta mu  = dmu [ 1 + phi ] / 2 */

  fe_es_deltamu_set(nk, dmu);

  phi0 = 0.0;
  field_scalar_set(phi, index, phi0);
  fe_es_mu_ion_solv(index, 0, &dmu_test);
  assert(fabs(2.0*dmu_test - dmu[0]) < DBL_EPSILON);
  fe_es_mu_ion_solv(index, 1, &dmu_test);
  assert(fabs(2.0*dmu_test - dmu[1]) < DBL_EPSILON);

  phi0 = 1.0;
  field_scalar_set(phi, index, phi0);
  fe_es_mu_ion_solv(index, 0, &dmu_test);
  assert(fabs(dmu_test - dmu[0]) < DBL_EPSILON);
  fe_es_mu_ion_solv(index, 1, &dmu_test);
  assert(fabs(dmu_test - dmu[1]) < DBL_EPSILON);


  /* Check epsilon e = ebar [ 1 - gamma phi ] */

  fe_es_epsilon_set(epsilon1, epsilon2);

  phi0 = 0.0;
  field_scalar_set(phi, index, phi0);
  fe_es_var_epsilon(index, &eps_test);
  eps_expect = ebar*(1.0 - gamma*phi0);
  assert(fabs(eps_expect - eps_test) < DBL_EPSILON);

  phi0 = 1.0;
  field_scalar_set(phi, index, phi0);
  fe_es_var_epsilon(index, &eps_test);
  eps_expect = ebar*(1.0 - gamma*phi0);
  assert(fabs(eps_expect - eps_test) < DBL_EPSILON);

  /* Finish. */

  fe_es_free();
  field_grad_free(dphi);
  field_free(phi);
  psi_free(psi);

  return 0;
}
