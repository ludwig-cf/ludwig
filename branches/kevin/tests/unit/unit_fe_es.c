/*****************************************************************************
 *
 *  test_fe_es.c
 *
 *  Electrokinetic plus Symetric free energy "Electrosymmetric"
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

#include "coords.h"
#include "leesedwards.h"
#include "fe_electro_symmetric.h"
#include "unit_control.h"

int do_test_fe_es1(control_t * ctrl);
int do_test_fe_es2(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_fe_es
 *
 *****************************************************************************/

int do_ut_fe_es(control_t * ctrl) {

  assert(ctrl);

  do_test_fe_es1(ctrl);
  do_test_fe_es2(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_es1
 *
 *****************************************************************************/

int do_test_fe_es1(control_t * ctrl) {

  int nk = 2;

  pe_t * pe = NULL;
  coords_t * coords = NULL;
  le_t * le = NULL;
  field_t * phi = NULL;
  field_grad_t * phi_grad = NULL;
  psi_t * psi = NULL;

  assert(ctrl);

  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Electrosymmetric object\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &coords);
  coords_commit(coords);

  le_create(coords, &le);
  le_commit(le);

  try {
    field_create(coords, 1, "phi", &phi);
    field_grad_create(phi, 2, &phi_grad);
    psi_create(coords, nk, &psi);

    control_macro_test(ctrl, phi != NULL);
    control_macro_test(ctrl, phi_grad != NULL);
    control_macro_test(ctrl, psi != NULL);

    fe_es_create(phi, phi_grad, psi);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    fe_es_free();
    psi_free(psi);
    field_grad_free(phi_grad);
    field_free(phi);
    le_free(le);
    coords_free(coords);
    pe_free(pe);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_es2
 *
 *****************************************************************************/

int do_test_fe_es2(control_t * ctrl) {

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

  pe_t * pe = NULL;
  coords_t * coords = NULL;
  le_t * le = NULL;
  psi_t * psi = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Electrosymmetric free energy\n");

  pe_create_parent(MPI_COMM_WORLD, &pe);
  coords_create(pe, &coords);
  coords_nhalo_set(coords, 2);
  coords_commit(coords);

  le_create(coords, &le);
  le_commit(le);

  psi_create(coords, nk, &psi);
  assert(psi);

  field_create(coords, 1, "phi", &phi);
  assert(phi);
  field_init(phi, 1, NULL);

  field_grad_create(phi, 2, &dphi);
  assert(dphi);

  fe_es_create(phi, dphi, psi);

  try {

    /* Check delta mu  = dmu [ 1 + phi ] / 2 */

    fe_es_deltamu_set(nk, dmu);

    phi0 = 0.0;
    field_scalar_set(phi, index, phi0);

    fe_es_mu_solv(index, 0, &dmu_test);
    control_verb(ctrl, "Delta mu[0] %10.4f (%10.4f)\n", 2.0*dmu_test, dmu[0]);
    control_macro_test_dbl_eq(ctrl, 2.0*dmu_test, dmu[0], DBL_EPSILON);
 
    fe_es_mu_solv(index, 1, &dmu_test);
    control_verb(ctrl, "Delta mu[1] %10.4f (%10.4f)\n", 2.0*dmu_test, dmu[1]);
    control_macro_test_dbl_eq(ctrl, 2.0*dmu_test, dmu[1], DBL_EPSILON);

    phi0 = 1.0;
    field_scalar_set(phi, index, phi0);
    fe_es_mu_solv(index, 0, &dmu_test);
    control_macro_test_dbl_eq(ctrl, dmu_test, dmu[0], DBL_EPSILON);

    fe_es_mu_solv(index, 1, &dmu_test);
    control_macro_test_dbl_eq(ctrl, dmu_test, dmu[1], DBL_EPSILON);


    /* Check epsilon e = ebar [ 1 - gamma phi ] */

    fe_es_epsilon_set(epsilon1, epsilon2);

    phi0 = 0.0;
    field_scalar_set(phi, index, phi0);
    fe_es_var_epsilon(index, &eps_test);
    eps_expect = ebar*(1.0 - gamma*phi0);

    control_macro_test_dbl_eq(ctrl, eps_expect, eps_test, DBL_EPSILON);

    phi0 = 1.0;
    field_scalar_set(phi, index, phi0);
    fe_es_var_epsilon(index, &eps_test);
    eps_expect = ebar*(1.0 - gamma*phi0);

    control_macro_test_dbl_eq(ctrl, eps_expect, eps_test, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    fe_es_free();
    field_grad_free(dphi);
    field_free(phi);
    psi_free(psi);
    le_free(le);
    coords_free(coords);
    pe_free(pe);
  }

  control_report(ctrl);

  return 0;
}
