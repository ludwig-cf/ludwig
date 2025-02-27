/*****************************************************************************
 *
 *  test_fe_electro_symm.c
 *
 *  Electrokinetic + symmetric free energy
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

static int do_test1(pe_t * pe);

/*****************************************************************************
 *
 *  test_fe_electro_symm_suite
 *
 *****************************************************************************/

int test_fe_electro_symm_suite(void) {

  int ndevice;
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_fe_electro_symm\n");
  }
  else {
    do_test1(pe);
    pe_info(pe, "PASS     ./unit/test_fe_electro_symm\n");
  }

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *****************************************************************************/

static int do_test1(pe_t * pe) {

  int nk = 2;
  int nhalo = 2;
  int index = 1;
  double dmu[2] = {1.0, 2.0};  /* Solvation free energy differences */
  double dmu_test;

  double epsilon1 = 1.0;
  double epsilon2 = 2.0;
  double ebar = 0.5*(epsilon1 + epsilon2);
  double gamma = (epsilon1 - epsilon2) / (epsilon1 + epsilon2);
  double eps_test, eps_expect;
  double phi0;

  cs_t * cs = NULL;
  psi_t * psi = NULL;
  field_t * phi = NULL;
  field_grad_t * dphi = NULL;
  fe_symm_t * fe_symm = NULL;
  fe_electro_t * fe_elec = NULL;
  fe_es_t * fe = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  {
    psi_options_t opts = psi_options_default(nhalo);
    opts.epsilon1 = epsilon1;
    opts.epsilon2 = epsilon2;
    psi_create(pe, cs, &opts, &psi);
  }
  fe_electro_create(pe, psi, &fe_elec);

  {
    field_options_t opts = field_options_ndata_nhalo(1, 1);
    field_create(pe, cs, NULL, "phi", &opts, &phi);
  }

  field_grad_create(pe, phi, 2, &dphi);
  assert(dphi);
  fe_symm_create(pe, cs, phi, dphi, &fe_symm);


  fe_es_create(pe, cs, fe_symm, fe_elec, psi, &fe);

  /* Check delta mu  = dmu [ 1 + phi ] / 2 */

  fe_es_deltamu_set(fe, nk, dmu);

  phi0 = 0.0;
  field_scalar_set(phi, index, phi0);
  fe_es_mu_ion_solv(fe, index, 0, &dmu_test);
  test_assert(fabs(2.0*dmu_test - dmu[0]) < DBL_EPSILON);
  fe_es_mu_ion_solv(fe, index, 1, &dmu_test);
  test_assert(fabs(2.0*dmu_test - dmu[1]) < DBL_EPSILON);

  phi0 = 1.0;
  field_scalar_set(phi, index, phi0);
  fe_es_mu_ion_solv(fe, index, 0, &dmu_test);
  test_assert(fabs(dmu_test - dmu[0]) < DBL_EPSILON);
  fe_es_mu_ion_solv(fe, index, 1, &dmu_test);
  test_assert(fabs(dmu_test - dmu[1]) < DBL_EPSILON);


  /* Check epsilon e = ebar [ 1 - gamma phi ] */

  phi0 = 0.0;
  field_scalar_set(phi, index, phi0);
  fe_es_var_epsilon(fe, index, &eps_test);
  eps_expect = ebar*(1.0 - gamma*phi0);
  test_assert(fabs(eps_expect - eps_test) < DBL_EPSILON);

  phi0 = 1.0;
  field_scalar_set(phi, index, phi0);
  fe_es_var_epsilon(fe, index, &eps_test);
  eps_expect = ebar*(1.0 - gamma*phi0);
  test_assert(fabs(eps_expect - eps_test) < DBL_EPSILON);

  /* Finish. */

  fe_es_free(fe);
  fe_symm_free(fe_symm);
  fe_electro_free(fe_elec);
  field_grad_free(dphi);
  field_free(phi);
  psi_free(&psi);
  cs_free(cs);

  return 0;
}
