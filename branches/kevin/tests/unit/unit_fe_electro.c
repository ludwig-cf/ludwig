/*****************************************************************************
 *
 *  unit_fe_electro.c
 *
 *  Unit test for the electrokinetic free energy.
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "util.h"
#include "coords.h"
#include "physics.h"
#include "fe_electro.h"
#include "unit_control.h"

int do_test_fe_electro1(control_t * ctrl);
int do_test_fe_electro2(control_t * ctrl);
int do_test_fe_electro3(control_t * ctrl);

/*****************************************************************************
 *
 *  do_ut_fe_electro
 *
 *****************************************************************************/

int do_ut_fe_electro(control_t * ctrl) {

  physics_t * param = NULL;

  assert(ctrl);
  pe_init_quiet();
  coords_init();
  physics_ref(&param);

  do_test_fe_electro1(ctrl);
  do_test_fe_electro2(ctrl);
  do_test_fe_electro3(ctrl);

  coords_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_electro1
 *
 *  Free energy density.
 *
 *****************************************************************************/

int do_test_fe_electro1(control_t * ctrl) {

  int nk = 2;
  double valency[2] = {1, 2};
  double kt = 1.0;       /* Passes with scaled potential but REMOVE */
  double eunit = 1.0;    /* Passes with scaled potential but REMOVE */
  psi_t * psi = NULL;

  int index = 1;
  double rho0, rho1;     /* Test charge densities */
  double fed0, fed1;     /* Expected free energy contributions */
  double psi0;           /* Test potential */
  double fed;

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Free energy densities\n");

  psi_create(nk, &psi);
  psi_unit_charge_set(psi, eunit);
  physics_kt_set(kt);

  fe_electro_create(psi);

  try {

    /* psi = 0 so have \sum rho kT*(log(rho) - 1) */

    rho0 = 1.0;
    rho1 = 1.0;
    fed0 = rho0*kt*(log(rho0 + DBL_EPSILON) - 1.0);
    fed1 = rho0*kt*(log(rho1 + DBL_EPSILON) - 1.0);

    psi_rho_set(psi, index, 0, rho0);
    psi_rho_set(psi, index, 1, rho1);
    psi_psi_set(psi, index, 0.0);

    fed = fe_electro_fed(index);
    control_verb(ctrl, "rho = 1   fed = %22.15e, %22.15e\n", fed, fed0 + fed1);
    control_macro_test_dbl_eq(ctrl, fed, fed0 + fed1, DBL_EPSILON);

    rho0 = exp(1.0);
    fed0 = rho0*kt*(log(rho0) - 1.0);

    psi_rho_set(psi, index, 0, rho0);
    fed = fe_electro_fed(index);
    control_verb(ctrl, "rho = e   fed = %22.15e, %22.15e\n", fed, fed0 + fed1);
    control_macro_test_dbl_eq(ctrl, fed, fed0 + fed1, DBL_EPSILON);

    rho1 = exp(2.0);
    fed1 = rho1*kt*(log(rho1) - 1.0);

    psi_rho_set(psi, index, 1, rho1);
    fed = fe_electro_fed(index);
    control_verb(ctrl, "rho = e^2 fed = %22.15e, %22.15e\n", fed, fed0 + fed1);
    control_macro_test_dbl_eq(ctrl, fed, fed0 + fed1, DBL_EPSILON);

    /* For psi > 0 we add \sum rho 0.5 Z e psi */

    psi0 = 2.0;
    fed0 += rho0*0.5*valency[0]*eunit*psi0;
    fed1 += rho1*0.5*valency[1]*eunit*psi0;

    psi_psi_set(psi, index, psi0);
    psi_valency_set(psi, 0, valency[0]);
    psi_valency_set(psi, 1, valency[1]);
    fed = fe_electro_fed(index);
    control_verb(ctrl, "psi = 2   fed = %22.15e, %22.15e\n", fed, fed0 + fed1);
    control_macro_test_dbl_eq(ctrl, fed, fed0 + fed1, DBL_EPSILON);
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    fe_electro_free();
    psi_free(psi);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_electro2
 *
 *  Chemical potential
 *
 *****************************************************************************/

int do_test_fe_electro2(control_t * ctrl) {

  int n;
  int nk = 3;
  double kt = 0.1;
  double eunit = 1.0;
  double valency[3] = {3, 2, 1};
  psi_t * psi = NULL;

  int index = 1;
  double rho0;    /* Test charge density */
  double psi0;    /* Test potential */
  double mu0;     /* Expected chemical potential */
  double mu;      /* Actual chemical potential */

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Chemical potentias\n");

  psi_create(nk, &psi);
  psi_unit_charge_set(psi, eunit);
  physics_kt_set(kt);

  fe_electro_create(psi);

  try {

    for (n = 0; n < nk; n++) {
      rho0 = 1.0 + 1.0*n;
      psi_rho_set(psi, index, n, rho0);
      psi_valency_set(psi, n, valency[n]);
    
      /* For psi = 0, have mu_a = kT log(rho_a) */
      psi0 = 0.0;
      psi_psi_set(psi, index, psi0);
      mu0 = kt*log(rho0);
      mu = fe_electro_mu(index, n);
      control_macro_test_dbl_eq(ctrl, mu, mu0, DBL_EPSILON);
      /* assert(fabs(mu - mu0) < DBL_EPSILON);*/

      /* Complete mu_a = kT log(rho) + Z_a e psi */
      psi0 = 1.0;
      psi_psi_set(psi, index, psi0);
      mu0 = kt*log(rho0) + valency[n]*eunit*psi0;
      mu = fe_electro_mu(index, n);
      control_macro_test_dbl_eq(ctrl, mu, mu0, DBL_EPSILON);
      /* assert(fabs(mu - mu0) < DBL_EPSILON);*/
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    fe_electro_free();
    psi_free(psi);
  }

  control_report(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_fe_electro3
 *
 *  Stress. This is a bit more work, as the stress depends on the
 *  gradient of the current potential.
 *
 *****************************************************************************/

int do_test_fe_electro3(control_t * ctrl) {

  int nk = 2;
  int index;
  int ia, ib;
  psi_t * psi = NULL;

  double kt = 1..0;                 /* Reset following previous test */
  double epsilon = 0.5;             /* Permeativity */
  double s[3][3];                   /* A stress */
  double e0[3];                     /* A field */

  double psi0, psi1;
  double emod;
  double sexpect;

  assert(ctrl);
  control_test(ctrl, __CONTROL_INFO__);
  control_verb(ctrl, "Stress calculation\n");

  psi_create(nk, &psi);
  psi_epsilon_set(psi, epsilon);
  fe_electro_create(psi);
  physics_kt_set(kt);

  try {

    /* No external field, no potential; note index must allow for a
     * spatial gradient */

    index = coords_index(1, 1, 1);
    fe_electro_stress(index, s);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	sexpect = 0.0;
	control_verb(ctrl, "s[%d][%d] %22.15e %22.15e\n", ia, ib,
		     s[ia][ib], sexpect);
	control_macro_test_dbl_eq(ctrl, s[ia][ib], sexpect, DBL_EPSILON);
      }
    }

    /* With a potential (only): explicitly set the relevant terms
     * (we need to know the differencing scheme in fe_electro.c). */

    psi0 = 1.0;
    psi1 = 2.0;
    psi_psi_set(psi, coords_index(1+1, 1, 1), psi1);
    psi_psi_set(psi, coords_index(1-1, 1, 1), psi0);
    e0[X] = -0.5*(psi1 - psi0);

    psi0 = 3.0;
    psi1 = 4.0;
    psi_psi_set(psi, coords_index(1, 1+1, 1), psi1);
    psi_psi_set(psi, coords_index(1, 1-1, 1), psi0);
    e0[Y] = -0.5*(psi1 - psi0);

    psi0 = 6.0;
    psi1 = 5.0;
    psi_psi_set(psi, coords_index(1, 1, 1+1), psi1);
    psi_psi_set(psi, coords_index(1, 1, 1-1), psi0);
    e0[Z] = -0.5*(psi1 - psi0);
    emod = modulus(e0);

    fe_electro_stress(coords_index(1, 1, 1), s);

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	sexpect = -epsilon*(e0[ia]*e0[ib] - 0.5*d_[ia][ib]*emod*emod);
	control_verb(ctrl, "s[%d][%d] %22.15e %22.15e\n", ia, ib,
		     s[ia][ib], sexpect);
	control_macro_test_dbl_eq(ctrl, s[ia][ib], sexpect, DBL_EPSILON);
      }
    }
  }
  catch (TestFailedException) {
    control_option_set(ctrl, CONTROL_FAIL);
  }
  finally {
    fe_electro_free();
    psi_free(psi);
  }

  control_report(ctrl);

  return 0;
}
