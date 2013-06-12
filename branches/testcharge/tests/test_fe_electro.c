/*****************************************************************************
 *
 *  test_fe_electro.c
 *
 *  Unit test for the electrokinetic free energy.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Phsyics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2013)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
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
#include "fe_electro.h"

static int do_test1(void);
static int do_test2(void);
static int do_test3(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  physics_t * param = NULL;

  MPI_Init(&argc, &argv);
  pe_init();
  coords_init();
  physics_ref(&param);

  do_test1();
  do_test2();
  do_test3();

  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *  Free energy density.
 *
 *****************************************************************************/

static int do_test1(void) {

  int nk = 2;
  double valency[2] = {1, 2};
  double kt = 2.0;
  double eunit = 3.0;
  psi_t * psi = NULL;

  int index = 1;
  double rho0, rho1;     /* Test charge densities */
  double fed0, fed1;     /* Expected free energy contributions */
  double psi0;           /* Test potential */
  double fed;

  psi_create(nk, &psi);
  psi_unit_charge_set(psi, eunit);
  physics_kt_set(kt);

  fe_electro_create(psi);

  /* psi = 0 so have \sum rho kT*(log(rho) - 1) */

  rho0 = 1.0;
  rho1 = 1.0;
  fed0 = rho0*kt*(log(rho0 + DBL_EPSILON) - 1.0);
  fed1 = rho0*kt*(log(rho1 + DBL_EPSILON) - 1.0);

  psi_rho_set(psi, index, 0, rho0);
  psi_rho_set(psi, index, 1, rho1);
  psi_psi_set(psi, index, 0.0);

  fed = fe_electro_fed(index);
  assert(fabs(fed - (fed0 + fed1)) < DBL_EPSILON);

  rho0 = exp(1.0);
  fed0 = rho0*kt*(log(rho0) - 1.0);

  psi_rho_set(psi, index, 0, rho0);
  fed = fe_electro_fed(index);
  assert(fabs(fed - (fed0 + fed1)) < DBL_EPSILON);

  rho1 = exp(2.0);
  fed1 = rho1*kt*(log(rho1) - 1.0);

  psi_rho_set(psi, index, 1, rho1);
  fed = fe_electro_fed(index);
  assert(fabs(fed - (fed0 + fed1)) < DBL_EPSILON);

  /* For psi > 0 we add \sum rho 0.5 Z e psi */

  psi0 = 2.0;
  fed0 += rho0*0.5*valency[0]*eunit*psi0;
  fed1 += rho1*0.5*valency[1]*eunit*psi0;

  psi_psi_set(psi, index, psi0);
  psi_valency_set(psi, 0, valency[0]);
  psi_valency_set(psi, 1, valency[1]);
  fed = fe_electro_fed(index);
  assert(fabs(fed - (fed0 + fed1)) < DBL_EPSILON);

  fe_electro_free();
  psi_free(psi);

  return 0;
}

/*****************************************************************************
 *
 *  do_test2
 *
 *  Chemical potential
 *
 *****************************************************************************/

int do_test2(void) {

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

  psi_create(nk, &psi);
  psi_unit_charge_set(psi, eunit);
  physics_kt_set(kt);

  fe_electro_create(psi);

  for (n = 0; n < nk; n++) {
    rho0 = 1.0 + 1.0*n;
    psi_rho_set(psi, index, n, rho0);
    psi_valency_set(psi, n, valency[n]);
    
    /* For psi = 0, have mu_a = kT log(rho_a) */
    psi0 = 0.0;
    psi_psi_set(psi, index, psi0);
    mu0 = kt*log(rho0);
    mu = fe_electro_mu(index, n);
    assert(fabs(mu - mu0) < DBL_EPSILON);

    /* Complete mu_a = kT log(rho) + Z_a e psi */
    psi0 = 1.0;
    psi_psi_set(psi, index, psi0);
    mu0 = kt*log(rho0) + valency[n]*eunit*psi0;
    mu = fe_electro_mu(index, n);
    assert(fabs(mu - mu0) < DBL_EPSILON);
  }

  fe_electro_free();
  psi_free(psi);

  return 0;
}

/*****************************************************************************
 *
 *  do_test3
 *
 *  Stress. This is a bit more work, as the stress depends on the
 *  gradient of the current potential.
 *
 *****************************************************************************/

static int do_test3(void) {

  int nk = 2;
  int index = 1;
  int ia, ib;
  psi_t * psi = NULL;

  double epsilon = 0.5;             /* Permeativity */
  double ex[3] = {1.0, 2.0, 3.0};   /* External field */
  double s[3][3];                   /* A stress */
  double e0[3];                     /* A field */

  double psi0, psi1;
  double emod;
  double sexpect;

  psi_create(nk, &psi);
  psi_epsilon_set(psi, epsilon);
  fe_electro_create(psi);

  /* No external field, no potential */

  fe_electro_stress(index, s);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sexpect = 0.0;
      assert(fabs(s[ia][ib] - sexpect) < DBL_EPSILON);
    }
  }

  /* External field, no potential */

  physics_e0_set(ex);
  fe_electro_stress(index, s);
  emod = modulus(ex);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sexpect = -epsilon*(ex[ia]*ex[ib] - 0.5*d_[ia][ib]*emod*emod);
      assert(fabs(s[ia][ib] - sexpect) < DBL_EPSILON);
    }
  }

  /* With a potential (only): explicitly set the relevant terms
   * (we need to know the differencing scheme in fe_electro.c). */

  ex[X] = 0.0;
  ex[Y] = 0.0;
  ex[Z] = 0.0;
  physics_e0_set(ex);

  /* The 'true' field */

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
      assert(fabs(s[ia][ib] - sexpect) < DBL_EPSILON);
    }
  }

  fe_electro_free();
  psi_free(psi);

  return 0;
}
